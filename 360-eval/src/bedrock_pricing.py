# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Amazon Bedrock model pricing with 3-tier fallback and local file caching.

Adapted from multitenancy-and-observability-for-bedrock/backend/shared/pricing.py.
Replaces DynamoDB caching with local file cache using the same hash-based
invalidation pattern as model_capability_validator.py.

Flow:
    1. Check local cache -- if profiles_hash matches, return immediately
    2. Tier 1: Query AWS Price List API (all 3 service codes)
       -> If both input + output found: pricing_source = "api"
    3. Tier 2: If output missing, re-query AmazonAmazon BedrockService only
       -> If filled: pricing_source = "api_partial"
    4. Tier 3: Scrape the Amazon Bedrock pricing webpage bulk JSON
       -> If filled: pricing_source = "webpage"
    5. If still missing -> "unavailable" with None costs
    6. Cache the result locally in .cache/model_pricing.json
"""

import gzip
import hashlib
import html as html_module
import json
import logging
import re
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import boto3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
PRICING_CACHE_FILE = CACHE_DIR / "model_pricing.json"
MODELS_PROFILE_PATH = PROJECT_ROOT / "config" / "models_profiles.jsonl"
JUDGE_PROFILE_PATH = PROJECT_ROOT / "config" / "judge_profiles.jsonl"

SERVICE_CODES = [
    "AmazonBedrock",
    "AmazonAmazon BedrockService",
    "AmazonAmazon BedrockFoundationModels",
]
PRICING_CLIENT_REGION = "us-east-1"
PRICING_REFRESH_DAYS = 7  # Re-fetch pricing after this many days

# Amazon Bedrock pricing page and bulk JSON endpoints
_PRICING_PAGE_URL = "https://aws.amazon.com/bedrock/pricing/"
_BULK_JSON_URL = "https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/{svc}/USD/current/{svc}.json"

# Map service paths from priceOf tokens to bulk JSON service names
_SERVICE_PATH_MAP = {
    "bedrock/bedrock": "bedrock",
    "bedrockfoundationmodels/bedrockfoundationmodels": "bedrockfoundationmodels",
    "bedrockservice/bedrockservice": "bedrockservice",
}

# Region code to display name (as used on the AWS pricing page)
_REGION_DISPLAY_NAMES = {
    "us-east-1": "US East (N. Virginia)",
    "us-east-2": "US East (Ohio)",
    "us-west-1": "US West (N. California)",
    "us-west-2": "US West (Oregon)",
    "af-south-1": "Africa (Cape Town)",
    "ap-east-1": "Asia Pacific (Hong Kong)",
    "ap-east-2": "Asia Pacific (Taipei)",
    "ap-south-1": "Asia Pacific (Mumbai)",
    "ap-south-2": "Asia Pacific (Hyderabad)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
    "ap-southeast-2": "Asia Pacific (Sydney)",
    "ap-southeast-3": "Asia Pacific (Jakarta)",
    "ap-southeast-4": "Asia Pacific (Melbourne)",
    "ap-southeast-5": "Asia Pacific (Malaysia)",
    "ap-southeast-7": "Asia Pacific (Thailand)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "ap-northeast-2": "Asia Pacific (Seoul)",
    "ap-northeast-3": "Asia Pacific (Osaka)",
    "ap-south-3": "Asia Pacific (New Zealand)",
    "ca-central-1": "Canada (Central)",
    "ca-west-1": "Canada West (Calgary)",
    "eu-central-1": "EU (Frankfurt)",
    "eu-central-2": "EU (Zurich)",
    "eu-west-1": "EU (Ireland)",
    "eu-west-2": "EU (London)",
    "eu-west-3": "EU (Paris)",
    "eu-north-1": "EU (Stockholm)",
    "eu-south-1": "EU (Milan)",
    "eu-south-2": "EU (Spain)",
    "me-south-1": "Middle East (Bahrain)",
    "me-central-1": "Middle East (UAE)",
    "mx-central-1": "Mexico (Central)",
    "sa-east-1": "South America (Sao Paulo)",
    "il-central-1": "Israel (Tel Aviv)",
    "us-gov-west-1": "AWS GovCloud (US)",
    "us-gov-east-1": "AWS GovCloud (US-East)",
}


# ---------------------------------------------------------------------------
# Local file cache (mirrors model_capability_validator.py pattern)
# ---------------------------------------------------------------------------

def get_profiles_hash() -> str:
    """Generate SHA256 hash of models_profiles.jsonl + judge_profiles.jsonl content."""
    h = hashlib.sha256()
    for path in (MODELS_PROFILE_PATH, JUDGE_PROFILE_PATH):
        try:
            with open(path, "r", encoding="utf-8") as f:
                h.update(f.read().encode())
        except FileNotFoundError:
            logger.warning("Profile file not found: %s", path)
        except Exception as e:
            logger.error("Error reading profile %s: %s", path, e)
    return h.hexdigest()


def load_pricing_cache() -> dict:
    """Load cached pricing from local JSON file."""
    if not PRICING_CACHE_FILE.exists():
        return {"last_updated": None, "profiles_hash": "", "pricing": {}}
    try:
        with open(PRICING_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Error loading pricing cache: %s", e)
        return {"last_updated": None, "profiles_hash": "", "pricing": {}}


def save_pricing_cache(cache: dict) -> bool:
    """Save pricing cache to local JSON file."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(PRICING_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        logger.info("Pricing cache saved to %s", PRICING_CACHE_FILE)
        return True
    except Exception as e:
        logger.error("Error saving pricing cache: %s", e)
        return False


def is_pricing_cache_valid() -> bool:
    """Check if the pricing cache is still valid (profiles haven't changed)."""
    cache = load_pricing_cache()
    if not cache.get("profiles_hash"):
        return False
    return cache["profiles_hash"] == get_profiles_hash()


# ---------------------------------------------------------------------------
# Model ID helpers
# ---------------------------------------------------------------------------

def strip_model_id_for_pricing(model_id: str) -> str:
    """Strip 'bedrock/' prefix and cross-region prefixes for API lookup.

    Examples:
        bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0
            -> anthropic.claude-sonnet-4-5-20250929-v1:0
        bedrock/deepseek.v3-v1:0
            -> deepseek.v3-v1:0
        bedrock/converse/us.amazon.nova-2-lite-v1:0
            -> amazon.nova-2-lite-v1:0
    """
    cleaned = model_id
    for prefix in ("bedrock/converse/", "bedrock/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    # Remove cross-region inference prefixes (us., eu., ap., etc.)
    cleaned = re.sub(r"^[a-z]{2}\.", "", cleaned)
    return cleaned


def _cache_key(model_id: str, region: str) -> str:
    """Build cache key from model_id (without bedrock/ prefix) and region."""
    # Strip bedrock/ but keep cross-region prefix for uniqueness
    stripped = model_id
    for prefix in ("bedrock/converse/", "bedrock/"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    return f"{region}#{stripped}"


# ---------------------------------------------------------------------------
# Pricing API helpers
# ---------------------------------------------------------------------------

def get_all_products(pricing_client, service_code: str) -> list:
    """Paginate through all products for a service code."""
    products = []
    next_token = None
    while True:
        kwargs = {
            "ServiceCode": service_code,
            "MaxResults": 100,
            "FormatVersion": "aws_v1",
        }
        if next_token:
            kwargs["NextToken"] = next_token
        response = pricing_client.get_products(**kwargs)
        for item_json in response.get("PriceList", []):
            products.append(json.loads(item_json))
        next_token = response.get("NextToken")
        if not next_token:
            break
    return products


def extract_pricing(product: dict) -> dict:
    """Extract pricing details from a single product entry."""
    attrs = product.get("product", {}).get("attributes", {})
    terms = product.get("terms", {})

    price_per_unit = None
    for offer in terms.get("OnDemand", {}).values():
        for dim in offer.get("priceDimensions", {}).values():
            price_per_unit = dim.get("pricePerUnit", {}).get("USD")
            break

    return {
        "model_id": attrs.get("modelId", ""),
        "usagetype": attrs.get("usagetype", ""),
        "inference_type": attrs.get("inferenceType", ""),
        "feature": attrs.get("feature", ""),
        "feature_type": attrs.get("featuretype", ""),
        "region_code": attrs.get("regionCode", ""),
        "group_description": attrs.get("groupDescription", ""),
        "price_per_unit_usd": price_per_unit,
    }


def is_on_demand_standard(entry: dict) -> bool:
    """Filter to only on-demand, non-reserved, non-batch, non-cache entries."""
    usagetype = entry["usagetype"].lower()
    feature = entry["feature"].lower()
    feature_type = entry["feature_type"].lower()

    for skip in ("reserved", "batch", "cache", "long-context", "latency-optimized"):
        if skip in usagetype or skip in feature or skip in feature_type:
            return False
    return True


def classify_token_type(entry: dict) -> str:
    """Determine if this is input or output token pricing."""
    usagetype = entry["usagetype"].lower()
    inference_type = entry["inference_type"].lower()
    group_desc = entry["group_description"].lower()

    if "output" in usagetype or "output" in inference_type or "output" in group_desc:
        return "output"
    elif "input" in usagetype or "input" in inference_type or "input" in group_desc:
        return "input"
    return "unknown"


def _matches_model(entry: dict, model_id: str, region: str) -> bool:
    """Check if a pricing entry matches the given model_id and region."""
    if entry["region_code"] and entry["region_code"] != region:
        return False

    # Strip bare trailing version suffix like ":0" that isn't part of a "-vN:N" pattern
    # e.g., "kimi-k2-thinking:0" -> "kimi-k2-thinking"
    # but keep "claude-sonnet-4-5-20250929-v1:0" intact
    clean_id = re.sub(r"(?<!-v\d):(\d+)$", "", model_id)
    model_lower = clean_id.lower()
    entry_model_id = entry.get("model_id", "").lower()
    entry_usagetype = entry.get("usagetype", "").lower()

    if entry_model_id and entry_model_id == model_lower:
        return True
    if model_lower in entry_usagetype:
        return True

    # Normalized substring match (strips dots, dashes, underscores, colons)
    a = re.sub(r"[-._:]", "", model_lower)
    if entry_model_id:
        b = re.sub(r"[-._:]", "", entry_model_id)
        if a in b or b in a:
            return True
    if entry_usagetype:
        b = re.sub(r"[-._:]", "", entry_usagetype)
        if a in b:
            return True

    # Token-overlap match: split on separators and check overlap
    # Handles vendor name mismatches (e.g., "moonshot" vs "moonshotai")
    model_parts = re.split(r"[-._:]", model_lower)
    model_parts = [p for p in model_parts if p]
    if entry_usagetype and len(model_parts) > 1:
        usage_tokens = set(re.split(r"[-._:]", entry_usagetype)) - {""}
        # For "vendor.model-name" format, skip the vendor prefix (first part before ".")
        # and require all remaining tokens to appear in usagetype
        if "." in model_id:
            dot_pos = model_lower.index(".")
            non_vendor_parts = re.split(r"[-._:]", model_lower[dot_pos + 1:])
            non_vendor = set(p for p in non_vendor_parts if p)
        else:
            non_vendor = set(model_parts)
        if non_vendor and non_vendor.issubset(usage_tokens):
            return True

    return False


def _extract_costs_from_products(products: list, model_id: str, region: str) -> dict:
    """Extract input/output costs from a list of products for a model+region."""
    input_cost = None
    output_cost = None

    for p in products:
        entry = extract_pricing(p)

        if not _matches_model(entry, model_id, region):
            continue
        if not is_on_demand_standard(entry):
            continue

        token_type = classify_token_type(entry)
        price = entry["price_per_unit_usd"]
        if price is None:
            continue

        price_float = float(price)
        if token_type == "input" and input_cost is None:
            input_cost = price_float
        elif token_type == "output" and output_cost is None:
            output_cost = price_float

        if input_cost is not None and output_cost is not None:
            break

    return {"input_cost": input_cost, "output_cost": output_cost}


# ---------------------------------------------------------------------------
# Tier 1: Query all 3 service codes
# ---------------------------------------------------------------------------

def _fetch_all_products_parallel() -> list:
    """Fetch all products from all 3 service codes in parallel.

    Returns a combined list of all product entries.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pricing_client = boto3.client("pricing", region_name=PRICING_CLIENT_REGION)
    all_products = []

    def fetch_service(sc):
        try:
            products = get_all_products(pricing_client, sc)
            logger.info("Fetched %d products from %s", len(products), sc)
            return products
        except Exception as exc:
            logger.warning("Failed to query %s: %s", sc, exc)
            return []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch_service, sc): sc for sc in SERVICE_CODES}
        for future in as_completed(futures):
            all_products.extend(future.result())

    logger.info("Total products fetched: %d", len(all_products))
    return all_products


def _resolve_bulk_from_products(
    products: list, models: list[tuple[str, str]]
) -> dict[str, dict]:
    """Match all (model_id, region) pairs against a pre-fetched product list.

    Args:
        products: Combined product list from all service codes.
        models: List of (stripped_model_id, region) tuples.

    Returns:
        Dict keyed by "region#model_id" with {input_cost, output_cost} values.
    """
    # Pre-extract all pricing entries once
    entries = []
    for p in products:
        entry = extract_pricing(p)
        if not is_on_demand_standard(entry):
            continue
        token_type = classify_token_type(entry)
        if token_type == "unknown":
            continue
        price = entry["price_per_unit_usd"]
        if price is None:
            continue
        entry["_token_type"] = token_type
        entry["_price_float"] = float(price)
        entries.append(entry)

    logger.info("Pre-filtered to %d on-demand pricing entries", len(entries))

    results = {}
    for model_id, region in models:
        key = f"{region}#{model_id}"
        input_cost = None
        output_cost = None

        for entry in entries:
            if not _matches_model(entry, model_id, region):
                continue

            if entry["_token_type"] == "input" and input_cost is None:
                input_cost = entry["_price_float"]
            elif entry["_token_type"] == "output" and output_cost is None:
                output_cost = entry["_price_float"]

            if input_cost is not None and output_cost is not None:
                break

        results[key] = {"input_cost": input_cost, "output_cost": output_cost}

    return results


def _query_price_list_api(model_id: str, region: str) -> dict:
    """Tier 1: Query AWS Price List API across all 3 service codes.

    Returns {"input_cost": float|None, "output_cost": float|None}.
    """
    products = _fetch_all_products_parallel()
    return _extract_costs_from_products(products, model_id, region)


# ---------------------------------------------------------------------------
# Tier 2: Re-query AmazonAmazon BedrockService only for missing output pricing
# ---------------------------------------------------------------------------

def _query_bedrock_service_only(model_id: str, region: str) -> dict:
    """Tier 2: Re-query only AmazonAmazon BedrockService for missing pricing.

    Some models (older Anthropic) have input pricing in AmazonAmazon Bedrock
    but output pricing only in AmazonAmazon BedrockService.

    Returns {"input_cost": float|None, "output_cost": float|None}.
    """
    pricing_client = boto3.client("pricing", region_name=PRICING_CLIENT_REGION)

    try:
        products = get_all_products(pricing_client, "AmazonAmazon BedrockService")
    except Exception as exc:
        logger.warning("Failed to query AmazonAmazon BedrockService for tier 2: %s", exc)
        return {"input_cost": None, "output_cost": None}

    return _extract_costs_from_products(products, model_id, region)


# ---------------------------------------------------------------------------
# Tier 3: Scrape Amazon Bedrock pricing webpage bulk JSON
# ---------------------------------------------------------------------------

def _fetch_url(url: str, timeout: int = 15) -> bytes:
    """Fetch a URL and return raw bytes. Handles gzip encoding."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "BedrockBenchmark/1.0",
        "Accept-Encoding": "gzip, deflate",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
        return data


def _parse_pricing_page(page_html: str) -> list[dict]:
    """Parse the Amazon Bedrock pricing page HTML to extract model-to-hash mappings."""
    content = html_module.unescape(page_html)

    token_re = (
        r'\{priceOf!'
        r'(?P<path>[^!}]+/[^!}]+)!'
        r'(?P<hash>[A-Za-z0-9_-]+)'
        r'(?:!\*!(?P<mult>\d+))?'
        r'(?:!opt)?'
        r'\}'
    )

    token_re_unnamed = (
        r'\{priceOf!'
        r'[^!}]+/[^!}]+!'
        r'[A-Za-z0-9_-]+'
        r'(?:!\*!\d+)?'
        r'(?:!opt)?'
        r'\}'
    )

    row_re = (
        r'<td[^>]*>([^<]{2,80})</td>'
        r'\s*<td[^>]*>(' + token_re_unnamed + r')</td>'
        r'\s*<td[^>]*>(' + token_re_unnamed + r')</td>'
    )

    entries = []
    seen = set()

    for row_match in re.finditer(row_re, content):
        model_name = row_match.group(1).strip()
        input_cell = row_match.group(2)
        output_cell = row_match.group(3)

        input_m = re.search(token_re, input_cell)
        output_m = re.search(token_re, output_cell)
        if not input_m or not output_m:
            continue

        service_path = input_m.group("path")
        input_hash = input_m.group("hash")
        output_hash = output_m.group("hash")
        input_mult = int(input_m.group("mult") or "1")
        output_mult = int(output_m.group("mult") or "1")

        dedup_key = (model_name, service_path, input_hash)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        name_lower = model_name.lower()
        if any(skip in name_lower for skip in ("long context", "batch", "priority", "flex")):
            continue

        entries.append({
            "model_name": model_name,
            "service_path": service_path,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "input_mult": input_mult,
            "output_mult": output_mult,
        })

    return entries


def _normalize_for_match(s: str) -> str:
    """Normalize a model identifier for fuzzy matching."""
    s = s.lower()
    for prefix in ("anthropic.", "meta.", "amazon.", "cohere.", "ai21.",
                    "mistral.", "stability.", "deepseek.", "us.", "eu.",
                    "qwen.", "moonshot.", "openai."):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = re.sub(r"-\d{8}-v\d+:\d+$", "", s)
    s = re.sub(r"-v\d+:\d+$", "", s)
    s = re.sub(r"-v\d+$", "", s)
    s = re.sub(r"[-._:]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _find_model_entry(model_id: str, entries: list[dict]) -> Optional[dict]:
    """Find the best matching entry for a model_id using fuzzy matching."""
    norm_id = _normalize_for_match(model_id)
    norm_id_nospaces = norm_id.replace(" ", "")

    best_match = None
    best_score = 0

    for entry in entries:
        norm_name = _normalize_for_match(entry["model_name"])
        norm_name_nospaces = norm_name.replace(" ", "")

        if norm_id == norm_name:
            return entry

        # Substring match: require both sides to be at least 6 chars to avoid
        # false positives like "text" matching "titan embed text"
        if len(norm_id_nospaces) >= 6 and len(norm_name_nospaces) >= 6:
            if norm_id_nospaces in norm_name_nospaces or norm_name_nospaces in norm_id_nospaces:
                score = len(norm_name_nospaces)
                if score > best_score:
                    best_score = score
                    best_match = entry
                continue

        id_tokens = set(norm_id.split())
        name_tokens = set(norm_name.split())
        # Exclude pure numeric tokens from matching (e.g., "1", "2") to avoid
        # false positives like "pegasus 1 2" matching "claude 2 1"
        id_meaningful = {t for t in id_tokens if not t.isdigit()}
        name_meaningful = {t for t in name_tokens if not t.isdigit()}
        overlap = len(id_meaningful & name_meaningful)
        # Require at least 2 meaningful token matches AND >50% overlap
        min_tokens = min(len(id_meaningful), len(name_meaningful))
        if overlap >= 2 and min_tokens > 0 and overlap / min_tokens > 0.5 and overlap > best_score:
            best_score = overlap
            best_match = entry

    return best_match


def _fetch_bulk_json(service_path: str) -> Optional[dict]:
    """Fetch the bulk pricing JSON for a service."""
    svc = _SERVICE_PATH_MAP.get(service_path)
    if not svc:
        logger.warning("Unknown service path for bulk JSON: %s", service_path)
        return None

    url = _BULK_JSON_URL.format(svc=svc)
    try:
        data = _fetch_url(url, timeout=20)
        return json.loads(data)
    except Exception as exc:
        logger.warning("Failed to fetch bulk pricing JSON from %s: %s", url, exc)
        return None


def _scrape_webpage_pricing(model_id: str, region: str) -> dict:
    """Tier 3: Scrape Amazon Bedrock pricing webpage for a single model.

    Returns {"input_cost": float|None, "output_cost": float|None}.
    Costs are returned in per-1K-token format.
    """
    try:
        page_bytes = _fetch_url(_PRICING_PAGE_URL, timeout=20)
        page_html = page_bytes.decode("utf-8", errors="replace")
        entries = _parse_pricing_page(page_html)
        bulk_cache = {}
        return _resolve_single_from_webpage(model_id, region, entries, bulk_cache)
    except Exception as exc:
        logger.warning("Tier 3 webpage pricing scrape failed: %s", exc)
        return {"input_cost": None, "output_cost": None}


def _resolve_single_from_webpage(
    model_id: str, region: str, entries: list[dict], bulk_cache: dict
) -> dict:
    """Resolve pricing for a single model from pre-fetched webpage data.

    Args:
        model_id: Stripped model ID.
        region: AWS region.
        entries: Parsed pricing page entries (from _parse_pricing_page).
        bulk_cache: Dict to cache bulk JSON data by service_path (shared across calls).

    Returns {"input_cost": float|None, "output_cost": float|None}.
    """
    if not entries:
        return {"input_cost": None, "output_cost": None}

    matched = _find_model_entry(model_id, entries)
    if not matched:
        logger.info("Tier 3: No matching model found for %s", model_id)
        return {"input_cost": None, "output_cost": None}

    logger.info(
        "Tier 3: Matched %s -> '%s' (service=%s)",
        model_id, matched["model_name"], matched["service_path"],
    )

    # Use cached bulk JSON or fetch it
    sp = matched["service_path"]
    if sp not in bulk_cache:
        bulk_cache[sp] = _fetch_bulk_json(sp)
    bulk_data = bulk_cache[sp]

    if not bulk_data:
        return {"input_cost": None, "output_cost": None}

    region_name = _REGION_DISPLAY_NAMES.get(region)
    if not region_name:
        logger.warning("Tier 3: Unknown region code %s", region)
        return {"input_cost": None, "output_cost": None}

    region_prices = bulk_data.get("regions", {}).get(region_name, {})
    if not region_prices:
        logger.info("Tier 3: No pricing data for region %s", region_name)
        return {"input_cost": None, "output_cost": None}

    input_entry = region_prices.get(matched["input_hash"], {})
    output_entry = region_prices.get(matched["output_hash"], {})

    input_raw = float(input_entry["price"]) if input_entry.get("price") else None
    output_raw = float(output_entry["price"]) if output_entry.get("price") else None

    # Convert to per-1K-token format:
    #   {priceOf!svc!hash!*!1000} -> raw is per 1K tokens
    #   {priceOf!svc!hash}        -> raw is per 1M tokens, divide by 1000
    input_cost = None
    output_cost = None

    if input_raw is not None:
        input_cost = input_raw if matched["input_mult"] >= 1000 else input_raw / 1000.0

    if output_raw is not None:
        output_cost = output_raw if matched["output_mult"] >= 1000 else output_raw / 1000.0

    return {"input_cost": input_cost, "output_cost": output_cost}


def _scrape_webpage_pricing_bulk(models: list[tuple[str, str]]) -> dict[str, dict]:
    """Tier 3 bulk: Scrape webpage once and resolve all models.

    Fetches the pricing page HTML once, parses it, then resolves all models
    sharing the cached HTML and bulk JSON data.

    Args:
        models: List of (stripped_model_id, region) tuples.

    Returns:
        Dict keyed by "region#model_id" with {input_cost, output_cost} values.
    """
    results = {}

    try:
        page_bytes = _fetch_url(_PRICING_PAGE_URL, timeout=20)
        page_html = page_bytes.decode("utf-8", errors="replace")
        entries = _parse_pricing_page(page_html)

        if not entries:
            logger.warning("Tier 3 bulk: No model entries parsed from pricing page")
            return {f"{reg}#{mid}": {"input_cost": None, "output_cost": None} for mid, reg in models}

        logger.info("Tier 3 bulk: Parsed %d entries from pricing page", len(entries))

        # Shared cache for bulk JSON files (avoids re-fetching per service_path)
        bulk_cache = {}

        for model_id, region in models:
            key = f"{region}#{model_id}"
            results[key] = _resolve_single_from_webpage(model_id, region, entries, bulk_cache)

    except Exception as exc:
        logger.warning("Tier 3 bulk webpage scrape failed: %s", exc)
        for model_id, region in models:
            key = f"{region}#{model_id}"
            if key not in results:
                results[key] = {"input_cost": None, "output_cost": None}

    return results


# ---------------------------------------------------------------------------
# Main pricing resolver
# ---------------------------------------------------------------------------

def get_model_pricing(model_id: str, region: str) -> dict:
    """Get pricing for a model using 3-tier fallback.

    Args:
        model_id: Bedrock model ID without 'bedrock/' prefix
                  (e.g., 'anthropic.claude-sonnet-4-5-20250929-v1:0')
        region: AWS region (e.g., 'us-west-2')

    Returns:
        dict with keys: input_cost, output_cost, pricing_source
        Costs are per 1K tokens.
    """
    # Tier 1: Query Price List API (all 3 service codes)
    result = _query_price_list_api(model_id, region)

    if result["input_cost"] is not None and result["output_cost"] is not None:
        return {
            "input_cost": result["input_cost"],
            "output_cost": result["output_cost"],
            "pricing_source": "api",
        }

    # Tier 2: Re-query AmazonAmazon BedrockService for missing pricing
    if result["input_cost"] is not None or result["output_cost"] is not None:
        tier2 = _query_bedrock_service_only(model_id, region)
        if result["input_cost"] is None and tier2["input_cost"] is not None:
            result["input_cost"] = tier2["input_cost"]
        if result["output_cost"] is None and tier2["output_cost"] is not None:
            result["output_cost"] = tier2["output_cost"]

        if result["input_cost"] is not None and result["output_cost"] is not None:
            return {
                "input_cost": result["input_cost"],
                "output_cost": result["output_cost"],
                "pricing_source": "api_partial",
            }

    # Tier 3: Scrape pricing from AWS webpage bulk JSON
    tier3 = _scrape_webpage_pricing(model_id, region)
    if result["input_cost"] is None and tier3["input_cost"] is not None:
        result["input_cost"] = tier3["input_cost"]
    if result["output_cost"] is None and tier3["output_cost"] is not None:
        result["output_cost"] = tier3["output_cost"]

    if result["input_cost"] is not None and result["output_cost"] is not None:
        return {
            "input_cost": result["input_cost"],
            "output_cost": result["output_cost"],
            "pricing_source": "webpage",
        }

    return {
        "input_cost": result["input_cost"],
        "output_cost": result["output_cost"],
        "pricing_source": "unavailable",
    }


# ---------------------------------------------------------------------------
# Bulk resolution and enrichment (public API)
# ---------------------------------------------------------------------------

def _load_bedrock_entries_from_profile(path: Path) -> list[dict]:
    """Load Amazon Bedrock model entries from a JSONL profile file."""
    entries = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    data = json.loads(line)
                    model_id = data.get("model_id", "")
                    region = data.get("region", "")
                    if model_id.startswith("bedrock/") and region:
                        entries.append({"model_id": model_id, "region": region})
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        logger.warning("Profile file not found: %s", path)
    return entries


def resolve_all_pricing(force: bool = False) -> dict:
    """Resolve pricing for all Amazon Bedrock models in both profile files.

    Uses local file cache with hash-based invalidation. Only re-resolves
    when models_profiles.jsonl or judge_profiles.jsonl content changes.

    Args:
        force: If True, ignore cache and re-resolve all pricing.

    Returns:
        The pricing cache dict.
    """
    if not force and is_pricing_cache_valid():
        logger.info("Pricing cache is valid, using cached pricing")
        return load_pricing_cache()

    logger.info("Resolving Amazon Bedrock model pricing...")

    cache = {
        "last_updated": None,
        "profiles_hash": get_profiles_hash(),
        "pricing": {},
    }

    # Collect unique (model_id, region) pairs from both profiles
    entries = _load_bedrock_entries_from_profile(MODELS_PROFILE_PATH)
    entries.extend(_load_bedrock_entries_from_profile(JUDGE_PROFILE_PATH))

    seen = set()
    unique_entries = []
    for e in entries:
        key = _cache_key(e["model_id"], e["region"])
        if key not in seen:
            seen.add(key)
            unique_entries.append(e)

    logger.info("Resolving pricing for %d unique Amazon Bedrock model+region combinations", len(unique_entries))

    # Build list of (stripped_model_id, region) for bulk lookup
    model_region_pairs = []
    key_map = {}  # maps "region#stripped_id" -> cache_key
    for entry in unique_entries:
        stripped_id = strip_model_id_for_pricing(entry["model_id"])
        region = entry["region"]
        cache_key = _cache_key(entry["model_id"], region)
        bulk_key = f"{region}#{stripped_id}"
        model_region_pairs.append((stripped_id, region))
        key_map[bulk_key] = cache_key

    # Tier 1: Fetch all products in parallel, match all models at once
    logger.info("Tier 1: Fetching all products from Price List API (parallel)...")
    try:
        all_products = _fetch_all_products_parallel()
        api_results = _resolve_bulk_from_products(all_products, model_region_pairs)
    except Exception as exc:
        logger.warning("Bulk API fetch failed: %s", exc)
        api_results = {}

    # Tier 3: For models not fully resolved by API, try webpage scrape
    missing = [
        (mid, reg) for mid, reg in model_region_pairs
        if api_results.get(f"{reg}#{mid}", {}).get("input_cost") is None
        or api_results.get(f"{reg}#{mid}", {}).get("output_cost") is None
    ]

    if missing:
        logger.info("Tier 3: Scraping webpage pricing for %d remaining models (single fetch)...", len(missing))
        tier3_results = _scrape_webpage_pricing_bulk(missing)

        for stripped_id, region in missing:
            bulk_key = f"{region}#{stripped_id}"
            api_r = api_results.get(bulk_key, {"input_cost": None, "output_cost": None})
            tier3 = tier3_results.get(bulk_key, {"input_cost": None, "output_cost": None})

            if api_r["input_cost"] is None and tier3["input_cost"] is not None:
                api_r["input_cost"] = tier3["input_cost"]
            if api_r["output_cost"] is None and tier3["output_cost"] is not None:
                api_r["output_cost"] = tier3["output_cost"]
            api_results[bulk_key] = api_r

    # Build cache from results
    for stripped_id, region in model_region_pairs:
        bulk_key = f"{region}#{stripped_id}"
        cache_key = key_map[bulk_key]
        r = api_results.get(bulk_key, {"input_cost": None, "output_cost": None})

        inp = r["input_cost"]
        out = r["output_cost"]
        if inp is not None and out is not None:
            source = "api" if bulk_key not in {f"{reg}#{mid}" for mid, reg in missing} else "webpage"
        else:
            source = "unavailable"

        cache["pricing"][cache_key] = {
            "input_cost_per_1k": inp,
            "output_cost_per_1k": out,
            "pricing_source": source,
            "last_resolved": datetime.now(timezone.utc).isoformat(),
        }

        if source == "unavailable":
            logger.warning("Pricing unavailable for %s @ %s", stripped_id, region)
        else:
            logger.info("  %s @ %s -> input: %s, output: %s (%s)", stripped_id, region, inp, out, source)

    save_pricing_cache(cache)
    return cache


def enrich_model_entry(entry: dict) -> dict:
    """Overlay dynamic pricing onto a model/judge entry from JSONL.

    For Amazon Bedrock models, looks up cached pricing and replaces cost fields.
    Falls back to original JSONL values if API pricing is unavailable.
    Non-Bedrock models are returned unchanged.

    Handles both key formats:
        - Models: input_token_cost / output_token_cost
        - Judges: input_cost_per_1k / output_cost_per_1k

    Args:
        entry: A dict loaded from models_profiles.jsonl or judge_profiles.jsonl.

    Returns:
        The entry dict with pricing potentially updated.
    """
    model_id = entry.get("model_id", "")
    region = entry.get("region", "")

    if not model_id.startswith("bedrock/") or not region:
        return entry

    cache = load_pricing_cache()
    key = _cache_key(model_id, region)
    cached = cache.get("pricing", {}).get(key)

    if not cached or cached.get("pricing_source") == "unavailable":
        if cached:
            logger.warning(
                "Dynamic pricing unavailable for %s @ %s, using static JSONL values",
                model_id, region,
            )
        return entry

    input_price = cached.get("input_cost_per_1k")
    output_price = cached.get("output_cost_per_1k")

    # Update model profile keys
    if "input_token_cost" in entry and input_price is not None:
        entry["input_token_cost"] = input_price
    if "output_token_cost" in entry and output_price is not None:
        entry["output_token_cost"] = output_price

    # Update judge profile keys
    if "input_cost_per_1k" in entry and input_price is not None:
        entry["input_cost_per_1k"] = input_price
    if "output_cost_per_1k" in entry and output_price is not None:
        entry["output_cost_per_1k"] = output_price

    return entry


# ---------------------------------------------------------------------------
# Profile generation from Amazon Bedrock APIs
# ---------------------------------------------------------------------------

def fetch_foundation_models(region: str = "us-east-1") -> tuple[dict, dict]:
    """Fetch all foundation models and build lookup maps.

    Returns:
        (name_to_info, id_to_name) where:
        - name_to_info: modelName -> {modelId, inferenceTypes, provider, ...}
        - id_to_name: modelId -> modelName
    """
    client = boto3.client("bedrock", region_name=region)
    resp = client.list_foundation_models()
    models = resp.get("modelSummaries", [])

    name_to_info = {}
    id_to_name = {}

    for m in models:
        name = m["modelName"]
        mid = m["modelId"]
        inference_types = m.get("inferenceTypesSupported", [])

        # Skip provisioned-only variants (e.g., :256k, :300k suffixes)
        if re.search(r":\d+k$", mid) or re.search(r":mm$", mid):
            continue
        if inference_types == ["PROVISIONED"]:
            continue

        # If duplicate name, prefer ON_DEMAND or INFERENCE_PROFILE over PROVISIONED
        if name in name_to_info:
            existing = name_to_info[name]
            if "PROVISIONED" not in inference_types:
                pass  # overwrite with better variant
            else:
                continue  # keep existing non-provisioned variant

        name_to_info[name] = {
            "modelId": mid,
            "inferenceTypes": inference_types,
            "provider": m.get("providerName", ""),
            "inputModalities": m.get("inputModalities", []),
            "outputModalities": m.get("outputModalities", []),
        }
        id_to_name[mid] = name

    return name_to_info, id_to_name


def fetch_inference_profiles(region: str = "us-east-1") -> dict:
    """Fetch inference profiles and build cross-region prefix map.

    Returns:
        Dict mapping base_model_id -> list of cross-region profile IDs.
        E.g., anthropic.claude-sonnet-4-5-v1:0 -> [us.anthropic..., global.anthropic...]
    """
    client = boto3.client("bedrock", region_name=region)

    profiles = []
    resp = client.list_inference_profiles(maxResults=100)
    profiles.extend(resp.get("inferenceProfileSummaries", []))
    while resp.get("nextToken"):
        resp = client.list_inference_profiles(maxResults=100, nextToken=resp["nextToken"])
        profiles.extend(resp.get("inferenceProfileSummaries", []))

    cross_region_map = defaultdict(list)
    for p in profiles:
        pid = p["inferenceProfileId"]
        base_models = [m.get("modelArn", "").split("/")[-1] for m in p.get("models", [])]
        for base in set(base_models):
            if base:
                cross_region_map[base].append(pid)

    return cross_region_map


def _extract_model_key_from_usagetype(usagetype: str) -> str:
    """Extract model identifier from usagetype for matching.

    E.g., USE1-moonshotai.kimi-k2-thinking-mantle-input-tokens-standard
        -> moonshotai.kimi-k2-thinking
    """
    stripped = re.sub(r"^[A-Z]{2,4}\d?-", "", usagetype)
    match = re.match(r"(.+?)-(input|output)-(tokens?|token-count|video-token)", stripped)
    if not match:
        return ""
    model_part = match.group(1)
    model_part = re.sub(r"-mantle$", "", model_part)
    return model_part


def _normalize_name_for_generation(s: str) -> str:
    """Normalize a name for fuzzy matching during profile generation."""
    s = s.lower()
    # Strip minor version suffixes like ".0" (e.g., "2.0" -> "2", "4.7" stays)
    # Only strip ".0" specifically, not meaningful decimals
    s = re.sub(r"\.0(?=\b|[^0-9])", "", s)
    s = re.sub(r"[-._:\s]+", "", s)
    return s


def _tokenize_for_matching(s: str) -> set[str]:
    """Split a normalized name into meaningful tokens for set matching.

    Splits on transitions between letters and digits to handle cases like
    'claude4sonnet' -> {'claude', '4', 'sonnet'} and
    'claudesonnet4' -> {'claude', 'sonnet', '4'}.
    """
    # Insert separator at letter-digit and digit-letter boundaries
    s = re.sub(r"([a-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([a-z])", r"\1 \2", s)
    # Also split CamelCase-like runs of lowercase that were originally separate words
    # by splitting at any point where a common model keyword starts
    keywords = ["claude", "sonnet", "opus", "haiku", "nova", "lite", "pro", "micro",
                "premier", "llama", "mistral", "qwen", "deepseek", "gemma", "kimi",
                "minimax", "nemotron", "sonic", "canvas", "reel"]
    for kw in keywords:
        s = s.replace(kw, f" {kw} ")
    return set(s.split()) - {""}


def _match_usagetype_to_model(
    usage_model_key: str, name_matcher: dict, name_to_info: dict
) -> Optional[str]:
    """Try to match a usagetype model key to a foundation model.

    Returns the modelId if matched, None otherwise.
    """
    norm_key = _normalize_name_for_generation(usage_model_key)

    # Direct match
    if norm_key in name_matcher:
        name = name_matcher[norm_key]
        return name_to_info[name]["modelId"]

    # Substring match (either direction)
    for norm_name, name in name_matcher.items():
        if norm_key in norm_name or norm_name in norm_key:
            return name_to_info[name]["modelId"]

    # Token-set match: handles word reordering (e.g., 'claude4sonnet' vs 'claudesonnet4')
    key_tokens = _tokenize_for_matching(norm_key)
    if len(key_tokens) >= 2:
        best_match = None
        best_overlap = 0
        for norm_name, name in name_matcher.items():
            name_tokens = _tokenize_for_matching(norm_name)
            overlap = len(key_tokens & name_tokens)
            # Require all key tokens to be in the name tokens (or vice versa)
            if key_tokens == name_tokens:
                return name_to_info[name]["modelId"]
            if key_tokens.issubset(name_tokens) or name_tokens.issubset(key_tokens):
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = name
        if best_match:
            return name_to_info[best_match]["modelId"]

    return None


def _get_preferred_cross_region_id(model_id: str, cross_region_map: dict) -> str:
    """Get the preferred cross-region model ID.

    Prefers us. prefix, falls back to global., then base model_id.
    """
    profiles = cross_region_map.get(model_id, [])
    if not profiles:
        return model_id

    for p in profiles:
        if p.startswith("us."):
            return p

    return profiles[0]


def _fetch_supported_models_page() -> list[dict]:
    """Fetch model catalog from Amazon Bedrock supported models page.

    Parses https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
    to get the definitive list of models, their IDs, and region availability.

    Returns:
        List of dicts with keys: provider, model_name, model_id,
        single_regions, cross_regions, input_modalities, output_modalities
    """
    url = "https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html"
    req = urllib.request.Request(url, headers={"User-Agent": "BedrockBenchmark/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    tables = re.findall(r"<table[^>]*>(.*?)</table>", html, re.DOTALL)
    if not tables:
        logger.warning("No tables found on models-supported page")
        return []

    models = []
    for table in tables:
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.DOTALL)
        headers = re.findall(r"<th[^>]*>(.*?)</th>", rows[0], re.DOTALL) if rows else []
        headers = [re.sub(r"<[^>]+>", "", h).strip().lower() for h in headers]

        if "model id" not in headers:
            continue

        for row in rows[1:]:
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
            cells_text = [re.sub(r"<[^>]+>", " ", c).strip() for c in cells]

            if len(cells_text) < 3:
                continue

            model_id = cells_text[headers.index("model id")].strip() if "model id" in headers else ""
            if not model_id:
                continue

            provider = cells_text[headers.index("provider")].strip() if "provider" in headers else ""
            model_name = cells_text[headers.index("model")].strip() if "model" in headers else ""

            # Extract regions
            single_regions = []
            cross_regions = []
            if "single-region model support" in headers:
                idx = headers.index("single-region model support")
                if idx < len(cells_text):
                    single_regions = re.findall(r"[a-z]{2}-[a-z]+-\d+", cells_text[idx])
            if "cross-region inference profile support" in headers:
                idx = headers.index("cross-region inference profile support")
                if idx < len(cells_text):
                    cross_regions = re.findall(r"[a-z]{2}-[a-z]+-\d+", cells_text[idx])

            # Extract modalities
            input_mod = ""
            output_mod = ""
            if "input modalities" in headers:
                idx = headers.index("input modalities")
                if idx < len(cells_text):
                    input_mod = cells_text[idx].strip()
            if "output modalities" in headers:
                idx = headers.index("output modalities")
                if idx < len(cells_text):
                    output_mod = cells_text[idx].strip()

            models.append({
                "provider": provider,
                "model_name": model_name,
                "model_id": model_id,
                "single_regions": single_regions,
                "cross_regions": cross_regions,
                "input_modalities": input_mod,
                "output_modalities": output_mod,
            })

    logger.info("Parsed %d models from supported models page", len(models))
    return models


def _find_pricing_for_region(
    model_id: str, target_region: str, pricing: dict
) -> Optional[dict]:
    """Find pricing for a model in a region with geo-prefix fallback.

    Lookup order:
    1. Exact (model_id, target_region) match
    2. Same geo-prefix region (e.g., ca-central-1 falls back to ca-west-1)
    3. us-west-2 as global fallback

    Returns {"input": float, "output": float} or None.
    """
    # 1. Exact match
    key = (model_id, target_region)
    if key in pricing and pricing[key]["input"] is not None and pricing[key]["output"] is not None:
        return pricing[key]

    # 2. Same geo-prefix fallback (e.g., ca- for ca-central-1)
    geo_prefix = target_region.split("-")[0] + "-"
    for (mid, reg), costs in pricing.items():
        if mid == model_id and reg.startswith(geo_prefix) and costs["input"] is not None and costs["output"] is not None:
            return costs

    # 3. us-west-2 fallback
    fallback_key = (model_id, "us-west-2")
    if fallback_key in pricing and pricing[fallback_key]["input"] is not None and pricing[fallback_key]["output"] is not None:
        return pricing[fallback_key]

    # 4. Any region with pricing
    for (mid, reg), costs in pricing.items():
        if mid == model_id and costs["input"] is not None and costs["output"] is not None:
            return costs

    return None


def generate_models_profiles() -> list[dict]:
    """Generate model profile entries from Amazon Bedrock APIs.

    Uses the AWS models-supported page as the authoritative source for which
    models exist and which regions they support. Uses the Price List API for
    pricing, with geo-prefix and us-west-2 fallback for missing regions.

    Returns:
        List of dicts suitable for writing to models_profiles.jsonl.
    """
    logger.info("Generating models_profiles from Bedrock APIs...")

    # Step 1: Fetch model catalog from models-supported page
    catalog = _fetch_supported_models_page()
    if not catalog:
        logger.error("Failed to fetch model catalog, falling back to ListFoundationModels")
        catalog = []

    # Step 2: Inference profiles for cross-region IDs
    cross_region_map = fetch_inference_profiles()
    logger.info("Fetched %d cross-region profile mappings", len(cross_region_map))

    # Step 3: Fetch pricing from Price List API
    name_to_info, id_to_name = fetch_foundation_models()
    name_matcher = {_normalize_name_for_generation(n): n for n in name_to_info}
    products = _fetch_all_products_parallel()

    # Build pricing and tier maps from Price List API
    pricing = defaultdict(lambda: {"input": None, "output": None})
    service_tiers = defaultdict(set)

    for p in products:
        attrs = p.get("product", {}).get("attributes", {})
        region = attrs.get("regionCode", "")
        usagetype = attrs.get("usagetype", "").lower()
        if not region or not usagetype:
            continue

        usage_model_key = _extract_model_key_from_usagetype(usagetype)
        if not usage_model_key:
            continue

        model_id = _match_usagetype_to_model(usage_model_key, name_matcher, name_to_info)
        if not model_id:
            continue

        key = (model_id, region)

        # Extract service tier
        if "-priority" in usagetype:
            service_tiers[key].add("priority")
        elif "-flex" in usagetype:
            service_tiers[key].add("flex")
        elif "-standard" in usagetype:
            service_tiers[key].add("standard")
        else:
            service_tiers[key].add("default")

        # Extract on-demand standard pricing
        entry = extract_pricing(p)
        if not is_on_demand_standard(entry):
            continue

        token_type = classify_token_type(entry)
        price = entry["price_per_unit_usd"]
        if price is None or token_type == "unknown":
            continue

        if pricing[key][token_type] is None:
            pricing[key][token_type] = float(price)

    # Step 4: Webpage scrape fallback for models with no API pricing at all
    models_with_pricing = set(mid for (mid, _) in pricing.keys() if pricing[(mid, _)]["input"] is not None)
    models_needing_scrape = []
    for entry in catalog:
        mid = entry["model_id"]
        if mid not in models_with_pricing and not re.search(r":\d+k$", mid):
            models_needing_scrape.append(mid)

    if models_needing_scrape:
        logger.info("Tier 3: %d models without API pricing, trying webpage scrape...", len(models_needing_scrape))
        scrape_pairs = [(strip_model_id_for_pricing(f"bedrock/{mid}"), "us-east-1") for mid in models_needing_scrape]
        scrape_results = _scrape_webpage_pricing_bulk(scrape_pairs)
        for mid, (stripped, _) in zip(models_needing_scrape, scrape_pairs):
            result = scrape_results.get(f"us-east-1#{stripped}", {})
            if result.get("input_cost") is not None and result.get("output_cost") is not None:
                pricing[(mid, "us-east-1")]["input"] = result["input_cost"]
                pricing[(mid, "us-east-1")]["output"] = result["output_cost"]
                if (mid, "us-east-1") not in service_tiers:
                    service_tiers[(mid, "us-east-1")].add("default")

    # Step 5: Build JSONL entries from catalog + pricing
    entries = []
    for model in catalog:
        model_id = model["model_id"]

        # Skip provisioned-only variants
        if re.search(r":\d+k$", model_id) or re.search(r":mm$", model_id):
            continue

        # Only include models with Text output (skip image, video, embedding, reranking)
        if model.get("output_modalities") and "Text" not in model["output_modalities"]:
            continue

        # Determine cross-region model ID
        full_model_id = _get_preferred_cross_region_id(model_id, cross_region_map)

        # All regions where this model is available
        all_regions = list(set(model["single_regions"] + model["cross_regions"]))
        if not all_regions:
            # Model has no region info on the page (Table 2 models) — use us-east-1
            all_regions = ["us-east-1"]

        for region in sorted(all_regions):
            # Find pricing with geo-prefix fallback
            costs = _find_pricing_for_region(model_id, region, pricing)
            if not costs:
                continue

            # Build tier list
            tiers = sorted(service_tiers.get((model_id, region), {"default"}))
            if "default" not in tiers:
                tiers = ["default"] + tiers

            entries.append({
                "model_id": f"bedrock/{full_model_id}",
                "region": region,
                "input_token_cost": costs["input"],
                "output_token_cost": costs["output"],
                "service_tiers": tiers,
            })

    logger.info("Generated %d model+region entries", len(entries))
    return entries


def _is_pricing_stale() -> bool:
    """Check if pricing is older than PRICING_REFRESH_DAYS."""
    cache = load_pricing_cache()
    last_updated = cache.get("last_updated")
    if not last_updated:
        return True
    try:
        updated_dt = datetime.fromisoformat(last_updated)
        age = datetime.now(timezone.utc) - updated_dt
        return age.days >= PRICING_REFRESH_DAYS
    except (ValueError, TypeError):
        return True


def ensure_models_profiles(models_path: Optional[Path] = None) -> Path:
    """helps models_profiles.jsonl exists and pricing is fresh.

    - If file doesn't exist: generate from Bedrock APIs
    - If file exists but pricing is >7 days old: refresh Amazon Bedrock model pricing
    - Non-Bedrock entries (openai/, gemini/) are preserved unchanged during refresh

    Args:
        models_path: Path to models_profiles.jsonl. Defaults to config/models_profiles.jsonl.

    Returns:
        Path to the models_profiles.jsonl file.
    """
    if models_path is None:
        models_path = MODELS_PROFILE_PATH

    if not models_path.exists():
        print("models_profiles.jsonl not found. Generating model catalog from AWS...")
        print("This may take 20-30 seconds (fetching models, regions, and pricing)...")
        logger.info("models_profiles.jsonl not found, generating from Bedrock official sources...")
        try:
            entries = generate_models_profiles()
            if entries:
                models_path.parent.mkdir(parents=True, exist_ok=True)
                with open(models_path, "w", encoding="utf-8") as f:
                    for e in entries:
                        f.write(json.dumps(e) + "\n")
                print(f"Generated {len(entries)} model entries across {len(set(e['region'] for e in entries))} regions.")
                logger.info("Generated %s with %d entries", models_path, len(entries))
            else:
                logger.warning("No entries generated from Bedrock APIs")
        except Exception as exc:
            logger.error("Failed to generate models_profiles.jsonl: %s", exc)
        return models_path

    # File exists — check if refresh is needed
    if not _is_pricing_stale():
        logger.info("Pricing is fresh (<%d days old), no refresh needed", PRICING_REFRESH_DAYS)
        return models_path

    print(f"Pricing is over {PRICING_REFRESH_DAYS} days old. Refreshing Amazon Bedrock model pricing...")
    print("This may take 20-30 seconds...")
    logger.info("Pricing is stale (>%d days), refreshing Amazon Bedrock model pricing...", PRICING_REFRESH_DAYS)
    try:
        # Read existing entries, preserve non-Amazon Bedrock
        existing = []
        with open(models_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))

        non_bedrock = [e for e in existing if not e.get("model_id", "").startswith("bedrock/")]

        # Re-generate Amazon Bedrock entries from APIs
        bedrock_entries = generate_models_profiles()

        # Combine: fresh Amazon Bedrock + preserved non-Amazon Bedrock
        all_entries = bedrock_entries + non_bedrock

        with open(models_path, "w", encoding="utf-8") as f:
            for e in all_entries:
                f.write(json.dumps(e) + "\n")

        logger.info(
            "Refreshed %s: %d Amazon Bedrock + %d non-Bedrock entries",
            models_path, len(bedrock_entries), len(non_bedrock),
        )
    except Exception as exc:
        logger.error("Failed to refresh models_profiles.jsonl: %s", exc)

    return models_path
