#!/usr/bin/env python3
"""Standalone Bedrock model catalog generator (live, multi-region).

Generates config/models_profiles.jsonl from a live API sweep — NOT the (lagging) AWS
docs page — so it always reflects the latest models and their real region availability.

    generate_models_profiles(regions, include_mantle, include_no_pricing)
      1. discover_across_regions()      list_foundation_models + list_inference_profiles
                                        + Mantle /v1/models, swept across all regions
      2. build_modelid_pricing()        Converse pricing (usagetype -> model-name match)
         build_price_index()            token-keyed pricing for Mantle-only models
         _scrape_webpage_pricing_bulk() webpage fallback for unmatched models
      3. classify endpoint + build:
         - Foundation models  -> Converse (bare id / cross-region profile id)
         - Mantle-only models -> bedrock_mantle, gated by a Responses-API probe
           (a Mantle listing does NOT imply invocability); the prior catalog is the
           anchor of record for routing, so production models never regress.

Pricing resolution order: manual override -> Price List (name match / token index, with
geo + us-east-1 fallback) -> webpage scrape. Models that still lack a price are emitted
with $0 + pricing_source="MISSING" and listed in the run summary (never silently dropped).

CLI: python generate_models_profiles.py [--regions r1,r2] [--no-mantle]
                                         [--drop-no-pricing] [--dry-run] [--output PATH]

Verifying Mantle endpoints needs a Bedrock bearer token: pip install
aws-bedrock-token-generator (without it, Mantle models are included unverified + warned).
"""

import argparse
import gzip
import html as html_module
import json
import logging
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import boto3

# Reuse the project's Mantle helpers (OpenAI-compatible surface discovery + the
# /v1/responses capability probe used for endpoint classification).
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
try:
    from mantle_client import list_mantle_models  # noqa: E402
    _MANTLE_AVAILABLE = True
except Exception as _e:  # pragma: no cover - mantle helpers optional
    _MANTLE_AVAILABLE = False
    _MANTLE_IMPORT_ERROR = _e

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = PROJECT_ROOT / "config" / "models_profiles.jsonl"

SERVICE_CODES = [
    "AmazonBedrock",
    "AmazonBedrockService",
    "AmazonBedrockFoundationModels",
]
PRICING_CLIENT_REGION = "us-east-1"

# AWS Bedrock pricing page and bulk JSON endpoints
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
# Model ID helpers
# ---------------------------------------------------------------------------

def strip_model_id_for_pricing(model_id: str) -> str:
    """Strip 'bedrock/' prefix and cross-region prefixes for API lookup."""
    cleaned = model_id
    for prefix in ("bedrock/converse/", "bedrock/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    cleaned = re.sub(r"^[a-z]{2}\.", "", cleaned)
    return cleaned


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
    price_unit = ""
    for offer in terms.get("OnDemand", {}).values():
        for dim in offer.get("priceDimensions", {}).values():
            price_per_unit = dim.get("pricePerUnit", {}).get("USD")
            price_unit = dim.get("unit", "")
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
        "price_unit": price_unit,
    }


def _normalize_to_per_1m(price: float, unit: str) -> float:
    """Convert a price to per-1M-token format based on the unit string."""
    unit_lower = unit.lower()
    if "1k" in unit_lower or "thousand" in unit_lower:
        return round(price * 1000, 2)
    if "1m" in unit_lower or "million" in unit_lower or not unit:
        return round(price, 2)
    logger.warning("Unrecognized pricing unit '%s', assuming per-1M tokens", unit)
    return round(price, 2)


def _is_on_demand_tier(entry: dict) -> bool:
    """Filter to on-demand entries including tier variants (flex/priority/standard)."""
    usagetype = entry["usagetype"].lower()
    feature = entry["feature"].lower()
    feature_type = entry["feature_type"].lower()

    for skip in ("reserved", "batch", "cache", "long-context", "latency-optimized"):
        if skip in usagetype or skip in feature or skip in feature_type:
            return False
    return True


def _detect_tier(usagetype: str) -> str:
    """Detect service tier from usagetype string."""
    usagetype = usagetype.lower()
    if "-priority" in usagetype:
        return "priority"
    if "-flex" in usagetype:
        return "flex"
    return "default"


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


# ---------------------------------------------------------------------------
# Step 1: Fetch model catalog from AWS docs
# ---------------------------------------------------------------------------

def _fetch_supported_models_page() -> list[dict]:
    """Fetch model catalog from AWS Bedrock supported models page.

    Parses https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
    to get the definitive list of models, their IDs, and region availability.
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


# ---------------------------------------------------------------------------
# Step 2: Fetch cross-region inference profiles
# ---------------------------------------------------------------------------

def fetch_inference_profiles(region: str = "us-east-1") -> dict:
    """Fetch inference profiles and build cross-region prefix map.

    Returns:
        Dict mapping base_model_id -> list of cross-region profile IDs.
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


def _get_preferred_cross_region_id(model_id: str, cross_region_map: dict) -> str:
    """Get the preferred cross-region model ID (prefers us. prefix)."""
    profiles = cross_region_map.get(model_id, [])
    if not profiles:
        return model_id

    for p in profiles:
        if p.startswith("us."):
            return p

    return profiles[0]


# ---------------------------------------------------------------------------
# Step 3: Fetch pricing from Price List API
# ---------------------------------------------------------------------------

def fetch_foundation_models(region: str = "us-east-1") -> tuple[dict, dict]:
    """Fetch all foundation models and build lookup maps.

    Returns:
        (name_to_info, id_to_name)
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

        if re.search(r":\d+k$", mid) or re.search(r":mm$", mid):
            continue
        if inference_types == ["PROVISIONED"]:
            continue

        if name in name_to_info:
            if "PROVISIONED" not in inference_types:
                pass
            else:
                continue

        name_to_info[name] = {
            "modelId": mid,
            "inferenceTypes": inference_types,
            "provider": m.get("providerName", ""),
            "inputModalities": m.get("inputModalities", []),
            "outputModalities": m.get("outputModalities", []),
        }
        id_to_name[mid] = name

    return name_to_info, id_to_name


def _fetch_all_products_parallel() -> list:
    """Fetch all products from all 3 service codes in parallel."""
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


def _extract_model_key_from_usagetype(usagetype: str) -> str:
    """Extract model identifier from usagetype for matching."""
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
    s = re.sub(r"\.0(?=\b|[^0-9])", "", s)
    s = re.sub(r"[-._:\s]+", "", s)
    return s


def _tokenize_for_matching(s: str) -> set[str]:
    """Split a normalized name into meaningful tokens for set matching."""
    s = re.sub(r"([a-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([a-z])", r"\1 \2", s)
    keywords = ["claude", "sonnet", "opus", "haiku", "nova", "lite", "pro", "micro",
                "premier", "llama", "mistral", "qwen", "deepseek", "gemma", "kimi",
                "minimax", "nemotron", "sonic", "canvas", "reel"]
    for kw in keywords:
        s = s.replace(kw, f" {kw} ")
    return set(s.split()) - {""}


def _match_usagetype_to_model(
    usage_model_key: str, name_matcher: dict, name_to_info: dict
) -> Optional[str]:
    """Try to match a usagetype model key to a foundation model."""
    norm_key = _normalize_name_for_generation(usage_model_key)

    if norm_key in name_matcher:
        name = name_matcher[norm_key]
        return name_to_info[name]["modelId"]

    for norm_name, name in name_matcher.items():
        if norm_key in norm_name or norm_name in norm_key:
            return name_to_info[name]["modelId"]

    key_tokens = _tokenize_for_matching(norm_key)
    if len(key_tokens) >= 2:
        best_match = None
        best_overlap = 0
        for norm_name, name in name_matcher.items():
            name_tokens = _tokenize_for_matching(norm_name)
            overlap = len(key_tokens & name_tokens)
            if key_tokens == name_tokens:
                return name_to_info[name]["modelId"]
            if key_tokens.issubset(name_tokens) or name_tokens.issubset(key_tokens):
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = name
        if best_match:
            return name_to_info[best_match]["modelId"]

    return None


# ---------------------------------------------------------------------------
# Step 4: Webpage pricing scrape (fallback)
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
    """Parse the AWS Bedrock pricing page HTML to extract model-to-hash mappings."""
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

        if len(norm_id_nospaces) >= 6 and len(norm_name_nospaces) >= 6:
            if norm_id_nospaces in norm_name_nospaces or norm_name_nospaces in norm_id_nospaces:
                score = len(norm_name_nospaces)
                if score > best_score:
                    best_score = score
                    best_match = entry
                continue

        id_tokens = set(norm_id.split())
        name_tokens = set(norm_name.split())
        id_meaningful = {t for t in id_tokens if not t.isdigit()}
        name_meaningful = {t for t in name_tokens if not t.isdigit()}
        overlap = len(id_meaningful & name_meaningful)
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


def _resolve_single_from_webpage(
    model_id: str, region: str, entries: list[dict], bulk_cache: dict
) -> dict:
    """Resolve pricing for a single model from pre-fetched webpage data."""
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

    input_cost = None
    output_cost = None

    if input_raw is not None:
        input_cost = round(input_raw * 1000 if matched["input_mult"] >= 1000 else input_raw, 2)

    if output_raw is not None:
        output_cost = round(output_raw * 1000 if matched["output_mult"] >= 1000 else output_raw, 2)

    return {"input_cost": input_cost, "output_cost": output_cost}


def _scrape_webpage_pricing_bulk(models: list[tuple[str, str]]) -> dict[str, dict]:
    """Tier 3 bulk: Scrape webpage once and resolve all models."""
    results = {}

    try:
        page_bytes = _fetch_url(_PRICING_PAGE_URL, timeout=20)
        page_html = page_bytes.decode("utf-8", errors="replace")
        entries = _parse_pricing_page(page_html)

        if not entries:
            logger.warning("Tier 3 bulk: No model entries parsed from pricing page")
            return {f"{reg}#{mid}": {"input_cost": None, "output_cost": None} for mid, reg in models}

        logger.info("Tier 3 bulk: Parsed %d entries from pricing page", len(entries))

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
# Step 5: Region pricing lookup with fallback
# ---------------------------------------------------------------------------

def _find_pricing_for_region(
    model_id: str, target_region: str, pricing: dict
) -> Optional[dict]:
    """Find pricing for a model in a region with geo-prefix fallback.

    Lookup order:
    1. Exact (model_id, target_region) match
    2. Same geo-prefix region
    3. us-west-2 as global fallback
    4. Any region with pricing
    """
    key = (model_id, target_region)
    if key in pricing and pricing[key]["input"] is not None and pricing[key]["output"] is not None:
        return pricing[key]

    geo_prefix = target_region.split("-")[0] + "-"
    for (mid, reg), costs in pricing.items():
        if mid == model_id and reg.startswith(geo_prefix) and costs["input"] is not None and costs["output"] is not None:
            return costs

    fallback_key = (model_id, "us-west-2")
    if fallback_key in pricing and pricing[fallback_key]["input"] is not None and pricing[fallback_key]["output"] is not None:
        return pricing[fallback_key]

    for (mid, reg), costs in pricing.items():
        if mid == model_id and costs["input"] is not None and costs["output"] is not None:
            return costs

    return None


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Multi-region, API-driven discovery (replaces the docs-page scrape)
# ---------------------------------------------------------------------------

# Brand-new models lag in the Price List API (commercial rows are published days
# after launch). Until then, source pricing here ($/1M tokens, default tier). A live
# Price List row always takes precedence over these.
MANUAL_PRICE_OVERRIDES = {
    "anthropic.claude-opus-4-7": {"input": 5.0, "output": 25.0},
    "anthropic.claude-opus-4-8": {"input": 5.0, "output": 25.0},
    "zai.glm-5": {"input": 1.0, "output": 3.2},
    "openai.gpt-5.4": {"input": 2.5, "output": 15.0},
    "openai.gpt-5.5": {"input": 5.0, "output": 30.0},
}

# Dated snapshot aliases (openai.gpt-5.5-2026-04-23) — skip; the undated id is the
# rolling "latest" pointer the catalog should carry.
_DATED_SNAPSHOT_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")

# Non-text surfaces we never want in an LLM benchmarking catalog. Matched against the
# model id (covers embeddings, rerank, image, video, speech) for models where the API
# doesn't return clean modality info (e.g. Mantle-listed models).
_NON_TEXT_PATTERNS = re.compile(
    r"(embed|rerank|canvas|reel|image|video|stable|titan-image|sonic|"
    r"marengo|pegasus|ray|sd3|upscale|speech|tts|transcribe|vision)",
    re.I,
)


def _accessible_regions() -> list[str]:
    """All commercial Bedrock regions the account can reach (excludes GovCloud)."""
    try:
        ec2 = boto3.client("ec2", region_name="us-east-1")
        regions = [r["RegionName"] for r in ec2.describe_regions()["Regions"]]
        regions = [r for r in regions if not r.startswith("us-gov-")]
        if regions:
            return sorted(regions)
    except Exception as exc:  # pragma: no cover - fall back to the static list
        logger.warning("describe_regions failed (%s); using static region list", exc)
    return sorted(r for r in _REGION_DISPLAY_NAMES if not r.startswith("us-gov-"))


def discover_across_regions(regions: list[str], include_mantle: bool = True) -> dict:
    """Sweep Bedrock APIs across regions to discover the live model catalog.

    For each region (in parallel, errors isolated per region) this collects:
      - list_foundation_models   -> Converse-capable models + where they're available
      - list_inference_profiles  -> cross-region profile IDs + the regions they serve
      - Mantle /v1/models        -> OpenAI-surface models (GPT-5.x, Grok, Gemma-4, ...)

    Returns a dict with foundation_regions, on_demand_regions, model_info,
    profile_regions, base_profiles, mantle_regions, and per-region errors.
    """
    foundation_regions: dict = defaultdict(set)   # modelId -> {region}
    on_demand_regions: dict = defaultdict(set)    # modelId -> {region} (ON_DEMAND)
    model_info: dict = {}                          # modelId -> {provider,in,out,it}
    profile_regions: dict = defaultdict(set)       # profileId -> {region}
    base_profiles: dict = defaultdict(set)         # base modelId -> {profileId}
    mantle_regions: dict = defaultdict(set)        # mantle id -> {region}
    errors: dict = {}

    def scan(region: str) -> tuple:
        out = {"fm": [], "prof": [], "mantle": [], "err": None}
        try:
            cl = boto3.client("bedrock", region_name=region)
            out["fm"] = cl.list_foundation_models().get("modelSummaries", [])
            profs, r = [], cl.list_inference_profiles(maxResults=100)
            profs += r.get("inferenceProfileSummaries", [])
            while r.get("nextToken"):
                r = cl.list_inference_profiles(maxResults=100, nextToken=r["nextToken"])
                profs += r.get("inferenceProfileSummaries", [])
            out["prof"] = profs
        except Exception as exc:
            out["err"] = str(exc)
        if include_mantle and _MANTLE_AVAILABLE:
            try:
                out["mantle"] = list_mantle_models(region, timeout=12)
            except Exception:
                pass  # region simply has no Mantle surface
        return region, out

    with ThreadPoolExecutor(max_workers=12) as ex:
        for region, out in ex.map(scan, regions):
            if out["err"]:
                errors[region] = out["err"]
            for m in out["fm"]:
                mid = m["modelId"]
                it = m.get("inferenceTypesSupported", [])
                if re.search(r":\d+k$", mid) or re.search(r":mm$", mid) or it == ["PROVISIONED"]:
                    continue
                foundation_regions[mid].add(region)
                if "ON_DEMAND" in it:
                    on_demand_regions[mid].add(region)
                model_info.setdefault(mid, {
                    "provider": m.get("providerName", ""),
                    "in": m.get("inputModalities", []),
                    "out": m.get("outputModalities", []),
                    "it": it,
                })
            for p in out["prof"]:
                pid = p["inferenceProfileId"]
                profile_regions[pid].add(region)
                for mm in p.get("models", []):
                    base = mm.get("modelArn", "").split("/")[-1]
                    if base:
                        base_profiles[base].add(pid)
            for mm in out["mantle"]:
                mantle_regions[mm["model_id"]].add(region)

    logger.info("Discovery: %d foundation models, %d profiles, %d Mantle models across %d/%d regions",
                len(foundation_regions), len(profile_regions), len(mantle_regions),
                len(regions) - len(errors), len(regions))
    return {
        "foundation_regions": foundation_regions,
        "on_demand_regions": on_demand_regions,
        "model_info": model_info,
        "profile_regions": profile_regions,
        "base_profiles": base_profiles,
        "mantle_regions": mantle_regions,
        "errors": errors,
    }


def build_price_index(products: list) -> dict:
    """Index Price List products by (normalized model token, region) -> {tier: {in,out}}.

    Token-keyed (derived from the usagetype) so it serves BOTH Converse models and
    Mantle-only models (whose ids never appear in ListFoundationModels). Only on-demand
    tiers are kept; the first value seen per (token, region, tier, io) wins.
    """
    idx: dict = defaultdict(lambda: defaultdict(dict))
    for p in products:
        e = extract_pricing(p)
        region, ut = e["region_code"], e["usagetype"]
        if not region or not ut or not _is_on_demand_tier(e):
            continue
        token = _extract_model_key_from_usagetype(ut)
        tt = classify_token_type(e)
        if not token or tt == "unknown" or e["price_per_unit_usd"] is None:
            continue
        # Strip a leading provider prefix (e.g. "minimax.minimax-m2.5" -> "minimax-m2.5")
        # so the key matches _model_token(), which also drops the provider.
        token = re.sub(r"^[a-z0-9]+\.", "", token)
        norm = _normalize_name_for_generation(token)
        tier = _detect_tier(ut)
        val = _normalize_to_per_1m(float(e["price_per_unit_usd"]), e["price_unit"])
        idx[(norm, region)][tier].setdefault(tt, val)
    return idx


def build_modelid_pricing(products: list, name_to_info: dict) -> tuple:
    """Build modelId-keyed pricing via the proven usagetype->model-name fuzzy match.

    Used for Converse models, whose Price List usagetypes carry the model *name*
    (no date/version stamp) — so a direct token join off the model *id* fails. Returns
    (pricing, tier_pricing, service_tiers), each keyed by (modelId, region).
    """
    name_matcher = {_normalize_name_for_generation(n): n for n in name_to_info}
    pricing: dict = defaultdict(lambda: {"input": None, "output": None})
    service_tiers: dict = defaultdict(set)
    tier_pricing: dict = defaultdict(lambda: defaultdict(lambda: {"input": None, "output": None}))

    for p in products:
        attrs = p.get("product", {}).get("attributes", {})
        region = attrs.get("regionCode", "")
        usagetype = attrs.get("usagetype", "").lower()
        if not region or not usagetype:
            continue
        usage_key = _extract_model_key_from_usagetype(usagetype)
        if not usage_key:
            continue
        model_id = _match_usagetype_to_model(usage_key, name_matcher, name_to_info)
        if not model_id:
            continue
        key = (model_id, region)
        tier = _detect_tier(usagetype)
        service_tiers[key].add(tier)
        entry = extract_pricing(p)
        if not _is_on_demand_tier(entry):
            continue
        token_type = classify_token_type(entry)
        price = entry["price_per_unit_usd"]
        if price is None or token_type == "unknown":
            continue
        norm_price = _normalize_to_per_1m(float(price), entry.get("price_unit", ""))
        if tier_pricing[key][tier][token_type] is None:
            tier_pricing[key][tier][token_type] = norm_price
        if tier == "default" and pricing[key][token_type] is None:
            pricing[key][token_type] = norm_price
    return pricing, tier_pricing, service_tiers


def _model_token(model_id: str) -> str:
    """Normalized pricing token for a model id (drops provider + version suffix).

    e.g. 'anthropic.claude-opus-4-7-v1:0' / Mantle 'zai.glm-5' -> matches the
    usagetype-derived token used by build_price_index.
    """
    core = model_id
    for p in ("bedrock/converse/", "bedrock/"):      # drop routing prefix (anchor ids carry it)
        if core.startswith(p):
            core = core[len(p):]
            break
    core = re.sub(r"^[a-z]{2}\.", "", core)          # drop cross-region prefix (us./eu./...)
    core = re.sub(r"^[a-z0-9-]+\.", "", core)        # drop provider prefix (anthropic./zai./...)
    core = re.sub(r"(-v?\d+)?:\d+$", "", core)       # drop trailing -v1:0 / -1:0 / :0
    core = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", core)  # drop dated snapshot (-2026-03-05)
    core = re.sub(r"-\d{8}$", "", core)              # drop date stamp (-20251001)
    return _normalize_name_for_generation(core)


def price_for_model(model_id: str, region: str, price_idx: dict) -> Optional[dict]:
    """Resolve {input, output, tier_pricing} for a model+region, or None.

    Order: manual override -> Price List index (default tier, with an output>=input
    sanity guard that signals a bad usagetype match). Returns per-1M $ values.
    """
    base = re.sub(r"(-v\d+)?:\d+$", "", re.sub(r"^[a-z0-9-]+\.", "", model_id))
    for ov_key, ov in MANUAL_PRICE_OVERRIDES.items():
        if _model_token(ov_key) == _model_token(model_id):
            return {"input": ov["input"], "output": ov["output"],
                    "tiers": {"default": dict(ov)}, "source": "override"}

    tok = _model_token(model_id)
    tiers = price_idx.get((tok, region))
    if not tiers:
        # Token pricing is often published in us-east-1 only; fall back across regions
        # (same-geo first, then us-east-1, then any) rather than reporting $0.
        geo = region.split("-")[0] + "-"
        cand = ([t for (t, r) in price_idx if t == tok and r.startswith(geo)]
                + ([region] if (tok, region) in price_idx else []))
        for r in (sorted(set(cand)) + ["us-east-1"] + sorted(r for (t, r) in price_idx if t == tok)):
            if (tok, r) in price_idx:
                tiers = price_idx[(tok, r)]
                break
    if not tiers:
        return None
    default = tiers.get("default", {})
    inp, outp = default.get("input"), default.get("output")
    if inp is None or outp is None:
        return None
    if outp < inp:   # implausible for chat models -> bad match, don't trust it
        logger.warning("Skipping implausible price for %s in %s (in=%s > out=%s)",
                       model_id, region, inp, outp)
        return None
    tp = {t: {"input": v["input"], "output": v["output"]}
          for t, v in tiers.items() if v.get("input") is not None and v.get("output") is not None}
    tp.setdefault("default", {"input": inp, "output": outp})
    return {"input": inp, "output": outp, "tiers": tp, "source": "pricelist"}


def _mint_bedrock_bearer(region: str = "us-east-1") -> Optional[str]:
    """Mint a short-term Bedrock bearer token (for the Mantle Responses probe).

    The Mantle Responses API authenticates with a Bedrock API key, NOT raw SigV4 —
    which is why the SigV4-based probe_responses_api misclassifies everything. Returns
    None if the token generator isn't installed (probe then degrades to "include all").
    """
    try:
        from aws_bedrock_token_generator import provide_token
        return provide_token(region=region)
    except Exception as exc:
        logger.warning("Could not mint Bedrock bearer token (%s); Mantle models won't be "
                       "endpoint-verified and will all be included.", exc)
        return None


def _mantle_serves_responses(model_id: str, region: str, token: str) -> bool:
    """True if a Mantle model actually serves the OpenAI Responses API.

    This is the lesson-learned guard: a Mantle /v1/models listing does NOT imply the
    model is invocable via Responses (the product's Mantle path). The only reliable
    signal is a real call — models that don't serve the route raise "isn't supported on
    this route". Uses a 16-token probe (a few cents across a run). On ambiguous/transient
    errors it returns True, since false-excluding a working model is worse than including
    one that a later run or human review can prune.
    """
    import litellm
    litellm.drop_params = True
    try:
        litellm.responses(
            model=f"openai/{model_id}", input="hi",
            api_base=f"https://bedrock-mantle.{region}.api.aws/openai/v1",
            api_key=token, max_output_tokens=16,
        )
        return True   # clean success only — any error (unsupported route, invalid id) -> False
    except Exception:
        return False


def _mantle_invocable_id(model_id: str, region: str, token: str) -> Optional[str]:
    """Return the invocable Responses-API id for a Mantle model, or None.

    The Mantle /v1/models listing sometimes omits the version suffix the runtime needs
    (e.g. lists ``openai.gpt-oss-120b`` but only ``openai.gpt-oss-120b-1:0`` is callable).
    Try the bare id then common version variants; the first that serves Responses wins.
    """
    for cand in (model_id, f"{model_id}-1:0", f"{model_id}:0"):
        if _mantle_serves_responses(cand, region, token):
            return cand
    return None


def _load_anchor_mantle(anchor_path: Path, live_mantle_tokens: set) -> list[dict]:
    """Existing bedrock_mantle entries to preserve verbatim.

    The Mantle surface is fragile to auto-classify (version suffixes, probe flakiness),
    so trust what already works: keep prior Mantle entries whose model is still live in
    the Mantle listing. New Mantle models go through the Responses probe instead.
    """
    try:
        with open(anchor_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except Exception:
        return []
    return [e for e in rows
            if e.get("endpoint") == "bedrock_mantle"
            and _model_token(e["model_id"]) in live_mantle_tokens]


def generate_models_profiles(regions: Optional[list[str]] = None,
                             include_mantle: bool = True,
                             include_no_pricing: bool = True,
                             anchor_path: Optional[Path] = None,
                             merge_anchor_regions: bool = True) -> list[dict]:
    """Generate model profile entries from a live, multi-region Bedrock API sweep.

    Flow:
        1. discover_across_regions()   -> live models + regions (Converse + Mantle)
        2. _fetch_all_products_parallel() + build_price_index() -> pricing
        3. Classify endpoint (Converse vs Mantle via /v1/responses probe) and build
           entries. Mantle-listed models that are NOT invocable (reject the Responses
           API and aren't Converse models) are excluded.

    include_no_pricing: keep discovered+invocable models even when no price is found,
    tagging them so a human can fill pricing in (vs. silently dropping them).
    """
    logger.info("Generating models_profiles from a live multi-region Bedrock sweep...")

    # Step 1: Discover live models across regions (Converse + Mantle surfaces)
    if regions is None:
        regions = _accessible_regions()
    print(f"Step 1/3: Discovering live models across {len(regions)} regions "
          f"(foundation + inference profiles{' + Mantle' if include_mantle else ''})...")
    disc = discover_across_regions(regions, include_mantle=include_mantle)
    fr, od = disc["foundation_regions"], disc["on_demand_regions"]
    minfo, preg = disc["model_info"], disc["profile_regions"]
    bprof, mreg = disc["base_profiles"], disc["mantle_regions"]
    reachable = len(regions) - len(disc["errors"])
    print(f"  {len(fr)} foundation models, {len(mreg)} Mantle models, "
          f"{reachable}/{len(regions)} regions reachable")

    # Step 2: Pricing from the Price List API. Two views:
    #   - modelId-keyed (name-matched) for Converse models, where usagetypes carry the
    #     model *name* without the date/version stamp that's in the model id.
    #   - token-keyed for Mantle-only models, whose ids aren't in ListFoundationModels.
    print("Step 2/3: Fetching pricing from AWS Price List API (3 service codes in parallel)...")
    products = _fetch_all_products_parallel()
    name_to_info, _ = fetch_foundation_models()
    pricing, tier_pricing, service_tiers = build_modelid_pricing(products, name_to_info)
    price_idx = build_price_index(products)

    # Webpage-scrape fallback: the Price List API only matches a subset of models by
    # name; fill the rest (older models like Jamba/Cohere/Llama) from the pricing page.
    def _has_price(mid: str) -> bool:
        return any(k[0] == mid and pricing[k]["input"] is not None for k in pricing)

    unpriced = [mid for mid in fr if not _has_price(mid)]
    if unpriced:
        scrape_pairs = [(strip_model_id_for_pricing(f"bedrock/{mid}"), "us-east-1") for mid in unpriced]
        scraped = _scrape_webpage_pricing_bulk(scrape_pairs)
        filled = 0
        for mid, (stripped, _) in zip(unpriced, scrape_pairs):
            r = scraped.get(f"us-east-1#{stripped}", {})
            if r.get("input_cost") is not None and r.get("output_cost") is not None:
                pricing[(mid, "us-east-1")]["input"] = r["input_cost"]
                pricing[(mid, "us-east-1")]["output"] = r["output_cost"]
                service_tiers[(mid, "us-east-1")].add("default")
                filled += 1
        logger.info("Webpage scrape filled pricing for %d/%d unmatched models", filled, len(unpriced))
    print(f"  Priced {len(set(k[0] for k in pricing if pricing[k]['input'] is not None))} Converse models, "
          f"{len(set(k[0] for k in price_idx))} Mantle tokens")

    # Step 3: Classify endpoint + build entries
    print("Step 3/3: Classifying endpoints (Converse vs Mantle) and building entries...")
    entries: list[dict] = []
    stats = {"converse": 0, "mantle": 0, "no_price": 0,
             "excluded_nontext": 0, "excluded_not_invocable": 0}
    no_price: list[str] = []

    def _text_ok(model_id: str) -> bool:
        out = (minfo.get(model_id, {}) or {}).get("out") or []
        if out and "TEXT" not in [o.upper() for o in out]:
            return False
        return not _NON_TEXT_PATTERNS.search(model_id)

    def _override(model_id: str):
        for ov_key, ov in MANUAL_PRICE_OVERRIDES.items():
            if _model_token(ov_key) == _model_token(model_id):
                return {"input": ov["input"], "output": ov["output"],
                        "tiers": {"default": dict(ov)}, "source": "override"}
        return None

    def _converse_costs(model_id: str, region: str):
        ov = _override(model_id)
        if ov:
            return ov
        base = _find_pricing_for_region(model_id, region, pricing)
        if not base:
            # Many newer Converse models are priced only under their Mantle usagetype
            # (clean ids with no date stamp) — resolve those from the token index.
            return price_for_model(model_id, region, price_idx)
        raw = set(service_tiers.get((model_id, region), {"default"}))
        raw.discard("standard")
        raw.add("default")
        tp = {}
        for t in sorted(raw):
            tc = tier_pricing.get((model_id, region), {}).get(t, {"input": None, "output": None})
            if tc["input"] is not None and tc["output"] is not None:
                tp[t] = {"input": tc["input"], "output": tc["output"]}
            else:
                tp[t] = {"input": base["input"], "output": base["output"]}
        return {"input": base["input"], "output": base["output"], "tiers": tp, "source": "pricelist"}

    def _mantle_costs(model_id: str, region: str):
        return _override(model_id) or price_for_model(model_id, region, price_idx)

    def _costs_or_placeholder(model_id: str, region: str, resolver):
        c = resolver(model_id, region)
        if c:
            return c
        if not include_no_pricing:
            return None
        return {"input": 0.0, "output": 0.0,
                "tiers": {"default": {"input": 0.0, "output": 0.0}}, "source": "MISSING"}

    def _append(core: str, region: str, costs: dict, endpoint: Optional[str] = None):
        e = {
            "model_id": f"bedrock/{core}",
            "region": region,
            "input_token_cost": costs["input"],
            "output_token_cost": costs["output"],
            "service_tiers": sorted(costs["tiers"].keys()),
            "tier_pricing": costs["tiers"],
        }
        if costs.get("source") and costs["source"] != "pricelist":
            e["pricing_source"] = costs["source"]
        if endpoint == "bedrock_mantle":
            e["endpoint"] = "bedrock_mantle"
            e["mantle_region"] = region
        entries.append(e)

    # Anchor owns routing for models already in the prior catalog. The Mantle surface
    # is fragile to auto-classify (dual-surface models, probe flakiness), so a model the
    # current catalog routes as Mantle stays Mantle — it must NOT be re-routed to Converse
    # just because it also appears in ListFoundationModels (e.g. gpt-oss).
    live_mantle_tokens = {_model_token(m) for m in mreg} if include_mantle else set()
    anchor_mantle = (_load_anchor_mantle(anchor_path or OUTPUT_PATH, live_mantle_tokens)
                     if include_mantle else [])
    anchor_mantle_tokens = {_model_token(e["model_id"]) for e in anchor_mantle}

    # --- Converse models (ListFoundationModels) ---
    seen_tokens: set = set()
    for mid in sorted(fr):
        if not _text_ok(mid):
            stats["excluded_nontext"] += 1
            continue
        if _model_token(mid) in anchor_mantle_tokens:
            continue  # production routes this via Mantle -> preserved below, not Converse
        # Routing convention (matches the existing catalog): if the model has a
        # cross-region inference profile, use it (us./global.) and serve its profile
        # regions plus any on-demand regions. Otherwise use the bare on-demand id.
        profiles = sorted(bprof.get(mid, []))
        if profiles:
            core = _get_preferred_cross_region_id(mid, {mid: profiles})
            regs = sorted(set(preg.get(core, set())) | set(od.get(mid, set())))
        elif mid in od:
            core, regs = mid, sorted(od[mid])
        else:
            continue  # neither a usable profile nor on-demand -> not invocable
        regs = regs or sorted(fr[mid])
        seen_tokens.add(_model_token(mid))
        wrote, missing = False, False
        for region in regs:
            costs = _costs_or_placeholder(mid, region, _converse_costs)
            if not costs:
                continue
            _append(core, region, costs)
            wrote = True
            missing = missing or costs["source"] == "MISSING"
        if wrote:
            stats["converse"] += 1
            if missing:
                no_price.append(f"bedrock/{core}")

    # --- Mantle-only models (OpenAI surface: GPT-5.x, Grok, Gemma-4, ...) ---
    if include_mantle:
        # Preserve known-good Mantle entries from the prior catalog verbatim (no probe),
        # so models already in production (e.g. gpt-oss) never regress on a flaky probe.
        for e in anchor_mantle:
            tok = _model_token(e["model_id"])
            if tok in seen_tokens:
                continue
            seen_tokens.add(tok)
            entries.append(e)
            stats["mantle"] += 1

        bearer = _mint_bedrock_bearer()  # for the Responses-API capability probe
        for mid in sorted(mreg):
            if _model_token(mid) in seen_tokens:
                continue  # also a Converse model -> already added on that surface
            if _DATED_SNAPSHOT_RE.search(mid):
                continue  # dated alias of a rolling "latest" id we already carry
            if _NON_TEXT_PATTERNS.search(mid):
                stats["excluded_nontext"] += 1
                continue
            seen_tokens.add(_model_token(mid))  # dedup dated/undated variants of same model
            regs = sorted(mreg[mid])
            # Endpoint classification: a Mantle listing alone doesn't mean invocable.
            # Verify the model actually serves the Responses API (the product's Mantle
            # path) and resolve the invocable id (the listing may omit a version suffix).
            # Without a bearer token we can't verify, so include all as-is and warn.
            core = mid
            if bearer is not None:
                probe_region = "us-east-1" if "us-east-1" in regs else regs[0]
                inv = _mantle_invocable_id(mid, probe_region, bearer)
                if inv is None:
                    stats["excluded_not_invocable"] += 1
                    logger.info("Excluding Mantle model %s (not invocable via /v1/responses)", mid)
                    continue
                core = inv
            wrote, missing = False, False
            for region in regs:
                costs = _costs_or_placeholder(mid, region, _mantle_costs)
                if not costs:
                    continue
                _append(core, region, costs, endpoint="bedrock_mantle")
                wrote = True
                missing = missing or costs["source"] == "MISSING"
            if wrote:
                stats["mantle"] += 1
                if missing:
                    no_price.append(f"bedrock/{core} (mantle)")

    # Merge entries for regions the live sweep couldn't reach (opt-in regions the account
    # isn't enabled for) from the anchor, so refreshing never drops existing coverage.
    # Only for models still offered somewhere live; regions we DID scan are trusted as-is.
    if merge_anchor_regions:
        covered = set(regions) - set(disc["errors"])
        have = {(e["model_id"], e["region"]) for e in entries}
        live_tokens = {_model_token(e["model_id"]) for e in entries}
        try:
            with open(anchor_path or OUTPUT_PATH) as f:
                anchor_all = [json.loads(line) for line in f if line.strip()]
        except Exception:
            anchor_all = []
        merged = 0
        for e in anchor_all:
            if e["region"] in covered:
                continue  # scanned -> trust the live result (incl. genuine removals)
            if _model_token(e["model_id"]) not in live_tokens:
                continue  # model no longer offered anywhere live
            if (e["model_id"], e["region"]) in have:
                continue
            entries.append(e)
            have.add((e["model_id"], e["region"]))
            merged += 1
        if merged:
            logger.info("Merged %d anchor entries from unreachable regions", merged)
        stats["merged_anchor_regions"] = merged

    stats["no_price"] = len(no_price)
    logger.info("Built %d entries | converse=%d mantle=%d no_price=%d "
                "excluded_nontext=%d excluded_not_invocable=%d",
                len(entries), stats["converse"], stats["mantle"], stats["no_price"],
                stats["excluded_nontext"], stats["excluded_not_invocable"])
    # Stash run metadata for main()'s summary.
    generate_models_profiles.last_stats = stats          # type: ignore[attr-defined]
    generate_models_profiles.last_no_price = no_price     # type: ignore[attr-defined]
    generate_models_profiles.last_errors = disc["errors"] # type: ignore[attr-defined]
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate models_profiles.jsonl from a live multi-region Bedrock sweep.")
    parser.add_argument("--regions", default=None,
                        help="Comma-separated regions to scan (default: all accessible commercial regions).")
    parser.add_argument("--output", default=str(OUTPUT_PATH),
                        help=f"Output path (default: {OUTPUT_PATH}).")
    parser.add_argument("--no-mantle", action="store_true",
                        help="Skip Mantle (OpenAI-surface) discovery.")
    parser.add_argument("--drop-no-pricing", action="store_true",
                        help="Exclude models with no resolvable price (default: keep + flag them).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and summarize but do not write the output file.")
    parser.add_argument("--no-merge-regions", action="store_true",
                        help="Don't merge anchor entries for unreachable (opt-in) regions.")
    args = parser.parse_args()

    regions = [r.strip() for r in args.regions.split(",")] if args.regions else None
    output_path = Path(args.output)

    print("=" * 60)
    print("Bedrock Model Catalog Generator (live, multi-region)")
    print("=" * 60)
    if not _MANTLE_AVAILABLE and not args.no_mantle:
        print(f"  NOTE: Mantle helpers unavailable ({_MANTLE_IMPORT_ERROR}); Mantle models skipped.")
    print()

    entries = generate_models_profiles(
        regions=regions,
        include_mantle=not args.no_mantle,
        include_no_pricing=not args.drop_no_pricing,
        merge_anchor_regions=not args.no_merge_regions,
    )
    if not entries:
        print("\nNo entries generated. Check your AWS credentials and permissions.")
        return 1

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    stats = getattr(generate_models_profiles, "last_stats", {})
    no_price = getattr(generate_models_profiles, "last_no_price", [])
    errors = getattr(generate_models_profiles, "last_errors", {})
    models = set(e["model_id"] for e in entries)
    all_regions = set(e["region"] for e in entries)
    mantle = [e for e in entries if e.get("endpoint") == "bedrock_mantle"]

    print()
    print("=" * 60)
    print("GENERATION COMPLETE" + ("  (dry-run, not written)" if args.dry_run else ""))
    print("=" * 60)
    print(f"Total entries:              {len(entries)}")
    print(f"Unique models:              {len(models)}  (converse={stats.get('converse',0)}, mantle={stats.get('mantle',0)})")
    if stats.get("merged_anchor_regions"):
        print(f"Merged (unreachable rgns):  {stats['merged_anchor_regions']}  (preserved from prior catalog)")
    print(f"Mantle entries:             {len(mantle)}")
    print(f"Regions covered:            {len(all_regions)}")
    print(f"Excluded (non-text):        {stats.get('excluded_nontext',0)}")
    print(f"Excluded (not invocable):   {stats.get('excluded_not_invocable',0)}")
    print(f"Output file:                {output_path}")
    if errors:
        print(f"\nRegions unreachable ({len(errors)}): {', '.join(sorted(errors))}")
    if no_price:
        print(f"\n⚠  {len(no_price)} model(s) need manual pricing (emitted with $0 + pricing_source=MISSING):")
        for m in no_price:
            print(f"     - {m}")
        print("   Add a MANUAL_PRICE_OVERRIDES entry or wait for the Price List API to publish them.")
    print()
    print("Sample entries:")
    for e in entries[:5]:
        ep = " [mantle]" if e.get("endpoint") == "bedrock_mantle" else ""
        print(f"  {e['model_id']:<50}{ep:<10} {e['region']:<14} in=${e['input_token_cost']} out=${e['output_token_cost']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
