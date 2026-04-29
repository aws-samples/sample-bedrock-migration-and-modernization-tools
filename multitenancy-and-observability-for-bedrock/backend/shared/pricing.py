# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Amazon Bedrock model pricing with 3-tier fallback and DynamoDB caching.

Flow:
    1. Check DynamoDB cache -- if < 24h old, return immediately
    2. Tier 1: Query AWS Price List API (all 3 service codes)
       -> If both input + output found: pricing_source = "api"
    3. Tier 2: If output missing, re-query AmazonBedrockService only
       -> If filled: pricing_source = "api_partial"
    4. Tier 3: Scrape the Amazon Bedrock pricing webpage bulk JSON
       -> If filled: pricing_source = "webpage"
    5. If still missing -> "unavailable" with None costs
    6. Cache the result in DynamoDB with 24h TTL

Environment variables:
    PRICING_CACHE_TABLE - DynamoDB table for pricing cache
"""

import gzip
import html as html_module
import io
import json
import logging
import os
import re
import time
import urllib.request
from decimal import Decimal
from typing import Optional

import boto3

logger = logging.getLogger(__name__)

SERVICE_CODES = [
    "AmazonBedrock",
    "AmazonBedrockService",
    "AmazonBedrockFoundationModels",
]
PRICING_CLIENT_REGION = "us-east-1"
CACHE_TTL_SECONDS = 86400  # 24 hours

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

    model_lower = model_id.lower()
    entry_model_id = entry.get("model_id", "").lower()
    entry_usagetype = entry.get("usagetype", "").lower()

    if entry_model_id and entry_model_id == model_lower:
        return True
    if model_lower in entry_usagetype:
        return True
    if entry_model_id:
        a = re.sub(r"[-._:]", "", model_lower)
        b = re.sub(r"[-._:]", "", entry_model_id)
        if a in b or b in a:
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

def _query_price_list_api(model_id: str, region: str) -> dict:
    """Tier 1: Query AWS Price List API across all 3 service codes.

    Returns {"input_cost": float|None, "output_cost": float|None}.
    """
    pricing_client = boto3.client("pricing", region_name=PRICING_CLIENT_REGION)
    input_cost = None
    output_cost = None

    for sc in SERVICE_CODES:
        try:
            products = get_all_products(pricing_client, sc)
        except Exception as exc:
            logger.warning("Failed to query %s: %s", sc, exc)
            continue

        costs = _extract_costs_from_products(products, model_id, region)

        if input_cost is None and costs["input_cost"] is not None:
            input_cost = costs["input_cost"]
        if output_cost is None and costs["output_cost"] is not None:
            output_cost = costs["output_cost"]

        if input_cost is not None and output_cost is not None:
            break

    return {"input_cost": input_cost, "output_cost": output_cost}


# ---------------------------------------------------------------------------
# Tier 2: Re-query AmazonBedrockService only for missing output pricing
# ---------------------------------------------------------------------------

def _query_bedrock_service_only(model_id: str, region: str) -> dict:
    """Tier 2: Re-query only AmazonBedrockService for missing pricing.

    Some models (older Anthropic) have input pricing in AmazonBedrock
    but output pricing only in AmazonBedrockService.

    Returns {"input_cost": float|None, "output_cost": float|None}.
    """
    pricing_client = boto3.client("pricing", region_name=PRICING_CLIENT_REGION)

    try:
        products = get_all_products(pricing_client, "AmazonBedrockService")
    except Exception as exc:
        logger.warning("Failed to query AmazonBedrockService for tier 2: %s", exc)
        return {"input_cost": None, "output_cost": None}

    return _extract_costs_from_products(products, model_id, region)


# ---------------------------------------------------------------------------
# Tier 3: Scrape Amazon Bedrock pricing webpage bulk JSON
# ---------------------------------------------------------------------------

def _fetch_url(url: str, timeout: int = 15) -> bytes:
    """Fetch a URL and return raw bytes. Handles gzip encoding."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "ISVAmazon BedrockObservability/1.0",
        "Accept-Encoding": "gzip, deflate",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
        return data


def _parse_pricing_page(page_html: str) -> list[dict]:
    """Parse the Amazon Bedrock pricing page HTML to extract model-to-hash mappings.

    Extracts entries of the form:
        ModelName</td><td>{priceOf!service/sub!HASH[!*!N][!opt]}</td>
                       <td>{priceOf!service/sub!HASH[!*!N][!opt]}</td>

    Returns a list of dicts:
        [{"model_name": str, "service_path": str,
          "input_hash": str, "output_hash": str,
          "input_mult": int, "output_mult": int}, ...]
    """
    content = html_module.unescape(page_html)

    # Regex for a priceOf token (with named groups for extraction)
    token_re = (
        r'\{priceOf!'
        r'(?P<path>[^!}]+/[^!}]+)!'
        r'(?P<hash>[A-Za-z0-9_-]+)'
        r'(?:!\*!(?P<mult>\d+))?'
        r'(?:!opt)?'
        r'\}'
    )

    # Unnamed version for use in the row-level regex
    token_re_unnamed = (
        r'\{priceOf!'
        r'[^!}]+/[^!}]+!'
        r'[A-Za-z0-9_-]+'
        r'(?:!\*!\d+)?'
        r'(?:!opt)?'
        r'\}'
    )

    # Match table rows: <td>ModelName</td><td>{priceOf...}</td><td>{priceOf...}</td>
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

        # Deduplicate by (model_name, service_path, input_hash)
        dedup_key = (model_name, service_path, input_hash)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # Skip long-context, batch, priority, flex variants
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
    # Strip common prefixes
    for prefix in ("anthropic.", "meta.", "amazon.", "cohere.", "ai21.",
                    "mistral.", "stability.", "deepseek.", "us.", "eu."):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    # Remove version suffixes like -20240307-v1:0, -v1:0
    s = re.sub(r"-\d{8}-v\d+:\d+$", "", s)
    s = re.sub(r"-v\d+:\d+$", "", s)
    s = re.sub(r"-v\d+$", "", s)
    # Normalize separators
    s = re.sub(r"[-._:]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _find_model_entry(model_id: str, entries: list[dict]) -> Optional[dict]:
    """Find the best matching entry for a model_id using fuzzy matching.

    Tries exact substring match first, then normalized token matching.
    """
    norm_id = _normalize_for_match(model_id)
    norm_id_nospaces = norm_id.replace(" ", "")

    best_match = None
    best_score = 0

    for entry in entries:
        norm_name = _normalize_for_match(entry["model_name"])
        norm_name_nospaces = norm_name.replace(" ", "")

        # Exact normalized match
        if norm_id == norm_name:
            return entry

        # Substring containment (either direction)
        if norm_id_nospaces in norm_name_nospaces or norm_name_nospaces in norm_id_nospaces:
            score = len(norm_name_nospaces)
            if score > best_score:
                best_score = score
                best_match = entry
            continue

        # Token overlap: count shared words
        id_tokens = set(norm_id.split())
        name_tokens = set(norm_name.split())
        overlap = len(id_tokens & name_tokens)
        if overlap >= 2 and overlap > best_score:
            best_score = overlap
            best_match = entry

    return best_match


def _fetch_bulk_json(service_path: str) -> Optional[dict]:
    """Fetch the bulk pricing JSON for a service.

    The service_path comes from a priceOf token (e.g. "bedrock/bedrock").
    Maps to a bulk JSON URL at b0.p.awsstatic.com.
    """
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
    """Tier 3: Scrape Amazon Bedrock pricing webpage for missing prices.

    Fetches the pricing page HTML to build a model→hash mapping, then
    resolves prices from the public bulk JSON at b0.p.awsstatic.com.

    The bulk JSON is the same data source that powers the AWS pricing
    webpage -- not web scraping per se, but reading the underlying
    structured data.

    Returns {"input_cost": float|None, "output_cost": float|None}.
    """
    try:
        # 1. Fetch and parse the pricing page HTML
        page_bytes = _fetch_url(_PRICING_PAGE_URL, timeout=20)
        page_html = page_bytes.decode("utf-8", errors="replace")
        entries = _parse_pricing_page(page_html)

        if not entries:
            logger.warning("Tier 3: No model entries parsed from pricing page")
            return {"input_cost": None, "output_cost": None}

        # 2. Find matching model entry
        matched = _find_model_entry(model_id, entries)
        if not matched:
            logger.info("Tier 3: No matching model found for %s", model_id)
            return {"input_cost": None, "output_cost": None}

        logger.info(
            "Tier 3: Matched %s -> '%s' (service=%s)",
            model_id, matched["model_name"], matched["service_path"],
        )

        # 3. Fetch bulk pricing JSON for the matched service
        bulk_data = _fetch_bulk_json(matched["service_path"])
        if not bulk_data:
            return {"input_cost": None, "output_cost": None}

        # 4. Look up the target region
        region_name = _REGION_DISPLAY_NAMES.get(region)
        if not region_name:
            # Try partial match (some regions use slightly different names)
            for code, name in _REGION_DISPLAY_NAMES.items():
                if code == region:
                    region_name = name
                    break
            if not region_name:
                logger.warning("Tier 3: Unknown region code %s", region)
                return {"input_cost": None, "output_cost": None}

        region_prices = bulk_data.get("regions", {}).get(region_name, {})
        if not region_prices:
            logger.info("Tier 3: No pricing data for region %s", region_name)
            return {"input_cost": None, "output_cost": None}

        # 5. Resolve prices from hashes
        input_entry = region_prices.get(matched["input_hash"], {})
        output_entry = region_prices.get(matched["output_hash"], {})

        input_raw = float(input_entry["price"]) if input_entry.get("price") else None
        output_raw = float(output_entry["price"]) if output_entry.get("price") else None

        # 6. Convert to per-1K-token format (matching Price List API convention)
        #
        # The priceOf multiplier tells us the raw unit:
        #   {priceOf!svc!hash!*!1000} → raw is per 1K tokens, display = raw * 1000 (per 1M)
        #   {priceOf!svc!hash}        → raw is per 1M tokens, display = raw (per 1M)
        #
        # We want per 1K tokens (matching Price List API). So:
        #   With mult=1000: per_1K = raw (already per 1K)
        #   With mult=1:    per_1K = raw / 1000 (convert per 1M to per 1K)
        input_cost = None
        output_cost = None

        if input_raw is not None:
            if matched["input_mult"] >= 1000:
                input_cost = input_raw
            else:
                input_cost = input_raw / 1000.0

        if output_raw is not None:
            if matched["output_mult"] >= 1000:
                output_cost = output_raw
            else:
                output_cost = output_raw / 1000.0

        return {"input_cost": input_cost, "output_cost": output_cost}

    except Exception as exc:
        logger.warning("Tier 3 webpage pricing scrape failed: %s", exc)
        return {"input_cost": None, "output_cost": None}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_model_pricing(model_id: str, region: str) -> dict:
    """Get pricing for a model using 3-tier fallback.

    Tier 1: Query Price List API (all 3 service codes) -> "api"
    Tier 2: Re-query AmazonBedrockService for gaps -> "api_partial"
    Tier 3: Scrape AWS pricing webpage bulk JSON -> "webpage"
    Otherwise -> "unavailable" with None costs.

    Returns:
        dict with keys: input_cost, output_cost, pricing_source
    """
    # Tier 1: Query Price List API (all 3 service codes)
    result = _query_price_list_api(model_id, region)

    if result["input_cost"] is not None and result["output_cost"] is not None:
        return {
            "input_cost": result["input_cost"],
            "output_cost": result["output_cost"],
            "pricing_source": "api",
        }

    # Tier 2: Re-query AmazonBedrockService for missing pricing
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

    # Pricing not fully available
    return {
        "input_cost": result["input_cost"],
        "output_cost": result["output_cost"],
        "pricing_source": "unavailable",
    }


# ---------------------------------------------------------------------------
# DynamoDB pricing cache
# ---------------------------------------------------------------------------

def _get_pricing_table():
    """Get DynamoDB Table resource for pricing cache."""
    table_name = os.environ.get("PRICING_CACHE_TABLE", "")
    if not table_name:
        raise EnvironmentError("PRICING_CACHE_TABLE environment variable not set")
    dynamodb = boto3.resource("dynamodb")
    return dynamodb.Table(table_name)


def cache_pricing(table, cache_key: str, pricing_data: dict) -> None:
    """Write pricing data to DynamoDB with 24h TTL."""
    now = time.time()
    item = {
        "cache_key": cache_key,
        "region#model_id": cache_key,
        "model_id": pricing_data.get("model_id", cache_key.split("#", 1)[-1]),
        "region": pricing_data.get("region", cache_key.split("#", 1)[0]),
        "input_cost": Decimal(str(pricing_data["input_cost"])) if pricing_data.get("input_cost") is not None else Decimal("0"),
        "output_cost": Decimal(str(pricing_data["output_cost"])) if pricing_data.get("output_cost") is not None else Decimal("0"),
        "pricing_source": pricing_data.get("pricing_source", "unknown"),
        "cached_at": Decimal(str(now)),
        "ttl": int(now) + CACHE_TTL_SECONDS,
    }
    table.put_item(Item=item)


def get_cached_pricing(table, cache_key: str) -> Optional[dict]:
    """Read pricing from DynamoDB cache. Returns None if missing or expired."""
    try:
        response = table.get_item(Key={"region#model_id": cache_key})
    except Exception as exc:
        logger.warning("Failed to read pricing cache for %s: %s", cache_key, exc)
        return None

    item = response.get("Item")
    if not item:
        return None

    # Check TTL expiry
    ttl = int(item.get("ttl", 0))
    if ttl > 0 and ttl < int(time.time()):
        return None

    return {
        "cache_key": cache_key,
        "model_id": item.get("model_id", ""),
        "region": item.get("region", ""),
        "input_cost": float(item.get("input_cost", 0)),
        "output_cost": float(item.get("output_cost", 0)),
        "pricing_source": item.get("pricing_source", "unknown"),
        "cached_at": float(item.get("cached_at", 0)),
        "ttl": ttl,
    }
