"""Bedrock Mantle API client.

The Mantle endpoint (``bedrock-mantle.{region}.api.aws``) is an OpenAI-compatible
surface in Amazon Bedrock that exposes models — including third-party ones such as
OpenAI GPT-5.x — that do NOT show up in ``bedrock:ListFoundationModels`` or the AWS
Price List API. Use this to discover/validate Mantle model availability per region.

Auth: SigV4-signed HTTPS (signing service name ``bedrock``). Requires valid AWS
credentials in the environment (same as every other boto3 call in this project).

Helpers:
- ``list_mantle_models(region)``            -> normalized model dicts in one region
- ``find_mantle_models(substr, regions)``   -> search model ids across regions
- ``mantle_models_by_region(regions)``      -> {region: [model_id, ...]}
- ``probe_responses_api(model_id, region)`` -> does the model support /v1/responses
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

logger = logging.getLogger(__name__)

MANTLE_ENDPOINT_PATTERN = "bedrock-mantle.{region}.api.aws"
SIGNING_SERVICE = "bedrock"
DEFAULT_TIMEOUT = 10

# Reused across calls so credential resolution isn't repeated each request.
_session = boto3.Session()


def mantle_endpoint(region: str) -> str:
    """Return the Mantle host for a region."""
    return MANTLE_ENDPOINT_PATTERN.format(region=region)


def _signed_request(method: str, region: str, path: str, body: dict | None = None):
    """Build a SigV4-signed urllib Request for the Mantle endpoint."""
    host = mantle_endpoint(region)
    url = f"https://{host}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None

    # Set Host before signing so SigV4 includes it in the canonical request.
    headers = {"Content-Type": "application/json", "Host": host}
    aws_request = AWSRequest(method=method, url=url, headers=headers, data=data)

    credentials = _session.get_credentials().get_frozen_credentials()
    SigV4Auth(credentials, SIGNING_SERVICE, region).add_auth(aws_request)

    return urllib.request.Request(url, data=data, headers=dict(aws_request.headers), method=method)


def list_mantle_models(region: str = "us-east-1", timeout: int = DEFAULT_TIMEOUT) -> list[dict]:
    """List all models from Mantle ``/v1/models`` in one region.

    Returns a list of ``{"model_id", "provider", "region", "raw"}`` dicts.
    Raises ``urllib.error.HTTPError``/``URLError`` on transport failure so callers
    can distinguish "region has no Mantle" from "empty model list".
    """
    req = _signed_request("GET", region, "/v1/models")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    # Accept both {"data": [...]} (OpenAI shape) and a flat array.
    if isinstance(payload, dict):
        raw = payload.get("data", [])
    elif isinstance(payload, list):
        raw = payload
    else:
        raw = []

    return [
        {
            "model_id": m.get("id", ""),
            "provider": m.get("owned_by", ""),
            "region": region,
            "raw": m,
        }
        for m in raw
        if isinstance(m, dict) and m.get("id")
    ]


def get_mantle_model(model_id: str, region: str = "us-east-1",
                     timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Fetch the per-model detail object from Mantle ``/v1/models/{id}``.

    Use this to confirm whether the detail endpoint returns any richer fields
    (e.g. pricing/limits) than the minimal list object. Returns the raw JSON.
    """
    req = _signed_request("GET", region, f"/v1/models/{model_id}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def mantle_models_by_region(regions: list[str], timeout: int = DEFAULT_TIMEOUT) -> dict[str, list[str]]:
    """Return ``{region: [model_id, ...]}`` for each region (errors -> empty list)."""
    out: dict[str, list[str]] = {}
    for region in regions:
        try:
            out[region] = [m["model_id"] for m in list_mantle_models(region, timeout)]
        except Exception as e:  # noqa: BLE001 - report per-region, don't abort the sweep
            logger.warning("Mantle list failed in %s: %s", region, e)
            out[region] = []
    return out


def find_mantle_models(substring: str, regions: list[str] | None = None,
                       timeout: int = DEFAULT_TIMEOUT) -> list[dict]:
    """Find Mantle models whose id contains ``substring`` (case-insensitive).

    Returns matched model dicts (with their region). Per-region transport errors
    are returned as ``{"region", "error"}`` entries so failures are visible.
    """
    regions = regions or ["us-east-1"]
    sub = substring.lower()
    hits: list[dict] = []
    for region in regions:
        try:
            for m in list_mantle_models(region, timeout):
                if sub in m["model_id"].lower():
                    hits.append(m)
        except Exception as e:  # noqa: BLE001
            hits.append({"region": region, "error": str(e)})
    return hits


def fetch_mantle_pricing(model_substr: str | None = None,
                         pricing_region: str = "us-east-1") -> list[dict]:
    """Fetch Mantle (OpenAI-compatible) token pricing from the AWS Price List API.

    Mantle models are NOT priced via ``/v1/models`` (that endpoint carries no price).
    Instead AWS publishes them in the Price List API under usagetypes shaped like
    ``<REGIONPFX>-openai.<model>-mantle-<input|output>-tokens-<tier>`` where tier is
    one of standard/flex/priority/batch. NOTE: as of this writing the Price List API
    lags availability — commercial-region rows for the newest models (e.g. gpt-5.5)
    may not be published yet even though Mantle lists them as available.

    Args:
        model_substr: optional case-insensitive filter on the model token
            (e.g. "gpt-5", "gpt-oss-120b").
        pricing_region: region for the Price List client (only us-east-1 / ap-south-1).

    Returns:
        List of ``{model, region, tier, io, price_per_1m, usagetype}`` dicts.
        (Price List prices are per-1K tokens; converted here to per-1M.)
    """
    import re as _re

    pc = _session.client("pricing", region_name=pricing_region)
    service_codes = ["AmazonBedrock", "AmazonBedrockFoundationModels", "AmazonBedrockService"]
    pat = _re.compile(r"openai\.([a-z0-9.\-]+?)-mantle-(input|output)-tokens-(\w+)", _re.I)
    sub = (model_substr or "").lower()

    rows: list[dict] = []
    for sc in service_codes:
        token = None
        while True:
            kw = {"ServiceCode": sc, "MaxResults": 100, "FormatVersion": "aws_v1"}
            if token:
                kw["NextToken"] = token
            resp = pc.get_products(**kw)
            for item in resp.get("PriceList", []):
                prod = json.loads(item) if isinstance(item, str) else item
                attrs = prod.get("product", {}).get("attributes", {})
                ut = attrs.get("usagetype", "")
                m = pat.search(ut)
                if not m:
                    continue
                model, io, tier = m.group(1), m.group(2).lower(), m.group(3).lower()
                if sub and sub not in model.lower():
                    continue
                price = None
                for offer in prod.get("terms", {}).get("OnDemand", {}).values():
                    for dim in offer.get("priceDimensions", {}).values():
                        usd = dim.get("pricePerUnit", {}).get("USD")
                        unit = (dim.get("unit", "") or "").lower()
                        if usd is not None:
                            # Normalize to per-1M tokens. Older rows are per "1K tokens";
                            # newer ones (e.g. gpt-5.x) are already per "1M tokens".
                            mult = 1000.0 if ("1k" in unit or "thousand" in unit) else 1.0
                            price = round(float(usd) * mult, 4)
                        break
                rows.append({
                    "model": f"openai.{model}", "region": attrs.get("regionCode", ""),
                    "tier": tier, "io": io, "price_per_1m": price, "usagetype": ut,
                })
            token = resp.get("NextToken")
            if not token:
                break
    return rows


def probe_responses_api(model_id: str, region: str = "us-east-1",
                        timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Probe whether a Mantle model supports the OpenAI Responses API.

    Sends ``POST /v1/responses`` with no input (free — no tokens consumed). The
    endpoint accepts the model then rejects the empty input if it's supported.
    Returns True if supported, False otherwise.
    """
    req = _signed_request("POST", region, "/v1/responses", body={"model": model_id})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        # HTTP 200 + an input-validation error means the model itself was accepted.
        err_code = (data.get("error") or {}).get("code", "") if isinstance(data, dict) else ""
        return err_code in ("invalid_prompt", "invalid_request_error") or "error" not in data
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        # 400 validation_error = not supported; 404 = unknown model.
        return e.code == 200 or "invalid_prompt" in body
    except Exception as e:  # noqa: BLE001
        logger.warning("Responses probe failed for %s in %s: %s", model_id, region, e)
        return False
