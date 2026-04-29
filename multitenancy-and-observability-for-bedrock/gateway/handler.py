# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Gateway Lambda handler for the ISV Amazon Bedrock Observability platform.

Sits behind API Gateway, proxies profile requests to Amazon Bedrock
``converse()`` and emits per-profile CloudWatch metrics (tokens, cost,
latency).
"""

import json
import logging
import os
import time
from decimal import Decimal

import boto3

from status_cache import ProfileStatusCache

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Module-level clients & cache – reused across warm invocations
# ---------------------------------------------------------------------------
bedrock_runtime = boto3.client("bedrock-runtime")
dynamodb = boto3.resource("dynamodb")
cloudwatch = boto3.client("cloudwatch")

TENANTS_TABLE = os.environ.get("TENANTS_TABLE", "Tenants")
PRICING_CACHE_TABLE = os.environ.get("PRICING_CACHE_TABLE", "PricingCache")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

tenants_table = dynamodb.Table(TENANTS_TABLE)
pricing_table = dynamodb.Table(PRICING_CACHE_TABLE)

profile_cache = ProfileStatusCache(ttl_seconds=30)

METRIC_NAMESPACE = "ISVBedrock/Gateway"

TAG_DIMENSION_PREFIX = "Tag_"


def _tags_to_cw_dimensions(tags: dict) -> list:
    """Convert profile tags to CloudWatch dimension entries."""
    dims = []
    for k, v in (tags or {}).items():
        val = " / ".join(v) if isinstance(v, list) else str(v)
        if val:
            dims.append({"Name": f"{TAG_DIMENSION_PREFIX}{k}", "Value": val})
    return dims


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_response(status_code: int, body: dict) -> dict:
    """Build an API Gateway proxy-integration response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Profile-Id,Tenant-Id",
            "Access-Control-Allow-Methods": "POST,OPTIONS",
        },
        "body": json.dumps(body, default=str),
    }


def _resolve_profile_id(event: dict) -> str | None:
    """Extract profile_id from path parameters or headers.

    Accepts both ``Profile-Id`` and ``Tenant-Id`` headers for backward
    compatibility.
    """
    path_params = event.get("pathParameters") or {}
    profile_id = path_params.get("tenant_id")
    if profile_id:
        return profile_id

    headers = event.get("headers") or {}
    # API Gateway lowercases header names in v2 payloads
    return (
        headers.get("Profile-Id")
        or headers.get("profile-id")
        or headers.get("profile_id")
        or headers.get("tenant_id")
        or headers.get("Tenant-Id")
        or headers.get("tenant-id")
    )


def _get_profile(profile_id: str) -> dict | None:
    """Return the profile record from cache or DynamoDB."""
    cached = profile_cache.get(profile_id)
    if cached is not None:
        return cached

    resp = tenants_table.get_item(Key={"tenant_id": profile_id})
    item = resp.get("Item")
    if item is None:
        return None

    profile_cache.set(profile_id, item)
    return item


def _get_pricing(model_id: str) -> dict | None:
    """Fetch pricing record from PricingCache table.

    Key format: ``{region}#{model_id}``
    """
    pricing_key = f"{AWS_REGION}#{model_id}"
    try:
        resp = pricing_table.get_item(Key={"region#model_id": pricing_key})
        return resp.get("Item")
    except Exception as exc:
        logger.warning("Pricing lookup failed for %s: %s", pricing_key, exc)
        return None


def _calculate_cost(token_count: int, per_thousand_rate) -> float:
    """Cost = rate_per_1K_tokens * (tokens / 1_000)."""
    rate = float(per_thousand_rate) if isinstance(per_thousand_rate, Decimal) else per_thousand_rate
    return rate * (token_count / 1_000)


def _emit_metrics(
    inference_profile: str,
    profile_id: str,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    input_cost: float,
    output_cost: float,
    latency_ms: float,
    success: bool,
    tags: dict | None = None,
) -> None:
    """Publish CloudWatch metrics with multiple dimension sets.

    Emits each metric twice:
    1. Base dimensions (TenantId + InferenceProfile) — used by dashboard widgets
    2. Full dimensions (base + ModelId + Tag_*) — used for detailed/tag-based queries

    CloudWatch requires exact dimension matching, so dashboards that query by
    (TenantId, InferenceProfile) won't find metrics stored only with
    (TenantId, InferenceProfile, ModelId, Tag_*).
    """
    base_dimensions = [
        {"Name": "InferenceProfile", "Value": inference_profile},
        {"Name": "TenantId", "Value": profile_id},
    ]

    full_dimensions = list(base_dimensions) + [
        {"Name": "ModelId", "Value": model_id},
    ]
    if tags:
        full_dimensions.extend(_tags_to_cw_dimensions(tags))
    if len(full_dimensions) > 30:
        logger.warning("Truncated CloudWatch dimensions to 30 (CW limit)")
        full_dimensions = full_dimensions[:30]

    metric_names = [
        ("InputTokens", input_tokens, "Count"),
        ("OutputTokens", output_tokens, "Count"),
        ("InputTokensCost", input_cost, "None"),
        ("OutputTokensCost", output_cost, "None"),
        ("InvocationLatencyMs", latency_ms, "Milliseconds"),
    ]

    if success:
        metric_names.append(("InvocationSuccess", 1, "Count"))
    else:
        metric_names.append(("InvocationFailure", 1, "Count"))

    # Emit with base dimensions (for dashboards) and full dimensions (for detailed queries)
    metric_data = []
    for name, value, unit in metric_names:
        metric_data.append({
            "MetricName": name,
            "Dimensions": base_dimensions,
            "Value": value,
            "Unit": unit,
        })
        metric_data.append({
            "MetricName": name,
            "Dimensions": full_dimensions,
            "Value": value,
            "Unit": unit,
        })

    cloudwatch.put_metric_data(Namespace=METRIC_NAMESPACE, MetricData=metric_data)


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------

def handler(event, context):  # noqa: ARG001 – context unused but required
    """API Gateway proxy Lambda handler."""
    logger.info("Received event: %s", json.dumps(event, default=str))

    # 1. Resolve profile --------------------------------------------------
    profile_id = _resolve_profile_id(event)
    if not profile_id:
        return _json_response(400, {"error": "Missing profile_id in path or headers"})

    # 2. Profile lookup (cache / DynamoDB) ---------------------------------
    profile = _get_profile(profile_id)
    if profile is None:
        return _json_response(404, {"error": f"Profile '{profile_id}' not found"})

    # 3. Profile status enforcement ----------------------------------------
    status = profile.get("status", "active")
    if status == "suspended":
        return _json_response(403, {"error": "Profile is suspended"})
    if status == "throttled":
        return _json_response(429, {"error": "Profile is throttled"})

    # 4. Parse request body -----------------------------------------------
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _json_response(400, {"error": "Invalid JSON in request body"})

    messages = body.get("messages")
    if not messages:
        return _json_response(400, {"error": "Request body must include 'messages'"})

    # Determine model: prefer profile's inference profile ARN, fall back to
    # explicit model_id in the request, then profile-level model_id.
    inference_profile_id = profile.get("inference_profile_id", "")
    inference_profile_arn = profile.get("inference_profile_arn", "")
    model_id = body.get("model_id") or profile.get("model_id")
    converse_model_id = inference_profile_arn or model_id

    if not converse_model_id:
        return _json_response(400, {"error": "No model_id or inference_profile_id configured"})

    # 5. Pricing lookup ---------------------------------------------------
    pricing_lookup_key = model_id or inference_profile_id or ""
    pricing = _get_pricing(pricing_lookup_key)
    input_cost_rate = float(pricing.get("input_cost", 0)) if pricing else 0.0
    output_cost_rate = float(pricing.get("output_cost", 0)) if pricing else 0.0

    # 6. Call Amazon Bedrock converse() ------------------------------------------
    latency_ms = 0.0
    try:
        start = time.monotonic()
        response = bedrock_runtime.converse(
            modelId=converse_model_id,
            messages=messages,
        )
        end = time.monotonic()
        latency_ms = (end - start) * 1000
    except Exception as exc:
        logger.exception("Bedrock converse() failed for profile %s", profile_id)
        # Emit failure metric even on error
        _emit_metrics(
            inference_profile=inference_profile_id or converse_model_id,
            profile_id=profile_id,
            model_id=model_id or converse_model_id,
            input_tokens=0,
            output_tokens=0,
            input_cost=0.0,
            output_cost=0.0,
            latency_ms=latency_ms,
            success=False,
            tags=profile.get("tags", {}),
        )
        return _json_response(502, {"error": f"Bedrock invocation failed: {exc}"})

    # 7. Extract token usage & compute cost -------------------------------
    usage = response.get("usage", {})
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)

    input_cost = _calculate_cost(input_tokens, input_cost_rate)
    output_cost = _calculate_cost(output_tokens, output_cost_rate)

    # 8. Emit CloudWatch metrics ------------------------------------------
    _emit_metrics(
        inference_profile=inference_profile_id or converse_model_id,
        profile_id=profile_id,
        model_id=model_id or converse_model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        latency_ms=latency_ms,
        success=True,
        tags=profile.get("tags", {}),
    )

    # 9. Return response --------------------------------------------------
    output_message = response.get("output", {}).get("message", {})
    return _json_response(200, {
        "output": output_message,
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
        },
        "cost": {
            "inputCost": input_cost,
            "outputCost": output_cost,
            "totalCost": input_cost + output_cost,
        },
        "latencyMs": round(latency_ms, 2),
    })
