# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Lambda handler for profile CRUD operations.

Routes based on httpMethod + resource path from API Gateway proxy integration.

Environment variables:
    TENANTS_TABLE          - DynamoDB table for profile records
    PROFILE_MAPPINGS_TABLE - DynamoDB table for profile-to-tenant mappings
"""

import json
import logging
import os
import uuid
import datetime

import boto3

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared import dynamo_utils
from shared.tag_utils import parse_tag_filters, filter_by_tags, validate_tags, PREDEFINED_TAG_CATEGORIES

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TENANTS_TABLE = os.environ.get("TENANTS_TABLE", "")
PROFILE_MAPPINGS_TABLE = os.environ.get("PROFILE_MAPPINGS_TABLE", "")


def handler(event, context):
    """Main Lambda handler - routes to appropriate operation."""
    http_method = event.get("httpMethod", "")
    path = event.get("path", "")
    path_params = event.get("pathParameters") or {}
    # With {proxy+} routing, path params come as {"proxy": "profiles/abc-123"}
    # not {"tenant_id": "abc-123"}. Extract profile_id from the path.
    profile_id = path_params.get("tenant_id")
    if not profile_id:
        # Parse from path: /profiles/{id}, /profiles/{id}/activate, etc.
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[0] == "profiles" and parts[1] != "tags":
            profile_id = parts[1]

    try:
        if path.endswith("/profiles/tags") and http_method == "GET":
            return _list_profile_tags()
        elif path.endswith("/activate") and http_method == "POST":
            return _activate_profile(profile_id)
        elif path.endswith("/suspend") and http_method == "POST":
            return _suspend_profile(profile_id)
        elif profile_id:
            if http_method == "GET":
                return _get_profile(profile_id)
            elif http_method == "PUT":
                return _update_profile(profile_id, event)
            elif http_method == "DELETE":
                return _delete_profile(profile_id, event)
        else:
            if http_method == "POST":
                return _create_profile(event)
            elif http_method == "GET":
                return _list_profiles(event)

        return _response(404, {"error": "Not found"})

    except Exception as exc:
        logger.exception("Unhandled error in profiles handler")
        return _response(500, {"error": str(exc)})


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def _create_profile(event):
    """POST /profiles - Create a new profile with an inference profile."""
    body = _parse_body(event)
    if not body:
        return _response(400, {"error": "Invalid or missing request body"})

    profile_name = body.get("tenant_name", "").strip()
    model_id = body.get("model_id", "").strip()
    region = body.get("region", "").strip()
    raw_tags = body.get("tags") or {}

    if not profile_name or not model_id:
        return _response(400, {"error": "tenant_name and model_id are required"})

    try:
        tags = validate_tags(raw_tags, profile_name)
    except ValueError as e:
        return _response(400, {"error": str(e)})

    profile_id = str(uuid.uuid4())
    if not region:
        region = os.environ.get("AWS_REGION", "us-east-1")

    # Create Amazon Bedrock inference profile
    inference_profile_id = ""
    inference_profile_arn = ""
    profile_strategy = "dedicated"

    bedrock_client = boto3.client("bedrock", region_name=region)
    inference_profile_name = f"isv-obs-{profile_name}-{profile_id[:8]}"

    # Resolve the model source ARN for create_inference_profile.
    # Models with a regional prefix (us., eu., ap.) are system-defined
    # inference profiles — look up their real ARN from Amazon Bedrock.
    # Plain model IDs are foundation models — construct the ARN directly.
    _REGIONAL_PREFIXES = ("us.", "eu.", "ap.")
    if any(model_id.startswith(p) for p in _REGIONAL_PREFIXES):
        try:
            profile_resp = bedrock_client.get_inference_profile(
                inferenceProfileIdentifier=model_id
            )
            model_source_arn = profile_resp["inferenceProfileArn"]
        except Exception as exc:
            logger.error("Failed to resolve inference profile %s: %s", model_id, exc)
            return _response(400, {
                "error": f"Could not find system inference profile '{model_id}': {exc}"
            })
    else:
        model_source_arn = f"arn:aws:bedrock:{region}::foundation-model/{model_id}"

    # Format tags for Amazon Bedrock API — skip empty values (Amazon Bedrock rejects them)
    bedrock_tags = []
    for k, v in tags.items():
        val = " / ".join(v) if isinstance(v, list) else str(v)
        if val:
            bedrock_tags.append({"key": k, "value": val})
    bedrock_tags.append({"key": "tenant_id", "value": profile_id})
    bedrock_tags.append({"key": "managed_by", "value": "isv-bedrock-observability"})

    logger.info("Creating inference profile with copyFrom=%s", model_source_arn)
    try:
        response = bedrock_client.create_inference_profile(
            inferenceProfileName=inference_profile_name,
            modelSource={"copyFrom": model_source_arn},
            description=f"Inference profile for {profile_name}",
            tags=bedrock_tags,
        )
        inference_profile_arn = response.get("inferenceProfileArn", "")
        inference_profile_id = inference_profile_arn.split("/")[-1] if inference_profile_arn else ""
    except bedrock_client.exceptions.ServiceQuotaExceededException:
        logger.warning(
            "ServiceQuotaExceededException for profile %s - falling back to shared strategy",
            profile_id,
        )
        profile_strategy = "shared"
    except Exception as exc:
        logger.error("Failed to create inference profile for profile %s (copyFrom=%s): %s", profile_id, model_source_arn, exc)
        return _response(500, {"error": f"Failed to create inference profile (source: {model_source_arn}): {exc}"})

    # Activate cost allocation tags in Cost Explorer (best effort)
    try:
        ce_client = boto3.client("ce", region_name="us-east-1")
        tag_keys = list(tags.keys()) + ["tenant_id", "managed_by"]
        ce_client.update_cost_allocation_tags_status(
            CostAllocationTagsStatus=[{"TagKey": k, "Status": "Active"} for k in tag_keys]
        )
        logger.info("Activated cost allocation tags: %s", tag_keys)
    except Exception as exc:
        logger.warning("Failed to activate cost allocation tags (non-fatal): %s", exc)

    capacity_limit = body.get("capacity_limit")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    profile = {
        "tenant_id": profile_id,
        "tenant_name": profile_name,
        "status": "active",
        "model_id": model_id,
        "region": region,
        "inference_profile_id": inference_profile_id,
        "inference_profile_arn": inference_profile_arn,
        "profile_strategy": profile_strategy,
        "tags": tags,
        "created_at": now,
        "updated_at": now,
    }
    if capacity_limit is not None:
        profile["capacity_limit"] = int(capacity_limit)

    dynamo_utils.put_profile(TENANTS_TABLE, profile)
    logger.info("Created profile %s (%s) with strategy=%s", profile_id, profile_name, profile_strategy)

    return _response(201, profile)


def _list_profiles(event):
    """GET /profiles - List all profiles (paginated)."""
    params = event.get("queryStringParameters") or {}
    limit = int(params.get("limit", "100"))
    last_key_raw = params.get("last_key")

    last_key = None
    if last_key_raw:
        try:
            last_key = json.loads(last_key_raw)
        except json.JSONDecodeError:
            return _response(400, {"error": "Invalid last_key parameter"})

    result = dynamo_utils.list_profiles(TENANTS_TABLE, limit=limit, last_key=last_key)

    # Apply tag filters if provided
    tag_filters_raw = params.get("tag_filters", "")
    tag_filters = parse_tag_filters(tag_filters_raw)
    profiles_list = result["profiles"]
    if tag_filters:
        profiles_list = filter_by_tags(profiles_list, tag_filters)

    response_body = {
        "profiles": profiles_list,
        "count": len(profiles_list),
    }
    if result["last_key"]:
        response_body["last_key"] = result["last_key"]

    return _response(200, response_body)


def _get_profile(profile_id):
    """GET /profiles/{profile_id} - Get a single profile."""
    profile = dynamo_utils.get_profile(TENANTS_TABLE, profile_id)
    if not profile:
        return _response(404, {"error": f"Profile {profile_id} not found"})
    return _response(200, profile)


def _update_profile(profile_id, event):
    """PUT /profiles/{profile_id} - Update profile fields."""
    body = _parse_body(event)
    if not body:
        return _response(400, {"error": "Invalid or missing request body"})

    existing = dynamo_utils.get_profile(TENANTS_TABLE, profile_id)
    if not existing:
        return _response(404, {"error": f"Profile {profile_id} not found"})

    # Only allow updating specific fields
    updatable_fields = {"tenant_name", "tags", "capacity_limit"}

    if "tags" in body:
        profile_name_for_validation = body.get("tenant_name", existing.get("tenant_name", "")).strip()
        try:
            body["tags"] = validate_tags(body["tags"], profile_name_for_validation)
        except ValueError as e:
            return _response(400, {"error": str(e)})

    for field in updatable_fields:
        if field in body:
            existing[field] = body[field]

    dynamo_utils.put_profile(TENANTS_TABLE, existing)
    logger.info("Updated profile %s", profile_id)

    return _response(200, existing)


def _delete_profile(profile_id, event):
    """DELETE /profiles/{profile_id} - Delete profile and optionally its inference profile."""
    profile = dynamo_utils.get_profile(TENANTS_TABLE, profile_id)
    if not profile:
        return _response(404, {"error": f"Profile {profile_id} not found"})

    # Optionally delete the inference profile
    params = event.get("queryStringParameters") or {}
    delete_inf_profile = params.get("delete_profile", "false").lower() == "true"

    if delete_inf_profile and profile.get("inference_profile_id"):
        try:
            region = profile.get("region", os.environ.get("AWS_REGION", "us-east-1"))
            bedrock_client = boto3.client("bedrock", region_name=region)
            bedrock_client.delete_inference_profile(
                inferenceProfileIdentifier=profile["inference_profile_id"]
            )
            logger.info("Deleted inference profile %s", profile["inference_profile_id"])
        except Exception as exc:
            logger.warning(
                "Failed to delete inference profile %s: %s",
                profile.get("inference_profile_id"),
                exc,
            )

    dynamo_utils.delete_profile(TENANTS_TABLE, profile_id)
    logger.info("Deleted profile %s", profile_id)

    return _response(200, {"message": f"Profile {profile_id} deleted"})


def _activate_profile(profile_id):
    """POST /profiles/{profile_id}/activate - Set status to active."""
    profile = dynamo_utils.get_profile(TENANTS_TABLE, profile_id)
    if not profile:
        return _response(404, {"error": f"Profile {profile_id} not found"})

    updated = dynamo_utils.update_profile_status(TENANTS_TABLE, profile_id, "active")
    logger.info("Activated profile %s", profile_id)
    return _response(200, updated)


def _suspend_profile(profile_id):
    """POST /profiles/{profile_id}/suspend - Set status to suspended."""
    profile = dynamo_utils.get_profile(TENANTS_TABLE, profile_id)
    if not profile:
        return _response(404, {"error": f"Profile {profile_id} not found"})

    updated = dynamo_utils.update_profile_status(TENANTS_TABLE, profile_id, "suspended")
    logger.info("Suspended profile %s", profile_id)
    return _response(200, updated)


def _list_profile_tags():
    """GET /profiles/tags - Return unique tag keys and their values across all profiles."""
    result = dynamo_utils.list_profiles(TENANTS_TABLE, limit=1000)
    tag_map = {}  # key -> set of values
    for t in result["profiles"]:
        for k, v in t.get("tags", {}).items():
            tag_map.setdefault(k, set()).add(v)
    return _response(200, {
        "tags": {k: sorted(v) for k, v in tag_map.items()},
        "predefined_categories": list(PREDEFINED_TAG_CATEGORIES),
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_body(event) -> dict:
    """Parse JSON body from API Gateway event."""
    body = event.get("body", "")
    if not body:
        return {}
    if isinstance(body, str):
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {}
    return body


def _response(status_code: int, body: dict) -> dict:
    """Build API Gateway proxy response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
        },
        "body": json.dumps(body, default=str),
    }
