"""
Mantle Collector Lambda

Collects model lists from the Mantle API endpoint for a single region.
Uses SigV4-signed HTTP requests to the bedrock-mantle.{region}.api.aws endpoint.
"""

import json
import logging
import os
import time
import urllib.request
import urllib.error
from typing import Any

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from shared import (
    write_to_s3,
    get_s3_client,
    validate_required_params,
    ValidationError,
)

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

MANTLE_ENDPOINT_PATTERN = "bedrock-mantle.{region}.api.aws"
REQUEST_TIMEOUT_SECONDS = 10

# Module-level session — reused across Lambda invocations for performance
_boto3_session = boto3.Session()


def call_mantle_endpoint(region: str) -> list[dict]:
    """
    Call Mantle /v1/models endpoint with SigV4 signing.

    Args:
        region: AWS region code to query.

    Returns:
        List of normalized model dicts with model_id, model_name, provider, region.

    Raises:
        Exception: On any HTTP or parsing error (caller should handle).
    """
    host = MANTLE_ENDPOINT_PATTERN.format(region=region)
    url = f"https://{host}/v1/models"

    # Create and sign the request with SigV4
    # Explicitly set Host header before signing so SigV4 includes it in
    # the signature calculation — prevents host mismatch between the
    # canonical request and what urllib actually sends.
    headers = {
        "Content-Type": "application/json",
        "Host": host,
    }
    aws_request = AWSRequest(method="GET", url=url, headers=headers)

    credentials = _boto3_session.get_credentials().get_frozen_credentials()
    signer = SigV4Auth(credentials, "bedrock", region)
    signer.add_auth(aws_request)

    # Transfer ALL signed headers to urllib request (includes Authorization,
    # X-Amz-Date, X-Amz-Security-Token, Host, etc.)
    signed_headers = dict(aws_request.headers)
    req = urllib.request.Request(url, headers=signed_headers, method="GET")

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Log the response body for debugging auth issues (e.g. 401/403)
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        logger.error(
            "Mantle HTTP error in %s: %s %s | Body: %s",
            region,
            e.code,
            e.reason,
            body,
        )
        raise

    # Handle both {"data": [...]} and flat array responses
    if isinstance(data, dict):
        models_raw = data.get("data", [])
    elif isinstance(data, list):
        models_raw = data
    else:
        models_raw = []

    if not isinstance(models_raw, list):
        models_raw = []

    return [
        {
            "model_id": m.get("id", ""),
            "model_name": m.get("id", "").split(".")[-1] if m.get("id") else "",
            "provider": m.get("owned_by", ""),
            "region": region,
        }
        for m in models_raw
        if m.get("id")
    ]


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for Mantle model collection (single region).

    Invoked per-region by the Step Functions Map state.

    Input:
        {
            "region": "us-east-1",
            "s3Bucket": "bucket-name",
            "s3Key": "executions/{id}/mantle/us-east-1.json"
        }

    Output (success):
        {
            "status": "SUCCESS",
            "region": "us-east-1",
            "s3Key": "executions/{id}/mantle/us-east-1.json",
            "mantleModelCount": 5,
            "durationMs": 1200
        }

    Output (failure):
        {
            "status": "FAILED",
            "region": "us-east-1",
            "errorType": "ConnectionError",
            "errorMessage": "Mantle endpoint not available in us-east-1",
            "retryable": false
        }
    """
    start_time = time.time()

    # Validate required parameters
    try:
        validate_required_params(event, ["region"], "MantleCollector")
    except ValidationError as e:
        return {
            "status": "FAILED",
            "errorType": "ValidationError",
            "errorMessage": str(e),
            "retryable": False,
        }

    region = event["region"]
    s3_bucket = event.get("s3Bucket")
    s3_key = event.get("s3Key", f"test/mantle/{region}.json")
    dry_run = event.get("dryRun", False)

    logger.info(f"Collecting Mantle models from region: {region}")

    try:
        models = call_mantle_endpoint(region)
        logger.info(f"Mantle: {region} returned {len(models)} models")

        output_data = {
            "metadata": {
                "region": region,
                "mantle_model_count": len(models),
                "collection_timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "endpoint": MANTLE_ENDPOINT_PATTERN.format(region=region),
            },
            "mantle_models": models,
        }

        if not dry_run and s3_bucket:
            s3_client = get_s3_client()
            write_to_s3(s3_client, s3_bucket, s3_key, output_data)
        else:
            logger.info(
                f"Dry run - would write {len(models)} Mantle models "
                f"to s3://{s3_bucket}/{s3_key}"
            )

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            "status": "SUCCESS",
            "region": region,
            "s3Key": s3_key,
            "mantleModelCount": len(models),
            "durationMs": duration_ms,
        }

    except urllib.error.HTTPError as e:
        # HTTPError is a subclass of URLError — must be caught first
        duration_ms = int((time.time() - start_time) * 1000)
        is_retryable = e.code >= 500
        log_level = logging.WARNING if is_retryable else logging.ERROR
        logger.log(log_level, f"Mantle HTTP error in {region}: {e.code} {e.reason}")
        return {
            "status": "FAILED",
            "region": region,
            "errorType": "HTTPError",
            "errorMessage": f"HTTP {e.code}: {e.reason}",
            "retryable": is_retryable,
            "durationMs": duration_ms,
        }

    except urllib.error.URLError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        reason = str(e.reason) if hasattr(e, "reason") else str(e)
        logger.debug(f"Mantle not available in {region}: URLError: {reason}")
        return {
            "status": "FAILED",
            "region": region,
            "errorType": "URLError",
            "errorMessage": f"Mantle endpoint not available in {region}: {reason}",
            "retryable": False,
            "durationMs": duration_ms,
        }

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.warning(
            f"Unexpected error collecting Mantle models from {region}: "
            f"{type(e).__name__}: {e}"
        )
        return {
            "status": "FAILED",
            "region": region,
            "errorType": type(e).__name__,
            "errorMessage": str(e),
            "retryable": "Throttling" in str(e),
            "durationMs": duration_ms,
        }
