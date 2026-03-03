"""
Mantle Collector Lambda

Collects model lists from the Mantle API endpoint for a single region.
Uses SigV4-signed HTTP requests to the bedrock-mantle.{region}.api.aws endpoint.
"""

import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from shared import (
    write_to_s3,
    get_s3_client,
    validate_required_params,
    ValidationError,
)
from shared.powertools import logger, tracer, metrics, LambdaContext
from aws_lambda_powertools.metrics import MetricUnit

MANTLE_ENDPOINT_PATTERN = "bedrock-mantle.{region}.api.aws"
REQUEST_TIMEOUT_SECONDS = 10

# Module-level session — reused across Lambda invocations for performance
_boto3_session = boto3.Session()


@tracer.capture_method
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


def probe_responses_api(model_id: str, region: str) -> bool:
    """Probe whether a Mantle model supports the Responses API.

    Sends POST /v1/responses with {"model": model_id} (no input).
    This is free — no tokens consumed. The response pattern tells us:
    - HTTP 200 + error.code "invalid_prompt" = SUPPORTED (model accepted, input validation failed)
    - HTTP 400 + error.code "validation_error" = NOT SUPPORTED
    - HTTP 404 = model not found (shouldn't happen for known models)

    Args:
        model_id: The Mantle model ID to probe.
        region: AWS region code.

    Returns:
        True if the model supports the Responses API, False otherwise.
    """
    host = MANTLE_ENDPOINT_PATTERN.format(region=region)
    url = f"https://{host}/v1/responses"
    body = json.dumps({"model": model_id}).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Host": host,
    }
    aws_request = AWSRequest(method="POST", url=url, headers=headers, data=body)

    credentials = _boto3_session.get_credentials().get_frozen_credentials()
    signer = SigV4Auth(credentials, "bedrock", region)
    signer.add_auth(aws_request)

    signed_headers = dict(aws_request.headers)
    req = urllib.request.Request(url, data=body, headers=signed_headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            response_body = json.loads(response.read().decode("utf-8"))
            # HTTP 200 + error.code "invalid_prompt" → model supports Responses API
            error_code = response_body.get("error", {}).get("code", "")
            if error_code == "invalid_prompt":
                return True
            # HTTP 200 with no error or unexpected shape — treat as supported
            # (the endpoint accepted the model, just rejected the empty input)
            return True
    except urllib.error.HTTPError as e:
        try:
            error_body = json.loads(e.read().decode("utf-8"))
        except Exception:
            logger.debug(
                "Responses API probe for %s: HTTP %s, unreadable body",
                model_id,
                e.code,
            )
            return False

        error_code = error_body.get("error", {}).get("code", "")

        if e.code == 400 and error_code == "validation_error":
            # Model does NOT support Responses API
            return False
        if e.code == 404 and error_code == "not_found_error":
            # Model not found in Mantle (unexpected for known models)
            logger.warning(
                "Responses API probe: model %s not found (404) in %s",
                model_id,
                region,
            )
            return False

        # Any other HTTP error — default to False
        logger.debug(
            "Responses API probe for %s: HTTP %s, error_code=%s",
            model_id,
            e.code,
            error_code,
        )
        return False
    except Exception as e:
        logger.debug(
            "Responses API probe for %s failed: %s: %s",
            model_id,
            type(e).__name__,
            e,
        )
        return False


def probe_all_responses_support(
    model_ids: list[str], region: str, max_workers: int = 10
) -> dict[str, bool]:
    """Probe Responses API support for all models in parallel.

    Args:
        model_ids: List of Mantle model IDs to probe.
        region: AWS region code.
        max_workers: Maximum number of concurrent probe threads.

    Returns:
        Dict mapping model_id → True (supports Responses API) / False.
    """
    results: dict[str, bool] = {}

    if not model_ids:
        return results

    with ThreadPoolExecutor(max_workers=min(max_workers, len(model_ids))) as executor:
        future_to_model = {
            executor.submit(probe_responses_api, mid, region): mid for mid in model_ids
        }
        for future in as_completed(future_to_model):
            model_id = future_to_model[future]
            try:
                results[model_id] = future.result()
            except Exception as e:
                logger.debug(
                    "Responses API probe future failed for %s: %s",
                    model_id,
                    e,
                )
                results[model_id] = False

    return results


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
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

    logger.info("Starting Mantle collection", extra={"region": region})

    try:
        models = call_mantle_endpoint(region)
        logger.info(
            "Mantle models retrieved", extra={"region": region, "count": len(models)}
        )

        # Probe Responses API support for each model
        model_ids = [m["model_id"] for m in models]
        probe_start = time.time()
        responses_support = probe_all_responses_support(model_ids, region)
        probe_duration_ms = int((time.time() - probe_start) * 1000)

        supported_count = sum(1 for v in responses_support.values() if v)
        logger.info(
            "Responses API probe complete",
            extra={
                "region": region,
                "supported": supported_count,
                "total": len(model_ids),
                "duration_ms": probe_duration_ms,
            },
        )

        # Enrich each model dict with Responses API support flag
        for model in models:
            model["supports_responses_api"] = responses_support.get(
                model["model_id"], False
            )

        output_data = {
            "metadata": {
                "region": region,
                "mantle_model_count": len(models),
                "collection_timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "endpoint": MANTLE_ENDPOINT_PATTERN.format(region=region),
                "responses_api_probe": {
                    "probed": len(model_ids),
                    "supported": supported_count,
                    "duration_ms": probe_duration_ms,
                },
            },
            "mantle_models": models,
        }

        if not dry_run and s3_bucket:
            s3_client = get_s3_client()
            write_to_s3(s3_client, s3_bucket, s3_key, output_data)
        else:
            logger.info(
                "Dry run - would write Mantle models",
                extra={"count": len(models), "bucket": s3_bucket, "key": s3_key},
            )

        duration_ms = int((time.time() - start_time) * 1000)

        # Emit metrics
        metrics.add_metric(
            name="MantleModelsCollected", unit=MetricUnit.Count, value=len(models)
        )
        metrics.add_dimension(name="Region", value=region)

        logger.info(
            "Mantle collection complete",
            extra={
                "region": region,
                "model_count": len(models),
                "duration_ms": duration_ms,
            },
        )

        return {
            "status": "SUCCESS",
            "region": region,
            "s3Key": s3_key,
        }

    except urllib.error.HTTPError as e:
        # HTTPError is a subclass of URLError — must be caught first
        duration_ms = int((time.time() - start_time) * 1000)
        is_retryable = e.code >= 500
        if is_retryable:
            logger.warning(
                "Mantle HTTP error (retryable)",
                extra={"region": region, "code": e.code, "reason": e.reason},
            )
        else:
            logger.error(
                "Mantle HTTP error",
                extra={"region": region, "code": e.code, "reason": e.reason},
            )
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
        logger.debug("Mantle not available", extra={"region": region, "reason": reason})
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
            "Unexpected error collecting Mantle models",
            extra={"region": region, "error_type": type(e).__name__, "error": str(e)},
        )
        return {
            "status": "FAILED",
            "region": region,
            "errorType": type(e).__name__,
            "errorMessage": str(e),
            "retryable": "Throttling" in str(e),
            "durationMs": duration_ms,
        }
