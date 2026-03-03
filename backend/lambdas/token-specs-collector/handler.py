"""
Token Specs Collector Lambda

Fetches token specifications (context window, max output) from LiteLLM.
Works with the correct snake_case schema.

Includes TTL-based caching to reduce external API calls.
"""

import json
import time
from datetime import datetime
from typing import Any, Callable, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import boto3
from botocore.config import Config

from shared.powertools import logger, tracer, metrics, LambdaContext
from shared import get_config_loader
from aws_lambda_powertools.metrics import MetricUnit

RETRY_CONFIG = Config(
    retries={"max_attempts": 3, "mode": "adaptive"}, connect_timeout=10, read_timeout=30
)

# Cache configuration
LITELLM_CACHE_KEY = "cache/litellm_model_prices.json"
DEFAULT_CACHE_TTL_HOURS = 24


def get_litellm_urls() -> dict:
    """Get LiteLLM URLs from config."""
    config = get_config_loader()
    return {
        "primary": config.get_litellm_url(),
        "fallback": config.get_litellm_fallback_url(),
    }


def get_s3_client():
    return boto3.client("s3", config=RETRY_CONFIG)


def read_from_s3(
    s3_client, bucket: str, key: str, default_on_missing: Optional[dict] = None
) -> Optional[dict]:
    """Read JSON data from S3.

    Args:
        s3_client: Boto3 S3 client instance.
        bucket: S3 bucket name.
        key: S3 object key.
        default_on_missing: If provided, return this value when object doesn't exist.

    Returns:
        Parsed JSON data as a dictionary, or default_on_missing if object not found.
    """
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except s3_client.exceptions.NoSuchKey:
        if default_on_missing is not None:
            return default_on_missing
        raise
    except Exception as e:
        # Check if it's a NoSuchKey error from ClientError
        if (
            hasattr(e, "response")
            and e.response.get("Error", {}).get("Code") == "NoSuchKey"
        ):
            if default_on_missing is not None:
                return default_on_missing
        raise


def write_to_s3(s3_client, bucket: str, key: str, data: dict) -> None:
    """Write JSON data to S3."""
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2, default=str),
        ContentType="application/json",
    )
    logger.info("Written to S3", extra={"bucket": bucket, "key": key})


def get_cached_or_fetch(
    s3_client: Any,
    bucket: str,
    cache_key: str,
    fetch_fn: Callable[[], dict],
    source_url: str,
    ttl_hours: int = DEFAULT_CACHE_TTL_HOURS,
) -> Tuple[dict, bool]:
    """
    Get data from cache if valid, otherwise fetch and cache.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        cache_key: S3 key for cache file
        fetch_fn: Function to call if cache miss (returns data dict)
        source_url: URL of the data source (for metadata)
        ttl_hours: Cache TTL in hours (default 24)

    Returns:
        Tuple of (data, from_cache)
    """
    try:
        cached = read_from_s3(s3_client, bucket, cache_key, default_on_missing=None)
        if cached:
            cached_at_str = cached.get("cached_at")
            if cached_at_str:
                cached_at = datetime.strptime(cached_at_str, "%Y-%m-%dT%H:%M:%SZ")
                age_hours = (datetime.utcnow() - cached_at).total_seconds() / 3600
                cache_ttl = cached.get("ttl_hours", ttl_hours)

                if age_hours < cache_ttl:
                    logger.info(
                        "Cache hit - using cached data",
                        extra={
                            "cache_key": cache_key,
                            "age_hours": round(age_hours, 1),
                            "ttl_hours": cache_ttl,
                        },
                    )
                    return cached.get("data", {}), True
                else:
                    logger.info(
                        "Cache expired - fetching fresh data",
                        extra={
                            "cache_key": cache_key,
                            "age_hours": round(age_hours, 1),
                            "ttl_hours": cache_ttl,
                        },
                    )
    except Exception as e:
        logger.warning(
            "Cache read failed, fetching fresh data",
            extra={"cache_key": cache_key, "error": str(e)},
        )

    # Cache miss or expired - fetch fresh data
    logger.info("Fetching fresh data from source", extra={"source_url": source_url})
    data = fetch_fn()

    # Update cache
    cache_data = {
        "cached_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ttl_hours": ttl_hours,
        "source_url": source_url,
        "data": data,
    }
    try:
        write_to_s3(s3_client, bucket, cache_key, cache_data)
        logger.info("Cache updated", extra={"cache_key": cache_key})
    except Exception as e:
        logger.warning(
            "Cache write failed - continuing without caching",
            extra={"cache_key": cache_key, "error": str(e)},
        )

    return data, False


@tracer.capture_method
def fetch_litellm_data() -> dict:
    """Fetch model data from LiteLLM GitHub repository with fallback support."""
    urls = get_litellm_urls()

    # Try primary URL first
    try:
        request = Request(
            urls["primary"],
            headers={
                "User-Agent": "BedrockProfiler/1.0",
                "Cache-Control": "no-cache, no-store",
                "Pragma": "no-cache",
            },
        )
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            logger.info(
                "Fetched models from LiteLLM (primary URL)", extra={"count": len(data)}
            )
            return data
    except (URLError, HTTPError) as e:
        logger.warning(
            "Primary LiteLLM URL failed, trying fallback", extra={"error": str(e)}
        )

    # Try fallback URL
    try:
        request = Request(
            urls["fallback"],
            headers={
                "User-Agent": "BedrockProfiler/1.0",
                "Cache-Control": "no-cache, no-store",
                "Pragma": "no-cache",
            },
        )
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            logger.info(
                "Fetched models from LiteLLM (fallback URL)", extra={"count": len(data)}
            )
            return data
    except (URLError, HTTPError) as e:
        logger.error(
            "Failed to fetch LiteLLM data from both URLs", extra={"error": str(e)}
        )
        return {}


def filter_bedrock_models(litellm_data: dict) -> dict:
    """Filter LiteLLM data to only include Bedrock models."""
    bedrock_models = {}

    for model_key, model_data in litellm_data.items():
        # Check if it's a Bedrock model
        # Include models with 'bedrock' in key or provider containing 'bedrock'
        litellm_provider = model_data.get("litellm_provider", "")
        is_bedrock = (
            "bedrock" in model_key.lower() or "bedrock" in litellm_provider.lower()
            if litellm_provider
            else False
        )
        if is_bedrock:
            bedrock_models[model_key] = model_data

    logger.info(
        "Filtered Bedrock models from LiteLLM data",
        extra={"count": len(bedrock_models)},
    )
    return bedrock_models


def match_token_specs(models_data: dict, litellm_bedrock: dict) -> dict:
    """
    Match token specs from LiteLLM to our models.

    Returns dict of model_id -> token_specs (in snake_case)
    """
    token_specs = {}

    # Build lookup maps for flexible matching
    litellm_lookup = {}
    for key, data in litellm_bedrock.items():
        # Extract just the model portion from keys like "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
        normalized = key.lower()
        if "/" in normalized:
            normalized = normalized.split("/")[-1]

        # Use snake_case for output schema
        litellm_lookup[normalized] = {
            "context_window": data.get("max_input_tokens") or data.get("max_tokens"),
            "max_output_tokens": data.get("max_output_tokens"),
            "source": "litellm",
            "original_key": key,
            "litellm_verified": True,
        }

    # Match against our models
    for provider, provider_data in models_data.get("providers", {}).items():
        for model_id, model in provider_data.get("models", {}).items():
            model_id_normalized = model_id.lower()

            # Try exact match first
            if model_id_normalized in litellm_lookup:
                token_specs[model_id] = litellm_lookup[model_id_normalized]
                continue

            # Try partial matching
            for litellm_key, specs in litellm_lookup.items():
                if (
                    model_id_normalized in litellm_key
                    or litellm_key in model_id_normalized
                ):
                    token_specs[model_id] = specs
                    break

    return token_specs


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    """
    Lambda handler for token specs collection.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "modelsS3Key": "executions/{id}/merged/models.json"
        }

    Output:
        {
            "status": "SUCCESS",
            "s3Key": "executions/{id}/intermediate/token-specs.json",
            "modelsWithSpecs": 104,
            "modelsWithoutSpecs": 4,
            "fromCache": true
        }
    """
    start_time = time.time()

    s3_bucket = event["s3Bucket"]
    execution_id = event["executionId"]
    models_s3_key = event["modelsS3Key"]
    dry_run = event.get("dryRun", False)

    if ":" in execution_id:
        execution_id = execution_id.split(":")[-1]

    output_key = f"executions/{execution_id}/intermediate/token-specs.json"

    logger.info("Starting token specs collection")

    try:
        s3_client = get_s3_client()

        # Get LiteLLM source URL for cache metadata
        urls = get_litellm_urls()
        source_url = urls["primary"]

        # Fetch LiteLLM data with caching
        litellm_data, from_cache = get_cached_or_fetch(
            s3_client=s3_client,
            bucket=s3_bucket,
            cache_key=LITELLM_CACHE_KEY,
            fetch_fn=fetch_litellm_data,
            source_url=source_url,
            ttl_hours=DEFAULT_CACHE_TTL_HOURS,
        )

        bedrock_models = filter_bedrock_models(litellm_data)

        if not dry_run:
            # Read our models
            models_data = read_from_s3(s3_client, s3_bucket, models_s3_key)

            if models_data is None:
                raise ValueError(f"Models data not found at {models_s3_key}")

            # Match token specs
            token_specs = match_token_specs(models_data, bedrock_models)

            # Count statistics
            total_models = sum(
                len(p["models"]) for p in models_data.get("providers", {}).values()
            )
            models_with_specs = len(token_specs)
            models_without_specs = total_models - models_with_specs

            output_data = {
                "metadata": {
                    "models_with_specs": models_with_specs,
                    "models_without_specs": models_without_specs,
                    "litellm_models_available": len(bedrock_models),
                    "source": "litellm",
                    "from_cache": from_cache,
                    "collection_timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                },
                "token_specs": token_specs,
            }

            write_to_s3(s3_client, s3_bucket, output_key, output_data)
        else:
            logger.info("Dry run - skipping S3 operations")
            models_with_specs = len(bedrock_models)
            models_without_specs = 0

        duration_ms = int((time.time() - start_time) * 1000)

        # Emit metrics
        metrics.add_metric(
            name="TokenSpecsCollected", unit=MetricUnit.Count, value=models_with_specs
        )

        logger.info(
            "Token specs collection complete",
            extra={
                "models_with_specs": models_with_specs,
                "models_without_specs": models_without_specs,
                "from_cache": from_cache,
                "duration_ms": duration_ms,
            },
        )

        return {
            "status": "SUCCESS",
            "s3Key": output_key,
            "modelsWithSpecs": models_with_specs,
            "modelsWithoutSpecs": models_without_specs,
            "litellmModelsAvailable": len(bedrock_models),
            "fromCache": from_cache,
            "durationMs": duration_ms,
        }

    except Exception as e:
        logger.exception("Failed to collect token specs", extra={"error": str(e)})
        return {
            "status": "FAILED",
            "errorType": type(e).__name__,
            "errorMessage": str(e),
        }
