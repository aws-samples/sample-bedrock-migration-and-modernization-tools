"""
Feature Collector Lambda

Collects inference profiles and enhanced features from a single region.
Supports reading from cache (written by region-discovery) to avoid duplicate API calls.
"""

import time
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from shared import (
    RETRY_CONFIG,
    write_to_s3,
    read_from_s3,
    parse_execution_id,
    validate_required_params,
    ValidationError,
    S3WriteError,
    S3ReadError,
)
from shared.powertools import logger, tracer, metrics, LambdaContext
from aws_lambda_powertools.metrics import MetricUnit


def get_bedrock_client(region: str):
    """Create Bedrock client for a specific region."""
    return boto3.client("bedrock", region_name=region, config=RETRY_CONFIG)


def get_s3_client():
    return boto3.client("s3", config=RETRY_CONFIG)


@tracer.capture_method
def read_cached_profiles(
    s3_client, bucket: str, cache_key: str, region: str
) -> Optional[list[dict]]:
    """
    Read cached inference profiles from S3.

    Returns:
        List of profile dicts if cache is valid, None otherwise.
    """
    try:
        cache_data = read_from_s3(s3_client, bucket, cache_key)

        # Validate cache data structure
        if not isinstance(cache_data, dict):
            logger.warning(
                "Invalid cache data structure", extra={"cache_key": cache_key}
            )
            return None

        # Verify the cache is for the correct region
        cached_region = cache_data.get("region")
        if cached_region != region:
            logger.warning(
                "Cache region mismatch",
                extra={
                    "expected": region,
                    "cached": cached_region,
                    "cache_key": cache_key,
                },
            )
            return None

        profiles = cache_data.get("profiles", [])
        if not isinstance(profiles, list):
            logger.warning("Invalid profiles in cache", extra={"cache_key": cache_key})
            return None

        logger.info(
            "Read profiles from cache",
            extra={
                "region": region,
                "count": len(profiles),
                "cache_key": cache_key,
                "cache_timestamp": cache_data.get("timestamp"),
            },
        )
        return profiles

    except S3ReadError as e:
        logger.warning(
            "Cache miss - will call API",
            extra={"region": region, "cache_key": cache_key, "error": str(e)},
        )
        return None
    except Exception as e:
        logger.warning(
            "Error reading cache - will call API",
            extra={"region": region, "cache_key": cache_key, "error": str(e)},
        )
        return None


@tracer.capture_method
def collect_inference_profiles(bedrock_client, region: str) -> list[dict]:
    """
    Collect inference profiles from Bedrock API.

    Returns list of inference profile dictionaries.
    """
    profiles = []

    try:
        # List inference profiles
        paginator = bedrock_client.get_paginator("list_inference_profiles")

        for page in paginator.paginate():
            for profile in page.get("inferenceProfileSummaries", []):
                normalized = {
                    "inferenceProfileId": profile.get("inferenceProfileId", ""),
                    "inferenceProfileArn": profile.get("inferenceProfileArn", ""),
                    "inferenceProfileName": profile.get("inferenceProfileName", ""),
                    "description": profile.get("description", ""),
                    "status": profile.get("status", ""),
                    "type": profile.get("type", ""),
                    "models": profile.get("models", []),
                    "region": region,
                }
                profiles.append(normalized)

        logger.info(
            "Collected inference profiles",
            extra={"region": region, "count": len(profiles)},
        )

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code in ("AccessDeniedException", "UnrecognizedClientException"):
            logger.warning(
                "Access denied or region not enabled",
                extra={"region": region, "error_code": error_code},
            )
        elif error_code == "ValidationException":
            logger.warning("Inference profiles not available", extra={"region": region})
        elif error_code == "InvalidIdentityToken":
            logger.warning(
                "Invalid token - region may require opt-in", extra={"region": region}
            )
        else:
            logger.error(
                "Error collecting inference profiles",
                extra={"region": region, "error": str(e)},
            )
            # Don't raise - continue with empty profiles

    except Exception as e:
        logger.warning(
            "Unexpected error collecting inference profiles",
            extra={"region": region, "error": str(e)},
        )

    return profiles


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    """
    Lambda handler for feature collection.

    Input:
        {
            "region": "us-east-1",
            "s3Bucket": "bucket-name",
            "s3Key": "executions/{id}/features/us-east-1.json",
            "cacheKey": "executions/{id}/cache/inference_profiles_us-east-1.json"  # optional
        }

    Output:
        {
            "status": "SUCCESS",
            "region": "us-east-1",
            "s3Key": "executions/{id}/features/us-east-1.json",
            "inferenceProfileCount": 12,
            "fromCache": true|false
        }
    """
    start_time = time.time()

    # Validate required parameters
    try:
        validate_required_params(event, ["region"], "FeatureCollector")
    except ValidationError as e:
        return {
            "status": "FAILED",
            "errorType": "ValidationError",
            "errorMessage": str(e),
        }

    region = event["region"]
    s3_bucket = event.get("s3Bucket")
    s3_key = event.get("s3Key", f"test/features/{region}.json")
    dry_run = event.get("dryRun", False)

    # Get cache key - either directly passed or look up from inferenceProfileCacheKeys map
    cache_key = event.get("cacheKey")
    if not cache_key:
        cache_keys_map = event.get("inferenceProfileCacheKeys", {})
        cache_key = (
            cache_keys_map.get(region) if isinstance(cache_keys_map, dict) else None
        )

    logger.info(
        "Starting feature collection",
        extra={
            "region": region,
            "cache_key": cache_key,
            "cache_available": bool(cache_key and s3_bucket),
        },
    )

    try:
        s3_client = get_s3_client()
        profiles = None
        from_cache = False

        # Try to read from cache first if cache key is provided
        if cache_key and s3_bucket:
            cached_profiles = read_cached_profiles(
                s3_client, s3_bucket, cache_key, region
            )
            if cached_profiles is not None:
                # Normalize cached profiles to match expected format
                profiles = []
                for profile in cached_profiles:
                    normalized = {
                        "inferenceProfileId": profile.get("inferenceProfileId", ""),
                        "inferenceProfileArn": profile.get("inferenceProfileArn", ""),
                        "inferenceProfileName": profile.get("inferenceProfileName", ""),
                        "description": profile.get("description", ""),
                        "status": profile.get("status", ""),
                        "type": profile.get("type", ""),
                        "models": profile.get("models", []),
                        "region": region,
                    }
                    profiles.append(normalized)
                from_cache = True
                logger.info(
                    "Using cached profiles",
                    extra={"region": region, "count": len(profiles)},
                )

        # Fall back to API call if cache miss or not available
        if profiles is None:
            bedrock_client = get_bedrock_client(region)
            profiles = collect_inference_profiles(bedrock_client, region)
            logger.info(
                "Fetched profiles from API",
                extra={"region": region, "count": len(profiles)},
            )

        output_data = {
            "metadata": {
                "region": region,
                "inference_profile_count": len(profiles),
                "collection_timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "from_cache": from_cache,
            },
            "inference_profiles": profiles,
        }

        if not dry_run and s3_bucket:
            write_to_s3(s3_client, s3_bucket, s3_key, output_data)
        else:
            logger.info(
                "Dry run - would write profiles",
                extra={"count": len(profiles), "bucket": s3_bucket, "key": s3_key},
            )

        duration_ms = int((time.time() - start_time) * 1000)

        # Emit metrics
        metrics.add_metric(
            name="ProfilesCollected", unit=MetricUnit.Count, value=len(profiles)
        )
        metrics.add_metric(
            name="CacheHits", unit=MetricUnit.Count, value=1 if from_cache else 0
        )
        metrics.add_dimension(name="Region", value=region)

        logger.info(
            "Feature collection complete",
            extra={
                "region": region,
                "profile_count": len(profiles),
                "from_cache": from_cache,
                "duration_ms": duration_ms,
            },
        )

        return {
            "status": "SUCCESS",
            "region": region,
            "s3_key": s3_key,
        }

    except Exception as e:
        logger.exception(
            "Failed to collect features", extra={"region": region, "error": str(e)}
        )
        return {
            "status": "FAILED",
            "region": region,
            "errorType": type(e).__name__,
            "errorMessage": str(e),
            "retryable": "Throttling" in str(e),
        }
