"""
Region Discovery Lambda

Dynamically discovers all AWS regions where Bedrock inference profiles are available.
This replaces hardcoded region lists with dynamic discovery.

Also caches the full inference profiles response to S3 for downstream consumers
(e.g., feature-collector) to avoid duplicate API calls.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from shared.powertools import logger, tracer, metrics, LambdaContext
from shared.config import RETRY_CONFIG
from aws_lambda_powertools.metrics import MetricUnit

# EC2 region for DescribeRegions API call (configurable via environment variable)
EC2_REGION = os.environ.get("EC2_REGION", "us-east-1")

# S3 client singleton for caching profiles
_s3_client = None


def get_s3_client():
    """Get or create S3 client singleton."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", config=RETRY_CONFIG)
    return _s3_client


def write_cache_to_s3(bucket: str, key: str, data: dict) -> bool:
    """
    Write cache data to S3.

    Returns True if successful, False otherwise.
    """
    try:
        s3 = get_s3_client()
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, indent=2, default=str),
            ContentType="application/json",
        )
        logger.debug("Cached data written to S3", extra={"bucket": bucket, "key": key})
        return True
    except ClientError as e:
        logger.warning(
            "Failed to write cache to S3",
            extra={"bucket": bucket, "key": key, "error": str(e)},
        )
        return False


@tracer.capture_method
def get_all_enabled_regions() -> list[str]:
    """Get all regions enabled in this AWS account."""
    ec2 = boto3.client("ec2", region_name=EC2_REGION)

    try:
        response = ec2.describe_regions(
            AllRegions=False,  # Only enabled regions
            Filters=[
                {"Name": "opt-in-status", "Values": ["opt-in-not-required", "opted-in"]}
            ],
        )
        regions = [r["RegionName"] for r in response.get("Regions", [])]
        logger.info("Found enabled regions", extra={"region_count": len(regions)})
        return regions
    except ClientError as e:
        logger.error("Error getting regions", extra={"error": str(e)})
        # Fallback to common regions
        return [
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "eu-central-1",
            "eu-north-1",
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "sa-east-1",
        ]


@tracer.capture_method
def check_bedrock_available(
    region: str, s3_bucket: Optional[str] = None, execution_id: Optional[str] = None
) -> tuple[str, bool, Optional[str]]:
    """
    Check if Bedrock inference profiles are available in a region.

    If S3 bucket and execution_id are provided, caches the full inference profiles
    response to S3 for downstream consumers.

    Returns:
        Tuple of (region, is_available, cache_key or None)
    """
    try:
        bedrock = boto3.client("bedrock", region_name=region)

        # Fetch ALL inference profiles (not just maxResults=1)
        profiles = []
        paginator = bedrock.get_paginator("list_inference_profiles")
        for page in paginator.paginate():
            profiles.extend(page.get("inferenceProfileSummaries", []))

        # Cache the profiles if S3 info provided and we got profiles
        cache_key = None
        if s3_bucket and execution_id and profiles:
            cache_key = (
                f"executions/{execution_id}/cache/inference_profiles_{region}.json"
            )
            cache_data = {
                "region": region,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "profileCount": len(profiles),
                "profiles": profiles,
            }
            if write_cache_to_s3(s3_bucket, cache_key, cache_data):
                logger.debug(
                    "Cached inference profiles",
                    extra={
                        "region": region,
                        "count": len(profiles),
                        "cache_key": cache_key,
                    },
                )
            else:
                cache_key = None  # Reset if write failed

        return (region, True, cache_key)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("UnrecognizedClientException", "InvalidIdentityToken"):
            # Region not enabled or Bedrock not available
            logger.debug(
                "Bedrock not available in region",
                extra={"region": region, "error_code": error_code},
            )
            return (region, False, None)
        elif error_code == "AccessDeniedException":
            # Bedrock exists but we don't have access - still count it
            logger.debug("Bedrock exists but access denied", extra={"region": region})
            return (region, True, None)
        else:
            logger.warning(
                "Error checking Bedrock in region",
                extra={"region": region, "error": str(e)},
            )
            return (region, False, None)
    except Exception as e:
        logger.warning(
            "Unexpected error checking region",
            extra={"region": region, "error": str(e)},
        )
        return (region, False, None)


@tracer.capture_method
def discover_bedrock_regions(
    all_regions: list[str],
    s3_bucket: Optional[str] = None,
    execution_id: Optional[str] = None,
) -> tuple[list[str], dict[str, str]]:
    """
    Discover which regions have Bedrock inference profiles available.

    If S3 bucket and execution_id are provided, caches the full inference profiles
    to S3 for each region.

    Returns:
        Tuple of (list of regions with Bedrock, dict mapping region to cache key)
    """
    bedrock_regions = []
    cache_keys = {}

    # Check regions in parallel for speed
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(
                check_bedrock_available, region, s3_bucket, execution_id
            ): region
            for region in all_regions
        }

        for future in as_completed(futures):
            region, available, cache_key = future.result()
            if available:
                bedrock_regions.append(region)
                if cache_key:
                    cache_keys[region] = cache_key

    # Sort for consistent ordering
    bedrock_regions.sort()
    logger.info(
        "Regions with Bedrock discovered",
        extra={"region_count": len(bedrock_regions), "cached_regions": len(cache_keys)},
    )
    return bedrock_regions, cache_keys


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    """
    Lambda handler for region discovery.

    Input:
        {
            "s3Bucket": "optional-bucket-name",  # If provided, caches profiles
            "executionId": "optional-execution-id"  # Required if s3Bucket provided
        }

    Output:
        {
            "status": "SUCCESS",
            "featureRegions": ["us-east-1", "us-west-2", ...],
            "totalRegions": 27,
            "discoveryTimestamp": "2024-01-01T00:00:00Z",
            "inferenceProfileCacheKeys": {
                "us-east-1": "executions/{id}/cache/inference_profiles_us-east-1.json",
                ...
            }
        }
    """
    start_time = time.time()

    # Extract optional S3 caching parameters
    s3_bucket = event.get("s3Bucket")
    execution_id = event.get("executionId")

    # Parse execution ID from ARN format if needed
    if execution_id and ":" in execution_id:
        # Format: arn:aws:states:region:account:execution:state-machine:execution-name
        execution_id = execution_id.split(":")[-1]

    logger.info(
        "Starting region discovery",
        extra={
            "caching_enabled": bool(s3_bucket and execution_id),
            "s3_bucket": s3_bucket,
            "execution_id": execution_id,
        },
    )

    try:
        # Get all enabled regions in the account
        all_regions = get_all_enabled_regions()
        logger.info(
            "Checking enabled regions for Bedrock availability",
            extra={"region_count": len(all_regions)},
        )

        # Filter to regions with Bedrock inference profiles
        # Also cache profiles if S3 info provided
        bedrock_regions, cache_keys = discover_bedrock_regions(
            all_regions, s3_bucket, execution_id
        )

        elapsed = time.time() - start_time

        metrics.add_metric(
            name="RegionsDiscovered", unit=MetricUnit.Count, value=len(bedrock_regions)
        )
        metrics.add_metric(
            name="ProfilesCached", unit=MetricUnit.Count, value=len(cache_keys)
        )
        logger.info(
            "Region discovery complete",
            extra={
                "regions_discovered": len(bedrock_regions),
                "all_enabled_regions": len(all_regions),
                "profiles_cached": len(cache_keys),
                "elapsed_seconds": round(elapsed, 2),
            },
        )

        return {
            "status": "SUCCESS",
            "featureRegions": bedrock_regions,
            "totalRegions": len(bedrock_regions),
            "allEnabledRegions": len(all_regions),
            "discoveryTimestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "inferenceProfileCacheKeys": cache_keys,
        }

    except Exception as e:
        logger.error(
            "Region discovery failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        metrics.add_metric(name="RegionDiscoveryErrors", unit=MetricUnit.Count, value=1)
        return {
            "status": "FAILED",
            "errorType": type(e).__name__,
            "errorMessage": str(e),
            # Fallback to known good regions
            "featureRegions": [
                "us-east-1",
                "us-east-2",
                "us-west-1",
                "us-west-2",
                "eu-west-1",
                "eu-west-2",
                "eu-west-3",
                "eu-central-1",
                "eu-north-1",
                "ap-northeast-1",
                "ap-northeast-2",
                "ap-south-1",
                "ap-southeast-1",
                "ap-southeast-2",
                "ca-central-1",
                "sa-east-1",
            ],
            "totalRegions": 16,
            "inferenceProfileCacheKeys": {},
        }
