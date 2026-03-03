"""
Region Discovery Lambda

Dynamically discovers all AWS regions where Bedrock inference profiles are available.
This replaces hardcoded region lists with dynamic discovery.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import ClientError

from shared.powertools import logger, tracer, metrics, LambdaContext
from aws_lambda_powertools.metrics import MetricUnit

# EC2 region for DescribeRegions API call (configurable via environment variable)
EC2_REGION = os.environ.get("EC2_REGION", "us-east-1")


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
def check_bedrock_available(region: str) -> tuple[str, bool]:
    """Check if Bedrock inference profiles are available in a region."""
    try:
        bedrock = boto3.client("bedrock", region_name=region)
        # Try to list inference profiles - this will fail if Bedrock isn't available
        bedrock.list_inference_profiles(maxResults=1)
        return (region, True)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("UnrecognizedClientException", "InvalidIdentityToken"):
            # Region not enabled or Bedrock not available
            logger.debug(
                "Bedrock not available in region",
                extra={"region": region, "error_code": error_code},
            )
            return (region, False)
        elif error_code == "AccessDeniedException":
            # Bedrock exists but we don't have access - still count it
            logger.debug("Bedrock exists but access denied", extra={"region": region})
            return (region, True)
        else:
            logger.warning(
                "Error checking Bedrock in region",
                extra={"region": region, "error": str(e)},
            )
            return (region, False)
    except Exception as e:
        logger.warning(
            "Unexpected error checking region",
            extra={"region": region, "error": str(e)},
        )
        return (region, False)


@tracer.capture_method
def discover_bedrock_regions(all_regions: list[str]) -> list[str]:
    """Discover which regions have Bedrock inference profiles available."""
    bedrock_regions = []

    # Check regions in parallel for speed
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(check_bedrock_available, region): region
            for region in all_regions
        }

        for future in as_completed(futures):
            region, available = future.result()
            if available:
                bedrock_regions.append(region)

    # Sort for consistent ordering
    bedrock_regions.sort()
    logger.info(
        "Regions with Bedrock discovered", extra={"region_count": len(bedrock_regions)}
    )
    return bedrock_regions


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    """
    Lambda handler for region discovery.

    Input:
        {} (no input required)

    Output:
        {
            "status": "SUCCESS",
            "featureRegions": ["us-east-1", "us-west-2", ...],
            "totalRegions": 27,
            "discoveryTimestamp": "2024-01-01T00:00:00Z"
        }
    """
    start_time = time.time()

    logger.info("Starting region discovery")

    try:
        # Get all enabled regions in the account
        all_regions = get_all_enabled_regions()
        logger.info(
            "Checking enabled regions for Bedrock availability",
            extra={"region_count": len(all_regions)},
        )

        # Filter to regions with Bedrock inference profiles
        bedrock_regions = discover_bedrock_regions(all_regions)

        elapsed = time.time() - start_time

        metrics.add_metric(
            name="RegionsDiscovered", unit=MetricUnit.Count, value=len(bedrock_regions)
        )
        logger.info(
            "Region discovery complete",
            extra={
                "regions_discovered": len(bedrock_regions),
                "all_enabled_regions": len(all_regions),
                "elapsed_seconds": round(elapsed, 2),
            },
        )

        return {
            "status": "SUCCESS",
            "featureRegions": bedrock_regions,
            "totalRegions": len(bedrock_regions),
            "allEnabledRegions": len(all_regions),
            "discoveryTimestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
        }
