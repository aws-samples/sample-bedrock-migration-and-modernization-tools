"""
Feature Collector Lambda

Collects inference profiles and enhanced features from a single region.
"""

import time

import boto3
from botocore.exceptions import ClientError

from shared import (
    RETRY_CONFIG,
    write_to_s3,
    parse_execution_id,
    validate_required_params,
    ValidationError,
    S3WriteError,
)
from shared.powertools import logger, tracer, metrics, LambdaContext
from aws_lambda_powertools.metrics import MetricUnit


def get_bedrock_client(region: str):
    """Create Bedrock client for a specific region."""
    return boto3.client("bedrock", region_name=region, config=RETRY_CONFIG)


def get_s3_client():
    return boto3.client("s3", config=RETRY_CONFIG)


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
            "s3Key": "executions/{id}/features/us-east-1.json"
        }

    Output:
        {
            "status": "SUCCESS",
            "region": "us-east-1",
            "s3Key": "executions/{id}/features/us-east-1.json",
            "inferenceProfileCount": 12
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

    logger.info("Starting feature collection", extra={"region": region})

    try:
        bedrock_client = get_bedrock_client(region)
        profiles = collect_inference_profiles(bedrock_client, region)

        output_data = {
            "metadata": {
                "region": region,
                "inferenceProfileCount": len(profiles),
                "collectionTimestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            },
            "inferenceProfiles": profiles,
        }

        if not dry_run and s3_bucket:
            s3_client = get_s3_client()
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
        metrics.add_dimension(name="Region", value=region)

        logger.info(
            "Feature collection complete",
            extra={
                "region": region,
                "profile_count": len(profiles),
                "duration_ms": duration_ms,
            },
        )

        return {
            "status": "SUCCESS",
            "region": region,
            "s3Key": s3_key,
            "inferenceProfileCount": len(profiles),
            "durationMs": duration_ms,
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
