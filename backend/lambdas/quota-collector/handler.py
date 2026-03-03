"""
Quota Collector Lambda

Collects Bedrock service quotas from a single AWS region.

Configuration (environment variables):
    QUOTA_BATCH_SIZE: Number of quotas per API call (default: 100)
"""

import os
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

# Configuration with defaults
QUOTA_BATCH_SIZE = int(os.environ.get("QUOTA_BATCH_SIZE", "100"))

SERVICE_CODE = "bedrock"


def get_quotas_client(region: str):
    """Create Service Quotas client for a specific region."""
    return boto3.client("service-quotas", region_name=region, config=RETRY_CONFIG)


def get_s3_client():
    return boto3.client("s3", config=RETRY_CONFIG)


@tracer.capture_method
def collect_quotas(quotas_client, region: str) -> list[dict]:
    """
    Collect all Bedrock service quotas from Service Quotas API.

    Returns list of quota dictionaries.
    """
    quotas = []
    next_token = None

    try:
        while True:
            params = {"ServiceCode": SERVICE_CODE, "MaxResults": QUOTA_BATCH_SIZE}

            if next_token:
                params["NextToken"] = next_token

            response = quotas_client.list_service_quotas(**params)

            for quota in response.get("Quotas", []):
                normalized = {
                    "quota_code": quota.get("QuotaCode", ""),
                    "quota_name": quota.get("QuotaName", ""),
                    "quota_arn": quota.get("QuotaArn", ""),
                    "value": quota.get("Value"),
                    "unit": quota.get("Unit", ""),
                    "adjustable": quota.get("Adjustable", False),
                    "global_quota": quota.get("GlobalQuota", False),
                    "usage_metric": quota.get("UsageMetric", {}),
                    "period": quota.get("Period", {}),
                    "region": region,
                }
                quotas.append(normalized)

            next_token = response.get("NextToken")
            if not next_token:
                break

        logger.info(
            "Quotas collected", extra={"quota_count": len(quotas), "region": region}
        )

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "NoSuchResourceException":
            logger.warning("Bedrock service not available", extra={"region": region})
        elif error_code in ("AccessDeniedException", "UnrecognizedClientException"):
            logger.warning(
                "Access denied or region not enabled",
                extra={"region": region, "error_code": error_code},
            )
        elif error_code == "InvalidIdentityToken":
            logger.warning(
                "Invalid token for region - region may require opt-in",
                extra={"region": region},
            )
        else:
            logger.error(
                "Error collecting quotas", extra={"region": region, "error": str(e)}
            )
            # Don't raise - continue with empty quotas for this region

    except Exception as e:
        logger.warning(
            "Unexpected error collecting quotas",
            extra={"region": region, "error": str(e)},
        )
        # Continue with empty quotas

    return quotas


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    """
    Lambda handler for quota collection.

    Input:
        {
            "region": "us-east-1",
            "s3Bucket": "bucket-name",
            "s3Key": "executions/{id}/quotas/us-east-1.json"
        }

    Output:
        {
            "status": "SUCCESS",
            "region": "us-east-1",
            "s3Key": "executions/{id}/quotas/us-east-1.json",
            "quotaCount": 45
        }
    """
    start_time = time.time()

    # Validate required parameters
    try:
        validate_required_params(event, ["region"], "QuotaCollector")
    except ValidationError as e:
        return {
            "status": "FAILED",
            "errorType": "ValidationError",
            "errorMessage": str(e),
        }

    region = event["region"]
    s3_bucket = event.get("s3Bucket")
    s3_key = event.get("s3Key", f"test/quotas/{region}.json")
    dry_run = event.get("dryRun", False)

    logger.info("Starting quota collection", extra={"region": region})

    try:
        quotas_client = get_quotas_client(region)
        quotas = collect_quotas(quotas_client, region)

        output_data = {
            "metadata": {
                "region": region,
                "quota_count": len(quotas),
                "service_code": SERVICE_CODE,
                "collection_timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            },
            "quotas": quotas,
        }

        if not dry_run and s3_bucket:
            s3_client = get_s3_client()
            write_to_s3(s3_client, s3_bucket, s3_key, output_data)
        else:
            logger.info(
                "Dry run - skipping S3 write",
                extra={"quota_count": len(quotas), "bucket": s3_bucket, "key": s3_key},
            )

        duration_ms = int((time.time() - start_time) * 1000)

        # Add metrics
        metrics.add_metric(
            name="QuotasCollected", unit=MetricUnit.Count, value=len(quotas)
        )
        metrics.add_metric(
            name="CollectionDurationMs", unit=MetricUnit.Milliseconds, value=duration_ms
        )
        metrics.add_dimension(name="Region", value=region)

        logger.info(
            "Quota collection complete",
            extra={
                "quota_count": len(quotas),
                "region": region,
                "duration_ms": duration_ms,
            },
        )

        return {
            "status": "SUCCESS",
            "region": region,
            "s3Key": s3_key,
        }

    except Exception as e:
        logger.exception(
            "Failed to collect quotas", extra={"region": region, "error": str(e)}
        )
        return {
            "status": "FAILED",
            "region": region,
            "errorType": type(e).__name__,
            "errorMessage": str(e),
            "retryable": "Throttling" in str(e),
        }
