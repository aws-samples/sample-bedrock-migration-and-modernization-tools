"""
CloudFront Cache Invalidator Lambda

Creates a CloudFront invalidation for /latest/* after the pipeline
copies new data files. Runs as the final step before ExecutionSucceeded.
"""

import os
import time

import boto3

from shared import RETRY_CONFIG, validate_required_params, ValidationError
from shared.powertools import logger, tracer, metrics, LambdaContext
from aws_lambda_powertools.metrics import MetricUnit


DISTRIBUTION_ID = os.environ.get("CLOUDFRONT_DISTRIBUTION_ID", "")
INVALIDATION_PATHS = ["/latest/*"]


def get_cloudfront_client():
    return boto3.client("cloudfront", config=RETRY_CONFIG)


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    """
    Create a CloudFront invalidation for data files.

    Input: Receives the full pipeline state (passthrough from CopyToLatest).
    Output: Original state plus invalidationId.
    """
    start_time = time.time()

    if not DISTRIBUTION_ID:
        logger.warning("CLOUDFRONT_DISTRIBUTION_ID not set, skipping invalidation")
        return {
            "status": "SKIPPED",
            "reason": "No distribution ID configured",
        }

    try:
        client = get_cloudfront_client()
        caller_ref = f"pipeline-{int(time.time())}"

        response = client.create_invalidation(
            DistributionId=DISTRIBUTION_ID,
            InvalidationBatch={
                "Paths": {
                    "Quantity": len(INVALIDATION_PATHS),
                    "Items": INVALIDATION_PATHS,
                },
                "CallerReference": caller_ref,
            },
        )

        invalidation_id = response["Invalidation"]["Id"]
        duration_ms = int((time.time() - start_time) * 1000)

        metrics.add_metric(
            name="CacheInvalidations", unit=MetricUnit.Count, value=1
        )
        metrics.add_metric(
            name="DurationMs", unit=MetricUnit.Milliseconds, value=duration_ms
        )

        logger.info(
            "CloudFront invalidation created",
            extra={
                "invalidation_id": invalidation_id,
                "distribution_id": DISTRIBUTION_ID,
                "paths": INVALIDATION_PATHS,
                "duration_ms": duration_ms,
            },
        )

        return {
            "status": "SUCCESS",
            "invalidationId": invalidation_id,
            "distributionId": DISTRIBUTION_ID,
            "paths": INVALIDATION_PATHS,
            "durationMs": duration_ms,
        }

    except Exception as e:
        logger.exception(
            "Failed to create CloudFront invalidation",
            extra={"error_type": type(e).__name__},
        )
        # Non-fatal: pipeline data is already in S3, cache will expire naturally
        return {
            "status": "FAILED",
            "errorType": type(e).__name__,
            "errorMessage": str(e),
        }
