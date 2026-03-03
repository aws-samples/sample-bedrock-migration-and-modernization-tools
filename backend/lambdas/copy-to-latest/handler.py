"""
Copy to Latest Lambda

Copies final outputs to the latest/ prefix for easy access.
Preserves date_added for existing models and stamps new models with the current date.
"""

import json
import time
from typing import Any

import boto3

from shared import (
    RETRY_CONFIG,
    parse_execution_id,
    validate_required_params,
    ValidationError,
)
from shared.powertools import logger, tracer, metrics, LambdaContext
from aws_lambda_powertools.metrics import MetricUnit


def get_s3_client():
    return boto3.client("s3", config=RETRY_CONFIG)


def copy_s3_object(s3_client: Any, bucket: str, source_key: str, dest_key: str) -> None:
    """Copy an S3 object to a new location."""
    copy_source = {"Bucket": bucket, "Key": source_key}
    s3_client.copy_object(
        Bucket=bucket,
        CopySource=copy_source,
        Key=dest_key,
        MetadataDirective="REPLACE",
        ContentType="application/json",
    )
    logger.info(
        "Copied S3 object",
        extra={
            "source": f"s3://{bucket}/{source_key}",
            "destination": f"s3://{bucket}/{dest_key}",
        },
    )


def read_s3_json(s3_client: Any, bucket: str, key: str) -> dict:
    """Read and parse a JSON file from S3. Returns empty dict on failure."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except Exception as e:
        logger.warning(
            "Could not read S3 object",
            extra={"bucket": bucket, "key": key, "error": str(e)},
        )
        return {}


def stamp_date_added(
    s3_client: Any, bucket: str, new_models_key: str, latest_models_key: str
) -> None:
    """
    Read both the new models file and the previous latest, then:
    - For models that existed before: preserve their date_added
    - For new models: set date_added to today's date (YYYY-MM-DD)
    Write the updated data back to the new models file.
    """
    new_data = read_s3_json(s3_client, bucket, new_models_key)
    if not new_data or "providers" not in new_data:
        logger.warning(
            "New models data is empty or has no providers; skipping date_added stamping"
        )
        return

    # Build a lookup of existing model date_added from previous latest
    previous_data = read_s3_json(s3_client, bucket, latest_models_key)
    existing_dates = {}
    if previous_data and "providers" in previous_data:
        for provider_data in previous_data["providers"].values():
            models = provider_data.get("models", {})
            for model_id, model in models.items():
                date_val = model.get("date_added")
                if date_val:
                    existing_dates[model_id] = date_val

    today = time.strftime("%Y-%m-%d", time.gmtime())
    new_count = 0
    preserved_count = 0

    for provider_data in new_data["providers"].values():
        models = provider_data.get("models", {})
        for model_id, model in models.items():
            if model_id in existing_dates:
                model["date_added"] = existing_dates[model_id]
                preserved_count += 1
            else:
                model["date_added"] = today
                new_count += 1

    logger.info(
        "date_added stamping complete",
        extra={
            "new_models": new_count,
            "preserved_models": preserved_count,
            "stamp_date": today,
        },
    )

    # Write the updated data back to the new models key (before the copy)
    s3_client.put_object(
        Bucket=bucket,
        Key=new_models_key,
        Body=json.dumps(new_data, indent=2),
        ContentType="application/json",
    )


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    """
    Lambda handler for copying to latest.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "finalResult": {
                "modelsS3Key": "executions/{id}/final/bedrock_models.json",
                "pricingS3Key": "executions/{id}/final/bedrock_pricing.json"
            }
        }

    Output:
        {
            "status": "SUCCESS",
            "latestModelsKey": "latest/bedrock_models.json",
            "latestPricingKey": "latest/bedrock_pricing.json"
        }
    """
    start_time = time.time()

    # Validate required parameters
    try:
        validate_required_params(event, ["s3Bucket", "executionId"], "CopyToLatest")
    except ValidationError as e:
        return {
            "status": "FAILED",
            "errorType": "ValidationError",
            "errorMessage": str(e),
        }

    s3_bucket = event["s3Bucket"]
    execution_id = parse_execution_id(event["executionId"])
    final_result = event.get("finalResult", {})
    dry_run = event.get("dryRun", False)

    models_source_key = final_result.get("modelsS3Key")
    pricing_source_key = final_result.get("pricingS3Key")

    latest_models_key = "latest/bedrock_models.json"
    latest_pricing_key = "latest/bedrock_pricing.json"

    logger.info("Starting copy to latest", extra={"execution_id": execution_id})

    copied_files = []
    try:
        if not dry_run:
            s3_client = get_s3_client()

            # Stamp date_added before copying to latest
            if models_source_key:
                stamp_date_added(
                    s3_client, s3_bucket, models_source_key, latest_models_key
                )

            # Copy models
            if models_source_key:
                copy_s3_object(
                    s3_client, s3_bucket, models_source_key, latest_models_key
                )
                copied_files.append(latest_models_key)

            # Copy pricing
            if pricing_source_key:
                copy_s3_object(
                    s3_client, s3_bucket, pricing_source_key, latest_pricing_key
                )
                copied_files.append(latest_pricing_key)

            # Also create a manifest file with execution info
            manifest = {
                "lastUpdated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "executionId": execution_id,
                "files": {"models": latest_models_key, "pricing": latest_pricing_key},
            }
            s3_client.put_object(
                Bucket=s3_bucket,
                Key="latest/manifest.json",
                Body=json.dumps(manifest, indent=2),
                ContentType="application/json",
            )
            copied_files.append("latest/manifest.json")
        else:
            logger.info("Dry run - skipping copy")

        duration_ms = int((time.time() - start_time) * 1000)

        # Emit metrics
        metrics.add_metric(
            name="FilesCopied", unit=MetricUnit.Count, value=len(copied_files)
        )
        metrics.add_metric(
            name="DurationMs", unit=MetricUnit.Milliseconds, value=duration_ms
        )

        logger.info(
            "Copy to latest complete",
            extra={"files_copied": copied_files, "duration_ms": duration_ms},
        )

        return {
            "status": "SUCCESS",
            "latestModelsKey": latest_models_key,
            "latestPricingKey": latest_pricing_key,
            "durationMs": duration_ms,
        }

    except Exception as e:
        logger.exception(
            "Failed to copy to latest", extra={"error_type": type(e).__name__}
        )
        return {
            "status": "FAILED",
            "errorType": type(e).__name__,
            "errorMessage": str(e),
        }
