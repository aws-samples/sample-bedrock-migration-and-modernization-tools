"""
Copy to Latest Lambda

Copies final outputs to the latest/ prefix for easy access.
"""

import json
import logging
import os
import time
from typing import Any

import boto3
from botocore.config import Config

logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

RETRY_CONFIG = Config(
    retries={'max_attempts': 3, 'mode': 'adaptive'},
    connect_timeout=10,
    read_timeout=30
)


def get_s3_client():
    return boto3.client('s3', config=RETRY_CONFIG)


def copy_s3_object(s3_client: Any, bucket: str, source_key: str, dest_key: str) -> None:
    """Copy an S3 object to a new location."""
    copy_source = {'Bucket': bucket, 'Key': source_key}
    s3_client.copy_object(
        Bucket=bucket,
        CopySource=copy_source,
        Key=dest_key,
        MetadataDirective='REPLACE',
        ContentType='application/json'
    )
    logger.info(f"Copied s3://{bucket}/{source_key} to s3://{bucket}/{dest_key}")


def lambda_handler(event: dict, context: Any) -> dict:
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

    s3_bucket = event['s3Bucket']
    execution_id = event['executionId']
    final_result = event.get('finalResult', {})
    dry_run = event.get('dryRun', False)

    models_source_key = final_result.get('modelsS3Key')
    pricing_source_key = final_result.get('pricingS3Key')

    latest_models_key = 'latest/bedrock_models.json'
    latest_pricing_key = 'latest/bedrock_pricing.json'

    logger.info(f"Copying final outputs to latest/")

    try:
        if not dry_run:
            s3_client = get_s3_client()

            # Copy models
            if models_source_key:
                copy_s3_object(s3_client, s3_bucket, models_source_key, latest_models_key)

            # Copy pricing
            if pricing_source_key:
                copy_s3_object(s3_client, s3_bucket, pricing_source_key, latest_pricing_key)

            # Also create a manifest file with execution info
            manifest = {
                'lastUpdated': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'executionId': execution_id,
                'files': {
                    'models': latest_models_key,
                    'pricing': latest_pricing_key
                }
            }
            s3_client.put_object(
                Bucket=s3_bucket,
                Key='latest/manifest.json',
                Body=json.dumps(manifest, indent=2),
                ContentType='application/json'
            )
        else:
            logger.info("Dry run - skipping copy")

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            'status': 'SUCCESS',
            'latestModelsKey': latest_models_key,
            'latestPricingKey': latest_pricing_key,
            'durationMs': duration_ms
        }

    except Exception as e:
        logger.error(f"Failed to copy to latest: {e}", exc_info=True)
        return {
            'status': 'FAILED',
            'errorType': type(e).__name__,
            'errorMessage': str(e)
        }
