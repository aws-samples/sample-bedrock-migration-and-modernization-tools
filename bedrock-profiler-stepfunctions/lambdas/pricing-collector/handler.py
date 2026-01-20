"""
Pricing Collector Lambda

Collects pricing data from AWS Pricing API for a single Bedrock service code.
"""

import json
import logging
import os
import time
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

# AWS clients
PRICING_REGION = os.environ.get('PRICING_API_REGION', 'us-east-1')
DATA_BUCKET = os.environ.get('DATA_BUCKET')

# Retry configuration
RETRY_CONFIG = Config(
    retries={
        'max_attempts': 3,
        'mode': 'adaptive'
    },
    connect_timeout=10,
    read_timeout=30
)


def get_pricing_client():
    """Create Pricing API client (only available in us-east-1 or ap-south-1)."""
    return boto3.client('pricing', region_name=PRICING_REGION, config=RETRY_CONFIG)


def get_s3_client():
    """Create S3 client."""
    return boto3.client('s3', config=RETRY_CONFIG)


def collect_pricing_for_service(pricing_client: Any, service_code: str) -> list[dict]:
    """
    Collect all pricing products for a given service code.

    Args:
        pricing_client: Boto3 Pricing client
        service_code: AWS service code (e.g., 'AmazonBedrock')

    Returns:
        List of pricing product dictionaries
    """
    products = []
    next_token = None
    batch_count = 0
    max_batches = 100  # Safety limit

    logger.info(f"Starting pricing collection for service: {service_code}")

    while batch_count < max_batches:
        try:
            params = {
                'ServiceCode': service_code,
                'MaxResults': 100,
                'FormatVersion': 'aws_v1'
            }

            if next_token:
                params['NextToken'] = next_token

            response = pricing_client.get_products(**params)

            # Parse price list items
            for price_item in response.get('PriceList', []):
                try:
                    product = json.loads(price_item) if isinstance(price_item, str) else price_item
                    products.append(product)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse price item: {e}")
                    continue

            batch_count += 1

            # Check for more results
            next_token = response.get('NextToken')
            if not next_token:
                break

            # Brief pause to avoid throttling
            if batch_count % 10 == 0:
                logger.info(f"Processed {batch_count} batches, {len(products)} products so far...")
                time.sleep(0.5)

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                logger.warning(f"Throttled, waiting before retry...")
                time.sleep(2)
                continue
            else:
                logger.error(f"ClientError collecting pricing: {e}")
                raise

    logger.info(f"Completed: {len(products)} products from {batch_count} batches")
    return products


def write_to_s3(s3_client: Any, bucket: str, key: str, data: dict) -> None:
    """Write JSON data to S3."""
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2, default=str),
        ContentType='application/json'
    )
    logger.info(f"Written to s3://{bucket}/{key}")


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for pricing collection.

    Input:
        {
            "serviceCode": "AmazonBedrock",
            "s3Bucket": "bucket-name",
            "s3Key": "executions/{id}/pricing/AmazonBedrock.json",
            "dryRun": false  // Optional: skip S3 write for testing
        }

    Output:
        {
            "status": "SUCCESS",
            "serviceCode": "AmazonBedrock",
            "s3Key": "executions/{id}/pricing/AmazonBedrock.json",
            "recordCount": 1250,
            "durationMs": 45000
        }
    """
    start_time = time.time()

    # Extract parameters
    service_code = event['serviceCode']
    s3_bucket = event.get('s3Bucket', DATA_BUCKET)
    s3_key = event.get('s3Key', f'test/{service_code}.json')
    dry_run = event.get('dryRun', False)

    logger.info(f"Starting pricing collection: service={service_code}, bucket={s3_bucket}, dryRun={dry_run}")

    try:
        # Initialize clients
        pricing_client = get_pricing_client()

        # Collect pricing data
        products = collect_pricing_for_service(pricing_client, service_code)

        # Structure the output
        output_data = {
            'metadata': {
                'serviceCode': service_code,
                'recordCount': len(products),
                'collectionTimestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'pricingRegion': PRICING_REGION
            },
            'products': products
        }

        # Write to S3 (skip in dry run mode)
        if not dry_run and s3_bucket:
            s3_client = get_s3_client()
            write_to_s3(s3_client, s3_bucket, s3_key, output_data)
        else:
            logger.info(f"Dry run mode - skipping S3 write. Would write to s3://{s3_bucket}/{s3_key}")

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            'status': 'SUCCESS',
            'serviceCode': service_code,
            's3Key': s3_key,
            'recordCount': len(products),
            'durationMs': duration_ms,
            'dryRun': dry_run
        }

    except Exception as e:
        logger.error(f"Failed to collect pricing: {e}", exc_info=True)

        return {
            'status': 'FAILED',
            'serviceCode': service_code,
            'errorType': type(e).__name__,
            'errorMessage': str(e),
            'retryable': isinstance(e, (ClientError,)) and 'Throttling' in str(e)
        }
