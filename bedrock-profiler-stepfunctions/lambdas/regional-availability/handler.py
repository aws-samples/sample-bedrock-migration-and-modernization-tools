"""
Regional Availability Lambda

Computes regional availability map from pricing data.
Works with the correct snake_case schema.
"""

import json
import logging
import os
import time
from typing import Any
from collections import defaultdict

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


def read_from_s3(s3_client: Any, bucket: str, key: str) -> dict:
    """Read JSON data from S3."""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response['Body'].read().decode('utf-8'))


def write_to_s3(s3_client: Any, bucket: str, key: str, data: dict) -> None:
    """Write JSON data to S3."""
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2, default=str),
        ContentType='application/json'
    )
    logger.info(f"Written to s3://{bucket}/{key}")


def compute_regional_availability(pricing_data: dict) -> dict:
    """
    Compute regional availability from pricing data.

    Handles both old and new pricing structures:
    - New: providers > {model_id} > regions > {region} > pricing_groups
    - Old: providers > {provider} > models > {model_id} > regions

    Returns:
    {
        "regions": {
            "us-east-1": {
                "bedrock_available": true,
                "models_in_region": 95,
                "providers": ["Amazon", "Anthropic", ...],
                "model_count": 95
            }
        },
        "model_availability": {
            "anthropic.claude-3-sonnet-v1": ["us-east-1", "us-west-2", ...]
        }
    }
    """
    regions = defaultdict(lambda: {
        'bedrock_available': True,
        'models_in_region': 0,
        'providers': set(),
        'models': []
    })

    model_availability = defaultdict(list)

    # Extract region information from pricing data
    providers_data = pricing_data.get('providers', {})

    for key, data in providers_data.items():
        # Check if this is the new structure (model_id -> data with regions)
        # or old structure (provider -> models -> model_id -> data)
        if isinstance(data, dict):
            if 'regions' in data and 'model_provider' in data:
                # New structure: key is model_id, data contains regions directly
                model_id = key
                provider = data.get('model_provider', 'Unknown')
                model_regions = data.get('regions', {})

                for region in model_regions.keys():
                    regions[region]['models_in_region'] += 1
                    regions[region]['providers'].add(provider)
                    regions[region]['models'].append(model_id)
                    model_availability[model_id].append(region)

            elif 'models' in data:
                # Old structure: key is provider, data contains models
                provider = key
                for model_id, model_data in data.get('models', {}).items():
                    model_regions = model_data.get('regions', {})

                    for region in model_regions.keys():
                        regions[region]['models_in_region'] += 1
                        regions[region]['providers'].add(provider)
                        regions[region]['models'].append(model_id)
                        model_availability[model_id].append(region)

    # Convert sets to lists for JSON serialization (snake_case output)
    result_regions = {}
    for region, data in regions.items():
        result_regions[region] = {
            'bedrock_available': data['bedrock_available'],
            'models_in_region': data['models_in_region'],
            'providers': sorted(list(data['providers'])),
            'model_count': len(data['models'])
        }

    return {
        'regions': result_regions,
        'model_availability': dict(model_availability)
    }


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for regional availability computation.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "pricingS3Key": "executions/{id}/merged/pricing.json"
        }

    Output:
        {
            "status": "SUCCESS",
            "s3Key": "executions/{id}/intermediate/regional-availability.json",
            "regionsWithBedrock": 20
        }
    """
    start_time = time.time()

    s3_bucket = event['s3Bucket']
    execution_id = event['executionId']
    pricing_s3_key = event['pricingS3Key']
    dry_run = event.get('dryRun', False)

    if ':' in execution_id:
        execution_id = execution_id.split(':')[-1]

    output_key = f"executions/{execution_id}/intermediate/regional-availability.json"

    logger.info("Computing regional availability")

    try:
        s3_client = get_s3_client()

        if not dry_run:
            pricing_data = read_from_s3(s3_client, s3_bucket, pricing_s3_key)
            availability = compute_regional_availability(pricing_data)

            output_data = {
                'metadata': {
                    'regions_with_bedrock': len(availability['regions']),
                    'total_models_tracked': len(availability['model_availability']),
                    'collection_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                },
                'regions': availability['regions'],
                'model_availability': availability['model_availability']
            }

            write_to_s3(s3_client, s3_bucket, output_key, output_data)
            regions_count = len(availability['regions'])
        else:
            logger.info("Dry run - skipping processing")
            regions_count = 0

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            'status': 'SUCCESS',
            's3Key': output_key,
            'regionsWithBedrock': regions_count,
            'durationMs': duration_ms
        }

    except Exception as e:
        logger.error(f"Failed to compute availability: {e}", exc_info=True)
        return {
            'status': 'FAILED',
            'errorType': type(e).__name__,
            'errorMessage': str(e)
        }
