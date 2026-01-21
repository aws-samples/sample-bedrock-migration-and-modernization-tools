"""
Feature Collector Lambda

Collects inference profiles and enhanced features from a single region.
"""

import logging
import os
import time
from typing import Any

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

logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))


def get_bedrock_client(region: str):
    """Create Bedrock client for a specific region."""
    return boto3.client('bedrock', region_name=region, config=RETRY_CONFIG)


def get_s3_client():
    return boto3.client('s3', config=RETRY_CONFIG)


def collect_inference_profiles(bedrock_client: Any, region: str) -> list[dict]:
    """
    Collect inference profiles from Bedrock API.

    Returns list of inference profile dictionaries.
    """
    profiles = []

    try:
        # List inference profiles
        paginator = bedrock_client.get_paginator('list_inference_profiles')

        for page in paginator.paginate():
            for profile in page.get('inferenceProfileSummaries', []):
                normalized = {
                    'inferenceProfileId': profile.get('inferenceProfileId', ''),
                    'inferenceProfileArn': profile.get('inferenceProfileArn', ''),
                    'inferenceProfileName': profile.get('inferenceProfileName', ''),
                    'description': profile.get('description', ''),
                    'status': profile.get('status', ''),
                    'type': profile.get('type', ''),
                    'models': profile.get('models', []),
                    'region': region
                }
                profiles.append(normalized)

        logger.info(f"Collected {len(profiles)} inference profiles from {region}")

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code in ('AccessDeniedException', 'UnrecognizedClientException'):
            logger.warning(f"Access denied or region not enabled: {region} ({error_code})")
        elif error_code == 'ValidationException':
            logger.warning(f"Inference profiles not available in {region}")
        elif error_code == 'InvalidIdentityToken':
            logger.warning(f"Invalid token for region {region} - region may require opt-in")
        else:
            logger.error(f"Error collecting inference profiles in {region}: {e}")
            # Don't raise - continue with empty profiles

    except Exception as e:
        logger.warning(f"Unexpected error collecting inference profiles in {region}: {e}")

    return profiles


def lambda_handler(event: dict, context: Any) -> dict:
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
        validate_required_params(event, ['region'], 'FeatureCollector')
    except ValidationError as e:
        return {
            'status': 'FAILED',
            'errorType': 'ValidationError',
            'errorMessage': str(e)
        }

    region = event['region']
    s3_bucket = event.get('s3Bucket')
    s3_key = event.get('s3Key', f'test/features/{region}.json')
    dry_run = event.get('dryRun', False)

    logger.info(f"Collecting features from region: {region}")

    try:
        bedrock_client = get_bedrock_client(region)
        profiles = collect_inference_profiles(bedrock_client, region)

        output_data = {
            'metadata': {
                'region': region,
                'inferenceProfileCount': len(profiles),
                'collectionTimestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            },
            'inferenceProfiles': profiles
        }

        if not dry_run and s3_bucket:
            s3_client = get_s3_client()
            write_to_s3(s3_client, s3_bucket, s3_key, output_data)
        else:
            logger.info(f"Dry run - would write {len(profiles)} profiles to s3://{s3_bucket}/{s3_key}")

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            'status': 'SUCCESS',
            'region': region,
            's3Key': s3_key,
            'inferenceProfileCount': len(profiles),
            'durationMs': duration_ms
        }

    except Exception as e:
        logger.error(f"Failed to collect features from {region}: {e}", exc_info=True)
        return {
            'status': 'FAILED',
            'region': region,
            'errorType': type(e).__name__,
            'errorMessage': str(e),
            'retryable': 'Throttling' in str(e)
        }
