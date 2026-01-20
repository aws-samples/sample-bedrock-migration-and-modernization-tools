"""
Quota Collector Lambda

Collects Bedrock service quotas from a single AWS region.
"""

import json
import logging
import os
import time
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

RETRY_CONFIG = Config(
    retries={'max_attempts': 3, 'mode': 'adaptive'},
    connect_timeout=10,
    read_timeout=30
)

SERVICE_CODE = 'bedrock'


def get_quotas_client(region: str):
    """Create Service Quotas client for a specific region."""
    return boto3.client('service-quotas', region_name=region, config=RETRY_CONFIG)


def get_s3_client():
    return boto3.client('s3', config=RETRY_CONFIG)


def write_to_s3(s3_client: Any, bucket: str, key: str, data: dict) -> None:
    """Write JSON data to S3."""
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2, default=str),
        ContentType='application/json'
    )
    logger.info(f"Written to s3://{bucket}/{key}")


def collect_quotas(quotas_client: Any, region: str) -> list[dict]:
    """
    Collect all Bedrock service quotas from Service Quotas API.

    Returns list of quota dictionaries.
    """
    quotas = []
    next_token = None

    try:
        while True:
            params = {
                'ServiceCode': SERVICE_CODE,
                'MaxResults': 100
            }

            if next_token:
                params['NextToken'] = next_token

            response = quotas_client.list_service_quotas(**params)

            for quota in response.get('Quotas', []):
                normalized = {
                    'quotaCode': quota.get('QuotaCode', ''),
                    'quotaName': quota.get('QuotaName', ''),
                    'quotaArn': quota.get('QuotaArn', ''),
                    'value': quota.get('Value'),
                    'unit': quota.get('Unit', ''),
                    'adjustable': quota.get('Adjustable', False),
                    'globalQuota': quota.get('GlobalQuota', False),
                    'usageMetric': quota.get('UsageMetric', {}),
                    'period': quota.get('Period', {}),
                    'region': region
                }
                quotas.append(normalized)

            next_token = response.get('NextToken')
            if not next_token:
                break

        logger.info(f"Collected {len(quotas)} quotas from {region}")

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchResourceException':
            logger.warning(f"Bedrock service not available in {region}")
        elif error_code in ('AccessDeniedException', 'UnrecognizedClientException'):
            logger.warning(f"Access denied or region not enabled: {region} ({error_code})")
        elif error_code == 'InvalidIdentityToken':
            logger.warning(f"Invalid token for region {region} - region may require opt-in")
        else:
            logger.error(f"Error collecting quotas in {region}: {e}")
            # Don't raise - continue with empty quotas for this region

    except Exception as e:
        logger.warning(f"Unexpected error collecting quotas in {region}: {e}")
        # Continue with empty quotas

    return quotas


def lambda_handler(event: dict, context: Any) -> dict:
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

    region = event['region']
    s3_bucket = event.get('s3Bucket')
    s3_key = event.get('s3Key', f'test/quotas/{region}.json')
    dry_run = event.get('dryRun', False)

    logger.info(f"Collecting quotas from region: {region}")

    try:
        quotas_client = get_quotas_client(region)
        quotas = collect_quotas(quotas_client, region)

        output_data = {
            'metadata': {
                'region': region,
                'quotaCount': len(quotas),
                'serviceCode': SERVICE_CODE,
                'collectionTimestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            },
            'quotas': quotas
        }

        if not dry_run and s3_bucket:
            s3_client = get_s3_client()
            write_to_s3(s3_client, s3_bucket, s3_key, output_data)
        else:
            logger.info(f"Dry run - would write {len(quotas)} quotas to s3://{s3_bucket}/{s3_key}")

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            'status': 'SUCCESS',
            'region': region,
            's3Key': s3_key,
            'quotaCount': len(quotas),
            'durationMs': duration_ms
        }

    except Exception as e:
        logger.error(f"Failed to collect quotas from {region}: {e}", exc_info=True)
        return {
            'status': 'FAILED',
            'region': region,
            'errorType': type(e).__name__,
            'errorMessage': str(e),
            'retryable': 'Throttling' in str(e)
        }
