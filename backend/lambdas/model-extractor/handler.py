"""
Model Extractor Lambda

Extracts foundation models from a single AWS region using the Bedrock API.
Outputs models in the correct snake_case schema matching the original collector.
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

# Documentation links per provider
DOCUMENTATION_LINKS = {
    'Anthropic': {
        'aws_bedrock_guide': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html',
        'pricing_guide': 'https://aws.amazon.com/bedrock/pricing/'
    },
    'Amazon': {
        'aws_bedrock_guide': 'https://docs.aws.amazon.com/bedrock/latest/userguide/titan-models.html',
        'pricing_guide': 'https://aws.amazon.com/bedrock/pricing/'
    },
    'Meta': {
        'aws_bedrock_guide': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html',
        'pricing_guide': 'https://aws.amazon.com/bedrock/pricing/'
    },
    'Mistral AI': {
        'aws_bedrock_guide': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html',
        'pricing_guide': 'https://aws.amazon.com/bedrock/pricing/'
    },
    'Cohere': {
        'aws_bedrock_guide': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere.html',
        'pricing_guide': 'https://aws.amazon.com/bedrock/pricing/'
    },
    'AI21 Labs': {
        'aws_bedrock_guide': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-ai21.html',
        'pricing_guide': 'https://aws.amazon.com/bedrock/pricing/'
    },
    'Stability AI': {
        'aws_bedrock_guide': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-stability-diffusion.html',
        'pricing_guide': 'https://aws.amazon.com/bedrock/pricing/'
    },
    'default': {
        'aws_bedrock_guide': 'https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html',
        'pricing_guide': 'https://aws.amazon.com/bedrock/pricing/'
    }
}


def get_bedrock_client(region: str):
    """Create Bedrock client for a specific region."""
    return boto3.client('bedrock', region_name=region, config=RETRY_CONFIG)


def get_s3_client():
    return boto3.client('s3', config=RETRY_CONFIG)


def get_documentation_links(model_id: str, provider: str) -> dict:
    """Get documentation links based on provider and model."""
    # Check for Nova models (Amazon's newer models)
    if 'nova' in model_id.lower():
        return {
            'aws_bedrock_guide': 'https://docs.aws.amazon.com/nova/latest/userguide/what-is-nova.html',
            'pricing_guide': 'https://aws.amazon.com/bedrock/pricing/'
        }

    # Get provider-specific docs or default
    return DOCUMENTATION_LINKS.get(provider, DOCUMENTATION_LINKS['default']).copy()


def process_model_data(raw_model: dict, region: str) -> dict:
    """
    Process and structure model data to match the expected schema.

    Converts AWS API response to snake_case schema matching the original collector.
    """
    model_id = raw_model.get('modelId', '')
    provider = raw_model.get('providerName', '')
    collection_timestamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())

    return {
        # Core identification (snake_case)
        'model_id': model_id,
        'model_arn': raw_model.get('modelArn', ''),
        'model_name': raw_model.get('modelName', ''),
        'model_provider': provider,

        # Capabilities from API (nested structure)
        'model_modalities': {
            'input_modalities': raw_model.get('inputModalities', []),
            'output_modalities': raw_model.get('outputModalities', [])
        },
        'streaming_supported': raw_model.get('responseStreamingSupported', False),
        'customization': {
            'customization_supported': raw_model.get('customizationsSupported', []),
            'customization_options': {}
        },
        'inference_types_supported': raw_model.get('inferenceTypesSupported', []),
        'model_lifecycle': {
            'status': raw_model.get('modelLifecycle', {}).get('status', 'UNKNOWN'),
            'release_date': ''
        },

        # Regional information
        'regions_available': [region],

        # Fields to be enhanced in later phases
        'model_capabilities': [],
        'model_use_cases': [],
        'languages_supported': [],
        'consumption_options': [],
        'cross_region_inference': {},
        'documentation_links': get_documentation_links(model_id, provider),
        'model_pricing': {'is_pricing_available': False},
        'model_service_quotas': {},

        # Collection metadata
        'collection_metadata': {
            'first_discovered_at': collection_timestamp,
            'first_discovered_in_region': region,
            'api_source': 'list_foundation_models',
            'dual_region_collection': True,
            'regions_collected_from': [region]
        }
    }


def extract_models(bedrock_client: Any, region: str) -> list[dict]:
    """
    Extract all foundation models from Bedrock API.

    Returns list of model dictionaries with correct snake_case schema.
    """
    models = []

    try:
        response = bedrock_client.list_foundation_models()
        model_summaries = response.get('modelSummaries', [])

        for raw_model in model_summaries:
            processed = process_model_data(raw_model, region)
            models.append(processed)

        logger.info(f"Extracted {len(models)} models from {region}")

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code in ('AccessDeniedException', 'UnrecognizedClientException'):
            logger.warning(f"Access denied or region not enabled: {region} ({error_code})")
        elif error_code == 'InvalidIdentityToken':
            logger.warning(f"Invalid token for region {region} - region may require opt-in")
        else:
            logger.error(f"Error listing models in {region}: {e}")

    except Exception as e:
        logger.warning(f"Unexpected error extracting models in {region}: {e}")

    return models


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for model extraction.

    Input:
        {
            "region": "us-east-1",
            "s3Bucket": "bucket-name",
            "s3Key": "executions/{id}/models/us-east-1.json"
        }

    Output:
        {
            "status": "SUCCESS",
            "region": "us-east-1",
            "s3Key": "executions/{id}/models/us-east-1.json",
            "modelCount": 108
        }
    """
    start_time = time.time()

    # Validate required parameters
    try:
        validate_required_params(event, ['region'], 'ModelExtractor')
    except ValidationError as e:
        return {
            'status': 'FAILED',
            'errorType': 'ValidationError',
            'errorMessage': str(e)
        }

    region = event['region']
    s3_bucket = event.get('s3Bucket')
    s3_key = event.get('s3Key', f'test/models/{region}.json')
    dry_run = event.get('dryRun', False)

    logger.info(f"Extracting models from region: {region}")

    try:
        bedrock_client = get_bedrock_client(region)
        models = extract_models(bedrock_client, region)

        output_data = {
            'metadata': {
                'region': region,
                'model_count': len(models),
                'collection_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            },
            'models': models
        }

        if not dry_run and s3_bucket:
            s3_client = get_s3_client()
            write_to_s3(s3_client, s3_bucket, s3_key, output_data)
        else:
            logger.info(f"Dry run - would write {len(models)} models to s3://{s3_bucket}/{s3_key}")

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            'status': 'SUCCESS',
            'region': region,
            's3Key': s3_key,
            'modelCount': len(models),
            'durationMs': duration_ms
        }

    except Exception as e:
        logger.error(f"Failed to extract models from {region}: {e}", exc_info=True)
        return {
            'status': 'FAILED',
            'region': region,
            'errorType': type(e).__name__,
            'errorMessage': str(e),
            'retryable': 'Throttling' in str(e)
        }
