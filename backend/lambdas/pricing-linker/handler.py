"""
Pricing Linker Lambda

Links pricing data to models, creating price references per model per region.
Works with the correct snake_case schema.
"""

import json
import logging
import os
import time
from typing import Any
from difflib import SequenceMatcher

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


def similarity_score(a: str, b: str) -> float:
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_best_pricing_match(model_id: str, model_name: str, pricing_models: dict) -> tuple[str, float]:
    """
    Find the best matching pricing entry for a model.

    Returns (matched_pricing_key, confidence_score)
    """
    best_match = None
    best_score = 0.0

    # Normalize model ID for matching
    model_id_normalized = model_id.lower().replace('-', '').replace('_', '').replace('.', '')
    model_name_normalized = model_name.lower().replace('-', '').replace('_', '').replace('.', '')

    for pricing_key, pricing_data in pricing_models.items():
        pricing_model_name = pricing_data.get('model_name', '')
        pricing_key_normalized = pricing_key.lower().replace('-', '').replace('_', '').replace('.', '')
        pricing_name_normalized = pricing_model_name.lower().replace('-', '').replace('_', '').replace('.', '')

        # Check for exact matches first
        if model_id_normalized == pricing_key_normalized:
            return pricing_key, 1.0

        if model_name_normalized == pricing_name_normalized:
            return pricing_key, 1.0

        # Check for partial matches
        score = max(
            similarity_score(model_id_normalized, pricing_key_normalized),
            similarity_score(model_name_normalized, pricing_name_normalized),
            similarity_score(model_id_normalized, pricing_name_normalized)
        )

        if score > best_score:
            best_score = score
            best_match = pricing_key

    return best_match, best_score


def link_pricing_to_models(models_data: dict, pricing_data: dict) -> dict:
    """
    Link pricing information to each model.

    Returns updated models structure with pricing references in correct schema.
    """
    models_with_pricing = 0
    models_without_pricing = 0

    # Flatten pricing models for easier matching
    all_pricing_models = {}
    for key, data in pricing_data.get('providers', {}).items():
        if isinstance(data, dict):
            if 'regions' in data:
                # Flat structure: model_id -> {model_name, model_provider, regions}
                # The key is the model_id directly at top level of providers
                all_pricing_models[key] = data
            elif 'models' in data:
                # Old nested structure: provider -> models -> model_id -> pricing
                for model_key, model_pricing in data.get('models', {}).items():
                    all_pricing_models[model_key] = model_pricing
            else:
                # New nested structure: provider -> model_id -> {model_name, model_provider, regions}
                # The key is the provider name, data contains model entries
                for model_key, model_pricing in data.items():
                    if isinstance(model_pricing, dict) and 'regions' in model_pricing:
                        all_pricing_models[model_key] = model_pricing

    # Process each provider and model
    for provider, provider_data in models_data.get('providers', {}).items():
        for model_id, model in provider_data.get('models', {}).items():
            # Use snake_case field names
            model_name = model.get('model_name', model_id)

            # Find matching pricing
            matched_key, confidence = find_best_pricing_match(
                model_id, model_name, all_pricing_models
            )

            if matched_key and confidence >= 0.7:
                pricing_info = all_pricing_models[matched_key]
                pricing_regions = pricing_info.get('regions', {})

                # Store as model_pricing to match expected schema
                model['model_pricing'] = {
                    'is_pricing_available': True,
                    'pricing_reference_id': matched_key,
                    'confidence': round(confidence, 3),
                    'regions': pricing_regions,
                    'total_regions': len(pricing_regions)
                }
                model['has_pricing'] = True
                models_with_pricing += 1
            else:
                model['model_pricing'] = {
                    'is_pricing_available': False,
                    'pricing_reference_id': None,
                    'confidence': round(confidence, 3) if matched_key else 0,
                    'regions': {},
                    'total_regions': 0
                }
                model['has_pricing'] = False
                models_without_pricing += 1

    return {
        'models_with_pricing': models_with_pricing,
        'models_without_pricing': models_without_pricing,
        'providers': models_data.get('providers', {})
    }


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for pricing linking.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "pricingS3Key": "executions/{id}/merged/pricing.json",
            "modelsS3Key": "executions/{id}/merged/models.json"
        }

    Output:
        {
            "status": "SUCCESS",
            "s3Key": "executions/{id}/intermediate/models-with-pricing.json",
            "modelsWithPricing": 86,
            "modelsWithoutPricing": 22
        }
    """
    start_time = time.time()

    s3_bucket = event['s3Bucket']
    execution_id = event['executionId']
    pricing_s3_key = event['pricingS3Key']
    models_s3_key = event['modelsS3Key']
    dry_run = event.get('dryRun', False)

    if ':' in execution_id:
        execution_id = execution_id.split(':')[-1]

    output_key = f"executions/{execution_id}/intermediate/models-with-pricing.json"

    logger.info(f"Linking pricing to models")

    try:
        s3_client = get_s3_client()

        if not dry_run:
            # Read pricing and models data
            pricing_data = read_from_s3(s3_client, s3_bucket, pricing_s3_key)
            models_data = read_from_s3(s3_client, s3_bucket, models_s3_key)

            # Link pricing to models
            result = link_pricing_to_models(models_data, pricing_data)

            output_data = {
                'metadata': {
                    'models_with_pricing': result['models_with_pricing'],
                    'models_without_pricing': result['models_without_pricing'],
                    'collection_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                },
                'providers': result['providers']
            }

            write_to_s3(s3_client, s3_bucket, output_key, output_data)

            models_with_pricing = result['models_with_pricing']
            models_without_pricing = result['models_without_pricing']
        else:
            logger.info("Dry run - skipping processing")
            models_with_pricing = 0
            models_without_pricing = 0

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            'status': 'SUCCESS',
            's3Key': output_key,
            'modelsWithPricing': models_with_pricing,
            'modelsWithoutPricing': models_without_pricing,
            'durationMs': duration_ms
        }

    except Exception as e:
        logger.error(f"Failed to link pricing: {e}", exc_info=True)
        return {
            'status': 'FAILED',
            'errorType': type(e).__name__,
            'errorMessage': str(e)
        }
