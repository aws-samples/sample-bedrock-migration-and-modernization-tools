"""
Pricing Linker Lambda - V1 (Original Implementation)

Links pricing data to models, creating price references per model per region.
Works with the correct snake_case schema.

This is the original implementation before PORT features were added.
Kept for A/B testing and comparison purposes.
"""

import logging
import os
import time
from typing import Any
from difflib import SequenceMatcher

from shared import (
    get_s3_client,
    read_from_s3,
    write_to_s3,
    parse_execution_id,
    validate_required_params,
    ValidationError,
    S3ReadError,
)

logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))


def similarity_score(a: str, b: str) -> float:
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def has_on_demand_pricing(pricing_data: dict) -> bool:
    """Check if pricing data has On-Demand pricing in at least one region."""
    regions = pricing_data.get('regions', {})
    for region_data in regions.values():
        pricing_groups = region_data.get('pricing_groups', {})
        on_demand = pricing_groups.get('On-Demand', [])
        if on_demand:
            return True
    return False


def normalize_model_id(model_id: str) -> str:
    """Normalize model ID for matching by removing common suffixes and normalizing format."""
    normalized = model_id.lower()

    # Remove common suffixes that differ between APIs
    # e.g., google.gemma-3-12b-it -> google.gemma-3-12b
    suffixes_to_remove = ['-it', '-instruct', '-chat', '-v1', '-v2', '-v3']
    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]

    # Remove all separators for fuzzy matching
    return normalized.replace('-', '').replace('_', '').replace('.', '')


def find_best_pricing_match(model_id: str, model_name: str, pricing_models: dict) -> tuple[str, float]:
    """
    Find the best matching pricing entry for a model.
    Prioritizes matches that have On-Demand pricing over those that don't.

    Returns (matched_pricing_key, confidence_score)
    """
    # Track best matches separately for On-Demand and non-On-Demand
    best_on_demand_match = None
    best_on_demand_score = 0.0
    best_other_match = None
    best_other_score = 0.0

    # Normalize model ID for matching (including suffix removal)
    model_id_normalized = normalize_model_id(model_id)
    model_name_normalized = model_name.lower().replace('-', '').replace('_', '').replace('.', '').replace(' ', '')

    for pricing_key, pricing_data in pricing_models.items():
        pricing_model_name = pricing_data.get('model_name', '')
        pricing_key_normalized = normalize_model_id(pricing_key)
        pricing_name_normalized = pricing_model_name.lower().replace('-', '').replace('_', '').replace('.', '').replace(' ', '')

        # Calculate match score
        score = 0.0

        # Check for exact matches first (after normalization)
        if model_id_normalized == pricing_key_normalized:
            score = 1.0
        elif model_name_normalized == pricing_name_normalized:
            score = 1.0
        # Check if one is a prefix of the other (handles version suffix differences)
        elif model_id_normalized.startswith(pricing_key_normalized) or pricing_key_normalized.startswith(model_id_normalized):
            score = 0.95
        elif model_name_normalized.startswith(pricing_name_normalized) or pricing_name_normalized.startswith(model_name_normalized):
            score = 0.95
        else:
            # Check for partial matches using similarity
            score = max(
                similarity_score(model_id_normalized, pricing_key_normalized),
                similarity_score(model_name_normalized, pricing_name_normalized),
                similarity_score(model_id_normalized, pricing_name_normalized)
            )

        # Track separately based on whether pricing has On-Demand tier
        if has_on_demand_pricing(pricing_data):
            if score > best_on_demand_score:
                best_on_demand_score = score
                best_on_demand_match = pricing_key
        else:
            if score > best_other_score:
                best_other_score = score
                best_other_match = pricing_key

    # Prefer On-Demand matches if score is reasonable (>= 0.7)
    if best_on_demand_match and best_on_demand_score >= 0.7:
        return best_on_demand_match, best_on_demand_score

    # Fall back to other matches if no good On-Demand match
    if best_other_match and best_other_score > best_on_demand_score:
        return best_other_match, best_other_score

    # Return best On-Demand match even if score is low
    if best_on_demand_match:
        return best_on_demand_match, best_on_demand_score

    return best_other_match, best_other_score


def link_pricing_to_models(models_data: dict, pricing_data: dict) -> dict:
    """
    Link pricing information to each model.

    Returns updated models structure with pricing references in correct schema.
    """
    models_with_pricing = 0
    models_without_pricing = 0

    # Flatten pricing models for easier matching, tracking provider for each
    # Structure: { model_key: { 'provider': provider_name, 'data': pricing_data } }
    all_pricing_models = {}
    for provider_name, data in pricing_data.get('providers', {}).items():
        if isinstance(data, dict):
            if 'regions' in data:
                # Flat structure: model_id -> {model_name, model_provider, regions}
                # The key is the model_id directly at top level of providers
                all_pricing_models[provider_name] = {'provider': provider_name, 'data': data}
            elif 'models' in data:
                # Old nested structure: provider -> models -> model_id -> pricing
                for model_key, model_pricing in data.get('models', {}).items():
                    all_pricing_models[model_key] = {'provider': provider_name, 'data': model_pricing}
            else:
                # New nested structure: provider -> model_id -> {model_name, model_provider, regions}
                # The key is the provider name, data contains model entries
                for model_key, model_pricing in data.items():
                    if isinstance(model_pricing, dict) and 'regions' in model_pricing:
                        all_pricing_models[model_key] = {'provider': provider_name, 'data': model_pricing}

    # Create a simplified dict for matching (just the data)
    pricing_data_only = {k: v['data'] for k, v in all_pricing_models.items()}

    # Process each provider and model
    for provider, provider_data in models_data.get('providers', {}).items():
        for model_id, model in provider_data.get('models', {}).items():
            # Use snake_case field names
            model_name = model.get('model_name', model_id)

            # Find matching pricing
            matched_key, confidence = find_best_pricing_match(
                model_id, model_name, pricing_data_only
            )

            if matched_key and confidence >= 0.7:
                pricing_entry = all_pricing_models[matched_key]
                pricing_info = pricing_entry['data']
                pricing_provider = pricing_entry['provider']
                pricing_regions = pricing_info.get('regions', {})

                # Store as model_pricing to match expected schema
                # Include pricing_file_reference for frontend compatibility
                model['model_pricing'] = {
                    'is_pricing_available': True,
                    'pricing_reference_id': matched_key,
                    'pricing_file_reference': {
                        'provider': pricing_provider,
                        'model_key': matched_key
                    },
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
                    'pricing_file_reference': None,
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

    # Validate required parameters
    try:
        validate_required_params(event, ['s3Bucket', 'executionId', 'pricingS3Key', 'modelsS3Key'], 'PricingLinker')
    except ValidationError as e:
        return {
            'status': 'FAILED',
            'errorType': 'ValidationError',
            'errorMessage': str(e)
        }

    s3_bucket = event['s3Bucket']
    execution_id = parse_execution_id(event['executionId'])
    pricing_s3_key = event['pricingS3Key']
    models_s3_key = event['modelsS3Key']
    dry_run = event.get('dryRun', False)

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
