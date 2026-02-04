"""
Model Merger Lambda

Merges and deduplicates models collected from multiple regions.
Works with the correct snake_case schema.
"""

import logging
import os
import time
from typing import Any

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


def get_base_model_id(model_id: str) -> str:
    """
    Extract the base model ID by removing context window suffixes.

    Examples:
        'anthropic.claude-3-5-sonnet-20240620-v1:0:18k' -> 'anthropic.claude-3-5-sonnet-20240620-v1:0'
        'anthropic.claude-3-5-sonnet-20240620-v1:0:200k' -> 'anthropic.claude-3-5-sonnet-20240620-v1:0'
        'anthropic.claude-3-5-sonnet-20240620-v1:0' -> 'anthropic.claude-3-5-sonnet-20240620-v1:0'
    """
    # Check for context window suffixes like :18k, :200k, :51k, :28k
    import re
    # Pattern matches :NNNk at the end (where N is a digit)
    pattern = r':\d+k$'
    return re.sub(pattern, '', model_id)


def merge_models(all_models: list[dict]) -> dict:
    """
    Merge models from multiple regions, deduplicating by model_id.

    Also deduplicates context window variants (e.g., :18k, :200k, :51k)
    by keeping only the base model.

    Preserves the snake_case schema and merges regions_available.

    Returns a provider-grouped structure:
    {
        "providers": {
            "Anthropic": {
                "models": {
                    "anthropic.claude-3-sonnet-v1": { ... }
                }
            }
        }
    }
    """
    # Use dict to deduplicate by model_id
    models_by_id = {}

    for model in all_models:
        model_id = model.get('model_id')
        if not model_id:
            continue

        # Get base model ID (remove context window suffixes like :18k, :200k)
        base_model_id = get_base_model_id(model_id)

        # Skip context window variants - only keep base models
        if model_id != base_model_id:
            logger.debug(f"Skipping context variant: {model_id} (base: {base_model_id})")
            continue

        # Keep first occurrence or merge regions_available
        if model_id not in models_by_id:
            models_by_id[model_id] = model.copy()
            # Ensure regions_available is a list
            if 'regions_available' not in models_by_id[model_id]:
                models_by_id[model_id]['regions_available'] = []
        else:
            # Merge regions_available
            existing_regions = set(models_by_id[model_id].get('regions_available', []))
            new_regions = set(model.get('regions_available', []))
            merged_regions = sorted(list(existing_regions | new_regions))
            models_by_id[model_id]['regions_available'] = merged_regions

            # Update collection_metadata.regions_collected_from
            existing_collected = set(
                models_by_id[model_id].get('collection_metadata', {}).get('regions_collected_from', [])
            )
            new_collected = set(
                model.get('collection_metadata', {}).get('regions_collected_from', [])
            )
            merged_collected = sorted(list(existing_collected | new_collected))
            if 'collection_metadata' not in models_by_id[model_id]:
                models_by_id[model_id]['collection_metadata'] = {}
            models_by_id[model_id]['collection_metadata']['regions_collected_from'] = merged_collected

    # Group by provider
    providers = {}
    for model_id, model in models_by_id.items():
        provider = model.get('model_provider', 'Unknown')

        if provider not in providers:
            providers[provider] = {'models': {}}

        providers[provider]['models'][model_id] = model

    return providers


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for model merging.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "modelResults": [
                {"status": "SUCCESS", "region": "us-east-1", "s3Key": "..."},
                ...
            ]
        }

    Output:
        {
            "status": "SUCCESS",
            "s3Key": "executions/{id}/merged/models.json",
            "totalModels": 108,
            "providersCount": 17
        }
    """
    start_time = time.time()

    # Validate required parameters
    try:
        validate_required_params(event, ['s3Bucket', 'executionId', 'modelResults'], 'ModelMerger')
    except ValidationError as e:
        return {
            'status': 'FAILED',
            'errorType': 'ValidationError',
            'errorMessage': str(e)
        }

    s3_bucket = event['s3Bucket']
    execution_id = parse_execution_id(event['executionId'])
    model_results = event['modelResults']
    dry_run = event.get('dryRun', False)

    output_key = f"executions/{execution_id}/merged/models.json"

    logger.info(f"Merging models from {len(model_results)} regions")

    try:
        s3_client = get_s3_client()

        # Collect all models from successful extractors
        all_models = []
        regions_processed = []

        for item in model_results:
            # Handle nested result structure from Map state
            # Successful: { region, result: { status, s3Key } }
            # Failed: { status: "FAILED", region, error }
            nested_result = item.get('result', {})
            status = item.get('status') or nested_result.get('status')
            s3_key = item.get('s3Key') or nested_result.get('s3Key')
            region = item.get('region')

            if status == 'SUCCESS' and s3_key:
                if not dry_run:
                    data = read_from_s3(s3_client, s3_bucket, s3_key)
                    models = data.get('models', [])
                    all_models.extend(models)
                    regions_processed.append(region)
                    logger.info(f"Loaded {len(models)} models from {region}")
            else:
                logger.warning(f"Skipping non-successful result: {item}")

        # Merge and deduplicate
        providers = merge_models(all_models)

        # Calculate statistics
        providers_count = len(providers)
        total_models = sum(len(p['models']) for p in providers.values())

        output_data = {
            'metadata': {
                'total_models': total_models,
                'providers_count': providers_count,
                'regions_processed': regions_processed,
                'collection_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            },
            'providers': providers
        }

        if not dry_run:
            write_to_s3(s3_client, s3_bucket, output_key, output_data)
        else:
            logger.info(f"Dry run - would write to s3://{s3_bucket}/{output_key}")

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            'status': 'SUCCESS',
            's3Key': output_key,
            'totalModels': total_models,
            'providersCount': providers_count,
            'durationMs': duration_ms
        }

    except Exception as e:
        logger.error(f"Failed to merge models: {e}", exc_info=True)
        return {
            'status': 'FAILED',
            'errorType': type(e).__name__,
            'errorMessage': str(e)
        }
