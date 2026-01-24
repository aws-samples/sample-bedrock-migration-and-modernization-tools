"""
Model Enricher Lambda

Enriches models with capabilities, use cases, and documentation links.
Derives this information from model metadata (modalities, provider, model name).
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
    get_config_loader,
)

logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

# Configuration loader - initialized on first use
_config_loader = None


def _get_config():
    """Get the configuration loader (lazy initialization)."""
    global _config_loader
    if _config_loader is None:
        _config_loader = get_config_loader()
        _config_loader.load_config()
    return _config_loader




def extract_capabilities(model_data: dict) -> list[str]:
    """Extract model capabilities from modalities and model info."""
    capabilities = set()

    # Get modalities from nested structure (snake_case schema)
    modalities = model_data.get('model_modalities', {})
    input_modalities = modalities.get('input_modalities', [])
    output_modalities = modalities.get('output_modalities', [])
    model_name = model_data.get('model_name', '').lower()
    model_id = model_data.get('model_id', '').lower()

    # Text capabilities
    if 'TEXT' in input_modalities and 'TEXT' in output_modalities:
        capabilities.update(['text_generation', 'chat'])

    # Vision capabilities
    if 'IMAGE' in input_modalities:
        capabilities.update(['vision', 'image_understanding', 'multimodal'])

    # Video capabilities
    if 'VIDEO' in input_modalities:
        capabilities.update(['video_understanding', 'multimodal'])

    # Image generation
    if 'IMAGE' in output_modalities:
        capabilities.update(['image_generation', 'text_to_image'])

    # Video generation
    if 'VIDEO' in output_modalities:
        capabilities.update(['video_generation'])

    # Embedding capabilities
    if 'EMBEDDING' in output_modalities or 'embed' in model_name:
        capabilities.update(['embeddings', 'semantic_search'])
        if 'image' in model_name:
            capabilities.add('image_embeddings')
        else:
            capabilities.add('text_embeddings')

    # Code capabilities (inferred from model name/family)
    if any(term in model_id for term in ['code', 'codestral', 'devstral']):
        capabilities.add('code_generation')

    # Reasoning capability for advanced models
    if any(term in model_id for term in ['claude', 'sonnet', 'opus', 'nova', 'llama', 'mistral', 'command']):
        capabilities.add('reasoning')

    # Function calling (most modern LLMs support this)
    if any(term in model_id for term in ['claude', 'nova', 'llama-3', 'mistral', 'command-r']):
        capabilities.add('function_calling')

    # Streaming support (snake_case field name)
    if model_data.get('streaming_supported', False):
        capabilities.add('streaming')

    return sorted(list(capabilities))


def extract_use_cases(capabilities: list[str], model_data: dict) -> list[str]:
    """Extract use cases based on capabilities."""
    use_cases = set()

    capability_to_use_cases = {
        'chat': ['conversational_ai', 'customer_support', 'virtual_assistants'],
        'text_generation': ['content_creation', 'creative_writing', 'summarization', 'translation'],
        'reasoning': ['complex_analysis', 'problem_solving', 'decision_support'],
        'code_generation': ['software_development', 'code_review', 'debugging', 'documentation'],
        'vision': ['image_analysis', 'visual_qa', 'document_analysis', 'ocr'],
        'image_understanding': ['content_moderation', 'product_cataloging'],
        'video_understanding': ['video_analysis', 'content_moderation'],
        'image_generation': ['creative_design', 'marketing_assets', 'prototyping'],
        'video_generation': ['video_production', 'animation'],
        'embeddings': ['semantic_search', 'recommendation_systems', 'clustering'],
        'text_embeddings': ['document_retrieval', 'similarity_matching'],
        'image_embeddings': ['image_search', 'visual_similarity'],
        'function_calling': ['api_integration', 'workflow_automation', 'tool_use'],
    }

    for capability in capabilities:
        if capability in capability_to_use_cases:
            use_cases.update(capability_to_use_cases[capability])

    return sorted(list(use_cases))


def get_documentation_links(model_data: dict) -> dict:
    """Get documentation links based on provider and model."""
    provider = model_data.get('model_provider', '')
    model_id = model_data.get('model_id', '').lower()

    # Get documentation links from config
    config = _get_config()
    all_docs = config.get_documentation_links()

    # Check for Nova models (Amazon's newer models)
    if 'nova' in model_id:
        nova_docs = all_docs.get('nova', all_docs.get('default', {}))
        return nova_docs.copy()

    # Get provider-specific docs or default
    return all_docs.get(provider, all_docs.get('default', {})).copy()


def enrich_model(model_data: dict) -> dict:
    """Enrich a single model with capabilities, use cases, and documentation."""
    enriched = model_data.copy()

    # Extract capabilities
    capabilities = extract_capabilities(model_data)
    enriched['model_capabilities'] = capabilities

    # Extract use cases
    use_cases = extract_use_cases(capabilities, model_data)
    enriched['model_use_cases'] = use_cases

    # Get documentation links (matching expected schema)
    enriched['documentation_links'] = get_documentation_links(model_data)

    return enriched


def enrich_providers(providers: dict) -> dict:
    """Enrich all models in the providers structure."""
    enriched_providers = {}

    for provider_name, provider_data in providers.items():
        enriched_providers[provider_name] = {'models': {}}

        for model_id, model_data in provider_data.get('models', {}).items():
            enriched_providers[provider_name]['models'][model_id] = enrich_model(model_data)

    return enriched_providers


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for model enrichment.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "modelsS3Key": "executions/{id}/merged/models.json"
        }

    Output:
        {
            "status": "SUCCESS",
            "s3Key": "executions/{id}/enriched/models.json",
            "modelsEnriched": 147
        }
    """
    start_time = time.time()

    # Validate required parameters
    try:
        validate_required_params(event, ['s3Bucket', 'executionId', 'modelsS3Key'], 'ModelEnricher')
    except ValidationError as e:
        return {
            'status': 'FAILED',
            'errorType': 'ValidationError',
            'errorMessage': str(e)
        }

    s3_bucket = event['s3Bucket']
    execution_id = parse_execution_id(event['executionId'])
    models_s3_key = event['modelsS3Key']
    dry_run = event.get('dryRun', False)

    output_key = f"executions/{execution_id}/enriched/models.json"

    logger.info(f"Enriching models from {models_s3_key}")

    try:
        s3_client = get_s3_client()

        if not dry_run:
            # Read models data
            models_data = read_from_s3(s3_client, s3_bucket, models_s3_key)
            providers = models_data.get('providers', {})

            # Enrich all models
            enriched_providers = enrich_providers(providers)

            # Calculate statistics
            total_models = sum(
                len(p.get('models', {})) for p in enriched_providers.values()
            )

            # Build output
            output_data = {
                'metadata': {
                    'total_models': total_models,
                    'providers_count': len(enriched_providers),
                    'enrichment_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                },
                'providers': enriched_providers
            }

            # Write to S3
            write_to_s3(s3_client, s3_bucket, output_key, output_data)
        else:
            logger.info("Dry run - skipping enrichment")
            total_models = 0

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            'status': 'SUCCESS',
            's3Key': output_key,
            'modelsEnriched': total_models,
            'durationMs': duration_ms
        }

    except Exception as e:
        logger.error(f"Failed to enrich models: {e}", exc_info=True)
        return {
            'status': 'FAILED',
            'errorType': type(e).__name__,
            'errorMessage': str(e)
        }
