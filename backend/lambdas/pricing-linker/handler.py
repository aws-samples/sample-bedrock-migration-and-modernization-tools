"""
Pricing Linker Lambda - V2 (With PORT Features)

Links pricing data to models, creating price references per model per region.
Works with the correct snake_case schema.

V2 Features (ported from reference implementation):
- Provider-scoped matching: Only matches within same provider
- Conflict detection: Blocks semantic mismatches (haiku/sonnet, 8b/405b)
- Enhanced normalization: Provider-specific rules for edge cases
"""

import logging
import os
import re
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

# Minimum confidence threshold for accepting a pricing match
MIN_CONFIDENCE_THRESHOLD = 0.7

# Provider name aliases for matching variations
PROVIDER_ALIASES = {
    'amazon': {'amazon', 'aws'},
    'anthropic': {'anthropic'},
    'meta': {'meta', 'facebook'},
    'mistral ai': {'mistral', 'mistralai', 'mistral ai'},
    'stability ai': {'stability', 'stabilityai', 'stability ai'},
    'cohere': {'cohere'},
    'ai21 labs': {'ai21', 'ai21labs', 'ai21 labs'},
    'luma ai': {'luma', 'lumaai', 'luma ai'},
    'twelvelabs': {'twelvelabs', 'twelve labs', 'twelverlabs'},
    'minimax': {'minimax', 'minimax ai', 'minimax-ai'},
    'moonshot ai': {'moonshot', 'moonshot ai', 'kimi', 'kimi ai'},
    'deepseek': {'deepseek'},
    'qwen': {'qwen', 'qwen2', 'alibaba'},
    'google': {'google'},
    'nvidia': {'nvidia'},
    'openai': {'openai'},
    'writer': {'writer'},
}


def similarity_score(a: str, b: str) -> float:
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def has_on_demand_pricing(pricing_data: dict) -> bool:
    """Check if pricing data has On-Demand pricing in at least one region."""
    if not pricing_data or not isinstance(pricing_data, dict):
        return False

    regions = pricing_data.get('regions', {})
    if not isinstance(regions, dict):
        return False

    for region_data in regions.values():
        if not isinstance(region_data, dict):
            continue
        pricing_groups = region_data.get('pricing_groups', {})
        on_demand = pricing_groups.get('On-Demand', [])
        if on_demand:
            return True
    return False


# =============================================================================
# PORT Feature 1: Provider-Scoped Matching
# =============================================================================

def extract_provider_from_model_id(model_id: str) -> str:
    """
    Extract provider name from model ID.

    Examples:
        'meta.llama3-8b' -> 'meta'
        'anthropic.claude-3-sonnet' -> 'anthropic'
        'amazon.titan-text-express' -> 'amazon'
    """
    if '.' in model_id:
        return model_id.split('.')[0].lower()
    return ''


def providers_match(model_provider: str, pricing_provider: str) -> bool:
    """
    Check if model provider matches pricing provider with alias support.

    Handles variations like:
        - "Stability AI" vs "stability"
        - "AI21 Labs" vs "ai21"
        - "Mistral AI" vs "mistralai"
    """
    if not model_provider or not pricing_provider:
        return False

    model_provider_lower = model_provider.lower().strip()
    pricing_provider_lower = pricing_provider.lower().strip()

    # Direct match
    if model_provider_lower == pricing_provider_lower:
        return True

    # Check alias mappings
    for canonical, aliases in PROVIDER_ALIASES.items():
        model_in_aliases = model_provider_lower in aliases or any(
            alias in model_provider_lower or model_provider_lower in alias
            for alias in aliases
        )
        pricing_in_aliases = pricing_provider_lower in aliases or any(
            alias in pricing_provider_lower or pricing_provider_lower in alias
            for alias in aliases
        )
        if model_in_aliases and pricing_in_aliases:
            return True

    return False


# =============================================================================
# PORT Feature 2: Conflict Detection
# =============================================================================

def has_semantic_conflict(model_name: str, pricing_name: str, model_id: str = '', pricing_key: str = '') -> bool:
    """
    Detect semantic conflicts that should block a match.
    Returns True if there's a conflict (models shouldn't match).

    Prevents:
        - Claude variant mismatches (haiku/sonnet/opus)
        - Model type mismatches (embed/generator)
        - Size mismatches (8b vs 405b)
    """
    model_lower = (model_name + ' ' + model_id).lower()
    pricing_lower = (pricing_name + ' ' + pricing_key).lower()

    # Claude model variant conflicts - these are distinct models
    claude_variants = ['haiku', 'sonnet', 'opus']
    model_claude_variant = None
    pricing_claude_variant = None

    for variant in claude_variants:
        if variant in model_lower:
            model_claude_variant = variant
        if variant in pricing_lower:
            pricing_claude_variant = variant

    if model_claude_variant and pricing_claude_variant and model_claude_variant != pricing_claude_variant:
        return True

    # Nova model variant conflicts
    nova_variants = ['micro', 'lite', 'pro', 'premier', 'canvas', 'reel', 'sonic']
    model_nova_variant = None
    pricing_nova_variant = None

    for variant in nova_variants:
        if f'nova-{variant}' in model_lower or f'nova {variant}' in model_lower:
            model_nova_variant = variant
        if f'nova-{variant}' in pricing_lower or f'nova {variant}' in pricing_lower:
            pricing_nova_variant = variant

    if model_nova_variant and pricing_nova_variant and model_nova_variant != pricing_nova_variant:
        return True

    # Llama size conflicts
    llama_sizes = ['8b', '70b', '405b', '11b', '90b', '1b', '3b']
    if 'llama' in model_lower and 'llama' in pricing_lower:
        model_llama_size = None
        pricing_llama_size = None
        for size in llama_sizes:
            if size in model_lower:
                model_llama_size = size
            if size in pricing_lower:
                pricing_llama_size = size
        if model_llama_size and pricing_llama_size and model_llama_size != pricing_llama_size:
            return True

    # Model type conflicts - embedding vs generation models
    type_conflicts = [
        (['embed', 'embedding'], ['generator', 'generation', 'chat', 'instruct']),
        (['rerank'], ['embed', 'embedding', 'chat']),
        (['image-generator', 'imagegenerator'], ['text', 'chat', 'embed']),
    ]

    for type_group1, type_group2 in type_conflicts:
        model_has_type1 = any(t in model_lower for t in type_group1)
        pricing_has_type2 = any(t in pricing_lower for t in type_group2)
        model_has_type2 = any(t in model_lower for t in type_group2)
        pricing_has_type1 = any(t in pricing_lower for t in type_group1)

        if (model_has_type1 and pricing_has_type2) or (model_has_type2 and pricing_has_type1):
            return True

    # General size mismatch detection (e.g., 8B vs 70B)
    size_pattern = re.compile(r'(\d+)b\b', re.IGNORECASE)
    model_sizes = size_pattern.findall(model_lower)
    pricing_sizes = size_pattern.findall(pricing_lower)

    if model_sizes and pricing_sizes:
        model_size = int(model_sizes[0])
        pricing_size = int(pricing_sizes[0])
        # Allow 20% variance for rounding, but block major mismatches
        max_size = max(model_size, pricing_size)
        if max_size > 0 and abs(model_size - pricing_size) > max_size * 0.3:
            return True

    return False


# =============================================================================
# PORT Feature 3: Enhanced Normalization
# =============================================================================

def normalize_model_id(model_id: str, provider: str = '') -> str:
    """
    Normalize model ID for matching by removing common suffixes and normalizing format.
    Includes provider-specific normalization rules.

    Args:
        model_id: The model identifier to normalize
        provider: Optional provider name for provider-specific rules
    """
    normalized = model_id.lower()
    provider_lower = provider.lower() if provider else ''

    # Remove common suffixes that differ between APIs
    suffixes_to_remove = ['-it', '-instruct', '-chat', '-v1', '-v2', '-v3', ':0', ':1', ':2']
    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]

    # Provider-specific normalization rules
    if 'qwen' in provider_lower or 'qwen' in normalized:
        # Qwen models: remove 'instruct' variations that may differ
        normalized = re.sub(r'[-_]?instruct', '', normalized)

    if 'deepseek' in provider_lower or 'deepseek' in normalized:
        # DeepSeek: normalize version formats (DeepSeek-V3 -> deepseek3)
        normalized = re.sub(r'[-_]?v(\d+)', r'\1', normalized)

    if 'cohere' in provider_lower or 'cohere' in normalized:
        # Cohere: remove 'model' keyword that may differ
        normalized = re.sub(r'[-_]?model', '', normalized)

    if 'stability' in provider_lower or 'stability' in normalized:
        # Stability: normalize SD versions (sd3, sdxl, etc.)
        normalized = re.sub(r'stable[-_]?diffusion[-_]?', 'sd', normalized)

    # Remove all separators for fuzzy matching
    return normalized.replace('-', '').replace('_', '').replace('.', '').replace(' ', '')


def find_best_pricing_match(
    model_id: str,
    model_name: str,
    model_provider: str,
    pricing_models: dict
) -> tuple[str, float]:
    """
    Find the best matching pricing entry for a model.

    Features:
        - Provider-scoped matching: Only matches within same provider
        - Conflict detection: Blocks semantic mismatches
        - On-Demand prioritization: Prefers entries with On-Demand pricing

    Args:
        model_id: The model identifier
        model_name: Human-readable model name
        model_provider: The model's provider name
        pricing_models: Dict of pricing entries with provider info

    Returns:
        (matched_pricing_key, confidence_score)
    """
    # Track best matches separately for On-Demand and non-On-Demand
    best_on_demand_match = None
    best_on_demand_score = 0.0
    best_other_match = None
    best_other_score = 0.0

    # Normalize model identifiers
    model_id_normalized = normalize_model_id(model_id, model_provider)
    model_name_normalized = model_name.lower().replace('-', '').replace('_', '').replace('.', '').replace(' ', '')

    for pricing_key, pricing_entry in pricing_models.items():
        pricing_data = pricing_entry['data']
        pricing_provider = pricing_entry['provider']
        pricing_model_name = pricing_data.get('model_name', '')

        # PORT Feature 1: Provider-scoped matching
        if not providers_match(model_provider, pricing_provider):
            continue

        # PORT Feature 2: Conflict detection
        if has_semantic_conflict(model_name, pricing_model_name, model_id, pricing_key):
            continue

        # Normalize pricing identifiers
        pricing_key_normalized = normalize_model_id(pricing_key, pricing_provider)
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

    # Prefer On-Demand matches if score is reasonable
    if best_on_demand_match and best_on_demand_score >= MIN_CONFIDENCE_THRESHOLD:
        return best_on_demand_match, best_on_demand_score

    # Fall back to other matches if no good On-Demand match
    if best_other_match and best_other_score >= best_on_demand_score:
        return best_other_match, best_other_score

    # Return best On-Demand match even if score is low (as last resort)
    if best_on_demand_match:
        return best_on_demand_match, best_on_demand_score

    return best_other_match, best_other_score if best_other_match else 0.0


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
                all_pricing_models[provider_name] = {'provider': provider_name, 'data': data}
            elif 'models' in data:
                # Old nested structure: provider -> models -> model_id -> pricing
                for model_key, model_pricing in data.get('models', {}).items():
                    all_pricing_models[model_key] = {'provider': provider_name, 'data': model_pricing}
            else:
                # New nested structure: provider -> model_id -> {model_name, model_provider, regions}
                for model_key, model_pricing in data.items():
                    if isinstance(model_pricing, dict) and 'regions' in model_pricing:
                        all_pricing_models[model_key] = {'provider': provider_name, 'data': model_pricing}

    # Process each provider and model
    for provider, provider_data in models_data.get('providers', {}).items():
        for model_id, model in provider_data.get('models', {}).items():
            model_name = model.get('model_name', model_id)
            model_provider = model.get('model_provider', provider)

            # Find matching pricing (now with provider scoping and conflict detection)
            matched_key, confidence = find_best_pricing_match(
                model_id, model_name, model_provider, all_pricing_models
            )

            if matched_key and confidence >= MIN_CONFIDENCE_THRESHOLD:
                pricing_entry = all_pricing_models[matched_key]
                pricing_info = pricing_entry['data']
                pricing_provider = pricing_entry['provider']
                pricing_regions = pricing_info.get('regions', {})

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

    logger.info("Linking pricing to models (V2 with PORT features)")

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
                    'version': 'v2-port-features',
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
