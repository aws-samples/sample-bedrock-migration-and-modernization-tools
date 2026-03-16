"""
Pricing Linker - Local Handler (V2 with PORT Features)

Links pricing data to models, creating price references per model per region.
Features: Provider-scoped matching, conflict detection, enhanced normalization.

This is a standalone version of backend/lambdas/pricing-linker/handler.py
"""

import re
from difflib import SequenceMatcher

# Configuration
PROVIDER_ALIASES = {
    'anthropic': {'anthropic', 'claude'},
    'amazon': {'amazon', 'titan', 'nova'},
    'meta': {'meta', 'llama'},
    'mistral': {'mistral', 'mistralai', 'mistral ai'},
    'cohere': {'cohere'},
    'ai21': {'ai21', 'ai21 labs', 'ai21labs'},
    'stability': {'stability', 'stability ai', 'stable'},
}

CLAUDE_VARIANTS = ['opus', 'sonnet', 'haiku', 'instant']
NOVA_VARIANTS = ['micro', 'lite', 'pro', 'premier', 'canvas', 'reel']
LLAMA_SIZES = ['8b', '11b', '70b', '90b', '405b', '1b', '3b']
MIN_CONFIDENCE_THRESHOLD = 0.65


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


def providers_match(model_provider: str, pricing_provider: str) -> bool:
    """Check if model provider matches pricing provider with alias support."""
    if not model_provider or not pricing_provider:
        return False

    model_provider_lower = model_provider.lower().strip()
    pricing_provider_lower = pricing_provider.lower().strip()

    if model_provider_lower == pricing_provider_lower:
        return True

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


def has_semantic_conflict(model_name: str, pricing_name: str, model_id: str = '', pricing_key: str = '') -> bool:
    """Detect semantic conflicts that should block a match."""
    model_lower = (model_name + ' ' + model_id).lower()
    pricing_lower = (pricing_name + ' ' + pricing_key).lower()

    # Claude variant conflicts
    model_claude_variant = None
    pricing_claude_variant = None
    for variant in CLAUDE_VARIANTS:
        if variant in model_lower:
            model_claude_variant = variant
        if variant in pricing_lower:
            pricing_claude_variant = variant
    if model_claude_variant and pricing_claude_variant and model_claude_variant != pricing_claude_variant:
        return True

    # Nova variant conflicts
    model_nova_variant = None
    pricing_nova_variant = None
    for variant in NOVA_VARIANTS:
        if f'nova-{variant}' in model_lower or f'nova {variant}' in model_lower:
            model_nova_variant = variant
        if f'nova-{variant}' in pricing_lower or f'nova {variant}' in pricing_lower:
            pricing_nova_variant = variant
    if model_nova_variant and pricing_nova_variant and model_nova_variant != pricing_nova_variant:
        return True

    # Llama size conflicts
    if 'llama' in model_lower and 'llama' in pricing_lower:
        model_llama_size = None
        pricing_llama_size = None
        for size in LLAMA_SIZES:
            if size in model_lower:
                model_llama_size = size
            if size in pricing_lower:
                pricing_llama_size = size
        if model_llama_size and pricing_llama_size and model_llama_size != pricing_llama_size:
            return True

    # Model type conflicts
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

    # General size mismatch detection
    size_pattern = re.compile(r'(\d+)b\b', re.IGNORECASE)
    model_sizes = size_pattern.findall(model_lower)
    pricing_sizes = size_pattern.findall(pricing_lower)
    if model_sizes and pricing_sizes:
        model_size = int(model_sizes[0])
        pricing_size = int(pricing_sizes[0])
        max_size = max(model_size, pricing_size)
        if max_size > 0 and abs(model_size - pricing_size) > max_size * 0.3:
            return True

    return False


def normalize_model_id(model_id: str, provider: str = '') -> str:
    """Normalize model ID for matching."""
    normalized = model_id.lower()
    provider_lower = provider.lower() if provider else ''

    suffixes_to_remove = ['-it', '-instruct', '-chat', '-v1', '-v2', '-v3', ':0', ':1', ':2']
    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]

    if 'qwen' in provider_lower or 'qwen' in normalized:
        normalized = re.sub(r'[-_]?instruct', '', normalized)
    if 'deepseek' in provider_lower or 'deepseek' in normalized:
        normalized = re.sub(r'[-_]?v(\d+)', r'\1', normalized)
    if 'cohere' in provider_lower or 'cohere' in normalized:
        normalized = re.sub(r'[-_]?model', '', normalized)
    if 'stability' in provider_lower or 'stability' in normalized:
        normalized = re.sub(r'stable[-_]?diffusion[-_]?', 'sd', normalized)

    return normalized.replace('-', '').replace('_', '').replace('.', '').replace(' ', '')


def find_best_pricing_match(model_id: str, model_name: str, model_provider: str, pricing_models: dict) -> tuple:
    """Find the best matching pricing entry for a model."""
    best_on_demand_match = None
    best_on_demand_score = 0.0
    best_other_match = None
    best_other_score = 0.0

    model_id_normalized = normalize_model_id(model_id, model_provider)
    model_name_normalized = model_name.lower().replace('-', '').replace('_', '').replace('.', '').replace(' ', '')

    for pricing_key, pricing_entry in pricing_models.items():
        pricing_data = pricing_entry['data']
        pricing_provider = pricing_entry['provider']
        pricing_model_name = pricing_data.get('model_name', '')

        if not providers_match(model_provider, pricing_provider):
            continue

        if has_semantic_conflict(model_name, pricing_model_name, model_id, pricing_key):
            continue

        pricing_key_normalized = normalize_model_id(pricing_key, pricing_provider)
        pricing_name_normalized = pricing_model_name.lower().replace('-', '').replace('_', '').replace('.', '').replace(' ', '')

        score = 0.0
        if model_id_normalized == pricing_key_normalized:
            score = 1.0
        elif model_name_normalized == pricing_name_normalized:
            score = 1.0
        elif model_id_normalized.startswith(pricing_key_normalized) or pricing_key_normalized.startswith(model_id_normalized):
            score = 0.95
        elif model_name_normalized.startswith(pricing_name_normalized) or pricing_name_normalized.startswith(model_name_normalized):
            score = 0.95
        else:
            score = max(
                similarity_score(model_id_normalized, pricing_key_normalized),
                similarity_score(model_name_normalized, pricing_name_normalized),
                similarity_score(model_id_normalized, pricing_name_normalized)
            )

        if has_on_demand_pricing(pricing_data):
            if score > best_on_demand_score:
                best_on_demand_score = score
                best_on_demand_match = pricing_key
        else:
            if score > best_other_score:
                best_other_score = score
                best_other_match = pricing_key

    if best_on_demand_match and best_on_demand_score >= MIN_CONFIDENCE_THRESHOLD:
        return best_on_demand_match, best_on_demand_score
    if best_other_match and best_other_score >= best_on_demand_score:
        return best_other_match, best_other_score
    if best_on_demand_match:
        return best_on_demand_match, best_on_demand_score
    return best_other_match, best_other_score if best_other_match else 0.0


def link_pricing_to_models(models_data: dict, pricing_data: dict) -> dict:
    """Link pricing information to each model."""
    models_with_pricing = 0
    models_without_pricing = 0

    # Flatten pricing models for matching
    all_pricing_models = {}
    for provider_name, data in pricing_data.get('providers', {}).items():
        if isinstance(data, dict):
            if 'regions' in data:
                all_pricing_models[provider_name] = {'provider': provider_name, 'data': data}
            elif 'models' in data:
                for model_key, model_pricing in data.get('models', {}).items():
                    all_pricing_models[model_key] = {'provider': provider_name, 'data': model_pricing}
            else:
                for model_key, model_pricing in data.items():
                    if isinstance(model_pricing, dict) and 'regions' in model_pricing:
                        all_pricing_models[model_key] = {'provider': provider_name, 'data': model_pricing}

    for provider, provider_data in models_data.get('providers', {}).items():
        for model_id, model in provider_data.get('models', {}).items():
            model_name = model.get('model_name', model_id)
            model_provider = model.get('model_provider', provider)

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
