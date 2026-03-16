"""
Model Merger - Local Handler

Merges and deduplicates models collected from multiple regions.
Extracts context window sizes from model ID variants before deduplication.

This is a standalone version of backend/lambdas/model-merger/handler.py
"""

import re


def get_base_model_id(model_id: str) -> str:
    """Extract the base model ID by removing context window suffixes."""
    return re.sub(r':\d+k$', '', model_id)


def parse_variant_size(model_id: str) -> int | None:
    """Extract context window size from model ID variant suffix."""
    match = re.search(r':(\d+)k$', model_id)
    if match:
        return int(match.group(1)) * 1000
    return None


def merge_models(all_models: list) -> dict:
    """
    Merge models from multiple regions, deduplicating by model_id.

    Also deduplicates context window variants (e.g., :18k, :200k, :51k)
    by keeping only the base model.

    Returns a provider-grouped structure.
    """
    models_by_id = {}
    variant_context_windows = {}

    for model in all_models:
        model_id = model.get('model_id')
        if not model_id:
            continue

        base_model_id = get_base_model_id(model_id)

        # Skip context window variants - but extract size info first
        if model_id != base_model_id:
            size_tokens = parse_variant_size(model_id)
            if size_tokens:
                current_max = variant_context_windows.get(base_model_id, 0)
                variant_context_windows[base_model_id] = max(current_max, size_tokens)
            continue

        if model_id not in models_by_id:
            models_by_id[model_id] = model.copy()
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

            # Merge console_metadata
            existing_console_meta = models_by_id[model_id].get('console_metadata')
            new_console_meta = model.get('console_metadata')
            if not existing_console_meta and new_console_meta:
                models_by_id[model_id]['console_metadata'] = new_console_meta

    # Attach variant context windows to base models
    for model_id, max_size in variant_context_windows.items():
        if model_id in models_by_id:
            models_by_id[model_id]['variant_context_window'] = max_size

    # Group by provider
    providers = {}
    for model_id, model in models_by_id.items():
        provider = model.get('model_provider', 'Unknown')
        if provider not in providers:
            providers[provider] = {'models': {}}
        providers[provider]['models'][model_id] = model

    return providers
