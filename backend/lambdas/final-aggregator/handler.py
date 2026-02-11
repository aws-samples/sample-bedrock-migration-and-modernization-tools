"""
Final Aggregator Lambda

Merges all collected data into the final comprehensive JSON outputs.
Works with the correct snake_case schema from upstream Lambdas.
"""

import logging
import os
import re
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


def aggregate_quotas(quota_results: list[dict], s3_client: Any, bucket: str) -> dict:
    """Aggregate quotas from all regions."""
    quotas_by_region = {}

    for item in quota_results:
        nested_result = item.get('result', {})
        status = item.get('status') or nested_result.get('status')
        s3_key = item.get('s3Key') or nested_result.get('s3Key')
        region = item.get('region')

        if status == 'SUCCESS' and s3_key:
            try:
                data = read_from_s3(s3_client, bucket, s3_key, default_on_missing={})
                quotas_by_region[region] = data.get('quotas', [])
            except S3ReadError as e:
                logger.warning(f"Failed to read quotas for {region}: {e}")
                quotas_by_region[region] = []

    return quotas_by_region


def aggregate_features(feature_results: list[dict], s3_client: Any, bucket: str) -> dict:
    """Aggregate inference profiles from all regions."""
    profiles_by_region = {}

    for item in feature_results:
        nested_result = item.get('result', {})
        status = item.get('status') or nested_result.get('status')
        s3_key = item.get('s3Key') or nested_result.get('s3Key')
        region = item.get('region')

        if status == 'SUCCESS' and s3_key:
            try:
                data = read_from_s3(s3_client, bucket, s3_key, default_on_missing={})
                # Handle both snake_case and camelCase from feature extractor
                profiles_by_region[region] = data.get('inference_profiles', data.get('inferenceProfiles', []))
            except S3ReadError as e:
                logger.warning(f"Failed to read features for {region}: {e}")
                profiles_by_region[region] = []

    return profiles_by_region


def get_size_category(context_window: int) -> dict:
    """Get size category based on context window."""
    if context_window is None:
        return {"category": "Unknown", "color": "#6B7280", "tier": 0}
    elif context_window >= 128000:
        return {"category": "Large", "color": "#10B981", "tier": 3}
    elif context_window >= 32000:
        return {"category": "Medium", "color": "#3B82F6", "tier": 2}
    else:
        return {"category": "Small", "color": "#F59E0B", "tier": 1}


def get_context_window_from_config(model_id: str) -> dict:
    """
    Get context window specs from config for a model.

    Uses pattern matching to find the best match in context_window_specs.
    Returns dict with context window data or empty dict if not found.
    """
    config = _get_config()
    context_specs = config.config.get('model_configuration', {}).get('context_window_specs', {})

    # Remove version suffix for matching (e.g., anthropic.claude-opus-4-6-v1:0 -> anthropic.claude-opus-4-6)
    model_id_clean = model_id.lower()
    # Remove common suffixes
    for suffix in ['-v1:0', '-v1', '-v2:0', '-v2', ':0', ':1']:
        if model_id_clean.endswith(suffix):
            model_id_clean = model_id_clean[:-len(suffix)]

    # Try exact match first
    if model_id_clean in context_specs:
        return context_specs[model_id_clean]

    # Try prefix matching (longest match wins)
    best_match = None
    best_match_len = 0
    for pattern, specs in context_specs.items():
        if pattern.startswith('_'):  # Skip comment keys
            continue
        if model_id_clean.startswith(pattern) and len(pattern) > best_match_len:
            best_match = specs
            best_match_len = len(pattern)

    return best_match or {}


def build_cross_region_inference(model_id: str, features_by_region: dict) -> dict:
    """Build cross-region inference data for a model.

    Deduplicates profiles by (profile_id, source_region) to avoid duplicates
    when a profile contains multiple model variants.
    """
    profiles = []
    source_regions = set()
    seen_profile_regions = set()  # Track (profile_id, region) pairs to avoid duplicates

    for region, region_profiles in features_by_region.items():
        for profile in region_profiles:
            profile_id = profile.get('inference_profile_id', profile.get('inferenceProfileId'))

            # Skip if we've already added this profile for this region
            profile_region_key = (profile_id, region)
            if profile_region_key in seen_profile_regions:
                continue

            # Check if any model in this profile matches
            profile_models = profile.get('models', [])
            matches = False
            for pm in profile_models:
                # Handle both snake_case and camelCase model ARN
                model_arn = pm.get('model_arn', pm.get('modelArn', ''))
                if model_id in model_arn:
                    matches = True
                    break  # Found a match, no need to check other models

            if matches:
                profiles.append({
                    'profile_id': profile_id,
                    'profile_name': profile.get('inference_profile_name', profile.get('inferenceProfileName')),
                    'source_region': region,
                    'type': profile.get('type'),
                    'status': profile.get('status', 'ACTIVE'),
                    'description': profile.get('description', '')
                })
                source_regions.add(region)
                seen_profile_regions.add(profile_region_key)

    return {
        'supported': len(profiles) > 0,
        'profiles_count': len(profiles),
        'source_regions': sorted(list(source_regions)),
        'profiles': profiles
    }


def _normalize_for_quota_matching(name: str) -> str:
    """
    Normalize a model name or quota model reference for exact matching.
    - Lowercase
    - Replace hyphens/underscores with spaces
    - Strip default version tags (v1, v1.0) — keeps v2, v2.1, etc.
    - Strip 8-digit date codes (e.g. 20240307)
    - Join standalone single-digit pairs: "3 5" -> "3.5"
    - Strip trailing context length qualifiers (200K, 1M Context Length)
    - Collapse whitespace
    """
    n = name.lower().strip()
    # Strip trailing punctuation (AWS quota typos like "Claude Sonnet 4.5.")
    n = re.sub(r'[.;,!]+$', '', n)
    n = n.replace('-', ' ').replace('_', ' ')
    # Normalize "+" to " plus" (e.g. "Command R+" → "Command R plus")
    n = n.replace('+', ' plus')
    # Strip v1/V1/V1.0 (default version, not a distinguishing identifier)
    n = re.sub(r'\bv1(?:\.0)?\b', '', n, flags=re.I)
    # Strip 8-digit date patterns (e.g. 20240307, 20250929)
    n = re.sub(r'\b\d{8}\b', '', n)
    # Join standalone single-digit pairs not adjacent to letters/digits/dots
    # e.g. "3 5" -> "3.5" but NOT "v1 1m" or "3.1 70b"
    n = re.sub(r'(?<![a-zA-Z\d.])(\d)\s+(\d)(?![a-zA-Z\d])', r'\1.\2', n)
    # Strip trailing context length qualifiers (200K, 1M, 256K, 1M Context Length)
    n = re.sub(r'\s+\d+[kKmM](?:\s+context\s+length)?\s*$', '', n, flags=re.I)
    # Collapse whitespace
    n = ' '.join(n.split())
    return n


# Known provider prefixes that appear in quota names but not in model names.
# Ordered longest-first to avoid partial matches.
_PROVIDER_PREFIXES = [
    'anthropic ', 'ai21 labs ', 'stability.ai ', 'stability ai ',
    'mistral ai ', 'moonshot ai ', 'writer ai ', 'luma ai ',
    'twelvelabs ', 'deepseek ', 'minimax ', 'openai ', 'nvidia ',
    'amazon ', 'google ', 'cohere ', 'meta ', 'luma ', 'qwen ',
    'mistral ',  # After 'mistral ai ' — catches "Mistral Mixtral..." in quotas
]


def _strip_provider_prefix(name: str) -> str:
    """Strip a known provider prefix from a normalized (lowercase) name."""
    for prefix in _PROVIDER_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _extract_quota_model_ref(quota_name: str) -> str:
    """
    Extract the model reference string from a quota name.

    Quota names follow patterns like:
      "On-demand model inference requests per minute for Anthropic Claude 3.5 Sonnet"
      "Batch inference job size (in GB) for Claude Sonnet 4.5"
      "(Model customization) ... for a Claude 3 Haiku v1 Fine-tuning job"
      "Model units per provisioned model for the 128k context length variant for Amazon Nova Micro"
      "No-commitment model units for Provisioned Throughput created for base model Amazon Nova 2 Lite V1.0 256K"

    Returns the model name portion, or None if not found.
    """
    name = quota_name.strip()

    # Remove leading category prefix like "(Model customization)"
    name = re.sub(r'^\([^)]+\)\s*', '', name)

    # Remove trailing "(doubled for cross-region calls)" qualifier
    name = re.sub(r'\s*\(doubled\s+for[^)]*\)\s*$', '', name, flags=re.I)

    # Split by "for" and take the LAST segment (the model reference)
    parts = re.split(r'\bfor\s+', name, flags=re.I)
    if len(parts) < 2:
        return None

    ref = parts[-1].strip()

    # Clean up extracted ref:
    # Remove leading articles "a "/"an "
    ref = re.sub(r'^(?:a|an)\s+', '', ref, flags=re.I)
    # Remove "base model"/"custom model" prefix
    ref = re.sub(r'^(?:base|custom)\s+model\s+', '', ref, flags=re.I)
    # Remove trailing job type suffixes (Fine-tuning, Continued Pre-Training, distillation)
    ref = re.sub(r'\s+(?:Fine[- ]?tuning|Continued Pre[- ]?Training|distillation)\b.*$', '', ref, flags=re.I)
    # Remove trailing "per month"
    ref = re.sub(r'\s+per\s+month$', '', ref, flags=re.I)

    return ref.strip() if ref.strip() else None


def _build_model_aliases(model_id: str, model_name: str, model_provider: str) -> set:
    """
    Build a set of normalized aliases for a model that quotas might reference.

    Generates aliases from:
    1. model_name (primary)
    2. provider + model_name (for quotas that include provider prefix)
    3. model_name without parenthetical (for Mistral date versions like "(24.07)")
    4. model_id-derived name (catches naming variants like "2407" vs "(24.07)")
    """
    aliases = set()
    if not model_name:
        return aliases

    # Normalize provider name
    prov = (model_provider or '').lower().strip()

    # Alias 1: from model_name
    norm_name = _normalize_for_quota_matching(model_name)
    aliases.add(norm_name)
    # Also without provider prefix (in case model_name includes it, e.g. "DeepSeek-R1")
    aliases.add(_strip_provider_prefix(norm_name))

    # Alias 2: provider + model_name (for quotas like "Anthropic Claude 3.5 Sonnet")
    if prov and not norm_name.startswith(prov):
        aliases.add(_normalize_for_quota_matching(prov + ' ' + model_name))

    # Alias 3: model_name without parenthetical (e.g. "Mistral Large (24.07)" -> "Mistral Large")
    if '(' in model_name:
        name_no_parens = re.sub(r'\s*\([^)]*\)', '', model_name).strip()
        if name_no_parens:
            np = _normalize_for_quota_matching(name_no_parens)
            aliases.add(np)
            aliases.add(_strip_provider_prefix(np))
            if prov and not np.startswith(prov):
                aliases.add(_normalize_for_quota_matching(prov + ' ' + name_no_parens))
        # Also: parens removed but content kept (e.g. "Pixtral Large (25.02)" -> "Pixtral Large 25.02")
        name_flat_parens = model_name.replace('(', '').replace(')', '').strip()
        fp = _normalize_for_quota_matching(name_flat_parens)
        aliases.add(fp)
        aliases.add(_strip_provider_prefix(fp))
        if prov and not fp.startswith(prov):
            aliases.add(_normalize_for_quota_matching(prov + ' ' + name_flat_parens))

    # Alias 5: short name without trailing size+task suffix
    # e.g. "Llama 4 Maverick 17B Instruct" -> "Llama 4 Maverick"
    short_name = re.sub(r'\s+\d+[Bb]\s+(?:Instruct|Chat|IT|PT)\s*$', '', model_name)
    if short_name != model_name:
        sn = _normalize_for_quota_matching(short_name)
        aliases.add(sn)
        aliases.add(_strip_provider_prefix(sn))
        if prov and not sn.startswith(prov):
            aliases.add(_normalize_for_quota_matching(prov + ' ' + short_name))

    # Alias 6: without trailing version number (e.g. "Stable Image Core 1.0" -> "Stable Image Core")
    short_ver = re.sub(r'\s+\d+\.\d+\s*$', '', model_name)
    if short_ver != model_name:
        sv = _normalize_for_quota_matching(short_ver)
        aliases.add(sv)
        aliases.add(_strip_provider_prefix(sv))
        if prov and not sv.startswith(prov):
            aliases.add(_normalize_for_quota_matching(prov + ' ' + short_ver))

    # Alias 4: from model_id (catches naming variants)
    if model_id:
        clean_id = model_id.split(':')[0]  # Remove :0, :18k etc.
        # Extract model part after provider prefix (e.g. "anthropic.claude-sonnet-4-5-20250929-v1")
        id_parts = clean_id.split('.', 1)
        model_part = id_parts[1] if len(id_parts) > 1 else clean_id
        # Remove trailing date+v1 or just v1 (default version only; keep v2+ as they distinguish models)
        model_part = re.sub(r'(-\d{8})?-v1$', '', model_part)
        # Remove trailing standalone 8-digit date (for models without version suffix)
        model_part = re.sub(r'-\d{8}$', '', model_part)
        if model_part:
            id_alias = _normalize_for_quota_matching(model_part)
            aliases.add(id_alias)

    # Remove any empty strings
    aliases.discard('')
    return aliases


# Cached quota index: maps normalized model ref -> {region -> [quotas]}
_quota_index = None


def _build_quota_index(quotas_by_region: dict) -> dict:
    """
    Pre-index all quotas by their normalized model reference.
    This enables O(1) lookup per model instead of O(quotas) scanning.
    """
    index = {}
    for region, quotas in quotas_by_region.items():
        for quota in quotas:
            quota_name = quota.get('quotaName', quota.get('quota_name', ''))
            ref = _extract_quota_model_ref(quota_name)
            if not ref:
                continue
            # Normalize and index both with and without provider prefix
            norm = _normalize_for_quota_matching(ref)
            norm_no_prov = _strip_provider_prefix(norm)
            for key in {norm, norm_no_prov}:
                if key:
                    index.setdefault(key, {}).setdefault(region, []).append(quota)
    return index


def build_model_quotas(model_id: str, model_name: str, quotas_by_region: dict,
                       model_provider: str = '') -> dict:
    """
    Build model-specific quotas by region using exact name matching.

    Uses a pre-built index of quota model references for efficient lookup.
    Matches quota names against model aliases derived from model_name,
    model_provider, and model_id — no hardcoded model lists or keyword matching.
    """
    global _quota_index
    if _quota_index is None:
        _quota_index = _build_quota_index(quotas_by_region)

    aliases = _build_model_aliases(model_id, model_name, model_provider)
    model_quotas = {}
    seen_codes_per_region = {}  # Dedup: same quota found via multiple aliases

    for alias in aliases:
        matched = _quota_index.get(alias, {})
        for region, quotas in matched.items():
            if region not in seen_codes_per_region:
                seen_codes_per_region[region] = set()
            for quota in quotas:
                code = quota.get('quotaCode', quota.get('quota_code', ''))
                if code in seen_codes_per_region[region]:
                    continue
                seen_codes_per_region[region].add(code)
                model_quotas.setdefault(region, []).append({
                    'quota_code': code,
                    'quota_name': quota.get('quotaName', quota.get('quota_name', '')),
                    'quota_arn': quota.get('quotaArn', quota.get('quota_arn', '')),
                    'description': quota.get('description', ''),
                    'quota_applied_at_level': quota.get('quotaAppliedAtLevel', quota.get('quota_applied_at_level', 'ACCOUNT')),
                    'value': quota.get('value', 0),
                    'unit': quota.get('unit', 'None'),
                    'adjustable': quota.get('adjustable', False),
                    'global_quota': quota.get('globalQuota', quota.get('global_quota', False)),
                    'usage_metric': quota.get('usageMetric', quota.get('usage_metric', {})),
                    'period': quota.get('period', {})
                })

    return model_quotas


def get_consumption_options(inference_types: list, pricing_data: dict = None, pricing_ref: dict = None) -> list:
    """Determine consumption options from inference types and pricing data.

    Always includes 'on_demand' if there's On-Demand pricing available.
    Adds 'batch' if there's Batch pricing available.
    """
    options = set()

    # Map inference types to consumption options
    type_mapping = {
        'ON_DEMAND': 'on_demand',
        'PROVISIONED': 'provisioned_throughput',
        'INFERENCE_PROFILE': 'cross_region_inference'
    }
    for inf_type in inference_types:
        if inf_type in type_mapping:
            options.add(type_mapping[inf_type])

    # Check pricing data for additional consumption options
    if pricing_data and pricing_ref:
        provider = pricing_ref.get('provider', '')
        model_key = pricing_ref.get('model_key', '')

        if provider and model_key:
            providers = pricing_data.get('providers', {})
            prov_data = providers.get(provider, {})
            model_pricing = prov_data.get(model_key, {})

            if isinstance(model_pricing, dict) and 'regions' in model_pricing:
                # Check first available region for pricing groups
                for region_data in model_pricing.get('regions', {}).values():
                    pricing_groups = region_data.get('pricing_groups', {})

                    # Check for On-Demand pricing
                    if any(g.startswith('On-Demand') for g in pricing_groups.keys()):
                        options.add('on_demand')

                    # Check for Batch pricing
                    if any(g.startswith('Batch') for g in pricing_groups.keys()):
                        options.add('batch')

                    # Check for Provisioned Throughput pricing
                    if 'Provisioned Throughput' in pricing_groups:
                        options.add('provisioned_throughput')

                    break  # Only need to check one region

    # Always include on_demand as a default if no other options found
    if not options:
        options.add('on_demand')

    # Sort for consistent ordering: on_demand, batch, cross_region_inference, provisioned_throughput
    order = ['on_demand', 'batch', 'cross_region_inference', 'provisioned_throughput']
    return sorted(list(options), key=lambda x: order.index(x) if x in order else len(order))


def check_batch_inference(model_id: str, pricing_data: dict, pricing_ref: dict = None, regional_availability: list = None) -> dict:
    """Check if batch inference is supported based on pricing data.

    Uses pricing_file_reference.model_key for accurate matching when available.
    Calculates coverage_percentage based on regional_availability.
    """
    supported_regions = []

    # Use pricing reference model_key if available, otherwise fall back to model_id
    lookup_keys = []
    if pricing_ref:
        provider = pricing_ref.get('provider', '')
        model_key = pricing_ref.get('model_key', '')
        if provider and model_key:
            lookup_keys.append((provider, model_key))

    # Also try with the original model_id
    lookup_keys.append((None, model_id))

    providers = pricing_data.get('providers', {})

    for provider_hint, lookup_key in lookup_keys:
        if supported_regions:
            break  # Already found, no need to continue

        for prov_name, prov_data in providers.items():
            if supported_regions:
                break

            # Skip if provider hint doesn't match
            if provider_hint and prov_name.lower() != provider_hint.lower():
                continue

            # Check for provider -> model structure (new schema)
            if isinstance(prov_data, dict):
                # Direct model lookup
                if lookup_key in prov_data:
                    model_data = prov_data[lookup_key]
                    if isinstance(model_data, dict) and 'regions' in model_data:
                        for region, region_data in model_data.get('regions', {}).items():
                            pricing_groups = region_data.get('pricing_groups', {})
                            # Check for any Batch pricing group
                            if any(g.startswith('Batch') for g in pricing_groups.keys()):
                                if region not in supported_regions:
                                    supported_regions.append(region)

                # Fuzzy matching as fallback
                if not supported_regions:
                    lookup_lower = lookup_key.lower()
                    for mid, model_data in prov_data.items():
                        if isinstance(model_data, dict) and 'regions' in model_data:
                            # Check if lookup_key is contained in mid or vice versa
                            mid_lower = mid.lower()
                            if lookup_lower in mid_lower or mid_lower in lookup_lower:
                                for region, region_data in model_data.get('regions', {}).items():
                                    pricing_groups = region_data.get('pricing_groups', {})
                                    if any(g.startswith('Batch') for g in pricing_groups.keys()):
                                        if region not in supported_regions:
                                            supported_regions.append(region)

    # Calculate coverage percentage based on model's regional availability
    total_regions = len(regional_availability) if regional_availability else 0
    batch_region_count = len(supported_regions)
    coverage = (batch_region_count / total_regions * 100) if total_regions > 0 else 0.0
    # Cap at 100% - values > 100% indicate batch pricing in more regions than model availability
    coverage = min(coverage, 100.0)

    return {
        'supported': len(supported_regions) > 0,
        'supported_regions': sorted(supported_regions),
        'coverage_percentage': round(coverage, 1),
        'detection_method': 'pricing_data' if supported_regions else 'no_pricing_data'
    }


def transform_model_to_schema(
    model_id: str,
    model: dict,
    regional_availability: list,
    token_specs: dict,
    quotas_by_region: dict,
    features_by_region: dict,
    enriched_model: dict,
    pricing_data: dict,
    collection_timestamp: str
) -> dict:
    """
    Merge model data from all sources into final schema.

    Input model data is already in snake_case from upstream Lambdas.
    """
    # Get enriched data (already in snake_case)
    capabilities = enriched_model.get('model_capabilities', model.get('model_capabilities', []))
    use_cases = enriched_model.get('model_use_cases', model.get('model_use_cases', []))
    doc_links = enriched_model.get('documentation_links', model.get('documentation_links', {}))

    # Build token/converse data (upstream uses snake_case)
    # 4-tier priority: 1) Console API metadata, 2) Model ID variants,
    # 3) Config (extended fields always, context as fallback), 4) LiteLLM
    existing_converse = enriched_model.get('converse_data', model.get('converse_data', {}))

    context_window = None
    max_output = None
    source = None
    extended_context = None
    extended_context_beta = None
    extended_output = None
    extended_output_beta = None

    # --- TIER 1: Console API metadata (from model-extractor REST call) ---
    console_meta = model.get('console_metadata', {})
    console_languages = console_meta.get('languages', []) if console_meta else []
    console_use_cases = console_meta.get('use_cases', []) if console_meta else []
    console_description = console_meta.get('description', '') if console_meta else ''
    console_short_description = console_meta.get('short_description', '') if console_meta else ''
    if console_meta:
        api_context = console_meta.get('max_context_window')
        if api_context and isinstance(api_context, (int, float)):
            context_window = int(api_context)
            source = 'bedrock_console_api'
        api_output = console_meta.get('max_output_tokens')
        if api_output and isinstance(api_output, (int, float)):
            max_output = int(api_output)

    # --- TIER 2: Model ID size variant (from model-merger) ---
    if context_window is None:
        variant_cw = model.get('variant_context_window')
        if variant_cw and isinstance(variant_cw, (int, float)):
            context_window = int(variant_cw)
            source = 'model_id_variant'

    # --- TIER 3: profiler-config.json ---
    config_specs = get_context_window_from_config(model_id)
    if config_specs:
        config_standard = config_specs.get('standard_context')
        config_extended = config_specs.get('extended_context')

        # If API returned the extended value as context_window, prefer config's standard
        # (e.g., API says 1M for Opus 4.6, but standard is 200K with 1M extended)
        if config_standard and config_extended and context_window == config_extended:
            context_window = config_standard
            source = config_specs.get('source', 'config')

        # Use standard_context only if Tiers 1-2 didn't provide context_window
        if context_window is None:
            context_window = config_standard
            source = config_specs.get('source', 'config')

        # Use max_output from config if not yet set
        if max_output is None:
            max_output = config_specs.get('max_output')

        # Extended fields: ALWAYS apply from config regardless of tier
        # These fields only exist in config (Claude dual context, extended output)
        extended_context = config_extended
        extended_context_beta = config_specs.get('extended_context_beta')
        extended_output = config_specs.get('extended_output')
        extended_output_beta = config_specs.get('extended_output_beta')

    # --- TIER 4: LiteLLM token_specs (last resort) ---
    if context_window is None:
        context_window = token_specs.get('context_window')
    if max_output is None:
        max_output = token_specs.get('max_output_tokens')
    if source is None and token_specs.get('source'):
        source = token_specs.get('source')

    # --- Fallback: existing converse_data ---
    if context_window is None:
        context_window = existing_converse.get('context_window')
    if max_output is None:
        max_output = existing_converse.get('max_output_tokens')
    if source is None:
        source = existing_converse.get('source')

    converse_data = {
        'context_window': context_window,
        'max_output_tokens': max_output,
        'size_category': get_size_category(context_window),
        'verified': source is not None and source != 'unknown',
        'source': source or 'unknown',
        'litellm_verified': token_specs.get('litellm_verified', existing_converse.get('litellm_verified', False)),
        'capabilities_count': len(capabilities),
        'use_cases_count': len(use_cases),
        'regions_count': len(regional_availability)
    }

    # Add extended context info if available
    if extended_context:
        converse_data['extended_context'] = extended_context
        converse_data['has_extended_context'] = True
        if extended_context_beta:
            converse_data['extended_context_beta'] = extended_context_beta
    else:
        converse_data['has_extended_context'] = False

    # Add extended output info if available
    if extended_output:
        converse_data['extended_output'] = extended_output
        if extended_output_beta:
            converse_data['extended_output_beta'] = extended_output_beta

    # Build cross-region inference
    cross_region = build_cross_region_inference(model_id, features_by_region)

    # Build model quotas (using snake_case model_name)
    model_quotas = build_model_quotas(
        model_id,
        model.get('model_name', ''),
        quotas_by_region,
        model_provider=model.get('model_provider', '')
    )

    # Get model pricing from upstream (already in snake_case)
    # Preserve pricing_file_reference from pricing-linker which has correct provider mapping
    model_pricing_data = model.get('model_pricing', {})
    has_pricing = model_pricing_data.get('is_pricing_available', model.get('has_pricing', False))
    pricing_ref_id = model_pricing_data.get('pricing_reference_id', '')

    # Use upstream pricing_file_reference if available (from pricing-linker)
    # This preserves the correct provider name from the pricing file
    upstream_pricing_ref = model_pricing_data.get('pricing_file_reference')

    # Check batch inference support - pass pricing reference and regional availability for accurate lookup
    # Use fallback to model.regions_available if regional_availability is empty (same logic as total_regions_available)
    regions_for_coverage = regional_availability if regional_availability else model.get('regions_available', [])
    batch_inference = check_batch_inference(model_id, pricing_data, upstream_pricing_ref, regions_for_coverage)

    # Union regional_availability with batch supported_regions
    # (CRIS source_regions are kept separate — they represent cross-region inference, not direct availability)
    regions_set = set(regional_availability)
    if batch_inference.get('supported_regions'):
        regions_set.update(batch_inference['supported_regions'])
    if len(regions_set) > len(regional_availability):
        regional_availability = sorted(list(regions_set))
        # Recalculate batch coverage with expanded regional_availability
        if batch_inference.get('supported'):
            total_regs = len(regional_availability)
            batch_regs = len(batch_inference['supported_regions'])
            batch_inference['coverage_percentage'] = round(
                min(batch_regs / total_regs * 100, 100.0), 1
            ) if total_regs > 0 else 0.0

    if upstream_pricing_ref and isinstance(upstream_pricing_ref, dict):
        pricing_provider = upstream_pricing_ref.get('provider', model.get('model_provider', ''))
        pricing_model_key = upstream_pricing_ref.get('model_key', pricing_ref_id or model_id)
    else:
        # Fallback: use model's provider (may not match pricing file)
        pricing_provider = model.get('model_provider', '')
        pricing_model_key = pricing_ref_id if pricing_ref_id else model_id

    model_pricing = {
        'is_pricing_available': has_pricing,
        'pricing_reference_id': pricing_ref_id or model_id,
        'pricing_file_reference': {
            'provider': pricing_provider,
            'model_key': pricing_model_key,
            'model_name': model.get('model_name', '')
        },
        'pricing_summary': {
            'integration_source': 'amazon-bedrock-pricing-collector',
            'has_pricing_data': has_pricing,
            'integration_timestamp': collection_timestamp,
            'reference_based': True
        }
    }

    # Build documentation links (pass through all from enricher, with defaults)
    documentation_links = doc_links.copy() if doc_links else {}
    # Ensure minimum required links
    if 'aws_bedrock_guide' not in documentation_links:
        documentation_links['aws_bedrock_guide'] = 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html'
    if 'pricing_guide' not in documentation_links:
        documentation_links['pricing_guide'] = 'https://aws.amazon.com/bedrock/pricing/'

    # Get modalities (already in snake_case nested structure)
    model_modalities = model.get('model_modalities', {})
    if not model_modalities:
        # Fallback for legacy data
        model_modalities = {
            'input_modalities': model.get('input_modalities', []),
            'output_modalities': model.get('output_modalities', [])
        }

    # Get collection metadata (already in snake_case)
    existing_metadata = model.get('collection_metadata', {})
    collection_metadata = {
        'first_discovered_at': existing_metadata.get('first_discovered_at', collection_timestamp),
        'first_discovered_in_region': existing_metadata.get('first_discovered_in_region', regional_availability[0] if regional_availability else 'unknown'),
        'api_source': existing_metadata.get('api_source', 'list_foundation_models'),
        'dual_region_collection': existing_metadata.get('dual_region_collection', True),
        'regions_collected_from': existing_metadata.get('regions_collected_from', []),
        'phase2_regional_discovery': True,
        'regional_data_source': 'api_discovery'
    }

    # Get model lifecycle (already in snake_case)
    model_lifecycle = model.get('model_lifecycle', {})
    if not model_lifecycle:
        model_lifecycle = {
            'status': 'ACTIVE',
            'release_date': ''
        }

    # Get customization (already in snake_case)
    customization = model.get('customization', {})
    if not customization:
        customization = {
            'customization_supported': model.get('customization_supported', []),
            'customization_options': {}
        }

    return {
        'model_id': model_id,
        'model_arn': model.get('model_arn', ''),
        'model_name': model.get('model_name', ''),
        'model_provider': model.get('model_provider', ''),
        'model_modalities': model_modalities,
        'streaming_supported': model.get('streaming_supported', False),
        'customization': customization,
        'inference_types_supported': model.get('inference_types_supported', []),
        'model_lifecycle': model_lifecycle,
        'regions_available': regional_availability if regional_availability else model.get('regions_available', []),
        'model_capabilities': capabilities,
        'model_use_cases': console_use_cases,
        'languages_supported': console_languages,
        'description': console_description,
        'short_description': console_short_description,
        'consumption_options': get_consumption_options(model.get('inference_types_supported', []), pricing_data, upstream_pricing_ref),
        'cross_region_inference': cross_region,
        'documentation_links': documentation_links,
        'model_pricing': model_pricing,
        'model_service_quotas': model_quotas,
        'collection_metadata': collection_metadata,
        'regional_availability_source': 'api_discovery',
        'total_regions_available': len(regional_availability) if regional_availability else len(model.get('regions_available', [])),
        'batch_inference_supported': batch_inference,
        'converse_data': converse_data,
        'has_pricing': has_pricing,
        'has_quotas': len(model_quotas) > 0
    }


def find_matching_availability(model_id: str, model_availability: dict) -> list:
    """
    Find regional availability for a model, handling ID format differences.

    Model IDs from Bedrock API: anthropic.claude-3-5-sonnet-20241022-v2:0
    Model IDs from Pricing API: anthropic.claude-3-sonnet

    Strategy: Try exact match first, then find the best (longest) match.
    """
    # Try exact match first
    if model_id in model_availability:
        return model_availability[model_id]

    # Normalize model_id for matching (remove version suffix like :0, :18k, etc.)
    base_model_id = model_id.split(':')[0] if ':' in model_id else model_id

    # Try matching without version suffix
    if base_model_id in model_availability:
        return model_availability[base_model_id]

    # Find the best (longest) matching pricing key
    # This prevents "claude-3-sonnet" from incorrectly matching "claude-3-5-sonnet-xxx"
    model_id_lower = model_id.lower()
    best_match_key = None
    best_match_length = 0

    for pricing_key in model_availability.keys():
        pricing_key_lower = pricing_key.lower()

        # Check if pricing key is contained in model_id or model_id starts with pricing key
        if pricing_key_lower in model_id_lower or model_id_lower.startswith(pricing_key_lower):
            if len(pricing_key) > best_match_length:
                best_match_key = pricing_key
                best_match_length = len(pricing_key)
            continue

        # Also check by removing common prefixes/suffixes and comparing core name
        pricing_parts = pricing_key_lower.replace('anthropic.', '').replace('amazon.', '').replace('meta.', '').replace('mistral.', '').replace('cohere.', '').replace('ai21.', '').replace('stability.', '').replace('nvidia.', '').replace('luma.', '')
        model_parts = model_id_lower.replace('anthropic.', '').replace('amazon.', '').replace('meta.', '').replace('mistral.', '').replace('cohere.', '').replace('ai21.', '').replace('stability.', '').replace('nvidia.', '').replace('luma.', '')

        # Check if core names overlap significantly
        if pricing_parts and model_parts:
            # Remove date/version suffixes from model_parts for comparison
            model_core = re.sub(r'-\d{8}-v\d+.*$', '', model_parts)
            if pricing_parts == model_core or pricing_parts in model_core or model_core.startswith(pricing_parts):
                if len(pricing_key) > best_match_length:
                    best_match_key = pricing_key
                    best_match_length = len(pricing_key)

    if best_match_key:
        return model_availability[best_match_key]

    return []


def build_final_models(
    models_with_pricing: dict,
    regional_availability: dict,
    token_specs: dict,
    quotas_by_region: dict,
    features_by_region: dict,
    enriched_models: dict,
    pricing_data: dict,
    collection_timestamp: str
) -> dict:
    """Build the final comprehensive models structure in expected schema."""
    providers = models_with_pricing.get('providers', {})
    enriched_providers = enriched_models.get('providers', {})
    # Upstream uses snake_case: model_availability
    model_availability = regional_availability.get('model_availability', {})
    # Upstream uses snake_case: token_specs
    token_specs_data = token_specs.get('token_specs', {})

    result_providers = {}

    for provider, provider_data in providers.items():
        result_providers[provider] = {'models': {}}

        for model_id, model in provider_data.get('models', {}).items():
            # Get regional availability for this model (with fuzzy matching)
            regions = find_matching_availability(model_id, model_availability)

            # Get token specs for this model
            specs = token_specs_data.get(model_id, {})

            # Get enriched data for this model
            enriched = enriched_providers.get(provider, {}).get('models', {}).get(model_id, {})

            # Transform to expected schema
            transformed = transform_model_to_schema(
                model_id=model_id,
                model=model,
                regional_availability=regions,
                token_specs=specs,
                quotas_by_region=quotas_by_region,
                features_by_region=features_by_region,
                enriched_model=enriched,
                pricing_data=pricing_data,
                collection_timestamp=collection_timestamp
            )

            result_providers[provider]['models'][model_id] = transformed

    return result_providers


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for final aggregation.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "pricingS3Key": "...",
            "modelsS3Key": "...",
            "quotaResults": [...],
            "pricingLinked": {...},
            "regionalAvailability": {...},
            "featureResults": [...],
            "tokenSpecs": {...},
            "enrichedModels": {...}
        }

    Output:
        {
            "status": "SUCCESS",
            "modelsS3Key": "executions/{id}/final/bedrock_models.json",
            "pricingS3Key": "executions/{id}/final/bedrock_pricing.json",
            "totalModels": 108,
            "totalProviders": 17
        }
    """
    start_time = time.time()
    collection_timestamp = time.strftime('%Y-%m-%dT%H:%M:%S.000000+00:00', time.gmtime())

    # Reset quota index cache for each invocation (Lambda containers may be reused)
    global _quota_index
    _quota_index = None

    # Validate required parameters
    try:
        validate_required_params(event, ['s3Bucket', 'executionId'], 'FinalAggregator')
    except ValidationError as e:
        return {
            'status': 'FAILED',
            'errorType': 'ValidationError',
            'errorMessage': str(e)
        }

    s3_bucket = event['s3Bucket']
    execution_id = parse_execution_id(event['executionId'])
    pricing_s3_key = event.get('pricingS3Key')
    quota_results = event.get('quotaResults', [])
    pricing_linked = event.get('pricingLinked', {})
    regional_availability = event.get('regionalAvailability', {})
    feature_results = event.get('featureResults', [])
    token_specs_result = event.get('tokenSpecs', {})
    enriched_models_result = event.get('enrichedModels', {})
    dry_run = event.get('dryRun', False)

    models_output_key = f"executions/{execution_id}/final/bedrock_models.json"
    pricing_output_key = f"executions/{execution_id}/final/bedrock_pricing.json"

    logger.info("Building final aggregated output")

    try:
        s3_client = get_s3_client()

        if not dry_run:
            # Read intermediate data
            models_with_pricing_key = pricing_linked.get('s3Key')
            models_with_pricing = read_from_s3(s3_client, s3_bucket, models_with_pricing_key) if models_with_pricing_key else {}

            availability_key = regional_availability.get('s3Key')
            availability_data = read_from_s3(s3_client, s3_bucket, availability_key) if availability_key else {}

            token_specs_key = token_specs_result.get('s3Key')
            token_specs_data = read_from_s3(s3_client, s3_bucket, token_specs_key) if token_specs_key else {}

            pricing_data = read_from_s3(s3_client, s3_bucket, pricing_s3_key) if pricing_s3_key else {}

            enriched_models_key = enriched_models_result.get('s3Key')
            enriched_models_data = read_from_s3(s3_client, s3_bucket, enriched_models_key) if enriched_models_key else {}

            # Aggregate quotas and features
            quotas_by_region = aggregate_quotas(quota_results, s3_client, s3_bucket)
            features_by_region = aggregate_features(feature_results, s3_client, s3_bucket)

            # Build final models in expected schema
            final_providers = build_final_models(
                models_with_pricing,
                availability_data,
                token_specs_data,
                quotas_by_region,
                features_by_region,
                enriched_models_data,
                pricing_data,
                collection_timestamp
            )

            # Calculate statistics
            total_models = sum(len(p.get('models', {})) for p in final_providers.values())
            total_providers = len(final_providers)
            total_regions = len(availability_data.get('regions', {}))

            # Count models with pricing and quotas
            models_with_pricing_count = sum(
                1 for p in final_providers.values()
                for m in p.get('models', {}).values()
                if m.get('has_pricing', False)
            )
            models_with_quotas_count = sum(
                1 for p in final_providers.values()
                for m in p.get('models', {}).values()
                if m.get('has_quotas', False)
            )
            total_quotas = sum(
                len(quotas) for region_quotas in quotas_by_region.values()
                for quotas in ([region_quotas] if isinstance(region_quotas, list) else region_quotas.values())
            )

            # Build final models output in expected schema
            models_output = {
                'metadata': {
                    'collection_timestamp': collection_timestamp,
                    'providers_count': total_providers,
                    'total_models': total_models,
                    'models_with_pricing': models_with_pricing_count,
                    'models_with_quotas': models_with_quotas_count,
                    'regions_covered': total_regions,
                    'total_quotas_available': total_quotas,
                    'collection_method': 'comprehensive_structure_with_quota_assignment'
                },
                'providers': final_providers
            }

            # Write models output
            write_to_s3(s3_client, s3_bucket, models_output_key, models_output)

            # Copy pricing data as-is (pricing-aggregator already formats it)
            # The pricing schema transformation happens in pricing-aggregator
            write_to_s3(s3_client, s3_bucket, pricing_output_key, pricing_data)

        else:
            logger.info("Dry run - skipping final aggregation")
            total_models = 0
            total_providers = 0
            total_regions = 0

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            'status': 'SUCCESS',
            'modelsS3Key': models_output_key,
            'pricingS3Key': pricing_output_key,
            'totalModels': total_models,
            'totalProviders': total_providers,
            'totalRegions': total_regions,
            'durationMs': duration_ms
        }

    except Exception as e:
        logger.error(f"Failed to aggregate: {e}", exc_info=True)
        return {
            'status': 'FAILED',
            'errorType': type(e).__name__,
            'errorMessage': str(e)
        }
