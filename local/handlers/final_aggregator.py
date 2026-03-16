"""
Final Aggregator - Local Handler

Merges all collected data into the final comprehensive JSON outputs.
Produces the exact same schema as the AWS Step Functions pipeline.

This is a standalone version of backend/lambdas/final-aggregator/handler.py
"""

import re
from collections import defaultdict
from local.handlers.config import get_context_window_specs


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
    """Get context window specs from config for a model."""
    specs = get_context_window_specs()

    # Try exact match first
    if model_id in specs:
        return specs[model_id]

    # Try base model ID (without version)
    base_id = model_id.split(':')[0] if ':' in model_id else model_id
    if base_id in specs:
        return specs[base_id]

    # Try prefix matching
    for spec_key, spec_data in specs.items():
        if model_id.startswith(spec_key) or base_id.startswith(spec_key):
            return spec_data

    return None


def build_cross_region_inference(model_id: str, features_by_region: dict) -> dict:
    """Build cross-region inference (CRIS) info from inference profiles."""
    supported_regions = []
    profile_ids = []

    for region, profiles in features_by_region.items():
        for profile in profiles:
            profile_models = profile.get('models', [])
            for model_ref in profile_models:
                model_arn = model_ref.get('modelArn', model_ref.get('model_arn', ''))
                if model_id in model_arn:
                    if region not in supported_regions:
                        supported_regions.append(region)
                    profile_id = profile.get('inference_profile_id', profile.get('inferenceProfileId', ''))
                    if profile_id and profile_id not in profile_ids:
                        profile_ids.append(profile_id)

    return {
        'supported': len(supported_regions) > 0,
        'source_regions': sorted(supported_regions),
        'profile_ids': profile_ids,
        'detection_method': 'inference_profiles' if supported_regions else 'none'
    }


def _normalize_for_quota_matching(name: str) -> str:
    """Normalize a string for quota matching."""
    n = name.lower().strip()
    n = re.sub(r'[^\w\s.-]', ' ', n)
    n = ' '.join(n.split())
    return n


def _extract_quota_model_ref(quota_name: str) -> str:
    """Extract the model reference string from a quota name."""
    name = quota_name.strip()
    name = re.sub(r'^\([^)]+\)\s*', '', name)
    name = re.sub(r'\s*\(doubled\s+for[^)]*\)\s*$', '', name, flags=re.I)

    parts = re.split(r'\bfor\s+', name, flags=re.I)
    if len(parts) < 2:
        return None

    ref = parts[-1].strip()
    ref = re.sub(r'^(?:a|an)\s+', '', ref, flags=re.I)
    ref = re.sub(r'^(?:base|custom)\s+model\s+', '', ref, flags=re.I)
    ref = re.sub(r'\s+(?:Fine[- ]?tuning|Continued Pre[- ]?Training|distillation)\b.*$', '', ref, flags=re.I)
    ref = re.sub(r'\s+per\s+month$', '', ref, flags=re.I)

    return ref.strip() if ref.strip() else None


def build_model_quotas(model_id: str, model_name: str, quotas_by_region: dict, model_provider: str = '') -> dict:
    """Build model quotas by matching quota names to model."""
    model_quotas = {}
    model_name_norm = _normalize_for_quota_matching(model_name)

    for region, quotas in quotas_by_region.items():
        for quota in quotas:
            quota_name = quota.get('quotaName', quota.get('quota_name', ''))
            quota_ref = _extract_quota_model_ref(quota_name)

            if not quota_ref:
                continue

            quota_ref_norm = _normalize_for_quota_matching(quota_ref)

            # Check if this quota matches this model
            if model_name_norm in quota_ref_norm or quota_ref_norm in model_name_norm:
                model_quotas.setdefault(region, []).append({
                    'quota_code': quota.get('quotaCode', quota.get('quota_code', '')),
                    'quota_name': quota_name,
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
    """Determine consumption options from inference types and pricing data."""
    options = set()

    type_mapping = {
        'ON_DEMAND': 'on_demand',
        'PROVISIONED': 'provisioned_throughput',
        'INFERENCE_PROFILE': 'cross_region_inference'
    }
    for inf_type in inference_types:
        if inf_type in type_mapping:
            options.add(type_mapping[inf_type])

    if pricing_data and pricing_ref:
        provider = pricing_ref.get('provider', '')
        model_key = pricing_ref.get('model_key', '')

        if provider and model_key:
            providers = pricing_data.get('providers', {})
            prov_data = providers.get(provider, {})
            model_pricing = prov_data.get(model_key, {})

            if isinstance(model_pricing, dict) and 'regions' in model_pricing:
                for region_data in model_pricing.get('regions', {}).values():
                    pricing_groups = region_data.get('pricing_groups', {})
                    if any(g.startswith('On-Demand') for g in pricing_groups.keys()):
                        options.add('on_demand')
                    if any(g.startswith('Batch') for g in pricing_groups.keys()):
                        options.add('batch')
                    if 'Provisioned Throughput' in pricing_groups:
                        options.add('provisioned_throughput')
                    break

    if not options:
        options.add('on_demand')

    order = ['on_demand', 'batch', 'cross_region_inference', 'provisioned_throughput']
    return sorted(list(options), key=lambda x: order.index(x) if x in order else len(order))


def check_batch_inference(model_id: str, pricing_data: dict, pricing_ref: dict = None, regional_availability: list = None) -> dict:
    """Check if batch inference is supported based on pricing data."""
    supported_regions = []

    lookup_keys = []
    if pricing_ref:
        provider = pricing_ref.get('provider', '')
        model_key = pricing_ref.get('model_key', '')
        if provider and model_key:
            lookup_keys.append((provider, model_key))
    lookup_keys.append((None, model_id))

    providers = pricing_data.get('providers', {})

    for provider_hint, lookup_key in lookup_keys:
        if supported_regions:
            break

        for prov_name, prov_data in providers.items():
            if supported_regions:
                break
            if provider_hint and prov_name.lower() != provider_hint.lower():
                continue

            if isinstance(prov_data, dict):
                if lookup_key in prov_data:
                    model_data = prov_data[lookup_key]
                    if isinstance(model_data, dict) and 'regions' in model_data:
                        for region, region_data in model_data.get('regions', {}).items():
                            pricing_groups = region_data.get('pricing_groups', {})
                            if any(g.startswith('Batch') for g in pricing_groups.keys()):
                                if region not in supported_regions:
                                    supported_regions.append(region)

    total_regions = len(regional_availability) if regional_availability else 0
    batch_region_count = len(supported_regions)
    coverage = (batch_region_count / total_regions * 100) if total_regions > 0 else 0.0
    coverage = min(coverage, 100.0)

    return {
        'supported': len(supported_regions) > 0,
        'supported_regions': sorted(supported_regions),
        'coverage_percentage': round(coverage, 1),
        'detection_method': 'pricing_data' if supported_regions else 'no_pricing_data'
    }


def find_matching_availability(model_id: str, model_availability: dict) -> list:
    """Find regional availability for a model, handling ID format differences."""
    if model_id in model_availability:
        return model_availability[model_id]

    base_model_id = model_id.split(':')[0] if ':' in model_id else model_id
    if base_model_id in model_availability:
        return model_availability[base_model_id]

    model_id_lower = model_id.lower()
    best_match_key = None
    best_match_length = 0

    for pricing_key in model_availability.keys():
        pricing_key_lower = pricing_key.lower()

        if pricing_key_lower in model_id_lower or model_id_lower.startswith(pricing_key_lower):
            if len(pricing_key) > best_match_length:
                best_match_key = pricing_key
                best_match_length = len(pricing_key)
            continue

        pricing_parts = pricing_key_lower.replace('anthropic.', '').replace('amazon.', '').replace('meta.', '').replace('mistral.', '').replace('cohere.', '').replace('ai21.', '').replace('stability.', '').replace('nvidia.', '').replace('luma.', '')
        model_parts = model_id_lower.replace('anthropic.', '').replace('amazon.', '').replace('meta.', '').replace('mistral.', '').replace('cohere.', '').replace('ai21.', '').replace('stability.', '').replace('nvidia.', '').replace('luma.', '')

        if pricing_parts and model_parts:
            model_core = re.sub(r'-\d{8}-v\d+.*$', '', model_parts)
            if pricing_parts == model_core or pricing_parts in model_core or model_core.startswith(pricing_parts):
                if len(pricing_key) > best_match_length:
                    best_match_key = pricing_key
                    best_match_length = len(pricing_key)

    if best_match_key:
        return model_availability[best_match_key]

    return []


def transform_model_to_schema(
    model_id: str,
    model: dict,
    regional_availability: list,
    quotas_by_region: dict,
    features_by_region: dict,
    enriched_model: dict,
    pricing_data: dict,
    collection_timestamp: str
) -> dict:
    """Transform model data into the final schema."""
    capabilities = enriched_model.get('model_capabilities', model.get('model_capabilities', []))
    use_cases = enriched_model.get('model_use_cases', model.get('model_use_cases', []))
    doc_links = enriched_model.get('documentation_links', model.get('documentation_links', {}))

    # Build converse data
    context_window = None
    max_output = None
    source = None

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

    if context_window is None:
        variant_cw = model.get('variant_context_window')
        if variant_cw and isinstance(variant_cw, (int, float)):
            context_window = int(variant_cw)
            source = 'model_id_variant'

    config_specs = get_context_window_from_config(model_id)
    if config_specs:
        if context_window is None:
            context_window = config_specs.get('standard_context')
            source = config_specs.get('source', 'config')
        if max_output is None:
            max_output = config_specs.get('max_output')

    converse_data = {
        'context_window': context_window,
        'max_output_tokens': max_output,
        'size_category': get_size_category(context_window),
        'verified': source is not None and source != 'unknown',
        'source': source or 'unknown',
        'litellm_verified': False,
        'capabilities_count': len(capabilities),
        'use_cases_count': len(use_cases),
        'regions_count': len(regional_availability)
    }

    # Build cross-region inference
    cross_region = build_cross_region_inference(model_id, features_by_region)

    # Build model quotas
    model_quotas = build_model_quotas(
        model_id,
        model.get('model_name', ''),
        quotas_by_region,
        model_provider=model.get('model_provider', '')
    )

    # Get pricing info
    model_pricing_data = model.get('model_pricing', {})
    has_pricing = model_pricing_data.get('is_pricing_available', model.get('has_pricing', False))
    pricing_ref_id = model_pricing_data.get('pricing_reference_id', '')
    upstream_pricing_ref = model_pricing_data.get('pricing_file_reference')

    # Check batch inference
    regions_for_coverage = regional_availability if regional_availability else model.get('regions_available', [])
    batch_inference = check_batch_inference(model_id, pricing_data, upstream_pricing_ref, regions_for_coverage)

    # Expand regions with batch supported regions
    regions_set = set(regional_availability)
    if batch_inference.get('supported_regions'):
        regions_set.update(batch_inference['supported_regions'])
    if len(regions_set) > len(regional_availability):
        regional_availability = sorted(list(regions_set))

    if upstream_pricing_ref and isinstance(upstream_pricing_ref, dict):
        pricing_provider = upstream_pricing_ref.get('provider', model.get('model_provider', ''))
        pricing_model_key = upstream_pricing_ref.get('model_key', pricing_ref_id or model_id)
    else:
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
            'integration_source': 'local-collector',
            'has_pricing_data': has_pricing,
            'integration_timestamp': collection_timestamp,
            'reference_based': True
        }
    }

    # Documentation links
    documentation_links = doc_links.copy() if doc_links else {}
    if 'aws_bedrock_guide' not in documentation_links:
        documentation_links['aws_bedrock_guide'] = 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html'
    if 'pricing_guide' not in documentation_links:
        documentation_links['pricing_guide'] = 'https://aws.amazon.com/bedrock/pricing/'

    # Model modalities
    model_modalities = model.get('model_modalities', {})
    if not model_modalities:
        model_modalities = {
            'input_modalities': model.get('input_modalities', []),
            'output_modalities': model.get('output_modalities', [])
        }

    # Model lifecycle
    model_lifecycle = model.get('model_lifecycle', {})
    if not model_lifecycle:
        model_lifecycle = {'status': 'ACTIVE'}

    # Collection metadata
    collection_metadata = model.get('collection_metadata', {})
    collection_metadata['last_aggregated_at'] = collection_timestamp

    # Customization
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


def build_final_models(
    models_with_pricing: dict,
    regional_availability: dict,
    quotas_by_region: dict,
    features_by_region: dict,
    enriched_models: dict,
    pricing_data: dict,
    collection_timestamp: str
) -> dict:
    """Build the final comprehensive models structure."""
    providers = models_with_pricing.get('providers', {})
    enriched_providers = enriched_models.get('providers', {})
    model_availability = regional_availability.get('model_availability', {})

    result_providers = {}

    for provider, provider_data in providers.items():
        result_providers[provider] = {'models': {}}

        for model_id, model in provider_data.get('models', {}).items():
            regions = find_matching_availability(model_id, model_availability)
            enriched = enriched_providers.get(provider, {}).get('models', {}).get(model_id, {})

            transformed = transform_model_to_schema(
                model_id=model_id,
                model=model,
                regional_availability=regions,
                quotas_by_region=quotas_by_region,
                features_by_region=features_by_region,
                enriched_model=enriched,
                pricing_data=pricing_data,
                collection_timestamp=collection_timestamp
            )

            result_providers[provider]['models'][model_id] = transformed

    return result_providers
