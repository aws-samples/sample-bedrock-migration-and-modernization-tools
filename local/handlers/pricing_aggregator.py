"""
Pricing Aggregator - Local Handler

Merges pricing data from all three Bedrock service codes into a unified structure.
Transforms data to match the expected frontend schema with pricing_groups.

This is a standalone version of backend/lambdas/pricing-aggregator/handler.py
"""

import re
from collections import defaultdict
from local.handlers.config import get_config


def get_region_locations() -> dict:
    """Get region locations from configuration."""
    return get_config().get('region_configuration', {}).get('region_locations', {})


def get_provider_patterns() -> dict:
    """Get provider patterns from configuration."""
    return get_config().get('provider_configuration', {}).get('provider_patterns', {})


def get_explicit_provider_names() -> dict:
    """Get explicit provider name mappings from configuration."""
    return get_config().get('provider_configuration', {}).get('explicit_provider_names', {})


def determine_pricing_type(usage_type: str, unit: str, description: str) -> dict:
    """Determine the pricing type and unit from usage type, unit, and description."""
    usage_lower = usage_type.lower()
    unit_lower = (unit or '').lower()
    desc_lower = (description or '').lower()

    is_input = 'input' in usage_lower or 'input' in desc_lower
    is_output = 'output' in usage_lower or 'output' in desc_lower

    # Check for per-image pricing
    is_image_pricing = (
        'per image' in desc_lower or
        unit_lower == 'images' or
        unit_lower == 'image' or
        'images processed' in desc_lower or
        'created_image' in usage_lower or
        'output image' in desc_lower or
        ('stable' in desc_lower and 'image' in desc_lower)
    )

    if is_image_pricing:
        if 't2i' in usage_lower or 'i2i' in usage_lower or 'created_image' in usage_lower or ('stable' in desc_lower and 'image' in desc_lower):
            return {'pricing_type': 'image_generation', 'unit_label': 'per image', 'is_input': None, 'is_output': None}
        return {'pricing_type': 'image', 'unit_label': 'per image', 'is_input': is_input or not is_output, 'is_output': is_output}

    # Check for video generation
    is_video_generation = 'i2v' in usage_lower or 't2v' in usage_lower or ('video' in usage_lower and ('generation' in desc_lower or 'generated' in desc_lower))
    if is_video_generation:
        return {'pricing_type': 'video_generation', 'unit_label': 'per video', 'is_input': None, 'is_output': None}

    # Check for video pricing
    if 'video' in usage_lower and ('second' in unit_lower or 'frame' in unit_lower):
        return {'pricing_type': 'video', 'unit_label': f'per {unit_lower}', 'is_input': is_input, 'is_output': is_output}

    # Check for model units
    if 'modelunit' in usage_lower or 'model-unit' in usage_lower or 'modelunits' in unit_lower:
        return {'pricing_type': 'model_unit', 'unit_label': 'per hour', 'is_input': None, 'is_output': None}

    # Check for search units
    if 'search' in unit_lower or 'search' in desc_lower or 'rerank' in usage_lower or 'rerank' in desc_lower:
        return {'pricing_type': 'search_unit', 'unit_label': 'per 1K search units', 'is_input': None, 'is_output': None}

    # Check for video per-second pricing
    if ('second' in unit_lower or 'per second' in desc_lower) and ('video' in desc_lower or 'ray' in usage_lower):
        return {'pricing_type': 'video_second', 'unit_label': 'per second', 'is_input': None, 'is_output': None}

    # Default to token-based
    return {'pricing_type': 'token', 'unit_label': 'per 1K tokens', 'is_input': is_input, 'is_output': is_output}


def determine_pricing_group(usage_type: str, inference_type: str) -> str:
    """Determine the pricing group based on usage type and inference type."""
    usage_lower = usage_type.lower()
    inference_lower = inference_type.lower() if inference_type else ''

    is_global = 'global' in usage_lower or 'cross-region' in usage_lower
    is_batch = 'batch' in usage_lower
    is_long_context = 'long-context' in usage_lower or 'long context' in inference_lower or '_lctx' in usage_lower or 'longcontext' in usage_lower
    is_provisioned = 'provisioned' in usage_lower or 'provisioned' in inference_lower or 'reserved' in usage_lower or '_tpm_' in usage_lower
    is_custom = 'custom' in usage_lower or 'fine-tun' in usage_lower

    if is_custom:
        return 'Custom Model'
    elif is_provisioned:
        return 'Provisioned Throughput'
    elif is_batch and is_long_context and is_global:
        return 'Batch Long Context Global'
    elif is_batch and is_long_context:
        return 'Batch Long Context'
    elif is_batch and is_global:
        return 'Batch Global'
    elif is_batch:
        return 'Batch'
    elif is_long_context and is_global:
        return 'On-Demand Long Context Global'
    elif is_long_context:
        return 'On-Demand Long Context'
    elif is_global:
        return 'On-Demand Global'
    else:
        return 'On-Demand'


def clean_model_name(raw_name: str) -> str:
    """Clean model name by removing AWS-specific suffixes."""
    if not raw_name or raw_name.lower() in ['unknown', 'unknown model']:
        return raw_name

    cleaned = raw_name.strip()
    suffixes_to_remove = ['(Amazon Bedrock Edition)', '(Amazon Bedrock)', 'Amazon Bedrock Edition', 'Amazon Bedrock']
    for suffix in suffixes_to_remove:
        if suffix in cleaned:
            cleaned = cleaned.replace(suffix, '').strip()
    return cleaned if cleaned else raw_name


def extract_from_usagetype(usagetype: str) -> str:
    """Extract model name from usagetype as fallback."""
    if not usagetype:
        return None

    parts = usagetype.split('-')
    if len(parts) < 2:
        return None

    skip_parts = ['mp', 'input', 'output', 'tokens', 'count', 'units', 'cache', 'read', 'write']
    for part in parts[1:]:
        if part.lower() in skip_parts:
            continue
        if len(part) > 3 and any(c.isalpha() for c in part):
            formatted = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
            if len(formatted) > 3:
                return formatted
    return None


def extract_raw_model_name(attributes: dict) -> str:
    """Extract raw model name using multi-strategy approach."""
    servicename = attributes.get('servicename', '').strip()
    if servicename and servicename not in ['Amazon Bedrock', 'Amazon Bedrock Service']:
        return servicename

    model = attributes.get('model', '').strip()
    if model and model.lower() != 'unknown':
        return model

    titan_model = attributes.get('titanModel', '').strip()
    if titan_model:
        return titan_model

    usagetype = attributes.get('usagetype', '')
    if usagetype:
        extracted = extract_from_usagetype(usagetype)
        if extracted:
            return extracted

    return 'Unknown Model'


def extract_model_info(product: dict) -> dict:
    """Extract model information from a pricing product."""
    attributes = product.get('product', {}).get('attributes', {})
    terms = product.get('terms', {})

    price_per_unit = None
    unit = None
    currency = 'USD'
    description = ''

    on_demand = terms.get('OnDemand', {})
    for term_key, term_value in on_demand.items():
        price_dimensions = term_value.get('priceDimensions', {})
        for dim_key, dim_value in price_dimensions.items():
            price_per_unit = dim_value.get('pricePerUnit', {}).get('USD')
            unit = dim_value.get('unit')
            description = dim_value.get('description', '')
            break
        break

    try:
        price = float(price_per_unit) if price_per_unit else None
    except (ValueError, TypeError):
        price = None

    original_price = price
    desc_lower = description.lower()
    is_per_million = 'per 1m' in desc_lower or 'million' in desc_lower or 'per 1,000,000' in desc_lower or '1000000' in desc_lower
    if price and is_per_million:
        price = price / 1000

    raw_model_name = extract_raw_model_name(attributes)
    model_name = clean_model_name(raw_model_name)

    return {
        'model': model_name,
        'region': attributes.get('regionCode', 'Unknown'),
        'inferenceType': attributes.get('inferenceType', ''),
        'usageType': attributes.get('usagetype', ''),
        'operation': attributes.get('operation', ''),
        'price': price,
        'original_price': original_price,
        'unit': unit,
        'currency': currency,
        'sku': product.get('product', {}).get('sku', ''),
        'description': description,
        'serviceCode': attributes.get('servicecode', 'AmazonBedrock'),
        'attributes': attributes
    }


def detect_custom_model_type(description: str, dimension: str) -> str:
    """Detect if this is a Custom Model Import vs Custom Model Training."""
    desc_lower = description.lower()
    dim_lower = dimension.lower()

    import_indicators = ['flan architecture', 'llama architecture', 'inference for', 'storage for',
                        'custom model unit per min for inference', 'custom model unit/month storage',
                        'imported model', 'model import']
    training_indicators = ['customization-training', 'customization-storage', 'fine', 'finetun',
                          'training', 'custom training', 'model customization']

    if any(indicator in desc_lower or indicator in dim_lower for indicator in import_indicators):
        return 'Custom Model Import'
    if any(indicator in desc_lower or indicator in dim_lower for indicator in training_indicators):
        return 'Custom Model Training'
    return None


def normalize_provider_name(provider: str) -> str:
    """Normalize provider name to match model data provider names."""
    if not provider:
        return provider
    provider_lower = provider.lower().strip()
    explicit_names = get_explicit_provider_names()
    if provider_lower in explicit_names:
        return explicit_names[provider_lower]
    return provider


def infer_provider(model_name: str, attributes: dict = None) -> str:
    """Infer the provider from the model name and attributes."""
    model_lower = model_name.lower()

    if attributes:
        explicit_provider = attributes.get('provider', '').strip()
        if explicit_provider and explicit_provider.lower() != 'unknown':
            return normalize_provider_name(explicit_provider)

    explicit_names = get_explicit_provider_names()
    provider_patterns = get_provider_patterns()

    for explicit_name, provider in explicit_names.items():
        if explicit_name in model_lower:
            return provider

    for provider, patterns in provider_patterns.items():
        for pattern in patterns:
            if pattern in model_lower:
                return provider

    if attributes:
        all_text = ' '.join(str(v) for v in attributes.values()).lower()
        for provider, patterns in provider_patterns.items():
            for pattern in patterns:
                if pattern in all_text:
                    return provider

    return 'Unknown Models'


def normalize_model_id(model_name: str, provider: str) -> str:
    """Normalize model name to a consistent ID format."""
    provider_prefix = provider.lower().replace(' ', '-').replace('_', '-')
    if provider_prefix == 'unknown-models':
        provider_prefix = 'unknown'
    model_clean = model_name.lower().replace(' ', '-').replace('.', '-')
    return f"{provider_prefix}.{model_clean}"


def aggregate_pricing(all_products: list) -> tuple:
    """Aggregate all pricing products into the expected schema structure."""
    models_data = defaultdict(lambda: {
        'model_name': '',
        'model_provider': '',
        'regions': defaultdict(lambda: {'pricing_groups': defaultdict(list)})
    })

    group_types_seen = set()
    total_entries = 0

    for product in all_products:
        info = extract_model_info(product)
        model_name = info['model']
        region = info['region']

        if model_name == 'Unknown' or model_name == 'Unknown Model' or region == 'Unknown':
            continue

        custom_model_type = detect_custom_model_type(info['description'], info['usageType'])

        if custom_model_type == 'Custom Model Import':
            provider = 'Custom Model Import'
        else:
            provider = infer_provider(model_name, info.get('attributes'))

        model_id = normalize_model_id(model_name, provider)
        pricing_group = determine_pricing_group(info['usageType'], info['inferenceType'])
        group_types_seen.add(pricing_group)

        region_locations = get_region_locations()
        location = region_locations.get(region, region)

        pricing_type_info = determine_pricing_type(info['usageType'], info['unit'], info['description'])

        pricing_entry = {
            'dimension': info['usageType'],
            'price_per_unit': info['price'],
            'price_per_thousand': info['price'] if pricing_type_info['pricing_type'] == 'token' else None,
            'original_price': info['original_price'],
            'unit': info['unit'] or 'tokens',
            'description': info['description'],
            'source_dataset': 'aws_pricing_api',
            'model_id': model_id,
            'model_name': model_name,
            'provider': provider,
            'model_provider': provider,
            'location': location,
            'operation': info['operation'],
            'service_code': info['serviceCode'],
            'pricing_type': pricing_type_info['pricing_type'],
            'unit_label': pricing_type_info['unit_label'],
            'is_input': pricing_type_info['is_input'],
            'is_output': pricing_type_info['is_output'],
            'pricing_characteristics': {
                'inference_type': 'on_demand' if 'on-demand' in pricing_group.lower() else ('batch' if 'batch' in pricing_group.lower() else 'other'),
                'context_type': 'long_context' if 'long context' in pricing_group.lower() else 'standard',
                'geographic_scope': 'global' if 'global' in pricing_group.lower() else 'regional'
            },
            'pricing_group': pricing_group
        }

        models_data[model_id]['model_name'] = model_name
        models_data[model_id]['model_provider'] = provider
        models_data[model_id]['pricing_types'] = models_data[model_id].get('pricing_types', set())
        models_data[model_id]['pricing_types'].add(pricing_type_info['pricing_type'])
        models_data[model_id]['regions'][region]['pricing_groups'][pricing_group].append(pricing_entry)
        total_entries += 1

    # Convert to final structure
    result = defaultdict(dict)
    total_regions_processed = 0
    total_groups_created = 0

    for model_id, model_data in models_data.items():
        provider = model_data['model_provider']
        pricing_types_list = sorted(list(model_data.get('pricing_types', set())))

        primary_pricing_type = 'token'
        for pt in ['video_generation', 'image_generation', 'video_second', 'video', 'image', 'search_unit', 'token', 'model_unit']:
            if pt in pricing_types_list:
                primary_pricing_type = pt
                break

        model_entry = {
            'model_name': model_data['model_name'],
            'model_provider': provider,
            'pricing_types': pricing_types_list,
            'primary_pricing_type': primary_pricing_type,
            'regions': {}
        }

        for region, region_data in model_data['regions'].items():
            pricing_groups = dict(region_data['pricing_groups'])

            if 'On-Demand' not in pricing_groups and 'On-Demand Global' in pricing_groups:
                pricing_groups['On-Demand'] = pricing_groups['On-Demand Global']
            if 'Batch' not in pricing_groups and 'Batch Global' in pricing_groups:
                pricing_groups['Batch'] = pricing_groups['Batch Global']

            total_dimensions = sum(len(entries) for entries in pricing_groups.values())
            groups_count = len(pricing_groups)

            group_sizes = {group: len(entries) for group, entries in pricing_groups.items()}
            largest_groups = sorted(group_sizes.items(), key=lambda x: x[1], reverse=True)[:5]

            model_entry['regions'][region] = {
                'pricing_groups': pricing_groups,
                'total_dimensions': total_dimensions,
                'groups_count': groups_count,
                'group_statistics': {
                    'total_entries': total_dimensions,
                    'total_groups': groups_count,
                    'group_sizes': group_sizes,
                    'largest_groups': largest_groups,
                    'average_entries_per_group': total_dimensions / groups_count if groups_count > 0 else 0
                }
            }

            total_regions_processed += 1
            total_groups_created += groups_count

        result[provider][model_id] = model_entry

    metadata_stats = {
        'total_entries': total_entries,
        'total_regions_processed': total_regions_processed,
        'total_groups_created': total_groups_created,
        'group_types_seen': sorted(list(group_types_seen))
    }

    return result, metadata_stats
