"""
Pricing Aggregator Lambda

Merges pricing data from all three Bedrock service codes into a unified structure.
Transforms data to match the expected frontend schema with pricing_groups.
"""

import logging
import os
import re
import time
from typing import Any
from collections import defaultdict

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

# Region code to location name mapping
REGION_LOCATIONS = {
    'us-east-1': 'US East (N. Virginia)',
    'us-east-2': 'US East (Ohio)',
    'us-west-1': 'US West (N. California)',
    'us-west-2': 'US West (Oregon)',
    'eu-west-1': 'Europe (Ireland)',
    'eu-west-2': 'Europe (London)',
    'eu-west-3': 'Europe (Paris)',
    'eu-central-1': 'Europe (Frankfurt)',
    'eu-north-1': 'Europe (Stockholm)',
    'eu-south-1': 'Europe (Milan)',
    'eu-south-2': 'Europe (Spain)',
    'ap-northeast-1': 'Asia Pacific (Tokyo)',
    'ap-northeast-2': 'Asia Pacific (Seoul)',
    'ap-northeast-3': 'Asia Pacific (Osaka)',
    'ap-southeast-1': 'Asia Pacific (Singapore)',
    'ap-southeast-2': 'Asia Pacific (Sydney)',
    'ap-southeast-3': 'Asia Pacific (Jakarta)',
    'ap-southeast-5': 'Asia Pacific (Malaysia)',
    'ap-south-1': 'Asia Pacific (Mumbai)',
    'sa-east-1': 'South America (Sao Paulo)',
    'ca-central-1': 'Canada (Central)',
    'ca-west-1': 'Canada West (Calgary)',
    'me-south-1': 'Middle East (Bahrain)',
    'me-central-1': 'Middle East (UAE)',
    'il-central-1': 'Israel (Tel Aviv)',
    'af-south-1': 'Africa (Cape Town)',
}


def determine_pricing_type(usage_type: str, unit: str, description: str) -> dict:
    """
    Determine the pricing type and unit from usage type, unit, and description.

    Returns:
        {
            'pricing_type': 'token' | 'image' | 'video_second' | 'model_unit' | 'other',
            'unit_label': 'per 1K tokens' | 'per image' | etc.,
            'is_input': True/False/None,
            'is_output': True/False/None,
        }
    """
    usage_lower = usage_type.lower()
    unit_lower = (unit or '').lower()
    desc_lower = (description or '').lower()

    # Determine if input/output
    is_input = 'input' in usage_lower or 'input' in desc_lower
    is_output = 'output' in usage_lower or 'output' in desc_lower

    # Check for per-image pricing
    # Patterns: 'per image', 'image', 'images', 'images processed', 'created_image', 'output image'
    is_image_pricing = (
        'per image' in desc_lower or
        unit_lower == 'images' or
        unit_lower == 'image' or  # Support singular form (e.g., Nova Canvas)
        'images processed' in desc_lower or
        'created_image' in usage_lower or
        'output image' in desc_lower or
        ('stable' in desc_lower and 'image' in desc_lower)  # Stability AI pattern
    )

    if is_image_pricing:
        # Image generation models (Canvas, Titan Image Generator, Stability AI, etc.)
        if 't2i' in usage_lower or 'i2i' in usage_lower or 'created_image' in usage_lower or ('stable' in desc_lower and 'image' in desc_lower):
            return {
                'pricing_type': 'image_generation',
                'unit_label': 'per image',
                'is_input': None,
                'is_output': None,
            }
        # Image embedding/processing
        return {
            'pricing_type': 'image',
            'unit_label': 'per image',
            'is_input': is_input or not is_output,
            'is_output': is_output,
        }

    # Check for video generation (I2V = image-to-video, T2V = text-to-video)
    # Patterns: NovaReel-I2V-Medfps-HDRes, NovaReel-T2V-Lowfps-SDRes
    is_video_generation = (
        'i2v' in usage_lower or  # image-to-video
        't2v' in usage_lower or  # text-to-video
        ('video' in usage_lower and ('generation' in desc_lower or 'generated' in desc_lower))
    )

    if is_video_generation:
        return {
            'pricing_type': 'video_generation',
            'unit_label': 'per video',
            'is_input': None,
            'is_output': None,
        }

    # Check for video pricing (per second or per frame) - for video processing, not generation
    if 'video' in usage_lower and ('second' in unit_lower or 'frame' in unit_lower):
        return {
            'pricing_type': 'video',
            'unit_label': f'per {unit_lower}',
            'is_input': is_input,
            'is_output': is_output,
        }

    # Check for model units (provisioned throughput)
    if 'modelunit' in usage_lower or 'model-unit' in usage_lower or 'modelunits' in unit_lower:
        return {
            'pricing_type': 'model_unit',
            'unit_label': 'per hour',
            'is_input': None,
            'is_output': None,
        }

    # Check for token-based pricing (most common)
    if 'token' in usage_lower or 'token' in desc_lower or '1k token' in desc_lower or '1m token' in desc_lower:
        return {
            'pricing_type': 'token',
            'unit_label': 'per 1K tokens',
            'is_input': is_input,
            'is_output': is_output,
        }

    # Default to token-based for text models
    return {
        'pricing_type': 'token',
        'unit_label': 'per 1K tokens',
        'is_input': is_input,
        'is_output': is_output,
    }


def determine_pricing_group(usage_type: str, inference_type: str) -> str:
    """Determine the pricing group based on usage type and inference type."""
    usage_lower = usage_type.lower()
    inference_lower = inference_type.lower() if inference_type else ''

    # Check for global (cross-region)
    is_global = 'global' in usage_lower or 'cross-region' in usage_lower

    # Check for batch
    is_batch = 'batch' in usage_lower

    # Check for long context
    is_long_context = 'long-context' in usage_lower or 'long context' in inference_lower

    # Check for provisioned
    is_provisioned = 'provisioned' in usage_lower or 'provisioned' in inference_lower

    # Check for custom model
    is_custom = 'custom' in usage_lower or 'fine-tun' in usage_lower

    # Determine group
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
    """Clean model name by removing AWS-specific suffixes.

    Examples:
        'Stable Diffusion 3 Large v1.0 (Amazon Bedrock Edition)' -> 'Stable Diffusion 3 Large v1.0'
        'Claude 3.5 Sonnet (Amazon Bedrock Edition)' -> 'Claude 3.5 Sonnet'
    """
    if not raw_name or raw_name.lower() in ['unknown', 'unknown model']:
        return raw_name

    cleaned = raw_name.strip()

    # Remove AWS-specific suffixes
    suffixes_to_remove = [
        '(Amazon Bedrock Edition)',
        '(Amazon Bedrock)',
        'Amazon Bedrock Edition',
        'Amazon Bedrock'
    ]

    for suffix in suffixes_to_remove:
        if suffix in cleaned:
            cleaned = cleaned.replace(suffix, '').strip()

    return cleaned if cleaned else raw_name


def extract_from_usagetype(usagetype: str) -> str:
    """Extract model name from usagetype as fallback.

    Patterns like:
    - "USE1-NovaLite-input-tokens" -> "Nova Lite"
    - "APN1-Claude3Sonnet-output" -> "Claude 3 Sonnet"
    """
    if not usagetype:
        return None

    # Remove region prefix (e.g., "USE1-", "APN1-")
    parts = usagetype.split('-')
    if len(parts) < 2:
        return None

    # Skip common non-model parts
    skip_parts = ['mp', 'input', 'output', 'tokens', 'count', 'units', 'cache', 'read', 'write']

    for part in parts[1:]:
        if part.lower() in skip_parts:
            continue

        # If part looks like a model name (contains letters and is substantial)
        if len(part) > 3 and any(c.isalpha() for c in part):
            # Try to format it nicely (camelCase -> Title Case)
            formatted = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
            if len(formatted) > 3:
                return formatted

    return None


def extract_raw_model_name(attributes: dict) -> str:
    """Extract raw model name using multi-strategy approach.

    Priority order:
    1. servicename (for AmazonBedrockFoundationModels)
    2. model (for AmazonBedrock, AmazonBedrockService)
    3. titanModel (special case for Titan models)
    4. Fallback extraction from usagetype
    """
    # Strategy 1: servicename (most common in AmazonBedrockFoundationModels)
    servicename = attributes.get('servicename', '').strip()
    if servicename and servicename not in ['Amazon Bedrock', 'Amazon Bedrock Service']:
        return servicename

    # Strategy 2: model field (most common in AmazonBedrock, AmazonBedrockService)
    model = attributes.get('model', '').strip()
    if model and model.lower() != 'unknown':
        return model

    # Strategy 3: titanModel field (special case)
    titan_model = attributes.get('titanModel', '').strip()
    if titan_model:
        return titan_model

    # Strategy 4: Extract from usagetype (fallback)
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

    # Extract pricing from OnDemand terms
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

    # Parse the price
    try:
        price = float(price_per_unit) if price_per_unit else None
    except (ValueError, TypeError):
        price = None

    # Normalize price to per-thousand if needed (some prices are per-million)
    original_price = price
    if price and 'per 1M' in description.lower():
        price = price / 1000  # Convert to per-thousand

    # Get model name using multi-strategy extraction
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
        'attributes': attributes  # Pass all attributes for provider detection fallback
    }


def detect_custom_model_type(description: str, dimension: str) -> str:
    """Detect if this is a Custom Model Import vs Custom Model Training.

    Args:
        description: Price description
        dimension: Price dimension (usagetype)

    Returns:
        'Custom Model Import', 'Custom Model Training', or None
    """
    desc_lower = description.lower()
    dim_lower = dimension.lower()

    # Custom Model Import indicators
    import_indicators = [
        'flan architecture', 'llama architecture', 'inference for', 'storage for',
        'custom model unit per min for inference', 'custom model unit/month storage',
        'imported model', 'model import'
    ]

    # Custom Model Training/Customization indicators
    training_indicators = [
        'customization-training', 'customization-storage', 'fine', 'finetun',
        'training', 'custom training', 'model customization'
    ]

    # Check for import patterns
    if any(indicator in desc_lower or indicator in dim_lower for indicator in import_indicators):
        return 'Custom Model Import'

    # Check for training/customization patterns
    if any(indicator in desc_lower or indicator in dim_lower for indicator in training_indicators):
        return 'Custom Model Training'

    return None


# Provider keyword patterns - expanded to match old version's coverage
PROVIDER_PATTERNS = {
    'Amazon': ['titan', 'nova', 'amazon-bedrock', 'rerank'],
    'Anthropic': ['claude', 'anthropic'],
    'Meta': ['llama', 'mllama'],
    'Mistral AI': ['mistral', 'mixtral', 'ministral', 'magistral', 'pixtral', 'voxtral'],
    'Cohere': ['cohere', 'command', 'embed'],  # 'rerank' moved to Amazon per old version
    'AI21 Labs': ['ai21', 'jamba', 'jurassic'],
    'Stability AI': ['stable', 'stability', 'sdxl'],
    'Luma AI': ['luma', 'ray'],  # Expanded: 'ray' not just 'ray-v'
    'Writer': ['writer', 'palmyra'],
    'NVIDIA': ['nvidia', 'nemotron'],
    'Qwen': ['qwen'],
    'OpenAI': ['gpt', 'openai'],  # Expanded: 'gpt' not just 'gpt-oss'
    'DeepSeek': ['deepseek', 'r1'],
    'Google': ['gemma', 'gemini'],
    'TwelveLabs': ['twelve', 'twelvelabs', 'marengo', 'pegasus'],  # Expanded: 'twelve'
    'MiniMax': ['minimax'],
    'Moonshot AI': ['kimi', 'moonshot'],
}

# Explicit provider name mappings (high confidence matches)
# Used both for extraction from model names AND for normalizing provider attributes
EXPLICIT_PROVIDER_NAMES = {
    'twelvelabs': 'TwelveLabs',
    'twelve labs': 'TwelveLabs',
    'cohere': 'Cohere',
    'luma ai': 'Luma AI',
    'luma': 'Luma AI',
    'anthropic': 'Anthropic',
    'stability ai': 'Stability AI',
    'ai21 labs': 'AI21 Labs',
    'ai21': 'AI21 Labs',
    'mistral ai': 'Mistral AI',
    'mistral': 'Mistral AI',
    'deepseek': 'DeepSeek',
    'writer': 'Writer',
    'meta': 'Meta',
    'amazon': 'Amazon',
    'google': 'Google',
    'nvidia': 'NVIDIA',
    'openai': 'OpenAI',
    'qwen': 'Qwen',
    'minimax': 'MiniMax',
}


def normalize_provider_name(provider: str) -> str:
    """Normalize provider name to match model data provider names.

    E.g., 'Mistral' -> 'Mistral AI', 'mistral' -> 'Mistral AI'
    """
    if not provider:
        return provider

    provider_lower = provider.lower().strip()

    # Check explicit mappings first
    if provider_lower in EXPLICIT_PROVIDER_NAMES:
        return EXPLICIT_PROVIDER_NAMES[provider_lower]

    # Return as-is if no mapping found
    return provider


def infer_provider(model_name: str, attributes: dict = None) -> str:
    """Infer the provider from the model name and attributes.

    Uses multi-strategy approach:
    1. Check explicit 'provider' attribute (normalized to match model data)
    2. Check explicit provider names in model name
    3. Check generic keywords in model name
    4. Fallback: search ALL attributes for provider keywords
    """
    model_lower = model_name.lower()

    # Strategy 1: Check explicit 'provider' attribute (AmazonBedrockService has this)
    if attributes:
        explicit_provider = attributes.get('provider', '').strip()
        if explicit_provider and explicit_provider.lower() != 'unknown':
            # Normalize to match model data provider names (e.g., 'Mistral' -> 'Mistral AI')
            return normalize_provider_name(explicit_provider)

    # Strategy 2: Check for explicit provider names in model name (high confidence)
    for explicit_name, provider in EXPLICIT_PROVIDER_NAMES.items():
        if explicit_name in model_lower:
            return provider

    # Strategy 3: Check generic keywords in model name
    for provider, patterns in PROVIDER_PATTERNS.items():
        for pattern in patterns:
            if pattern in model_lower:
                return provider

    # Strategy 4: Fallback - search ALL attributes for provider keywords
    if attributes:
        all_text = ' '.join(str(v) for v in attributes.values()).lower()
        for provider, patterns in PROVIDER_PATTERNS.items():
            for pattern in patterns:
                if pattern in all_text:
                    return provider

    return 'Unknown Models'


def normalize_model_id(model_name: str, provider: str) -> str:
    """Normalize model name to a consistent ID format."""
    # Create a provider prefix
    provider_prefix = provider.lower().replace(' ', '-').replace('_', '-')
    if provider_prefix == 'unknown-models':
        provider_prefix = 'unknown'

    # Clean the model name
    model_clean = model_name.lower().replace(' ', '-').replace('.', '-')

    return f"{provider_prefix}.{model_clean}"


def aggregate_pricing(all_products: list[dict]) -> tuple[dict, dict]:
    """
    Aggregate all pricing products into the expected schema structure.

    Output structure:
    {
        "providers": {
            "provider.model-id": {
                "model_name": "Model Name",
                "model_provider": "Provider",
                "regions": {
                    "us-east-1": {
                        "pricing_groups": {
                            "On-Demand": [...],
                            "Batch": [...]
                        },
                        "total_dimensions": 10,
                        "groups_count": 2,
                        "group_statistics": {...}
                    }
                }
            }
        }
    }
    """
    # Structure: provider_model_id -> region -> pricing_group -> entries
    models_data = defaultdict(lambda: {
        'model_name': '',
        'model_provider': '',
        'regions': defaultdict(lambda: {
            'pricing_groups': defaultdict(list)
        })
    })

    group_types_seen = set()
    total_entries = 0

    for product in all_products:
        info = extract_model_info(product)

        model_name = info['model']
        region = info['region']

        if model_name == 'Unknown' or model_name == 'Unknown Model' or region == 'Unknown':
            continue

        # Check for Custom Model Import/Training first
        custom_model_type = detect_custom_model_type(info['description'], info['usageType'])

        # Infer provider with all attributes for fallback detection
        if custom_model_type == 'Custom Model Import':
            provider = 'Custom Model Import'
        else:
            provider = infer_provider(model_name, info.get('attributes'))

        # Create model ID
        model_id = normalize_model_id(model_name, provider)

        # Determine pricing group
        pricing_group = determine_pricing_group(info['usageType'], info['inferenceType'])
        group_types_seen.add(pricing_group)

        # Get location name
        location = REGION_LOCATIONS.get(region, region)

        # Determine pricing type
        pricing_type_info = determine_pricing_type(
            info['usageType'],
            info['unit'],
            info['description']
        )

        # Build pricing entry in expected schema
        pricing_entry = {
            'dimension': info['usageType'],
            'price_per_unit': info['price'],  # Generic price per unit
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
                'inference_type': 'on_demand' if 'on-demand' in pricing_group.lower() else (
                    'batch' if 'batch' in pricing_group.lower() else 'other'
                ),
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

    # Convert to final structure nested by provider: providers -> Provider -> model_id -> data
    # This matches the frontend expected schema
    result = defaultdict(dict)
    total_regions_processed = 0
    total_groups_created = 0

    for model_id, model_data in models_data.items():
        provider = model_data['model_provider']

        # Convert pricing_types set to list for JSON serialization
        pricing_types_list = sorted(list(model_data.get('pricing_types', set())))

        # Determine primary pricing type for the model
        # Priority: video_generation > image_generation > video > image > model_unit > token
        # Image/video generation models should show per-image/video pricing, not token pricing
        primary_pricing_type = 'token'  # default
        for pt in ['video_generation', 'image_generation', 'video', 'image', 'model_unit', 'token']:
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

            # Ensure 'On-Demand' exists for frontend compatibility
            # Copy 'On-Demand Global' entries to 'On-Demand' if 'On-Demand' doesn't exist
            if 'On-Demand' not in pricing_groups and 'On-Demand Global' in pricing_groups:
                pricing_groups['On-Demand'] = pricing_groups['On-Demand Global']
            # Same for Batch
            if 'Batch' not in pricing_groups and 'Batch Global' in pricing_groups:
                pricing_groups['Batch'] = pricing_groups['Batch Global']

            total_dimensions = sum(len(entries) for entries in pricing_groups.values())
            groups_count = len(pricing_groups)

            # Calculate group statistics
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

        # Nest under provider name
        result[provider][model_id] = model_entry

    metadata_stats = {
        'total_entries': total_entries,
        'total_regions_processed': total_regions_processed,
        'total_groups_created': total_groups_created,
        'group_types_seen': sorted(list(group_types_seen))
    }

    return result, metadata_stats


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for pricing aggregation.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "pricingResults": [
                {"status": "SUCCESS", "serviceCode": "AmazonBedrock", "s3Key": "..."},
                ...
            ]
        }

    Output:
        {
            "status": "SUCCESS",
            "s3Key": "executions/{id}/merged/pricing.json",
            "providersCount": 17,
            "totalPricingEntries": 8716
        }
    """
    start_time = time.time()
    collection_timestamp = time.strftime('%Y-%m-%dT%H:%M:%S.%fZ', time.gmtime())

    # Validate required parameters
    try:
        validate_required_params(event, ['s3Bucket', 'executionId', 'pricingResults'], 'PricingAggregator')
    except ValidationError as e:
        return {
            'status': 'FAILED',
            'errorType': 'ValidationError',
            'errorMessage': str(e)
        }

    s3_bucket = event['s3Bucket']
    execution_id = parse_execution_id(event['executionId'])
    pricing_results = event['pricingResults']
    dry_run = event.get('dryRun', False)

    output_key = f"executions/{execution_id}/merged/pricing.json"

    logger.info(f"Aggregating pricing from {len(pricing_results)} sources")

    try:
        s3_client = get_s3_client()

        # Collect all products from successful collectors
        all_products = []
        successful_sources = []

        for item in pricing_results:
            # Handle nested result structure from Map state
            nested_result = item.get('result', {})
            status = item.get('status') or nested_result.get('status')
            s3_key = item.get('s3Key') or nested_result.get('s3Key')
            service_code = item.get('serviceCode')

            if status == 'SUCCESS' and s3_key:
                logger.info(f"Reading from s3://{s3_bucket}/{s3_key}")

                if not dry_run:
                    data = read_from_s3(s3_client, s3_bucket, s3_key)
                    products = data.get('products', [])
                    all_products.extend(products)
                    successful_sources.append({
                        'service_code': service_code,
                        's3_key': s3_key,
                        'count': len(products)
                    })
                    logger.info(f"Loaded {len(products)} products from {service_code}")
            else:
                logger.warning(f"Skipping non-successful result: {item}")

        if dry_run:
            all_products = []

        logger.info(f"Total products to aggregate: {len(all_products)}")

        # Aggregate pricing data in expected schema
        aggregated, metadata_stats = aggregate_pricing(all_products)

        # Convert defaultdict to regular dict for JSON serialization
        aggregated = dict(aggregated)

        # Count unique providers (now keys of aggregated since it's nested by provider)
        providers_count = len(aggregated)

        # Build output in expected schema
        output_data = {
            'metadata': {
                'generated_at': collection_timestamp,
                'version': '1.0.0',
                'total_pricing_entries': metadata_stats['total_entries'],
                'data_sources': {
                    'aws_pricing_api': {
                        'success': True,
                        'count': metadata_stats['total_entries'],
                        'error': None
                    }
                },
                'providers_count': providers_count,
                'total_regions_processed': metadata_stats['total_regions_processed'],
                'total_groups_created': metadata_stats['total_groups_created'],
                'unique_group_types': len(metadata_stats['group_types_seen']),
                'average_groups_per_region': (
                    metadata_stats['total_groups_created'] / metadata_stats['total_regions_processed']
                    if metadata_stats['total_regions_processed'] > 0 else 0
                ),
                'currency': 'USD',
                'pricing_standardization': 'Smart conversion applied: per-million to per-thousand when needed, unit extraction from descriptions',
                'structure': 'provider > model > region > pricing_groups > dimensions',
                'group_types_available': metadata_stats['group_types_seen']
            },
            'providers': aggregated
        }

        # Write to S3
        if not dry_run:
            write_to_s3(s3_client, s3_bucket, output_key, output_data)
        else:
            logger.info(f"Dry run - would write to s3://{s3_bucket}/{output_key}")

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            'status': 'SUCCESS',
            's3Key': output_key,
            'providersCount': providers_count,
            'totalPricingEntries': metadata_stats['total_entries'],
            'durationMs': duration_ms
        }

    except Exception as e:
        logger.error(f"Failed to aggregate pricing: {e}", exc_info=True)
        return {
            'status': 'FAILED',
            'errorType': type(e).__name__,
            'errorMessage': str(e)
        }
