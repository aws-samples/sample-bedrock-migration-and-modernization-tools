"""
Pricing Aggregator Lambda

Merges pricing data from all three Bedrock service codes into a unified structure.
Transforms data to match the expected frontend schema with pricing_groups.
"""

import json
import logging
import os
import re
import time
from typing import Any
from collections import defaultdict

import boto3
from botocore.config import Config

logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

RETRY_CONFIG = Config(
    retries={'max_attempts': 3, 'mode': 'adaptive'},
    connect_timeout=10,
    read_timeout=30
)

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


def get_s3_client():
    return boto3.client('s3', config=RETRY_CONFIG)


def read_from_s3(s3_client: Any, bucket: str, key: str) -> dict:
    """Read JSON data from S3."""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response['Body'].read().decode('utf-8'))


def write_to_s3(s3_client: Any, bucket: str, key: str, data: dict) -> None:
    """Write JSON data to S3."""
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2, default=str),
        ContentType='application/json'
    )
    logger.info(f"Written to s3://{bucket}/{key}")


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

    return {
        'model': attributes.get('model', 'Unknown'),
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
        'serviceCode': attributes.get('servicecode', 'AmazonBedrock')
    }


def infer_provider(model_name: str) -> str:
    """Infer the provider from the model name."""
    model_lower = model_name.lower()

    provider_patterns = {
        'Amazon': ['titan', 'nova'],
        'Anthropic': ['claude'],
        'Meta': ['llama'],
        'Mistral AI': ['mistral', 'mixtral'],
        'Cohere': ['cohere', 'command', 'embed'],
        'AI21 Labs': ['ai21', 'jamba', 'jurassic'],
        'Stability AI': ['stable', 'stability', 'sdxl'],
        'Luma': ['luma'],
        'Writer': ['writer', 'palmyra'],
        'NVIDIA': ['nvidia', 'nemotron'],
    }

    for provider, patterns in provider_patterns.items():
        for pattern in patterns:
            if pattern in model_lower:
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

        if model_name == 'Unknown' or region == 'Unknown':
            continue

        # Infer provider
        provider = infer_provider(model_name)

        # Create model ID
        model_id = normalize_model_id(model_name, provider)

        # Determine pricing group
        pricing_group = determine_pricing_group(info['usageType'], info['inferenceType'])
        group_types_seen.add(pricing_group)

        # Get location name
        location = REGION_LOCATIONS.get(region, region)

        # Build pricing entry in expected schema
        pricing_entry = {
            'dimension': info['usageType'],
            'price_per_thousand': info['price'],
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
        models_data[model_id]['regions'][region]['pricing_groups'][pricing_group].append(pricing_entry)
        total_entries += 1

    # Convert to final structure nested by provider: providers -> Provider -> model_id -> data
    # This matches the frontend expected schema
    result = defaultdict(dict)
    total_regions_processed = 0
    total_groups_created = 0

    for model_id, model_data in models_data.items():
        provider = model_data['model_provider']

        model_entry = {
            'model_name': model_data['model_name'],
            'model_provider': provider,
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

    s3_bucket = event['s3Bucket']
    execution_id = event['executionId']
    pricing_results = event['pricingResults']
    dry_run = event.get('dryRun', False)

    # Extract just the execution ID portion if full ARN provided
    if ':' in execution_id:
        execution_id = execution_id.split(':')[-1]

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
