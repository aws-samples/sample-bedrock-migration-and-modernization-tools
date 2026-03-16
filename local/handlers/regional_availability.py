"""
Regional Availability - Local Handler

Discovers model availability across all AWS regions using Bedrock API
and supplemented by pricing data.

This is a standalone version of backend/lambdas/regional-availability/handler.py
"""

import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config

RETRY_CONFIG = Config(
    retries={'max_attempts': 3, 'mode': 'adaptive'},
    connect_timeout=5,
    read_timeout=30,
)


def discover_via_api(regions: list, session: boto3.Session = None) -> tuple:
    """
    Call ListFoundationModels across all regions in parallel.

    Returns:
        model_availability: {model_id: set(regions)}
        region_stats: {region: {models_found, error}}
    """
    model_availability = defaultdict(set)
    region_stats = {}

    def query_region(region):
        try:
            if session:
                client = session.client('bedrock', region_name=region, config=RETRY_CONFIG)
            else:
                client = boto3.client('bedrock', region_name=region, config=RETRY_CONFIG)
            response = client.list_foundation_models()
            models = response.get('modelSummaries', [])
            model_ids = [m['modelId'] for m in models if 'modelId' in m]
            return region, model_ids, None
        except Exception as e:
            return region, [], str(e)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(query_region, r): r for r in regions}
        for future in as_completed(futures):
            region, model_ids, error = future.result()
            region_stats[region] = {
                'models_found': len(model_ids),
                'error': error,
            }
            for mid in model_ids:
                model_availability[mid].add(region)

    return model_availability, region_stats


def extract_from_pricing(pricing_data: dict) -> dict:
    """Extract regional availability from pricing data."""
    model_availability = defaultdict(set)

    providers_data = pricing_data.get('providers', {})
    for provider_name, provider_models in providers_data.items():
        if not isinstance(provider_models, dict):
            continue
        for model_id, model_data in provider_models.items():
            if not isinstance(model_data, dict):
                continue
            model_regions = model_data.get('regions', {})
            for region in model_regions.keys():
                model_availability[model_id].add(region)

    return model_availability


def merge_pricing_into_api(api_availability: dict, pricing_availability: dict) -> dict:
    """
    Merge pricing regions into API-discovered models using fuzzy matching.

    This handles cases where API returns model ID like "anthropic.claude-sonnet-4-20250514-v1:0"
    but pricing uses "anthropic.claude-sonnet-4".
    """
    for api_model_id in list(api_availability.keys()):
        api_id_lower = api_model_id.lower()
        # Remove version suffix for matching
        api_base = re.sub(r'-\d{8}-v\d+.*$', '', api_id_lower.split(':')[0])

        for pricing_key, pricing_regions in pricing_availability.items():
            pricing_key_lower = pricing_key.lower()
            # Check if pricing key matches this API model
            if (pricing_key_lower in api_id_lower or
                api_base == pricing_key_lower or
                api_base.startswith(pricing_key_lower + '-') or
                pricing_key_lower.startswith(api_base)):
                # Merge pricing regions into this API model
                api_availability[api_model_id].update(pricing_regions)

    return api_availability


def compute_regional_availability(
    regions: list,
    pricing_data: dict,
    session: boto3.Session = None
) -> dict:
    """
    Compute regional availability combining API discovery and pricing data.

    Args:
        regions: List of regions to query
        pricing_data: Pricing data dictionary
        session: Optional boto3 session

    Returns:
        {
            'model_availability': {model_id: [regions]},
            'regions': {region: {stats}}
        }
    """
    # Source 1: API discovery
    api_availability, region_stats = discover_via_api(regions, session)
    api_models = len(api_availability)

    # Source 2: Pricing data
    pricing_availability = extract_from_pricing(pricing_data)

    # Merge pricing into API-discovered models
    api_availability = merge_pricing_into_api(api_availability, pricing_availability)

    # Add pricing-only models
    for pricing_key, pricing_regions in pricing_availability.items():
        if pricing_key not in api_availability:
            api_availability[pricing_key].update(pricing_regions)

    # Build region summary
    regions_summary = defaultdict(lambda: {
        'bedrock_available': True,
        'models_in_region': 0,
        'providers': set(),
    })

    for model_id, regs in api_availability.items():
        provider = model_id.split('.')[0].capitalize() if '.' in model_id else 'Unknown'
        for region in regs:
            regions_summary[region]['models_in_region'] += 1
            regions_summary[region]['providers'].add(provider)

    result_regions = {}
    for region, data in regions_summary.items():
        result_regions[region] = {
            'bedrock_available': data['bedrock_available'],
            'models_in_region': data['models_in_region'],
            'providers': sorted(list(data['providers'])),
            'model_count': data['models_in_region'],
        }

    return {
        'model_availability': {mid: sorted(list(regs)) for mid, regs in api_availability.items()},
        'regions': result_regions,
        'api_models_discovered': api_models,
        'total_models': len(api_availability),
    }
