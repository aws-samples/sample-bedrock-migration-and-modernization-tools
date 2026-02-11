"""
Regional Availability Lambda

Discovers model availability across all AWS regions using the Bedrock API
(ListFoundationModels), supplemented by pricing data for additional coverage.
"""

import logging
import os
import time
from typing import Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config

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

RETRY_CONFIG = Config(
    retries={'max_attempts': 3, 'mode': 'adaptive'},
    connect_timeout=5,
    read_timeout=30,
)


def _discover_via_api(regions: list) -> tuple[dict, dict]:
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
            client = boto3.client('bedrock', region_name=region, config=RETRY_CONFIG)
            response = client.list_foundation_models()
            models = response.get('modelSummaries', [])
            model_ids = [m['modelId'] for m in models if 'modelId' in m]
            return region, model_ids, None
        except Exception as e:
            logger.warning(f"Failed to query region {region}: {e}")
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

    successful = sum(1 for s in region_stats.values() if s['error'] is None)
    logger.info(f"API discovery: {len(model_availability)} models across "
                f"{successful}/{len(regions)} successful regions")

    return model_availability, region_stats


def _extract_from_pricing(pricing_data: dict) -> dict:
    """
    Extract regional availability from pricing data.

    Returns: {model_id: set(regions)}
    """
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


def _combine_availability(api_availability: dict, pricing_availability: dict) -> dict:
    """
    Combine API discovery and pricing data into a unified availability map.
    """
    combined = defaultdict(set)

    for model_id, regions in api_availability.items():
        combined[model_id].update(regions)

    for model_id, regions in pricing_availability.items():
        combined[model_id].update(regions)

    # Build region summary
    regions_summary = defaultdict(lambda: {
        'bedrock_available': True,
        'models_in_region': 0,
        'providers': set(),
    })

    for model_id, regions in combined.items():
        provider = model_id.split('.')[0].capitalize() if '.' in model_id else 'Unknown'
        for region in regions:
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

    model_availability = {mid: sorted(list(regs)) for mid, regs in combined.items()}

    return {
        'regions': result_regions,
        'model_availability': model_availability,
    }


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for regional availability computation.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "pricingS3Key": "executions/{id}/merged/pricing.json",
            "regions": ["us-east-1", "us-west-2", ...]
        }

    Output:
        {
            "status": "SUCCESS",
            "s3Key": "executions/{id}/intermediate/regional-availability.json",
            "regionsWithBedrock": 27
        }
    """
    start_time = time.time()

    try:
        validate_required_params(event, ['s3Bucket', 'executionId', 'pricingS3Key'], 'RegionalAvailability')
    except ValidationError as e:
        return {
            'status': 'FAILED',
            'errorType': 'ValidationError',
            'errorMessage': str(e)
        }

    s3_bucket = event['s3Bucket']
    execution_id = parse_execution_id(event['executionId'])
    pricing_s3_key = event['pricingS3Key']
    regions = event.get('regions', [])
    dry_run = event.get('dryRun', False)

    output_key = f"executions/{execution_id}/intermediate/regional-availability.json"

    logger.info(f"Computing regional availability (API regions: {len(regions)})")

    try:
        s3_client = get_s3_client()

        if not dry_run:
            # Source 1: API discovery across all regions
            api_availability = {}
            region_stats = {}
            if regions:
                api_availability, region_stats = _discover_via_api(regions)

            # Source 2: Pricing data
            pricing_data = read_from_s3(s3_client, s3_bucket, pricing_s3_key)
            pricing_availability = _extract_from_pricing(pricing_data)

            # Combine sources
            availability = _combine_availability(api_availability, pricing_availability)

            output_data = {
                'metadata': {
                    'regions_with_bedrock': len(availability['regions']),
                    'total_models_tracked': len(availability['model_availability']),
                    'api_regions_queried': len(regions),
                    'api_models_discovered': len(api_availability),
                    'pricing_models_discovered': len(pricing_availability),
                    'collection_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'discovery_method': 'api_and_pricing',
                },
                'regions': availability['regions'],
                'model_availability': availability['model_availability'],
            }

            write_to_s3(s3_client, s3_bucket, output_key, output_data)
            regions_count = len(availability['regions'])
        else:
            logger.info("Dry run - skipping processing")
            regions_count = 0

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            'status': 'SUCCESS',
            's3Key': output_key,
            'regionsWithBedrock': regions_count,
            'durationMs': duration_ms,
        }

    except Exception as e:
        logger.error(f"Failed to compute availability: {e}", exc_info=True)
        return {
            'status': 'FAILED',
            'errorType': type(e).__name__,
            'errorMessage': str(e)
        }
