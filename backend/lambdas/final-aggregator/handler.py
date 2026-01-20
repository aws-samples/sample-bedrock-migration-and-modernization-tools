"""
Final Aggregator Lambda

Merges all collected data into the final comprehensive JSON outputs.
Works with the correct snake_case schema from upstream Lambdas.
"""

import json
import logging
import os
import re
import time
from typing import Any

import boto3
from botocore.config import Config

logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

RETRY_CONFIG = Config(
    retries={'max_attempts': 3, 'mode': 'adaptive'},
    connect_timeout=10,
    read_timeout=30
)


def get_s3_client():
    return boto3.client('s3', config=RETRY_CONFIG)


def read_from_s3(s3_client: Any, bucket: str, key: str) -> dict:
    """Read JSON data from S3."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        logger.warning(f"Failed to read {key}: {e}")
        return {}


def write_to_s3(s3_client: Any, bucket: str, key: str, data: dict) -> None:
    """Write JSON data to S3."""
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2, default=str),
        ContentType='application/json'
    )
    logger.info(f"Written to s3://{bucket}/{key}")


def aggregate_quotas(quota_results: list[dict], s3_client: Any, bucket: str) -> dict:
    """Aggregate quotas from all regions."""
    quotas_by_region = {}

    for item in quota_results:
        nested_result = item.get('result', {})
        status = item.get('status') or nested_result.get('status')
        s3_key = item.get('s3Key') or nested_result.get('s3Key')
        region = item.get('region')

        if status == 'SUCCESS' and s3_key:
            data = read_from_s3(s3_client, bucket, s3_key)
            quotas_by_region[region] = data.get('quotas', [])

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
            data = read_from_s3(s3_client, bucket, s3_key)
            # Handle both snake_case and camelCase from feature extractor
            profiles_by_region[region] = data.get('inference_profiles', data.get('inferenceProfiles', []))

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


def build_cross_region_inference(model_id: str, features_by_region: dict) -> dict:
    """Build cross-region inference data for a model."""
    profiles = []
    source_regions = set()

    for region, region_profiles in features_by_region.items():
        for profile in region_profiles:
            profile_models = profile.get('models', [])
            for pm in profile_models:
                # Handle both snake_case and camelCase model ARN
                model_arn = pm.get('model_arn', pm.get('modelArn', ''))
                if model_id in model_arn:
                    profiles.append({
                        'inference_profile_id': profile.get('inference_profile_id', profile.get('inferenceProfileId')),
                        'inference_profile_name': profile.get('inference_profile_name', profile.get('inferenceProfileName')),
                        'region': region,
                        'type': profile.get('type'),
                        'status': profile.get('status', 'ACTIVE')
                    })
                    source_regions.add(region)

    return {
        'supported': len(profiles) > 0,
        'profiles_count': len(profiles),
        'source_regions': sorted(list(source_regions)),
        'profiles': profiles
    }


def build_model_quotas(model_id: str, model_name: str, quotas_by_region: dict) -> dict:
    """Build model-specific quotas by region."""
    model_quotas = {}
    model_id_lower = model_id.lower()
    model_name_lower = model_name.lower()

    for region, quotas in quotas_by_region.items():
        region_quotas = []
        for quota in quotas:
            quota_name = quota.get('quota_name', quota.get('QuotaName', '')).lower()
            # Check if quota is related to this model
            if model_id_lower in quota_name or model_name_lower in quota_name:
                region_quotas.append({
                    'quota_code': quota.get('quota_code', quota.get('QuotaCode', '')),
                    'quota_name': quota.get('quota_name', quota.get('QuotaName', '')),
                    'quota_arn': quota.get('quota_arn', quota.get('QuotaArn', '')),
                    'description': quota.get('description', quota.get('Description', '')),
                    'quota_applied_at_level': quota.get('quota_applied_at_level', 'ACCOUNT'),
                    'value': quota.get('value', quota.get('Value', 0)),
                    'unit': quota.get('unit', quota.get('Unit', 'None')),
                    'adjustable': quota.get('adjustable', quota.get('Adjustable', False)),
                    'global_quota': quota.get('global_quota', quota.get('GlobalQuota', False)),
                    'usage_metric': quota.get('usage_metric', quota.get('UsageMetric', {})),
                    'period': quota.get('period', quota.get('Period', {}))
                })
        if region_quotas:
            model_quotas[region] = region_quotas

    return model_quotas


def get_consumption_options(inference_types: list) -> list:
    """Convert inference types to consumption options."""
    options = []
    type_mapping = {
        'ON_DEMAND': 'on_demand',
        'PROVISIONED': 'provisioned_throughput',
        'INFERENCE_PROFILE': 'cross_region_inference'
    }
    for inf_type in inference_types:
        if inf_type in type_mapping:
            options.append(type_mapping[inf_type])
    return options if options else ['on_demand']


def check_batch_inference(model_id: str, pricing_data: dict) -> dict:
    """Check if batch inference is supported based on pricing data."""
    supported_regions = []

    # Look through pricing data for batch entries
    # Handle both nested provider structure and flat model structure
    providers = pricing_data.get('providers', {})
    for key, data in providers.items():
        # Check for model_id -> regions structure (new schema)
        if isinstance(data, dict) and 'regions' in data:
            if model_id.lower() in key.lower():
                for region, region_data in data.get('regions', {}).items():
                    pricing_groups = region_data.get('pricing_groups', {})
                    if 'Batch' in pricing_groups:
                        if region not in supported_regions:
                            supported_regions.append(region)
        # Check for provider -> models structure (old schema)
        elif isinstance(data, dict) and 'models' in data:
            for mid, model_data in data.get('models', {}).items():
                if model_id.lower() in mid.lower():
                    for region, region_data in model_data.get('regions', {}).items():
                        pricing_groups = region_data.get('pricing_groups', {})
                        if 'Batch' in pricing_groups:
                            if region not in supported_regions:
                                supported_regions.append(region)

    return {
        'supported': len(supported_regions) > 0,
        'supported_regions': sorted(supported_regions),
        'coverage_percentage': 0.0,
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
    # Use token_specs first, fall back to enriched model's converse_data or model's existing data
    existing_converse = enriched_model.get('converse_data', model.get('converse_data', {}))

    context_window = token_specs.get('context_window')
    max_output = token_specs.get('max_output_tokens')
    source = token_specs.get('source')

    # Fall back to existing converse_data values if token_specs doesn't have them
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

    # Build cross-region inference
    cross_region = build_cross_region_inference(model_id, features_by_region)

    # Build model quotas (using snake_case model_name)
    model_quotas = build_model_quotas(
        model_id,
        model.get('model_name', ''),
        quotas_by_region
    )

    # Check batch inference support
    batch_inference = check_batch_inference(model_id, pricing_data)

    # Get model pricing from upstream (already in snake_case)
    # Build flat structure with pricing_file_reference to match frontend schema
    model_pricing_data = model.get('model_pricing', {})
    has_pricing = model_pricing_data.get('is_pricing_available', model.get('has_pricing', False))
    pricing_ref_id = model_pricing_data.get('pricing_reference_id', '')

    # Extract provider and model_key from pricing_reference_id
    # Format: "provider.model-key" -> provider="Provider", model_key="provider.model-key"
    provider_name = model.get('model_provider', '')
    model_key = pricing_ref_id if pricing_ref_id else model_id

    model_pricing = {
        'is_pricing_available': has_pricing,
        'pricing_reference_id': f"{provider_name}.{model_key}" if provider_name else model_key,
        'pricing_file_reference': {
            'provider': provider_name,
            'model_key': model_key,
            'model_name': model.get('model_name', '')
        },
        'pricing_summary': {
            'integration_source': 'amazon-bedrock-pricing-collector',
            'has_pricing_data': has_pricing,
            'integration_timestamp': collection_timestamp,
            'reference_based': True
        }
    }

    # Build documentation links (already in snake_case from enricher)
    documentation_links = {
        'aws_bedrock_guide': doc_links.get('aws_bedrock_guide', 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html'),
        'pricing_guide': doc_links.get('pricing_guide', 'https://aws.amazon.com/bedrock/pricing/')
    }

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
        'model_use_cases': use_cases,
        'languages_supported': model.get('languages_supported', ['English']),
        'consumption_options': get_consumption_options(model.get('inference_types_supported', [])),
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

    Strategy: Try exact match first, then try prefix/substring matching.
    """
    # Try exact match first
    if model_id in model_availability:
        return model_availability[model_id]

    # Normalize model_id for matching (remove version suffix like :0, :18k, etc.)
    base_model_id = model_id.split(':')[0] if ':' in model_id else model_id

    # Try matching without version suffix
    if base_model_id in model_availability:
        return model_availability[base_model_id]

    # Try to find a pricing key that's a prefix of the model_id
    # e.g., "anthropic.claude-3-sonnet" should match "anthropic.claude-3-5-sonnet-20241022-v2:0"
    model_id_lower = model_id.lower()
    for pricing_key, regions in model_availability.items():
        pricing_key_lower = pricing_key.lower()
        # Check if pricing key is contained in model_id (handles claude-3-sonnet matching claude-3-5-sonnet)
        # or if model_id starts with pricing key
        if pricing_key_lower in model_id_lower or model_id_lower.startswith(pricing_key_lower):
            return regions

        # Also check by removing common prefixes/suffixes and comparing core name
        # Extract core model name (e.g., "claude-3-sonnet" from "anthropic.claude-3-sonnet")
        pricing_parts = pricing_key_lower.replace('anthropic.', '').replace('amazon.', '').replace('meta.', '').replace('mistral.', '').replace('cohere.', '').replace('ai21.', '').replace('stability.', '')
        model_parts = model_id_lower.replace('anthropic.', '').replace('amazon.', '').replace('meta.', '').replace('mistral.', '').replace('cohere.', '').replace('ai21.', '').replace('stability.', '')

        # Check if core names overlap significantly
        if pricing_parts and model_parts:
            # Remove date/version suffixes from model_parts for comparison
            model_core = re.sub(r'-\d{8}-v\d+.*$', '', model_parts)
            if pricing_parts == model_core or pricing_parts in model_core or model_core.startswith(pricing_parts):
                return regions

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

    s3_bucket = event['s3Bucket']
    execution_id = event['executionId']
    pricing_s3_key = event.get('pricingS3Key')
    quota_results = event.get('quotaResults', [])
    pricing_linked = event.get('pricingLinked', {})
    regional_availability = event.get('regionalAvailability', {})
    feature_results = event.get('featureResults', [])
    token_specs_result = event.get('tokenSpecs', {})
    enriched_models_result = event.get('enrichedModels', {})
    dry_run = event.get('dryRun', False)

    if ':' in execution_id:
        execution_id = execution_id.split(':')[-1]

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
