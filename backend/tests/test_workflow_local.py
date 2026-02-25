#!/usr/bin/env python3
"""
Local End-to-End Workflow Test

Simulates the Step Functions workflow by running each Lambda in sequence,
using local files instead of S3.

This test fully replicates the state machine behavior including:
- Wave 1: Pricing, model extraction, quotas (parallel)
- Wave 2: Regional availability (API-based), feature collection, mantle collection,
          token specs, pricing linker (parallel)
- Wave 3: Final aggregation

Usage:
    cd backend/tests
    python test_workflow_local.py [--quick]

Options:
    --quick    Run with reduced data (fewer regions, limited batches)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Add lambdas and shared layer to path
LAMBDAS_DIR = Path(__file__).parent.parent / 'lambdas'
SHARED_LAYER_DIR = Path(__file__).parent.parent / 'layers' / 'common' / 'python'
sys.path.insert(0, str(LAMBDAS_DIR))
sys.path.insert(0, str(SHARED_LAYER_DIR))

# Output directory for test data
OUTPUT_DIR = Path(__file__).parent / 'workflow_output'

# All known Bedrock regions (used for regional availability and feature collection)
ALL_BEDROCK_REGIONS = [
    'us-east-1', 'us-west-2', 'us-east-2',
    'eu-west-1', 'eu-west-2', 'eu-west-3', 'eu-central-1', 'eu-north-1',
    'ap-southeast-1', 'ap-southeast-2', 'ap-southeast-3',
    'ap-northeast-1', 'ap-northeast-2', 'ap-south-1',
    'ca-central-1', 'sa-east-1',
    'me-south-1', 'me-central-1',
    'il-central-1',
    'ap-southeast-4', 'ap-southeast-5',
]


def setup_output_dir():
    """Create output directory structure."""
    dirs = [
        OUTPUT_DIR / 'pricing',
        OUTPUT_DIR / 'models',
        OUTPUT_DIR / 'quotas',
        OUTPUT_DIR / 'features',
        OUTPUT_DIR / 'mantle',
        OUTPUT_DIR / 'merged',
        OUTPUT_DIR / 'intermediate',
        OUTPUT_DIR / 'enriched',
        OUTPUT_DIR / 'final',
        OUTPUT_DIR / 'latest',
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def save_json(path: Path, data: dict):
    """Save JSON to file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path.name} ({path.stat().st_size / 1024:.1f} KB)")


def load_json(path: Path) -> dict:
    """Load JSON from file."""
    with open(path) as f:
        return json.load(f)


def run_pricing_collectors(output_dir: Path, quick: bool = False):
    """Wave 1: Run pricing collectors in parallel."""
    print("\n" + "="*60)
    print("WAVE 1A: Pricing Collection")
    print("="*60)

    from importlib import import_module
    sys.path.insert(0, str(LAMBDAS_DIR / 'pricing-collector'))

    # Import handler
    spec = import_module('pricing-collector.handler')

    service_codes = ['AmazonBedrock', 'AmazonBedrockService', 'AmazonBedrockFoundationModels']
    results = []

    def local_handler(event, context=None):
        service_code = event['serviceCode']
        print(f"\n  Collecting pricing for: {service_code}")

        # Run the actual collection logic
        start = time.time()
        pricing_client = spec.get_pricing_client()

        # Limit batches in quick mode
        max_batches = 5 if quick else 100
        products = []
        next_token = None
        batch_count = 0

        while batch_count < max_batches:
            params = {'ServiceCode': service_code, 'MaxResults': 100, 'FormatVersion': 'aws_v1'}
            if next_token:
                params['NextToken'] = next_token

            response = pricing_client.get_products(**params)
            for item in response.get('PriceList', []):
                products.append(json.loads(item) if isinstance(item, str) else item)

            batch_count += 1
            next_token = response.get('NextToken')
            if not next_token:
                break

        duration = int((time.time() - start) * 1000)

        # Save locally
        output_data = {
            'metadata': {'serviceCode': service_code, 'recordCount': len(products)},
            'products': products
        }
        output_path = output_dir / 'pricing' / f'{service_code}.json'
        save_json(output_path, output_data)

        return {
            'status': 'SUCCESS',
            'serviceCode': service_code,
            's3Key': str(output_path),
            'recordCount': len(products),
            'durationMs': duration
        }

    # Run in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(local_handler, {'serviceCode': sc}): sc
            for sc in service_codes
        }
        for future in as_completed(futures):
            results.append(future.result())

    return results


def run_model_extractors(output_dir: Path, quick: bool = False):
    """Wave 1: Run model extractors in parallel."""
    print("\n" + "="*60)
    print("WAVE 1B: Model Extraction")
    print("="*60)

    sys.path.insert(0, str(LAMBDAS_DIR / 'model-extractor'))
    from importlib import import_module
    spec = import_module('model-extractor.handler')

    regions = ['us-east-1', 'us-west-2']
    results = []

    for region in regions:
        print(f"\n  Extracting models from: {region}")
        start = time.time()

        bedrock_client = spec.get_bedrock_client(region)
        models = spec.extract_models(bedrock_client, region)

        duration = int((time.time() - start) * 1000)

        output_data = {
            'metadata': {'region': region, 'modelCount': len(models)},
            'models': models
        }
        output_path = output_dir / 'models' / f'{region}.json'
        save_json(output_path, output_data)

        results.append({
            'status': 'SUCCESS',
            'region': region,
            's3Key': str(output_path),
            'modelCount': len(models),
            'durationMs': duration
        })

    return results


def run_quota_collectors(output_dir: Path, quick: bool = False):
    """Wave 1: Run quota collectors in parallel."""
    print("\n" + "="*60)
    print("WAVE 1C: Quota Collection")
    print("="*60)

    sys.path.insert(0, str(LAMBDAS_DIR / 'quota-collector'))
    from importlib import import_module
    spec = import_module('quota-collector.handler')

    # Use fewer regions in quick mode
    if quick:
        regions = ['us-east-1', 'us-west-2']
    else:
        regions = [
            'us-east-1', 'us-west-2', 'us-east-2',
            'eu-west-1', 'eu-west-2', 'eu-west-3', 'eu-central-1', 'eu-north-1',
            'ap-southeast-1', 'ap-southeast-2', 'ap-northeast-1', 'ap-northeast-2', 'ap-south-1',
            'ca-central-1', 'sa-east-1', 'ap-southeast-3'
        ]

    results = []

    def collect_region(region):
        print(f"  Collecting quotas from: {region}")
        start = time.time()

        try:
            quotas_client = spec.get_quotas_client(region)
            quotas = spec.collect_quotas(quotas_client, region)
            duration = int((time.time() - start) * 1000)

            if quotas:
                output_data = {
                    'metadata': {'region': region, 'quotaCount': len(quotas)},
                    'quotas': quotas
                }
                output_path = output_dir / 'quotas' / f'{region}.json'
                save_json(output_path, output_data)

                return {
                    'status': 'SUCCESS',
                    'region': region,
                    's3Key': str(output_path),
                    'quotaCount': len(quotas),
                    'durationMs': duration
                }
            else:
                print(f"  - Skipped {region} (no quotas or region unavailable)")
                return {
                    'status': 'SKIPPED',
                    'region': region,
                    'quotaCount': 0,
                    'durationMs': duration
                }
        except Exception as e:
            print(f"  - Failed {region}: {e}")
            return {'status': 'FAILED', 'region': region, 'error': str(e)}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(collect_region, r) for r in regions]
        for future in as_completed(futures):
            results.append(future.result())

    return results


def run_pricing_aggregator(output_dir: Path, pricing_results: list):
    """Aggregate pricing data."""
    print("\n" + "="*60)
    print("WAVE 1D: Pricing Aggregation")
    print("="*60)

    sys.path.insert(0, str(LAMBDAS_DIR / 'pricing-aggregator'))
    from importlib import import_module
    spec = import_module('pricing-aggregator.handler')

    # Load all pricing files
    all_products = []
    for result in pricing_results:
        if result['status'] == 'SUCCESS':
            data = load_json(Path(result['s3Key']))
            all_products.extend(data.get('products', []))
            print(f"  Loaded {len(data.get('products', []))} from {result['serviceCode']}")

    print(f"  Total products: {len(all_products)}")

    # Aggregate - returns (providers_dict, metadata_stats)
    providers_dict, metadata_stats = spec.aggregate_pricing(all_products)

    output_data = {
        'metadata': {
            'providersCount': len(providers_dict),
            'totalProducts': len(all_products),
            **metadata_stats
        },
        'providers': providers_dict
    }
    output_path = output_dir / 'merged' / 'pricing.json'
    save_json(output_path, output_data)

    return {'status': 'SUCCESS', 's3Key': str(output_path)}


def run_model_merger(output_dir: Path, model_results: list):
    """Merge models from all regions."""
    print("\n" + "="*60)
    print("WAVE 1E: Model Merging")
    print("="*60)

    sys.path.insert(0, str(LAMBDAS_DIR / 'model-merger'))
    from importlib import import_module
    spec = import_module('model-merger.handler')

    # Load all model files
    all_models = []
    for result in model_results:
        if result['status'] == 'SUCCESS':
            data = load_json(Path(result['s3Key']))
            all_models.extend(data.get('models', []))
            print(f"  Loaded {len(data.get('models', []))} from {result['region']}")

    print(f"  Total models before dedup: {len(all_models)}")

    # Merge
    providers = spec.merge_models(all_models)
    total_models = sum(len(p['models']) for p in providers.values())
    print(f"  Total models after dedup: {total_models}")

    output_data = {
        'metadata': {'totalModels': total_models, 'providersCount': len(providers)},
        'providers': providers
    }
    output_path = output_dir / 'merged' / 'models.json'
    save_json(output_path, output_data)

    return {'status': 'SUCCESS', 's3Key': str(output_path), 'totalModels': total_models}


def run_token_specs_collector(output_dir: Path, models_path: Path):
    """Collect token specs from LiteLLM."""
    print("\n" + "="*60)
    print("WAVE 2A: Token Specs Collection")
    print("="*60)

    sys.path.insert(0, str(LAMBDAS_DIR / 'token-specs-collector'))
    from importlib import import_module
    spec = import_module('token-specs-collector.handler')

    print("  Fetching from LiteLLM...")
    litellm_data = spec.fetch_litellm_data()
    bedrock_models = spec.filter_bedrock_models(litellm_data)
    print(f"  Found {len(bedrock_models)} Bedrock models in LiteLLM")

    # Match with our models
    models_data = load_json(models_path)
    token_specs = spec.match_token_specs(models_data, bedrock_models)
    print(f"  Matched {len(token_specs)} models")

    output_data = {
        'metadata': {'modelsWithSpecs': len(token_specs)},
        'tokenSpecs': token_specs
    }
    output_path = output_dir / 'intermediate' / 'token-specs.json'
    save_json(output_path, output_data)

    return {'status': 'SUCCESS', 's3Key': str(output_path)}


def run_regional_availability(output_dir: Path, regions: list, quick: bool = False):
    """
    Compute regional availability using the actual Bedrock API.

    This replicates the regional-availability Lambda behavior:
    - Queries ListFoundationModels with byInferenceType="ON_DEMAND" filter
    - Queries ListFoundationModels with byInferenceType="PROVISIONED" filter
    - Returns accurate on-demand availability (not from pricing data)
    """
    print("\n" + "="*60)
    print("WAVE 2B: Regional Availability (API-based)")
    print("="*60)

    import boto3
    from botocore.config import Config

    retry_config = Config(
        retries={'max_attempts': 3, 'mode': 'adaptive'},
        connect_timeout=5,
        read_timeout=30,
    )

    on_demand_availability = defaultdict(set)
    provisioned_availability = defaultdict(set)
    region_stats = {}

    def query_region(region: str):
        """Query a single region for ON_DEMAND and PROVISIONED models."""
        try:
            client = boto3.client('bedrock', region_name=region, config=retry_config)

            # ON_DEMAND: models that can be invoked directly via Converse / InvokeModel
            od_response = client.list_foundation_models(byInferenceType='ON_DEMAND')
            od_models = [
                m['modelId']
                for m in od_response.get('modelSummaries', [])
                if 'modelId' in m
            ]

            # PROVISIONED: models available for Provisioned Throughput
            prov_response = client.list_foundation_models(byInferenceType='PROVISIONED')
            prov_models = [
                m['modelId']
                for m in prov_response.get('modelSummaries', [])
                if 'modelId' in m
            ]

            return region, od_models, prov_models, None
        except Exception as e:
            print(f"  - Failed {region}: {e}")
            return region, [], [], str(e)

    print(f"  Querying {len(regions)} regions with byInferenceType=ON_DEMAND filter...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(query_region, r): r for r in regions}
        for future in as_completed(futures):
            region, od_models, prov_models, error = future.result()
            region_stats[region] = {
                'on_demand_count': len(od_models),
                'provisioned_count': len(prov_models),
                'error': error,
            }
            for mid in od_models:
                on_demand_availability[mid].add(region)
            for mid in prov_models:
                provisioned_availability[mid].add(region)

            if not error:
                print(f"  {region}: {len(od_models)} on-demand, {len(prov_models)} provisioned")

    successful = sum(1 for s in region_stats.values() if s['error'] is None)
    print(f"\n  API discovery: {len(on_demand_availability)} on-demand models, "
          f"{len(provisioned_availability)} provisioned models across "
          f"{successful}/{len(regions)} successful regions")

    # Build region summary
    regions_summary = {}
    for model_id, regs in on_demand_availability.items():
        provider = model_id.split('.')[0].capitalize() if '.' in model_id else 'Unknown'
        for region in regs:
            if region not in regions_summary:
                regions_summary[region] = {
                    'bedrock_available': True,
                    'models_in_region': 0,
                    'providers': set(),
                }
            regions_summary[region]['models_in_region'] += 1
            regions_summary[region]['providers'].add(provider)

    # Convert sets to sorted lists
    for region in regions_summary:
        regions_summary[region]['providers'] = sorted(list(regions_summary[region]['providers']))
        regions_summary[region]['model_count'] = regions_summary[region]['models_in_region']

    model_availability = {
        mid: sorted(list(regs)) for mid, regs in on_demand_availability.items()
    }
    prov_availability = {
        mid: sorted(list(regs)) for mid, regs in provisioned_availability.items()
    }

    output_data = {
        'metadata': {
            'regions_with_bedrock': len(regions_summary),
            'total_models_tracked': len(model_availability),
            'total_provisioned_models': len(prov_availability),
            'api_regions_queried': len(regions),
            'collection_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'discovery_method': 'api_on_demand_filtered',
        },
        'region_summary': regions_summary,
        'model_availability': model_availability,
        'provisioned_availability': prov_availability,
    }
    output_path = output_dir / 'intermediate' / 'regional-availability.json'
    save_json(output_path, output_data)

    return {'status': 'SUCCESS', 's3Key': str(output_path), 'regionsWithBedrock': len(regions_summary)}


def run_feature_collectors(output_dir: Path, regions: list, quick: bool = False):
    """
    Wave 2: Run feature collectors (inference profiles) in parallel.

    This replicates the feature-collector Lambda behavior:
    - Queries ListInferenceProfiles in each region
    - Returns CRIS (Cross-Region Inference) profile data
    """
    print("\n" + "="*60)
    print("WAVE 2C: Feature Collection (Inference Profiles)")
    print("="*60)

    sys.path.insert(0, str(LAMBDAS_DIR / 'feature-collector'))
    from importlib import import_module
    spec = import_module('feature-collector.handler')

    results = []

    def collect_region(region):
        print(f"  Collecting inference profiles from: {region}")
        start = time.time()

        try:
            bedrock_client = spec.get_bedrock_client(region)
            profiles = spec.collect_inference_profiles(bedrock_client, region)
            duration = int((time.time() - start) * 1000)

            output_data = {
                'metadata': {
                    'region': region,
                    'inferenceProfileCount': len(profiles),
                    'collectionTimestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                },
                'inferenceProfiles': profiles
            }
            output_path = output_dir / 'features' / f'{region}.json'
            save_json(output_path, output_data)

            return {
                'status': 'SUCCESS',
                'region': region,
                's3Key': str(output_path),
                'inferenceProfileCount': len(profiles),
                'durationMs': duration
            }
        except Exception as e:
            print(f"  - Failed {region}: {e}")
            return {'status': 'FAILED', 'region': region, 'error': str(e)}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(collect_region, r) for r in regions]
        for future in as_completed(futures):
            results.append(future.result())

    successful = sum(1 for r in results if r['status'] == 'SUCCESS')
    total_profiles = sum(r.get('inferenceProfileCount', 0) for r in results if r['status'] == 'SUCCESS')
    print(f"\n  Collected {total_profiles} inference profiles from {successful}/{len(regions)} regions")

    return results


def run_mantle_collectors(output_dir: Path, regions: list, quick: bool = False):
    """
    Wave 2: Run Mantle collectors in parallel.

    This replicates the mantle-collector Lambda behavior:
    - Queries Mantle /v1/models endpoint in each region
    - Probes Responses API support for each model
    - Returns Mantle model data with API support flags
    """
    print("\n" + "="*60)
    print("WAVE 2D: Mantle Collection")
    print("="*60)

    sys.path.insert(0, str(LAMBDAS_DIR / 'mantle-collector'))
    from importlib import import_module
    spec = import_module('mantle-collector.handler')

    results = []

    def collect_region(region):
        print(f"  Collecting Mantle models from: {region}")
        start = time.time()

        try:
            models = spec.call_mantle_endpoint(region)

            # Probe Responses API support for each model
            model_ids = [m['model_id'] for m in models]
            probe_start = time.time()
            responses_support = spec.probe_all_responses_support(model_ids, region)
            probe_duration_ms = int((time.time() - probe_start) * 1000)

            supported_count = sum(1 for v in responses_support.values() if v)

            # Enrich each model dict with Responses API support flag
            for model in models:
                model['supports_responses_api'] = responses_support.get(model['model_id'], False)

            duration = int((time.time() - start) * 1000)

            output_data = {
                'metadata': {
                    'region': region,
                    'mantle_model_count': len(models),
                    'collection_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'endpoint': f'bedrock-mantle.{region}.api.aws',
                    'responses_api_probe': {
                        'probed': len(model_ids),
                        'supported': supported_count,
                        'duration_ms': probe_duration_ms,
                    },
                },
                'mantle_models': models,
            }
            output_path = output_dir / 'mantle' / f'{region}.json'
            save_json(output_path, output_data)

            return {
                'status': 'SUCCESS',
                'region': region,
                's3Key': str(output_path),
                'mantleModelCount': len(models),
                'durationMs': duration
            }
        except Exception as e:
            # Mantle may not be available in all regions
            print(f"  - Mantle not available in {region}: {type(e).__name__}")
            return {'status': 'FAILED', 'region': region, 'error': str(e)}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(collect_region, r) for r in regions]
        for future in as_completed(futures):
            results.append(future.result())

    successful = sum(1 for r in results if r['status'] == 'SUCCESS')
    total_models = sum(r.get('mantleModelCount', 0) for r in results if r['status'] == 'SUCCESS')
    print(f"\n  Collected {total_models} Mantle models from {successful}/{len(regions)} regions")

    return results


def run_pricing_linker(output_dir: Path, pricing_path: Path, models_path: Path):
    """Link pricing to models."""
    print("\n" + "="*60)
    print("WAVE 2E: Pricing Linker")
    print("="*60)

    sys.path.insert(0, str(LAMBDAS_DIR / 'pricing-linker'))
    from importlib import import_module
    spec = import_module('pricing-linker.handler')

    pricing_data = load_json(pricing_path)
    models_data = load_json(models_path)

    result = spec.link_pricing_to_models(models_data, pricing_data)

    print(f"  Models with pricing: {result['models_with_pricing']}")
    print(f"  Models without pricing: {result['models_without_pricing']}")

    output_data = {
        'metadata': {
            'models_with_pricing': result['models_with_pricing'],
            'models_without_pricing': result['models_without_pricing']
        },
        'providers': result['providers']
    }
    output_path = output_dir / 'intermediate' / 'models-with-pricing.json'
    save_json(output_path, output_data)

    return {'status': 'SUCCESS', 's3Key': str(output_path)}


def run_final_aggregator(output_dir: Path, models_with_pricing_path: Path,
                         token_specs_path: Path, pricing_path: Path, quota_results: list,
                         availability_path: Path, feature_results: list, mantle_results: list):
    """Run final aggregation to produce the complete output."""
    print("\n" + "="*60)
    print("WAVE 3: Final Aggregation")
    print("="*60)

    sys.path.insert(0, str(LAMBDAS_DIR / 'final-aggregator'))
    from importlib import import_module
    spec = import_module('final-aggregator.handler')

    # Load data
    models_with_pricing = load_json(models_with_pricing_path)
    token_specs = load_json(token_specs_path)
    pricing_data = load_json(pricing_path)

    # Build quotas by region
    quotas_by_region = {}
    for result in quota_results:
        if result.get('status') == 'SUCCESS' and result.get('s3Key'):
            region = result.get('region')
            data = load_json(Path(result['s3Key']))
            quotas_by_region[region] = data.get('quotas', [])

    # Load regional availability (API-based)
    if availability_path and availability_path.exists():
        avail_data = load_json(availability_path)
        regional_availability = {
            'model_availability': avail_data.get('model_availability', {}),
            'provisioned_availability': avail_data.get('provisioned_availability', {})
        }
        print(f"  Loaded regional availability: {len(regional_availability['model_availability'])} models")
    else:
        regional_availability = {'model_availability': {}, 'provisioned_availability': {}}

    # Build features by region from feature collector results
    features_by_region = {}
    for result in feature_results:
        if result.get('status') == 'SUCCESS' and result.get('s3Key'):
            region = result.get('region')
            data = load_json(Path(result['s3Key']))
            features_by_region[region] = data.get('inferenceProfiles', [])
    print(f"  Loaded features from {len(features_by_region)} regions")

    # Build mantle by model from mantle collector results
    mantle_by_model = {}
    for result in mantle_results:
        if result.get('status') == 'SUCCESS' and result.get('s3Key'):
            data = load_json(Path(result['s3Key']))
            region = result.get('region')
            for model in data.get('mantle_models', []):
                model_id = model.get('model_id')
                if model_id:
                    if model_id not in mantle_by_model:
                        mantle_by_model[model_id] = {
                            'supported': True,
                            'mantle_regions': [],
                            'supports_responses_api': model.get('supports_responses_api', False),
                        }
                    mantle_by_model[model_id]['mantle_regions'].append(region)
                    # Update responses API support if any region supports it
                    if model.get('supports_responses_api'):
                        mantle_by_model[model_id]['supports_responses_api'] = True
    print(f"  Loaded Mantle data for {len(mantle_by_model)} models")

    enriched_models = {}
    collection_timestamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

    # Call build_final_models
    final_providers = spec.build_final_models(
        models_with_pricing=models_with_pricing,
        regional_availability=regional_availability,
        token_specs={'token_specs': token_specs.get('tokenSpecs', token_specs.get('token_specs', {}))},
        quotas_by_region=quotas_by_region,
        features_by_region=features_by_region,
        enriched_models=enriched_models,
        pricing_data=pricing_data,
        collection_timestamp=collection_timestamp,
        mantle_by_model=mantle_by_model,
    )

    total_models = sum(len(p['models']) for p in final_providers.values())
    print(f"  Total models in final output: {total_models}")
    print(f"  Providers: {len(final_providers)}")

    # Count models with in_region availability
    models_with_in_region = 0
    models_without_in_region = 0
    for provider_data in final_providers.values():
        for model in provider_data['models'].values():
            if model.get('in_region') and len(model['in_region']) > 0:
                models_with_in_region += 1
            else:
                models_without_in_region += 1
    print(f"  Models with in_region: {models_with_in_region}")
    print(f"  Models without in_region (INFERENCE_PROFILE-only): {models_without_in_region}")

    # Save final models output
    output_data = {
        'metadata': {
            'total_models': total_models,
            'providers_count': len(final_providers),
            'collection_timestamp': collection_timestamp,
        },
        'providers': final_providers
    }
    output_path = output_dir / 'final' / 'bedrock_models.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, output_data)

    # Save pricing output (copy from aggregated pricing)
    pricing_output_path = output_dir / 'final' / 'bedrock_pricing.json'
    save_json(pricing_output_path, pricing_data)

    return {'status': 'SUCCESS', 's3Key': str(output_path), 'totalModels': total_models}


def print_summary(output_dir: Path):
    """Print summary of generated files."""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total_size = 0
    for f in output_dir.rglob('*.json'):
        size = f.stat().st_size
        total_size += size
        rel_path = f.relative_to(output_dir)
        print(f"  {rel_path}: {size / 1024:.1f} KB")

    print(f"\n  Total: {total_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Local workflow test')
    parser.add_argument('--quick', action='store_true', help='Quick mode with reduced data')
    args = parser.parse_args()

    print("="*60)
    print("BEDROCK PROFILER - LOCAL WORKFLOW TEST")
    print("="*60)
    print(f"Mode: {'Quick' if args.quick else 'Full'}")

    start_time = time.time()

    # Setup
    output_dir = setup_output_dir()
    print(f"Output directory: {output_dir}")

    # Determine regions to use
    if args.quick:
        feature_regions = ['us-east-1', 'us-west-2', 'eu-west-1']
    else:
        feature_regions = ALL_BEDROCK_REGIONS

    # ============================================
    # WAVE 1: Parallel Collection
    # ============================================
    pricing_results = run_pricing_collectors(output_dir, args.quick)
    model_results = run_model_extractors(output_dir, args.quick)
    quota_results = run_quota_collectors(output_dir, args.quick)

    # Wave 1: Aggregation
    pricing_agg = run_pricing_aggregator(output_dir, pricing_results)
    models_merged = run_model_merger(output_dir, model_results)

    # ============================================
    # WAVE 2: Enrichment (parallel)
    # ============================================
    pricing_path = Path(pricing_agg['s3Key'])
    models_path = Path(models_merged['s3Key'])

    # Run Wave 2 collectors in parallel
    print("\n" + "="*60)
    print("WAVE 2: Enrichment Processing (parallel)")
    print("="*60)

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all Wave 2 tasks
        future_token_specs = executor.submit(run_token_specs_collector, output_dir, models_path)
        future_availability = executor.submit(run_regional_availability, output_dir, feature_regions, args.quick)
        future_features = executor.submit(run_feature_collectors, output_dir, feature_regions, args.quick)
        future_mantle = executor.submit(run_mantle_collectors, output_dir, feature_regions, args.quick)
        future_pricing_linked = executor.submit(run_pricing_linker, output_dir, pricing_path, models_path)

        # Wait for all to complete
        token_specs = future_token_specs.result()
        availability = future_availability.result()
        feature_results = future_features.result()
        mantle_results = future_mantle.result()
        pricing_linked = future_pricing_linked.result()

    # ============================================
    # WAVE 3: Final Aggregation
    # ============================================
    token_specs_path = output_dir / 'intermediate' / 'token-specs.json'
    availability_path = output_dir / 'intermediate' / 'regional-availability.json'
    models_with_pricing_path = Path(pricing_linked['s3Key'])

    final_result = run_final_aggregator(
        output_dir,
        models_with_pricing_path,
        token_specs_path,
        pricing_path,
        quota_results,
        availability_path=availability_path,
        feature_results=feature_results,
        mantle_results=mantle_results,
    )

    # Summary
    print_summary(output_dir)

    total_time = time.time() - start_time
    print(f"\n  Total time: {total_time:.1f} seconds")
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
