#!/usr/bin/env python3
"""
Local End-to-End Workflow Test

Simulates the Step Functions workflow by running each Lambda in sequence,
using local files instead of S3.

Usage:
    cd bedrock-profiler-stepfunctions
    source venv/bin/activate
    python tests/test_workflow_local.py [--quick]

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

# Add lambdas to path
LAMBDAS_DIR = Path(__file__).parent.parent / 'lambdas'
sys.path.insert(0, str(LAMBDAS_DIR))

# Output directory for test data
OUTPUT_DIR = Path(__file__).parent / 'workflow_output'


def setup_output_dir():
    """Create output directory structure."""
    dirs = [
        OUTPUT_DIR / 'pricing',
        OUTPUT_DIR / 'models',
        OUTPUT_DIR / 'quotas',
        OUTPUT_DIR / 'features',
        OUTPUT_DIR / 'merged',
        OUTPUT_DIR / 'intermediate',
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

    # Monkey-patch to write locally instead of S3
    original_handler = spec.lambda_handler

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
        # All 16 known Bedrock regions from the original profiler
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
                print(f"  ⚠ Skipped {region} (no quotas or region unavailable)")
                return {
                    'status': 'SKIPPED',
                    'region': region,
                    'quotaCount': 0,
                    'durationMs': duration
                }
        except Exception as e:
            print(f"  ⚠ Failed {region}: {e}")
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

    # Aggregate
    aggregated = spec.aggregate_pricing(all_products)

    output_data = {
        'metadata': {
            'providersCount': len(aggregated),
            'totalProducts': len(all_products)
        },
        'providers': aggregated
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


def run_regional_availability(output_dir: Path, pricing_path: Path):
    """Compute regional availability."""
    print("\n" + "="*60)
    print("WAVE 2B: Regional Availability")
    print("="*60)

    sys.path.insert(0, str(LAMBDAS_DIR / 'regional-availability'))
    from importlib import import_module
    spec = import_module('regional-availability.handler')

    pricing_data = load_json(pricing_path)
    availability = spec.compute_regional_availability(pricing_data)

    print(f"  Regions with Bedrock: {len(availability['regions'])}")
    print(f"  Models tracked: {len(availability['modelAvailability'])}")

    output_data = {
        'metadata': {'regionsWithBedrock': len(availability['regions'])},
        'regions': availability['regions'],
        'modelAvailability': availability['modelAvailability']
    }
    output_path = output_dir / 'intermediate' / 'regional-availability.json'
    save_json(output_path, output_data)

    return {'status': 'SUCCESS', 's3Key': str(output_path)}


def run_pricing_linker(output_dir: Path, pricing_path: Path, models_path: Path):
    """Link pricing to models."""
    print("\n" + "="*60)
    print("WAVE 2C: Pricing Linker")
    print("="*60)

    sys.path.insert(0, str(LAMBDAS_DIR / 'pricing-linker'))
    from importlib import import_module
    spec = import_module('pricing-linker.handler')

    pricing_data = load_json(pricing_path)
    models_data = load_json(models_path)

    result = spec.link_pricing_to_models(models_data, pricing_data)

    print(f"  Models with pricing: {result['modelsWithPricing']}")
    print(f"  Models without pricing: {result['modelsWithoutPricing']}")

    output_data = {
        'metadata': {
            'modelsWithPricing': result['modelsWithPricing'],
            'modelsWithoutPricing': result['modelsWithoutPricing']
        },
        'providers': result['providers']
    }
    output_path = output_dir / 'intermediate' / 'models-with-pricing.json'
    save_json(output_path, output_data)

    return {'status': 'SUCCESS', 's3Key': str(output_path)}


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

    # Wave 1: Parallel collection
    pricing_results = run_pricing_collectors(output_dir, args.quick)
    model_results = run_model_extractors(output_dir, args.quick)
    quota_results = run_quota_collectors(output_dir, args.quick)

    # Wave 1: Aggregation
    pricing_agg = run_pricing_aggregator(output_dir, pricing_results)
    models_merged = run_model_merger(output_dir, model_results)

    # Wave 2: Enrichment
    pricing_path = Path(pricing_agg['s3Key'])
    models_path = Path(models_merged['s3Key'])

    token_specs = run_token_specs_collector(output_dir, models_path)
    availability = run_regional_availability(output_dir, pricing_path)
    pricing_linked = run_pricing_linker(output_dir, pricing_path, models_path)

    # Summary
    print_summary(output_dir)

    total_time = time.time() - start_time
    print(f"\n  Total time: {total_time:.1f} seconds")
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
