"""
Local data collector for Bedrock Model Profiler.

Collects Bedrock model and pricing data using local AWS credentials,
producing JSON files IDENTICAL to the AWS Step Functions pipeline.

Uses the same transformation logic as the Lambda handlers.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.config import Config
from botocore.exceptions import ClientError

# Import local handlers (same logic as Lambda handlers)
from local.handlers import (
    aggregate_pricing,
    merge_models,
    enrich_providers,
    link_pricing_to_models,
    compute_regional_availability,
    build_final_models,
)

logger = logging.getLogger(__name__)

RETRY_CONFIG = Config(
    retries={'max_attempts': 3, 'mode': 'adaptive'},
    connect_timeout=10,
    read_timeout=30,
)

# Bulk Pricing API URL template
BULK_PRICING_URL = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/{service_code}/current/us-east-1/index.json"

# Default regions for collection (extended to match cloud deployment)
DEFAULT_MODEL_REGIONS = ['us-east-1', 'us-west-2']
DEFAULT_QUOTA_REGIONS = [
    # Americas
    'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2',
    'ca-central-1', 'sa-east-1',
    # Europe
    'eu-west-1', 'eu-west-2', 'eu-west-3', 'eu-central-1', 'eu-north-1',
    'eu-south-1', 'eu-south-2',
    # Asia Pacific
    'ap-east-2', 'ap-southeast-1', 'ap-southeast-2', 'ap-southeast-5', 'ap-southeast-7',
    'ap-northeast-1', 'ap-northeast-2', 'ap-northeast-3', 'ap-south-1', 'ap-south-2',
    # Middle East & Israel
    'me-central-1', 'il-central-1',
]


class LocalCollector:
    """
    Collects Bedrock model and pricing data using local AWS credentials.

    Uses the same transformation functions as the AWS Lambda pipeline to ensure
    identical output between local and cloud execution.
    """

    def __init__(self, profile_name: str = None, output_dir: Path = None):
        self.profile_name = profile_name
        self.output_dir = Path(output_dir) if output_dir else Path("data")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = boto3.Session(profile_name=profile_name)
        self.model_regions = DEFAULT_MODEL_REGIONS
        self.quota_regions = DEFAULT_QUOTA_REGIONS

        self.pricing_service_codes = [
            'AmazonBedrock',
            'AmazonBedrockFoundationModels',
            'AmazonBedrockService'
        ]

        logger.info(f"Initialized LocalCollector with profile={profile_name}, output_dir={self.output_dir}")

    def _get_client(self, service: str, region: str = None):
        kwargs = {'config': RETRY_CONFIG}
        if region:
            kwargs['region_name'] = region
        return self.session.client(service, **kwargs)

    def collect_all(self) -> dict:
        """Run the full data collection pipeline (same as Step Functions)."""
        start_time = time.time()
        results = {}
        collection_timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.000000+00:00')

        print("\n" + "=" * 60)
        print("Bedrock Model Profiler - Local Data Collection")
        print("=" * 60)
        print(f"AWS Profile: {self.profile_name or 'default'}")
        print(f"Output Directory: {self.output_dir.absolute()}")
        print("Using same handlers as AWS Step Functions pipeline")
        print("=" * 60 + "\n")

        # Phase 1: Collect raw pricing
        print("[1/8] Collecting pricing data...")
        raw_pricing_products = self._collect_pricing()
        results['pricing_raw'] = {'products': len(raw_pricing_products)}
        print(f"       Found {len(raw_pricing_products)} pricing products")

        # Phase 2: Aggregate pricing (same as pricing-aggregator Lambda)
        print("\n[2/8] Aggregating pricing data...")
        aggregated_data, metadata_stats = aggregate_pricing(raw_pricing_products)
        aggregated_pricing = {
            'metadata': {
                'generated_at': collection_timestamp,
                'version': '1.0.0',
                'total_pricing_entries': metadata_stats['total_entries'],
                'providers_count': len(aggregated_data),
                'currency': 'USD',
            },
            'providers': dict(aggregated_data)
        }
        print(f"       Aggregated into {len(aggregated_data)} providers")

        # Phase 3: Extract models
        print(f"\n[3/8] Extracting models from {len(self.model_regions)} regions...")
        models_by_region = self._extract_models()
        total_raw = sum(len(m) for m in models_by_region.values())
        print(f"       Found {total_raw} raw model entries")

        # Phase 4: Merge models (same as model-merger Lambda)
        print("\n[4/8] Merging and deduplicating models...")
        all_models = []
        for region, models in models_by_region.items():
            all_models.extend(models)
        merged_providers = merge_models(all_models)
        total_merged = sum(len(p.get('models', {})) for p in merged_providers.values())
        merged_models = {
            'metadata': {'total_models': total_merged},
            'providers': merged_providers
        }
        print(f"       {total_merged} unique models after merge")

        # Phase 5: Enrich models (same as model-enricher Lambda)
        print("\n[5/8] Enriching models with capabilities...")
        enriched_providers = enrich_providers(merged_providers)
        enriched_models = {
            'metadata': {'total_models': total_merged},
            'providers': enriched_providers
        }
        print(f"       Added capabilities and use cases")

        # Phase 6: Link pricing (same as pricing-linker Lambda)
        print("\n[6/8] Linking pricing to models...")
        link_result = link_pricing_to_models(enriched_models, aggregated_pricing)
        models_with_pricing = {
            'metadata': {
                'models_with_pricing': link_result['models_with_pricing'],
                'models_without_pricing': link_result['models_without_pricing'],
            },
            'providers': link_result['providers']
        }
        print(f"       {link_result['models_with_pricing']} models linked to pricing")

        # Phase 7: Collect quotas and features
        print(f"\n[7/8] Collecting quotas from {len(self.quota_regions)} regions...")
        quotas_by_region = self._collect_quotas()
        total_quotas = sum(len(q) for q in quotas_by_region.values())
        print(f"       Collected {total_quotas} quotas")

        print(f"       Collecting inference profiles...")
        features_by_region = self._collect_features()
        total_profiles = sum(len(f) for f in features_by_region.values())
        print(f"       Collected {total_profiles} inference profiles")

        # Phase 8: Final aggregation (same as regional-availability + final-aggregator)
        print("\n[8/8] Building final output...")
        print(f"       Computing regional availability across {len(self.quota_regions)} regions...")
        regional_availability = compute_regional_availability(
            self.quota_regions,
            aggregated_pricing,
            self.session
        )
        print(f"       Discovered {regional_availability['total_models']} models across regions")

        final_providers = build_final_models(
            models_with_pricing,
            regional_availability,
            quotas_by_region,
            features_by_region,
            enriched_models,
            aggregated_pricing,
            collection_timestamp
        )

        total_models = sum(len(p.get('models', {})) for p in final_providers.values())

        final_models = {
            'metadata': {
                'collection_timestamp': collection_timestamp,
                'providers_count': len(final_providers),
                'total_models': total_models,
                'collection_method': 'local_collector',
                'pipeline_version': 'identical_to_step_functions'
            },
            'providers': final_providers
        }

        # Write output files
        self._write_json('bedrock_models.json', final_models)
        self._write_json('bedrock_pricing.json', aggregated_pricing)

        # Copy to frontend/public/latest/ so dev server can serve them
        frontend_dir = Path(__file__).parent.parent / "frontend" / "public" / "latest"
        frontend_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(self.output_dir / 'bedrock_models.json', frontend_dir / 'bedrock_models.json')
        shutil.copy2(self.output_dir / 'bedrock_pricing.json', frontend_dir / 'bedrock_pricing.json')

        duration = time.time() - start_time

        print("\n" + "=" * 60)
        print("Collection Complete!")
        print("=" * 60)
        print(f"Duration: {duration:.1f} seconds")
        print(f"Models collected: {total_models}")
        print(f"Providers: {len(final_providers)}")
        print(f"\nOutput files:")
        print(f"  {self.output_dir}/bedrock_models.json")
        print(f"  {self.output_dir}/bedrock_pricing.json")
        print(f"\nFrontend dev server files:")
        print(f"  {frontend_dir}/bedrock_models.json")
        print(f"  {frontend_dir}/bedrock_pricing.json")
        print("=" * 60 + "\n")

        return {'final': {'models': total_models, 'providers': len(final_providers)}}

    def _collect_pricing(self) -> list:
        """Collect pricing data from AWS Pricing API."""
        pricing_client = self._get_client('pricing', 'us-east-1')
        all_products = []
        existing_skus = set()

        for service_code in self.pricing_service_codes:
            products = self._collect_pricing_for_service(pricing_client, service_code)
            for p in products:
                sku = p.get('product', {}).get('sku')
                if sku and sku not in existing_skus:
                    all_products.append(p)
                    existing_skus.add(sku)

            bulk_products = self._fetch_bulk_pricing(service_code)
            for p in bulk_products:
                sku = p.get('product', {}).get('sku')
                if sku and sku not in existing_skus:
                    all_products.append(p)
                    existing_skus.add(sku)

        return all_products

    def _collect_pricing_for_service(self, pricing_client, service_code: str) -> list:
        products = []
        next_token = None

        try:
            while True:
                params = {'ServiceCode': service_code, 'MaxResults': 100, 'FormatVersion': 'aws_v1'}
                if next_token:
                    params['NextToken'] = next_token

                response = pricing_client.get_products(**params)

                for price_item in response.get('PriceList', []):
                    try:
                        product = json.loads(price_item) if isinstance(price_item, str) else price_item
                        products.append(product)
                    except json.JSONDecodeError:
                        continue

                next_token = response.get('NextToken')
                if not next_token:
                    break
                time.sleep(0.1)

        except ClientError as e:
            logger.warning(f"Error collecting pricing for {service_code}: {e}")

        return products

    def _fetch_bulk_pricing(self, service_code: str) -> list:
        url = BULK_PRICING_URL.format(service_code=service_code)
        try:
            with urlopen(url, timeout=60) as response:
                data = json.loads(response.read().decode('utf-8'))
        except (HTTPError, URLError):
            return []

        products = []
        bulk_products = data.get('products', {})
        bulk_terms = data.get('terms', {}).get('OnDemand', {})

        for sku, product_info in bulk_products.items():
            products.append({
                'product': {'sku': sku, 'attributes': product_info.get('attributes', {})},
                'terms': {'OnDemand': bulk_terms.get(sku, {})},
            })

        return products

    def _extract_models(self) -> dict:
        models_by_region = {}

        def extract_from_region(region: str) -> tuple:
            models = []
            try:
                client = self._get_client('bedrock', region)
                response = client.list_foundation_models()

                for raw_model in response.get('modelSummaries', []):
                    models.append({
                        'model_id': raw_model.get('modelId', ''),
                        'model_arn': raw_model.get('modelArn', ''),
                        'model_name': raw_model.get('modelName', ''),
                        'model_provider': raw_model.get('providerName', ''),
                        'model_modalities': {
                            'input_modalities': raw_model.get('inputModalities', []),
                            'output_modalities': raw_model.get('outputModalities', [])
                        },
                        'streaming_supported': raw_model.get('responseStreamingSupported', False),
                        'customization_supported': raw_model.get('customizationsSupported', []),
                        'inference_types_supported': raw_model.get('inferenceTypesSupported', []),
                        'model_lifecycle': {'status': raw_model.get('modelLifecycle', {}).get('status', 'ACTIVE')},
                        'regions_available': [region],
                        'collection_metadata': {
                            'first_discovered_in_region': region,
                            'regions_collected_from': [region]
                        }
                    })

                console_meta = self._fetch_console_metadata(region)
                for model in models:
                    model_id = model.get('model_id', '')
                    if model_id in console_meta:
                        model['console_metadata'] = console_meta[model_id]

            except ClientError as e:
                logger.warning(f"Error extracting models from {region}: {e}")

            return region, models

        with ThreadPoolExecutor(max_workers=len(self.model_regions)) as executor:
            futures = {executor.submit(extract_from_region, r): r for r in self.model_regions}
            for future in as_completed(futures):
                region, models = future.result()
                models_by_region[region] = models

        return models_by_region

    def _fetch_console_metadata(self, region: str) -> dict:
        try:
            credentials = self.session.get_credentials()
            if not credentials:
                return {}

            frozen_credentials = credentials.get_frozen_credentials()
            url = f'https://bedrock.{region}.amazonaws.com/foundation-models'

            request = AWSRequest(method='GET', url=url, headers={'Content-Type': 'application/json', 'x-console-consumer': 'true'})
            SigV4Auth(frozen_credentials, 'bedrock', region).add_auth(request)

            http_request = Request(url, headers=dict(request.headers), method='GET')
            with urlopen(http_request, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))

            metadata_by_id = {}
            for model in data.get('modelSummaries', []):
                model_id = model.get('modelId', '')
                if not model_id:
                    continue

                meta = {}
                console_ide_raw = model.get('consoleIDEMetadata')
                if console_ide_raw:
                    try:
                        console_ide = json.loads(console_ide_raw)
                        desc = console_ide.get('description', {})
                        if desc.get('fullDescription'):
                            meta['description'] = desc['fullDescription']
                        if desc.get('shortDescription'):
                            meta['short_description'] = desc['shortDescription']
                        if desc.get('maxContextWindow'):
                            meta['max_context_window'] = self._parse_context_window(desc['maxContextWindow'])
                        if desc.get('maxOutputTokens'):
                            meta['max_output_tokens'] = self._parse_context_window(desc['maxOutputTokens'])
                        if desc.get('useCases'):
                            meta['use_cases'] = desc['useCases']
                        if desc.get('languages'):
                            meta['languages'] = desc['languages']
                    except json.JSONDecodeError:
                        pass

                if meta:
                    metadata_by_id[model_id] = meta

            return metadata_by_id

        except Exception:
            return {}

    def _parse_context_window(self, value: str) -> int:
        import re
        if not value:
            return None
        value = str(value).strip().replace(',', '')
        try:
            return int(value)
        except ValueError:
            pass
        match = re.match(r'^([\d.]+)\s*([KkMm])', value)
        if match:
            num = float(match.group(1))
            unit = match.group(2).upper()
            return int(num * 1000) if unit == 'K' else int(num * 1000000)
        return None

    def _collect_quotas(self) -> dict:
        quotas_by_region = {}

        def collect_from_region(region: str) -> tuple:
            quotas = []
            try:
                client = self._get_client('service-quotas', region)
                paginator = client.get_paginator('list_service_quotas')
                for page in paginator.paginate(ServiceCode='bedrock'):
                    for quota in page.get('Quotas', []):
                        quotas.append({
                            'quotaCode': quota.get('QuotaCode', ''),
                            'quotaName': quota.get('QuotaName', ''),
                            'quotaArn': quota.get('QuotaArn', ''),
                            'description': quota.get('Description', ''),
                            'quotaAppliedAtLevel': quota.get('QuotaAppliedAtLevel', 'ACCOUNT'),
                            'value': quota.get('Value'),
                            'unit': quota.get('Unit', ''),
                            'adjustable': quota.get('Adjustable', False),
                            'globalQuota': quota.get('GlobalQuota', False),
                            'usageMetric': quota.get('UsageMetric', {}),
                            'period': quota.get('Period', {}),
                            'region': region
                        })
            except ClientError:
                pass
            return region, quotas

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(collect_from_region, r): r for r in self.quota_regions}
            for future in as_completed(futures):
                region, quotas = future.result()
                quotas_by_region[region] = quotas

        return quotas_by_region

    def _collect_features(self) -> dict:
        features_by_region = {}

        def collect_from_region(region: str) -> tuple:
            profiles = []
            try:
                client = self._get_client('bedrock', region)
                paginator = client.get_paginator('list_inference_profiles')
                for page in paginator.paginate():
                    for profile in page.get('inferenceProfileSummaries', []):
                        profiles.append({
                            'inference_profile_id': profile.get('inferenceProfileId', ''),
                            'inference_profile_name': profile.get('inferenceProfileName', ''),
                            'status': profile.get('status', ''),
                            'type': profile.get('type', ''),
                            'description': profile.get('description', ''),
                            'models': profile.get('models', []),
                            'region': region
                        })
            except ClientError:
                pass
            return region, profiles

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(collect_from_region, r): r for r in self.quota_regions}
            for future in as_completed(futures):
                region, profiles = future.result()
                features_by_region[region] = profiles

        return features_by_region

    def _write_json(self, filename: str, data: Any):
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        size_kb = filepath.stat().st_size / 1024
        print(f"       Wrote {filepath} ({size_kb:.1f} KB)")
