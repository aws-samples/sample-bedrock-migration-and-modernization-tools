# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Local data collector for Amazon Bedrock Model Profiler.

Collects Amazon Bedrock model and pricing data using local AWS credentials,
producing JSON files IDENTICAL to the AWS Step Functions pipeline.

All transformation functions are imported directly from the actual Lambda
handler code to guarantee identical output between local and cloud execution.
"""

import json
import logging
import os
import sys
import time
import importlib.util as _ilu
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup: shared layer must be importable before any Lambda handler code
# ---------------------------------------------------------------------------
_project_root = Path(__file__).parent.parent
_shared_layer_path = str(_project_root / "backend" / "layers" / "common" / "python")
if _shared_layer_path not in sys.path:
    sys.path.insert(0, _shared_layer_path)

# Pre-load the backend config into the shared ConfigLoader singleton
# so all Lambda handlers get context_window_specs, hidden_models, etc.
_config_file = _project_root / "backend" / "config" / "profiler-config.json"
if _config_file.exists():
    os.environ.setdefault("POWERTOOLS_SERVICE_NAME", "local-profiler")
    from shared.config_loader import get_config_loader as _get_config_loader
    _loader = _get_config_loader(force_new=True)
    with open(_config_file) as _f:
        _loader._config = json.load(_f)


# ---------------------------------------------------------------------------
# Import helpers: load functions from Lambda handlers via importlib
# ---------------------------------------------------------------------------
def _import_lambda_module(lambda_name: str):
    """Import a Lambda handler module by name, returning the module object."""
    handler_path = str(_project_root / "backend" / "lambdas" / lambda_name / "handler.py")
    spec = _ilu.spec_from_file_location(f"{lambda_name.replace('-', '_')}_handler", handler_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Import actual Lambda handler functions
_model_extractor_mod = _import_lambda_module("model-extractor")
_process_model_data = _model_extractor_mod.process_model_data
_fetch_console_metadata = _model_extractor_mod.fetch_console_metadata

_model_merger_mod = _import_lambda_module("model-merger")
_merge_models = _model_merger_mod.merge_models

_pricing_aggregator_mod = _import_lambda_module("pricing-aggregator")
_aggregate_pricing = _pricing_aggregator_mod.aggregate_pricing

_pricing_linker_mod = _import_lambda_module("pricing-linker")
_link_pricing_to_models = _pricing_linker_mod.link_pricing_to_models

_regional_availability_mod = _import_lambda_module("regional-availability")
_discover_via_api = _regional_availability_mod._discover_via_api
_build_availability_output = _regional_availability_mod._build_availability_output

_final_aggregator_mod = _import_lambda_module("final-aggregator")
_build_final_models = _final_aggregator_mod.build_final_models

_mantle_mod = _import_lambda_module("mantle-collector")
_call_mantle_endpoint = _mantle_mod.call_mantle_endpoint
_probe_all_responses_support = _mantle_mod.probe_all_responses_support


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RETRY_CONFIG = Config(
    retries={'max_attempts': 3, 'mode': 'adaptive'},
    connect_timeout=10,
    read_timeout=30,
)

BULK_PRICING_URL = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/{service_code}/current/us-east-1/index.json"

DEFAULT_MODEL_REGIONS = ['us-east-1', 'us-west-2']


class LocalCollector:
    """
    Collects Amazon Bedrock model and pricing data using local AWS credentials.

    Uses the actual Lambda handler functions imported from the backend to helps
    identical output between local and cloud execution.
    """

    def __init__(self, profile_name: str = None, output_dir: Path = None):
        self.profile_name = profile_name
        self.output_dir = Path(output_dir) if output_dir else Path("data")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = boto3.Session(profile_name=profile_name)
        self.model_regions = DEFAULT_MODEL_REGIONS
        self.quota_regions = []

        self.pricing_service_codes = [
            'AmazonBedrock',
            'AmazonBedrockFoundationModels',
            'AmazonBedrockService'
        ]

        # Set boto3 default session so Lambda handlers that call
        # boto3.client() / boto3.Session() directly use our profile
        boto3.setup_default_session(profile_name=profile_name)

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
        print("Using actual Lambda handler code from backend/lambdas/")
        print("=" * 60 + "\n")

        # Phase 0: Discover regions (same as region-discovery Lambda)
        print("[0/8] Discovering Amazon Bedrock regions...")
        self.quota_regions = self._discover_regions()
        print(f"       Found {len(self.quota_regions)} Bedrock-enabled regions")

        # Phase 1: Collect raw pricing
        print("\n[1/8] Collecting pricing data...")
        raw_pricing_products = self._collect_pricing()
        results['pricing_raw'] = {'products': len(raw_pricing_products)}
        print(f"       Found {len(raw_pricing_products)} pricing products")

        # Phase 2: Aggregate pricing (pricing-aggregator Lambda)
        print("\n[2/8] Aggregating pricing data...")
        aggregated_data, metadata_stats = _aggregate_pricing(raw_pricing_products)
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

        # Phase 3: Extract models (model-extractor Lambda)
        print(f"\n[3/8] Extracting models from {len(self.model_regions)} regions...")
        models_by_region = self._extract_models()
        total_raw = sum(len(m) for m in models_by_region.values())
        print(f"       Found {total_raw} raw model entries")

        # Phase 4: Merge models (model-merger Lambda)
        print("\n[4/8] Merging and deduplicating models...")
        all_models = []
        for region, models in models_by_region.items():
            all_models.extend(models)
        merged_providers = _merge_models(all_models)
        total_merged = sum(len(p.get('models', {})) for p in merged_providers.values())
        merged_models = {
            'metadata': {'total_models': total_merged},
            'providers': merged_providers
        }
        print(f"       {total_merged} unique models after merge")

        # Phase 5: Link pricing (pricing-linker Lambda)
        # Note: enrichment is no longer a separate step — it happens during
        # extraction via console metadata (same as the backend pipeline).
        print("\n[5/8] Linking pricing to models...")
        link_result = _link_pricing_to_models(merged_models, aggregated_pricing)
        models_with_pricing = {
            'metadata': {
                'models_with_pricing': link_result['models_with_pricing'],
                'models_without_pricing': link_result['models_without_pricing'],
            },
            'providers': link_result['providers']
        }
        print(f"       {link_result['models_with_pricing']} models linked to pricing")

        # Phase 6: Collect quotas, features, and Mantle models
        print(f"\n[6/8] Collecting quotas from {len(self.quota_regions)} regions...")
        quotas_by_region = self._collect_quotas()
        total_quotas = sum(len(q) for q in quotas_by_region.values())
        print(f"       Collected {total_quotas} quotas")

        print(f"       Collecting inference profiles...")
        features_by_region = self._collect_features()
        total_profiles = sum(len(f) for f in features_by_region.values())
        print(f"       Collected {total_profiles} inference profiles")

        # Phase 7: Collect Mantle models (mantle-collector Lambda)
        print(f"\n[7/8] Collecting Mantle models...")
        mantle_by_model = self._collect_mantle()
        print(f"       Found {len(mantle_by_model)} Mantle models")

        # Phase 8: Regional availability + final aggregation
        print("\n[8/8] Building final output...")
        print(f"       Computing regional availability across {len(self.quota_regions)} regions...")
        regional_availability = self._compute_regional_availability()
        on_demand_count = len(regional_availability.get('model_availability', {}))
        print(f"       Discovered {on_demand_count} on-demand models across regions")

        final_providers = _build_final_models(
            models_with_pricing=models_with_pricing,
            regional_availability=regional_availability,
            token_specs={},
            quotas_by_region=quotas_by_region,
            features_by_region=features_by_region,
            enriched_models=merged_models,
            pricing_data=aggregated_pricing,
            collection_timestamp=collection_timestamp,
            mantle_by_model=mantle_by_model,
            lifecycle_by_model={},
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

    # ------------------------------------------------------------------
    # Phase 0: Region discovery
    # ------------------------------------------------------------------
    def _discover_regions(self) -> list:
        """Dynamically discover all Amazon Bedrock-enabled regions (same as region-discovery Lambda)."""
        try:
            ec2 = self.session.client('ec2', region_name='us-east-1')
            response = ec2.describe_regions(
                AllRegions=False,
                Filters=[{'Name': 'opt-in-status', 'Values': ['opt-in-not-required', 'opted-in']}],
            )
            all_regions = [r['RegionName'] for r in response.get('Regions', [])]
        except ClientError as e:
            logger.warning(f"Failed to discover regions: {e}, using defaults")
            return sorted([
                'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2',
                'eu-west-1', 'eu-west-2', 'eu-west-3', 'eu-central-1', 'eu-north-1',
                'ap-northeast-1', 'ap-northeast-2', 'ap-south-1',
                'ap-southeast-1', 'ap-southeast-2', 'ca-central-1', 'sa-east-1',
            ])

        bedrock_regions = []

        def check_region(region):
            try:
                client = self._get_client('bedrock', region)
                client.list_inference_profiles(maxResults=1)
                return region, True
            except ClientError as e:
                code = e.response.get('Error', {}).get('Code', '')
                if code == 'AccessDeniedException':
                    return region, True
                return region, False
            except Exception:
                return region, False

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(check_region, r): r for r in all_regions}
            for future in as_completed(futures):
                region, available = future.result()
                if available:
                    bedrock_regions.append(region)

        return sorted(bedrock_regions)

    # ------------------------------------------------------------------
    # Phase 1: Pricing collection (API calls — kept local)
    # ------------------------------------------------------------------
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
                'source': 'bulk_pricing_api',
            })

        return products

    # ------------------------------------------------------------------
    # Phase 3: Model extraction (uses Lambda's process_model_data + fetch_console_metadata)
    # ------------------------------------------------------------------
    def _extract_models(self) -> dict:
        """Extract models using the same logic as the model-extractor Lambda."""
        models_by_region = {}

        def extract_from_region(region: str) -> tuple:
            models = []
            try:
                client = self._get_client('bedrock', region)
                response = client.list_foundation_models()

                for raw_model in response.get('modelSummaries', []):
                    # Use the Lambda's process_model_data for identical schema
                    processed = _process_model_data(raw_model, region)
                    models.append(processed)

                # Use the Lambda's fetch_console_metadata for full enrichment
                console_metadata = _fetch_console_metadata(region)
                if console_metadata:
                    for model in models:
                        model_id = model.get('model_id', '')
                        if model_id in console_metadata:
                            meta = console_metadata[model_id]

                            # Populate model fields from console metadata
                            # (same as model-extractor Lambda lines 628-656)
                            if meta.get('model_family'):
                                model['model_family'] = meta['model_family']
                            if meta.get('max_context_window'):
                                model['max_context_window'] = meta['max_context_window']
                            if meta.get('max_output_tokens'):
                                model['max_output_tokens'] = meta['max_output_tokens']
                            if meta.get('description'):
                                model['description'] = meta['description']
                            if meta.get('short_description'):
                                model['short_description'] = meta['short_description']
                            if meta.get('languages'):
                                model['languages_supported'] = meta['languages']
                            if meta.get('use_cases'):
                                model['model_use_cases'] = meta['use_cases']
                            if meta.get('model_attributes'):
                                model['model_capabilities'] = meta['model_attributes']
                            if meta.get('release_date'):
                                model['model_lifecycle']['release_date'] = meta['release_date']
                            if meta.get('feature_support'):
                                model['feature_support'] = meta['feature_support']
                            if meta.get('chat_features'):
                                model['chat_features'] = meta['chat_features']
                            if meta.get('guardrails_supported'):
                                model['guardrails_supported'] = meta['guardrails_supported']
                            if meta.get('batch_supported'):
                                model['batch_supported'] = meta['batch_supported']

                            model['console_metadata'] = meta

            except ClientError as e:
                logger.warning(f"Error extracting models from {region}: {e}")

            return region, models

        with ThreadPoolExecutor(max_workers=len(self.model_regions)) as executor:
            futures = {executor.submit(extract_from_region, r): r for r in self.model_regions}
            for future in as_completed(futures):
                region, models = future.result()
                models_by_region[region] = models

        return models_by_region

    # ------------------------------------------------------------------
    # Phase 6: Quotas and features (API calls — kept local)
    # ------------------------------------------------------------------
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
                            'quota_code': quota.get('QuotaCode', ''),
                            'quota_name': quota.get('QuotaName', ''),
                            'quota_arn': quota.get('QuotaArn', ''),
                            'quota_applied_at_level': quota.get('QuotaAppliedAtLevel', 'ACCOUNT'),
                            'value': quota.get('Value'),
                            'unit': quota.get('Unit', ''),
                            'adjustable': quota.get('Adjustable', False),
                            'global_quota': quota.get('GlobalQuota', False),
                            'usage_metric': quota.get('UsageMetric', {}),
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
                            'inferenceProfileId': profile.get('inferenceProfileId', ''),
                            'inferenceProfileArn': profile.get('inferenceProfileArn', ''),
                            'inferenceProfileName': profile.get('inferenceProfileName', ''),
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

    # ------------------------------------------------------------------
    # Phase 7: Mantle collection (uses Lambda's call_mantle_endpoint)
    # ------------------------------------------------------------------
    def _collect_mantle(self) -> dict:
        """Collect Mantle models from all regions (same as mantle-collector Lambda)."""
        _mantle_mod._boto3_session = self.session

        mantle_by_model = {}

        def collect_from_region(region):
            try:
                models = _call_mantle_endpoint(region)
                model_ids = [m['model_id'] for m in models]
                responses_support = _probe_all_responses_support(model_ids, region)
                for m in models:
                    m['supports_responses_api'] = responses_support.get(m['model_id'], False)
                return region, models
            except Exception as e:
                logger.debug(f"Mantle not available in {region}: {e}")
                return region, []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(collect_from_region, r): r for r in self.quota_regions}
            for future in as_completed(futures):
                region, models = future.result()
                for m in models:
                    model_id = m.get('model_id', '')
                    if model_id:
                        if model_id not in mantle_by_model:
                            mantle_by_model[model_id] = {'regions': set(), 'supports_responses_api': False}
                        mantle_by_model[model_id]['regions'].add(region)
                        if m.get('supports_responses_api', False):
                            mantle_by_model[model_id]['supports_responses_api'] = True

        return {
            mid: {'regions': sorted(list(info['regions'])), 'supports_responses_api': info['supports_responses_api']}
            for mid, info in mantle_by_model.items()
        }

    # ------------------------------------------------------------------
    # Phase 8: Regional availability (uses Lambda's _discover_via_api)
    # ------------------------------------------------------------------
    def _compute_regional_availability(self) -> dict:
        """Compute regional availability using the same logic as regional-availability Lambda.

        Uses ON_DEMAND + PROVISIONED inference type filtering (no pricing data union)
        to produce accurate availability without false positives.
        """
        on_demand, provisioned, region_stats, cache_hits, api_calls = _discover_via_api(
            self.quota_regions
            # No S3 cache params — local execution typically uses API
        )

        availability = _build_availability_output(on_demand, provisioned)

        return {
            'model_availability': availability.get('model_availability', {}),
            'provisioned_availability': availability.get('provisioned_availability', {}),
            'regions': availability.get('regions', {}),
            'total_models': len(availability.get('model_availability', {})),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _write_json(self, filename: str, data: Any):
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        size_kb = filepath.stat().st_size / 1024
        print(f"       Wrote {filepath} ({size_kb:.1f} KB)")
