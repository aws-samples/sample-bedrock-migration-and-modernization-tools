"""
Bedrock Model Collector
Collects foundation models from all Bedrock regions using multi-threading
"""

import boto3
import logging
import time
import glob
import os
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from botocore.exceptions import ClientError

from config import Config


class BedrockModelCollector:
    """Multi-threaded collector for Bedrock foundation models across regions"""

    def __init__(self, profile_name: Optional[str] = None, regions: List[str] = None, max_workers: int = 2, use_direct_api: bool = False):
        """
        Initialize the model collector

        Args:
            profile_name: AWS profile name to use
            regions: List of regions to collect models from
            max_workers: Maximum number of concurrent worker threads
            use_direct_api: Use direct REST API for faster collection
        """
        self.profile_name = profile_name
        self.regions = regions or []
        self.max_workers = max_workers
        self.use_direct_api = use_direct_api
        self.session = None

        # Data storage
        self.all_models = {}  # model_id -> model_data
        self.model_regions = defaultdict(set)  # model_id -> set of regions
        self.region_statistics = {}  # region -> stats

        self.logger = logging.getLogger(__name__)
        self._initialize_session()
        self._validate_pricing_dependency()

    def _initialize_session(self):
        """Initialize AWS session with credentials"""
        try:
            if self.profile_name:
                self.session = boto3.Session(profile_name=self.profile_name)
            else:
                self.session = boto3.Session()

            self.logger.info(f"✅ Model collector initialized with {self.max_workers} workers")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize model collector: {e}")
            raise

    def _validate_pricing_dependency(self):
        """
        Validate that pricing JSON exists before starting collection

        Raises:
            RuntimeError: If no pricing data is found
        """
        try:
            pricing_path = Config.get_pricing_collector_path()
            pricing_pattern = os.path.join(pricing_path, Config.PRICING_FILE_PATTERN)
            pricing_files = glob.glob(pricing_pattern)

            if not pricing_files:
                raise RuntimeError(
                    "❌ DEPENDENCY ERROR: No pricing data found!\n"
                    f"Expected location: {pricing_path}\n"
                    f"Pattern: {Config.PRICING_FILE_PATTERN}\n"
                    "The model-collector requires pricing-collector to run first.\n"
                    "Please run the pricing-collector before collecting models."
                )

            # Find the latest pricing file
            latest_pricing_file = max(pricing_files, key=lambda f: os.path.getmtime(f))
            self.logger.info(f"✅ Pricing dependency validated: {os.path.basename(latest_pricing_file)}")

        except Exception as e:
            self.logger.error(f"❌ Pricing dependency validation failed: {e}")
            raise



    def collect_models_all_regions(self) -> Dict[str, Any]:
        """
        Collect models using optimized dual-region approach

        Returns:
            Dictionary of all unique models with their regional availability
        """
        # NEW PHASE 1: Use optimized dual-region collection
        self.logger.info("🚀 Using Optimized Dual-Region Collection Mode")

        # Collect from us-east-1 and us-west-2 in parallel for complete model catalog
        models = self._collect_models_dual_region()

        # Convert to expected format for compatibility
        self.all_models = models

        return models

    def _collect_models_dual_region(self) -> Dict[str, Any]:
        """
        Collect models from us-east-1 and us-west-2 in parallel with deduplication

        Returns:
            Dictionary of all unique models with regional information
        """
        self.logger.info("📡 Collecting models from us-east-1 and us-west-2 in parallel...")

        # Define the two key regions for comprehensive model collection
        key_regions = ['us-east-1', 'us-west-2']

        start_time = time.time()

        # Use ThreadPoolExecutor for parallel collection from key regions
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit tasks for both key regions
            future_to_region = {
                executor.submit(self._collect_models_from_region, region): region
                for region in key_regions
            }

            region_results = {}

            # Process completed tasks
            for future in as_completed(future_to_region):
                region = future_to_region[future]
                try:
                    region_data = future.result()
                    region_results[region] = region_data
                    self.logger.info(f"✅ Collected {region_data.get('models_count', 0)} models from {region}")

                except Exception as e:
                    self.logger.error(f"❌ Failed to collect models from {region}: {e}")
                    region_results[region] = {
                        'status': 'failed',
                        'error': str(e),
                        'models': {},
                        'models_count': 0
                    }

        # Deduplicate and merge models from both regions
        deduplicated_models = self._deduplicate_dual_region_models(region_results)

        end_time = time.time()
        duration = end_time - start_time

        self.logger.info(f"✅ Dual-region collection complete in {duration:.2f} seconds")
        self.logger.info(f"📊 Total unique models: {len(deduplicated_models)}")

        return deduplicated_models

    def _deduplicate_dual_region_models(self, region_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Deduplicate models collected from multiple regions

        Args:
            region_results: Dictionary of region -> collection results

        Returns:
            Dictionary of deduplicated models
        """
        self.logger.info("🔧 Deduplicating models from dual-region collection...")

        deduplicated_models = {}
        model_regions_map = defaultdict(set)

        for region, result in region_results.items():
            if result.get('status') == 'success':
                models = result.get('models', {})

                for model_id, model_data in models.items():
                    # Track which regions have this model
                    model_regions_map[model_id].add(region)

                    # Smart deduplication: prefer models with context window patterns
                    if model_id not in deduplicated_models:
                        # First time seeing this model, store it
                        deduplicated_models[model_id] = model_data.copy()
                        deduplicated_models[model_id]['regions_available'] = []
                    else:
                        # Model already exists, check if this variant is better
                        current_model_id = deduplicated_models[model_id]['model_id']
                        current_arn = deduplicated_models[model_id].get('model_arn', '')
                        new_arn = model_data.get('model_arn', '')

                        # Extract original IDs from ARNs to check for context patterns
                        def has_context_pattern(arn):
                            if not arn or '/' not in arn:
                                return False
                            original_id = arn.split('/')[-1]
                            # Check for context patterns like :4k, :200k, :1000k
                            return (original_id.count(':') >= 2 and
                                   len(original_id.split(':')) >= 3 and
                                   original_id.split(':')[-1].lower().endswith('k'))

                        def extract_context_value(arn):
                            """Extract context window size from ARN"""
                            if not arn or '/' not in arn:
                                return None
                            original_id = arn.split('/')[-1]
                            if original_id.count(':') >= 2:
                                parts = original_id.split(':')
                                if len(parts) >= 3:
                                    last_part = parts[-1].lower()
                                    if last_part.endswith('k') and len(last_part) > 1:
                                        number_part = last_part[:-1]
                                        try:
                                            context_k = float(number_part)
                                            return int(context_k * 1000)
                                        except (ValueError, TypeError):
                                            pass
                            return None

                        current_has_context = has_context_pattern(current_arn)
                        new_has_context = has_context_pattern(new_arn)

                        # Enhanced logic: prefer highest context value
                        if new_has_context and current_has_context:
                            # Both have context patterns - choose the one with higher context value
                            current_context = extract_context_value(current_arn) or 0
                            new_context = extract_context_value(new_arn) or 0

                            if new_context > current_context:
                                self.logger.info(f"🔄 Upgrading {model_id} to higher context variant: {new_context:,} > {current_context:,} ({new_arn})")
                                deduplicated_models[model_id] = model_data.copy()
                                deduplicated_models[model_id]['regions_available'] = []
                            # If current is higher or equal, keep it
                        elif new_has_context and not current_has_context:
                            # New has context, current doesn't - upgrade to context variant
                            new_context = extract_context_value(new_arn) or 0
                            self.logger.info(f"🔄 Upgrading {model_id} to context-bearing variant: {new_context:,} ({new_arn})")
                            deduplicated_models[model_id] = model_data.copy()
                            deduplicated_models[model_id]['regions_available'] = []
                        # If both or neither have context (and new doesn't have higher context), keep the current one

        # Update regional availability for each model
        for model_id in deduplicated_models:
            regions = sorted(list(model_regions_map[model_id]))
            deduplicated_models[model_id]['regions_available'] = regions

            # Store in class attributes for compatibility
            self.model_regions[model_id] = model_regions_map[model_id]

            # Update collection metadata
            deduplicated_models[model_id]['collection_metadata']['dual_region_collection'] = True
            deduplicated_models[model_id]['collection_metadata']['regions_collected_from'] = regions

        self.logger.info(f"✅ Deduplication complete: {len(deduplicated_models)} unique models")

        # Log regional coverage statistics
        region_coverage = {}
        for model_id, regions_set in model_regions_map.items():
            coverage = len(regions_set)
            region_coverage[coverage] = region_coverage.get(coverage, 0) + 1

        for coverage, count in region_coverage.items():
            self.logger.info(f"   - {count} models available in {coverage} region(s)")

        return deduplicated_models

    def _collect_models_from_region(self, region: str) -> Dict[str, Any]:
        """
        Collect models from a specific region

        Args:
            region: AWS region to collect models from

        Returns:
            Dictionary with models data and statistics for this region
        """
        self.logger.info(f"🌍 Collecting models from region: {region}")

        try:
            bedrock = self.session.client('bedrock', region_name=region)

            # Get all foundation models in this region
            response = bedrock.list_foundation_models()
            all_models = response.get('modelSummaries', [])

            self.logger.info(f"✅ Found {len(all_models)} models in {region}")

            # Process and enhance model data
            processed_models = {}
            for model in all_models:
                model_id = model.get('modelId', '')
                if model_id:
                    # Extract comprehensive model information
                    processed_model = self._process_model_data(model, region)
                    processed_models[model_id] = processed_model

            return {
                'status': 'success',
                'models': processed_models,
                'models_count': len(processed_models),
                'region': region
            }

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_msg = str(e)
            self.logger.error(f"❌ AWS error in {region}: {error_code} - {error_msg}")
            raise

        except Exception as e:
            self.logger.error(f"❌ Unexpected error in {region}: {e}")
            raise

    def _process_model_data(self, raw_model: Dict[str, Any], region: str) -> Dict[str, Any]:
        """
        Process and enhance raw model data from the API

        Args:
            raw_model: Raw model data from list_foundation_models API
            region: Region where this model was found

        Returns:
            Enhanced model data dictionary
        """
        # Extract all available fields from the API response
        model_id = raw_model.get('modelId', '')

        processed_model = {
            # Core identification
            'model_id': model_id,
            'model_arn': raw_model.get('modelArn', ''),
            'model_name': raw_model.get('modelName', ''),
            'model_provider': raw_model.get('providerName', ''),

            # Capabilities from API
            'model_modalities': {
                'input_modalities': raw_model.get('inputModalities', []),
                'output_modalities': raw_model.get('outputModalities', [])
            },
            'streaming_supported': raw_model.get('responseStreamingSupported', False),
            'customization': {
                'customization_supported': raw_model.get('customizationsSupported', []),
                'customization_options': {}  # Will be enhanced later
            },
            'inference_types_supported': raw_model.get('inferenceTypesSupported', []),
            'model_lifecycle': {
                'status': raw_model.get('modelLifecycle', {}).get('status', 'UNKNOWN'),
                'release_date': ''  # Will be enhanced later
            },

            # Regional information
            'regions_available': [region],  # Will be merged from other regions

            # Fields to be enhanced in later phases
            'model_capabilities': [],
            'model_use_cases': [],
            'languages_supported': [],
            'consumption_options': [],
            'cross_region_inference': {},
            'documentation_links': {},
            'model_pricing': {'is_pricing_available': False},
            'model_service_quotas': {},

            # Collection metadata
            'collection_metadata': {
                'first_discovered_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                'first_discovered_in_region': region,
                'api_source': 'list_foundation_models'
            }
        }

        # Generate initial documentation links
        processed_model['documentation_links'] = Config.generate_documentation_links(
            model_id, processed_model['model_provider']
        )

        # Consumption options will be determined in Phase 4 by ModelFeaturesEnhancer using pricing data
        # This provides more accurate consumption options based on actual pricing groups
        processed_model['consumption_options'] = []  # Placeholder, will be set in Phase 4

        # Enhanced metadata will be added in Phase 4 by ModelFeaturesEnhancer
        # No longer using EOE extractor - replaced by intelligent metadata generation

        return processed_model

    def _merge_region_data(self, region: str, region_data: Dict[str, Any]):
        """
        Merge data from a region into the global model collection

        Args:
            region: Region name
            region_data: Data collected from this region
        """
        if region_data['status'] == 'success':
            models = region_data['models']

            for model_id, model_data in models.items():
                # Track regions for this model
                self.model_regions[model_id].add(region)

                # If this is the first time we see this model, store it
                if model_id not in self.all_models:
                    self.all_models[model_id] = model_data
                else:
                    # Merge regional availability
                    existing_regions = set(self.all_models[model_id].get('regions_available', []))
                    existing_regions.add(region)
                    self.all_models[model_id]['regions_available'] = sorted(list(existing_regions))

            self.logger.info(f"📊 Merged {len(models)} models from {region}")

        # Store region statistics
        self.region_statistics[region] = {
            'status': region_data['status'],
            'models_count': region_data.get('models_count', 0),
            'processed_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        }

    def _finalize_model_data(self):
        """Finalize model data after all regions have been processed"""
        # Update all models with final regional availability
        for model_id, regions_set in self.model_regions.items():
            if model_id in self.all_models:
                self.all_models[model_id]['regions_available'] = sorted(list(regions_set))

        # Add collection statistics to each model
        for model_id, model_data in self.all_models.items():
            model_data['collection_metadata']['total_regions_available'] = len(self.model_regions[model_id])

        self.logger.info("✅ Model data finalization complete")

    def get_collection_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the model collection

        Returns:
            Dictionary with collection statistics
        """
        successful_regions = len([r for r in self.region_statistics.values() if r.get('status') == 'success'])

        # Provider statistics
        providers = {}
        for model in self.all_models.values():
            provider = model.get('model_provider', 'Unknown')
            providers[provider] = providers.get(provider, 0) + 1

        # Regional coverage statistics
        region_coverage = {}
        for _, regions in self.model_regions.items():
            coverage = len(regions)
            region_coverage[coverage] = region_coverage.get(coverage, 0) + 1

        return {
            'total_unique_models': len(self.all_models),
            'total_regions_processed': len(self.regions),
            'successful_regions': successful_regions,
            'failed_regions': len(self.regions) - successful_regions,
            'providers': providers,
            'regional_coverage': region_coverage,
            'collection_method': f'multi_threaded_{self.max_workers}_workers',
            'region_statistics': self.region_statistics
        }