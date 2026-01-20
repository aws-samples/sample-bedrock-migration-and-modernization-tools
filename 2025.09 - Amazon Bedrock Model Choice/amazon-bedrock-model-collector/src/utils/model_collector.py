"""
Bedrock Model Collector
Collects foundation models from all Bedrock regions using multi-threading
"""

import boto3
import logging
import time
from typing import Dict, List, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from botocore.exceptions import ClientError

from config import Config
from collectors.direct_api_collector import DirectBedrockAPICollector
from utils.eoe_model_extractor import EoeModelExtractor


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

        # Eoe model extractor (lazy initialization)
        self._eoe_extractor = None

        self.logger = logging.getLogger(__name__)
        self._initialize_session()

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

    def _get_eoe_extractor(self) -> Optional[EoeModelExtractor]:
        """Get Eoe model extractor with lazy initialization"""
        if self._eoe_extractor is None:
            try:
                self.logger.info("🔧 Initializing comprehensive model extractor...")
                self._eoe_extractor = EoeModelExtractor()
                eoe_models = self._eoe_extractor.extract_all_models()
                if eoe_models:
                    self.logger.info(f"✅ Comprehensive extractor ready: {len(eoe_models)} models with complete data")
                else:
                    self.logger.warning("⚠️ Comprehensive extractor initialization failed, enhanced data will not be available")
                    self._eoe_extractor = None
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize comprehensive extractor: {e}")
                self._eoe_extractor = None

        return self._eoe_extractor

    def _enhance_direct_api_models(self):
        """Enhance direct API collected models with comprehensive Eoe data"""
        extractor = self._get_eoe_extractor()
        if extractor:
            self.logger.info("🔧 Enhancing direct API models with comprehensive Eoe data...")
            enhanced_count = 0
            eoe_models = extractor.get_all_models()

            for model_id, model_data in self.all_models.items():
                # Try to find matching Eoe model by various ID patterns
                eoe_match = self._find_eoe_match(model_id, eoe_models)
                if eoe_match:
                    # Add comprehensive Eoe data in consolidated field
                    model_data['EoeModelData'] = {
                        'context_window': eoe_match.get('context_window'),
                        'description': eoe_match.get('description'),
                        'short_description': eoe_match.get('shortDescription'),
                        'content_policy': eoe_match.get('contentPolicy'),
                        'model_name': eoe_match.get('modelName'),
                        'provider_name': eoe_match.get('providerName'),
                        'model_family': eoe_match.get('modelFamily'),
                        'modality': eoe_match.get('modality'),
                        'version': eoe_match.get('version'),
                        'languages': eoe_match.get('languages'),
                        'supported_use_cases': eoe_match.get('supportedUseCases'),
                        'raw_id_reference': eoe_match.get('raw_id_reference')
                    }

                    enhanced_count += 1
                    self.logger.debug(f"🔧 Enhanced {model_id} with comprehensive Eoe data")

            self.logger.info(f"✅ Enhanced {enhanced_count} direct API models with comprehensive data")

    def _find_eoe_match(self, aws_model_id: str, eoe_models: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find matching static model data for AWS API model ID."""
        # Simple mapping for Claude models only (since that's what we have static data for)
        base_id = aws_model_id.split(':')[0]

        # Simple Claude mappings
        claude_mappings = {
            'anthropic.claude-3-5-sonnet': 'ANTHROPIC_CLAUDE_3_5_SONNET_20241022_V2',
            'anthropic.claude-3-5-haiku': 'ANTHROPIC_CLAUDE_3_5_HAIKU_20241022_V1',
            'anthropic.claude-3-sonnet': 'ANTHROPIC_CLAUDE_3_SONNET_20240229_V1',
            'anthropic.claude-3-opus': 'ANTHROPIC_CLAUDE_3_OPUS_20240229_V1',
            'anthropic.claude-3-haiku': 'ANTHROPIC_CLAUDE_3_HAIKU_20240307_V1',
        }

        for aws_pattern, eoe_id in claude_mappings.items():
            if base_id.startswith(aws_pattern) and eoe_id in eoe_models:
                return eoe_models[eoe_id]

        return None


    def collect_models_all_regions(self) -> Dict[str, Any]:
        """
        Collect models from all regions using multi-threading

        Returns:
            Dictionary of all unique models with their regional availability
        """
        # Use direct API if enabled (faster and more efficient)
        if self.use_direct_api:
            self.logger.info("🚀 Using Direct API Collection Mode")
            direct_collector = DirectBedrockAPICollector(self.profile_name)
            models = direct_collector.collect_models()

            # Convert to expected format for compatibility
            self.all_models = models
            for model_id in models:
                self.model_regions[model_id].add('us-east-1')  # Direct API uses us-east-1

            # Enhance with context window information
            self._enhance_direct_api_models()

            return models

        if not self.regions:
            raise ValueError("No regions provided for model collection")

        self.logger.info(f"Starting multi-threaded model collection across {len(self.regions)} regions")
        self.logger.info(f"Using {self.max_workers} concurrent workers")

        start_time = time.time()

        # Use ThreadPoolExecutor for concurrent region processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for each region
            future_to_region = {
                executor.submit(self._collect_models_from_region, region): region
                for region in self.regions
            }

            # Process completed tasks
            for future in as_completed(future_to_region):
                region = future_to_region[future]
                try:
                    region_data = future.result()
                    self._merge_region_data(region, region_data)

                except Exception as e:
                    self.logger.error(f"❌ Failed to collect models from {region}: {e}")
                    self.region_statistics[region] = {
                        'status': 'failed',
                        'error': str(e),
                        'models_count': 0
                    }

        end_time = time.time()
        duration = end_time - start_time

        # Process and log final results
        self._finalize_model_data()

        self.logger.info(f"✅ Model collection complete in {duration:.2f} seconds")
        self.logger.info(f"📊 Total unique models: {len(self.all_models)}")
        self.logger.info(f"📊 Successful regions: {len([r for r in self.region_statistics.values() if r.get('status') == 'success'])}/{len(self.regions)}")

        return self.all_models

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
            'discovered_in_region': region,

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

        # Determine consumption options from inference types
        consumption_options = []
        if 'ON_DEMAND' in processed_model['inference_types_supported']:
            consumption_options.append('on_demand')
        if 'PROVISIONED' in processed_model['inference_types_supported']:
            consumption_options.append('provisioned_throughput')
        processed_model['consumption_options'] = consumption_options

        # Extract comprehensive Eoe data
        extractor = self._get_eoe_extractor()
        if extractor:
            eoe_models = extractor.get_all_models()
            eoe_match = self._find_eoe_match(model_id, eoe_models)
            if eoe_match:
                # Add comprehensive Eoe data in consolidated field
                processed_model['EoeModelData'] = {
                    'context_window': eoe_match.get('context_window'),
                    'description': eoe_match.get('description'),
                    'short_description': eoe_match.get('shortDescription'),
                    'content_policy': eoe_match.get('contentPolicy'),
                    'model_name': eoe_match.get('modelName'),
                    'provider_name': eoe_match.get('providerName'),
                    'model_family': eoe_match.get('modelFamily'),
                    'modality': eoe_match.get('modality'),
                    'version': eoe_match.get('version'),
                    'languages': eoe_match.get('languages'),
                    'supported_use_cases': eoe_match.get('supportedUseCases'),
                    'raw_id_reference': eoe_match.get('raw_id_reference')
                }

                self.logger.debug(f"🔧 Enhanced {model_id} with comprehensive Eoe data")

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
        for model_id, regions in self.model_regions.items():
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