"""
Regional Availability Discovery
Discovers regional availability of models using pricing JSON data
"""

import json
import logging
import glob
import os
from typing import Dict, List, Set, Optional, Any
from pathlib import Path

from config import Config


class RegionalAvailabilityDiscovery:
    """Discovers regional availability of Bedrock models from pricing data"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pricing_data = None
        self.regional_availability_map = {}
        self.bedrock_regions = set()

    def discover_regional_availability(self, models: Dict[str, Any], pricing_references: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Discover regional availability for models using pricing JSON and pricing references

        Args:
            models: Dictionary of models from Phase 1
            pricing_references: Optional pricing references from Phase 2 (for better matching)

        Returns:
            Dictionary with enhanced regional availability data
        """
        self.logger.info("🌍 Discovering regional availability from pricing data...")

        # Load pricing data
        self._load_pricing_data()

        # Extract regional availability from pricing data
        regional_models = self._extract_regional_availability_from_pricing()

        # Cross-reference with models to determine actual availability
        enhanced_models = self._cross_reference_models_with_pricing(models, regional_models, pricing_references)

        # Generate bedrock regions summary
        bedrock_regions_summary = self._generate_bedrock_regions_summary()

        result = {
            'enhanced_models': enhanced_models,
            'regional_availability_map': self.regional_availability_map,
            'bedrock_regions': sorted(list(self.bedrock_regions)),
            'summary': bedrock_regions_summary
        }

        self.logger.info(f"✅ Regional availability discovery complete")
        self.logger.info(f"📊 Bedrock available in {len(self.bedrock_regions)} regions")
        self.logger.info(f"📊 Enhanced {len(enhanced_models)} models with regional data")

        return result

    def _load_pricing_data(self) -> bool:
        """Load the latest pricing data file"""
        try:
            pricing_path = Config.get_pricing_collector_path()
            pricing_pattern = os.path.join(pricing_path, Config.PRICING_FILE_PATTERN)
            pricing_files = glob.glob(pricing_pattern)

            if not pricing_files:
                raise FileNotFoundError(f"No pricing files found in {pricing_path}")

            # Find the latest pricing file
            latest_file = max(pricing_files, key=lambda f: os.path.getmtime(f))
            self.logger.info(f"📄 Loading pricing data from: {os.path.basename(latest_file)}")

            with open(latest_file, 'r', encoding='utf-8') as f:
                self.pricing_data = json.load(f)

            providers_count = len(self.pricing_data.get('providers', {}))
            self.logger.info(f"✅ Loaded pricing data with {providers_count} providers")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load pricing data: {e}")
            raise

    def _extract_regional_availability_from_pricing(self) -> Dict[str, Set[str]]:
        """
        Extract regional availability data from pricing JSON

        Returns:
            Dictionary mapping model names to sets of available regions
        """
        self.logger.info("🔧 Extracting regional availability from pricing data...")

        regional_models = {}  # model_name -> set of regions

        providers = self.pricing_data.get('providers', {})

        for provider_name, provider_data in providers.items():
            self.logger.debug(f"Processing provider: {provider_name}")

            # Handle different provider data structures
            models_data = provider_data.get('models', provider_data) if isinstance(provider_data, dict) else {}

            for model_key, model_info in models_data.items():
                if not isinstance(model_info, dict):
                    continue

                model_name = model_info.get('model_name', model_key)
                regions_data = model_info.get('regions', {})

                if isinstance(regions_data, dict):
                    # Extract regions where this model has pricing (hence is available)
                    available_regions = set(regions_data.keys())

                    if available_regions:
                        # Map both the key and model_name for matching flexibility
                        regional_models[model_key] = available_regions
                        if model_name != model_key:
                            regional_models[model_name] = available_regions

                        # Track all Bedrock regions
                        self.bedrock_regions.update(available_regions)

                        self.logger.debug(f"Model {model_name}: available in {len(available_regions)} regions")

        self.logger.info(f"✅ Extracted regional data for {len(regional_models)} pricing entries")
        self.logger.info(f"🌍 Found {len(self.bedrock_regions)} unique Bedrock regions in pricing data")

        return regional_models

    def _cross_reference_models_with_pricing(self, models: Dict[str, Any], regional_models: Dict[str, Set[str]], pricing_references: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Cross-reference collected models with pricing regional data

        Args:
            models: Models from Phase 1 collection
            regional_models: Regional availability from pricing data
            pricing_references: Pricing references from Phase 2 (for better matching)

        Returns:
            Enhanced models with accurate regional availability
        """
        self.logger.info("🔗 Cross-referencing models with pricing regional data...")

        enhanced_models = {}
        matches_found = 0

        for model_id, model_data in models.items():
            enhanced_model = model_data.copy()

            # Try to find regional availability for this model
            # Use pricing references for better matching if available
            available_regions = self._find_model_regions(model_id, model_data, regional_models, pricing_references)

            if available_regions:
                enhanced_model['regions_available'] = sorted(list(available_regions))
                enhanced_model['regional_availability_source'] = 'pricing_data'
                enhanced_model['total_regions_available'] = len(available_regions)
                matches_found += 1

                # Store in regional availability map
                self.regional_availability_map[model_id] = {
                    'regions': sorted(list(available_regions)),
                    'source': 'pricing_data',
                    'match_method': self._determine_match_method(model_id, model_data, regional_models)
                }

            else:
                # Fallback to discovered regions from Phase 1
                discovered_regions = model_data.get('discovered_in_regions', model_data.get('regions_available', []))
                enhanced_model['regions_available'] = discovered_regions
                enhanced_model['regional_availability_source'] = 'api_discovery'
                enhanced_model['total_regions_available'] = len(discovered_regions)

                self.regional_availability_map[model_id] = {
                    'regions': discovered_regions,
                    'source': 'api_discovery',
                    'match_method': 'no_pricing_match'
                }

            # Update collection metadata
            enhanced_model['collection_metadata']['phase2_regional_discovery'] = True
            enhanced_model['collection_metadata']['regional_data_source'] = enhanced_model['regional_availability_source']

            enhanced_models[model_id] = enhanced_model

        self.logger.info(f"✅ Cross-reference complete: {matches_found}/{len(models)} models matched with pricing regional data")

        return enhanced_models

    def _find_model_regions(self, model_id: str, model_data: Dict[str, Any], regional_models: Dict[str, Set[str]], pricing_references: Dict[str, Any] = None) -> Optional[Set[str]]:
        """
        Find regional availability for a specific model

        Args:
            model_id: Model ID from AWS API
            model_data: Model data from Phase 1
            regional_models: Regional availability from pricing
            pricing_references: Pricing references from Phase 2 (for better matching)

        Returns:
            Set of regions where model is available, or None if not found
        """
        model_name = model_data.get('model_name', '')
        provider = model_data.get('model_provider', '')

        # Strategy 1: Use pricing references if available (most accurate)
        if pricing_references and model_id in pricing_references:
            pricing_data = pricing_references[model_id]
            if 'model_pricing' in pricing_data:
                model_pricing = pricing_data['model_pricing']

                # Check if pricing summary has available_regions
                if 'pricing_summary' in model_pricing:
                    pricing_summary = model_pricing['pricing_summary']
                    available_regions = pricing_summary.get('available_regions', [])
                    if available_regions:
                        regions_from_pricing = set(available_regions)
                        self.logger.debug(f"Found regions from pricing references for {model_id}: {regions_from_pricing}")
                        return regions_from_pricing

        # Strategy 2: Try multiple matching strategies (fallback)
        match_candidates = [
            model_id,
            model_name,
            model_id.lower(),
            model_name.lower(),
        ]

        # Add provider-specific matching patterns
        if provider:
            provider_lower = provider.lower()
            model_id_base = model_id.split(':')[0] if ':' in model_id else model_id

            if 'anthropic' in provider_lower:
                # Try Claude-specific patterns
                if 'claude' in model_id_base:
                    base_name = model_id_base.replace('anthropic.', '')
                    match_candidates.extend([base_name, base_name.replace('-', ' ').title()])

            elif 'amazon' in provider_lower:
                # Try Titan-specific patterns
                if 'titan' in model_id_base:
                    base_name = model_id_base.replace('amazon.', '')
                    match_candidates.extend([base_name, base_name.replace('-', ' ').title()])

        # Try to find matches
        for candidate in match_candidates:
            if candidate in regional_models:
                return regional_models[candidate]

            # Partial matching for complex model names
            for pricing_key, regions in regional_models.items():
                if candidate in pricing_key.lower() or pricing_key.lower() in candidate:
                    return regions

        return None

    def _determine_match_method(self, model_id: str, model_data: Dict[str, Any], regional_models: Dict[str, Set[str]]) -> str:
        """Determine how the model was matched with pricing data"""
        model_name = model_data.get('model_name', '')

        # Check direct matches first
        if model_id in regional_models:
            return 'direct_model_id'
        elif model_name in regional_models:
            return 'direct_model_name'
        else:
            return 'fuzzy_match'

    def _generate_bedrock_regions_summary(self) -> Dict[str, Any]:
        """
        Generate summary of Bedrock regions and availability

        Returns:
            Dictionary with regional availability summary
        """
        summary = {
            'total_bedrock_regions': len(self.bedrock_regions),
            'bedrock_regions': sorted(list(self.bedrock_regions)),
            'regional_model_counts': {},
            'top_regions_by_model_count': [],
            'generation_method': 'pricing_data_analysis'
        }

        # Count models per region
        region_model_counts = {}
        for model_id, availability_info in self.regional_availability_map.items():
            regions = availability_info.get('regions', [])
            for region in regions:
                region_model_counts[region] = region_model_counts.get(region, 0) + 1

        summary['regional_model_counts'] = region_model_counts

        # Find top regions by model count
        sorted_regions = sorted(region_model_counts.items(), key=lambda x: x[1], reverse=True)
        summary['top_regions_by_model_count'] = sorted_regions[:10]  # Top 10 regions

        return summary

    def get_bedrock_regions(self) -> List[str]:
        """
        Get list of Bedrock-enabled regions

        Returns:
            Sorted list of region names
        """
        return sorted(list(self.bedrock_regions))

    def get_regional_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about regional availability discovery

        Returns:
            Dictionary with discovery statistics
        """
        return {
            'total_bedrock_regions': len(self.bedrock_regions),
            'total_models_processed': len(self.regional_availability_map),
            'models_with_pricing_regions': len([
                info for info in self.regional_availability_map.values()
                if info.get('source') == 'pricing_data'
            ]),
            'models_with_api_regions': len([
                info for info in self.regional_availability_map.values()
                if info.get('source') == 'api_discovery'
            ]),
            'bedrock_regions': self.get_bedrock_regions()
        }