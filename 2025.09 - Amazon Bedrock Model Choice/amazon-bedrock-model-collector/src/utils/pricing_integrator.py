"""
Pricing Integrator
Integrates pricing data from amazon-bedrock-pricing-collector
"""

import json
import logging
import glob
from pathlib import Path
from typing import Dict, Any, Optional, List


class PricingIntegrator:
    """Integrates comprehensive pricing data from amazon-bedrock-pricing-collector"""

    def __init__(self, pricing_collector_path: str = '../amazon-bedrock-pricing-collector/out'):
        self.pricing_collector_path = pricing_collector_path
        self.logger = logging.getLogger(__name__)
        self.pricing_data = None

    def integrate_pricing_data(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate pricing data with model information"""
        self.logger.info("Integrating pricing data from amazon-bedrock-pricing-collector...")

        # Load latest pricing data
        pricing_data = self._load_latest_pricing_data()
        if not pricing_data:
            self.logger.warning("No pricing data available - continuing without pricing integration")
            return self._create_empty_pricing_integration(models)

        # Match models with pricing data
        integrated_data = {}
        matches_found = 0

        for model_id, model_data in models.items():
            pricing_info = self._find_model_pricing(model_id, model_data, pricing_data)

            integrated_data[model_id] = {
                'model_pricing': pricing_info,
                'pricing_metadata': {
                    'integration_source': 'amazon-bedrock-pricing-collector',
                    'has_pricing_data': pricing_info['is_pricing_available'],
                    'integration_timestamp': self._get_timestamp()
                }
            }

            if pricing_info['is_pricing_available']:
                matches_found += 1

        self.logger.info(f"✅ Integrated pricing for {matches_found}/{len(models)} models")
        return integrated_data

    def _load_latest_pricing_data(self) -> Optional[Dict[str, Any]]:
        """Load the latest pricing data file"""
        try:
            pricing_path = Path(self.pricing_collector_path)
            if not pricing_path.exists():
                self.logger.warning(f"Pricing collector path does not exist: {pricing_path}")
                return None

            # Find latest pricing file
            pricing_files = list(pricing_path.glob('bedrock-pricing-*.json'))
            if not pricing_files:
                self.logger.warning("No pricing files found")
                return None

            latest_file = max(pricing_files, key=lambda f: f.stat().st_mtime)
            self.logger.info(f"Loading pricing data from: {latest_file.name}")

            with open(latest_file, 'r') as f:
                data = json.load(f)

            self.logger.info(f"✅ Loaded pricing data with {len(data.get('providers', {}))} providers")
            return data

        except Exception as e:
            self.logger.error(f"Failed to load pricing data: {e}")
            return None

    def _find_model_pricing(self, model_id: str, model_data: Dict[str, Any], pricing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find pricing information for a specific model with comprehensive pricing structure"""
        model_name = model_data.get('model_name', '')
        model_provider = model_data.get('model_provider', '')

        # Search through pricing data structure
        providers = pricing_data.get('providers', {})

        for provider_name, provider_data in providers.items():
            if self._provider_matches(model_provider, provider_name):
                # Check if provider_data is the models dict directly or has a models key
                if isinstance(provider_data, dict):
                    # Try both structures: direct models or nested under 'models' key
                    models_data = provider_data.get('models', provider_data) if 'models' in provider_data else provider_data

                    for pricing_model_id, pricing_model_info in models_data.items():
                        if self._model_matches(model_id, model_name, pricing_model_id, pricing_model_info):
                            # Create comprehensive pricing structure organized by region
                            comprehensive_pricing = self._create_comprehensive_pricing_structure(pricing_model_info)

                            # Return the comprehensive pricing structure directly, with metadata
                            result = comprehensive_pricing.copy()
                            result['is_pricing_available'] = True
                            result['pricing_metadata'] = {
                                'matched_provider': provider_name,
                                'matched_model_id': pricing_model_id,
                                'model_name': pricing_model_info.get('model_name', ''),
                                'total_regions': pricing_model_info.get('total_regions', 0),
                                'total_pricing_entries': pricing_model_info.get('total_pricing_entries', 0)
                            }
                            return result

        return {'is_pricing_available': False}

    def _create_comprehensive_pricing_structure(self, pricing_model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive pricing structure organized by region and pricing type"""
        comprehensive_pricing = {
            'comprehensive_pricing': {},
            'summary': {
                'total_regions_with_pricing': 0,
                'total_pricing_dimensions': 0,
                'available_pricing_types': set(),
                'available_regions': [],
                'integration_timestamp': self._get_timestamp()
            }
        }

        # Extract regions data
        regions_data = pricing_model_info.get('regions', {})

        if not isinstance(regions_data, dict):
            return comprehensive_pricing

        for region_name, region_info in regions_data.items():
            if not isinstance(region_info, dict) or 'pricing_groups' not in region_info:
                continue

            pricing_groups = region_info['pricing_groups']
            region_pricing = {}

            # Organize pricing by type (on_demand, batch, etc.)
            for group_type, group_data in pricing_groups.items():
                if isinstance(group_data, list) and group_data:
                    # Categorize pricing dimensions
                    pricing_type = self._categorize_pricing_type(group_type, group_data)

                    if pricing_type not in region_pricing:
                        region_pricing[pricing_type] = {
                            'pricing_dimensions': {},
                            'total_entries': 0
                        }

                    # Process pricing dimensions
                    dimensions = self._process_pricing_dimensions(group_data)
                    region_pricing[pricing_type]['pricing_dimensions'][group_type] = dimensions
                    region_pricing[pricing_type]['total_entries'] += len(group_data)

                    comprehensive_pricing['summary']['available_pricing_types'].add(pricing_type)
                    comprehensive_pricing['summary']['total_pricing_dimensions'] += len(group_data)

            if region_pricing:
                comprehensive_pricing['comprehensive_pricing'][region_name] = region_pricing
                comprehensive_pricing['summary']['available_regions'].append(region_name)
                comprehensive_pricing['summary']['total_regions_with_pricing'] += 1

        # Convert set to sorted list for JSON serialization and sort regions
        comprehensive_pricing['summary']['available_pricing_types'] = sorted(list(comprehensive_pricing['summary']['available_pricing_types']))
        comprehensive_pricing['summary']['available_regions'] = sorted(comprehensive_pricing['summary']['available_regions'])

        return comprehensive_pricing

    def _categorize_pricing_type(self, group_type: str, group_data: List[Dict]) -> str:
        """Categorize pricing group into pricing types (on_demand, batch, etc.)"""
        group_type_lower = group_type.lower()

        # Check for batch pricing indicators
        if any(indicator in group_type_lower for indicator in ['batch', 'bulk', 'async']):
            return 'batch'

        # Check for provisioned throughput indicators
        if any(indicator in group_type_lower for indicator in ['provisioned', 'reserved', 'dedicated']):
            return 'provisioned'

        # Check for specialized pricing types
        if any(indicator in group_type_lower for indicator in ['training', 'fine-tune', 'custom']):
            return 'training'

        if any(indicator in group_type_lower for indicator in ['embed', 'vector', 'search']):
            return 'embedding'

        # Default to on_demand for standard token pricing
        return 'on_demand'

    def _process_pricing_dimensions(self, group_data: List[Dict]) -> Dict[str, Any]:
        """Process pricing dimensions from group data"""
        dimensions_summary = {
            'pricing_entries': group_data,
            'entry_count': len(group_data),
            'dimension_types': set(),
            'price_ranges': {
                'min_price': None,
                'max_price': None,
                'currency': 'USD'
            }
        }

        prices = []
        for entry in group_data:
            # Extract dimension type
            if isinstance(entry, dict):
                if 'dimension' in entry:
                    dimensions_summary['dimension_types'].add(entry['dimension'])
                elif 'pricing_dimension' in entry:
                    dimensions_summary['dimension_types'].add(entry['pricing_dimension'])

                # Extract pricing information for ranges
                price_value = None
                if 'price_per_unit' in entry:
                    try:
                        price_value = float(entry['price_per_unit'])
                    except (ValueError, TypeError):
                        pass
                elif 'price' in entry:
                    try:
                        price_value = float(entry['price'])
                    except (ValueError, TypeError):
                        pass

                if price_value is not None:
                    prices.append(price_value)

        # Calculate price ranges
        if prices:
            dimensions_summary['price_ranges']['min_price'] = min(prices)
            dimensions_summary['price_ranges']['max_price'] = max(prices)

        # Convert set to sorted list for JSON serialization
        dimensions_summary['dimension_types'] = sorted(list(dimensions_summary['dimension_types']))

        return dimensions_summary

    def _provider_matches(self, model_provider: str, pricing_provider: str) -> bool:
        """Check if model provider matches pricing provider"""
        if not model_provider or not pricing_provider:
            return False

        model_provider_lower = model_provider.lower()
        pricing_provider_lower = pricing_provider.lower()

        return (model_provider_lower == pricing_provider_lower or
                model_provider_lower in pricing_provider_lower or
                pricing_provider_lower in model_provider_lower)

    def _model_matches(self, model_id: str, model_name: str, pricing_model_name: str, pricing_data: Dict[str, Any]) -> bool:
        """Check if model matches pricing model"""
        # Try multiple matching strategies
        model_id_lower = model_id.lower()
        model_name_lower = model_name.lower()
        pricing_name_lower = pricing_model_name.lower()

        # Direct name match
        if model_name_lower == pricing_name_lower:
            return True

        # Model ID contains pricing name or vice versa
        if pricing_name_lower in model_id_lower or model_name_lower in pricing_name_lower:
            return True

        # Check pricing data for model ID references
        pricing_str = json.dumps(pricing_data).lower()
        if model_id_lower in pricing_str:
            return True

        return False

    def _create_empty_pricing_integration(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Create empty pricing integration when no data is available"""
        integrated_data = {}

        for model_id in models.keys():
            integrated_data[model_id] = {
                'is_pricing_available': False,
                'comprehensive_pricing': {},
                'summary': {
                    'total_regions_with_pricing': 0,
                    'total_pricing_dimensions': 0,
                    'available_pricing_types': [],
                    'available_regions': [],
                    'integration_timestamp': self._get_timestamp()
                },
                'pricing_metadata': {
                    'integration_source': 'none',
                    'has_pricing_data': False,
                    'integration_timestamp': self._get_timestamp(),
                    'reason': 'pricing_collector_data_not_available'
                }
            }

        return integrated_data

    def _get_timestamp(self) -> str:
        """Get current UTC timestamp"""
        import time
        return time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())