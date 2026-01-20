"""
Data Processor
Handles data standardization, region extraction, and processing pipeline
"""

import logging
import re
from typing import Dict, List
from collections import defaultdict
from datetime import datetime

from utils.pricing_groups import PricingGroupsOrganizer
from utils.smart_extractor import SmartModelExtractor
from config import Config


logger = logging.getLogger(__name__)


class BedrockPricingDataProcessor:
    """Processes and standardizes Bedrock pricing data"""

    def __init__(self):
        """Initialize the data processor"""
        self.pricing_organizer = PricingGroupsOrganizer()
        self.conversion_stats = {
            'conversions_applied': 0,
            'units_extracted': 0,
            'dimensions_enhanced': 0,
            'duplicates_removed': 0,
            'regions_standardized': 0,
            'commitment_lengths_added': 0,
            'custom_model_types_classified': 0
        }

    def extract_unit_from_description(self, description: str, current_unit: str) -> str:
        """
        Extract proper unit from description when current unit is too vague

        Args:
            description: Pricing description text
            current_unit: Current unit (often 'Units' or 'hour')

        Returns:
            Extracted specific unit or current unit if no better match found
        """
        if not description or current_unit not in ['Units', 'hour', 'Hour']:
            return current_unit

        desc_lower = description.lower()

        # Token-based units
        if 'token' in desc_lower:
            if 'million' in desc_lower:
                return '1M tokens'
            elif 'thousand' in desc_lower or '1k' in desc_lower:
                return '1K tokens'
            else:
                return 'tokens'

        # Time-based units from description
        if 'second' in desc_lower:
            return 'second'
        elif 'minute' in desc_lower:
            return 'minute'
        elif 'hour' in desc_lower and current_unit != 'hour':
            return 'hour'

        # Image/media units
        if 'image' in desc_lower:
            return 'image'

        # Keep original if no better match
        return current_unit

    def standardize_price_and_unit(self, price: float, unit: str, description: str) -> tuple:
        """
        Convert per-million pricing to per-thousand for consistency

        Args:
            price: Original price
            unit: Current unit
            description: Price description

        Returns:
            Tuple of (standardized_price, standardized_unit)
        """
        desc_lower = description.lower()

        # Check if this needs million → thousand conversion
        if 'million' in desc_lower and ('token' in desc_lower or unit in ['Units', '1M tokens']):
            # Convert from per-million to per-thousand
            standardized_price = price / 1000  # 1 million = 1000 thousands
            standardized_unit = '1K tokens'
            self.conversion_stats['conversions_applied'] += 1
            return standardized_price, standardized_unit

        # No conversion needed
        return price, unit

    def update_description_for_conversion(self, description: str, was_converted: bool) -> str:
        """
        Update description after million→thousand conversion

        Args:
            description: Original description
            was_converted: Whether conversion was applied

        Returns:
            Updated description
        """
        if not was_converted:
            return description

        # Replace million references with thousand in description
        desc = description
        desc = re.sub(r'\b(\d+\s*)?million\b', r'\1thousand', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\b1\s*thousand\b', '1K', desc, flags=re.IGNORECASE)

        return desc

    def detect_custom_model_type(self, description: str, dimension: str) -> str:
        """
        Distinguish between Custom Model Import vs Custom Model Training/Customization

        Args:
            description: Price description
            dimension: Price dimension

        Returns:
            'Custom Model Import', 'Custom Model Training', or 'Custom Model'
        """
        desc_lower = description.lower()
        dim_lower = dimension.lower()

        # Custom Model Import indicators (like Flan architecture models for inference/storage)
        import_indicators = [
            'flan architecture', 'llama architecture', 'inference for', 'storage for',
            'custom model unit per min for inference', 'custom model unit/month storage',
            'imported model', 'model import'
        ]

        # Custom Model Training/Customization indicators
        training_indicators = [
            'customization-training', 'customization-storage', 'fine', 'finetun',
            'training', 'custom training', 'model customization'
        ]

        # Check for import patterns
        if any(indicator in desc_lower or indicator in dim_lower for indicator in import_indicators):
            return 'Custom Model Import'

        # Check for training/customization patterns
        if any(indicator in desc_lower or indicator in dim_lower for indicator in training_indicators):
            return 'Custom Model Training'

        # Default to generic custom model
        return 'Custom Model'

    def enhance_unit_with_context(self, unit: str, description: str, model_id: str) -> str:
        """
        Enhance units with specific context from descriptions

        Args:
            unit: Current unit
            description: Full description
            model_id: Model identifier

        Returns:
            Enhanced unit with context
        """
        desc_lower = description.lower()

        # Nova Sonic: Distinguish speech vs text tokens
        if 'nova' in model_id.lower() and 'sonic' in model_id.lower() and unit == '1K tokens':
            if 'speech-input' in desc_lower or 'speech-output' in desc_lower:
                if 'input' in desc_lower:
                    return '1K speech input tokens'
                elif 'output' in desc_lower:
                    return '1K speech output tokens'
                else:
                    return '1K speech tokens'
            elif 'text-input' in desc_lower or 'text-output' in desc_lower:
                if 'input' in desc_lower:
                    return '1K text input tokens'
                elif 'output' in desc_lower:
                    return '1K text output tokens'
                else:
                    return '1K text tokens'

        # Luma: Add resolution/fps specs to video units
        if 'luma' in model_id.lower() and unit == 'second':
            if '540p' in desc_lower and '24fps' in desc_lower:
                return 'second (540p, 24fps)'
            elif '720p' in desc_lower and '24fps' in desc_lower:
                return 'second (720p, 24fps)'
            elif 'video' in desc_lower:
                return 'second (video)'

        return unit

    def extract_provisioned_commitment(self, description: str, dimension: str) -> str:
        """
        Extract commitment term from provisioned throughput descriptions

        Args:
            description: Price description
            dimension: Price dimension

        Returns:
            Commitment term (e.g., 'No Commitment', '1 Month', '6 months')
        """
        desc_lower = description.lower()
        dim_lower = dimension.lower()

        # Check for commitment terms
        if 'nocommit' in desc_lower or 'no-commit' in desc_lower:
            return 'No Commitment'
        elif '1month' in desc_lower or '1-month' in desc_lower:
            return '1 Month'
        elif '6months' in desc_lower or '6-months' in desc_lower:
            return '6 Months'
        elif '12months' in desc_lower or '12-months' in desc_lower or '1year' in desc_lower:
            return '12 Months'
        elif 'custom' in desc_lower:
            return 'Custom'

        return 'Standard'

    def standardize_region_name(self, region: str) -> str:
        """
        Convert location names to region codes

        Args:
            region: Region name (could be location format)

        Returns:
            Standardized region code
        """
        # If already a region code, return as-is
        if region and not ('(' in region and ')' in region):
            return region

        # Location to region code mapping
        location_mappings = {
            'US East (N. Virginia)': 'us-east-1',
            'US East (Ohio)': 'us-east-2',
            'US West (Oregon)': 'us-west-2',
            'US West (N. California)': 'us-west-1',
            'EU (Ireland)': 'eu-west-1',
            'EU (London)': 'eu-west-2',
            'EU (Paris)': 'eu-west-3',
            'EU (Frankfurt)': 'eu-central-1',
            'EU (Stockholm)': 'eu-north-1',
            'EU (Milan)': 'eu-south-1',
            'Asia Pacific (Tokyo)': 'ap-northeast-1',
            'Asia Pacific (Seoul)': 'ap-northeast-2',
            'Asia Pacific (Mumbai)': 'ap-south-1',
            'Asia Pacific (Singapore)': 'ap-southeast-1',
            'Asia Pacific (Sydney)': 'ap-southeast-2',
            'South America (Sao Paulo)': 'sa-east-1',
            'AWS GovCloud (US-West)': 'us-gov-west-1',
            'AWS GovCloud (US-East)': 'us-gov-east-1'
        }

        standardized = location_mappings.get(region, region)
        if standardized != region:
            self.conversion_stats['regions_standardized'] += 1

        return standardized

    def enhance_dimension_with_specs(self, dimension: str, description: str, model_id: str) -> str:
        """
        Enhance dimension with specific details from description

        Args:
            dimension: Original dimension
            description: Full description
            model_id: Model identifier

        Returns:
            Enhanced dimension with specifications
        """
        desc_lower = description.lower()

        # Add provisioned throughput commitment terms
        if 'provisioned' in desc_lower:
            commitment = self.extract_provisioned_commitment(description, dimension)
            if commitment != 'Standard':
                enhanced = f"{dimension} ({commitment})"
                self.conversion_stats['dimensions_enhanced'] += 1
                return enhanced

        # Add resolution specs for video models
        if 'luma' in model_id.lower() and 'video' in desc_lower:
            if '540p' in desc_lower and '24fps' in desc_lower:
                enhanced = f"{dimension} (540p, 24fps)"
                self.conversion_stats['dimensions_enhanced'] += 1
                return enhanced
            elif '720p' in desc_lower and '24fps' in desc_lower:
                enhanced = f"{dimension} (720p, 24fps)"
                self.conversion_stats['dimensions_enhanced'] += 1
                return enhanced

        # Add speech/text context for Nova Sonic
        if 'nova' in model_id.lower() and 'sonic' in model_id.lower():
            if 'speech-input' in desc_lower:
                enhanced = f"{dimension} (Speech Input)"
                self.conversion_stats['dimensions_enhanced'] += 1
                return enhanced
            elif 'speech-output' in desc_lower:
                enhanced = f"{dimension} (Speech Output)"
                self.conversion_stats['dimensions_enhanced'] += 1
                return enhanced
            elif 'text-input' in desc_lower:
                enhanced = f"{dimension} (Text Input)"
                self.conversion_stats['dimensions_enhanced'] += 1
                return enhanced
            elif 'text-output' in desc_lower:
                enhanced = f"{dimension} (Text Output)"
                self.conversion_stats['dimensions_enhanced'] += 1
                return enhanced

        return dimension

    def is_duplicate_entry(self, entry1: Dict, entry2: Dict) -> bool:
        """
        Check if two entries are true duplicates

        Args:
            entry1: First pricing entry
            entry2: Second pricing entry

        Returns:
            True if entries are duplicates
        """
        # Same price, unit, and description = duplicate
        return (
            entry1.get('price_per_thousand') == entry2.get('price_per_thousand') and
            entry1.get('unit') == entry2.get('unit') and
            entry1.get('description') == entry2.get('description')
        )

    def extract_region_from_entry(self, entry: Dict) -> str:
        """
        Extract region from pricing entry

        Args:
            entry: Pricing entry dictionary

        Returns:
            Extracted region code
        """
        # Check direct region field
        if 'region' in entry:
            return entry['region']

        # Check location field and map to region codes
        if 'location' in entry:
            location = entry['location']
            return Config.LOCATION_TO_REGION_MAP.get(location, location)

        # Check dimension for region codes
        dimension = entry.get('dimension', '')
        region_match = re.search(r'^([A-Z]{2,4}\d?)[-_]', dimension)
        if region_match:
            region_code = region_match.group(1)
            # Map common region codes
            return Config.REGION_CODE_MAP.get(region_code, region_code.lower())

        # Default to global if no region found
        return 'global'

    def process_pricing_entry(self, entry: Dict) -> Dict:
        """
        Process a pricing entry with comprehensive enhancements

        Args:
            entry: Raw pricing entry from AWS Pricing API

        Returns:
            Enhanced and processed pricing entry
        """
        original_price = entry.get('original_price', 0.0)
        unit = entry.get('unit', 'unknown')
        description = entry.get('description', '')
        dimension = entry.get('dimension', '')
        model_id = entry.get('model_id', 'unknown')

        # Smart unit extraction for vague units
        if unit in ['Units', 'hour', 'Hour']:
            extracted_unit = self.extract_unit_from_description(description, unit)
            if extracted_unit != unit:
                unit = extracted_unit
                self.conversion_stats['units_extracted'] += 1

        # Smart price standardization (million → thousand conversion when needed)
        standardized_price, standardized_unit = self.standardize_price_and_unit(original_price, unit, description)
        was_converted = standardized_price != original_price

        # Enhance unit with context (Nova Sonic speech/text, Luma resolution, etc.)
        enhanced_unit = self.enhance_unit_with_context(standardized_unit, description, model_id)

        # Enhance dimension with specifications (commitment terms, resolution, input types)
        enhanced_dimension = self.enhance_dimension_with_specs(dimension, description, model_id)

        # Update description if conversion was applied
        final_description = self.update_description_for_conversion(description, was_converted)

        # Detect custom model type early (before provider assignment)
        custom_model_type = None
        if 'custom' in description.lower() or 'custom' in dimension.lower():
            custom_model_type = self.detect_custom_model_type(description, dimension)
            self.conversion_stats['custom_model_types_classified'] += 1

        # Create processed entry
        processed_entry = {
            'dimension': enhanced_dimension,
            'price_per_thousand': round(standardized_price, 6),
            'original_price': original_price,
            'unit': enhanced_unit,
            'description': final_description,
            'source_dataset': entry.get('source_dataset', 'unknown')
        }

        # Add commitment_length for provisioned throughput models
        if 'provisioned' in description.lower():
            commitment = self.extract_provisioned_commitment(description, dimension)
            if commitment != 'Standard':
                processed_entry['commitment_length'] = commitment
                self.conversion_stats['commitment_lengths_added'] += 1

        # Add custom_model_type if detected
        if custom_model_type:
            processed_entry['custom_model_type'] = custom_model_type

        # Add additional fields if present
        for field in ['model_id', 'model_name', 'provider', 'model_provider', 'location', 'operation', 'service_code']:
            if field in entry:
                processed_entry[field] = entry[field]

        return processed_entry

    def organize_pricing_data(self, all_pricing_data: List[Dict]) -> Dict:
        """
        Organize all pricing data into the final structure

        Args:
            all_pricing_data: List of all pricing entries

        Returns:
            Organized pricing data in final structure
        """
        logger.info(f"Organizing {len(all_pricing_data)} pricing entries")

        # Group by provider and model
        providers = defaultdict(lambda: defaultdict(list))

        processed_count = 0

        for entry in all_pricing_data:
            # Process entry (AWS API data is already clean and standardized)
            processed_entry = self.process_pricing_entry(entry)

            # Extract and standardize region
            region = self.extract_region_from_entry(entry)
            standardized_region = self.standardize_region_name(region)

            # Group by provider and model (with special handling)
            provider = processed_entry.get('provider', 'Unknown')
            model_id = processed_entry.get('model_id', 'unknown')

            # Consolidate Mistral providers under 'Mistral'
            if provider == 'Mistral AI':
                provider = 'Mistral'

            # Move Custom Model Import entries to dedicated provider
            custom_type = processed_entry.get('custom_model_type')
            if custom_type == 'Custom Model Import':
                provider = 'Custom Model Import'

            providers[provider][model_id].append((standardized_region, processed_entry))
            processed_count += 1

        logger.info(f"Processed {processed_count} entries - Conversions: {self.conversion_stats['conversions_applied']}, Units extracted: {self.conversion_stats['units_extracted']}, Dimensions enhanced: {self.conversion_stats['dimensions_enhanced']}, Regions standardized: {self.conversion_stats['regions_standardized']}, Commitment lengths: {self.conversion_stats['commitment_lengths_added']}, Custom models classified: {self.conversion_stats['custom_model_types_classified']}")


        # Organize by regions and create pricing groups
        final_structure = {}

        # Collect all unknown/cryptic providers under "Unknown Models"
        unknown_models = {}
        for provider, models in providers.items():
            if self._is_unknown_provider(provider):
                unknown_models.update(models)

        # Add unknown models group if any exist
        if unknown_models:
            final_structure['Unknown Models'] = {}
            for model_id, entries in unknown_models.items():
                # Process unknown models the same way as known providers
                regions = defaultdict(list)
                for region, processed_entry in entries:
                    regions[region].append(processed_entry)

                model_regions = {}
                for region, region_entries in regions.items():
                    region_organized = self.pricing_organizer.organize_region_pricing(region_entries)
                    model_regions[region] = region_organized

                model_name = entries[0][1].get('model_name', 'Unknown Model') if entries else 'Unknown Model'
                final_structure['Unknown Models'][model_id] = {
                    'model_name': model_name,
                    'model_provider': 'Unknown Models',
                    'regions': model_regions,
                    'total_regions': len(model_regions),
                    'total_pricing_entries': len(entries)
                }

        # Process known providers
        for provider, models in providers.items():
            if self._is_unknown_provider(provider):
                continue  # Already handled above

            # Special handling for Amazon provider with sub-grouping
            if provider == 'Amazon':
                amazon_groups = self._organize_amazon_models(models)

                # Create main Amazon provider entry with sub-groups
                amazon_structure = {}
                for group_name, group_models in amazon_groups.items():
                    for model_id, entries in group_models.items():
                        regions = defaultdict(list)
                        for region, processed_entry in entries:
                            regions[region].append(processed_entry)

                        model_regions = {}
                        for region, region_entries in regions.items():
                            region_organized = self.pricing_organizer.organize_region_pricing(region_entries)
                            model_regions[region] = region_organized

                        model_name = entries[0][1].get('model_name', 'Unknown Model') if entries else 'Unknown Model'
                        amazon_structure[model_id] = {
                            'model_name': model_name,
                            'model_provider': 'Amazon',
                            'model_group': group_name,
                            'regions': model_regions,
                            'total_regions': len(model_regions),
                            'total_pricing_entries': len(entries)
                        }

                final_structure['Amazon'] = amazon_structure
            else:
                # Regular provider processing
                final_structure[provider] = {}
                for model_id, entries in models.items():
                    regions = defaultdict(list)
                    for region, processed_entry in entries:
                        regions[region].append(processed_entry)

                    model_regions = {}
                    for region, region_entries in regions.items():
                        region_organized = self.pricing_organizer.organize_region_pricing(region_entries)
                        model_regions[region] = region_organized

                    model_name = entries[0][1].get('model_name', 'Unknown Model') if entries else 'Unknown Model'
                    model_provider = entries[0][1].get('model_provider', provider) if entries else provider
                    final_structure[provider][model_id] = {
                        'model_name': model_name,
                        'model_provider': model_provider,
                        'regions': model_regions,
                        'total_regions': len(model_regions),
                        'total_pricing_entries': len(entries)
                    }

        return final_structure

    def _is_unknown_provider(self, provider: str) -> bool:
        """
        Determine if a provider should be grouped under "Unknown Models"
        Uses flexible matching to handle provider name variations

        Args:
            provider: Provider name to check

        Returns:
            True if provider should be considered unknown
        """
        # If provider is explicitly "Unknown" or variations
        if provider in ['Unknown', 'Unknown Provider', 'Unknown Models']:
            return True

        # Use flexible keyword matching instead of exact SmartModelExtractor formatting
        # This handles cases where extracted provider names don't exactly match formatting
        provider_lower = provider.lower()

        # List of known provider patterns (case-insensitive)
        known_patterns = [
            'anthropic', 'meta', 'amazon', 'stability', 'ai21', 'cohere',
            'mistral', 'openai', 'deepseek', 'qwen', 'writer', 'twelve', 'luma',
            'custom model import'  # Add our new provider
        ]

        # Check if any known pattern matches the provider name
        for pattern in known_patterns:
            if pattern in provider_lower:
                return False  # It's a known provider

        return True  # Unknown provider

    def _organize_amazon_models(self, amazon_models: Dict) -> Dict:
        """
        Organize Amazon models into sub-groups based on model families

        Args:
            amazon_models: Dictionary of Amazon model entries

        Returns:
            Dictionary with organized Amazon sub-groups
        """
        amazon_groups = defaultdict(dict)

        for model_id, entries in amazon_models.items():
            model_id_lower = model_id.lower()

            # Determine group based on model family keywords
            if 'nova' in model_id_lower:
                amazon_groups['Amazon Nova'][model_id] = entries
            elif 'titan' in model_id_lower:
                amazon_groups['Amazon Titan'][model_id] = entries
            else:
                amazon_groups['Other Amazon'][model_id] = entries

        # Convert defaultdict back to regular dict and filter out empty groups
        return {k: dict(v) for k, v in amazon_groups.items() if v}

    def create_final_structure(self, all_pricing_data: List[Dict], data_sources_stats: Dict) -> Dict:
        """
        Create the final JSON structure with metadata

        Args:
            all_pricing_data: List of all pricing entries
            data_sources_stats: Statistics from data sources

        Returns:
            Complete final structure
        """
        # Organize pricing data
        organized_data = self.organize_pricing_data(all_pricing_data)

        # Generate statistics
        total_regions = 0
        total_groups = 0
        all_group_types = set()

        for provider, models in organized_data.items():
            for model_id, model_data in models.items():
                for region, region_data in model_data['regions'].items():
                    total_regions += 1
                    groups = region_data.get('pricing_groups', {})
                    total_groups += len(groups)
                    all_group_types.update(groups.keys())

        # Create final structure
        final_structure = {
            'metadata': {
                'generated_at': datetime.utcnow().isoformat() + 'Z',
                'version': Config.VERSION,
                'total_pricing_entries': len(all_pricing_data),
                'data_sources': data_sources_stats,
                'providers_count': len(organized_data),
                'total_regions_processed': total_regions,
                'total_groups_created': total_groups,
                'unique_group_types': len(all_group_types),
                'average_groups_per_region': round(total_groups / total_regions, 1) if total_regions > 0 else 0,
                'currency': Config.CURRENCY,
                'pricing_standardization': Config.PRICING_STANDARDIZATION_MESSAGE,
                'structure': Config.STRUCTURE_DESCRIPTION,
                'group_types_available': sorted(list(all_group_types))
            },
            'providers': organized_data
        }

        return final_structure