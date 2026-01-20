"""
Pricing Groups
Logic for organizing pricing dimensions into logical groups

Note: This file contains string constants like 'input_tokens', 'output_tokens', etc.
These are legitimate AWS pricing category names, NOT passwords.
The B105 bandit warnings in this file are false positives.
"""
# nosec B105 - The string constants in this file are AWS pricing categories, not passwords

import logging
import re
from typing import Dict, List
from collections import defaultdict


logger = logging.getLogger(__name__)


class PricingGroupsOrganizer:
    """Organizes pricing dimensions into logical groups"""


    def analyze_dimension_characteristics(self, dimension: str) -> Dict[str, str]:
        """
        Analyze a pricing dimension to determine its characteristics (simplified)

        Args:
            dimension: The pricing dimension string

        Returns:
            Dictionary of characteristics (only those used in group names)
        """
        dim_lower = dimension.lower()

        # Only track characteristics actually used in group name creation
        characteristics = {
            'inference_type': 'on_demand',
            'context_type': 'standard',
            'geographic_scope': 'regional'
        }

        # Determine inference type (simplified keyword matching)
        if 'batch' in dim_lower:
            characteristics['inference_type'] = 'batch'
        elif 'provisioned' in dim_lower:
            characteristics['inference_type'] = 'provisioned_throughput'
        elif 'custom' in dim_lower:
            characteristics['inference_type'] = 'custom_model'

        # Determine context type (simplified)
        if 'lctx' in dim_lower or 'longcontext' in dim_lower:
            characteristics['context_type'] = 'long_context'

        # Determine geographic scope (simplified)
        if 'global' in dim_lower:
            characteristics['geographic_scope'] = 'global'

        return characteristics

    def create_group_name(self, characteristics: Dict[str, str]) -> str:
        """
        Create a human-readable group name from characteristics (simplified)

        Args:
            characteristics: Dictionary of pricing characteristics

        Returns:
            Human-readable group name
        """
        parts = []

        # Map inference types to display names
        inference_map = {
            'on_demand': 'On-Demand',
            'batch': 'Batch',
            'provisioned_throughput': 'Provisioned Throughput',
            'custom_model': 'Custom Model'
        }

        inference_type = characteristics.get('inference_type', 'on_demand')
        parts.append(inference_map.get(inference_type, 'On-Demand'))

        # Add modifiers
        if characteristics.get('context_type') == 'long_context':
            parts.append('Long Context')

        if characteristics.get('geographic_scope') == 'global':
            parts.append('Global')

        return ' '.join(parts)

    def remove_duplicates(self, pricing_entries: List[Dict]) -> List[Dict]:
        """
        Remove duplicate pricing entries

        Args:
            pricing_entries: List of pricing entries

        Returns:
            List with duplicates removed
        """
        seen = set()
        unique_entries = []

        for entry in pricing_entries:
            # Create a unique key based on price, unit, and description
            key = (
                entry.get('price_per_thousand'),
                entry.get('unit'),
                entry.get('description')
            )

            if key not in seen:
                seen.add(key)
                unique_entries.append(entry)

        return unique_entries

    def organize_pricing_by_groups(self, pricing_entries: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Organize pricing entries into logical groups with deduplication

        Args:
            pricing_entries: List of pricing entries

        Returns:
            Dictionary with group names as keys and deduplicated pricing entries as values
        """
        # Remove duplicates first
        deduplicated_entries = self.remove_duplicates(pricing_entries)

        groups = defaultdict(list)

        for entry in deduplicated_entries:
            dimension = entry.get('dimension', '')

            # Analyze characteristics
            characteristics = self.analyze_dimension_characteristics(dimension)
            group_name = self.create_group_name(characteristics)

            # Add characteristics to entry
            entry['pricing_characteristics'] = characteristics
            entry['pricing_group'] = group_name

            # Add to group
            groups[group_name].append(entry)

        return dict(groups)

    def get_group_statistics(self, grouped_pricing: Dict[str, List[Dict]]) -> Dict:
        """
        Generate statistics about the pricing groups

        Args:
            grouped_pricing: Dictionary of grouped pricing entries

        Returns:
            Statistics dictionary
        """
        total_entries = sum(len(entries) for entries in grouped_pricing.values())
        group_count = len(grouped_pricing)

        group_sizes = {
            group_name: len(entries)
            for group_name, entries in grouped_pricing.items()
        }

        # Sort groups by size
        largest_groups = sorted(
            group_sizes.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'total_entries': total_entries,
            'total_groups': group_count,
            'group_sizes': group_sizes,
            'largest_groups': largest_groups[:5],  # Top 5 largest groups
            'average_entries_per_group': round(total_entries / group_count, 1) if group_count > 0 else 0
        }

    def organize_region_pricing(self, region_pricing_entries: List[Dict]) -> Dict:
        """
        Organize pricing entries for a single region into groups

        Args:
            region_pricing_entries: List of pricing entries for a region

        Returns:
            Dictionary with grouped pricing and metadata
        """
        if not region_pricing_entries:
            return {
                'pricing_groups': {},
                'total_dimensions': 0,
                'groups_count': 0,
                'group_statistics': {}
            }

        # Organize into groups
        grouped_pricing = self.organize_pricing_by_groups(region_pricing_entries)

        # Generate statistics
        statistics = self.get_group_statistics(grouped_pricing)

        return {
            'pricing_groups': grouped_pricing,
            'total_dimensions': len(region_pricing_entries),
            'groups_count': len(grouped_pricing),
            'group_statistics': statistics
        }