"""
UI pages for the Amazon Bedrock Model Explorer application.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from utils.common import (
    get_provider_badge_class,
    get_provider_color,
    format_price,
)
from ui.filters import get_region_display_name
from ui.converse_data_helpers import get_converse_context_window, get_converse_max_output_tokens

def get_cross_region_inference_support(model_data):
    """
    Extract cross-region inference support from model data.

    Args:
        model_data: Full model dictionary or pandas Series containing cross_region_inference

    Returns:
        bool: True if cross-region inference is supported, False otherwise
    """
    try:
        # Handle both dict and pandas Series
        if hasattr(model_data, 'get'):  # Works for both dict and Series
            data = model_data
        elif hasattr(model_data, 'to_dict'):  # pandas Series
            data = model_data
        else:
            return False

        # Get cross_region_inference data
        cross_region_inference = data.get('cross_region_inference', {})

        # Handle pandas Series NaN values and ensure we have a dict
        if cross_region_inference is None or str(cross_region_inference) == 'nan':
            return False

        # If it's a string representation, try to handle it
        if isinstance(cross_region_inference, str) and cross_region_inference.strip() == '{}':
            return False

        # Check if it's a proper dict with the supported field
        if isinstance(cross_region_inference, dict):
            supported = cross_region_inference.get('supported', False)
            # Ensure supported is actually a boolean and not NaN
            if supported is True or supported == 'True' or supported == 'true':
                return True
            elif supported is False or supported == 'False' or supported == 'false':
                return False

        return False

    except Exception as e:
        # In case of any error, default to False
        return False


def get_consumption_options(model_data):
    """
    Extract consumption options from model data, including batch inference support.
    
    Args:
        model_data: Full model dictionary or pandas Series containing model information
        
    Returns:
        list: List of consumption options available for the model
    """
    # Handle both dict and pandas Series
    if hasattr(model_data, 'get'):  # Works for both dict and Series
        data = model_data
    elif hasattr(model_data, 'to_dict'):  # pandas Series
        data = model_data
    else:
        return ['on_demand']
    
    # Start with the base consumption options from the model
    consumption_options = data.get('consumption_options', ['on_demand']).copy()
    
    # Remove cross_region_inference as it's handled separately
    consumption_options = [opt for opt in consumption_options if opt != 'cross_region_inference']
    
    # Check batch inference support from the new field
    batch_inference = data.get('batch_inference_supported', {})
    if isinstance(batch_inference, dict) and batch_inference.get('supported', False):
        if 'batch' not in consumption_options:
            consumption_options.append('batch')
    else:
        # Remove batch if it exists but is not supported
        consumption_options = [opt for opt in consumption_options if opt != 'batch']
    
    return consumption_options


def format_quota_value(value):
    """
    Safely format quota values, handling both numeric and string types.
    
    Args:
        value: Quota value (int, float, or string)
        
    Returns:
        str: Formatted quota value or "N/A"
    """
    if isinstance(value, (int, float)) and value > 0:
        return f"{value:,}"
    elif isinstance(value, str) and value.replace(',', '').replace('.', '').isdigit():
        try:
            numeric_val = float(value.replace(',', ''))
            return f"{numeric_val:,.0f}" if numeric_val > 0 else "N/A"
        except:
            return "N/A"
    else:
        return "N/A"

def extract_quota_info_for_region(model_data, preferred_region=None):
    """
    Extract quota information for a specific region.
    
    Args:
        model_data: Full model dictionary or pandas Series containing quota_limits
        preferred_region: Preferred AWS region to extract quotas from
        
    Returns:
        dict: Dictionary with tokens_per_minute and requests_per_minute for the region
    """
    # Handle both dict and pandas Series
    if hasattr(model_data, 'get'):  # Works for both dict and Series
        data = model_data
    elif hasattr(model_data, 'to_dict'):  # pandas Series
        data = model_data
    else:
        return {'tokens_per_minute': 'N/A', 'requests_per_minute': 'N/A'}
    
    quota_limits = data.get('quota_limits', {})
    if not quota_limits or not isinstance(quota_limits, dict):
        return {'tokens_per_minute': 'N/A', 'requests_per_minute': 'N/A'}
    
    regions_data = quota_limits.get('regions', {})
    if not regions_data:
        # Fallback to legacy quota_limits structure
        tokens_per_min = quota_limits.get('tokens_per_minute', 'N/A')
        requests_per_min = quota_limits.get('requests_per_minute', 'N/A')
        return {'tokens_per_minute': tokens_per_min, 'requests_per_minute': requests_per_min}
    
    # If preferred region is specified and available, try it first
    if preferred_region and preferred_region in regions_data:
        region_quotas = regions_data[preferred_region]
        if isinstance(region_quotas, dict):
            # Look for on-demand quotas in the region
            for quota_code, quota_info in region_quotas.items():
                if isinstance(quota_info, dict):
                    quota_name = quota_info.get('name', '').lower()
                    if 'on-demand' in quota_name and 'tokens per minute' in quota_name:
                        return {
                            'tokens_per_minute': quota_info.get('value', 'N/A'),
                            'requests_per_minute': 'N/A'  # Will be filled if found
                        }
    
    # Fallback to any available region data
    for region, region_quotas in regions_data.items():
        if isinstance(region_quotas, dict):
            tokens_per_min = 'N/A'
            requests_per_min = 'N/A'
            
            for quota_code, quota_info in region_quotas.items():
                if isinstance(quota_info, dict):
                    quota_name = quota_info.get('name', '').lower()
                    if 'tokens per minute' in quota_name:
                        tokens_per_min = quota_info.get('value', 'N/A')
                    elif 'requests per minute' in quota_name:
                        requests_per_min = quota_info.get('value', 'N/A')
            
            if tokens_per_min != 'N/A' or requests_per_min != 'N/A':
                return {'tokens_per_minute': tokens_per_min, 'requests_per_minute': requests_per_min}
    
    return {'tokens_per_minute': 'N/A', 'requests_per_minute': 'N/A'}


def extract_available_regions_from_pricing(model_data):
    """
    Extract available regions from model pricing data.

    Args:
        model_data: Full model dictionary or pandas Series containing model_pricing

    Returns:
        list: List of regions where the model has pricing data available
    """
    # Handle both dict and pandas Series
    if hasattr(model_data, 'get'):  # Works for both dict and Series
        data = model_data
    elif hasattr(model_data, 'to_dict'):  # pandas Series
        data = model_data
    else:
        return []

    # Get pricing from the new structure: model_pricing.comprehensive_pricing
    model_pricing = data.get('model_pricing', {})
    if not model_pricing:
        # Fallback to legacy regions field if no pricing data
        return data.get('regions', [])

    comprehensive_pricing = model_pricing.get('comprehensive_pricing', {})
    if not comprehensive_pricing:
        # Fallback to legacy regions field if no comprehensive pricing
        return data.get('regions', [])

    # Return the keys (regions) from comprehensive_pricing
    available_regions = list(comprehensive_pricing.keys())

    # If no regions found in pricing, fallback to legacy regions field
    if not available_regions:
        return data.get('regions', [])

    return available_regions


def extract_comprehensive_pricing_info(model_data, preferred_region=None):
    """
    Extract pricing information from multiple sources with primary region support.

    Args:
        model_data: Full model dictionary or pandas Series containing model_pricing
        preferred_region: Preferred AWS region to extract pricing from

    Returns:
        tuple: (input_price, output_price, region_used) per 1K tokens
    """
    # Handle both dict and pandas Series
    if hasattr(model_data, 'get'):  # Works for both dict and Series
        data = model_data
    elif hasattr(model_data, 'to_dict'):  # pandas Series
        data = model_data
    else:
        return 0, 0, None

    # Set default region if none provided
    if not preferred_region:
        preferred_region = 'us-east-1'

    # Try detailed_pricing first (from pricing resolution)
    detailed_pricing = data.get('detailed_pricing', {})
    if isinstance(detailed_pricing, dict) and detailed_pricing:
        input_price, output_price, region = _extract_from_detailed_pricing(detailed_pricing, preferred_region)
        if input_price > 0 or output_price > 0:
            return input_price, output_price, region

    # Try comprehensive_pricing (legacy structure)
    model_pricing = data.get('model_pricing', {})
    if isinstance(model_pricing, dict):
        comprehensive_pricing = model_pricing.get('comprehensive_pricing', {})
        if isinstance(comprehensive_pricing, dict) and comprehensive_pricing:
            input_price, output_price, region = _extract_from_comprehensive_pricing(comprehensive_pricing, preferred_region)
            if input_price > 0 or output_price > 0:
                return input_price, output_price, region

    # Fallback to direct pricing resolution
    try:
        return _resolve_pricing_from_reference(data, preferred_region)
    except:
        return 0, 0, None


def _extract_from_detailed_pricing(detailed_pricing, preferred_region):
    """Extract pricing from detailed_pricing structure"""
    regions = detailed_pricing.get('regions', {})
    if not regions:
        return 0, 0, None

    # Try preferred region first
    if preferred_region in regions:
        region_data = regions[preferred_region]
        input_price, output_price = _extract_pricing_from_region_data(region_data)
        if input_price > 0 or output_price > 0:
            return input_price, output_price, preferred_region

    # Try other regions if preferred not found
    priority_regions = ['us-east-1', 'us-west-2', 'eu-west-1']
    for region in priority_regions:
        if region != preferred_region and region in regions:
            region_data = regions[region]
            input_price, output_price = _extract_pricing_from_region_data(region_data)
            if input_price > 0 or output_price > 0:
                return input_price, output_price, region

    return 0, 0, None


def _extract_pricing_from_region_data(region_data):
    """Extract input and output pricing from region data structure"""
    input_price = 0
    output_price = 0

    if not isinstance(region_data, dict):
        return input_price, output_price

    # Look for on_demand pricing first
    on_demand = region_data.get('on_demand', {})
    if isinstance(on_demand, dict):
        # Try direct token pricing fields
        input_price = on_demand.get('input_tokens', 0)
        output_price = on_demand.get('output_tokens', 0)

        # If no direct pricing, look in pricing_dimensions
        if input_price == 0 and output_price == 0:
            pricing_dimensions = on_demand.get('pricing_dimensions', {})
            if isinstance(pricing_dimensions, dict):
                # Look for standard On-Demand dimension
                for dim_name, dim_data in pricing_dimensions.items():
                    if 'on-demand' in dim_name.lower() or dim_name in ['On-Demand', 'OnDemand']:
                        entries = dim_data.get('pricing_entries', [])
                        for entry in entries:
                            description = entry.get('description', '').lower()
                            price = entry.get('price_per_thousand', 0)

                            # Match input tokens
                            if any(term in description for term in ['input tokens', 'input token', 'input']):
                                if 'long context' not in description and 'global' not in description:
                                    input_price = max(input_price, price)

                            # Match output tokens
                            elif any(term in description for term in ['output tokens', 'output token', 'response tokens', 'response']):
                                if 'long context' not in description and 'global' not in description:
                                    output_price = max(output_price, price)

                        # If we found prices in this dimension, break
                        if input_price > 0 or output_price > 0:
                            break

    return input_price, output_price


def _resolve_pricing_from_reference(model_data, preferred_region):
    """Resolve pricing from reference system using NewModelRepository"""
    try:
        # Import here to avoid circular imports
        from models.new_model_repository import NewModelRepository

        model_id = model_data.get('model_id', '')
        if not model_id:
            return 0, 0, None

        # Initialize repository and get pricing details
        repo = NewModelRepository()

        # Get the pricing reference from the model data
        model_pricing = model_data.get('model_pricing', {})
        if isinstance(model_pricing, dict):
            pricing_reference = model_pricing  # Remove double-nesting - model_pricing already contains the data
        else:
            return 0, 0, None

        if not isinstance(pricing_reference, dict) or not pricing_reference.get('is_pricing_available'):
            return 0, 0, None

        pricing_details = repo.get_model_pricing_details(pricing_reference)

        if not pricing_details or not isinstance(pricing_details, dict):
            return 0, 0, None

        # The new pricing structure: regions > pricing_groups > dimensions
        regions_data = pricing_details.get('regions', pricing_details)

        # Try preferred region first
        if preferred_region and preferred_region in regions_data:
            input_price, output_price = _extract_from_pricing_region(regions_data[preferred_region])
            if input_price > 0 or output_price > 0:
                return input_price, output_price, preferred_region

        # Try other priority regions
        priority_regions = ['us-east-1', 'us-west-2', 'eu-west-1']
        for region in priority_regions:
            if region != preferred_region and region in regions_data:
                input_price, output_price = _extract_from_pricing_region(regions_data[region])
                if input_price > 0 or output_price > 0:
                    return input_price, output_price, region

        # Try any available region
        for region, region_data in regions_data.items():
            input_price, output_price = _extract_from_pricing_region(region_data)
            if input_price > 0 or output_price > 0:
                return input_price, output_price, region

        return 0, 0, None

    except Exception as e:
        import logging
        logging.warning(f"Error resolving pricing from reference for model {model_data.get('model_id', '')}: {e}")
        return 0, 0, None


def _extract_from_pricing_region(region_data):
    """Extract clean input and output pricing from new pricing collector format"""
    input_price = 0
    output_price = 0

    if not isinstance(region_data, dict):
        return input_price, output_price

    # Look in pricing_groups for On-Demand pricing
    pricing_groups = region_data.get('pricing_groups', {})
    on_demand_group = pricing_groups.get('On-Demand', [])

    if isinstance(on_demand_group, list):
        for dimension in on_demand_group:
            if isinstance(dimension, dict):
                description = dimension.get('description', '').lower()
                price_per_thousand = dimension.get('price_per_thousand', 0)

                # Skip all non-standard pricing types
                excluded_terms = [
                    'long context', 'global', 'cache', 'cached', 'batch', 'training',
                    'provisioned', 'reserved', 'spot', 'commitment', 'upfront',
                    'storage', 'custom', 'fine-tuning', 'embedding'
                ]

                # Skip if description contains any excluded terms
                if any(term in description for term in excluded_terms):
                    continue

                # Match clean input tokens - be very specific
                if any(exact_match in description for exact_match in ['input tokens', 'input token']):
                    # Ensure it's not output-related
                    if 'output' not in description and 'response' not in description:
                        input_price = max(input_price, price_per_thousand)

                # Match clean output tokens - be very specific
                elif any(exact_match in description for exact_match in ['output tokens', 'output token', 'response tokens']):
                    # Ensure it's not input-related
                    if 'input' not in description:
                        output_price = max(output_price, price_per_thousand)

                # Alternative matching for simpler descriptions
                elif 'input' in description and 'token' in description:
                    # Double check it's clean standard pricing
                    if ('output' not in description and
                        'response' not in description and
                        all(term not in description for term in excluded_terms)):
                        input_price = max(input_price, price_per_thousand)

                elif ('output' in description or 'response' in description) and 'token' in description:
                    # Double check it's clean standard pricing
                    if ('input' not in description and
                        all(term not in description for term in excluded_terms)):
                        output_price = max(output_price, price_per_thousand)

                # Handle image/video pricing (for non-text models)
                elif ('image' in description or 'video' in description):
                    # For image/video models, show the pricing as "input" since it's the cost per generation
                    if all(term not in description for term in excluded_terms):
                        # Use input_price to store per-image/per-video pricing
                        input_price = max(input_price, price_per_thousand)

    return input_price, output_price


def _extract_from_comprehensive_pricing(comprehensive_pricing, preferred_region):
    """Extract pricing from comprehensive_pricing structure (legacy)"""

    # In the new structure, regions are top-level keys in comprehensive_pricing
    def extract_pricing_from_new_region(region_data):
        """Extract input and output pricing from the new region structure"""
        input_price = 0
        output_price = 0

        # Look for on_demand pricing first
        on_demand = region_data.get('on_demand', {})
        if on_demand and 'pricing_dimensions' in on_demand:
            dimensions = on_demand['pricing_dimensions']

            # Look for standard On-Demand dimension first
            target_dimensions = ['On-Demand', 'OnDemand']
            for dim_name in target_dimensions:
                if dim_name in dimensions:
                    entries = dimensions[dim_name].get('pricing_entries', [])
                    for entry in entries:
                        description = entry.get('description', '').lower()
                        price = entry.get('price_per_thousand', 0)

                        # Match input tokens
                        if any(term in description for term in ['input tokens', 'input token', 'input']):
                            if 'long context' not in description and 'global' not in description:
                                input_price = max(input_price, price)  # Take highest if multiple

                        # Match output tokens
                        elif any(term in description for term in ['output tokens', 'output token', 'response tokens', 'response']):
                            if 'long context' not in description and 'global' not in description:
                                output_price = max(output_price, price)  # Take highest if multiple

                    # If we found prices in this dimension, break
                    if input_price > 0 or output_price > 0:
                        break

            # If no standard pricing found, try any On-Demand dimension
            if input_price == 0 and output_price == 0:
                for dim_name, dim_data in dimensions.items():
                    if 'on-demand' in dim_name.lower() or 'ondemand' in dim_name.lower():
                        entries = dim_data.get('pricing_entries', [])
                        for entry in entries:
                            description = entry.get('description', '').lower()
                            price = entry.get('price_per_thousand', 0)

                            # Match input tokens
                            if any(term in description for term in ['input tokens', 'input token', 'input']):
                                input_price = max(input_price, price)

                            # Match output tokens
                            elif any(term in description for term in ['output tokens', 'output token', 'response tokens', 'response']):
                                output_price = max(output_price, price)

                        # If we found prices, break
                        if input_price > 0 or output_price > 0:
                            break

        return input_price, output_price

    def extract_non_token_pricing_from_region(region_data):
        """Extract non-token pricing (image, video, etc.) from region structure"""
        image_price = 0
        video_price = 0
        other_price = 0
        other_unit = ""

        # Look for on_demand pricing first
        on_demand = region_data.get('on_demand', {})
        if on_demand and 'pricing_dimensions' in on_demand:
            dimensions = on_demand['pricing_dimensions']

            # Look through all dimensions for non-token pricing
            for dim_name, dim_data in dimensions.items():
                if 'on-demand' in dim_name.lower() or 'ondemand' in dim_name.lower() or dim_name in ['On-Demand', 'OnDemand']:
                    entries = dim_data.get('pricing_entries', [])
                    for entry in entries:
                        description = entry.get('description', '').lower()
                        price = entry.get('price_per_thousand', 0)
                        unit = entry.get('unit', '').lower()

                        # Match image pricing
                        if 'image' in description or 'image' in unit:
                            image_price = max(image_price, price)

                        # Match video pricing
                        elif 'video' in description or 'video' in unit:
                            video_price = max(video_price, price)

                        # Match other pricing (audio, document, etc.)
                        elif price > 0 and not any(term in description for term in ['input tokens', 'output tokens', 'token']):
                            other_price = max(other_price, price)
                            other_unit = entry.get('unit', 'unit')

        return image_price, video_price, other_price, other_unit

    # If preferred region is specified and available, try it first
    if preferred_region and preferred_region in comprehensive_pricing:
        region_data = comprehensive_pricing[preferred_region]
        input_price, output_price = extract_pricing_from_new_region(region_data)

        if input_price > 0 or output_price > 0:
            return input_price, output_price, preferred_region

    # If user has selected a specific region but no pricing is available for it,
    # return 0, 0, None to show "check pricing details" message
    if preferred_region and preferred_region != 'us-east-1':
        # User explicitly selected a non-default region, respect their choice
        # Don't fallback to US regions automatically
        return 0, 0, None

    # Only fall back to other regions if no preferred region was specified or if it was the default us-east-1
    # Fallback to priority order: us-east-1, then us-west-2, then any other US region, then any region
    priority_regions = ['us-east-1', 'us-west-2']

    # First try priority US regions
    for region in priority_regions:
        if region in comprehensive_pricing:
            region_data = comprehensive_pricing[region]
            input_price, output_price = extract_pricing_from_new_region(region_data)

            if input_price > 0 or output_price > 0:
                return input_price, output_price, region

    # Then try any other US region
    us_regions = [r for r in comprehensive_pricing.keys() if r.startswith('us-')]
    for region in us_regions:
        if region not in priority_regions:  # Skip already checked regions
            region_data = comprehensive_pricing[region]
            input_price, output_price = extract_pricing_from_new_region(region_data)

            if input_price > 0 or output_price > 0:
                return input_price, output_price, region

    # Finally, try any region with pricing data
    for region, region_data in comprehensive_pricing.items():
        input_price, output_price = extract_pricing_from_new_region(region_data)

        if input_price > 0 or output_price > 0:
            return input_price, output_price, region

    # If no token pricing found, try to get image/video pricing
    for region, region_data in comprehensive_pricing.items():
        image_price, video_price, other_price, other_unit = extract_non_token_pricing_from_region(region_data)

        if image_price > 0:
            return image_price, 0, region  # Return image price as "input" price
        elif video_price > 0:
            return video_price, 0, region  # Return video price as "input" price
        elif other_price > 0:
            return other_price, 0, region  # Return other price as "input" price

    return 0, 0, None

def extract_pricing_info(pricing_data):
    """
    Extract pricing information from model data, handling multiple pricing structures.
    Prioritizes US regions (us-east-1, then us-west-2) from comprehensive_pricing.
    
    Args:
        pricing_data: Dictionary containing pricing information
        
    Returns:
        tuple: (input_price, output_price) per 1K tokens
    """
    if not isinstance(pricing_data, dict):
        return 0, 0
    
    # Standard structure: pricing.on_demand (preferred)
    if 'on_demand' in pricing_data:
        on_demand_pricing = pricing_data.get('on_demand', {})
        if isinstance(on_demand_pricing, dict):
            input_price = on_demand_pricing.get('input_tokens', 0)
            output_price = on_demand_pricing.get('output_tokens', 0)
            return input_price, output_price
    
    # Legacy complex structure: pricing.categories.on_demand
    if 'categories' in pricing_data:
        categories = pricing_data.get('categories', {})
        on_demand = categories.get('on_demand', {})
        if on_demand:
            input_price = on_demand.get('input_tokens', 0)
            output_price = on_demand.get('output_tokens', 0)
            return input_price, output_price
    
    # Legacy complex structure: pricing.detailed_pricing.on_demand
    if 'detailed_pricing' in pricing_data:
        detailed = pricing_data.get('detailed_pricing', {})
        on_demand = detailed.get('on_demand', {})
        if on_demand:
            input_price = on_demand.get('input_tokens', 0)
            output_price = on_demand.get('output_tokens', 0)
            return input_price, output_price
    
    # AWS Pricing API structure (legacy)
    if 'input_cost_per_1k_tokens' in pricing_data:
        input_price = pricing_data.get('input_cost_per_1k_tokens', 0)
        output_price = pricing_data.get('output_cost_per_1k_tokens', 0)
        return input_price, output_price
    
    # Old structure: direct pricing fields
    if 'input_tokens_per_1k' in pricing_data:
        input_price = pricing_data.get('input_tokens_per_1k', 0)
        output_price = pricing_data.get('output_tokens_per_1k', 0)
        return input_price, output_price
    
    # Fallback: try to find any pricing data
    input_price = 0
    output_price = 0
    
    # Check for various possible field names
    possible_input_fields = ['input_tokens', 'input_cost', 'input_price', 'inputTokens']
    possible_output_fields = ['output_tokens', 'output_cost', 'output_price', 'outputTokens']
    
    for field in possible_input_fields:
        if field in pricing_data and pricing_data[field] > 0:
            input_price = pricing_data[field]
            break
    
    for field in possible_output_fields:
        if field in pricing_data and pricing_data[field] > 0:
            output_price = pricing_data[field]
            break
    
    return input_price, output_price


def extract_all_consumption_pricing(pricing_data):
    """
    Extract pricing information for all consumption options.
    
    Args:
        pricing_data: Dictionary containing pricing information
        
    Returns:
        dict: Dictionary with pricing for each consumption type
    """
    if not isinstance(pricing_data, dict):
        return {}
    
    consumption_pricing = {}
    
    # Standard structure: pricing directly contains consumption types
    pricing_source = pricing_data
    
    # Legacy complex structure: pricing.categories or pricing.detailed_pricing
    if 'categories' in pricing_data:
        pricing_source = pricing_data['categories']
    elif 'detailed_pricing' in pricing_data:
        pricing_source = pricing_data['detailed_pricing']
    
    # Check for different consumption types
    consumption_types = ['on_demand', 'batch', 'provisioned_throughput']
    
    for consumption_type in consumption_types:
        if consumption_type in pricing_source:
            type_pricing = pricing_source[consumption_type]
            if isinstance(type_pricing, dict):
                if consumption_type == 'provisioned_throughput':
                    consumption_pricing[consumption_type] = {
                        'model_units_per_hour': type_pricing.get('model_units_per_hour', 0),
                        'minimum_commitment': type_pricing.get('minimum_commitment', '')
                    }
                else:
                    consumption_pricing[consumption_type] = {
                        'input_tokens': type_pricing.get('input_tokens', 0),
                        'output_tokens': type_pricing.get('output_tokens', 0),
                        'per_image': type_pricing.get('per_image', 0),
                        'per_video': type_pricing.get('per_video', 0)
                    }
    
    return consumption_pricing


def categorize_models_by_consumption_options(models_data):
    """
    Categorize models by their consumption options.

    Args:
        models_data: List or DataFrame of model data

    Returns:
        dict: Dictionary with consumption options as keys and model lists as values
    """
    categories = {
        'on_demand': [],
        'batch': [],
        'provisioned_throughput': [],
        'cross_region_inference': []
    }

    for model in models_data:
        model_id = model.get('model_id', '')
        consumption_options = get_consumption_options(model)

        # Check cross-region inference separately
        cris_supported = get_cross_region_inference_support(model)

        for option in consumption_options:
            if option in categories:
                categories[option].append({
                    'model_id': model_id,
                    'name': model.get('name', ''),
                    'provider': model.get('provider', ''),
                    'data': model
                })

        if cris_supported:
            categories['cross_region_inference'].append({
                'model_id': model_id,
                'name': model.get('name', ''),
                'provider': model.get('provider', ''),
                'data': model
            })

    return categories


def categorize_models_by_type(models_data):
    """
    Categorize models by their input/output modalities.

    Args:
        models_data: List or DataFrame of model data

    Returns:
        dict: Dictionary with model types as keys and model lists as values
    """
    categories = {
        'text_only': [],
        'multimodal_text_image': [],
        'multimodal_text_video': [],
        'multimodal_text_audio': [],
        'image_generation': [],
        'video_generation': [],
        'audio_generation': [],
        'other_multimodal': []
    }

    for model in models_data:
        model_id = model.get('model_id', '')
        input_modalities = set(model.get('input_modalities', []))
        output_modalities = set(model.get('output_modalities', []))

        model_info = {
            'model_id': model_id,
            'name': model.get('name', ''),
            'provider': model.get('provider', ''),
            'input_modalities': list(input_modalities),
            'output_modalities': list(output_modalities),
            'data': model
        }

        # Text-only models
        if input_modalities == {'TEXT'} and output_modalities == {'TEXT'}:
            categories['text_only'].append(model_info)

        # Image generation models (text input, image output)
        elif 'TEXT' in input_modalities and 'IMAGE' in output_modalities:
            if len(output_modalities) == 1:  # Only image output
                categories['image_generation'].append(model_info)
            else:
                categories['multimodal_text_image'].append(model_info)

        # Video generation models (text input, video output)
        elif 'TEXT' in input_modalities and 'VIDEO' in output_modalities:
            if len(output_modalities) == 1:  # Only video output
                categories['video_generation'].append(model_info)
            else:
                categories['multimodal_text_video'].append(model_info)

        # Audio generation models (text input, audio output)
        elif 'TEXT' in input_modalities and 'AUDIO' in output_modalities:
            if len(output_modalities) == 1:  # Only audio output
                categories['audio_generation'].append(model_info)
            else:
                categories['multimodal_text_audio'].append(model_info)

        # Multimodal text + image (can accept both text and images)
        elif 'TEXT' in input_modalities and 'IMAGE' in input_modalities:
            categories['multimodal_text_image'].append(model_info)

        # Multimodal text + video
        elif 'TEXT' in input_modalities and 'VIDEO' in input_modalities:
            categories['multimodal_text_video'].append(model_info)

        # Multimodal text + audio
        elif 'TEXT' in input_modalities and 'AUDIO' in input_modalities:
            categories['multimodal_text_audio'].append(model_info)

        # Other multimodal combinations
        elif len(input_modalities) > 1 or len(output_modalities) > 1:
            categories['other_multimodal'].append(model_info)

        # If none of the above, default to text_only if it has text capabilities
        elif 'TEXT' in input_modalities or 'TEXT' in output_modalities:
            categories['text_only'].append(model_info)
        else:
            categories['other_multimodal'].append(model_info)

    return categories


def extract_pricing_dimensions_by_model_type(model_data, region=None):
    """
    Extract pricing dimensions based on model type (output modalities).

    Args:
        model_data: Model data dictionary
        region: AWS region to extract pricing from

    Returns:
        dict: Dictionary with pricing dimensions relevant to the model type
    """
    output_modalities = set(model_data.get('output_modalities', []))

    # Get comprehensive pricing for the region
    if region:
        input_price, output_price, pricing_region = extract_comprehensive_pricing_info(model_data, region)
    else:
        input_price, output_price, pricing_region = extract_comprehensive_pricing_info(model_data)

    # Get model pricing structure
    model_pricing = model_data.get('model_pricing', {})
    comprehensive_pricing = model_pricing.get('comprehensive_pricing', {})

    pricing_dimensions = {
        'region_used': pricing_region,
        'cross_region_supported': get_cross_region_inference_support(model_data),
        'consumption_options': get_consumption_options(model_data)
    }

    # Text-based pricing (input/output tokens)
    if 'TEXT' in output_modalities:
        pricing_dimensions.update({
            'input_tokens_per_1k': input_price,
            'output_tokens_per_1k': output_price,
            'pricing_unit': 'per 1K tokens'
        })

    # Image-based pricing
    if 'IMAGE' in output_modalities:
        # Try to extract image pricing from comprehensive pricing
        image_price = 0
        if pricing_region and pricing_region in comprehensive_pricing:
            region_data = comprehensive_pricing[pricing_region]
            on_demand = region_data.get('on_demand', {})
            if on_demand and 'pricing_dimensions' in on_demand:
                dimensions = on_demand['pricing_dimensions']
                for dim_name, dim_data in dimensions.items():
                    if 'on-demand' in dim_name.lower() or dim_name in ['On-Demand', 'OnDemand']:
                        entries = dim_data.get('pricing_entries', [])
                        for entry in entries:
                            description = entry.get('description', '').lower()
                            unit = entry.get('unit', '').lower()
                            if 'image' in description or 'image' in unit:
                                image_price = entry.get('price_per_thousand', 0)
                                break

        pricing_dimensions.update({
            'per_image': image_price,
            'pricing_unit': 'per image' if image_price > 0 else 'per unit'
        })

    # Video-based pricing
    if 'VIDEO' in output_modalities:
        video_price = 0
        if pricing_region and pricing_region in comprehensive_pricing:
            region_data = comprehensive_pricing[pricing_region]
            on_demand = region_data.get('on_demand', {})
            if on_demand and 'pricing_dimensions' in on_demand:
                dimensions = on_demand['pricing_dimensions']
                for dim_name, dim_data in dimensions.items():
                    if 'on-demand' in dim_name.lower() or dim_name in ['On-Demand', 'OnDemand']:
                        entries = dim_data.get('pricing_entries', [])
                        for entry in entries:
                            description = entry.get('description', '').lower()
                            unit = entry.get('unit', '').lower()
                            if 'video' in description or 'video' in unit:
                                video_price = entry.get('price_per_thousand', 0)
                                break

        pricing_dimensions.update({
            'per_video': video_price,
            'pricing_unit': 'per video' if video_price > 0 else 'per unit'
        })

    # Audio-based pricing
    if 'AUDIO' in output_modalities:
        audio_price = 0
        if pricing_region and pricing_region in comprehensive_pricing:
            region_data = comprehensive_pricing[pricing_region]
            on_demand = region_data.get('on_demand', {})
            if on_demand and 'pricing_dimensions' in on_demand:
                dimensions = on_demand['pricing_dimensions']
                for dim_name, dim_data in dimensions.items():
                    if 'on-demand' in dim_name.lower() or dim_name in ['On-Demand', 'OnDemand']:
                        entries = dim_data.get('pricing_entries', [])
                        for entry in entries:
                            description = entry.get('description', '').lower()
                            unit = entry.get('unit', '').lower()
                            if 'audio' in description or 'audio' in unit:
                                audio_price = entry.get('price_per_thousand', 0)
                                break

        pricing_dimensions.update({
            'per_audio_unit': audio_price,
            'pricing_unit': 'per audio unit' if audio_price > 0 else 'per unit'
        })

    return pricing_dimensions


from utils.recommendation import (
    analyze_use_case,
    generate_follow_up_questions,
    extract_keywords,
    detect_intent,
    recommend_models,
    explain_recommendation,
    suggest_third_party_models,
    explain_third_party_suggestion
)
from ui.filters import (
    show_filter_controls,
    show_sort_controls,
    apply_filters,
    apply_sorting,
    show_filter_summary
)
from ui.selection import (
    initialize_selection_state,
    toggle_model_selection,
    is_model_selected,
    get_selected_models,
    clear_selection
)



def show_model_card_view(df: pd.DataFrame, is_favorites_view=False):
    """Display models in card view format with consistent row heights"""
    # Import favorites utilities
    from utils.favorites import is_favorite, toggle_favorite
    
    # Sort models by provider and name for consistent display
    df_sorted = df.sort_values(by=['provider', 'model_name'])
    
    # CSS is loaded from custom_styles.css in app.py
    
    # Get primary region from filter state for pricing and quota display
    primary_region = st.session_state.get('filter_state', {}).get('primary_region', 'us-east-1')
    
    # Process models in rows of 4 for proper row-based layout
    models_list = list(df_sorted.iterrows())
    
    # Display models in rows of 4
    for i in range(0, len(models_list), 4):
        row_models = models_list[i:i+4]
        
        # Create columns for this row (4 columns)
        cols = st.columns(4)
        
        # Display each model in the row
        for col_idx, (_, model) in enumerate(row_models):
            with cols[col_idx]:
                # All the existing model card content goes here
                # Ensure provider is a string (handle NaN/float values)
                provider = model.get('provider', 'Unknown')
                if not isinstance(provider, str):
                    provider = str(provider) if provider is not None and not pd.isna(provider) else 'Unknown'

                # Ensure model_id is a string (handle NaN/float values)
                model_id = model.get('model_id', '')
                if not isinstance(model_id, str):
                    model_id = str(model_id) if model_id is not None and not pd.isna(model_id) else ''

                # Get provider badge class (status badge now handled inline)
                provider_badge_class = get_provider_badge_class(provider)

                # Get pricing information - use comprehensive pricing with selected primary region
                input_price, output_price, region_used = extract_comprehensive_pricing_info(model, primary_region)

                # Format model ID for display
                model_id_short = model_id.split(':')[0] if ':' in model_id else model_id
                

                
                # === HEADER SECTION with gradient background ===
                # Import converse data helpers for size category
                from ui.converse_data_helpers import get_converse_size_category

                # Prepare badges for provider, status, and size category
                provider_badge = f"<span style='background-color: {get_provider_color(provider)}; color: white; padding: 5px 10px; border-radius: 16px; font-size: 0.75rem; font-weight: bold; margin-right: 8px;'>{provider}</span>"

                # Status logic: get from model_lifecycle field
                model_lifecycle = model.get('model_lifecycle', {})
                if isinstance(model_lifecycle, dict):
                    lifecycle_status = model_lifecycle.get('status', 'ACTIVE')
                else:
                    lifecycle_status = 'ACTIVE'

                # Ensure lifecycle_status is a string (handle NaN/float values)
                if not isinstance(lifecycle_status, str):
                    lifecycle_status = 'ACTIVE'

                if lifecycle_status == 'LEGACY':
                    status_display = 'LEGACY'
                    status_color = '#F59E0B'  # Orange for Legacy
                else:
                    status_display = 'ACTIVE'
                    status_color = '#10B981'  # Green for Active

                status_badge = f"<span style='background: {status_color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: bold; margin-right: 6px;'>{status_display}</span>"

                # Size category badge from converse_data
                size_category = get_converse_size_category(model)
                size_color = size_category.get('color', '#6B7280')
                size_name = size_category.get('category', 'Unknown')
                size_badge = f"<span style='background: {size_color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: bold; margin-right: 6px;'>{size_name}</span>"
                
                # Get model description and release date
                description = model.get('description', '')
                # Ensure description is a string (safeguard against dict or other types)
                if not isinstance(description, str):
                    description = str(description)
                truncated_description = description[:180] + ('...' if len(description) > 180 else '')

                release_date = model.get('release_date', '')
                # Ensure release_date is a string (handle NaN/float values)
                if not isinstance(release_date, str):
                    release_date = str(release_date) if release_date is not None and not pd.isna(release_date) else ''
                
                # Prepare favorite and compare button states
                is_fav = is_favorite(model_id)
                fav_icon = "⭐" if is_fav else "☆"

                if 'comparison_models_with_regions' not in st.session_state:
                    st.session_state.comparison_models_with_regions = []
                is_comparing = any(item['model_id'] == model_id 
                                 for item in st.session_state.comparison_models_with_regions)
                compare_icon = "⚖️" if is_comparing else "⚖"
                
                # Create the header with badges (removed description and model size badge)
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #1f4e79 0%, #2d5aa0 100%);
                            padding: 15px; border-radius: 10px; margin-bottom: 15px; min-height: 120px; overflow: visible;">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                        <div style="display: flex; align-items: center;">
                            {provider_badge}
                            {status_badge}
                            {size_badge}
                        </div>
                    </div>
                    <h3 style="color: white; margin: 5px 0;">{model['name']}{' <span style="font-size: 0.8rem; font-weight: normal;">(' + release_date + ')</span>' if release_date and release_date != 'N/A' else ''}</h3>
                    <p style="color: #ffffff; margin: 5px 0; font-size: 0.9rem; font-weight: 600; opacity: 0.9;">{model_id_short}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a row for the buttons with 3 columns and improved styling
                button_col1, button_col2 = st.columns([2, 2])

                # Favorite button
                with button_col1:
                    if st.button(f"{fav_icon} Favorite", key=f"fav_{model_id}", width='stretch'):
                        toggle_favorite(model_id)
                        st.rerun()

                # Compare button
                with button_col2:
                    if st.button(f"{compare_icon} Compare", key=f"comp_{model_id}", width='stretch'):
                        if 'comparison_models_with_regions' not in st.session_state:
                            st.session_state.comparison_models_with_regions = []

                        # Check if this model is already in comparison
                        existing_item = next((item for item in st.session_state.comparison_models_with_regions
                                            if item['model_id'] == model_id), None)

                        if existing_item:
                            # Remove from comparison
                            st.session_state.comparison_models_with_regions = [
                                item for item in st.session_state.comparison_models_with_regions
                                if item['model_id'] != model_id
                            ]
                        else:
                            # Add to comparison with primary region if available, otherwise default region
                            available_regions = extract_available_regions_from_pricing(model)
                            # Ensure we have at least one region available
                            if not available_regions:
                                available_regions = ['us-east-1']

                            primary_region = st.session_state.get('filter_state', {}).get('primary_region', 'us-east-1')

                            # Use primary region if model is available in that region, otherwise use first available
                            selected_region = primary_region if primary_region in available_regions else available_regions[0]

                            new_item = {
                                'model_id': model_id,
                                'region': selected_region,
                                'comparison_id': f"{model_id}_{selected_region}"
                            }
                            st.session_state.comparison_models_with_regions.append(new_item)
                        st.rerun()
                
                # Create 2-column layout for model details (new organized structure)
                detail_col1, detail_col2 = st.columns(2)
                
                # LEFT COLUMN
                with detail_col1:
                    # Row 1: Technical Specifications
                    st.markdown("**⚙️ Technical Specifications:**")
                    
                    # Streaming and Cross-Region status (emoji only)
                    streaming_supported = model.get('streaming_supported', False)
                    cris_supported = get_cross_region_inference_support(model)
                    context_window = get_converse_context_window(model)
                    st.markdown(f"Context window: {context_window:,} tokens")
                    st.markdown(f"Streaming {'✅' if streaming_supported else '❌'}")
                    st.markdown(f"Cross-Region {'✅' if cris_supported else '❌'}")

                    # Row 2: Allowed Formats
                    st.markdown("**📥 Allowed Formats:**")

                    # Get modalities from model_modalities nested structure
                    model_modalities = model.get('model_modalities', {})
                    input_modalities = model_modalities.get('input_modalities', []) if isinstance(model_modalities, dict) else []
                    output_modalities = model_modalities.get('output_modalities', []) if isinstance(model_modalities, dict) else []

                    # Fallback to direct fields for backward compatibility
                    if not input_modalities:
                        input_modalities = model.get('input_modalities', [])
                    if not output_modalities:
                        output_modalities = model.get('output_modalities', [])

                    input_text = ', '.join(input_modalities) if input_modalities else 'Not specified'
                    output_text = ', '.join(output_modalities) if output_modalities else 'Not specified'

                    st.markdown(f"Input: {input_text}")
                    st.markdown(f"Output: {output_text}")
                    
                    # Row 3: Pricing
                    # Determine model type for appropriate pricing display (reuse extracted modalities)
                    # input_modalities and output_modalities already extracted above

                    # Determine pricing unit based on model type
                    is_text_model = 'TEXT' in input_modalities and 'TEXT' in output_modalities
                    is_image_model = 'IMAGE' in input_modalities or 'IMAGE' in output_modalities
                    is_video_model = 'VIDEO' in input_modalities or 'VIDEO' in output_modalities

                    if is_text_model:
                        st.markdown("**💰 Pricing per 1K tokens:**")
                    elif is_image_model:
                        st.markdown("**💰 Pricing per image:**")
                    elif is_video_model:
                        st.markdown("**💰 Pricing per video:**")
                    else:
                        st.markdown("**💰 Pricing:**")

                    # Show pricing if we have input pricing (output can be 0) or any other pricing
                    if input_price > 0 or output_price > 0:
                        # Show actual pricing from the selected region
                        if is_text_model:
                            # Standard token-based pricing display
                            if input_price > 0:
                                st.markdown(f"Input: {format_price(input_price)}")
                            if output_price > 0:
                                st.markdown(f"Output: {format_price(output_price)}")
                        else:
                            # Non-text model pricing (image, video, etc.)
                            if input_price > 0:
                                if is_image_model:
                                    st.markdown(f"Per image: {format_price(input_price)}")
                                elif is_video_model:
                                    st.markdown(f"Per video: {format_price(input_price)}")
                                else:
                                    st.markdown(f"Per unit: {format_price(input_price)}")

                        # Show region used for pricing if available
                        if region_used:
                            st.markdown(f"*Region: {region_used}*", help="Pricing shown for this region")

                    else:
                        # Fallback to legacy pricing extraction
                        input_price_legacy, output_price_legacy = extract_pricing_info(model.get('pricing', {}))
                        if input_price_legacy > 0 or output_price_legacy > 0:
                            if is_text_model:
                                if input_price_legacy > 0:
                                    st.markdown(f"Input: {format_price(input_price_legacy)}")
                                if output_price_legacy > 0:
                                    st.markdown(f"Output: {format_price(output_price_legacy)}")
                            else:
                                # Non-text model legacy pricing
                                if input_price_legacy > 0:
                                    if is_image_model:
                                        st.markdown(f"Per image: {format_price(input_price_legacy)}")
                                    elif is_video_model:
                                        st.markdown(f"Per video: {format_price(input_price_legacy)}")
                                    else:
                                        st.markdown(f"Per unit: {format_price(input_price_legacy)}")
                        else:
                            # Show link to AWS Bedrock pricing when no pricing data is available
                            st.markdown("[Check AWS Bedrock Pricing →](https://aws.amazon.com/bedrock/pricing/)",
                                      help="Pricing data not available - click to view public pricing")
                    
                    # Row 4: Consumption Options
                    st.markdown("**💳 Consumption Options:**")
                    # Get consumption options directly from bedrock model json
                    consumption_options = model.get('consumption_options', [])

                    consumption_html = "<div style='margin: 2px 0 8px 0;'>"
                    # Expanded consumption options mapping
                    consumption_map = {
                        'on_demand': 'On-Demand',
                        'batch': 'Batch',
                        'provisioned_throughput': 'Provisioned',
                        'cross_region_inference': 'Cross-Region',
                        'general': 'General',
                        'batch_inference': 'Batch Inference',
                        'inference_profile': 'Inference Profile'
                    }

                    # Display all consumption options from bedrock model json
                    for option in consumption_options:
                        display_name = consumption_map.get(option, option.replace('_', ' ').title())
                        consumption_html += f"<span style='display: inline-block; margin: 1px; padding: 3px 8px; background: #f8f9fa; color: #495057; border: 1px solid #dee2e6; border-radius: 6px; font-size: 0.7rem; font-weight: 500;'>{display_name}</span> "

                    # If no consumption options, show default
                    if not consumption_options:
                        consumption_html += f"<span style='display: inline-block; margin: 1px; padding: 3px 8px; background: #f8f9fa; color: #495057; border: 1px solid #dee2e6; border-radius: 6px; font-size: 0.7rem; font-weight: 500;'>On-Demand</span> "

                    consumption_html += "</div>"
                    st.markdown(consumption_html, unsafe_allow_html=True)
                
                # RIGHT COLUMN  
                with detail_col2:
                    # Row 1: Capabilities
                    st.markdown("**🎯 Capabilities:**")
                    # Use the correct field name from the new data structure
                    capabilities = model.get('model_capabilities', [])
                    if capabilities:
                        caps_html = "<div style='margin: 2px 0 8px 0;'>"
                        for cap in capabilities[:6]:
                            caps_html += f"<span style='display: inline-block; margin: 2px; padding: 4px 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 14px; font-size: 0.7rem; font-weight: 500;'>{cap}</span> "
                        if len(capabilities) > 6:
                            caps_html += f"<span style='display: inline-block; margin: 2px; padding: 4px 10px; background: #6B7280; color: white; border-radius: 14px; font-size: 0.7rem;'>+{len(capabilities) - 6}</span>"
                        caps_html += "</div>"
                        st.markdown(caps_html, unsafe_allow_html=True)
                    else:
                        st.markdown("*Not specified*")
                    
                    st.markdown("---")  # Separator
                    
                    # Row 2: Regions
                    regions = extract_available_regions_from_pricing(model)
                    st.markdown(f"**🗺️ Regions ({len(regions)} available):**")
                    if regions:
                        # Define gradient colors for different geographic regions
                        def get_region_gradient(region):
                            if region.startswith('us-'):
                                return 'linear-gradient(135deg, #F59E0B 0%, #D97706 100%)'  # Orange gradient for US
                            elif region.startswith('eu-'):
                                return 'linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%)'  # Blue gradient for EU
                            elif region.startswith('ap-'):
                                return 'linear-gradient(135deg, #10B981 0%, #059669 100%)'  # Green gradient for Asia Pacific
                            elif region.startswith('ca-'):
                                return 'linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%)'  # Purple gradient for Canada
                            else:
                                return 'linear-gradient(135deg, #6B7280 0%, #4B5563 100%)'  # Gray gradient for unknown
                        
                        regions_html = "<div style='margin: 2px 0 8px 0;'>"
                        for region in regions[:6]:  # Show up to 6 regions
                            gradient = get_region_gradient(region)
                            regions_html += f"<span style='display: inline-block; margin: 1px; padding: 3px 8px; background: {gradient}; color: white; border-radius: 10px; font-size: 0.7rem; font-weight: 500;'>{region}</span> "
                        
                        if len(regions) > 6:
                            regions_html += f"<span style='display: inline-block; margin: 1px; padding: 3px 8px; background: #6B7280; color: white; border-radius: 10px; font-size: 0.7rem; font-weight: 500;'>+{len(regions)-6}</span>"
                        
                        regions_html += "</div>"
                        st.markdown(regions_html, unsafe_allow_html=True)
                    else:
                        st.markdown("*No regions specified*")
                    
                    st.markdown("---")  # Separator
                    
                    # Row 3: Use Cases
                    st.markdown("**💡 Use Cases:**")
                    # Use the correct field name from the new data structure
                    use_cases = model.get('model_use_cases', [])
                    if use_cases:
                        use_cases_html = "<div style='margin: 2px 0 8px 0;'>"
                        for use_case in use_cases[:4]:
                            use_cases_html += f"<span style='display: inline-block; margin: 1px; padding: 3px 8px; background: #f8f9fa; color: #495057; border: 1px solid #dee2e6; border-radius: 6px; font-size: 0.7rem; font-weight: 500;'>{use_case}</span> "
                        if len(use_cases) > 4:
                            use_cases_html += f"<span style='display: inline-block; margin: 1px; padding: 3px 8px; background: #f8f9fa; color: #495057; border: 1px solid #dee2e6; border-radius: 6px; font-size: 0.7rem; font-weight: 500;'>+{len(use_cases)-4}</span>"
                        use_cases_html += "</div>"
                        st.markdown(use_cases_html, unsafe_allow_html=True)
                    else:
                        st.markdown("*Not specified*")
            
                # === ACTION BUTTONS ===
                st.markdown("---")
                
                # Check if details are currently shown
                details_shown = st.session_state.get(f"show_details_{model_id}", False)

                # Toggle Details button
                button_text = "❌ Close Details" if details_shown else "📋 View Details"
                if st.button(button_text, key=f"details_{model_id}", width="stretch"):
                    st.session_state[f"show_details_{model_id}"] = not details_shown
                    st.rerun()

                # Show detailed modal if requested
                if st.session_state.get(f"show_details_{model_id}", False):
                    st.markdown('<div class="model-details-section">', unsafe_allow_html=True)
                    from .model_details_modal import display_model_details_modal
                    display_model_details_modal(model)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # Add empty space to maintain consistent height when details are not shown
                    st.markdown('<div class="model-details-section" style="min-height: 0;"></div>', unsafe_allow_html=True)

                # Add some vertical spacing between cards
                st.write("")
        
        # Fill empty columns if needed (up to 4 columns)
        for col_idx in range(len(row_models), 4):
            with cols[col_idx]:
                st.empty()


def show_model_explorer(df: pd.DataFrame, is_favorites_view=False):
    """Display the main model explorer interface
    
    Args:
        df: DataFrame containing model data
        is_favorites_view: Whether this is the favorites view
    """
    if df.empty:
        st.warning("No models match your current filters. Please adjust your selection.")
        return
    
    # Store original count for search comparison
    original_count = len(df)
    
    # Display filter controls
    key_suffix = "_favorites" if is_favorites_view else "_explorer"
    filter_state = show_filter_controls(df, key_suffix)
    
    # Ensure filter_state is not None
    if filter_state is None:
        # Initialize default filter state
        filter_state = {
            'providers': [],
            'capabilities': [],
            'geo_region': 'All Regions',
            'use_cases': [],
            'modalities': [],
            'streaming': 'All Models',
            'search_query': '',
            'consumption_filter': 'All Models',
            'cris_support': 'All Models'
        }
    
    # Apply filters
    filtered_df = apply_filters(df, filter_state)
    
    # Show filter summary
    show_filter_summary(filter_state, original_count, len(filtered_df))
    
    # If no models match filters, show warning and return
    if filtered_df.empty:
        st.warning("No models match your current filters. Please adjust your selection.")
        return
        
    # Initialize pagination state if not exists - separate for each view
    if 'card_page_number' not in st.session_state:
        st.session_state.card_page_number = 1
    if 'table_page_number' not in st.session_state:
        st.session_state.table_page_number = 1
    if 'page_size' not in st.session_state:
        st.session_state.page_size = 10

    # Flag to prevent dropdown interference after button clicks
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False
    
    # View selection tabs
    view_tab1, view_tab2 = st.tabs(["🃏 Card View", "📋 Table View"])
    
    with view_tab1:
        # Consolidated controls in one row
        control_col1, control_col2, control_col3, control_col4 = st.columns([2, 1, 1, 2])
        
        with control_col1:
            # Sort selector
            sort_options = {
                "Name (A-Z)": ("name", True),
                "Name (Z-A)": ("name", False),
                "Provider (A-Z)": ("provider", True),
                "Provider (Z-A)": ("provider", False),
                "Input Price (Low-High)": ("input_price", True),
                "Input Price (High-Low)": ("input_price", False),
                "Output Price (Low-High)": ("output_price", True),
                "Output Price (High-Low)": ("output_price", False),
                "Regions (Most-Least)": ("region_count", False),
                "Regions (Least-Most)": ("region_count", True),
                "Release Date (Newest First)": ("release_date", False),
                "Release Date (Oldest First)": ("release_date", True)
            }
            
            # Get current sort option from session state or default
            current_sort_key = "Name (A-Z)"  # default
            if 'sort_option' in st.session_state:
                current_sort_key = st.session_state.sort_option
            
            sort_by = st.selectbox(
                "Sort by",
                options=list(sort_options.keys()),
                index=list(sort_options.keys()).index(current_sort_key) if current_sort_key in sort_options else 0,
                key=f"sort_selector_card{'_favorites' if is_favorites_view else '_explorer'}"
            )
            
            # Get sort column and direction
            sort_column, ascending = sort_options[sort_by]
            st.session_state.sort_option = sort_by
        
        with control_col2:
            # Items per page
            page_size = st.selectbox(
                "Items per page",
                options=[5, 10, 20, 50, 100],
                index=[5, 10, 20, 50, 100].index(st.session_state.page_size),
                key="page_size_selector"
            )
            st.session_state.page_size = page_size
        
        with control_col3:
            # Apply sorting first to get correct pagination
            sorted_df = apply_sorting(filtered_df, sort_column, ascending)
            
            # Calculate pagination values
            total_pages = max(1, (len(sorted_df) + page_size - 1) // page_size)
            
            # Page selector (dropdown instead of number input) - FIXED to sync properly
            page_options = list(range(1, total_pages + 1))

            # Ensure dropdown shows correct page after button clicks
            actual_page = st.session_state.card_page_number
            current_page_index = min(max(0, actual_page - 1), len(page_options) - 1)

            # Force dropdown to sync by using a new key when button is clicked
            dropdown_key = f"page_selector_card_{actual_page}" if st.session_state.button_clicked else "page_selector_card"

            selected_page = st.selectbox(
                "Page",
                options=page_options,
                index=current_page_index,
                key=dropdown_key
            )

            # Only update session state if no button was clicked (prevents interference)
            if selected_page != st.session_state.card_page_number and not st.session_state.button_clicked:
                st.session_state.card_page_number = selected_page

            # Reset the button flag after processing
            st.session_state.button_clicked = False
        
        with control_col4:
            # Page navigation info and buttons with vertical alignment
            st.markdown(f"""
            <div class="pagination-controls-container">
                <div class="pagination-total-items">
                    <strong>{len(sorted_df)} total items</strong>
                </div>
                <div class="pagination-nav-buttons">
                    <!-- Navigation buttons will be rendered below -->
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Navigation buttons in a single row
            nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

            current_page = st.session_state.card_page_number

            with nav_col1:
                if st.button("⏮️", disabled=(current_page == 1), key="first_page_btn", help="First page"):
                    st.session_state.card_page_number = 1
                    st.session_state.button_clicked = True  # Prevent dropdown interference
                    st.rerun()

            with nav_col2:
                if st.button("◀️", disabled=(current_page == 1), key="prev_page_btn", help="Previous page"):
                    new_page = max(1, current_page - 1)
                    st.session_state.card_page_number = new_page
                    st.session_state.button_clicked = True  # Prevent dropdown interference
                    st.rerun()

            with nav_col3:
                if st.button("▶️", disabled=(current_page == total_pages), key="next_page_btn", help="Next page"):
                    new_page = min(total_pages, current_page + 1)
                    st.session_state.card_page_number = new_page
                    st.session_state.button_clicked = True  # Prevent dropdown interference
                    st.rerun()

            with nav_col4:
                if st.button("⏭️", disabled=(current_page == total_pages), key="last_page_btn", help="Last page"):
                    st.session_state.card_page_number = total_pages
                    st.session_state.button_clicked = True  # Prevent dropdown interference
                    st.rerun()
        
        # Paginate the dataframe - Card View
        current_page = st.session_state.card_page_number
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, len(sorted_df))
        paginated_df = sorted_df.iloc[start_idx:end_idx].copy()

        show_model_card_view(paginated_df, is_favorites_view=is_favorites_view)
    
    with view_tab2:
        # Same consolidated controls for table view
        control_col1, control_col2, control_col3, control_col4 = st.columns([2, 1, 1, 2])
        
        with control_col1:
            # Sort selector
            sort_options = {
                "Name (A-Z)": ("name", True),
                "Name (Z-A)": ("name", False),
                "Provider (A-Z)": ("provider", True),
                "Provider (Z-A)": ("provider", False),
                "Input Price (Low-High)": ("input_price", True),
                "Input Price (High-Low)": ("input_price", False),
                "Output Price (Low-High)": ("output_price", True),
                "Output Price (High-Low)": ("output_price", False),
                "Regions (Most-Least)": ("region_count", False),
                "Regions (Least-Most)": ("region_count", True),
                "Release Date (Newest First)": ("release_date", False),
                "Release Date (Oldest First)": ("release_date", True)
            }
            
            # Get current sort option from session state or default
            current_sort_key = "Name (A-Z)"  # default
            if 'sort_option' in st.session_state:
                current_sort_key = st.session_state.sort_option
            
            sort_by = st.selectbox(
                "Sort by",
                options=list(sort_options.keys()),
                index=list(sort_options.keys()).index(current_sort_key) if current_sort_key in sort_options else 0,
                key=f"sort_selector_table{'_favorites' if is_favorites_view else '_explorer'}"
            )
            
            # Get sort column and direction
            sort_column, ascending = sort_options[sort_by]
            st.session_state.sort_option = sort_by
        
        with control_col2:
            page_size = st.selectbox(
                "Items per page",
                options=[5, 10, 20, 50, 100],
                index=[5, 10, 20, 50, 100].index(st.session_state.page_size),
                key="page_size_selector_table"
            )
            st.session_state.page_size = page_size
        
        with control_col3:
            # Apply sorting first
            sorted_df = apply_sorting(filtered_df, sort_column, ascending)
            
            total_pages = max(1, (len(sorted_df) + page_size - 1) // page_size)
            page_options = list(range(1, total_pages + 1))
            current_page_index = min(st.session_state.table_page_number - 1, len(page_options) - 1)

            selected_page = st.selectbox(
                "Page",
                options=page_options,
                index=current_page_index,
                key="page_selector_table"
            )
            # Only update session state if selectbox value actually changed and user manually selected it
            # (Don't interfere with button navigation)
            if selected_page != st.session_state.table_page_number:
                st.session_state.table_page_number = selected_page
        
        with control_col4:
            # Page navigation info and buttons with vertical alignment
            st.markdown(f"""
            <div class="pagination-controls-container">
                <div class="pagination-total-items">
                    <strong>{len(sorted_df)} total items</strong>
                </div>
                <div class="pagination-nav-buttons">
                    <!-- Navigation buttons will be rendered below -->
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

            current_page = st.session_state.table_page_number

            with nav_col1:
                if st.button("⏮️", disabled=(current_page == 1), key="first_page_btn_table", help="First page"):
                    st.session_state.table_page_number = 1
                    st.rerun()

            with nav_col2:
                if st.button("◀️", disabled=(current_page == 1), key="prev_page_btn_table", help="Previous page"):
                    new_page = max(1, current_page - 1)
                    st.session_state.table_page_number = new_page
                    st.rerun()

            with nav_col3:
                if st.button("▶️", disabled=(current_page == total_pages), key="next_page_btn_table", help="Next page"):
                    new_page = min(total_pages, current_page + 1)
                    st.session_state.table_page_number = new_page
                    st.rerun()

            with nav_col4:
                if st.button("⏭️", disabled=(current_page == total_pages), key="last_page_btn_table", help="Last page"):
                    st.session_state.table_page_number = total_pages
                    st.rerun()

        # Paginate the dataframe
        current_page = st.session_state.table_page_number
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, len(sorted_df))
        paginated_df = sorted_df.iloc[start_idx:end_idx].copy()
        
        show_model_table_view(paginated_df, is_favorites_view=is_favorites_view)


def show_model_table_view(df: pd.DataFrame, is_favorites_view=False):
    """Display models in table view format with sortable columns and row selection
    
    Args:
        df: DataFrame containing model data
        is_favorites_view: Whether this is the favorites view
    """
    # Import favorites utilities
    from utils.favorites import is_favorite, toggle_favorite
    
    # Create a copy of the dataframe for display
    display_df = df.copy()
    
    # Initialize selection state
    initialize_selection_state()
    
    # Add selection controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("Clear Selection", key="clear_table_selection"):
            clear_selection()
            st.rerun()
    
    # Extract pricing information for display using comprehensive_pricing
    def extract_input_price_for_table(row):
        # Get primary region from filter state for consistent pricing display
        primary_region = st.session_state.get('filter_state', {}).get('primary_region', 'us-east-1')
        input_price, _, _ = extract_comprehensive_pricing_info(row, primary_region)
        return input_price

    def extract_output_price_for_table(row):
        # Get primary region from filter state for consistent pricing display
        primary_region = st.session_state.get('filter_state', {}).get('primary_region', 'us-east-1')
        _, output_price, _ = extract_comprehensive_pricing_info(row, primary_region)
        return output_price
    
    display_df['input_price'] = display_df.apply(extract_input_price_for_table, axis=1)
    display_df['output_price'] = display_df.apply(extract_output_price_for_table, axis=1)

    # Extract quota limits for display from model_service_quotas
    def extract_tokens_per_minute(model_service_quotas):
        if not isinstance(model_service_quotas, dict):
            return 'N/A'

        # Look through all regions and quotas for tokens per minute
        for region_data in model_service_quotas.values():
            if isinstance(region_data, dict):
                for quota_code, quota_data in region_data.items():
                    if isinstance(quota_data, dict) and quota_code != 'quota_metadata':
                        quota_name = quota_data.get('quota_name', '').lower()
                        if 'tokens per minute' in quota_name:
                            value = quota_data.get('value', 0)
                            return f"{int(value)}" if value > 0 else 'N/A'
        return 'N/A'

    def extract_requests_per_minute(model_service_quotas):
        if not isinstance(model_service_quotas, dict):
            return 'N/A'

        # Look through all regions and quotas for requests per minute
        for region_data in model_service_quotas.values():
            if isinstance(region_data, dict):
                for quota_code, quota_data in region_data.items():
                    if isinstance(quota_data, dict) and quota_code != 'quota_metadata':
                        quota_name = quota_data.get('quota_name', '').lower()
                        if 'requests per minute' in quota_name and 'on-demand' in quota_name:
                            value = quota_data.get('value', 0)
                            return f"{int(value)}" if value > 0 else 'N/A'
        return 'N/A'

    display_df['tokens_per_minute'] = display_df['model_service_quotas'].apply(extract_tokens_per_minute)
    display_df['requests_per_minute'] = display_df['model_service_quotas'].apply(extract_requests_per_minute)

    # Count regions for display
    display_df['region_count'] = display_df.apply(lambda row: len(extract_available_regions_from_pricing(row)), axis=1)
    
    
    # Initialize comparison state
    if 'comparison_models' not in st.session_state:
        st.session_state.comparison_models = set()
    
    # Helper function for status display
    def get_status_display(row):
        # Get status from model_lifecycle field
        model_lifecycle = row.get('model_lifecycle', {})
        if isinstance(model_lifecycle, dict):
            lifecycle_status = model_lifecycle.get('status', 'ACTIVE')
        else:
            lifecycle_status = 'ACTIVE'
        
        # Map AWS status to display status
        if lifecycle_status == 'LEGACY':
            return 'LEGACY'
        else:
            return 'ACTIVE'
    
    # Create a dataframe for display with simplified columns
    table_df = pd.DataFrame({
        'Provider': display_df['provider'],
        'Status': display_df.apply(get_status_display, axis=1),
        'Name': display_df['model_name'],
        'Model ID': display_df['model_id'],
        'Context Window': display_df.apply(lambda row: get_converse_context_window(row), axis=1),
        'Input Price': display_df['input_price'].apply(format_price),
        'Output Price': display_df['output_price'].apply(format_price),
        'Tokens/Min': display_df['tokens_per_minute'],
        'Requests/Min': display_df['requests_per_minute'],
        'Capabilities': display_df['model_capabilities'].apply(lambda x: ', '.join(x[:3]) + (f' +{len(x)-3} more' if len(x) > 3 else '') if isinstance(x, list) else 'Not specified'),
        'Regions': display_df['region_count'].apply(lambda x: f"{x} regions"),
        'Selected': display_df['model_id'].apply(lambda x: x in st.session_state.comparison_models),
        'Favorite': display_df['model_id'].apply(lambda x: is_favorite(x))
    })
    
    # Display the table with data_editor for interactive selection
    edited_df = st.data_editor(
        table_df,
        column_config={
            'Provider': st.column_config.TextColumn(
                "Provider",
                width="medium"
            ),
            'Status': st.column_config.TextColumn(
                "Status",
                width="small"
            ),
            'Name': st.column_config.TextColumn(
                "Name",
                width="large"
            ),
            'Model ID': st.column_config.TextColumn(
                "Model ID",
                width="medium"
            ),
            'Context Window': st.column_config.NumberColumn(
                "Context Window",
                format="%d tokens",
                width="medium"
            ),
            'Input Price': st.column_config.TextColumn(
                "Input Price",
                width="small"
            ),
            'Output Price': st.column_config.TextColumn(
                "Output Price",
                width="small"
            ),
            'Capabilities': st.column_config.TextColumn(
                "Capabilities",
                width="large"
            ),
            'Regions': st.column_config.TextColumn(
                "Regions",
                width="small"
            ),
            'Selected': st.column_config.CheckboxColumn(
                "Select",
                help="Select models for comparison",
                width="small"
            ),
            'Favorite': st.column_config.CheckboxColumn(
                "Favorite",
                help="Add to favorites",
                width="small"
            )
        },
        hide_index=True,
        width="stretch",
        disabled=["Provider", "Name", "Model ID", "Context Window", "Input Price", "Output Price", "Capabilities", "Regions"]
    )
    
    # Handle selection and favorite changes from edited dataframe
    if 'comparison_models' not in st.session_state:
        st.session_state.comparison_models = set()
    
    for _, row in edited_df.iterrows():
        model_id = row['Model ID']
        
        # Handle model selection for comparison
        if row['Selected']:
            st.session_state.comparison_models.add(model_id)
        else:
            st.session_state.comparison_models.discard(model_id)
        
        # Also handle the general selection state
        toggle_model_selection(model_id, row['Selected'])
        
        # Handle favorites toggle
        current_favorite = is_favorite(model_id)
        if current_favorite != row['Favorite']:
            toggle_favorite(model_id)
    


def show_model_compact_view(df: pd.DataFrame, is_favorites_view=False):
    """Display models in improved compact format with full-width rows
    
    Args:
        df: DataFrame containing model data
        is_favorites_view: Whether this is the favorites view
    """
    # Import favorites utilities
    from utils.favorites import is_favorite, toggle_favorite
    
    # Create a copy of the dataframe for display
    display_df = df.copy()
    
    # Initialize selection state
    initialize_selection_state()
    
    # Extract pricing information for display using comprehensive_pricing
    def extract_input_price_for_compact(row):
        # Get primary region from filter state for consistent pricing display
        primary_region = st.session_state.get('filter_state', {}).get('primary_region', 'us-east-1')
        input_price, _, _ = extract_comprehensive_pricing_info(row, primary_region)
        return input_price

    def extract_output_price_for_compact(row):
        # Get primary region from filter state for consistent pricing display
        primary_region = st.session_state.get('filter_state', {}).get('primary_region', 'us-east-1')
        _, output_price, _ = extract_comprehensive_pricing_info(row, primary_region)
        return output_price
    
    display_df['input_price'] = display_df.apply(extract_input_price_for_compact, axis=1)
    display_df['output_price'] = display_df.apply(extract_output_price_for_compact, axis=1)

    # Extract quota limits for display from model_service_quotas
    def extract_tokens_per_minute_compact(model_service_quotas):
        if not isinstance(model_service_quotas, dict):
            return 'N/A'

        # Look through all regions and quotas for tokens per minute
        for region_data in model_service_quotas.values():
            if isinstance(region_data, dict):
                for quota_code, quota_data in region_data.items():
                    if isinstance(quota_data, dict) and quota_code != 'quota_metadata':
                        quota_name = quota_data.get('quota_name', '').lower()
                        if 'tokens per minute' in quota_name:
                            value = quota_data.get('value', 0)
                            return f"{int(value)}" if value > 0 else 'N/A'
        return 'N/A'

    def extract_requests_per_minute_compact(model_service_quotas):
        if not isinstance(model_service_quotas, dict):
            return 'N/A'

        # Look through all regions and quotas for requests per minute
        for region_data in model_service_quotas.values():
            if isinstance(region_data, dict):
                for quota_code, quota_data in region_data.items():
                    if isinstance(quota_data, dict) and quota_code != 'quota_metadata':
                        quota_name = quota_data.get('quota_name', '').lower()
                        if 'requests per minute' in quota_name and 'on-demand' in quota_name:
                            value = quota_data.get('value', 0)
                            return f"{int(value)}" if value > 0 else 'N/A'
        return 'N/A'

    display_df['tokens_per_minute'] = display_df['model_service_quotas'].apply(extract_tokens_per_minute_compact)
    display_df['requests_per_minute'] = display_df['model_service_quotas'].apply(extract_requests_per_minute_compact)

    # Count regions for display
    display_df['region_count'] = display_df.apply(lambda row: len(extract_available_regions_from_pricing(row)), axis=1)
    
    # Add CSS for compact view styling
    st.markdown("""
    <style>
    .compact-model-row {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }
    
    .compact-model-row:hover {
        background: #ffffff;
        border-color: #cbd5e1;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .compact-header {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .compact-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 6px;
    }
    
    .compact-info {
        color: #64748b;
        font-size: 0.9rem;
        margin: 0.25rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Iterate through models and display in improved compact format
    for _, model in display_df.iterrows():
        model_id = model['model_id']
        
        # Format pricing
        input_price = format_price(model['input_price'])
        output_price = format_price(model['output_price'])
        
        # Format model ID for display
        model_id_short = model['model_id'].split(':')[0] if ':' in model['model_id'] else model['model_id']
        
        # Start compact row container
        st.markdown('<div class="compact-model-row">', unsafe_allow_html=True)
        
        # === ROW 1: Header with Provider, Name, Actions ===
        header_col1, header_col2 = st.columns([3, 0.5])
        
        with header_col1:
            # Provider badge and model info with consistent colors from card view
            provider_color = get_provider_color(model['provider'])
            
            # Status logic: get from model_lifecycle field
            model_lifecycle = model.get('model_lifecycle', {})
            if isinstance(model_lifecycle, dict):
                lifecycle_status = model_lifecycle.get('status', 'ACTIVE')
            else:
                lifecycle_status = 'ACTIVE'
            
            if lifecycle_status == 'LEGACY':
                status_display = 'LEGACY'
                status_color = '#F59E0B'  # Orange for Legacy
            else:
                status_display = 'ACTIVE'
                status_color = '#10B981'  # Green for Active
            
            # Model size based on context window with consistent colors from card view
            def get_model_size_from_context_window(context_window):
                """Determine model size based on context window size"""
                if context_window < 100000:  # < 100K tokens
                    return 'Small', '#F59E0B'  # Orange
                elif context_window <= 200000:  # 100K - 200K tokens
                    return 'Medium', '#3B82F6'  # Blue
                elif context_window <= 500000:  # > 200K - 500K tokens
                    return 'Large', '#10B981'  # Green
                else:  # > 500K tokens
                    return 'XLarge', '#8B5CF6'  # Purple
            
            context_window = get_converse_context_window(model)
            category, category_color = get_model_size_from_context_window(context_window)
            
            st.markdown(f"""
            <div class="compact-header">
                <span class="compact-badge" style="background: {provider_color}; color: white;">{model['provider']}</span>
                <span class="compact-badge" style="background: {status_color}; color: white;">{status_display}</span>
                <span class="compact-badge" style="background: {category_color}; color: white;">{category}</span>
            </div>
            <div style="margin-top: 0.5rem;">
                <div style="font-weight: bold; font-size: 1.1rem; color: #1e293b; margin-bottom: 0.25rem;">{model['name']}</div>
                <div style="color: #374151; font-size: 0.9rem; font-weight: 500; margin: 0.25rem 0;">
                    {model_id_short}{(' • Released: ' + model.get('release_date', '')) if model.get('release_date') and model.get('release_date') != 'N/A' else ''}{' • Legacy: ' + str(model.get('legacy_date', '')) if model.get('legacy_date') else ''}{' • EOL: ' + str(model.get('eol_date', '')) if model.get('eol_date') else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with header_col2:
            # Action buttons
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                # Favorite button
                is_fav = is_favorite(model['model_id'])
                if st.button("⭐" if is_fav else "☆", key=f"fav_compact_{model['model_id']}", help="Toggle favorite"):
                    toggle_favorite(model['model_id'])
                    st.rerun()
            
            with action_col2:
                # Compare button
                if 'comparison_models_with_regions' not in st.session_state:
                    st.session_state.comparison_models_with_regions = []
                is_comparing = any(item['model_id'] == model['model_id'] 
                                 for item in st.session_state.comparison_models_with_regions)
                if st.button("⚖️" if is_comparing else "⚖", key=f"compare_compact_{model['model_id']}", help="Toggle comparison"):
                    if is_comparing:
                        # Remove from comparison
                        st.session_state.comparison_models_with_regions = [
                            item for item in st.session_state.comparison_models_with_regions 
                            if item['model_id'] != model['model_id']
                        ]
                    else:
                        # Add to comparison with primary region if available, otherwise default region
                        available_regions = extract_available_regions_from_pricing(model)
                        # Ensure we have at least one region available
                        if not available_regions:
                            available_regions = ['us-east-1']

                        primary_region = st.session_state.get('filter_state', {}).get('primary_region', 'us-east-1')

                        # Use primary region if model is available in that region, otherwise use first available
                        selected_region = primary_region if primary_region in available_regions else available_regions[0]
                        
                        new_item = {
                            'model_id': model['model_id'],
                            'region': selected_region,
                            'comparison_id': f"{model['model_id']}_{selected_region}"
                        }
                        st.session_state.comparison_models_with_regions.append(new_item)
                    st.rerun()
        
        # === ROW 2: Technical Information Grid (6 columns) ===
        info_col1, info_col2, info_col3, info_col4, info_col5, info_col6 = st.columns(6)
        
        with info_col1:
            st.markdown(f"""
            <div>
                <div style="font-weight: 600; color: #374151;">⚙️ Technical Specifications</div>
                <div class="compact-info">Context window: {get_converse_context_window(model):,} tokens</div>
                <div class="compact-info">Max output: {get_converse_max_output_tokens(model) if get_converse_max_output_tokens(model) > 0 else 'N/A'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Service quotas available in View Details modal only
        
        with info_col3:
            # Features
            streaming_supported = model.get('streaming_supported', False)
            cris_supported = get_cross_region_inference_support(model)
            st.markdown(f"""
            <div>
                <div style="font-weight: 600; color: #374151;">🔧 Features</div>
                <div class="compact-info">🔄 Streaming: {'✅' if streaming_supported else '❌'}</div>
                <div class="compact-info">🌐 Cross-Region: {'✅' if cris_supported else '❌'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with info_col4:
            # Modalities
            input_modalities = model.get('input_modalities', [])
            output_modalities = model.get('output_modalities', [])
            st.markdown(f"""
            <div>
                <div style="font-weight: 600; color: #374151;">📥📤 I/O</div>
                <div class="compact-info">In: {', '.join(input_modalities[:2]) if input_modalities else 'N/A'}</div>
                <div class="compact-info">Out: {', '.join(output_modalities[:2]) if output_modalities else 'N/A'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with info_col5:
            # Availability - regions and consumption options (excluding cross-region)
            regions_count = len(extract_available_regions_from_pricing(model))
            filtered_options = get_consumption_options(model)
            
            # Map consumption options to shorter display names
            option_names = {
                'on_demand': 'On-Demand',
                'provisioned_throughput': 'Provisioned',
                'batch': 'Batch'
            }
            
            consumption_display = ', '.join([option_names.get(opt, opt.replace('_', ' ').title()) for opt in filtered_options[:3]])
            if len(filtered_options) > 3:
                consumption_display += f' +{len(filtered_options)-3}'
            
            st.markdown(f"""
            <div>
                <div style="font-weight: 600; color: #374151;">🗺️ Availability</div>
                <div class="compact-info">{regions_count} regions</div>
                <div class="compact-info">{consumption_display if consumption_display else 'On-Demand'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with info_col6:
            # Pricing
            st.markdown(f"""
            <div>
                <div style="font-weight: 600; color: #374151;">💰 Pricing</div>
                <div class="compact-info">In: {input_price}/1K</div>
                <div class="compact-info">Out: {output_price}/1K</div>
            </div>
            """, unsafe_allow_html=True)
        
        # === ROW 3: Description, Capabilities, and Use Cases ===
        desc_col1, desc_col2, desc_col3 = st.columns([2, 1, 1])
        
        with desc_col1:
            # Description
            '''
            description = model.get('description', 'No description available')
            description_short = description[:120] + ('...' if len(description) > 120 else '')
            st.markdown(f"""
            <div>
                <div style="font-weight: 600; color: #374151; margin-bottom: 0.25rem;">📝 Description</div>
                <div class="compact-info" style="font-style: italic;">{description_short}</div>
            </div>
            """, unsafe_allow_html=True)
            '''
        
        with desc_col2:
            # Capabilities with consistent styling from card view
            capabilities = model.get('model_capabilities', [])
            if capabilities:
                caps_html = "<div style='margin: 4px 0;'>"
                for cap in capabilities[:3]:  # Show first 3 capabilities
                    caps_html += f"<span style='display: inline-block; margin: 2px; padding: 4px 8px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; font-size: 0.7rem; font-weight: 500;'>{cap}</span> "
                if len(capabilities) > 3:
                    caps_html += f"<span style='color: #6B7280; font-size: 0.7rem;'>+{len(capabilities)-3} more</span>"
                caps_html += "</div>"
                
                st.markdown(f"""
                <div>
                    <div style="font-weight: 600; color: #374151;">🎯 Capabilities</div>
                    {caps_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div>
                    <div style="font-weight: 600; color: #374151;">🎯 Capabilities</div>
                    <div class="compact-info">N/A</div>
                </div>
                """, unsafe_allow_html=True)
        
        with desc_col3:
            # Use Cases with consistent styling from card view
            use_cases = model.get('model_use_cases', [])
            if use_cases:
                use_cases_html = "<div style='margin: 4px 0;'>"
                for use_case in use_cases[:3]:  # Show first 3 use cases
                    use_cases_html += f"<span style='display: inline-block; margin: 2px; padding: 4px 8px; background: #F0F9FF; color: #0369A1; border: 1px solid #BAE6FD; border-radius: 12px; font-size: 0.7rem;'>{use_case}</span> "
                if len(use_cases) > 3:
                    use_cases_html += f"<span style='color: #6B7280; font-size: 0.7rem;'>+{len(use_cases)-3} more</span>"
                use_cases_html += "</div>"
                
                st.markdown(f"""
                <div>
                    <div style="font-weight: 600; color: #374151;">💡 Use Cases</div>
                    {use_cases_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div>
                    <div style="font-weight: 600; color: #374151;">💡 Use Cases</div>
                    <div class="compact-info">N/A</div>
                </div>
                """, unsafe_allow_html=True)
        
        # End compact row container
        st.markdown('</div>', unsafe_allow_html=True)
def show_model_comparison(df: pd.DataFrame):
    """Display the enhanced model comparison interface with region selection"""
    # Initialize comparison models with region support
    if 'comparison_models_with_regions' not in st.session_state:
        st.session_state.comparison_models_with_regions = []
    
    # Migrate old comparison_models to new format if needed
    if 'comparison_models' in st.session_state and st.session_state.comparison_models:
        for model_id in st.session_state.comparison_models:
            # Check if this model is already in the new format
            existing = any(item['model_id'] == model_id for item in st.session_state.comparison_models_with_regions)
            if not existing:
                # Get the model's available regions and use primary region if available
                model_row = df[df['model_id'] == model_id]
                if not model_row.empty:
                    available_regions = extract_available_regions_from_pricing(model_row.iloc[0])
                    # Ensure we have at least one region available
                    if not available_regions:
                        available_regions = ['us-east-1']

                    primary_region = st.session_state.get('filter_state', {}).get('primary_region', 'us-east-1')

                    # Use primary region if model is available in that region, otherwise use first available
                    selected_region = primary_region if primary_region in available_regions else available_regions[0]
                    
                    st.session_state.comparison_models_with_regions.append({
                        'model_id': model_id,
                        'region': selected_region,
                        'comparison_id': f"{model_id}_{selected_region}"
                    })
        # Clear old format
        st.session_state.comparison_models = set()
    
    comparison_items = st.session_state.comparison_models_with_regions
    
    if not comparison_items:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 12px; margin: 2rem 0;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">⚖️</div>
            <h2 style="color: #1e293b; margin-bottom: 1rem;">No Models Selected for Comparison</h2>
            <p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">Select models from the Model Explorer to compare them side-by-side with region-specific data</p>
            <div style="background: white; padding: 1.5rem; border-radius: 8px; text-align: left; max-width: 600px; margin: 0 auto;">
                <h4 style="color: #374151; margin-bottom: 1rem;">How to select models:</h4>
                <div style="color: #6b7280;">
                    <p>📋 <strong>Card View:</strong> Click the ⚖️ button on any model card</p>
                    <p>📊 <strong>Table View:</strong> Check the "Select" boxes</p>
                    <p>📝 <strong>Compact View:</strong> Click the ⚖️ button in the header</p>
                </div>
                <div style="margin-top: 1rem; padding: 1rem; background: #f0f9ff; border-radius: 6px; border-left: 4px solid #3b82f6;">
                    <h5 style="color: #1e40af; margin: 0 0 0.5rem 0;">🌍 Region-Specific Comparison</h5>
                    <p style="color: #1e40af; margin: 0; font-size: 0.9rem;">Each model can be compared with different regions to see region-specific pricing, quotas, and availability.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Get unique model IDs for filtering
    model_ids = [item['model_id'] for item in comparison_items]
    comparison_df = df[df['model_id'].isin(model_ids)]
    
    if comparison_df.empty:
        st.warning("Selected models not found in database.")
        return
    
    # Add CSS for comparison view
    st.markdown("""
    <style>
    
    .comparison-header {
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }
    
    .comparison-section {
        margin-bottom: 1.5rem;
    }
    
    .comparison-label {
        font-weight: 600;
        color: #374151;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .comparison-value {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }
    
    .comparison-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 2px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display comparison header with background
    st.markdown("""
    <div style="background: linear-gradient(90deg, rgb(31, 78, 121) 0%, rgb(45, 90, 160) 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0; text-align: center;">⚖️ Models Comparison</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Add custom CSS for button alignment and styling
    st.markdown("""
    <style>
    /* Ensure buttons are vertically aligned */
    div[data-testid="column"] {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
    }
    
    /* Custom styling for the add model button */
    div[data-testid="stButton"] button[key="add_model_region"] {
        background-color: #e3f2fd !important;
        color: #1565c0 !important;
        border: 1px solid #90caf9 !important;
    }
    div[data-testid="stButton"] button[key="add_model_region"]:hover {
        background-color: #bbdefb !important;
        border-color: #64b5f6 !important;
    }
    
    /* Ensure both buttons have the same height and alignment */
    .stButton > button {
        height: 2.5rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear All Selected Models", key="clear_comparison_view", width="stretch", type="secondary"):
            st.session_state.comparison_models_with_regions.clear()
            st.rerun()
    
    with col2:
        if st.button("➕ Add Model to comparison", key="add_model_region", width="stretch"):
            st.session_state.show_add_model_region = True
    
    # Add model with different region functionality
    if st.session_state.get('show_add_model_region', False):
        with st.expander("➕ Add Model with Different Region", expanded=True):
            # Select model
            available_models = [(row['model_id'], row['model_name']) for _, row in df.iterrows()]
            selected_model_tuple = st.selectbox(
                "Select Model:",
                options=available_models,
                format_func=lambda x: f"{x[1]} ({x[0]})",
                key="add_model_select"
            )
            
            if selected_model_tuple:
                selected_model_id = selected_model_tuple[0]
                model_row = df[df['model_id'] == selected_model_id].iloc[0]
                available_regions = extract_available_regions_from_pricing(model_row)
                # Ensure we have at least one region available
                if not available_regions:
                    available_regions = ['us-east-1']
                
                # Select region
                selected_region = st.selectbox(
                    "Select Region:",
                    options=available_regions,
                    key="add_region_select"
                )
                
                # Check if this combination already exists
                existing_combo = any(
                    item['model_id'] == selected_model_id and item['region'] == selected_region 
                    for item in comparison_items
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Add to Comparison", disabled=existing_combo, width="stretch"):
                        new_item = {
                            'model_id': selected_model_id,
                            'region': selected_region,
                            'comparison_id': f"{selected_model_id}_{selected_region}"
                        }
                        st.session_state.comparison_models_with_regions.append(new_item)
                        st.session_state.show_add_model_region = False
                        st.rerun()
                
                with col2:
                    if st.button("Cancel", width="stretch"):
                        st.session_state.show_add_model_region = False
                        st.rerun()
                
                if existing_combo:
                    st.warning("This model-region combination is already in the comparison.")
    
    # Extract pricing information for display using comprehensive pricing with region-specific data
    def get_pricing_from_model_region(model_row, region):
        """Extract pricing from model using comprehensive pricing for specific region"""
        input_price, output_price, _ = extract_comprehensive_pricing_info(model_row, region)
        if input_price > 0 or output_price > 0:
            return input_price, output_price
        
        # Fallback to legacy pricing if comprehensive pricing not available
        pricing_data = model_row.get('pricing', {})
        return extract_pricing_info(pricing_data)
    
    # Create enhanced comparison data with region-specific information
    enhanced_comparison_data = []
    for item in comparison_items:
        model_row = comparison_df[comparison_df['model_id'] == item['model_id']].iloc[0]
        input_price, output_price = get_pricing_from_model_region(model_row, item['region'])
        
        enhanced_data = model_row.copy()
        enhanced_data['selected_region'] = item['region']
        enhanced_data['comparison_id'] = item['comparison_id']
        enhanced_data['input_price'] = input_price
        enhanced_data['output_price'] = output_price
        enhanced_comparison_data.append(enhanced_data)
    
    # Create hybrid model cards for comparison with region selection - ALWAYS 5 COLUMNS LAYOUT
    with st.expander("## Model Overview with Region Selection", expanded=True):
        # Display models in consistent 5-column layout regardless of model count
        rows = [enhanced_comparison_data[i:i+5] for i in range(0, len(enhanced_comparison_data), 5)]
        
        for row_idx, row in enumerate(rows):
            # Always create 5 columns for consistent sizing
            cols = st.columns(5)
            
            for col_idx, model in enumerate(row):
                with cols[col_idx]:
                    # Start comparison card (no white background div)
                    
                    # === HEADER SECTION WITH REGION SELECTOR ===
                    provider_color = get_provider_color(model['provider'])
                    
                    # Status logic: get from model_lifecycle field
                    model_lifecycle = model.get('model_lifecycle', {})
                    if isinstance(model_lifecycle, dict):
                        lifecycle_status = model_lifecycle.get('status', 'ACTIVE')
                    else:
                        lifecycle_status = 'ACTIVE'
                    
                    if lifecycle_status == 'LEGACY':
                        status_display = 'LEGACY'
                        status_color = '#F59E0B'
                    else:
                        status_display = 'ACTIVE'
                        status_color = '#10B981'
                    
                    # Model size based on context window
                    def get_model_size_from_context_window(context_window):
                        """Determine model size based on context window size"""
                        if context_window < 100000:  # < 100K tokens
                            return 'Small', '#F59E0B'  # Orange
                        elif context_window <= 200000:  # 100K - 200K tokens
                            return 'Medium', '#3B82F6'  # Blue
                        elif context_window <= 500000:  # > 200K - 500K tokens
                            return 'Large', '#10B981'  # Green
                        else:  # > 500K tokens
                            return 'XLarge', '#8B5CF6'  # Purple
                    
                    context_window = get_converse_context_window(model)
                    category, category_color = get_model_size_from_context_window(context_window)
                    
                    st.markdown(f"""
                    <div class="comparison-header">
                        <div style="margin-bottom: 0.75rem;">
                            <span class="comparison-badge" style="background: {provider_color}; color: white;">{model['provider']}</span>
                            <span class="comparison-badge" style="background: {status_color}; color: white;">{status_display}</span>
                            <span class="comparison-badge" style="background: {category_color}; color: white;">{category}</span>
                        </div>
                        <h3 style="margin: 0; color: #1e293b; font-size: 1.2rem;">{model['name']}</h3>
                        <div style="color: #6b7280; font-size: 0.85rem; margin-top: 0.25rem;">
                            {model['model_id'].split(':')[0] if ':' in model['model_id'] else model['model_id']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # === REGION SELECTOR ===
                    # Get available regions from pricing section in models JSON
                    available_regions = extract_available_regions_from_pricing(model)

                    # Ensure we have at least one region available
                    if not available_regions:
                        available_regions = ['us-east-1']

                    current_region = model['selected_region']

                    # Region selector
                    new_region = st.selectbox(
                        "🌍 Region:",
                        options=available_regions,
                        index=available_regions.index(current_region) if current_region in available_regions else 0,
                        key=f"region_select_{model['comparison_id']}_{row_idx}_{col_idx}"
                    )
                    
                    # Update region if changed
                    if new_region != current_region:
                        # Find and update the item in session state
                        for item in st.session_state.comparison_models_with_regions:
                            if item['comparison_id'] == model['comparison_id']:
                                item['region'] = new_region
                                item['comparison_id'] = f"{item['model_id']}_{new_region}"
                                break
                        st.rerun()
                    
                    # === TECHNICAL SPECS ===
                    streaming_supported = model.get('streaming_supported', False)
                    cris_supported = get_cross_region_inference_support(model)
                    st.markdown(f"""
                    <div class="comparison-section">
                        <div class="comparison-label">⚙️ Technical Specifications</div>
                        <div class="comparison-value">Context: {get_converse_context_window(model):,} tokens</div>
                        <div class="comparison-value">Streaming: {'✅ Yes' if streaming_supported else '❌ No'}</div>
                        <div class="comparison-value">Cross-Region: {'✅ Yes' if cris_supported else '❌ No'}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # === AVAILABILITY ===
                    # Get all available regions from pricing data (not just the first 3)
                    all_available_regions = extract_available_regions_from_pricing(model)
                    regions_count = len(all_available_regions)
                    filtered_options = get_consumption_options(model)
                    consumption_display = ', '.join([opt.replace('_', ' ').title() for opt in filtered_options[:2]])

                    st.markdown(f"""
                    <div class="comparison-section">
                        <div class="comparison-label">🗺️ Availability</div>
                        <div class="comparison-value">{regions_count} total regions</div>
                        <div class="comparison-value">{consumption_display if consumption_display else 'On-Demand'}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # === CAPABILITIES ===
                    capabilities = model.get('model_capabilities', [])
                    if capabilities:
                        caps_html = "<div style='margin-top: 0.5rem;'>"
                        for cap in capabilities[:4]:  # Show first 4 capabilities
                            caps_html += f"<span class='comparison-badge' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>{cap}</span>"
                        if len(capabilities) > 4:
                            caps_html += f"<span style='color: #6B7280; font-size: 0.75rem;'>+{len(capabilities)-4} more</span>"
                        caps_html += "</div>"
                        
                        st.markdown(f"""
                        <div class="comparison-section">
                            <div class="comparison-label">🎯 Capabilities</div>
                            {caps_html}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # === REMOVE BUTTON ===
                    if st.button("Remove", key=f"remove_{model['comparison_id']}_{row_idx}_{col_idx}", width="stretch"):
                        # Remove from session state
                        st.session_state.comparison_models_with_regions = [
                            item for item in st.session_state.comparison_models_with_regions 
                            if item['comparison_id'] != model['comparison_id']
                        ]
                        st.rerun()
                    
                    # End comparison card (no closing div needed)
    
    # === DETAILED COMPARISON TABLES ===
    st.markdown("---")
    st.markdown("## Detailed Comparison")
    
    # Create tabs for different comparison aspects
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Radar Overview", "🗺️ Availability", "💰 Pricing", "⚙️ Technical Specs"])
    
    with tab1:
        # Simple radar chart comparison
        st.markdown("### Model Comparison Radar")

        try:
            import plotly.graph_objects as go

            radar_data = []
            all_costs = [m['input_price'] + m['output_price'] for m in enhanced_comparison_data if m['input_price'] + m['output_price'] > 0]
            max_cost = max(all_costs) if all_costs else 1

            for model in enhanced_comparison_data:
                # Simple scoring based on key differentiators

                # Cost (lower = better, 0-10 scale)
                total_cost = model['input_price'] + model['output_price']
                cost_score = 10 - (total_cost / max_cost * 10) if total_cost > 0 else 5

                # Context window (0-10 scale)
                context = get_converse_context_window(model)
                context_score = min(context / 100000, 10) if context > 0 else 2

                # Regions availability
                regions = len(extract_available_regions_from_pricing(model))
                region_score = min(regions / 2, 10)

                # Features (batch, streaming, etc.)
                features = 0
                if model.get('streaming_supported'): features += 3
                if model.get('batch_inference_supported', {}).get('supported'): features += 3
                if get_cross_region_inference_support(model): features += 4
                feature_score = min(features, 10)

                radar_data.append({
                    'Model': model['name'],
                    'Cost Efficiency': cost_score,
                    'Context Window': context_score,
                    'Availability': region_score,
                    'Features': feature_score
                })

            # Create simple radar chart
            fig = go.Figure()
            categories = ['Cost Efficiency', 'Context Window', 'Availability', 'Features']

            for data in radar_data:
                fig.add_trace(go.Scatterpolar(
                    r=[data[cat] for cat in categories],
                    theta=categories,
                    fill='toself',
                    name=data['Model']
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=True,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

            # Scoring methodology legend
            st.markdown("""
            **📊 Radar Scoring Methodology:**
            - **Cost Efficiency**: Lower pricing = higher score (10 = cheapest, 0 = most expensive)
            - **Context Window**: Token capacity (10 = 1M+ tokens, 2 = <100K tokens)
            - **Availability**: Regional coverage (10 = 20+ regions, lower = fewer regions)
            - **Features**: Advanced capabilities (Streaming +3, Batch +3, CRIS +4, max 10)

            *All scores on 0-10 scale for easy comparison*
            """)

        except ImportError:
            st.error("Plotly is required for radar charts. Please install plotly to view this visualization.")
        except Exception as e:
            st.error(f"Error creating radar chart: {e}")
    
    with tab2:
        # Simplified Availability Overview with World Map
        # Consolidate models (remove duplicates, show unique models with all regions)
        unique_models = {}
        for model in enhanced_comparison_data:
            model_id = model['model_id']
            if model_id not in unique_models:
                unique_models[model_id] = {
                    'name': model['name'],
                    'provider': model.get('provider', 'Unknown'),
                    'regions': set(extract_available_regions_from_pricing(model)),
                    'consumption_options': get_consumption_options(model),
                    'cross_region_supported': get_cross_region_inference_support(model)
                }
            else:
                # Add regions from this instance
                unique_models[model_id]['regions'].update(extract_available_regions_from_pricing(model))

        # Add some spacing before the summary section
        st.markdown("<br>", unsafe_allow_html=True)

        # === SECTION 1: SUMMARY ===
        col0, col1, col2, col3 = st.columns([.5, 1, 1, 1])

        with col1:
            total_unique_models = len(unique_models)
            st.metric("🤖 Unique Models", f"{total_unique_models}")

        with col2:
            avg_regions_per_model = sum(len(info['regions']) for info in unique_models.values()) / len(unique_models) if unique_models else 0
            st.metric("🗺️ Avg Regions", f"{avg_regions_per_model:.1f}")

        with col3:
            cris_count = sum(1 for info in unique_models.values() if info['cross_region_supported'])
            st.metric("🌐 CRIS Models", f"{cris_count}/{total_unique_models}")

        st.markdown("---")
        
        # === SECTION 2: MODEL AVAILABILITY COMPARISON ===
        st.markdown("### 📊 Model Availability Comparison")

        table_data = []
        for model_id, model_info in unique_models.items():
            regions_list = sorted(list(model_info['regions']))
            consumption_opts = ', '.join([opt.replace('_', ' ').title() for opt in model_info['consumption_options']])

            table_data.append({
                'Model': model_info['name'],
                'Provider': model_info['provider'],
                'Regions': f"{len(regions_list)} regions",
                'Region List': ', '.join(regions_list[:3]) + (f' +{len(regions_list)-3} more' if len(regions_list) > 3 else ''),
                'Consumption Options': consumption_opts,
                'Cross-Region Inference': '✅ Yes' if model_info['cross_region_supported'] else '❌ No'
            })

        availability_df = pd.DataFrame(table_data)

        # Display clean availability table
        st.dataframe(availability_df, width="stretch", hide_index=True)

        # === SECTION 3: REGION LOOKUP ===
        st.markdown("### 🔍 Region Lookup")

        # Create region-to-models mapping - only use valid AWS region codes
        region_models = {}
        valid_region_pattern = r'^[a-z]{2}-[a-z]+-\d+$'  # Pattern for AWS region codes like us-east-1

        for model_id, model_info in unique_models.items():
            for region in model_info['regions']:
                # Only include proper AWS region codes
                if isinstance(region, str) and region and '-' in region and len(region) <= 20:
                    if region not in region_models:
                        region_models[region] = []
                    region_models[region].append({
                        'name': model_info['name'],
                        'provider': model_info['provider']
                    })

        # Sort regions by number of models (descending)
        sorted_regions = sorted(region_models.items(), key=lambda x: len(x[1]), reverse=True)

        if sorted_regions:
            # Create region name mapping for better display
            region_name_map = {
                'us-east-1': 'N. Virginia',
                'us-east-2': 'Ohio',
                'us-west-1': 'N. California',
                'us-west-2': 'Oregon',
                'eu-west-1': 'Ireland',
                'eu-west-2': 'London',
                'eu-west-3': 'Paris',
                'eu-central-1': 'Frankfurt',
                'ap-southeast-1': 'Singapore',
                'ap-southeast-2': 'Sydney',
                'ap-northeast-1': 'Tokyo',
                'ap-northeast-2': 'Seoul',
                'ap-south-1': 'Mumbai',
                'ca-central-1': 'Canada Central',
                'sa-east-1': 'São Paulo'
            }

            # Compact dropdown selector
            col1, col2 = st.columns([1, 2])

            with col1:
                # Create better formatted options with region names
                region_options = []
                for region, models in sorted_regions:
                    region_name = region_name_map.get(region, region.replace('-', ' ').title())
                    display_text = f"{region_name} ({region}) - {len(models)} models"
                    region_options.append(display_text)

                selected_region_display = st.selectbox(
                    "Select a region:",
                    options=region_options,
                    key="region_availability_selector"
                )

            with col2:
                if selected_region_display:
                    # Extract actual region code from the display text
                    # Format: "N. Virginia (us-east-1) - 3 models"
                    import re
                    match = re.search(r'\(([^)]+)\)', selected_region_display)
                    selected_region = match.group(1) if match else selected_region_display.split(' - ')[0]
                    models = region_models.get(selected_region, [])

                    if models:
                        # Group by provider and display with visual enhancements
                        provider_groups = {}
                        for model in models:
                            provider = model['provider']
                            if provider not in provider_groups:
                                provider_groups[provider] = []
                            provider_groups[provider].append(model['name'])

                        # Display with visual badges using simpler HTML
                        for provider, model_names in sorted(provider_groups.items()):
                            # Get provider color for visual consistency
                            provider_color = get_provider_color(provider)

                            # Provider header with badge
                            st.markdown(f"""
                            <div style="margin: 12px 0 8px 0;">
                                <span style="background: {provider_color}; color: white; padding: 6px 12px; border-radius: 20px; font-size: 0.9rem; font-weight: bold; margin-right: 8px;">
                                    {provider}
                                </span>
                                <span style="color: #6b7280; font-size: 0.85rem;">
                                    {len(model_names)} model{"s" if len(model_names) > 1 else ""}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)

                            # Distribute models across columns
                            sorted_models = sorted(model_names)
                            models_per_row = 4  # 4 columns for better space utilization

                            # Create column grid for models
                            models_html = '<div style="margin: 8px 0;">'

                            for i in range(0, len(sorted_models), models_per_row):
                                models_html += '<div style="display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px;">'

                                row_models = sorted_models[i:i + models_per_row]
                                for model_name in row_models:
                                    models_html += f'''
                                    <div style="flex: 1; min-width: 150px; max-width: 200px;">
                                        <span style="display: block; padding: 6px 10px; background: #f1f5f9; color: #334155;
                                              border-radius: 12px; font-size: 0.8rem; border: 1px solid #e2e8f0;
                                              text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                                            {model_name}
                                        </span>
                                    </div>'''

                                models_html += '</div>'

                            models_html += '</div>'
                            st.markdown(models_html, unsafe_allow_html=True)
                    else:
                        st.info("No models found for this region")
        else:
            st.info("No valid regions found in the data")

        # Add some spacing before the world map section
        st.markdown("<br>", unsafe_allow_html=True)

        # === SECTION 4: MODEL AVAILABILITY MAP ===
        st.markdown("### 🌍 Model Availability Map")

        # Selected models legend placeholder - will be updated after processing
        legend_placeholder = st.empty()

        try:
            import plotly.express as px
            # AWS region coordinates mapping
            region_coords = {
                'us-east-1': {'lat': 39.0458, 'lon': -77.5081, 'name': 'N. Virginia'},
                'us-east-2': {'lat': 39.9612, 'lon': -82.9988, 'name': 'Ohio'},
                'us-west-1': {'lat': 37.3541, 'lon': -121.9552, 'name': 'N. California'},
                'us-west-2': {'lat': 47.6062, 'lon': -122.3321, 'name': 'Oregon'},
                'eu-west-1': {'lat': 53.4084, 'lon': -8.2439, 'name': 'Ireland'},
                'eu-west-2': {'lat': 51.5074, 'lon': -0.1278, 'name': 'London'},
                'eu-west-3': {'lat': 48.8566, 'lon': 2.3522, 'name': 'Paris'},
                'eu-central-1': {'lat': 50.1109, 'lon': 8.6821, 'name': 'Frankfurt'},
                'ap-southeast-1': {'lat': 1.3521, 'lon': 103.8198, 'name': 'Singapore'},
                'ap-southeast-2': {'lat': -33.8688, 'lon': 151.2093, 'name': 'Sydney'},
                'ap-northeast-1': {'lat': 35.6762, 'lon': 139.6503, 'name': 'Tokyo'},
                'ap-northeast-2': {'lat': 37.5665, 'lon': 126.9780, 'name': 'Seoul'},
                'ap-south-1': {'lat': 19.0760, 'lon': 72.8777, 'name': 'Mumbai'},
                'ca-central-1': {'lat': 45.4215, 'lon': -75.6972, 'name': 'Canada'},
                'sa-east-1': {'lat': -23.5505, 'lon': -46.6333, 'name': 'São Paulo'}
            }

            # Create individual markers for each model in each region (like Endpoint Legend)
            map_data = []
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF']

            # Group models by region to create offset positions
            region_models = {}
            for model_id, model_info in unique_models.items():
                for region in model_info['regions']:
                    if region not in region_models:
                        region_models[region] = []
                    region_models[region].append((model_id, model_info))

            for region, models_in_region in region_models.items():
                if region in region_coords:
                    coords = region_coords[region]
                    base_lat = coords['lat']
                    base_lon = coords['lon']

                    # Create offset positions for multiple models in same region
                    offset = 0.3  # degrees offset for visibility
                    positions = [
                        (base_lat, base_lon),  # center
                        (base_lat + offset, base_lon + offset),  # top-right
                        (base_lat - offset, base_lon + offset),  # bottom-right
                        (base_lat + offset, base_lon - offset),  # top-left
                        (base_lat - offset, base_lon - offset),  # bottom-left
                        (base_lat, base_lon + offset),  # right
                        (base_lat, base_lon - offset),  # left
                        (base_lat + offset, base_lon),  # top
                        (base_lat - offset, base_lon),  # bottom
                    ]

                    for i, (model_id, model_info) in enumerate(models_in_region):
                        # Use position based on model index, cycle through positions
                        pos = positions[i % len(positions)]

                        model_display = model_info['name'][:25] + '...' if len(model_info['name']) > 25 else model_info['name']

                        map_data.append({
                            'Location': coords['name'],
                            'Region': region,
                            'Provider': model_info['provider'],
                            'Model': model_display,
                            'Hover_Name': model_display,
                            'lat': pos[0],
                            'lon': pos[1],
                            'Color': colors[list(unique_models.keys()).index(model_id) % len(colors)],
                            'Model_ID': model_id
                        })

            if map_data:
                map_df = pd.DataFrame(map_data)

                # Convert to pydeck format with model-based colors
                try:
                    import pydeck as pdk
                    import random

                    # Generate random colors for each unique model
                    unique_model_ids = list(unique_models.keys())
                    model_colors = {}

                    for i, model_id in enumerate(unique_model_ids):
                        # Generate deterministic bright colors for each model (avoid random for security compliance)
                        # Use deterministic color generation based on index
                        r = 100 + (i * 37 % 156)  # 100-255 range
                        g = 100 + (i * 67 % 156)  # 100-255 range
                        b = 100 + (i * 97 % 156)  # 100-255 range
                        model_colors[model_id] = [r, g, b, 200]

                    pydeck_data = []
                    for item in map_data:
                        model_id = item['Model_ID']
                        color = model_colors.get(model_id, [128, 128, 128, 200])  # Default grey

                        pydeck_data.append({
                            'lat': item['lat'],
                            'lon': item['lon'],
                            'location': item['Location'],
                            'region': item['Region'],
                            'provider': item['Provider'],
                            'model': item['Model'],
                            'model_id': model_id,
                            'color': color
                        })

                    points_df = pd.DataFrame(pydeck_data)

                    # Use same pydeck configuration as Endpoint Legend & Summary
                    st.pydeck_chart(pdk.Deck(
                        map_style=None,
                        initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1.5, pitch=0),
                        layers=[pdk.Layer('ScatterplotLayer', data=points_df, get_position='[lon, lat]',
                                          get_color='color', get_radius=80, radius_scale=1000,
                                          radius_min_pixels=8, radius_max_pixels=25, pickable=True)],
                        tooltip={'html': '<b>Model:</b> {model}<br><b>Provider:</b> {provider}<br><b>Region:</b> {location} ({region})',
                                 'style': {'backgroundColor': 'steelblue', 'color': 'white'}}
                    ))

                    # Calculate model totals and update legend
                    model_counts = {}
                    model_regions = {}
                    for item in pydeck_data:
                        model_name = item['model']
                        model_id = item['model_id']
                        if model_id not in model_counts:
                            model_counts[model_id] = 0
                            model_regions[model_id] = set()
                        model_counts[model_id] += 1
                        model_regions[model_id].add(item['region'])

                    # Update legend with selected models
                    with legend_placeholder.container():
                        models_list = [(model_id, unique_models[model_id]['name'], len(model_regions[model_id]))
                                      for model_id in model_counts.keys()]
                        num_cols = min(len(models_list), 4)  # Max 4 columns
                        cols = st.columns(num_cols)

                        # Generate color indicators for legend
                        for i, (model_id, model_name, region_count) in enumerate(models_list):
                            col_index = i % num_cols
                            with cols[col_index]:
                                # Create color indicator using the model's color
                                color = model_colors[model_id]
                                color_style = f"rgb({color[0]}, {color[1]}, {color[2]})"

                                provider = unique_models[model_id]['provider']
                                st.markdown(f"""
                                <div style="text-align: center; padding: 10px;">
                                    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 8px;">
                                        <div style="width: 15px; height: 15px; border-radius: 50%; background-color: {color_style}; margin-right: 8px;"></div>
                                        <strong>{model_name}</strong>
                                    </div>
                                    <div style="margin-bottom: 6px; font-size: 0.9rem;">
                                        Available in <strong>{region_count} regions</strong>
                                    </div>
                                    <div style="color: #666; font-size: 0.85rem;">
                                        {provider}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                    total_regions = len(set(item['region'] for item in pydeck_data))
                    st.info(f"🗺️ **Selected Models Available in {total_regions} regions** across {len(model_counts)} models")

                except ImportError:
                    st.error("Pydeck is required for the map visualization.")
            else:
                st.info("No region coordinate data available for mapping")

        except ImportError as e:
            st.warning(f"Map visualization unavailable: Plotly is required for map visualization. Please install plotly: {e}")
        except Exception as e:
            st.warning(f"Map visualization unavailable: {e}")

    with tab3:
        # === SECTION 1: KEY PRICING METRICS ===
        st.markdown("<br>", unsafe_allow_html=True)

        # Calculate pricing metrics
        total_costs = [model['input_price'] + model['output_price'] for model in enhanced_comparison_data if model['input_price'] + model['output_price'] > 0]

        # Summary metrics using st.metric (like availability tab)
        col0, col1, col2, col3 = st.columns([.5, 1, 1, 1])

        with col1:
            avg_cost = sum(total_costs) / len(total_costs) if total_costs else 0
            st.metric("💰 Average Cost", f"${avg_cost:.2f}/1K tokens")

        with col2:
            min_cost = min(total_costs) if total_costs else 0
            max_cost = max(total_costs) if total_costs else 0
            st.metric("📊 Price Range", f"${min_cost:.2f} - ${max_cost:.2f}")

        with col3:
            budget_models = len([c for c in total_costs if c < 5]) if total_costs else 0
            premium_models = len([c for c in total_costs if c >= 15]) if total_costs else 0
            st.metric("🎯 Budget/Premium", f"{budget_models}/{premium_models}")

        st.markdown("---")

        # === SECTION 2: REGIONAL PRICE ANALYSIS ===

        # Create comprehensive regional pricing analysis
        regional_analysis = {}
        savings_opportunities = []

        for model in enhanced_comparison_data:
            model_id = model['model_id']
            model_name = model['name']

            # Get all available regions for this model
            available_regions = extract_available_regions_from_pricing(model)

            if not available_regions:
                continue

            regional_prices = {}
            for region in available_regions:
                # Get pricing for each region
                input_price, output_price, pricing_region = extract_comprehensive_pricing_info(model, region)
                total_cost = input_price + output_price
                if total_cost > 0:
                    regional_prices[region] = {
                        'input': input_price,
                        'output': output_price,
                        'total': total_cost
                    }

            if len(regional_prices) > 1:  # Only analyze models with multiple regions
                regional_analysis[model_id] = {
                    'name': model_name,
                    'provider': model.get('provider', 'Unknown'),
                    'prices': regional_prices,
                    'cheapest_region': min(regional_prices.keys(), key=lambda r: regional_prices[r]['total']),
                    'most_expensive_region': max(regional_prices.keys(), key=lambda r: regional_prices[r]['total'])
                }

                # Calculate savings opportunity
                min_cost = min(p['total'] for p in regional_prices.values())
                max_cost = max(p['total'] for p in regional_prices.values())
                if max_cost > min_cost:
                    savings_pct = ((max_cost - min_cost) / max_cost) * 100
                    savings_opportunities.append({
                        'model': model_name,
                        'savings_pct': savings_pct,
                        'savings_amount': max_cost - min_cost,
                        'cheapest_region': regional_analysis[model_id]['cheapest_region'],
                        'expensive_region': regional_analysis[model_id]['most_expensive_region'],
                        'min_cost': min_cost,
                        'max_cost': max_cost
                    })

        if regional_analysis:
            # === SAVINGS OPPORTUNITIES SUMMARY ===
            if savings_opportunities:
                st.markdown("#### 💡 Cost Optimization Opportunities")

                # Sort by savings percentage
                savings_opportunities.sort(key=lambda x: x['savings_pct'], reverse=True)

                for i, opportunity in enumerate(savings_opportunities[:3]):  # Show top 3
                    savings_color = "#10B981" if opportunity['savings_pct'] >= 20 else "#F59E0B" if opportunity['savings_pct'] >= 10 else "#6B7280"

                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {savings_color}15 0%, {savings_color}05 100%);
                                border: 1px solid {savings_color}40; border-radius: 8px; padding: 12px; margin: 8px 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="color: #1f2937;">{opportunity['model']}</strong><br>
                                <small style="color: #6b7280;">
                                    Switch from {get_region_display_name(opportunity['expensive_region'])}
                                    → {get_region_display_name(opportunity['cheapest_region'])}
                                </small>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 1.2rem; font-weight: bold; color: {savings_color};">
                                    {opportunity['savings_pct']:.1f}% savings
                                </div>
                                <small style="color: #6b7280;">
                                    ${opportunity['savings_amount']:.4f}/1K tokens
                                </small>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")

            # === PRICING MATRIX TABLE ===
            st.markdown("#### 📊 Regional Pricing Matrix")

            # Get selected region from filter state for comparison
            selected_region = st.session_state.get('filter_state', {}).get('primary_region', 'us-east-1')
            selected_region_name = get_region_display_name(selected_region)

            # Create a comprehensive pricing matrix table
            matrix_data = []
            for model_id, analysis in regional_analysis.items():
                cheapest_region = analysis['cheapest_region']
                expensive_region = analysis['most_expensive_region']
                cheapest_price = analysis['prices'][cheapest_region]['total']
                expensive_price = analysis['prices'][expensive_region]['total']
                savings_pct = 0 if abs(expensive_price - cheapest_price) < 0.0001 else ((expensive_price - cheapest_price) / expensive_price) * 100

                # Find all regions with the best price
                best_price_regions = [region for region, prices in analysis['prices'].items()
                                    if abs(prices['total'] - cheapest_price) < 0.0001]

                # Format best regions display
                if len(best_price_regions) > 3:
                    best_regions_display = f"Any of {len(best_price_regions)} regions"
                else:
                    best_regions_display = ", ".join([get_region_display_name(region) for region in best_price_regions])

                # Get pricing for selected region
                if selected_region in analysis['prices']:
                    selected_prices = analysis['prices'][selected_region]
                    selected_total = f"${selected_prices['total']:.4f}"
                    selected_input = f"${selected_prices['input']:.4f}"
                    selected_output = f"${selected_prices['output']:.4f}"
                else:
                    selected_total = "N/A"
                    selected_input = "N/A"
                    selected_output = "N/A"

                matrix_data.append({
                    'Provider': analysis['provider'],
                    'Model': analysis['name'],
                    f'{selected_region_name} Input': selected_input,
                    f'{selected_region_name} Output': selected_output,
                    f'{selected_region_name} Total': selected_total,
                    'Best Price': f"${cheapest_price:.4f}",
                    'Best Regions': best_regions_display,
                    'Worst Price': f"${expensive_price:.4f}" if savings_pct > 0 else "Same Price",
                    'Max Savings': f"{savings_pct:.1f}%" if savings_pct > 0 else "0%"
                })

            # Create DataFrame and display
            matrix_df = pd.DataFrame(matrix_data)

            # Dynamic column configuration based on selected region (ordered as requested)
            column_config = {
                'Provider': st.column_config.TextColumn("Provider", width="small"),
                'Model': st.column_config.TextColumn("Model", width="medium"),
                f'{selected_region_name} Input': st.column_config.TextColumn(f"{selected_region_name} Input", width="small"),
                f'{selected_region_name} Output': st.column_config.TextColumn(f"{selected_region_name} Output", width="small"),
                f'{selected_region_name} Total': st.column_config.TextColumn(f"{selected_region_name} Total", width="small"),
                'Best Price': st.column_config.TextColumn("Best Price", width="small"),
                'Best Regions': st.column_config.TextColumn("Best Regions", width="large"),
                'Worst Price': st.column_config.TextColumn("Worst Price", width="small"),
                'Max Savings': st.column_config.TextColumn("Max Savings", width="small")
            }

            st.dataframe(
                matrix_df,
                width='stretch',
                column_config=column_config,
                hide_index=True
            )

            st.markdown("---")

            # === SIMPLE REGIONAL COMPARISON ===
            st.markdown("#### 🎯 Simple Regional Price Comparison")

            # Dynamic column layout with responsive design
            num_models = len(regional_analysis)
            columns_per_row = min(4, num_models) if num_models > 0 else 4

            # Create rows of regional analysis data
            regional_items = list(regional_analysis.items())
            regional_rows = [regional_items[i:i+columns_per_row] for i in range(0, len(regional_items), columns_per_row)]

            for row_idx, row in enumerate(regional_rows):
                # Create dynamic columns for this row
                cols = st.columns(columns_per_row)

                for col_idx, (model_id, analysis) in enumerate(row):
                    with cols[col_idx]:
                        provider_color = get_provider_color(analysis['provider'])
                        cheapest_region = analysis['cheapest_region']
                        expensive_region = analysis['most_expensive_region']
                        cheapest_price = analysis['prices'][cheapest_region]['total']
                        expensive_price = analysis['prices'][expensive_region]['total']

                        # Check if best and worst are the same
                        same_price = abs(expensive_price - cheapest_price) < 0.0001
                        savings_pct = 0 if same_price else ((expensive_price - cheapest_price) / expensive_price) * 100

                        # Get all regions with the best price for recommendation
                        best_price_regions = [region for region, prices in analysis['prices'].items()
                                            if abs(prices['total'] - cheapest_price) < 0.0001]

                        # Prepare conditional content
                        savings_badge_color = '#10b981' if savings_pct > 0 else '#6b7280'
                        savings_text = f'Save {int(savings_pct)}%' if savings_pct > 0 else 'Same Price'

                        # Worst price section (only if different from best)
                        worst_price_html = ""
                        if not same_price:
                            worst_price_html = f"""<div style="background: #fee2e2; border: 1px solid #fecaca; border-radius: 8px; padding: 8px; margin-bottom: 8px; text-align: center;">
        <div style="color: #dc2626; font-size: 0.75rem; font-weight: bold; margin-bottom: 4px;">💸 Highest Price</div>
        <div style="color: #dc2626; font-size: 1.1rem; font-weight: bold;">${expensive_price:.4f}/1K</div>
        <div style="color: #dc2626; font-size: 0.8rem;">{get_region_display_name(expensive_region)}</div>
    </div>"""

                        # Recommendation text (escape any potential HTML-breaking characters)
                        if len(best_price_regions) > 1:
                            recommendation_text = f'Deploy in any of {len(best_price_regions)} regions with best pricing'
                        else:
                            recommendation_text = f'Deploy in {get_region_display_name(cheapest_region)}'

                        savings_text_rec = f' to save ${expensive_price - cheapest_price:.4f}/1K' if not same_price else ''

                        # Create full recommendation content
                        full_recommendation = f"{recommendation_text}{savings_text_rec}"

                        # Mini regional price card - split into sections to avoid HTML parsing issues

                        # Use Streamlit container with simple styling
                        with st.container():
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {provider_color}15 0%, {provider_color}05 100%);
                                        border: 1px solid {provider_color}40;
                                        border-radius: 10px;
                                        padding: 15px;
                                        margin-bottom: 15px;">
                                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                                    <h4 style="margin: 0; color: #1f2937; font-size: 1rem; font-weight: bold;">
                                        {analysis['name']}
                                    </h4>
                                    <span style="background: {savings_badge_color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold;">
                                        {savings_text}
                                    </span>
                                </div>
                                <div style="margin-bottom: 12px;">
                                    <span style="background: {provider_color}; color: white; padding: 2px 6px; border-radius: 8px; font-size: 0.7rem;">
                                        {analysis['provider']}
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Use simple markdown for recommendation
                            st.markdown(f"**💡 Recommendation:** {full_recommendation}")

                            # Simple best price display
                            st.success(f"💚 **Best Price:** ${cheapest_price:.4f}/1K in {get_region_display_name(cheapest_region)}")

                            # Worst price if different
                            if not same_price:
                                st.error(f"💸 **Highest Price:** ${expensive_price:.4f}/1K in {get_region_display_name(expensive_region)}")

                # Fill empty columns if needed for consistent layout
                for col_idx in range(len(row), columns_per_row):
                    with cols[col_idx]:
                        st.empty()

            # === INTERACTIVE MODEL SELECTOR ===
            st.markdown("---")
            st.markdown("#### 🔍 Detailed Regional Breakdown")

            # Model selector
            model_options = {analysis['name']: model_id for model_id, analysis in regional_analysis.items()}
            selected_model_name = st.selectbox(
                "Select a model to see all regional prices:",
                options=list(model_options.keys()),
                key="regional_price_model_selector"
            )

            if selected_model_name:
                selected_model_id = model_options[selected_model_name]
                selected_analysis = regional_analysis[selected_model_id]

                st.markdown(f"**{selected_model_name}** - Price Tiers by Region")

                # Group regions by unique price points
                price_groups = {}
                for region, prices in selected_analysis['prices'].items():
                    price_key = prices['total']  # Use total cost as key
                    if price_key not in price_groups:
                        price_groups[price_key] = {
                            'regions': [],
                            'input_price': prices['input'],
                            'output_price': prices['output']
                        }
                    price_groups[price_key]['regions'].append(region)

                # Sort by price (cheapest first)
                sorted_price_groups = sorted(price_groups.items(), key=lambda x: x[0])

                # Create columns for unique prices (max 3 per row for better readability)
                prices_per_row = 3
                price_rows = [sorted_price_groups[i:i+prices_per_row] for i in range(0, len(sorted_price_groups), prices_per_row)]

                for row in price_rows:
                    cols = st.columns(len(row))
                    for i, (total_price, price_info) in enumerate(row):
                        with cols[i]:
                            # Determine styling based on price tier
                            if total_price == min(price_groups.keys()):
                                bg_color = "#dcfce7"
                                border_color = "#bbf7d0"
                                text_color = "#15803d"
                                badge = "🥇 Cheapest"
                            elif total_price == max(price_groups.keys()):
                                bg_color = "#fee2e2"
                                border_color = "#fecaca"
                                text_color = "#dc2626"
                                badge = "💸 Most Expensive"
                            else:
                                bg_color = "#fef3c7"
                                border_color = "#fde68a"
                                text_color = "#d97706"
                                badge = "💰 Mid-tier"

                            # Format regions list
                            region_names = [get_region_display_name(region) for region in price_info['regions']]
                            if len(region_names) > 2:
                                regions_display = f"{', '.join(region_names[:2])} +{len(region_names)-2} more"
                            else:
                                regions_display = ', '.join(region_names)

                            st.markdown(f"""
                            <div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 8px; padding: 12px; text-align: center; margin-bottom: 8px;">
                                <div style="color: {text_color}; font-size: 0.8rem; font-weight: bold; margin-bottom: 6px;">{badge}</div>
                                <div style="color: {text_color}; font-size: 1.2rem; font-weight: bold; margin-bottom: 8px;">${total_price:.4f}/1K</div>
                                <div style="color: {text_color}; font-size: 0.75rem; margin-bottom: 6px;">
                                    In: ${price_info['input_price']:.4f} | Out: ${price_info['output_price']:.4f}
                                </div>
                                <div style="background: rgba(255,255,255,0.3); border-radius: 6px; padding: 6px; margin-top: 8px;">
                                    <div style="color: {text_color}; font-size: 0.8rem; font-weight: 500; margin-bottom: 2px;">Available in:</div>
                                    <div style="color: {text_color}; font-size: 0.75rem; line-height: 1.3;">{regions_display}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Show full regions list if there are many
                            if len(price_info['regions']) > 2:
                                with st.expander(f"All {len(price_info['regions'])} regions with ${total_price:.4f}/1K pricing"):
                                    for region in price_info['regions']:
                                        st.write(f"• {get_region_display_name(region)} ({region})")

        else:
            st.info("💡 Regional price analysis requires models with pricing data in multiple regions.")

        st.markdown("---")

        # === SECTION 3: MODELS BY TYPE ===
        st.markdown("### 🎯 Models by Type")

        # Categorize models by type
        model_type_categories = categorize_models_by_type(enhanced_comparison_data)

        col1, col2 = st.columns([1, 2])

        with col1:
            # Type selector (like region selector in availability tab)
            type_options = {}
            for type_name, models in model_type_categories.items():
                if models:  # Only show categories that have models
                    display_name = type_name.replace('_', ' ').title()
                    type_options[display_name] = type_name

            if type_options:
                selected_type = st.selectbox(
                    "Select model type:",
                    options=list(type_options.keys()),
                    key="pricing_type_selector"
                )
            else:
                selected_type = None

        with col2:
            if selected_type and selected_type in type_options:
                type_key = type_options[selected_type]
                models_of_type = model_type_categories[type_key]

                st.markdown(f"**{selected_type} Models** ({len(models_of_type)} total)")

                # Group models by provider for better organization
                provider_groups = {}
                for model_info in models_of_type:
                    provider = model_info.get('provider', 'Unknown')
                    if provider not in provider_groups:
                        provider_groups[provider] = []
                    provider_groups[provider].append(model_info)

                # Display models grouped by provider
                for provider, provider_models in sorted(provider_groups.items()):
                    provider_color = get_provider_color(provider)

                    st.markdown(f"""
                    <div style="margin: 12px 0 8px 0;">
                        <span style="background: {provider_color}; color: white; padding: 6px 12px; border-radius: 20px; font-size: 0.9rem; font-weight: bold; margin-right: 8px;">
                            {provider}
                        </span>
                        <span style="color: #6b7280; font-size: 0.85rem;">
                            {len(provider_models)} model{"s" if len(provider_models) > 1 else ""}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display models for this provider
                    for model_info in provider_models[:5]:  # Show up to 5 models per provider
                        model_data = model_info['data']
                        input_price, output_price, region = extract_comprehensive_pricing_info(model_data)
                        total_cost = input_price + output_price

                        # Get the model from enhanced_comparison_data to show region
                        enhanced_model = None
                        for m in enhanced_comparison_data:
                            if m['model_id'] == model_info['model_id']:
                                enhanced_model = m
                                break

                        region_display = enhanced_model['selected_region'] if enhanced_model is not None else region

                        st.markdown(f"""
                        <div style="margin: 4px 0; padding: 6px; background: #f8f9fa; border-radius: 6px; border-left: 3px solid {provider_color};">
                            <strong>{model_info['name']}</strong> ({region_display}) - ${total_cost:.4f}/1K tokens
                        </div>
                        """, unsafe_allow_html=True)

                    if len(provider_models) > 5:
                        st.markdown(f"*... and {len(provider_models)-5} more models*")

    with tab4:
        # === SECTION 1: KEY TECHNICAL METRICS ===
        st.markdown("<br>", unsafe_allow_html=True)

        # Calculate technical metrics
        context_windows = [get_converse_context_window(model) for model in enhanced_comparison_data if get_converse_context_window(model) > 0]
        modality_counts = []
        streaming_support = []
        batch_support = []
        cris_support = []

        for model in enhanced_comparison_data:
            # Count modalities
            input_mods = set(model.get('input_modalities', ['TEXT']))
            output_mods = set(model.get('output_modalities', ['TEXT']))
            all_mods = input_mods.union(output_mods)
            modality_counts.append(len(all_mods))

            # Use same approach as model cards - access directly from model object
            # Note: For comparison models, data is directly in the pandas Series, not nested under 'model_data'

            # Check streaming support - same as model cards
            streaming_supported = model.get('streaming_supported', False)
            streaming_support.append(streaming_supported)

            # Check batch support - only explicit BATCH inference type
            inference_types = model.get('inference_types_supported', [])
            batch_supported = 'BATCH' in inference_types
            batch_support.append(batch_supported)

            # Check CRIS support - same as model cards
            cris_supported = get_cross_region_inference_support(model)
            cris_support.append(cris_supported)

        # Summary metrics (matching pricing tab style)
        col0, col1, col2, col3 = st.columns([.5, 1, 1, 1])

        with col1:
            avg_context = sum(context_windows) / len(context_windows) if context_windows else 0
            if avg_context >= 1000000:
                context_display = f"{avg_context/1000000:.1f}M tokens"
            elif avg_context >= 1000:
                context_display = f"{avg_context/1000:.0f}K tokens"
            else:
                context_display = f"{avg_context:.0f} tokens"
            st.metric("🧠 Avg Context Window", context_display)

        with col2:
            multimodal_count = len([c for c in modality_counts if c > 1])
            text_only_count = len(modality_counts) - multimodal_count
            st.metric("🎨 Multimodal Models", f"{multimodal_count}/{len(modality_counts)}")

        with col3:
            streaming_count = sum(streaming_support)
            batch_count = sum(batch_support)
            st.metric("⚡ Streaming/Batch", f"{streaming_count}/{batch_count}")

        st.markdown("---")

        # === SECTION 2: TECHNICAL SPECIFICATIONS COMPARISON TABLE ===
        st.markdown("### ⚙️ Technical Specifications Comparison")

        # Helper function to get modality icons with text
        def get_modality_display(modality):
            display_map = {
                'TEXT': '📝 Text',
                'IMAGE': '🖼️ Image',
                'AUDIO': '🎵 Audio',
                'VIDEO': '🎥 Video'
            }
            return display_map.get(modality.upper(), f'❓ {modality.title()}')

        # Helper function to format context window
        def format_context_window(tokens):
            if tokens == 0:
                return "N/A"
            elif tokens >= 1000000:
                return f"{tokens/1000000:.1f}M"
            elif tokens >= 1000:
                return f"{tokens/1000:.0f}K"
            else:
                return f"{tokens:,}"

        # Create technical specifications table
        tech_data = []
        for model in enhanced_comparison_data:
            # Note: For comparison models, data is directly in the pandas Series, not nested under 'model_data'

            # Get modalities with icon + text display
            input_modalities = model.get('input_modalities', ['TEXT'])
            output_modalities = model.get('output_modalities', ['TEXT'])
            input_display = ', '.join([get_modality_display(mod) for mod in input_modalities])
            output_display = ', '.join([get_modality_display(mod) for mod in output_modalities])

            # Get technical features - same approach as model cards
            streaming = '✅' if model.get('streaming_supported', False) else '❌'

            # Check batch support using correct batch_inference_supported field
            batch_info = model.get('batch_inference_supported', {})
            batch = '✅' if batch_info.get('supported', False) else '❌'

            # Get cross-region support - same as model cards
            cris = '✅' if get_cross_region_inference_support(model) else '❌'

            # Context window
            context_window = get_converse_context_window(model)
            context_formatted = format_context_window(context_window)

            # Consumption options
            consumption_opts = get_consumption_options(model)
            consumption_display = ', '.join([opt.replace('_', ' ').title() for opt in consumption_opts])

            tech_data.append({
                'Model': model['name'],
                'Provider': model.get('provider', 'Unknown'),
                'Input Types': input_display,
                'Output Types': output_display,
                'Context Window': context_formatted,
                'Streaming': streaming,
                'Batch Inference': batch,
                'Cross-Region': cris,
                'Deployment Options': consumption_display
            })

        tech_df = pd.DataFrame(tech_data)
        st.dataframe(tech_df, width="stretch", hide_index=True)

        st.markdown("---")

        # === SECTION 3: BATCH INFERENCE SUPPORT ===
        st.markdown("### 📦 Batch Inference Support")

        batch_models = []
        non_batch_models = []

        for model in enhanced_comparison_data:
            # Note: For comparison models, data is directly in the pandas Series, not nested under 'model_data'
            # Check batch support using the correct batch_inference_supported field
            batch_info = model.get('batch_inference_supported', {})
            if batch_info.get('supported', False):
                batch_models.append(model)
            else:
                non_batch_models.append(model)

        # Enhanced metrics display
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📦 Batch Enabled", f"{len(batch_models)}/{len(enhanced_comparison_data)}")

        with col2:
            total_batch_regions = set()
            for model in batch_models:
                batch_info = model.get('batch_inference_supported', {})
                regions = batch_info.get('supported_regions', [])
                total_batch_regions.update(regions)
            st.metric("🗺️ Regions with Batch", len(total_batch_regions))

        with col3:
            avg_coverage = 0
            if batch_models:
                total_regions = sum(len(model.get('batch_inference_supported', {}).get('supported_regions', [])) for model in batch_models)
                avg_coverage = total_regions / len(batch_models) if batch_models else 0
            st.metric("📊 Avg Regional Coverage", f"{avg_coverage:.1f}")

        if batch_models:
            st.markdown("#### 🗺️ Regional Batch Coverage")

            # Group models by region
            region_groups = {}
            for model in batch_models:
                batch_info = model.get('batch_inference_supported', {})
                regions = batch_info.get('supported_regions', [])
                for region in regions:
                    if region not in region_groups:
                        region_groups[region] = []
                    region_groups[region].append(model)

            col1, col2 = st.columns([1, 2])

            with col1:
                # Region filter similar to consumption/modality sections
                region_options = {"All Regions": "all"}
                for region in sorted(region_groups.keys()):
                    count = len(region_groups[region])
                    region_name = get_region_display_name(region)
                    region_options[f"{region_name} ({count})"] = region

                selected_region = st.selectbox(
                    "Filter by batch-supported region:",
                    options=list(region_options.keys()),
                    key="batch_region_selector"
                )

            with col2:
                if selected_region and region_options[selected_region] != "all":
                    region_code = region_options[selected_region]
                    filtered_models = region_groups.get(region_code, [])
                else:
                    filtered_models = batch_models

                st.markdown(f"**{selected_region.split(' (')[0]}** ({len(filtered_models)} models)")

                # Group by provider like other sections - show all when "All Regions" selected
                provider_groups = {}
                batch_limit = len(batch_models) if selected_region == "All Regions" else 8
                for model in filtered_models[:batch_limit]:
                    provider = model.get('provider', 'Unknown')
                    if provider not in provider_groups:
                        provider_groups[provider] = []
                    provider_groups[provider].append(model)

                for provider, provider_models in sorted(provider_groups.items()):
                    provider_color = get_provider_color(provider)

                    st.markdown(f"""
                    <div style="margin: 12px 0 8px 0;">
                        <span style="background: {provider_color}; color: white; padding: 6px 12px; border-radius: 20px; font-size: 0.9rem; font-weight: bold;">
                            {provider}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    for model in provider_models:
                        batch_info = model.get('batch_inference_supported', {})
                        regions = batch_info.get('supported_regions', [])
                        region_count = len(regions)

                        st.markdown(f"""
                        <div style="margin: 4px 0; padding: 8px; background: #f8f9fa; border-radius: 6px; border-left: 3px solid {provider_color};">
                            <strong>{model['name']}</strong><br>
                            <small style="color: #be185d;">📦 Batch supported in {region_count} regions</small>
                        </div>
                        """, unsafe_allow_html=True)

                if len(filtered_models) > batch_limit:
                    st.markdown(f"*... and {len(filtered_models)-batch_limit} more models*")

        else:
            st.info("🔍 No models currently support batch inference in the selected comparison.")

        # === SECTION 4: DEPLOYMENT OPTIONS & MODALITY SUPPORT ===
        st.markdown("---")

        # Prepare data for both sections
        # Group models by consumption options
        consumption_groups = {}
        for model in enhanced_comparison_data:
            consumption_opts = get_consumption_options(model)
            consumption_key = ', '.join(sorted(consumption_opts))
            if consumption_key not in consumption_groups:
                consumption_groups[consumption_key] = []
            consumption_groups[consumption_key].append(model)

        # Create dynamic modality options based on selected models
        modality_combinations = {}
        for model in enhanced_comparison_data:
            input_mods = set(model.get('input_modalities', ['TEXT']))
            output_mods = set(model.get('output_modalities', ['TEXT']))
            all_mods = input_mods.union(output_mods)

            # Create a readable key for this combination
            mod_key = ', '.join(sorted(all_mods))
            if mod_key not in modality_combinations:
                modality_combinations[mod_key] = []
            modality_combinations[mod_key].append(model)

        # 2-column layout
        deploy_col, modality_col = st.columns(2)

        # LEFT COLUMN: DEPLOYMENT & CONSUMPTION OPTIONS
        with deploy_col:
            st.markdown("### 🏗️ Deployment & Consumption Options")

            # Selector first
            consumption_options = {"All Options": "all"}
            for key in sorted(consumption_groups.keys()):
                if key:  # Only non-empty keys
                    display_name = key.replace('_', ' ').title()
                    consumption_options[display_name] = key

            selected_consumption = st.selectbox(
                "Filter by deployment options:",
                options=list(consumption_options.keys()),
                key="tech_consumption_selector"
            )

            # Models below selector
            if selected_consumption and consumption_options[selected_consumption] != "all":
                consumption_key = consumption_options[selected_consumption]
                filtered_models = consumption_groups.get(consumption_key, [])
            else:
                filtered_models = enhanced_comparison_data

            st.markdown(f"**{selected_consumption}** ({len(filtered_models)} models)")

            # Group by provider - show all models when "All Options" selected
            provider_groups = {}
            model_limit = len(enhanced_comparison_data) if selected_consumption == "All Options" else 6
            for model in filtered_models[:model_limit]:
                provider = model.get('provider', 'Unknown')
                if provider not in provider_groups:
                    provider_groups[provider] = []
                provider_groups[provider].append(model)

            for provider, provider_models in sorted(provider_groups.items()):
                provider_color = get_provider_color(provider)

                st.markdown(f"""
                <div style="margin: 8px 0 6px 0;">
                    <span style="background: {provider_color}; color: white; padding: 4px 8px; border-radius: 16px; font-size: 0.8rem; font-weight: bold;">
                        {provider}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # Show all models per provider when "All Options" selected, otherwise limit to 2
                per_provider_limit = len(provider_models) if selected_consumption == "All Options" else 2
                for model in provider_models[:per_provider_limit]:
                    consumption_opts = get_consumption_options(model)
                    consumption_display = ', '.join([opt.replace('_', ' ').title() for opt in consumption_opts])

                    st.markdown(f"""
                    <div style="margin: 3px 0; padding: 6px; background: #fefce8; border-radius: 6px; border-left: 3px solid {provider_color};">
                        <strong style="font-size: 0.9rem;">{model['name']}</strong><br>
                        <small style="color: #a16207; font-size: 0.8rem;">Options: {consumption_display}</small>
                    </div>
                    """, unsafe_allow_html=True)

            if len(filtered_models) > model_limit:
                st.markdown(f"*... and {len(filtered_models)-model_limit} more models*")

        # RIGHT COLUMN: MODALITY SUPPORT
        with modality_col:
            st.markdown("### 🎨 Modality Support Breakdown")

            # Selector first
            modality_options = {"All Models": "all"}
            for key in sorted(modality_combinations.keys()):
                count = len(modality_combinations[key])
                display_key = ', '.join([get_modality_display(mod) for mod in sorted(key.split(', '))])
                modality_options[f"{display_key} ({count})"] = key

            selected_modality = st.selectbox(
                "Filter by modality support:",
                options=list(modality_options.keys()),
                key="tech_modality_selector"
            )

            # Models below selector
            if selected_modality and modality_options[selected_modality] != "all":
                modality_key = modality_options[selected_modality]
                filtered_models_mod = modality_combinations.get(modality_key, [])
            else:
                filtered_models_mod = enhanced_comparison_data

            st.markdown(f"**{selected_modality.split(' (')[0]}** ({len(filtered_models_mod)} models)")

            # Group by provider - show all models when "All Models" selected
            provider_groups_mod = {}
            model_limit_mod = len(enhanced_comparison_data) if selected_modality == "All Models" else 6
            for model in filtered_models_mod[:model_limit_mod]:
                provider = model.get('provider', 'Unknown')
                if provider not in provider_groups_mod:
                    provider_groups_mod[provider] = []
                provider_groups_mod[provider].append(model)

            for provider, provider_models in sorted(provider_groups_mod.items()):
                provider_color = get_provider_color(provider)

                st.markdown(f"""
                <div style="margin: 8px 0 6px 0;">
                    <span style="background: {provider_color}; color: white; padding: 4px 8px; border-radius: 16px; font-size: 0.8rem; font-weight: bold;">
                        {provider}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # Show all models per provider when "All Models" selected, otherwise limit to 2
                per_provider_limit_mod = len(provider_models) if selected_modality == "All Models" else 2
                for model in provider_models[:per_provider_limit_mod]:
                    input_mods = model.get('input_modalities', ['TEXT'])
                    output_mods = model.get('output_modalities', ['TEXT'])
                    input_display = ', '.join([get_modality_display(mod) for mod in input_mods])
                    output_display = ', '.join([get_modality_display(mod) for mod in output_mods])
                    context = format_context_window(get_converse_context_window(model))

                    st.markdown(f"""
                    <div style="margin: 3px 0; padding: 6px; background: #f8f9fa; border-radius: 6px; border-left: 3px solid {provider_color};">
                        <strong style="font-size: 0.9rem;">{model['name']}</strong><br>
                        <small style="color: #6b7280; font-size: 0.8rem;">In: {input_display} | Out: {output_display}<br>Context: {context}</small>
                    </div>
                    """, unsafe_allow_html=True)

            if len(filtered_models_mod) > model_limit_mod:
                st.markdown(f"*... and {len(filtered_models_mod)-model_limit_mod} more models*")

        # === SECTION 6: CROSS-REGION INFERENCE (CRIS) SUPPORT ===
        st.markdown("---")
        st.markdown("### 🌐 Cross-Region Inference (CRIS) Support")

        cris_models = []
        for model in enhanced_comparison_data:
            if get_cross_region_inference_support(model):
                cris_models.append(model)

        if not cris_models:
            st.info("🔍 No models currently support Cross-Region Inference.")
            return

        # Enhanced metrics display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌐 CRIS Enabled", f"{len(cris_models)}/{len(enhanced_comparison_data)}")
        with col2:
            total_endpoints = sum(model.get('cross_region_inference', {}).get('profiles_count', 0) for model in cris_models)
            st.metric("🔗 Total Endpoints", total_endpoints)
        with col3:
            all_regions = set()
            for model in cris_models:
                cris_info = model.get('cross_region_inference', {})
                source_regions = cris_info.get('source_regions', [])
                all_regions.update(source_regions)
            st.metric("🗺️ Source Regions", len(all_regions))

        # Model selector and CRIS visualization
        selected_model = st.selectbox("Select a model to visualize CRIS endpoints:",
                                      [model['name'] for model in cris_models], key="cris_model_selector")
        selected_model_data = next(model for model in cris_models if model['name'] == selected_model)
        profiles = selected_model_data.get('cross_region_inference', {}).get('profiles', [])

        if profiles:
            # Color legend with counts - will be updated after processing
            st.markdown("#### 🎨 Endpoint Legend & Summary")
            legend_placeholder = st.empty()

            # Create region coordinates and process profiles
            region_coordinates = {
                'us-east-1': {'lat': 39.0458, 'lon': -77.4413, 'name': 'N. Virginia'},
                'us-west-2': {'lat': 45.5152, 'lon': -122.6784, 'name': 'Oregon'},
                'eu-west-1': {'lat': 53.3498, 'lon': -6.2603, 'name': 'Ireland'},
                'eu-central-1': {'lat': 50.1109, 'lon': 8.6821, 'name': 'Frankfurt'},
                'ap-northeast-1': {'lat': 35.6762, 'lon': 139.6503, 'name': 'Tokyo'},
                'ap-southeast-2': {'lat': -33.8688, 'lon': 151.2093, 'name': 'Sydney'}
            }

            # Group profiles by region to show all endpoint types per region
            region_endpoints = {}
            for profile in profiles:
                profile_id = profile.get('profile_id', '')
                source_region = profile.get('source_region', '')

                if source_region not in region_endpoints:
                    region_endpoints[source_region] = {
                        'global': 0,
                        'eu_regional': 0,
                        'us_regional': 0,
                        'other_regional': 0,
                        'profiles': []
                    }

                # Count endpoint types per region
                if profile_id.startswith('global.'):
                    region_endpoints[source_region]['global'] += 1
                elif profile_id.startswith('eu.'):
                    region_endpoints[source_region]['eu_regional'] += 1
                elif profile_id.startswith('us.'):
                    region_endpoints[source_region]['us_regional'] += 1
                else:
                    region_endpoints[source_region]['other_regional'] += 1

                region_endpoints[source_region]['profiles'].append(profile)

            # Create multiple points per region based on endpoint types
            endpoint_points = []
            for source_region, region_data in region_endpoints.items():
                if source_region in region_coordinates:
                    coord = region_coordinates[source_region]
                    base_lat = coord['lat']
                    base_lon = coord['lon']

                    # Create offset positions for different endpoint types in same region
                    offset = 0.5  # degrees offset for visibility
                    positions = [
                        (base_lat, base_lon),  # center
                        (base_lat + offset, base_lon + offset),  # top-right
                        (base_lat - offset, base_lon + offset),  # bottom-right
                        (base_lat + offset, base_lon - offset),  # top-left
                    ]

                    pos_idx = 0

                    # Add global endpoints
                    if region_data['global'] > 0:
                        lat, lon = positions[pos_idx % len(positions)]
                        endpoint_points.append({
                            'lat': lat,
                            'lon': lon,
                            'region': source_region,
                            'name': coord['name'],
                            'type': 'Global',
                            'color': [255, 100, 100, 200],  # Red
                            'size': min(region_data['global'] * 20 + 80, 200),
                            'count': region_data['global']
                        })
                        pos_idx += 1

                    # Add EU regional endpoints
                    if region_data['eu_regional'] > 0:
                        lat, lon = positions[pos_idx % len(positions)]
                        endpoint_points.append({
                            'lat': lat,
                            'lon': lon,
                            'region': source_region,
                            'name': coord['name'],
                            'type': 'EU Regional',
                            'color': [100, 100, 255, 200],  # Blue
                            'size': min(region_data['eu_regional'] * 20 + 80, 200),
                            'count': region_data['eu_regional']
                        })
                        pos_idx += 1

                    # Add US regional endpoints
                    if region_data['us_regional'] > 0:
                        lat, lon = positions[pos_idx % len(positions)]
                        endpoint_points.append({
                            'lat': lat,
                            'lon': lon,
                            'region': source_region,
                            'name': coord['name'],
                            'type': 'US Regional',
                            'color': [100, 255, 100, 200],  # Green
                            'size': min(region_data['us_regional'] * 20 + 80, 200),
                            'count': region_data['us_regional']
                        })
                        pos_idx += 1

                    # Add other regional endpoints
                    if region_data['other_regional'] > 0:
                        lat, lon = positions[pos_idx % len(positions)]
                        endpoint_points.append({
                            'lat': lat,
                            'lon': lon,
                            'region': source_region,
                            'name': coord['name'],
                            'type': 'Other Regional',
                            'color': [255, 255, 100, 200],  # Yellow
                            'size': min(region_data['other_regional'] * 20 + 80, 200),
                            'count': region_data['other_regional']
                        })

            # Create and display pydeck map
            if endpoint_points:
                points_df = pd.DataFrame(endpoint_points)
                st.pydeck_chart(pdk.Deck(
                    map_style=None,
                    initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1.5, pitch=0),
                    layers=[pdk.Layer('ScatterplotLayer', data=points_df, get_position='[lon, lat]',
                                      get_color='color', get_radius='size', radius_scale=1000,
                                      radius_min_pixels=8, radius_max_pixels=30, pickable=True)],
                    tooltip={'html': '<b>Region:</b> {name} ({region})<br><b>Type:</b> {type}<br><b>Count:</b> {count} endpoints',
                             'style': {'backgroundColor': 'steelblue', 'color': 'white'}}
                ))

                # Calculate totals and update legend with counts
                global_total = sum(row['count'] for _, row in points_df.iterrows() if row['type'] == 'Global')
                eu_total = sum(row['count'] for _, row in points_df.iterrows() if row['type'] == 'EU Regional')
                us_total = sum(row['count'] for _, row in points_df.iterrows() if row['type'] == 'US Regional')
                other_total = sum(row['count'] for _, row in points_df.iterrows() if row['type'] == 'Other Regional')

                # Update legend with actual counts
                with legend_placeholder.container():
                    legend_col1, legend_col2, legend_col3, legend_col4 = st.columns(4)

                    with legend_col1:
                        if global_total > 0:
                            st.metric("🔴 Global Endpoints", f"{global_total}")
                            st.caption("Cross all regions worldwide")
                        else:
                            st.markdown("🔴 **Global Endpoints**")
                            st.caption("Cross all regions worldwide")

                    with legend_col2:
                        if eu_total > 0:
                            st.metric("🔵 EU Regional", f"{eu_total}")
                            st.caption("European regions only")
                        else:
                            st.markdown("🔵 **EU Regional Endpoints**")
                            st.caption("European regions only")

                    with legend_col3:
                        if us_total > 0:
                            st.metric("🟢 US Regional", f"{us_total}")
                            st.caption("US regions only")
                        else:
                            st.markdown("🟢 **US Regional Endpoints**")
                            st.caption("US regions only")

                    with legend_col4:
                        if other_total > 0:
                            st.metric("🟡 Other Regional", f"{other_total}")
                            st.caption("JP, Asia-Pacific, etc.")
                        else:
                            st.markdown("🟡 **Other Regional Endpoints**")
                            st.caption("JP, Asia-Pacific, etc.")

                total_endpoints = global_total + eu_total + us_total + other_total
                st.info(f"📊 **Total CRIS Endpoints for {selected_model}: {total_endpoints}**")
        else:
            st.info("No CRIS profile data available for selected model.")
