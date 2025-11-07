"""
Model Details Modal UI Component
Displays comprehensive pricing and quota information for a model
"""
import streamlit as st
import logging
from typing import Dict, Any, List
from ui.converse_data_helpers import (
    get_converse_context_window,
    get_converse_max_output_tokens,
    get_converse_size_category,
    get_converse_function_calling,
    get_converse_recommendation
)


# Legacy function removed - no longer needed with unified pricing structure


def display_model_details_modal(model: Dict[str, Any]):
    """Display detailed model information in a modal-like interface"""
    
    # Create tabs for different information sections
    tab1, tab2, tab3 = st.tabs(["⚙️ Technical Specifications", "🔢 Service Quotas", "📊 Pricing Details"])
    
    with tab1:
        display_technical_specifications(model)
    
    with tab2:
        display_quota_details(model)
    
    with tab3:
        display_pricing_details(model)


def display_pricing_details(model: Dict[str, Any]):
    """Display comprehensive pricing information with geographical grouping"""
    st.subheader("💰 Pricing Information")

    try:
        # Get pricing reference from new structure
        model_pricing_ref = model.get('model_pricing', {})

        # Ensure pricing data is in the expected format
        if not isinstance(model_pricing_ref, dict):
            model_pricing_ref = {}

        # Extract pricing reference info
        pricing_info = model_pricing_ref  # Remove double-nesting - model_pricing_ref already contains the data
        is_pricing_available = pricing_info.get('is_pricing_available', False)
        pricing_reference_id = pricing_info.get('pricing_reference_id')

        # Check if we have detailed pricing data (same approach as model cards)
        has_pricing_data = False
        detailed_pricing = {}

        if is_pricing_available:
            # Check for detailed_pricing field that's populated by the repository
            detailed_pricing = model.get('detailed_pricing', {})
            if detailed_pricing and isinstance(detailed_pricing, dict):
                has_pricing_data = True

        # Show no pricing if either no pricing available OR no detailed pricing loaded
        if not is_pricing_available or not has_pricing_data:
            # Show why pricing is not available - get from pricing metadata if available
            pricing_metadata = model.get('pricing_metadata', {})
            integration_reason = pricing_metadata.get('reason', 'Unknown reason')
            integration_source = pricing_metadata.get('integration_source', 'Unknown')
            last_updated = pricing_metadata.get('integration_timestamp', 'Unknown')

            st.warning("💸 Pricing information is currently not available for this model.")

            # Create an informative display about the status
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
            <div style="background-color: #fef3c7; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #f59e0b;">
                <h5 style="margin: 0 0 8px 0; color: #92400e; font-size: 14px;">📊 Data Status</h5>
                <p style="margin: 0; color: #b45309; font-size: 12px;">
                    <strong>Source:</strong> {integration_source}<br>
                    <strong>Status:</strong> {integration_reason.replace('_', ' ').title()}<br>
                    <strong>Last Check:</strong> {last_updated}
                </p>
            </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style="background-color: #dbeafe; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #3b82f6;">
                    <h5 style="margin: 0 0 8px 0; color: #1e40af; font-size: 14px;">💡 What this means</h5>
                    <p style="margin: 0; color: #1e3a8a; font-size: 12px;">
                        The pricing collector hasn't successfully gathered pricing data for this model yet.
                        This could mean the model is new, has custom pricing, or requires special access.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.info("""
            🔄 **To get pricing information:**
            - Check the [AWS Bedrock Pricing page](https://aws.amazon.com/bedrock/pricing/) directly
            - Run the pricing collector manually if available
            - Contact your AWS account team for enterprise pricing
            - Review the model's AWS documentation for standard rates
            """)
            return

        # Display pricing using unified pricing collector format
        if detailed_pricing and 'regions' in detailed_pricing:
            display_pricing_collector_format(detailed_pricing)
        else:
            # Pricing data format not recognized
            st.warning("Pricing data is available but structure is unexpected.")
            st.info("💡 The model indicates pricing is available, but the data format is not recognized. Please contact support for assistance.")

    except Exception as e:
        st.error(f"❌ Error loading pricing information: {str(e)}")
        st.info("💡 Please try refreshing the page or contact support if the issue persists.")


# Legacy function removed - replaced by unified pricing collector format display


def categorize_pricing_by_type(pricing_data: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Categorize pricing data by pricing type and then by region, similar to quota structure"""
    categories = {
        'on_demand': {},
        'batch': {},
        'provisioned_throughput': {},
        'cross_region': {},
        'model_customization': {},
        'general': {}
    }

    try:
        for region, region_data in pricing_data.items():
            if not isinstance(region_data, dict):
                continue

            for pricing_type, type_data in region_data.items():
                if not isinstance(type_data, dict):
                    continue

                # Determine category based on pricing type
                category = 'general'  # default category
                pricing_type_lower = pricing_type.lower()

                if 'on_demand' in pricing_type_lower or pricing_type_lower == 'on_demand':
                    category = 'on_demand'
                elif 'batch' in pricing_type_lower:
                    category = 'batch'
                elif 'provisioned' in pricing_type_lower:
                    category = 'provisioned_throughput'
                elif 'cross_region' in pricing_type_lower or 'cross-region' in pricing_type_lower:
                    category = 'cross_region'
                elif 'customization' in pricing_type_lower or 'fine' in pricing_type_lower:
                    category = 'model_customization'

                # Extract pricing entries from the type data
                pricing_entries = []
                if 'pricing_dimensions' in type_data:
                    dimensions = type_data['pricing_dimensions']
                    for dimension_name, dimension_data in dimensions.items():
                        if 'pricing_entries' in dimension_data:
                            entries = dimension_data['pricing_entries']
                            for entry in entries:
                                # Add dimension name to the entry for context
                                enhanced_entry = entry.copy()
                                enhanced_entry['dimension_name'] = dimension_name
                                pricing_entries.append(enhanced_entry)

                # Add entries to the appropriate category and region
                if pricing_entries:
                    if region not in categories[category]:
                        categories[category][region] = []
                    categories[category][region].extend(pricing_entries)

    except Exception as e:
        # If there's an error processing pricing data, return empty categories
        logging.debug(f"Error processing pricing data for categorization: {e}. Returning empty categories.")

    return categories


def group_pricing_by_geography_for_type(type_regions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Group pricing regions by geographical areas for a specific pricing type"""
    geo_groups = {
        'North America': {},
        'Europe': {},
        'Asia Pacific': {},
        'South America': {},
        'Other Regions': {}
    }

    for region, pricing_items in type_regions.items():
        geo_region = map_region_to_geography(region)
        geo_groups[geo_region][region] = pricing_items

    # Remove empty groups
    return {k: v for k, v in geo_groups.items() if v}


def group_pricing_by_dimension(pricing_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group pricing items by their dimension names within a region"""
    dimension_groups = {}

    for item in pricing_items:
        # Get the dimension name from the item
        dimension_name = item.get('dimension_name', 'Standard')

        # Initialize the group if it doesn't exist
        if dimension_name not in dimension_groups:
            dimension_groups[dimension_name] = []

        # Add the item to the appropriate dimension group
        dimension_groups[dimension_name].append(item)

    # Sort dimensions for consistent display (On-Demand first, then alphabetically)
    def sort_dimensions(dim_name):
        if 'on-demand' in dim_name.lower() and 'long context' not in dim_name.lower():
            return (0, dim_name)  # On-Demand first
        elif 'on-demand' in dim_name.lower() and 'long context' in dim_name.lower():
            return (1, dim_name)  # On-Demand Long Context second
        elif 'batch' in dim_name.lower():
            return (2, dim_name)  # Batch third
        else:
            return (3, dim_name)  # Everything else alphabetically

    sorted_dimension_names = sorted(dimension_groups.keys(), key=sort_dimensions)
    return {dim_name: dimension_groups[dim_name] for dim_name in sorted_dimension_names}


def _should_skip_dimension_grouping(dimension_groups: Dict[str, List[Dict[str, Any]]], pricing_type: str) -> bool:
    """Determine if dimension sub-grouping should be skipped based on whether dimensions add meaningful differentiation"""

    # Always skip if there's only one dimension
    if len(dimension_groups) == 1:
        return True

    # Check if dimensions have meaningful variations beyond just the base pricing type
    pricing_type_lower = pricing_type.lower().replace('_', '-')
    dimension_names = list(dimension_groups.keys())

    # Look for meaningful differentiators in dimension names
    differentiators = set()
    for dim_name in dimension_names:
        dim_lower = dim_name.lower()

        # Extract parts that are not just the base pricing type
        if pricing_type_lower in dim_lower:
            # Remove the base pricing type and common separators to see what's left
            remaining = dim_lower.replace(pricing_type_lower, '').replace('-', ' ').replace('_', ' ').strip()
            if remaining:
                # Split on spaces and collect non-empty meaningful parts
                parts = [part.strip() for part in remaining.split() if part.strip()]
                for part in parts:
                    # Skip very common/generic terms
                    if part not in ['pricing', 'inference', 'model']:
                        differentiators.add(part)
        else:
            # If the dimension name doesn't contain the pricing type, it's likely meaningful
            differentiators.add(dim_lower.strip())

    # If we found meaningful differentiators (like 'global', 'long', 'context'), show sub-groups
    # Examples of meaningful differentiators:
    # - "global" (for On-Demand Global vs On-Demand)
    # - "long context" (for On-Demand Long Context)
    # - "commitment" variations for provisioned throughput
    meaningful_differentiators = {
        'global', 'long', 'context', 'commitment', 'regional', 'cross', 'region',
        'standard', 'premium', 'enhanced', 'lite', 'pro', 'micro', 'small', 'medium', 'large'
    }

    has_meaningful_differentiation = bool(differentiators.intersection(meaningful_differentiators))

    # Show sub-groups if:
    # 1. We have multiple dimensions AND
    # 2. The dimensions have meaningful differentiators beyond just the base pricing type
    return not has_meaningful_differentiation


def display_individual_pricing_item(pricing_info: Dict[str, Any], color: str, is_even: bool):
    """Display an individual pricing item with clean, minimal information"""
    try:
        # Extract pricing information
        description = pricing_info.get('description', 'Unknown Pricing')
        price_per_thousand = pricing_info.get('price_per_thousand', 0)
        unit = pricing_info.get('unit', '1K tokens')

        # Format the price value
        try:
            if isinstance(price_per_thousand, (int, float)) and price_per_thousand is not None:
                if price_per_thousand < 0.001:
                    formatted_price = f"${price_per_thousand:.6f}"
                elif price_per_thousand < 0.01:
                    formatted_price = f"${price_per_thousand:.4f}"
                else:
                    formatted_price = f"${price_per_thousand:.3f}"
            else:
                formatted_price = str(price_per_thousand) if price_per_thousand is not None else "$0.000"
        except (TypeError, ValueError):
            formatted_price = "N/A"

        # Background color alternation for readability
        bg_color = "#f8fafc" if is_even else "#ffffff"

        # Format unit display
        unit_display = unit if unit and unit != 'None' else '1K tokens'

        # Create the pricing card with clean, minimal information
        st.markdown(f"""
        <div style="background-color: {bg_color};
                    padding: 12px;
                    border-radius: 8px;
                    margin-bottom: 8px;
                    border: 1px solid #e2e8f0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="flex: 1;">
                    <h5 style="margin: 0; color: #1f2937; font-size: 14px; line-height: 1.4;">
                        {description}
                    </h5>
                </div>
                <div style="text-align: right; margin-left: 15px;">
                    <div style="font-size: 18px; font-weight: bold; color: {color};">
                        {formatted_price}
                    </div>
                    <div style="font-size: 11px; color: #9ca3af; margin-top: 2px;">
                        per {unit_display}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        # Fallback display if there's any error
        st.error(f"❌ Error displaying pricing item: {str(e)}")


# Legacy functions removed - no longer needed with unified pricing structure


def display_pricing_collector_format(pricing_data: Dict[str, Any]):
    """Display pricing from AWS Pricing Collector format with full detailed information"""
    regions_data = pricing_data.get('regions', {})

    if not regions_data:
        st.warning("No regional pricing data available.")
        return

    # Process pricing data to extract statistics and organize by individual pricing groups
    pricing_groups_organized = {}
    consumption_options = set()
    unique_regions = set(regions_data.keys())

    # Organize pricing by individual group name across all regions
    for region, region_info in regions_data.items():
        pricing_groups = region_info.get('pricing_groups', {})

        for group_name, group_data in pricing_groups.items():
            if not isinstance(group_data, list) or not group_data:
                continue

            # Use the actual group name instead of categorizing it
            consumption_options.add(group_name)

            if group_name not in pricing_groups_organized:
                pricing_groups_organized[group_name] = {}

            if region not in pricing_groups_organized[group_name]:
                pricing_groups_organized[group_name][region] = []

            # Process each pricing entry in the group
            for entry in group_data:
                if isinstance(entry, dict):
                    # Extract price from the pricing collector format
                    price_value = entry.get('price_per_thousand', 0)

                    # Fallback to other possible price fields
                    if price_value == 0:
                        price_fields_to_try = [
                            'price_per_unit', 'price', 'pricePerUnit', 'original_price',
                            'cost', 'rate', 'amount', 'value'
                        ]

                        for field in price_fields_to_try:
                            if field in entry:
                                try:
                                    if isinstance(entry[field], dict) and 'USD' in entry[field]:
                                        # Handle nested price structure like {'USD': '0.00025'}
                                        price_value = float(entry[field]['USD'])
                                        break
                                    else:
                                        price_value = float(entry[field])
                                        break
                                except (ValueError, TypeError):
                                    continue

                    # Extract unit from various possible fields
                    unit_value = entry.get('unit', entry.get('Unit', '1K tokens'))

                    # Extract description from various possible fields
                    description_value = (
                        entry.get('description') or
                        entry.get('Description') or
                        entry.get('usageType') or
                        entry.get('usage_type') or
                        group_name
                    )

                    processed_entry = {
                        'description': description_value,
                        'price_per_thousand': price_value,
                        'unit': unit_value,
                        'dimension': entry.get('dimension', entry.get('pricing_dimension', entry.get('Dimension', ''))),
                        'usage_type': entry.get('usage_type', entry.get('usageType', '')),
                        'raw_entry': entry
                    }

                    pricing_groups_organized[group_name][region].append(processed_entry)

    # Display simplified metrics banner at the top
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("⚙️ Pricing Groups", f"{len(pricing_groups_organized)}")

    with col2:
        st.metric("🌍 Regions Available", f"{len(unique_regions)}")

    with col3:
        total_entries = sum(len(entries) for group_data in pricing_groups_organized.values()
                           for entries in group_data.values())
        st.metric("📊 Total Pricing Entries", f"{total_entries}")

    st.markdown("---")  # Add separator line

    # Define group styling based on actual group names
    def get_group_style(group_name: str) -> dict:
        """Get styling for a specific pricing group"""
        group_lower = group_name.lower()

        if 'global' in group_lower:
            return {'name': f'🌍 {group_name}', 'color': '#3b82f6'}  # Globe for Global
        elif 'long context' in group_lower:
            return {'name': f'🚀 {group_name}', 'color': '#8b5cf6'}  # Rocket for Long Context
        elif 'on-demand' in group_lower:
            return {'name': f'⚡ {group_name}', 'color': '#10b981'}  # Lightning for standard On-Demand
        elif 'batch' in group_lower:
            return {'name': f'📦 {group_name}', 'color': '#8b5cf6'}  # Package for Batch
        elif 'provisioned' in group_lower:
            return {'name': f'🔧 {group_name}', 'color': '#f59e0b'}  # Wrench for Provisioned
        elif 'custom' in group_lower:
            return {'name': f'🎯 {group_name}', 'color': '#ef4444'}  # Target for Custom
        else:
            return {'name': f'💰 {group_name}', 'color': '#6b7280'}  # Money for others

    # Sort groups for logical display order
    def sort_groups(group_name: str) -> tuple:
        """Sort groups with On-Demand first, then by name"""
        if 'on-demand' in group_name.lower():
            if 'long context' in group_name.lower():
                return (1, group_name)  # On-Demand Long Context second
            elif 'global' in group_name.lower():
                return (2, group_name)  # On-Demand Global third
            else:
                return (0, group_name)  # Regular On-Demand first
        elif 'batch' in group_name.lower():
            return (3, group_name)  # Batch fourth
        elif 'provisioned' in group_name.lower():
            return (4, group_name)  # Provisioned fifth
        else:
            return (5, group_name)  # Everything else last

    # Display pricing by individual group with geographical grouping
    sorted_groups = sorted(pricing_groups_organized.keys(), key=sort_groups)

    for group_name in sorted_groups:
        group_data = pricing_groups_organized[group_name]
        group_style = get_group_style(group_name)

        # Create collapsible expander for each pricing group
        with st.expander(f"{group_style['name']} ({len(group_data)} regions)", expanded=False):
            # Group by geography
            geo_groups = group_pricing_by_geography_comprehensive(group_data)

            # Display geographical groups
            for geo_name, geo_regions in geo_groups.items():
                with st.expander(f"{get_geo_icon(geo_name)} {geo_name} ({len(geo_regions)} regions)", expanded=False):
                    # Sort regions alphabetically and display each region
                    sorted_regions = sorted(geo_regions.items())

                    for region, pricing_items in sorted_regions:
                        region_display_name = get_region_display_name(region)
                        region_code = region
                        with st.expander(f"📍 {region_display_name} ({region_code}) - {len(pricing_items)} items", expanded=False):
                            # Display all pricing items for this region
                            for i, item in enumerate(pricing_items):
                                display_individual_pricing_item(item, group_style['color'], i % 2 == 0)


def categorize_pricing_group(group_name: str) -> str:
    """Categorize pricing group based on name"""
    group_lower = group_name.lower()

    if any(keyword in group_lower for keyword in ['on-demand', 'on_demand', 'standard']):
        return 'on_demand'
    elif any(keyword in group_lower for keyword in ['batch', 'bulk', 'async']):
        return 'batch'
    elif any(keyword in group_lower for keyword in ['provisioned', 'throughput', 'reserved']):
        return 'provisioned_throughput'
    elif any(keyword in group_lower for keyword in ['cross-region', 'cross_region', 'inference']):
        return 'cross_region'
    elif any(keyword in group_lower for keyword in ['customization', 'training', 'fine-tune', 'fine_tune']):
        return 'model_customization'
    elif any(keyword in group_lower for keyword in ['embed', 'embedding', 'vector']):
        return 'embedding'
    else:
        return 'general'


def display_agreement_offers_pricing(comprehensive_pricing: Dict[str, Any]):
    """Display pricing from Agreement Offers API structure"""
    by_region = comprehensive_pricing.get('by_region', {})
    
    if not by_region:
        st.warning("No regional pricing data available.")
        return
    
    # Process pricing data to extract statistics
    consumption_options = set()
    unique_regions = set(by_region.keys())
    
    for region, region_pricing in by_region.items():
        # Check which consumption options are available
        if region_pricing.get('on_demand'):
            consumption_options.add('On-Demand')
        if region_pricing.get('provisioned_throughput'):
            consumption_options.add('Provisioned Throughput')
    
    # Display simplified metrics banner at the top
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("⚙️ Consumption Options", f"{len(consumption_options)}")
    
    with col2:
        st.metric("🌍 Regions Available", f"{len(unique_regions)}")
    
    st.markdown("---")  # Add separator line
    
    # Define category styling
    category_styles = {
        'on_demand': {'name': '🚀 On-Demand Pricing', 'color': '#10b981'},
        'provisioned_throughput': {'name': '⚡ Provisioned Throughput', 'color': '#f59e0b'}
    }
    
    # Process and display pricing by category
    for category_key in ['on_demand', 'provisioned_throughput']:
        category_data = extract_agreement_offers_category_pricing(by_region, category_key)
        
        if not category_data:
            continue
            
        category_style = category_styles.get(category_key, {'name': f'❓ {str(category_key).title()}', 'color': '#6b7280'})
        
        # Create collapsible expander for each category
        with st.expander(f"{category_style['name']} ({len(category_data)} regions)", expanded=False):
            # Group by geography
            geo_groups = group_pricing_by_geography_comprehensive(category_data)
            
            # Display geographical groups
            for geo_name, geo_regions in geo_groups.items():
                with st.expander(f"{get_geo_icon(geo_name)} {geo_name} ({len(geo_regions)} regions)", expanded=False):
                    # Sort regions alphabetically and display each region
                    sorted_regions = sorted(geo_regions.items())
                    
                    for region, pricing_items in sorted_regions:
                        region_display_name = get_region_display_name(region)
                        region_code = region
                        with st.expander(f"📍 {region_display_name} ({region_code}) - {len(pricing_items)} items", expanded=False):
                            # Display all pricing items for this region
                            for i, item in enumerate(pricing_items):
                                display_agreement_offers_pricing_item(item, category_style['color'], i % 2 == 0, region)


def display_pricing_api_pricing(comprehensive_pricing: Dict[str, Any]):
    """Display pricing from AWS Pricing API structure"""
    by_region = comprehensive_pricing.get('by_region', {})
    
    if not by_region:
        st.warning("No regional pricing data available.")
        return
    
    # Process pricing data to extract statistics
    token_types = set()
    unique_regions = set(by_region.keys())
    
    for region, region_pricing in by_region.items():
        # Check which token types are available
        if region_pricing.get('input_tokens'):
            token_types.add('Input Tokens')
        if region_pricing.get('output_tokens'):
            token_types.add('Output Tokens')
    
    # Display simplified metrics banner at the top
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🔤 Token Types", f"{len(token_types)}")
    
    with col2:
        st.metric("🌍 Regions Available", f"{len(unique_regions)}")
    
    st.markdown("---")  # Add separator line
    
    # Define token type styling
    token_styles = {
        'input_tokens': {'name': '📥 Input Tokens', 'color': '#10b981'},
        'output_tokens': {'name': '📤 Output Tokens', 'color': '#3b82f6'}
    }
    
    # Process and display pricing by token type
    for token_type in ['input_tokens', 'output_tokens']:
        token_data = extract_pricing_api_token_pricing(by_region, token_type)
        
        if not token_data:
            continue
            
        token_style = token_styles.get(token_type, {'name': f'🔤 {str(token_type).replace("_", " ").title()}', 'color': '#6b7280'})
        
        # Create collapsible expander for each token type
        with st.expander(f"{token_style['name']} ({len(token_data)} regions)", expanded=False):
            # Group by geography
            geo_groups = group_pricing_by_geography_comprehensive(token_data)
            
            # Display geographical groups
            for geo_name, geo_regions in geo_groups.items():
                with st.expander(f"{get_geo_icon(geo_name)} {geo_name} ({len(geo_regions)} regions)", expanded=False):
                    # Sort regions alphabetically and display each region
                    sorted_regions = sorted(geo_regions.items())
                    
                    for region, pricing_items in sorted_regions:
                        region_display_name = get_region_display_name(region)
                        region_code = region
                        with st.expander(f"📍 {region_display_name} ({region_code}) - {len(pricing_items)} items", expanded=False):
                            # Display all pricing items for this region
                            for i, item in enumerate(pricing_items):
                                display_pricing_api_pricing_item(item, token_style['color'], i % 2 == 0, region)


def display_pricing_item(item: Dict[str, Any], color: str, is_even: bool):
    """Display an individual pricing item with creative styling"""
    # Extract pricing information
    if isinstance(item, dict):
        if 'name' in item:
            display_name = item['name']
        elif 'description' in item:
            # Extract name from description
            desc = item.get('description', '')
            if 'for' in desc:
                display_name = desc.split('for')[1].strip()
            else:
                display_name = desc
        else:
            display_name = str(item.get('usage_type', 'Unknown')).replace('-', ' ').title()
        
        price_value = item.get('price_usd', 0)
        unit = item.get('unit', '')
        
        # Format the price value
        if unit == '1K tokens':
            display_value = f"${price_value:.6f}"
        elif unit == 'image':
            display_value = f"${price_value:.4f}"
        elif unit == 'hour':
            display_value = f"${price_value:.2f}"
        else:
            display_value = f"${price_value:.6f}"
        
        # Format the unit
        if unit == '1K tokens':
            display_unit = "per 1K tokens"
        elif unit:
            display_unit = f"per {unit}"
        else:
            display_unit = ""
    else:
        # Handle non-dict items
        display_name = str(item)
        display_value = "N/A"
        display_unit = ""
    
    # Background color alternation for readability
    bg_color = "#f8fafc" if is_even else "#ffffff"
    
    # Create the pricing card
    st.markdown(f"""
    <div style="background-color: {bg_color}; 
                padding: 15px; 
                border-radius: 8px; 
                margin-bottom: 10px;
                border: 1px solid #e2e8f0;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="flex: 1;">
                <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px; line-height: 1.4;">
                    {display_name}
                </h5>
                <p style="margin: 0; color: #6b7280; font-size: 12px; line-height: 1.3;">{display_unit}</p>
            </div>
            <div style="text-align: right; margin-left: 15px;">
                <div style="font-size: 18px; font-weight: bold; color: {color};">
                    {display_value}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_quota_details(model: Dict[str, Any]):
    """Display comprehensive quota information grouped by region, similar to pricing details"""
    st.subheader("🔢 Model Service Quotas")

    try:
        # Get quota data from the model (updated for new data structure)
        # Try multiple possible field names for quotas
        model_service_quotas = model.get('model_service_quotas', model.get('service_quotas', model.get('quotas', {})))

        # Ensure quota data is in the expected format (handle NaN/float values)
        if not isinstance(model_service_quotas, dict):
            model_service_quotas = {}

        # The regions_data is directly in model_service_quotas, excluding metadata
        regions_data = {k: v for k, v in model_service_quotas.items() if k != 'quota_metadata'}

        if not regions_data:
            st.warning("No service quota information available for this model.")
            st.info("💡 This model may not have established quotas yet, or quotas may be managed at a different level.")
            return

        # Convert list-based quota structure to dict-based structure expected by categorize_quotas_by_region
        converted_regions_data = {}
        for region, quota_list in regions_data.items():
            if isinstance(quota_list, list):
                # Convert list of quota objects to dict with quota_code as key
                converted_regions_data[region] = {}
                for quota_obj in quota_list:
                    if isinstance(quota_obj, dict):
                        quota_code = quota_obj.get('quota_code', f'quota_{len(converted_regions_data[region])}')
                        converted_regions_data[region][quota_code] = quota_obj
            else:
                # If it's already a dict, use as-is
                converted_regions_data[region] = quota_list

        # Process and categorize quotas by category and region
        categorized_quotas_by_region = categorize_quotas_by_region(converted_regions_data)
        
        if not categorized_quotas_by_region or not any(categorized_quotas_by_region.values()):
            st.warning("No service quota information available for this model.")
            return
            
    except Exception as e:
        st.error(f"❌ Error loading quota information: {str(e)}")
        st.info("💡 Please try refreshing the page or contact support if the issue persists.")
        return
    
    # Calculate statistics for the top banner
    total_quotas = 0
    adjustable_count = 0
    regions_with_quotas = set()
    categories_with_quotas = set()
    
    for category, regions in categorized_quotas_by_region.items():
        if regions:
            categories_with_quotas.add(category)
            for region, quotas in regions.items():
                regions_with_quotas.add(region)
                total_quotas += len(quotas)
                adjustable_count += sum(1 for quota in quotas if quota.get('adjustable', False))
    
    # Display metrics banner at the top
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Quotas", total_quotas)
    
    with col2:
        st.metric("🔧 Adjustable", f"{adjustable_count}/{total_quotas}")
    
    with col3:
        st.metric("🌍 Regions", len(regions_with_quotas))
    
    with col4:
        st.metric("📈 Categories", len(categories_with_quotas))
    
    st.markdown("---")  # Add separator line
    
    # Define category styling - Updated order: on-demand, cross-region, batch, provisioned, general
    category_styles = {
        'on_demand': {'name': '🚀 On-Demand Inference', 'color': '#10b981'},
        'cross_region': {'name': '🌍 Cross-Region Inference', 'color': '#3b82f6'},
        'batch': {'name': '📦 Batch Inference', 'color': '#8b5cf6'},
        'provisioned_throughput': {'name': '⚡ Provisioned Throughput', 'color': '#f59e0b'},
        'model_customization': {'name': '🎯 Model Customization', 'color': '#ef4444'},
        'general': {'name': '⚙️ General Limits', 'color': '#6b7280'}
    }
    
    # Display quotas by category with regional grouping in the specified order
    ordered_categories = ['on_demand', 'cross_region', 'batch', 'provisioned_throughput', 'model_customization', 'general']
    
    try:
        # First display categories in the specified order
        for category_key in ordered_categories:
            if category_key in categorized_quotas_by_region and categorized_quotas_by_region[category_key]:
                category_regions = categorized_quotas_by_region[category_key]
                category_info = category_styles.get(category_key, {
                    'name': f'📋 {str(category_key).replace("_", " ").title()}', 
                    'color': '#6b7280'
                })
                
                # Count total quotas in this category
                total_category_quotas = sum(len(quotas) for quotas in category_regions.values())
                
                # Create collapsible expander for each category (collapsed by default)
                with st.expander(
                    f"{category_info['name']} ({len(category_regions)} regions)", 
                    expanded=False
                ):
                    # Group regions by geography
                    geo_groups = group_quotas_by_geography(category_regions)
                    
                    # Display geographical groups
                    for geo_name, geo_regions in geo_groups.items():
                        with st.expander(f"{get_geo_icon(geo_name)} {geo_name} ({len(geo_regions)} regions)", expanded=False):
                            # Sort regions alphabetically and display each region
                            sorted_regions = sorted(geo_regions.items())
                            
                            for region, quotas in sorted_regions:
                                region_display_name = get_region_display_name(region)
                                region_code = region
                                with st.expander(f"📍 {region_display_name} ({region_code}) - {len(quotas)} quotas", expanded=False):
                                    # Display all quotas for this region
                                    for i, quota in enumerate(quotas):
                                        display_individual_quota_item(quota, category_info['color'], i % 2 == 0)
        
        # Then display any remaining categories not in the ordered list
        for category_key, category_regions in categorized_quotas_by_region.items():
            if category_key not in ordered_categories and category_regions:
                category_info = category_styles.get(category_key, {
                    'name': f'📋 {str(category_key).replace("_", " ").title()}', 
                    'color': '#6b7280'
                })
                
                # Count total quotas in this category
                total_category_quotas = sum(len(quotas) for quotas in category_regions.values())
                
                # Create collapsible expander for each category (collapsed by default)
                with st.expander(
                    f"{category_info['name']} ({len(category_regions)} regions)", 
                    expanded=False
                ):
                    # Group regions by geography
                    geo_groups = group_quotas_by_geography(category_regions)
                    
                    # Display geographical groups
                    for geo_name, geo_regions in geo_groups.items():
                        with st.expander(f"{get_geo_icon(geo_name)} {geo_name} ({len(geo_regions)} regions)", expanded=False):
                            # Sort regions alphabetically and display each region
                            sorted_regions = sorted(geo_regions.items())
                            
                            for region, quotas in sorted_regions:
                                region_display_name = get_region_display_name(region)
                                region_code = region
                                with st.expander(f"📍 {region_display_name} ({region_code}) - {len(quotas)} quotas", expanded=False):
                                    # Display all quotas for this region
                                    for i, quota in enumerate(quotas):
                                        display_individual_quota_item(quota, category_info['color'], i % 2 == 0)
                            
    except Exception as e:
        st.error(f"❌ Error displaying quota categories: {str(e)}")
        st.info("💡 Please try refreshing the page or contact support if the issue persists.")


def categorize_quotas_by_region(regions_data: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Categorize quotas by category and then by region, similar to pricing structure"""
    categories = {
        'on_demand': {},
        'cross_region': {},
        'batch': {},
        'provisioned_throughput': {},
        'model_customization': {},
        'general': {}
    }

    try:
        for region, quotas in regions_data.items():
            if not isinstance(quotas, dict):
                continue

            for quota_code, quota_info in quotas.items():
                if not isinstance(quota_info, dict):
                    continue

                quota_name = quota_info.get('quota_name', '')
                
                # Skip empty quota names
                if not quota_name:
                    continue
                
                # Create quota entry with safe value extraction
                try:
                    value = quota_info.get('value', 0)
                    # Ensure value is numeric
                    if not isinstance(value, (int, float)):
                        value = 0
                except (TypeError, ValueError):
                    value = 0
                
                processed_quota = {
                    'code': quota_code,
                    'name': quota_name,
                    'value': value,
                    'unit': str(quota_info.get('unit', '')),
                    'adjustable': bool(quota_info.get('adjustable', False))
                }
                
                # Categorize based on quota name
                quota_name_lower = quota_name.lower()
                category = 'general'  # default category
                
                try:
                    if any(keyword in quota_name_lower for keyword in ['on-demand', 'on demand']):
                        category = 'on_demand'
                    elif any(keyword in quota_name_lower for keyword in ['cross-region', 'cross region']):
                        category = 'cross_region'
                    elif 'batch' in quota_name_lower:
                        category = 'batch'
                    elif any(keyword in quota_name_lower for keyword in ['provisioned', 'model units']):
                        category = 'provisioned_throughput'
                    elif any(keyword in quota_name_lower for keyword in ['customization', 'fine-tuning', 'fine tuning', 'training']):
                        category = 'model_customization'
                    
                    # Initialize region list for this category if it doesn't exist
                    if region not in categories[category]:
                        categories[category][region] = []
                    
                    # Add quota to the appropriate category and region
                    categories[category][region].append(processed_quota)
                    
                except Exception as e:
                    # If categorization fails, put in general category
                    if region not in categories['general']:
                        categories['general'][region] = []
                    categories['general'][region].append(processed_quota)

    except Exception as e:
        # If there's an error processing quotas, return empty categories
        # This will be handled by the calling function
        logging.debug(f"Error processing quota data for categorization: {e}. Returning empty categories.")
    
    return categories


def group_quotas_by_geography(category_regions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Group quota regions by geographical areas"""
    geo_groups = {
        'North America': {},
        'Europe': {},
        'Asia Pacific': {},
        'South America': {},
        'Other Regions': {}
    }
    
    for region, quotas in category_regions.items():
        geo_region = map_region_to_geography(region)
        geo_groups[geo_region][region] = quotas
    
    # Remove empty groups
    return {k: v for k, v in geo_groups.items() if v}


def display_individual_quota_item(quota_info: Dict[str, Any], color: str, is_even: bool):
    """Display an individual quota item without region information (since region is already in the hierarchy)"""
    try:
        quota_name = quota_info.get('name', 'Unknown Quota')
        value = quota_info.get('value', 0)
        unit = quota_info.get('unit', '')
        adjustable = quota_info.get('adjustable', False)
        quota_code = quota_info.get('code', '')
        
        # Safely format the value
        try:
            if isinstance(value, (int, float)) and value is not None:
                if value >= 1000000:
                    formatted_value = f"{value/1000000:.1f}M"
                elif value >= 1000:
                    formatted_value = f"{value/1000:.1f}K"
                else:
                    formatted_value = f"{value:,}"
            else:
                formatted_value = str(value) if value is not None else "0"
        except (TypeError, ValueError):
            formatted_value = "N/A"
        
        # Background color alternation for readability
        bg_color = "#f8fafc" if is_even else "#ffffff"
        
        # Safely format unit display
        unit_display = unit if unit and unit != 'None' else ''
        
        # Create the quota card
        st.markdown(f"""
        <div style="background-color: {bg_color}; 
                    padding: 15px; 
                    border-radius: 8px; 
                    margin-bottom: 10px;
                    border: 1px solid #e2e8f0;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="flex: 1;">
                    <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px; line-height: 1.4;">
                        {quota_name}
                    </h5>
                    <p style="margin: 4px 0 0 0; color: #9ca3af; font-size: 11px;">
                        Code: {quota_code}
                    </p>
                </div>
                <div style="text-align: right; margin-left: 15px;">
                    <div style="font-size: 18px; font-weight: bold; color: {color};">
                        {formatted_value} {unit_display}
                    </div>
                    <div style="font-size: 11px; color: #9ca3af; margin-top: 4px;">
                        {'🔧 Adjustable' if adjustable else '🔒 Fixed'}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        # Fallback display if there's any error
        st.error(f"❌ Error displaying quota: {str(e)}")


def categorize_quotas_from_regions(regions_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize quotas from regions data into logical groups (legacy function for compatibility)"""
    categories = {
        'on_demand': [],
        'cross_region': [],
        'batch': [],
        'provisioned_throughput': [],
        'model_customization': [],
        'general': []
    }
    
    # Track unique quotas by their name to avoid duplicates
    unique_quotas = {}
    
    try:
        for region, quotas in regions_data.items():
            if not isinstance(quotas, dict):
                continue
                
            for quota_code, quota_info in quotas.items():
                if not isinstance(quota_info, dict):
                    continue

                quota_name = quota_info.get('quota_name', '')

                # Skip empty quota names
                if not quota_name:
                    continue
                
                # Skip if we've already processed this quota
                if quota_name in unique_quotas:
                    # Add this region to the existing quota
                    if region not in unique_quotas[quota_name]['regions']:
                        unique_quotas[quota_name]['regions'].append(region)
                    continue
                
                # Create new quota entry with safe value extraction
                try:
                    value = quota_info.get('value', 0)
                    # Ensure value is numeric
                    if not isinstance(value, (int, float)):
                        value = 0
                except (TypeError, ValueError):
                    value = 0
                
                processed_quota = {
                    'code': quota_code,
                    'name': quota_name,
                    'value': value,
                    'unit': str(quota_info.get('unit', '')),
                    'adjustable': bool(quota_info.get('adjustable', False)),
                    'regions': [region]
                }
                
                # Categorize based on quota name
                quota_name_lower = quota_name.lower()
                
                try:
                    if any(keyword in quota_name_lower for keyword in ['on-demand', 'on demand']):
                        categories['on_demand'].append(processed_quota)
                    elif any(keyword in quota_name_lower for keyword in ['cross-region', 'cross region']):
                        categories['cross_region'].append(processed_quota)
                    elif 'batch' in quota_name_lower:
                        categories['batch'].append(processed_quota)
                    elif any(keyword in quota_name_lower for keyword in ['provisioned', 'model units']):
                        categories['provisioned_throughput'].append(processed_quota)
                    elif any(keyword in quota_name_lower for keyword in ['customization', 'fine-tuning', 'fine tuning', 'training']):
                        categories['model_customization'].append(processed_quota)
                    else:
                        categories['general'].append(processed_quota)
                    
                    unique_quotas[quota_name] = processed_quota
                    
                except Exception as e:
                    # If categorization fails, put in general category
                    categories['general'].append(processed_quota)
                    unique_quotas[quota_name] = processed_quota

    except Exception as e:
        # If there's an error processing quotas, return empty categories
        # This will be handled by the calling function
        logging.debug(f"Error processing quota data: {e}. Returning empty categories.")
    
    return categories


def display_individual_quota_with_regions(quota_info: Dict[str, Any], color: str, is_even: bool):
    """Display an individual quota with regions information"""
    try:
        quota_name = quota_info.get('name', 'Unknown Quota')
        value = quota_info.get('value', 0)
        unit = quota_info.get('unit', '')
        adjustable = quota_info.get('adjustable', False)
        regions = quota_info.get('regions', [])
        quota_code = quota_info.get('code', '')
        
        # Safely format the value
        try:
            if isinstance(value, (int, float)) and value is not None:
                if value >= 1000000:
                    formatted_value = f"{value/1000000:.1f}M"
                elif value >= 1000:
                    formatted_value = f"{value/1000:.1f}K"
                else:
                    formatted_value = f"{value:,}"
            else:
                formatted_value = str(value) if value is not None else "0"
        except (TypeError, ValueError):
            formatted_value = "N/A"
        
        # Safely format regions list
        try:
            if isinstance(regions, list) and regions:
                if len(regions) <= 3:
                    regions_display = ", ".join(str(r) for r in regions)
                else:
                    regions_display = f"{', '.join(str(r) for r in regions[:3])} +{len(regions)-3} more"
            else:
                regions_display = "No regions"
        except (TypeError, AttributeError):
            regions_display = "Unknown regions"
        
        # Background color alternation for readability
        bg_color = "#f8fafc" if is_even else "#ffffff"
        
        # Safely format unit display
        unit_display = unit if unit and unit != 'None' else ''
        
        # Create the quota card
        st.markdown(f"""
        <div style="background-color: {bg_color}; 
                    padding: 15px; 
                    border-radius: 8px; 
                    margin-bottom: 10px;
                    border: 1px solid #e2e8f0;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="flex: 1;">
                    <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px; line-height: 1.4;">
                        {quota_name}
                    </h5>
                    <p style="margin: 0; color: #6b7280; font-size: 12px; line-height: 1.3;">
                        📍 {regions_display}
                    </p>
                    <p style="margin: 4px 0 0 0; color: #9ca3af; font-size: 11px;">
                        Code: {quota_code}
                    </p>
                </div>
                <div style="text-align: right; margin-left: 15px;">
                    <div style="font-size: 18px; font-weight: bold; color: {color};">
                        {formatted_value} {unit_display}
                    </div>
                    <div style="font-size: 11px; color: #9ca3af; margin-top: 4px;">
                        {'🔧 Adjustable' if adjustable else '🔒 Fixed'}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        # Fallback display if there's any error
        st.error(f"❌ Error displaying quota: {str(e)}")


def display_individual_quota(quota_info: Dict[str, Any], color: str, is_even: bool):
    """Display an individual quota with creative styling (legacy function for compatibility)"""
    quota_name = quota_info.get('quota_name', quota_info.get('name', 'Unknown Quota'))
    value = quota_info.get('value', 0)
    unit = quota_info.get('unit', '')
    adjustable = quota_info.get('adjustable', False)
    global_quota = quota_info.get('global_quota', False)
    description = quota_info.get('description', '')
    
    # Format the value nicely
    if isinstance(value, (int, float)):
        if value >= 1000000:
            formatted_value = f"{value/1000000:.1f}M"
        elif value >= 1000:
            formatted_value = f"{value/1000:.1f}K"
        else:
            formatted_value = f"{value:,}"
    else:
        formatted_value = str(value)
    
    # Background color alternation for readability
    bg_color = "#f8fafc" if is_even else "#ffffff"
    
    # Create the quota card
    st.markdown(f"""
    <div style="background-color: {bg_color}; 
                padding: 15px; 
                border-radius: 8px; 
                margin-bottom: 10px;
                border: 1px solid #e2e8f0;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="flex: 1;">
                <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px; line-height: 1.4;">
                    {quota_name}
                </h5>
                {f'<p style="margin: 0; color: #6b7280; font-size: 12px; line-height: 1.3;">{description}</p>' if description else ''}
            </div>
            <div style="text-align: right; margin-left: 15px;">
                <div style="font-size: 18px; font-weight: bold; color: {color};">
                    {formatted_value} {unit}
                </div>
                <div style="font-size: 11px; color: #9ca3af; margin-top: 4px;">
                    {'🔧 Adjustable' if adjustable else '🔒 Fixed'} 
                    {' • 🌍 Global' if global_quota else ' • 🏠 Account'}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def group_pricing_by_geography(pricing_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group pricing items by geographical regions"""
    geo_groups = {
        'North America': [],
        'Europe': [],
        'Asia Pacific': [],
        'South America': [],
        'Other Regions': []
    }
    
    for item in pricing_items:
        region = extract_region_from_pricing(item)
        geo_region = map_region_to_geography(region)
        geo_groups[geo_region].append(item)
    
    # Remove empty groups
    return {k: v for k, v in geo_groups.items() if v}


def extract_region_from_pricing(pricing_item: Dict[str, Any]) -> str:
    """Extract AWS region from pricing item"""
    # Try to extract from usage_type first
    usage_type = pricing_item.get('usage_type', '').lower()
    
    # Common region prefixes in usage types
    region_mappings = {
        'use1': 'us-east-1',
        'use2': 'us-east-2', 
        'usw1': 'us-west-1',
        'usw2': 'us-west-2',
        'eu': 'eu-west-1',
        'euw1': 'eu-west-1',
        'euw2': 'eu-west-2',
        'euw3': 'eu-west-3',
        'euc1': 'eu-central-1',
        'eus1': 'eu-south-1',
        'eun1': 'eu-north-1',
        'aps1': 'ap-southeast-1',
        'aps2': 'ap-southeast-2',
        'aps3': 'ap-south-1',
        'apn1': 'ap-northeast-1',
        'apn2': 'ap-northeast-2',
        'apn3': 'ap-northeast-3',
        'ape1': 'ap-east-1',
        'can1': 'ca-central-1',
        'sae1': 'sa-east-1'
    }
    
    # Extract region prefix from usage_type
    for prefix, region in region_mappings.items():
        if usage_type.startswith(prefix + '-'):
            return region
    
    # Try to extract from description
    description = pricing_item.get('description', '').lower()
    
    # Common region names in descriptions
    description_mappings = {
        'us east (n. virginia)': 'us-east-1',
        'us east (ohio)': 'us-east-2',
        'us west (oregon)': 'us-west-2',
        'us west (n. california)': 'us-west-1',
        'europe (ireland)': 'eu-west-1',
        'europe (london)': 'eu-west-2',
        'europe (paris)': 'eu-west-3',
        'europe (frankfurt)': 'eu-central-1',
        'eu (frankfurt)': 'eu-central-1',
        'eu (paris)': 'eu-west-3',
        'eu (milan)': 'eu-south-1',
        'asia pacific (singapore)': 'ap-southeast-1',
        'asia pacific (sydney)': 'ap-southeast-2',
        'asia pacific (mumbai)': 'ap-south-1',
        'asia pacific (tokyo)': 'ap-northeast-1',
        'asia pacific (seoul)': 'ap-northeast-2',
        'canada (central)': 'ca-central-1',
        'south america (sao paulo)': 'sa-east-1'
    }
    
    for desc_region, region in description_mappings.items():
        if desc_region in description:
            return region
    
    return 'unknown'


def map_region_to_geography(region: str) -> str:
    """Map AWS region to geographical area"""
    if region.startswith('us-') or region.startswith('ca-'):
        return 'North America'
    elif region.startswith('eu-'):
        return 'Europe'
    elif region.startswith('ap-'):
        return 'Asia Pacific'
    elif region.startswith('sa-'):
        return 'South America'
    else:
        return 'Other Regions'


def get_geo_icon(geo_name: str) -> str:
    """Get emoji icon for geographical region"""
    icons = {
        'North America': '🇺🇸',
        'Europe': '🇪🇺', 
        'Asia Pacific': '🌏',
        'South America': '🇧🇷',
        'Other Regions': '🌍'
    }
    return icons.get(geo_name, '🌍')


def display_pricing_item_with_region(item: Dict[str, Any], color: str, is_even: bool, region: str):
    """Display an individual pricing item with region information"""
    # Extract pricing information
    if isinstance(item, dict):
        if 'name' in item:
            display_name = item['name']
        elif 'description' in item:
            # Extract name from description
            desc = item.get('description', '')
            if 'for' in desc:
                display_name = desc.split('for')[1].strip()
            else:
                display_name = desc
        else:
            display_name = str(item.get('usage_type', 'Unknown')).replace('-', ' ').title()
        
        price_value = item.get('price_usd', 0)
        unit = item.get('unit', '')
        
        # Format the price value
        if unit == '1K tokens':
            display_value = f"${price_value:.6f}"
        elif unit == 'image':
            display_value = f"${price_value:.4f}"
        elif unit == 'hour':
            display_value = f"${price_value:.2f}"
        else:
            display_value = f"${price_value:.6f}"
        
        # Format the unit
        if unit == '1K tokens':
            display_unit = "per 1K tokens"
        elif unit:
            display_unit = f"per {unit}"
        else:
            display_unit = ""
    else:
        # Handle non-dict items
        display_name = str(item)
        display_value = "N/A"
        display_unit = ""
    
    # Background color alternation for readability
    bg_color = "#f8fafc" if is_even else "#ffffff"
    
    # Format region display
    region_display = region if region != 'unknown' else 'Unknown Region'
    
    # Create the pricing card with region information
    st.markdown(f"""
    <div style="background-color: {bg_color}; 
                padding: 15px; 
                border-radius: 8px; 
                margin-bottom: 10px;
                border: 1px solid #e2e8f0;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="flex: 1;">
                <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px; line-height: 1.4;">
                    {display_name}
                </h5>
                <p style="margin: 0; color: #6b7280; font-size: 12px; line-height: 1.3;">{display_unit}</p>
                <p style="margin: 4px 0 0 0; color: #9ca3af; font-size: 11px; font-weight: 500;">📍 {region_display}</p>
            </div>
            <div style="text-align: right; margin-left: 15px;">
                <div style="font-size: 18px; font-weight: bold; color: {color};">
                    {display_value}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def extract_enhanced_feature(pricing_item: Dict[str, Any]) -> str:
    """Extract and enhance feature classification from pricing item"""
    # Get the original feature
    original_feature = pricing_item.get('feature', 'Unknown')
    
    # Get usage type and description for enhanced detection
    usage_type = pricing_item.get('usage_type', '').lower()
    description = pricing_item.get('description', '').lower()
    
    # Enhanced detection for provisioned throughput variants
    if 'provisionedthroughput' in usage_type or 'provisioned throughput' in description:
        if 'nocommit' in usage_type or 'no commitment' in description:
            return 'provisioned throughput inference - no commitment'
        elif '1month' in usage_type or '1 month' in description:
            return 'provisioned throughput inference - 1 month'
        elif '6months' in usage_type or '6 months' in description:
            return 'provisioned throughput inference - 6 months'
        else:
            return 'provisioned throughput'
    
    # Enhanced detection for batch inference
    if 'batch' in usage_type or 'batch' in description:
        return 'batch inference'
    
    # Enhanced detection for model customization
    if 'customization' in usage_type or 'customization' in description:
        return 'model customization'
    
    # Enhanced detection for fine-tuning
    if 'fine-tuning' in usage_type or 'fine-tuning' in description or 'finetuning' in usage_type:
        return 'fine-tuning'
    
    # Return original feature if no enhanced detection matches
    return original_feature

def has_non_zero_pricing(pricing_category: Dict[str, Any]) -> bool:
    """Check if a pricing category has any non-zero values"""
    if not pricing_category:
        return False
    
    for key, value in pricing_category.items():
        if isinstance(value, (int, float)) and value > 0:
            return True
    return False


def extract_category_pricing(regions_pricing: Dict[str, Any], category: str) -> Dict[str, List[Dict[str, Any]]]:
    """Extract pricing data for a specific category across all regions"""
    category_data = {}
    
    for region, region_pricing in regions_pricing.items():
        category_pricing = region_pricing.get(category, {})
        
        if not has_non_zero_pricing(category_pricing):
            continue
            
        pricing_items = []
        
        # Convert pricing data to display items
        for pricing_type, value in category_pricing.items():
            if isinstance(value, (int, float)) and value > 0:
                pricing_items.append({
                    'name': format_pricing_type_name(pricing_type),
                    'value': value,
                    'unit': get_pricing_unit(pricing_type),
                    'type': pricing_type
                })
        
        if pricing_items:
            category_data[region] = pricing_items
    
    return category_data


def format_pricing_type_name(pricing_type: str) -> str:
    """Format pricing type name for display"""
    name_mappings = {
        'input_tokens': 'Input Tokens',
        'output_tokens': 'Output Tokens',
        'per_image': 'Image Processing',
        'per_video': 'Video Processing',
        'per_request': 'Per Request',
        'model_units_per_hour': 'Model Units',
        'per_hour': 'Per Hour',
        'training_per_hour': 'Training',
        'storage_per_month': 'Storage'
    }
    
    return name_mappings.get(pricing_type, str(pricing_type).replace('_', ' ').title())


def get_pricing_unit(pricing_type: str) -> str:
    """Get the unit for a pricing type"""
    unit_mappings = {
        'input_tokens': 'per 1K tokens',
        'output_tokens': 'per 1K tokens',
        'per_image': 'per image',
        'per_video': 'per second',
        'per_request': 'per request',
        'model_units_per_hour': 'per hour',
        'per_hour': 'per hour',
        'training_per_hour': 'per hour',
        'storage_per_month': 'per month'
    }
    
    return unit_mappings.get(pricing_type, '')


def group_pricing_by_geography_comprehensive(category_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Group comprehensive pricing data by geographical regions"""
    geo_groups = {
        'North America': {},
        'Europe': {},
        'Asia Pacific': {},
        'South America': {},
        'Other Regions': {}
    }
    
    for region, pricing_items in category_data.items():
        geo_region = map_region_to_geography(region)
        geo_groups[geo_region][region] = pricing_items
    
    # Remove empty groups
    return {k: v for k, v in geo_groups.items() if v}


def extract_agreement_offers_category_pricing(by_region: Dict[str, Any], category: str) -> Dict[str, List[Dict[str, Any]]]:
    """Extract pricing data for a specific category from Agreement Offers structure"""
    category_data = {}
    
    for region, region_pricing in by_region.items():
        category_pricing = region_pricing.get(category, {})
        
        if not category_pricing:
            continue
            
        pricing_items = []
        
        if category == 'on_demand':
            # Handle on-demand pricing structure
            for token_type in ['input_tokens', 'output_tokens']:
                token_data = category_pricing.get(token_type, [])
                if isinstance(token_data, list):
                    for item in token_data:
                        pricing_items.append({
                            'name': f"{str(token_type).replace('_', ' ').title()}",
                            'price': float(item.get('price', 0)),
                            'unit': item.get('unit', ''),
                            'description': item.get('description', ''),
                            'dimension': item.get('dimension', ''),
                            'type': token_type
                        })
        
        elif category == 'provisioned_throughput':
            # Handle provisioned throughput pricing structure
            for commitment_type in ['no_commitment', 'one_month_commitment', 'six_months_commitment']:
                commitment_data = category_pricing.get(commitment_type, [])
                if isinstance(commitment_data, list):
                    for item in commitment_data:
                        pricing_items.append({
                            'name': f"Provisioned Throughput - {str(commitment_type).replace('_', ' ').title()}",
                            'price': float(item.get('price', 0)),
                            'unit': item.get('unit', ''),
                            'description': item.get('description', ''),
                            'dimension': item.get('dimension', ''),
                            'type': commitment_type
                        })
        
        if pricing_items:
            category_data[region] = pricing_items
    
    return category_data


def extract_pricing_api_token_pricing(by_region: Dict[str, Any], token_type: str) -> Dict[str, List[Dict[str, Any]]]:
    """Extract pricing data for a specific token type from Pricing API structure"""
    token_data = {}
    
    for region, region_pricing in by_region.items():
        token_pricing = region_pricing.get(token_type, [])
        
        if not isinstance(token_pricing, list) or not token_pricing:
            continue
            
        pricing_items = []
        
        for item in token_pricing:
            # Extract pricing from the complex Pricing API structure
            terms = item.get('terms', {}).get('OnDemand', {})
            
            for term_key, term_data in terms.items():
                price_dimensions = term_data.get('priceDimensions', {})
                
                for price_key, price_data in price_dimensions.items():
                    price_per_unit = price_data.get('pricePerUnit', {}).get('USD', '0')
                    
                    pricing_items.append({
                        'name': f"{str(token_type).replace('_', ' ').title()}",
                        'price': float(price_per_unit),
                        'unit': price_data.get('unit', ''),
                        'description': price_data.get('description', ''),
                        'sku': item.get('sku', ''),
                        'type': token_type
                    })
        
        if pricing_items:
            token_data[region] = pricing_items
    
    return token_data


def display_agreement_offers_pricing_item(item: Dict[str, Any], color: str, is_even: bool, region: str):
    """Display a pricing item from Agreement Offers structure"""
    name = item.get('name', 'Unknown')
    price = item.get('price', 0)
    unit = item.get('unit', '')
    description = item.get('description', '')
    dimension = item.get('dimension', '')
    
    # Format the price value
    if 'tokens' in unit.lower():
        display_value = f"${price:.6f}"
    elif 'units' in unit.lower():
        display_value = f"${price:.2f}"
    else:
        display_value = f"${price:.6f}"
    
    # Format the unit
    display_unit = unit if unit else ""
    
    # Background color alternation for readability
    bg_color = "#f8fafc" if is_even else "#ffffff"
    
    # Create the pricing card
    st.markdown(f"""
    <div style="background-color: {bg_color}; 
                padding: 15px; 
                border-radius: 8px; 
                margin-bottom: 10px;
                border: 1px solid #e2e8f0;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="flex: 1;">
                <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px; line-height: 1.4;">
                    {name}
                </h5>
                <p style="margin: 0; color: #6b7280; font-size: 12px; line-height: 1.3;">{display_unit}</p>
                {f'<p style="margin: 4px 0 0 0; color: #9ca3af; font-size: 11px;">{description}</p>' if description else ''}
                {f'<p style="margin: 4px 0 0 0; color: #9ca3af; font-size: 10px;">Dimension: {dimension}</p>' if dimension else ''}
            </div>
            <div style="text-align: right; margin-left: 15px;">
                <div style="font-size: 18px; font-weight: bold; color: {color};">
                    {display_value}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_pricing_api_pricing_item(item: Dict[str, Any], color: str, is_even: bool, region: str):
    """Display a pricing item from Pricing API structure"""
    name = item.get('name', 'Unknown')
    price = item.get('price', 0)
    unit = item.get('unit', '')
    description = item.get('description', '')
    sku = item.get('sku', '')
    
    # Format the price value
    if 'tokens' in unit.lower():
        display_value = f"${price:.6f}"
    elif 'image' in unit.lower():
        display_value = f"${price:.4f}"
    elif 'hour' in unit.lower():
        display_value = f"${price:.2f}"
    else:
        display_value = f"${price:.6f}"
    
    # Format the unit
    display_unit = unit if unit else ""
    
    # Background color alternation for readability
    bg_color = "#f8fafc" if is_even else "#ffffff"
    
    # Create the pricing card
    st.markdown(f"""
    <div style="background-color: {bg_color}; 
                padding: 15px; 
                border-radius: 8px; 
                margin-bottom: 10px;
                border: 1px solid #e2e8f0;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="flex: 1;">
                <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px; line-height: 1.4;">
                    {name}
                </h5>
                <p style="margin: 0; color: #6b7280; font-size: 12px; line-height: 1.3;">{display_unit}</p>
                {f'<p style="margin: 4px 0 0 0; color: #9ca3af; font-size: 11px;">{description}</p>' if description else ''}
                {f'<p style="margin: 4px 0 0 0; color: #9ca3af; font-size: 10px;">SKU: {sku}</p>' if sku else ''}
            </div>
            <div style="text-align: right; margin-left: 15px;">
                <div style="font-size: 18px; font-weight: bold; color: {color};">
                    {display_value}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def get_region_display_name(region: str) -> str:
    """Get human-readable display name for AWS region"""
    region_names = {
        'us-east-1': 'US East (N. Virginia)',
        'us-east-2': 'US East (Ohio)',
        'us-west-1': 'US West (N. California)',
        'us-west-2': 'US West (Oregon)',
        'eu-west-1': 'Europe (Ireland)',
        'eu-west-2': 'Europe (London)',
        'eu-west-3': 'Europe (Paris)',
        'eu-central-1': 'Europe (Frankfurt)',
        'eu-north-1': 'Europe (Stockholm)',
        'eu-south-1': 'Europe (Milan)',
        'ap-northeast-1': 'Asia Pacific (Tokyo)',
        'ap-northeast-2': 'Asia Pacific (Seoul)',
        'ap-northeast-3': 'Asia Pacific (Osaka)',
        'ap-southeast-1': 'Asia Pacific (Singapore)',
        'ap-southeast-2': 'Asia Pacific (Sydney)',
        'ap-south-1': 'Asia Pacific (Mumbai)',
        'ap-east-1': 'Asia Pacific (Hong Kong)',
        'ca-central-1': 'Canada (Central)',
        'sa-east-1': 'South America (São Paulo)',
        'me-south-1': 'Middle East (Bahrain)',
        'af-south-1': 'Africa (Cape Town)'
    }
    
    return region_names.get(region, str(region).title())


def display_comprehensive_pricing_item(item: Dict[str, Any], color: str, is_even: bool, region: str):
    """Display a comprehensive pricing item with region information"""
    name = item.get('name', 'Unknown')
    value = item.get('value', 0)
    unit = item.get('unit', '')
    pricing_type = item.get('type', '')
    
    # Format the price value based on type
    if pricing_type in ['input_tokens', 'output_tokens']:
        display_value = f"${value:.6f}"
    elif pricing_type in ['per_image', 'per_video']:
        display_value = f"${value:.4f}"
    elif pricing_type in ['per_hour', 'training_per_hour', 'storage_per_month']:
        display_value = f"${value:.2f}"
    else:
        display_value = f"${value:.6f}"
    
    # Background color alternation for readability
    bg_color = "#f8fafc" if is_even else "#ffffff"
    
    # Format region display
    region_display = region
    
    # Create the pricing card with region information
    st.markdown(f"""
    <div style="background-color: {bg_color}; 
                padding: 15px; 
                border-radius: 8px; 
                margin-bottom: 10px;
                border: 1px solid #e2e8f0;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="flex: 1;">
                <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px; line-height: 1.4;">
                    {name}
                </h5>
                <p style="margin: 0; color: #6b7280; font-size: 12px; line-height: 1.3;">{unit}</p>
                <p style="margin: 4px 0 0 0; color: #9ca3af; font-size: 11px; font-weight: 500;">📍 {region_display}</p>
            </div>
            <div style="text-align: right; margin-left: 15px;">
                <div style="font-size: 18px; font-weight: bold; color: {color};">
                    {display_value}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def get_region_display_name(region_code: str) -> str:
    """Get human-readable region name from AWS region code"""
    region_names = {
        # US Regions
        'us-east-1': 'N. Virginia',
        'us-east-2': 'Ohio',
        'us-west-1': 'N. California',
        'us-west-2': 'Oregon',
        
        # Europe Regions
        'eu-west-1': 'Ireland',
        'eu-west-2': 'London',
        'eu-west-3': 'Paris',
        'eu-central-1': 'Frankfurt',
        'eu-central-2': 'Zurich',
        'eu-south-1': 'Milan',
        'eu-south-2': 'Spain',
        'eu-north-1': 'Stockholm',
        
        # Asia Pacific Regions
        'ap-southeast-1': 'Singapore',
        'ap-southeast-2': 'Sydney',
        'ap-southeast-3': 'Jakarta',
        'ap-southeast-4': 'Melbourne',
        'ap-south-1': 'Mumbai',
        'ap-south-2': 'Hyderabad',
        'ap-northeast-1': 'Tokyo',
        'ap-northeast-2': 'Seoul',
        'ap-northeast-3': 'Osaka',
        'ap-east-1': 'Hong Kong',
        
        # Canada Regions
        'ca-central-1': 'Canada Central',
        'ca-west-1': 'Calgary',
        
        # South America Regions
        'sa-east-1': 'São Paulo',
        
        # Middle East Regions
        'me-south-1': 'Bahrain',
        'me-central-1': 'UAE',
        
        # Africa Regions
        'af-south-1': 'Cape Town',
        
        # Other/Special Regions
        'us-gov-east-1': 'AWS GovCloud (US-East)',
        'us-gov-west-1': 'AWS GovCloud (US-West)',
        'cn-north-1': 'Beijing',
        'cn-northwest-1': 'Ningxia'
    }
    
    return region_names.get(str(region_code).lower(), str(region_code).replace('-', ' ').title())


def display_technical_specifications(model: Dict[str, Any]):
    """Display comprehensive technical specifications for the model"""
    st.subheader("⚙️ Technical Specifications")
    
    # Extract technical data - use new field names with fallbacks to old names
    model_id = model.get('model_id', 'Unknown')
    provider = model.get('provider', model.get('model_provider', 'Unknown'))
    capabilities = model.get('model_capabilities', model.get('capabilities', []))
    use_cases = model.get('model_use_cases', model.get('use_cases', []))
    context_window = model.get('context_window', 0)
    max_output_tokens = model.get('max_output_tokens', 0)
    languages = model.get('languages_supported', model.get('languages', []))
    input_modalities = model.get('input_modalities', model.get('model_input_modalities', []))
    output_modalities = model.get('output_modalities', model.get('model_output_modalities', []))
    consumption_options = model.get('consumption_options', model.get('model_consumption_options', []))
    # Enhanced customization data extraction - try multiple sources
    def get_customizations_data(model):
        """Extract customization data from different possible sources"""
        # Try direct fields first
        for field_name in ['customizations_supported', 'model_customizations_supported', 'customization_supported']:
            if field_name in model and model[field_name]:
                data = model[field_name]
                if isinstance(data, list):
                    return data

        # Try nested customization field
        if 'customization' in model and isinstance(model['customization'], dict):
            custom_data = model['customization']
            for nested_field in ['customization_supported', 'supported', 'options']:
                if nested_field in custom_data and custom_data[nested_field]:
                    data = custom_data[nested_field]
                    if isinstance(data, list):
                        return data

        return []  # Return empty list if no customization data found

    customizations_supported = get_customizations_data(model)
    inference_types_supported = model.get('inference_types_supported', model.get('model_inference_types_supported', []))
    streaming_supported = model.get('streaming_supported', model.get('response_streaming_supported', False))
    cross_region_inference = model.get('cross_region_inference', {})
    batch_inference_supported = model.get('batch_inference_supported', {})
    model_lifecycle = model.get('model_lifecycle', {})
    documentation_links = model.get('documentation_links', {})

    # Ensure extracted data is in the expected format (handle NaN/float values)
    if not isinstance(cross_region_inference, dict):
        cross_region_inference = {}
    if not isinstance(batch_inference_supported, dict):
        batch_inference_supported = {}
    if not isinstance(model_lifecycle, dict):
        model_lifecycle = {}
    if not isinstance(documentation_links, dict):
        documentation_links = {}
    
    # Calculate statistics for the top banner
    total_languages = len(languages) if isinstance(languages, list) else 0
    total_capabilities = len(capabilities) if isinstance(capabilities, list) else 0
    total_use_cases = len(use_cases) if isinstance(use_cases, list) else 0

    # Extract regions - try multiple possible field names
    regions_list = model.get('regions', model.get('model_regions', []))
    if not regions_list:
        # Try extracting from cross_region_inference if available
        if isinstance(cross_region_inference, dict):
            source_regions = cross_region_inference.get('source_regions', [])
            dest_regions = cross_region_inference.get('destination_regions', [])
            regions_list = list(set(source_regions + dest_regions)) if source_regions or dest_regions else []

    regions_available = len(regions_list) if isinstance(regions_list, list) else 0
    
    # Display metrics banner at the top
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌍 Regions", regions_available)
    
    with col2:
        st.metric("🗣️ Languages", total_languages)
    
    with col3:
        st.metric("⚡ Capabilities", total_capabilities)
    
    with col4:
        lifecycle_status = model_lifecycle.get('status', 'Unknown').upper()
        if lifecycle_status == 'ACTIVE':
            status_color = '#10b981'  # Green for Active
        elif lifecycle_status == 'LEGACY':
            status_color = '#F59E0B'  # Orange for Legacy
        else:
            status_color = '#6b7280'  # Gray for Unknown
        st.markdown(f"""
        <div style="text-align: center;">
            <p style="margin: 0; color: #6b7280; font-size: 14px;">📊 Status</p>
            <div style="background-color: {status_color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-top: 4px; display: inline-block;">
                {lifecycle_status}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")  # Add separator line
    
    # Core Model Information (now includes Enhanced Analysis)
    with st.expander("🔧 Core Model Information", expanded=False):
        display_core_model_info(model)

    # Capabilities and Use Cases
    with st.expander("⚡ Capabilities & Use Cases", expanded=False):
        display_capabilities_and_use_cases(capabilities, use_cases, input_modalities, output_modalities)
    
    # Language Support
    with st.expander("🗣️ Language Support", expanded=False):
        # Safely get language support level
        language_support_level = model.get('language_support_level', 'Unknown')
        if not isinstance(language_support_level, str):
            language_support_level = str(language_support_level) if language_support_level is not None else 'Unknown'
        
        # Safely get primary languages
        primary_languages = model.get('primary_languages', [])
        if not isinstance(primary_languages, list):
            primary_languages = [primary_languages] if primary_languages else []
        
        display_language_support(languages, language_support_level, primary_languages)
    
    # Regional Availability
    with st.expander("🌍 Regional Availability", expanded=False):
        display_regional_availability(regions_list)
    
    # Consumption & Deployment Options
    with st.expander("🚀 Consumption & Deployment Options", expanded=False):
        display_consumption_options(consumption_options, inference_types_supported, customizations_supported)
    
    # Cross-Region Inference
    with st.expander("🌐 Cross-Region Inference", expanded=False):
        display_cross_region_inference(cross_region_inference)
    
    # Batch Inference Support
    with st.expander("📦 Batch Inference Support", expanded=False):
        display_batch_inference_support(batch_inference_supported)


def get_provider_documentation_url(provider: str) -> str:
    """Get documentation URL for a provider"""
    provider_docs = {
        'Anthropic': 'https://docs.anthropic.com/en/docs/about-claude/models',
        'Amazon': 'https://docs.aws.amazon.com/bedrock/latest/userguide/foundation-models.html',
        'AI21 Labs': 'https://docs.ai21.com/docs/overview',
        'Cohere': 'https://docs.cohere.com/docs/models',
        'Meta': 'https://llama.meta.com/docs/',
        'Mistral AI': 'https://docs.mistral.ai/models/',
        'Stability AI': 'https://platform.stability.ai/docs/getting-started'
    }
    return provider_docs.get(provider, None)

def display_core_model_info(model: Dict[str, Any]):
    """Display core model information with integrated enhanced analysis"""
    # Extract data from model
    model_id = model.get('model_id', 'Unknown')
    provider = model.get('provider', 'Unknown')
    streaming_supported = model.get('streaming_supported', False)
    model_lifecycle = model.get('model_lifecycle', {})
    documentation_links = model.get('documentation_links', {})

    # Get enhanced context window data from converse_data
    context_window = get_converse_context_window(model)
    max_output_tokens = get_converse_max_output_tokens(model)
    size_category = get_converse_size_category(model)
    # Ensure model_lifecycle is a dictionary (handle NaN/float values)
    if not isinstance(model_lifecycle, dict):
        model_lifecycle = {}

    # Ensure documentation_links is a dictionary (handle NaN/float values)
    if documentation_links is None or not isinstance(documentation_links, dict):
        documentation_links = {}

    # 2x2 Layout for Core Model Information
    # First row: Model ID and Lifecycle Status
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #e2e8f0;">
            <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px;">Model ID</h5>
            <p style="margin: 0; color: #6b7280; font-size: 12px; font-family: monospace;">{model_id}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        lifecycle_status = model_lifecycle.get('status', 'Unknown').upper()
        if lifecycle_status == 'ACTIVE':
            lifecycle_color = '#10b981'  # Green for Active
        elif lifecycle_status == 'LEGACY':
            lifecycle_color = '#F59E0B'  # Orange for Legacy
        else:
            lifecycle_color = '#6b7280'  # Gray for Unknown
        st.markdown(f"""
        <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #e2e8f0;">
            <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px;">Lifecycle Status</h5>
            <div style="background-color: {lifecycle_color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; display: inline-block;">
                {lifecycle_status}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Second row: Provider and Streaming Support
    col3, col4 = st.columns(2)

    with col3:
        # Get documentation URL from model data (or fallback to provider default)
        if documentation_links and documentation_links.get('primary'):
            provider_doc_url = documentation_links.get('primary')
        else:
            provider_doc_url = get_provider_documentation_url(provider)

        st.markdown(f"""
        <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #e2e8f0;">
            <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px;">Provider</h5>
            <p style="margin: 0; color: #10b981; font-size: 14px; font-weight: bold;">{provider}</p>
            {f'<a href="{provider_doc_url}" target="_blank" style="color: #3b82f6; font-size: 12px; text-decoration: none;">📚 View Documentation →</a>' if provider_doc_url else ''}
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #e2e8f0;">
            <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px;">Streaming Support</h5>
            <p style="margin: 0; color: {'#10b981' if streaming_supported else '#ef4444'}; font-size: 14px; font-weight: bold;">
                {'✅ Supported' if streaming_supported else '❌ Not Supported'}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Add Enhanced Analysis Integration - Context Window and Output Tokens
    st.markdown("---")
    st.markdown("**📏 Enhanced Context Analysis**")

    col5, col6 = st.columns(2)

    with col5:
        # Context Window with proper display
        if context_window > 0:
            # Format context window display
            if context_window >= 1000000:
                context_display = f"{context_window/1000000:.1f}M tokens"
            elif context_window >= 1000:
                context_display = f"{context_window/1000:.0f}K tokens"
            else:
                context_display = f"{context_window:,} tokens"

            st.markdown(f"""
            <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #e2e8f0;">
                <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px;">Context Window</h5>
                <p style="margin: 0; color: #3b82f6; font-size: 16px; font-weight: bold;">{context_display}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #e2e8f0;">
                <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px;">Context Window</h5>
                <p style="margin: 0; color: #6b7280; font-size: 14px;">Not specified</p>
            </div>
            """, unsafe_allow_html=True)

    with col6:
        # Max Output Tokens
        if max_output_tokens > 0:
            # Format max output tokens display
            if max_output_tokens >= 1000000:
                output_display = f"{max_output_tokens/1000000:.1f}M tokens"
            elif max_output_tokens >= 1000:
                output_display = f"{max_output_tokens/1000:.0f}K tokens"
            else:
                output_display = f"{max_output_tokens:,} tokens"

            st.markdown(f"""
            <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #e2e8f0;">
                <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px;">Max Output Tokens</h5>
                <p style="margin: 0; color: #10b981; font-size: 16px; font-weight: bold;">{output_display}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #e2e8f0;">
                <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px;">Max Output Tokens</h5>
                <p style="margin: 0; color: #6b7280; font-size: 14px;">Not specified</p>
            </div>
            """, unsafe_allow_html=True)




def display_capabilities_and_use_cases(capabilities: List[str], use_cases: List[str], input_modalities: List[str], output_modalities: List[str]):
    """Display capabilities and use cases with modalities on top, then divider, then capabilities/use cases"""

    # === TOP SECTION: INPUT/OUTPUT MODALITIES ===
    st.markdown("### 📥📤 Input & Output Modalities")

    modality_col1, modality_col2 = st.columns(2)

    with modality_col1:
        st.markdown("**📥 Input Modalities**")
        if input_modalities:
            for i, modality in enumerate(input_modalities):
                bg_color = "#f8fafc" if i % 2 == 0 else "#ffffff"
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 12px; border-radius: 8px; margin-bottom: 6px; border: 1px solid #e2e8f0;">
                    <p style="margin: 0; color: #3b82f6; font-size: 14px; font-weight: 500;">📥 {modality}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No input modalities specified")

    with modality_col2:
        st.markdown("**📤 Output Modalities**")
        if output_modalities:
            for i, modality in enumerate(output_modalities):
                bg_color = "#f8fafc" if i % 2 == 0 else "#ffffff"
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 12px; border-radius: 8px; margin-bottom: 6px; border: 1px solid #e2e8f0;">
                    <p style="margin: 0; color: #10b981; font-size: 14px; font-weight: 500;">📤 {modality}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No output modalities specified")

    # === DIVIDING LINE ===
    st.markdown("---")

    # === BOTTOM SECTION: CAPABILITIES & USE CASES ===
    st.markdown("### 🎯💼 Capabilities & Use Cases")

    cap_col1, cap_col2 = st.columns(2)

    with cap_col1:
        st.markdown("**🎯 Capabilities**")
        if capabilities:
            for i, capability in enumerate(capabilities):
                bg_color = "#f8fafc" if i % 2 == 0 else "#ffffff"
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 12px; border-radius: 8px; margin-bottom: 6px; border: 1px solid #e2e8f0;">
                    <p style="margin: 0; color: #1f2937; font-size: 13px;">• {str(capability).replace('_', ' ').title()}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No capabilities specified")

    with cap_col2:
        st.markdown("**💼 Use Cases**")
        if use_cases:
            for i, use_case in enumerate(use_cases):
                bg_color = "#f8fafc" if i % 2 == 0 else "#ffffff"
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 12px; border-radius: 8px; margin-bottom: 6px; border: 1px solid #e2e8f0;">
                    <p style="margin: 0; color: #1f2937; font-size: 13px;">• {str(use_case).title()}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No use cases specified")


def display_language_support(languages: List[str], support_level: str, primary_languages: List[str]):
    """Display language support information"""
    # Use full container width for languages display
    st.markdown("**🗣️ All Supported Languages**")
    if languages:
        # Group languages in pairs for better display
        for i in range(0, len(languages), 2):
            lang_pair = languages[i:i+2]
            cols = st.columns(len(lang_pair))
            for j, lang in enumerate(lang_pair):
                with cols[j]:
                    bg_color = "#f8fafc" if (i + j) % 2 == 0 else "#ffffff"
                    st.markdown(f"""
                    <div style="background-color: {bg_color}; padding: 8px; border-radius: 6px; margin-bottom: 5px; border: 1px solid #e2e8f0;">
                        <p style="margin: 0; color: #6b7280; font-size: 12px; text-align: center;">{lang}</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No languages specified")


def display_consumption_options(consumption_options: List[str], inference_types: List[str], customizations: List[str]):
    """Display consumption and deployment options"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🚀 Consumption Options**")
        if consumption_options:
            for i, option in enumerate(consumption_options):
                bg_color = "#f8fafc" if i % 2 == 0 else "#ffffff"
                option_display = str(option).replace('_', ' ').title()
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 12px; border-radius: 6px; margin-bottom: 8px; border: 1px solid #e2e8f0;">
                    <p style="margin: 0; color: #10b981; font-size: 13px; font-weight: bold;">🚀 {option_display}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No consumption options specified")
        
        st.markdown("**⚡ Inference Types**")
        if inference_types:
            for i, inf_type in enumerate(inference_types):
                bg_color = "#f8fafc" if i % 2 == 0 else "#ffffff"
                type_display = str(inf_type).replace('_', ' ').title()
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 12px; border-radius: 6px; margin-bottom: 8px; border: 1px solid #e2e8f0;">
                    <p style="margin: 0; color: #3b82f6; font-size: 13px; font-weight: bold;">⚡ {type_display}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No inference types specified")
    
    with col2:
        st.markdown("**🎯 Customization Options**")
        if customizations:
            # Enhanced display with better formatting and descriptions
            def format_customization_type(custom_type):
                """Format and add descriptions to customization types"""
                custom_str = str(custom_type).upper()

                descriptions = {
                    'FINE_TUNING': ('Fine-Tuning', 'Customize model with your own training data'),
                    'CONTINUED_PRE_TRAINING': ('Continued Pre-Training', 'Continue training the base model on domain-specific data'),
                    'DISTILLATION': ('Distillation', 'Create smaller, faster model from larger one'),
                    'FINETUNING': ('Fine-Tuning', 'Customize model with your own training data'),
                    'CPT': ('Continued Pre-Training', 'Continue training on domain-specific data'),
                }

                return descriptions.get(custom_str, (custom_str.replace('_', ' ').title(), 'Model customization available'))

            for i, custom in enumerate(customizations):
                bg_color = "#f8fafc" if i % 2 == 0 else "#ffffff"
                display_name, description = format_customization_type(custom)

                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 16px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #e2e8f0;">
                    <div style="margin: 0; color: #ef4444; font-size: 14px; font-weight: bold; margin-bottom: 4px;">
                        🎯 {display_name}
                    </div>
                    <div style="margin: 0; color: #6b7280; font-size: 12px; line-height: 1.4;">
                        {description}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Add informational note about customization
            st.markdown("""
            <div style="background: linear-gradient(90deg, #fef3c7 0%, #fde68a 100%); padding: 12px; border-radius: 8px; margin-top: 12px; border: 1px solid #f59e0b;">
                <p style="margin: 0; color: #92400e; font-size: 12px;">
                    💡 <strong>Note:</strong> Customization options may require additional setup and incur separate costs.
                    Check AWS Bedrock pricing for customization-specific charges.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No customization options available for this model")


def display_cross_region_inference(cross_region_info: Dict[str, Any]):
    """Display cross-region inference information with geographical grouping"""
    # Ensure cross_region_info is a dictionary (handle NaN/float values)
    if not isinstance(cross_region_info, dict):
        cross_region_info = {}

    supported = cross_region_info.get('supported', False)
    profiles_count = cross_region_info.get('profiles_count', 0)
    source_regions = cross_region_info.get('source_regions', [])
    destination_regions = cross_region_info.get('destination_regions', [])
    profiles = cross_region_info.get('profiles', [])
    
    # Display summary metrics in 2x2 layout
    # Main status
    status_color = '#10b981' if supported else '#ef4444'
    status_text = '✅ Supported' if supported else '❌ Not Supported'

    # First row: Status and Profiles
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #e2e8f0;">
            <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px;">Status</h5>
            <p style="margin: 0; color: {status_color}; font-size: 14px; font-weight: bold;">{status_text}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #e2e8f0;">
            <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px;">Profiles</h5>
            <p style="margin: 0; color: #3b82f6; font-size: 14px; font-weight: bold;">{profiles_count}</p>
        </div>
        """, unsafe_allow_html=True)

    # Second row: Source Regions and Destination Regions
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"""
        <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #e2e8f0;">
            <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px;">Source Regions</h5>
            <p style="margin: 0; color: #f59e0b; font-size: 14px; font-weight: bold;">{len(source_regions)}</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #e2e8f0;">
            <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px;">Destination Regions</h5>
            <p style="margin: 0; color: #8b5cf6; font-size: 14px; font-weight: bold;">{len(destination_regions)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    if supported and profiles_count > 0:
        # Display CRIS inference endpoints/profiles - make this more prominent
        if profiles:
            st.markdown("---")
            st.markdown("**🌐 CRIS (Cross-Region Inference Service) Endpoints**")

            # Group profiles by source region geography for better organization
            profile_geo_groups = {}
            for profile in profiles:
                source_region = profile.get('source_region', '')
                if source_region:
                    geo = map_region_to_geography(source_region)
                    if geo not in profile_geo_groups:
                        profile_geo_groups[geo] = []
                    profile_geo_groups[geo].append(profile)

            # Display profiles grouped by geography with endpoint focus
            for geo_name, geo_profiles in profile_geo_groups.items():
                # Get unique profiles by profile_id + source_region combination to show all different source regions
                unique_profiles = {}
                for profile in geo_profiles:
                    profile_id = profile.get('profile_id', '')
                    source_region = profile.get('source_region', '')
                    # Create unique key using profile_id + source_region to show all regional variations
                    unique_key = f"{profile_id}|{source_region}"
                    if unique_key not in unique_profiles:
                        unique_profiles[unique_key] = profile

                with st.expander(f"{get_geo_icon(geo_name)} {geo_name} Inference Endpoints ({len(unique_profiles)} available)", expanded=False):
                    for i, (unique_key, profile) in enumerate(unique_profiles.items()):
                        display_cross_region_profile(profile, i % 2 == 0)
        else:
            st.markdown("---")
            st.markdown("**🌐 CRIS (Cross-Region Inference Service) Endpoints**")
            st.warning("""
            ⚠️ **No CRIS endpoints found** for this model in the current data collection.

            This could mean:
            - The model doesn't support cross-region inference
            - No inference profiles are configured yet for your account
            - The data collection needs to be refreshed to capture recent profile changes
            """)
    
    elif not supported:
        st.info("💡 Cross-region inference is not available for this model. The model can only be used in its native regions.")


def display_cross_region_profile(profile: Dict[str, Any], is_even: bool):
    """Display an individual cross-region inference profile"""
    try:
        profile_id = profile.get('profile_id', 'Unknown')
        profile_name = profile.get('profile_name', 'Unknown Profile')
        profile_type = profile.get('type', 'Unknown')
        status = profile.get('status', 'Unknown')
        source_region = profile.get('source_region', 'Unknown')
        description = profile.get('description', 'No description available')
        created_at = profile.get('created_at', 'Unknown')
        updated_at = profile.get('updated_at', 'Unknown')
        
        # Background color alternation for readability
        bg_color = "#f8fafc" if is_even else "#ffffff"
        
        # Status color coding
        status_color = '#10b981' if status == 'ACTIVE' else '#6b7280'
        
        # Type color coding
        type_color = '#3b82f6' if profile_type == 'SYSTEM_DEFINED' else '#f59e0b'
        
        # Format dates
        try:
            if created_at and created_at != 'Unknown':
                created_display = created_at.split('T')[0] if 'T' in created_at else created_at.split(' ')[0]
            else:
                created_display = 'Unknown'
        except:
            created_display = 'Unknown'
        
        # Get source region display name
        source_region_name = get_region_display_name(source_region) if source_region != 'Unknown' else 'Unknown'
        
        # Create the profile card
        st.markdown(f"""
        <div style="background-color: {bg_color}; 
                    padding: 15px; 
                    border-radius: 8px; 
                    margin-bottom: 10px;
                    border: 1px solid #e2e8f0;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">
                <div style="flex: 1;">
                    <h5 style="margin: 0 0 4px 0; color: #1f2937; font-size: 14px; font-weight: bold;">
                        {profile_name}
                    </h5>
                    <p style="margin: 0 0 4px 0; color: #6b7280; font-size: 11px; font-family: monospace;">
                        ID: {profile_id}
                    </p>
                    <div style="display: flex; gap: 8px; margin-top: 6px;">
                        <span style="background-color: {status_color}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold;">
                            {status}
                        </span>
                    </div>
                </div>
                <div style="text-align: right; margin-left: 15px;">
                    <div style="color: #1f2937; font-size: 12px; font-weight: bold; margin-bottom: 4px;">
                        📍 {source_region_name} ({source_region})
                    </div>
                </div>
            </div>
            <div style="border-top: 1px solid #e2e8f0; padding-top: 10px;">
                <p style="margin: 0 0 6px 0; color: #4b5563; font-size: 12px; line-height: 1.4;">
                    {description}
                </p>
                <p style="margin: 0; color: #9ca3af; font-size: 10px;">
                    Created: {created_display}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        # Fallback display if there's any error
        st.error(f"❌ Error displaying profile: {str(e)}")


def display_batch_inference_support(batch_info: Dict[str, Any]):
    """Display batch inference support information"""
    # Ensure batch_info is a dictionary (handle NaN/float values)
    if not isinstance(batch_info, dict):
        batch_info = {}

    supported = batch_info.get('supported', False)
    supported_regions = batch_info.get('supported_regions', [])
    total_regions = batch_info.get('total_regions', 0)
    coverage_percentage = batch_info.get('coverage_percentage', 0)
    
    # Follow language support styling - metrics in left column
    col1, col2 = st.columns(2)
    
    # Main status
    status_color = '#10b981' if supported else '#ef4444'
    status_text = '✅ Supported' if supported else '❌ Not Supported'
    
    with col1:
        st.markdown(f"""
        <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #e2e8f0;">
            <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px;">Batch Inference Status</h5>
            <p style="margin: 0; color: {status_color}; font-size: 14px; font-weight: bold;">{status_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if supported and supported_regions:
            st.markdown(f"""
            <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #e2e8f0;">
                <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px;">Supported Regions</h5>
                <p style="margin: 0; color: #3b82f6; font-size: 14px; font-weight: bold;">{total_regions}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #e2e8f0;">
                <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px;">Coverage</h5>
                <p style="margin: 0; color: #10b981; font-size: 14px; font-weight: bold;">{coverage_percentage:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if supported and supported_regions:
            st.markdown("**📦 Batch-Enabled Regions**")
            # Group regions by geography for better organization
            geo_groups = {}
            for region in supported_regions:
                geo = map_region_to_geography(region)
                if geo not in geo_groups:
                    geo_groups[geo] = []
                geo_groups[geo].append(region)
            
            for geo, regions in geo_groups.items():
                st.markdown(f"**{get_geo_icon(geo)} {geo}**")
                for i, region in enumerate(regions):
                    bg_color = "#f8fafc" if i % 2 == 0 else "#ffffff"
                    region_name = get_region_display_name(region)
                    st.markdown(f"""
                    <div style="background-color: {bg_color}; padding: 8px; border-radius: 6px; margin-bottom: 5px; border: 1px solid #e2e8f0;">
                        <p style="margin: 0; color: #6b7280; font-size: 12px;">📦 {region_name} ({region})</p>
                    </div>
                    """, unsafe_allow_html=True)
        elif not supported:
            st.info("💡 Batch inference is not available for this model. Use on-demand or provisioned throughput options instead.")


def display_regional_availability(regions: List[str]):
    """Display regional availability information with creative geographical grouping"""
    if not regions:
        st.warning("No regional availability information available for this model.")
        return
    
    # Group regions by geography for better organization
    geo_groups = {}
    for region in regions:
        geo = map_region_to_geography(region)
        if geo not in geo_groups:
            geo_groups[geo] = []
        geo_groups[geo].append(region)
    
    # Display summary metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #e2e8f0;">
            <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px;">Total Regions</h5>
            <p style="margin: 0; color: #10b981; font-size: 18px; font-weight: bold;">{len(regions)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #e2e8f0;">
            <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px;">Geographies</h5>
            <p style="margin: 0; color: #3b82f6; font-size: 18px; font-weight: bold;">{len(geo_groups)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display regions grouped by geography
    for geo_name, geo_regions in geo_groups.items():
        st.markdown(f"**{get_geo_icon(geo_name)} {geo_name} ({len(geo_regions)} regions)**")
        
        # Create a grid layout for regions within each geography
        cols_per_row = 3
        for i in range(0, len(geo_regions), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, region in enumerate(geo_regions[i:i+cols_per_row]):
                with cols[j]:
                    region_name = get_region_display_name(region)
                    bg_color = "#f8fafc" if (i + j) % 2 == 0 else "#ffffff"
                    st.markdown(f"""
                    <div style="background-color: {bg_color}; padding: 12px; border-radius: 8px; margin-bottom: 8px; border: 1px solid #e2e8f0; text-align: center;">
                        <div style="color: #1f2937; font-size: 13px; font-weight: bold; margin-bottom: 4px;">
                            {region_name}
                        </div>
                        <div style="color: #6b7280; font-size: 11px; font-family: monospace;">
                            {region}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("")  # Add some spacing between geographies


def display_converse_data_analysis(model: Dict[str, Any]):
    """Display enhanced analysis from converse_data"""

    # Extract converse data
    context_window = get_converse_context_window(model)
    max_output_tokens = get_converse_max_output_tokens(model)
    size_category = get_converse_size_category(model)
    function_calling = get_converse_function_calling(model)
    recommendation = get_converse_recommendation(model)

    # Check if we have converse data
    has_converse_data = model.get('converse_data') is not None

    if not has_converse_data:
        st.info("Enhanced analysis data is not available for this model.")
        return

    st.markdown("**Enhanced model analysis with context window and token specifications from community-verified data.**")
    st.markdown("---")

    # Context & Output Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📏 Context Analysis**")

        # Context window with size badge
        if context_window > 0:
            category = size_category.get('category', 'Unknown')
            color = size_category.get('color', '#6B7280')
            tier = size_category.get('tier', 0)

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); padding: 15px; border-radius: 10px; margin: 10px 0;">
                <h4 style="color: white; margin: 0 0 8px 0;">Context Window</h4>
                <div style="color: white; font-size: 24px; font-weight: bold; margin: 5px 0;">{context_window:,} tokens</div>
                <div style="background: rgba(255,255,255,0.2); color: white; padding: 4px 8px; border-radius: 6px; font-size: 12px; display: inline-block;">
                    {category} (Tier {tier})
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Context window not specified")

        # Max output tokens
        if max_output_tokens > 0:
            st.metric("Max Output Tokens", f"{max_output_tokens:,}")
        else:
            st.metric("Max Output Tokens", "Not specified")

    with col2:
        # Empty column to maintain layout
        pass

