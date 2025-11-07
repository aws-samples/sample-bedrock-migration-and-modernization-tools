"""
Filtering and sorting components for the Amazon Bedrock Expert application.
"""
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional


def get_all_available_regions(df: pd.DataFrame) -> List[str]:
    """
    Get all available regions from model data, sorted alphabetically.

    Args:
        df: DataFrame containing model data

    Returns:
        List of AWS region codes sorted alphabetically
    """
    # Use metadata as the primary source since individual models only show 3 regions each
    # but the metadata shows all 20 regions where models are actually available
    try:
        from models.new_model_repository import NewModelRepository
        repo = NewModelRepository()
        metadata = repo.get_metadata()
        metadata_regions = metadata.get('regions', [])

        if isinstance(metadata_regions, list) and len(metadata_regions) > 0:
            # Use metadata regions (all 20 regions)
            all_regions = set(metadata_regions)
        else:
            raise Exception("No metadata regions found")

    except Exception as e:
        # Fallback: Extract from individual models if metadata fails
        all_regions = set()
        for _, row in df.iterrows():
            # Check multiple possible region fields
            regions = row.get('regions', [])
            available_regions = row.get('available_regions', [])

            if isinstance(regions, list):
                all_regions.update(regions)
            if isinstance(available_regions, list):
                all_regions.update(available_regions)

        # If still no regions, use common defaults
        if not all_regions:
            all_regions = {'us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'}

    # Convert to sorted list, prioritizing common regions first
    common_regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']
    other_regions = sorted([r for r in all_regions if r not in common_regions])

    return [r for r in common_regions if r in all_regions] + other_regions


def get_geo_display_name(geo: str) -> str:
    """
    Get display name for geographic area.
    
    Args:
        geo: Geographic area code (US, EU, AP, CA, Global)
        
    Returns:
        Display name for the geographic area
    """
    geo_display_mapping = {
        'US': 'US (North America)',
        'EU': 'EU (Europe)',
        'AP': 'AP (Asia Pacific)',
        'CA': 'CA (Canada)',
        'Global': 'Global'
    }
    
    return geo_display_mapping.get(geo, geo)


def get_region_display_name(region: str) -> str:
    """
    Get human-readable display name for AWS region.
    
    Args:
        region: AWS region code
        
    Returns:
        Display name for the region
    """
    region_display_mapping = {
        # US Regions
        'us-east-1': 'N. Virginia (us-east-1)',
        'us-east-2': 'Ohio (us-east-2)',
        'us-west-1': 'N. California (us-west-1)',
        'us-west-2': 'Oregon (us-west-2)',

        # Europe Regions
        'eu-west-1': 'Ireland (eu-west-1)',
        'eu-west-2': 'London (eu-west-2)',
        'eu-west-3': 'Paris (eu-west-3)',
        'eu-central-1': 'Frankfurt (eu-central-1)',
        'eu-central-2': 'Zurich (eu-central-2)',
        'eu-north-1': 'Stockholm (eu-north-1)',
        'eu-south-1': 'Milan (eu-south-1)',
        'eu-south-2': 'Spain (eu-south-2)',

        # Asia Pacific Regions
        'ap-northeast-1': 'Tokyo (ap-northeast-1)',
        'ap-northeast-2': 'Seoul (ap-northeast-2)',
        'ap-northeast-3': 'Osaka (ap-northeast-3)',
        'ap-southeast-1': 'Singapore (ap-southeast-1)',
        'ap-southeast-2': 'Sydney (ap-southeast-2)',
        'ap-south-1': 'Mumbai (ap-south-1)',
        'ap-east-1': 'Hong Kong (ap-east-1)',

        # Canada Regions
        'ca-central-1': 'Canada Central (ca-central-1)',

        # South America Regions
        'sa-east-1': 'São Paulo (sa-east-1)'
    }
    
    return region_display_mapping.get(region, f'{region.upper()} ({region})')


def clean_filter_state_for_data(df: pd.DataFrame):
    """
    Clean filter state to ensure all values are valid for the current dataset.
    This prevents errors when switching between different views (e.g., favorites).
    
    Args:
        df: Current DataFrame to validate against
    """
    if 'filter_state' not in st.session_state:
        return
    
    # Get current valid options from the data
    providers = sorted(df['provider'].unique().tolist()) if not df.empty else []
    
    # Clean capabilities
    all_capabilities = []
    if not df.empty:
        for caps in df['capabilities']:
            if isinstance(caps, list):
                all_capabilities.extend(caps)
    unique_capabilities = sorted(list(set(all_capabilities)))
    
    # Clean regions
    all_regions = []
    if not df.empty:
        for regions in df['regions']:
            if isinstance(regions, list):
                all_regions.extend(regions)
    unique_regions = sorted(list(set(all_regions)))
    
    # Clean modalities
    all_modalities = []
    if not df.empty:
        for modalities in df['input_modalities']:
            if isinstance(modalities, list):
                all_modalities.extend(modalities)
        for modalities in df['output_modalities']:
            if isinstance(modalities, list):
                all_modalities.extend(modalities)
    unique_modalities = sorted(list(set(all_modalities)))
    
    # Clean use cases
    all_use_cases = []
    if not df.empty:
        for use_cases in df['use_cases']:
            if isinstance(use_cases, list):
                all_use_cases.extend(use_cases)
    unique_use_cases = sorted(list(set(all_use_cases)))
    
    # Clean languages
    all_languages = []
    if not df.empty:
        for languages in df['languages']:
            if isinstance(languages, list):
                all_languages.extend(languages)
    unique_languages = sorted(list(set(all_languages)))
    
    # Update filter state with only valid values
    filter_state = st.session_state.filter_state
    
    filter_state['providers'] = [p for p in filter_state.get('providers', []) if p in providers]
    filter_state['capabilities'] = [c for c in filter_state.get('capabilities', []) if c in unique_capabilities]
    # Note: geo_region is handled as a selectbox, no cleanup needed
    # Clean up modality filter (single select)
    current_modality = filter_state.get('modality_filter', 'All Modalities')
    if current_modality not in unique_modalities and current_modality != 'All Modalities':
        filter_state['modality_filter'] = 'All Modalities'
    filter_state['use_cases'] = [uc for uc in filter_state.get('use_cases', []) if uc in unique_use_cases]
    filter_state['languages'] = [lang for lang in filter_state.get('languages', []) if lang in unique_languages]


def show_filter_controls(df: pd.DataFrame, key_suffix: str = "") -> Dict[str, Any]:
    """
    Display filter controls for the model explorer.
    
    Args:
        df: DataFrame containing model data
        
    Returns:
        Dictionary containing selected filter values
    """
    # Clean filter state to ensure all values are valid for current data
    clean_filter_state_for_data(df)
    
    # Initialize filter state in session state if not exists
    if 'filter_state' not in st.session_state:
        st.session_state.filter_state = {
            'providers': [],  # Empty selection shows all providers by default
            'capabilities': [],
            'geo_region': 'All Regions',
            'model_status': 'All Status',
            'use_cases': [],
            'modality_filter': 'All Modalities',
            'streaming': 'All Models',
            'search_query': '',
            'consumption_filter': 'All Models',
            'customization_filter': 'All Models',
            'cris_support': 'All Models',
            'languages': [],
            'context_filter': 'All Models',
            'primary_geo': 'US',
            'primary_region': 'us-east-1'
        }
    
    # Create filter container with gradient background
    # Add custom CSS for better filter styling
    st.markdown("""
    <style>
    .filter-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    .filter-header {
        color: #495057;
        font-weight: 600;
        margin-bottom: 0.5rem;
        padding-bottom: 0.25rem;
        border-bottom: 2px solid #007bff;
    }
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 6px;
    }
    .stMultiSelect > div > div {
        background-color: white;
        border-radius: 6px;
    }
    .stTextInput > div > div > input {
        background-color: white;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    .search-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .filter-container {
        background: linear-gradient(135deg, #f6f8fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add CSS for vertical alignment of filter controls
    st.markdown("""
    <style>
    /* Align all filter controls vertically */
    .filter-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Ensure consistent height for all input elements */
    .stTextInput > div > div > input {
        height: 2.5rem;
        padding: 0.5rem 1rem;
    }
    
    .stSelectbox > div > div {
        min-height: 2.5rem;
    }
    
    /* Align toggle button properly - target the actual toggle component */
    .stToggle {
        display: flex;
        align-items: center;
        height: 2.5rem;
        margin-top: 1.5rem;
    }
    
    .stToggle > label {
        display: flex;
        align-items: center;
        height: 2.5rem;
        margin: 0;
        padding-top: 0;
    }
    
    /* Custom styling for the toggle container */
    .toggle-container {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        height: 2.5rem;
        margin-top: 1.5rem;
        padding-top: 0;
    }
    
    /* Override Streamlit's default toggle margins */
    div[data-testid="stToggle"] {
        margin-top: 1.5rem !important;
        display: flex;
        align-items: center;
    }
    
    div[data-testid="stToggle"] > label {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        display: flex;
        align-items: center;
        height: 2.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Search bar with region selector
    search_col, region_col = st.columns([5, 1])
    
    # Add CSS to align Primary Region selector using its specific key
    st.markdown("""
    <style>
    /* Target Primary Region selector by its specific key class */
    .st-key-primary_region_selector_explorer {
        margin-top: -1.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with search_col:
        search_query = st.text_input(
            "Search Models",
            value=st.session_state.filter_state.get('search_query', ''),
            placeholder="🔍 Search by model name, provider, capabilities, or description...",
            help="Search across all model information including names, providers, capabilities, and descriptions",
            key=f"model_search_input{key_suffix}",
            label_visibility="collapsed"
        )
        st.session_state.filter_state['search_query'] = search_query
    with region_col:
        # Primary Region selector with all available regions
        region_options = get_all_available_regions(df)

        # Initialize primary region if not exists
        if 'primary_region' not in st.session_state.filter_state:
            st.session_state.filter_state['primary_region'] = region_options[0] if region_options else 'us-east-1'

        # Ensure current selection is still valid
        current_region = st.session_state.filter_state.get('primary_region', 'us-east-1')
        if current_region not in region_options and region_options:
            current_region = region_options[0]
            st.session_state.filter_state['primary_region'] = current_region

        selected_primary_region = st.selectbox(
            "📍 Primary Region",
            options=region_options,
            index=region_options.index(current_region) if current_region in region_options else 0,
            format_func=get_region_display_name,
            help="Select the AWS region for displaying pricing and quotas. Model cards will show pricing and quota information from this region when available.",
            key=f"primary_region_selector{key_suffix}"
        )
        st.session_state.filter_state['primary_region'] = selected_primary_region
    
    # Show search results count if there's a search query
    if search_query:
        st.markdown(f"*Searching for: **{search_query}***")
    
    # Advanced filters in collapsed section
    with st.expander("🔧 Advanced Filters", expanded=False):
        # Create filter layout with 4 columns
        col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### 🏢 **Provider & Location**")
        
        # Provider filter with multiselect
        providers = sorted(df['provider'].unique().tolist()) if not df.empty else []
        
        # Get current provider selection (empty by default to show all)
        current_providers = st.session_state.filter_state.get('providers', [])
        # Keep only valid providers from current selection
        valid_providers = [p for p in current_providers if p in providers]
        
        selected_providers = st.multiselect(
            "🏢 Providers",
            providers,
            default=valid_providers,  # Empty selection by default
            help="Filter by model providers (Amazon, Anthropic, etc.). Leave empty to show all providers.",
            key=f"providers_multiselect{key_suffix}"
        )
        st.session_state.filter_state['providers'] = selected_providers
        
        # Geo-based regions filter (moved below providers)
        geo_regions = ['All Regions', 'US (North America)', 'EU (Europe)', 'AP (Asia Pacific)', 'CA (Canada)']
        
        selected_geo = st.selectbox(
            "🌍 Geographic Regions",
            options=geo_regions,
            index=geo_regions.index(st.session_state.filter_state.get('geo_region', 'All Regions')),
            help="Filter by geographic regions where models are available",
            key=f"geo_region_selector{key_suffix}"
        )
        st.session_state.filter_state['geo_region'] = selected_geo
        
        # Model status filter (Active/Legacy)
        status_options = ['All Status', 'ACTIVE', 'LEGACY']
        
        selected_status = st.selectbox(
            "📊 Model Status",
            options=status_options,
            index=status_options.index(st.session_state.filter_state.get('model_status', 'All Status')),
            help="Filter by model lifecycle status (Active or Legacy)",
            key=f"model_status_selector{key_suffix}"
        )
        st.session_state.filter_state['model_status'] = selected_status
    
    with col2:
        st.markdown("#### 💳 **Consumption & Features**")
        
        # Consumption Options filter (selectbox)
        consumption_options = ['All Models', 'On-Demand Only', 'Provisioned Throughput', 'Batch Processing']
        
        consumption_filter = st.selectbox(
            "💳 Consumption Options",
            options=consumption_options,
            index=consumption_options.index(st.session_state.filter_state.get('consumption_filter', 'All Models')),
            help="Filter by available consumption and pricing options",
            key=f"consumption_filter{key_suffix}"
        )
        st.session_state.filter_state['consumption_filter'] = consumption_filter
        
        # Cross-Region Inference filter
        cris_filter = st.selectbox(
            "🌐 Cross-Region Inference",
            ["All Models", "CRIS Supported", "CRIS Not Supported"],
            index=["All Models", "CRIS Supported", "CRIS Not Supported"].index(st.session_state.filter_state.get('cris_support', 'All Models')),
            help="Filter by Cross-Region Inference support",
            key=f"cris_filter{key_suffix}"
        )
        st.session_state.filter_state['cris_support'] = cris_filter
        
        # Streaming support filter
        streaming_options = ["All Models", "Streaming Supported", "No Streaming"]
        streaming_filter = st.selectbox(
            "📡 Streaming Support",
            options=streaming_options,
            index=streaming_options.index(st.session_state.filter_state.get('streaming', 'All Models')) if st.session_state.filter_state.get('streaming', 'All Models') in streaming_options else 0,
            help="Filter by streaming capability",
            key=f"streaming_filter{key_suffix}"
        )
        st.session_state.filter_state['streaming'] = streaming_filter
    
    with col3:
        st.markdown("#### 🎯 **Use Cases & Content**")
        
        # Use case filter
        all_use_cases = []
        if not df.empty:
            for use_cases in df['use_cases']:
                if isinstance(use_cases, list):
                    all_use_cases.extend(use_cases)
        unique_use_cases = sorted(list(set(all_use_cases)))
        
        if unique_use_cases:
            current_use_cases = st.session_state.filter_state.get('use_cases', [])
            valid_use_case_defaults = [uc for uc in current_use_cases if uc in unique_use_cases] if current_use_cases else []
            
            selected_use_cases = st.multiselect(
                "🎯 Use Cases",
                unique_use_cases,
                default=valid_use_case_defaults,
                help="Filter by intended use cases",
                key=f"use_cases_multiselect{key_suffix}"
            )
            st.session_state.filter_state['use_cases'] = selected_use_cases
        
        # Modality filter
        all_modalities = []
        if not df.empty:
            for modalities in df['input_modalities']:
                if isinstance(modalities, list):
                    all_modalities.extend(modalities)
            for modalities in df['output_modalities']:
                if isinstance(modalities, list):
                    all_modalities.extend(modalities)
        unique_modalities = sorted(list(set(all_modalities)))
        
        if unique_modalities:
            # Add "All Modalities" option at the beginning
            modality_options = ['All Modalities'] + unique_modalities

            current_modality = st.session_state.filter_state.get('modality_filter', 'All Modalities')
            # Validate current selection exists in options
            if current_modality not in modality_options:
                current_modality = 'All Modalities'

            selected_modality = st.selectbox(
                "📊 Modalities",
                options=modality_options,
                index=modality_options.index(current_modality),
                help="Filter by input/output modalities (text, image, etc.)",
                key=f"modality_selectbox{key_suffix}"
            )
            st.session_state.filter_state['modality_filter'] = selected_modality
        
        # Capabilities filter
        all_capabilities = []
        if not df.empty:
            for caps in df['capabilities']:
                if isinstance(caps, list):
                    all_capabilities.extend(caps)
        unique_capabilities = sorted(list(set(all_capabilities)))
        
        if unique_capabilities:
            valid_capability_defaults = [c for c in st.session_state.filter_state.get('capabilities', []) if c in unique_capabilities]
            
            selected_capabilities = st.multiselect(
                "🔧 Capabilities",
                unique_capabilities,
                default=valid_capability_defaults if valid_capability_defaults else [],
                help="Filter by model capabilities (text, image, etc.)",
                key=f"capabilities_multiselect{key_suffix}"
            )
            st.session_state.filter_state['capabilities'] = selected_capabilities
    
    with col4:
        st.markdown("#### 🧠 **Model Capabilities**")
        
        # Customization Options filter
        customization_options = ['All Models', 'Fine-Tuning Available', 'Continued Pre-Training', 'Distillation Available', 'No Customization']
        
        customization_filter = st.selectbox(
            "🎯 Customization Options",
            options=customization_options,
            index=customization_options.index(st.session_state.filter_state.get('customization_filter', 'All Models')),
            help="Filter by model customization capabilities (fine-tuning, continued pre-training, distillation)",
            key=f"customization_filter{key_suffix}"
        )
        st.session_state.filter_state['customization_filter'] = customization_filter
        
        # Language filter
        all_languages = []
        if not df.empty:
            for languages in df['languages']:
                if isinstance(languages, list):
                    all_languages.extend(languages)
        unique_languages = sorted(list(set(all_languages)))
        
        if unique_languages:
            current_languages = st.session_state.filter_state.get('languages', [])
            valid_language_defaults = [lang for lang in current_languages if lang in unique_languages] if current_languages else []
            
            selected_languages = st.multiselect(
                "🗣️ Languages",
                unique_languages,
                default=valid_language_defaults,
                help="Filter by supported languages",
                key=f"languages_multiselect{key_suffix}"
            )
            st.session_state.filter_state['languages'] = selected_languages
        
        # Context Window filter - aligned with size categories
        context_options = [
            'All Models',
            'Small (< 32K tokens)',
            'Medium (32K - 128K tokens)',
            'Large (128K - 500K tokens)',
            'XL (> 500K tokens)'
        ]
        
        context_filter = st.selectbox(
            "📏 Context Window",
            options=context_options,
            index=context_options.index(st.session_state.filter_state.get('context_filter', 'All Models')),
            help="Filter by context window size (matches model size categories)",
            key=f"context_filter{key_suffix}"
        )
        st.session_state.filter_state['context_filter'] = context_filter
    
    return st.session_state.filter_state

def show_sort_controls() -> Tuple[str, bool]:
    """
    Display sorting controls for the model explorer.
    
    Returns:
        Tuple containing sort column and ascending flag
    """
    # Define sort options
    sort_options = {
        "Provider (A-Z)": ("provider", True),
        "Provider (Z-A)": ("provider", False),
        "Name (A-Z)": ("name", True),
        "Name (Z-A)": ("name", False),
        "Context Window (High-Low)": ("context_window", False),
        "Context Window (Low-High)": ("context_window", True),
        "Input Price (Low-High)": ("input_price", True),
        "Input Price (High-Low)": ("input_price", False),
        "Output Price (Low-High)": ("output_price", True),
        "Output Price (High-Low)": ("output_price", False),
        "Regions (Most-Least)": ("region_count", False),
        "Regions (Least-Most)": ("region_count", True),
        "Release Date (Newest First)": ("release_date", False),
        "Release Date (Oldest First)": ("release_date", True)
    }
    
    # Initialize sort state in session state if not exists
    if 'sort_option' not in st.session_state:
        st.session_state.sort_option = "Provider (A-Z)"
    
    # Create sort control
    sort_by = st.selectbox(
        "Sort by:",
        options=list(sort_options.keys()),
        index=list(sort_options.keys()).index(st.session_state.sort_option),
        key="sort_selector"
    )
    
    # Update session state
    st.session_state.sort_option = sort_by
    
    # Return sort column and direction
    return sort_options[sort_by]


def apply_filters(df: pd.DataFrame, filter_state: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply filters to the model DataFrame.
    
    Args:
        df: DataFrame containing model data
        filter_state: Dictionary containing filter values
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    # Handle None filter_state
    if filter_state is None:
        return df
    
    filtered_df = df.copy()
    
    # Apply search query filter
    if filter_state.get('search_query'):
        query = filter_state['search_query'].lower()
        filtered_df = filtered_df[
            filtered_df['model_name'].str.lower().str.contains(query, na=False) |
            filtered_df['provider'].str.lower().str.contains(query, na=False) |
            filtered_df['description'].str.lower().str.contains(query, na=False) |
            filtered_df['model_id'].str.lower().str.contains(query, na=False) |
            filtered_df['capabilities'].apply(lambda x: any(query in cap.lower() for cap in x if isinstance(x, list)))
        ]
    
    # Filter by providers
    if filter_state.get('providers', []):
        filtered_df = filtered_df[filtered_df['provider'].isin(filter_state.get('providers', []))]
    
    # Filter by geo regions
    geo_region = filter_state.get('geo_region', 'All Regions')
    if geo_region != 'All Regions':
        if geo_region == 'US (North America)':
            geo_filter = filtered_df['regions'].apply(
                lambda x: isinstance(x, list) and any('us-' in region for region in x)
            )
        elif geo_region == 'EU (Europe)':
            geo_filter = filtered_df['regions'].apply(
                lambda x: isinstance(x, list) and any('eu-' in region for region in x)
            )
        elif geo_region == 'AP (Asia Pacific)':
            geo_filter = filtered_df['regions'].apply(
                lambda x: isinstance(x, list) and any('ap-' in region for region in x)
            )
        elif geo_region == 'CA (Canada)':
            geo_filter = filtered_df['regions'].apply(
                lambda x: isinstance(x, list) and any('ca-' in region for region in x)
            )
        else:
            geo_filter = pd.Series([True] * len(filtered_df))
        
        filtered_df = filtered_df[geo_filter]
    
    # Filter by model status (Active/Legacy)
    model_status = filter_state.get('model_status', 'All Status')
    if model_status != 'All Status':
        def get_model_status(row):
            # Try multiple possible sources for model lifecycle status

            # Option 1: Direct status field
            if 'status' in row:
                status = row['status']
            # Option 2: Model lifecycle field (dict)
            elif 'model_lifecycle' in row and isinstance(row['model_lifecycle'], dict):
                status = row['model_lifecycle'].get('status', 'ACTIVE')
            # Option 3: Look in model_id or name for legacy indicators
            elif any(field in row for field in ['model_id', 'name', 'model_name']):
                model_name = str(row.get('model_id', '') or row.get('name', '') or row.get('model_name', '')).lower()
                # Check for legacy indicators in model name/id
                if any(legacy_indicator in model_name for legacy_indicator in ['legacy', 'deprecated', 'old', 'v1', 'v0']):
                    status = 'LEGACY'
                else:
                    status = 'ACTIVE'
            else:
                status = 'ACTIVE'  # Default to Active

            # Normalize status to uppercase for comparison
            status = str(status).upper()

            # Map AWS status to filter values
            if status in ['ACTIVE', 'AVAILABLE']:
                return 'ACTIVE'
            elif status in ['LEGACY', 'DEPRECATED', 'DISCONTINUED']:
                return 'LEGACY'
            else:
                return 'ACTIVE'  # Default to Active for unknown status
        
        status_filter = filtered_df.apply(get_model_status, axis=1) == model_status
        filtered_df = filtered_df[status_filter]
    

    
    # Filter by capabilities
    if filter_state.get('capabilities', []):
        filtered_df = filtered_df[filtered_df['capabilities'].apply(
            lambda x: isinstance(x, list) and any(cap in x for cap in filter_state.get('capabilities', []))
        )]
    
    # Filter by use cases
    if filter_state.get('use_cases', []):
        filtered_df = filtered_df[filtered_df['use_cases'].apply(
            lambda x: isinstance(x, list) and any(use_case in x for use_case in filter_state.get('use_cases', []))
        )]
    
    # Filter by modalities (single select) - Enhanced with safe column checking
    modality_filter = filter_state.get('modality_filter', 'All Modalities')
    if modality_filter != 'All Modalities':

        def has_modality(row):
            """Safely check if row has the selected modality in input or output"""
            # Check input modalities
            input_modalities = []
            for col_name in ['input_modalities', 'model_input_modalities']:
                if col_name in row and isinstance(row[col_name], list):
                    input_modalities = row[col_name]
                    break

            # Check output modalities
            output_modalities = []
            for col_name in ['output_modalities', 'model_output_modalities']:
                if col_name in row and isinstance(row[col_name], list):
                    output_modalities = row[col_name]
                    break

            # Check if modality exists in either input or output
            return (modality_filter in input_modalities) or (modality_filter in output_modalities)

        # Apply the filter
        modality_mask = filtered_df.apply(has_modality, axis=1)
        filtered_df = filtered_df[modality_mask]
    
    # Filter by streaming support
    if filter_state.get('streaming') == "Streaming Supported":
        filtered_df = filtered_df[filtered_df['streaming_supported'] == True]
    elif filter_state.get('streaming') == "No Streaming":
        filtered_df = filtered_df[filtered_df['streaming_supported'] == False]
    
    # Filter by consumption options
    consumption_filter = filter_state.get('consumption_filter', 'All Models')
    if consumption_filter != 'All Models':
        if consumption_filter == 'On-Demand Only':
            filtered_df = filtered_df[filtered_df['consumption_options'].apply(
                lambda x: isinstance(x, list) and 'on_demand' in x
            )]
        elif consumption_filter == 'Provisioned Throughput':
            filtered_df = filtered_df[filtered_df['consumption_options'].apply(
                lambda x: isinstance(x, list) and 'provisioned_throughput' in x
            )]
        elif consumption_filter == 'Batch Processing':
            filtered_df = filtered_df[filtered_df['batch_inference_supported'].apply(
                lambda x: isinstance(x, dict) and x.get('supported', False) == True
            )]
    
    # Filter by customization options
    customization_filter = filter_state.get('customization_filter', 'All Models')
    if customization_filter != 'All Models':

        # Helper function to safely get customization data
        def get_customizations(row):
            """Safely extract customization data from different possible column names"""
            # Try different possible column names
            for col_name in ['customizations_supported', 'model_customizations_supported', 'customization_supported']:
                if col_name in row:
                    return row[col_name] if isinstance(row[col_name], list) else []

            # Fallback: check if customization data is nested
            if 'customization' in row and isinstance(row['customization'], dict):
                return row['customization'].get('customization_supported', [])

            return []  # Default empty list if no customization data found

        if customization_filter == 'Fine-Tuning Available':
            filtered_df = filtered_df[filtered_df.apply(
                lambda row: 'FINE_TUNING' in get_customizations(row), axis=1
            )]
        elif customization_filter == 'Continued Pre-Training':
            filtered_df = filtered_df[filtered_df.apply(
                lambda row: 'CONTINUED_PRE_TRAINING' in get_customizations(row), axis=1
            )]
        elif customization_filter == 'Distillation Available':
            filtered_df = filtered_df[filtered_df.apply(
                lambda row: 'DISTILLATION' in get_customizations(row), axis=1
            )]
        elif customization_filter == 'No Customization':
            filtered_df = filtered_df[filtered_df.apply(
                lambda row: len(get_customizations(row)) == 0, axis=1
            )]
    
    # Filter by languages
    if filter_state.get('languages', []):
        filtered_df = filtered_df[filtered_df['languages'].apply(
            lambda x: isinstance(x, list) and any(lang in x for lang in filter_state.get('languages', []))
        )]
    
    # Filter by context window - use converse_data context windows
    context_filter = filter_state.get('context_filter', 'All Models')
    if context_filter != 'All Models':
        # Helper function to get context window from converse_data or fallback
        def get_model_context_window(row):
            from ui.converse_data_helpers import get_converse_context_window
            return get_converse_context_window(row)

        # Apply context window ranges matching size categories
        if context_filter == 'Small (< 32K tokens)':
            mask = filtered_df.apply(lambda row: get_model_context_window(row) < 32000, axis=1)
            filtered_df = filtered_df[mask]
        elif context_filter == 'Medium (32K - 128K tokens)':
            mask = filtered_df.apply(lambda row: 32000 <= get_model_context_window(row) < 128000, axis=1)
            filtered_df = filtered_df[mask]
        elif context_filter == 'Large (128K - 500K tokens)':
            mask = filtered_df.apply(lambda row: 128000 <= get_model_context_window(row) < 500000, axis=1)
            filtered_df = filtered_df[mask]
        elif context_filter == 'XL (> 500K tokens)':
            mask = filtered_df.apply(lambda row: get_model_context_window(row) >= 500000, axis=1)
            filtered_df = filtered_df[mask]
    
    # Apply CRIS support filter
    if filter_state.get('cris_support', 'All Models') != 'All Models':
        if filter_state['cris_support'] == 'CRIS Supported':
            cris_mask = filtered_df['cross_region_inference'].apply(
                lambda x: isinstance(x, dict) and x.get('supported', False) == True
            )
        else:  # CRIS Not Supported
            cris_mask = filtered_df['cross_region_inference'].apply(
                lambda x: isinstance(x, dict) and x.get('supported', False) == False
            )
        filtered_df = filtered_df[cris_mask]
    
    return filtered_df


def apply_sorting(df: pd.DataFrame, sort_column: str, ascending: bool) -> pd.DataFrame:
    """
    Apply sorting to the model DataFrame.
    
    Args:
        df: DataFrame containing model data
        sort_column: Column to sort by
        ascending: Sort direction (True for ascending, False for descending)
        
    Returns:
        Sorted DataFrame
    """
    if df.empty:
        return df
    
    try:
        # Handle different sort columns
        if sort_column == "name":
            return df.sort_values('model_name', ascending=ascending)
        
        elif sort_column == "provider":
            return df.sort_values('provider', ascending=ascending)
        
        elif sort_column == "input_price":
            # Extract input price for sorting from comprehensive_pricing
            def get_input_price(row):
                # Try comprehensive_pricing first (new format)
                comprehensive_pricing = row.get('comprehensive_pricing', {})
                if isinstance(comprehensive_pricing, dict):
                    regions_data = comprehensive_pricing.get('regions', {})
                    # Try US regions first for consistent pricing
                    for region in ['us-east-1', 'us-west-2']:
                        if region in regions_data:
                            on_demand = regions_data[region].get('on_demand', {})
                            input_price = on_demand.get('input_tokens', 0)
                            if input_price > 0:
                                return input_price
                    # Try any region with pricing
                    for region_data in regions_data.values():
                        on_demand = region_data.get('on_demand', {})
                        input_price = on_demand.get('input_tokens', 0)
                        if input_price > 0:
                            return input_price
                
                # Fallback to old pricing format
                pricing = row.get('pricing', {})
                if isinstance(pricing, dict):
                    if 'input_tokens_per_1k' in pricing:
                        return pricing.get('input_tokens_per_1k', 0)
                    elif 'on_demand' in pricing and isinstance(pricing['on_demand'], dict):
                        return pricing['on_demand'].get('input_tokens', 0)
                return 0
            
            df_copy = df.copy()
            df_copy['sort_price'] = df_copy.apply(get_input_price, axis=1)
            return df_copy.sort_values('sort_price', ascending=ascending).drop('sort_price', axis=1)
        
        elif sort_column == "output_price":
            # Extract output price for sorting from comprehensive_pricing
            def get_output_price(row):
                # Try comprehensive_pricing first (new format)
                comprehensive_pricing = row.get('comprehensive_pricing', {})
                if isinstance(comprehensive_pricing, dict):
                    regions_data = comprehensive_pricing.get('regions', {})
                    # Try US regions first for consistent pricing
                    for region in ['us-east-1', 'us-west-2']:
                        if region in regions_data:
                            on_demand = regions_data[region].get('on_demand', {})
                            output_price = on_demand.get('output_tokens', 0)
                            if output_price > 0:
                                return output_price
                    # Try any region with pricing
                    for region_data in regions_data.values():
                        on_demand = region_data.get('on_demand', {})
                        output_price = on_demand.get('output_tokens', 0)
                        if output_price > 0:
                            return output_price
                
                # Fallback to old pricing format
                pricing = row.get('pricing', {})
                if isinstance(pricing, dict):
                    if 'output_tokens_per_1k' in pricing:
                        return pricing.get('output_tokens_per_1k', 0)
                    elif 'on_demand' in pricing and isinstance(pricing['on_demand'], dict):
                        return pricing['on_demand'].get('output_tokens', 0)
                return 0
            
            df_copy = df.copy()
            df_copy['sort_price'] = df_copy.apply(get_output_price, axis=1)
            return df_copy.sort_values('sort_price', ascending=ascending).drop('sort_price', axis=1)
        
        elif sort_column == "region_count":
            # Sort by number of regions
            def get_region_count(regions):
                if isinstance(regions, list):
                    return len(regions)
                return 0
            
            df_copy = df.copy()
            df_copy['sort_regions'] = df_copy['regions'].apply(get_region_count)
            return df_copy.sort_values('sort_regions', ascending=ascending).drop('sort_regions', axis=1)
        
        elif sort_column == "context_window":
            # Sort by context window from converse_data
            def get_model_context_window(row):
                from ui.converse_data_helpers import get_converse_context_window
                return get_converse_context_window(row)

            df_copy = df.copy()
            df_copy['sort_context'] = df_copy.apply(get_model_context_window, axis=1)
            return df_copy.sort_values('sort_context', ascending=ascending).drop('sort_context', axis=1)

        elif sort_column == "release_date":
            # Sort by release date if available
            if 'release_date' in df.columns:
                return df.sort_values('release_date', ascending=ascending)
            else:
                # Fallback to name sorting if no release date
                return df.sort_values('model_name', ascending=ascending)

        else:
            # Default to name sorting
            return df.sort_values('model_name', ascending=ascending)
            
    except Exception as e:
        # If sorting fails, return original dataframe
        print(f"Sorting error: {e}")
        return df

def show_filter_summary(filter_state: Dict[str, Any], original_count: int, filtered_count: int) -> None:
    """
    Display a summary of active filters.
    
    Args:
        filter_state: Dictionary containing filter values
        original_count: Original number of models
        filtered_count: Number of models after filtering
    """
    # Count active filters
    active_filters = 0
    if filter_state.get('search_query'):
        active_filters += 1
    if filter_state.get('providers', []) and len(filter_state.get('providers', [])) < original_count:
        active_filters += 1
    if filter_state.get('capabilities', []) and len(filter_state.get('capabilities', [])) < original_count:
        active_filters += 1
    if filter_state.get('geo_region', 'All Regions') != 'All Regions':
        active_filters += 1
    if filter_state.get('model_status', 'All Status') != 'All Status':
        active_filters += 1
    if filter_state.get('consumption_filter', 'All Models') != 'All Models':
        active_filters += 1
    if filter_state.get('customization_filter', 'All Models') != 'All Models':
        active_filters += 1
    if filter_state.get('cris_support', 'All Models') != 'All Models':
        active_filters += 1
    if filter_state.get('use_cases', []) and len(filter_state.get('use_cases', [])) < original_count:
        active_filters += 1
    if filter_state.get('modality_filter', 'All Modalities') != 'All Modalities':
        active_filters += 1
    if filter_state.get('languages', []) and len(filter_state.get('languages', [])) < original_count:
        active_filters += 1
    if filter_state.get('context_filter', 'All Models') != 'All Models':
        active_filters += 1
    if filter_state.get('streaming', 'All Models') != "All Models":
        active_filters += 1
    
    # Display summary
    if active_filters > 0:
        st.info(f"📊 Showing {filtered_count} of {original_count} models with {active_filters} active filters")
    else:
        st.info(f"📊 Showing all {original_count} models")