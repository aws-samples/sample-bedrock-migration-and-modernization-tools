"""
Common utility functions for the Amazon Bedrock Model Expert application.
"""
import streamlit as st
import pandas as pd
import json
import time
from typing import Dict, List, Any, Union

# Set pandas option to avoid FutureWarning about downcasting
pd.set_option('future.no_silent_downcasting', True)


def derive_model_size_from_context_window(context_window: int) -> str:
    """
    Derive model size category from context window size.

    Args:
        context_window: Context window size in tokens

    Returns:
        Model size category string
    """
    if context_window == 0 or pd.isna(context_window) or context_window is None:
        return "Unknown"
    elif context_window <= 4096:
        return "Micro"
    elif context_window <= 8192:
        return "Small"
    elif context_window <= 32768:
        return "Medium"
    elif context_window <= 128000:
        return "Large"
    else:
        return "Extra Large"


def derive_model_size_from_model_info(row) -> str:
    """
    Derive model size from various model information fields.

    Args:
        row: DataFrame row with model data

    Returns:
        Model size category string
    """
    # First try context window
    context_window = row.get('context_window')
    if context_window and not pd.isna(context_window) and context_window != 0:
        return derive_model_size_from_context_window(context_window)

    # Try to derive from model family or model ID
    model_id = (row.get('model_id') or '').lower()
    model_family = (row.get('model_family') or '').lower()

    # Image models are typically smaller in terms of context window
    input_modalities = row.get('input_modalities', [])
    output_modalities = row.get('output_modalities', [])

    if 'IMAGE' in input_modalities or 'IMAGE' in output_modalities:
        return "Medium"  # Image models default to medium

    # Text models without context window info
    if any(keyword in model_id for keyword in ['claude', 'gpt', 'llama', 'titan-text']):
        if any(size in model_id for size in ['large', 'xl', 'pro']):
            return "Large"
        elif any(size in model_id for size in ['small', 'lite', 'express']):
            return "Small"
        else:
            return "Medium"

    # Default based on provider patterns
    provider = (row.get('provider') or '').lower()
    if 'amazon' in provider:
        return "Medium"
    elif 'anthropic' in provider:
        return "Large"
    elif 'meta' in provider:
        return "Large"
    elif 'stability' in provider:
        return "Medium"

    return "Unknown"


def get_provider_badge_class(provider: str) -> str:
    """
    Get the CSS class for a provider badge.
    
    Args:
        provider: Provider name
        
    Returns:
        CSS class name
    """
    provider_lower = provider.lower()
    
    if 'amazon' in provider_lower:
        return 'amazon-badge'
    elif 'anthropic' in provider_lower:
        return 'anthropic-badge'
    elif 'meta' in provider_lower:
        return 'meta-badge'
    elif 'cohere' in provider_lower:
        return 'cohere-badge'
    elif 'mistral' in provider_lower:
        return 'mistral-badge'
    elif 'stability' in provider_lower:
        return 'stability-badge'
    elif 'deepseek' in provider_lower:
        return 'deepseek-badge'
    elif 'writer' in provider_lower:
        return 'writer-badge'
    elif 'luma' in provider_lower:
        return 'luma-badge'
    else:
        return 'amazon-badge'  # Default to Amazon


def get_provider_color(provider: str) -> str:
    """
    Get the color for a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        Color hex code
    """
    provider_lower = provider.lower()
    
    if 'amazon' in provider_lower:
        return '#FF9900'
    elif 'anthropic' in provider_lower:
        return '#D4A574'
    elif 'meta' in provider_lower:
        return '#1877F2'
    elif 'cohere' in provider_lower:
        return '#39594C'
    elif 'mistral' in provider_lower:
        return '#FF6B35'
    elif 'stability' in provider_lower:
        return '#000000'
    elif 'deepseek' in provider_lower:
        return '#8B5CF6'
    elif 'writer' in provider_lower:
        return '#10B981'
    elif 'luma' in provider_lower:
        return '#F59E0B'
    else:
        return '#667eea'  # Default color


def get_status_badge_html(model: Dict[str, Any]) -> str:
    """
    Get HTML for a model status badge.
    
    Args:
        model: Model data dictionary
        
    Returns:
        HTML string for status badge
    """
    # Get status from model_lifecycle field
    model_lifecycle = model.get('model_lifecycle', {})
    if isinstance(model_lifecycle, dict):
        lifecycle_status = model_lifecycle.get('status', 'ACTIVE')
    else:
        lifecycle_status = 'ACTIVE'
    
    if lifecycle_status == 'ACTIVE':
        return '<span style="background-color: #10B981; color: white; padding: 2px 8px; border-radius: 20px; font-size: 0.7rem; font-weight: bold;">Active</span>'
    elif lifecycle_status == 'LEGACY':
        return '<span style="background-color: #F59E0B; color: white; padding: 2px 8px; border-radius: 20px; font-size: 0.7rem; font-weight: bold;">Legacy</span>'
    else:
        return '<span style="background-color: #10B981; color: white; padding: 2px 8px; border-radius: 20px; font-size: 0.7rem; font-weight: bold;">Active</span>'


def format_price(price: float) -> str:
    """
    Format a price value for display.
    
    Args:
        price: Price value
        
    Returns:
        Formatted price string
    """
    if price == 0:
        return "$0.00"
    elif price < 0.00001:
        return "<$0.00001"
    elif price < 0.001:
        return f"${price:.5f}"
    elif price < 0.01:
        return f"${price:.4f}"
    elif price < 0.1:
        return f"${price:.3f}"
    else:
        return f"${price:.2f}"


def clean_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean model data to handle NaN values and ensure consistent data types.
    
    Args:
        df: DataFrame with model data
        
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Fill missing values with defaults and handle dictionary descriptions
    def clean_description(desc):
        if pd.isna(desc):
            return 'No description available'
        elif isinstance(desc, dict):
            # Extract the best description from dictionary
            # Priority: shortDescription > fullDescription > first available text
            if 'shortDescription' in desc and desc['shortDescription']:
                return str(desc['shortDescription'])
            elif 'fullDescription' in desc and desc['fullDescription']:
                return str(desc['fullDescription'])
            elif 'supportedUseCases' in desc and desc['supportedUseCases']:
                return f"Supported use cases: {str(desc['supportedUseCases'])}"
            else:
                # Convert dict to string representation as fallback
                return str(desc)
        else:
            return str(desc)

    # Add description column if not present and handle existing ones
    if 'description' not in df.columns:
        df['description'] = 'No description available'
    else:
        df['description'] = df['description'].apply(clean_description)

    # Handle context_window - check if it exists in new structure
    if 'context_window' in df.columns:
        df['context_window'] = df['context_window'].fillna(0).astype(int)
    else:
        # Add default context_window column if missing
        df['context_window'] = 0

    # Handle max_output_tokens - add if missing
    if 'max_output_tokens' not in df.columns:
        # Try to derive from context_window or set default
        df['max_output_tokens'] = (df['context_window'] * 0.25).fillna(0).astype(int)  # Typical ratio
    else:
        df['max_output_tokens'] = df['max_output_tokens'].fillna(0).astype(int)

    # Handle streaming support - check response_streaming_supported field
    if 'streaming_supported' not in df.columns:
        if 'response_streaming_supported' in df.columns:
            df['streaming_supported'] = df['response_streaming_supported'].fillna(False)
        else:
            df['streaming_supported'] = False
    else:
        df['streaming_supported'] = df['streaming_supported'].fillna(False)

    # Derive model_size from available information if not present
    if 'model_size' not in df.columns:
        df['model_size'] = df.apply(derive_model_size_from_model_info, axis=1)
    else:
        df['model_size'] = df['model_size'].fillna('Unknown')
    
    # Extract modalities from model_modalities nested structure
    def extract_modalities(row):
        """Extract input and output modalities from model_modalities field"""
        model_modalities = row.get('model_modalities', {})

        # Get modalities from nested structure
        input_modalities = []
        output_modalities = []

        if isinstance(model_modalities, dict):
            input_modalities = model_modalities.get('input_modalities', [])
            output_modalities = model_modalities.get('output_modalities', [])

        # Fallback to direct fields for backward compatibility
        if not input_modalities and 'input_modalities' in row:
            input_modalities = row.get('input_modalities', [])
        if not output_modalities and 'output_modalities' in row:
            output_modalities = row.get('output_modalities', [])

        # Ensure they are lists
        if not isinstance(input_modalities, list):
            input_modalities = []
        if not isinstance(output_modalities, list):
            output_modalities = []

        return pd.Series({
            'input_modalities': input_modalities,
            'output_modalities': output_modalities
        })

    # Apply modalities extraction
    modalities_df = df.apply(extract_modalities, axis=1)
    df['input_modalities'] = modalities_df['input_modalities']
    df['output_modalities'] = modalities_df['output_modalities']

    # Add missing fields with defaults
    field_defaults = {
        'languages': [],  # Some models might not have this, use languages_supported instead
        'use_cases': [],
        'capabilities': [],
        'regions': []
    }

    # Ensure model_name exists (legacy compatibility)
    if 'name' in df.columns and 'model_name' not in df.columns:
        df['model_name'] = df['name']
    elif 'model_name' not in df.columns:
        df['model_name'] = df['model_id']  # Fallback to model_id

    # Add 'name' column for backward compatibility with UI code
    if 'name' not in df.columns:
        df['name'] = df['model_name']

    # Map field names from new structure to legacy names for UI compatibility
    field_mappings = {
        'capabilities': 'model_capabilities',
        'use_cases': 'model_use_cases',
        'languages': 'languages_supported'
    }

    for ui_field, data_field in field_mappings.items():
        if ui_field not in df.columns and data_field in df.columns:
            df[ui_field] = df[data_field]

    # Check for alternative field names and set defaults
    for field, default in field_defaults.items():
        if field not in df.columns:
            if field == 'languages' and 'languages_supported' in df.columns:
                df['languages'] = df['languages_supported']
            else:
                df[field] = [default] * len(df)

    # Ensure lists for array fields (using both old and new field names)
    array_fields = ['capabilities', 'model_capabilities', 'regions', 'languages', 'use_cases', 'model_use_cases', 'input_modalities', 'output_modalities']
    for col in array_fields:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

    # Fix empty regions lists (this was causing IndexError in UI)
    if 'regions' in df.columns:
        df['regions'] = df['regions'].apply(
            lambda x: ['us-east-1', 'us-west-2', 'eu-west-1'] if not x or len(x) == 0 else x
        )
    
    # Ensure dictionary for comprehensive_pricing
    if 'comprehensive_pricing' in df.columns:
        df['comprehensive_pricing'] = df['comprehensive_pricing'].apply(lambda x: x if isinstance(x, dict) else {})
    
    # For backward compatibility, also handle old pricing field if it exists
    if 'pricing' in df.columns:
        df['pricing'] = df['pricing'].apply(lambda x: x if isinstance(x, dict) else {})
    
    return df


def filter_models(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Filter models based on filter criteria.
    
    Args:
        df: DataFrame with model data
        filters: Dictionary with filter criteria
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Provider filter
    if 'providers' in filters and filters['providers']:
        filtered_df = filtered_df[filtered_df['provider'].isin(filters['providers'])]
    
    # Capability filter
    if 'capabilities' in filters and filters['capabilities']:
        filtered_df = filtered_df[filtered_df['capabilities'].apply(
            lambda caps: any(cap in caps for cap in filters['capabilities'])
        )]
    
    # Region filter
    if 'regions' in filters and filters['regions']:
        filtered_df = filtered_df[filtered_df['regions'].apply(
            lambda regions: any(region in regions for region in filters['regions'])
        )]
    
    # Price range filter
    if 'price_range' in filters and filters['price_range']:
        min_price, max_price = filters['price_range']
        
        # Extract input price for filtering from comprehensive_pricing
        def extract_input_price(row):
            # Try comprehensive_pricing first (new format)
            if 'comprehensive_pricing' in row and isinstance(row['comprehensive_pricing'], dict):
                regions_data = row['comprehensive_pricing'].get('regions', {})
                # Try US regions first
                for region in ['us-east-1', 'us-west-2']:
                    if region in regions_data:
                        on_demand = regions_data[region].get('on_demand', {})
                        if on_demand.get('input_tokens', 0) > 0:
                            return on_demand['input_tokens']
                # Try any region with pricing
                for region_data in regions_data.values():
                    on_demand = region_data.get('on_demand', {})
                    if on_demand.get('input_tokens', 0) > 0:
                        return on_demand['input_tokens']
            
            # Fallback to old pricing format
            if 'pricing' in row and isinstance(row['pricing'], dict):
                return row['pricing'].get('on_demand', {}).get('input_tokens', 0)
            
            return 0
        
        filtered_df['input_price'] = filtered_df.apply(extract_input_price, axis=1)
        
        # Apply price filter
        if max_price < 100:  # If max_price is less than the maximum possible value
            filtered_df = filtered_df[(filtered_df['input_price'] >= min_price) & 
                                     (filtered_df['input_price'] <= max_price)]
        else:
            filtered_df = filtered_df[filtered_df['input_price'] >= min_price]
    
    # Context window filter
    if 'context_window' in filters and filters['context_window']:
        min_context, max_context = filters['context_window']
        
        if max_context < 1000000:  # If max_context is less than the maximum possible value
            filtered_df = filtered_df[(filtered_df['context_window'] >= min_context) & 
                                     (filtered_df['context_window'] <= max_context)]
        else:
            filtered_df = filtered_df[filtered_df['context_window'] >= min_context]
    
    # Model size filter
    if 'model_size' in filters and filters['model_size']:
        filtered_df = filtered_df[filtered_df['model_size'].isin(filters['model_size'])]
    
    # Streaming support filter
    if 'streaming' in filters:
        filtered_df = filtered_df[filtered_df['streaming_supported'] == filters['streaming']]
    
    # Status filter
    if 'status' in filters and filters['status']:
        filtered_df = filtered_df[filtered_df['status'].isin(filters['status'])]
    
    # Search filter
    if 'search' in filters and filters['search']:
        search_term = filters['search'].lower()
        
        # Search in multiple columns
        filtered_df = filtered_df[
            filtered_df['name'].str.lower().str.contains(search_term, na=False) |
            filtered_df['model_id'].str.lower().str.contains(search_term, na=False) |
            filtered_df['provider'].str.lower().str.contains(search_term, na=False) |
            filtered_df['description'].str.lower().str.contains(search_term, na=False) |
            filtered_df['capabilities'].apply(lambda caps: any(search_term in cap.lower() for cap in caps))
        ]
    
    return filtered_df


def get_aws_documentation_url(model_id: str) -> str:
    """
    Get AWS documentation URL for a model.
    
    Args:
        model_id: Model ID
        
    Returns:
        Documentation URL
    """
    # Extract model name from ID
    model_name = model_id.split(':')[0] if ':' in model_id else model_id
    
    # Map model names to documentation URLs
    model_docs = {
        'anthropic.claude-3-sonnet-20240229-v1': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude-3.html',
        'anthropic.claude-3-haiku-20240307-v1': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude-3.html',
        'anthropic.claude-3-opus-20240229-v1': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude-3.html',
        'anthropic.claude-v2': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html',
        'anthropic.claude-v2:1': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html',
        'anthropic.claude-instant-v1': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html',
        'meta.llama2-13b-chat-v1': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html',
        'meta.llama2-70b-chat-v1': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html',
        'amazon.titan-text-express-v1': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html',
        'amazon.titan-text-lite-v1': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html',
        'amazon.titan-embed-text-v1': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed.html',
        'amazon.titan-embed-image-v1': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed.html',
        'amazon.titan-image-generator-v1': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html',
        'cohere.command-text-v14': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command.html',
        'cohere.command-light-text-v14': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command.html',
        'cohere.embed-english-v3': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-embed.html',
        'cohere.embed-multilingual-v3': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-embed.html',
        'stability.stable-diffusion-xl-v1': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-stability-diffusion.html',
        'stability.stable-diffusion-xl-v0': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-stability-diffusion.html',
        'ai21.j2-mid-v1': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-ai21-jurassic2.html',
        'ai21.j2-ultra-v1': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-ai21-jurassic2.html',
        'mistral.mistral-7b-instruct-v0:2': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html',
        'mistral.mixtral-8x7b-instruct-v0:1': 'https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html'
    }
    
    # Return documentation URL if available, otherwise return general Bedrock documentation
    return model_docs.get(model_name, 'https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html')








