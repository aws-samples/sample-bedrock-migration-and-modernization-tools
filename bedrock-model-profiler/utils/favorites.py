"""
Favorites management utilities for the Amazon Bedrock Model Expert application.

This module provides functions for managing user favorites, including:
- Adding and removing models from favorites
- Persisting favorites across sessions using local storage
- Loading favorites from local storage
- Checking if a model is in favorites
"""
import streamlit as st
import json
import os
from typing import Dict, List, Any, Optional

def initialize_favorites():
    """Initialize favorites in session state if not already present"""
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    
    # Load favorites from local storage if available
    load_favorites_from_storage()

def is_favorite(model_id: str) -> bool:
    """Check if a model is in favorites
    
    Args:
        model_id: The model ID to check
        
    Returns:
        True if the model is in favorites, False otherwise
    """
    initialize_favorites()
    return model_id in st.session_state.favorites

def toggle_favorite(model_id: str) -> bool:
    """Toggle favorite status for a model
    
    Args:
        model_id: The model ID to toggle
        
    Returns:
        True if the model is now a favorite, False if it was removed
    """
    initialize_favorites()
    
    if model_id in st.session_state.favorites:
        st.session_state.favorites.remove(model_id)
        save_favorites_to_storage()
        return False
    else:
        st.session_state.favorites.append(model_id)
        save_favorites_to_storage()
        return True

def add_to_favorites(model_id: str) -> bool:
    """Add a model to favorites
    
    Args:
        model_id: The model ID to add
        
    Returns:
        True if the model was added, False if it was already in favorites
    """
    initialize_favorites()
    
    if model_id not in st.session_state.favorites:
        st.session_state.favorites.append(model_id)
        save_favorites_to_storage()
        return True
    return False

def remove_from_favorites(model_id: str) -> bool:
    """Remove a model from favorites
    
    Args:
        model_id: The model ID to remove
        
    Returns:
        True if the model was removed, False if it wasn't in favorites
    """
    initialize_favorites()
    
    if model_id in st.session_state.favorites:
        st.session_state.favorites.remove(model_id)
        save_favorites_to_storage()
        return True
    return False

def get_favorites() -> List[str]:
    """Get the list of favorite model IDs
    
    Returns:
        List of model IDs in favorites
    """
    initialize_favorites()
    return st.session_state.favorites

def clear_favorites():
    """Clear all favorites"""
    st.session_state.favorites = []
    save_favorites_to_storage()

def save_favorites_to_storage():
    """Save favorites to local storage using Streamlit's experimental_set_query_params
    
    This allows favorites to persist across sessions.
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save favorites to a JSON file
        with open('data/favorites.json', 'w', encoding='utf-8') as f:
            json.dump({
                'favorites': st.session_state.favorites,
                'timestamp': str(st.session_state.get('last_activity', ''))
            }, f)
    except Exception as e:
        st.error(f"Error saving favorites: {str(e)}")

def load_favorites_from_storage():
    """Load favorites from local storage
    
    This allows favorites to persist across sessions.
    """
    try:
        # Check if favorites file exists
        if os.path.exists('data/favorites.json'):
            with open('data/favorites.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'favorites' in data and isinstance(data['favorites'], list):
                    st.session_state.favorites = data['favorites']
    except Exception as e:
        st.error(f"Error loading favorites: {str(e)}")

def get_favorite_models(df) -> Any:
    """Get favorite models from the dataframe
    
    Args:
        df: DataFrame containing model data
        
    Returns:
        DataFrame containing only favorite models
    """
    initialize_favorites()
    
    if not st.session_state.favorites:
        return df.head(0)  # Return empty DataFrame with same structure
    
    return df[df['model_id'].isin(st.session_state.favorites)]