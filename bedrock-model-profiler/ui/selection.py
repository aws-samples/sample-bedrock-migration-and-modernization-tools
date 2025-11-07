"""
Model selection utilities for the Amazon Bedrock Expert application.
"""
import streamlit as st
from typing import List, Set, Dict, Any


def initialize_selection_state():
    """Initialize selection-related session state variables"""
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = set()
    
    if 'comparison_models' not in st.session_state:
        st.session_state.comparison_models = set()
    

    
    if 'pricing_models' not in st.session_state:
        st.session_state.pricing_models = set()


def toggle_model_selection(model_id: str, selected: bool = None) -> bool:
    """
    Toggle or set the selection state of a model.
    
    Args:
        model_id: The ID of the model to toggle
        selected: If provided, set the selection state to this value
        
    Returns:
        The new selection state (True if selected, False if not)
    """
    initialize_selection_state()
    
    if selected is None:
        # Toggle the current state
        if model_id in st.session_state.selected_models:
            st.session_state.selected_models.remove(model_id)
            return False
        else:
            st.session_state.selected_models.add(model_id)
            return True
    else:
        # Set to the specified state
        if selected:
            st.session_state.selected_models.add(model_id)
        else:
            if model_id in st.session_state.selected_models:
                st.session_state.selected_models.remove(model_id)
        return selected


def is_model_selected(model_id: str) -> bool:
    """
    Check if a model is currently selected.
    
    Args:
        model_id: The ID of the model to check
        
    Returns:
        True if the model is selected, False otherwise
    """
    initialize_selection_state()
    return model_id in st.session_state.selected_models


def get_selected_models() -> Set[str]:
    """
    Get the set of currently selected model IDs.
    
    Returns:
        A set of selected model IDs
    """
    initialize_selection_state()
    return st.session_state.selected_models


def clear_selection():
    """Clear all selected models"""
    initialize_selection_state()
    st.session_state.selected_models = set()


def set_models_for_comparison():
    """Set the currently selected models for comparison"""
    initialize_selection_state()
    st.session_state.comparison_models = st.session_state.selected_models.copy()



def set_models_for_pricing():
    """Set the currently selected models for pricing calculation"""
    initialize_selection_state()
    st.session_state.pricing_models = st.session_state.selected_models.copy()


def get_comparison_models() -> Set[str]:
    """Get the set of models selected for comparison"""
    initialize_selection_state()
    return st.session_state.comparison_models



def get_pricing_models() -> Set[str]:
    """Get the set of models selected for pricing calculation"""
    initialize_selection_state()
    return st.session_state.pricing_models


def show_selection_summary(view_type=""):
    """
    Display a summary of the current selection
    
    Args:
        view_type: A string to differentiate between different views (table, compact, etc.)
    """
    initialize_selection_state()
    
    if st.session_state.selected_models:
        st.success(f"✅ {len(st.session_state.selected_models)} models selected for comparison")
        
        # Show clear selection button with unique key
        if st.button("Clear Selection", key=f"clear_selection_{view_type}"):
            clear_selection()
            st.rerun()


def show_selection_actions(view_type=""):
    """
    Display action buttons for selected models
    
    Args:
        view_type: A string to differentiate between different views (table, compact, etc.)
    """
    initialize_selection_state()
    
    if not st.session_state.selected_models:
        return
    
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Compare Selected Models", key=f"compare_selected_action_{view_type}", width="stretch"):
            set_models_for_comparison()
            st.success(f"Comparing {len(st.session_state.selected_models)} models")
            # Set the active tab to comparison
            if 'active_tab' in st.session_state:
                st.session_state.active_tab = "⚖️ Comparison"
    
    with col2:
        if st.button("Calculate Pricing", key=f"pricing_selected_action_{view_type}", width="stretch"):
            set_models_for_pricing()
            st.success(f"Calculating pricing for {len(st.session_state.selected_models)} models")
            # Set the active tab to pricing
            if 'active_tab' in st.session_state:
                st.session_state.active_tab = "💰 Pricing Calculator"