"""
Amazon Bedrock Model Profiler - Main Application

This is the main entry point for the Amazon Bedrock Model Profiler application.
It provides comprehensive visibility of available Amazon Bedrock foundation models
and helps you select the best option for your specific use case.

The application is built using Streamlit and provides the following features:
- Model Profiler: Browse and filter Amazon Bedrock models with detailed specifications
- Favorites: Save and organize your preferred models
- Model Comparison: Compare models side-by-side to find the best fit
"""
import streamlit as st
import pandas as pd
import json
import time

# Import utility functions
from utils.common import clean_model_data
from utils.credentials import get_available_aws_profiles
from utils.error_handling import show_error_message

# Import UI components
from ui.pages import (
    show_model_explorer, 
    show_model_comparison
)

# Import data updater
from utils.data_updater import ModelDataUpdater

# Import updated model repository
from models.new_model_repository import NewModelRepository as ModelRepository

# Page configuration
st.set_page_config(
    page_title="🤖 Amazon Bedrock Model Profiler",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS immediately
with open('ui/custom_styles.css', 'r', encoding='utf-8') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load CSS styles
def load_css():
    """Load custom CSS styles for the application"""
    # Load custom CSS file
    try:
        with open('ui/custom_styles.css', 'r', encoding='utf-8') as f:
            custom_css = f.read()
            st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Custom CSS file not found. Please check if ui/custom_styles.css exists.")

def show_overview_boxes(df, is_favorites=False):
    """Show overview statistics boxes"""
    if df.empty:
        return
    
    # Calculate statistics
    total_models = len(df)
    total_providers = df['provider'].nunique()
    
    # Count cross-region inference support
    cross_region_count = 0
    for _, row in df.iterrows():
        cross_region_inference = row.get('cross_region_inference', {})
        if isinstance(cross_region_inference, dict) and cross_region_inference.get('supported', False):
            cross_region_count += 1
    
    # Count unique regions - use metadata as primary source since individual models only show 3 each
    try:
        from models.new_model_repository import NewModelRepository
        repo = NewModelRepository()
        metadata = repo.get_metadata()
        metadata_regions = metadata.get('regions', [])

        if isinstance(metadata_regions, list) and len(metadata_regions) > 0:
            total_regions = len(metadata_regions)  # All 20 regions from metadata
        else:
            # Fallback: count from individual models
            all_regions = set()
            for _, row in df.iterrows():
                regions = row.get('regions', [])
                available_regions = row.get('available_regions', [])
                if isinstance(regions, list):
                    all_regions.update(regions)
                if isinstance(available_regions, list):
                    all_regions.update(available_regions)
            total_regions = len(all_regions)
    except Exception:
        # Final fallback
        all_regions = set()
        for _, row in df.iterrows():
            regions = row.get('regions', [])
            if isinstance(regions, list):
                all_regions.update(regions)
        total_regions = len(all_regions)
    
    # Count multimodal models
    multimodal_count = 0
    for _, row in df.iterrows():
        input_modalities = row.get('input_modalities', [])
        output_modalities = row.get('output_modalities', [])
        if isinstance(input_modalities, list) and isinstance(output_modalities, list):
            all_modalities = set(input_modalities + output_modalities)
            if len(all_modalities) > 1 or ('IMAGE' in all_modalities or 'AUDIO' in all_modalities):
                multimodal_count += 1
    
    # Add banner for overview sections
    if is_favorites:
        st.markdown("""
        <div style="background: linear-gradient(90deg, rgb(31, 78, 121) 0%, rgb(45, 90, 160) 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem;">
            <h3 style="color: white; margin: 0; text-align: center;">⭐ Favorites Overview</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(90deg, rgb(31, 78, 121) 0%, rgb(45, 90, 160) 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem;">
            <h3 style="color: white; margin: 0; text-align: center;">📊 Models Overview</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Create 5 columns for the metrics
    col0, col1, col2, col3, col4, col5 = st.columns([.4, 1, 1, 1, 1, 1])
    
    with col1:
        st.metric(
            label="📚 Models",
            value=f"{total_models:,}",
            help="Total number of models available"
        )
    
    with col2:
        st.metric(
            label="🏢 Providers",
            value=f"{total_providers}",
            help="Number of different model providers"
        )
    
    with col3:
        st.metric(
            label="🌍 Regions",
            value=f"{total_regions}",
            help="Number of AWS regions where models are available"
        )
    
    with col4:
        st.metric(
            label="🌐 Cross-Region",
            value=f"{cross_region_count}",
            help="Models supporting cross-region inference"
        )
    
    with col5:
        st.metric(
            label="🎨 Multimodal",
            value=f"{multimodal_count}",
            help="Models supporting multiple input/output types (text, image, audio)"
        )


# Load model data with optimized caching
@st.cache_data(ttl=300)  # Cache for 5 minutes, then refresh
def load_model_data():
    """Load Bedrock model data from JSON file with data cleaning"""
    try:
        # Use ModelRepository to load data
        repository = ModelRepository()
        df = repository.load_models_df()
        
        if df.empty:
            return pd.DataFrame()
        
        # Clean the data to handle NaN values
        df = clean_model_data(df)
        
        # Add timestamp for cache validation
        df.attrs['cache_timestamp'] = time.time()
        
        return df
    except Exception as e:
        # Use error handling utilities
        show_error_message(e, context="load_model_data")
        return pd.DataFrame()

# Paginate dataframe for better performance
def paginate_dataframe(df: pd.DataFrame, page_size: int = 10, page_num: int = 1) -> pd.DataFrame:
    """
    Paginate a dataframe for better performance with large datasets.
    
    Args:
        df: DataFrame to paginate
        page_size: Number of items per page
        page_num: Page number (1-indexed)
        
    Returns:
        Paginated DataFrame
    """
    total_pages = max(1, (len(df) + page_size - 1) // page_size)
    page_num = min(max(1, page_num), total_pages)
    
    start_idx = (page_num - 1) * page_size
    end_idx = min(start_idx + page_size, len(df))
    
    return df.iloc[start_idx:end_idx].copy(), total_pages, page_num

def update_models_from_aws(profile_name=None, region='us-east-1'):
    """Update models using the new data collection system"""
    with st.spinner("🔄 Updating models using integrated collectors..."):
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Display information about data sources
            st.info("""
            📊 **New Data Collection System:**
            - Amazon Bedrock Model Collector: Comprehensive model data extraction
            - Amazon Bedrock Pricing Collector: Real-time pricing information
            - Multi-threaded collection across all AWS regions
            - Integrated service quotas and inference profiles
            """)

            status_text.text("🔗 Initializing comprehensive data update...")
            progress_bar.progress(10)

            # Initialize the new data updater
            updater = ModelDataUpdater()

            # Step 1: Run pricing collector
            status_text.text("💰 Step 1/2: Running pricing collector (Foundation Models included)...")
            progress_bar.progress(25)
            pricing_success = updater.run_pricing_collector()

            # Step 2: Run model collector
            status_text.text("📡 Step 2/2: Running model collector with pricing integration...")
            progress_bar.progress(50)
            model_success = updater.run_model_collector(profile_name=profile_name)

            if not model_success:
                raise Exception("Model data update failed - check AWS credentials and permissions")

            # Verify and validate the data
            if not updater.verify_model_data_exists() or not updater.validate_new_data():
                raise Exception("Model data validation failed")

            status_text.text("✅ Finalizing update...")
            progress_bar.progress(90)

            status_text.text("✅ Update completed successfully!")
            progress_bar.progress(100)

            # Clear the progress indicators
            progress_bar.empty()
            status_text.empty()

            # Load and analyze the updated data
            repo = ModelRepository()
            stats = repo.get_statistics()

            # Show success message with statistics
            st.success(f"""
            🎉 **Model Database Updated Successfully!**

            📊 **Updated Statistics:**
            • **Total Models:** {stats['total_models']} models
            • **Providers:** {stats['total_providers']} providers
            • **Regions Covered:** {stats['regions_covered']} regions
            • **Models with Pricing:** {stats['models_with_pricing']} models
            • **Models with Quotas:** {stats['models_with_quotas']} models

            🔄 **Data Sources:**
            • Bedrock API, Pricing API, Service Quotas API
            • Multi-region data collection
            • Real-time model availability

            📅 **Last Updated:** {stats['last_updated']}
            🔢 **Data Version:** {stats['version']}

            The application will refresh automatically to show the updated data.
            """)

            # Clear cache and force a rerun to reload the data
            st.cache_data.clear()  # Clear cached data to ensure fresh data is loaded
            st.rerun()

        except Exception as e:
            st.error(f"""
            ❌ **Update Failed**

            **Error:** {str(e)}

            **Possible causes:**
            - AWS credentials not configured properly
            - Network connectivity issues
            - Insufficient AWS permissions for Bedrock API
            - Model collector or pricing collector issues

            **Solutions:**
            - Verify AWS profile is configured
            - Check AWS credentials with: `aws sts get-caller-identity --profile <your-aws-profile-name>`
            - Ensure Bedrock permissions in your AWS account
            """)

            # Show manual update options
            st.info("""
            💡 **Manual Update Options:**

            **Run Collector Manually:**
            ```bash
            cd collectors/model-collector
            python3 main.py
            ```

            **Or use the unified launcher:**
            ```bash
            python3 -m utils.data_updater
            ```

            **Note:** Model collector now saves directly to `data/bedrock_models.json`
            """)

def main():
    """Main application entry point"""
    # Load CSS styles
    load_css()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1><span class="title-icon">🤖</span> Amazon Bedrock Model Profiler</h1>
        <p>Comprehensive model analysis and selection for migration planning and workload optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_model_data()
    
    # Initialize session state
    if 'filters' not in st.session_state:
        st.session_state.filters = {
            'providers': [],
            'capabilities': [],
            'regions': [],
            'price_range': [0, 100]
        }
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "explorer"
    
    # Sidebar for navigation and model updates
    with st.sidebar:
        st.header("🔍 Navigation & Updates")
        
        # Current Database Status - Clean and Simple
        try:
            with open('data/bedrock_models.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Get metadata (new structure)
                metadata = data.get('metadata', {})

                # Get total models and providers from metadata (preferred) or calculate
                total_models = metadata.get('total_models', 0)
                providers_count = metadata.get('providers_count', 0)

                # Fallback: calculate from actual data structure if metadata is incomplete
                if total_models == 0 or providers_count == 0:
                    providers_data = data.get('providers', {})
                    if providers_data:
                        # New hierarchical structure
                        providers_count = len(providers_data)
                        total_models = sum(len(provider_data.get('models', {})) for provider_data in providers_data.values())
                    else:
                        # Old flat structure fallback
                        models = data.get('models', [])
                        if isinstance(models, list):
                            providers_list = list(set(model.get('provider', '') for model in models if model.get('provider')))
                            providers_count = len(providers_list)
                            total_models = len(models)

                # Get last updated from metadata (check multiple possible field names)
                last_updated = (metadata.get('collection_timestamp') or
                              metadata.get('extraction_date') or
                              metadata.get('generated_at') or
                              data.get('last_updated', 'Unknown'))
                
                # Clean database info display
                st.info(f"""**📊 Current Database:**
• **{providers_count}** Providers
• **{total_models}** Models
• **Updated:** {last_updated.split(' ')[0] if last_updated != 'Unknown' else 'Unknown'}""")
        except:
            st.warning("📊 **Database:** Not available")
        
        st.divider()
        
        # Combined AWS Setup & Update Section
        with st.expander("⚙️ AWS Setup & Update", expanded=False):
            # Simple credentials selection
            st.write("**AWS Configuration:**")
            
            # Get available profiles
            profiles = get_available_aws_profiles()
            
            # Profile selection
            selected_profile = st.selectbox(
                "Profile:",
                options=profiles,
                index=0,  # Default to first profile (which is "default")
                key="profile_select"
            )
            
            st.divider()
            
            # What gets updated - with Foundation Models
            st.write("**📋 Update Process (2 steps):**")
            st.write("**Step 1:** Latest pricing data (Foundation Models included)")
            st.write("**Step 2:** Complete model database with pricing integration")
            st.write("• Models available • Enhanced pricing • Feature updates • Service quotas")
            
            st.write("**📡 Data sources:**")
            st.write("• Service Quotas, Pricing, Bedrock API • Official AWS Documentation")
            
            st.divider()

            # Timing reminder for users
            st.info("⏱️ **Expected update time:** up to 5 minutes\n\nPlease be patient while we collect the latest model data.")

            # Update button - simplified
            if st.button("🔄 Update models database", width="stretch", type="primary"):
                # Validate credentials first
                try:
                    import boto3
                    from botocore.exceptions import ClientError, ProfileNotFound
                    
                    # Test credentials with default region
                    default_region = 'us-west-2'
                    session = boto3.Session(profile_name=selected_profile if selected_profile != 'default' else None)
                    bedrock_client = session.client('bedrock', region_name=default_region)
                    bedrock_client.list_foundation_models()
                    
                    # If we get here, credentials are valid
                    st.success("✅ Credentials validated")
                    
                    # Proceed with update (region is handled internally by the scalable approach)
                    profile_name = selected_profile if selected_profile != 'default' else None
                    update_models_from_aws(
                        profile_name=profile_name,
                        region=default_region
                    )
                    
                except ProfileNotFound:
                    st.error(f"❌ Profile '{selected_profile}' not found")
                except ClientError as e:
                    st.error(f"❌ AWS credentials error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")
        
        st.divider()
        
        # Initialize selection state
        if 'selected_models' not in st.session_state:
            st.session_state.selected_models = set()
        
        # Display current selection summary
        if st.session_state.selected_models:
            st.success(f"✅ {len(st.session_state.selected_models)} models selected for comparison")
    
    # We no longer need to filter the dataframe here since filtering is now done in the model explorer page
    filtered_df = df
    
    # === NAVIGATION ===
    # Radio button navigation - compact and left-aligned
    selected_page = st.radio(
        "Navigation Menu",
        options=["🏠 Model Explorer", "⭐ Favorites", "⚖️ Comparison"],
        horizontal=True,
        key="main_navigation",
        label_visibility="collapsed"
    )
    
    # Display content based on selection
    if selected_page == "🏠 Model Explorer":
            if not df.empty:
                # Add overview boxes before search
                show_overview_boxes(df)
                show_model_explorer(filtered_df)
            else:
                st.markdown("""
                    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 12px; margin: 2rem 0;">
                        <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
                        <h2 style="color: #1e293b; margin-bottom: 1rem;">Welcome to Amazon Bedrock Model Profiler!</h2>
                        <p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">No model data found. This appears to be your first time using the application.</p>
                        <div style="background: white; padding: 1.5rem; border-radius: 8px; text-align: left; max-width: 600px; margin: 0 auto;">
                            <h4 style="color: #374151; margin-bottom: 1rem;">How to get started:</h4>
                            <div style="color: #6b7280;">
                                <p>1. Look at the sidebar on the left</p>
                                <p>2. Expand "<strong>⚙️ AWS Setup & Update</strong>"</p>
                                <p>3. Click "<strong>🔄 Update models database</strong>"</p>
                            </div>
                        </div>
                        <p style="color: #64748b; font-size: 1.1rem; margin-top: 2rem;">The system will automatically collect comprehensive model data from AWS Bedrock API.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
    elif selected_page == "⭐ Favorites":
            if not df.empty:
                from utils.favorites import get_favorite_models, initialize_favorites, clear_favorites, get_favorites
                initialize_favorites()
                favorites_df = get_favorite_models(df)
                
                # Add overview boxes for favorites
                if not favorites_df.empty:
                    show_overview_boxes(favorites_df, is_favorites=True)
                
                # Add favorites management controls
                if not favorites_df.empty:
                    # Add clear favorites button (full width)
                    if st.button("🗑️ Clear All Favorites", type="secondary", width="stretch"):
                        clear_favorites()
                        st.success("All favorites have been cleared!")
                        st.rerun()
                    
                    # Show the favorites WITHOUT is_favorites_view to avoid duplicate keys
                    show_model_explorer(favorites_df)
                    
                else:
                    st.markdown("""
                        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 12px; margin: 2rem 0;">
                            <div style="font-size: 4rem; margin-bottom: 1rem;">⭐</div>
                            <h2 style="color: #1e293b; margin-bottom: 1rem;">No Favorites Yet</h2>
                            <div style="background: white; padding: 1.5rem; border-radius: 8px; text-align: left; max-width: 600px; margin: 0 auto;">
                                <h4 style="color: #374151; margin-bottom: 1rem;">How to add favorites:</h4>
                                <div style="color: #6b7280;">
                                    <p>1. Go to the <strong>Model Explorer section</strong></p>
                                    <p>2. Browse through the available models</p>
                                    <p>3. Click the ⭐ button on any model card to add it to your favorites</p>
                                    <p>4. Return to this tab to see all your favorite models in one place</p>
                                </div>
                            </div>
                            <div style="background: white; padding: 1.5rem; border-radius: 8px; text-align: left; max-width: 600px; margin: 0 auto;">
                                <h4 style="color: #374151; margin-bottom: 1rem;">Benefits of using Favorites:</h4>
                                <div style="color: #6b7280;">
                                    <p>- Quick access to models you use frequently</p>
                                    <p>- Easy comparison of your preferred models</p>
                                    <p>- Persistent across sessions - your favorites will be saved even when you close the browser</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # Enhanced no-data guidance for Favorites
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;">
                    <h2 style="color: white; margin: 0;">⭐ Favorites Section</h2>
                    <p style="color: white; margin: 1rem 0;">Model data needed to use favorites</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                ### 📋 How Favorites Work:

                **Once you have model data:**
                1. Browse models in the **Model Explorer** tab
                2. Click the ⭐ button on model cards to save favorites
                3. Return here to see your curated collection

                **Current Status:** No model data available yet.

                **Next Step:** Please collect model data first using the sidebar ⬅️
                """)

                if st.button("🔄 Go to Model Explorer", type="secondary", use_container_width=True):
                    st.session_state.selected_page = "🏠 Model Explorer"
                    st.rerun()
        
    elif selected_page == "⚖️ Comparison":
        show_model_comparison(df)

if __name__ == "__main__":
    main()