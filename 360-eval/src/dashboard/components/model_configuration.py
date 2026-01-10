"""Model configuration component for the Streamlit dashboard."""

import streamlit as st
import pandas as pd
from ..utils.constants import (
    DEFAULT_BEDROCK_MODELS,
    DEFAULT_OPENAI_MODELS,
    DEFAULT_COST_MAP,
    DEFAULT_JUDGES_COST,
    DEFAULT_JUDGES,
    AWS_REGIONS,
    MODEL_TO_REGIONS,
    REGION_TO_MODELS,
    JUDGE_MODEL_TO_REGIONS,
    JUDGE_REGION_TO_MODELS
)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))
from model_capability_validator import (
    get_available_service_tiers,
    get_model_capabilities,
    is_cache_valid,
    load_capability_cache,
    validate_all_models,
    CACHE_FILE
)
from ..utils.state_management import save_current_evaluation


class ModelConfigurationComponent:
    """Component for configuring models and judge models."""
    
    def __init__(self):
        # Initialize session state for model/region filtering
        if 'selected_bedrock_model' not in st.session_state:
            st.session_state.selected_bedrock_model = None
        if 'filtered_regions' not in st.session_state:
            st.session_state.filtered_regions = list(REGION_TO_MODELS.keys()) if REGION_TO_MODELS else AWS_REGIONS
        if 'filtered_models' not in st.session_state:
            st.session_state.filtered_models = {}
    
    def _on_region_change(self):
        """Handle region selection change and auto-correct model selection."""
        selected_region = st.session_state.aws_region
        if selected_region in REGION_TO_MODELS:
            available_models = REGION_TO_MODELS[selected_region]
            # Auto-select first available model if current selection is invalid
            if st.session_state.selected_bedrock_model not in available_models and available_models:
                st.session_state.selected_bedrock_model = available_models[0]
            st.session_state.filtered_models = available_models
    
    def _on_model_change(self):
        """Handle model selection change and auto-correct region selection."""
        selected_model = st.session_state.bedrock_model_select
        if selected_model in MODEL_TO_REGIONS:
            available_regions = MODEL_TO_REGIONS[selected_model]
            # Auto-select first available region if current selection is invalid
            if st.session_state.aws_region not in available_regions and available_regions:
                st.session_state.aws_region = available_regions[0]
            st.session_state.filtered_regions = available_regions
    
    def render(self):
        """Render the model configuration component."""

        # Check if model capability validation is needed
        self._render_validation_banner()

        # Determine available regions based on selected model
        available_regions = st.session_state.filtered_regions if hasattr(st.session_state, 'filtered_regions') else list(REGION_TO_MODELS.keys())
        if not available_regions:
            available_regions = AWS_REGIONS
        
        # Region selection with dynamic filtering
        selected_region = st.selectbox(
            "AWS Region",
            options=available_regions,
            index=0 if available_regions else 0,
            key="aws_region",
            on_change=self._on_region_change
        )
        
        # Available models tabs (Bedrock, OpenAI)
        tab1, tab2 = st.tabs(["Bedrock Models", "Other Models"])
        
        with tab1:
            # Get models available in selected region
            if selected_region in REGION_TO_MODELS:
                bedrock_models = REGION_TO_MODELS[selected_region]
            else:
                bedrock_models = [model[0] for model in DEFAULT_BEDROCK_MODELS]
            self._render_model_dropdown(bedrock_models, "bedrock", selected_region)
        
        with tab2:
            openai_models = [model[0] for model in DEFAULT_OPENAI_MODELS]
            self._render_model_dropdown(openai_models, "openai", selected_region)
        
        # Selected models display
        st.subheader("Selected Models")
        if not st.session_state.current_evaluation_config["selected_models"]:
            st.info("No models selected. Please select at least one model to evaluate.")
        else:
            selected_models_df = pd.DataFrame(st.session_state.current_evaluation_config["selected_models"])

            # Build rename mapping based on what columns exist
            rename_map = {
                "id": "Model ID",
                "region": "AWS Region",
                "input_cost": "Input Cost (per token)",
                "output_cost": "Output Cost (per token)",
                "target_rpm": "Target RPM"
            }

            # Add service_tier to rename map if it exists
            if "service_tier" in selected_models_df.columns:
                rename_map["service_tier"] = "Service Tier"

            selected_models_df = selected_models_df.rename(columns=rename_map)

            # Replace None/NaN in Target RPM with "No Limit"
            if "Target RPM" in selected_models_df.columns:
                selected_models_df["Target RPM"] = selected_models_df["Target RPM"].fillna("No Limit")

            # Replace None/NaN in Service Tier with "default"
            if "Service Tier" in selected_models_df.columns:
                selected_models_df["Service Tier"] = selected_models_df["Service Tier"].fillna("default")

            # Drop service_tier_label column if it exists (redundant - used only for display suffix)
            columns_to_drop = ["service_tier_label"]
            selected_models_df = selected_models_df.drop(columns=[col for col in columns_to_drop if col in selected_models_df.columns], errors='ignore')

            st.dataframe(selected_models_df, hide_index=True)
            
            # Button to remove all selected models
            st.button(
                "Clear Selected Models",
                on_click=self._clear_selected_models
            )
        
        # Judge model selection
        # Read from widget key directly for immediate reactivity
        is_latency_only = st.session_state.get("latency_only_mode", False)

        st.subheader("Judge Models" + (" (Not used in latency-only mode)" if is_latency_only else ""))

        if is_latency_only:
            st.info("ℹ️ Judge models are not needed for latency-only evaluations. Only performance metrics will be collected.")

        self._render_judge_selection(selected_region, disabled=is_latency_only)

        # If we have selected judge models, display them
        if st.session_state.current_evaluation_config["judge_models"]:
            judge_models_df = pd.DataFrame(st.session_state.current_evaluation_config["judge_models"])
            judge_models_df = judge_models_df.rename(columns={
                "id": "Model ID",
                "region": "AWS Region",
                "input_cost": "Input Cost (per token)",
                "output_cost": "Output Cost (per token)"
            })
            st.dataframe(judge_models_df, hide_index=True)

            # Button to remove all judge models
            st.button(
                "Clear Judge Models",
                on_click=self._clear_judge_models,
                key="clear_judges",
                disabled=is_latency_only
            )
        
        # Show validation status
        is_valid = self._is_configuration_valid()
        missing_items = self._get_missing_configuration_items()
        
        if not is_valid and missing_items:
            st.warning(f"Please complete the following before saving: {', '.join(missing_items)}")
        
        # Action buttons - only save and reset, no direct run
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(
                "Save Configuration",
                disabled=not is_valid,
            ):
                save_current_evaluation()
                st.success(f"Configuration profile saved successfully!")
                # Debug information
                print(f"Saved configuration to session state. Total evaluations: {len(st.session_state.evaluations)}")
                print(f"Evaluation IDs: {[e['id'] for e in st.session_state.evaluations]}")
        
        with col2:
            st.button(
                "Reset Configuration",
                on_click=self._reset_configuration
            )
    
    def _render_model_dropdown(self, model_list, prefix, region):
        """Render the model selection UI with dropdown."""
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])

        with col1:
            if prefix == "bedrock":
                # Check availability for each Bedrock model and annotate
                annotated_models = []
                model_availability = {}  # Store availability status

                for model_id in model_list if model_list else []:
                    capabilities = get_model_capabilities(model_id, region)

                    if capabilities and capabilities.get("available"):
                        # Model is available
                        annotated_models.append(model_id)
                        model_availability[model_id] = {
                            "available": True,
                            "error": None,
                            "service_tiers": capabilities.get("service_tiers", ["default"])
                        }
                    elif capabilities and not capabilities.get("available"):
                        # Model is unavailable (tested and failed)
                        unavailable_label = f"⚠️ {model_id} (unavailable)"
                        annotated_models.append(unavailable_label)
                        model_availability[unavailable_label] = {
                            "available": False,
                            "error": capabilities.get("error", "Not available in this region"),
                            "service_tiers": [],
                            "original_id": model_id
                        }
                    else:
                        # No cache data available
                        annotated_models.append(model_id)
                        model_availability[model_id] = {
                            "available": None,  # Unknown
                            "error": "Not validated",
                            "service_tiers": ["default"]
                        }

                # Store availability map in session state for later use
                st.session_state[f"{prefix}_model_availability"] = model_availability

                # For Bedrock models, add on_change callback
                selected_model = st.selectbox(
                    "Select Model",
                    options=annotated_models if annotated_models else ["No models available in this region"],
                    key=f"{prefix}_model_select",
                    on_change=self._on_model_change if annotated_models else None,
                    disabled=not annotated_models,
                    help="Models marked with ⚠️ are not available in the selected region"
                )
            else:
                # For non-Bedrock models, no region filtering
                selected_model = st.selectbox(
                    "Select Model",
                    options=model_list,
                    key=f"{prefix}_model_select"
                )
                model_availability = {}

        # Get default costs
        default_input_cost = DEFAULT_COST_MAP.get(selected_model, {"input": 0.001, "output": 0.002})["input"]
        default_output_cost = DEFAULT_COST_MAP.get(selected_model, {"input": 0.001, "output": 0.002})["output"]

        with col2:
            input_cost = st.number_input(
                "Input Cost",
                min_value=0.0,
                max_value=1.0,
                value=default_input_cost,
                step=0.0001,
                format="%.6f",
                key=f"{prefix}_input_cost"
            )

        with col3:
            output_cost = st.number_input(
                "Output Cost",
                min_value=0.0,
                max_value=1.0,
                value=default_output_cost,
                step=0.0001,
                format="%.6f",
                key=f"{prefix}_output_cost"
            )

        with col4:
            # Service tier dropdown for Bedrock models
            if prefix == "bedrock":
                # Get available tiers for this specific model+region
                available_tiers = get_available_service_tiers(selected_model, region)

                if available_tiers and len(available_tiers) > 0:
                    selected_tier = st.selectbox(
                        "Service Tier",
                        options=available_tiers,
                        key=f"{prefix}_service_tier_select",
                        help="Select the service tier for this model. You can add the same model multiple times with different tiers."
                    )
                else:
                    # Fallback to default if no tiers available
                    selected_tier = st.selectbox(
                        "Service Tier",
                        options=["default"],
                        key=f"{prefix}_service_tier_select",
                        help="Service tier for this model"
                    )
            else:
                # Non-Bedrock models - show Target RPM in col4
                target_rpm = st.number_input(
                    "Target RPM",
                    min_value=0,
                    max_value=600,
                    value=0,
                    step=10,
                    key=f"{prefix}_target_rpm",
                    help="Requests per minute (0 = no rate limiting). Use to test model reliability at specific load levels."
                )
                # Convert 0 to None for storage (0 means no rate limiting)
                target_rpm = target_rpm if target_rpm > 0 else None

        with col5:
            # For Bedrock models, show Target RPM in col5; for others, show Add button
            if prefix == "bedrock":
                target_rpm = st.number_input(
                    "Target RPM",
                    min_value=0,
                    max_value=600,
                    value=0,
                    step=10,
                    key=f"{prefix}_target_rpm",
                    help="Requests per minute (0 = no rate limiting). Use to test model reliability at specific load levels."
                )
                # Convert 0 to None for storage (0 means no rate limiting)
                target_rpm = target_rpm if target_rpm > 0 else None

            # Check if model is unavailable (for Bedrock models)
            model_info = model_availability.get(selected_model, {}) if prefix == "bedrock" else {}
            is_unavailable = model_info.get("available") == False

            # Get the selected tier for Bedrock models
            selected_tier = st.session_state.get(f"{prefix}_service_tier_select", "default") if prefix == "bedrock" else None

            st.button(
                "Add Model",
                key=f"{prefix}_add_model",
                on_click=self._add_model,
                args=(selected_model, region, input_cost, output_cost, target_rpm, prefix, selected_tier),
                disabled=is_unavailable,
                help="This model is not available in the selected region" if is_unavailable else "Add this model to your evaluation"
            )

        # Show availability status for Bedrock models
        if prefix == "bedrock" and model_availability:
            model_info = model_availability.get(selected_model, {})

            if model_info.get("available") == False:
                # Model is unavailable
                error_msg = model_info.get("error", "Not available in this region")
                st.error(f"⚠️ **Model Unavailable:** {error_msg}")
            elif model_info.get("available") == None:
                # Model not validated
                st.info("ℹ️ **Not Validated:** This model hasn't been validated yet. Run validation to check availability and service tier support.")
            elif model_info.get("available") == True:
                # Model is available - show service tier count
                tiers = model_info.get("service_tiers", [])
                if len(tiers) > 1:
                    st.success(f"✅ **Available** with {len(tiers)} service tiers: {', '.join(tiers)}")
                else:
                    st.success(f"✅ **Available** (default tier only)")

    
    def _render_judge_selection(self, region, disabled=False):
        """Render the judge model selection UI."""
        # Ignore the passed region parameter - use judge's own regions from config
        judge_options = [m[0] for m in DEFAULT_JUDGES]
        judge_regions = {m[0]: m[1] for m in DEFAULT_JUDGES}
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

        with col1:
            selected_judge = st.selectbox(
                "Select Judge Model",
                options=judge_options,
                key="judge_model_select",
                disabled=disabled
            )
        
        # Handle case where selectbox returns index instead of value
        if isinstance(selected_judge, int):
            selected_judge = judge_options[selected_judge] if selected_judge < len(judge_options) else judge_options[0]
        
        # Get default costs and region from judge config
        default_input_cost = DEFAULT_JUDGES_COST.get(selected_judge, {"input": 0.001, "output": 0.002})["input"]
        default_output_cost = DEFAULT_JUDGES_COST.get(selected_judge, {"input": 0.001, "output": 0.002})["output"]
        # Use the judge's predefined region from the config file
        judge_region = judge_regions.get(selected_judge, "us-east-1")
        with col2:
            judge_input_cost = st.number_input(
                "Input Cost",
                min_value=0.0,
                max_value=1.0,
                value=default_input_cost,
                step=0.0001,
                format="%.6f",
                key="judge_input_cost",
                disabled=disabled
            )

        with col3:
            judge_output_cost = st.number_input(
                "Output Cost",
                min_value=0.0,
                max_value=1.0,
                value=default_output_cost,
                step=0.0001,
                format="%.6f",
                key="judge_output_cost",
                disabled=disabled
            )

        with col4:
            st.button(
                "Add Judge",
                key="add_judge",
                on_click=self._add_judge_model,
                args=(selected_judge, judge_region, judge_input_cost, judge_output_cost),
                disabled=disabled
            )
    
    def _add_model(self, model_id, region, input_cost, output_cost, target_rpm=None, prefix="bedrock", selected_tier=None):
        """Add a model to the selected models list.

        For Bedrock models, adds the model with the specified service tier.
        """
        # For Bedrock models, use the selected tier; for others, set to None
        if prefix == "bedrock":
            tier = selected_tier if selected_tier else "default"
            # Create label suffix for display purposes
            tier_label = f"_{tier}" if tier != "default" else ""
        else:
            tier = None
            tier_label = None

        # Check if this model+region+tier combination already exists
        existing_model = None
        for model in st.session_state.current_evaluation_config["selected_models"]:
            if (model["id"] == model_id and
                model.get("region", "") == region and
                model.get("service_tier") == tier):
                existing_model = model
                break

        if existing_model:
            # Update existing model entry
            existing_model["input_cost"] = input_cost
            existing_model["output_cost"] = output_cost
            existing_model["target_rpm"] = target_rpm
        else:
            # Add new model entry
            st.session_state.current_evaluation_config["selected_models"].append({
                "id": model_id,
                "region": region,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "target_rpm": target_rpm,
                "service_tier": tier,
                "service_tier_label": tier_label
            })
    
    def _add_judge_model(self, model_id, region, input_cost, output_cost):
        """Add a judge model to the judge models list."""
        # Check if model is already selected with same region
        for model in st.session_state.current_evaluation_config["judge_models"]:
            # Check if the model ID matches and either region matches or isn't present
            if model["id"] == model_id and model.get("region", "") == region:
                # Update costs and region if model already exists
                model["input_cost"] = input_cost
                model["output_cost"] = output_cost
                model["region"] = region
                return
        
        # Add new model
        st.session_state.current_evaluation_config["judge_models"].append({
            "id": model_id,
            "region": region,
            "input_cost": input_cost,
            "output_cost": output_cost
        })
    
    def _clear_selected_models(self):
        """Clear all selected models."""
        st.session_state.current_evaluation_config["selected_models"] = []
    
    def _clear_judge_models(self):
        """Clear all judge models."""
        st.session_state.current_evaluation_config["judge_models"] = []
    
    def _reset_configuration(self):
        """Reset the current configuration to default values."""
        # Keep CSV data and column selections, reset everything else
        csv_data = st.session_state.current_evaluation_config["csv_data"]
        prompt_column = st.session_state.current_evaluation_config["prompt_column"]
        golden_answer_column = st.session_state.current_evaluation_config["golden_answer_column"]
        
        st.session_state.current_evaluation_config = {
            "id": None,
            "name": f"Benchmark-{pd.Timestamp.now().strftime('%Y%m%d')}",
            "csv_data": csv_data,
            "prompt_column": prompt_column,
            "golden_answer_column": golden_answer_column,
            "task_type": "",
            "task_criteria": "",
            "output_dir": st.session_state.current_evaluation_config["output_dir"],
            "parallel_calls": st.session_state.current_evaluation_config["parallel_calls"],
            "invocations_per_scenario": st.session_state.current_evaluation_config["invocations_per_scenario"],
            "sleep_between_invocations": st.session_state.current_evaluation_config["sleep_between_invocations"],
            "experiment_counts": st.session_state.current_evaluation_config["experiment_counts"],
            "temperature_variations": st.session_state.current_evaluation_config["temperature_variations"],
            "failure_threshold": st.session_state.current_evaluation_config["failure_threshold"],
            "selected_models": [],
            "judge_models": [],
            "user_defined_metrics": "",
            "status": "configuring",
            "progress": 0,
            "created_at": None,
            "updated_at": None,
            "results": None
        }
    
    def _get_missing_configuration_items(self):
        """Get a list of missing configuration items."""
        config = st.session_state.current_evaluation_config
        missing_items = []
        
        # Check for CSV data with prompt and golden answer columns
        if config["csv_data"] is None:
            missing_items.append("CSV data")
        elif not config["prompt_column"] or not config["golden_answer_column"]:
            missing_items.append("prompt and golden answer column selection")
        
        # Check if latency-only mode is enabled
        is_latency_only = config.get("latency_only_mode", False)

        # Check for task type and criteria (support both old and new format)
        # Skip task type and criteria validation in latency-only mode
        if not is_latency_only:
            task_evaluations = config.get("task_evaluations", [])
            if task_evaluations:
                # New format: check each task evaluation
                for i, task_eval in enumerate(task_evaluations):
                    if not task_eval.get("task_type", "").strip():
                        missing_items.append(f"task type for evaluation {i+1}")
                    if not task_eval.get("task_criteria", "").strip():
                        missing_items.append(f"task criteria for evaluation {i+1}")
            else:
                # Fallback to old format for backward compatibility
                if not config.get("task_type", "").strip():
                    missing_items.append("task type")
                if not config.get("task_criteria", "").strip():
                    missing_items.append("task criteria")

        # Check for at least one target model
        if not config["selected_models"]:
            missing_items.append("at least one target model")

        # Check for at least one judge model (only in full 360 mode)
        if not is_latency_only and not config["judge_models"]:
            missing_items.append("at least one judge model")
        
        return missing_items
    
    def _is_configuration_valid(self):
        """Check if the current configuration is valid."""
        return len(self._get_missing_configuration_items()) == 0

    def _render_validation_banner(self):
        """Render validation status banner for model capabilities."""
        cache_exists = CACHE_FILE.exists()
        cache_valid = is_cache_valid() if cache_exists else False

        # Determine banner state
        if not cache_exists:
            banner_type = "warning"
            banner_title = "⚠️ Model Capabilities Not Validated"
            banner_message = "Service tier availability hasn't been validated. Run validation to see which tiers are available for each model."
            show_banner = True
        elif not cache_valid:
            banner_type = "info"
            banner_title = "ℹ️ Model Configuration Changed"
            banner_message = "Your models_profiles.jsonl has changed since last validation. Re-validate to update service tier availability."
            show_banner = True
        else:
            # Cache is valid, show compact success message
            cache = load_capability_cache()
            last_updated = cache.get("last_updated", "Unknown")
            st.success(f"✅ Model capabilities validated (Last updated: {last_updated})")
            show_banner = False

        if show_banner:
            # Show expandable validation banner
            with st.expander(banner_title, expanded=True):
                if banner_type == "warning":
                    st.warning(banner_message)
                else:
                    st.info(banner_message)

                st.markdown("""
                **What validation does:**
                - Tests each model+region combination with a tiny API request
                - Discovers which service tiers (default, priority, flex) are available
                - Caches results to avoid repeated calls
                - Cost: ~$0.001 for full validation
                """)

                col1, col2 = st.columns([1, 3])

                with col1:
                    if st.button("🔍 Validate Now", key="validate_models_button", help="Run validation (takes 2-3 minutes)"):
                        with st.spinner("⏳ Validating models... This may take 2-3 minutes"):
                            try:
                                # Run validation
                                cache = validate_all_models(force=True)

                                # Show success with summary
                                capabilities = cache.get("capabilities", {})
                                total_combos = sum(len(regions) for regions in capabilities.values())
                                available_combos = sum(
                                    1 for regions in capabilities.values()
                                    for caps in regions.values()
                                    if caps.get("available")
                                )

                                st.success(f"✅ Validation complete! {available_combos}/{total_combos} model+region combinations available.")
                                st.info("💡 Refresh the page to see updated service tier options.")

                                # Offer to rerun the app
                                if st.button("🔄 Refresh Dashboard", key="refresh_after_validation"):
                                    st.rerun()

                            except Exception as e:
                                st.error(f"❌ Validation failed: {str(e)}")
                                st.info("Check logs for details. You can also run validation manually: `python src/validate_model_capabilities.py`")

                with col2:
                    st.info("**Alternative:** Run validation via command line:\n```bash\npython src/validate_model_capabilities.py\n```")

