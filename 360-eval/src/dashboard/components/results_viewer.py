"""Results viewer component for the Streamlit dashboard."""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from ..utils.benchmark_runner import sync_evaluations_from_files
from ..utils.constants import DEFAULT_OUTPUT_DIR, PROJECT_ROOT

class ResultsViewerComponent:
    """Component for viewing evaluation results."""

    def _count_unprocessed_records(self, eval_name):
        """Count unprocessed records for a given evaluation name."""
        try:
            unprocessed_dir = Path(PROJECT_ROOT) / DEFAULT_OUTPUT_DIR / "unprocessed"
            if not unprocessed_dir.exists():
                return 0

            # Look for unprocessed files matching this evaluation name
            # New format: unprocessed_<experiment_name>_<timestamp>_<uuid>.json
            matching_files = list(unprocessed_dir.glob(f"unprocessed_{eval_name}_*.json"))

            total_count = 0
            for file_path in matching_files:
                try:
                    with open(file_path, 'r') as f:
                        records = json.load(f)
                        total_count += len(records)
                except:
                    pass  # Skip files that can't be parsed

            return total_count
        except:
            return 0

    def _display_rpm_metrics(self, eval_config, eval_id):
        """Display RPM (Requests Per Minute) metrics if available."""
        try:
            # Find the CSV results file
            output_dir = Path(DEFAULT_OUTPUT_DIR)
            eval_name = eval_config.get("name", "")
            composite_id = f"{eval_id}_{eval_name}"

            # Look for CSV files
            csv_files = list(output_dir.glob(f"invocations_*{composite_id}*.csv"))
            if not csv_files:
                # Fallback to original name pattern
                csv_files = list(output_dir.glob(f"*{eval_name}*.csv"))

            if not csv_files:
                return  # No CSV file found, skip RPM metrics

            # Use the latest CSV file
            latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)

            # Read the CSV file
            df = pd.read_csv(latest_csv)

            # Check if RPM metrics columns exist
            rpm_columns = ['target_rpm', 'actual_rpm', 'throttled', 'throttle_wait_time', 'throttle_events_count']
            has_rpm_metrics = any(col in df.columns for col in rpm_columns)

            if not has_rpm_metrics:
                return  # No RPM metrics in this evaluation

            # Check if any models had RPM configured
            if 'target_rpm' not in df.columns or df['target_rpm'].isna().all():
                return  # No models with RPM configured

            st.divider()
            st.write("#### ⚡ Rate Limiting & Reliability Metrics")

            # Group by model_id and region to calculate per-model metrics
            if 'model_id' in df.columns:
                # Filter to only models with target_rpm configured
                rpm_df = df[df['target_rpm'].notna() & (df['target_rpm'] > 0)].copy()

                if len(rpm_df) == 0:
                    st.info("No models were configured with rate limiting (target RPM) in this evaluation.")
                    return

                # Group by model_id and region
                group_cols = ['model_id']
                if 'region' in df.columns:
                    group_cols.append('region')

                metrics_list = []
                for group_key, group_df in rpm_df.groupby(group_cols):
                    if isinstance(group_key, tuple):
                        model_id, region = group_key
                    else:
                        model_id = group_key
                        region = group_df['region'].iloc[0] if 'region' in group_df.columns else 'N/A'

                    # Calculate metrics
                    target_rpm = group_df['target_rpm'].iloc[0] if 'target_rpm' in group_df.columns else 0
                    actual_rpm = group_df['actual_rpm'].mean() if 'actual_rpm' in group_df.columns else 0
                    total_requests = len(group_df)

                    # Throttle metrics
                    if 'throttled' in group_df.columns:
                        throttled_count = group_df['throttled'].sum() if pd.api.types.is_bool_dtype(group_df['throttled']) else (group_df['throttled'] == True).sum()
                    else:
                        throttled_count = 0

                    if 'throttle_wait_time' in group_df.columns:
                        total_wait_time = group_df['throttle_wait_time'].sum()
                        avg_wait_time = group_df[group_df['throttle_wait_time'] > 0]['throttle_wait_time'].mean() if (group_df['throttle_wait_time'] > 0).any() else 0
                    else:
                        total_wait_time = 0
                        avg_wait_time = 0

                    if 'throttle_events_count' in group_df.columns:
                        total_throttle_events = group_df['throttle_events_count'].iloc[0] if len(group_df) > 0 else 0
                    else:
                        total_throttle_events = throttled_count

                    # Success/Error rate
                    if 'api_call_status' in group_df.columns:
                        success_count = (group_df['api_call_status'] == 'success').sum()
                        error_count = (group_df['api_call_status'] != 'success').sum()
                    else:
                        # Try to infer from other columns
                        success_count = total_requests  # Assume all succeeded if no status column
                        error_count = 0

                    success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0
                    error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0

                    metrics_list.append({
                        'Model ID': model_id,
                        'Region': region,
                        'Target RPM': int(target_rpm) if target_rpm else 0,
                        'Actual RPM': f"{actual_rpm:.1f}" if actual_rpm else "N/A",
                        'Total Requests': total_requests,
                        'Throttle Events': int(total_throttle_events),
                        'Total Wait Time (s)': f"{total_wait_time:.2f}",
                        'Avg Wait Time (s)': f"{avg_wait_time:.3f}" if avg_wait_time > 0 else "0.000",
                        'Success Rate': f"{success_rate:.1f}%",
                        'Error Rate': f"{error_rate:.1f}%"
                    })

                if metrics_list:
                    metrics_df = pd.DataFrame(metrics_list)
                    st.dataframe(metrics_df, width='stretch', hide_index=True)

                    # Add explanation
                    with st.expander("ℹ️ About Rate Limiting Metrics"):
                        st.markdown("""
                        **Rate limiting metrics help you understand model reliability at specific load levels:**

                        - **Target RPM**: The configured requests per minute limit for testing
                        - **Actual RPM**: The actual average rate achieved during the evaluation
                        - **Throttle Events**: Number of times requests were delayed to maintain the target RPM
                        - **Wait Time**: Total and average time spent waiting due to rate limiting
                        - **Success/Error Rates**: Reliability of the model at the configured RPM

                        Use these metrics to identify the optimal RPM for production workloads and detect throttling thresholds.
                        """)
                else:
                    st.info("RPM metrics are being collected but no data is available yet.")

        except Exception as e:
            # Silently fail - RPM metrics are optional
            st.warning(f"Could not load RPM metrics: {str(e)}")

    def render(self):
        """Render the results viewer component."""
        # Sync evaluation statuses from files
        sync_evaluations_from_files()
        
        st.subheader("Completed Evaluations")
        
        # Check if there are any completed evaluations
        completed_evals = [
            e for e in st.session_state.evaluations 
            if e["status"] == "completed"
        ]
        
        if not completed_evals:
            st.info("No completed evaluations yet. Run evaluations to see results here.")
        else:
            # Create a table of completed evaluations
            eval_data = []
            for eval_config in completed_evals:
                # Get model and judge details for display
                models_info = eval_config.get("selected_models", [])
                judges_info = eval_config.get("judge_models", [])
                
                # Create model summary
                if isinstance(models_info, list) and len(models_info) > 0:
                    if isinstance(models_info[0], dict):
                        models_summary = f"{len(models_info)} models"
                        models_details = ", ".join([m.get("model_id", "Unknown") for m in models_info])
                    else:
                        models_summary = f"{len(models_info)} models"
                        models_details = ", ".join(models_info)
                else:
                    models_summary = "0 models"
                    models_details = "None"
                
                # Create judge summary
                if isinstance(judges_info, list) and len(judges_info) > 0:
                    if isinstance(judges_info[0], dict):
                        judges_summary = f"{len(judges_info)} judges"
                        judges_details = ", ".join([j.get("model_id", "Unknown") for j in judges_info])
                    else:
                        judges_summary = f"{len(judges_info)} judges"
                        judges_details = ", ".join(judges_info)
                else:
                    judges_summary = "0 judges"
                    judges_details = "None"
                
                # Extract file name from persistent storage or CSV data
                csv_file_name = "Unknown"
                
                # First check if we have the persisted file name (from status files)
                if eval_config.get("csv_file_name"):
                    csv_file_name = eval_config.get("csv_file_name")
                # Fallback to CSV data if available (for active session)
                elif eval_config.get("csv_data") is not None:
                    csv_file_name = eval_config.get("csv_file_name", "Uploaded CSV")
                elif hasattr(eval_config.get("csv_data"), "name"):
                    csv_file_name = eval_config["csv_data"].name
                
                # Get temperature used
                temperature = eval_config.get("temperature", "Not specified")
                
                # Check if custom metrics were used
                user_metrics = eval_config.get("user_defined_metrics", "")
                has_custom_metrics = "Yes" if user_metrics and user_metrics.strip() else "No"
                
                eval_data.append({
                    "Name": eval_config["name"],
                    "Task Type": eval_config["task_type"],
                    "Data File": csv_file_name,
                    "Temperature": temperature,
                    "Custom Metrics": has_custom_metrics,
                    "Models": models_summary,
                    "Judges": judges_summary,
                    "Completed": pd.to_datetime(eval_config["updated_at"]).strftime("%Y-%m-%d %H:%M")
                })
            
            eval_df = pd.DataFrame(eval_data)
            st.dataframe(eval_df, hide_index=True)
            
            # Add refresh button
            st.button(
                "Refresh Results",
                on_click=sync_evaluations_from_files
            )
            
            # Select an evaluation to view results
            selected_eval_id = st.selectbox(
                "Select evaluation to view results",
                options=[e["id"] for e in completed_evals],
                format_func=lambda x: next((e["name"] for e in completed_evals if e["id"] == x), x)
            )
            
            if selected_eval_id:
                self._show_evaluation_results(selected_eval_id)
    
    def _show_evaluation_results(self, eval_id):
        """Show detailed results for a specific evaluation."""
        # First try to find status file for the most up-to-date information (now in logs directory)
        from ..utils.constants import STATUS_FILES_DIR
        status_dir = Path(STATUS_FILES_DIR)
        
        # Try both composite and legacy formats for status file
        status_file = None
        # First try to find composite format by looking at evaluation name
        for eval_config in st.session_state.evaluations:
            if eval_config["id"] == eval_id:
                eval_name = eval_config.get("name", "")
                composite_id = f"{eval_id}_{eval_name}"
                composite_status_file = status_dir / f"eval_{composite_id}_status.json"
                if composite_status_file.exists():
                    status_file = composite_status_file
                    break
        
        # Fallback to legacy format
        if not status_file:
            legacy_status_file = status_dir / f"eval_{eval_id}_status.json"
            if legacy_status_file.exists():
                status_file = legacy_status_file
        
        
        # Find the evaluation configuration
        eval_config = None
        for e in st.session_state.evaluations:
            if e["id"] == eval_id:
                eval_config = e
                break
        
        if not eval_config:
            st.error("Evaluation not found")
            return
        
        # Display evaluation details
        st.subheader(f"📊 {eval_config['name']}")

        # Check for unprocessed records and show warning if any exist
        unprocessed_count = self._count_unprocessed_records(eval_config['name'])
        if unprocessed_count > 0:
            col_warn1, col_warn2 = st.columns([3, 1])
            with col_warn1:
                st.warning(f"⚠️ This evaluation has **{unprocessed_count}** unprocessed (failed) records. "
                          f"Click the button to view details and troubleshoot.")
            with col_warn2:
                if st.button("View Failed Records", key=f"view_unprocessed_{eval_id}"):
                    st.session_state.nav_radio = "Unprocessed"
                    st.rerun()

        # Display basic info in columns for better layout
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Task Type:** {eval_config.get('task_type', 'Unknown')}")
            st.write(f"**Task Criteria:** {eval_config.get('task_criteria', 'Unknown')}")
            st.write(f"**Status:** {eval_config.get('status', 'Unknown')}")
            
            # Display data file name
            data_file = eval_config.get('csv_file_name', 'Unknown')
            st.write(f"**Data File:** {data_file}")
            
        with col2:
            st.write(f"**Created:** {pd.to_datetime(eval_config.get('created_at', '')).strftime('%Y-%m-%d %H:%M') if eval_config.get('created_at') else 'Unknown'}")
            st.write(f"**Completed:** {pd.to_datetime(eval_config.get('updated_at', '')).strftime('%Y-%m-%d %H:%M') if eval_config.get('updated_at') else 'Unknown'}")
            if eval_config.get('duration'):
                st.write(f"**Duration:** {eval_config['duration']:.1f} seconds")
            
            # Display temperature and custom metrics
            temperature = eval_config.get('temperature', 'Not specified')
            st.write(f"**Temperature:** {temperature}")
            
            user_metrics = eval_config.get('user_defined_metrics', '')
            if user_metrics and user_metrics.strip():
                st.write(f"**Custom Metrics:** {user_metrics}")
            else:
                st.write(f"**Custom Metrics:** None")
        
        # Show error if present
        if eval_config.get('error'):
            st.error(f"**Error:** {eval_config['error']}")
        
        # Display models used
        st.write("#### Models Evaluated")
        models_info = eval_config.get("selected_models", [])
        if isinstance(models_info, list) and len(models_info) > 0:
            if isinstance(models_info[0], dict):
                # New format with complete information
                models_df = pd.DataFrame(models_info)
                st.dataframe(models_df, width='stretch', hide_index=True)
            else:
                # Legacy format - just model IDs
                st.write("Models (legacy format):")
                for model in models_info:
                    st.write(f"- {model}")
        else:
            st.write("No model information available")
        
        # Display judges used
        st.write("#### Judge Models")
        judges_info = eval_config.get("judge_models", [])
        if isinstance(judges_info, list) and len(judges_info) > 0:
            if isinstance(judges_info[0], dict):
                # New format with complete information
                judges_df = pd.DataFrame(judges_info)
                st.dataframe(judges_df, width='stretch', hide_index=True)
            else:
                # Legacy format - just model IDs
                st.write("Judges (legacy format):")
                for judge in judges_info:
                    st.write(f"- {judge}")
        else:
            st.write("No judge information available")
        
        # Display additional evaluation metadata
        st.write("#### Evaluation Configuration")
        config_col1, config_col2 = st.columns(2)
        with config_col1:
            st.write(f"**Parallel API Calls:** {eval_config.get('parallel_calls', 'Unknown')}")
            st.write(f"**Invocations per Scenario:** {eval_config.get('invocations_per_scenario', 'Unknown')}")
            st.write(f"**Experiment Counts:** {eval_config.get('experiment_counts', 'Unknown')}")
        with config_col2:
            st.write(f"**Temperature Variations:** {eval_config.get('temperature_variations', 'Unknown')}")
            st.write(f"**Failure Threshold:** {eval_config.get('failure_threshold', 'Unknown')}")
            st.write(f"**Sleep Between Invocations:** {eval_config.get('sleep_between_invocations', 'Unknown')}s")
        
        if eval_config.get('user_defined_metrics'):
            st.write(f"**User-Defined Metrics:** {eval_config['user_defined_metrics']}")

        # Display RPM Metrics if available
        self._display_rpm_metrics(eval_config, eval_id)

        # Add Load Config button
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("📋 Load Config", key=f"load_config_{eval_id}",
                        help="Load this evaluation's configuration to create a new evaluation"):
                st.session_state.load_from_eval_config = eval_config.copy()
                st.session_state.navigate_to_setup = True
                st.rerun()

    
