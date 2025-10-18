"""Unprocessed records viewer component for the Streamlit dashboard."""

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from ..utils.constants import DEFAULT_OUTPUT_DIR, PROJECT_ROOT

class UnprocessedRecordsViewer:
    """Component for viewing and analyzing unprocessed (failed) records."""

    def __init__(self):
        """Initialize the unprocessed viewer component."""
        self.unprocessed_dir = Path(PROJECT_ROOT) / DEFAULT_OUTPUT_DIR / "unprocessed"

    def render(self):
        """Render the unprocessed records viewer component."""
        st.subheader("Unprocessed Records Analysis")

        st.markdown("""
        This page shows records that failed to be evaluated due to API errors, timeouts, or other issues.
        Use this information to troubleshoot problems and retry failed evaluations.
        """)

        # Check if unprocessed directory exists
        if not self.unprocessed_dir.exists():
            st.info("No unprocessed records directory found. Failed records will appear here when they occur.")
            return

        # Get all unprocessed files
        unprocessed_files = sorted(
            [f for f in self.unprocessed_dir.glob("unprocessed_*.json")],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if not unprocessed_files:
            st.success("✅ No unprocessed records found! All evaluations completed successfully.")
            return

        # Display summary statistics
        st.write(f"📁 Found **{len(unprocessed_files)}** unprocessed record files")

        # Parse all files and collect data
        all_records = []
        file_info = []

        for file_path in unprocessed_files:
            try:
                with open(file_path, 'r') as f:
                    records = json.load(f)

                # Extract experiment name from filename
                filename = file_path.name
                # New format: unprocessed_<experiment_name>_<timestamp>_<uuid>.json
                parts = filename.replace('.json', '').split('_')
                if len(parts) >= 4:
                    # Extract experiment name (everything between 'unprocessed_' and timestamp)
                    timestamp_index = next((i for i, p in enumerate(parts) if '-' in p and 'T' in p), None)
                    if timestamp_index:
                        experiment_name = '_'.join(parts[1:timestamp_index])
                    else:
                        experiment_name = parts[1] if len(parts) > 1 else "Unknown"
                else:
                    experiment_name = "Legacy"

                # Add file info and experiment name to each record
                for record in records:
                    record['_file'] = file_path.name
                    record['_file_size'] = file_path.stat().st_size
                    record['_experiment_name'] = experiment_name
                    all_records.append(record)

                file_info.append({
                    'filename': file_path.name,
                    'experiment': experiment_name,
                    'records': len(records),
                    'size_kb': file_path.stat().st_size / 1024,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                })
            except Exception as e:
                st.warning(f"Could not parse {file_path.name}: {str(e)}")

        if not all_records:
            st.warning("No valid records found in unprocessed files.")
            return

        # Display file summary table
        with st.expander(f"📊 File Summary ({len(file_info)} files)", expanded=True):
            file_df = pd.DataFrame(file_info)

            # Strip UUID prefix from experiment names for display (UUID format: 8-4-4-4-12 hex characters)
            import re
            uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_'
            file_df['experiment'] = file_df['experiment'].apply(lambda x: re.sub(uuid_pattern, '', x))

            # Select and rename columns (exclude filename)
            file_df = file_df[['experiment', 'records', 'size_kb', 'modified']]
            file_df = file_df.rename(columns={
                'experiment': 'Experiment',
                'records': 'Failed Records',
                'size_kb': 'Size (KB)',
                'modified': 'Last Modified'
            })
            file_df['Size (KB)'] = file_df['Size (KB)'].round(2)
            st.dataframe(file_df, width='stretch', hide_index=True)

        # Summary statistics
        st.divider()
        st.subheader("Summary Statistics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Failed Records", len(all_records))
        with col2:
            unique_experiments = len(set(r['_experiment_name'] for r in all_records))
            st.metric("Affected Experiments", unique_experiments)
        with col3:
            unique_models = len(set(r['scenario'].get('model_id', 'Unknown') for r in all_records if 'scenario' in r))
            st.metric("Affected Models", unique_models)
        with col4:
            unique_tasks = len(set(r['scenario'].get('task_types', 'Unknown') for r in all_records if 'scenario' in r))
            st.metric("Affected Task Types", unique_tasks)

        # Error distribution charts
        st.divider()
        st.subheader("Error Analysis")

        # Prepare data for charts
        error_data = []
        for record in all_records:
            scenario = record.get('scenario', {})
            result = record.get('result', {})
            reason = record.get('reason', 'Unknown')

            error_data.append({
                'experiment': record.get('_experiment_name', 'Unknown'),
                'model_id': scenario.get('model_id', 'Unknown'),
                'task_type': scenario.get('task_types', 'Unknown'),
                'reason': reason,
                'api_status': result.get('api_call_status', 'Unknown'),
                'error_code': result.get('error_code', 'N/A')
            })

        error_df = pd.DataFrame(error_data)

        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["By Error Type", "By Model", "By Task Type"])

        with tab1:
            # Error type distribution
            error_counts = error_df['reason'].value_counts()
            fig_errors = px.pie(
                values=error_counts.values,
                names=error_counts.index,
                title="Distribution of Error Types",
                hole=0.4
            )
            fig_errors.update_layout(template="plotly_dark")
            st.plotly_chart(fig_errors, width='stretch')

        with tab2:
            # Errors by model
            model_errors = error_df.groupby('model_id').size().reset_index(name='count')
            model_errors = model_errors.sort_values('count', ascending=False)
            fig_models = px.bar(
                model_errors,
                x='model_id',
                y='count',
                title="Failed Records by Model",
                labels={'model_id': 'Model', 'count': 'Failed Records'}
            )
            fig_models.update_layout(template="plotly_dark", xaxis_tickangle=-45)
            st.plotly_chart(fig_models, width='stretch')

        with tab3:
            # Errors by task type
            task_errors = error_df.groupby('task_type').size().reset_index(name='count')
            task_errors = task_errors.sort_values('count', ascending=False)
            fig_tasks = px.bar(
                task_errors,
                x='task_type',
                y='count',
                title="Failed Records by Task Type",
                labels={'task_type': 'Task Type', 'count': 'Failed Records'}
            )
            fig_tasks.update_layout(template="plotly_dark", xaxis_tickangle=-45)
            st.plotly_chart(fig_tasks, width='stretch')

        # Detailed records table
        st.divider()
        st.subheader("Detailed Records")

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_experiment = st.selectbox(
                "Filter by Experiment",
                options=["All"] + sorted(error_df['experiment'].unique().tolist())
            )
        with col2:
            selected_model = st.selectbox(
                "Filter by Model",
                options=["All"] + sorted(error_df['model_id'].unique().tolist())
            )
        with col3:
            selected_task = st.selectbox(
                "Filter by Task Type",
                options=["All"] + sorted(error_df['task_type'].unique().tolist())
            )

        # Apply filters
        filtered_df = error_df.copy()
        if selected_experiment != "All":
            filtered_df = filtered_df[filtered_df['experiment'] == selected_experiment]
        if selected_model != "All":
            filtered_df = filtered_df[filtered_df['model_id'] == selected_model]
        if selected_task != "All":
            filtered_df = filtered_df[filtered_df['task_type'] == selected_task]

        st.write(f"Showing **{len(filtered_df)}** of **{len(error_df)}** failed records")

        # Display filtered records
        display_df = filtered_df[['experiment', 'model_id', 'task_type', 'reason', 'api_status']].copy()
        display_df = display_df.rename(columns={
            'experiment': 'Experiment',
            'model_id': 'Model',
            'task_type': 'Task Type',
            'reason': 'Failure Reason',
            'api_status': 'API Status'
        })
        st.dataframe(display_df, width='stretch', hide_index=True)

        # Detailed view expander
        st.divider()
        with st.expander("🔍 View Detailed Record Information"):
            if len(filtered_df) > 0:
                # Get filtered records
                filtered_records = [r for r in all_records
                                   if (selected_experiment == "All" or r.get('_experiment_name') == selected_experiment)
                                   and (selected_model == "All" or r.get('scenario', {}).get('model_id') == selected_model)
                                   and (selected_task == "All" or r.get('scenario', {}).get('task_types') == selected_task)]

                record_index = st.selectbox(
                    "Select record to view",
                    range(len(filtered_records)),
                    format_func=lambda i: f"Record {i+1}: {filtered_records[i].get('scenario', {}).get('model_id', 'Unknown')} - {filtered_records[i].get('reason', 'Unknown')}"
                )

                if record_index is not None:
                    selected_record = filtered_records[record_index]

                    st.write("#### Scenario Details")
                    scenario = selected_record.get('scenario', {})
                    st.json(scenario)

                    st.write("#### Result Details")
                    result = selected_record.get('result', {})
                    st.json(result)

                    st.write("#### Failure Information")
                    st.write(f"**Reason:** {selected_record.get('reason', 'Unknown')}")
                    st.write(f"**File:** {selected_record.get('_file', 'Unknown')}")
            else:
                st.info("No records match the selected filters.")

        # Export functionality
        st.divider()
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("📥 Export to CSV"):
                self._export_to_csv(filtered_df)
        with col2:
            st.caption("Export the filtered records to CSV for further analysis")

    def _export_to_csv(self, df):
        """Export filtered data to CSV."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = self.unprocessed_dir.parent / f"unprocessed_export_{timestamp}.csv"
            df.to_csv(export_path, index=False)
            st.success(f"✅ Exported to: {export_path}")
        except Exception as e:
            st.error(f"Failed to export: {str(e)}")
