"""
Metrics calculation functions for benchmark results.
"""

import logging
from .constants import EPSILON_DIVISION, VALUE_RATIO_MULTIPLIER

logger = logging.getLogger(__name__)


def calculate_metrics_by_model_task(df):
    """Calculate detailed metrics for each model-task-config combination.

    Properly handles cases where same task name has different configurations.
    """
    # Check if this is latency-only mode
    has_task_success = 'task_success' in df.columns and df['task_success'].notna().any()

    # Fill NaN task_types with a default value (for latency-only mode where task_types may be empty)
    if 'task_types' in df.columns:
        df['task_types'] = df['task_types'].fillna('Latency Benchmark')

    # Build aggregation dict based on available columns
    agg_dict = {
        'time_to_first_byte': ['mean', 'min', 'max'],
        'time_to_last_byte': ['mean', 'min', 'max'],
        'OTPS': ['mean', 'min', 'max'],
        'response_cost': ['mean', 'sum'],
        'output_tokens': ['mean', 'sum'],
        'input_tokens': ['mean', 'sum']
    }

    # Only include task_success if it exists and has valid data (not latency-only mode)
    if has_task_success:
        agg_dict['task_success'] = ['mean', 'count']
    else:
        # Use a different column for count in latency-only mode
        agg_dict['time_to_first_byte'] = agg_dict['time_to_first_byte'] + ['count']

    # Group by model, task, AND config signature
    metrics = df.groupby(['model_name', 'task_types', 'config_signature']).agg(agg_dict)

    # Flatten multi-level column index
    metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]

    # Rename columns for clarity
    rename_dict = {
        'time_to_first_byte_mean': 'avg_ttft',
        'time_to_last_byte_mean': 'avg_latency',
        'OTPS_mean': 'avg_otps',
        'response_cost_mean': 'avg_cost',
        'output_tokens_mean': 'avg_output_tokens',
        'input_tokens_mean': 'avg_input_tokens'
    }

    if has_task_success:
        rename_dict['task_success_mean'] = 'success_rate'
        rename_dict['task_success_count'] = 'sample_count'
    else:
        # In latency-only mode, use ttft count as sample_count
        rename_dict['time_to_first_byte_count'] = 'sample_count'

    metrics = metrics.rename(columns=rename_dict)
    metrics = metrics.reset_index()

    # Add task_display_name column with disambiguation
    # Count how many unique config_signatures exist for each task_type
    task_config_counts = metrics.groupby('task_types')['config_signature'].nunique()

    def generate_task_display_name(row):
        """Generate display name with numeric suffix if multiple configs exist."""
        task = row['task_types']
        if task_config_counts.get(task, 1) > 1:
            # Multiple configurations exist - need disambiguation
            # Get all configs for this task, sort them, and find index
            configs_for_task = sorted(metrics[metrics['task_types'] == task]['config_signature'].unique())
            config_index = configs_for_task.index(row['config_signature']) + 1
            return f"{task} ({config_index})"
        else:
            # Single configuration - use original name
            return task

    metrics['task_display_name'] = metrics.apply(generate_task_display_name, axis=1)
    logger.info(f"Generated task display names with disambiguation for {len(metrics)} metric rows")

    # Calculate value_ratio only if success_rate exists (360 mode)
    if has_task_success:
        # Use only non-NaN success_rate values for calculating max (to handle mixed evaluations)
        valid_success_rates = metrics['success_rate'].dropna()
        if not valid_success_rates.empty:
            max_raw_ratio = valid_success_rates.max() / (metrics['avg_cost'].min() + EPSILON_DIVISION)
            # Calculate value_ratio, will be NaN for rows where success_rate is NaN (latency-only tasks)
            metrics['value_ratio'] = VALUE_RATIO_MULTIPLIER * (metrics['success_rate'] / (metrics['avg_cost'] + EPSILON_DIVISION)) / max_raw_ratio

    return metrics


def calculate_metrics_by_model_task_temperature(df):
    """Calculate detailed metrics for each model-task-temperature-config combination.

    This function groups data by model, task, temperature, and config signature to enable
    temperature-based performance analysis while respecting configuration boundaries.

    Args:
        df: DataFrame with model evaluation data including TEMPERATURE column

    Returns:
        DataFrame with metrics grouped by model_name, task_types, TEMPERATURE, and config_signature
    """
    # Check if TEMPERATURE column exists
    if 'TEMPERATURE' not in df.columns:
        return None

    # Check if this is latency-only mode
    has_task_success = 'task_success' in df.columns and df['task_success'].notna().any()

    # Check if config_signature exists (should be added by load_data)
    if 'config_signature' not in df.columns:
        logger.warning("config_signature column not found in dataframe, temperature metrics may be incorrectly aggregated")
        # Fall back to grouping without config_signature
        groupby_cols = ['model_name', 'task_types', 'TEMPERATURE']
    else:
        groupby_cols = ['model_name', 'task_types', 'config_signature', 'TEMPERATURE']

    # Build aggregation dict based on available columns
    agg_dict = {
        'time_to_first_byte': ['mean', 'min', 'max'],
        'time_to_last_byte': ['mean', 'min', 'max'],
        'OTPS': ['mean', 'min', 'max'],
        'response_cost': ['mean', 'sum'],
        'output_tokens': ['mean', 'sum'],
        'input_tokens': ['mean', 'sum']
    }

    # Only include task_success if it exists and has valid data (not latency-only mode)
    if has_task_success:
        agg_dict['task_success'] = ['mean', 'count']
    else:
        # Use a different column for count in latency-only mode
        agg_dict['time_to_first_byte'] = agg_dict['time_to_first_byte'] + ['count']

    # Group by model, task, config, and temperature
    metrics = df.groupby(groupby_cols).agg(agg_dict)

    # Flatten multi-level column index
    metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]

    # Rename columns for clarity
    rename_dict = {
        'time_to_first_byte_mean': 'avg_ttft',
        'time_to_last_byte_mean': 'avg_latency',
        'OTPS_mean': 'avg_otps',
        'response_cost_mean': 'avg_cost',
        'output_tokens_mean': 'avg_output_tokens',
        'input_tokens_mean': 'avg_input_tokens'
    }

    if has_task_success:
        rename_dict['task_success_mean'] = 'success_rate'
        rename_dict['task_success_count'] = 'sample_count'
    else:
        # In latency-only mode, use ttft count as sample_count
        rename_dict['time_to_first_byte_count'] = 'sample_count'

    metrics = metrics.rename(columns=rename_dict)

    metrics = metrics.reset_index()

    # Add task_display_name if config_signature exists
    if 'config_signature' in metrics.columns:
        task_config_counts = metrics.groupby('task_types')['config_signature'].nunique()

        def generate_task_display_name(row):
            """Generate display name with numeric suffix if multiple configs exist."""
            task = row['task_types']
            if task_config_counts.get(task, 1) > 1:
                # Multiple configurations exist - need disambiguation
                configs_for_task = sorted(metrics[metrics['task_types'] == task]['config_signature'].unique())
                config_index = configs_for_task.index(row['config_signature']) + 1
                return f"{task} ({config_index})"
            else:
                # Single configuration - use original name
                return task

        metrics['task_display_name'] = metrics.apply(generate_task_display_name, axis=1)
        logger.info(f"Generated task display names for temperature metrics: {len(metrics)} rows")

    return metrics


def calculate_latency_metrics(df):
    """Calculate aggregated latency metrics by model (with service tier if available)."""
    # Use model_name_with_tier to preserve service tier distinctions
    group_col = 'model_name_with_tier' if 'model_name_with_tier' in df.columns else 'model_name'
    latency = df.groupby([group_col]).agg({
        'time_to_first_byte': ['mean', 'min', 'max', 'std'],
        'time_to_last_byte': ['mean', 'min', 'max', 'std'],
        'OTPS': ['mean', 'min', 'max', 'std']
    })

    # Flatten multi-level column index
    latency.columns = ['_'.join(col).strip() for col in latency.columns.values]

    # Rename columns for clarity
    latency = latency.rename(columns={
        'time_to_first_byte_mean': 'avg_ttft',
        'time_to_last_byte_mean': 'avg_latency',
        'OTPS_mean': 'avg_otps'
    })

    latency = latency.reset_index()
    # Rename the grouping column back to model_name for consistency with visualizations
    if group_col == 'model_name_with_tier':
        latency = latency.rename(columns={'model_name_with_tier': 'model_name'})

    return latency


def calculate_cost_metrics(df):
    """Calculate aggregated cost metrics by model (with service tier if available)."""
    # Use model_name_with_tier to preserve service tier distinctions
    group_col = 'model_name_with_tier' if 'model_name_with_tier' in df.columns else 'model_name'
    cost = df.groupby([group_col]).agg({
        'response_cost': ['mean', 'min', 'max', 'sum'],
        'input_tokens': ['mean', 'sum'],
        'output_tokens': ['mean', 'sum']
    })

    # Flatten multi-level column index
    cost.columns = ['_'.join(col).strip() for col in cost.columns.values]

    # Rename columns for clarity
    cost = cost.rename(columns={
        'response_cost_mean': 'avg_cost',
        'response_cost_sum': 'total_cost',
        'input_tokens_mean': 'avg_input_tokens',
        'output_tokens_mean': 'avg_output_tokens'
    })

    cost = cost.reset_index()
    # Rename the grouping column back to model_name for consistency with visualizations
    if group_col == 'model_name_with_tier':
        cost = cost.rename(columns={'model_name_with_tier': 'model_name'})

    return cost
