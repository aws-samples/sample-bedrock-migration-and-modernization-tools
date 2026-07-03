"""
Metrics calculation functions for benchmark results.
"""

import logging
import pandas as pd
from .constants import EPSILON_DIVISION, VALUE_RATIO_MULTIPLIER

logger = logging.getLogger(__name__)


def expected_attempts_per_model(df_all):
    """Derive the number of *expected* invocations per model from the dataset.

    Used as the denominator for reliability scoring so the metric reflects
    the dataset shape (n_prompts x invocations x temp_variants x experiments)
    instead of the count of CSV rows actually produced. Without this, attempts
    that crashed hard enough to land only in the unprocessed JSON would
    silently vanish from the denominator, flattering broken models.

    n_prompts is the *per-model* scenario count (max unique prompts over
    model_name), NOT the global unique-prompt count. This matters whenever the
    same scenarios are run under more than one model_name:
      - APO produces an extra "<model>_Prompt_Optimized" model_name whose prompt
        text differs from the original, and
      - one-to-many prompt optimization / multi-model runs evaluate the same
        scenarios across several target models.
    A global df['prompt'].nunique() would sum the prompts across all those
    model_names and inflate the denominator (e.g. APO doubles it), capping every
    model's reliability at ~1/N. Taking the max per model_name yields the true
    scenario count, while still penalizing a model that crashed on some prompts
    (its successes are divided by the fullest model's count).

    Returns 0 if df_all is empty or lacks a 'prompt' column.
    """
    if df_all is None or df_all.empty or 'prompt' not in df_all.columns:
        return 0
    if 'model_name' in df_all.columns and df_all['model_name'].notna().any():
        per_model = df_all.dropna(subset=['model_name']).groupby('model_name')['prompt'].nunique()
        n_prompts = int(per_model.max()) if not per_model.empty else df_all['prompt'].nunique()
    else:
        n_prompts = df_all['prompt'].nunique()

    def _first_int(col, default):
        if col not in df_all.columns:
            return default
        try:
            v = pd.to_numeric(df_all[col], errors='coerce').dropna()
            return int(v.iloc[0]) if not v.empty else default
        except Exception:
            return default

    inv = max(_first_int('invocations_per_scenario', 1), 1)
    experiments = max(_first_int('experiment_counts', 1), 1)

    # Number of distinct temperatures actually used per model — take the MAX so
    # a model that crashed before exhausting variants doesn't pull the count down.
    n_temps = 1
    if 'TEMPERATURE' in df_all.columns and 'model_id' in df_all.columns:
        try:
            t = df_all.groupby('model_id')['TEMPERATURE'].nunique()
            n_temps = max(int(t.max()), 1)
        except Exception:
            n_temps = 1

    return int(n_prompts * inv * n_temps * experiments)


def _build_task_display_name_map(metrics_df):
    """
    Pre-compute a mapping from (task_type, config_signature) to display name.

    This avoids O(n²) complexity when generating display names for each row.
    Instead of filtering the DataFrame inside apply(), we build the mapping once O(n).

    Args:
        metrics_df: DataFrame with 'task_types' and 'config_signature' columns

    Returns:
        dict: Mapping of (task_type, config_signature) -> display_name
    """
    # Count unique config_signatures per task_type
    task_config_counts = metrics_df.groupby('task_types')['config_signature'].nunique()

    # Build sorted configs for each task that has multiple configs
    task_sorted_configs = {}
    for task in task_config_counts.index:
        if task_config_counts[task] > 1:
            configs = sorted(metrics_df[metrics_df['task_types'] == task]['config_signature'].unique())
            task_sorted_configs[task] = {cfg: idx + 1 for idx, cfg in enumerate(configs)}

    # Build the display name map
    display_name_map = {}
    for _, row in metrics_df[['task_types', 'config_signature']].drop_duplicates().iterrows():
        task = row['task_types']
        config = row['config_signature']

        if task in task_sorted_configs:
            # Multiple configs - add numeric suffix
            config_index = task_sorted_configs[task][config]
            display_name_map[(task, config)] = f"{task} ({config_index})"
        else:
            # Single config - use task name directly
            display_name_map[(task, config)] = task

    return display_name_map


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

    # Add task_display_name column with disambiguation using pre-computed map (O(n) instead of O(n²))
    display_name_map = _build_task_display_name_map(metrics)
    metrics['task_display_name'] = metrics.apply(
        lambda row: display_name_map[(row['task_types'], row['config_signature'])],
        axis=1
    )
    logger.info(f"Generated task display names with disambiguation for {len(metrics)} metric rows")

    # Calculate value_ratio only if success_rate exists (360 mode)
    if has_task_success:
        # Use only non-NaN success_rate values for calculating max (to handle mixed evaluations)
        valid_success_rates = metrics['success_rate'].dropna()
        if not valid_success_rates.empty:
            max_raw_ratio = valid_success_rates.max() / (metrics['avg_cost'].min() + EPSILON_DIVISION)
            # Guard against division by zero (all success_rates are 0)
            if max_raw_ratio == 0:
                metrics['value_ratio'] = 0.0
            else:
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

    # Add task_display_name if config_signature exists using pre-computed map (O(n) instead of O(n²))
    if 'config_signature' in metrics.columns:
        display_name_map = _build_task_display_name_map(metrics)
        metrics['task_display_name'] = metrics.apply(
            lambda row: display_name_map[(row['task_types'], row['config_signature'])],
            axis=1
        )
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
