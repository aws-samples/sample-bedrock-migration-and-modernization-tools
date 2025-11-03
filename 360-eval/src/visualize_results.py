import logging, glob, re, ast, os
import pytz
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots
from jinja2 import Template
from collections import Counter
from datetime import datetime
from scipy import stats
from utils import run_inference, report_summary_template, convert_scientific_to_decimal

# Configuration constants
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Analysis constants
MIN_RECORDS_FOR_ANALYSIS = 1000
MIN_RECORDS_FOR_HISTOGRAM = 2000
EPSILON_DIVISION = 0.001  # Small value to prevent division by zero
VALUE_RATIO_MULTIPLIER = 10

# Statistical constants
PERCENTILES = [0.50, 0.90, 0.95, 0.99]
NORMAL_DISTRIBUTION_RANGE_MULTIPLIER = 0.5
NORMAL_DISTRIBUTION_POINTS = 100

# Visualization constants
COEFFICIENT_VARIATION_THRESHOLD = 0.3  # CV < 30% indicates good consistency
GRID_OPACITY = 0.3
COMPOSITE_SCORE_WEIGHTS = {
    'latency': 0.5,
    'cost': 0.5
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'success_rate': {'good': 0.95, 'medium': 0.85},
    # 'avg_latency': {'good': 1.5, 'medium': 2},
    'avg_cost': {'good': 0.5, 'medium': 1.0},
    'avg_otps': {'good': 100, 'medium': 35},
}

# LLM inference settings
INFERENCE_MAX_TOKENS = 750
INFERENCE_TEMPERATURE = 0.3
INFERENCE_REGION = 'us-west-2'

# Get project root directory
PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Setup logger
logger = logging.getLogger(__name__)
log_dir = PROJECT_ROOT / "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    # filename=log_dir / f"visualize_results_{TIMESTAMP}.log",
    filemode='a'
)
logger.info(f"Starting visualization with project root: {PROJECT_ROOT}")

# Load HTML template with absolute path
template_path = PROJECT_ROOT / "assets" / "html_template.txt"
try:
    with open(template_path, 'r') as file:
        HTML_TEMPLATE = file.read()
    logger.info(f"Loaded HTML template from {template_path}")
except FileNotFoundError:
    logger.error(f"HTML template not found at {template_path}")
    HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head><title>LLM Benchmark Report</title></head>
<body><h1>LLM Benchmark Report</h1><p>Template not found, using fallback.</p></body>
</html>"""


def extract_model_name(model_id):
    """Extract clean model name from ID."""
    # Check if this is an optimized prompt variant
    optimization_suffix = ""
    if "_Prompt_Optimized" in model_id:
        optimization_suffix = "_Prompt_Optimized"
        # Remove suffix temporarily for processing
        model_id = model_id.replace("_Prompt_Optimized", "")

    if '.' in model_id:
        parts = model_id.split('.')
        if len(parts) == 3:
            model_name = parts[-1].split(':')[0].split('-v')[0]
        else:
            model_name = parts[-2] + '.' + parts[-1]
        return model_name + optimization_suffix
    return model_id.split(':')[0] + optimization_suffix

def parse_json_string(json_str):
    try:
        if isinstance(json_str, list):
            json_str = json_str[0]
        # Use ast.literal_eval to safely evaluate the string as a Python literal
        # This handles the single-quoted JSON-like strings
        dict_data = ast.literal_eval(json_str)
        return dict_data
    except Exception as e:
        # Return error information if parsing fails
        return {"error": str(e)}


def get_evaluation_config_signature(df):
    """
    Create a unique signature for evaluation configuration.
    Evaluations with matching signatures can be safely merged.

    Returns:
        tuple: Configuration components that define a unique evaluation
    """
    try:
        # Extract unique values for each config component
        models = tuple(sorted(df['model_id'].unique()))
        task_criteria = tuple(sorted(df['task_criteria'].unique()))
        task_types = tuple(sorted(df['task_types'].unique()))

        # Extract judge models from performance_metrics column
        judge_models = set()
        for perf_metrics in df['performance_metrics'].dropna():
            try:
                parsed = parse_json_string(perf_metrics)
                if isinstance(parsed, dict) and 'judge_details' in parsed:
                    for judge in parsed['judge_details']:
                        if 'model' in judge:
                            judge_models.add(judge['model'])
            except:
                pass
        judges = tuple(sorted(judge_models))

        # User-defined metrics (if exists in data)
        user_metrics = tuple(sorted(df['user_defined_metrics'].unique())) if 'user_defined_metrics' in df.columns else tuple()

        # Temperature values
        temperatures = tuple(sorted(df['TEMPERATURE'].unique())) if 'TEMPERATURE' in df.columns else tuple()

        # Create signature tuple
        config_sig = (models, task_criteria, task_types, judges, user_metrics, temperatures)

        return config_sig
    except Exception as e:
        logger.warning(f"Error creating config signature: {e}")
        # Return a random signature to keep this group separate
        import uuid
        return (str(uuid.uuid4()),)


def load_data(directory, evaluation_names=None):
    """Load and prepare benchmark data with proper configuration-based grouping.

    Only merges evaluations with identical configurations (models, judges, criteria, etc.)

    Args:
        directory: Directory containing CSV files
        evaluation_names: Optional list of evaluation names to filter by
    """
    directory = Path(directory)
    logger.info(f"Looking for CSV files in: {directory}")

    # Load CSV files
    all_files = glob.glob(str(directory / "invocations_*.csv"))
    if not all_files:
        logger.error(f"No invocation CSVs found in {directory}")
        raise FileNotFoundError(f"No invocation CSVs found in {directory}")

    # Filter files by evaluation names if specified
    if evaluation_names:
        files = []
        for file_path in all_files:
            file_name = Path(file_path).name
            if any(eval_name in file_name for eval_name in evaluation_names):
                files.append(file_path)

        if not files:
            logger.warning(f"No CSV files found matching evaluations: {evaluation_names}")
            logger.info(f"Available files: {[Path(f).name for f in all_files]}")
            raise FileNotFoundError(f"No CSV files found for evaluations: {evaluation_names}")

        logger.info(f"Filtered to {len(files)} CSV files matching evaluations {evaluation_names}")
    else:
        files = all_files
        logger.info(f"Found {len(files)} CSV files (no filter applied)")

    # Step 1: Load all files and group by configuration signature
    config_groups = {}
    file_metadata = {}

    for f in files:
        try:
            logger.info(f"Reading file: {f}")
            df_file = pd.read_csv(f)
            logger.info(f"Read {len(df_file)} rows from {f}")

            # Add source file tracking
            df_file['source_file'] = Path(f).name

            # Get configuration signature for this file
            config_sig = get_evaluation_config_signature(df_file)

            # Group files by config signature
            if config_sig not in config_groups:
                config_groups[config_sig] = []
                file_metadata[config_sig] = {
                    'files': [],
                    'task_type': df_file['task_types'].iloc[0] if not df_file.empty else 'Unknown',
                    'total_records': 0
                }

            config_groups[config_sig].append(df_file)
            file_metadata[config_sig]['files'].append(Path(f).name)
            file_metadata[config_sig]['total_records'] += len(df_file)

        except Exception as e:
            logger.error(f"Error reading {f}: {str(e)}")
            continue

    if not config_groups:
        logger.error("No valid data found in any CSV files")
        raise ValueError("No valid data found in any CSV files")

    # Step 2: Log configuration grouping information
    logger.info(f"Found {len(config_groups)} unique evaluation configurations")
    for i, (config_sig, metadata) in enumerate(file_metadata.items(), 1):
        logger.info(f"  Config {i}: task_type='{metadata['task_type']}', "
                   f"files={len(metadata['files'])}, records={metadata['total_records']}")
        for fname in metadata['files']:
            logger.info(f"    - {fname}")

    # Step 3: Merge ONLY within same configuration groups
    merged_dfs = []
    for config_sig, dfs_in_group in config_groups.items():
        # These all have matching configs, safe to merge
        merged_df = pd.concat(dfs_in_group, ignore_index=True)

        # Add config signature as a column for later grouping
        merged_df['config_signature'] = str(hash(config_sig))

        merged_dfs.append(merged_df)
        logger.info(f"Merged {len(dfs_in_group)} files with matching config into {len(merged_df)} records")

    # Step 4: Combine all config groups
    df = pd.concat(merged_dfs, ignore_index=True)
    logger.info(f"Combined data has {len(df)} total rows across {len(config_groups)} unique configurations")

    # Clean and prepare data (optimized with method chaining)
    df = (df[df['api_call_status'] == 'Success']
          .reset_index(drop=True)
          .assign(model_name=lambda x: x['model_id'].apply(extract_model_name)))
    parsed_dicts = df['performance_metrics'].apply(parse_json_string)
    del df['performance_metrics']
    # Convert the Series of dictionaries to a DataFrame
    unpacked_findings = pd.DataFrame(list(parsed_dicts))
    df = pd.concat([df, unpacked_findings], axis=1)
    df['task_success'] = df['judge_success']
    # Calculate tokens per second
    df['OTPS'] = df['output_tokens'] / (df['time_to_last_byte'] + EPSILON_DIVISION)

    judge_scores = pd.DataFrame(df['judge_scores'].to_dict()).transpose()
    # Identify numeric index values
    numeric_index_mask = pd.to_numeric(judge_scores.index, errors='coerce').notna()
    # Filter and process judge scores (optimized with method chaining)
    judge_scores_df = (judge_scores[numeric_index_mask]
                       .reset_index(drop=True)
                       .assign(mean_scores=lambda x: x.mean(axis=1)))
    df = pd.concat([df, judge_scores_df], axis=1)
    # ── Cost summary ───────────────────────────────────────────────────────────
    cost_stats = (
        df.groupby(["model_name"])["response_cost"]
          .agg(avg_cost="mean", total_cost="sum", num_invocations="count")
    )

    # ── Latency percentiles (50/90/95/99) ──────────────────────────────────────
    latency_stats = (
        df.groupby(["model_name"])["time_to_last_byte"]
          .quantile(PERCENTILES)         # returns MultiIndex
          .unstack(level=-1)                          # percentiles → columns
    )
    latency_stats.columns = [f"p{int(q*100)}" for q in latency_stats.columns]

    # ── Combine both sets of metrics ──────────────────────────────────────────
    summary = cost_stats.join(latency_stats)

    # Optional: forecast spend per model/profile (30-day projection)
    summary["monthly_forecast"] = (
        summary["avg_cost"]
        * (summary["num_invocations"] / df.shape[0])
        * 30
    )
    df = pd.concat([df, summary], axis=1)

    return df


def calculate_metrics_by_model_task(df):
    """Calculate detailed metrics for each model-task-config combination.

    Properly handles cases where same task name has different configurations.
    """
    # Group by model, task, AND config signature
    metrics = df.groupby(['model_name', 'task_types', 'config_signature']).agg({
        'task_success': ['mean', 'count'],
        'time_to_first_byte': ['mean', 'min', 'max'],
        'time_to_last_byte': ['mean', 'min', 'max'],
        'OTPS': ['mean', 'min', 'max'],
        'response_cost': ['mean', 'sum'],
        'output_tokens': ['mean', 'sum'],
        'input_tokens': ['mean', 'sum']
    })

    # Flatten multi-level column index
    metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]

    # Rename columns for clarity
    metrics = metrics.rename(columns={
        'task_success_mean': 'success_rate',
        'task_success_count': 'sample_count',
        'time_to_first_byte_mean': 'avg_ttft',
        'time_to_last_byte_mean': 'avg_latency',
        'OTPS_mean': 'avg_otps',
        'response_cost_mean': 'avg_cost',
        'output_tokens_mean': 'avg_output_tokens',
        'input_tokens_mean': 'avg_input_tokens'
    })

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

    max_raw_ratio = metrics['success_rate'].max() / (metrics['avg_cost'].min() + EPSILON_DIVISION)
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

    # Check if config_signature exists (should be added by load_data)
    if 'config_signature' not in df.columns:
        logger.warning("config_signature column not found in dataframe, temperature metrics may be incorrectly aggregated")
        # Fall back to grouping without config_signature
        groupby_cols = ['model_name', 'task_types', 'TEMPERATURE']
    else:
        groupby_cols = ['model_name', 'task_types', 'config_signature', 'TEMPERATURE']

    # Group by model, task, config, and temperature
    metrics = df.groupby(groupby_cols).agg({
        'task_success': ['mean', 'count'],
        'time_to_first_byte': ['mean', 'min', 'max'],
        'time_to_last_byte': ['mean', 'min', 'max'],
        'OTPS': ['mean', 'min', 'max'],
        'response_cost': ['mean', 'sum'],
        'output_tokens': ['mean', 'sum'],
        'input_tokens': ['mean', 'sum']
    })

    # Flatten multi-level column index
    metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]

    # Rename columns for clarity
    metrics = metrics.rename(columns={
        'task_success_mean': 'success_rate',
        'task_success_count': 'sample_count',
        'time_to_first_byte_mean': 'avg_ttft',
        'time_to_last_byte_mean': 'avg_latency',
        'OTPS_mean': 'avg_otps',
        'response_cost_mean': 'avg_cost',
        'output_tokens_mean': 'avg_output_tokens',
        'input_tokens_mean': 'avg_input_tokens'
    })

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
    """Calculate aggregated latency metrics by model."""
    latency = df.groupby(['model_name']).agg({
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

    return latency.reset_index()


def calculate_cost_metrics(df):
    """Calculate aggregated cost metrics by model."""
    cost = df.groupby(['model_name']).agg({
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

    return cost.reset_index()


def create_normal_distribution_histogram(df,
                                         key='time_to_first_byte',
                                         label='Time to First Token (seconds)'):
    """
    Creates overlapping histogram plots with normal distribution curves for time_to_first_byte by model.
    Only creates the plot if there are more than 2000 records available.

    Args:
        df: DataFrame containing the benchmark data
        label: label for the histogram plot
        key: data column to create the histogram plot
    Returns:
        Plotly figure or None if insufficient data
    """
    min_vals = MIN_RECORDS_FOR_ANALYSIS
    # Check if we have enough data
    # Check if we have enough data
    value_counts = df['model_name'].value_counts()
    # Get values that appear more than 2000 times
    frequent_values = value_counts[value_counts > min_vals].index
    # Filter the dataframe to only include rows where the column value is in our frequent_values list
    df_match = df[df['model_name'].isin(frequent_values)]

    if df_match.empty:
        logger.info(f"Insufficient data for {label} Distribution by Model histogram: {len(df)} records (need >{MIN_RECORDS_FOR_HISTOGRAM})")
        return None

    # Filter out any null values
    df_clean = df_match[df_match[key].notna()].copy()

    if df_clean.empty:
        return ["No valid time_to_first_byte data found"]

    logger.info(f"Creating {label} Distribution by Model histogram with {len(df)} records")

    # Create figure
    fig = go.Figure()

    # Get unique models and assign colors
    unique_models = df_clean['model_name'].unique()
    colors = px.colors.qualitative.Set1[:len(unique_models)]

    # Create histogram and normal distribution for each model
    for i, model in enumerate(unique_models):
        model_data = df_clean[df_clean['model_name'] == model][key]

        if len(model_data) < 10:  # Skip models with too few data points
            continue

        # Calculate statistics for normal distribution
        mean = model_data.mean()
        std = model_data.std()

        # Add histogram
        fig.add_trace(go.Histogram(
            x=model_data,
            name=f'{model} (n={len(model_data)})',
            opacity=0.6,
            marker_color=colors[i % len(colors)],
            histnorm='probability density',  # Normalize to match normal curve
            nbinsx=50,
            showlegend=True
        ))

        # Generate points for normal distribution curve
        x_range = np.linspace(
            model_data.min() - NORMAL_DISTRIBUTION_RANGE_MULTIPLIER * std,
            model_data.max() + NORMAL_DISTRIBUTION_RANGE_MULTIPLIER * std,
            NORMAL_DISTRIBUTION_POINTS
        )
        normal_curve = stats.norm.pdf(x_range, mean, std)

        # Add normal distribution curve
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_curve,
            mode='lines',
            name=f'{model} Normal (μ={mean:.3f}, σ={std:.3f})',
            line=dict(
                color=colors[i % len(colors)],
                width=2,
                dash='dash'
            ),
            opacity=0.8,
            showlegend=True
        ))

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#2d2d2d",
        title={
            'text': f'{label} Distribution by Model<br><sub>Histograms with Normal Distribution Overlays</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=label,
        yaxis_title='Probability Density',
        barmode='overlay',  # Allow histograms to overlap
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        height=800,
        margin=dict(r=250)  # Extra margin for legend
    )

    # Update x and y axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=f'rgba(128,128,128,{GRID_OPACITY})')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=f'rgba(128,128,128,{GRID_OPACITY})')

    return fig


def identify_unique_task_configs(df):
    """
    Identify unique task configurations based on metric sets.
    If same task name has different metric configurations (different judges, criteria),
    assign numbered suffixes.

    Args:
        df: DataFrame with 'task_types' and 'judge_scores' columns

    Returns:
        dict: {unique_task_name: (original_task_name, metric_signature, indices)}
    """
    from collections import defaultdict

    # Extract metric signatures for each task evaluation
    df['parsed_scores'] = df['judge_scores'].apply(extract_judge_scores)

    # Group by task and collect metric signatures
    task_metrics = defaultdict(list)

    for idx, row in df.iterrows():
        task = row['task_types']
        if pd.isna(task):
            continue

        scores = row.get('parsed_scores', {})
        if not isinstance(scores, dict):
            continue

        # Get metric names (AVG_ prefixed keys)
        metrics = sorted([k.replace('AVG_', '') for k in scores.keys() if k.startswith('AVG_')])
        metric_sig = tuple(metrics) if metrics else tuple()

        task_metrics[task].append((idx, metric_sig))

    # Identify unique configurations per task
    unique_configs = {}

    for task, evaluations in task_metrics.items():
        # Get unique metric signatures for this task
        unique_sigs = {}
        for idx, sig in evaluations:
            if sig not in unique_sigs:
                unique_sigs[sig] = []
            unique_sigs[sig].append(idx)

        # If multiple configurations exist, add numeric suffixes
        if len(unique_sigs) > 1:
            for config_num, (sig, indices) in enumerate(sorted(unique_sigs.items()), start=1):
                unique_task_name = f"{task}({config_num})"
                unique_configs[unique_task_name] = (task, sig, indices)
        else:
            # Single configuration, use original name
            sig, indices = list(unique_sigs.items())[0]
            unique_configs[task] = (task, sig, indices)

    return unique_configs


def create_visualizations(df, model_task_metrics, latency_metrics, cost_metrics):
    """Create visualizations for the report."""
    visualizations = {}

    latency_metrics_round = latency_metrics
    average_cost_round = latency_metrics_round.round({'avg_ttft': 4})
    # 1. TTFT Comparison
    ttft_fig = px.bar(
        average_cost_round.sort_values('avg_ttft'),
        template="plotly_dark",  # Use the built-in dark template as a base
        x='model_name',
        y='avg_ttft',
        labels={'model_name': 'Model', 'avg_ttft': 'Time to First Token (Secs)'},
        title='Time to First Token by Model',
        color='avg_ttft',
        color_continuous_scale='Viridis_r'  # Reversed so lower is better (green)
    )

    # Improve overall chart visibility
    ttft_fig.update_layout(
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#2d2d2d",  # Slightly lighter than paper for contrast
    )

    visualizations['ttft_comparison'] = ttft_fig

    tokens_per_sec_round = latency_metrics
    tokens_per_sec_round = tokens_per_sec_round.round({'avg_otps': 2})

    # 2. OTPS Comparison
    otps_fig = px.bar(
        tokens_per_sec_round.sort_values('avg_otps', ascending=False),
        template="plotly_dark",  # Use the built-in dark template as a base
        x='model_name',
        y='avg_otps',
        error_y=tokens_per_sec_round['OTPS_std'],
        labels={'model_name': 'Model', 'avg_otps': 'Tokens/sec'},
        title='Output Tokens Per Second by Model',
        color='avg_otps',
        color_continuous_scale='Viridis'
    )

    # Improve overall chart visibility
    otps_fig.update_layout(
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#2d2d2d",  # Slightly lighter than paper for contrast
    )

    visualizations['otps_comparison'] = otps_fig
    average_cost_round = (cost_metrics
                          .sort_values('avg_cost')
                          .round({'avg_cost': 5}))
    # 3. Cost Comparison
    cost_fig = px.bar(
        average_cost_round.sort_values('avg_cost'),
        template="plotly_dark",  # Use the built-in dark template as a base
        x='model_name',
        y='avg_cost',
        labels={'model_name': 'Model', 'avg_cost': 'Cost per Response (USD)'},
        title='Using μ (Micro) Symbol for Small Numbers',
        color='avg_cost',
        color_continuous_scale='Viridis_r'  # Reversed so lower is better (green)
    )

    # Improve overall chart visibility
    cost_fig.update_layout(
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#2d2d2d",  # Slightly lighter than paper for contrast
    )

    visualizations['cost_comparison'] = cost_fig

    # 5. Task-Model Success Rate Heatmap
    # Pivot to create model vs task matrix (use task_display_name for proper disambiguation)
    pivot_success = pd.pivot_table(
        model_task_metrics,
        values='success_rate',
        index='model_name',
        columns='task_display_name',  # Use task_display_name instead of task_types
        aggfunc='mean'
    ).infer_objects(copy=False).fillna(0)

    heatmap_fig = px.imshow(
        pivot_success,
        template="plotly_dark",  # Use the built-in dark template as a base
        labels={'x': 'Task Type', 'y': 'Model', 'color': 'Success Rate'},
        title='Success Rate by Model and Task Type',
        color_continuous_scale='Earth', #'Viridis',
        text_auto='.2f',
        aspect='auto'
    )
    # Improve overall chart visibility
    heatmap_fig.update_layout(
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#2d2d2d",  # Slightly lighter than paper for contrast
    )

    visualizations['model_task_heatmap'] = heatmap_fig

    model_task_metrics_round = model_task_metrics
    average_cost_round = model_task_metrics_round.round({'avg_otps': 2, 'value_ratio': 2})
    # 6. Model-Task Bubble Chart
    bubble_fig = px.scatter(
        average_cost_round,
        template="plotly_dark",  # Keep the dark template for the base layout
        x='avg_latency',
        y='success_rate',
        size='avg_otps',
        color='avg_cost',
        facet_col='task_display_name',  # Use task_display_name instead of task_types
        facet_col_wrap=3,
        hover_data=['model_name', 'value_ratio'],
        labels={
            'avg_latency': 'Latency (Secs)',
            'success_rate': 'Success Rate',
            'avg_cost': 'Cost (USD)',
            'avg_otps': 'Tokens/sec',
            'task_display_name': 'Task Type'  # Add label for task_display_name
        },
        title='Model Performance by Task Type',
        color_continuous_scale='Earth',  # Use a brighter color scale
        opacity=0.85  # Slightly increase transparency for better contrast
    )

    # Additional customizations to improve visibility
    bubble_fig.update_traces(
        marker=dict(
            line=dict(width=1, color="rgba(255, 255, 255, 0.3)")  # Add subtle white outline
        )
    )

    # You can also brighten the color bar
    bubble_fig.update_layout(
        coloraxis_colorbar=dict(
            title_font_color="#ffffff",
            tickfont_color="#ffffff",
        )
    )

    # Make facet titles more visible
    bubble_fig.for_each_annotation(lambda a: a.update(font=dict(color="#90caf9", size=12)))

    # Improve overall chart visibility
    bubble_fig.update_layout(
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#2d2d2d",  # Slightly lighter than paper for contrast
        font=dict(color="#e0e0e0"),
        title_font=dict(color="#90caf9", size=18)
    )
    visualizations['model_task_bubble'] = bubble_fig

    # 7. Error Analysis
    if 'judge_explanation' in df.columns:
        fails = df[df['task_success'] == False].copy()
        if not fails.empty:
            fails['error'] = fails['judge_explanation'].fillna("Unknown").replace("", "Unknown")
            # Extract error categories using regex
            fails['error_category'] = fails['error'].apply(
                lambda x: '<br>'.join(list(set(re.findall(r'[A-Za-z-]+', str(x.replace(" ", "-").replace("&", "and")))))) if pd.notnull(x) else "Unknown"
            )

            counts = fails.groupby(['model_name', 'task_types', 'error_category']).size().reset_index(name='count')
            # counts['error_category'] = counts['error_category']
            # if counts
            error_fig = px.treemap(
                counts,
                template="plotly_dark",  # Use the built-in dark template as a base
                path=['task_types', 'model_name', 'error_category'],
                values='count',
                title='Error Analysis by Task, Model, and Error Type',
                color='count',
                color_continuous_scale='Reds'
            )
            error_fig.update_traces(
                hovertemplate='<br>Error Judgment: %{label}<br>Count: %{value:.0f}<br>Model: %{parent}<extra></extra>'
            )
            # Improve overall chart visibility
            error_fig.update_layout(
                paper_bgcolor="#1e1e1e",
                plot_bgcolor="#2d2d2d",  # Slightly lighter than paper for contrast
                )

            visualizations['error_analysis'] = error_fig.to_html(full_html=False)
        else:
            visualizations['error_analysis'] = '<div id="not-found">No Errors found in the Evaluation</div>'
    else:
        visualizations['error_analysis'] = '<div id="not-found">No Jury Evaluation Found</div>'

    # Add this inside create_visualizations() function
    # Create one radar chart per task with all models overlaid

    # Identify unique task configurations (handles edge case of same task with different metrics)
    unique_task_configs = identify_unique_task_configs(df)

    radar_charts = {}

    # Define color palette for models
    color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2

    # Create one chart per unique task configuration
    for unique_task_name, (original_task, metric_sig, indices) in unique_task_configs.items():
        # Filter data for this task configuration
        task_data = df.loc[indices].copy()

        if task_data.empty:
            continue

        # Get all unique models for this task
        models_in_task = task_data['model_name'].dropna().unique()

        if len(models_in_task) == 0:
            continue

        # Determine the metric categories for this task (from metric signature)
        task_categories = sorted(list(metric_sig)) if metric_sig else []

        if not task_categories:
            continue

        # Create figure for this task
        fig = go.Figure()

        # Add one trace per model
        for model_idx, model in enumerate(models_in_task):
            # Filter data for this model
            model_data = task_data[task_data['model_name'] == model]

            # Extract scores for this model
            scores_dicts = model_data['parsed_scores'].dropna().tolist()

            if not scores_dicts:
                continue

            # Calculate average scores for each category
            avg_scores = {}
            for score_dict in scores_dicts:
                for key, value in score_dict.items():
                    if key.startswith('AVG_'):
                        category = key.replace('AVG_', '')
                        if category in task_categories:
                            if category not in avg_scores:
                                avg_scores[category] = []
                            avg_scores[category].append(value)

            # Fill in values for each category
            values = []
            for category in task_categories:
                scores = avg_scores.get(category, [])
                if scores:
                    values.append(sum(scores) / len(scores))
                else:
                    values.append(0)

            # Get color for this model
            color = color_palette[model_idx % len(color_palette)]

            # Convert hex color to rgb for fill
            if color.startswith('#'):
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                fill_color = f'rgba({r}, {g}, {b}, 0.2)'
            else:
                fill_color = f'rgba(100, 100, 100, 0.2)'

            # Add trace for this model
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=task_categories + [task_categories[0]],  # Close the polygon
                fill='toself',
                name=model,
                opacity=0.7,
                line=dict(color=color, width=2),
                fillcolor=fill_color
            ))

        # Update layout
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#2d2d2d",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5],  # Assuming scores are on a 0-5 scale
                    gridcolor='rgba(128,128,128,0.3)'
                ),
                angularaxis=dict(
                    gridcolor='rgba(128,128,128,0.3)'
                )
            ),
            title=f"{unique_task_name} - Model Comparison",
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            ),
            height=600,
            width=1000,
            margin=dict(l=100, r=200, b=100, t=120)
        )

        # Store the chart with unique task name
        radar_charts[unique_task_name] = fig

    visualizations['judge_score_radars'] = radar_charts

    # 8. Task-specific charts with temperature breakdown
    task_charts = {}

    # Try to calculate temperature-based metrics if TEMPERATURE column exists
    temp_metrics = calculate_metrics_by_model_task_temperature(df)
    has_temperature_data = temp_metrics is not None and not temp_metrics.empty

    # Loop through unique task_display_name to handle multiple configs per task type
    for task_display in model_task_metrics['task_display_name'].unique():
        if has_temperature_data:
            # Use temperature-based grouped bar charts
            task_temp_data = temp_metrics[temp_metrics['task_display_name'] == task_display]

            if not task_temp_data.empty:
                # Create subplot with 2x2 grid
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        "Success Rate by Temperature",
                        "Latency (Secs) by Temperature",
                        'Cost per Response (USD) by Temperature<br><span style="font-size: 12px;">Using μ (Micro) Symbol for Small Numbers</span>',
                        "Tokens per Second by Temperature"
                    )
                )

                # Get unique temperatures and assign colors
                unique_temps = sorted(task_temp_data['TEMPERATURE'].unique())

                # Use solid color when only one temperature, otherwise use Viridis scale
                if len(unique_temps) == 1:
                    temp_color_map = {unique_temps[0]: '#66b2b2'}
                else:
                    viridis_colors = px.colors.sequential.Viridis
                    temp_color_map = {temp: viridis_colors[int((i / max(len(unique_temps) - 1, 1)) * (len(viridis_colors) - 1))]
                                      for i, temp in enumerate(unique_temps)}

                # Get unique models for consistent x-axis ordering
                unique_models = sorted(task_temp_data['model_name'].unique())

                # Success Rate subplot (row=1, col=1)
                for temp in unique_temps:
                    temp_data = task_temp_data[task_temp_data['TEMPERATURE'] == temp].set_index('model_name')
                    y_values = [temp_data.loc[model, 'success_rate'] if model in temp_data.index else None
                                for model in unique_models]

                    fig.add_trace(
                        go.Bar(
                            x=unique_models,
                            y=y_values,
                            name=f'T={temp}',
                            marker_color=temp_color_map[temp],
                            hovertemplate='<b>%{x}</b><br>Success Rate: %{y:.1%}<br>Temperature: ' + str(temp) + '<extra></extra>',
                            showlegend=True,
                            legendgroup=f'temp_{temp}'
                        ),
                        row=1, col=1
                    )

                # Latency subplot (row=1, col=2)
                for temp in unique_temps:
                    temp_data = task_temp_data[task_temp_data['TEMPERATURE'] == temp].set_index('model_name')
                    y_values = [temp_data.loc[model, 'avg_latency'] if model in temp_data.index else None
                                for model in unique_models]

                    fig.add_trace(
                        go.Bar(
                            x=unique_models,
                            y=y_values,
                            name=f'T={temp}',
                            marker_color=temp_color_map[temp],
                            hovertemplate='<b>%{x}</b><br>Latency: %{y:.2f}s<br>Temperature: ' + str(temp) + '<extra></extra>',
                            showlegend=False,
                            legendgroup=f'temp_{temp}'
                        ),
                        row=1, col=2
                    )

                # Cost subplot (row=2, col=1)
                for temp in unique_temps:
                    temp_data = task_temp_data[task_temp_data['TEMPERATURE'] == temp].set_index('model_name')
                    y_values = [temp_data.loc[model, 'avg_cost'] if model in temp_data.index else None
                                for model in unique_models]

                    fig.add_trace(
                        go.Bar(
                            x=unique_models,
                            y=y_values,
                            name=f'T={temp}',
                            marker_color=temp_color_map[temp],
                            hovertemplate='<b>%{x}</b><br>Cost: $%{y:.4f}<br>Temperature: ' + str(temp) + '<extra></extra>',
                            showlegend=False,
                            legendgroup=f'temp_{temp}'
                        ),
                        row=2, col=1
                    )

                # Tokens per Second subplot (row=2, col=2)
                for temp in unique_temps:
                    temp_data = task_temp_data[task_temp_data['TEMPERATURE'] == temp].set_index('model_name')
                    y_values = [temp_data.loc[model, 'avg_otps'] if model in temp_data.index else None
                                for model in unique_models]

                    fig.add_trace(
                        go.Bar(
                            x=unique_models,
                            y=y_values,
                            name=f'T={temp}',
                            marker_color=temp_color_map[temp],
                            hovertemplate='<b>%{x}</b><br>Tokens/sec: %{y:.1f}<br>Temperature: ' + str(temp) + '<extra></extra>',
                            showlegend=False,
                            legendgroup=f'temp_{temp}'
                        ),
                        row=2, col=2
                    )

                fig.update_layout(
                    height=800,
                    title_text=f"Performance Metrics for {task_display} (Grouped by Temperature)",
                    barmode='group',
                    template="plotly_dark",
                    paper_bgcolor="#1e1e1e",
                    plot_bgcolor="#2d2d2d",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        title="Temperature"
                    )
                )

                task_charts[task_display] = fig
        else:
            # Fallback to simple bar charts if no temperature data
            task_data = model_task_metrics[model_task_metrics['task_display_name'] == task_display]

            if not task_data.empty:
                # Create subplot with 2x2 grid
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Success Rate", "Latency (Secs)", 'Cost per Response (USD)<br><span style="font-size: 12px;">Using μ (Micro) Symbol for Small Numbers</span>', "Tokens per Second")
                )

                # Sort data for each subplot (using method chaining for efficiency)
                by_success = task_data.sort_values('success_rate', ascending=False)
                by_latency = task_data.sort_values('avg_latency')
                by_cost = task_data.sort_values('avg_cost')
                by_otps = task_data.sort_values('avg_otps', ascending=False)

                # Add traces for each subplot
                fig.add_trace(
                    go.Bar(x=by_success['model_name'], y=by_success['success_rate'], marker_color='green'),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Bar(x=by_latency['model_name'], y=by_latency['avg_latency'], marker_color='orange'),
                    row=1, col=2
                )

                fig.add_trace(
                    go.Bar(x=by_cost['model_name'], y=by_cost['avg_cost'], marker_color='red'),
                    row=2, col=1
                )

                fig.add_trace(
                    go.Bar(x=by_otps['model_name'], y=by_otps['avg_otps'], marker_color='blue'),
                    row=2, col=2
                )

                fig.update_layout(
                    height=725,
                    title_text=f"Performance Metrics for {task_display}",
                    showlegend=False,
                    template="plotly_dark",
                    paper_bgcolor="#1e1e1e",
                    plot_bgcolor="#2d2d2d",
                )

                task_charts[task_display] = fig

    visualizations['task_charts'] = task_charts
    visualizations['integrated_analysis_tables'], analysis_df = create_integrated_analysis_table(model_task_metrics)



    visualizations['regional_performance'] = create_regional_performance_analysis(df)

    # Add TTFB histogram with normal distribution (only if sufficient data)
    ttfb_histogram = create_normal_distribution_histogram(df)
    if ttfb_histogram is not None:
        visualizations['ttfb_histogram'] = ttfb_histogram

    accuracy_histogram = create_normal_distribution_histogram(df, key='mean_scores', label='Accuracy Distribution by Model')
    if accuracy_histogram is not None:
        visualizations['accuracy_histogram'] = accuracy_histogram

    return visualizations


def generate_task_findings(df, model_task_metrics):
    """Generate key findings for each task configuration (using task_display_name)."""
    task_findings = {}

    # Loop through unique task_display_name to handle multiple configs
    for task_display in model_task_metrics['task_display_name'].unique():
        task_data = model_task_metrics[model_task_metrics['task_display_name'] == task_display]
        findings = []

        if not task_data.empty:
            # Best accuracy model
            best_acc_idx = task_data['success_rate'].idxmax()
            best_acc = task_data.loc[best_acc_idx]
            findings.append(f"{best_acc['model_name']} had the highest success rate ({best_acc['success_rate']:.1%})")

            # Best speed model
            best_speed_idx = task_data['avg_latency'].idxmin()
            best_speed = task_data.loc[best_speed_idx]
            findings.append(
                f"{best_speed['model_name']} was the fastest with {best_speed['avg_latency']:.2f}s average latency")

            # Best throughput model
            best_otps_idx = task_data['avg_otps'].idxmax()
            best_otps = task_data.loc[best_otps_idx]
            findings.append(
                f"{best_otps['model_name']} had the highest throughput ({best_otps['avg_otps']:.1f} tokens/sec)")

            # Best value model
            best_value_idx = task_data['value_ratio'].idxmax()
            best_value = task_data.loc[best_value_idx]
            findings.append(
                f"{best_value['model_name']} offered the best value (success/cost ratio: {best_value['value_ratio']:.2f})")

            # Average success rate
            avg_success = task_data['success_rate'].mean()
            findings.append(f"Average success rate for this task was {avg_success:.1%}")

            # Error analysis - filter by both task_types and config_signature
            task_types = task_data['task_types'].iloc[0]
            config_sig = task_data['config_signature'].iloc[0] if 'config_signature' in task_data.columns else None

            if config_sig:
                fails = df[(df['task_types'] == task_types) & (df['config_signature'] == config_sig) & (df['task_success'] == False)]
            else:
                fails = df[(df['task_types'] == task_types) & (df['task_success'] == False)]

            if not fails.empty and 'judge_explanation' in fails.columns:
                # Extract common error patterns
                error_patterns = []
                unique_explanations = fails['judge_explanation'].dropna()
                all_errors = unique_explanations.apply(lambda x: [i for i in x.split(';') if i != '']).tolist()
                [error_patterns.extend(exp) for exp in all_errors]
                if error_patterns:
                    common_errors = Counter(error_patterns).most_common(2)
                    errors_text = ", ".join([f"{err[0]} ({err[1]} occurrences)" for err in common_errors])
                    findings.append(f"Most common errors: {errors_text}")

        task_findings[task_display] = findings

    return task_findings


def generate_task_recommendations(model_task_metrics):
    """Generate task-specific model recommendations (using task_display_name)."""
    recommendations = []

    # Loop through unique task_display_name to handle multiple configs
    for task_display in model_task_metrics['task_display_name'].unique():
        task_data = model_task_metrics[model_task_metrics['task_display_name'] == task_display]

        if not task_data.empty:
            # Find best models by different metrics
            best_suc = task_data['success_rate'].max()
            best_acc_model = '<br>'.join(task_data[task_data['success_rate'] == best_suc]['model_name'].tolist())

            best_lat = task_data['avg_latency'].min()
            best_speed_model = '<br>'.join(task_data[task_data['avg_latency'] == best_lat]['model_name'].tolist())

            best_value = task_data['value_ratio'].max()
            best_value_model = '<br>'.join(task_data[task_data['value_ratio'] == best_value]['model_name'].tolist())

            # Create recommendation entry (use task_display for display)
            recommendations.append({
                'task': task_display,
                'best_accuracy_model': best_acc_model,
                'accuracy': f"{best_suc:.1%}",
                'best_speed_model': best_speed_model,
                'speed': f"{best_lat:.2f}s",
                'best_value_model': best_value_model,
                'value': f"{best_value:.2f}"
            })

    return sorted(recommendations, key=lambda x: x['task'])


def generate_histogram_findings(df, key='time_to_first_byte', label='Time to First Token'):
    """
    Generate key findings for the TTFB histogram analysis.
    Returns either meaningful findings or a message about insufficient data.

    Args:
        df: DataFrame containing the benchmark data
        key: Key used to measure
        label: Label used to label the findings
    Returns:
        List of finding strings or single message about insufficient data
    """
    min_records = MIN_RECORDS_FOR_ANALYSIS
    # Check if we have enough data
    value_counts = df['model_name'].value_counts()
    # Get values that appear more than 2000 times
    frequent_values = value_counts[value_counts > min_records].index
    # Filter the dataframe to only include rows where the column value is in our frequent_values list
    df_match = df[df['model_name'].isin(frequent_values)]
    if df_match.empty:
        return [f"Not enough data to perform measurements (need at minimum over {MIN_RECORDS_FOR_HISTOGRAM} measurements per model)"]

    # Filter out any null values
    df_clean = df_match[df_match[key].notna()].copy()

    if df_clean.empty:
        return [f"No valid {key} data found"]

    findings = []

    for model in df_clean['model_name'].unique().tolist():
        df_model = df_clean[df_clean['model_name'] == model]
        # Overall statistics
        overall_mean = df_model[key].mean()
        overall_std = df_model[key].std()
        findings.append(f"Model <b>{model}</b> {label}: μ={overall_mean:.3f}s, σ={overall_std:.3f}s across {len(df_model)} measurements")

    # Model-specific analysis (optimized with method chaining)
    model_stats = (df_clean.groupby('model_name')[key]
                   .agg(['mean', 'std', 'count'])
                   .reset_index()
                   .query(f'count >= {min_records}'))  # Only models with sufficient data

    if not model_stats.empty:
        # Fastest model (lowest mean)
        fastest_model = model_stats.loc[model_stats['mean'].idxmin()]
        findings.append(f"Highest achieving model: <b>{fastest_model['model_name']}</b> with {fastest_model['mean']:.3f}s average {label}")

        # Most consistent model (lowest standard deviation)
        most_consistent = model_stats.loc[model_stats['std'].idxmin()]
        findings.append(f"Most consistent model: <b>{most_consistent['model_name']}</b> with {most_consistent['std']:.3f}s standard deviation")

        # Model with highest variability
        most_variable = model_stats.loc[model_stats['std'].idxmax()]
        findings.append(f"Most variable model (fat-tails): <b>{most_variable['model_name']}</b> with {most_variable['std']:.3f}s standard deviation")

        # Distribution characteristics
        # Check for normality using coefficient of variation
        model_stats['cv'] = model_stats['std'] / model_stats['mean']  # Coefficient of variation

        # Models with good normal distribution characteristics (low CV)
        well_distributed = model_stats[model_stats['cv'] < COEFFICIENT_VARIATION_THRESHOLD]  # CV < 30% indicates good consistency
        if not well_distributed.empty:
            best_distributed = well_distributed.loc[well_distributed['cv'].idxmin()]
            findings.append(f"Best distribution characteristics: <b>{best_distributed['model_name']}</b> (Coefficient of Variation/CV={best_distributed['cv']:.2f})")

        # Performance spread analysis
        fastest_mean = model_stats['mean'].min()
        slowest_mean = model_stats['mean'].max()
        performance_spread = ((slowest_mean - fastest_mean) / fastest_mean) * 100
        findings.append(f"Performance spread: {performance_spread:.1f}% difference between best and worst achieving models")
        for model in df_clean['model_name'].unique().tolist():
            # Outlier detection
            df_model = df_clean[df_clean['model_name'] == model]
            q1 = df_model[key].quantile(0.25)
            q3 = df_model[key].quantile(0.75)
            iqr = q3 - q1
            outlier_threshold = q3 + 1.5 * iqr
            outliers = df_model[df_model[key] > outlier_threshold]
            if not outliers.empty:
                outlier_pct = (len(outliers) / len(df_clean)) * 100
                findings.append(f"Outliers for <b>{model}</b>: {len(outliers)} measurements ({outlier_pct:.1f}%) exceed {outlier_threshold:.3f}s")

    return findings


def create_html_report(output_dir, timestamp, evaluation_names=None):
    """Generate HTML benchmark report with task-specific analysis.

    Args:
        output_dir: Directory containing CSV files and where report will be saved
        timestamp: Timestamp for report filename
        evaluation_names: Optional list of evaluation names to filter by
    """
    # Ensure output_dir is an absolute path
    if isinstance(output_dir, str):
        if not os.path.isabs(output_dir):
            output_dir = PROJECT_ROOT / output_dir
        output_dir = Path(output_dir)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using output directory: {output_dir}")

    # Use log directory from project root
    log_dir = PROJECT_ROOT / "logs"
    os.makedirs(log_dir, exist_ok=True)
    report_log_file = log_dir / f"report_generation-{timestamp}.log"
    logger.info(f"Report generation logs will be saved to: {report_log_file}")

    # Load and process data
    if evaluation_names:
        logger.info(f"Loading and processing data for evaluations: {evaluation_names}")
    else:
        logger.info("Loading and processing data for all evaluations...")
    try:
        df = load_data(output_dir, evaluation_names)
        evaluation_info = f" for evaluations {evaluation_names}" if evaluation_names else " (all evaluations)"
        logger.info(f"Loaded data with {len(df)} records from {output_dir}{evaluation_info}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


    # Calculate metrics
    logger.info("Calculating model-task metrics...")
    model_task_metrics = calculate_metrics_by_model_task(df)

    logger.info("Calculating latency metrics...")
    latency_metrics = calculate_latency_metrics(df)

    logger.info("Calculating cost metrics...")
    cost_metrics = calculate_cost_metrics(df)



    # Create visualizations
    logger.info("Creating visualizations...")
    visualizations = create_visualizations(df, model_task_metrics, latency_metrics, cost_metrics)

    # Generate findings and recommendations
    logger.info("Generating task findings...")
    task_findings = generate_task_findings(df, model_task_metrics)

    logger.info("Generating recommendations...")
    task_recommendations = generate_task_recommendations(model_task_metrics)
    task_level_analysis = '# Task Level Analysis:\n'
    # Prepare task analysis data for template
    task_analysis = []
    for task, chart in visualizations['task_charts'].items():
        task_level_analysis += f'# Task Name: {task}\n\n'
        task_level_analysis += '- ' + '\n- '.join(task_findings.get(task, ["No specific findings available."])) + '\n\n'
        task_analysis.append({
            'name': task,
            'chart': chart.to_html(full_html=False),
            'findings': task_findings.get(task, ["No specific findings available."])
        })

    # Render HTML template
    logger.info("Rendering HTML report...")

    # Parse the string into a datetime object
    datetime_object = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")

    # Format the datetime object into the desired string representation
    formatted_date = datetime_object.strftime("%B %d, %Y at %I:%M %p")
    # Add this to extract unique models
    unique_models = df['model_name'].dropna().unique().tolist()

    logger.info("Generating TTFT histogram findings...")
    time_to_first_token_findings = generate_histogram_findings(df)
    perf_analysis = '# Performance Analysis across all models:\n- ' + '\n- '.join(time_to_first_token_findings)

    logger.info("Generating Accuracy histogram findings...")
    accuracy_findings = generate_histogram_findings(df, key='mean_scores', label="Average Accuracy")     #TODO: BY TASK??
    acc_analysis = '# Accuracy Analysis across all models:\n- ' + '\n- '.join(accuracy_findings)

    whole_number_cost_metrics = convert_scientific_to_decimal(cost_metrics)
    cost_analysis = '# Cost Analysis across all models on all Task:\n' + '\n'.join([str(i) for i in whole_number_cost_metrics.to_dict(orient='records')])

    recommendations = '# Recommendations:\n* ' + '\n* '.join([str(i) for i in task_recommendations])

    prompt_template = report_summary_template(models=unique_models, evaluations=f'{acc_analysis}\n\n{cost_analysis}\n\n{perf_analysis}\n\n{task_level_analysis}\n\n{recommendations}')  ## Append AND Format all evals ++ rename the columns to help the model
    # Model ID preparation for litellm (/converse addition) is now handled centrally in run_inference()
    inference = run_inference(model_name='bedrock/us.amazon.nova-premier-v1:0',
                              prompt_text=prompt_template,
                              stream=False,
                              provider_params={"maxTokens": INFERENCE_MAX_TOKENS,
                                               "temperature": INFERENCE_TEMPERATURE,
                                               "aws_region_name": INFERENCE_REGION})['text']
    html = Template(HTML_TEMPLATE).render(
        timestamp=formatted_date,
        inference=inference,

        # Latency charts
        ttft_comparison_div=visualizations['ttft_comparison'].to_html(full_html=False),
        otps_comparison_div=visualizations['otps_comparison'].to_html(full_html=False),

        # Cost charts
        cost_comparison_div=visualizations['cost_comparison'].to_html(full_html=False),

        # Task analysis
        task_analysis=task_analysis,

        # Model-Task performance
        model_task_heatmap_div=visualizations['model_task_heatmap'].to_html(full_html=False),
        model_task_bubble_div=visualizations['model_task_bubble'].to_html(full_html=False),

        unique_models = unique_models,
        # Radar charts are now keyed by task name (not model-task tuples)
        judge_score_radars = {task: chart.to_html(full_html=False)
                              for task, chart in visualizations.get('judge_score_radars', {}).items()},

        # Error and regional Analysis
        error_analysis_div=visualizations['error_analysis'],
        integrated_analysis_tables={task: table.to_html(full_html=False) for task, table in visualizations['integrated_analysis_tables'].items()},
        unique_tasks=list(visualizations['integrated_analysis_tables'].keys()),
        regional_performance_div=visualizations['regional_performance'].to_html(full_html=False),

        # TTFB histogram (only if sufficient data)
        ttfb_histogram_div=visualizations['ttfb_histogram'].to_html(full_html=False) if 'ttfb_histogram' in visualizations else '',
        ttfb_findings=time_to_first_token_findings,

        # Accuracy histogram (only if sufficient data)
        accuracy_histogram_div=visualizations['accuracy_histogram'].to_html(
            full_html=False) if 'accuracy_histogram' in visualizations else '',
        accuracy_findings=accuracy_findings,
        # Recommendations
        task_recommendations=task_recommendations,
    )

    # Write report to file with evaluation-specific naming
    if evaluation_names:
        eval_suffix = "_" + "_".join(evaluation_names[:3])  # Limit to first 3 for filename length
        if len(evaluation_names) > 3:
            eval_suffix += f"_and_{len(evaluation_names)-3}_more"
        out_file = output_dir / f"llm_benchmark_report_{timestamp}{eval_suffix}.html"
    else:
        out_file = output_dir / f"llm_benchmark_report_{timestamp}.html"

    logger.info(f"Writing HTML report to: {out_file}")
    out_file.write_text(html, encoding="utf-8")
    evaluation_scope = f"for {len(evaluation_names)} specific evaluations" if evaluation_names else "for all evaluations"
    logger.info(f"HTML report written successfully {evaluation_scope}")

    return out_file

#############################
#############################

def extract_judge_scores(json_str):
    try:
        if isinstance(json_str, dict):
            return json_str
        if isinstance(json_str, list):
            json_str = json_str[0]
        # Use ast.literal_eval to safely evaluate the string as a Python literal
        dict_data = ast.literal_eval(json_str)
        return dict_data
    except Exception as e:
        return {}


from collections import defaultdict
import numpy as np
def build_task_latency_thresholds(records, method="percentile", value=0.75, round_ndigits=3):
    """
    Build latency thresholds per task across models.
    Parameters
    ----------
    """
    by_task = defaultdict(list)
    # group latencies by task
    for r in records:
        tt = r.get("task_types")
        lat = r.get("avg_latency")
        if tt and isinstance(lat, (int, float)) and lat > 0:
            by_task[tt].append(float(lat))
    out = {}
    for tt, lats in by_task.items():
        arr = np.array(lats, dtype=float)
        med = float(np.median(arr))
        if method == "percentile":
            medium_cutoff = float(np.quantile(arr, value))
        elif method == "tolerance":
            medium_cutoff = med * (1 + value)
        else:
            raise ValueError("method must be 'percentile' or 'tolerance'")
        out[tt] = {
                "good": round(med, round_ndigits),
                "medium": round(medium_cutoff, round_ndigits)
        }
    return out


##############################
##############################
def create_integrated_analysis_table(model_task_metrics):
    """
    Creates interactive tables for each task with distance-from-best color coding.
    Colors are based on how far each model is from the best performer.
    """
    # Define colors - now with more gradations for distance-based coloring
    colors = {
        'best': '#2ecc71',      # Dark green - the best performer
        'excellent': '#52d68a',  # Medium green - within 5% of best
        'good': '#c6efce',      # Light green - within 10% of best
        'medium': '#ffffcc',    # Yellow - within 20% of best
        'below': '#ffd4a3',     # Orange - within 30% of best
        'poor': '#ffcccc'       # Light red - more than 30% behind
    }

    # Initialize task_tables dictionary and thresholds
    task_tables = {}
    thresholds = PERFORMANCE_THRESHOLDS.copy()

    # Prepare the data for the table
    table_data = model_task_metrics.copy()

    thresholds['avg_latency'] = build_task_latency_thresholds(table_data[['model_name', 'task_types', 'avg_latency']].to_dict(orient='records'))

    # Format Model Name
    table_data['model_name'] = table_data['model_name'].apply(lambda x: x.split('/')[-1])

    # Format metrics for display
    table_data['success_rate_fmt'] = table_data['success_rate'].apply(lambda x: f"{x:.1%}")
    table_data['avg_latency_fmt'] = table_data['avg_latency'].apply(lambda x: f"{x:.2f}s")
    table_data['avg_cost_fmt'] = table_data['avg_cost'].apply(lambda x: f"${x:.4f}")
    table_data['avg_otps_fmt'] = table_data['avg_otps'].apply(lambda x: f"{x:.1f}")

    # Calculate composite score (higher is better)
    # Normalize metrics to 0-1 range and combine them
    max_latency = table_data['avg_latency'].max() or 1
    max_cost = table_data['avg_cost'].max() or 1

    table_data['composite_score'] = (
            table_data['success_rate'] +
            (1 - (table_data['avg_latency'] / max_latency)) * COMPOSITE_SCORE_WEIGHTS['latency'] +
            (1 - (table_data['avg_cost'] / max_cost)) * COMPOSITE_SCORE_WEIGHTS['cost']
    )

    # Helper function to determine color based on value and thresholds
    def get_color(value, metric):
        if metric == 'success_rate' or metric == 'avg_otps':
            if value >= thresholds[metric]['good']:
                return colors['good']
            elif value >= thresholds[metric]['medium']:
                return colors['medium']
            else:
                return colors['poor']
        elif metric == 'avg_latency':
            if value['avg_latency'] <= thresholds[metric][value['task_types']]['good']:
                return colors['good']
            else:
                return colors['medium']
        else:  # For latency and cost, lower is better
            if value <= thresholds[metric]['good']:
                return colors['good']
            elif value <= thresholds[metric]['medium']:
                return colors['medium']
            else:
                return colors['poor']

    # Loop through each unique task_display_name and create a table for each
    for task_display in table_data['task_display_name'].unique():
        # Filter data for this task
        task_data = table_data[table_data['task_display_name'] == task_display].copy()

        # Create figure
        fig = go.Figure()

        # Create table cells with conditional formatting
        fig.add_trace(go.Table(
            header=dict(
                values=['Model', 'Task Type', 'Success Rate', 'Latency', 'Cost', 'Tokens/sec', 'Score'],
                font=dict(size=12, color='white'),
                fill_color='#2E5A88',
                align='left'
            ),
            cells=dict(
                values=[
                    task_data['model_name'],
                    task_data['task_display_name'],  # Use task_display_name for display
                    task_data['success_rate_fmt'],
                    task_data['avg_latency_fmt'],
                    task_data['avg_cost_fmt'],
                    task_data['avg_otps_fmt'],
                    task_data['composite_score'].apply(lambda x: f"{x:.2f}")
                ],
                align='left',
                font=dict(size=11),
                # Conditional formatting based on thresholds
                fill_color=[
                    ['#3a3a3a'] * len(task_data),  # Model column (dark gray)
                    ['#3a3a3a'] * len(task_data),  # Task column (dark gray)
                    # Success rate coloring (three-color)
                    [get_color(sr, 'success_rate') for sr in task_data['success_rate']],
                    # Latency coloring (three-color)
                    [get_color(lt, 'avg_latency') for lt in task_data[['avg_latency','task_types']].to_dict(orient='records')],
                    # Cost coloring (three-color)
                    [get_color(cost, 'avg_cost') for cost in task_data['avg_cost']],
                    # OTPS coloring
                    [get_color(tps, 'avg_otps') for tps in task_data['avg_otps']],
                    # Composite score coloring based on quantiles
                    [colors['good'] if score >= task_data['composite_score'].quantile(0.67) else
                     colors['medium'] if score >= task_data['composite_score'].quantile(0.33) else
                     colors['poor'] for score in task_data['composite_score']]
                ],
                # Text color: white for dark columns, black for colored columns
                font_color=[
                    ['white'] * len(task_data),  # Model column (white text on dark background)
                    ['white'] * len(task_data),  # Task column (white text on dark background)
                    ['black'] * len(task_data),  # Success rate (black text on colored background)
                    ['black'] * len(task_data),  # Latency (black text on colored background)
                    ['black'] * len(task_data),  # Cost (black text on colored background)
                    ['black'] * len(task_data),  # Tokens/sec (black text on colored background)
                    ['black'] * len(task_data),  # Score (black text on colored background)
                ]
            )
        ))

        # Update layout with dark theme and dynamic height
        # Calculate height based on number of rows (header + rows)
        row_height = 35  # pixels per row
        header_height = 40  # header row height
        total_height = header_height + (len(task_data) * row_height)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#2d2d2d",
            margin=dict(l=0, r=0, t=0, b=0),
            height=total_height
        )

        # Store the table for this task (using task_display_name as key)
        task_tables[task_display] = fig

    # Return dictionary of tables for dropdown display
    return task_tables, model_task_metrics.to_dict(orient='records')


def create_regional_performance_analysis(df):
    """
    Creates a plot showing latency and cost metrics grouped by region,
    including time of day analysis and region-specific recommendations.
    """

    # Map regions to their time zones
    region_timezones = {
        # North America
        'us-east-1': pytz.timezone('America/New_York'),  # N. Virginia
        'us-east-2': pytz.timezone('America/Chicago'),  # Ohio
        'us-west-1': pytz.timezone('America/Los_Angeles'),  # N. California
        'us-west-2': pytz.timezone('America/Los_Angeles'),  # Oregon

        # Africa
        'af-south-1': pytz.timezone('Africa/Johannesburg'),  # Cape Town

        # Asia Pacific
        'ap-east-1': pytz.timezone('Asia/Hong_Kong'),  # Hong Kong
        'ap-south-2': pytz.timezone('Asia/Kolkata'),  # Hyderabad
        'ap-southeast-3': pytz.timezone('Asia/Jakarta'),  # Jakarta
        'ap-southeast-5': pytz.timezone('Asia/Kuala_Lumpur'),  # Malaysia
        'ap-southeast-4': pytz.timezone('Australia/Melbourne'),  # Melbourne
        'ap-south-1': pytz.timezone('Asia/Kolkata'),  # Mumbai
        'ap-northeast-3': pytz.timezone('Asia/Tokyo'),  # Osaka
        'ap-northeast-2': pytz.timezone('Asia/Seoul'),  # Seoul
        'ap-southeast-1': pytz.timezone('Asia/Singapore'),  # Singapore
        'ap-southeast-2': pytz.timezone('Australia/Sydney'),  # Sydney
        'ap-southeast-7': pytz.timezone('Asia/Bangkok'),  # Thailand
        'ap-northeast-1': pytz.timezone('Asia/Tokyo'),  # Tokyo

        # Canada
        'ca-central-1': pytz.timezone('America/Toronto'),  # Central
        'ca-west-1': pytz.timezone('America/Edmonton'),  # Calgary

        # Europe
        'eu-central-1': pytz.timezone('Europe/Berlin'),  # Frankfurt
        'eu-west-1': pytz.timezone('Europe/Dublin'),  # Ireland
        'eu-west-2': pytz.timezone('Europe/London'),  # London
        'eu-south-1': pytz.timezone('Europe/Rome'),  # Milan
        'eu-west-3': pytz.timezone('Europe/Paris'),  # Paris
        'eu-south-2': pytz.timezone('Europe/Madrid'),  # Spain
        'eu-north-1': pytz.timezone('Europe/Stockholm'),  # Stockholm
        'eu-central-2': pytz.timezone('Europe/Zurich'),  # Zurich

        # Israel
        'il-central-1': pytz.timezone('Asia/Jerusalem'),  # Tel Aviv

        # Mexico
        'mx-central-1': pytz.timezone('America/Mexico_City'),  # Central

        # Middle East
        'me-south-1': pytz.timezone('Asia/Bahrain'),  # Bahrain
        'me-central-1': pytz.timezone('Asia/Dubai'),  # UAE

        # South America
        'sa-east-1': pytz.timezone('America/Sao_Paulo'),  # São Paulo

        # AWS GovCloud
        'us-gov-east-1': pytz.timezone('America/New_York'),  # US-East
        'us-gov-west-1': pytz.timezone('America/Los_Angeles'),  # US-West
    }

    # df = df[df['model_id'].str.contains('bedrock', case=False, na=False)]
    # Add local time information
    def get_local_time(row):
        if row['region'] in region_timezones:
            try:
                # Parse ISO timestamp
                utc_time = datetime.strptime(row['job_timestamp_iso'], '%Y-%m-%dT%H:%M:%SZ')
                utc_time = utc_time.replace(tzinfo=pytz.UTC)
                # Convert to local time
                local_time = utc_time.astimezone(region_timezones[row['region']])
                # Return formatted time and hour for grouping
                return pd.Series({
                    'local_time': local_time.strftime('%H:%M:%S'),
                    'hour_of_day': local_time.hour
                })
            except (ValueError, TypeError):
                return pd.Series({'local_time': 'Unknown', 'hour_of_day': -1})
        return pd.Series({'local_time': 'Unknown', 'hour_of_day': -1})

    # Add local time columns
    time_data = df.apply(get_local_time, axis=1)
    df = pd.concat([df, time_data], axis=1)
    df['average_input_output_token_size'] = df['input_tokens'] + df['output_tokens']
    # Group data by region
    regional_metrics = df.groupby(['region', 'task_types']).agg({
        'average_input_output_token_size': 'mean',
        'time_to_first_byte': 'mean',
        'time_to_last_byte': 'mean',
        'response_cost': 'mean',
        'inference_request_count': 'mean',
        'throughput_tps': 'mean',
        'hour_of_day': lambda x: x.mode()[0] if not x.empty else -1,
        'local_time': lambda x: x.iloc[0] if not x.empty else 'Unknown'
    }).reset_index()

    regional_metrics['average_input_output_token_size'] = regional_metrics['average_input_output_token_size'].round(1).astype("string")
    # Calculate time of day periods
    def get_time_period(hour):
        if hour == -1:
            return "Unknown"
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 22:
            return "Evening"
        else:
            return "Night"

    regional_metrics['time_period'] = regional_metrics['hour_of_day'].apply(get_time_period)

    # Calculate a composite score (lower latency, higher success, lower cost is better)
    max_latency = regional_metrics['time_to_last_byte'].max() or 1
    max_cost = regional_metrics['response_cost'].max() or 1

    regional_metrics['composite_score'] = (
            # regional_metrics['task_success'] +
            regional_metrics['inference_request_count'] +
            (1 - (regional_metrics['time_to_last_byte'] / max_latency)) +
            (1 - (regional_metrics['response_cost'] / max_cost))
    )

    regional_metrics['composite_label'] = regional_metrics['region'] + "<br>Mean of Total Token Size: " + regional_metrics['average_input_output_token_size']

    # Normalize the composite score
    min_score = regional_metrics['composite_score'].min()
    max_score = regional_metrics['composite_score'].max()
    regional_metrics['normalized_score'] = (regional_metrics['composite_score'] - min_score) / (max_score - min_score)

    # Create a figure with two subplots: latency vs cost, and time of day analysis
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Latency vs Cost by Region", 'Hourly Performance by Region<br><span style="font-size: 12px;">Using μ (Micro) Symbol for Small Numbers</span>'),
        vertical_spacing=0.30,  # Increased for more space between plots
        specs=[[{"type": "scatter"}], [{"type": "bar"}]],
    )

    fig.update_layout(template="plotly_dark")

    # Calculate min and max for scaling
    min_count = regional_metrics['inference_request_count'].min()
    max_count = regional_metrics['inference_request_count'].max()

    # Create a more dramatic size scale (20-100 instead of default)
    size_values = 20 + ((regional_metrics['inference_request_count'] - min_count) / (((max_count - min_count) * 50) + 1))

    # Add scatter plot for latency vs cost
    scatter = go.Scatter(
        x=regional_metrics['time_to_last_byte'],
        y=regional_metrics['response_cost'],
        mode='markers+text',
        marker=dict(
            size=size_values, #Size based on success rate
            # size=regional_metrics['inference_request_count'] * 50,
            color=regional_metrics['composite_score'],
            colorscale='Viridis',
            colorbar=dict(title="Composite Score", y=0.75, len=0.5),  # Positioned in top half
            showscale=True
        ),
        text=regional_metrics['composite_label'],
        textposition="top center",
        hovertemplate=
        '<b>%{text}</b><br>' +
        'Latency: %{x:.2f}s<br>' +
        'Cost: $%{y:.4f}<br>' +
        'Average Number of Retries: ' + regional_metrics['inference_request_count'].apply(lambda x: str(round(x,2)))+ '<br>' +
        'Local Time at Inference: ' + regional_metrics['local_time'] + '<br>' +
        'Time Period: ' + regional_metrics['time_period'] + '<br>',
        name='',
        showlegend=False
    )

    fig.add_trace(scatter, row=1, col=1)

    # Group data by region and hour for hourly analysis
    hourly_data = df.groupby(['region', 'hour_of_day']).agg({
        'throughput_tps': 'mean',
        'time_to_last_byte': 'mean'
    }).reset_index()

    hourly_data = hourly_data[hourly_data['hour_of_day'] != -1]  # Remove unknown hours

    # Add bar chart for hourly performance
    for region in regional_metrics['region'].unique():
        region_data = hourly_data[hourly_data['region'] == region]
        #### EQUALIZE N OF TASKS AND DATA PER REGION
        if not region_data.empty:
            bar = go.Bar(
                x=region_data['hour_of_day'],
                y=region_data['throughput_tps'],
                name=region,
                marker_color=px.colors.qualitative.Plotly[
                    list(regional_metrics['region']).index(region) % len(px.colors.qualitative.Plotly)],
                hovertemplate=
                'Region Inference Hour: %{x}:00<br>' +
                'Tokens Per Second: %{y:.2f}<br>' +
                'Avg Latency: ' + region_data['time_to_last_byte'].apply(lambda x: f"{x:.2f}s") + '<br>' +
                'Region: ' +  region_data['region']
            )

            fig.add_trace(bar, row=2, col=1)

    # Update layout with more spacing
    fig.update_layout(
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#2d2d2d",  # Slightly lighter than paper for contrast

        template="plotly_dark",  # Use the built-in dark template as a base
        title={
            'text': 'Regional Performance Analysis with Time of Day',
            'y': 0.98,  # Position title a bit lower from top
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=1000,  # Increased height
        legend_title_text='Region',
        # showlegend=False,
        margin=dict(t=150, b=150),   # More top and bottom margin
    legend=dict(
        y=0.30,  # Adjust this value to position the legend vertically (0.35 is approximately at the bottom subplot)
        x=1.05,  # Position legend to the right of the plot
        xanchor='left',  # Anchor legend to its left side
        yanchor='middle',  # Center legend vertically at the specified y position
        orientation='v'  # Arrange legend items vertically
    )
    )

    # Update x and y axes
    fig.update_xaxes(title_text="Average Latency (Secs)", row=1, col=1)
    fig.update_yaxes(title_text="Average Cost (USD)", row=1, col=1)

    fig.update_xaxes(
        title_text="Hour of Day (24-hour format)",
        tickmode='array',
        tickvals=list(range(0, 24, 3)),
        ticktext=[f"{h}:00" for h in range(0, 24, 3)],
        row=2, col=1
    )
    fig.update_yaxes(title_text="Throughput (TPS)", row=2, col=1)

    # Add recommendations based on data
    best_region_idx = regional_metrics['composite_score'].idxmax()
    best_region = regional_metrics.loc[best_region_idx]

    # Add annotations with recommendations - positioned with better spacing
    fig.add_annotation(
        x=0.5,
        y=0.99,  # Positioned right below title
        xref="paper",
        yref="paper",
        text=f"<b>Recommendation:</b> {best_region['region']} performed best with {str(round(best_region['throughput_tps'],3))} Tokens Per Second {best_region['local_time']} local time ({best_region['time_period']})",
        showarrow=False,
        font=dict(size=14, color="darkgreen"),
        bgcolor="rgba(200, 240, 200, 0.6)",
        bordercolor="green",
        borderwidth=2,
        borderpad=10,
        align="center"
    )
    return fig



# ==========================================
# UNIFIED LLM/RAG VISUALIZATION SUPPORT
# ==========================================

def detect_evaluation_types(directory, evaluation_names=None):
    """Detect what types of evaluations are present in the directory.

    Args:
        directory: Directory containing CSV files
        evaluation_names: Optional list of evaluation names to filter by

    Returns:
        dict: Dictionary with keys 'llm_files', 'rag_files', 'types' (['llm'], ['rag'], or ['llm', 'rag'])
    """
    directory = Path(directory)

    # Find LLM files (invocations_*.csv but NOT rag_invocations_*.csv)
    all_llm_files = glob.glob(str(directory / "invocations_*.csv"))
    # Filter out RAG files
    llm_files = [f for f in all_llm_files if not Path(f).name.startswith("rag_invocations_")]

    # Find RAG files (rag_invocations_*.csv)
    rag_files = glob.glob(str(directory / "rag_invocations_*.csv"))

    # Filter by evaluation names if specified
    if evaluation_names:
        filtered_llm = []
        filtered_rag = []

        for file_path in llm_files:
            if any(eval_name in Path(file_path).name for eval_name in evaluation_names):
                filtered_llm.append(file_path)

        for file_path in rag_files:
            if any(eval_name in Path(file_path).name for eval_name in evaluation_names):
                filtered_rag.append(file_path)

        llm_files = filtered_llm
        rag_files = filtered_rag

    # Determine types present
    types = []
    if llm_files:
        types.append('llm')
    if rag_files:
        types.append('rag')

    logger.info(f"Detected evaluation types: {types}")
    logger.info(f"LLM files: {len(llm_files)}, RAG files: {len(rag_files)}")

    return {
        'llm_files': llm_files,
        'rag_files': rag_files,
        'types': types
    }


def load_rag_data(files):
    """Load RAG evaluation data from CSV files.

    Args:
        files: List of RAG CSV file paths

    Returns:
        pd.DataFrame: Combined RAG data
    """
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            logger.info(f"Reading RAG file: {f}")
            df = pd.read_csv(f)
            logger.info(f"Read {len(df)} RAG records from {f}")
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading RAG file {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined RAG data: {len(combined)} total records")
    return combined


def detect_similarity_methods(df_rag):
    """
    Detect similarity methods used in the RAG evaluation from column names.

    Args:
        df_rag: DataFrame with RAG evaluation data

    Returns:
        list: List of detected similarity method names (e.g., ['jaccard', 'cosine'])
    """
    if df_rag.empty:
        return []

    detected_methods = set()

    # Look for suffixed metric columns
    # Pattern 1: context_precision_<method>, context_recall_<method>
    for col in df_rag.columns:
        if col.startswith('context_precision_') and col != 'context_precision':
            method = col.replace('context_precision_', '')
            detected_methods.add(method)
        elif col.startswith('context_recall_') and col != 'context_recall':
            method = col.replace('context_recall_', '')
            detected_methods.add(method)
        # Pattern 2: precision@K_<method>, recall@K_<method>
        elif 'precision@' in col and '_' in col:
            parts = col.split('_')
            if len(parts) == 2 and parts[0].startswith('precision@'):
                detected_methods.add(parts[1])
        elif 'recall@' in col and '_' in col:
            parts = col.split('_')
            if len(parts) == 2 and parts[0].startswith('recall@'):
                detected_methods.add(parts[1])

    # If no methods detected, check if base metrics exist (single method or no suffix)
    if not detected_methods:
        has_base_metrics = ('context_precision' in df_rag.columns or
                           'context_recall' in df_rag.columns or
                           any('precision@' in col and '_' not in col for col in df_rag.columns))
        if has_base_metrics:
            # Single method with no suffix
            return ['default']

    logger.info(f"Detected similarity methods from columns: {sorted(list(detected_methods))}")
    return sorted(list(detected_methods))


def get_metric_column_name(metric_base, similarity_method=None):
    """
    Get the actual column name for a metric, considering similarity method suffix.

    Args:
        metric_base: Base metric name (e.g., 'context_precision')
        similarity_method: Similarity method name (e.g., 'jaccard', 'cosine', or None/'default')

    Returns:
        str: Actual column name (e.g., 'context_precision' or 'context_precision_jaccard')
    """
    if similarity_method is None or similarity_method == 'default':
        return metric_base
    return f"{metric_base}_{similarity_method}"


def create_rag_visualizations(df_rag, similarity_method=None):
    """Create RAG-specific visualizations.

    Args:
        df_rag: DataFrame with RAG evaluation data
        similarity_method: Specific similarity method to visualize (None for default/single method)

    Returns:
        dict: Dictionary of RAG visualization figures
    """
    viz = {}

    if df_rag.empty:
        return viz

    try:
        # Get metric column names with method suffix
        precision_col = get_metric_column_name('context_precision', similarity_method)
        recall_col = get_metric_column_name('context_recall', similarity_method)
        relevancy_col = get_metric_column_name('context_relevancy', similarity_method)

        # 1. Embedding Model Comparison (Precision, Recall, Relevancy)
        if 'embedding_model' in df_rag.columns and precision_col in df_rag.columns:
            # Build aggregation dict dynamically based on available columns
            agg_dict = {
                precision_col: 'mean',
                recall_col: 'mean'
            }
            if relevancy_col in df_rag.columns:
                agg_dict[relevancy_col] = 'mean'

            metrics_by_model = df_rag.groupby('embedding_model').agg(agg_dict).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Bar(name='Precision', x=metrics_by_model['embedding_model'], y=metrics_by_model[precision_col]))
            fig.add_trace(go.Bar(name='Recall', x=metrics_by_model['embedding_model'], y=metrics_by_model[recall_col]))
            if relevancy_col in metrics_by_model.columns:
                fig.add_trace(go.Bar(name='Relevancy', x=metrics_by_model['embedding_model'], y=metrics_by_model[relevancy_col]))

            # Add similarity method to title if specified
            title = 'Embedding Model Performance Comparison'
            if similarity_method and similarity_method != 'default':
                title += f' ({similarity_method.replace("_", " ").title()})'

            fig.update_layout(
                title=title,
                xaxis_title='Embedding Model',
                yaxis_title='Score',
                barmode='group',
                template='plotly_dark'
            )
            viz['embedding_comparison'] = fig

        # 2. Retrieval Latency Analysis - Stacked Bar Breakdown
        if 'embedding_model' in df_rag.columns:
            # Calculate average latencies per embedding model
            latency_cols = ['query_embedding_latency', 'retrieval_latency', 'reranking_latency']
            available_cols = [col for col in latency_cols if col in df_rag.columns]

            if available_cols:
                # Group by embedding model and calculate mean latencies
                latency_data = df_rag.groupby('embedding_model')[available_cols].mean()

                # Convert to milliseconds (assume input is in seconds)
                latency_data_ms = latency_data * 1000

                # Sort by total latency (ascending - fastest first)
                latency_data_ms['total'] = latency_data_ms.sum(axis=1)
                latency_data_ms = latency_data_ms.sort_values('total')

                # Create stacked horizontal bar chart
                fig = go.Figure()

                # Add bars for each latency component
                colors = {'query_embedding_latency': '#3498db', 'retrieval_latency': '#2ecc71', 'reranking_latency': '#f39c12'}
                labels = {'query_embedding_latency': 'Query Embedding', 'retrieval_latency': 'Retrieval', 'reranking_latency': 'Reranking'}

                for col in available_cols:
                    fig.add_trace(go.Bar(
                        name=labels.get(col, col),
                        y=latency_data_ms.index,
                        x=latency_data_ms[col],
                        orientation='h',
                        marker_color=colors.get(col, '#95a5a6'),
                        text=[f"{val:.1f}ms" for val in latency_data_ms[col]],
                        textposition='inside',
                        hovertemplate='%{y}<br>' + labels.get(col, col) + ': %{x:.2f}ms<extra></extra>'
                    ))

                # Update layout
                fig.update_layout(
                    title='Latency Breakdown by Embedding Model',
                    xaxis_title='Latency (ms)',
                    yaxis_title='Embedding Model',
                    barmode='stack',
                    template='plotly_dark',
                    showlegend=True,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    height=max(400, len(latency_data_ms) * 60)  # Dynamic height based on number of models
                )

                viz['retrieval_latency'] = fig

        # 3. Cost Analysis by Embedding Model
        if 'total_cost' in df_rag.columns and 'embedding_model' in df_rag.columns:
            cost_by_model = df_rag.groupby('embedding_model')['total_cost'].sum().reset_index()
            fig = px.bar(cost_by_model, x='embedding_model', y='total_cost',
                        title='Total Cost by Embedding Model',
                        labels={'total_cost': 'Total Cost ($)', 'embedding_model': 'Embedding Model'},
                        template='plotly_dark')
            viz['cost_analysis'] = fig

        # 4. Precision@K and Recall@K Analysis
        try:
            # Find precision and recall columns with method suffix
            if similarity_method and similarity_method != 'default':
                method_suffix = f'_{similarity_method}'
                precision_cols = [col for col in df_rag.columns if col.startswith('precision@') and col.endswith(method_suffix)]
                recall_cols = [col for col in df_rag.columns if col.startswith('recall@') and col.endswith(method_suffix)]
            else:
                # For default, look for columns without method suffix
                precision_cols = [col for col in df_rag.columns if col.startswith('precision@') and '_' not in col.split('@')[1]]
                recall_cols = [col for col in df_rag.columns if col.startswith('recall@') and '_' not in col.split('@')[1]]

            if precision_cols and recall_cols and 'embedding_model' in df_rag.columns:
                fig = make_subplots(rows=1, cols=2, subplot_titles=('Precision@K', 'Recall@K'))

                for model in df_rag['embedding_model'].unique():
                    model_data = df_rag[df_rag['embedding_model'] == model]

                    # Extract K values from column names
                    k_values = []
                    for col in precision_cols:
                        # Handle both 'precision@5' and 'precision@5_jaccard' formats
                        k_part = col.split('@')[1].split('_')[0]
                        k_values.append(int(k_part))

                    precision_vals = [model_data[col].mean() for col in precision_cols]
                    recall_vals = [model_data[col].mean() for col in recall_cols]

                    fig.add_trace(go.Scatter(x=k_values, y=precision_vals, mode='lines+markers', name=f'{model} (P)', showlegend=True), row=1, col=1)
                    fig.add_trace(go.Scatter(x=k_values, y=recall_vals, mode='lines+markers', name=f'{model} (R)', showlegend=True), row=1, col=2)

                title = 'Precision@K and Recall@K by Embedding Model'
                if similarity_method and similarity_method != 'default':
                    title += f' ({similarity_method.replace("_", " ").title()})'

                fig.update_layout(title=title, template='plotly_dark')
                fig.update_xaxes(title_text='K', row=1, col=1)
                fig.update_xaxes(title_text='K', row=1, col=2)
                fig.update_yaxes(title_text='Precision', row=1, col=1)
                fig.update_yaxes(title_text='Recall', row=1, col=2)
                viz['precision_recall_at_k'] = fig
                logger.info("Created Precision@K and Recall@K visualization")
        except Exception as e:
            logger.error(f"Error creating Precision@K and Recall@K visualization: {e}", exc_info=True)

        # 5. Top-K Analysis (Fixed Metrics)
        try:
            # Get new metric columns
            concentration_cols = [col for col in df_rag.columns if 'concentration@' in col]
            diversity_cols = [col for col in df_rag.columns if 'diversity@' in col]
            dropoff_cols = [col for col in df_rag.columns if 'dropoff_k' in col]
            churn_cols = [col for col in df_rag.columns if 'churn_k' in col]

            # A. Rank Concentration Chart (Higher = quality at top)
            if concentration_cols and 'embedding_model' in df_rag.columns:
                concentration_data = df_rag.groupby('embedding_model')[concentration_cols].mean()

                if not concentration_data.empty:
                    fig = go.Figure()

                    for model in concentration_data.index:
                        x_labels = [col.replace('concentration@', 'K=') for col in concentration_cols]
                        y_values = concentration_data.loc[model].values

                        fig.add_trace(go.Scatter(
                            x=x_labels,
                            y=y_values,
                            mode='lines+markers',
                            name=model,
                            line=dict(width=2),
                            marker=dict(size=8)
                        ))

                    fig.update_layout(
                        title='Rank Concentration by Embedding Model (Higher = Better)',
                        xaxis_title='Top-K Value',
                        yaxis_title='Concentration Score',
                        template='plotly_dark',
                        hovermode='x unified',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                    )
                    viz['rank_concentration'] = fig
                    logger.info("Created Rank Concentration visualization")

            # B. Quality Drop-off Chart (Lower = better)
            if dropoff_cols and 'embedding_model' in df_rag.columns:
                dropoff_data = df_rag.groupby('embedding_model')[dropoff_cols].mean()

                if not dropoff_data.empty:
                    fig = go.Figure()

                    for model in dropoff_data.index:
                        x_labels = [col.replace('dropoff_', '').replace('_to_', ' → ') for col in dropoff_cols]
                        y_values = dropoff_data.loc[model].values

                        fig.add_trace(go.Scatter(
                            x=x_labels,
                            y=y_values,
                            mode='lines+markers',
                            name=model,
                            line=dict(width=2),
                            marker=dict(size=8)
                        ))

                    fig.update_layout(
                        title='Quality Drop-off as K Increases (Lower = Better)',
                        xaxis_title='K Transition',
                        yaxis_title='Drop-off Rate',
                        template='plotly_dark',
                        hovermode='x unified',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                    )
                    viz['quality_dropoff'] = fig
                    logger.info("Created Quality Drop-off visualization")

            # 7. Result Churn Rate Analysis
            if churn_cols and 'embedding_model' in df_rag.columns:
                churn_data = df_rag.groupby('embedding_model')[churn_cols].mean()

                if not churn_data.empty:
                    fig = go.Figure()

                    for model in churn_data.index:
                        x_labels = [col.replace('churn_', '').replace('_to_', ' → ') for col in churn_cols]
                        y_values = churn_data.loc[model].values

                        fig.add_trace(go.Scatter(
                            x=x_labels,
                            y=y_values,
                            mode='lines+markers',
                            name=model,
                            line=dict(width=2),
                            marker=dict(size=8)
                        ))

                    fig.update_layout(
                        title='Result Churn Rate as K Increases',
                        xaxis_title='K Transition',
                        yaxis_title='Churn Rate (New Results / K)',
                        template='plotly_dark',
                        hovermode='x unified',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                    )
                    viz['result_churn'] = fig
                    logger.info("Created Result Churn visualization")

            # 8. Average Rank Concentration by Model
            if 'avg_rank_concentration' in df_rag.columns and 'embedding_model' in df_rag.columns:
                concentration_by_model = df_rag.groupby('embedding_model')['avg_rank_concentration'].mean().sort_values(ascending=False).reset_index()

                fig = go.Figure(go.Bar(
                    y=concentration_by_model['embedding_model'],
                    x=concentration_by_model['avg_rank_concentration'],
                    orientation='h',
                    marker=dict(
                        color=concentration_by_model['avg_rank_concentration'],
                        colorscale='RdYlGn',
                        cmin=0,
                        cmax=1,
                        colorbar=dict(title='Concentration Score')
                    ),
                    text=[f'{val:.3f}' for val in concentration_by_model['avg_rank_concentration']],
                    textposition='inside',
                    hovertemplate='%{y}<br>Avg Concentration: %{x:.3f}<extra></extra>'
                ))

                fig.update_layout(
                    title='Average Rank Concentration by Embedding Model (Higher = Better)',
                    xaxis_title='Average Concentration Score',
                    yaxis_title='Embedding Model',
                    template='plotly_dark',
                    height=max(400, len(concentration_by_model) * 50)
                )
                viz['avg_concentration'] = fig
                logger.info("Created Average Concentration visualization")

        except Exception as e:
            logger.error(f"Error creating Top-K Stability visualizations: {e}", exc_info=True)

        # 9. Cost Efficiency Frontier
        try:
            if 'context_precision' in df_rag.columns and 'total_cost' in df_rag.columns and 'embedding_model' in df_rag.columns:
                # Calculate average metrics per model
                efficiency_data = df_rag.groupby('embedding_model').agg({
                    'context_precision': 'mean',
                    'context_recall': 'mean',
                    'total_cost': 'mean',
                    'total_latency': 'mean'
                }).reset_index()

                # Calculate F1 score (harmonic mean of precision and recall)
                efficiency_data['f1_score'] = 2 * (efficiency_data['context_precision'] * efficiency_data['context_recall']) / \
                                              (efficiency_data['context_precision'] + efficiency_data['context_recall'])
                efficiency_data['f1_score'] = efficiency_data['f1_score'].fillna(0)

                # Cost Efficiency: Calculate raw efficiency first
                cost_per_1k_queries = efficiency_data['total_cost'] * 1000
                raw_efficiency = efficiency_data['f1_score'] / (cost_per_1k_queries + 0.00001)

                # Normalize to 0-100 scale for intuitive interpretation
                min_eff = raw_efficiency.min()
                max_eff = raw_efficiency.max()
                if max_eff > min_eff:
                    efficiency_data['cost_efficiency'] = ((raw_efficiency - min_eff) / (max_eff - min_eff)) * 100
                else:
                    efficiency_data['cost_efficiency'] = 50.0  # Default if all same

                # Latency Efficiency: F1 Score per Second (multiply by 10 for readability)
                efficiency_data['latency_efficiency'] = (efficiency_data['f1_score'] / (efficiency_data['total_latency'] + 0.00001)) * 10

                # Create scatter plot
                fig = go.Figure()

                # Add scatter points
                fig.add_trace(go.Scatter(
                    x=efficiency_data['total_cost'],
                    y=efficiency_data['f1_score'],
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color=efficiency_data['cost_efficiency'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(
                            title='Cost<br>Efficiency<br>(0-100)',
                            x=1.15,  # Move colorbar further right
                            thickness=15,
                            len=0.7
                        )
                    ),
                    text=efficiency_data['embedding_model'].apply(lambda x: x.split('/')[-1][:15]),
                    textposition='top center',
                    textfont=dict(size=9),
                    hovertemplate='<b>%{text}</b><br>' +
                                  'F1 Score: %{y:.3f}<br>' +
                                  'Avg Cost per Query: $%{x:.6f}<br>' +
                                  'Cost Efficiency: %{marker.color:.0f}/100<extra></extra>',
                    name=''
                ))

                # Find Pareto frontier (models that are not dominated by any other)
                pareto_models = []
                for i, row in efficiency_data.iterrows():
                    is_pareto = True
                    for j, other_row in efficiency_data.iterrows():
                        if i != j:
                            # If another model has better F1 AND lower cost, current model is dominated
                            if other_row['f1_score'] >= row['f1_score'] and other_row['total_cost'] <= row['total_cost']:
                                if other_row['f1_score'] > row['f1_score'] or other_row['total_cost'] < row['total_cost']:
                                    is_pareto = False
                                    break
                    if is_pareto:
                        pareto_models.append(i)

                # Draw Pareto frontier line
                if len(pareto_models) > 1:
                    pareto_data = efficiency_data.loc[pareto_models].sort_values('total_cost')
                    fig.add_trace(go.Scatter(
                        x=pareto_data['total_cost'],
                        y=pareto_data['f1_score'],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name='Pareto Frontier',
                        hoverinfo='skip'
                    ))

                fig.update_layout(
                    title='Cost Efficiency Frontier (F1 Score vs Cost)',
                    xaxis_title='Average Total Cost ($)',
                    yaxis_title='F1 Score (Precision + Recall)',
                    template='plotly_dark',
                    hovermode='closest',
                    height=600,
                    showlegend=True,
                    margin=dict(r=150)  # Add right margin for colorbar
                )
                viz['cost_efficiency_frontier'] = fig
                logger.info("Created Cost Efficiency Frontier visualization")

        except Exception as e:
            logger.error(f"Error creating Cost Efficiency Frontier: {e}", exc_info=True)

        # 10. Latency-Accuracy Tradeoff
        try:
            if 'context_precision' in df_rag.columns and 'total_latency' in df_rag.columns and 'embedding_model' in df_rag.columns:
                latency_acc_data = df_rag.groupby('embedding_model').agg({
                    'context_precision': 'mean',
                    'context_recall': 'mean',
                    'total_latency': 'mean'
                }).reset_index()

                # Calculate F1 score
                latency_acc_data['f1_score'] = 2 * (latency_acc_data['context_precision'] * latency_acc_data['context_recall']) / \
                                               (latency_acc_data['context_precision'] + latency_acc_data['context_recall'])
                latency_acc_data['f1_score'] = latency_acc_data['f1_score'].fillna(0)

                # Convert latency to milliseconds
                latency_acc_data['latency_ms'] = latency_acc_data['total_latency'] * 1000

                # Create scatter plot
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=latency_acc_data['latency_ms'],
                    y=latency_acc_data['f1_score'],
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color=latency_acc_data['f1_score'],
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title='F1<br>Score')
                    ),
                    text=latency_acc_data['embedding_model'].apply(lambda x: x.split('/')[-1][:15]),
                    textposition='top center',
                    textfont=dict(size=9),
                    hovertemplate='<b>%{text}</b><br>' +
                                  'F1 Score: %{y:.3f}<br>' +
                                  'Latency: %{x:.1f}ms<extra></extra>',
                    name=''
                ))

                fig.update_layout(
                    title='Latency-Accuracy Tradeoff',
                    xaxis_title='Average Total Latency (ms)',
                    yaxis_title='F1 Score (Precision + Recall)',
                    template='plotly_dark',
                    hovermode='closest',
                    height=600
                )
                viz['latency_accuracy_tradeoff'] = fig
                logger.info("Created Latency-Accuracy Tradeoff visualization")

        except Exception as e:
            logger.error(f"Error creating Latency-Accuracy Tradeoff: {e}", exc_info=True)

        # 11. Cost per Query Breakdown
        try:
            if 'embedding_cost' in df_rag.columns and 'embedding_model' in df_rag.columns:
                # Calculate average costs per embedding model
                cost_cols = []
                if 'embedding_cost' in df_rag.columns:
                    cost_cols.append('embedding_cost')
                if 'reranking_cost' in df_rag.columns:
                    cost_cols.append('reranking_cost')
                if 'total_cost' in df_rag.columns:
                    cost_cols.append('total_cost')

                if cost_cols:
                    cost_data = df_rag.groupby('embedding_model')[cost_cols].mean().reset_index()

                    # Sort by total cost
                    if 'total_cost' in cost_data.columns:
                        cost_data = cost_data.sort_values('total_cost', ascending=False)

                    fig = go.Figure()

                    # Add stacked bars for cost breakdown
                    if 'embedding_cost' in cost_data.columns:
                        fig.add_trace(go.Bar(
                            name='Embedding Cost',
                            y=cost_data['embedding_model'],
                            x=cost_data['embedding_cost'],
                            orientation='h',
                            marker_color='#3498db',
                            text=[f'${val:.6f}' for val in cost_data['embedding_cost']],
                            textposition='inside',
                            hovertemplate='Embedding: $%{x:.6f}<extra></extra>'
                        ))

                    if 'reranking_cost' in cost_data.columns:
                        fig.add_trace(go.Bar(
                            name='Reranking Cost',
                            y=cost_data['embedding_model'],
                            x=cost_data['reranking_cost'],
                            orientation='h',
                            marker_color='#f39c12',
                            text=[f'${val:.6f}' for val in cost_data['reranking_cost']],
                            textposition='inside',
                            hovertemplate='Reranking: $%{x:.6f}<extra></extra>'
                        ))

                    fig.update_layout(
                        title='Cost per Query Breakdown by Embedding Model',
                        xaxis_title='Cost per Query ($)',
                        yaxis_title='Embedding Model',
                        barmode='stack',
                        template='plotly_dark',
                        showlegend=True,
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        height=max(400, len(cost_data) * 60)
                    )

                    viz['cost_per_query'] = fig
                    logger.info("Created Cost per Query visualization")

        except Exception as e:
            logger.error(f"Error creating Cost per Query visualization: {e}", exc_info=True)

        # 12. Context Window Fit Analysis
        try:
            # Find all token count columns
            token_cols = [col for col in df_rag.columns if col.startswith('tokens@')]

            if token_cols and 'embedding_model' in df_rag.columns:
                # Calculate average token counts per model and K value
                token_data = df_rag.groupby('embedding_model')[token_cols].mean().reset_index()

                # Extract K values
                k_values = [int(col.split('@')[1]) for col in token_cols]

                # Create line chart showing token counts at different K values
                fig = go.Figure()

                for _, row in token_data.iterrows():
                    model_name = row['embedding_model']
                    token_counts = [row[col] for col in token_cols]

                    fig.add_trace(go.Scatter(
                        x=k_values,
                        y=token_counts,
                        mode='lines+markers',
                        name=model_name,
                        hovertemplate=f'<b>{model_name}</b><br>K=%{{x}}<br>Tokens: %{{y:.0f}}<extra></extra>'
                    ))

                # Add horizontal reference lines for common context windows
                context_windows = [
                    (16000, 'GPT-3.5 16K', '#e74c3c'),
                    (32000, 'GPT-4 32K', '#f39c12'),
                    (128000, 'GPT-4 128K', '#3498db'),
                    (200000, 'Claude 200K', '#2ecc71')
                ]

                for limit, label, color in context_windows:
                    fig.add_hline(
                        y=limit,
                        line_dash='dash',
                        line_color=color,
                        annotation_text=label,
                        annotation_position='right'
                    )

                fig.update_layout(
                    title='Context Window Fit: Token Count by K Value',
                    xaxis_title='K (Number of Retrieved Chunks)',
                    yaxis_title='Total Tokens',
                    template='plotly_dark',
                    hovermode='x unified',
                    legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.05),
                    height=600,
                    yaxis_type='log'  # Log scale for better visualization
                )

                viz['context_window_fit'] = fig
                logger.info("Created Context Window Fit visualization")

        except Exception as e:
            logger.error(f"Error creating Context Window visualizations: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Error creating RAG visualizations: {e}", exc_info=True)

    return viz


def generate_rag_llm_summary(df_rag, similarity_method='default'):
    """
    Generate an LLM-powered executive summary of RAG evaluation results.

    Args:
        df_rag: DataFrame with RAG evaluation data
        similarity_method: The similarity calculation method used.
                          If None, generates comprehensive summary across all methods.
                          If 'default', uses all data without filtering.
                          Otherwise filters by specific method.

    Returns:
        str: HTML-formatted executive summary, or empty string if generation fails
    """
    try:
        import litellm

        if df_rag.empty:
            return ""

        # Filter by similarity method if specified
        if similarity_method is None:
            # None = comprehensive summary across ALL methods
            df_filtered = df_rag.copy()
            similarity_method = 'all'  # For display purposes
        elif similarity_method != 'default':
            df_filtered = df_rag[df_rag['similarity_method'] == similarity_method].copy()
            if df_filtered.empty:
                df_filtered = df_rag.copy()  # Fallback to all data
        else:
            df_filtered = df_rag.copy()

        # Calculate key metrics for the summary
        metrics_summary = {}

        # Overall metrics
        metrics_summary['total_queries'] = len(df_filtered)
        metrics_summary['total_scenarios'] = df_filtered['scenario_id'].nunique() if 'scenario_id' in df_filtered.columns else len(df_filtered)

        # Embedding model performance
        if 'embedding_model' in df_filtered.columns:
            # Determine which k value to use (prefer @5, fallback to @3, then @1)
            # Look for columns with OR without similarity method suffix
            k_value = None
            for k in [5, 3, 1]:
                # Check for base columns (no suffix) OR any columns containing @k
                precision_cols = [col for col in df_filtered.columns if f'precision@{k}' in col]
                recall_cols = [col for col in df_filtered.columns if f'recall@{k}' in col]

                if precision_cols and recall_cols:
                    k_value = k
                    break

            if k_value is None:
                logger.warning("No precision@k/recall@k columns found")
                return ""

            # Calculate F1 from precision and recall
            # Use the first available column (handles both suffixed and non-suffixed columns)
            precision_cols = [col for col in df_filtered.columns if f'precision@{k_value}' in col]
            recall_cols = [col for col in df_filtered.columns if f'recall@{k_value}' in col]
            ndcg_cols = [col for col in df_filtered.columns if f'ndcg@{k_value}' in col]
            hit_rate_cols = [col for col in df_filtered.columns if f'hit_rate@{k_value}' in col]

            precision_col = precision_cols[0] if precision_cols else f'precision@{k_value}'
            recall_col = recall_cols[0] if recall_cols else f'recall@{k_value}'
            ndcg_col = ndcg_cols[0] if ndcg_cols else f'ndcg@{k_value}'
            hit_rate_col = hit_rate_cols[0] if hit_rate_cols else f'hit_rate@{k_value}'

            # Add F1 calculation
            df_filtered['f1_score'] = df_filtered.apply(
                lambda row: 2 * (row[precision_col] * row[recall_col]) / (row[precision_col] + row[recall_col])
                if (row[precision_col] + row[recall_col]) > 0 else 0,
                axis=1
            )

            # Aggregate metrics - include NDCG and Hit Rate if available
            agg_dict = {
                precision_col: ['mean', 'std'],
                recall_col: ['mean', 'std'],
                'f1_score': ['mean', 'std'],
                'retrieval_latency': ['mean', 'median'],
                'total_cost': 'sum'
            }

            # Add NDCG if available
            if ndcg_col in df_filtered.columns:
                agg_dict[ndcg_col] = ['mean', 'std']

            # Add Hit Rate if available
            if hit_rate_col in df_filtered.columns:
                agg_dict[hit_rate_col] = ['mean']

            embedding_stats = df_filtered.groupby('embedding_model').agg(agg_dict).round(4)

            # Find best performing model
            best_f1_model = embedding_stats[('f1_score', 'mean')].idxmax()
            best_f1_score = embedding_stats.loc[best_f1_model, ('f1_score', 'mean')]

            fastest_model = embedding_stats[('retrieval_latency', 'median')].idxmin()
            fastest_latency = embedding_stats.loc[fastest_model, ('retrieval_latency', 'median')]

            cheapest_model = embedding_stats[('total_cost', 'sum')].idxmin()
            cheapest_cost = embedding_stats.loc[cheapest_model, ('total_cost', 'sum')]

            metrics_summary['k_value'] = k_value
            metrics_summary['precision_col'] = precision_col
            metrics_summary['recall_col'] = recall_col
            metrics_summary['best_f1_model'] = best_f1_model
            metrics_summary['best_f1_score'] = best_f1_score
            metrics_summary['fastest_model'] = fastest_model
            metrics_summary['fastest_latency'] = fastest_latency
            metrics_summary['cheapest_model'] = cheapest_model
            metrics_summary['cheapest_cost'] = cheapest_cost
            metrics_summary['num_models'] = len(embedding_stats)

        # Re-ranker analysis
        if 'reranker_id' in df_filtered.columns and k_value:
            # Calculate F1 for reranker analysis
            df_filtered['f1_rerank'] = df_filtered.apply(
                lambda row: 2 * (row[precision_col] * row[recall_col]) / (row[precision_col] + row[recall_col])
                if (row[precision_col] + row[recall_col]) > 0 else 0,
                axis=1
            )

            reranker_stats = df_filtered.groupby('reranker_id').agg({
                precision_col: 'mean',
                recall_col: 'mean',
                'f1_rerank': 'mean'
            }).round(4)

            if len(reranker_stats) > 1:
                # Calculate improvement from re-ranking
                no_rerank = reranker_stats.loc[reranker_stats.index == 'none'] if 'none' in reranker_stats.index else None
                with_rerank = reranker_stats.loc[reranker_stats.index != 'none']

                if no_rerank is not None and not with_rerank.empty:
                    best_reranker = with_rerank['f1_rerank'].idxmax()
                    improvement = (with_rerank.loc[best_reranker, 'f1_rerank'] - no_rerank['f1_rerank'].iloc[0]) * 100
                    metrics_summary['best_reranker'] = best_reranker
                    metrics_summary['reranker_improvement'] = improvement

        # Create summary text for LLM
        k_val = metrics_summary.get('k_value', 5)
        summary_data = f"""
RAG Evaluation Results Summary:
- Total Queries Evaluated: {metrics_summary.get('total_queries', 0)}
- Total Scenarios: {metrics_summary.get('total_scenarios', 0)}
- Similarity Method: {similarity_method}
- Evaluation Metric: Top-{k_val} retrieval (Precision@{k_val}, Recall@{k_val}, F1@{k_val})
- Number of Embedding Models Tested: {metrics_summary.get('num_models', 0)}

Top Performers:
- Best F1@{k_val} Score: {metrics_summary.get('best_f1_model', 'N/A')} with F1={metrics_summary.get('best_f1_score', 0):.4f}
- Fastest Retrieval: {metrics_summary.get('fastest_model', 'N/A')} with {metrics_summary.get('fastest_latency', 0):.4f}s median latency
- Most Cost-Effective: {metrics_summary.get('cheapest_model', 'N/A')} with ${metrics_summary.get('cheapest_cost', 0):.6f} total cost
"""

        if 'best_reranker' in metrics_summary:
            summary_data += f"\nRe-ranking Impact:\n- Best Re-ranker: {metrics_summary['best_reranker']}\n- F1 Improvement: +{metrics_summary['reranker_improvement']:.2f}%\n"

        # Add detailed stats table
        if 'embedding_model' in df_filtered.columns and 'embedding_stats' in locals():
            summary_data += f"\nDetailed Model Performance (at K={k_val}):\n"
            for model in embedding_stats.index:
                p = embedding_stats.loc[model, (precision_col, 'mean')]
                r = embedding_stats.loc[model, (recall_col, 'mean')]
                f1 = embedding_stats.loc[model, ('f1_score', 'mean')]
                lat = embedding_stats.loc[model, ('retrieval_latency', 'median')]
                cost = embedding_stats.loc[model, ('total_cost', 'sum')]

                stats_line = f"- {model}: P@{k_val}={p:.4f}, R@{k_val}={r:.4f}, F1@{k_val}={f1:.4f}"

                # Add NDCG if available
                if ndcg_col in df_filtered.columns:
                    ndcg = embedding_stats.loc[model, (ndcg_col, 'mean')]
                    stats_line += f", NDCG@{k_val}={ndcg:.4f}"

                # Add Hit Rate if available
                if hit_rate_col in df_filtered.columns:
                    hit = embedding_stats.loc[model, (hit_rate_col, 'mean')]
                    hit_pct = hit * 100
                    stats_line += f", Hit Rate={hit_pct:.1f}%"

                stats_line += f", Latency={lat:.4f}s, Cost=${cost:.6f}\n"
                summary_data += stats_line

        # Generate LLM summary
        prompt = f"""Based on the data below, write a clear summary (2-3 paragraphs) using simple, non-technical language that explains:

1. What was tested - How many queries were run and how many embedding models were compared
2. Key results - Which models performed best for accuracy, speed, and cost (state the actual numbers)
3. Performance patterns - How the models compare to each other across different metrics

Important formatting rules:
- Use **bold** (markdown format) for all embedding model names
- Focus only on factual results - do NOT give recommendations or suggestions
- Use simple language - avoid technical jargon where possible
- Include specific numbers from the data (F1 scores, latency, costs)
- Explain what the metrics mean in plain terms (e.g., "F1 score measures how well the model finds relevant information")
- Do NOT include a title or heading in your response - start directly with the content

Data:
{summary_data}

Write in a clear, factual tone that presents the test results objectively."""

        # Call LLM API (using Haiku 4.5 for cost-effectiveness)
        logger.info("Generating LLM summary for RAG evaluation...")
        response = litellm.completion(
            model="bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )

        summary_text = response.choices[0].message.content

        # Convert markdown bold to HTML bold and handle paragraph spacing
        import re
        summary_html = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', summary_text)

        # Replace double newlines with paragraph breaks, single newlines with just a break
        summary_html = summary_html.replace('\n\n', '</p><p>')
        summary_html = summary_html.replace('\n', '<br>')
        summary_html = f'<p>{summary_html}</p>'

        # Format as HTML
        html_summary = f"""
    <div class="llm-summary" style="background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 100%); padding: 25px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #4CAF50;">
        <h3 style="color: #4CAF50; margin-top: 0; display: flex; align-items: center; gap: 10px;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
            AI-Generated Summary
        </h3>
        <div style="color: #e0e0e0; line-height: 1.8; font-size: 15px;">
            {summary_html}
        </div>
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 13px; color: #888;">
            <em>Generated by Claude 3.5 Haiku • Based on {metrics_summary.get('total_queries', 0)} queries across {metrics_summary.get('num_models', 0)} embedding models</em>
        </div>
    </div>
"""

        logger.info("Successfully generated LLM summary")
        return html_summary

    except Exception as e:
        logger.error(f"Error generating LLM summary: {e}", exc_info=True)
        return ""


def create_unified_html_report(output_dir, timestamp, evaluation_names=None):
    """Generate unified HTML report supporting LLM, RAG, or Combined evaluations.

    Args:
        output_dir: Directory containing CSV files
        timestamp: Timestamp for report filename
        evaluation_names: Optional list of evaluation names to filter by

    Returns:
        str: Path to generated HTML report
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect what types of evaluations are present
    detection = detect_evaluation_types(output_dir, evaluation_names)
    eval_types = detection['types']

    if not eval_types:
        raise FileNotFoundError(f"No evaluation files found in {output_dir}")

    logger.info(f"Creating unified report for evaluation types: {eval_types}")

    # Load data based on detected types
    df_llm = pd.DataFrame()
    df_rag = pd.DataFrame()

    if 'llm' in eval_types:
        logger.info("Loading LLM evaluation data...")
        try:
            df_llm = load_data(output_dir, evaluation_names)
            logger.info(f"Loaded {len(df_llm)} LLM records")
        except Exception as e:
            logger.error(f"Error loading LLM data: {e}")

    if 'rag' in eval_types:
        logger.info("Loading RAG evaluation data...")
        try:
            df_rag = load_rag_data(detection['rag_files'])
            logger.info(f"Loaded {len(df_rag)} RAG records")
        except Exception as e:
            logger.error(f"Error loading RAG data: {e}")

    # Determine report type for naming
    if len(eval_types) == 2:
        report_type = 'unified'
        logger.info("Creating combined LLM+RAG report")
    elif 'rag' in eval_types:
        report_type = 'rag'
        logger.info("Creating RAG-only report")
    else:
        report_type = 'llm'
        logger.info("Creating LLM-only report")
        # For LLM-only, use existing function
        return create_html_report(output_dir, timestamp, evaluation_names)

    # Create visualizations for each type
    llm_viz = {}
    rag_viz = {}

    if not df_llm.empty:
        logger.info("Creating LLM visualizations...")
        try:
            model_task_metrics = calculate_metrics_by_model_task(df_llm)
            latency_metrics = calculate_latency_metrics(df_llm)
            cost_metrics = calculate_cost_metrics(df_llm)
            llm_viz = create_visualizations(df_llm, model_task_metrics, latency_metrics, cost_metrics)
        except Exception as e:
            logger.error(f"Error creating LLM visualizations: {e}", exc_info=True)

    # Detect similarity methods and create visualizations for each
    similarity_methods = []
    rag_viz_by_method = {}

    if not df_rag.empty:
        logger.info("Detecting similarity methods...")
        similarity_methods = detect_similarity_methods(df_rag)
        logger.info(f"Found similarity methods: {similarity_methods}")

        if not similarity_methods:
            # No methods detected, use default
            similarity_methods = ['default']

        logger.info("Creating RAG visualizations for each similarity method...")
        for method in similarity_methods:
            logger.info(f"Creating visualizations for method: {method}")
            viz = create_rag_visualizations(df_rag, similarity_method=method)
            rag_viz_by_method[method] = viz

        # For backward compatibility, set rag_viz to first method
        rag_viz = rag_viz_by_method.get(similarity_methods[0], {})

    # Generate report filename
    if evaluation_names:
        eval_suffix = "_" + "_".join(evaluation_names[:3])
        if len(evaluation_names) > 3:
            eval_suffix += f"_and_{len(evaluation_names)-3}_more"
    else:
        eval_suffix = ""

    report_filename = f"llm_benchmark_report_{timestamp}_{report_type}{eval_suffix}.html"
    report_path = output_dir / report_filename

    # For now, create a simple HTML report
    # TODO: This will be enhanced when html_template.txt is updated
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report_type.upper()} Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #1e1e1e; color: #e0e0e0; }}
        h1 {{ color: #4CAF50; }}
        h2 {{ color: #2196F3; border-bottom: 2px solid #2196F3; padding-bottom: 5px; }}
        .section {{ margin: 30px 0; }}
        .info {{ background-color: #2d2d2d; padding: 15px; border-left: 4px solid #2196F3; }}
    </style>
</head>
<body>
    <h1>{report_type.upper()} Benchmark Report</h1>
    <div class="info">
        <p>Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
        <p>Report Type: {', '.join([t.upper() for t in eval_types])}</p>
        <p>Evaluations: {', '.join(evaluation_names) if evaluation_names else 'All'}</p>
    </div>
"""

    # Add LLM section if present
    if llm_viz:
        html_content += """
    <div class="section">
        <h2>LLM Evaluation Results</h2>
"""
        if 'ttft_comparison' in llm_viz:
            html_content += f"<div>{llm_viz['ttft_comparison'].to_html(full_html=False)}</div>"
        if 'cost_comparison' in llm_viz:
            html_content += f"<div>{llm_viz['cost_comparison'].to_html(full_html=False)}</div>"
        html_content += "</div>"

    # Add RAG section if present
    if rag_viz_by_method:
        html_content += """
    <div class="section">
        <h2>RAG Evaluation Results</h2>
"""

        # Generate ONE comprehensive AI summary for all data (before the dropdown)
        logger.info("Generating comprehensive LLM summary for all RAG data...")
        llm_summary = generate_rag_llm_summary(df_rag, similarity_method=None)  # None = all data
        if llm_summary:
            html_content += llm_summary

        # Add similarity method dropdown if multiple methods
        if len(similarity_methods) > 1:
            html_content += """
        <div style="margin: 20px 0; padding: 15px; background-color: #2d2d2d; border-radius: 5px;">
            <label for="similarityMethodSelect" style="font-weight: bold; margin-right: 10px;">
                Similarity Method:
            </label>
            <select id="similarityMethodSelect" onchange="switchSimilarityMethod()" style="padding: 5px 10px; background-color: #1e1e1e; color: #e0e0e0; border: 1px solid #4CAF50; border-radius: 3px; cursor: pointer;">
"""
            for method in similarity_methods:
                display_name = method.replace('_', ' ').title() if method != 'default' else 'Default (Jaccard)'
                html_content += f'                <option value="{method}">{display_name}</option>\n'

            html_content += """
            </select>
            <p style="margin-top: 10px; font-size: 0.9em; color: #888;">
                Select a similarity calculation method to view corresponding metrics and visualizations.
            </p>
        </div>

        <script>
        function switchSimilarityMethod() {
            const selectedMethod = document.getElementById('similarityMethodSelect').value;

            // Hide all method sections
            const allSections = document.querySelectorAll('.similarity-method-section');
            allSections.forEach(section => {
                section.style.display = 'none';
            });

            // Show selected method section
            const selectedSection = document.getElementById('similarity-method-' + selectedMethod);
            if (selectedSection) {
                selectedSection.style.display = 'block';

                // Force Plotly charts to resize properly
                setTimeout(function() {
                    // Find all Plotly charts in the selected section
                    const plotlyCharts = selectedSection.querySelectorAll('.plotly-graph-div');

                    plotlyCharts.forEach(function(chart) {
                        // Method 1: Use Plotly's resize function if available
                        if (window.Plotly && chart._fullLayout) {
                            try {
                                window.Plotly.Plots.resize(chart);
                            } catch (e) {
                                console.log('Plotly resize failed:', e);
                            }
                        }

                        // Method 2: Redraw if resize didn't work
                        if (window.Plotly && chart._fullData) {
                            try {
                                window.Plotly.redraw(chart);
                            } catch (e) {
                                console.log('Plotly redraw failed:', e);
                            }
                        }
                    });

                    // Method 3: Dispatch global resize event
                    window.dispatchEvent(new Event('resize'));
                }, 50);
            }
        }
        </script>
"""

        # Create sections for each similarity method
        for i, method in enumerate(similarity_methods):
            display_style = 'block' if i == 0 else 'none'  # Show first method by default
            html_content += f"""
        <div id="similarity-method-{method}" class="similarity-method-section" style="display: {display_style};">
"""

            method_viz = rag_viz_by_method.get(method, {})

            if method_viz:
                html_content += """
            <h3>Core Performance Metrics</h3>
"""
                if 'embedding_comparison' in method_viz:
                    html_content += f"<div>{method_viz['embedding_comparison'].to_html(full_html=False)}</div>"
                    html_content += """
            <div class="info">
                <p><b>What this shows:</b> Precision = % of retrieved chunks that are relevant | Recall = % of relevant chunks that were retrieved | Higher is better for both</p>
            </div>
"""
                if 'precision_recall_at_k' in method_viz:
                    html_content += f"<div>{method_viz['precision_recall_at_k'].to_html(full_html=False)}</div>"
                    html_content += """
            <div class="info">
                <p><b>What this shows:</b> How precision and recall change as K (number of retrieved chunks) increases | Helps you find the optimal K value for your use case</p>
            </div>
"""

                html_content += """
            <h3>Cost & Performance Analysis</h3>
"""
                if 'cost_efficiency_frontier' in method_viz:
                    html_content += f"<div>{method_viz['cost_efficiency_frontier'].to_html(full_html=False)}</div>"
                    html_content += """
            <div class="info">
                <p><b>What this shows:</b> F1 Score vs Cost trade-off | Models on the Pareto frontier (red dashed line) are optimal choices | Look for high F1 with low cost</p>
            </div>
"""
                if 'latency_accuracy_tradeoff' in method_viz:
                    html_content += f"<div>{method_viz['latency_accuracy_tradeoff'].to_html(full_html=False)}</div>"
                    html_content += """
            <div class="info">
                <p><b>What this shows:</b> F1 Score vs Latency trade-off | Find the sweet spot between accuracy and speed for your application</p>
            </div>
"""
                if 'cost_analysis' in method_viz:
                    html_content += f"<div>{method_viz['cost_analysis'].to_html(full_html=False)}</div>"
                    html_content += """
            <div class="info">
                <p><b>What this shows:</b> Total cost aggregated across all queries | Helps identify the most cost-effective embedding models</p>
            </div>
"""
                if 'cost_per_query' in method_viz:
                    html_content += f"<div>{method_viz['cost_per_query'].to_html(full_html=False)}</div>"
                    html_content += """
            <div class="info">
                <p><b>What this shows:</b> Average cost breakdown per query | Embedding Cost = cost to generate query/document embeddings | Reranking Cost = cost to reorder results | Lower is cheaper</p>
            </div>
"""

                html_content += """
            <h3>Latency Breakdown</h3>
"""
                if 'retrieval_latency' in method_viz:
                    html_content += f"<div>{method_viz['retrieval_latency'].to_html(full_html=False)}</div>"
                    html_content += """
            <div class="info">
                <p><b>What this shows:</b> Time breakdown for each retrieval step | Query Embedding = convert query to vector | Retrieval = search vector DB | Reranking = reorder results | Lower is faster</p>
            </div>
"""

                html_content += """
            <h3>Top-K Stability Analysis</h3>
"""
                if 'topk_stability' in method_viz:
                    html_content += f"<div>{method_viz['topk_stability'].to_html(full_html=False)}</div>"
                if 'avg_stability' in method_viz:
                    html_content += f"<div>{method_viz['avg_stability'].to_html(full_html=False)}</div>"
                if 'result_churn' in method_viz:
                    html_content += f"<div>{method_viz['result_churn'].to_html(full_html=False)}</div>"
                html_content += """
            <div class="info">
                <p><b>What this shows:</b> Consistency of retrieval results across different K values | Stability Score = how similar results are when changing K | Result Churn = rate of change in retrieved documents | Higher stability means more consistent results</p>
            </div>
"""

                # Context Window Analysis (new)
                if 'context_window_fit' in method_viz:
                    html_content += """
            <h3>Context Window Analysis</h3>
"""
                    html_content += f"<div>{method_viz['context_window_fit'].to_html(full_html=False)}</div>"
                    html_content += """
            <div class="info">
                <p><b>What this shows:</b> Total tokens needed to fit K retrieved chunks | Dashed lines show LLM context window limits | If your line is below a dashed line, all chunks fit in that LLM's context</p>
            </div>
"""

            # Close the similarity method section div
            html_content += """
        </div>
"""

        # Close the main RAG section div (already handled above)
        pass

    html_content += """
</body>
</html>"""

    # Write report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"Unified report saved to: {report_path}")
    return str(report_path)


if __name__ == "__main__":
    # Use absolute path relative to project root
    OUTPUT_DIR = PROJECT_ROOT / "benchmark-results"
    logger.info(f"Starting LLM benchmark report generation with timestamp: {TIMESTAMP}")
    try:
        # Use unified report generation (auto-detects LLM/RAG/Combined)
        report_file = create_unified_html_report(OUTPUT_DIR, TIMESTAMP)
        logger.info(f"Report generation complete: {report_file}")
        print(f"Report generated successfully: {report_file}")
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)
        print(f"Error generating report: {str(e)}")
        sys.exit(1)