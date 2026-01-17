"""
Analysis and findings generation for benchmark results.
"""

import logging
import pandas as pd
import ast
from collections import Counter
from .constants import (
    MIN_RECORDS_FOR_ANALYSIS, NORMAL_DISTRIBUTION_RANGE_MULTIPLIER,
    NORMAL_DISTRIBUTION_POINTS, COEFFICIENT_VARIATION_THRESHOLD
)
from scipy import stats
import numpy as np

logger = logging.getLogger(__name__)


def extract_judge_scores(json_str):
    """Extract judge scores from JSON string."""
    try:
        if isinstance(json_str, list):
            json_str = json_str[0]
        dict_data = ast.literal_eval(json_str) if isinstance(json_str, str) else json_str
        return dict_data if isinstance(dict_data, dict) else {}
    except:
        return {}

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


def generate_task_findings(df, model_task_metrics, has_latency_only=False):
    """Generate key findings for each task configuration (using task_display_name)."""
    task_findings = {}

    # Loop through unique task_display_name to handle multiple configs
    for task_display in model_task_metrics['task_display_name'].unique():
        task_data = model_task_metrics[model_task_metrics['task_display_name'] == task_display]
        findings = []

        if not task_data.empty:
            # Check if this specific task has success_rate data (for mixed evaluations)
            task_has_success_rate = 'success_rate' in task_data.columns and task_data['success_rate'].notna().any()

            # Skip accuracy-related findings only if this specific task has no success_rate
            if task_has_success_rate and not has_latency_only:
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

            # Skip value ratio only if this task has no success_rate
            if task_has_success_rate and not has_latency_only:
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


def generate_task_recommendations(model_task_metrics, has_latency_only=False):
    """Generate task-specific model recommendations (using task_display_name)."""
    recommendations_360 = []
    recommendations_latency = []

    # Separate tasks with success_rate (360) from those without (latency-only)
    if 'success_rate' in model_task_metrics.columns:
        metrics_360 = model_task_metrics[model_task_metrics['success_rate'].notna()].copy()
        metrics_latency = model_task_metrics[model_task_metrics['success_rate'].isna()].copy()
    else:
        metrics_360 = pd.DataFrame()
        metrics_latency = model_task_metrics.copy()

    # Process 360 evaluation tasks
    for task_display in metrics_360['task_display_name'].unique() if not metrics_360.empty else []:
        task_data = metrics_360[metrics_360['task_display_name'] == task_display]

        if not task_data.empty:
            best_lat = task_data['avg_latency'].min()
            best_speed_model = '<br>'.join(task_data[task_data['avg_latency'] == best_lat]['model_name'].tolist())

            best_suc = task_data['success_rate'].max()
            best_acc_model = '<br>'.join(task_data[task_data['success_rate'] == best_suc]['model_name'].tolist())

            best_value = task_data['value_ratio'].max()
            best_value_model = '<br>'.join(task_data[task_data['value_ratio'] == best_value]['model_name'].tolist())

            recommendations_360.append({
                'task': str(task_display),
                'best_accuracy_model': str(best_acc_model),
                'accuracy': f"{best_suc:.1%}",
                'best_speed_model': str(best_speed_model),
                'speed': f"{best_lat:.2f}s",
                'best_value_model': str(best_value_model),
                'value': f"{best_value:.2f}"
            })

    # Process latency-only tasks
    for task_display in metrics_latency['task_display_name'].unique() if not metrics_latency.empty else []:
        task_data = metrics_latency[metrics_latency['task_display_name'] == task_display]

        if not task_data.empty:
            best_lat = task_data['avg_latency'].min()
            best_speed_model = '<br>'.join(task_data[task_data['avg_latency'] == best_lat]['model_name'].tolist())

            recommendations_latency.append({
                'task': str(task_display),
                'best_accuracy_model': "N/A",
                'accuracy': "N/A",
                'best_speed_model': str(best_speed_model),
                'speed': f"{best_lat:.2f}s",
                'best_value_model': "N/A",
                'value': "N/A"
            })

    # Combine both lists
    all_recommendations = recommendations_360 + recommendations_latency

    return sorted(all_recommendations, key=lambda x: x['task'])


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

