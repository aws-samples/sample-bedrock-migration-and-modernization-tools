"""
Chart generation functions for benchmark visualizations.
"""

import logging
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from collections import Counter
import pytz
from datetime import datetime

from .constants import (
    GRID_OPACITY, PERCENTILES, NORMAL_DISTRIBUTION_RANGE_MULTIPLIER,
    NORMAL_DISTRIBUTION_POINTS, MIN_RECORDS_FOR_HISTOGRAM, MIN_RECORDS_FOR_ANALYSIS,
    EPSILON_DIVISION
)
from .analysis import identify_unique_task_configs
from .metrics_calculation import calculate_metrics_by_model_task_temperature

# Import from html_report (to avoid circular import, we'll handle this differently)
# These functions should be called from html_report, not from chart_generators

logger = logging.getLogger(__name__)

def create_placeholder_chart(message="No data available for accuracy evaluation"):
    """Create a placeholder chart for latency-only evaluations."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="#999999"),
        align="center"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#2d2d2d",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig
def create_normal_distribution_histogram(df,
                                         key='time_to_first_byte',
                                         label='Time to First Token (seconds)'):
    """
    Creates overlapping histogram plots with normal distribution curves for time_to_first_byte by model.
    Only creates the plot if there are more than 2000 records available.
    X-axis is capped at 99th percentile to prevent outliers from compressing the visualization.

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

    # Determine if this is an accuracy metric (higher is better) or latency metric (lower is better)
    is_accuracy_metric = 'accuracy' in label.lower() or 'score' in label.lower()

    # Calculate percentiles across all data for x-axis limit
    p99 = df_clean[key].quantile(0.99)
    p95 = df_clean[key].quantile(0.95)
    p90 = df_clean[key].quantile(0.90)
    p50 = df_clean[key].quantile(0.50)
    p10 = df_clean[key].quantile(0.10)
    p5 = df_clean[key].quantile(0.05)
    p1 = df_clean[key].quantile(0.01)

    # For accuracy metrics, outliers are low values (poor performance)
    # For latency metrics, outliers are high values (slow performance)
    if is_accuracy_metric:
        outliers = df_clean[df_clean[key] < p1]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df_clean)) * 100
        x_min = p1
        x_max = df_clean[key].max()
        outlier_threshold = p1
        outlier_direction = "below"
    else:
        outliers = df_clean[df_clean[key] > p99]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df_clean)) * 100
        x_min = df_clean[key].min()
        x_max = p99
        outlier_threshold = p99
        outlier_direction = "beyond"

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

        # Use percent normalization for all metrics - much more interpretable
        # Include mean and std in legend for easy reference
        fig.add_trace(go.Histogram(
            x=model_data,
            name=f'{model} (n={len(model_data)}, Average={mean:.3f}, Standard Deviation={std:.3f})',
            opacity=0.6,
            marker_color=colors[i % len(colors)],
            histnorm='percent',  # Y-axis shows percentage of observations
            nbinsx=200,  # Increased from 100 for even finer granularity
            showlegend=True
        ))

    # Add percentile lines (different for accuracy vs latency metrics)
    if is_accuracy_metric:
        percentile_lines = [
            (p1, 'p1', 'solid'),
            (p5, 'p5', 'dashdot'),
            (p10, 'p10', 'dash'),
            (p50, 'p50 (Median)', 'dot')
        ]
    else:
        percentile_lines = [
            (p50, 'p50 (Median)', 'dot'),
            (p90, 'p90', 'dash'),
            (p95, 'p95', 'dashdot'),
            (p99, 'p99', 'solid')
        ]

    for percentile_val, percentile_label, line_style in percentile_lines:
        fig.add_vline(
            x=percentile_val,
            line_dash=line_style,
            line_color='rgba(255, 255, 255, 0.4)',
            line_width=1,
            annotation_text=f"{percentile_label}<br>{percentile_val:.3f}",
            annotation_position="top" if is_accuracy_metric else "top",
            annotation=dict(
                font=dict(size=10, color='rgba(255, 255, 255, 0.7)'),
                showarrow=False
            )
        )

    # Add annotation box for outlier information
    if is_accuracy_metric:
        annotation_text = (
            f"<b>Showing 99% of data</b><br>"
            f"Range: {p1:.3f} - 1.0<br>"
            f"Outliers: {outlier_count} ({outlier_pct:.1f}%)<br>"
            f"{outlier_direction} {p1:.3f}"
        )
        subtitle_text = f'{label} Distribution by Model<br><sub>Percentage Distribution (Capped at 1st Percentile)</sub>'
        x_range_min = p1 * 0.98
        x_range_max = df_clean[key].max() * 1.02
    else:
        annotation_text = (
            f"<b>Showing 99% of data</b><br>"
            f"Range: 0 - {p99:.3f}s<br>"
            f"Outliers: {outlier_count} ({outlier_pct:.1f}%)<br>"
            f"{outlier_direction} {p99:.3f}s"
        )
        subtitle_text = f'{label} Distribution by Model<br><sub>Percentage Distribution (Capped at 99th Percentile)</sub>'
        x_range_min = 0
        x_range_max = p99 * 1.02

    # All metrics now use percentage normalization for clearer interpretation
    y_axis_title = 'Percentage of Observations (%)'

    fig.add_annotation(
        text=annotation_text,
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.98,
        xanchor="right",
        yanchor="top",
        showarrow=False,
        bgcolor="rgba(50, 50, 50, 0.8)",
        bordercolor="rgba(255, 255, 255, 0.3)",
        borderwidth=1,
        borderpad=8,
        font=dict(size=11, color='#e0e0e0'),
        align='left'
    )

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#2d2d2d",
        title={
            'text': subtitle_text,
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=label,
        yaxis_title=y_axis_title,
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

    # Update x and y axes - cap x-axis appropriately based on metric type
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=f'rgba(128,128,128,{GRID_OPACITY})',
        range=[x_range_min, x_range_max]  # Add padding for better visibility
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=f'rgba(128,128,128,{GRID_OPACITY})')

    return fig


def create_outlier_boxplot(df, key='time_to_first_byte', label='Time to First Token (seconds)'):
    """
    Creates a percentile-based distribution chart showing granular breakdown of latency buckets.
    Uses horizontal bars to show percentile ranges for better visual clarity.

    Args:
        df: DataFrame containing the benchmark data
        key: data column to analyze
        label: label for the plot
    Returns:
        Plotly figure or None if insufficient data
    """
    min_vals = MIN_RECORDS_FOR_ANALYSIS
    # Check if we have enough data
    value_counts = df['model_name'].value_counts()
    frequent_values = value_counts[value_counts > min_vals].index
    df_match = df[df['model_name'].isin(frequent_values)]

    if df_match.empty:
        logger.info(f"Insufficient data for {label} Outlier Analysis: {len(df)} records")
        return None

    # Filter out any null values
    df_clean = df_match[df_match[key].notna()].copy()

    if df_clean.empty:
        return None

    logger.info(f"Creating {label} Percentile Distribution Chart with {len(df_clean)} records")

    # Determine if this is an accuracy metric (higher is better) or latency metric (lower is better)
    is_accuracy_metric = 'accuracy' in label.lower() or 'score' in label.lower()

    # Calculate global percentiles for reference
    p99 = df_clean[key].quantile(0.99)
    p1 = df_clean[key].quantile(0.01)

    # Create figure
    fig = go.Figure()

    # Get unique models and assign colors
    unique_models = df_clean['model_name'].unique()
    colors = px.colors.qualitative.Set1[:len(unique_models)]

    # Define percentile buckets (more granular)
    percentiles = [0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    percentile_labels = ['0-25%', '25-50%', '50-75%', '75-90%', '90-95%', '95-99%', '99-100%']

    # Color scheme for percentile ranges
    # For accuracy: green (low/bad) to red (high/good) - REVERSED
    # For latency: green (low/fast) to red (high/slow) - NORMAL
    if is_accuracy_metric:
        # Reverse colors for accuracy - worst performance is at bottom percentiles
        bucket_colors = [
            'rgba(192, 57, 43, 0.7)',    # Dark red - worst scores (0-25%)
            'rgba(231, 76, 60, 0.7)',    # Red - poor scores
            'rgba(230, 126, 34, 0.7)',   # Dark orange - below average
            'rgba(243, 156, 18, 0.7)',   # Orange - average
            'rgba(241, 196, 15, 0.7)',   # Yellow - above average
            'rgba(82, 214, 138, 0.7)',   # Light green - good scores
            'rgba(46, 204, 113, 0.7)'    # Green - best scores (99-100%)
        ]
    else:
        # Normal colors for latency - worst performance is at top percentiles
        bucket_colors = [
            'rgba(46, 204, 113, 0.7)',   # Green - fastest quartile
            'rgba(82, 214, 138, 0.7)',   # Light green
            'rgba(241, 196, 15, 0.7)',   # Yellow - median
            'rgba(243, 156, 18, 0.7)',   # Orange
            'rgba(230, 126, 34, 0.7)',   # Dark orange
            'rgba(231, 76, 60, 0.7)',    # Red - slow
            'rgba(192, 57, 43, 0.7)'     # Dark red - outliers
        ]

    y_offset = 0
    model_positions = {}

    # Create stacked horizontal bars for each model
    for i, model in enumerate(unique_models):
        model_data = df_clean[df_clean['model_name'] == model][key]

        if len(model_data) < 10:
            continue

        model_positions[model] = y_offset

        # Calculate percentile values for this model
        percentile_values = [model_data.quantile(p) for p in percentiles]

        # Count outliers for this model
        outlier_count = len(model_data[model_data > p99])
        outlier_pct = (outlier_count / len(model_data)) * 100 if len(model_data) > 0 else 0

        # Create bars for each percentile bucket
        for j in range(len(percentiles) - 1):
            start_val = percentile_values[j]
            end_val = percentile_values[j + 1]
            width = end_val - start_val

            # Calculate percentage of data in this bucket
            bucket_data = model_data[(model_data >= start_val) & (model_data < end_val)]
            bucket_pct = (len(bucket_data) / len(model_data)) * 100

            fig.add_trace(go.Bar(
                x=[width],
                y=[model],
                base=[start_val],
                orientation='h',
                name=percentile_labels[j] if i == 0 else '',  # Only show legend for first model
                marker_color=bucket_colors[j],
                hovertemplate=f'<b>{model}</b><br>' +
                             f'{percentile_labels[j]}<br>' +
                             f'Range: {start_val:.3f} - {end_val:.3f}<br>' +
                             f'Width: {width:.3f}<br>' +
                             f'Data in bucket: {bucket_pct:.1f}%<br>' +
                             '<extra></extra>',
                showlegend=(i == 0),  # Only show in legend once
                legendgroup=percentile_labels[j]
            ))

        y_offset += 1

    # Add vertical line at appropriate percentile threshold
    if is_accuracy_metric:
        threshold_val = p1
        threshold_label = "p1"
        total_outliers = len(df_clean[df_clean[key] < p1])
        outlier_desc = f"Outliers (<p1): {total_outliers}"
        threshold_text = f"{threshold_label}: {threshold_val:.3f}<br>(Low Outlier Threshold)"
    else:
        threshold_val = p99
        threshold_label = "p99"
        total_outliers = len(df_clean[df_clean[key] > p99])
        outlier_desc = f"Outliers (>p99): {total_outliers}"
        threshold_text = f"{threshold_label}: {threshold_val:.3f}<br>(High Outlier Threshold)"

    fig.add_vline(
        x=threshold_val,
        line_dash='solid',
        line_color='rgba(255, 100, 100, 0.8)',
        line_width=2,
        annotation_text=threshold_text,
        annotation_position="top right" if not is_accuracy_metric else "top left",
        annotation=dict(
            font=dict(size=11, color='rgba(255, 100, 100, 0.9)'),
            showarrow=False,
            bgcolor="rgba(50, 50, 50, 0.8)",
            bordercolor="rgba(255, 100, 100, 0.5)",
            borderwidth=1,
            borderpad=4
        )
    )

    # Add summary statistics annotation
    total_records = len(df_clean)
    outlier_pct_total = (total_outliers / total_records) * 100 if total_records > 0 else 0

    annotation_text = (
        f"<b>Distribution Summary</b><br>"
        f"Total Records: {total_records}<br>"
        f"{outlier_desc} ({outlier_pct_total:.1f}%)<br>"
        f"{threshold_label} Threshold: {threshold_val:.3f}<br><br>"
        f"<i>Each bar shows percentile ranges</i>"
    )

    fig.add_annotation(
        text=annotation_text,
        xref="paper",
        yref="paper",
        x=0.98,  # Move to right side
        y=0.02,  # Position at bottom of chart area
        xanchor="right",
        yanchor="bottom",
        showarrow=False,
        bgcolor="rgba(50, 50, 50, 0.8)",
        bordercolor="rgba(255, 255, 255, 0.3)",
        borderwidth=1,
        borderpad=8,
        font=dict(size=11, color='#e0e0e0'),
        align='left'
    )

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#2d2d2d",
        title={
            'text': f'{label} - Percentile Distribution Analysis<br><sub>Stacked Percentile Ranges Showing Distribution Spread</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=label,
        yaxis_title='Model',
        barmode='stack',
        height=max(400, len(unique_models) * 60),  # Dynamic height based on number of models
        margin=dict(l=300, r=150, t=100, b=150),  # Extra margin for model names and bottom annotation
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            title="Percentile Range"
        )
    )

    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=f'rgba(128,128,128,{GRID_OPACITY})'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=f'rgba(128,128,128,{GRID_OPACITY})'
    )

    return fig


def create_visualizations(df, model_task_metrics, latency_metrics, cost_metrics, has_latency_only=False):
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
    # Only skip if there's NO success_rate data at all (filter out NaN rows first)
    has_success_rate_data = 'success_rate' in model_task_metrics.columns and model_task_metrics['success_rate'].notna().any()

    if not has_success_rate_data:
        # Use placeholder only if there's absolutely no success_rate data
        heatmap_fig = create_placeholder_chart("Success rate heatmap not available in latency-only mode")
    else:
        # Filter to only rows with success_rate data (exclude latency-only tasks in mixed reports)
        metrics_with_success = model_task_metrics[model_task_metrics['success_rate'].notna()].copy()

        # Pivot to create model vs task matrix (use task_display_name for proper disambiguation)
        pivot_success = pd.pivot_table(
            metrics_with_success,
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

    # 6. Model-Task Bubble Chart
    if not has_success_rate_data:
        # For latency-only mode, create a simplified bubble chart with only latency vs cost
        model_task_metrics_round = model_task_metrics
        average_cost_round = model_task_metrics_round.round({'avg_otps': 2, 'value_ratio': 2})
        bubble_fig = px.scatter(
            average_cost_round,
            template="plotly_dark",
            x='avg_latency',
            y='avg_cost',
            size='avg_otps',
            color='avg_otps',
            facet_col='task_display_name',
            facet_col_wrap=3,
            hover_data=['model_name'],
            labels={
                'avg_latency': 'Latency (Secs)',
                'avg_cost': 'Cost (USD)',
                'avg_otps': 'Tokens/sec',
                'task_display_name': 'Task Type'
            },
            title='Model Performance by Task Type (Latency-Only Mode)',
            color_continuous_scale='Viridis',
            opacity=0.85
        )
        bubble_fig.update_traces(
            marker=dict(
                line=dict(width=1, color="rgba(255, 255, 255, 0.3)")
            )
        )
        bubble_fig.update_layout(
            coloraxis_colorbar=dict(
                title_font_color="#ffffff",
                tickfont_color="#ffffff",
            ),
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#2d2d2d",
            font=dict(color="#e0e0e0"),
            title_font=dict(color="#90caf9", size=18)
        )
        bubble_fig.for_each_annotation(lambda a: a.update(font=dict(color="#90caf9", size=12)))
    else:
        # Full 360 evaluation mode with success_rate - filter to only rows with success_rate
        metrics_with_success = model_task_metrics[model_task_metrics['success_rate'].notna()].copy()
        model_task_metrics_round = metrics_with_success
        average_cost_round = model_task_metrics_round.round({'avg_otps': 2, 'value_ratio': 2})
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
    if has_latency_only:
        visualizations['error_analysis'] = '<div id="not-found">Error analysis not available in latency-only mode</div>'
    elif 'judge_explanation' in df.columns:
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

    radar_charts = {}

    # Only skip radar charts if there's NO success_rate data at all
    if has_success_rate_data:
        # Filter dataframe to only rows with judge scores (exclude latency-only in mixed reports)
        df_with_scores = df[df['judge_success'] != 'N/A'].copy()

        # Identify unique task configurations (handles edge case of same task with different metrics)
        unique_task_configs = identify_unique_task_configs(df_with_scores)

        # Define color palette for models
        color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2

        # Create one chart per unique task configuration
        for unique_task_name, (original_task, metric_sig, indices) in unique_task_configs.items():
            # Filter data for this task configuration (use df_with_scores to exclude latency-only)
            task_data = df_with_scores.loc[indices].copy()

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
                # Create subplot with 2x2 or 1x3 grid depending on mode
                if has_latency_only:
                    # Latency-only mode: skip success rate, use 1x3 grid
                    fig = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=(
                            "Latency (Secs) by Temperature",
                            'Cost per Response (USD) by Temperature<br><span style="font-size: 12px;">Using μ (Micro) Symbol for Small Numbers</span>',
                            "Tokens per Second by Temperature"
                        )
                    )
                else:
                    # Full 360 mode: include success rate, use 2x2 grid
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

                # Success Rate subplot (row=1, col=1) - only in 360 mode
                if not has_latency_only:
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

                # Latency subplot - position depends on mode
                latency_row, latency_col = (1, 1) if has_latency_only else (1, 2)
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
                            showlegend=True if has_latency_only else False,
                            legendgroup=f'temp_{temp}'
                        ),
                        row=latency_row, col=latency_col
                    )

                # Cost subplot - position depends on mode
                cost_row, cost_col = (1, 2) if has_latency_only else (2, 1)
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
                        row=cost_row, col=cost_col
                    )

                # Tokens per Second subplot - position depends on mode
                otps_row, otps_col = (1, 3) if has_latency_only else (2, 2)
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
                        row=otps_row, col=otps_col
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
                # Create subplot with 2x2 or 1x3 grid depending on mode
                if has_latency_only:
                    # Latency-only mode: skip success rate, use 1x3 grid
                    fig = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=("Latency (Secs)", 'Cost per Response (USD)<br><span style="font-size: 12px;">Using μ (Micro) Symbol for Small Numbers</span>', "Tokens per Second")
                    )
                else:
                    # Full 360 mode: include success rate, use 2x2 grid
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=("Success Rate", "Latency (Secs)", 'Cost per Response (USD)<br><span style="font-size: 12px;">Using μ (Micro) Symbol for Small Numbers</span>', "Tokens per Second")
                    )

                # Sort data for each subplot (using method chaining for efficiency)
                by_latency = task_data.sort_values('avg_latency')
                by_cost = task_data.sort_values('avg_cost')
                by_otps = task_data.sort_values('avg_otps', ascending=False)

                # Add success rate trace only in 360 mode
                if not has_latency_only:
                    by_success = task_data.sort_values('success_rate', ascending=False)
                    fig.add_trace(
                        go.Bar(x=by_success['model_name'], y=by_success['success_rate'], marker_color='green'),
                        row=1, col=1
                    )

                # Add other traces with position based on mode
                latency_row, latency_col = (1, 1) if has_latency_only else (1, 2)
                fig.add_trace(
                    go.Bar(x=by_latency['model_name'], y=by_latency['avg_latency'], marker_color='orange'),
                    row=latency_row, col=latency_col
                )

                cost_row, cost_col = (1, 2) if has_latency_only else (2, 1)
                fig.add_trace(
                    go.Bar(x=by_cost['model_name'], y=by_cost['avg_cost'], marker_color='red'),
                    row=cost_row, col=cost_col
                )

                otps_row, otps_col = (1, 3) if has_latency_only else (2, 2)
                fig.add_trace(
                    go.Bar(x=by_otps['model_name'], y=by_otps['avg_otps'], marker_color='blue'),
                    row=otps_row, col=otps_col
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

    # Note: integrated_analysis_tables and regional_performance are created in html_report.py
    # to avoid circular imports

    # Create TTFT histograms and boxplots per task
    ttfb_histograms_by_task = {}
    ttfb_boxplots_by_task = {}

    # Loop through unique task_display_name for consistency with other charts
    for task_display in model_task_metrics['task_display_name'].unique():
        # Get original task_types for filtering the main df
        task_data = model_task_metrics[model_task_metrics['task_display_name'] == task_display]
        task_types = task_data['task_types'].iloc[0]
        config_sig = task_data['config_signature'].iloc[0] if 'config_signature' in task_data.columns else None

        # Filter df by task_types and config_signature
        if config_sig:
            task_df = df[(df['task_types'] == task_types) & (df['config_signature'] == config_sig)]
        else:
            task_df = df[df['task_types'] == task_types]

        # Create TTFT histogram for this task
        ttfb_histogram = create_normal_distribution_histogram(task_df, label=f'Time to First Token (seconds) - {task_display}')
        if ttfb_histogram is not None:
            ttfb_histograms_by_task[task_display] = ttfb_histogram

        # Create TTFT boxplot for this task
        ttfb_boxplot = create_outlier_boxplot(task_df, label=f'Time to First Token (seconds) - {task_display}')
        if ttfb_boxplot is not None:
            ttfb_boxplots_by_task[task_display] = ttfb_boxplot

    visualizations['ttfb_histograms_by_task'] = ttfb_histograms_by_task
    visualizations['ttfb_boxplots_by_task'] = ttfb_boxplots_by_task

    # Create accuracy histograms and boxplots per task
    accuracy_histograms_by_task = {}
    accuracy_boxplots_by_task = {}

    for task_display in model_task_metrics['task_display_name'].unique():
        task_data = model_task_metrics[model_task_metrics['task_display_name'] == task_display]
        task_types = task_data['task_types'].iloc[0]
        config_sig = task_data['config_signature'].iloc[0] if 'config_signature' in task_data.columns else None

        if config_sig:
            task_df = df[(df['task_types'] == task_types) & (df['config_signature'] == config_sig)]
        else:
            task_df = df[df['task_types'] == task_types]

        accuracy_histogram = create_normal_distribution_histogram(task_df, key='mean_scores', label=f'Accuracy Distribution - {task_display}')
        if accuracy_histogram is not None:
            accuracy_histograms_by_task[task_display] = accuracy_histogram

        accuracy_boxplot = create_outlier_boxplot(task_df, key='mean_scores', label=f'Average Accuracy - {task_display}')
        if accuracy_boxplot is not None:
            accuracy_boxplots_by_task[task_display] = accuracy_boxplot

    visualizations['accuracy_histograms_by_task'] = accuracy_histograms_by_task
    visualizations['accuracy_boxplots_by_task'] = accuracy_boxplots_by_task

    # Create total tokens histograms and boxplots per task
    total_tokens_histograms_by_task = {}
    total_tokens_boxplots_by_task = {}

    for task_display in model_task_metrics['task_display_name'].unique():
        task_data = model_task_metrics[model_task_metrics['task_display_name'] == task_display]
        task_types = task_data['task_types'].iloc[0]
        config_sig = task_data['config_signature'].iloc[0] if 'config_signature' in task_data.columns else None

        if config_sig:
            task_df = df[(df['task_types'] == task_types) & (df['config_signature'] == config_sig)]
        else:
            task_df = df[df['task_types'] == task_types]

        # Create Output Tokens histogram for this task
        total_tokens_histogram = create_normal_distribution_histogram(task_df, key='output_tokens', label=f'Output Tokens Per Response - {task_display}')
        if total_tokens_histogram is not None:
            total_tokens_histograms_by_task[task_display] = total_tokens_histogram

        # Create Output Tokens boxplot for this task
        total_tokens_boxplot = create_outlier_boxplot(task_df, key='output_tokens', label=f'Output Tokens Per Response - {task_display}')
        if total_tokens_boxplot is not None:
            total_tokens_boxplots_by_task[task_display] = total_tokens_boxplot

    visualizations['total_tokens_histograms_by_task'] = total_tokens_histograms_by_task
    visualizations['total_tokens_boxplots_by_task'] = total_tokens_boxplots_by_task

    return visualizations


