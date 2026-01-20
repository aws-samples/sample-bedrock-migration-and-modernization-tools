"""
HTML report generation for benchmark results.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import pytz
from jinja2 import Template
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .constants import (
    PROJECT_ROOT, TIMESTAMP, HTML_TEMPLATE, COMPOSITE_SCORE_WEIGHTS,
    PERFORMANCE_THRESHOLDS, INFERENCE_MAX_TOKENS, INFERENCE_TEMPERATURE,
    INFERENCE_REGION, COEFFICIENT_VARIATION_THRESHOLD
)
from .data_loading import load_data
from .metrics_calculation import (
    calculate_metrics_by_model_task, calculate_metrics_by_model_task_temperature,
    calculate_latency_metrics, calculate_cost_metrics
)
from .chart_generators import create_visualizations
from .analysis import (
    generate_task_findings, generate_task_recommendations,
    identify_unique_task_configs, generate_histogram_findings
)

# Import utils from parent directory
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import run_inference, report_summary_template, convert_scientific_to_decimal

logger = logging.getLogger(__name__)

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

    # Check if this is latency-only mode
    has_success_rate = 'success_rate' in table_data.columns

    thresholds['avg_latency'] = build_task_latency_thresholds(table_data[['model_name', 'task_types', 'avg_latency']].to_dict(orient='records'))

    # Format Model Name
    table_data['model_name'] = table_data['model_name'].apply(lambda x: x.split('/')[-1])

    # Format metrics for display
    if has_success_rate:
        table_data['success_rate_fmt'] = table_data['success_rate'].apply(lambda x: f"{x:.1%}")
    table_data['avg_latency_fmt'] = table_data['avg_latency'].apply(lambda x: f"{x:.2f}s")
    table_data['avg_cost_fmt'] = table_data['avg_cost'].apply(lambda x: f"${x:.4f}")
    table_data['avg_otps_fmt'] = table_data['avg_otps'].apply(lambda x: f"{x:.1f}")

    # Calculate composite score (higher is better)
    # Normalize metrics to 0-1 range and combine them
    max_latency = table_data['avg_latency'].max() or 1
    max_cost = table_data['avg_cost'].max() or 1

    if has_success_rate:
        table_data['composite_score'] = (
                table_data['success_rate'] +
                (1 - (table_data['avg_latency'] / max_latency)) * COMPOSITE_SCORE_WEIGHTS['latency'] +
                (1 - (table_data['avg_cost'] / max_cost)) * COMPOSITE_SCORE_WEIGHTS['cost']
        )
    else:
        # Latency-only mode: composite score based only on latency and cost
        table_data['composite_score'] = (
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
        # Prepare headers and values based on mode
        if has_success_rate:
            header_values = ['Model', 'Task Type', 'Success Rate', 'Latency', 'Cost', 'Tokens/sec', 'Score']
            cell_values = [
                task_data['model_name'],
                task_data['task_display_name'],
                task_data['success_rate_fmt'],
                task_data['avg_latency_fmt'],
                task_data['avg_cost_fmt'],
                task_data['avg_otps_fmt'],
                task_data['composite_score'].apply(lambda x: f"{x:.2f}")
            ]
            fill_colors = [
                ['#3a3a3a'] * len(task_data),  # Model column (dark gray)
                ['#3a3a3a'] * len(task_data),  # Task column (dark gray)
                [get_color(sr, 'success_rate') for sr in task_data['success_rate']],
                [get_color(lt, 'avg_latency') for lt in task_data[['avg_latency','task_types']].to_dict(orient='records')],
                [get_color(cost, 'avg_cost') for cost in task_data['avg_cost']],
                [get_color(tps, 'avg_otps') for tps in task_data['avg_otps']],
                [colors['good'] if score >= task_data['composite_score'].quantile(0.67) else
                 colors['medium'] if score >= task_data['composite_score'].quantile(0.33) else
                 colors['poor'] for score in task_data['composite_score']]
            ]
            font_colors = [
                ['white'] * len(task_data),  # Model column
                ['white'] * len(task_data),  # Task column
                ['black'] * len(task_data),  # Success rate
                ['black'] * len(task_data),  # Latency
                ['black'] * len(task_data),  # Cost
                ['black'] * len(task_data),  # Tokens/sec
                ['black'] * len(task_data),  # Score
            ]
        else:
            # Latency-only mode: skip success rate column
            header_values = ['Model', 'Task Type', 'Latency', 'Cost', 'Tokens/sec', 'Score']
            cell_values = [
                task_data['model_name'],
                task_data['task_display_name'],
                task_data['avg_latency_fmt'],
                task_data['avg_cost_fmt'],
                task_data['avg_otps_fmt'],
                task_data['composite_score'].apply(lambda x: f"{x:.2f}")
            ]
            fill_colors = [
                ['#3a3a3a'] * len(task_data),  # Model column (dark gray)
                ['#3a3a3a'] * len(task_data),  # Task column (dark gray)
                [get_color(lt, 'avg_latency') for lt in task_data[['avg_latency','task_types']].to_dict(orient='records')],
                [get_color(cost, 'avg_cost') for cost in task_data['avg_cost']],
                [get_color(tps, 'avg_otps') for tps in task_data['avg_otps']],
                [colors['good'] if score >= task_data['composite_score'].quantile(0.67) else
                 colors['medium'] if score >= task_data['composite_score'].quantile(0.33) else
                 colors['poor'] for score in task_data['composite_score']]
            ]
            font_colors = [
                ['white'] * len(task_data),  # Model column
                ['white'] * len(task_data),  # Task column
                ['black'] * len(task_data),  # Latency
                ['black'] * len(task_data),  # Cost
                ['black'] * len(task_data),  # Tokens/sec
                ['black'] * len(task_data),  # Score
            ]

        fig.add_trace(go.Table(
            header=dict(
                values=header_values,
                font=dict(size=12, color='white'),
                fill_color='#2E5A88',
                align='left'
            ),
            cells=dict(
                values=cell_values,
                align='left',
                font=dict(size=11),
                fill_color=fill_colors,
                font_color=font_colors
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


    # Check evaluation types in the dataset
    has_latency = 'eval_type' in df.columns and (df['eval_type'] == 'latency').any()
    has_360 = 'eval_type' in df.columns and (df['eval_type'] == '360').any()
    has_mixed = has_latency and has_360
    has_only_latency = has_latency and not has_360

    if has_mixed:
        logger.info("Detected mixed evaluation types: both latency-only and 360 evaluations present")
    elif has_only_latency:
        logger.info("Detected latency-only evaluation")
    else:
        logger.info("Detected 360 evaluation (with judge scoring)")

    # Calculate metrics
    logger.info("Calculating model-task metrics...")
    model_task_metrics = calculate_metrics_by_model_task(df)

    logger.info("Calculating latency metrics...")
    latency_metrics = calculate_latency_metrics(df)

    logger.info("Calculating cost metrics...")
    cost_metrics = calculate_cost_metrics(df)



    # Create visualizations
    logger.info("Creating visualizations...")
    visualizations = create_visualizations(df, model_task_metrics, latency_metrics, cost_metrics, has_only_latency)

    # Add integrated analysis and regional performance (handled here to avoid circular imports)
    visualizations['integrated_analysis_tables'], analysis_df = create_integrated_analysis_table(model_task_metrics)
    visualizations['regional_performance'] = create_regional_performance_analysis(df)

    # Generate findings and recommendations
    logger.info("Generating task findings...")
    task_findings = generate_task_findings(df, model_task_metrics, has_only_latency)

    logger.info("Generating recommendations...")
    task_recommendations = generate_task_recommendations(model_task_metrics, has_only_latency)
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

    logger.info("Generating Output Tokens histogram findings...")
    total_tokens_findings = generate_histogram_findings(df, key='output_tokens', label="Output Tokens Per Response")

    whole_number_cost_metrics = convert_scientific_to_decimal(cost_metrics)
    cost_analysis = '# Cost Analysis across all models on all Task:\n' + '\n'.join([str(i) for i in whole_number_cost_metrics.to_dict(orient='records')])

    recommendations = '# Recommendations:\n* ' + '\n* '.join([str(i) for i in task_recommendations])

    prompt_template = report_summary_template(models=unique_models, evaluations=f'{acc_analysis}\n\n{cost_analysis}\n\n{perf_analysis}\n\n{task_level_analysis}\n\n{recommendations}')  ## Append AND Format all evals ++ rename the columns to help the model
    # Model ID preparation for litellm (/converse addition) is now handled centrally in run_inference()
    inference = run_inference(model_name='bedrock/global.amazon.nova-2-lite-v1:0',
                              prompt_text=prompt_template,
                              stream=False,
                              provider_params={"maxTokens": INFERENCE_MAX_TOKENS,
                                               "temperature": INFERENCE_TEMPERATURE,
                                               "aws_region_name": INFERENCE_REGION},
                              judge_eval=True)['text']
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

        # TTFB histograms and boxplots by task
        ttfb_histograms_by_task={task: chart.to_html(full_html=False)
                                  for task, chart in visualizations.get('ttfb_histograms_by_task', {}).items()},
        ttfb_boxplots_by_task={task: chart.to_html(full_html=False)
                               for task, chart in visualizations.get('ttfb_boxplots_by_task', {}).items()},
        ttfb_findings=time_to_first_token_findings,

        # Accuracy histograms and boxplots by task
        accuracy_histograms_by_task={task: chart.to_html(full_html=False)
                                      for task, chart in visualizations.get('accuracy_histograms_by_task', {}).items()},
        accuracy_boxplots_by_task={task: chart.to_html(full_html=False)
                                    for task, chart in visualizations.get('accuracy_boxplots_by_task', {}).items()},
        accuracy_findings=accuracy_findings,

        # Total tokens histograms and boxplots by task
        total_tokens_histograms_by_task={task: chart.to_html(full_html=False)
                                          for task, chart in visualizations.get('total_tokens_histograms_by_task', {}).items()},
        total_tokens_boxplots_by_task={task: chart.to_html(full_html=False)
                                        for task, chart in visualizations.get('total_tokens_boxplots_by_task', {}).items()},
        total_tokens_findings=total_tokens_findings,

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

