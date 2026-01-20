"""
Visualization module for LLM benchmark results.

This module provides functionality for:
- Loading and processing benchmark data
- Calculating metrics
- Generating charts and visualizations
- Creating HTML reports
"""

from .data_loading import load_data, extract_model_name, parse_json_string, get_evaluation_config_signature
from .metrics_calculation import (
    calculate_metrics_by_model_task,
    calculate_metrics_by_model_task_temperature,
    calculate_latency_metrics,
    calculate_cost_metrics
)
from .chart_generators import create_visualizations, create_placeholder_chart
from .analysis import (
    generate_task_findings,
    generate_task_recommendations,
    generate_histogram_findings,
    identify_unique_task_configs
)
from .html_report import create_html_report
from .constants import *

__all__ = [
    # Data loading
    'load_data',
    'extract_model_name',
    'parse_json_string',
    'get_evaluation_config_signature',

    # Metrics
    'calculate_metrics_by_model_task',
    'calculate_metrics_by_model_task_temperature',
    'calculate_latency_metrics',
    'calculate_cost_metrics',

    # Charts
    'create_visualizations',
    'create_placeholder_chart',

    # Analysis
    'generate_task_findings',
    'generate_task_recommendations',
    'generate_histogram_findings',
    'identify_unique_task_configs',

    # HTML
    'create_html_report',
]
