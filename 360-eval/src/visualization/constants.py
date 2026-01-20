"""
Configuration constants for visualization and reporting.
"""

import os
import logging
from pathlib import Path
from datetime import datetime

# Timestamp for current session
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
PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Setup logger
logger = logging.getLogger(__name__)
log_dir = PROJECT_ROOT / "logs"
os.makedirs(log_dir, exist_ok=True)

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
