"""
Main entry point for LLM benchmark visualization and report generation.

This module has been refactored into a modular structure:
- visualization.constants: Configuration and constants
- visualization.data_loading: Data loading and preprocessing
- visualization.metrics_calculation: Metrics computation
- visualization.chart_generators: Chart and visualization generation
- visualization.analysis: Analysis and findings generation
- visualization.html_report: HTML report creation
"""

import logging
import sys
from pathlib import Path

# Import from visualization module
from visualization import create_html_report
from visualization.constants import PROJECT_ROOT, TIMESTAMP

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)
logger.info(f"Starting visualization with project root: {PROJECT_ROOT}")


if __name__ == "__main__":
    # Use absolute path relative to project root
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    logger.info(f"Starting LLM benchmark report generation with timestamp: {TIMESTAMP}")
    try:
        report_file = create_html_report(OUTPUT_DIR, TIMESTAMP)
        logger.info(f"Report generation complete: {report_file}")
        print(f"Report generated successfully: {report_file}")
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)
        print(f"Error generating report: {str(e)}")
        sys.exit(1)
