"""
Configuration file for Amazon Bedrock Model Profiler

Configure your AWS profile and other settings here.
"""

# AWS Configuration
# AWS profile is now selected through the UI

# You can also set these to None to use default credentials
# AWS_ACCESS_KEY_ID = None
# AWS_SECRET_ACCESS_KEY = None

# Default AWS region
AWS_DEFAULT_REGION = "us-east-1"

# Application Configuration
STREAMLIT_PORT = 8501
LOG_LEVEL = "INFO"

# Data Configuration
DATA_DIR = "data"
MODEL_DATA_FILE = "bedrock_models.json"

# Virtual Environment Configuration
VENV_DIR = "venv"
PYTHON_REQUIREMENTS = "requirements.txt"

# Data Collection Configuration
# Parallel processing is now automatically optimized per collector:
#   - model-collector Phase 1: 2 workers (dual-region)
#   - model-collector Phase 5: 10 workers (~20 regions for quotas)
#   - pricing-collector: 3 workers (one per service code)
# These values are optimized and don't need manual configuration.
ENABLE_PARALLEL_COLLECTION = True  # Set to False to disable all parallel processing


# Shared Configuration Class
# This contains values that should be shared between the main app and collectors
class SharedConfig:
    """Shared configuration values accessible to all components"""

    # AWS Configuration
    AWS_DEFAULT_REGION = AWS_DEFAULT_REGION

    # Collection Configuration
    # Note: Individual collectors use optimized worker counts internally:
    #   - model-collector Phase 1: 2 workers (dual-region collection)
    #   - model-collector Phase 5: 10 workers (service quotas from ~20 regions)
    #   - pricing-collector: 3 workers (one per service code)
    ENABLE_PARALLEL_COLLECTION = ENABLE_PARALLEL_COLLECTION

    # Data Configuration
    DATA_DIR = DATA_DIR
    MODEL_DATA_FILE = MODEL_DATA_FILE