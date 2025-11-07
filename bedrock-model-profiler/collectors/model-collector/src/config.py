"""
Configuration file for Amazon Bedrock Model Collector
Centralizes all configuration variables and settings for comprehensive model data collection
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Import shared configuration from root
root_dir = Path(__file__).parent.parent.parent.parent
root_config_file = root_dir / 'config.py'

# Read and execute root config to get SharedConfig class
SharedConfig = None
if root_config_file.exists():
    # Read the root config file and extract SharedConfig
    import importlib.util
    spec = importlib.util.spec_from_file_location("root_config", str(root_config_file))
    root_config_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(root_config_module)
        if hasattr(root_config_module, 'SharedConfig'):
            SharedConfig = root_config_module.SharedConfig
    except Exception as e:
        # Log config loading failure for debugging, then fallback to defaults
        logging.debug(f"Could not load SharedConfig from root config: {e}. Using fallback configuration.")

# Fallback if SharedConfig not loaded
if SharedConfig is None:
    class SharedConfig:
        AWS_DEFAULT_REGION = 'us-east-1'
        ENABLE_PARALLEL_COLLECTION = True
        DATA_DIR = 'data'
        MODEL_DATA_FILE = 'bedrock_models.json'


class Config:
    """Centralized configuration for the Bedrock Model Collector"""

    # AWS Configuration from shared config
    AWS_PROFILE_NAME = None  # Will be overridden by command line argument
    AWS_REGION = SharedConfig.AWS_DEFAULT_REGION

    # Threading Configuration
    # Parallel processing enabled with optimized worker counts per phase:
    #   - Phase 1: 2 workers (dual-region collection: us-east-1 + us-west-2)
    #   - Phase 5: 10 workers (service quotas from ~20 regions)
    USE_PARALLEL_COLLECTION = SharedConfig.ENABLE_PARALLEL_COLLECTION

    USE_DIRECT_API = True  # Enable direct REST API for faster collection

    # Service Configuration
    BEDROCK_SERVICE_CODE = 'bedrock'  # For service quotas

    # Request Configuration
    REQUEST_TIMEOUT = 30  # seconds
    MAX_RESULTS_PER_PAGE = 100

    # Retry Configuration
    RETRY_TOTAL = 3
    RETRY_BACKOFF_FACTOR = 1
    RETRY_STATUS_FORCELIST = [429, 500, 502, 503, 504]

    # Progress Reporting
    PROGRESS_REPORT_INTERVAL = 25  # Report progress every N models

    # Known Bedrock Regions (fallback)
    KNOWN_BEDROCK_REGIONS = [
        'us-east-1', 'us-west-2', 'us-east-2',
        'eu-west-1', 'eu-west-2', 'eu-west-3', 'eu-central-1', 'eu-north-1',
        'ap-southeast-1', 'ap-southeast-2', 'ap-northeast-1', 'ap-northeast-2', 'ap-south-1',
        'ca-central-1', 'sa-east-1', 'ap-southeast-3'
    ]

    # Web Sources for Enhanced Data
    WEB_STATIC_URLS = {
        'foundation_models_data': 'https://c.d.cdn.console.awsstatic.com/a/v1/FDU25WTVKMYQHNV5VTD3B5RGXLY5RJDGK4TISNITRUFJXTTFYFBA/assets/foundationModelsData-5bb59c86.js'
    }

    # Provider Detection Patterns (for enhanced data matching)
    PROVIDER_PATTERNS = {
        'Anthropic': ['claude', 'anthropic'],
        'Amazon': ['nova', 'titan', 'amazon'],
        'Meta': ['llama', 'meta'],
        'Mistral AI': ['mistral', 'mixtral', 'pixtral'],
        'Qwen': ['qwen'],
        'DeepSeek': ['deepseek'],
        'OpenAI': ['gpt', 'openai'],
        'Stability AI': ['stable', 'stability'],
        'AI21 Labs': ['jamba', 'ai21'],
        'Cohere': ['cohere', 'command', 'embed'],
        'TwelveLabs': ['twelve', 'pegasus', 'marengo'],
        'Writer': ['writer', 'palmyra'],
        'Luma AI': ['luma', 'dream']
    }

    # File and Directory Configuration
    # Output directly to the root data/ directory (avoid duplication)
    @classmethod
    def get_root_data_dir(cls):
        """Get absolute path to root data directory"""
        # From collectors/model-collector/src/config.py -> root/data/
        # Go up 4 levels: src -> model-collector -> collectors -> root
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        return os.path.join(current_dir, SharedConfig.DATA_DIR)

    OUTPUT_DIR = None  # Will be set dynamically to root data/ directory
    LOGS_DIR = 'logs'
    TESTS_DIR = 'tests'
    LOG_FILENAME = 'model_collection.log'
    OUTPUT_FILE_PREFIX = 'bedrock-models'
    FINAL_OUTPUT_FILENAME = SharedConfig.MODEL_DATA_FILE  # Final filename in root data/

    # Pricing Collector Integration
    @classmethod
    def get_pricing_collector_path(cls):
        """Get absolute path to pricing collector output directory"""
        import os
        # Get the directory where this config.py file is located
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Navigate to the sibling pricing collector directory
        return os.path.join(os.path.dirname(current_dir), 'pricing-collector', 'out')

    PRICING_FILE_PATTERN = 'bedrock-pricing-*.json'  # Pattern to find latest pricing file

    # Documentation URL Patterns
    DOCUMENTATION_PATTERNS = {
        'aws_bedrock_base': 'https://docs.aws.amazon.com/bedrock/latest/userguide/',
        'aws_pricing': 'https://aws.amazon.com/bedrock/pricing/',
        'anthropic_claude': 'https://docs.claude.com/en/docs/about-claude/models/overview#model-comparison-table',
        'meta_llama': 'https://www.llama.com/docs/overview/',
        'mistral_docs': 'https://docs.mistral.ai/getting-started/models/models_overview/',
        'ai21_labs': 'https://docs.ai21.com/docs/overview',
        'cohere_docs': 'https://docs.cohere.com/docs/models',
        'stability_ai': 'https://stability.ai/core-models',
        'amazon_titan': 'https://docs.aws.amazon.com/bedrock/latest/userguide/titan-models.html'
    }

    # Metadata
    VERSION = '1.0.0'
    DESCRIPTION = 'Comprehensive Amazon Bedrock Model Database with Enhanced Features'

    @classmethod
    def get_log_file_path(cls) -> str:
        """Get the full path for the log file"""
        return os.path.join(cls.LOGS_DIR, cls.LOG_FILENAME)

    @classmethod
    def get_output_file_path(cls, timestamp: str) -> str:
        """Get the full path for the output file with timestamp"""
        filename = f"{cls.OUTPUT_FILE_PREFIX}-{timestamp}.json"
        return os.path.join(cls.OUTPUT_DIR, filename)

    @classmethod
    def get_user_agent(cls) -> str:
        """Get the user agent string for web requests"""
        return 'Mozilla/5.0 (compatible; bedrock-model-collector)'

    @classmethod
    def get_request_headers(cls) -> Dict[str, str]:
        """Get standard request headers"""
        return {
            'User-Agent': cls.get_user_agent(),
            'Accept': 'application/json'
        }

    @classmethod
    def detect_provider_from_model_id(cls, model_id: str) -> Optional[str]:
        """
        Detect provider from model ID using configured patterns

        Args:
            model_id: The model ID string to analyze

        Returns:
            Provider name if detected, None if no match
        """
        model_id_lower = model_id.lower()

        for provider, patterns in cls.PROVIDER_PATTERNS.items():
            if any(pattern in model_id_lower for pattern in patterns):
                return provider

        return None

    @classmethod
    def generate_documentation_links(cls, model_id: str, provider: str) -> Dict[str, str]:
        """
        Generate documentation links for a model based on provider and model ID

        Args:
            model_id: The model ID
            provider: The model provider

        Returns:
            Dictionary of documentation links
        """
        links = {
            'aws_bedrock_guide': f"{cls.DOCUMENTATION_PATTERNS['aws_bedrock_base']}model-ids-arns.html",
            'pricing_guide': cls.DOCUMENTATION_PATTERNS['aws_pricing']
        }

        # Add provider-specific documentation
        provider_lower = provider.lower()
        if 'anthropic' in provider_lower:
            links['provider_docs'] = cls.DOCUMENTATION_PATTERNS['anthropic_claude']
        elif 'meta' in provider_lower:
            links['provider_docs'] = cls.DOCUMENTATION_PATTERNS['meta_llama']
        elif 'mistral' in provider_lower:
            links['provider_docs'] = cls.DOCUMENTATION_PATTERNS['mistral_docs']
        elif 'ai21' in provider_lower:
            links['provider_docs'] = cls.DOCUMENTATION_PATTERNS['ai21_labs']
        elif 'cohere' in provider_lower:
            links['provider_docs'] = cls.DOCUMENTATION_PATTERNS['cohere_docs']
        elif 'stability' in provider_lower:
            links['provider_docs'] = cls.DOCUMENTATION_PATTERNS['stability_ai']
        elif 'amazon' in provider_lower and 'titan' in model_id.lower():
            links['provider_docs'] = cls.DOCUMENTATION_PATTERNS['amazon_titan']

        return links