"""
Configuration file for Amazon Bedrock Pricing Collector
Centralizes all configuration variables and settings
"""

import os
import sys
import re
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    """Centralized configuration for the Bedrock Pricing Collector"""

    # AWS Configuration from shared config
    AWS_PROFILE_NAME = None  # Will be overridden by command line argument
    AWS_REGION = SharedConfig.AWS_DEFAULT_REGION

    # API Configuration
    PRICING_API_REGION = 'us-east-1'  # Pricing API is only available in us-east-1 and ap-south-1

    # Service Codes for AWS Pricing API
    AWS_PRICING_SERVICE_CODES = ['AmazonBedrock', 'AmazonBedrockService', 'AmazonBedrockFoundationModels']


    # Request Configuration
    REQUEST_TIMEOUT = 30  # seconds
    MAX_RESULTS_PER_PAGE = 100
    MAX_BATCHES_LIMIT = 50

    # Retry Configuration
    RETRY_TOTAL = 3
    RETRY_BACKOFF_FACTOR = 1
    RETRY_STATUS_FORCELIST = [429, 500, 502, 503, 504]

    # Progress Reporting
    PROGRESS_REPORT_INTERVAL = 25  # Report progress every N models

    # AWS Region Documentation URL
    AWS_REGIONS_URL = 'https://docs.aws.amazon.com/global-infrastructure/latest/regions/aws-region-billing-codes.html'

    # Region Mapping (Static fallback - will be updated dynamically)
    LOCATION_TO_REGION_MAP = {
        'US East (N. Virginia)': 'us-east-1',
        'US East (Ohio)': 'us-east-2',
        'US West (Oregon)': 'us-west-2',
        'US West (N. California)': 'us-west-1',
        'Europe (Ireland)': 'eu-west-1',
        'Europe (London)': 'eu-west-2',
        'Europe (Paris)': 'eu-west-3',
        'Europe (Frankfurt)': 'eu-central-1',
        'Asia Pacific (Tokyo)': 'ap-northeast-1',
        'Asia Pacific (Seoul)': 'ap-northeast-2',
        'Asia Pacific (Singapore)': 'ap-southeast-1',
        'Asia Pacific (Sydney)': 'ap-southeast-2',
        'Asia Pacific (Mumbai)': 'ap-south-1',
        'Canada (Central)': 'ca-central-1',
        'South America (São Paulo)': 'sa-east-1'
    }

    # Region Code Mapping
    REGION_CODE_MAP = {
        'USE1': 'us-east-1', 'USE2': 'us-east-2',
        'USW1': 'us-west-1', 'USW2': 'us-west-2',
        'EUW1': 'eu-west-1', 'EUW2': 'eu-west-2', 'EUW3': 'eu-west-3',
        'EUC1': 'eu-central-1', 'EUC2': 'eu-central-2',
        'EUN1': 'eu-north-1', 'EUS1': 'eu-south-1', 'EUS2': 'eu-south-2',
        'APN1': 'ap-northeast-1', 'APN2': 'ap-northeast-2', 'APN3': 'ap-northeast-3',
        'APS1': 'ap-southeast-1', 'APS2': 'ap-southeast-2', 'APS3': 'ap-southeast-3',
        'APS4': 'ap-southeast-4', 'APS5': 'ap-southeast-5', 'APS6': 'ap-southeast-6',
        'APS7': 'ap-southeast-7', 'APS9': 'ap-southeast-9',
        'APE1': 'ap-east-1', 'APE2': 'ap-east-2',
        'CAN1': 'ca-central-1', 'SAE1': 'sa-east-1',
        'MEC1': 'me-central-1', 'ILC1': 'il-central-1'
    }


    # File and Directory Configuration
    OUTPUT_DIR = 'out'
    LOGS_DIR = 'logs'
    LOG_FILENAME = 'pricing_collection.log'
    OUTPUT_FILE_PREFIX = 'bedrock-pricing'

    # Metadata
    VERSION = '1.0.0'
    CURRENCY = 'USD'
    PRICING_STANDARDIZATION_MESSAGE = 'Smart conversion applied: per-million to per-thousand when needed, unit extraction from descriptions'
    STRUCTURE_DESCRIPTION = 'provider > model > region > pricing_groups > dimensions'

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
        return 'Mozilla/5.0 (compatible; bedrock-pricing-collector)'

    @classmethod
    def get_request_headers(cls) -> Dict[str, str]:
        """Get standard request headers"""
        return {
            'User-Agent': cls.get_user_agent(),
            'Accept': 'application/json'
        }



    @classmethod
    def fetch_aws_region_data(cls) -> Optional[str]:
        """
        Fetch region data from AWS documentation

        Returns:
            Raw HTML content or None if failed
        """
        try:
            headers = cls.get_request_headers()
            response = requests.get(cls.AWS_REGIONS_URL, headers=headers, timeout=cls.REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logging.warning(f"Failed to fetch AWS region data: {str(e)}")
            return None

    @classmethod
    def parse_region_mappings(cls, html_content: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Parse region mappings from AWS documentation HTML

        Args:
            html_content: Raw HTML content from AWS docs

        Returns:
            Tuple of (location_to_region_map, region_code_map)
        """
        location_to_region = {}
        region_code_map = {}

        try:
            # Extract region information using regex patterns
            # Pattern matches the AWS documentation table structure: billing code, region code, and description
            region_pattern = r'<td[^>]*><code[^>]*>([A-Z]{2,4}\d?)</code></td>\s*<td[^>]*><code[^>]*>([a-z0-9-]+)</code></td>\s*<td[^>]*>([^<]+)</td>'

            matches = re.findall(region_pattern, html_content, re.IGNORECASE | re.MULTILINE)

            for billing_code, region_code, description in matches:
                # Clean up the description
                description = description.strip()
                if description.endswith('.'):
                    description = description[:-1]

                # Build mappings
                location_to_region[description] = region_code
                region_code_map[billing_code.upper()] = region_code

            logging.info(f"Parsed {len(location_to_region)} location mappings and {len(region_code_map)} region code mappings")

        except Exception as e:
            logging.warning(f"Failed to parse region mappings: {str(e)}")

        return location_to_region, region_code_map

    @classmethod
    def update_region_mappings(cls) -> bool:
        """
        Update region mappings dynamically from AWS documentation

        Returns:
            True if successful, False if failed (will use static fallbacks)
        """
        logging.info("Attempting to update region mappings from AWS documentation...")

        # Fetch region data
        html_content = cls.fetch_aws_region_data()
        if not html_content:
            logging.warning("Using static region mappings as fallback")
            return False

        # Parse mappings
        location_to_region, region_code_map = cls.parse_region_mappings(html_content)

        if not location_to_region or not region_code_map:
            logging.warning("Failed to parse region mappings, using static fallbacks")
            return False

        # Update class attributes
        cls.LOCATION_TO_REGION_MAP = location_to_region
        cls.REGION_CODE_MAP = region_code_map

        logging.info(f"✅ Successfully updated region mappings: {len(location_to_region)} locations, {len(region_code_map)} codes")
        return True

    @classmethod
    def get_region_mappings(cls) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Get current region mappings (either dynamic or static fallback)

        Returns:
            Tuple of (location_to_region_map, region_code_map)
        """
        return cls.LOCATION_TO_REGION_MAP, cls.REGION_CODE_MAP