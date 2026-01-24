"""
Shared utilities for Bedrock Model Profiler Lambda functions.

This module provides common functionality used across all Lambda handlers:
- S3 operations with proper exception handling
- AWS client configuration with retry logic
- Execution ID parsing
- Input validation utilities
- Configuration loading from S3 with fallback defaults
"""

from shared.config import RETRY_CONFIG, get_logger
from shared.s3_utils import get_s3_client, read_from_s3, write_to_s3, S3ReadError, S3WriteError
from shared.execution import parse_execution_id
from shared.validation import validate_required_params, ValidationError
from shared.config_loader import ConfigLoader, get_config_loader

__all__ = [
    # Config
    'RETRY_CONFIG',
    'get_logger',
    # S3 utilities
    'get_s3_client',
    'read_from_s3',
    'write_to_s3',
    'S3ReadError',
    'S3WriteError',
    # Execution utilities
    'parse_execution_id',
    # Validation utilities
    'validate_required_params',
    'ValidationError',
    # Configuration loader
    'ConfigLoader',
    'get_config_loader',
]
