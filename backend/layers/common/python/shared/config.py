"""
Shared configuration for Lambda functions.

Configuration (environment variables):
    AWS_CONNECT_TIMEOUT: Connection timeout for AWS SDK clients in seconds (default: 10)
    AWS_READ_TIMEOUT: Read timeout for AWS SDK clients in seconds (default: 30)
    AWS_MAX_RETRIES: Maximum retry attempts for AWS SDK clients (default: 3)
    LOG_LEVEL: Logging level (default: INFO)
"""

import logging
import os

from botocore.config import Config

# Configuration with defaults
AWS_CONNECT_TIMEOUT = int(os.environ.get("AWS_CONNECT_TIMEOUT", "10"))
AWS_READ_TIMEOUT = int(os.environ.get("AWS_READ_TIMEOUT", "30"))
AWS_MAX_RETRIES = int(os.environ.get("AWS_MAX_RETRIES", "3"))

# Standard retry configuration for AWS SDK clients
RETRY_CONFIG = Config(
    retries={"max_attempts": AWS_MAX_RETRIES, "mode": "adaptive"},
    connect_timeout=AWS_CONNECT_TIMEOUT,
    read_timeout=AWS_READ_TIMEOUT,
)


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a configured logger for Lambda functions.

    Args:
        name: Logger name. If None, returns the root logger.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    return logger
