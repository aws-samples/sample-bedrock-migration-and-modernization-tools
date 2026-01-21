"""
Shared configuration for Lambda functions.
"""

import logging
import os

from botocore.config import Config

# Standard retry configuration for AWS SDK clients
RETRY_CONFIG = Config(
    retries={'max_attempts': 3, 'mode': 'adaptive'},
    connect_timeout=10,
    read_timeout=30
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
    logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))
    return logger
