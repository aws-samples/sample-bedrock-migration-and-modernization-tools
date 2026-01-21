"""
Input validation utilities for Lambda handlers.
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, message: str, missing_params: Optional[List[str]] = None):
        super().__init__(message)
        self.missing_params = missing_params or []


def validate_required_params(
    event: dict,
    required_params: List[str],
    handler_name: str = 'Lambda'
) -> None:
    """
    Validate that all required parameters are present in the event.

    Args:
        event: Lambda event dictionary.
        required_params: List of required parameter names.
        handler_name: Name of the handler for logging purposes.

    Raises:
        ValidationError: When required parameters are missing.

    Example:
        >>> validate_required_params(event, ['s3Bucket', 'executionId'], 'PricingAggregator')
    """
    missing = [param for param in required_params if param not in event]

    if missing:
        message = f"{handler_name}: Missing required parameters: {', '.join(missing)}"
        logger.error(message)
        raise ValidationError(message, missing_params=missing)


def build_error_response(
    error: Exception,
    retryable: bool = False
) -> dict:
    """
    Build a standardized error response for Lambda handlers.

    Args:
        error: The exception that occurred.
        retryable: Whether the error is retryable by Step Functions.

    Returns:
        Standardized error response dictionary.
    """
    return {
        'status': 'FAILED',
        'errorType': type(error).__name__,
        'errorMessage': str(error),
        'retryable': retryable
    }
