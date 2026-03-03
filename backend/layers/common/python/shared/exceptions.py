"""Custom exception hierarchy for Bedrock Model Profiler.

This module provides a structured exception hierarchy with retry classification
for use with Step Functions error handling.

Usage:
    from shared.exceptions import ValidationError, S3ReadError, ThrottlingError

    try:
        validate_input(event)
    except ValidationError as e:
        return {'status': 'FAILED', 'errorType': e.error_code, 'retryable': e.retryable}
"""

from typing import Any, Optional


class ProfilerError(Exception):
    """Base exception for Bedrock Model Profiler.

    All custom exceptions inherit from this class, providing consistent
    attributes for error handling and logging.

    Attributes:
        message: Human-readable error message
        retryable: Whether this error should trigger a retry in Step Functions
        error_code: Machine-readable error code for categorization
        context: Additional context dictionary for debugging
    """

    def __init__(
        self,
        message: str,
        retryable: bool = False,
        error_code: str = "PROFILER_ERROR",
        context: Optional[dict] = None,
    ):
        self.message = message
        self.retryable = retryable
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} (context: {self.context})"
        return self.message

    def to_dict(self) -> dict:
        """Convert exception to dictionary for structured logging.

        Returns:
            Dictionary with error_code, message, retryable, and context.
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "retryable": self.retryable,
            "context": self.context,
        }


# =============================================================================
# Validation Exceptions
# =============================================================================


class ValidationError(ProfilerError):
    """Input validation error - never retryable.

    Raised when Lambda input fails validation. Since invalid input won't
    change on retry, these errors are never retryable.
    """

    def __init__(self, message: str, field: str = None, value: Any = None):
        context = {}
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)[:100]  # Truncate long values
        super().__init__(
            message, retryable=False, error_code="VALIDATION_ERROR", context=context
        )
        self.field = field
        self.value = value


# =============================================================================
# S3 Exceptions
# =============================================================================


class S3Error(ProfilerError):
    """Base exception for S3 operations."""

    def __init__(
        self,
        message: str,
        bucket: str = None,
        key: str = None,
        retryable: bool = True,
        error_code: str = "S3_ERROR",
    ):
        context = {}
        if bucket:
            context["bucket"] = bucket
        if key:
            context["key"] = key
        super().__init__(
            message, retryable=retryable, error_code=error_code, context=context
        )
        self.bucket = bucket
        self.key = key


class S3ReadError(S3Error):
    """Failed to read from S3 - retryable.

    Raised when an S3 GetObject operation fails. These are typically
    transient and should be retried.
    """

    def __init__(self, bucket: str, key: str, cause: str = None):
        message = f"Failed to read s3://{bucket}/{key}"
        if cause:
            message = f"{message}: {cause}"
        super().__init__(
            message, bucket=bucket, key=key, retryable=True, error_code="S3_READ_ERROR"
        )
        if cause:
            self.context["cause"] = cause


class S3WriteError(S3Error):
    """Failed to write to S3 - retryable.

    Raised when an S3 PutObject operation fails. These are typically
    transient and should be retried.
    """

    def __init__(self, bucket: str, key: str, cause: str = None):
        message = f"Failed to write s3://{bucket}/{key}"
        if cause:
            message = f"{message}: {cause}"
        super().__init__(
            message, bucket=bucket, key=key, retryable=True, error_code="S3_WRITE_ERROR"
        )
        if cause:
            self.context["cause"] = cause


# =============================================================================
# API Exceptions
# =============================================================================


class APIError(ProfilerError):
    """Base exception for AWS API errors."""

    def __init__(
        self,
        message: str,
        service: str = None,
        operation: str = None,
        region: str = None,
        retryable: bool = False,
        error_code: str = "API_ERROR",
    ):
        context = {}
        if service:
            context["service"] = service
        if operation:
            context["operation"] = operation
        if region:
            context["region"] = region
        super().__init__(
            message, retryable=retryable, error_code=error_code, context=context
        )
        self.service = service
        self.operation = operation
        self.region = region


class ThrottlingError(APIError):
    """API throttling error - always retryable.

    Raised when an AWS API returns a throttling error. These should
    always be retried with exponential backoff.
    """

    def __init__(self, service: str, operation: str, region: str = None):
        super().__init__(
            f"{service} {operation} throttled",
            service=service,
            operation=operation,
            region=region,
            retryable=True,
            error_code="THROTTLING_ERROR",
        )


class BedrockAPIError(APIError):
    """Error calling Bedrock API.

    Retryable for throttling and transient errors, not retryable for
    access denied or invalid requests.
    """

    def __init__(
        self, operation: str, region: str, cause: str, retryable: bool = False
    ):
        super().__init__(
            f"Bedrock {operation} failed in {region}: {cause}",
            service="bedrock",
            operation=operation,
            region=region,
            retryable=retryable,
            error_code="BEDROCK_API_ERROR",
        )
        self.context["cause"] = cause


class PricingAPIError(APIError):
    """Error calling Pricing API."""

    def __init__(self, operation: str, cause: str, retryable: bool = False):
        super().__init__(
            f"Pricing {operation} failed: {cause}",
            service="pricing",
            operation=operation,
            region="us-east-1",  # Pricing API is only in us-east-1
            retryable=retryable,
            error_code="PRICING_API_ERROR",
        )
        self.context["cause"] = cause


class QuotaAPIError(APIError):
    """Error calling Service Quotas API."""

    def __init__(
        self, operation: str, region: str, cause: str, retryable: bool = False
    ):
        super().__init__(
            f"ServiceQuotas {operation} failed in {region}: {cause}",
            service="service-quotas",
            operation=operation,
            region=region,
            retryable=retryable,
            error_code="QUOTA_API_ERROR",
        )
        self.context["cause"] = cause


# =============================================================================
# Configuration Exceptions
# =============================================================================


class ConfigurationError(ProfilerError):
    """Configuration error - never retryable.

    Raised when configuration is missing or invalid. These require
    manual intervention and should not be retried.
    """

    def __init__(self, message: str, config_key: str = None):
        context = {}
        if config_key:
            context["config_key"] = config_key
        super().__init__(
            message, retryable=False, error_code="CONFIGURATION_ERROR", context=context
        )
        self.config_key = config_key


# =============================================================================
# Data Processing Exceptions
# =============================================================================


class DataProcessingError(ProfilerError):
    """Base exception for data processing errors - not retryable."""

    def __init__(
        self,
        message: str,
        error_code: str = "DATA_PROCESSING_ERROR",
        context: dict = None,
    ):
        super().__init__(
            message, retryable=False, error_code=error_code, context=context
        )


class AggregationError(DataProcessingError):
    """Error during data aggregation."""

    def __init__(self, message: str, source: str = None):
        context = {"source": source} if source else {}
        super().__init__(message, error_code="AGGREGATION_ERROR", context=context)


class TransformationError(DataProcessingError):
    """Error during data transformation."""

    def __init__(self, message: str, model_id: str = None):
        context = {"model_id": model_id} if model_id else {}
        super().__init__(message, error_code="TRANSFORMATION_ERROR", context=context)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base
    "ProfilerError",
    # Validation
    "ValidationError",
    # S3
    "S3Error",
    "S3ReadError",
    "S3WriteError",
    # API
    "APIError",
    "ThrottlingError",
    "BedrockAPIError",
    "PricingAPIError",
    "QuotaAPIError",
    # Configuration
    "ConfigurationError",
    # Data Processing
    "DataProcessingError",
    "AggregationError",
    "TransformationError",
]
