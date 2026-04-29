# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Shared exception types for judge execution.

This module defines standard exception types used across the judge
execution pipeline for consistent error handling and classification.
"""


class JudgeExecutionError(Exception):
    """
    Base exception for all judge execution errors.
    
    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code
        retryable: Whether this error is transient and retryable
        context: Additional context (job_id, judge_id, retry_count, etc.)
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        retryable: bool = False,
        context: dict = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.retryable = retryable
        self.context = context or {}
    
    @property
    def job_id(self) -> str:
        """Get job_id from context if available."""
        return self.context.get('job_id')
    
    @property
    def judge_id(self) -> str:
        """Get judge_id from context if available."""
        return self.context.get('judge_id')
    
    @property
    def retry_count(self) -> int:
        """Get retry_count from context if available."""
        return self.context.get('retry_count', 0)
    
    @classmethod
    def from_job(
        cls,
        message: str,
        job_id: str,
        judge_id: str,
        error: str,
        retry_count: int = 0,
        error_code: str = "UNKNOWN_ERROR",
        retryable: bool = False
    ) -> "JudgeExecutionError":
        """
        Create exception with job context.
        
        Args:
            message: Human-readable error message
            job_id: Job identifier
            judge_id: Judge identifier
            error: Detailed error information
            retry_count: Number of retries attempted
            error_code: Machine-readable error code
            retryable: Whether error is retryable
            
        Returns:
            JudgeExecutionError with job context
        """
        return cls(
            message=message,
            error_code=error_code,
            retryable=retryable,
            context={
                'job_id': job_id,
                'judge_id': judge_id,
                'error': error,
                'retry_count': retry_count
            }
        )
    
    def __str__(self):
        return f"{self.error_code}: {self.message}"


class ValidationError(JudgeExecutionError):
    """
    Raised when judge response validation fails.
    
    This indicates the response format is invalid, not that the
    evaluation itself failed.
    
    Error codes:
    - INVALID_JSON: Response is not valid JSON
    - MISSING_FIELD: Required field missing from response
    - INVALID_SCORE: Score value outside allowed range/values
    - INVALID_FORMAT: Response structure doesn't match expected format
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "INVALID_FORMAT",
        field: str = None,
        expected: str = None,
        actual: str = None,
        context: dict = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            retryable=True,  # Validation errors are retryable (may be transient)
            context=context or {}
        )
        self.field = field
        self.expected = expected
        self.actual = actual
        
        # Add validation details to context
        if field:
            self.context['field'] = field
        if expected:
            self.context['expected'] = expected
        if actual:
            self.context['actual'] = actual


class APIError(JudgeExecutionError):
    """
    Raised when judge API call fails.
    
    This covers network errors, authentication failures, rate limiting,
    and other API-level issues.
    
    Error codes:
    - API_TIMEOUT: Request timed out
    - API_RATE_LIMIT: Rate limit exceeded
    - API_AUTH_FAILED: Authentication failed
    - API_UNAVAILABLE: Service unavailable (5xx)
    - API_BAD_REQUEST: Bad request (4xx)
    - API_NETWORK_ERROR: Network connectivity issue
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "API_ERROR",
        status_code: int = None,
        retryable: bool = True,
        context: dict = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            retryable=retryable,
            context=context or {}
        )
        self.status_code = status_code
        
        if status_code:
            self.context['status_code'] = status_code


class TimeoutError(JudgeExecutionError):
    """
    Raised when judge execution exceeds timeout.
    
    This is a wrapper around asyncio.TimeoutError for consistent
    error handling in the pipeline.
    """
    
    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        context: dict = None
    ):
        super().__init__(
            message=message,
            error_code="TIMEOUT",
            retryable=True,
            context=context or {}
        )
        self.timeout_seconds = timeout_seconds
        self.context['timeout_seconds'] = timeout_seconds


class ValidationResult:
    """
    Structured result from response validation.
    
    Provides detailed information about validation success/failure
    for better error reporting and retry decisions.
    
    Attributes:
        is_valid: Whether validation passed
        error_code: Machine-readable error code (if invalid)
        message: Human-readable error message (if invalid)
        field: Field that failed validation (if applicable)
        expected: Expected value/format (if applicable)
        actual: Actual value received (if applicable)
    """
    
    def __init__(
        self,
        is_valid: bool,
        error_code: str = None,
        message: str = None,
        field: str = None,
        expected: str = None,
        actual: str = None
    ):
        self.is_valid = is_valid
        self.error_code = error_code
        self.message = message
        self.field = field
        self.expected = expected
        self.actual = actual
    
    @classmethod
    def success(cls) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(is_valid=True)
    
    @classmethod
    def failure(
        cls,
        error_code: str,
        message: str,
        field: str = None,
        expected: str = None,
        actual: str = None
    ) -> "ValidationResult":
        """Create a failed validation result."""
        return cls(
            is_valid=False,
            error_code=error_code,
            message=message,
            field=field,
            expected=expected,
            actual=actual
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            'is_valid': self.is_valid,
            'error_code': self.error_code,
            'message': self.message,
            'field': self.field,
            'expected': self.expected,
            'actual': self.actual
        }
    
    def __repr__(self):
        if self.is_valid:
            return "ValidationResult(valid=True)"
        return f"ValidationResult(valid=False, error={self.error_code}, message={self.message})"
