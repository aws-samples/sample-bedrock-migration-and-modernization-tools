# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Retry policy implementation with exponential backoff.

This module provides configurable retry logic for handling transient
failures in judge API calls.
"""

import asyncio
import inspect
import logging
import random
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional, Set, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.
    
    Attributes:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        jitter: Enable jitter to prevent thundering herd (default: True)
        jitter_factor: Jitter randomization factor 0-1 (default: 0.5)
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.5
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be >= 0, got {self.max_retries}"
            )
        if self.base_delay <= 0:
            raise ValueError(
                f"base_delay must be > 0, got {self.base_delay}"
            )
        if self.max_delay <= 0:
            raise ValueError(
                f"max_delay must be > 0, got {self.max_delay}"
            )
        if self.max_delay < self.base_delay:
            raise ValueError(
                f"max_delay ({self.max_delay}) must be >= base_delay ({self.base_delay})"
            )
        if self.exponential_base <= 1:
            raise ValueError(
                f"exponential_base must be > 1, got {self.exponential_base}"
            )
        if not 0 <= self.jitter_factor <= 1:
            raise ValueError(
                f"jitter_factor must be between 0 and 1, got {self.jitter_factor}"
            )
    
    @classmethod
    def from_dict(cls, config: Optional[dict]) -> 'RetryConfig':
        """
        Create RetryConfig from dictionary.
        
        Args:
            config: Dictionary with retry configuration
            
        Returns:
            RetryConfig instance with defaults for missing fields
        """
        if not config:
            return cls()
        
        return cls(
            max_retries=config.get('max_retries', 3),
            base_delay=config.get('base_delay', 1.0),
            max_delay=config.get('max_delay', 60.0),
            exponential_base=config.get('exponential_base', 2.0),
            jitter=config.get('jitter', True),
            jitter_factor=config.get('jitter_factor', 0.5)
        )


class RetryPolicy:
    """
    Retry policy with exponential backoff and jitter.
    
    Implements retry logic with exponential backoff for handling
    transient failures. Default backoff sequence: 1s, 2s, 4s.
    Supports selective retryability and jitter to prevent thundering herd.
    """
    
    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        retry_on: Optional[Set[Type[Exception]]] = None,
        non_retry_on: Optional[Set[Type[Exception]]] = None
    ):
        """
        Initialize retry policy.
        
        Args:
            config: Retry configuration (uses defaults if None)
            retry_on: Set of exception types to retry (None = retry all)
            non_retry_on: Set of exception types to never retry
        """
        self.config = config or RetryConfig()
        self.retry_on = retry_on
        self.non_retry_on = non_retry_on or {
            # is not expected to retry these by default
            ValueError,
            TypeError,
            KeyError,
            AttributeError
        }
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for the given retry attempt with optional jitter.
        
        Uses exponential backoff: base_delay * (exponential_base ^ attempt)
        Capped at max_delay. Adds jitter if enabled to prevent thundering herd.
        
        Args:
            attempt: Retry attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        # Calculate base exponential delay
        delay = self.config.base_delay * (
            self.config.exponential_base ** attempt
        )
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            # Full jitter: randomize between 0 and delay * (1 + jitter_factor)
            jitter_range = delay * self.config.jitter_factor
            delay = delay + random.uniform(-jitter_range, jitter_range)
            # helps delay stays positive and within bounds
            delay = max(0.0, min(delay, self.config.max_delay))
        
        return delay
    
    def _should_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.
        
        Args:
            exception: The exception that occurred
            
        Returns:
            True if should retry, False otherwise
        """
        # is not expected to retry cancellation
        if isinstance(exception, asyncio.CancelledError):
            return False
        
        # Check non-retry list first
        if self.non_retry_on:
            for exc_type in self.non_retry_on:
                if isinstance(exception, exc_type):
                    return False
        
        # If retry_on is specified, only retry those types
        if self.retry_on:
            for exc_type in self.retry_on:
                if isinstance(exception, exc_type):
                    return True
            return False
        
        # Default: retry all exceptions except those in non_retry_on
        return True
    
    async def execute_with_retry(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        context: Optional[dict] = None,
        **kwargs: Any
    ) -> T:
        """
        Execute async function with retry logic.
        
        Retries the function up to max_retries times with exponential
        backoff between attempts. Logs each retry attempt with context.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            context: Optional context dict for structured logging
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from successful function execution
            
        Raises:
            asyncio.CancelledError: Immediately on cancellation
            Exception: The last exception if all retries fail
        """
        context = context or {}
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Handle both sync and async callables
                if inspect.isawaitable(result):
                    result = await result
                
                if attempt > 0:
                    logger.info(
                        f"Retry succeeded on attempt {attempt + 1}",
                        extra={"context": context, "attempt": attempt + 1}
                    )
                
                return result
                
            except asyncio.CancelledError:
                # typically re-raise cancellation immediately
                logger.info(
                    "Task cancelled during retry execution",
                    extra={"context": context, "attempt": attempt + 1}
                )
                raise
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if not self._should_retry(e):
                    logger.warning(
                        f"Non-retryable exception: {type(e).__name__}: {str(e)}",
                        extra={"context": context, "attempt": attempt + 1}
                    )
                    raise
                
                if attempt < self.config.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {type(e).__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f}s...",
                        extra={
                            "context": context,
                            "attempt": attempt + 1,
                            "delay_seconds": delay,
                            "exception_type": type(e).__name__
                        }
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {self.config.max_retries + 1} attempts failed. "
                        f"Last error: {type(e).__name__}: {str(e)}",
                        extra={
                            "context": context,
                            "total_attempts": self.config.max_retries + 1,
                            "exception_type": type(e).__name__
                        }
                    )
        
        # All retries exhausted
        raise last_exception
    
    def get_backoff_sequence(self) -> List[float]:
        """
        Get the complete backoff delay sequence.
        
        Returns:
            List of delays in seconds for each retry attempt
        """
        return [
            self.calculate_delay(i)
            for i in range(self.config.max_retries)
        ]
