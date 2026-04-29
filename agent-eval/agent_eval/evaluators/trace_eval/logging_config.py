# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Structured logging configuration for trace evaluation.

This module provides structured logging with contextual information
(run_id, job_id, timestamps) for better debugging and monitoring.
"""

import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class StructuredLogger:
    """
    Structured logger with contextual information.
    
    Provides logging methods that automatically include run_id, job_id,
    and timestamp context for better traceability.
    
    Attributes:
        logger: Underlying Python logger
        context: Default context to include in all log messages
    """
    
    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (typically module name)
            context: Default context to include in all messages
        """
        self.logger = logging.getLogger(name)
        self.context = context or {}
    
    def _format_message(self, message: str, extra_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Format message with context.
        
        Args:
            message: Base log message
            extra_context: Additional context for this message
            
        Returns:
            Formatted message with context
        """
        # Merge default context with extra context
        full_context = {**self.context, **(extra_context or {})}
        
        # Add timestamp
        full_context['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Format context as key=value pairs
        context_str = ' '.join(f'{k}={v}' for k, v in full_context.items())
        
        if context_str:
            return f"{message} | {context_str}"
        return message
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self.logger.debug(self._format_message(message, kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self.logger.info(self._format_message(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self.logger.warning(self._format_message(message, kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self.logger.error(self._format_message(message, kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self.logger.critical(self._format_message(message, kwargs))
    
    def with_context(self, **kwargs) -> "StructuredLogger":
        """
        Create a new logger with additional context.
        
        Args:
            **kwargs: Additional context to add
            
        Returns:
            New StructuredLogger with merged context
        """
        new_context = {**self.context, **kwargs}
        return StructuredLogger(self.logger.name, new_context)


def setup_logging(level: str = "INFO", format_style: str = "simple") -> None:
    """
    Configure logging for trace evaluation.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_style: Format style ("simple" or "detailed")
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure format
    if format_style == "detailed":
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        log_format = "%(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        stream=sys.stdout
    )


def get_logger(name: str, **context) -> StructuredLogger:
    """
    Get a structured logger with optional context.
    
    Args:
        name: Logger name (typically __name__)
        **context: Default context for this logger
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, context)


# Convenience function for creating job-scoped loggers
def get_job_logger(run_id: str, job_id: str) -> StructuredLogger:
    """
    Get a logger with job context.
    
    Args:
        run_id: Run identifier
        job_id: Job identifier
        
    Returns:
        StructuredLogger with run_id and job_id context
    """
    return StructuredLogger(
        "agent_eval.trace_eval.job",
        context={"run_id": run_id, "job_id": job_id}
    )


# Convenience function for creating run-scoped loggers
def get_run_logger(run_id: str) -> StructuredLogger:
    """
    Get a logger with run context.
    
    Args:
        run_id: Run identifier
        
    Returns:
        StructuredLogger with run_id context
    """
    return StructuredLogger(
        "agent_eval.trace_eval.run",
        context={"run_id": run_id}
    )
