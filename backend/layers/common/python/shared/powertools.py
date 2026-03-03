"""Centralized AWS Lambda Powertools configuration.

This module provides pre-configured Logger, Tracer, and Metrics instances
for use across all Lambda handlers in the Bedrock Model Profiler.

Usage:
    from shared.powertools import logger, tracer, metrics, LambdaContext

    @logger.inject_lambda_context(log_event=True)
    @tracer.capture_lambda_handler
    @metrics.log_metrics(capture_cold_start_metric=True)
    def lambda_handler(event: dict, context: LambdaContext) -> dict:
        logger.info("Processing request", extra={"key": "value"})
        metrics.add_metric(name="RequestCount", unit=MetricUnit.Count, value=1)
        return {"status": "SUCCESS"}
"""

import os
from aws_lambda_powertools import Logger, Tracer, Metrics
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools.metrics import MetricUnit

# Service name from environment or default
SERVICE_NAME = os.environ.get("POWERTOOLS_SERVICE_NAME", "bedrock-profiler")

# Metrics namespace from environment or default
METRICS_NAMESPACE = os.environ.get("POWERTOOLS_METRICS_NAMESPACE", "BedrockProfiler")

# Pre-configured Logger instance
# - Automatically includes Lambda context when using @logger.inject_lambda_context
# - Outputs structured JSON logs
# - Log level controlled by LOG_LEVEL environment variable
logger = Logger(service=SERVICE_NAME)

# Pre-configured Tracer instance
# - Automatically creates X-Ray segments for Lambda invocations
# - Use @tracer.capture_method for sub-segment tracing
tracer = Tracer(service=SERVICE_NAME)

# Pre-configured Metrics instance
# - Automatically flushes metrics at end of invocation
# - Use @metrics.log_metrics decorator on handler
metrics = Metrics(namespace=METRICS_NAMESPACE, service=SERVICE_NAME)

__all__ = [
    "logger",
    "tracer",
    "metrics",
    "LambdaContext",
    "MetricUnit",
    "SERVICE_NAME",
    "METRICS_NAMESPACE",
]
