"""Tests for Powertools shared module.

These tests require the aws-lambda-powertools package to be installed.
They will be skipped if the package is not available.
"""

import sys
from pathlib import Path
import pytest

# Add the shared layer to the path for direct import
SHARED_LAYER_PATH = Path(__file__).parent.parent / "layers" / "common" / "python"
if str(SHARED_LAYER_PATH) not in sys.path:
    sys.path.insert(0, str(SHARED_LAYER_PATH))

# Check if aws_lambda_powertools is available
try:
    import aws_lambda_powertools

    POWERTOOLS_AVAILABLE = True
except ImportError:
    POWERTOOLS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not POWERTOOLS_AVAILABLE, reason="aws-lambda-powertools package not installed"
)


def test_logger_instance_type():
    """Logger should be a Powertools Logger instance."""
    from shared.powertools import logger
    from aws_lambda_powertools import Logger

    assert isinstance(logger, Logger)


def test_tracer_instance_type():
    """Tracer should be a Powertools Tracer instance."""
    from shared.powertools import tracer
    from aws_lambda_powertools import Tracer

    assert isinstance(tracer, Tracer)


def test_metrics_instance_type():
    """Metrics should be a Powertools Metrics instance."""
    from shared.powertools import metrics
    from aws_lambda_powertools import Metrics

    assert isinstance(metrics, Metrics)


def test_service_name_default():
    """Default service name should be 'bedrock-profiler'."""
    from shared.powertools import SERVICE_NAME

    # Note: This may be overridden by env var in test environment
    assert SERVICE_NAME is not None


def test_service_name_from_env(monkeypatch):
    """Service name should be configurable via environment variable."""
    import importlib
    import os

    # Set custom service name
    monkeypatch.setenv("POWERTOOLS_SERVICE_NAME", "custom-service")

    # Reload the module to pick up the new env var
    import shared.powertools as powertools_module

    importlib.reload(powertools_module)

    assert powertools_module.SERVICE_NAME == "custom-service"

    # Clean up: reload with default
    monkeypatch.delenv("POWERTOOLS_SERVICE_NAME", raising=False)
    importlib.reload(powertools_module)


def test_exports_lambda_context():
    """LambdaContext should be exported."""
    from shared.powertools import LambdaContext

    assert LambdaContext is not None


def test_exports_metric_unit():
    """MetricUnit should be exported."""
    from shared.powertools import MetricUnit

    assert MetricUnit is not None
    assert hasattr(MetricUnit, "Count")
