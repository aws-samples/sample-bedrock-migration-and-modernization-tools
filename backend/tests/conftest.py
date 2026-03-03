"""
Pytest configuration for Phase 1 tests.

This conftest.py sets up mocks for aws_lambda_powertools and typing_extensions
before any test imports.
"""

import sys
from unittest.mock import MagicMock

# Create comprehensive mocks for aws_lambda_powertools
mock_powertools = MagicMock()
mock_powertools.Logger = MagicMock()
mock_powertools.Tracer = MagicMock()
mock_powertools.Metrics = MagicMock()

mock_utilities = MagicMock()
mock_utilities.typing = MagicMock()
mock_utilities.typing.LambdaContext = MagicMock()

mock_metrics = MagicMock()
mock_metrics.MetricUnit = MagicMock()
mock_metrics.MetricUnit.Count = "Count"
mock_metrics.MetricUnit.Milliseconds = "Milliseconds"

# Install mocks before any imports
sys.modules["aws_lambda_powertools"] = mock_powertools
sys.modules["aws_lambda_powertools.utilities"] = mock_utilities
sys.modules["aws_lambda_powertools.utilities.typing"] = mock_utilities.typing
sys.modules["aws_lambda_powertools.metrics"] = mock_metrics

# Mock typing_extensions for NotRequired (Python 3.9 compatibility)
try:
    from typing import NotRequired
except ImportError:
    # Python < 3.11, need to mock NotRequired
    import typing

    typing.NotRequired = MagicMock()

# Add paths for Lambda handlers
sys.path.insert(0, "layers/common/python")
sys.path.insert(0, "lambdas/gap-detection")
sys.path.insert(0, "lambdas/self-healing-agent")
sys.path.insert(0, "lambdas/config-sync")
