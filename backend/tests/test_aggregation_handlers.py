"""Tests for Aggregation handler Powertools integration.

Aggregation handlers (Task 08):
- final-aggregator
- copy-to-latest
- gap-detection

These tests verify that Aggregation handlers have been properly migrated
to use AWS Lambda Powertools decorators.
"""

import pytest
from pathlib import Path

HANDLERS_PATH = Path(__file__).parent.parent / "lambdas"

AGGREGATION_HANDLERS = [
    "final-aggregator",
    "copy-to-latest",
    "gap-detection",
]


def get_handler_source(handler_name: str) -> str:
    """Read the handler source code."""
    handler_path = HANDLERS_PATH / handler_name / "handler.py"
    return handler_path.read_text()


class TestFinalAggregatorPowertools:
    """Tests for final-aggregator Powertools integration."""

    def test_final_aggregator_imports_powertools(self):
        """final-aggregator should import from shared.powertools."""
        source = get_handler_source("final-aggregator")
        assert "from shared.powertools import" in source

    def test_final_aggregator_has_all_decorators(self):
        """final-aggregator should have all three Powertools decorators."""
        source = get_handler_source("final-aggregator")
        assert "@logger.inject_lambda_context" in source
        assert "@tracer.capture_lambda_handler" in source
        assert "@metrics.log_metrics" in source

    def test_final_aggregator_has_traced_methods(self):
        """final-aggregator should have @tracer.capture_method on helper functions.

        As a complex aggregation handler, final-aggregator should trace
        its internal methods for better observability.
        """
        source = get_handler_source("final-aggregator")
        assert "@tracer.capture_method" in source, (
            "final-aggregator should use @tracer.capture_method for helper functions"
        )


class TestCopyToLatestPowertools:
    """Tests for copy-to-latest Powertools integration."""

    def test_copy_to_latest_imports_powertools(self):
        """copy-to-latest should import from shared.powertools."""
        source = get_handler_source("copy-to-latest")
        assert "from shared.powertools import" in source

    def test_copy_to_latest_has_all_decorators(self):
        """copy-to-latest should have all three Powertools decorators."""
        source = get_handler_source("copy-to-latest")
        assert "@logger.inject_lambda_context" in source
        assert "@tracer.capture_lambda_handler" in source
        assert "@metrics.log_metrics" in source


class TestGapDetectionPowertools:
    """Tests for gap-detection Powertools integration."""

    def test_gap_detection_imports_powertools(self):
        """gap-detection should import from shared.powertools."""
        source = get_handler_source("gap-detection")
        assert "from shared.powertools import" in source

    def test_gap_detection_has_all_decorators(self):
        """gap-detection should have all three Powertools decorators."""
        source = get_handler_source("gap-detection")
        assert "@logger.inject_lambda_context" in source
        assert "@tracer.capture_lambda_handler" in source
        assert "@metrics.log_metrics" in source


class TestAggregationHandlersNoLegacyLogging:
    """Tests to verify Aggregation handlers don't use legacy logging."""

    @pytest.mark.parametrize("handler_name", AGGREGATION_HANDLERS)
    def test_no_legacy_logging(self, handler_name):
        """Aggregation handlers should not use legacy logging.getLogger()."""
        source = get_handler_source(handler_name)
        assert "logger = logging.getLogger()" not in source, (
            f"{handler_name} should not use legacy logging"
        )


class TestAggregationHandlersStructuredLogging:
    """Tests to verify Aggregation handlers use structured logging."""

    @pytest.mark.parametrize("handler_name", AGGREGATION_HANDLERS)
    def test_uses_structured_logging(self, handler_name):
        """Aggregation handlers should use structured logging with extra= parameter."""
        source = get_handler_source(handler_name)
        has_structured_logging = (
            "extra={" in source or "extra = {" in source or ", extra=" in source
        )
        assert has_structured_logging, (
            f"{handler_name} should use structured logging with extra= parameter"
        )


class TestAggregationHandlersMetrics:
    """Tests to verify Aggregation handlers emit custom metrics."""

    def test_final_aggregator_emits_metrics(self):
        """final-aggregator should emit custom metrics."""
        source = get_handler_source("final-aggregator")
        assert "metrics.add_metric" in source, (
            "final-aggregator should emit custom metrics"
        )

    def test_copy_to_latest_emits_metrics(self):
        """copy-to-latest should emit custom metrics."""
        source = get_handler_source("copy-to-latest")
        assert "metrics.add_metric" in source, (
            "copy-to-latest should emit custom metrics"
        )
