"""Tests for Wave 1 handler Powertools integration.

Wave 1 handlers (Task 06):
- pricing-collector
- pricing-aggregator
- model-extractor
- model-merger
- quota-collector

These tests verify that Wave 1 handlers have been properly migrated
to use AWS Lambda Powertools decorators.
"""

import pytest
from pathlib import Path

HANDLERS_PATH = Path(__file__).parent.parent / "lambdas"

WAVE1_HANDLERS = [
    "pricing-collector",
    "pricing-aggregator",
    "model-extractor",
    "model-merger",
    "quota-collector",
]


def get_handler_source(handler_name: str) -> str:
    """Read the handler source code."""
    handler_path = HANDLERS_PATH / handler_name / "handler.py"
    return handler_path.read_text()


class TestPricingCollectorPowertools:
    """Tests for pricing-collector Powertools integration."""

    def test_pricing_collector_imports_powertools(self):
        """pricing-collector should import from shared.powertools."""
        source = get_handler_source("pricing-collector")
        assert "from shared.powertools import" in source

    def test_pricing_collector_has_all_decorators(self):
        """pricing-collector should have all three Powertools decorators."""
        source = get_handler_source("pricing-collector")
        assert "@logger.inject_lambda_context" in source
        assert "@tracer.capture_lambda_handler" in source
        assert "@metrics.log_metrics" in source

    def test_pricing_collector_has_traced_methods(self):
        """pricing-collector should have @tracer.capture_method on helper functions."""
        source = get_handler_source("pricing-collector")
        assert "@tracer.capture_method" in source


class TestPricingAggregatorPowertools:
    """Tests for pricing-aggregator Powertools integration."""

    def test_pricing_aggregator_imports_powertools(self):
        """pricing-aggregator should import from shared.powertools."""
        source = get_handler_source("pricing-aggregator")
        assert "from shared.powertools import" in source

    def test_pricing_aggregator_has_all_decorators(self):
        """pricing-aggregator should have all three Powertools decorators."""
        source = get_handler_source("pricing-aggregator")
        assert "@logger.inject_lambda_context" in source
        assert "@tracer.capture_lambda_handler" in source
        assert "@metrics.log_metrics" in source


class TestModelExtractorPowertools:
    """Tests for model-extractor Powertools integration."""

    def test_model_extractor_imports_powertools(self):
        """model-extractor should import from shared.powertools."""
        source = get_handler_source("model-extractor")
        assert "from shared.powertools import" in source

    def test_model_extractor_has_all_decorators(self):
        """model-extractor should have all three Powertools decorators."""
        source = get_handler_source("model-extractor")
        assert "@logger.inject_lambda_context" in source
        assert "@tracer.capture_lambda_handler" in source
        assert "@metrics.log_metrics" in source


class TestModelMergerPowertools:
    """Tests for model-merger Powertools integration."""

    def test_model_merger_imports_powertools(self):
        """model-merger should import from shared.powertools."""
        source = get_handler_source("model-merger")
        assert "from shared.powertools import" in source

    def test_model_merger_has_all_decorators(self):
        """model-merger should have all three Powertools decorators."""
        source = get_handler_source("model-merger")
        assert "@logger.inject_lambda_context" in source
        assert "@tracer.capture_lambda_handler" in source
        assert "@metrics.log_metrics" in source


class TestQuotaCollectorPowertools:
    """Tests for quota-collector Powertools integration."""

    def test_quota_collector_imports_powertools(self):
        """quota-collector should import from shared.powertools."""
        source = get_handler_source("quota-collector")
        assert "from shared.powertools import" in source

    def test_quota_collector_has_all_decorators(self):
        """quota-collector should have all three Powertools decorators."""
        source = get_handler_source("quota-collector")
        assert "@logger.inject_lambda_context" in source
        assert "@tracer.capture_lambda_handler" in source
        assert "@metrics.log_metrics" in source


class TestWave1HandlersNoLegacyLogging:
    """Tests to verify Wave 1 handlers don't use legacy logging."""

    @pytest.mark.parametrize("handler_name", WAVE1_HANDLERS)
    def test_no_legacy_logging(self, handler_name):
        """Wave 1 handlers should not use legacy logging.getLogger()."""
        source = get_handler_source(handler_name)
        assert "logger = logging.getLogger()" not in source, (
            f"{handler_name} should not use legacy logging"
        )
