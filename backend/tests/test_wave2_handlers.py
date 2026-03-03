"""Tests for Wave 2 handler Powertools integration.

Wave 2 handlers (Task 07):
- pricing-linker
- regional-availability
- feature-collector
- token-specs-collector
- mantle-collector
- lifecycle-collector

These tests verify that Wave 2 handlers have been properly migrated
to use AWS Lambda Powertools decorators.
"""

import pytest
from pathlib import Path

HANDLERS_PATH = Path(__file__).parent.parent / "lambdas"

WAVE2_HANDLERS = [
    "pricing-linker",
    "regional-availability",
    "feature-collector",
    "token-specs-collector",
    "mantle-collector",
    "lifecycle-collector",
]


def get_handler_source(handler_name: str) -> str:
    """Read the handler source code."""
    handler_path = HANDLERS_PATH / handler_name / "handler.py"
    return handler_path.read_text()


class TestPricingLinkerPowertools:
    """Tests for pricing-linker Powertools integration."""

    def test_pricing_linker_imports_powertools(self):
        """pricing-linker should import from shared.powertools."""
        source = get_handler_source("pricing-linker")
        assert "from shared.powertools import" in source

    def test_pricing_linker_has_all_decorators(self):
        """pricing-linker should have all three Powertools decorators."""
        source = get_handler_source("pricing-linker")
        assert "@logger.inject_lambda_context" in source
        assert "@tracer.capture_lambda_handler" in source
        assert "@metrics.log_metrics" in source


class TestRegionalAvailabilityPowertools:
    """Tests for regional-availability Powertools integration."""

    def test_regional_availability_imports_powertools(self):
        """regional-availability should import from shared.powertools."""
        source = get_handler_source("regional-availability")
        assert "from shared.powertools import" in source

    def test_regional_availability_has_all_decorators(self):
        """regional-availability should have all three Powertools decorators."""
        source = get_handler_source("regional-availability")
        assert "@logger.inject_lambda_context" in source
        assert "@tracer.capture_lambda_handler" in source
        assert "@metrics.log_metrics" in source


class TestFeatureCollectorPowertools:
    """Tests for feature-collector Powertools integration."""

    def test_feature_collector_imports_powertools(self):
        """feature-collector should import from shared.powertools."""
        source = get_handler_source("feature-collector")
        assert "from shared.powertools import" in source

    def test_feature_collector_has_all_decorators(self):
        """feature-collector should have all three Powertools decorators."""
        source = get_handler_source("feature-collector")
        assert "@logger.inject_lambda_context" in source
        assert "@tracer.capture_lambda_handler" in source
        assert "@metrics.log_metrics" in source


class TestTokenSpecsCollectorPowertools:
    """Tests for token-specs-collector Powertools integration."""

    def test_token_specs_collector_imports_powertools(self):
        """token-specs-collector should import from shared.powertools."""
        source = get_handler_source("token-specs-collector")
        assert "from shared.powertools import" in source

    def test_token_specs_collector_has_all_decorators(self):
        """token-specs-collector should have all three Powertools decorators."""
        source = get_handler_source("token-specs-collector")
        assert "@logger.inject_lambda_context" in source
        assert "@tracer.capture_lambda_handler" in source
        assert "@metrics.log_metrics" in source


class TestMantleCollectorPowertools:
    """Tests for mantle-collector Powertools integration."""

    def test_mantle_collector_imports_powertools(self):
        """mantle-collector should import from shared.powertools."""
        source = get_handler_source("mantle-collector")
        assert "from shared.powertools import" in source

    def test_mantle_collector_has_all_decorators(self):
        """mantle-collector should have all three Powertools decorators."""
        source = get_handler_source("mantle-collector")
        assert "@logger.inject_lambda_context" in source
        assert "@tracer.capture_lambda_handler" in source
        assert "@metrics.log_metrics" in source


class TestLifecycleCollectorPowertools:
    """Tests for lifecycle-collector Powertools integration."""

    def test_lifecycle_collector_imports_powertools(self):
        """lifecycle-collector should import from shared.powertools."""
        source = get_handler_source("lifecycle-collector")
        assert "from shared.powertools import" in source

    def test_lifecycle_collector_has_all_decorators(self):
        """lifecycle-collector should have all three Powertools decorators."""
        source = get_handler_source("lifecycle-collector")
        assert "@logger.inject_lambda_context" in source
        assert "@tracer.capture_lambda_handler" in source
        assert "@metrics.log_metrics" in source


class TestWave2HandlersNoLegacyLogging:
    """Tests to verify Wave 2 handlers don't use legacy logging."""

    @pytest.mark.parametrize("handler_name", WAVE2_HANDLERS)
    def test_no_legacy_logging(self, handler_name):
        """Wave 2 handlers should not use legacy logging.getLogger()."""
        source = get_handler_source(handler_name)
        assert "logger = logging.getLogger()" not in source, (
            f"{handler_name} should not use legacy logging"
        )


class TestWave2HandlersStructuredLogging:
    """Tests to verify Wave 2 handlers use structured logging."""

    @pytest.mark.parametrize("handler_name", WAVE2_HANDLERS)
    def test_uses_structured_logging(self, handler_name):
        """Wave 2 handlers should use structured logging with extra= parameter."""
        source = get_handler_source(handler_name)
        has_structured_logging = (
            "extra={" in source or "extra = {" in source or ", extra=" in source
        )
        assert has_structured_logging, (
            f"{handler_name} should use structured logging with extra= parameter"
        )
