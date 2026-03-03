"""Tests for Utility handler Powertools integration.

Utility handlers (Task 09):
- region-discovery
- cognito-sync
- analytics

These tests verify that Utility handlers have been properly migrated
to use AWS Lambda Powertools decorators.

Note: self-healing-agent is explicitly OUT OF SCOPE and should NOT
have Powertools integration.
"""

import pytest
from pathlib import Path

HANDLERS_PATH = Path(__file__).parent.parent / "lambdas"

UTILITY_HANDLERS = [
    "region-discovery",
    "cognito-sync",
    "analytics",
]


def get_handler_source(handler_name: str) -> str:
    """Read the handler source code."""
    handler_path = HANDLERS_PATH / handler_name / "handler.py"
    return handler_path.read_text()


class TestRegionDiscoveryPowertools:
    """Tests for region-discovery Powertools integration."""

    def test_region_discovery_imports_powertools(self):
        """region-discovery should import from shared.powertools."""
        source = get_handler_source("region-discovery")
        assert "from shared.powertools import" in source

    def test_region_discovery_has_all_decorators(self):
        """region-discovery should have all three Powertools decorators."""
        source = get_handler_source("region-discovery")
        assert "@logger.inject_lambda_context" in source
        assert "@tracer.capture_lambda_handler" in source
        assert "@metrics.log_metrics" in source


class TestCognitoSyncPowertools:
    """Tests for cognito-sync Powertools integration."""

    def test_cognito_sync_imports_powertools(self):
        """cognito-sync should import from shared.powertools."""
        source = get_handler_source("cognito-sync")
        assert "from shared.powertools import" in source

    def test_cognito_sync_has_all_decorators(self):
        """cognito-sync should have all three Powertools decorators."""
        source = get_handler_source("cognito-sync")
        assert "@logger.inject_lambda_context" in source
        assert "@tracer.capture_lambda_handler" in source
        assert "@metrics.log_metrics" in source


class TestAnalyticsPowertools:
    """Tests for analytics Powertools integration."""

    def test_analytics_imports_powertools(self):
        """analytics should import from shared.powertools."""
        source = get_handler_source("analytics")
        assert "from shared.powertools import" in source

    def test_analytics_has_all_decorators(self):
        """analytics should have all three Powertools decorators."""
        source = get_handler_source("analytics")
        assert "@logger.inject_lambda_context" in source
        assert "@tracer.capture_lambda_handler" in source
        assert "@metrics.log_metrics" in source


class TestSelfHealingAgentNotModified:
    """Tests to verify self-healing-agent is NOT modified (out of scope)."""

    def test_self_healing_agent_no_powertools_import(self):
        """self-healing-agent should NOT import from shared.powertools."""
        source = get_handler_source("self-healing-agent")
        assert "from shared.powertools import" not in source, (
            "self-healing-agent should not be modified (out of scope)"
        )

    def test_self_healing_agent_uses_legacy_logging(self):
        """self-healing-agent should still use legacy logging."""
        source = get_handler_source("self-healing-agent")
        assert "logger = logging.getLogger()" in source, (
            "self-healing-agent should retain legacy logging"
        )

    def test_self_healing_agent_no_powertools_decorators(self):
        """self-healing-agent should NOT have Powertools decorators."""
        source = get_handler_source("self-healing-agent")
        assert "@logger.inject_lambda_context" not in source
        assert "@tracer.capture_lambda_handler" not in source
        assert "@metrics.log_metrics" not in source


class TestUtilityHandlersNoLegacyLogging:
    """Tests to verify Utility handlers don't use legacy logging."""

    @pytest.mark.parametrize("handler_name", UTILITY_HANDLERS)
    def test_no_legacy_logging(self, handler_name):
        """Utility handlers should not use legacy logging.getLogger()."""
        source = get_handler_source(handler_name)
        assert "logger = logging.getLogger()" not in source, (
            f"{handler_name} should not use legacy logging"
        )


class TestUtilityHandlersStructuredLogging:
    """Tests to verify Utility handlers use structured logging."""

    @pytest.mark.parametrize("handler_name", UTILITY_HANDLERS)
    def test_uses_structured_logging(self, handler_name):
        """Utility handlers should use structured logging with extra= parameter."""
        source = get_handler_source(handler_name)
        has_structured_logging = (
            "extra={" in source or "extra = {" in source or ", extra=" in source
        )
        assert has_structured_logging, (
            f"{handler_name} should use structured logging with extra= parameter"
        )
