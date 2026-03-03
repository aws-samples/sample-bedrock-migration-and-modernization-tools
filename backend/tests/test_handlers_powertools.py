"""Tests for Powertools integration across all handlers.

This module verifies that all 17 Lambda handlers (excluding self-healing-agent)
have been properly migrated to use AWS Lambda Powertools for:
- Structured logging via @logger.inject_lambda_context
- Distributed tracing via @tracer.capture_lambda_handler
- Custom metrics via @metrics.log_metrics

Tests use source code analysis (static analysis) rather than runtime checks,
avoiding the need to mock AWS services for decorator verification.
"""

import pytest
from pathlib import Path

# List of all handlers that should have Powertools (excluding self-healing-agent)
HANDLERS = [
    "pricing-collector",
    "pricing-aggregator",
    "model-extractor",
    "model-merger",
    "quota-collector",
    "pricing-linker",
    "regional-availability",
    "feature-collector",
    "token-specs-collector",
    "mantle-collector",
    "lifecycle-collector",
    "final-aggregator",
    "copy-to-latest",
    "gap-detection",
    "region-discovery",
    "cognito-sync",
    "analytics",
]

HANDLERS_PATH = Path(__file__).parent.parent / "lambdas"


def get_handler_source(handler_name: str) -> str:
    """Read the handler source code."""
    handler_path = HANDLERS_PATH / handler_name / "handler.py"
    return handler_path.read_text()


class TestPowertoolsImports:
    """Tests for Powertools import statements."""

    @pytest.mark.parametrize("handler_name", HANDLERS)
    def test_handler_imports_powertools(self, handler_name):
        """Handler should import from shared.powertools."""
        source = get_handler_source(handler_name)
        assert "from shared.powertools import" in source, (
            f"{handler_name} should import from shared.powertools"
        )


class TestLegacyLoggingRemoved:
    """Tests to verify legacy logging patterns have been removed."""

    @pytest.mark.parametrize("handler_name", HANDLERS)
    def test_handler_no_legacy_logging_getlogger(self, handler_name):
        """Handler should not use legacy logging.getLogger()."""
        source = get_handler_source(handler_name)
        # Check for common legacy logging patterns
        assert "logger = logging.getLogger()" not in source, (
            f"{handler_name} should not use logging.getLogger()"
        )

    @pytest.mark.parametrize("handler_name", HANDLERS)
    def test_handler_no_legacy_logging_setup(self, handler_name):
        """Handler should not have legacy logging setup with setLevel."""
        source = get_handler_source(handler_name)
        # Check for legacy pattern: logger.setLevel(...)
        assert "logger.setLevel(" not in source, (
            f"{handler_name} should not use logger.setLevel() (legacy pattern)"
        )


class TestPowertoolsDecorators:
    """Tests for Powertools decorator presence."""

    @pytest.mark.parametrize("handler_name", HANDLERS)
    def test_handler_has_inject_lambda_context(self, handler_name):
        """Handler should use @logger.inject_lambda_context decorator."""
        source = get_handler_source(handler_name)
        assert "@logger.inject_lambda_context" in source, (
            f"{handler_name} should use @logger.inject_lambda_context"
        )

    @pytest.mark.parametrize("handler_name", HANDLERS)
    def test_handler_has_capture_lambda_handler(self, handler_name):
        """Handler should use @tracer.capture_lambda_handler decorator."""
        source = get_handler_source(handler_name)
        assert "@tracer.capture_lambda_handler" in source, (
            f"{handler_name} should use @tracer.capture_lambda_handler"
        )

    @pytest.mark.parametrize("handler_name", HANDLERS)
    def test_handler_has_log_metrics(self, handler_name):
        """Handler should use @metrics.log_metrics decorator."""
        source = get_handler_source(handler_name)
        assert "@metrics.log_metrics" in source, (
            f"{handler_name} should use @metrics.log_metrics"
        )


class TestStructuredLogging:
    """Tests for structured logging usage."""

    @pytest.mark.parametrize("handler_name", HANDLERS)
    def test_handler_uses_structured_logging(self, handler_name):
        """Handler should use structured logging with extra= parameter."""
        source = get_handler_source(handler_name)
        # Check for structured logging pattern: logger.info/warning/error with extra=
        has_structured_logging = (
            "extra={" in source or "extra = {" in source or ", extra=" in source
        )
        assert has_structured_logging, (
            f"{handler_name} should use structured logging with extra= parameter"
        )


class TestSelfHealingAgentExclusion:
    """Tests to verify self-healing-agent is NOT modified (out of scope)."""

    def test_self_healing_agent_not_modified(self):
        """self-healing-agent should NOT have Powertools (out of scope)."""
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


class TestHandlerCount:
    """Tests to verify the expected number of handlers."""

    def test_expected_handler_count(self):
        """Should have exactly 17 handlers with Powertools."""
        handlers_with_powertools = []
        for handler_dir in HANDLERS_PATH.iterdir():
            if handler_dir.is_dir():
                handler_file = handler_dir / "handler.py"
                if handler_file.exists():
                    source = handler_file.read_text()
                    if "from shared.powertools import" in source:
                        handlers_with_powertools.append(handler_dir.name)

        assert len(handlers_with_powertools) == 17, (
            f"Expected 17 handlers with Powertools, found {len(handlers_with_powertools)}: "
            f"{handlers_with_powertools}"
        )

    def test_all_expected_handlers_exist(self):
        """All expected handlers should exist."""
        for handler_name in HANDLERS:
            handler_path = HANDLERS_PATH / handler_name / "handler.py"
            assert handler_path.exists(), f"Handler {handler_name} should exist"
