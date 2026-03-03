"""
Tests for config pattern consolidation (Task 12).

Validates that handlers no longer have _get_config() function and instead
use the shared get_config_loader() from the shared layer.
"""

import pytest
from pathlib import Path

HANDLERS_PATH = Path(__file__).parent.parent / "lambdas"

# Handlers that should have been consolidated
HANDLERS_TO_CHECK = [
    "pricing-aggregator",
    "model-extractor",
    "pricing-linker",
    "gap-detection",
    "final-aggregator",
]


def get_handler_source(handler_name: str) -> str:
    """Read the handler source code."""
    handler_path = HANDLERS_PATH / handler_name / "handler.py"
    return handler_path.read_text()


class TestNoGetConfigFunction:
    """Tests that _get_config() function has been removed from handlers."""

    @pytest.mark.parametrize("handler_name", HANDLERS_TO_CHECK)
    def test_no_get_config_function(self, handler_name):
        """Handler should not have _get_config() function definition."""
        # Arrange
        source = get_handler_source(handler_name)

        # Assert
        assert "def _get_config(" not in source, (
            f"{handler_name} should not have _get_config() function. "
            "Use get_config_loader() from shared layer instead."
        )


class TestNoConfigLoaderGlobal:
    """Tests that _config_loader global variable has been removed."""

    @pytest.mark.parametrize("handler_name", HANDLERS_TO_CHECK)
    def test_no_config_loader_global(self, handler_name):
        """Handler should not have _config_loader global variable."""
        # Arrange
        source = get_handler_source(handler_name)

        # Assert
        assert "_config_loader = None" not in source, (
            f"{handler_name} should not have _config_loader global variable. "
            "Use get_config_loader() from shared layer instead."
        )


class TestUsesGetConfigLoader:
    """Tests that handlers use get_config_loader from shared layer."""

    @pytest.mark.parametrize("handler_name", HANDLERS_TO_CHECK)
    def test_handlers_import_get_config_loader(self, handler_name):
        """Handler should import get_config_loader from shared."""
        # Arrange
        source = get_handler_source(handler_name)

        # Assert
        assert "get_config_loader" in source, (
            f"{handler_name} should use get_config_loader from shared layer"
        )

    @pytest.mark.parametrize("handler_name", HANDLERS_TO_CHECK)
    def test_handlers_import_from_shared(self, handler_name):
        """Handler should import from shared module."""
        # Arrange
        source = get_handler_source(handler_name)

        # Assert
        assert "from shared import" in source, (
            f"{handler_name} should import from shared module"
        )


class TestConfigLoaderUsage:
    """Tests for proper get_config_loader() usage patterns."""

    def test_pricing_aggregator_uses_config_loader(self):
        """Pricing aggregator should call get_config_loader() for config access."""
        # Arrange
        source = get_handler_source("pricing-aggregator")

        # Assert - should use get_config_loader() to get config
        assert "get_config_loader()" in source

    def test_model_extractor_uses_config_loader(self):
        """Model extractor should call get_config_loader() for config access."""
        # Arrange
        source = get_handler_source("model-extractor")

        # Assert - should use get_config_loader() to get config
        assert "get_config_loader()" in source

    def test_final_aggregator_uses_config_loader(self):
        """Final aggregator should call get_config_loader() for config access."""
        # Arrange
        source = get_handler_source("final-aggregator")

        # Assert - should use get_config_loader() to get config
        assert "get_config_loader()" in source
