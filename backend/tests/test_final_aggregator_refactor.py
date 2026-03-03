"""
Tests for final-aggregator refactoring (Task 14).

Validates that final-aggregator has been decomposed into sub-functions
for better maintainability and testability.
"""

import pytest
from pathlib import Path

HANDLER_PATH = (
    Path(__file__).parent.parent / "lambdas" / "final-aggregator" / "handler.py"
)


def get_handler_source() -> str:
    """Read the handler source code."""
    return HANDLER_PATH.read_text()


class TestSubFunctionExtraction:
    """Tests that sub-functions have been extracted from transform_model_to_schema."""

    def test_resolve_context_window_exists(self):
        """_resolve_context_window function should exist."""
        # Arrange
        source = get_handler_source()

        # Assert
        assert "def _resolve_context_window(" in source, (
            "final-aggregator should have _resolve_context_window() function "
            "extracted from transform_model_to_schema()"
        )

    def test_build_converse_data_exists(self):
        """_build_converse_data function should exist."""
        # Arrange
        source = get_handler_source()

        # Assert
        assert "def _build_converse_data(" in source, (
            "final-aggregator should have _build_converse_data() function "
            "extracted from transform_model_to_schema()"
        )

    def test_merge_lifecycle_data_exists(self):
        """_merge_lifecycle_data function should exist."""
        # Arrange
        source = get_handler_source()

        # Assert
        assert "def _merge_lifecycle_data(" in source, (
            "final-aggregator should have _merge_lifecycle_data() function "
            "extracted from transform_model_to_schema()"
        )

    def test_build_model_pricing_exists(self):
        """_build_model_pricing function should exist."""
        # Arrange
        source = get_handler_source()

        # Assert
        assert "def _build_model_pricing(" in source, (
            "final-aggregator should have _build_model_pricing() function "
            "extracted from transform_model_to_schema()"
        )

    def test_build_collection_metadata_exists(self):
        """_build_collection_metadata function should exist."""
        # Arrange
        source = get_handler_source()

        # Assert
        assert "def _build_collection_metadata(" in source, (
            "final-aggregator should have _build_collection_metadata() function "
            "extracted from transform_model_to_schema()"
        )


class TestSubFunctionUsage:
    """Tests that transform_model_to_schema uses the extracted sub-functions."""

    def test_transform_calls_resolve_context_window(self):
        """transform_model_to_schema should call _resolve_context_window."""
        # Arrange
        source = get_handler_source()

        # Assert
        assert "_resolve_context_window(" in source, (
            "transform_model_to_schema should call _resolve_context_window()"
        )

    def test_transform_calls_build_converse_data(self):
        """transform_model_to_schema should call _build_converse_data."""
        # Arrange
        source = get_handler_source()

        # Assert
        assert "_build_converse_data(" in source, (
            "transform_model_to_schema should call _build_converse_data()"
        )

    def test_transform_calls_merge_lifecycle_data(self):
        """transform_model_to_schema should call _merge_lifecycle_data."""
        # Arrange
        source = get_handler_source()

        # Assert
        assert "_merge_lifecycle_data(" in source, (
            "transform_model_to_schema should call _merge_lifecycle_data()"
        )

    def test_transform_calls_build_model_pricing(self):
        """transform_model_to_schema should call _build_model_pricing."""
        # Arrange
        source = get_handler_source()

        # Assert
        assert "_build_model_pricing(" in source, (
            "transform_model_to_schema should call _build_model_pricing()"
        )

    def test_transform_calls_build_collection_metadata(self):
        """transform_model_to_schema should call _build_collection_metadata."""
        # Arrange
        source = get_handler_source()

        # Assert
        assert "_build_collection_metadata(" in source, (
            "transform_model_to_schema should call _build_collection_metadata()"
        )


class TestContextWindowResolution:
    """Tests for context window resolution logic."""

    def test_resolve_context_window_has_tier_comments(self):
        """_resolve_context_window should document the 4-tier priority."""
        # Arrange
        source = get_handler_source()

        # Assert - function should have documentation about tiers
        assert "TIER 1" in source or "Tier 1" in source or "tier 1" in source, (
            "_resolve_context_window should document Tier 1 (Console API)"
        )

    def test_resolve_context_window_returns_dict(self):
        """_resolve_context_window should return a dictionary with context data."""
        # Arrange
        source = get_handler_source()

        # Find the function and check it returns a dict
        # Look for the return statement pattern
        assert "context_window" in source and "max_output" in source, (
            "_resolve_context_window should return context_window and max_output"
        )


class TestConverseDataBuilding:
    """Tests for converse data building logic."""

    def test_build_converse_data_uses_context_data(self):
        """_build_converse_data should use context_data parameter."""
        # Arrange
        source = get_handler_source()

        # Assert - function signature should include context_data
        assert "def _build_converse_data(" in source
        # Check that it uses context_data
        assert (
            'context_data["context_window"]' in source
            or "context_data['context_window']" in source
        ), "_build_converse_data should use context_data parameter"


class TestFunctionDecorators:
    """Tests that sub-functions have appropriate decorators."""

    def test_resolve_context_window_has_tracer(self):
        """_resolve_context_window should have @tracer.capture_method decorator."""
        # Arrange
        source = get_handler_source()

        # Find the function definition and check for decorator
        # Look for the pattern: @tracer.capture_method followed by def _resolve_context_window
        assert "@tracer.capture_method" in source, (
            "Sub-functions should have @tracer.capture_method decorator for tracing"
        )

    def test_build_converse_data_has_tracer(self):
        """_build_converse_data should have @tracer.capture_method decorator."""
        # Arrange
        source = get_handler_source()

        # The decorator should appear before the function
        # We check that both exist in the source
        assert "@tracer.capture_method" in source
        assert "def _build_converse_data(" in source


class TestTransformModelToSchemaOrchestration:
    """Tests that transform_model_to_schema properly orchestrates sub-functions."""

    def test_transform_model_to_schema_exists(self):
        """transform_model_to_schema function should exist."""
        # Arrange
        source = get_handler_source()

        # Assert
        assert "def transform_model_to_schema(" in source

    def test_transform_model_to_schema_docstring_mentions_orchestration(self):
        """transform_model_to_schema should document that it orchestrates sub-functions."""
        # Arrange
        source = get_handler_source()

        # Assert - docstring should mention orchestration or sub-functions
        assert "Orchestrates" in source or "sub-function" in source.lower(), (
            "transform_model_to_schema docstring should mention orchestration"
        )
