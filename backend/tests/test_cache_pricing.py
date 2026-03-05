"""Tests for Cache pricing detection (Task 07).

Tests for Phase 2 - Cache Write 1H Detection:
- detect_cache_type()
- determine_pricing_group_with_dimensions() for cache types
- aggregate_dimensions() for cache_types aggregation

These tests verify that cache pricing patterns (cache_read, cache_write,
cache_write_1h) are correctly detected from usage types and descriptions.
"""

from __future__ import annotations

import pytest
import re
from pathlib import Path

HANDLERS_PATH = Path(__file__).parent.parent / "lambdas"


def get_handler_source(handler_name: str) -> str:
    """Read the handler source code."""
    handler_path = HANDLERS_PATH / handler_name / "handler.py"
    return handler_path.read_text()


def _make_python39_compatible(source: str) -> str:
    """Convert Python 3.10+ type hints to Python 3.9 compatible syntax."""
    import re as re_module

    # Replace union type hints in function signatures
    source = re_module.sub(
        r"\) -> ([a-zA-Z]+) \| ([a-zA-Z]+):", r') -> "\1 | \2":', source
    )
    source = re_module.sub(r"\) -> tuple\[([^\]]+)\]:", r') -> "tuple[\1]":', source)

    return source


def _load_cache_detection_functions():
    """Load cache detection functions by extracting them from source.

    Since the handler has dependencies on shared modules that aren't available
    in the test environment, we extract and exec just the functions we need.
    """
    source = get_handler_source("pricing-aggregator")
    source = _make_python39_compatible(source)
    lines = source.split("\n")

    def extract_function(func_name: str) -> str:
        """Extract a function definition from source lines."""
        in_function = False
        func_lines = []
        base_indent = 0

        for line in lines:
            if f"def {func_name}(" in line:
                in_function = True
                base_indent = len(line) - len(line.lstrip())
                func_lines.append(line)
            elif in_function:
                # Check if we've exited the function (new top-level def or class)
                stripped = line.lstrip()
                current_indent = len(line) - len(line.lstrip()) if stripped else -1

                if stripped and current_indent >= 0 and current_indent <= base_indent:
                    if (
                        stripped.startswith("def ")
                        or stripped.startswith("class ")
                        or stripped.startswith("@")
                    ):
                        break
                func_lines.append(line)

        return "\n".join(func_lines)

    def extract_constant(const_name: str) -> str:
        """Extract a constant definition from source lines."""
        in_constant = False
        const_lines = []
        bracket_depth = 0

        for line in lines:
            if line.startswith(f"{const_name} = "):
                in_constant = True
                const_lines.append(line)
                # Count brackets to handle multi-line definitions
                bracket_depth += line.count("[") + line.count("{") + line.count("(")
                bracket_depth -= line.count("]") + line.count("}") + line.count(")")
                if bracket_depth <= 0:
                    break
            elif in_constant:
                const_lines.append(line)
                bracket_depth += line.count("[") + line.count("{") + line.count("(")
                bracket_depth -= line.count("]") + line.count("}") + line.count(")")
                if bracket_depth <= 0:
                    break

        return "\n".join(const_lines)

    # Extract constants needed by functions
    mantle_patterns_source = extract_constant("MANTLE_PATTERNS")
    cris_regional_patterns_source = extract_constant("CRIS_REGIONAL_PATTERNS")
    reserved_patterns_source = extract_constant("RESERVED_PATTERNS")
    commitment_patterns_source = extract_constant("COMMITMENT_PATTERNS")
    cache_patterns_source = extract_constant("CACHE_PATTERNS")

    # Extract functions
    detect_mantle_source = extract_function("detect_mantle_pricing")
    detect_cris_regional_source = extract_function("detect_cris_regional")
    detect_reserved_source = extract_function("detect_reserved_pricing")
    detect_cache_source = extract_function("detect_cache_type")
    determine_group_source = extract_function("determine_pricing_group_with_dimensions")
    aggregate_dims_source = extract_function("aggregate_dimensions")

    if not detect_cache_source:
        raise RuntimeError("Could not find detect_cache_type")
    if not determine_group_source:
        raise RuntimeError("Could not find determine_pricing_group_with_dimensions")
    if not aggregate_dims_source:
        raise RuntimeError("Could not find aggregate_dimensions")

    # Create a namespace with re module (needed for patterns)
    namespace = {"re": re}

    # Execute in order: constants first, then helper functions, then main functions
    if mantle_patterns_source:
        exec(mantle_patterns_source, namespace)
    if cris_regional_patterns_source:
        exec(cris_regional_patterns_source, namespace)
    if reserved_patterns_source:
        exec(reserved_patterns_source, namespace)
    if commitment_patterns_source:
        exec(commitment_patterns_source, namespace)
    if cache_patterns_source:
        exec(cache_patterns_source, namespace)

    # Execute helper functions
    if detect_mantle_source:
        exec(detect_mantle_source, namespace)
    if detect_cris_regional_source:
        exec(detect_cris_regional_source, namespace)
    if detect_reserved_source:
        exec(detect_reserved_source, namespace)
    if detect_cache_source:
        exec(detect_cache_source, namespace)

    # Execute main functions
    exec(determine_group_source, namespace)
    exec(aggregate_dims_source, namespace)

    return namespace


# Load functions once at module level
try:
    _funcs = _load_cache_detection_functions()
    detect_cache_type = _funcs["detect_cache_type"]
    determine_pricing_group_with_dimensions = _funcs[
        "determine_pricing_group_with_dimensions"
    ]
    aggregate_dimensions = _funcs["aggregate_dimensions"]
except Exception as e:
    # If loading fails, tests will be skipped
    detect_cache_type = None
    determine_pricing_group_with_dimensions = None
    aggregate_dimensions = None
    _load_error = str(e)


# =============================================================================
# Tests for Cache Type Detection
# =============================================================================


@pytest.mark.skipif(
    detect_cache_type is None,
    reason="Could not load detect_cache_type function",
)
class TestCachePricingDetection:
    """Tests for cache pricing detection."""

    def test_detect_cache_write_1h_usage(self):
        """detect_cache_type returns cache_write_1h for CacheWrite1H pattern."""
        # Arrange
        usage_type = "USE1-CacheWrite1H-InputTokens"
        description = ""

        # Act
        result = detect_cache_type(usage_type, description)

        # Assert
        assert result == "cache_write_1h"

    def test_detect_cache_write_1h_description(self):
        """detect_cache_type returns cache_write_1h for 1-hour-cache in description."""
        # Arrange
        usage_type = ""
        description = "1-hour-cache storage"

        # Act
        result = detect_cache_type(usage_type, description)

        # Assert
        assert result == "cache_write_1h"

    def test_detect_cache_write_standard(self):
        """detect_cache_type returns cache_write for standard CacheWrite pattern."""
        # Arrange
        usage_type = "USE1-CacheWrite-InputTokens"
        description = ""

        # Act
        result = detect_cache_type(usage_type, description)

        # Assert
        assert result == "cache_write"

    def test_detect_cache_read(self):
        """detect_cache_type returns cache_read for CacheRead pattern."""
        # Arrange
        usage_type = "USE1-CacheRead-InputTokens"
        description = ""

        # Act
        result = detect_cache_type(usage_type, description)

        # Assert
        assert result == "cache_read"

    def test_detect_non_cache(self):
        """detect_cache_type returns None for non-cache patterns."""
        # Arrange
        usage_type = "USE1-OnDemand-InputTokens"
        description = ""

        # Act
        result = detect_cache_type(usage_type, description)

        # Assert
        assert result is None

    def test_cache_1h_priority_over_standard(self):
        """cache_write_1h is detected before cache_write (priority check)."""
        # Arrange - pattern that contains both "cache-write" and "1h"
        usage_type = "cache-write-1h"
        description = ""

        # Act
        result = detect_cache_type(usage_type, description)

        # Assert - should be cache_write_1h, not cache_write
        assert result == "cache_write_1h"

    def test_detect_cache_storage_description(self):
        """detect_cache_type returns cache_write for cache-storage in description."""
        # Arrange
        usage_type = ""
        description = "cache-storage pricing"

        # Act
        result = detect_cache_type(usage_type, description)

        # Assert
        assert result == "cache_write"

    def test_detect_cached_input_description(self):
        """detect_cache_type returns cache_read for cached-input in description."""
        # Arrange
        usage_type = ""
        description = "cached-input tokens"

        # Act
        result = detect_cache_type(usage_type, description)

        # Assert
        assert result == "cache_read"

    def test_detect_cache_type_case_insensitive(self):
        """detect_cache_type is case-insensitive."""
        # Arrange - various case combinations
        test_cases = [
            ("USE1-CACHEWRITE1H-InputTokens", "", "cache_write_1h"),
            ("USE1-cachewrite1h-inputtokens", "", "cache_write_1h"),
            ("USE1-CACHEWRITE-InputTokens", "", "cache_write"),
            ("USE1-CACHEREAD-InputTokens", "", "cache_read"),
            ("", "1-HOUR-CACHE storage", "cache_write_1h"),
        ]

        # Act & Assert
        for usage_type, description, expected in test_cases:
            result = detect_cache_type(usage_type, description)
            assert result == expected, (
                f"Failed for usage_type={usage_type}, description={description}"
            )

    def test_detect_cache_underscore_separator(self):
        """detect_cache_type handles underscore separators."""
        # Arrange
        test_cases = [
            ("USE1_CacheWrite_1H_InputTokens", "", "cache_write_1h"),
            ("USE1_CacheWrite_InputTokens", "", "cache_write"),
            ("USE1_CacheRead_InputTokens", "", "cache_read"),
        ]

        # Act & Assert
        for usage_type, description, expected in test_cases:
            result = detect_cache_type(usage_type, description)
            assert result == expected, f"Failed for usage_type={usage_type}"


# =============================================================================
# Tests for Cache Dimensions
# =============================================================================


@pytest.mark.skipif(
    determine_pricing_group_with_dimensions is None,
    reason="Could not load determine_pricing_group_with_dimensions function",
)
class TestCacheDimensions:
    """Tests for cache dimension extraction."""

    def test_dimension_cache_type_write_1h(self):
        """dimensions include cache_type: cache_write_1h."""
        # Arrange
        usage_type = "USE1-CacheWrite1H-InputTokens"
        inference_type = ""
        description = ""

        # Act
        result = determine_pricing_group_with_dimensions(
            usage_type, inference_type, description
        )

        # Assert
        assert result["dimensions"]["cache_type"] == "cache_write_1h"

    def test_dimension_cache_type_write(self):
        """dimensions include cache_type: cache_write."""
        # Arrange
        usage_type = "USE1-CacheWrite-InputTokens"
        inference_type = ""
        description = ""

        # Act
        result = determine_pricing_group_with_dimensions(
            usage_type, inference_type, description
        )

        # Assert
        assert result["dimensions"]["cache_type"] == "cache_write"

    def test_dimension_cache_type_read(self):
        """dimensions include cache_type: cache_read."""
        # Arrange
        usage_type = "USE1-CacheRead-InputTokens"
        inference_type = ""
        description = ""

        # Act
        result = determine_pricing_group_with_dimensions(
            usage_type, inference_type, description
        )

        # Assert
        assert result["dimensions"]["cache_type"] == "cache_read"

    def test_dimension_cache_type_none_default(self):
        """dimensions default to cache_type: None for non-cache."""
        # Arrange
        usage_type = "USE1-OnDemand-InputTokens"
        inference_type = ""
        description = ""

        # Act
        result = determine_pricing_group_with_dimensions(
            usage_type, inference_type, description
        )

        # Assert
        assert result["dimensions"]["cache_type"] is None

    def test_cache_with_batch_mode(self):
        """cache_type is detected alongside batch inference mode."""
        # Arrange
        usage_type = "USE1-batch-CacheRead-InputTokens"
        inference_type = ""
        description = ""

        # Act
        result = determine_pricing_group_with_dimensions(
            usage_type, inference_type, description
        )

        # Assert
        assert result["dimensions"]["cache_type"] == "cache_read"
        assert result["group"] == "Batch"


# =============================================================================
# Tests for Cache Types Aggregation
# =============================================================================


@pytest.mark.skipif(
    aggregate_dimensions is None,
    reason="Could not load aggregate_dimensions function",
)
class TestCacheTypesAggregation:
    """Tests for cache_types aggregation in aggregate_dimensions."""

    def test_aggregate_dimensions_cache_types(self):
        """aggregate_dimensions includes cache_types list."""
        # Arrange
        entries = [
            {"dimensions": {"cache_type": "cache_read"}},
            {"dimensions": {"cache_type": "cache_write"}},
            {"dimensions": {"cache_type": "cache_write_1h"}},
        ]

        # Act
        result = aggregate_dimensions(entries)

        # Assert
        assert "cache_types" in result
        assert set(result["cache_types"]) == {
            "cache_read",
            "cache_write",
            "cache_write_1h",
        }

    def test_aggregate_dimensions_cache_types_sorted(self):
        """aggregate_dimensions returns sorted cache_types."""
        # Arrange
        entries = [
            {"dimensions": {"cache_type": "cache_write_1h"}},
            {"dimensions": {"cache_type": "cache_read"}},
            {"dimensions": {"cache_type": "cache_write"}},
        ]

        # Act
        result = aggregate_dimensions(entries)

        # Assert
        assert result["cache_types"] == sorted(result["cache_types"])

    def test_aggregate_dimensions_cache_types_empty(self):
        """aggregate_dimensions returns empty cache_types when no cache entries."""
        # Arrange
        entries = [
            {"dimensions": {"cache_type": None}},
            {"dimensions": {}},
        ]

        # Act
        result = aggregate_dimensions(entries)

        # Assert
        assert result["cache_types"] == []

    def test_aggregate_dimensions_cache_types_deduplication(self):
        """aggregate_dimensions deduplicates cache_types."""
        # Arrange
        entries = [
            {"dimensions": {"cache_type": "cache_read"}},
            {"dimensions": {"cache_type": "cache_read"}},
            {"dimensions": {"cache_type": "cache_write"}},
            {"dimensions": {"cache_type": "cache_write"}},
        ]

        # Act
        result = aggregate_dimensions(entries)

        # Assert
        assert len(result["cache_types"]) == 2
        assert set(result["cache_types"]) == {"cache_read", "cache_write"}

    def test_aggregate_dimensions_mixed_entries(self):
        """aggregate_dimensions handles mixed cache and non-cache entries."""
        # Arrange
        entries = [
            {"dimensions": {"cache_type": "cache_read", "inference_mode": "on_demand"}},
            {"dimensions": {"cache_type": None, "inference_mode": "batch"}},
            {
                "dimensions": {
                    "cache_type": "cache_write_1h",
                    "inference_mode": "on_demand",
                }
            },
        ]

        # Act
        result = aggregate_dimensions(entries)

        # Assert
        assert set(result["cache_types"]) == {"cache_read", "cache_write_1h"}
        assert "on_demand" in result["inference_modes"]
        assert "batch" in result["inference_modes"]
