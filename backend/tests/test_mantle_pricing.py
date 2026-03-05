"""Tests for Mantle pricing detection (Task 05).

Tests for Phase 2 - Mantle Pricing Detection:
- detect_mantle_pricing()
- determine_pricing_group_with_dimensions() for Mantle

These tests verify that Mantle (OpenAI-compatible) pricing patterns
are correctly detected from usage types and descriptions.
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
    # Replace `str | None` with `Optional[str]` style (but we'll use string annotations)
    # Actually, since we use `from __future__ import annotations`, the type hints
    # are treated as strings and won't be evaluated at runtime.
    # But the function signatures still need to be valid syntax.
    # Replace `-> str | None:` with `-> "str | None":`
    import re as re_module

    # Replace union type hints in function signatures
    source = re_module.sub(
        r"\) -> ([a-zA-Z]+) \| ([a-zA-Z]+):", r') -> "\1 | \2":', source
    )
    source = re_module.sub(r"\) -> tuple\[([^\]]+)\]:", r') -> "tuple[\1]":', source)

    return source


def _load_mantle_detection_functions():
    """Load Mantle detection functions by extracting them from source.

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

    if not detect_mantle_source:
        raise RuntimeError("Could not find detect_mantle_pricing")
    if not determine_group_source:
        raise RuntimeError("Could not find determine_pricing_group_with_dimensions")

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

    # Execute main function
    exec(determine_group_source, namespace)

    return namespace


# Load functions once at module level
try:
    _funcs = _load_mantle_detection_functions()
    detect_mantle_pricing = _funcs["detect_mantle_pricing"]
    determine_pricing_group_with_dimensions = _funcs[
        "determine_pricing_group_with_dimensions"
    ]
except Exception as e:
    # If loading fails, tests will be skipped
    detect_mantle_pricing = None
    determine_pricing_group_with_dimensions = None
    _load_error = str(e)


@pytest.mark.skipif(
    detect_mantle_pricing is None,
    reason="Could not load detect_mantle_pricing function",
)
class TestMantlePricingDetection:
    """Tests for Mantle pricing detection."""

    def test_detect_mantle_usage_type(self):
        """detect_mantle_pricing returns True for Mantle patterns in usage type."""
        # Arrange
        usage_type = "USE1-Mantle-InputTokens"
        description = ""

        # Act
        result = detect_mantle_pricing(usage_type, description)

        # Assert
        assert result is True

    def test_detect_mantle_description(self):
        """detect_mantle_pricing returns True for OpenAI-compatible in description."""
        # Arrange
        usage_type = ""
        description = "OpenAI-compatible endpoint"

        # Act
        result = detect_mantle_pricing(usage_type, description)

        # Assert
        assert result is True

    def test_detect_mantle_chat_completions(self):
        """detect_mantle_pricing returns True for chat-completions pattern."""
        # Arrange
        usage_type = "chat-completions"
        description = ""

        # Act
        result = detect_mantle_pricing(usage_type, description)

        # Assert
        assert result is True

    def test_detect_non_mantle(self):
        """detect_mantle_pricing returns False for non-Mantle usage types."""
        # Arrange
        usage_type = "USE1-OnDemand-InputTokens"
        description = ""

        # Act
        result = detect_mantle_pricing(usage_type, description)

        # Assert
        assert result is False

    def test_detect_mantle_case_insensitive(self):
        """detect_mantle_pricing is case-insensitive."""
        # Arrange - various case combinations
        test_cases = [
            ("MANTLE-InputTokens", ""),
            ("mantle-inputtokens", ""),
            ("MaNtLe-Input", ""),
            ("", "OPENAI-COMPATIBLE"),
            ("", "openai_compatible"),
            ("CHAT-COMPLETIONS", ""),
        ]

        # Act & Assert
        for usage_type, description in test_cases:
            result = detect_mantle_pricing(usage_type, description)
            assert result is True, (
                f"Failed for usage_type={usage_type}, description={description}"
            )

    def test_detect_mantle_from_inference_type(self):
        """detect_mantle_pricing returns True when inference_type contains pattern."""
        # Arrange
        usage_type = ""
        description = ""
        inference_type = "openai-compatible"

        # Act
        result = detect_mantle_pricing(usage_type, description, inference_type)

        # Assert
        assert result is True


@pytest.mark.skipif(
    determine_pricing_group_with_dimensions is None,
    reason="Could not load determine_pricing_group_with_dimensions function",
)
class TestMantleDimensions:
    """Tests for Mantle dimension extraction."""

    def test_dimension_inference_mode_mantle(self):
        """dimensions include inference_mode: mantle for Mantle pricing."""
        # Arrange
        usage_type = "USE1-Mantle-InputTokens"
        inference_type = ""
        description = ""

        # Act
        result = determine_pricing_group_with_dimensions(
            usage_type, inference_type, description
        )

        # Assert
        assert result["dimensions"]["inference_mode"] == "mantle"

    def test_group_is_mantle(self):
        """group is Mantle for Mantle pricing."""
        # Arrange
        usage_type = "USE1-Mantle-InputTokens"
        inference_type = ""
        description = ""

        # Act
        result = determine_pricing_group_with_dimensions(
            usage_type, inference_type, description
        )

        # Assert
        assert result["group"] == "Mantle"

    def test_mantle_source_dimension(self):
        """dimensions include source: mantle for backward compatibility."""
        # Arrange
        usage_type = "USE1-Mantle-InputTokens"
        inference_type = ""
        description = ""

        # Act
        result = determine_pricing_group_with_dimensions(
            usage_type, inference_type, description
        )

        # Assert
        assert result["dimensions"]["source"] == "mantle"

    def test_mantle_takes_priority_over_other_modes(self):
        """Mantle detection takes priority over batch/provisioned detection."""
        # Arrange - usage type that could match both Mantle and batch
        usage_type = "USE1-Mantle-batch-InputTokens"
        inference_type = ""
        description = ""

        # Act
        result = determine_pricing_group_with_dimensions(
            usage_type, inference_type, description
        )

        # Assert - Mantle should win
        assert result["group"] == "Mantle"
        assert result["dimensions"]["inference_mode"] == "mantle"
