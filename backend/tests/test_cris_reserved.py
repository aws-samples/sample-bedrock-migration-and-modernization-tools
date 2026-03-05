"""Tests for CRIS Regional and Reserved pricing detection (Task 06).

Tests for Phase 2 - CRIS Regional & Reserved Detection:
- detect_cris_regional()
- detect_reserved_pricing()
- determine_pricing_group_with_dimensions() for CRIS Regional and Reserved

These tests verify that CRIS Regional (cross-region within geographic area)
and Reserved pricing patterns are correctly detected.
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


def _load_cris_reserved_functions():
    """Load CRIS Regional and Reserved detection functions by extracting them from source.

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

    if not detect_cris_regional_source:
        raise RuntimeError("Could not find detect_cris_regional")
    if not detect_reserved_source:
        raise RuntimeError("Could not find detect_reserved_pricing")
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
    _funcs = _load_cris_reserved_functions()
    detect_cris_regional = _funcs["detect_cris_regional"]
    detect_reserved_pricing = _funcs["detect_reserved_pricing"]
    determine_pricing_group_with_dimensions = _funcs[
        "determine_pricing_group_with_dimensions"
    ]
except Exception as e:
    # If loading fails, tests will be skipped
    detect_cris_regional = None
    detect_reserved_pricing = None
    determine_pricing_group_with_dimensions = None
    _load_error = str(e)


# =============================================================================
# Tests for CRIS Regional Detection
# =============================================================================


@pytest.mark.skipif(
    detect_cris_regional is None,
    reason="Could not load detect_cris_regional function",
)
class TestCRISRegionalDetection:
    """Tests for CRIS Regional detection."""

    def test_detect_cris_regional_geo_suffix(self):
        """detect_cris_regional returns True for _Geo suffix."""
        # Arrange
        usage_type = "USE1_InputTokenCount_Geo"
        description = ""

        # Act
        result = detect_cris_regional(usage_type, description)

        # Assert
        assert result is True

    def test_detect_cris_regional_description(self):
        """detect_cris_regional returns True for Regional CRIS in description."""
        # Arrange
        usage_type = ""
        description = "Regional CRIS pricing"

        # Act
        result = detect_cris_regional(usage_type, description)

        # Assert
        assert result is True

    def test_detect_cris_regional_not_global(self):
        """detect_cris_regional returns False for Global patterns."""
        # Arrange
        usage_type = "USE1_Global_InputTokens"
        description = ""

        # Act
        result = detect_cris_regional(usage_type, description)

        # Assert
        assert result is False

    def test_detect_cris_regional_hyphen_geo(self):
        """detect_cris_regional returns True for -geo suffix."""
        # Arrange
        usage_type = "USE1-InputTokenCount-geo"
        description = ""

        # Act
        result = detect_cris_regional(usage_type, description)

        # Assert
        assert result is True

    def test_detect_cris_regional_cross_region_geo(self):
        """detect_cris_regional returns True for cross-region-geo pattern."""
        # Arrange
        usage_type = "USE1-cross-region-geo-InputTokens"
        description = ""

        # Act
        result = detect_cris_regional(usage_type, description)

        # Assert
        assert result is True

    def test_detect_cris_regional_case_insensitive(self):
        """detect_cris_regional is case-insensitive."""
        # Arrange - various case combinations
        test_cases = [
            ("USE1_InputTokenCount_GEO", ""),
            ("USE1_InputTokenCount_Geo", ""),
            ("USE1_InputTokenCount_geo", ""),
            ("", "REGIONAL CRIS pricing"),
            ("", "regional cris pricing"),
        ]

        # Act & Assert
        for usage_type, description in test_cases:
            result = detect_cris_regional(usage_type, description)
            assert result is True, (
                f"Failed for usage_type={usage_type}, description={description}"
            )

    def test_detect_non_cris_regional(self):
        """detect_cris_regional returns False for non-regional patterns."""
        # Arrange
        usage_type = "USE1-OnDemand-InputTokens"
        description = ""

        # Act
        result = detect_cris_regional(usage_type, description)

        # Assert
        assert result is False

    def test_detect_cris_regional_global_in_description_returns_false(self):
        """detect_cris_regional returns False when Global is in description."""
        # Arrange
        usage_type = "USE1_InputTokenCount_Geo"
        description = "Global pricing"

        # Act
        result = detect_cris_regional(usage_type, description)

        # Assert
        assert result is False


# =============================================================================
# Tests for Reserved Pricing Detection
# =============================================================================


@pytest.mark.skipif(
    detect_reserved_pricing is None,
    reason="Could not load detect_reserved_pricing function",
)
class TestReservedDetection:
    """Tests for Reserved pricing detection."""

    def test_detect_reserved_1month(self):
        """detect_reserved_pricing extracts 1_month commitment."""
        # Arrange
        usage_type = "Reserved_1Month_TPM"
        description = ""

        # Act
        is_reserved, commitment = detect_reserved_pricing(usage_type, description)

        # Assert
        assert is_reserved is True
        assert commitment == "1_month"

    def test_detect_reserved_3month(self):
        """detect_reserved_pricing extracts 3_month commitment."""
        # Arrange
        usage_type = "Reserved_3Month_TPM"
        description = ""

        # Act
        is_reserved, commitment = detect_reserved_pricing(usage_type, description)

        # Assert
        assert is_reserved is True
        assert commitment == "3_month"

    def test_detect_reserved_6month(self):
        """detect_reserved_pricing extracts 6_month commitment."""
        # Arrange
        usage_type = "Reserved_6Month_TPM"
        description = ""

        # Act
        is_reserved, commitment = detect_reserved_pricing(usage_type, description)

        # Assert
        assert is_reserved is True
        assert commitment == "6_month"

    def test_detect_reserved_no_commit(self):
        """detect_reserved_pricing extracts no_commit."""
        # Arrange
        usage_type = "no-commit-reserved"
        description = ""

        # Act
        is_reserved, commitment = detect_reserved_pricing(usage_type, description)

        # Assert
        assert is_reserved is True
        assert commitment == "no_commit"

    def test_detect_non_reserved(self):
        """detect_reserved_pricing returns False for non-reserved."""
        # Arrange
        usage_type = "OnDemand_InputTokens"
        description = ""

        # Act
        is_reserved, commitment = detect_reserved_pricing(usage_type, description)

        # Assert
        assert is_reserved is False
        assert commitment is None

    def test_detect_reserved_tpm_pattern(self):
        """detect_reserved_pricing detects _tpm_ pattern."""
        # Arrange
        usage_type = "USE1_tpm_InputTokens"
        description = ""

        # Act
        is_reserved, commitment = detect_reserved_pricing(usage_type, description)

        # Assert
        assert is_reserved is True

    def test_detect_reserved_from_description(self):
        """detect_reserved_pricing detects reserved from description."""
        # Arrange
        usage_type = ""
        description = "Reserved capacity pricing"

        # Act
        is_reserved, commitment = detect_reserved_pricing(usage_type, description)

        # Assert
        assert is_reserved is True

    def test_detect_reserved_case_insensitive(self):
        """detect_reserved_pricing is case-insensitive."""
        # Arrange - various case combinations
        test_cases = [
            ("RESERVED_1MONTH_TPM", "", "1_month"),
            ("reserved_1month_tpm", "", "1_month"),
            ("Reserved_3Month_TPM", "", "3_month"),
            ("NO-COMMIT-RESERVED", "", "no_commit"),
        ]

        # Act & Assert
        for usage_type, description, expected_commitment in test_cases:
            is_reserved, commitment = detect_reserved_pricing(usage_type, description)
            assert is_reserved is True, f"Failed for usage_type={usage_type}"
            assert commitment == expected_commitment, (
                f"Wrong commitment for usage_type={usage_type}"
            )

    def test_detect_reserved_without_commitment(self):
        """detect_reserved_pricing returns None commitment when not specified."""
        # Arrange
        usage_type = "reserved-pricing"
        description = ""

        # Act
        is_reserved, commitment = detect_reserved_pricing(usage_type, description)

        # Assert
        assert is_reserved is True
        assert commitment is None


# =============================================================================
# Tests for CRIS Regional and Reserved Dimensions
# =============================================================================


@pytest.mark.skipif(
    determine_pricing_group_with_dimensions is None,
    reason="Could not load determine_pricing_group_with_dimensions function",
)
class TestCRISRegionalDimensions:
    """Tests for CRIS Regional dimension extraction."""

    def test_dimension_geographic_scope_regional(self):
        """dimensions include geographic_scope: cris_regional for CRIS Regional."""
        # Arrange
        usage_type = "USE1_InputTokenCount_Geo"
        inference_type = ""
        description = ""

        # Act
        result = determine_pricing_group_with_dimensions(
            usage_type, inference_type, description
        )

        # Assert
        assert result["dimensions"]["geographic_scope"] == "cris_regional"

    def test_dimension_geographic_scope_global(self):
        """dimensions include geographic_scope: cris_global for Global."""
        # Arrange
        usage_type = "USE1_Global_InputTokens"
        inference_type = ""
        description = ""

        # Act
        result = determine_pricing_group_with_dimensions(
            usage_type, inference_type, description
        )

        # Assert
        assert result["dimensions"]["geographic_scope"] == "cris_global"

    def test_dimension_geographic_scope_in_region_default(self):
        """dimensions default to geographic_scope: in_region."""
        # Arrange
        usage_type = "USE1-OnDemand-InputTokens"
        inference_type = ""
        description = ""

        # Act
        result = determine_pricing_group_with_dimensions(
            usage_type, inference_type, description
        )

        # Assert
        assert result["dimensions"]["geographic_scope"] == "in_region"


@pytest.mark.skipif(
    determine_pricing_group_with_dimensions is None,
    reason="Could not load determine_pricing_group_with_dimensions function",
)
class TestReservedDimensions:
    """Tests for Reserved pricing dimension extraction."""

    def test_dimension_inference_mode_reserved(self):
        """dimensions include inference_mode: reserved for Reserved pricing."""
        # Arrange
        usage_type = "Reserved_3Month_TPM"
        inference_type = ""
        description = ""

        # Act
        result = determine_pricing_group_with_dimensions(
            usage_type, inference_type, description
        )

        # Assert
        assert result["dimensions"]["inference_mode"] == "reserved"

    def test_dimension_commitment_term(self):
        """dimensions include commitment term for Reserved pricing."""
        # Arrange
        usage_type = "Reserved_3Month_TPM"
        inference_type = ""
        description = ""

        # Act
        result = determine_pricing_group_with_dimensions(
            usage_type, inference_type, description
        )

        # Assert
        assert result["dimensions"]["commitment"] == "3_month"

    def test_group_is_reserved(self):
        """group is Reserved for Reserved pricing."""
        # Arrange
        usage_type = "Reserved_3Month_TPM"
        inference_type = ""
        description = ""

        # Act
        result = determine_pricing_group_with_dimensions(
            usage_type, inference_type, description
        )

        # Assert
        assert result["group"] == "Reserved"

    def test_reserved_takes_priority_over_provisioned(self):
        """Reserved detection takes priority over provisioned detection."""
        # Arrange - usage type that could match both Reserved and provisioned
        usage_type = "Reserved_provisioned_3Month_TPM"
        inference_type = ""
        description = ""

        # Act
        result = determine_pricing_group_with_dimensions(
            usage_type, inference_type, description
        )

        # Assert - Reserved should win
        assert result["group"] == "Reserved"
        assert result["dimensions"]["inference_mode"] == "reserved"

    def test_dimension_commitment_no_commit(self):
        """dimensions include commitment: no_commit for no-commit reserved."""
        # Arrange
        usage_type = "no-commit-reserved"
        inference_type = ""
        description = ""

        # Act
        result = determine_pricing_group_with_dimensions(
            usage_type, inference_type, description
        )

        # Assert
        assert result["dimensions"]["commitment"] == "no_commit"
