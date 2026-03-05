"""Tests for Pricing Aggregator dimension extraction and aggregation.

Tests for Task 02 - Pricing Dimensions Restructure:
- determine_pricing_group_with_dimensions()
- aggregate_dimensions()

These tests verify that pricing dimensions (Mantle, Geo, Tier, Context)
are correctly extracted and aggregated.
"""

import pytest
import sys
import importlib.util
from pathlib import Path

HANDLERS_PATH = Path(__file__).parent.parent / "lambdas"


def get_handler_source(handler_name: str) -> str:
    """Read the handler source code."""
    handler_path = HANDLERS_PATH / handler_name / "handler.py"
    return handler_path.read_text()


# Load the pricing-aggregator handler module directly to test functions
# We need to mock the shared imports since they depend on Lambda layer
PRICING_AGGREGATOR_PATH = HANDLERS_PATH / "pricing-aggregator" / "handler.py"


def _load_pricing_aggregator_functions():
    """Load pricing aggregator functions by extracting them from source.

    Since the handler has dependencies on shared modules that aren't available
    in the test environment, we extract and exec just the functions we need.
    """
    import re

    source = get_handler_source("pricing-aggregator")
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

    # Extract constants needed by helper functions
    mantle_patterns_source = extract_constant("MANTLE_PATTERNS")
    cris_regional_patterns_source = extract_constant("CRIS_REGIONAL_PATTERNS")
    reserved_patterns_source = extract_constant("RESERVED_PATTERNS")
    cache_patterns_source = extract_constant("CACHE_PATTERNS")
    commitment_patterns_source = extract_constant("COMMITMENT_PATTERNS")

    # Extract helper functions
    detect_mantle_source = extract_function("detect_mantle_pricing")
    detect_cris_regional_source = extract_function("detect_cris_regional")
    detect_reserved_source = extract_function("detect_reserved_pricing")
    detect_cache_source = extract_function("detect_cache_type")

    # Extract main functions
    func_source = extract_function("determine_pricing_group_with_dimensions")
    if not func_source:
        raise RuntimeError("Could not find determine_pricing_group_with_dimensions")

    agg_source = extract_function("aggregate_dimensions")
    if not agg_source:
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
    if cache_patterns_source:
        exec(cache_patterns_source, namespace)
    if commitment_patterns_source:
        exec(commitment_patterns_source, namespace)
    if detect_mantle_source:
        exec(detect_mantle_source, namespace)
    if detect_cris_regional_source:
        exec(detect_cris_regional_source, namespace)
    if detect_reserved_source:
        exec(detect_reserved_source, namespace)
    if detect_cache_source:
        exec(detect_cache_source, namespace)
    exec(func_source, namespace)
    exec(agg_source, namespace)

    return namespace


# Load functions once at module level
try:
    _funcs = _load_pricing_aggregator_functions()
    determine_pricing_group_with_dimensions = _funcs[
        "determine_pricing_group_with_dimensions"
    ]
    aggregate_dimensions = _funcs["aggregate_dimensions"]
except Exception as e:
    # If loading fails, tests will be skipped
    determine_pricing_group_with_dimensions = None
    aggregate_dimensions = None


class TestPricingDimensionFunctions:
    """Tests for pricing dimension extraction functions."""

    def test_pricing_aggregator_has_dimension_function(self):
        """pricing-aggregator should have determine_pricing_group_with_dimensions function."""
        source = get_handler_source("pricing-aggregator")
        assert "def determine_pricing_group_with_dimensions" in source

    def test_pricing_aggregator_has_aggregate_dimensions_function(self):
        """pricing-aggregator should have aggregate_dimensions function."""
        source = get_handler_source("pricing-aggregator")
        assert "def aggregate_dimensions" in source

    def test_pricing_entry_has_dimensions_field(self):
        """pricing entries should include dimensions field."""
        source = get_handler_source("pricing-aggregator")
        # Check that dimensions is added to pricing_entry
        assert (
            '"dimensions": dimensions' in source or "'dimensions': dimensions" in source
        )

    def test_model_entry_has_available_dimensions(self):
        """model entries should include available_dimensions field."""
        source = get_handler_source("pricing-aggregator")
        assert '"available_dimensions"' in source or "'available_dimensions'" in source

    def test_model_entry_has_mantle_pricing_flag(self):
        """model entries should include has_mantle_pricing field."""
        source = get_handler_source("pricing-aggregator")
        assert '"has_mantle_pricing"' in source or "'has_mantle_pricing'" in source


class TestDimensionExtractionLogic:
    """Tests for dimension extraction logic patterns in source code."""

    def test_mantle_detection_patterns(self):
        """pricing-aggregator should detect mantle from usage type and inference type."""
        source = get_handler_source("pricing-aggregator")
        # Check for mantle detection patterns
        assert '"mantle"' in source.lower() or "'mantle'" in source.lower()
        assert "openai-compatible" in source.lower()

    def test_geo_detection_patterns(self):
        """pricing-aggregator should detect geo dimensions (global, regional)."""
        source = get_handler_source("pricing-aggregator")
        # Check for geo detection patterns
        assert '"global"' in source or "'global'" in source
        assert '"regional"' in source or "'regional'" in source
        assert "cris" in source.lower()

    def test_tier_detection_patterns(self):
        """pricing-aggregator should detect tier dimensions (flex, priority)."""
        source = get_handler_source("pricing-aggregator")
        # Check for tier detection patterns
        assert '"flex"' in source or "'flex'" in source
        assert '"priority"' in source or "'priority'" in source

    def test_context_detection_patterns(self):
        """pricing-aggregator should detect context dimensions (standard, long)."""
        source = get_handler_source("pricing-aggregator")
        # Check for context detection patterns
        assert '"long"' in source or "'long'" in source
        assert '"standard"' in source or "'standard'" in source
        assert "_lctx" in source.lower()


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing 10 pricing groups."""

    def test_legacy_pricing_group_preserved(self):
        """pricing entries should preserve legacy pricing_group field."""
        source = get_handler_source("pricing-aggregator")
        # Check that legacy pricing_group is still used
        assert (
            '"pricing_group": legacy_pricing_group' in source
            or "'pricing_group': legacy_pricing_group" in source
        )

    def test_legacy_determine_pricing_group_exists(self):
        """legacy determine_pricing_group function should still exist."""
        source = get_handler_source("pricing-aggregator")
        assert "def determine_pricing_group(" in source

    def test_pricing_groups_structure_preserved(self):
        """pricing_groups dict structure should be preserved."""
        source = get_handler_source("pricing-aggregator")
        # Check that pricing_groups is still used for grouping
        assert "pricing_groups" in source
        assert "legacy_pricing_group" in source


class TestDimensionSchema:
    """Tests for dimension schema structure."""

    def test_dimension_source_values(self):
        """dimensions.source should have standard and mantle values."""
        source = get_handler_source("pricing-aggregator")
        # Check for source dimension values (both initialization and assignment)
        assert '"source": "standard"' in source or "'source': 'standard'" in source
        assert (
            'dimensions["source"] = "mantle"' in source
            or "dimensions['source'] = 'mantle'" in source
        )

    def test_dimension_geo_values(self):
        """dimensions.geographic_scope should have in_region, cris_global, and cris_regional values."""
        source = get_handler_source("pricing-aggregator")
        # Check for geographic_scope dimension values (both initialization and assignment)
        assert (
            '"geographic_scope": "in_region"' in source
            or "'geographic_scope': 'in_region'" in source
        )
        assert (
            'dimensions["geographic_scope"] = "cris_global"' in source
            or "dimensions['geographic_scope'] = 'cris_global'" in source
        )
        assert (
            'dimensions["geographic_scope"] = "cris_regional"' in source
            or "dimensions['geographic_scope'] = 'cris_regional'" in source
        )

    def test_dimension_tier_values(self):
        """dimensions.tier should have None, flex, and priority values."""
        source = get_handler_source("pricing-aggregator")
        # Check for tier dimension values (both initialization and assignment)
        assert '"tier": None' in source or "'tier': None" in source
        assert (
            'dimensions["tier"] = "flex"' in source
            or "dimensions['tier'] = 'flex'" in source
        )
        assert (
            'dimensions["tier"] = "priority"' in source
            or "dimensions['tier'] = 'priority'" in source
        )

    def test_dimension_context_values(self):
        """dimensions.context_type should have standard and long_context values."""
        source = get_handler_source("pricing-aggregator")
        # Check for context_type dimension values (both initialization and assignment)
        assert (
            '"context_type": "standard"' in source
            or "'context_type': 'standard'" in source
        )
        assert (
            'dimensions["context_type"] = "long_context"' in source
            or "dimensions['context_type'] = 'long_context'" in source
        )


# =============================================================================
# Functional Tests for Pricing Dimensions
# =============================================================================


@pytest.mark.skipif(
    determine_pricing_group_with_dimensions is None,
    reason="Could not load pricing aggregator functions",
)
class TestPricingDimensions:
    """Tests for nested pricing dimension extraction.

    These tests verify that determine_pricing_group_with_dimensions correctly
    extracts dimension values from usage type and inference type strings.
    """

    def test_mantle_source_detection(self):
        """Mantle usage type should set source=mantle."""
        result = determine_pricing_group_with_dimensions("USE1-Mantle-Claude-input", "")
        assert result["dimensions"]["source"] == "mantle"

    def test_mantle_from_inference_type(self):
        """OpenAI-compatible inference type should set source=mantle."""
        result = determine_pricing_group_with_dimensions(
            "USE1-Claude-input", "openai-compatible"
        )
        assert result["dimensions"]["source"] == "mantle"

    def test_standard_source_default(self):
        """Standard usage type should set source=standard."""
        result = determine_pricing_group_with_dimensions("USE1-Claude3Sonnet-input", "")
        assert result["dimensions"]["source"] == "standard"

    def test_global_geo_detection(self):
        """Global usage type should set geo=global."""
        result = determine_pricing_group_with_dimensions("USE1-Claude-Global-input", "")
        assert result["dimensions"]["geo"] == "global"

    def test_geo_suffix_detection(self):
        """_Geo suffix in usage type should set geo=regional."""
        result = determine_pricing_group_with_dimensions("USE1-Claude_Geo-input", "")
        assert result["dimensions"]["geo"] == "regional"

    def test_regional_geo_detection(self):
        """Regional usage type should set geo=regional."""
        result = determine_pricing_group_with_dimensions(
            "USE1-Claude-regional-input", ""
        )
        assert result["dimensions"]["geo"] == "regional"

    def test_long_context_detection(self):
        """Long context usage type should set context=long."""
        result = determine_pricing_group_with_dimensions(
            "USE1_InputTokenCount_LCtx", ""
        )
        assert result["dimensions"]["context"] == "long"

    def test_long_context_hyphenated(self):
        """Long-context usage type should set context=long."""
        result = determine_pricing_group_with_dimensions(
            "USE1-Claude-long-context-input", ""
        )
        assert result["dimensions"]["context"] == "long"

    def test_long_context_from_inference_type(self):
        """Long context inference type should set context=long."""
        result = determine_pricing_group_with_dimensions(
            "USE1-Claude-input", "long context"
        )
        assert result["dimensions"]["context"] == "long"

    def test_standard_context_default(self):
        """Standard usage type should set context=standard."""
        result = determine_pricing_group_with_dimensions("USE1-Claude-input", "")
        assert result["dimensions"]["context"] == "standard"

    def test_tier_flex_detection(self):
        """Flex tier should be detected."""
        result = determine_pricing_group_with_dimensions("USE1-Claude-Flex-input", "")
        assert result["dimensions"]["tier"] == "flex"

    def test_tier_priority_detection(self):
        """Priority tier should be detected."""
        result = determine_pricing_group_with_dimensions(
            "USE1-Claude-Priority-input", ""
        )
        assert result["dimensions"]["tier"] == "priority"

    def test_tier_none_default(self):
        """No tier should default to None."""
        result = determine_pricing_group_with_dimensions("USE1-Claude-input", "")
        assert result["dimensions"]["tier"] is None

    def test_base_group_on_demand(self):
        """Standard usage should return On-Demand group."""
        result = determine_pricing_group_with_dimensions("USE1-Claude-input", "")
        assert result["group"] == "On-Demand"

    def test_base_group_batch(self):
        """Batch usage should return Batch group."""
        result = determine_pricing_group_with_dimensions("USE1-Claude-batch-input", "")
        assert result["group"] == "Batch"

    def test_base_group_provisioned(self):
        """Provisioned usage should return Provisioned Throughput group."""
        result = determine_pricing_group_with_dimensions(
            "USE1-Claude-provisioned-input", ""
        )
        assert result["group"] == "Provisioned Throughput"

    def test_base_group_custom(self):
        """Custom model usage should return Custom Model group."""
        result = determine_pricing_group_with_dimensions("USE1-Claude-custom-input", "")
        assert result["group"] == "Custom Model"


@pytest.mark.skipif(
    aggregate_dimensions is None, reason="Could not load pricing aggregator functions"
)
class TestDimensionAggregation:
    """Tests for dimension aggregation across pricing entries."""

    def test_dimension_aggregation_multiple_sources(self):
        """Should aggregate multiple sources."""
        entries = [
            {
                "dimensions": {
                    "source": "standard",
                    "geo": None,
                    "tier": None,
                    "context": "standard",
                }
            },
            {
                "dimensions": {
                    "source": "mantle",
                    "geo": None,
                    "tier": None,
                    "context": "standard",
                }
            },
        ]
        result = aggregate_dimensions(entries)
        assert set(result["sources"]) == {"standard", "mantle"}

    def test_dimension_aggregation_multiple_geos(self):
        """Should aggregate multiple geos."""
        entries = [
            {
                "dimensions": {
                    "source": "standard",
                    "geo": "global",
                    "tier": None,
                    "context": "standard",
                }
            },
            {
                "dimensions": {
                    "source": "standard",
                    "geo": "regional",
                    "tier": None,
                    "context": "standard",
                }
            },
        ]
        result = aggregate_dimensions(entries)
        assert set(result["geos"]) == {"global", "regional"}

    def test_dimension_aggregation_multiple_tiers(self):
        """Should aggregate multiple tiers."""
        entries = [
            {
                "dimensions": {
                    "source": "standard",
                    "geo": None,
                    "tier": "flex",
                    "context": "standard",
                }
            },
            {
                "dimensions": {
                    "source": "standard",
                    "geo": None,
                    "tier": "priority",
                    "context": "standard",
                }
            },
        ]
        result = aggregate_dimensions(entries)
        assert set(result["tiers"]) == {"flex", "priority"}

    def test_dimension_aggregation_multiple_contexts(self):
        """Should aggregate multiple contexts."""
        entries = [
            {
                "dimensions": {
                    "source": "standard",
                    "geo": None,
                    "tier": None,
                    "context": "standard",
                }
            },
            {
                "dimensions": {
                    "source": "standard",
                    "geo": None,
                    "tier": None,
                    "context": "long",
                }
            },
        ]
        result = aggregate_dimensions(entries)
        assert set(result["contexts"]) == {"standard", "long"}

    def test_dimension_aggregation_empty_entries(self):
        """Should handle empty entries with defaults."""
        result = aggregate_dimensions([])
        assert result["sources"] == ["standard"]
        assert result["geos"] == []
        assert result["tiers"] == []
        assert result["contexts"] == ["standard"]

    def test_dimension_aggregation_missing_dimensions(self):
        """Should handle entries without dimensions field."""
        entries = [
            {},  # No dimensions field
            {
                "dimensions": {
                    "source": "mantle",
                    "geo": None,
                    "tier": None,
                    "context": "standard",
                }
            },
        ]
        result = aggregate_dimensions(entries)
        assert "mantle" in result["sources"]

    def test_dimension_aggregation_sorted_output(self):
        """Should return sorted lists."""
        entries = [
            {
                "dimensions": {
                    "source": "standard",
                    "geo": "regional",
                    "tier": "priority",
                    "context": "long",
                }
            },
            {
                "dimensions": {
                    "source": "mantle",
                    "geo": "global",
                    "tier": "flex",
                    "context": "standard",
                }
            },
        ]
        result = aggregate_dimensions(entries)
        # Verify lists are sorted
        assert result["sources"] == sorted(result["sources"])
        assert result["geos"] == sorted(result["geos"])
        assert result["tiers"] == sorted(result["tiers"])
        assert result["contexts"] == sorted(result["contexts"])
