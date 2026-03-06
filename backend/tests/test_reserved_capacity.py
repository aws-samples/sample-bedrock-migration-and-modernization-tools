"""
Tests for Reserved Capacity availability in final-aggregator.

Tests build_reserved_capacity(), _parse_commitment_term(),
get_consumption_options() with reserved, build_availability() with reserved,
and reconciliation logic in transform_model_to_schema().
"""

import pytest
from unittest.mock import MagicMock
import sys
import os

# Add lambda to path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "lambdas", "final-aggregator")
)

# Mock shared module before importing handler
mock_shared = MagicMock()
mock_shared.get_s3_client.return_value = MagicMock()
mock_shared.read_from_s3.return_value = {}
mock_shared.write_to_s3 = MagicMock()
mock_shared.parse_execution_id.return_value = "test-exec-123"
mock_shared.validate_required_params = MagicMock()
mock_shared.ValidationError = Exception
mock_shared.S3ReadError = Exception
mock_shared.get_config_loader.return_value = MagicMock(
    config={"model_configuration": {"context_window_specs": {}}}
)

sys.modules["shared"] = mock_shared

# Mock shared.model_matcher submodule
mock_model_matcher = MagicMock()
mock_model_matcher.get_canonical_model_id = (
    lambda x: x.lower().split(":")[0] if x else ""
)
mock_model_matcher.calculate_match_score = lambda x, y: 1.0 if x == y else 0.0
mock_model_matcher.get_model_variant_info = lambda x: {
    "base_id": x.split(":")[0] if x else "",
    "is_multimodal": False,
    "is_provisioned_only": False,
    "context_window": None,
    "version": None,
    "api_version": None,
    "has_dimension_suffix": False,
}
mock_model_matcher.has_semantic_conflict = lambda x, y: False
mock_model_matcher.get_provider_from_model_id = lambda x: (
    x.split(".")[0] if "." in x else "Unknown",
    x.split(".")[0].title() if "." in x else "Unknown",
)

sys.modules["shared.model_matcher"] = mock_model_matcher

# Mock shared.powertools submodule
mock_powertools = MagicMock()
mock_powertools.logger = MagicMock()
mock_powertools.tracer = MagicMock()
mock_powertools.tracer.capture_method = lambda f: f
mock_powertools.metrics = MagicMock()
mock_powertools.LambdaContext = MagicMock

sys.modules["shared.powertools"] = mock_powertools
sys.modules["aws_lambda_powertools"] = MagicMock()
sys.modules["aws_lambda_powertools.metrics"] = MagicMock()

from handler import (
    build_reserved_capacity,
    _parse_commitment_term,
    get_consumption_options,
    build_availability,
)


# ============================================================================
# _parse_commitment_term tests
# ============================================================================


class TestParseCommitmentTerm:
    def test_1_month(self):
        assert _parse_commitment_term("Reserved 1 Month Geo") == "1_month"

    def test_3_month(self):
        assert _parse_commitment_term("Reserved 3 Month Global") == "3_month"

    def test_6_month(self):
        assert _parse_commitment_term("Reserved 6 Month Geo") == "6_month"

    def test_no_match(self):
        assert _parse_commitment_term("On-Demand") == ""

    def test_partial_match(self):
        assert _parse_commitment_term("Reserved") == ""


# ============================================================================
# build_reserved_capacity tests
# ============================================================================


def _make_pricing_data(provider, model_key, regions_with_groups):
    """Helper to build pricing_data structure.

    regions_with_groups: dict of region -> list of pricing group names
    """
    regions = {}
    for region, groups in regions_with_groups.items():
        regions[region] = {
            "pricing_groups": {g: {"prices": []} for g in groups}
        }
    return {
        "providers": {
            provider: {
                model_key: {
                    "regions": regions
                }
            }
        }
    }


class TestBuildReservedCapacity:
    def test_no_reserved_groups(self):
        pricing_data = _make_pricing_data(
            "Anthropic", "claude-sonnet-4-5",
            {"us-east-1": ["On-Demand", "Batch"]}
        )
        pricing_ref = {"provider": "Anthropic", "model_key": "claude-sonnet-4-5"}

        result = build_reserved_capacity("anthropic.claude-sonnet-4-5-v1:0", pricing_data, pricing_ref)

        assert result["supported"] is False
        assert result["regions"] == []
        assert result["commitments"] == []

    def test_with_reserved_groups(self):
        pricing_data = _make_pricing_data(
            "Anthropic", "claude-sonnet-4-5",
            {
                "us-east-1": ["On-Demand", "Reserved 1 Month Geo", "Reserved 3 Month Global"],
                "us-west-2": ["On-Demand", "Reserved 1 Month Geo"],
                "eu-west-1": ["On-Demand"],  # No reserved here
            }
        )
        pricing_ref = {"provider": "Anthropic", "model_key": "claude-sonnet-4-5"}

        result = build_reserved_capacity("anthropic.claude-sonnet-4-5-v1:0", pricing_data, pricing_ref)

        assert result["supported"] is True
        assert sorted(result["regions"]) == ["us-east-1", "us-west-2"]
        assert sorted(result["commitments"]) == ["1_month", "3_month"]

    def test_all_regions_have_reserved(self):
        pricing_data = _make_pricing_data(
            "Anthropic", "claude-opus-4-5",
            {
                "us-east-1": ["On-Demand", "Reserved 1 Month Geo", "Reserved 3 Month Geo"],
                "us-west-2": ["On-Demand", "Reserved 1 Month Geo", "Reserved 3 Month Geo"],
            }
        )
        pricing_ref = {"provider": "Anthropic", "model_key": "claude-opus-4-5"}

        result = build_reserved_capacity("anthropic.claude-opus-4-5-v1:0", pricing_data, pricing_ref)

        assert result["supported"] is True
        assert result["regions"] == ["us-east-1", "us-west-2"]
        assert result["commitments"] == ["1_month", "3_month"]

    def test_fuzzy_matching_fallback(self):
        """When pricing_ref model_key doesn't match exactly, fuzzy match should work."""
        pricing_data = _make_pricing_data(
            "Anthropic", "claude-sonnet-4-6",
            {"us-east-1": ["On-Demand", "Reserved 1 Month Geo"]}
        )
        # pricing_ref with a key that won't match directly but will fuzzy match
        pricing_ref = {"provider": "Anthropic", "model_key": "claude-sonnet-4-6-v1"}

        result = build_reserved_capacity("anthropic.claude-sonnet-4-6-v1:0", pricing_data, pricing_ref)

        assert result["supported"] is True
        assert result["regions"] == ["us-east-1"]

    def test_no_pricing_ref(self):
        """Falls back to model_id when no pricing_ref provided."""
        pricing_data = _make_pricing_data(
            "Anthropic", "anthropic.claude-haiku-4-5-v1:0",
            {"us-east-1": ["On-Demand", "Reserved 1 Month Geo"]}
        )

        result = build_reserved_capacity("anthropic.claude-haiku-4-5-v1:0", pricing_data, None)

        assert result["supported"] is True

    def test_empty_pricing_data(self):
        result = build_reserved_capacity("anthropic.claude-sonnet-4-5-v1:0", {"providers": {}}, None)

        assert result["supported"] is False
        assert result["regions"] == []
        assert result["commitments"] == []

    def test_multiple_commitment_terms(self):
        pricing_data = _make_pricing_data(
            "Anthropic", "claude-sonnet-4-5",
            {
                "us-east-1": [
                    "On-Demand",
                    "Reserved 1 Month Geo",
                    "Reserved 1 Month Global",
                    "Reserved 3 Month Geo",
                    "Reserved 3 Month Global",
                ],
            }
        )
        pricing_ref = {"provider": "Anthropic", "model_key": "claude-sonnet-4-5"}

        result = build_reserved_capacity("anthropic.claude-sonnet-4-5-v1:0", pricing_data, pricing_ref)

        assert result["supported"] is True
        # Should deduplicate: 1 Month Geo + 1 Month Global = just "1_month"
        assert sorted(result["commitments"]) == ["1_month", "3_month"]


# ============================================================================
# get_consumption_options tests (reserved integration)
# ============================================================================


class TestGetConsumptionOptionsReserved:
    def test_reserved_from_pricing(self):
        pricing_data = _make_pricing_data(
            "Anthropic", "claude-sonnet-4-5",
            {"us-east-1": ["On-Demand", "Reserved 1 Month Geo"]}
        )
        pricing_ref = {"provider": "Anthropic", "model_key": "claude-sonnet-4-5"}

        result = get_consumption_options(
            ["ON_DEMAND"], pricing_data, pricing_ref, mantle_supported=False
        )

        assert "on_demand" in result
        assert "reserved" in result

    def test_no_reserved_without_pricing_groups(self):
        pricing_data = _make_pricing_data(
            "Anthropic", "claude-sonnet-4-5",
            {"us-east-1": ["On-Demand"]}
        )
        pricing_ref = {"provider": "Anthropic", "model_key": "claude-sonnet-4-5"}

        result = get_consumption_options(
            ["ON_DEMAND"], pricing_data, pricing_ref, mantle_supported=False
        )

        assert "reserved" not in result

    def test_reserved_sort_order(self):
        """Reserved should come after provisioned_throughput in sort order."""
        pricing_data = _make_pricing_data(
            "Anthropic", "claude-sonnet-4-5",
            {"us-east-1": ["On-Demand", "Batch", "Provisioned Throughput", "Reserved 1 Month Geo"]}
        )
        pricing_ref = {"provider": "Anthropic", "model_key": "claude-sonnet-4-5"}

        result = get_consumption_options(
            ["ON_DEMAND"], pricing_data, pricing_ref, mantle_supported=False
        )

        assert result.index("provisioned_throughput") < result.index("reserved")


# ============================================================================
# build_availability tests (reserved integration)
# ============================================================================


class TestBuildAvailabilityReserved:
    def _base_args(self):
        return {
            "regional_availability": ["us-east-1"],
            "cross_region_data": {"supported": False, "source_regions": [], "profiles": []},
            "batch_inference_data": {"supported": False, "supported_regions": []},
            "provisioned_data": {"supported": False, "provisioned_regions": []},
            "mantle_data": {"supported": False, "mantle_regions": []},
        }

    def test_reserved_supported(self):
        args = self._base_args()
        args["reserved_data"] = {
            "supported": True,
            "regions": ["us-east-1", "us-west-2"],
            "commitments": ["1_month", "3_month"],
        }

        result = build_availability(**args)

        assert result["reserved"]["supported"] is True
        assert result["reserved"]["regions"] == ["us-east-1", "us-west-2"]
        assert result["reserved"]["commitments"] == ["1_month", "3_month"]

    def test_reserved_not_supported(self):
        args = self._base_args()
        args["reserved_data"] = {
            "supported": False,
            "regions": [],
            "commitments": [],
        }

        result = build_availability(**args)

        assert result["reserved"]["supported"] is False
        assert result["reserved"]["regions"] == []
        assert result["reserved"]["commitments"] == []

    def test_reserved_default_none(self):
        """When reserved_data is None (default), should produce unsupported."""
        args = self._base_args()

        result = build_availability(**args)

        assert result["reserved"]["supported"] is False
        assert result["reserved"]["regions"] == []
        assert result["reserved"]["commitments"] == []

    def test_all_availability_sections_present(self):
        """Verify reserved is included alongside all other sections."""
        args = self._base_args()

        result = build_availability(**args)

        assert "on_demand" in result
        assert "cross_region" in result
        assert "batch" in result
        assert "provisioned" in result
        assert "mantle" in result
        assert "reserved" in result


# ============================================================================
# Integration tests: real pricing data from workflow_output/merged/pricing.json
# ============================================================================

_PRICING_PATH = os.path.join(
    os.path.dirname(__file__), "workflow_output", "merged", "pricing.json"
)


def _load_real_pricing_data():
    """Load real pricing data from the workflow output.

    The merged pricing uses providers -> provider -> models -> model_key structure.
    Unwrap the 'models' nesting so it matches the format final-aggregator expects:
    providers -> provider -> model_key -> {regions: ...}
    """
    import json

    with open(_PRICING_PATH) as f:
        raw = json.load(f)

    pricing_data = {"providers": {}}
    for prov, pdata in raw.get("providers", {}).items():
        if isinstance(pdata, dict) and "models" in pdata:
            pricing_data["providers"][prov] = pdata["models"]
        else:
            pricing_data["providers"][prov] = pdata
    return pricing_data


_has_real_pricing = os.path.exists(_PRICING_PATH)


@pytest.mark.skipif(not _has_real_pricing, reason="No real pricing data at workflow_output/merged/pricing.json")
class TestReservedCapacityRealData:
    """Integration tests using REAL pricing data from the Pricing API."""

    @pytest.fixture
    def pricing_data(self):
        return _load_real_pricing_data()

    def test_claude_sonnet_46_has_reserved(self, pricing_data):
        ref = {"provider": "Anthropic", "model_key": "anthropic.claude-sonnet-4-6"}
        result = build_reserved_capacity(
            "anthropic.claude-sonnet-4-6-v1:0", pricing_data, ref
        )

        assert result["supported"] is True
        assert len(result["regions"]) >= 10
        assert "1_month" in result["commitments"]
        assert "3_month" in result["commitments"]

    def test_claude_opus_46_has_reserved(self, pricing_data):
        ref = {"provider": "Anthropic", "model_key": "anthropic.claude-opus-4-6"}
        result = build_reserved_capacity(
            "anthropic.claude-opus-4-6-v1:0", pricing_data, ref
        )

        assert result["supported"] is True
        assert len(result["regions"]) >= 10
        assert "1_month" in result["commitments"]
        assert "3_month" in result["commitments"]

    def test_claude_sonnet_45_has_reserved(self, pricing_data):
        ref = {"provider": "Anthropic", "model_key": "anthropic.claude-sonnet-4-5"}
        result = build_reserved_capacity(
            "anthropic.claude-sonnet-4-5-v1:0", pricing_data, ref
        )

        assert result["supported"] is True
        assert len(result["regions"]) >= 10
        assert "1_month" in result["commitments"]
        assert "3_month" in result["commitments"]

    def test_claude_opus_45_has_reserved(self, pricing_data):
        ref = {"provider": "Anthropic", "model_key": "anthropic.claude-opus-4-5"}
        result = build_reserved_capacity(
            "anthropic.claude-opus-4-5-v1:0", pricing_data, ref
        )

        assert result["supported"] is True
        assert len(result["regions"]) >= 3
        assert len(result["commitments"]) >= 1

    def test_claude_haiku_45_has_reserved(self, pricing_data):
        ref = {"provider": "Anthropic", "model_key": "anthropic.claude-haiku-4-5"}
        result = build_reserved_capacity(
            "anthropic.claude-haiku-4-5-v1:0", pricing_data, ref
        )

        assert result["supported"] is True
        assert len(result["regions"]) >= 5
        assert len(result["commitments"]) >= 1

    def test_claude_3_haiku_no_reserved(self, pricing_data):
        """Legacy model should NOT have reserved capacity."""
        ref = {"provider": "Anthropic", "model_key": "anthropic.claude-3-haiku"}
        result = build_reserved_capacity(
            "anthropic.claude-3-haiku-20240307-v1:0", pricing_data, ref
        )

        assert result["supported"] is False
        assert result["regions"] == []
        assert result["commitments"] == []

    def test_claude_3_sonnet_no_reserved(self, pricing_data):
        """Legacy model should NOT have reserved capacity."""
        ref = {"provider": "Anthropic", "model_key": "anthropic.claude-3-sonnet"}
        result = build_reserved_capacity(
            "anthropic.claude-3-sonnet-20240229-v1:0", pricing_data, ref
        )

        assert result["supported"] is False
        assert result["regions"] == []
        assert result["commitments"] == []

    def test_unknown_model_no_reserved(self, pricing_data):
        result = build_reserved_capacity(
            "some.unknown-model-v1:0", pricing_data, None
        )

        assert result["supported"] is False

    def test_consumption_options_for_sonnet_46(self, pricing_data):
        """Claude Sonnet 4.6 should get 'reserved' in consumption_options."""
        ref = {"provider": "Anthropic", "model_key": "anthropic.claude-sonnet-4-6"}
        result = get_consumption_options(
            ["ON_DEMAND"], pricing_data, ref, mantle_supported=False
        )

        assert "on_demand" in result
        assert "reserved" in result

    def test_consumption_options_for_claude_3_haiku(self, pricing_data):
        """Claude 3 Haiku should NOT get 'reserved' in consumption_options."""
        ref = {"provider": "Anthropic", "model_key": "anthropic.claude-3-haiku"}
        result = get_consumption_options(
            ["ON_DEMAND"], pricing_data, ref, mantle_supported=False
        )

        assert "on_demand" in result
        assert "reserved" not in result

    def test_full_availability_for_sonnet_46(self, pricing_data):
        """End-to-end: build_reserved_capacity -> build_availability for Sonnet 4.6."""
        ref = {"provider": "Anthropic", "model_key": "anthropic.claude-sonnet-4-6"}
        reserved = build_reserved_capacity(
            "anthropic.claude-sonnet-4-6-v1:0", pricing_data, ref
        )

        availability = build_availability(
            regional_availability=["us-east-1", "us-west-2", "eu-west-1"],
            cross_region_data={"supported": True, "source_regions": ["us-east-1"], "profiles": []},
            batch_inference_data={"supported": True, "supported_regions": ["us-east-1", "us-west-2"]},
            provisioned_data={"supported": False, "provisioned_regions": []},
            mantle_data={"supported": True, "mantle_regions": ["us-east-1", "us-west-2"]},
            reserved_data=reserved,
        )

        assert availability["reserved"]["supported"] is True
        assert len(availability["reserved"]["regions"]) >= 10
        assert "1_month" in availability["reserved"]["commitments"]
        assert "3_month" in availability["reserved"]["commitments"]
        # Other sections unaffected
        assert availability["on_demand"]["supported"] is True
        assert availability["batch"]["supported"] is True
        assert availability["mantle"]["supported"] is True
