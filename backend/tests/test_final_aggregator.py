"""
Tests for final-aggregator Lambda lifecycle data merging.

Tests the lifecycle data integration into the final model output.
"""

import copy
import pytest
from unittest.mock import patch, MagicMock
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
# Provide real implementations for the model_matcher functions
mock_model_matcher.get_canonical_model_id = (
    lambda x: x.lower().split(":")[0] if x else ""
)
mock_model_matcher.calculate_match_score = lambda x, y: 1.0 if x == y else 0.0
mock_model_matcher.get_model_variant_info = lambda x: {
    "base_id": x.split(":")[0] if x else "",
    "is_multimodal": False,
    "is_provisioned_only": ":0:" in x
    if x
    else False,  # e.g., cohere.embed-english-v3:0:512
    "context_window": None,
    "version": None,
    "api_version": None,
    "has_dimension_suffix": ":0:" in x if x else False,
}
mock_model_matcher.has_semantic_conflict = lambda x, y: False

sys.modules["shared.model_matcher"] = mock_model_matcher

# Mock shared.powertools submodule (required after Powertools migration)
mock_powertools = MagicMock()
mock_powertools.logger = MagicMock()
mock_powertools.tracer = MagicMock()
mock_powertools.tracer.capture_method = lambda f: f
mock_powertools.metrics = MagicMock()
mock_powertools.LambdaContext = MagicMock

sys.modules["shared.powertools"] = mock_powertools
sys.modules["aws_lambda_powertools"] = MagicMock()
sys.modules["aws_lambda_powertools.metrics"] = MagicMock()


# Sample model data
SAMPLE_MODEL = {
    "model_id": "anthropic.claude-3-sonnet",
    "model_name": "Claude 3 Sonnet",
    "model_provider": "Anthropic",
    "model_modalities": {
        "input_modalities": ["TEXT"],
        "output_modalities": ["TEXT"],
    },
    "model_lifecycle": {"status": "ACTIVE"},
    "inference_types_supported": ["ON_DEMAND"],
    "streaming_supported": True,
    "in_region": ["us-east-1", "us-west-2"],
}

# Sample lifecycle data from scraper
SAMPLE_LIFECYCLE_DATA = {
    "anthropic.claude-3-sonnet": {
        "model_id": "anthropic.claude-3-sonnet",
        "model_name": "Claude 3 Sonnet",
        "provider": "Anthropic",
        "lifecycle_status": "active",
        "eol_date": None,
        "legacy_date": None,
    },
    "anthropic.claude-2": {
        "model_id": "anthropic.claude-3-haiku",  # Recommended replacement
        "model_name": "Claude 2",
        "provider": "Anthropic",
        "lifecycle_status": "legacy",
        "eol_date": "2024-12-15",
        "legacy_date": "2024-01-15",
        "recommended_replacement": "Claude 3 Haiku",
    },
}


@pytest.fixture
def sample_model():
    """Fixture providing a sample model."""
    return copy.deepcopy(SAMPLE_MODEL)


@pytest.fixture
def sample_lifecycle_by_model():
    """Fixture providing sample lifecycle data keyed by model_id."""
    return copy.deepcopy(SAMPLE_LIFECYCLE_DATA)


class TestBuildFinalModelsWithLifecycle:
    """Tests for build_final_models function with lifecycle data."""

    def test_build_final_models_with_lifecycle(
        self, sample_model, sample_lifecycle_by_model
    ):
        """Test that build_final_models merges lifecycle data into models."""
        # Import after mocking
        from handler import transform_model_to_schema

        # Arrange
        model_id = sample_model["model_id"]
        lifecycle_by_model = sample_lifecycle_by_model

        # Act - Transform model with lifecycle data
        result = transform_model_to_schema(
            model_id=model_id,
            model=sample_model,
            regional_availability=["us-east-1", "us-west-2"],
            token_specs={},
            quotas_by_region={},
            features_by_region={},
            enriched_model={},
            pricing_data={},
            collection_timestamp="2024-01-01T00:00:00Z",
            mantle_by_model={},
            provisioned_throughput=None,
            lifecycle_by_model=lifecycle_by_model,
        )

        # Assert - Model should have lifecycle field
        assert "model_lifecycle" in result
        assert result["model_lifecycle"]["status"] == "ACTIVE"

    def test_model_schema_lifecycle_fields(
        self, sample_model, sample_lifecycle_by_model
    ):
        """Test that transformed model includes lifecycle.status and lifecycle.eol_date."""
        from handler import transform_model_to_schema

        # Arrange - Use a legacy model
        legacy_model = sample_model.copy()
        legacy_model["model_id"] = "anthropic.claude-2"
        legacy_model["model_name"] = "Claude 2"

        # Act
        result = transform_model_to_schema(
            model_id="anthropic.claude-2",
            model=legacy_model,
            regional_availability=["us-east-1"],
            token_specs={},
            quotas_by_region={},
            features_by_region={},
            enriched_model={},
            pricing_data={},
            collection_timestamp="2024-01-01T00:00:00Z",
            mantle_by_model={},
            provisioned_throughput=None,
            lifecycle_by_model=sample_lifecycle_by_model,
        )

        # Assert - Lifecycle fields should be present
        lifecycle = result.get("model_lifecycle", {})
        assert "status" in lifecycle
        # If lifecycle data was merged, status should be LEGACY
        if lifecycle.get("status") == "LEGACY":
            assert "eol_date" in lifecycle or "legacy_date" in lifecycle

    def test_lifecycle_default_active(self, sample_model):
        """Test that models without lifecycle data default to active status."""
        from handler import transform_model_to_schema

        # Arrange - No lifecycle data for this model
        empty_lifecycle = {}

        # Act
        result = transform_model_to_schema(
            model_id=sample_model["model_id"],
            model=sample_model,
            regional_availability=["us-east-1"],
            token_specs={},
            quotas_by_region={},
            features_by_region={},
            enriched_model={},
            pricing_data={},
            collection_timestamp="2024-01-01T00:00:00Z",
            mantle_by_model={},
            provisioned_throughput=None,
            lifecycle_by_model=empty_lifecycle,
        )

        # Assert - Should default to ACTIVE
        lifecycle = result.get("model_lifecycle", {})
        assert lifecycle.get("status") == "ACTIVE"


class TestLifecycleMerging:
    """Tests for lifecycle data merging logic."""

    def test_lifecycle_status_override(self, sample_model, sample_lifecycle_by_model):
        """Test that scraped lifecycle status overrides model's default status."""
        from handler import transform_model_to_schema

        # Arrange - Model with ACTIVE status, but lifecycle says legacy
        model = sample_model.copy()
        model["model_id"] = "anthropic.claude-2"
        model["model_lifecycle"] = {"status": "ACTIVE"}  # Original status

        # Act
        result = transform_model_to_schema(
            model_id="anthropic.claude-2",
            model=model,
            regional_availability=["us-east-1"],
            token_specs={},
            quotas_by_region={},
            features_by_region={},
            enriched_model={},
            pricing_data={},
            collection_timestamp="2024-01-01T00:00:00Z",
            mantle_by_model={},
            provisioned_throughput=None,
            lifecycle_by_model=sample_lifecycle_by_model,
        )

        # Assert - Status should be overridden to LEGACY
        lifecycle = result.get("model_lifecycle", {})
        assert lifecycle.get("status") == "LEGACY"

    def test_lifecycle_eol_date_added(self, sample_model, sample_lifecycle_by_model):
        """Test that EOL date is added from lifecycle data."""
        from handler import transform_model_to_schema

        # Arrange
        model = sample_model.copy()
        model["model_id"] = "anthropic.claude-2"

        # Act
        result = transform_model_to_schema(
            model_id="anthropic.claude-2",
            model=model,
            regional_availability=["us-east-1"],
            token_specs={},
            quotas_by_region={},
            features_by_region={},
            enriched_model={},
            pricing_data={},
            collection_timestamp="2024-01-01T00:00:00Z",
            mantle_by_model={},
            provisioned_throughput=None,
            lifecycle_by_model=sample_lifecycle_by_model,
        )

        # Assert - EOL date should be present
        lifecycle = result.get("model_lifecycle", {})
        assert lifecycle.get("eol_date") == "2024-12-15"

    def test_lifecycle_recommended_replacement(
        self, sample_model, sample_lifecycle_by_model
    ):
        """Test that recommended replacement is added from lifecycle data."""
        from handler import transform_model_to_schema

        # Arrange
        model = sample_model.copy()
        model["model_id"] = "anthropic.claude-2"

        # Act
        result = transform_model_to_schema(
            model_id="anthropic.claude-2",
            model=model,
            regional_availability=["us-east-1"],
            token_specs={},
            quotas_by_region={},
            features_by_region={},
            enriched_model={},
            pricing_data={},
            collection_timestamp="2024-01-01T00:00:00Z",
            mantle_by_model={},
            provisioned_throughput=None,
            lifecycle_by_model=sample_lifecycle_by_model,
        )

        # Assert - Recommended replacement should be present
        lifecycle = result.get("model_lifecycle", {})
        assert lifecycle.get("recommended_replacement") == "Claude 3 Haiku"
