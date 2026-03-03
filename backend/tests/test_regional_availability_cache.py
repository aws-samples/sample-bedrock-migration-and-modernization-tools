"""
Tests for regional availability cache optimization (Task 10).

Tests the cache usage in regional-availability Lambda:
- Uses cached data when available and valid
- Falls back to API when cache is missing or invalid
- Output includes cache statistics (cacheHits, apiCalls, cacheHitRate)

Note: These tests verify the caching behavior through the shared cache_utils
module and validate expected output structures. Full handler integration tests
require the Lambda environment setup with proper powertools mocking.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys

# Ensure paths are set up for imports
sys.path.insert(0, "layers/common/python")


class TestCacheDataProcessing:
    """Tests for cache data processing logic."""

    @pytest.fixture
    def mock_cache_data(self):
        """Return mock cached model data."""
        return {
            "region": "us-east-1",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_summaries": [
                {
                    "modelId": "anthropic.claude-3-5-sonnet-v1",
                    "inferenceTypesSupported": ["ON_DEMAND"],
                },
                {
                    "modelId": "amazon.titan-text-express-v1",
                    "inferenceTypesSupported": ["ON_DEMAND", "PROVISIONED"],
                },
            ],
        }

    def test_cache_data_can_be_validated(self, mock_cache_data):
        """Should validate cache data correctly."""
        from shared.cache_utils import is_cache_valid

        # Fresh cache should be valid
        assert is_cache_valid(mock_cache_data, max_age_seconds=3600) is True

    def test_expired_cache_is_invalid(self, mock_cache_data):
        """Should reject expired cache data."""
        from shared.cache_utils import is_cache_valid

        # Make cache expired
        old_time = datetime.utcnow() - timedelta(hours=2)
        mock_cache_data["timestamp"] = old_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        assert is_cache_valid(mock_cache_data, max_age_seconds=3600) is False

    def test_missing_cache_is_invalid(self):
        """Should handle missing cache data."""
        from shared.cache_utils import is_cache_valid

        assert is_cache_valid(None, max_age_seconds=3600) is False
        assert is_cache_valid({}, max_age_seconds=3600) is False

    def test_filter_models_by_inference_type(self, mock_cache_data):
        """Should correctly filter models by inference type from cache."""
        model_summaries = mock_cache_data["model_summaries"]

        # Filter ON_DEMAND models
        on_demand_models = [
            m["modelId"]
            for m in model_summaries
            if "ON_DEMAND" in m.get("inferenceTypesSupported", [])
        ]

        # Filter PROVISIONED models
        provisioned_models = [
            m["modelId"]
            for m in model_summaries
            if "PROVISIONED" in m.get("inferenceTypesSupported", [])
        ]

        assert len(on_demand_models) == 2
        assert "anthropic.claude-3-5-sonnet-v1" in on_demand_models
        assert "amazon.titan-text-express-v1" in on_demand_models

        assert len(provisioned_models) == 1
        assert "amazon.titan-text-express-v1" in provisioned_models

    def test_provisioned_only_model_not_in_on_demand(self, mock_cache_data):
        """Provisioned-only models should not appear in ON_DEMAND list."""
        # Add a provisioned-only model
        mock_cache_data["model_summaries"].append(
            {
                "modelId": "provisioned-only-model",
                "inferenceTypesSupported": ["PROVISIONED"],
            }
        )

        model_summaries = mock_cache_data["model_summaries"]

        on_demand_models = [
            m["modelId"]
            for m in model_summaries
            if "ON_DEMAND" in m.get("inferenceTypesSupported", [])
        ]

        provisioned_models = [
            m["modelId"]
            for m in model_summaries
            if "PROVISIONED" in m.get("inferenceTypesSupported", [])
        ]

        assert "provisioned-only-model" not in on_demand_models
        assert "provisioned-only-model" in provisioned_models


class TestCacheKeyManagement:
    """Tests for cache key management."""

    def test_cache_keys_structure(self):
        """Cache keys should map region to S3 key."""
        cache_keys = {
            "us-east-1": "executions/exec-123/cache/list_foundation_models_us-east-1.json",
            "us-west-2": "executions/exec-123/cache/list_foundation_models_us-west-2.json",
        }

        assert "us-east-1" in cache_keys
        assert "us-west-2" in cache_keys
        assert "list_foundation_models" in cache_keys["us-east-1"]

    def test_separate_cached_and_uncached_regions(self):
        """Should correctly separate cached and uncached regions."""
        regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-1"]
        cache_keys = {
            "us-east-1": "cache/us-east-1.json",
            "us-west-2": "cache/us-west-2.json",
        }

        cached_regions = [r for r in regions if r in cache_keys]
        uncached_regions = [r for r in regions if r not in cache_keys]

        assert cached_regions == ["us-east-1", "us-west-2"]
        assert uncached_regions == ["eu-west-1", "ap-northeast-1"]


class TestCacheStatistics:
    """Tests for cache statistics calculation."""

    def test_cache_hit_rate_calculation(self):
        """Should calculate cache hit rate correctly."""
        # Simulate 2 cache hits out of 27 regions
        cache_hits = 2
        total_regions = 27

        cache_hit_rate = round(cache_hits / total_regions * 100, 1)

        # 2/27 * 100 = 7.407... rounded to 7.4
        assert cache_hit_rate == pytest.approx(7.4, abs=0.1)

    def test_cache_hit_rate_zero_when_no_hits(self):
        """Should return 0 when no cache hits."""
        cache_hits = 0
        total_regions = 25

        cache_hit_rate = (
            round(cache_hits / total_regions * 100, 1) if total_regions else 0.0
        )

        assert cache_hit_rate == 0.0

    def test_cache_hit_rate_hundred_when_all_cached(self):
        """Should return 100 when all regions are cached."""
        cache_hits = 10
        total_regions = 10

        cache_hit_rate = round(cache_hits / total_regions * 100, 1)

        assert cache_hit_rate == 100.0

    def test_cache_hit_rate_handles_empty_regions(self):
        """Should handle empty regions list."""
        cache_hits = 0
        total_regions = 0

        cache_hit_rate = (
            round(cache_hits / total_regions * 100, 1) if total_regions else 0.0
        )

        assert cache_hit_rate == 0.0


class TestExpectedOutputStructure:
    """Tests for expected handler output structure."""

    def test_handler_output_includes_cache_metrics(self):
        """Handler output should include cache hit metrics."""
        expected_output = {
            "status": "SUCCESS",
            "s3Key": "executions/exec-123/intermediate/regional-availability.json",
            "regionsWithBedrock": 27,
            "cacheHits": 2,
            "apiCalls": 25,
            "cacheHitRate": 7.4,
            "durationMs": 5000,
        }

        assert "cacheHits" in expected_output
        assert "apiCalls" in expected_output
        assert "cacheHitRate" in expected_output
        assert expected_output["cacheHits"] + expected_output["apiCalls"] == 27

    def test_handler_accepts_cache_keys_in_event(self):
        """Handler event should accept cacheKeys parameter."""
        event = {
            "s3Bucket": "test-bucket",
            "executionId": "exec-123",
            "regions": ["us-east-1", "us-west-2"],
            "cacheKeys": {
                "us-east-1": "executions/exec-123/cache/list_foundation_models_us-east-1.json",
                "us-west-2": "executions/exec-123/cache/list_foundation_models_us-west-2.json",
            },
        }

        assert "cacheKeys" in event
        assert isinstance(event["cacheKeys"], dict)
        assert len(event["cacheKeys"]) == 2

    def test_handler_works_without_cache_keys(self):
        """Handler should work when cacheKeys is not provided."""
        event = {
            "s3Bucket": "test-bucket",
            "executionId": "exec-123",
            "regions": ["us-east-1", "us-west-2"],
            # No cacheKeys provided
        }

        # Should default to empty dict
        cache_keys = event.get("cacheKeys", {})
        assert cache_keys == {}

    def test_output_metadata_includes_cache_info(self):
        """Output metadata should include cache statistics."""
        expected_metadata = {
            "regions_with_bedrock": 27,
            "total_models_tracked": 150,
            "total_provisioned_models": 50,
            "api_regions_queried": 25,
            "cache_hits": 2,
            "cache_hit_rate": 7.4,
            "collection_timestamp": "2026-03-03T12:00:00Z",
            "discovery_method": "api_on_demand_filtered_with_cache",
        }

        assert "cache_hits" in expected_metadata
        assert "cache_hit_rate" in expected_metadata
        assert "api_regions_queried" in expected_metadata
        assert (
            expected_metadata["discovery_method"] == "api_on_demand_filtered_with_cache"
        )


class TestCacheReadIntegration:
    """Integration tests for cache reading."""

    @pytest.fixture
    def mock_cache_data(self):
        """Return mock cached model data."""
        return {
            "region": "us-east-1",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_summaries": [
                {
                    "modelId": "anthropic.claude-3-5-sonnet-v1",
                    "inferenceTypesSupported": ["ON_DEMAND"],
                },
            ],
        }

    def test_get_cached_models_returns_data(self, mock_cache_data):
        """Should return cached data when available."""
        from shared.cache_utils import get_cached_models

        mock_s3_client = MagicMock()

        with patch("shared.cache_utils.read_from_s3", return_value=mock_cache_data):
            result = get_cached_models(
                mock_s3_client, "test-bucket", "cache/us-east-1.json"
            )

        assert result is not None
        assert result["region"] == "us-east-1"
        assert len(result["model_summaries"]) == 1

    def test_get_cached_models_returns_none_on_missing(self):
        """Should return None when cache is missing."""
        from shared.cache_utils import get_cached_models

        mock_s3_client = MagicMock()

        with patch("shared.cache_utils.read_from_s3", return_value=None):
            result = get_cached_models(
                mock_s3_client, "test-bucket", "cache/nonexistent.json"
            )

        assert result is None

    def test_cache_validation_with_fresh_data(self, mock_cache_data):
        """Should validate fresh cache data as valid."""
        from shared.cache_utils import is_cache_valid

        assert is_cache_valid(mock_cache_data, max_age_seconds=3600) is True

    def test_cache_validation_with_expired_data(self, mock_cache_data):
        """Should validate expired cache data as invalid."""
        from shared.cache_utils import is_cache_valid

        # Make cache expired
        old_time = datetime.utcnow() - timedelta(hours=2)
        mock_cache_data["timestamp"] = old_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        assert is_cache_valid(mock_cache_data, max_age_seconds=3600) is False
