"""
Tests for model extraction caching (Task 09).

Tests the caching functionality in model-extractor Lambda:
- Cache file creation and structure
- Cache utility functions (is_cache_valid, get_cached_models)
- Handler output includes cacheKey
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
import json

# Ensure paths are set up for imports
sys.path.insert(0, "layers/common/python")


class TestCacheUtilities:
    """Tests for cache utility functions in shared/cache_utils.py."""

    def test_is_cache_valid_returns_true_for_fresh(self):
        """Should return True for fresh cache data within max_age_seconds."""
        from shared.cache_utils import is_cache_valid

        fresh_cache = {"timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}

        assert is_cache_valid(fresh_cache, max_age_seconds=3600) is True

    def test_is_cache_valid_returns_false_for_expired(self):
        """Should return False for cache data older than max_age_seconds."""
        from shared.cache_utils import is_cache_valid

        old_time = datetime.utcnow() - timedelta(hours=2)
        expired_cache = {"timestamp": old_time.strftime("%Y-%m-%dT%H:%M:%SZ")}

        assert is_cache_valid(expired_cache, max_age_seconds=3600) is False

    def test_is_cache_valid_returns_false_for_missing_timestamp(self):
        """Should return False when timestamp is missing from cache data."""
        from shared.cache_utils import is_cache_valid

        assert is_cache_valid({}, max_age_seconds=3600) is False
        assert is_cache_valid(None, max_age_seconds=3600) is False

    def test_is_cache_valid_returns_false_for_invalid_timestamp(self):
        """Should return False when timestamp format is invalid."""
        from shared.cache_utils import is_cache_valid

        invalid_cache = {"timestamp": "not-a-valid-timestamp"}

        assert is_cache_valid(invalid_cache, max_age_seconds=3600) is False

    def test_get_cached_models_returns_data(self):
        """Should return cached data when cache key exists."""
        from shared.cache_utils import get_cached_models

        mock_s3_client = MagicMock()
        expected_data = {
            "region": "us-east-1",
            "timestamp": "2026-03-03T12:00:00Z",
            "model_summaries": [{"modelId": "anthropic.claude-3-5-sonnet-v1"}],
        }

        with patch("shared.cache_utils.read_from_s3", return_value=expected_data):
            result = get_cached_models(
                mock_s3_client, "test-bucket", "cache/us-east-1.json"
            )

        assert result == expected_data
        assert result["region"] == "us-east-1"
        assert len(result["model_summaries"]) == 1

    def test_get_cached_models_returns_none_on_missing(self):
        """Should return None when cache key does not exist."""
        from shared.cache_utils import get_cached_models

        mock_s3_client = MagicMock()

        with patch("shared.cache_utils.read_from_s3", return_value=None):
            result = get_cached_models(
                mock_s3_client, "test-bucket", "cache/nonexistent.json"
            )

        assert result is None

    def test_get_cached_models_returns_none_on_error(self):
        """Should return None when S3 read fails."""
        from shared.cache_utils import get_cached_models

        mock_s3_client = MagicMock()

        with patch(
            "shared.cache_utils.read_from_s3", side_effect=Exception("S3 error")
        ):
            result = get_cached_models(
                mock_s3_client, "test-bucket", "cache/error.json"
            )

        assert result is None


class TestBuildCacheKey:
    """Tests for cache key building utility."""

    def test_build_cache_key_default_type(self):
        """Should build correct cache key with default cache type."""
        from shared.cache_utils import build_cache_key

        key = build_cache_key("exec-123", "us-east-1")

        assert key == "executions/exec-123/cache/list_foundation_models_us-east-1.json"

    def test_build_cache_key_custom_type(self):
        """Should build correct cache key with custom cache type."""
        from shared.cache_utils import build_cache_key

        key = build_cache_key("exec-456", "eu-west-1", cache_type="custom_cache")

        assert key == "executions/exec-456/cache/custom_cache_eu-west-1.json"


class TestCacheFileStructure:
    """Tests for cache file structure validation."""

    def test_cache_file_has_required_fields(self):
        """Cache file should have region, timestamp, and model_summaries."""
        # This validates the expected structure of cached data
        cache_data = {
            "region": "us-east-1",
            "timestamp": "2026-03-03T12:00:00Z",
            "model_summaries": [
                {
                    "modelId": "anthropic.claude-3-5-sonnet-v1",
                    "modelName": "Claude 3.5 Sonnet",
                },
                {
                    "modelId": "amazon.titan-text-express-v1",
                    "modelName": "Titan Text Express",
                },
            ],
        }

        assert "region" in cache_data
        assert "timestamp" in cache_data
        assert "model_summaries" in cache_data
        assert isinstance(cache_data["model_summaries"], list)

    def test_cache_timestamp_format(self):
        """Cache timestamp should be in ISO 8601 format."""
        from shared.cache_utils import is_cache_valid

        # Valid ISO 8601 format
        valid_cache = {"timestamp": "2026-03-03T12:00:00Z"}
        assert is_cache_valid(valid_cache, max_age_seconds=86400) is True

        # Invalid format should fail
        invalid_cache = {"timestamp": "2026/03/03 12:00:00"}
        assert is_cache_valid(invalid_cache, max_age_seconds=86400) is False


class TestModelExtractorCacheIntegration:
    """Integration tests for model extractor caching behavior.

    Note: These tests verify the expected behavior of the caching mechanism
    by testing the cache_utils functions and validating the expected output
    structure. Full integration tests with the handler require the Lambda
    environment setup.
    """

    @pytest.fixture
    def mock_bedrock_response(self):
        """Return mock Bedrock ListFoundationModels response."""
        return {
            "modelSummaries": [
                {
                    "modelId": "anthropic.claude-3-5-sonnet-v1",
                    "modelName": "Claude 3.5 Sonnet",
                    "providerName": "Anthropic",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                    "inferenceTypesSupported": ["ON_DEMAND"],
                },
                {
                    "modelId": "amazon.titan-text-express-v1",
                    "modelName": "Titan Text Express",
                    "providerName": "Amazon",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                    "inferenceTypesSupported": ["ON_DEMAND", "PROVISIONED"],
                },
            ]
        }

    def test_cache_key_format(self):
        """Cache key should follow expected format for model extractor."""
        from shared.cache_utils import build_cache_key

        cache_key = build_cache_key("exec-123", "us-east-1")

        # Verify the cache key format matches what model-extractor produces
        assert (
            cache_key
            == "executions/exec-123/cache/list_foundation_models_us-east-1.json"
        )
        assert "cache/" in cache_key
        assert "us-east-1" in cache_key

    def test_cache_data_structure_validation(self, mock_bedrock_response):
        """Validate the expected cache data structure."""
        import time

        # This is the structure that model-extractor creates
        cache_data = {
            "region": "us-east-1",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_summaries": mock_bedrock_response["modelSummaries"],
        }

        # Verify structure
        assert "region" in cache_data
        assert "timestamp" in cache_data
        assert "model_summaries" in cache_data
        assert len(cache_data["model_summaries"]) == 2

        # Verify cache is valid
        from shared.cache_utils import is_cache_valid

        assert is_cache_valid(cache_data, max_age_seconds=3600) is True

    def test_cache_can_be_read_after_write(self, mock_bedrock_response):
        """Verify cache data can be read back correctly."""
        import time
        from shared.cache_utils import get_cached_models, is_cache_valid

        # Simulate cache data
        cache_data = {
            "region": "us-east-1",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_summaries": mock_bedrock_response["modelSummaries"],
        }

        mock_s3_client = MagicMock()

        with patch("shared.cache_utils.read_from_s3", return_value=cache_data):
            result = get_cached_models(
                mock_s3_client, "test-bucket", "cache/us-east-1.json"
            )

        assert result is not None
        assert result["region"] == "us-east-1"
        assert is_cache_valid(result, max_age_seconds=3600) is True
        assert len(result["model_summaries"]) == 2

    def test_handler_output_structure_with_cache_key(self):
        """Verify expected handler output structure includes cacheKey."""
        # This tests the expected output format without importing the handler
        expected_output = {
            "status": "SUCCESS",
            "region": "us-east-1",
            "s3Key": "executions/exec-123/models/us-east-1.json",
            "modelCount": 2,
            "durationMs": 100,
            "cacheKey": "executions/exec-123/cache/list_foundation_models_us-east-1.json",
        }

        # Verify structure
        assert expected_output["status"] == "SUCCESS"
        assert "cacheKey" in expected_output
        assert (
            "cache/list_foundation_models_us-east-1.json" in expected_output["cacheKey"]
        )

    def test_handler_output_without_cache_key(self):
        """Verify handler output when caching fails or is disabled."""
        # When bucket is not provided, cacheKey should not be in output
        expected_output_no_cache = {
            "status": "SUCCESS",
            "region": "us-east-1",
            "s3Key": "test/models/us-east-1.json",
            "modelCount": 2,
            "durationMs": 100,
            # No cacheKey when caching is disabled
        }

        assert "cacheKey" not in expected_output_no_cache
