"""
Tests for config-sync Lambda (Task 03).

Tests the config sync capabilities:
- Frontend config extraction from backend config
- JavaScript constants generation
- Lambda handler S3 writes
"""

import sys
import importlib.util
import json
import pytest
from unittest.mock import Mock, patch


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_backend_config():
    """Return a mock backend configuration."""
    return {
        "version": "1.0.0",
        "region_configuration": {
            "region_locations": {
                "us-east-1": "US East (N. Virginia)",
                "eu-west-1": "Europe (Ireland)",
            },
            "region_coordinates": {
                "us-east-1": {
                    "lat": 38.95,
                    "lng": -77.45,
                    "name": "N. Virginia",
                    "geo": "US",
                },
                "eu-west-1": {
                    "lat": 53.35,
                    "lng": -6.26,
                    "name": "Ireland",
                    "geo": "EU",
                },
            },
            "aws_regions": [
                {"value": "us-east-1", "label": "N. Virginia (us-east-1)", "geo": "US"},
                {"value": "eu-west-1", "label": "Ireland (eu-west-1)", "geo": "EU"},
            ],
            "geo_region_options": [
                {"value": "All Regions", "label": "All Regions"},
                {"value": "US", "label": "United States"},
            ],
            "geo_prefix_map": {"US": "us-", "EU": "eu-"},
        },
        "provider_configuration": {
            "provider_colors": {"Amazon": "#FF9900", "Anthropic": "#D4A27F"},
            "documentation_links": {
                "Amazon": {"aws_bedrock_guide": "https://docs.aws.amazon.com/bedrock/"}
            },
        },
        "model_configuration": {
            "context_window_thresholds": {
                "small": 32000,
                "medium": 128000,
                "large": 500000,
            },
            "model_families": ["claude", "titan", "nova"],
            "model_variants": ["haiku", "sonnet", "opus"],
        },
    }


@pytest.fixture
def config_sync_handler():
    """Import and return the config-sync handler module."""
    spec = importlib.util.spec_from_file_location(
        "config_sync_handler", "lambdas/config-sync/handler.py"
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ImportError("Could not load config-sync handler")


# ============================================================================
# Tests for Frontend Config Extraction
# ============================================================================


class TestExtractFrontendConfig:
    """Tests for extract_frontend_config function."""

    def test_extract_frontend_config_extracts_regions(
        self, mock_backend_config, config_sync_handler
    ):
        """Should extract regions from backend config."""
        # Act
        result = config_sync_handler.extract_frontend_config(mock_backend_config)

        # Assert
        assert "regions" in result
        assert "us-east-1" in result["regions"]
        assert result["regions"]["us-east-1"]["geo"] == "US"
        assert result["regions"]["us-east-1"]["lat"] == 38.95
        assert result["regions"]["us-east-1"]["lng"] == -77.45

    def test_extract_frontend_config_extracts_providers(
        self, mock_backend_config, config_sync_handler
    ):
        """Should extract provider colors from backend config."""
        # Act
        result = config_sync_handler.extract_frontend_config(mock_backend_config)

        # Assert
        assert "providers" in result
        assert "colors" in result["providers"]
        assert result["providers"]["colors"]["Amazon"] == "#FF9900"
        assert result["providers"]["colors"]["Anthropic"] == "#D4A27F"

    def test_extract_frontend_config_extracts_context_thresholds(
        self, mock_backend_config, config_sync_handler
    ):
        """Should extract context window thresholds from backend config."""
        # Act
        result = config_sync_handler.extract_frontend_config(mock_backend_config)

        # Assert
        assert "model_config" in result
        assert "context_thresholds" in result["model_config"]
        assert result["model_config"]["context_thresholds"]["small"] == 32000
        assert result["model_config"]["context_thresholds"]["medium"] == 128000
        assert result["model_config"]["context_thresholds"]["large"] == 500000


# ============================================================================
# Tests for JavaScript Constants Generation
# ============================================================================


class TestGenerateJsConstants:
    """Tests for generate_js_constants function."""

    def test_generate_js_constants_valid_javascript(
        self, mock_backend_config, config_sync_handler
    ):
        """Should produce valid JavaScript syntax."""
        # Arrange
        frontend_config = config_sync_handler.extract_frontend_config(
            mock_backend_config
        )

        # Act
        js_content = config_sync_handler.generate_js_constants(frontend_config)

        # Assert - check for valid export statements
        assert "export const providerColors" in js_content
        assert "export const regionCoordinates" in js_content
        assert "export const awsRegions" in js_content
        # Check that it contains actual data
        assert "#FF9900" in js_content  # Amazon color
        assert "us-east-1" in js_content

    def test_generate_js_constants_includes_exports(
        self, mock_backend_config, config_sync_handler
    ):
        """Should include proper ES6 exports."""
        # Arrange
        frontend_config = config_sync_handler.extract_frontend_config(
            mock_backend_config
        )

        # Act
        js_content = config_sync_handler.generate_js_constants(frontend_config)

        # Assert - count export statements
        export_count = js_content.count("export const")
        assert export_count >= 5  # At least 5 exports expected

        # Check specific exports
        assert "export const providerColors" in js_content
        assert "export const regionCoordinates" in js_content
        assert "export const awsRegions" in js_content
        assert "export const geoRegionOptions" in js_content
        assert "export const geoPrefixMap" in js_content
        assert "export const contextWindowThresholds" in js_content
        assert "export const configMetadata" in js_content


# ============================================================================
# Tests for Lambda Handler
# ============================================================================


class TestLambdaHandler:
    """Tests for lambda_handler function."""

    def test_lambda_handler_writes_frontend_config(
        self, mock_backend_config, config_sync_handler
    ):
        """Should write frontend-config.json to S3."""
        # Mock the config loader
        mock_config_loader = Mock()
        mock_config_loader.config = mock_backend_config

        mock_s3_client = Mock()
        mock_write_to_s3 = Mock()

        with patch.object(
            config_sync_handler, "get_config_loader", return_value=mock_config_loader
        ):
            with patch.object(
                config_sync_handler, "get_s3_client", return_value=mock_s3_client
            ):
                with patch.object(config_sync_handler, "write_to_s3", mock_write_to_s3):
                    # Arrange
                    event = {"s3Bucket": "test-bucket", "executionId": "test-exec-123"}

                    # Act
                    result = config_sync_handler.lambda_handler(event, None)

                    # Assert
                    assert result["status"] == "SUCCESS"
                    assert (
                        result["frontendConfigS3Key"] == "config/frontend-config.json"
                    )

                    # Verify write_to_s3 was called with frontend config
                    mock_write_to_s3.assert_called_once()
                    call_args = mock_write_to_s3.call_args
                    assert call_args[0][2] == "config/frontend-config.json"

    def test_lambda_handler_generates_js_when_requested(
        self, mock_backend_config, config_sync_handler
    ):
        """Should write JS file when generateJs=true."""
        # Mock the config loader
        mock_config_loader = Mock()
        mock_config_loader.config = mock_backend_config

        mock_s3_client = Mock()
        mock_write_to_s3 = Mock()

        with patch.object(
            config_sync_handler, "get_config_loader", return_value=mock_config_loader
        ):
            with patch.object(
                config_sync_handler, "get_s3_client", return_value=mock_s3_client
            ):
                with patch.object(config_sync_handler, "write_to_s3", mock_write_to_s3):
                    # Arrange
                    event = {
                        "s3Bucket": "test-bucket",
                        "executionId": "test-exec-123",
                        "generateJs": True,
                    }

                    # Act
                    result = config_sync_handler.lambda_handler(event, None)

                    # Assert
                    assert result["status"] == "SUCCESS"
                    assert (
                        result["frontendConfigS3Key"] == "config/frontend-config.json"
                    )
                    assert result["jsConstantsS3Key"] == "config/generated-constants.js"

                    # Verify both files were written
                    # write_to_s3 for JSON, put_object for JS
                    mock_write_to_s3.assert_called_once()
                    mock_s3_client.put_object.assert_called_once()

                    # Check JS file was written with correct content type
                    js_call = mock_s3_client.put_object.call_args
                    assert js_call[1]["Key"] == "config/generated-constants.js"
                    assert js_call[1]["ContentType"] == "application/javascript"
