"""
Tests for enhanced gap detection Lambda (Task 01).

Tests the new gap detection capabilities:
- Context window mismatch detection
- Unknown service code detection
- Frontend config drift detection
- Trigger decision with new gap types
"""

import sys
import importlib
import pytest
from unittest.mock import Mock, patch


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_config():
    """Return a mock configuration dictionary."""
    return {
        "version": "1.0.0-test",
        "gap_detection_config": {
            "context_window_variance_threshold": 0.1,
            "enable_frontend_drift_detection": True,
            "enable_context_window_detection": True,
            "enable_service_code_detection": True,
        },
        "model_configuration": {
            "context_window_specs": {
                "anthropic.claude-3-5-sonnet": {"standard_context": 200000},
                "anthropic.claude-3-haiku": {"standard_context": 200000},
                "meta.llama3": {"standard_context": 128000},
            }
        },
        "pricing_service_codes": ["AmazonBedrock", "AmazonBedrockService"],
        "region_configuration": {
            "region_locations": {
                "us-east-1": "US East (N. Virginia)",
                "us-west-2": "US West (Oregon)",
                "eu-west-1": "Europe (Ireland)",
            }
        },
        "provider_configuration": {
            "provider_colors": {
                "Amazon": "#FF9900",
                "Anthropic": "#D4A27F",
            }
        },
    }


@pytest.fixture
def mock_models_data():
    """Return mock models data with context window variance."""
    return {
        "providers": {
            "Anthropic": {
                "models": {
                    "anthropic.claude-3-5-sonnet-v1": {
                        "model_name": "Claude 3.5 Sonnet",
                        "model_provider": "Anthropic",
                        "converse_data": {
                            "context_window": 220000  # 10% higher than config (200000)
                        },
                    }
                }
            }
        }
    }


@pytest.fixture
def mock_pricing_data():
    """Return mock pricing data."""
    return {
        "providers": {
            "Anthropic": {
                "claude-3-5-sonnet": {
                    "service_code": "AmazonBedrock",
                    "regions": {"us-east-1": {"price": 0.003}},
                }
            }
        }
    }


@pytest.fixture
def mock_frontend_config():
    """Return mock frontend config for drift detection."""
    return {
        "regions": {
            "us-east-1": {"label": "N. Virginia", "geo": "US"},
            "us-west-2": {"label": "Oregon", "geo": "US"},
        },
        "providers": {
            "Amazon": {"color": "#FF9900"},
            "Anthropic": {"color": "#D4A27F"},
        },
    }


@pytest.fixture
def gap_detection_handler():
    """Import and return the gap-detection handler module."""
    # Ensure the correct path is first
    if "lambdas/gap-detection" not in sys.path:
        sys.path.insert(0, "lambdas/gap-detection")

    # Force reimport to get the correct module
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gap_detection_handler", "lambdas/gap-detection/handler.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================================
# Tests for Context Window Mismatch Detection
# ============================================================================


class TestContextWindowMismatchDetection:
    """Tests for detect_context_window_mismatches function."""

    def test_detect_context_window_mismatches_finds_variance(
        self, mock_config, gap_detection_handler
    ):
        """Should detect models with context window variance > threshold."""
        # Arrange - model with 15% variance (above 10% threshold)
        # The model ID must contain the config spec key for matching
        models_data = {
            "providers": {
                "Anthropic": {
                    "models": {
                        "anthropic.claude-3-5-sonnet-20240620-v1:0": {
                            "model_name": "Claude 3.5 Sonnet",
                            "model_provider": "Anthropic",
                            "converse_data": {
                                "context_window": 230000  # 15% higher than config (200000)
                            },
                        }
                    }
                }
            }
        }

        # Act
        result = gap_detection_handler.detect_context_window_mismatches(
            models_data, mock_config
        )

        # Assert
        assert len(result) == 1
        assert "anthropic.claude-3-5-sonnet" in result[0]["model_id"]
        assert result[0]["variance"] > 0.1  # Above threshold
        assert result[0]["actual_value"] == 230000
        assert result[0]["config_value"] == 200000

    def test_detect_context_window_mismatches_ignores_small_variance(
        self, mock_config, gap_detection_handler
    ):
        """Should ignore models with variance below threshold."""
        # Arrange - model with only 2.5% variance (below 10% threshold)
        models_data = {
            "providers": {
                "Anthropic": {
                    "models": {
                        "anthropic.claude-3-5-sonnet-v1": {
                            "converse_data": {"context_window": 205000}  # 2.5% variance
                        }
                    }
                }
            }
        }

        # Act
        result = gap_detection_handler.detect_context_window_mismatches(
            models_data, mock_config
        )

        # Assert
        assert len(result) == 0


# ============================================================================
# Tests for Unknown Service Code Detection
# ============================================================================


class TestServiceCodeDetection:
    """Tests for detect_unknown_service_codes function."""

    def test_detect_unknown_service_codes_finds_new_codes(
        self, mock_config, gap_detection_handler
    ):
        """Should identify service codes not in config."""
        # Arrange - pricing data with unknown service code
        pricing_data = {
            "providers": {
                "Anthropic": {
                    "claude-3-5-sonnet": {"service_code": "AmazonBedrockNewService"}
                }
            }
        }

        # Act
        result = gap_detection_handler.detect_unknown_service_codes(
            pricing_data, mock_config
        )

        # Assert
        assert "AmazonBedrockNewService" in result

    def test_detect_unknown_service_codes_all_known(
        self, mock_config, gap_detection_handler
    ):
        """Should return empty list when all codes are known."""
        # Arrange - pricing data with known service code
        pricing_data = {
            "providers": {
                "Anthropic": {"claude-3-5-sonnet": {"service_code": "AmazonBedrock"}}
            }
        }

        # Act
        result = gap_detection_handler.detect_unknown_service_codes(
            pricing_data, mock_config
        )

        # Assert
        assert len(result) == 0


# ============================================================================
# Tests for Frontend Config Drift Detection
# ============================================================================


class TestFrontendConfigDriftDetection:
    """Tests for detect_frontend_config_drift function."""

    def test_detect_frontend_config_drift_finds_missing_regions(
        self, mock_config, mock_frontend_config, gap_detection_handler
    ):
        """Should detect regions in backend but missing from frontend."""
        # Arrange - mock S3 client and read_from_s3
        mock_s3_client = Mock()

        with patch.object(
            gap_detection_handler, "read_from_s3", return_value=mock_frontend_config
        ):
            # Act
            result = gap_detection_handler.detect_frontend_config_drift(
                mock_config,
                "config/frontend-config.json",
                mock_s3_client,
                "test-bucket",
            )

            # Assert
            assert result["drift_detected"] is True
            # eu-west-1 is in backend but not in frontend
            assert "eu-west-1" in result["regions_missing_in_frontend"]

    def test_detect_frontend_config_drift_no_drift(
        self, mock_config, gap_detection_handler
    ):
        """Should return drift_detected=false when configs match."""
        # Arrange - frontend config that matches backend
        matching_frontend_config = {
            "regions": {
                "us-east-1": {"label": "N. Virginia"},
                "us-west-2": {"label": "Oregon"},
                "eu-west-1": {"label": "Ireland"},
            },
            "providers": {
                "Amazon": {"color": "#FF9900"},
                "Anthropic": {"color": "#D4A27F"},
            },
        }

        mock_s3_client = Mock()

        with patch.object(
            gap_detection_handler, "read_from_s3", return_value=matching_frontend_config
        ):
            # Act
            result = gap_detection_handler.detect_frontend_config_drift(
                mock_config,
                "config/frontend-config.json",
                mock_s3_client,
                "test-bucket",
            )

            # Assert
            assert result["drift_detected"] is False
            assert result["regions_missing_in_frontend"] == []
            assert result["regions_extra_in_frontend"] == []


# ============================================================================
# Tests for Trigger Decision
# ============================================================================


class TestTriggerDecision:
    """Tests for determine_trigger_decision function."""

    def test_trigger_decision_includes_new_gap_types(self, gap_detection_handler):
        """Should include new gap types in trigger decision reasons."""
        # Mock the config loader
        mock_config_loader = Mock()
        mock_config_loader.get_agent_thresholds.return_value = {
            "unmatched_models_trigger": 5,
            "low_confidence_threshold": 0.6,
            "max_low_confidence_matches": 3,
            "new_provider_trigger": True,
        }

        with patch.object(
            gap_detection_handler, "get_config_loader", return_value=mock_config_loader
        ):
            # Arrange - analysis with all new gap types
            analysis = {
                "models_without_pricing": [],
                "low_confidence_matches": [],
                "unknown_providers": [],
                "new_models": [],
                "context_window_mismatches": [
                    {"model_id": "test-model", "variance": 0.15}
                ],
                "unknown_service_codes": ["NewServiceCode"],
                "frontend_config_drift": {"drift_detected": True},
            }

            # Act
            result = gap_detection_handler.determine_trigger_decision(analysis)

            # Assert
            assert result["should_trigger"] is True
            reasons_text = " ".join(result["reasons"])
            assert "context window" in reasons_text.lower()
            assert (
                "service code" in reasons_text.lower()
                or "NewServiceCode" in reasons_text
            )
            assert "drift" in reasons_text.lower()
