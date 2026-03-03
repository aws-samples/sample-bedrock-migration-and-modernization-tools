"""
Tests for enhanced self-healing agent Lambda (Task 02).

Tests the new self-healing capabilities:
- Context window update application
- Service code addition
- Suggestion validation with thresholds
- Prompt building with all gap types
- Config backup creation
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
def mock_config():
    """Return a mock configuration dictionary."""
    return {
        "version": "1.0.0-test",
        "model_configuration": {
            "context_window_specs": {
                "anthropic.claude-3-5-sonnet": {"standard_context": 200000},
                "anthropic.claude-3-haiku": {"standard_context": 200000},
                "meta.llama3": {"standard_context": 128000},
                "model1": {},
                "model2": {},
                "model3": {},
                "model4": {},
                "model5": {},
                "model6": {},
                "model7": {},
                "model8": {},
                "model9": {},
                "model10": {},
            }
        },
        "pricing_service_codes": ["AmazonBedrock", "AmazonBedrockService"],
        "agent_configuration": {
            "auto_apply_rules": {
                "max_models_affected_for_auto_apply": 0.2,
                "safe_changes": [
                    "provider_pattern_addition",
                    "provider_alias_addition",
                    "context_window_update",
                    "service_code_addition",
                ],
            }
        },
        "provider_configuration": {
            "provider_patterns": {"Amazon": ["titan", "nova"]},
            "provider_aliases": {"amazon": ["amazon", "aws"]},
        },
    }


@pytest.fixture
def mock_gap_report():
    """Return a mock gap report with all gap types."""
    return {
        "execution_id": "test-exec-123",
        "details": {
            "models_without_pricing": [
                {"model_id": "test-model-1", "model_name": "Test Model 1"}
            ],
            "unknown_providers": ["NewProvider"],
            "low_confidence_matches": [{"model_id": "test-model-2", "confidence": 0.5}],
            "new_models": ["new-model-1", "new-model-2"],
            "context_window_mismatches": [
                {
                    "model_id": "anthropic.claude-3-5-sonnet-v1",
                    "actual_value": 220000,
                    "config_value": 200000,
                    "variance": 0.1,
                }
            ],
            "unknown_service_codes": ["AmazonBedrockNewService"],
            "frontend_config_drift": {
                "drift_detected": True,
                "regions_missing_in_frontend": ["eu-west-1"],
            },
        },
        "trigger_decision": {"should_trigger": True},
    }


@pytest.fixture
def self_healing_handler():
    """Import and return the self-healing-agent handler module."""
    spec = importlib.util.spec_from_file_location(
        "self_healing_handler", "lambdas/self-healing-agent/handler.py"
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ImportError("Could not load self-healing-agent handler")


# ============================================================================
# Tests for Context Window Update
# ============================================================================


class TestContextWindowUpdate:
    """Tests for apply_context_window_update function."""

    def test_apply_context_window_update_success(self, self_healing_handler):
        """Should correctly update context window specs."""
        # Arrange - use a simple key without dots to avoid path parsing issues
        suggestion = {
            "target_config_path": "model_configuration.context_window_specs.claude-new-model",
            "suggested_value": {"standard_context": 300000, "source": "litellm"},
        }
        current_config = {"model_configuration": {"context_window_specs": {}}}

        # Act
        success, message = self_healing_handler.apply_context_window_update(
            suggestion, current_config
        )

        # Assert
        assert success is True
        # The function extracts the model key from path_parts[2]
        assert (
            "claude-new-model"
            in current_config["model_configuration"]["context_window_specs"]
        )
        spec = current_config["model_configuration"]["context_window_specs"][
            "claude-new-model"
        ]
        assert spec["standard_context"] == 300000
        assert spec["source"] == "litellm"

    def test_apply_context_window_update_invalid_path(self, self_healing_handler):
        """Should reject suggestions with invalid target path."""
        # Arrange - path doesn't start with model_configuration.context_window_specs
        suggestion = {
            "target_config_path": "invalid.path.here",
            "suggested_value": {"standard_context": 300000},
        }
        current_config = {}

        # Act
        success, message = self_healing_handler.apply_context_window_update(
            suggestion, current_config
        )

        # Assert
        assert success is False
        assert "Invalid" in message


# ============================================================================
# Tests for Service Code Addition
# ============================================================================


class TestServiceCodeAddition:
    """Tests for apply_service_code_addition function."""

    def test_apply_service_code_addition_success(
        self, mock_config, self_healing_handler
    ):
        """Should add new service codes without duplicates."""
        # Arrange
        suggestion = {
            "suggested_value": ["AmazonBedrockNewService", "AnotherNewService"]
        }
        current_config = {
            "pricing_service_codes": ["AmazonBedrock", "AmazonBedrockService"]
        }

        # Act
        success, message = self_healing_handler.apply_service_code_addition(
            suggestion, current_config
        )

        # Assert
        assert success is True
        assert "AmazonBedrockNewService" in current_config["pricing_service_codes"]
        assert "AnotherNewService" in current_config["pricing_service_codes"]
        # Original codes should still be there
        assert "AmazonBedrock" in current_config["pricing_service_codes"]

    def test_apply_service_code_addition_no_duplicates(
        self, mock_config, self_healing_handler
    ):
        """Should skip existing codes and return False if no new codes."""
        # Arrange - all codes already exist
        suggestion = {"suggested_value": ["AmazonBedrock", "AmazonBedrockService"]}
        current_config = {
            "pricing_service_codes": ["AmazonBedrock", "AmazonBedrockService"]
        }

        # Act
        success, message = self_healing_handler.apply_service_code_addition(
            suggestion, current_config
        )

        # Assert
        assert success is False
        assert "No new" in message


# ============================================================================
# Tests for Suggestion Validation
# ============================================================================


class TestValidateSuggestion:
    """Tests for validate_suggestion function."""

    def test_validate_suggestion_enforces_threshold(
        self, mock_config, self_healing_handler
    ):
        """Should reject suggestions affecting too many models."""
        # Arrange - suggestion affects 5 out of 13 models (38%, above 20% threshold)
        suggestion = {
            "type": "provider_pattern_modification",
            "affected_models": ["model1", "model2", "model3", "model4", "model5"],
        }

        # Act
        is_valid, reason = self_healing_handler.validate_suggestion(
            suggestion, mock_config
        )

        # Assert
        assert is_valid is False
        assert "Affects" in reason

    def test_validate_suggestion_passes_valid(self, mock_config, self_healing_handler):
        """Should pass valid suggestions."""
        # Arrange - valid context window update affecting only 1 model
        suggestion = {
            "type": "context_window_update",
            "affected_models": ["model1"],
            "suggested_value": {"standard_context": 300000},
        }

        # Act
        is_valid, reason = self_healing_handler.validate_suggestion(
            suggestion, mock_config
        )

        # Assert
        assert is_valid is True
        assert reason == "Valid"


# ============================================================================
# Tests for Prompt Building
# ============================================================================


class TestBuildAnalysisPrompt:
    """Tests for build_analysis_prompt function."""

    def test_build_analysis_prompt_includes_all_gap_types(
        self, mock_gap_report, mock_config, self_healing_handler
    ):
        """Should include all new gap types in the prompt."""
        # Act
        prompt = self_healing_handler.build_analysis_prompt(
            mock_gap_report, mock_config
        )

        # Assert - check that all gap type sections are present
        assert "Context Window Mismatches" in prompt
        assert "Unknown Service Codes" in prompt
        assert "Frontend Config Drift" in prompt
        # Also check original gap types
        assert "Models Without Pricing" in prompt
        assert "Unknown Providers" in prompt
        assert "Low Confidence Matches" in prompt
        assert "New Models Detected" in prompt


# ============================================================================
# Tests for Safe Suggestions Application
# ============================================================================


class TestApplySafeSuggestions:
    """Tests for apply_safe_suggestions function."""

    def test_apply_safe_suggestions_creates_backup(
        self, mock_config, self_healing_handler
    ):
        """Should create config backup before auto-applying changes."""
        # Mock the config loader
        mock_config_loader = Mock()
        mock_config_loader.get_agent_config.return_value = {
            "auto_apply_rules": {
                "safe_changes": [
                    "provider_pattern_addition",
                    "context_window_update",
                    "service_code_addition",
                ]
            }
        }

        with patch.object(
            self_healing_handler, "_get_config", return_value=mock_config_loader
        ):
            # Arrange
            suggestions = [
                {
                    "id": "sugg-001",
                    "type": "context_window_update",
                    "auto_apply_safe": True,
                    "target_config_path": "model_configuration.context_window_specs.new-model",
                    "suggested_value": {"standard_context": 300000},
                    "affected_models": ["new-model"],
                }
            ]

            mock_s3_client = Mock()
            mock_write_to_s3 = Mock()

            with patch.object(self_healing_handler, "write_to_s3", mock_write_to_s3):
                # Act
                result = self_healing_handler.apply_safe_suggestions(
                    suggestions, mock_config.copy(), mock_s3_client, "test-bucket"
                )

                # Assert - backup should be created
                assert result["config_modified"] is True
                assert "sugg-001" in result["applied"]

                # Check that write_to_s3 was called (for backup and new config)
                assert mock_write_to_s3.call_count >= 1

                # Find the backup call
                backup_calls = [
                    call
                    for call in mock_write_to_s3.call_args_list
                    if "config-history" in str(call)
                ]
                assert len(backup_calls) >= 1
