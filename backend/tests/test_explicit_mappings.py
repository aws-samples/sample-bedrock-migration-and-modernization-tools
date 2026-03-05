"""
Tests for Phase 1 Task 02: Explicit Model Mappings

This module tests the explicit model mappings configuration for Z.AI and Moonshot AI
models that was added to support pricing matching for these providers.

Tests validate:
- Config file contains explicit mappings for Z.AI GLM models
- Config file contains explicit mapping for Moonshot AI Kimi model
- All mapping values are valid non-empty strings
- Existing Claude mappings are preserved (backward compatibility)
"""

import json
import pytest
from pathlib import Path


class TestExplicitMappingsConfig:
    """Tests for explicit model mappings in profiler-config.json."""

    @pytest.fixture
    def config(self):
        """Load the profiler configuration file."""
        config_path = Path(__file__).parent.parent / "config" / "profiler-config.json"
        with open(config_path) as f:
            return json.load(f)

    @pytest.fixture
    def mappings(self, config):
        """Extract explicit_model_mappings from config."""
        return config.get("matching_configuration", {}).get(
            "explicit_model_mappings", {}
        )

    def test_zai_glm_mapping_exists(self, mappings):
        """Z.AI GLM model should have explicit mapping."""
        # Arrange
        expected_key = "zai.glm-4.7-v1:0"

        # Act
        zai_keys = [
            k for k in mappings.keys() if "zai" in k.lower() and "glm" in k.lower()
        ]

        # Assert
        assert len(zai_keys) >= 1, (
            f"No Z.AI GLM mapping found. Available keys: {list(mappings.keys())}"
        )
        assert expected_key in mappings, (
            f"Expected '{expected_key}' in mappings, found Z.AI keys: {zai_keys}"
        )

    def test_zai_glm_flash_mapping_exists(self, mappings):
        """Z.AI GLM Flash model should have explicit mapping."""
        # Arrange
        expected_key = "zai.glm-4.7-flash-v1:0"

        # Act
        flash_keys = [
            k
            for k in mappings.keys()
            if "flash" in k.lower() and ("zai" in k.lower() or "glm" in k.lower())
        ]

        # Assert
        assert len(flash_keys) >= 1, (
            f"No Z.AI GLM Flash mapping found. Available keys: {list(mappings.keys())}"
        )
        assert expected_key in mappings, (
            f"Expected '{expected_key}' in mappings, found flash keys: {flash_keys}"
        )

    def test_moonshot_kimi_mapping_exists(self, mappings):
        """Moonshot AI Kimi model should have explicit mapping."""
        # Arrange
        expected_key = "moonshotai.kimi-k2.5-v1:0"

        # Act
        moonshot_keys = [
            k for k in mappings.keys() if "moonshot" in k.lower() or "kimi" in k.lower()
        ]

        # Assert
        assert len(moonshot_keys) >= 1, (
            f"No Moonshot/Kimi mapping found. Available keys: {list(mappings.keys())}"
        )
        assert expected_key in mappings, (
            f"Expected '{expected_key}' in mappings, found moonshot keys: {moonshot_keys}"
        )

    def test_mapping_values_are_valid(self, mappings):
        """All mapping values should be non-empty strings."""
        # Arrange & Act & Assert
        for key, value in mappings.items():
            assert isinstance(value, str), (
                f"Mapping value for '{key}' is not a string, got: {type(value).__name__}"
            )
            assert len(value) > 0, f"Mapping value for '{key}' is empty"

    def test_existing_claude_mappings_preserved(self, mappings):
        """Existing Claude mappings should still exist (backward compatibility)."""
        # Arrange
        claude_keys = [k for k in mappings.keys() if "claude" in k.lower()]

        # Assert
        assert len(claude_keys) >= 1, (
            f"Claude mappings were removed. Available keys: {list(mappings.keys())}"
        )

    def test_zai_glm_mapping_value_format(self, mappings):
        """Z.AI GLM mapping value should follow expected format."""
        # Arrange
        key = "zai.glm-4.7-v1:0"

        # Act
        value = mappings.get(key)

        # Assert
        assert value is not None, f"Mapping for '{key}' not found"
        assert value.startswith("zai."), (
            f"Expected mapping value to start with 'zai.', got: {value}"
        )
        assert "glm" in value.lower(), f"Expected 'glm' in mapping value, got: {value}"

    def test_zai_glm_flash_mapping_value_format(self, mappings):
        """Z.AI GLM Flash mapping value should follow expected format."""
        # Arrange
        key = "zai.glm-4.7-flash-v1:0"

        # Act
        value = mappings.get(key)

        # Assert
        assert value is not None, f"Mapping for '{key}' not found"
        assert value.startswith("zai."), (
            f"Expected mapping value to start with 'zai.', got: {value}"
        )
        assert "flash" in value.lower(), (
            f"Expected 'flash' in mapping value, got: {value}"
        )

    def test_moonshot_kimi_mapping_value_format(self, mappings):
        """Moonshot Kimi mapping value should follow expected format."""
        # Arrange
        key = "moonshotai.kimi-k2.5-v1:0"

        # Act
        value = mappings.get(key)

        # Assert
        assert value is not None, f"Mapping for '{key}' not found"
        assert value.startswith("moonshot."), (
            f"Expected mapping value to start with 'moonshot.', got: {value}"
        )
        assert "kimi" in value.lower(), (
            f"Expected 'kimi' in mapping value, got: {value}"
        )
