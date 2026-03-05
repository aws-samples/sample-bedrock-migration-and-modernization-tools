"""
Tests for Phase 1 Task 01: Provider Aliases

This module tests the provider alias configuration for kimi-ai and moonshotai
that was added to support Moonshot AI model matching.

Tests validate:
- Config file contains kimi-ai and moonshotai aliases for moonshot ai
- PROVIDER_ALIASES dict maps these aliases to moonshot
- get_canonical_model_id normalizes moonshotai/kimi-ai prefixes to moonshot
"""

import json
import pytest
import importlib.util
from pathlib import Path


# Load the model_matcher module directly from file (bypasses __init__.py and powertools dependency)
MODEL_MATCHER_PATH = (
    Path(__file__).parent.parent
    / "layers"
    / "common"
    / "python"
    / "shared"
    / "model_matcher.py"
)

spec = importlib.util.spec_from_file_location("model_matcher", MODEL_MATCHER_PATH)
model_matcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_matcher)

# Import functions and constants under test
get_canonical_model_id = model_matcher.get_canonical_model_id
PROVIDER_ALIASES = model_matcher.PROVIDER_ALIASES


class TestProviderAliasesConfig:
    """Tests for provider alias configuration in profiler-config.json."""

    @pytest.fixture
    def config(self):
        """Load the profiler configuration file."""
        config_path = Path(__file__).parent.parent / "config" / "profiler-config.json"
        with open(config_path) as f:
            return json.load(f)

    def test_kimi_ai_alias_in_config(self, config):
        """kimi-ai should be in moonshot ai aliases in config file."""
        # Arrange
        aliases = config["provider_configuration"]["provider_aliases"]["moonshot ai"]

        # Act & Assert
        assert "kimi-ai" in aliases, (
            f"Expected 'kimi-ai' in moonshot ai aliases, got: {aliases}"
        )

    def test_moonshotai_alias_in_config(self, config):
        """moonshotai should be in moonshot ai aliases in config file."""
        # Arrange
        aliases = config["provider_configuration"]["provider_aliases"]["moonshot ai"]

        # Act & Assert
        assert "moonshotai" in aliases, (
            f"Expected 'moonshotai' in moonshot ai aliases, got: {aliases}"
        )

    def test_kimi_alias_in_config(self, config):
        """kimi (without -ai) should also be in moonshot ai aliases."""
        # Arrange
        aliases = config["provider_configuration"]["provider_aliases"]["moonshot ai"]

        # Act & Assert
        # Check for either 'kimi' or 'kimi ai' (space variant)
        has_kimi = "kimi" in aliases or "kimi ai" in aliases
        assert has_kimi, (
            f"Expected 'kimi' or 'kimi ai' in moonshot ai aliases, got: {aliases}"
        )


class TestProviderAliasesDict:
    """Tests for PROVIDER_ALIASES dictionary in model_matcher.py."""

    def test_kimi_ai_in_provider_aliases(self):
        """kimi-ai should map to moonshot in PROVIDER_ALIASES."""
        # Arrange & Act
        result = PROVIDER_ALIASES.get("kimi-ai")

        # Assert
        assert result == "moonshot", (
            f"Expected 'kimi-ai' to map to 'moonshot', got: {result}"
        )

    def test_moonshotai_in_provider_aliases(self):
        """moonshotai should map to moonshot in PROVIDER_ALIASES."""
        # Arrange & Act
        result = PROVIDER_ALIASES.get("moonshotai")

        # Assert
        assert result == "moonshot", (
            f"Expected 'moonshotai' to map to 'moonshot', got: {result}"
        )

    def test_moonshot_in_provider_aliases(self):
        """moonshot should map to moonshot in PROVIDER_ALIASES (identity mapping)."""
        # Arrange & Act
        result = PROVIDER_ALIASES.get("moonshot")

        # Assert
        assert result == "moonshot", (
            f"Expected 'moonshot' to map to 'moonshot', got: {result}"
        )


class TestCanonicalModelIdNormalization:
    """Tests for get_canonical_model_id normalization of provider aliases."""

    def test_canonical_moonshotai_model(self):
        """moonshotai.kimi-k2.5 should normalize to moonshot.kimi-k2.5."""
        # Arrange
        input_id = "moonshotai.kimi-k2.5"

        # Act
        result = get_canonical_model_id(input_id)

        # Assert
        assert result.startswith("moonshot."), (
            f"Expected result to start with 'moonshot.', got: {result}"
        )
        assert "kimi" in result, f"Expected 'kimi' in result, got: {result}"

    def test_canonical_kimi_ai_model(self):
        """kimi-ai.model-name should normalize to moonshot.model-name."""
        # Arrange
        input_id = "kimi-ai.model-name"

        # Act
        result = get_canonical_model_id(input_id)

        # Assert
        assert result.startswith("moonshot."), (
            f"Expected result to start with 'moonshot.', got: {result}"
        )

    def test_canonical_preserves_model_name(self):
        """Canonicalization should preserve the model name part."""
        # Arrange
        input_id = "moonshotai.kimi-k2.5-thinking"

        # Act
        result = get_canonical_model_id(input_id)

        # Assert
        assert "kimi" in result, f"Expected 'kimi' in result, got: {result}"
        assert "thinking" in result, f"Expected 'thinking' in result, got: {result}"

    def test_canonical_moonshotai_with_version_suffix(self):
        """moonshotai.kimi-k2.5-v1:0 should normalize correctly."""
        # Arrange
        input_id = "moonshotai.kimi-k2.5-v1:0"

        # Act
        result = get_canonical_model_id(input_id)

        # Assert
        assert result.startswith("moonshot."), (
            f"Expected result to start with 'moonshot.', got: {result}"
        )
        # Version suffix :0 should be removed
        assert ":0" not in result, f"Expected ':0' to be removed, got: {result}"
