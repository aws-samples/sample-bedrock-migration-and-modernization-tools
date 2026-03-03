"""Tests for externalized URL configuration (Task 05).

Tests the URL externalization in config_loader.py:
- get_external_url method
- Specific URL accessor methods (get_bulk_pricing_url, get_litellm_url, etc.)
- Default values when config is missing
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

sys.path.insert(0, "layers/common/python")


@pytest.fixture
def mock_config_with_urls():
    """Return a mock configuration with external URLs."""
    return {
        "external_urls": {
            "pricing": {
                "bulk_pricing_api": "https://custom-pricing-url.com/api",
                "pricing_api_region": "us-west-2",
            },
            "documentation": {
                "bedrock_models_supported": "https://custom-docs.com/models",
                "bedrock_pricing": "https://custom-docs.com/pricing",
                "bedrock_model_lifecycle": "https://custom-docs.com/lifecycle",
                "bedrock_model_ids": "https://custom-docs.com/model-ids",
            },
            "external_data_sources": {
                "litellm_model_prices": "https://custom-litellm.com/prices.json",
                "litellm_model_prices_fallback": "https://custom-litellm.com/backup.json",
            },
        }
    }


@pytest.fixture
def empty_config():
    """Return an empty configuration."""
    return {}


@pytest.fixture
def config_without_external_urls():
    """Return a configuration without external_urls section."""
    return {
        "version": "1.0.0",
        "provider_configuration": {},
        "region_configuration": {},
    }


# ============================================================================
# Tests for get_external_url method
# ============================================================================


class TestGetExternalUrl:
    """Tests for get_external_url method."""

    def test_get_external_url_returns_configured_value(self, mock_config_with_urls):
        """Should return the configured URL value."""
        from shared.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = mock_config_with_urls

        result = loader.get_external_url("pricing", "bulk_pricing_api")

        assert result == "https://custom-pricing-url.com/api"

    def test_get_external_url_returns_default_when_missing(self, empty_config):
        """Should return default when URL not in config."""
        from shared.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = empty_config

        result = loader.get_external_url(
            "pricing", "bulk_pricing_api", "https://default.com"
        )

        assert result == "https://default.com"

    def test_get_external_url_returns_none_when_no_default(self, empty_config):
        """Should return None when URL not in config and no default provided."""
        from shared.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = empty_config

        result = loader.get_external_url("pricing", "bulk_pricing_api")

        assert result is None

    def test_get_external_url_handles_missing_category(self, mock_config_with_urls):
        """Should return default when category doesn't exist."""
        from shared.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = mock_config_with_urls

        result = loader.get_external_url(
            "nonexistent_category", "some_key", "https://fallback.com"
        )

        assert result == "https://fallback.com"


# ============================================================================
# Tests for specific URL accessor methods
# ============================================================================


class TestSpecificUrlAccessors:
    """Tests for specific URL accessor methods."""

    def test_get_bulk_pricing_url_returns_correct_url(self, mock_config_with_urls):
        """Should return bulk pricing URL from config."""
        from shared.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = mock_config_with_urls

        result = loader.get_bulk_pricing_url()

        assert result == "https://custom-pricing-url.com/api"

    def test_get_bulk_pricing_url_returns_default_when_missing(self, empty_config):
        """Should return default bulk pricing URL when not in config."""
        from shared.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = empty_config

        result = loader.get_bulk_pricing_url()

        # Should return the default URL template
        assert "pricing.us-east-1.amazonaws.com" in result
        assert "{service_code}" in result

    def test_get_litellm_url_returns_correct_url(self, mock_config_with_urls):
        """Should return LiteLLM URL from config."""
        from shared.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = mock_config_with_urls

        result = loader.get_litellm_url()

        assert result == "https://custom-litellm.com/prices.json"

    def test_get_litellm_url_returns_default_when_missing(self, empty_config):
        """Should return default LiteLLM URL when not in config."""
        from shared.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = empty_config

        result = loader.get_litellm_url()

        # Should return the default GitHub URL
        assert "raw.githubusercontent.com/BerriAI/litellm" in result
        assert "model_prices_and_context_window.json" in result

    def test_get_litellm_fallback_url_returns_correct_url(self, mock_config_with_urls):
        """Should return LiteLLM fallback URL from config."""
        from shared.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = mock_config_with_urls

        result = loader.get_litellm_fallback_url()

        assert result == "https://custom-litellm.com/backup.json"

    def test_get_litellm_fallback_url_returns_default_when_missing(self, empty_config):
        """Should return default LiteLLM fallback URL when not in config."""
        from shared.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = empty_config

        result = loader.get_litellm_fallback_url()

        # Should return the default backup URL
        assert "raw.githubusercontent.com/BerriAI/litellm" in result
        assert "backup" in result

    def test_get_documentation_url_returns_correct_url(self, mock_config_with_urls):
        """Should return documentation URL from config."""
        from shared.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = mock_config_with_urls

        result = loader.get_documentation_url("bedrock_model_lifecycle")

        assert result == "https://custom-docs.com/lifecycle"

    def test_get_documentation_url_returns_default_for_known_key(self, empty_config):
        """Should return default documentation URL for known keys."""
        from shared.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = empty_config

        result = loader.get_documentation_url("bedrock_models_supported")

        # Should return the default AWS docs URL
        assert "docs.aws.amazon.com/bedrock" in result
        assert "models-supported" in result

    def test_get_documentation_url_returns_default_for_pricing(self, empty_config):
        """Should return default pricing documentation URL."""
        from shared.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = empty_config

        result = loader.get_documentation_url("bedrock_pricing")

        # Should return the default AWS pricing URL
        assert "aws.amazon.com/bedrock/pricing" in result

    def test_get_documentation_url_returns_default_for_model_ids(self, empty_config):
        """Should return default model IDs documentation URL."""
        from shared.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = empty_config

        result = loader.get_documentation_url("bedrock_model_ids")

        # Should return the default model IDs URL
        assert "docs.aws.amazon.com/bedrock" in result
        assert "model-ids" in result
