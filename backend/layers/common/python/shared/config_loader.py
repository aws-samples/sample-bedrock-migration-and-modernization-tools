"""
Configuration Loader for Bedrock Model Profiler.

Loads externalized configuration from S3 with fallback to embedded defaults.
This allows dynamic updates to provider patterns, region lists, and other
configuration without code changes.
"""

import json
import logging
import os
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Embedded defaults - used as fallback if S3 config is unavailable
DEFAULT_CONFIG = {
    "version": "1.0.0-embedded",
    "provider_configuration": {
        "provider_aliases": {
            "amazon": ["amazon", "aws"],
            "anthropic": ["anthropic"],
            "meta": ["meta", "facebook"],
            "mistral ai": ["mistral", "mistralai", "mistral ai"],
            "stability ai": ["stability", "stabilityai", "stability ai"],
            "cohere": ["cohere"],
            "ai21 labs": ["ai21", "ai21labs", "ai21 labs"],
            "luma ai": ["luma", "lumaai", "luma ai"],
            "twelvelabs": ["twelvelabs", "twelve labs", "twelverlabs"],
            "minimax": ["minimax", "minimax ai", "minimax-ai"],
            "moonshot ai": ["moonshot", "moonshot ai", "kimi", "kimi ai"],
            "deepseek": ["deepseek"],
            "qwen": ["qwen", "qwen2", "alibaba"],
            "google": ["google"],
            "nvidia": ["nvidia"],
            "openai": ["openai"],
            "writer": ["writer"],
        },
        "provider_patterns": {
            "Amazon": ["titan", "nova", "amazon-bedrock", "rerank"],
            "Anthropic": ["claude", "anthropic"],
            "Meta": ["llama", "mllama"],
            "Mistral AI": [
                "mistral",
                "mixtral",
                "ministral",
                "magistral",
                "pixtral",
                "voxtral",
            ],
            "Cohere": ["cohere", "command", "embed"],
            "AI21 Labs": ["ai21", "jamba", "jurassic"],
            "Stability AI": ["stable", "stability", "sdxl"],
            "Luma AI": ["luma", "ray"],
            "Writer": ["writer", "palmyra"],
            "NVIDIA": ["nvidia", "nemotron"],
            "Qwen": ["qwen"],
            "OpenAI": ["gpt", "openai"],
            "DeepSeek": ["deepseek", "r1"],
            "Google": ["gemma", "gemini"],
            "TwelveLabs": ["twelve", "twelvelabs", "marengo", "pegasus"],
            "MiniMax": ["minimax"],
            "Moonshot AI": ["kimi", "moonshot"],
        },
        "explicit_provider_names": {
            "twelvelabs": "TwelveLabs",
            "twelve labs": "TwelveLabs",
            "cohere": "Cohere",
            "luma ai": "Luma AI",
            "luma": "Luma AI",
            "anthropic": "Anthropic",
            "stability ai": "Stability AI",
            "ai21 labs": "AI21 Labs",
            "ai21": "AI21 Labs",
            "mistral ai": "Mistral AI",
            "mistral": "Mistral AI",
            "deepseek": "DeepSeek",
            "writer": "Writer",
            "meta": "Meta",
            "amazon": "Amazon",
            "google": "Google",
            "nvidia": "NVIDIA",
            "openai": "OpenAI",
            "qwen": "Qwen",
            "minimax": "MiniMax",
        },
        "documentation_links": {
            "Anthropic": {
                "aws_bedrock_guide": "https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html",
                "pricing_guide": "https://aws.amazon.com/bedrock/pricing/",
            },
            "Amazon": {
                "aws_bedrock_guide": "https://docs.aws.amazon.com/bedrock/latest/userguide/titan-models.html",
                "pricing_guide": "https://aws.amazon.com/bedrock/pricing/",
            },
            "default": {
                "aws_bedrock_guide": "https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html",
                "pricing_guide": "https://aws.amazon.com/bedrock/pricing/",
            },
        },
    },
    "region_configuration": {
        "model_regions": ["us-east-1", "us-west-2"],
        "quota_regions": [
            "us-east-1",
            "us-west-2",
            "us-east-2",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "eu-central-1",
            "eu-north-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ap-southeast-3",
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-south-1",
            "ca-central-1",
            "sa-east-1",
        ],
        "feature_regions": [
            "us-east-1",
            "us-west-2",
            "us-east-2",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "eu-central-1",
            "eu-north-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ap-southeast-3",
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-south-1",
            "ca-central-1",
            "sa-east-1",
        ],
        "region_locations": {
            "us-east-1": "US East (N. Virginia)",
            "us-west-2": "US West (Oregon)",
            "eu-west-1": "Europe (Ireland)",
            "ap-northeast-1": "Asia Pacific (Tokyo)",
        },
    },
    "model_configuration": {
        "model_families": [
            "claude",
            "titan",
            "nova",
            "llama",
            "mistral",
            "command",
            "embed",
            "jamba",
            "stable",
        ],
        "model_variants": [
            "haiku",
            "sonnet",
            "opus",
            "lite",
            "pro",
            "micro",
            "premier",
            "express",
            "large",
            "small",
            "medium",
            "instant",
            "chat",
            "instruct",
            "ultra",
            "canvas",
            "reel",
        ],
        "claude_variants": ["haiku", "sonnet", "opus"],
        "nova_variants": ["micro", "lite", "pro", "premier", "canvas", "reel", "sonic"],
        "llama_sizes": ["8b", "70b", "405b", "11b", "90b", "1b", "3b"],
    },
    "matching_configuration": {
        "min_confidence_threshold": 0.7,
        "size_variance_threshold": 0.3,
        "suffixes_to_remove": [
            "-it",
            "-instruct",
            "-chat",
            "-v1",
            "-v2",
            "-v3",
            ":0",
            ":1",
            ":2",
        ],
    },
    "agent_configuration": {
        "bedrock_model_id": "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "thresholds": {
            "unmatched_models_trigger": 5,
            "low_confidence_threshold": 0.6,
            "new_provider_trigger": True,
        },
    },
}


class ConfigLoader:
    """
    Loads and provides access to profiler configuration.

    Configuration is loaded from S3 with fallback to embedded defaults.
    Uses caching to avoid repeated S3 reads within a Lambda invocation.
    """

    def __init__(
        self,
        s3_client: Any = None,
        bucket: Optional[str] = None,
        config_key: str = "config/profiler-config.json",
    ):
        """
        Initialize the config loader.

        Args:
            s3_client: Boto3 S3 client (optional, will create if not provided)
            bucket: S3 bucket name (optional, uses CONFIG_BUCKET env var)
            config_key: S3 key for the config file
        """
        self._s3_client = s3_client
        self._bucket = (
            bucket
            or os.environ.get("CONFIG_BUCKET")
            or os.environ.get("DATA_BUCKET")
            or os.environ.get("S3_BUCKET")
        )
        self._config_key = config_key
        self._config: Optional[dict] = None
        self._loaded_from_s3 = False

    def _get_s3_client(self) -> Any:
        """Get or create S3 client."""
        if self._s3_client is None:
            import boto3

            self._s3_client = boto3.client("s3")
        return self._s3_client

    def load_config(self, force_reload: bool = False) -> dict:
        """
        Load configuration from S3 with fallback to defaults.

        Args:
            force_reload: If True, bypass cache and reload from S3

        Returns:
            Configuration dictionary
        """
        if self._config is not None and not force_reload:
            return self._config

        # Try to load from S3
        if self._bucket:
            try:
                s3_client = self._get_s3_client()
                response = s3_client.get_object(
                    Bucket=self._bucket, Key=self._config_key
                )
                self._config = json.loads(response["Body"].read().decode("utf-8"))
                self._loaded_from_s3 = True
                logger.info(
                    f"Loaded config v{self._config.get('version', 'unknown')} from s3://{self._bucket}/{self._config_key}"
                )
                return self._config
            except Exception as e:
                logger.warning(f"Failed to load config from S3, using defaults: {e}")

        # Use embedded defaults
        self._config = DEFAULT_CONFIG.copy()
        self._loaded_from_s3 = False
        logger.info("Using embedded default configuration")
        return self._config

    @property
    def config(self) -> dict:
        """Get the current configuration (loads if not yet loaded)."""
        if self._config is None:
            self.load_config()
        return self._config

    @property
    def is_from_s3(self) -> bool:
        """Check if config was loaded from S3."""
        return self._loaded_from_s3

    # =========================================================================
    # Provider Configuration Accessors
    # =========================================================================

    def get_provider_aliases(self) -> dict:
        """Get provider name aliases for matching variations."""
        return self.config.get("provider_configuration", {}).get("provider_aliases", {})

    def get_provider_patterns(self) -> dict:
        """Get provider keyword patterns for detection."""
        return self.config.get("provider_configuration", {}).get(
            "provider_patterns", {}
        )

    def get_explicit_provider_names(self) -> dict:
        """Get explicit provider name mappings."""
        return self.config.get("provider_configuration", {}).get(
            "explicit_provider_names", {}
        )

    def get_provider_colors(self) -> dict:
        """Get provider brand colors."""
        return self.config.get("provider_configuration", {}).get("provider_colors", {})

    def get_documentation_links(self, provider: str = None) -> dict:
        """Get documentation links for a provider or all providers."""
        docs = self.config.get("provider_configuration", {}).get(
            "documentation_links", {}
        )
        if provider:
            return docs.get(provider, docs.get("default", {}))
        return docs

    # =========================================================================
    # Region Configuration Accessors
    # =========================================================================

    def get_region_list(self, region_type: str) -> list:
        """
        Get a list of regions by type.

        Args:
            region_type: One of 'model_regions', 'quota_regions', 'feature_regions'

        Returns:
            List of region codes
        """
        return self.config.get("region_configuration", {}).get(region_type, [])

    def get_region_locations(self) -> dict:
        """Get region code to location name mapping."""
        return self.config.get("region_configuration", {}).get("region_locations", {})

    def get_region_coordinates(self) -> dict:
        """Get region coordinates for map display."""
        return self.config.get("region_configuration", {}).get("region_coordinates", {})

    def get_aws_regions(self) -> list:
        """Get AWS regions with labels and geo info."""
        return self.config.get("region_configuration", {}).get("aws_regions", [])

    def get_geo_region_options(self) -> list:
        """Get geographic region filter options."""
        return self.config.get("region_configuration", {}).get("geo_region_options", [])

    # =========================================================================
    # Model Configuration Accessors
    # =========================================================================

    def get_model_families(self) -> list:
        """Get list of model families."""
        return self.config.get("model_configuration", {}).get("model_families", [])

    def get_model_variants(self) -> list:
        """Get list of model variants."""
        return self.config.get("model_configuration", {}).get("model_variants", [])

    def get_claude_variants(self) -> list:
        """Get Claude-specific variants."""
        return self.config.get("model_configuration", {}).get("claude_variants", [])

    def get_nova_variants(self) -> list:
        """Get Nova-specific variants."""
        return self.config.get("model_configuration", {}).get("nova_variants", [])

    def get_llama_sizes(self) -> list:
        """Get Llama model sizes."""
        return self.config.get("model_configuration", {}).get("llama_sizes", [])

    # =========================================================================
    # Matching Configuration Accessors
    # =========================================================================

    def get_min_confidence_threshold(self) -> float:
        """Get minimum confidence threshold for pricing matches."""
        return self.config.get("matching_configuration", {}).get(
            "min_confidence_threshold", 0.7
        )

    def get_size_variance_threshold(self) -> float:
        """Get size variance threshold for conflict detection."""
        return self.config.get("matching_configuration", {}).get(
            "size_variance_threshold", 0.3
        )

    def get_suffixes_to_remove(self) -> list:
        """Get list of suffixes to remove during normalization."""
        return self.config.get("matching_configuration", {}).get(
            "suffixes_to_remove", []
        )

    def get_type_conflicts(self) -> list:
        """Get type conflict definitions."""
        return self.config.get("matching_configuration", {}).get("type_conflicts", [])

    def get_explicit_model_mappings(self) -> dict:
        """Get explicit model ID to pricing key mappings."""
        return self.config.get("matching_configuration", {}).get(
            "explicit_model_mappings", {}
        )

    # =========================================================================
    # Agent Configuration Accessors
    # =========================================================================

    def get_agent_config(self) -> dict:
        """Get full agent configuration."""
        return self.config.get("agent_configuration", {})

    def get_agent_thresholds(self) -> dict:
        """Get agent trigger thresholds."""
        return self.config.get("agent_configuration", {}).get("thresholds", {})

    def get_bedrock_model_id(self) -> str:
        """Get Bedrock model ID for the self-healing agent."""
        return self.config.get("agent_configuration", {}).get(
            "bedrock_model_id", "us.anthropic.claude-opus-4-5-20251101-v1:0"
        )

    # =========================================================================
    # GovCloud Configuration Accessors
    # =========================================================================

    def get_govcloud_regions(self) -> list:
        """Get list of GovCloud regions."""
        return self.config.get("govcloud_configuration", {}).get(
            "govcloud_regions", ["us-gov-west-1", "us-gov-east-1"]
        )

    def get_govcloud_cris_models(self) -> list:
        """
        Get list of model patterns that should be marked as CRIS in GovCloud.

        Models matching these patterns will be categorized as CRIS in GovCloud regions.
        All other GovCloud models default to In-Region.

        Returns:
            List of model name patterns (e.g., ["claude-3-haiku", "claude-3-5-sonnet"])
        """
        return self.config.get("govcloud_configuration", {}).get(
            "govcloud_cris_models", []
        )

    def is_govcloud_cris_model(self, model_name: str) -> bool:
        """
        Check if a model should be marked as CRIS in GovCloud.

        Args:
            model_name: The model name to check (e.g., "Claude 3 Haiku", "claude-3-haiku")

        Returns:
            True if the model matches any CRIS pattern for GovCloud
        """
        if not model_name:
            return False

        model_lower = model_name.lower().replace(" ", "-").replace("_", "-")
        cris_patterns = self.get_govcloud_cris_models()

        for pattern in cris_patterns:
            pattern_lower = pattern.lower().replace(" ", "-").replace("_", "-")
            if pattern_lower in model_lower:
                return True

        return False

    # =========================================================================
    # External URL Accessors
    # =========================================================================

    def get_external_url(self, category: str, key: str, default: str = None) -> str:
        """
        Get an external URL from configuration.

        Args:
            category: URL category (pricing, documentation, external_data_sources)
            key: URL key within the category
            default: Default value if not found

        Returns:
            URL string or default
        """
        return self.config.get("external_urls", {}).get(category, {}).get(key, default)

    def get_bulk_pricing_url(self) -> str:
        """Get the bulk pricing API URL template."""
        return self.get_external_url(
            "pricing",
            "bulk_pricing_api",
            "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/{service_code}/current/{region}/index.json",
        )

    def get_litellm_url(self) -> str:
        """Get the LiteLLM model prices URL."""
        return self.get_external_url(
            "external_data_sources",
            "litellm_model_prices",
            "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
        )

    def get_litellm_fallback_url(self) -> str:
        """Get the LiteLLM fallback URL."""
        return self.get_external_url(
            "external_data_sources",
            "litellm_model_prices_fallback",
            "https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/model_prices_and_context_window_backup.json",
        )

    def get_documentation_url(self, key: str) -> str:
        """
        Get a documentation URL.

        Args:
            key: Documentation URL key (bedrock_models_supported, bedrock_pricing,
                 bedrock_model_lifecycle, bedrock_model_ids)

        Returns:
            URL string or default for the key
        """
        defaults = {
            "bedrock_models_supported": "https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html",
            "bedrock_pricing": "https://aws.amazon.com/bedrock/pricing/",
            "bedrock_model_lifecycle": "https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html",
            "bedrock_model_ids": "https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html",
        }
        return self.get_external_url("documentation", key, defaults.get(key))


# Global singleton instance for use within Lambda functions
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(
    s3_client: Any = None, bucket: Optional[str] = None, force_new: bool = False
) -> ConfigLoader:
    """
    Get or create a ConfigLoader singleton.

    Args:
        s3_client: Optional S3 client to use
        bucket: Optional bucket name override
        force_new: If True, create a new instance even if one exists

    Returns:
        ConfigLoader instance
    """
    global _config_loader

    if _config_loader is None or force_new:
        _config_loader = ConfigLoader(s3_client=s3_client, bucket=bucket)

    return _config_loader
