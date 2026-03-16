"""
Configuration module for local handlers.

Provides access to the same configuration used by Lambda handlers,
using embedded defaults when S3 config is not available.
"""

# Embedded default configuration (same as backend/layers/common/python/shared/config_loader.py)
DEFAULT_CONFIG = {
    "region_configuration": {
        "model_regions": ["us-east-1", "us-west-2"],
        "quota_regions": [
            "us-east-1", "us-west-2", "us-east-2",
            "eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1", "eu-north-1",
            "ap-southeast-1", "ap-southeast-2", "ap-southeast-3",
            "ap-northeast-1", "ap-northeast-2", "ap-south-1",
            "ca-central-1", "sa-east-1"
        ],
        "feature_regions": [
            "us-east-1", "us-west-2", "us-east-2",
            "eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1", "eu-north-1",
            "ap-southeast-1", "ap-southeast-2",
            "ap-northeast-1", "ap-northeast-2", "ap-south-1",
            "ca-central-1", "sa-east-1"
        ],
        "region_locations": {
            "us-east-1": "US East (N. Virginia)",
            "us-east-2": "US East (Ohio)",
            "us-west-1": "US West (N. California)",
            "us-west-2": "US West (Oregon)",
            "eu-west-1": "EU (Ireland)",
            "eu-west-2": "EU (London)",
            "eu-west-3": "EU (Paris)",
            "eu-central-1": "EU (Frankfurt)",
            "eu-north-1": "EU (Stockholm)",
            "eu-south-1": "EU (Milan)",
            "eu-south-2": "EU (Spain)",
            "ap-northeast-1": "Asia Pacific (Tokyo)",
            "ap-northeast-2": "Asia Pacific (Seoul)",
            "ap-northeast-3": "Asia Pacific (Osaka)",
            "ap-southeast-1": "Asia Pacific (Singapore)",
            "ap-southeast-2": "Asia Pacific (Sydney)",
            "ap-southeast-3": "Asia Pacific (Jakarta)",
            "ap-southeast-5": "Asia Pacific (Malaysia)",
            "ap-southeast-7": "Asia Pacific (Thailand)",
            "ap-south-1": "Asia Pacific (Mumbai)",
            "ap-south-2": "Asia Pacific (Hyderabad)",
            "ap-east-1": "Asia Pacific (Hong Kong)",
            "ap-east-2": "Asia Pacific (Philippines)",
            "sa-east-1": "South America (Sao Paulo)",
            "ca-central-1": "Canada (Central)",
            "me-south-1": "Middle East (Bahrain)",
            "me-central-1": "Middle East (UAE)",
            "af-south-1": "Africa (Cape Town)",
            "il-central-1": "Israel (Tel Aviv)"
        }
    },
    "provider_configuration": {
        "explicit_provider_names": {
            "anthropic": "Anthropic",
            "claude": "Anthropic",
            "amazon": "Amazon",
            "titan": "Amazon",
            "nova": "Amazon",
            "meta": "Meta",
            "llama": "Meta",
            "mistral": "Mistral AI",
            "cohere": "Cohere",
            "ai21": "AI21 Labs",
            "ai21labs": "AI21 Labs",
            "jamba": "AI21 Labs",
            "jurassic": "AI21 Labs",
            "stability": "Stability AI",
            "stable": "Stability AI",
            "sdxl": "Stability AI"
        },
        "provider_patterns": {
            "Anthropic": ["claude", "anthropic"],
            "Amazon": ["titan", "nova", "amazon"],
            "Meta": ["llama", "meta"],
            "Mistral AI": ["mistral", "mixtral", "pixtral", "codestral"],
            "Cohere": ["cohere", "command", "embed"],
            "AI21 Labs": ["ai21", "jamba", "jurassic"],
            "Stability AI": ["stability", "stable", "sdxl", "sd3"],
            "NVIDIA": ["nvidia", "nemotron"],
            "Writer AI": ["writer", "palmyra"],
            "Luma AI": ["luma", "ray"],
            "Moonshot AI": ["moonshot", "kimi"],
            "DeepSeek": ["deepseek"],
            "Qwen": ["qwen"],
            "OpenAI": ["openai", "gpt"],
            "Twelve Labs": ["twelve", "marengo", "pegasus"],
            "Minimax": ["minimax", "abab"]
        }
    },
    "context_window_specs": {
        "anthropic.claude-opus-4-6": {"standard_context": 200000, "extended_context": 1000000, "max_output": 32000, "source": "anthropic_docs"},
        "anthropic.claude-opus-4-5": {"standard_context": 200000, "extended_context": 1000000, "max_output": 32000, "source": "anthropic_docs"},
        "anthropic.claude-opus-4": {"standard_context": 200000, "extended_context": 1000000, "max_output": 32000, "source": "anthropic_docs"},
        "anthropic.claude-sonnet-4-5": {"standard_context": 200000, "extended_context": 1000000, "max_output": 64000, "source": "anthropic_docs"},
        "anthropic.claude-sonnet-4": {"standard_context": 200000, "max_output": 64000, "source": "anthropic_docs"},
        "anthropic.claude-3-7-sonnet": {"standard_context": 200000, "max_output": 128000, "source": "anthropic_docs"},
        "anthropic.claude-3-5-sonnet": {"standard_context": 200000, "max_output": 8192, "source": "anthropic_docs"},
        "anthropic.claude-3-5-haiku": {"standard_context": 200000, "max_output": 8192, "source": "anthropic_docs"},
        "anthropic.claude-3-opus": {"standard_context": 200000, "max_output": 4096, "source": "anthropic_docs"},
        "anthropic.claude-3-sonnet": {"standard_context": 200000, "max_output": 4096, "source": "anthropic_docs"},
        "anthropic.claude-3-haiku": {"standard_context": 200000, "max_output": 4096, "source": "anthropic_docs"},
        "meta.llama3-2": {"standard_context": 128000, "max_output": 4096, "source": "meta_docs"},
        "meta.llama3-1": {"standard_context": 128000, "max_output": 4096, "source": "meta_docs"},
        "meta.llama3-3": {"standard_context": 128000, "max_output": 4096, "source": "meta_docs"},
        "meta.llama4": {"standard_context": 128000, "max_output": 4096, "source": "meta_docs"},
        "mistral.mistral-large": {"standard_context": 128000, "max_output": 8192, "source": "mistral_docs"},
        "mistral.pixtral-large": {"standard_context": 128000, "max_output": 8192, "source": "mistral_docs"},
        "amazon.nova-pro": {"standard_context": 300000, "max_output": 5000, "source": "aws_docs"},
        "amazon.nova-lite": {"standard_context": 300000, "max_output": 5000, "source": "aws_docs"},
        "amazon.nova-micro": {"standard_context": 128000, "max_output": 5000, "source": "aws_docs"},
        "amazon.nova-premier": {"standard_context": 1000000, "max_output": 32000, "source": "aws_docs"},
        "cohere.command-r-plus": {"standard_context": 128000, "max_output": 4096, "source": "cohere_docs"},
        "cohere.command-r": {"standard_context": 128000, "max_output": 4096, "source": "cohere_docs"}
    }
}

_config = None


def get_config() -> dict:
    """Get the configuration dictionary."""
    global _config
    if _config is None:
        _config = DEFAULT_CONFIG
    return _config


def get_context_window_specs() -> dict:
    """Get context window specifications."""
    return get_config().get('context_window_specs', {})


def get_region_list(region_type: str) -> list:
    """Get a list of regions by type (model_regions, quota_regions, feature_regions)."""
    return get_config().get('region_configuration', {}).get(region_type, [])
