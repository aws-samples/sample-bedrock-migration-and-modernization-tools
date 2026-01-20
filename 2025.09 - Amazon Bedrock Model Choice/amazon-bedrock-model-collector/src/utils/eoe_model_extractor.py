#!/usr/bin/env python3
"""
Simple Context Window Extractor

Provides static context window information for known Bedrock models.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class EoeModelExtractor:
    """Provides static context window and basic model information."""

    def __init__(self):
        self.models = {}

    def extract_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Return static model information with context windows."""
        logger.info("Using static model information...")

        # Static context window data for known models
        static_models = {
            'ANTHROPIC_CLAUDE_3_5_SONNET_20241022_V2': {
                'model_id': 'ANTHROPIC_CLAUDE_3_5_SONNET_20241022_V2',
                'context_window': 200000,
                'modelName': 'Claude 3.5 Sonnet (New)',
                'providerName': 'Anthropic',
                'description': 'Most intelligent model with superior performance on highly complex tasks, including advanced math, coding, and reasoning.',
            },
            'ANTHROPIC_CLAUDE_3_5_HAIKU_20241022_V1': {
                'model_id': 'ANTHROPIC_CLAUDE_3_5_HAIKU_20241022_V1',
                'context_window': 200000,
                'modelName': 'Claude 3.5 Haiku',
                'providerName': 'Anthropic',
                'description': 'Fastest and most affordable model in the Claude 3 family.',
            },
            'ANTHROPIC_CLAUDE_3_SONNET_20240229_V1': {
                'model_id': 'ANTHROPIC_CLAUDE_3_SONNET_20240229_V1',
                'context_window': 200000,
                'modelName': 'Claude 3 Sonnet',
                'providerName': 'Anthropic',
                'description': 'Balance of intelligence and speed for a wide range of applications.',
            },
            'ANTHROPIC_CLAUDE_3_OPUS_20240229_V1': {
                'model_id': 'ANTHROPIC_CLAUDE_3_OPUS_20240229_V1',
                'context_window': 200000,
                'modelName': 'Claude 3 Opus',
                'providerName': 'Anthropic',
                'description': 'Most powerful model with top-level performance on highly complex tasks.',
            },
            'ANTHROPIC_CLAUDE_3_HAIKU_20240307_V1': {
                'model_id': 'ANTHROPIC_CLAUDE_3_HAIKU_20240307_V1',
                'context_window': 200000,
                'modelName': 'Claude 3 Haiku',
                'providerName': 'Anthropic',
                'description': 'Fast and compact model for near-instant responses.',
            },
        }

        self.models = static_models
        logger.info(f"Loaded {len(self.models)} static model definitions")
        return self.models

    def get_model_by_id(self, model_id: str) -> Dict[str, Any]:
        """Get specific model by ID."""
        return self.models.get(model_id)

    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all available models."""
        return self.models

    def get_models_by_provider(self, provider: str) -> Dict[str, Dict[str, Any]]:
        """Get models by provider name."""
        return {k: v for k, v in self.models.items()
                if v.get('providerName', '').upper().endswith(provider.upper())}


def extract_eoe_models() -> Dict[str, Dict[str, Any]]:
    """Convenience function to get static models."""
    extractor = EoeModelExtractor()
    return extractor.extract_all_models()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    models = extract_eoe_models()
    print(f"Available {len(models)} models")

    # Show samples
    for model_id, model_data in models.items():
        print(f"\n{model_id}:")
        print(f"  Provider: {model_data.get('providerName', 'N/A')}")
        print(f"  Context Window: {model_data.get('context_window', 'N/A')}")
        print(f"  Description: {model_data.get('description', 'N/A')}")