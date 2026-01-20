"""
Smart Model Extraction Logic
Advanced, adaptive extraction with minimal hardcoding for AWS Pricing API data
"""

import re
import logging
from typing import Dict, Tuple, Optional, Set


logger = logging.getLogger(__name__)


class SmartModelExtractor:
    """
    Smart, adaptive model extraction logic that minimizes hardcoding
    and adapts to different AWS service code structures
    """

    # Common provider keywords (minimal set for smart detection)
    PROVIDER_KEYWORDS = {
        'anthropic': ['claude'],
        'meta': ['llama'],
        'amazon': ['nova', 'titan', 'rerank'],
        'stability ai': ['stable'],
        'ai21 labs': ['jurassic', 'jamba'],
        'cohere': ['cohere', 'command', 'embed'],
        'mistral ai': ['mistral', 'mixtral'],
        'openai': ['gpt'],
        'deepseek': ['deepseek'],
        'qwen': ['qwen'],
        'writer': ['palmyra'],
        'twelvelabs': ['twelve', 'pegasus', 'marengo'],
        'luma ai': ['luma', 'ray']
    }

    # Suffixes to clean from model names
    CLEAN_SUFFIXES = [
        '(Amazon Bedrock Edition)',
        '(Amazon Bedrock)',
        'Amazon Bedrock Edition',
        'Amazon Bedrock'
    ]

    def __init__(self):
        """Initialize the smart extractor"""
        # Pre-compile regex patterns for performance
        self._suffix_patterns = [
            re.compile(re.escape(suffix) + r'\s*$', re.IGNORECASE)
            for suffix in self.CLEAN_SUFFIXES
        ]

    def extract_model_info(self, attrs: Dict, service_code: str) -> Tuple[str, str, str]:
        """
        Smart extraction of model name and provider from AWS Pricing attributes

        Args:
            attrs: Product attributes from AWS Pricing API
            service_code: AWS service code (AmazonBedrock, etc.)

        Returns:
            Tuple of (model_name, provider, model_id)
        """
        # Step 1: Extract raw model name using adaptive strategy
        raw_model_name = self._extract_raw_model_name(attrs, service_code)

        # Step 2: Clean the model name
        clean_model_name = self._clean_model_name(raw_model_name)

        # Step 3: Detect provider from clean model name
        provider = self._detect_provider(clean_model_name, attrs)

        # Step 4: Generate consistent model ID
        model_id = self._generate_model_id(provider, clean_model_name)

        return clean_model_name, provider, model_id

    def _extract_raw_model_name(self, attrs: Dict, service_code: str) -> str:
        """
        Extract raw model name using service-specific strategy

        Priority order (adaptive based on what's available):
        1. servicename (for AmazonBedrockFoundationModels)
        2. model (for AmazonBedrock, AmazonBedrockService)
        3. titanModel (special case for Titan models)
        4. Fallback extraction from usagetype
        """
        # Strategy 1: servicename (most common in AmazonBedrockFoundationModels)
        servicename = attrs.get('servicename', '').strip()
        if servicename and servicename not in ['Amazon Bedrock', 'Amazon Bedrock Service']:
            return servicename

        # Strategy 2: model field (most common in AmazonBedrock, AmazonBedrockService)
        model = attrs.get('model', '').strip()
        if model and model.lower() != 'unknown':
            return model

        # Strategy 3: titanModel field (special case)
        titan_model = attrs.get('titanModel', '').strip()
        if titan_model:
            return titan_model

        # Strategy 4: Extract from usagetype (fallback)
        usagetype = attrs.get('usagetype', '')
        if usagetype:
            # Try to extract meaningful model info from usage patterns
            extracted = self._extract_from_usagetype(usagetype)
            if extracted:
                return extracted

        # Final fallback
        return 'Unknown Model'

    def _clean_model_name(self, raw_name: str) -> str:
        """
        Clean model name by removing AWS-specific suffixes and normalizing

        Args:
            raw_name: Raw model name from AWS

        Returns:
            Cleaned model name
        """
        if not raw_name or raw_name.lower() in ['unknown', 'unknown model']:
            return raw_name

        cleaned = raw_name.strip()

        # Remove AWS-specific suffixes using pre-compiled regex
        for pattern in self._suffix_patterns:
            cleaned = pattern.sub('', cleaned).strip()

        # Remove extra whitespace and normalize
        cleaned = ' '.join(cleaned.split())

        return cleaned if cleaned else raw_name

    def _detect_provider(self, model_name: str, attrs: Dict) -> str:
        """
        Smart provider detection with minimal hardcoding

        Uses keyword matching but with adaptive fallbacks
        """
        # Check if provider is explicitly provided (AmazonBedrockService has this)
        explicit_provider = attrs.get('provider', '').strip()
        if explicit_provider and explicit_provider.lower() != 'unknown':
            return explicit_provider

        # Smart keyword detection
        model_lower = model_name.lower()
        search_text = f"{model_name} {attrs.get('usagetype', '')} {attrs.get('operation', '')}".lower()

        # Find matching provider based on keywords
        for provider, keywords in self.PROVIDER_KEYWORDS.items():
            if any(keyword in model_lower for keyword in keywords):
                return self._format_provider_name(provider)

        # Fallback: try broader search in all attributes
        all_text = ' '.join(str(v) for v in attrs.values()).lower()
        for provider, keywords in self.PROVIDER_KEYWORDS.items():
            if any(keyword in all_text for keyword in keywords):
                return self._format_provider_name(provider)

        return 'Unknown Provider'

    def _format_provider_name(self, provider: str) -> str:
        """Format provider name consistently"""
        # Convert to title case and handle special cases
        if provider == 'ai21 labs':
            return 'AI21 Labs'
        elif provider == 'stability ai':
            return 'Stability AI'
        elif provider == 'mistral ai':
            return 'Mistral AI'
        elif provider == 'luma ai':
            return 'Luma AI'
        elif provider == 'twelvelabs':
            return 'TwelveLabs'
        elif provider == 'openai':
            return 'OpenAI'
        else:
            return provider.title()

    def _generate_model_id(self, provider: str, model_name: str) -> str:
        """
        Generate consistent model ID from provider and model name

        Format: provider.model-name (lowercase, hyphenated)
        """
        if not provider or provider == 'Unknown Provider':
            provider_part = 'unknown'
        else:
            # Normalize provider name for ID
            provider_part = provider.lower().replace(' ', '').replace('-', '')

        if not model_name or model_name == 'Unknown Model':
            model_part = 'unknown'
        else:
            # Normalize model name for ID
            model_part = re.sub(r'[^\w\s-]', '', model_name.lower())
            model_part = re.sub(r'\s+', '-', model_part.strip())
            model_part = re.sub(r'-+', '-', model_part)

        return f"{provider_part}.{model_part}"

    def _extract_from_usagetype(self, usagetype: str) -> Optional[str]:
        """
        Extract model name from usagetype as fallback

        Patterns like:
        - "USE1-NovaLite-input-tokens" -> "Nova Lite"
        - "APN1-Claude3Sonnet-output" -> "Claude 3 Sonnet"
        """
        if not usagetype:
            return None

        # Remove region prefix (e.g., "USE1-", "APN1-")
        parts = usagetype.split('-')
        if len(parts) < 2:
            return None

        # Skip region part, look for model indicators
        for part in parts[1:]:
            # Skip common non-model parts
            if part.lower() in ['mp', 'input', 'output', 'tokens', 'count', 'units', 'cache', 'read', 'write']:
                continue

            # If part looks like a model name (contains letters and is substantial)
            if len(part) > 3 and any(c.isalpha() for c in part):
                # Try to format it nicely (camelCase -> Title Case)
                formatted = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
                if len(formatted) > 3:
                    return formatted

        return None

    def get_extraction_stats(self, all_extractions: list) -> Dict:
        """
        Generate statistics about extraction quality for monitoring
        """
        total = len(all_extractions)
        unknown_models = sum(1 for item in all_extractions if 'Unknown' in item[0])
        unknown_providers = sum(1 for item in all_extractions if 'Unknown' in item[1])

        # Count extraction methods used
        servicename_used = 0
        model_field_used = 0
        usagetype_fallback = 0

        # This would require tracking during extraction, simplified for now

        return {
            'total_extractions': total,
            'unknown_models': unknown_models,
            'unknown_providers': unknown_providers,
            'unknown_model_rate': unknown_models / total if total > 0 else 0,
            'unknown_provider_rate': unknown_providers / total if total > 0 else 0
        }