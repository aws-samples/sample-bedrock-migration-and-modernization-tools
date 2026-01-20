"""
Model Data Processor
Processes and structures comprehensive model data into final JSON format
"""

import logging
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict


class ModelDataProcessor:
    """Processes collected model data into comprehensive structured format"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_comprehensive_structure(self, raw_models: Dict[str, Any], enhanced_models: Dict[str, Any],
                                     pricing_data: Dict[str, Any], quotas_data: Dict[str, Any],
                                     regions: List[str]) -> Dict[str, Any]:
        """Create comprehensive structured JSON output"""

        self.logger.info("Creating comprehensive data structure...")

        # Organize models by provider
        providers = defaultdict(lambda: {"models": {}})

        for model_id, model_data in enhanced_models.items():
            provider = model_data.get('model_provider', 'Unknown')
            model_name_key = self._create_model_key(model_data)

            # Merge all data sources
            comprehensive_model = self._merge_model_data(
                model_data, pricing_data.get(model_id, {}), quotas_data, model_id
            )

            providers[provider]["models"][model_name_key] = comprehensive_model

        # Create metadata
        metadata = self._create_metadata(providers, regions, raw_models, pricing_data, quotas_data)

        return {
            "metadata": metadata,
            "providers": dict(providers)
        }

    def _create_model_key(self, model_data: Dict[str, Any]) -> str:
        """Create a clean model key from model name"""
        model_name = model_data.get('model_name', '')
        if model_name:
            # Clean model name for use as key
            return model_name.lower().replace(' ', '-').replace('.', '-')
        else:
            # Fallback to model ID
            return model_data.get('model_id', 'unknown').split('.')[-1]

    def _merge_model_data(self, model_data: Dict[str, Any], pricing_info: Dict[str, Any],
                         quotas_data: Dict[str, Any], model_id: str) -> Dict[str, Any]:
        """Merge all data sources into comprehensive model structure"""

        # Start with enhanced model data
        merged = model_data.copy()

        # Add comprehensive model pricing (grouped by region as requested)
        if pricing_info.get('model_pricing'):
            merged['model_pricing'] = pricing_info['model_pricing']
        else:
            merged['model_pricing'] = {'is_pricing_available': False}

        # Add pricing metadata
        if pricing_info.get('pricing_metadata'):
            merged['pricing_metadata'] = pricing_info['pricing_metadata']

        # Add comprehensive service quotas
        model_quotas = self._extract_model_quotas(model_data, quotas_data)
        merged['model_service_quotas'] = model_quotas

        # Ensure all required fields are present
        merged = self._ensure_complete_structure(merged)

        return merged

    def _extract_model_quotas(self, model_data: Dict[str, Any], quotas_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and organize service quotas for this model"""
        model_quotas = {}
        quota_metadata = {
            'total_quotas_retrieved': 0,
            'regions_queried': 0,
            'collection_timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        }

        # Get model regions from cross_region_inference.source_regions or fallback to empty list
        cross_region_info = model_data.get('cross_region_inference', {})
        model_regions = cross_region_info.get('source_regions', [])

        self.logger.debug(f"Model {model_data.get('model_id', 'unknown')} has regions: {model_regions}")

        for region in model_regions:
            if region in quotas_data:
                region_data = quotas_data[region]
                if 'quotas' in region_data:
                    region_quotas = region_data['quotas']

                    # Filter quotas relevant to this model using the enhanced quotas collector filtering
                    from utils.quotas_collector import ServiceQuotasCollector
                    quotas_collector = ServiceQuotasCollector()
                    relevant_quotas = quotas_collector._filter_model_quotas(model_data, region_quotas)

                    self.logger.debug(f"Region {region}: Found {len(region_quotas)} total quotas, {len(relevant_quotas)} relevant for model {model_data.get('model_id', 'unknown')}")

                    if relevant_quotas:
                        model_quotas[region] = relevant_quotas
                        quota_metadata['total_quotas_retrieved'] += len(relevant_quotas)

                    quota_metadata['regions_queried'] += 1

        model_quotas['quota_metadata'] = quota_metadata
        return model_quotas

    def _extract_model_version_info(self, model_id: str) -> Dict[str, str]:
        """Extract precise model version information from model ID"""
        model_info = {
            'base_model': '',
            'version': '',
            'variant': '',
            'context': ''
        }

        model_id_lower = model_id.lower()

        # Claude models - extract precise version information
        if 'claude' in model_id_lower:
            model_info['base_model'] = 'claude'

            # Extract model variant (sonnet, haiku, opus)
            if 'sonnet' in model_id_lower:
                model_info['variant'] = 'sonnet'
                # Extract version number more precisely
                if 'sonnet-4.5' in model_id_lower or 'sonnet-4-5' in model_id_lower:
                    model_info['version'] = '4.5'
                elif 'sonnet-4' in model_id_lower:
                    # Check if it's exactly version 4 (not 4.5)
                    if not any(x in model_id_lower for x in ['4.5', '4-5', '45']):
                        model_info['version'] = '4'
                elif '3.5' in model_id_lower or '3-5' in model_id_lower:
                    model_info['version'] = '3.5'
                elif '3' in model_id_lower:
                    model_info['version'] = '3'

            elif 'haiku' in model_id_lower:
                model_info['variant'] = 'haiku'
                if '3.5' in model_id_lower:
                    model_info['version'] = '3.5'
                elif '3' in model_id_lower:
                    model_info['version'] = '3'

            elif 'opus' in model_id_lower:
                model_info['variant'] = 'opus'
                if '3' in model_id_lower:
                    model_info['version'] = '3'

            # Extract context window information
            if '200k' in model_id_lower:
                model_info['context'] = '200k'
            elif '1m' in model_id_lower or '1000k' in model_id_lower:
                model_info['context'] = '1m'

        return model_info

    def _is_quota_exact_match(self, model_info: Dict[str, str], quota_name: str) -> bool:
        """Check if quota name exactly matches the model version"""
        quota_lower = quota_name.lower()

        if model_info['base_model'] != 'claude':
            return False

        # For Claude models, implement precise matching
        variant = model_info.get('variant', '')
        version = model_info.get('version', '')

        # Must contain the correct variant
        if variant and variant not in quota_lower:
            return False

        # Version-specific matching for Claude models
        if variant == 'sonnet' and version:
            if version == '4.5':
                # Sonnet 4.5 quotas should mention "4.5", "45", or "Sonnet 4 V1 1M" (specific context)
                version_indicators = ['4.5', 'sonnet 4 v1 1m', '45']
                if not any(indicator in quota_lower for indicator in version_indicators):
                    return False
            elif version == '4':
                # Sonnet 4 quotas should mention "sonnet 4" but NOT be 4.5 quotas
                if 'sonnet 4' not in quota_lower:
                    return False
                # Exclude 4.5 quotas
                exclusions = ['4.5', 'sonnet 4 v1 1m', '45']
                if any(exclusion in quota_lower for exclusion in exclusions):
                    return False
            elif version == '3.5':
                # Must mention 3.5 or be generic sonnet
                version_indicators = ['3.5', '35', 'sonnet v2']
                if not any(indicator in quota_lower for indicator in version_indicators):
                    # Allow generic "sonnet" quotas for backward compatibility
                    if 'claude 3' in quota_lower or 'sonnet 4' in quota_lower:
                        return False
            elif version == '3':
                # Claude 3 quotas (base version)
                if not any(indicator in quota_lower for indicator in ['claude 3', 'sonnet v2']):
                    return False
                # Exclude newer version quotas
                if any(exclusion in quota_lower for exclusion in ['3.5', '35', 'sonnet 4']):
                    return False

        elif variant == 'haiku' and version:
            if version == '3.5':
                version_indicators = ['3.5', '35', 'haiku v2']
                if not any(indicator in quota_lower for indicator in version_indicators):
                    return False
            elif version == '3':
                if not any(indicator in quota_lower for indicator in ['claude 3', 'haiku']):
                    return False
                # Exclude newer version quotas
                if any(exclusion in quota_lower for exclusion in ['3.5', '35']):
                    return False

        elif variant == 'opus' and version:
            if version == '3':
                if not any(indicator in quota_lower for indicator in ['claude 3', 'opus']):
                    return False
                # Opus typically doesn't have version variations yet, but future-proofing
                if any(exclusion in quota_lower for exclusion in ['3.5', '35']):
                    return False

        return True

    def _filter_relevant_quotas(self, model_data: Dict[str, Any], region_quotas: Dict[str, Any]) -> Dict[str, Any]:
        """Filter quotas relevant to the specific model with precise version matching"""
        model_id = model_data.get('model_id', '').lower()
        model_provider = model_data.get('model_provider', '').lower()

        relevant = {}

        # Extract precise model information
        model_info = self._extract_model_version_info(model_id)

        for quota_code, quota_info in region_quotas.items():
            quota_name = quota_info.get('quota_name', '').lower()
            is_relevant = False

            # General quotas that apply to all models
            general_quota_terms = [
                'inference profiles per account',
                'endpoints per inference profile',
                'model units no-commitment provisioned throughputs across',
                '(knowledge bases) maximum number of files'
            ]

            # Check for general quotas first
            for general_term in general_quota_terms:
                if general_term in quota_name:
                    is_relevant = True
                    break

            # If not a general quota, check for model-specific matches
            if not is_relevant:
                if 'claude' in model_id:
                    # Use precise Claude model matching
                    if 'claude' in quota_name or 'anthropic' in quota_name:
                        is_relevant = self._is_quota_exact_match(model_info, quota_name)

                elif 'llama' in model_id:
                    # Llama version matching
                    if 'llama' in quota_name or 'meta' in quota_name:
                        if '3.1' in model_id and '3.1' in quota_name:
                            is_relevant = True
                        elif '3.2' in model_id and '3.2' in quota_name:
                            is_relevant = True
                        elif '3.3' in model_id and '3.3' in quota_name:
                            is_relevant = True
                        elif 'llama' in quota_name and not any(v in quota_name for v in ['3.1', '3.2', '3.3']):
                            is_relevant = True

                elif 'titan' in model_id:
                    if 'titan' in quota_name:
                        # Match titan variants
                        if 'text' in model_id and 'text' in quota_name:
                            is_relevant = True
                        elif 'embed' in model_id and 'embed' in quota_name:
                            is_relevant = True
                        elif 'image' in model_id and 'image' in quota_name:
                            is_relevant = True
                        elif 'titan' in quota_name and not any(v in quota_name for v in ['text', 'embed', 'image']):
                            is_relevant = True

                elif 'nova' in model_id:
                    if 'nova' in quota_name:
                        # Match nova variants
                        if 'lite' in model_id and 'lite' in quota_name:
                            is_relevant = True
                        elif 'pro' in model_id and 'pro' in quota_name:
                            is_relevant = True
                        elif 'micro' in model_id and 'micro' in quota_name:
                            is_relevant = True
                        elif 'nova' in quota_name and not any(v in quota_name for v in ['lite', 'pro', 'micro']):
                            is_relevant = True

                # Other model types with simpler matching
                elif any(provider in quota_name for provider in [model_provider]):
                    is_relevant = True
                elif any(keyword in quota_name for keyword in ['deepseek', 'qwen', 'mistral', 'cohere'] if keyword in model_id):
                    is_relevant = True

            if is_relevant:
                relevant[quota_code] = quota_info

        return relevant

    def _ensure_complete_structure(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure model has all required fields with defaults"""

        defaults = {
            'capabilities': [],
            'use_cases': [],
            'context_window': None,
            'languages_supported': [],
            'consumption_options': [],
            'model_pricing': {'is_pricing_available': False},
            'model_service_quotas': {},
            'cross_region_inference': {'cris_supported': False},
            'documentation_links': {}
        }

        for field, default_value in defaults.items():
            if field not in model_data:
                model_data[field] = default_value

        # Ensure documentation links are populated (even if field exists but is empty)
        if not model_data.get('documentation_links'):
            model_data['documentation_links'] = self._generate_documentation_links(model_data)

        return model_data

    def _generate_documentation_links(self, model_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate appropriate documentation links based on model provider and family"""

        model_id = model_data.get('model_id', '').lower()
        model_provider = model_data.get('model_provider', '').lower()
        model_family_raw = model_data.get('model_family', '')
        model_family = model_family_raw.lower() if model_family_raw else ''

        documentation_links = {}

        # Amazon Nova models - should use Nova-specific documentation
        if 'nova' in model_id or 'nova' in model_family:
            documentation_links['primary'] = "https://docs.aws.amazon.com/nova/latest/userguide/what-is-nova.html"
            documentation_links['type'] = "amazon_nova"

        # Amazon Titan models - keep existing Titan documentation
        elif 'titan' in model_id or 'titan' in model_family:
            documentation_links['primary'] = "https://docs.aws.amazon.com/bedrock/latest/userguide/titan-models.html"
            documentation_links['type'] = "amazon_titan"

        # Anthropic Claude models
        elif 'claude' in model_id or 'anthropic' in model_provider:
            documentation_links['primary'] = "https://docs.aws.amazon.com/bedrock/latest/userguide/claude-models.html"
            documentation_links['type'] = "anthropic_claude"

        # DeepSeek models
        elif 'deepseek' in model_id or 'deepseek' in model_provider:
            documentation_links['primary'] = "https://github.com/deepseek-ai"
            documentation_links['type'] = "deepseek"

        # OpenAI models (if they appear in Bedrock in the future)
        elif 'openai' in model_provider or 'gpt' in model_id:
            documentation_links['primary'] = "https://platform.openai.com/docs/concepts"
            documentation_links['type'] = "openai"

        # Qwen models
        elif 'qwen' in model_id or 'qwen' in model_provider:
            documentation_links['primary'] = "https://qwen.readthedocs.io/en/latest/"
            documentation_links['type'] = "qwen"

        # TwelveLabs models
        elif 'twelvelabs' in model_provider or 'twelve' in model_id:
            documentation_links['primary'] = "https://docs.twelvelabs.io/docs/get-started/introduction"
            documentation_links['type'] = "twelvelabs"

        # Meta Llama models
        elif 'llama' in model_id or 'meta' in model_provider:
            documentation_links['primary'] = "https://docs.aws.amazon.com/bedrock/latest/userguide/llama-models.html"
            documentation_links['type'] = "meta_llama"

        # Mistral models
        elif 'mistral' in model_id or 'mistral' in model_provider:
            documentation_links['primary'] = "https://docs.mistral.ai/getting-started/models/models_overview/"
            documentation_links['type'] = "mistral"

        # Cohere models
        elif 'cohere' in model_id or 'cohere' in model_provider:
            documentation_links['primary'] = "https://docs.aws.amazon.com/bedrock/latest/userguide/cohere-models.html"
            documentation_links['type'] = "cohere"

        # Default to general Bedrock documentation
        else:
            documentation_links['primary'] = "https://docs.aws.amazon.com/bedrock/latest/userguide/foundation-models.html"
            documentation_links['type'] = "general_bedrock"

        # Always add Bedrock general documentation as secondary
        if documentation_links.get('type') != 'general_bedrock':
            documentation_links['bedrock_general'] = "https://docs.aws.amazon.com/bedrock/latest/userguide/foundation-models.html"

        return documentation_links

    def _create_metadata(self, providers: Dict, regions: List[str], raw_models: Dict,
                        pricing_data: Dict, quotas_data: Dict) -> Dict[str, Any]:
        """Create comprehensive metadata"""

        total_models = sum(len(provider["models"]) for provider in providers.values())
        models_with_pricing = sum(
            1 for provider in providers.values()
            for model in provider["models"].values()
            if model.get('model_pricing', {}).get('is_pricing_available', False)
        )
        models_with_quotas = sum(
            1 for provider in providers.values()
            for model in provider["models"].values()
            if model.get('model_service_quotas', {}) and len(model['model_service_quotas']) > 1
        )

        return {
            "generated_at": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            "version": "1.0.0",
            "description": "Comprehensive Amazon Bedrock Model Database",
            "providers_count": len(providers),
            "total_models": total_models,
            "regions_covered": len(regions),
            "regions": sorted(regions),
            "models_with_pricing": models_with_pricing,
            "models_with_quotas": models_with_quotas,
            "data_sources": [
                "bedrock_api",
                "model_pricing",
                "inference_profiles",
                "service_quotas",
                "pricing_collector"
            ],
            "collection_summary": {
                "raw_models_collected": len(raw_models),
                "pricing_integrations": len(pricing_data),
                "quota_regions": len(quotas_data),
                "multi_threaded_collection": True
            }
        }