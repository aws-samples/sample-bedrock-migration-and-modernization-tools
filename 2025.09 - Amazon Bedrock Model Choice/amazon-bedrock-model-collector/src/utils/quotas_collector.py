"""
Service Quotas Collector
Collects comprehensive Bedrock service quotas from all regions
"""

import boto3
import logging
from typing import Dict, List, Optional, Any
from botocore.exceptions import ClientError


class ServiceQuotasCollector:
    """Collector for Bedrock service quotas across all regions"""

    def __init__(self, profile_name: Optional[str] = None, regions: List[str] = None):
        self.profile_name = profile_name
        self.regions = regions or []
        self.session = None
        self.logger = logging.getLogger(__name__)
        self._initialize_session()

    def _initialize_session(self):
        """Initialize AWS session"""
        try:
            if self.profile_name:
                self.session = boto3.Session(profile_name=self.profile_name)
            else:
                self.session = boto3.Session()
            self.logger.info("✅ Service quotas collector initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize quotas collector: {e}")
            raise

    def collect_service_quotas(self) -> Dict[str, Any]:
        """
        Collect service quotas from all regions
        Strategy: One query per region, then parse for all models
        """
        self.logger.info(f"Collecting Bedrock service quotas from {len(self.regions)} regions...")

        all_quotas = {}

        for region in self.regions:
            self.logger.info(f"📊 Collecting quotas from {region}...")

            try:
                region_quotas = self._collect_region_quotas(region)
                if region_quotas:
                    all_quotas[region] = region_quotas
                    self.logger.info(f"✅ Collected {len(region_quotas)} quotas from {region}")
                else:
                    self.logger.warning(f"⚠️ No quotas found in {region}")

            except Exception as e:
                self.logger.error(f"❌ Failed to collect quotas from {region}: {e}")
                all_quotas[region] = {'error': str(e), 'quotas': {}}

        self.logger.info(f"✅ Quota collection complete from {len(all_quotas)} regions")
        return all_quotas

    def _collect_region_quotas(self, region: str) -> Dict[str, Any]:
        """Collect all Bedrock quotas from a specific region"""
        try:
            quotas_client = self.session.client('service-quotas', region_name=region)

            # Get all Bedrock service quotas in one call
            paginator = quotas_client.get_paginator('list_service_quotas')

            region_quotas = {}
            total_quotas = 0

            for page in paginator.paginate(ServiceCode='bedrock'):
                for quota in page.get('Quotas', []):
                    quota_code = quota.get('QuotaCode', '')
                    quota_name = quota.get('QuotaName', '')

                    if quota_code:
                        region_quotas[quota_code] = {
                            'quota_code': quota_code,
                            'quota_name': quota_name,
                            'value': quota.get('Value', 0),
                            'unit': quota.get('Unit', ''),
                            'adjustable': quota.get('Adjustable', False),
                            'global_quota': quota.get('GlobalQuota', False),
                            'usage_metric': quota.get('UsageMetric', {}),
                            'period': quota.get('Period', {})
                        }
                        total_quotas += 1

            return {
                'quotas': region_quotas,
                'total_quotas': total_quotas,
                'region': region,
                'collected_at': self._get_timestamp()
            }

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ['AccessDeniedException', 'NoSuchResourceException']:
                self.logger.debug(f"Service quotas not accessible in {region}: {error_code}")
            else:
                self.logger.error(f"AWS error in {region}: {error_code}")
            return {}

        except Exception as e:
            self.logger.error(f"Unexpected error in {region}: {e}")
            return {}

    def parse_quotas_for_models(self, quotas_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
        """Parse quotas and associate them with specific models"""
        model_quotas = {}

        for model_id, model_data in models.items():
            model_quotas[model_id] = {}

            # Get regions where this model is available
            model_regions = model_data.get('regions_available', [])

            for region in model_regions:
                if region in quotas_data:
                    region_data = quotas_data[region]
                    if 'quotas' in region_data:
                        # Filter quotas relevant to this model
                        relevant_quotas = self._filter_model_quotas(
                            model_data, region_data['quotas']
                        )
                        if relevant_quotas:
                            model_quotas[model_id][region] = relevant_quotas

            # Add quota metadata
            model_quotas[model_id]['quota_metadata'] = {
                'total_quotas_found': sum(
                    len(region_quotas) for region_quotas in model_quotas[model_id].values()
                    if isinstance(region_quotas, dict) and 'quota_metadata' not in str(region_quotas)
                ),
                'regions_with_quotas': len([
                    r for r in model_quotas[model_id].keys()
                    if r != 'quota_metadata'
                ]),
                'collection_timestamp': self._get_timestamp()
            }

        return model_quotas

    def _filter_model_quotas(self, model_data: Dict[str, Any], region_quotas: Dict[str, Any]) -> Dict[str, Any]:
        """Filter quotas relevant to a specific model with precise version matching"""
        model_id = model_data.get('model_id', '').lower()
        model_name = model_data.get('model_name', '').lower()
        model_provider = model_data.get('model_provider', '').lower()

        relevant_quotas = {}
        self.logger.debug(f"Filtering quotas for model: {model_id} (provider: {model_provider})")

        # Extract precise model information for Claude models
        model_info = self._extract_model_version_info(model_id)

        for quota_code, quota_info in region_quotas.items():
            quota_name = quota_info.get('quota_name', '').lower()
            is_relevant = False

            # Skip all general quotas that belong to Amazon Bedrock and not to specific models
            # Only check for model-specific matches

            # Claude models - precise matching
            if 'claude' in model_id:
                if 'claude' in quota_name or 'anthropic' in quota_name:
                    is_relevant = self._is_quota_exact_match(model_info, quota_name)

            # Llama models - version-specific matching
            elif 'llama' in model_id:
                if 'llama' in quota_name or 'meta' in quota_name:
                    if '3.1' in model_id and '3.1' in quota_name:
                        is_relevant = True
                    elif '3.2' in model_id and '3.2' in quota_name:
                        is_relevant = True
                    elif '3.3' in model_id and '3.3' in quota_name:
                        is_relevant = True
                    elif 'llama' in quota_name and not any(v in quota_name for v in ['3.1', '3.2', '3.3']):
                        is_relevant = True

            # Amazon Titan models - variant-specific matching
            elif 'titan' in model_id:
                if 'titan' in quota_name:
                    if 'text' in model_id and 'text' in quota_name:
                        is_relevant = True
                    elif 'embed' in model_id and 'embed' in quota_name:
                        is_relevant = True
                    elif 'image' in model_id and 'image' in quota_name:
                        is_relevant = True
                    elif 'titan' in quota_name and not any(v in quota_name for v in ['text', 'embed', 'image']):
                        is_relevant = True

            # Amazon Nova models - variant-specific matching
            elif 'nova' in model_id:
                if 'nova' in quota_name:
                    if 'lite' in model_id and 'lite' in quota_name:
                        is_relevant = True
                    elif 'pro' in model_id and 'pro' in quota_name:
                        is_relevant = True
                    elif 'micro' in model_id and 'micro' in quota_name:
                        is_relevant = True
                    elif 'canvas' in model_id and 'canvas' in quota_name:
                        is_relevant = True
                    elif 'reel' in model_id and 'reel' in quota_name:
                        is_relevant = True
                    elif 'nova' in quota_name and not any(v in quota_name for v in ['lite', 'pro', 'micro', 'canvas', 'reel']):
                        is_relevant = True

            # Cohere models
            elif 'cohere' in model_id:
                if 'cohere' in quota_name:
                    if 'command' in model_id and 'command' in quota_name:
                        is_relevant = True
                    elif 'embed' in model_id and 'embed' in quota_name:
                        is_relevant = True
                    elif 'cohere' in quota_name and not any(v in quota_name for v in ['command', 'embed']):
                        is_relevant = True

            # Mistral models
            elif 'mistral' in model_id:
                if 'mistral' in quota_name:
                    is_relevant = True

            # AI21 models
            elif 'ai21' in model_id or 'j2' in model_id:
                if 'ai21' in quota_name or 'j2' in quota_name or 'jamba' in quota_name:
                    is_relevant = True

            # Stability AI models
            elif 'stability' in model_id or 'stable' in model_id:
                if 'stability' in quota_name or 'stable' in quota_name:
                    is_relevant = True

            # DeepSeek models
            elif 'deepseek' in model_id:
                if 'deepseek' in quota_name:
                    is_relevant = True

            # Qwen models
            elif 'qwen' in model_id:
                if 'qwen' in quota_name:
                    is_relevant = True

            if is_relevant:
                relevant_quotas[quota_code] = quota_info
                self.logger.debug(f"  ✅ RELEVANT: {quota_info.get('quota_name', 'Unknown')}")
            # else:
            #     self.logger.debug(f"  ❌ SKIPPED: {quota_info.get('quota_name', 'Unknown')}")

        self.logger.debug(f"Final result: {len(relevant_quotas)}/{len(region_quotas)} quotas for {model_id}")
        return relevant_quotas

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

    def _is_general_bedrock_quota(self, quota_name: str) -> bool:
        """Check if quota is a general Bedrock quota applicable to all models"""
        general_patterns = [
            'bedrock', 'foundation model', 'inference', 'provisioned throughput',
            'on-demand', 'requests per', 'tokens per', 'batch'
        ]
        return any(pattern in quota_name for pattern in general_patterns)

    def _get_timestamp(self) -> str:
        """Get current UTC timestamp"""
        import time
        return time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())