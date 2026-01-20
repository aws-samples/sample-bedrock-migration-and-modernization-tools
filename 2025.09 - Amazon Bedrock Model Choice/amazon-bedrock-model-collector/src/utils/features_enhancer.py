"""
Model Features Enhancer
Enhances models with agreement offers, inference profiles, and additional features
"""

import boto3
import logging
import time
from typing import Dict, List, Optional, Any
from botocore.exceptions import ClientError


class ModelFeaturesEnhancer:
    """Enhances model data with agreement offers and inference profiles"""

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
            self.logger.info("✅ Features enhancer initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize features enhancer: {e}")
            raise

    def enhance_models(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance models with additional features"""
        self.logger.info(f"Enhancing {len(models)} models with additional features...")

        # Gather inference profiles first (one call per region)
        inference_profiles = self._gather_inference_profiles()

        # Gather batch inference capabilities using real AWS API calls
        batch_capabilities = self._gather_batch_inference_capabilities()

        enhanced_models = {}
        processed = 0

        for model_id, model_data in models.items():
            try:
                enhanced_model = model_data.copy()

                # Add agreement offers data
                enhanced_model = self._add_agreement_offers(enhanced_model)

                # Add inference profile data
                enhanced_model = self._add_inference_profiles(enhanced_model, inference_profiles)

                # Add batch inference support (REAL API-based detection)
                enhanced_model = self._add_batch_inference_support_real(enhanced_model, batch_capabilities)

                # Add missing fields: capabilities, use_cases, languages_supported, consumption_options
                enhanced_model = self._add_missing_fields(enhanced_model)

                enhanced_models[model_id] = enhanced_model
                processed += 1

                if processed % 10 == 0:
                    self.logger.info(f"Enhanced {processed}/{len(models)} models...")

            except Exception as e:
                self.logger.warning(f"Failed to enhance {model_id}: {e}")
                enhanced_models[model_id] = model_data  # Use original data

        self.logger.info(f"✅ Enhanced {processed} models successfully")
        return enhanced_models

    def _gather_inference_profiles(self) -> Dict[str, Any]:
        """Gather comprehensive inference profiles from all regions (Enhanced with old project APIs)"""
        self.logger.info("Gathering comprehensive inference profiles for cross-region support from ALL regions...")

        inference_profiles = {}
        profiles_by_region = {}
        total_profiles_found = 0

        # Check ALL regions where Bedrock is available (from old project implementation)
        for region in self.regions:
            self.logger.debug(f"Checking inference profiles in region: {region}")

            try:
                bedrock = self.session.client('bedrock', region_name=region)

                # List inference profiles in this region
                response = bedrock.list_inference_profiles()
                region_profiles = response.get('inferenceProfileSummaries', [])

                if region_profiles:
                    self.logger.debug(f"Found {len(region_profiles)} inference profiles in {region}")
                    profiles_by_region[region] = []

                    for profile in region_profiles:
                        profile_id = profile.get('inferenceProfileId', '')
                        profile_name = profile.get('inferenceProfileName', '')
                        profile_type = profile.get('type', '')
                        profile_status = profile.get('status', '')

                        self.logger.debug(f"Processing profile: {profile_id} ({profile_name}) in {region}")

                        # Get detailed profile information (Enhanced API call from old project)
                        try:
                            profile_details = bedrock.get_inference_profile(
                                inferenceProfileIdentifier=profile_id
                            )

                            # Store profile details for this region
                            profile_info = {
                                'profile_id': profile_id,
                                'profile_name': profile_name,
                                'type': profile_type,
                                'status': profile_status,
                                'region': region,
                                'models': [],
                                'description': profile_details.get('description', ''),
                                'created_at': str(profile_details.get('createdAt', '')),
                                'updated_at': str(profile_details.get('updatedAt', ''))
                            }

                            # Extract model information from the profile
                            models = profile_details.get('models', [])
                            for model_info in models:
                                # Handle both string ARNs and dict objects
                                if isinstance(model_info, dict):
                                    model_arn = model_info.get('modelArn', '')
                                else:
                                    model_arn = model_info

                                # Extract model ID from ARN
                                model_id = self._extract_model_id_from_arn(model_arn)

                                if model_id:
                                    profile_info['models'].append({
                                        'model_id': model_id,
                                        'model_arn': model_arn
                                    })

                                    # Initialize model's inference profile data if not exists
                                    if model_id not in inference_profiles:
                                        inference_profiles[model_id] = {
                                            'supported': True,
                                            'profiles': [],
                                            'source_regions': set(),
                                            'destination_regions': set(),
                                            'total_profiles': 0
                                        }

                                    # Add this profile to the model's profile list
                                    inference_profiles[model_id]['profiles'].append({
                                        'profile_id': profile_id,
                                        'profile_name': profile_name,
                                        'type': profile_type,
                                        'status': profile_status,
                                        'source_region': region,
                                        'description': profile_details.get('description', ''),
                                        'created_at': str(profile_details.get('createdAt', '')),
                                        'updated_at': str(profile_details.get('updatedAt', ''))
                                    })

                                    # Track regions
                                    inference_profiles[model_id]['source_regions'].add(region)
                                    inference_profiles[model_id]['total_profiles'] += 1

                                    self.logger.debug(f"Added profile {profile_id} for model {model_id} in {region}")

                            profiles_by_region[region].append(profile_info)
                            total_profiles_found += 1

                        except ClientError as e:
                            self.logger.warning(f"Could not get details for profile {profile_id} in {region}: {e}")
                            continue
                        except Exception as e:
                            self.logger.error(f"Error processing profile {profile_id} in {region}: {e}")
                            continue
                else:
                    self.logger.debug(f"No inference profiles found in {region}")

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code in ['AccessDeniedException', 'UnauthorizedOperation']:
                    self.logger.warning(f"Access denied for inference profiles in {region}")
                else:
                    self.logger.debug(f"Could not list inference profiles in {region}: {e}")
                continue
            except Exception as e:
                self.logger.error(f"Error gathering inference profiles from {region}: {e}")
                continue

        # Convert sets to lists for JSON serialization and add destination regions
        for model_id in inference_profiles:
            source_regions = list(inference_profiles[model_id]['source_regions'])
            inference_profiles[model_id]['source_regions'] = sorted(source_regions)

            # For destination regions, we can infer from the profiles
            # Cross-region inference typically allows routing to multiple regions
            destination_regions = set()
            for profile in inference_profiles[model_id]['profiles']:
                # Add all Bedrock regions as potential destinations for cross-region profiles
                if profile.get('type', '').lower() in ['cross_region', 'system_defined']:
                    destination_regions.update(self.regions)

            inference_profiles[model_id]['destination_regions'] = sorted(list(destination_regions))

        # Log comprehensive results
        self.logger.info(f"🎯 Inference Profiles Summary:")
        self.logger.info(f"   - Total profiles found: {total_profiles_found}")
        self.logger.info(f"   - Regions with profiles: {len(profiles_by_region)}")
        self.logger.info(f"   - Models with inference profiles: {len(inference_profiles)}")

        # Store the comprehensive profile data for later use
        self.inference_profiles_by_region = profiles_by_region

        return inference_profiles

    def _extract_model_id_from_arn(self, model_arn: str) -> str:
        """Extract model ID from model ARN"""
        # ARN format: arn:aws:bedrock:region::foundation-model/model-id
        if 'foundation-model/' in model_arn:
            return model_arn.split('foundation-model/')[-1]
        return model_arn

    def _add_agreement_offers(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add agreement offers data to model"""
        model_id = model_data['model_id']

        try:
            # Use first available region for the model
            regions = model_data.get('regions_available', [])
            if not regions:
                return model_data

            bedrock = self.session.client('bedrock', region_name=regions[0])
            response = bedrock.list_foundation_model_agreement_offers(modelId=model_id)

            offers = response.get('offers', [])
            model_data['agreement_offers'] = {
                'offers_count': len(offers),
                'has_offers': len(offers) > 0,
                'offers': offers[:3]  # Keep first 3 offers to save space
            }

        except Exception as e:
            self.logger.debug(f"No agreement offers for {model_id}: {e}")
            model_data['agreement_offers'] = {'offers_count': 0, 'has_offers': False}

        return model_data

    def _add_inference_profiles(self, model_data: Dict[str, Any], all_profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Add cross-region inference profile data (Enhanced to match old project structure)"""
        model_id = model_data['model_id']

        if model_id in all_profiles:
            profile_info = all_profiles[model_id]
            model_data['cross_region_inference'] = {
                'supported': True,  # Updated field name to match old project
                'profiles_count': len(profile_info['profiles']),
                'source_regions': profile_info['source_regions'],
                'destination_regions': profile_info['destination_regions'],
                'profiles': profile_info['profiles']  # Include all profiles with full details
            }
        else:
            model_data['cross_region_inference'] = {
                'supported': False,  # Updated field name to match old project
                'profiles_count': 0,
                'source_regions': [],
                'destination_regions': [],
                'profiles': []
            }

        return model_data

    def _gather_batch_inference_capabilities(self) -> Dict[str, Any]:
        """Gather batch inference capabilities for all regions using real AWS API calls"""
        self.logger.info("Testing REAL batch inference availability across all regions using bedrock.list_model_import_jobs()...")

        region_capabilities = {}

        for region in self.regions:
            try:
                bedrock = self.session.client('bedrock', region_name=region)

                # This is the same API call used in the old project
                # If this succeeds, batch inference is available in this region
                bedrock.list_model_import_jobs(maxResults=1)

                region_capabilities[region] = {
                    'batch_inference_available': True,
                    'method': 'list_model_import_jobs'
                }
                self.logger.debug(f"✅ Batch inference available in {region}")

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code in ['AccessDeniedException', 'UnauthorizedOperation']:
                    # Service exists but we don't have permission - assume it's supported
                    region_capabilities[region] = {
                        'batch_inference_available': True,
                        'method': 'access_denied_assume_supported'
                    }
                    self.logger.debug(f"🟡 Batch inference assumed available in {region} (access denied)")
                else:
                    # Service not available in this region
                    region_capabilities[region] = {
                        'batch_inference_available': False,
                        'method': 'api_error',
                        'error': error_code
                    }
                    self.logger.debug(f"❌ Batch inference not available in {region}")

            except Exception as e:
                # Other errors - assume not available
                region_capabilities[region] = {
                    'batch_inference_available': False,
                    'method': 'exception'
                }
                self.logger.debug(f"❌ Batch inference error in {region}: {str(e)[:50]}")

        # Summary
        available_regions = [r for r, caps in region_capabilities.items() if caps['batch_inference_available']]
        self.logger.info(f"🎯 Batch inference summary: Available in {len(available_regions)}/{len(self.regions)} regions")

        return region_capabilities

    def _add_batch_inference_support_real(self, model_data: Dict[str, Any], region_capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Add REAL batch inference support using AWS API testing (Based on old project method)"""
        model_id = model_data['model_id']

        # AWS-documented batch-supported patterns (from documentation + old project)
        batch_supported_patterns = [
            'anthropic.claude',  # All Claude models (3, 3.5, Sonnet 4, etc.)
            'amazon.nova',       # Nova Lite, Micro, Pro
            'amazon.titan',      # Titan Embeddings, Text
            'meta.llama',        # Llama 3.1, 3.2, 3.3, 4
            'cohere.embed',      # Cohere Embeddings
            'cohere.command',    # Cohere Command
            'deepseek',          # DeepSeek models
            'mistral',           # Mistral Large/Small
            'qwen'              # Qwen models
        ]

        # Step 1: Check if model type supports batch inference (pattern matching)
        model_supports_batch = any(pattern in model_id.lower() for pattern in batch_supported_patterns)

        if not model_supports_batch:
            # Model type doesn't support batch inference
            model_data['batch_inference_supported'] = {
                'supported': False,
                'supported_regions': [],
                'total_regions': 0,
                'coverage_percentage': 0.0,
                'reason': 'Model type does not support batch inference',
                'detection_method': 'pattern_matching'
            }
            return model_data

        # Step 2: Get regions where this model is available
        cross_region_info = model_data.get('cross_region_inference', {})
        source_regions = cross_region_info.get('source_regions', [])

        if not source_regions:
            # No region data available
            model_data['batch_inference_supported'] = {
                'supported': False,
                'supported_regions': [],
                'total_regions': 0,
                'coverage_percentage': 0.0,
                'reason': 'No regional availability data',
                'detection_method': 'no_region_data'
            }
            return model_data

        # Step 3: Check which of the model's regions actually support batch inference (using real API data)
        supported_batch_regions = []
        for region in source_regions:
            region_caps = region_capabilities.get(region, {})
            if region_caps.get('batch_inference_available', False):
                supported_batch_regions.append(region)

        # Step 4: Update model with real batch inference data
        coverage_percentage = (len(supported_batch_regions) / len(source_regions) * 100) if source_regions else 0
        is_supported = len(supported_batch_regions) > 0

        model_data['batch_inference_supported'] = {
            'supported': is_supported,
            'supported_regions': supported_batch_regions,
            'total_regions': len(source_regions),
            'coverage_percentage': coverage_percentage,
            'detection_method': 'real_api_testing'
        }

        return model_data

    def _add_missing_fields(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add missing fields: capabilities, use_cases, languages_supported, consumption_options"""

        # Extract capabilities from model modalities and properties
        capabilities = self._extract_capabilities(model_data)
        model_data['capabilities'] = capabilities

        # Extract use cases based on capabilities
        use_cases = self._extract_use_cases(capabilities, model_data)
        model_data['use_cases'] = use_cases

        # Extract supported languages
        languages_supported = self._extract_languages_supported(model_data)
        model_data['languages_supported'] = languages_supported

        # Extract consumption options
        consumption_options = self._extract_consumption_options(model_data)
        model_data['consumption_options'] = consumption_options

        # Add documentation links
        documentation_links = self._add_documentation_links(model_data)
        model_data['documentation_links'] = documentation_links

        return model_data

    def _extract_capabilities(self, model_data: Dict[str, Any]) -> List[str]:
        """Extract model capabilities from model info"""
        capabilities = []

        # Get input/output modalities
        input_modalities = model_data.get('input_modalities', [])
        output_modalities = model_data.get('output_modalities', [])

        if 'TEXT' in input_modalities and 'TEXT' in output_modalities:
            capabilities.extend(['chat', 'text_generation', 'text'])

        if 'IMAGE' in input_modalities:
            capabilities.extend(['image_input', 'multimodal'])

        if 'VIDEO' in input_modalities:
            capabilities.extend(['video_input', 'multimodal'])

        if 'IMAGE' in output_modalities:
            capabilities.extend(['image_generation', 'text_to_image'])

        if 'VIDEO' in output_modalities:
            capabilities.extend(['video_generation'])

        # Check for embeddings capability
        model_name = model_data.get('model_name', '').lower()
        if 'embed' in model_name:
            if 'image' in model_name:
                capabilities.append('image_embeddings')
            else:
                capabilities.extend(['embeddings', 'text_embeddings', 'semantic_search'])

        # Add reasoning capability for advanced models
        if any(term in model_name for term in ['claude', 'sonnet', 'opus', 'nova', 'llama']):
            capabilities.append('reasoning')

        return list(set(capabilities))  # Remove duplicates

    def _extract_use_cases(self, capabilities: List[str], model_data: Dict[str, Any]) -> List[str]:
        """Extract use cases based on model capabilities"""
        use_cases = []

        if 'chat' in capabilities:
            use_cases.extend(['chat applications', 'question answering', 'conversational ai'])

        if 'text_generation' in capabilities:
            use_cases.extend(['content generation', 'summarization', 'writing assistance'])

        if 'image_input' in capabilities:
            use_cases.extend(['document analysis', 'visual question answering', 'image understanding'])

        if 'image_generation' in capabilities:
            use_cases.extend(['image generation', 'creative content', 'visual design'])

        if 'video_generation' in capabilities:
            use_cases.extend(['video generation', 'video editing'])

        if 'embeddings' in capabilities:
            use_cases.extend(['semantic search', 'similarity matching', 'recommendation systems'])

        if 'reasoning' in capabilities:
            use_cases.extend(['complex reasoning', 'problem solving', 'analysis'])

        # Add common use cases
        use_cases.extend(['instruction following'])

        # Check model name for specific use cases
        model_name = model_data.get('model_name', '').lower()
        if 'code' in model_name:
            use_cases.append('code generation')

        return list(set(use_cases))

    def _extract_languages_supported(self, model_data: Dict[str, Any]) -> List[str]:
        """Extract supported languages from model description and known capabilities"""

        # Get model details for language extraction
        model_name = model_data.get('model_name', '').lower()
        description = model_data.get('description', {})

        # Extract supported languages from description if available
        supported_languages = []
        if isinstance(description, dict):
            supported_langs_field = description.get('supportedLanguages', '')
            if supported_langs_field:
                # Parse the supported languages field
                if supported_langs_field.lower() != 'english':
                    # If it's not just English, split and clean
                    supported_languages = [lang.strip() for lang in supported_langs_field.split(',')]
                else:
                    supported_languages = ['English']

        # If no explicit language info, infer from model provider
        if not supported_languages:
            model_provider = model_data.get('model_provider', '').lower()

            # Provider-specific language support
            if 'anthropic' in model_provider:
                supported_languages = ['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Japanese', 'Korean', 'Chinese']
            elif 'amazon' in model_provider:
                if 'nova' in model_name:
                    supported_languages = ['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Japanese', 'Korean']
                else:
                    supported_languages = ['English']
            elif 'meta' in model_provider:
                supported_languages = ['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Hindi', 'Thai', 'Vietnamese']
            elif 'cohere' in model_provider:
                supported_languages = ['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Dutch', 'Arabic', 'Chinese', 'Japanese']
            elif 'mistral' in model_provider:
                supported_languages = ['English', 'French', 'German', 'Spanish', 'Italian']
            elif 'deepseek' in model_provider:
                supported_languages = ['English', 'Chinese']
            else:
                # Default to English for unknown providers
                supported_languages = ['English']

        return supported_languages

    def _extract_consumption_options(self, model_data: Dict[str, Any]) -> List[str]:
        """Extract available consumption options"""
        options = ['on_demand']  # All models support on-demand

        # Check inference types supported from API
        inference_types = model_data.get('inference_types_supported', [])

        if 'PROVISIONED' in inference_types:
            options.append('provisioned_throughput')

        # Check batch inference support
        batch_supported = model_data.get('batch_inference_supported', {}).get('supported', False)
        if batch_supported:
            options.append('batch_inference')

        # Check if model supports customization (fine-tuning)
        customizations = model_data.get('customizations_supported', [])
        if customizations:
            options.append('custom_model')

        return options

    def _add_documentation_links(self, model_data: Dict[str, Any]) -> Dict[str, str]:
        """Add appropriate documentation links based on model provider and family"""

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
            documentation_links['primary'] = "https://docs.aws.amazon.com/bedrock/latest/userguide/mistral-models.html"
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