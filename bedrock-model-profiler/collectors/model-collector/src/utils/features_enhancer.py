"""
Model Features Enhancer - Optimized Version
Enhances models with cross-region inference profiles and intelligent metadata
"""

import boto3
import logging
from typing import Dict, List, Optional, Any
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class ModelFeaturesEnhancer:
    """Optimized model features enhancer for Phase 4"""

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
        """Enhance models with optimized features (Phase 4)"""
        self.logger.info(f"🔧 Enhancing {len(models)} models with optimized features...")

        # Gather inference profiles using boto3 (cross-region inference)
        inference_profiles = self._gather_inference_profiles_optimized()

        enhanced_models = {}
        processed = 0

        for model_id, model_data in models.items():
            try:
                enhanced_model = model_data.copy()

                # Add inference profile data (cross-region inference)
                enhanced_model = self._add_inference_profiles(enhanced_model, inference_profiles)

                # Extract batch inference support from pricing data (no API testing)
                enhanced_model = self._add_batch_inference_from_pricing(enhanced_model)

                # Generate intelligent metadata (capabilities, use cases, languages, docs)
                enhanced_model = self._add_intelligent_metadata(enhanced_model)

                enhanced_models[model_id] = enhanced_model
                processed += 1

                if processed % 10 == 0:
                    self.logger.info(f"Enhanced {processed}/{len(models)} models...")

            except Exception as e:
                self.logger.warning(f"Failed to enhance {model_id}: {e}")
                enhanced_models[model_id] = model_data  # Use original data

        self.logger.info(f"✅ Enhanced {processed} models successfully with optimized Phase 4")
        return enhanced_models

    def _gather_inference_profiles_optimized(self) -> Dict[str, Any]:
        """Parallelized inference profiles gathering using ThreadPoolExecutor"""
        self.logger.info("🔗 Gathering cross-region inference profiles with parallel workers...")

        start_time = time.time()
        max_workers = min(10, len(self.regions))  # Same as quotas collector

        self.logger.info(f"🚀 Starting parallel inference profile collection from {len(self.regions)} regions (workers={max_workers})...")

        # Collect results from all regions in parallel
        region_results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all regions for parallel processing
            future_to_region = {
                executor.submit(self._gather_profiles_from_region, region): region
                for region in self.regions
            }

            # Collect results as they complete
            for future in as_completed(future_to_region):
                region = future_to_region[future]
                try:
                    result = future.result()
                    if result:  # Only add if we got results
                        region_results.append(result)
                        self.logger.debug(f"✅ [{region}] Collected {result.get('profiles_count', 0)} profiles")
                except Exception as e:
                    self.logger.debug(f"❌ [{region}] Error collecting profiles: {e}")

        # Merge all results into final structure
        inference_profiles = self._merge_profile_results(region_results)
        total_profiles = sum(len(data.get('profiles', [])) for data in inference_profiles.values())

        elapsed_time = time.time() - start_time
        self.logger.info(f"✅ Parallel inference profile collection complete from {len(self.regions)} regions in {elapsed_time:.2f}s")
        self.logger.info(f"✅ Gathered {total_profiles} inference profiles for cross-region inference")

        return inference_profiles

    def _gather_profiles_from_region(self, region: str) -> Optional[Dict[str, Any]]:
        """Gather inference profiles from a single region (worker function)"""
        try:
            bedrock = self.session.client('bedrock', region_name=region)
            response = bedrock.list_inference_profiles()
            region_profiles = response.get('inferenceProfileSummaries', [])

            if not region_profiles:
                return None

            region_result = {
                'region': region,
                'profiles': [],
                'profiles_count': 0
            }

            for profile in region_profiles:
                profile_id = profile.get('inferenceProfileId', '')
                profile_name = profile.get('inferenceProfileName', '')
                profile_type = profile.get('type', '')

                try:
                    # Get detailed profile information
                    profile_details = bedrock.get_inference_profile(
                        inferenceProfileIdentifier=profile_id
                    )

                    # Process models in this profile
                    models = profile_details.get('models', [])
                    for model_info in models:
                        model_arn = model_info.get('modelArn', model_info) if isinstance(model_info, dict) else model_info
                        model_id = self._extract_model_id_from_arn(model_arn)

                        if model_id:
                            region_result['profiles'].append({
                                'model_id': model_id,
                                'profile_id': profile_id,
                                'profile_name': profile_name,
                                'type': profile_type,
                                'source_region': region,
                                'description': profile_details.get('description', '')
                            })

                    region_result['profiles_count'] += 1

                except Exception as e:
                    self.logger.debug(f"Could not get details for profile {profile_id} in {region}: {e}")

            return region_result if region_result['profiles'] else None

        except Exception as e:
            self.logger.debug(f"Error gathering inference profiles from {region}: {e}")
            return None

    def _merge_profile_results(self, region_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge profile results from all regions into final structure"""
        inference_profiles = {}

        for region_result in region_results:
            for profile_data in region_result.get('profiles', []):
                model_id = profile_data['model_id']

                if model_id not in inference_profiles:
                    inference_profiles[model_id] = {
                        'supported': True,
                        'profiles': [],
                        'source_regions': set(),
                        'unique_profile_keys': set(),  # Track unique profile_id + source_region combinations
                        'total_profiles': 0
                    }

                # Create unique key from profile_id + source_region to detect true duplicates
                profile_key = f"{profile_data['profile_id']}|{profile_data['source_region']}"

                # Only add if this profile_id + source_region combination is not already present
                if profile_key not in inference_profiles[model_id]['unique_profile_keys']:
                    inference_profiles[model_id]['profiles'].append({
                        'profile_id': profile_data['profile_id'],
                        'profile_name': profile_data['profile_name'],
                        'type': profile_data['type'],
                        'source_region': profile_data['source_region'],
                        'description': profile_data['description']
                    })
                    inference_profiles[model_id]['unique_profile_keys'].add(profile_key)

                inference_profiles[model_id]['source_regions'].add(profile_data['source_region'])

        # Convert sets to lists for JSON serialization and set final counts
        for model_id in inference_profiles:
            inference_profiles[model_id]['source_regions'] = sorted(list(inference_profiles[model_id]['source_regions']))
            # Count unique profiles (profile_id + source_region combinations)
            inference_profiles[model_id]['total_profiles'] = len(inference_profiles[model_id]['unique_profile_keys'])
            # Remove the temporary set as it's not needed in final output
            del inference_profiles[model_id]['unique_profile_keys']

        return inference_profiles

    def _add_inference_profiles(self, model_data: Dict[str, Any], all_profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Add inference profile data to model"""
        model_id = model_data.get('model_id', '')

        if model_id in all_profiles:
            profile_data = all_profiles[model_id]
            model_data['cross_region_inference'] = {
                'supported': profile_data.get('supported', False),
                'profiles_count': profile_data.get('total_profiles', 0),
                'source_regions': profile_data.get('source_regions', []),
                'profiles': profile_data.get('profiles', [])
            }
        else:
            model_data['cross_region_inference'] = {
                'supported': False,
                'profiles_count': 0,
                'source_regions': [],
                'profiles': []
            }

        return model_data

    def _add_batch_inference_from_pricing(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract batch inference support from pricing data instead of API testing"""
        # Check if model has pricing data with batch inference indicators
        pricing_info = model_data.get('model_pricing', {})

        if pricing_info.get('is_pricing_available'):
            pricing_summary = pricing_info.get('pricing_summary', {})
            has_batch_pricing = pricing_summary.get('has_batch_pricing', False)
            available_regions = pricing_summary.get('available_regions', [])

            model_data['batch_inference_supported'] = {
                'supported': has_batch_pricing,
                'supported_regions': available_regions if has_batch_pricing else [],
                'coverage_percentage': len(available_regions) / max(len(model_data.get('regions_available', [])), 1) * 100 if has_batch_pricing else 0.0,
                'detection_method': 'pricing_data_analysis'
            }
        else:
            # Fallback: no pricing data available
            model_data['batch_inference_supported'] = {
                'supported': False,
                'supported_regions': [],
                'coverage_percentage': 0.0,
                'detection_method': 'no_pricing_data'
            }

        return model_data

    def _add_intelligent_metadata(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent model metadata through AI-powered analysis"""
        # Capabilities extraction
        model_data = self._extract_capabilities(model_data)

        # Use cases derivation
        model_data = self._derive_use_cases(model_data)

        # Language support inference
        model_data = self._infer_language_support(model_data)

        # Consumption options detection
        model_data = self._detect_consumption_options(model_data)

        # Documentation links generation
        model_data = self._generate_documentation_links(model_data)

        return model_data

    def _extract_capabilities(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract capabilities from model modalities and metadata"""
        capabilities = set()

        input_modalities = model_data.get('model_modalities', {}).get('input_modalities', [])
        output_modalities = model_data.get('model_modalities', {}).get('output_modalities', [])

        # Base capabilities from modalities
        if 'TEXT' in input_modalities and 'TEXT' in output_modalities:
            capabilities.update(['chat', 'text_generation'])

        if 'IMAGE' in input_modalities:
            capabilities.update(['multimodal', 'vision', 'image_analysis'])

        if len(input_modalities) > 1:
            capabilities.add('multimodal')

        # Model-specific capabilities based on name/provider
        model_id = model_data.get('model_id', '').lower()
        if 'claude' in model_id:
            capabilities.update(['reasoning', 'analysis', 'code_generation'])
        elif 'titan' in model_id:
            capabilities.update(['embedding', 'summarization'])

        model_data['model_capabilities'] = sorted(list(capabilities))
        return model_data

    def _derive_use_cases(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Derive use cases from capabilities and model characteristics"""
        use_cases = set()
        capabilities = model_data.get('model_capabilities', [])

        # Map capabilities to use cases
        capability_use_case_map = {
            'chat': ['conversational_ai', 'customer_support', 'virtual_assistants'],
            'text_generation': ['content_creation', 'creative_writing', 'documentation'],
            'multimodal': ['document_analysis', 'visual_qa', 'content_understanding'],
            'vision': ['image_captioning', 'visual_analysis', 'ocr'],
            'reasoning': ['complex_analysis', 'problem_solving', 'decision_support'],
            'code_generation': ['software_development', 'code_review', 'debugging']
        }

        for capability in capabilities:
            if capability in capability_use_case_map:
                use_cases.update(capability_use_case_map[capability])

        model_data['model_use_cases'] = sorted(list(use_cases))
        return model_data

    def _infer_language_support(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Infer language support based on provider and model characteristics"""
        provider = model_data.get('model_provider', '').lower()

        # Provider-based language support
        if 'anthropic' in provider:
            languages = ['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Japanese']
        elif 'amazon' in provider:
            languages = ['English', 'Spanish', 'French', 'German', 'Portuguese']
        elif 'meta' in provider:
            languages = ['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Hindi', 'Thai']
        elif 'deepseek' in provider or 'qwen' in provider:
            languages = ['English', 'Chinese']
        elif 'mistral' in provider:
            languages = ['English', 'French', 'German', 'Spanish', 'Italian']
        else:
            languages = ['English']  # Default fallback

        model_data['languages_supported'] = languages
        return model_data

    def _detect_consumption_options(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect consumption options from pricing data groups"""
        consumption_options = []

        # First priority: Use pricing types from pricing integrator (most accurate)
        pricing_info = model_data.get('model_pricing', {})
        if pricing_info.get('is_pricing_available'):
            pricing_summary = pricing_info.get('pricing_summary', {})
            pricing_types = pricing_summary.get('pricing_types', [])

            if pricing_types:
                # Use pricing types directly as consumption options (no transformation)
                # User wants consumption_options to be exactly the same as pricing_types
                consumption_options = pricing_types.copy()

            # If we got consumption options from pricing data, use them
            if consumption_options:
                model_data['consumption_options'] = sorted(consumption_options)
                return model_data

        # Fallback: Use inference types and batch support (legacy logic)
        inference_types = model_data.get('inference_types_supported', [])
        batch_supported = model_data.get('batch_inference_supported', {}).get('supported', False)

        if 'ON_DEMAND' in inference_types:
            consumption_options.append('on_demand')
        if 'PROVISIONED' in inference_types:
            consumption_options.append('provisioned_throughput')
        if batch_supported:
            consumption_options.append('batch_inference')

        # Default fallback
        if not consumption_options:
            consumption_options = ['on_demand']

        model_data['consumption_options'] = sorted(consumption_options)
        return model_data

    def _generate_documentation_links(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate documentation links using Config patterns"""
        from config import Config

        model_id = model_data.get('model_id', '')
        provider = model_data.get('model_provider', '')

        # Use Config method for documentation links
        doc_links = Config.generate_documentation_links(model_id, provider)
        model_data['documentation_links'] = doc_links

        return model_data

    def _extract_model_id_from_arn(self, model_arn: str) -> str:
        """Extract model ID from ARN"""
        if not model_arn:
            return ''

        # ARN format: arn:aws:bedrock:region::foundation-model/model-id
        # or just model-id directly
        if model_arn.startswith('arn:'):
            parts = model_arn.split('/')
            if len(parts) > 1:
                return parts[-1]  # Last part after the final '/'

        return model_arn  # Assume it's already a model ID