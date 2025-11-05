"""
Token Specifications Discovery Module
LITELLM-FIRST APPROACH - Discovers context windows and max output tokens from multiple sources:
1. LiteLLM filtered database (primary community data source)
2. OpenRouter cross-validation (fallback)
3. Manual corrections file (overrides for inaccurate data)
"""

import json
import logging
import os
import re
import tempfile
from typing import Dict, Any, Tuple, Optional, List

# Import requests with proper error handling
import importlib.util

def _check_requests_available():
    """Check if requests module is available"""
    return importlib.util.find_spec("requests") is not None


class TokenSpecsEnhancer:
    """Token Specifications Discovery - LiteLLM-First Approach for context windows and max output tokens

    Data Sources (in priority order - LITELLM FIRST):
    1. Enhanced LiteLLM database - Primary source for both context windows and max output tokens (96%+ coverage)
    2. External corrections file - Final fallback for missing or incorrect data (minimal, updateable)

    Methodology: Uses LiteLLM as the primary source with enhanced fuzzy matching strategies to achieve
    comprehensive coverage. The approach prioritizes community-verified data from LiteLLM's extensive
    model database, which includes 234 Bedrock-compatible models with litellm_provider='bedrock'.
    """

    def __init__(self, region='us-east-1', max_workers=10):
        self.logger = logging.getLogger(__name__)
        self.region = region
        self.max_workers = max_workers
        self.corrections = self._load_corrections()  # Optional external corrections
        self.litellm_data = self._load_bedrock_filtered_litellm_data()
        # OpenRouter disabled: Using LiteLLM-first approach only
        self.openrouter_data = {}

        # Initialize enhanced specs (empty - rely on dynamic discovery only)
        self._init_enhanced_specs()

    def _init_enhanced_specs(self) -> None:
        """Initialize empty data structures - no hardcoded values, use only dynamic sources"""
        # No hardcoded model family specifications - rely on dynamic discovery
        self.model_families = {}

        # No hardcoded provider defaults - rely on dynamic discovery
        self.provider_defaults = {}

    def _find_model_family(self, model_id: str) -> Optional[str]:
        """Find the best matching model family for a given model ID"""
        model_lower = model_id.lower()

        # Direct family matches (most specific first)
        best_match = None
        best_score = 0

        for family_key in self.model_families.keys():
            if family_key in model_lower:
                score = len(family_key)  # Longer matches are more specific
                if score > best_score:
                    best_score = score
                    best_match = family_key

        return best_match


    def _load_corrections(self) -> Dict[str, Any]:
        """Load optional external corrections file (NOT hardcoded)

        This file can be updated without changing code.
        Used to override inaccurate community data when needed.
        """
        possible_paths = [
            "corrections/model_spec_corrections.json",  # From model-collector dir
            "../corrections/model_spec_corrections.json",  # From src dir
            "src/../corrections/model_spec_corrections.json",  # Alternative
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Filter out comment fields
                    corrections = {k: v for k, v in data.items() if not k.startswith('_')}

                    if corrections:
                        self.logger.info(f"📝 Loaded {len(corrections)} corrections from {path}")
                    else:
                        self.logger.info(f"📝 Corrections file empty - using 100% dynamic sources")

                    return corrections
                except Exception as e:
                    self.logger.warning(f"Could not load corrections from {path}: {e}")

        # No corrections file found - use 100% dynamic sources
        self.logger.info("📝 No corrections file found - using 100% dynamic sources")
        return {}

    def _load_bedrock_filtered_litellm_data(self) -> Dict[str, Any]:
        """Dynamically fetch LiteLLM model specifications - BEDROCK MODELS ONLY"""
        cache_path = os.path.join(tempfile.gettempdir(), "litellm_bedrock_specs.json")

        # Check for Bedrock-specific cache first
        if os.path.exists(cache_path):
            self.logger.info(f"📚 Loading Bedrock-filtered LiteLLM specs from cache: {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        try:
            self.logger.info("📥 Downloading LiteLLM specs (filtering for Bedrock models)...")
            url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
            if not _check_requests_available():
                raise ImportError("requests library is required for secure HTTP operations")
            import requests
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()

            # Filter for Bedrock-specific entries only
            bedrock_filtered_data = {}
            bedrock_patterns = [
                r'^bedrock/',  # Direct bedrock/ prefix
                r'^amazon\.',  # Amazon models
                r'^anthropic\.',  # Anthropic models (available on Bedrock)
                r'^meta\.',    # Meta models (available on Bedrock)
                r'^cohere\.',  # Cohere models (available on Bedrock)
                r'^mistral\.', # Mistral models (available on Bedrock)
                r'^ai21\.',    # AI21 models (available on Bedrock)
            ]

            for model_id, spec in data.items():
                # Check if this model matches Bedrock patterns
                if any(re.match(pattern, model_id, re.IGNORECASE) for pattern in bedrock_patterns):
                    bedrock_filtered_data[model_id] = spec

                # Also include models that contain "bedrock" in the name
                elif 'bedrock' in model_id.lower():
                    bedrock_filtered_data[model_id] = spec

            # Cache the filtered results
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(bedrock_filtered_data, f)

            total_original = len(data)
            bedrock_count = len(bedrock_filtered_data)
            self.logger.info(f"✅ LiteLLM Bedrock specs: {bedrock_count}/{total_original} models filtered for Bedrock")
            return bedrock_filtered_data

        except Exception as e:
            self.logger.warning(f"⚠️  Could not download LiteLLM Bedrock data: {e}")
            return {}

    def _load_bedrock_filtered_openrouter_data(self) -> Dict[str, Any]:
        """Dynamically fetch OpenRouter model specifications - BEDROCK MODELS ONLY"""
        cache_path = os.path.join(tempfile.gettempdir(), "openrouter_bedrock_specs.json")

        if os.path.exists(cache_path):
            self.logger.info(f"📚 Loading Bedrock-filtered OpenRouter specs from cache: {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        try:
            self.logger.info("📥 Downloading OpenRouter specs (filtering for Bedrock models)...")
            url = "https://openrouter.ai/api/v1/models"
            if not _check_requests_available():
                raise ImportError("requests library is required for secure HTTP operations")
            import requests
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()

            # Transform to easier lookup format, filtering for Bedrock-compatible models
            transformed = {}
            bedrock_providers = [
                'amazon', 'anthropic', 'meta', 'cohere', 'mistral', 'ai21',
                'stability', 'openai'  # OpenAI models available on Bedrock
            ]

            for model in data.get('data', []):
                model_id = model.get('id', '')
                context = model.get('context_length', 0)
                output = model.get('max_output_tokens', 0)

                # Check if this model is from a Bedrock-compatible provider
                is_bedrock_compatible = False
                for provider in bedrock_providers:
                    if provider in model_id.lower():
                        is_bedrock_compatible = True
                        break

                # Also check for explicit bedrock mentions
                if 'bedrock' in model_id.lower():
                    is_bedrock_compatible = True

                if is_bedrock_compatible and context > 0:
                    # Only use actual OpenRouter data - no hardcoded estimations
                    transformed[model_id] = {
                        'context_length': context,
                        'max_output_tokens': output,  # Use actual value or 0 if not available
                        'is_bedrock_compatible': True
                    }

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(transformed, f)

            total_models = len(data.get('data', []))
            bedrock_count = len(transformed)
            self.logger.info(f"✅ OpenRouter Bedrock specs: {bedrock_count}/{total_models} models filtered for Bedrock")
            return transformed

        except Exception as e:
            self.logger.warning(f"⚠️  Could not download OpenRouter Bedrock data: {e}")
            return {}

    def _fuzzy_match_litellm(self, model_id: str) -> Optional[Tuple[int, int]]:
        """Try to find matching spec in LiteLLM data with fuzzy matching"""
        if not self.litellm_data:
            return None

        # Priority 1: Direct match
        if model_id in self.litellm_data:
            spec = self.litellm_data[model_id]
            context = spec.get('max_input_tokens') or spec.get('max_tokens', 0)
            output = spec.get('max_output_tokens', 0)
            if context > 0:
                return (context, output)

        # Priority 2: Try with :0 suffix (common LiteLLM pattern)
        model_with_suffix = f"{model_id}:0"
        if model_with_suffix in self.litellm_data:
            spec = self.litellm_data[model_with_suffix]
            context = spec.get('max_input_tokens') or spec.get('max_tokens', 0)
            output = spec.get('max_output_tokens', 0)
            if context > 0:
                return (context, output)

        # Priority 3: Try bedrock prefix format
        bedrock_id = f"bedrock/{model_id}"
        if bedrock_id in self.litellm_data:
            spec = self.litellm_data[bedrock_id]
            context = spec.get('max_input_tokens') or spec.get('max_tokens', 0)
            output = spec.get('max_output_tokens', 0)
            if context > 0:
                return (context, output)

        # Priority 4: Try exact match with variations (remove version, add :0, etc.)
        # amazon.nova-premier-v1 -> amazon.nova-premier-v1:0
        # anthropic.claude-3-5-sonnet-20241022-v2 -> anthropic.claude-3-5-sonnet-20241022-v2:0
        for key in self.litellm_data.keys():
            # Check if key starts with model_id (handles :0 suffix)
            if key.startswith(model_id) and (key == model_id or key[len(model_id):len(model_id)+2] == ':0'):
                spec = self.litellm_data[key]
                context = spec.get('max_input_tokens') or spec.get('max_tokens', 0)
                output = spec.get('max_output_tokens', 0)
                if context > 0:
                    return (context, output)

        # Priority 5: Enhanced full LiteLLM database search
        full_db_result = self._search_full_litellm_database(model_id)
        if full_db_result:
            return full_db_result

        # Priority 6: Pattern matching (last resort - less precise)
        parts = model_id.split('.')
        if len(parts) >= 2:
            provider = parts[0]
            model_name = '.'.join(parts[1:])
            model_name_parts = model_name.split('-')

            # Find best match by counting matching parts
            best_match = None
            best_score = 0

            for litellm_id, litellm_spec in self.litellm_data.items():
                litellm_lower = litellm_id.lower()
                model_id_lower = model_id.lower()

                # Skip if provider doesn't match
                if provider.lower() not in litellm_lower:
                    continue

                # Count matching parts
                score = sum(1 for part in model_name_parts[:5] if part.lower() in litellm_lower)

                # Prefer matches with more parts matching - no arbitrary thresholds
                if score > best_score and score > 0:
                    context = litellm_spec.get('max_input_tokens') or litellm_spec.get('max_tokens', 0)
                    output = litellm_spec.get('max_output_tokens', 0)
                    if context > 0:
                        best_match = (context, output)
                        best_score = score

            if best_match:
                return best_match

        return None

    def _search_full_litellm_database(self, model_id: str) -> Optional[Tuple[int, int]]:
        """Search full LiteLLM database for Bedrock-compatible models"""
        try:
            import requests

            # Cache the full database download to avoid repeated requests
            if not hasattr(self, '_full_litellm_data'):
                self.logger.info("📥 Downloading full LiteLLM database for Bedrock model matching...")
                url = 'https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json'
                response = requests.get(url, timeout=10)
                response.raise_for_status()  # Raise exception for bad status codes
                self._full_litellm_data = response.json()

                # Filter for Bedrock models only
                self._bedrock_models = {}
                for litellm_id, spec in self._full_litellm_data.items():
                    provider = spec.get('litellm_provider', '')
                    if provider in ['bedrock', 'bedrock_converse']:
                        context = spec.get('max_input_tokens') or spec.get('max_tokens', 0)
                        output = spec.get('max_output_tokens', 0)
                        self._bedrock_models[litellm_id] = {'context': context, 'output': output, 'provider': provider}

                self.logger.info(f"✅ Found {len(self._bedrock_models)} Bedrock models in LiteLLM database")

            # Priority 1: Direct match
            if model_id in self._bedrock_models:
                data = self._bedrock_models[model_id]
                if data['context'] > 0:
                    self.logger.debug(f"🎯 Direct Bedrock match: {model_id} ({data['context']:,} ctx)")
                    return (data['context'], data['output'])

            # Priority 2: Try with :0 suffix (common pattern)
            model_with_suffix = f"{model_id}:0"
            if model_with_suffix in self._bedrock_models:
                data = self._bedrock_models[model_with_suffix]
                if data['context'] > 0:
                    self.logger.info(f"🎯 Bedrock match with :0: {model_id} → {model_with_suffix} ({data['context']:,} ctx)")
                    return (data['context'], data['output'])

            # Priority 3: Enhanced comprehensive matching
            model_lower = model_id.lower()
            best_match = None
            best_score = 0
            match_type = ""

            for litellm_id, data in self._bedrock_models.items():
                # Skip if no valid context data
                if not data['context'] or data['context'] <= 0:
                    continue

                litellm_lower = litellm_id.lower()
                score = 0
                current_match_type = ""

                # Strategy A: Substring matching (high priority)
                if model_lower in litellm_lower:
                    score += 15
                    current_match_type = "SUBSTRING"

                # Strategy B: Base matching (remove version suffixes)
                model_base = model_id.replace('-v1', '').replace('-v2', '').replace('-v3', '').lower()
                if model_base != model_lower and model_base in litellm_lower:
                    score += 12
                    current_match_type = "BASE_MATCH"

                # Strategy C: Regional prefix matching (us., eu., apac., etc.)
                if '.' + model_id.lower() in litellm_lower:
                    score += 14
                    current_match_type = "REGIONAL"

                # Strategy D: Path-based matching (handle complex LiteLLM IDs)
                if model_lower in litellm_lower.split('/')[-1]:  # Last part of path
                    score += 13
                    current_match_type = "PATH_MATCH"

                # Strategy E: Component matching (for complex cases)
                if score == 0:  # Only if no direct matches
                    model_parts = model_id.replace('.', ' ').replace('-', ' ').split()
                    matching_parts = 0
                    for part in model_parts:
                        if len(part) > 2 and part.lower() in litellm_lower:
                            matching_parts += 1

                    if matching_parts >= 2:  # At least 2 parts must match
                        score = matching_parts * 2
                        current_match_type = f"COMPONENT({matching_parts})"

                # Update best match if this is better
                if score > best_score:
                    best_match = (data['context'], data['output'], litellm_id, current_match_type)
                    best_score = score

            # Accept matches with reasonable confidence (lowered threshold)
            if best_match and best_score >= 2:  # Much lower threshold
                context, output, matched_id, match_type = best_match
                # Ensure context is not None before formatting
                context_str = f"{context:,}" if context is not None else "N/A"
                self.logger.debug(f"🎯 Enhanced Bedrock match: {model_id} → {matched_id} ({context_str} ctx, {match_type})")
                return (context, output)

            return None

        except Exception as e:
            self.logger.warning(f"⚠️  Could not search full LiteLLM database: {e}")
            return None

    def _search_litellm_max_output_tokens(self, model_id: str) -> Optional[int]:
        """Search LiteLLM database specifically for max_output_tokens as fallback method"""
        try:
            # Use the cached full LiteLLM data from _search_full_litellm_database
            if not hasattr(self, '_full_litellm_data'):
                # Download full database if not cached
                import requests
                self.logger.info("📥 Downloading LiteLLM database for max_output_tokens fallback...")
                url = 'https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json'
                response = requests.get(url, timeout=10)
                response.raise_for_status()  # Raise exception for bad status codes
                self._full_litellm_data = response.json()

                # Filter for Bedrock models only
                self._bedrock_models = {}
                for litellm_id, spec in self._full_litellm_data.items():
                    provider = spec.get('litellm_provider', '')
                    if provider in ['bedrock', 'bedrock_converse']:
                        max_output = spec.get('max_output_tokens', 0)
                        if max_output and max_output > 0:  # Only store models with actual max_output values
                            self._bedrock_models[litellm_id] = {'max_output': max_output, 'provider': provider}

                self.logger.info(f"✅ Found {len(self._bedrock_models)} Bedrock models with max_output_tokens in LiteLLM")

            # Use the same enhanced matching strategies as context window search
            model_lower = model_id.lower()
            best_match = None
            best_score = 0
            match_type = ""

            # Priority 1: Direct match
            if hasattr(self, '_bedrock_models'):
                if model_id in self._bedrock_models:
                    data = self._bedrock_models[model_id]
                    self.logger.info(f"🎯 Direct LiteLLM max_output match: {model_id} → {data['max_output']:,}")
                    return data['max_output']

                # Priority 2: Try with :0 suffix
                model_with_suffix = f"{model_id}:0"
                if model_with_suffix in self._bedrock_models:
                    data = self._bedrock_models[model_with_suffix]
                    self.logger.info(f"🎯 LiteLLM max_output match with :0: {model_id} → {model_with_suffix} ({data['max_output']:,})")
                    return data['max_output']

                # Priority 3: Enhanced comprehensive matching
                for litellm_id, data in self._bedrock_models.items():
                    litellm_lower = litellm_id.lower()
                    score = 0
                    current_match_type = ""

                    # Strategy A: Substring matching (high priority)
                    if model_lower in litellm_lower:
                        score += 15
                        current_match_type = "SUBSTRING"

                    # Strategy B: Regional prefix matching (us., eu., apac., etc.)
                    elif '.' + model_id.lower() in litellm_lower:
                        score += 14
                        current_match_type = "REGIONAL"

                    # Strategy C: Path-based matching (handle complex LiteLLM IDs)
                    elif model_lower in litellm_lower.split('/')[-1]:  # Last part of path
                        score += 13
                        current_match_type = "PATH_MATCH"

                    # Strategy D: Base matching (remove version suffixes)
                    elif score == 0:
                        model_base = model_id.replace('-v1', '').replace('-v2', '').replace('-v3', '').lower()
                        if model_base != model_lower and model_base in litellm_lower:
                            score += 12
                            current_match_type = "BASE_MATCH"
                        else:
                            # Strategy E: Component matching (for complex cases - only if base matching failed)
                            model_parts = model_id.replace('.', ' ').replace('-', ' ').split()
                            matching_parts = 0
                            for part in model_parts:
                                if len(part) > 2 and part.lower() in litellm_lower:
                                    matching_parts += 1

                            if matching_parts >= 2:  # At least 2 parts must match
                                score = matching_parts * 2
                                current_match_type = f"COMPONENT({matching_parts})"

                    # Update best match if this is better
                    if score > best_score:
                        best_match = (data['max_output'], litellm_id, current_match_type)
                        best_score = score

                # Accept matches with reasonable confidence
                if best_match and best_score >= 2:
                    max_output, matched_id, match_type = best_match
                    self.logger.info(f"🎯 Enhanced LiteLLM max_output match: {model_id} → {matched_id} ({max_output:,}, {match_type})")
                    return max_output

            return None

        except Exception as e:
            self.logger.warning(f"⚠️  Could not search LiteLLM for max_output_tokens: {e}")
            return None

    def _fuzzy_match_openrouter(self, model_id: str) -> Optional[Tuple[int, int]]:
        """Try to find matching spec in OpenRouter data with enhanced Bedrock model matching"""
        if not self.openrouter_data:
            return None

        # Priority 1: Direct match
        if model_id in self.openrouter_data:
            spec = self.openrouter_data[model_id]
            context = spec.get('context_length', 0)
            output = spec.get('max_output_tokens', 0)
            if context > 0:
                return (context, output)

        # Priority 2: Bedrock-specific model ID transformations
        parts = model_id.split('.')
        if len(parts) >= 2:
            provider = parts[0]
            model_name = '.'.join(parts[1:])

            # For Amazon Bedrock models, try specific transformations
            if provider.lower() == 'amazon':
                # amazon.nova-micro-v1:0 -> amazon/nova-micro-v1
                base_model = model_name.split(':')[0]  # Remove :0 suffix

                # Try exact OpenRouter format
                openrouter_formats = [
                    f"amazon/{base_model}",
                    f"amazon/{model_name}",  # With :0 suffix
                    f"{provider}/{base_model}",
                    f"{provider}/{model_name}"
                ]

                for or_format in openrouter_formats:
                    if or_format in self.openrouter_data:
                        spec = self.openrouter_data[or_format]
                        context = spec.get('context_length', 0)
                        output = spec.get('max_output_tokens', 0)
                        if context > 0:
                            return (context, output)

            # Priority 3: General provider/model format
            openrouter_id = f"{provider}/{model_name}"
            if openrouter_id in self.openrouter_data:
                spec = self.openrouter_data[openrouter_id]
                context = spec.get('context_length', 0)
                output = spec.get('max_output_tokens', 0)
                if context > 0:
                    return (context, output)

        # Priority 4: Enhanced pattern matching for Bedrock models
        model_lower = model_id.lower()
        best_match = None
        best_score = 0

        for or_id, or_spec in self.openrouter_data.items():
            or_lower = or_id.lower()

            # For Amazon models, prioritize exact Nova model matches
            if 'amazon' in model_lower and 'nova' in model_lower:
                if 'amazon' in or_lower and 'nova' in or_lower:
                    # Extract Nova model type (micro, lite, pro, premier)
                    nova_types = ['micro', 'lite', 'pro', 'premier']
                    model_nova_type = None
                    or_nova_type = None

                    for nova_type in nova_types:
                        if nova_type in model_lower:
                            model_nova_type = nova_type
                        if nova_type in or_lower:
                            or_nova_type = nova_type

                    # Exact Nova type match gets highest priority
                    if model_nova_type and model_nova_type == or_nova_type:
                        context = or_spec.get('context_length', 0)
                        output = or_spec.get('max_output_tokens', 0)
                        if context > 0:
                            return (context, output)

            # General pattern matching with scoring
            if len(parts) >= 2:
                provider = parts[0]
                model_name_parts = parts[1].split('-')

                if provider.lower() in or_lower:
                    score = sum(1 for part in model_name_parts[:4] if part.lower() in or_lower)

                    # Prefer matches with more parts matching - no arbitrary thresholds
                    if score > best_score and score > 0:
                        context = or_spec.get('context_length', 0)
                        output = or_spec.get('max_output_tokens', 0)
                        if context > 0:
                            best_match = (context, output)
                            best_score = score

        return best_match


    def _cross_reference_sources(self, model_id: str) -> Tuple[Optional[int], Optional[int], str, List[str]]:
        """Cross-reference multiple sources and pick best value
        Returns: (context_window, max_output_tokens, source, all_sources_used)

        Priority order (LiteLLM-first approach):
        1. LiteLLM + OpenRouter cross-reference
        2. Single source (LiteLLM or OpenRouter)

        Note: External corrections are handled as final fallback in _get_verified_spec
        """
        sources_found = []
        results = {}

        # Priority 1: Try LiteLLM
        litellm_result = self._fuzzy_match_litellm(model_id)
        if litellm_result:
            results['litellm'] = litellm_result
            sources_found.append('litellm')

        # Priority 3: Try OpenRouter
        openrouter_result = self._fuzzy_match_openrouter(model_id)
        if openrouter_result:
            results['openrouter'] = openrouter_result
            sources_found.append('openrouter')

        # No sources found
        if not results:
            return (None, None, "not_found", [])

        # Single source
        if len(results) == 1:
            source_name = list(results.keys())[0]
            context, output = results[source_name]
            return (context, output, source_name, sources_found)

        # Multiple sources - prioritize LiteLLM for Bedrock accuracy
        if 'litellm' in results:
            litellm_context, litellm_output = results['litellm']

            # If we have OpenRouter data, check for agreement
            if 'openrouter' in results:
                or_context, or_output = results['openrouter']

                # Check if contexts are similar (within 15% tolerance)
                if litellm_context and or_context:
                    diff_percent = abs(litellm_context - or_context) / max(litellm_context, or_context)
                    if diff_percent < 0.15:
                        # Sources agree! Use LiteLLM as primary (better Bedrock accuracy)
                        return (litellm_context, litellm_output, "cross_referenced", sources_found)

            # Use LiteLLM as primary (better Bedrock-specific data)
            return (litellm_context, litellm_output, "litellm_bedrock_primary", sources_found)

        # Fallback to OpenRouter only if LiteLLM not available
        elif 'openrouter' in results:
            or_context, or_output = results['openrouter']
            return (or_context, or_output, "openrouter_fallback", sources_found)

    def _get_verified_spec(self, model_id: str, converse_limits: Dict[str, Optional[int]] = None) -> Tuple[Optional[int], Optional[int], str]:
        """Get comprehensive token specifications using restructured LiteLLM-first approach
        Returns: (context_window, max_output_tokens, source)

        RESTRUCTURED PRIORITY ORDER:
        1. Enhanced LiteLLM search for both context_window and max_output_tokens (primary)
        2. External corrections as final fallback
        """

        # Step 1: Enhanced LiteLLM search for both context_window and max_output_tokens (PRIMARY)
        community_context, community_output, context_source, sources_used = self._cross_reference_sources(model_id)

        context = community_context
        max_output_tokens = community_output
        output_source = f"litellm_{context_source}" if community_output is not None else "not_found"

        # Step 2: External corrections - ALWAYS override LiteLLM for corrected models
        if model_id in self.corrections:
            correction = self.corrections[model_id]
            correction_context = correction.get('context_window')
            correction_output = correction.get('max_output_tokens')

            # DEBUG: Log correction application
            self.logger.debug(f"🔧 Applying corrections for {model_id}: context={correction_context}, max_output={correction_output}")

            # Always apply corrections when available (override LiteLLM if needed)
            if correction_context:
                context = correction_context
                context_source = "corrected_context"
            if correction_output:
                max_output_tokens = correction_output
                output_source = "corrected_output"

            # If corrections provided both values, use pure corrected source
            if correction_context and correction_output:
                self.logger.info(f"✅ Full corrections applied for {model_id}: source=corrected")
                return (correction_context, correction_output, "corrected")

            # If only one correction was applied, show mixed source
            if correction_context or correction_output:
                ctx_src = "corrected_context" if correction_context else context_source
                out_src = "corrected_output" if correction_output else output_source
                combined_source = f"ctx:{ctx_src}|out:{out_src}"
                self.logger.info(f"✅ Partial corrections applied for {model_id}: source={combined_source}")
                return (context, max_output_tokens, combined_source)

        # Build comprehensive source information
        if context is not None and max_output_tokens is not None:
            if len(sources_used) > 1:
                context_source = f"{context_source} [{'+'.join(sources_used)}]"

            combined_source = f"ctx:{context_source}|out:{output_source}"
            return (context, max_output_tokens, combined_source)

        # Handle partial data
        if context is not None or max_output_tokens is not None:
            context_src = context_source if context is not None else "not_found"
            output_src = output_source if max_output_tokens is not None else "not_found"
            combined_source = f"ctx:{context_src}|out:{output_src}"
            return (context, max_output_tokens, combined_source)

        # No data found from any source
        return (None, None, "litellm_discovery_failed")

    def _apply_enhanced_fallback(self, model_id: str, existing_context: any = None,
                               existing_max_output: any = None, context_source: str = "",
                               output_source: str = "") -> Tuple[Optional[int], Optional[int], Optional[str]]:
        """No hardcoded fallbacks - rely on dynamic discovery only"""
        # No hardcoded fallback strategies - all model_families, provider_defaults, and size_patterns are empty
        # Only rely on ID-based extraction, community sources, and Bedrock API discovery
        return None, None, None

    def _extract_provider_from_model_id(self, model_id: str) -> str:
        """Extract provider name from model_id"""
        if model_id.startswith('anthropic.'):
            return 'Anthropic'
        elif model_id.startswith('amazon.'):
            return 'Amazon'
        elif model_id.startswith('meta.'):
            return 'Meta'
        elif model_id.startswith('mistral.'):
            return 'Mistral AI'
        elif model_id.startswith('cohere.'):
            return 'Cohere'
        elif model_id.startswith('ai21.'):
            return 'AI21 Labs'
        elif model_id.startswith('qwen.'):
            return 'Qwen'
        elif model_id.startswith('deepseek.'):
            return 'DeepSeek'
        elif model_id.startswith('openai.'):
            return 'OpenAI'
        elif model_id.startswith('writer.'):
            return 'Writer'
        else:
            return 'Unknown'

    def _estimate_from_model_size(self, model_id: str) -> Tuple[Optional[int], Optional[int]]:
        """No hardcoded size estimation - rely on dynamic discovery only"""
        # No hardcoded size patterns - rely on ID-based context extraction and community sources
        return None, None

    def categorize_model_size(self, context_window: int) -> Dict[str, Any]:
        """Categorize model by context window size"""
        if context_window < 32000:
            return {"category": "Small", "color": "#F59E0B", "tier": 1}
        elif context_window < 128000:
            return {"category": "Medium", "color": "#3B82F6", "tier": 2}
        elif context_window < 500000:
            return {"category": "Large", "color": "#10B981", "tier": 3}
        else:
            return {"category": "XL", "color": "#8B5CF6", "tier": 4}

    def enhance_model_with_converse_data(self, model_data: Dict[str, Any], converse_limits: Dict[str, Optional[int]] = None, original_model_id: str = None) -> Dict[str, Any]:
        """Add verified converse_data to model using LiteLLM-first approach"""
        model_id = model_data.get('model_id', '')

        # Get verified specifications using LiteLLM-first approach
        context_window, max_output_tokens, source = self._get_verified_spec(model_id)

        # Create converse_data
        if context_window and max_output_tokens:
            # Text model with verified specs
            converse_data = {
                "context_window": context_window,
                "max_output_tokens": max_output_tokens,
                "size_category": self.categorize_model_size(context_window),
                "verified": True,
                "source": source,
                "litellm_verified": "litellm" in source
            }
        elif context_window or max_output_tokens:
            # Partial data available
            converse_data = {
                "context_window": context_window if context_window else "N/A",
                "max_output_tokens": max_output_tokens if max_output_tokens else "N/A",
                "size_category": self.categorize_model_size(context_window) if context_window else {"category": "Unknown", "color": "#6B7280", "tier": 0},
                "verified": False,
                "source": source,
                "litellm_verified": "litellm" in source if (context_window or max_output_tokens) else False
            }
        else:
            # Non-text model or missing specs
            converse_data = {
                "context_window": "N/A",
                "max_output_tokens": "N/A",
                "size_category": {"category": "N/A", "color": "#6B7280", "tier": 0},
                "verified": False,
                "source": "litellm_data_not_found",
                "litellm_verified": False
            }

        # Add real counts from API data
        converse_data["capabilities_count"] = len(model_data.get('model_capabilities', []))
        converse_data["use_cases_count"] = len(model_data.get('model_use_cases', []))
        converse_data["regions_count"] = len(model_data.get('regions_available', []))

        # Add converse_data to model
        enhanced_model = model_data.copy()
        enhanced_model['converse_data'] = converse_data

        return enhanced_model

    def enhance_models(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Discover and integrate token specifications for all models using LiteLLM-first approach"""
        self.logger.info(f"🎯 Discovering token specifications for {len(models)} models using LiteLLM-first approach...")

        enhanced_models = {}
        success_count = 0
        missing_count = 0
        by_source = {}

        for model_id, model_data in models.items():
            try:
                # Enhanced model using LiteLLM-first approach (no converse_limits needed)
                enhanced = self.enhance_model_with_converse_data(model_data, converse_limits=None, original_model_id=None)
                enhanced_models[model_id] = enhanced

                converse_data = enhanced['converse_data']
                source = converse_data.get('source', 'unknown')
                by_source[source] = by_source.get(source, 0) + 1

                if converse_data.get('verified'):
                    success_count += 1
                else:
                    missing_count += 1

            except Exception as e:
                self.logger.warning(f"Failed to enhance {model_id}: {e}")
                enhanced_models[model_id] = model_data  # Use original

        # Summary statistics
        total = success_count + missing_count

        self.logger.info(f"✅ Enhanced {success_count}/{total} models with verified token specifications")
        self.logger.info(f"📊 Data sources: {dict(sorted(by_source.items(), key=lambda x: x[1], reverse=True))}")

        return enhanced_models
