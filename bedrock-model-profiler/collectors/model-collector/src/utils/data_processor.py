"""
Clean Model Data Processor
Simple, clean implementation without complex quota assignment logic
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
from collections import defaultdict


class ModelDataProcessor:
    """Clean model data processor without quota assignment complexity"""

    def __init__(self, quotas_collector=None):
        self.logger = logging.getLogger(__name__)
        self.quotas_collector = quotas_collector

    def create_comprehensive_structure(self, raw_models: Dict[str, Any], enhanced_models: Dict[str, Any],
                                     pricing_data: Dict[str, Any], quotas_data: Dict[str, Any],
                                     regions: List[str], regional_availability: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create comprehensive structured JSON output (clean version)"""

        self.logger.info("Creating clean comprehensive data structure...")

        # STEP 1: Generate and display models grouped by providers
        models_by_provider = self._display_models_by_provider(enhanced_models)

        # STEP 2: Analyze model name conflicts and create priority groups
        no_conflicts_group, with_conflicts_group = self._analyze_model_conflicts(models_by_provider)

        # STEP 3: Assign quotas to models (Group 1 first, then Group 2)
        group1_assigned, group1_unassigned, remaining_quotas = self.assign_quotas_to_models(enhanced_models, quotas_data, no_conflicts_group, with_conflicts_group)
        self.logger.info(f"📋 Group 1 Assignment: {len(group1_assigned)} assigned, {len(group1_unassigned)} unassigned")

        # STEP 3b: Assign quotas to Group 2 models (conflicts) using remaining quotas
        group2_assigned, group2_unassigned, final_remaining_quotas = self.assign_group2_quotas(enhanced_models, remaining_quotas, with_conflicts_group)
        self.logger.info(f"📋 Group 2 Assignment: {len(group2_assigned)} assigned, {len(group2_unassigned)} unassigned")

        # STEP 3c: Special Mistral version-based assignment
        mistral_assigned, mistral_unassigned, final_remaining_quotas = self.assign_mistral_version_quotas(enhanced_models, final_remaining_quotas)
        self.logger.info(f"📋 Mistral Version Assignment: {len(mistral_assigned)} assigned, {len(mistral_unassigned)} unassigned")

        # Combine results from all groups
        assigned_models = {**group1_assigned, **group2_assigned, **mistral_assigned}
        unassigned_models = group1_unassigned + group2_unassigned + mistral_unassigned

        self.logger.info(f"📊 TOTAL ASSIGNMENT: {len(assigned_models)} assigned, {len(unassigned_models)} unassigned")


        # Organize models by provider
        providers = defaultdict(lambda: {"models": {}})

        # Process each model and create the final structure
        for model_id, model_data in enhanced_models.items():
            provider = model_data.get('model_provider', 'Unknown')

            # Clean model_id by removing sub-versions (everything after first ':')
            clean_model_id = self._clean_model_id(model_id)

            # Update the model_data with cleaned model_id while preserving all other data
            updated_model_data = model_data.copy()
            updated_model_data['model_id'] = clean_model_id

            # Use cleaned model_id as key
            model_key = clean_model_id

            # Merge model data (with assigned quotas from Step 3)
            # Check both original model_id and cleaned model_id for quota assignment (robust approach)
            if model_id in assigned_models:
                # Use assigned quotas from Step 3 (original model_id)
                model_quotas = assigned_models[model_id]['quotas_by_region']
                comprehensive_model = self._merge_model_data_with_quotas(
                    updated_model_data, pricing_data.get(model_id, {}), model_quotas
                )
            elif clean_model_id in assigned_models:
                # Use assigned quotas from Step 3 (cleaned model_id)
                model_quotas = assigned_models[clean_model_id]['quotas_by_region']
                comprehensive_model = self._merge_model_data_with_quotas(
                    updated_model_data, pricing_data.get(model_id, {}), model_quotas
                )
            else:
                # No quotas assigned - use clean version
                comprehensive_model = self._merge_model_data_clean(
                    updated_model_data, pricing_data.get(model_id, {})
                )

            # FIXED: Preserve models with corrected token specifications during deduplication
            # Check if this model key already exists (potential deduplication conflict)
            if model_key in providers[provider]["models"]:
                existing_model = providers[provider]["models"][model_key]

                # Preserve the model with better token specifications (corrected data priority)
                should_replace = self._should_replace_existing_model(existing_model, comprehensive_model)

                if should_replace:
                    providers[provider]["models"][model_key] = comprehensive_model
                    self.logger.debug(f"🔄 Replaced {model_key} with better token specifications from {model_id}")
                else:
                    self.logger.debug(f"📝 Kept existing {model_key}, discarded duplicate from {model_id}")
            else:
                # No conflict - add new model
                providers[provider]["models"][model_key] = comprehensive_model

        # Create metadata
        metadata = self._create_metadata(providers, regions, raw_models, pricing_data, quotas_data)

        self.logger.info(f"✅ Clean structure created with {sum(len(p['models']) for p in providers.values())} models")
        self.logger.info(f"📊 Ready for quota assignment - quotas_data contains {len(quotas_data)} regions")

        return {
            "metadata": metadata,
            "providers": dict(providers)
        }

    def _merge_model_data_with_quotas(self, model_data: Dict[str, Any], pricing_info: Dict[str, Any], assigned_quotas: Dict[str, Any]) -> Dict[str, Any]:
        """Merge model data with pricing info and assigned quotas"""
        # Start with enhanced model data
        merged = model_data.copy()

        # Add pricing information if available
        if pricing_info:
            # Extract the inner model_pricing data to avoid nested structure
            inner_model_pricing = pricing_info.get('model_pricing', {})
            merged.update({
                "model_pricing": inner_model_pricing,
                "has_pricing": inner_model_pricing.get('is_pricing_available', False)
            })
        else:
            merged.update({
                "model_pricing": {},
                "has_pricing": False
            })

        # Add assigned quotas (already organized by region from Step 3)
        if assigned_quotas:
            # Count total quotas across all regions
            total_quotas = sum(len(quotas) for quotas in assigned_quotas.values())

            merged.update({
                "model_service_quotas": assigned_quotas,  # Already in quotas_by_region format
                "has_quotas": True,
                "total_quotas_assigned": total_quotas
            })
        else:
            merged.update({
                "model_service_quotas": {},
                "has_quotas": False,
                "total_quotas_assigned": 0
            })

        return merged

    def _merge_model_data_clean(self, model_data: Dict[str, Any], pricing_info: Dict[str, Any]) -> Dict[str, Any]:
        """Merge model data with pricing info (no quotas yet)"""
        # Start with enhanced model data
        merged = model_data.copy()

        # Add pricing information if available
        if pricing_info:
            merged.update({
                "model_pricing": pricing_info,
                "has_pricing": True
            })
        else:
            merged.update({
                "model_pricing": {},
                "has_pricing": False
            })

        # Placeholder for quotas (empty for now - ready for your implementation)
        merged.update({
            "model_service_quotas": {},
            "has_quotas": False
        })

        return merged

    def _create_metadata(self, providers: Dict, regions: List[str], raw_models: Dict[str, Any],
                        pricing_data: Dict[str, Any], quotas_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for the final structure"""
        total_models = sum(len(provider_data["models"]) for provider_data in providers.values())
        models_with_pricing = sum(
            1 for provider_data in providers.values()
            for model_data in provider_data["models"].values()
            if model_data.get("has_pricing", False)
        )

        # Count models with quotas assigned
        models_with_quotas = sum(
            1 for provider_data in providers.values()
            for model_data in provider_data["models"].values()
            if model_data.get("has_quotas", False)
        )

        # Count total quotas available (for reference)
        total_quotas = 0
        for region_data in quotas_data.values():
            if isinstance(region_data, dict) and 'quotas' in region_data:
                total_quotas += len(region_data['quotas'])

        return {
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "providers_count": len(providers),
            "total_models": total_models,
            "models_with_pricing": models_with_pricing,
            "models_with_quotas": models_with_quotas,
            "regions_covered": len(regions),
            "total_quotas_available": total_quotas,
            "collection_method": "comprehensive_structure_with_quota_assignment"
        }

    # Helper method to get available quotas data for external quota assignment
    def get_quotas_data(self, quotas_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get clean quotas data for external processing"""
        clean_quotas = {}

        for region, region_data in quotas_data.items():
            if isinstance(region_data, dict) and 'quotas' in region_data:
                clean_quotas[region] = region_data['quotas']

        self.logger.info(f"📋 Quotas available: {sum(len(quotas) for quotas in clean_quotas.values())} across {len(clean_quotas)} regions")

        return clean_quotas

    # Helper method to get models grouped by provider
    def get_models_by_provider(self, enhanced_models: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Get models organized by provider for external quota assignment"""
        models_by_provider = defaultdict(list)

        for model_id, model_data in enhanced_models.items():
            provider = model_data.get('model_provider', 'Unknown')
            models_by_provider[provider].append({
                'model_id': model_id,
                'model_name': model_data.get('model_name', ''),
                'model_data': model_data
            })

        self.logger.info("📊 Models by provider:")
        for provider, models in models_by_provider.items():
            self.logger.info(f"   {provider}: {len(models)} models")

        return dict(models_by_provider)

    def _preprocess_model_name(self, model_name: str, provider: str) -> str:
        """Apply provider-specific model name preprocessing"""
        import re

        # Anthropic provider: Rename generic 'Claude' to 'Claude v2'
        if provider == 'Anthropic':
            if model_name.strip().lower() == 'claude':
                model_name = 'Claude v2'

        # Meta provider: Remove 'instruct'
        elif provider == 'Meta':
            model_name = re.sub(r'\s*instruct\s*', '', model_name, flags=re.IGNORECASE).strip()
            model_name = re.sub(r'\s*17B\s*', '', model_name, flags=re.IGNORECASE).strip()

        # Mistral provider: Remove 'instruct' + handle bracketed numbers like (24.02)
        elif provider == 'Mistral AI':
            # Remove 'instruct'
            model_name = re.sub(r'\s*instruct\s*', '', model_name, flags=re.IGNORECASE).strip()

            # Handle bracketed numbers like (24.02)
            # PRESERVE version patterns for version-based quota assignment in Step 3c
            # Only remove non-version patterns (e.g., remove "(Instruct)" but keep "(24.07)")
            version_pattern = r'\((\d+\.\d+|\d{4}\.\d+)\)'  # Match (24.07) or (2024.07) patterns
            if not re.search(version_pattern, model_name):
                # No version pattern - safe to remove all parentheses content
                model_name = re.sub(r'\s*\([^)]*\)\s*', '', model_name).strip()
            # If version pattern exists, preserve it for Step 3c processing

        # OpenAI provider: Replace '-' with ' '
        elif provider == 'OpenAI':
            model_name = model_name.replace('-', ' ').strip()

        # DeepSeek provider: Replace '-' with ' ' and remove '.1' from versions like 'v3.1'
        elif provider == 'DeepSeek':
            model_name = model_name.replace('-', ' ').strip()
            # Remove '.1' from version patterns like 'v3.1' -> 'v3'
            model_name = re.sub(r'(v\d+)\.1\b', r'\1', model_name, flags=re.IGNORECASE).strip()

        # Qwen provider: Remove brackets, replace '-' with ' ', and remove 'instruct'
        elif provider == 'Qwen':
            # Remove brackets and text inside them
            model_name = re.sub(r'\s*\([^)]*\)\s*', '', model_name).strip()
            # Replace '-' with ' '
            model_name = model_name.replace('-', ' ').strip()
            # Remove 'instruct' (case-insensitive)
            model_name = re.sub(r'\s*instruct\s*', '', model_name, flags=re.IGNORECASE).strip()

        # Cohere provider: Replace '+' with ' plus'
        elif provider == 'Cohere':
            model_name = model_name.replace('+', ' plus').strip()

        # Stability AI provider: Remove '1.0' from the end (could save separately if needed)
        elif provider == 'Stability AI':
            model_name = re.sub(r'\s+1\.0\s*$', '', model_name).strip()

        # TwelveLabs provider: Remove version patterns and 'embed' from Marengo
        elif provider == 'TwelveLabs':
            model_name = re.sub(r'\s+v\d+(\.\d+)?\s*$', '', model_name, flags=re.IGNORECASE).strip()
            # Remove 'embed' from model names (e.g., "Marengo Embed" -> "Marengo")
            model_name = re.sub(r'\s*embed\s*', '', model_name, flags=re.IGNORECASE).strip()

        return model_name

    def _generate_name_variations(self, model_name: str) -> List[str]:
        """Generate name variations to handle company name spacing differences"""
        variations = [model_name.lower()]  # Always include original

        # Handle specific company name variations
        name_lower = model_name.lower()

        # TwelveLabs <-> Twelve Labs variations
        if 'twelvelabs' in name_lower:
            spaced_version = name_lower.replace('twelvelabs', 'twelve labs')
            variations.append(spaced_version)
        elif 'twelve labs' in name_lower:
            unspaced_version = name_lower.replace('twelve labs', 'twelvelabs')
            variations.append(unspaced_version)

        # Add more company variations as needed
        # TODO: Add other company name variations here if discovered

        return list(set(variations))  # Remove duplicates

    def _clean_model_id(self, model_id: str) -> str:
        """Remove sub-versions from model ID (everything after first ':')"""
        # Split by ':' and take only the first part
        return model_id.split(':', 1)[0] if ':' in model_id else model_id

    def _display_models_by_provider(self, enhanced_models: Dict[str, Any]) -> Dict[str, List[str]]:
        """STEP 1: Generate simple list of models grouped by providers"""
        import json

        models_by_provider = {}

        # Group models by provider - just model names (deduplicated)
        for model_id, model_data in enhanced_models.items():
            provider = model_data.get('model_provider', 'Unknown')
            model_name = model_data.get('model_name', model_id)

            # Apply provider-specific model name preprocessing
            processed_name = self._preprocess_model_name(model_name, provider)

            if provider not in models_by_provider:
                models_by_provider[provider] = []

            # Add model name if not already in the list (deduplicate, case-insensitive)
            if processed_name.lower() not in [m.lower() for m in models_by_provider[provider]]:
                models_by_provider[provider].append(processed_name)

        return models_by_provider

    def _analyze_model_conflicts(self, models_by_provider: Dict[str, List[str]]) -> None:
        """STEP 2: Analyze model name conflicts and create priority groups"""
        # Groups for final output - organize by individual models, not providers
        no_conflicts_group = {}  # Models that can use direct matching
        with_conflicts_group = {}  # Models that have conflicts

        for provider, models in models_by_provider.items():
            # Track which models in this provider have conflicts and their priorities
            models_with_conflicts = set()
            model_priorities = {}  # Track priority for each conflicting model
            conflicts_found = []

            # Check each model against all others in the same provider
            for i, model_a in enumerate(models):
                for j, model_b in enumerate(models):
                    if i >= j:  # Skip same model and avoid duplicate pairs
                        continue

                    # Check containment (case-insensitive)
                    model_a_lower = model_a.lower()
                    model_b_lower = model_b.lower()

                    a_in_b = model_a_lower in model_b_lower
                    b_in_a = model_b_lower in model_a_lower

                    # If one is contained in the other (but not both ways)
                    if a_in_b and not b_in_a:
                        models_with_conflicts.add(model_a)
                        models_with_conflicts.add(model_b)
                        model_priorities[model_a] = 1  # contained = lower priority
                        model_priorities[model_b] = 2  # contains = higher priority
                        conflicts_found.append({
                            'lower_priority': model_a,
                            'higher_priority': model_b,
                            'reason': f"'{model_a}' is contained in '{model_b}'"
                        })

                    elif b_in_a and not a_in_b:
                        models_with_conflicts.add(model_a)
                        models_with_conflicts.add(model_b)
                        model_priorities[model_b] = 1  # contained = lower priority
                        model_priorities[model_a] = 2  # contains = higher priority
                        conflicts_found.append({
                            'lower_priority': model_b,
                            'higher_priority': model_a,
                            'reason': f"'{model_b}' is contained in '{model_a}'"
                        })

            # Separate models into conflict and no-conflict groups
            models_no_conflicts = [model for model in models if model not in models_with_conflicts]

            # Add to groups
            if models_no_conflicts:
                if provider not in no_conflicts_group:
                    no_conflicts_group[provider] = []
                no_conflicts_group[provider].extend(sorted(models_no_conflicts))

            if models_with_conflicts:
                if provider not in with_conflicts_group:
                    with_conflicts_group[provider] = []

                # Create sorted list of models with their priorities
                models_with_priority = []
                for model in sorted(models_with_conflicts):
                    models_with_priority.append({
                        'model_name': model,
                        'priority': model_priorities.get(model, 0)
                    })

                with_conflicts_group[provider] = models_with_priority

        # Return both groups for use in quota assignment
        return no_conflicts_group, with_conflicts_group

    def assign_quotas_to_models(self, enhanced_models: Dict[str, Any], quotas_data: Dict[str, Any], no_conflicts_group: Dict[str, Any], with_conflicts_group: Dict[str, Any]) -> tuple:
        """STEP 3: Assign quotas to models (start with Group 1 - no conflicts, then Group 2 - conflicts)"""
        # Convert no_conflicts_group to model_id -> model_data mapping
        group1_models = {}
        for provider, model_names in no_conflicts_group.items():
            for model_name in model_names:
                # Find the corresponding model_id in enhanced_models
                for model_id, model_data in enhanced_models.items():
                    if model_data.get('model_provider') == provider:
                        processed_name = self._preprocess_model_name(model_data.get('model_name', ''), provider)
                        if processed_name == model_name:
                            group1_models[model_id] = model_data
                            break

        # Process only Group 1 models for quota assignment
        # Create working copy of quotas data
        remaining_quotas = {}
        for region, region_data in quotas_data.items():
            if isinstance(region_data, dict) and 'quotas' in region_data:
                remaining_quotas[region] = region_data['quotas'].copy()

        # Track assignments
        assigned_models = {}
        unassigned_models = []

        # Process each model in Group 1 (simple models without conflicts)
        for model_id, model_data in group1_models.items():
            provider = model_data.get('model_provider', 'Unknown')
            model_name = model_data.get('model_name', '')
            processed_name = self._preprocess_model_name(model_name, provider)

            # Use cleaned model_id consistently (same as final save phase)
            clean_model_id = self._clean_model_id(model_id)

            # Generate name variations for better matching (handles company name spacing)
            name_variations = self._generate_name_variations(processed_name)

            # Initialize model quotas structure
            model_quotas_by_region = {}

            # Search for this model name in all regions' quotas
            for region, quotas_dict in remaining_quotas.items():
                # Direct text search for model name in quota names AND descriptions (iterate over copy to allow deletion)
                for quota_key in list(quotas_dict.keys()):
                    quota = quotas_dict[quota_key]
                    quota_name = quota.get('quota_name', '').lower()
                    quota_description = quota.get('description', '').lower()

                    # Does this quota name OR description contain any of our model name variations?
                    if any(variation in quota_name or variation in quota_description for variation in name_variations):
                        # Directly append the quota to this model's region
                        if region not in model_quotas_by_region:
                            model_quotas_by_region[region] = []
                        model_quotas_by_region[region].append(quota)
                        # Remove assigned quota from the pool immediately
                        del remaining_quotas[region][quota_key]

            # Track assignment results - check if model has any quotas assigned
            total_quotas = sum(len(rq) for rq in model_quotas_by_region.values())
            if total_quotas > 0:
                # Use cleaned model_id as key to match final save phase
                assigned_models[clean_model_id] = {
                    'model_name': processed_name,
                    'provider': provider,
                    'quotas_by_region': model_quotas_by_region,
                    'total_quotas': total_quotas
                }
            else:
                unassigned_models.append({
                    'model_id': clean_model_id,  # Use cleaned model_id for consistency
                    'model_name': processed_name,
                    'provider': provider
                })

        return assigned_models, unassigned_models, remaining_quotas

    def assign_group2_quotas(self, enhanced_models: Dict[str, Any], remaining_quotas: Dict[str, Any], with_conflicts_group: Dict[str, Any]) -> tuple:
        """STEP 3b: Assign quotas to Group 2 models (conflicts) using priority-based assignment"""
        # Convert with_conflicts_group to model_id -> model_data mapping with priorities
        group2_models = {}
        for provider, model_list in with_conflicts_group.items():
            for model_info in model_list:
                model_name = model_info['model_name']
                priority = model_info['priority']

                # Find the corresponding model_id in enhanced_models
                for model_id, model_data in enhanced_models.items():
                    if model_data.get('model_provider') == provider:
                        processed_name = self._preprocess_model_name(model_data.get('model_name', ''), provider)
                        if processed_name == model_name:
                            group2_models[model_id] = {
                                'model_data': model_data,
                                'priority': priority,
                                'processed_name': processed_name
                            }
                            break

        # Track assignments for Group 2
        assigned_models = {}
        unassigned_models = []

        # Process by priority: Priority 2 first (higher priority), then Priority 1
        for current_priority in [2, 1]:
            priority_models = {k: v for k, v in group2_models.items() if v['priority'] == current_priority}

            for model_id, model_info in priority_models.items():
                model_data = model_info['model_data']
                processed_name = model_info['processed_name']
                provider = model_data.get('model_provider', 'Unknown')

                # Use cleaned model_id consistently
                clean_model_id = self._clean_model_id(model_id)

                # Generate name variations for better matching (handles company name spacing)
                name_variations = self._generate_name_variations(processed_name)

                # Initialize model quotas structure
                model_quotas_by_region = {}

                # Search for this model name in all regions' quotas (same logic as Group 1)
                for region, quotas_dict in remaining_quotas.items():
                    # Direct text search for model name in quota names AND descriptions
                    for quota_key in list(quotas_dict.keys()):
                        quota = quotas_dict[quota_key]
                        quota_name = quota.get('quota_name', '').lower()
                        quota_description = quota.get('description', '').lower()

                        # Simple direct matching - model name variations in quota name OR description
                        if any(variation in quota_name or variation in quota_description for variation in name_variations):
                            # Directly append the quota to this model's region
                            if region not in model_quotas_by_region:
                                model_quotas_by_region[region] = []
                            model_quotas_by_region[region].append(quota)
                            # Remove assigned quota from the pool immediately
                            del remaining_quotas[region][quota_key]

                # Track assignment results - check if model has any quotas assigned
                total_quotas = sum(len(rq) for rq in model_quotas_by_region.values())
                if total_quotas > 0:
                    # Use cleaned model_id as key to match final save phase
                    assigned_models[clean_model_id] = {
                        'model_name': processed_name,
                        'provider': provider,
                        'priority': current_priority,
                        'quotas_by_region': model_quotas_by_region,
                        'total_quotas': total_quotas
                    }
                else:
                    unassigned_models.append({
                        'model_id': clean_model_id,
                        'model_name': processed_name,
                        'provider': provider,
                        'priority': current_priority
                    })

        return assigned_models, unassigned_models, remaining_quotas

    def assign_mistral_version_quotas(self, enhanced_models: Dict[str, Any], remaining_quotas: Dict[str, Any]) -> tuple:
        """Special assignment for Mistral models with version patterns like 'Mistral Large (24.07)'"""
        import re

        # Find Mistral models with version patterns
        mistral_version_models = {}

        for model_id, model_data in enhanced_models.items():
            provider = model_data.get('model_provider', 'Unknown')
            if provider == 'Mistral AI':
                model_name = model_data.get('model_name', '')
                processed_name = self._preprocess_model_name(model_name, provider)

                # Check if original name has version pattern with parentheses
                if '(' in model_name and ')' in model_name:
                    # Extract base name and version using regex
                    # Pattern: "Mistral Large (24.07)" -> base="Mistral Large", version="24.07"
                    version_pattern = r'^(.+?)\s*\(([^)]+)\)\s*$'
                    match = re.match(version_pattern, processed_name)

                    if match:
                        base_name = match.group(1).strip()
                        version = match.group(2).strip()

                        mistral_version_models[model_id] = {
                            'model_data': model_data,
                            'processed_name': processed_name,
                            'base_name': base_name,
                            'version': version,
                            'original_name': model_name
                        }

        if not mistral_version_models:
            self.logger.info("No Mistral version models found - skipping")
            return {}, [], remaining_quotas

        # Sort by version (higher versions first for priority)
        def version_sort_key(item):
            model_id, model_info = item
            version_str = model_info['version']
            try:
                # Convert version like "24.07" to float for sorting
                return float(version_str)
            except ValueError:
                # If can't convert to float, use string comparison
                return version_str

        sorted_models = sorted(mistral_version_models.items(), key=version_sort_key, reverse=True)

        # Track assignments
        assigned_models = {}
        unassigned_models = []
        lowest_priority_models_by_base = {}  # Track lowest priority model per base name

        # Track unassigned base-name matches for fallback (per base name)
        remaining_base_matches_by_base = {}  # base_name -> {region -> [(quota_key, quota, quota_name)]}

        # Process each Mistral version model using two-step search
        for idx, (model_id, model_info) in enumerate(sorted_models):
            model_data = model_info['model_data']
            processed_name = model_info['processed_name']
            base_name = model_info['base_name']
            version = model_info['version']
            provider = model_data.get('model_provider', 'Unknown')

            # Use cleaned model_id consistently
            clean_model_id = self._clean_model_id(model_id)

            # Track lowest priority model per base name (latest processed = lowest priority for that base)
            lowest_priority_models_by_base[base_name] = {
                'model_id': model_id,
                'clean_model_id': clean_model_id,
                'model_info': model_info,
                'provider': provider
            }

            # Generate name variations for base name (handles company name spacing)
            base_name_variations = self._generate_name_variations(base_name)

            # Initialize model quotas structure
            model_quotas_by_region = {}

            # Two-step search process
            for region, quotas_dict in remaining_quotas.items():
                step1_matches = []  # Quotas containing base name

                # Step 1: Find quotas containing base name (any variation)
                for quota_key, quota in quotas_dict.items():
                    quota_name = quota.get('quota_name', '').lower()
                    quota_description = quota.get('description', '').lower()

                    if any(variation in quota_name or variation in quota_description for variation in base_name_variations):
                        step1_matches.append((quota_key, quota, quota_name, quota_description))

                if step1_matches:
                    # Step 2: Within step1 matches, search for version
                    version_found = False
                    unassigned_base_matches = []

                    for quota_key, quota, quota_name, quota_description in step1_matches:
                        if version.lower() in quota_name or version.lower() in quota_description:
                            # Version found in quota name OR description - assign this quota
                            if region not in model_quotas_by_region:
                                model_quotas_by_region[region] = []
                            model_quotas_by_region[region].append(quota)
                            # Remove assigned quota from the pool immediately
                            del remaining_quotas[region][quota_key]

                            version_found = True
                            break  # Only assign first matching quota per region
                        else:
                            # This quota matches base name but not version - collect for potential fallback
                            unassigned_base_matches.append((quota_key, quota, quota_name))

                    if not version_found:
                        # Store unassigned base matches per base name for potential fallback
                        if unassigned_base_matches:
                            if base_name not in remaining_base_matches_by_base:
                                remaining_base_matches_by_base[base_name] = {}
                            if region not in remaining_base_matches_by_base[base_name]:
                                remaining_base_matches_by_base[base_name][region] = []
                            remaining_base_matches_by_base[base_name][region].extend(unassigned_base_matches)

            # Track assignment results
            total_quotas = sum(len(rq) for rq in model_quotas_by_region.values())
            if total_quotas > 0:
                assigned_models[clean_model_id] = {
                    'model_name': processed_name,
                    'provider': provider,
                    'base_name': base_name,
                    'version': version,
                    'quotas_by_region': model_quotas_by_region,
                    'total_quotas': total_quotas
                }
            else:
                unassigned_models.append({
                    'model_id': clean_model_id,
                    'model_name': processed_name,
                    'provider': provider,
                    'base_name': base_name,
                    'version': version
                })

        # FALLBACK: Assign remaining base matches to lowest priority model per base name
        if remaining_base_matches_by_base:
            for base_name, base_matches_by_region in remaining_base_matches_by_base.items():
                # Get lowest priority model for this base name
                if base_name in lowest_priority_models_by_base:
                    lowest_model = lowest_priority_models_by_base[base_name]
                    lowest_model_info = lowest_model['model_info']
                    lowest_clean_id = lowest_model['clean_model_id']
                    lowest_provider = lowest_model['provider']

                    fallback_quotas_by_region = {}
                    fallback_total_quotas = 0


                    for region, base_matches in base_matches_by_region.items():
                        for quota_key, quota, quota_name in base_matches:
                            # Assign this quota to the lowest priority model for this base name
                            if region not in fallback_quotas_by_region:
                                fallback_quotas_by_region[region] = []
                            fallback_quotas_by_region[region].append(quota)

                            # Remove from remaining quotas pool
                            if quota_key in remaining_quotas[region]:
                                del remaining_quotas[region][quota_key]

                            fallback_total_quotas += 1

                    # Update or create assignment for lowest priority model of this base name
                    if lowest_clean_id in assigned_models:
                        # Add to existing assignment
                        existing_quotas = assigned_models[lowest_clean_id]['quotas_by_region']
                        for region, quotas in fallback_quotas_by_region.items():
                            if region in existing_quotas:
                                existing_quotas[region].extend(quotas)
                            else:
                                existing_quotas[region] = quotas

                        assigned_models[lowest_clean_id]['total_quotas'] += fallback_total_quotas
                    else:
                        # Create new assignment for previously unassigned lowest priority model
                        assigned_models[lowest_clean_id] = {
                            'model_name': lowest_model_info['processed_name'],
                            'provider': lowest_provider,
                            'base_name': lowest_model_info['base_name'],
                            'version': lowest_model_info['version'],
                            'quotas_by_region': fallback_quotas_by_region,
                            'total_quotas': fallback_total_quotas
                        }

                        # Remove from unassigned list
                        unassigned_models = [m for m in unassigned_models if m['model_id'] != lowest_clean_id]
                else:
                    self.logger.warning(f"   ⚠️  No lowest priority model found for base name: {base_name}")

        return assigned_models, unassigned_models, remaining_quotas

    def _should_replace_existing_model(self, existing_model: Dict[str, Any], new_model: Dict[str, Any]) -> bool:
        """Determine if new model should replace existing model during deduplication

        Priority order:
        1. Models with corrected token specifications (source contains 'corrected')
        2. Models with verified token specifications (both context_window and max_output_tokens present)
        3. Models with partial specifications (either context_window or max_output_tokens present)
        4. Models with more complete data overall

        Returns:
            bool: True if new_model should replace existing_model, False otherwise
        """

        # Extract converse_data for comparison
        existing_converse = existing_model.get('converse_data', {})
        new_converse = new_model.get('converse_data', {})

        existing_source = existing_converse.get('source', '')
        new_source = new_converse.get('source', '')

        # Priority 1: Corrected data always wins (enhanced logic)
        existing_has_correction = 'corrected' in existing_source.lower()
        new_has_correction = 'corrected' in new_source.lower()

        # ENHANCED: Always prefer models with pure "corrected" source over mixed sources
        existing_is_pure_corrected = existing_source == 'corrected'
        new_is_pure_corrected = new_source == 'corrected'

        if new_is_pure_corrected and not existing_is_pure_corrected:
            return True  # New model has pure corrections, existing doesn't
        elif existing_is_pure_corrected and not new_is_pure_corrected:
            return False  # Existing model has pure corrections, new doesn't
        elif new_has_correction and not existing_has_correction:
            return True  # New model has some corrections, existing has none
        elif existing_has_correction and not new_has_correction:
            return False  # Existing model has some corrections, new has none
        elif new_has_correction and existing_has_correction:
            # Both have corrections - prefer pure corrected source
            if new_is_pure_corrected and not existing_is_pure_corrected:
                return True
            else:
                # Either existing is pure corrected (and new isn't), or both same correction level - keep existing
                return False

        # Priority 2: Verified complete specifications
        existing_context = existing_converse.get('context_window')
        existing_max_output = existing_converse.get('max_output_tokens')
        new_context = new_converse.get('context_window')
        new_max_output = new_converse.get('max_output_tokens')

        # Check if specifications are complete and valid (not "N/A" and not None)
        def is_valid_spec(value):
            return value is not None and value != "N/A" and value != 0

        existing_complete = is_valid_spec(existing_context) and is_valid_spec(existing_max_output)
        new_complete = is_valid_spec(new_context) and is_valid_spec(new_max_output)

        if new_complete and not existing_complete:
            return True  # New model has complete specs, existing doesn't
        elif existing_complete and not new_complete:
            return False  # Existing model has complete specs, new doesn't

        # Priority 3: Partial specifications
        existing_partial = is_valid_spec(existing_context) or is_valid_spec(existing_max_output)
        new_partial = is_valid_spec(new_context) or is_valid_spec(new_max_output)

        if new_partial and not existing_partial:
            return True  # New model has some specs, existing has none
        elif existing_partial and not new_partial:
            return False  # Existing model has some specs, new has none

        # Priority 4: Default to keeping existing model (first processed wins)
        return False
