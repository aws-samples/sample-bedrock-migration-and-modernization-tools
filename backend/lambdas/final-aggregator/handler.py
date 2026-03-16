"""
Final Aggregator Lambda

Merges all collected data into the final comprehensive JSON outputs.
Works with the correct snake_case schema from upstream Lambdas.
"""

import re
import time
from typing import Any, Optional

from shared import (
    get_s3_client,
    read_from_s3,
    write_to_s3,
    parse_execution_id,
    validate_required_params,
    ValidationError,
    S3ReadError,
    get_config_loader,
)
from shared.model_matcher import (
    get_provider_from_model_id,
    get_canonical_model_id,
    calculate_match_score,
    get_model_variant_info,
    has_semantic_conflict,
)
from shared.powertools import logger, tracer, metrics, LambdaContext
from aws_lambda_powertools.metrics import MetricUnit


@tracer.capture_method
def aggregate_quotas(quota_results: list[dict], s3_client: Any, bucket: str) -> dict:
    """Aggregate quotas from all regions with tracing."""
    logger.info("Aggregating quotas", extra={"result_count": len(quota_results)})
    quotas_by_region = {}

    for item in quota_results:
        nested_result = item.get("result", {})
        status = item.get("status") or nested_result.get("status")
        s3_key = (
            item.get("s3_key")
            or item.get("s3Key")
            or nested_result.get("s3_key")
            or nested_result.get("s3Key")
        )
        region = item.get("region")

        if status == "SUCCESS" and s3_key:
            try:
                data = read_from_s3(s3_client, bucket, s3_key, default_on_missing={})
                quotas_by_region[region] = data.get("quotas", [])
            except S3ReadError as e:
                logger.warning(
                    "Failed to read quotas", extra={"region": region, "error": str(e)}
                )
                quotas_by_region[region] = []

    return quotas_by_region


@tracer.capture_method
def aggregate_features(
    feature_results: list[dict], s3_client: Any, bucket: str
) -> dict:
    """Aggregate inference profiles from all regions with tracing."""
    logger.info("Aggregating features", extra={"result_count": len(feature_results)})
    profiles_by_region = {}

    for item in feature_results:
        nested_result = item.get("result", {})
        status = item.get("status") or nested_result.get("status")
        s3_key = (
            item.get("s3_key")
            or item.get("s3Key")
            or nested_result.get("s3_key")
            or nested_result.get("s3Key")
        )
        region = item.get("region")

        if status == "SUCCESS" and s3_key:
            try:
                data = read_from_s3(s3_client, bucket, s3_key, default_on_missing={})
                # Handle both snake_case and camelCase from feature extractor
                profiles_by_region[region] = data.get(
                    "inference_profiles", data.get("inferenceProfiles", [])
                )
            except S3ReadError as e:
                logger.warning(
                    "Failed to read features", extra={"region": region, "error": str(e)}
                )
                profiles_by_region[region] = []

    return profiles_by_region


@tracer.capture_method
def aggregate_mantle(mantle_results: list[dict], s3_client: Any, bucket: str) -> dict:
    """Aggregate Mantle models from all regions with tracing.

    Returns: { model_id: {"regions": [region1, ...], "supports_responses_api": bool} }
    """
    logger.info("Aggregating Mantle data", extra={"result_count": len(mantle_results)})
    mantle_by_model = {}

    for item in mantle_results:
        nested_result = item.get("result", {})
        status = item.get("status") or nested_result.get("status")
        s3_key = (
            item.get("s3_key")
            or item.get("s3Key")
            or nested_result.get("s3_key")
            or nested_result.get("s3Key")
        )
        region = item.get("region")

        if status == "SUCCESS" and s3_key:
            try:
                data = read_from_s3(s3_client, bucket, s3_key, default_on_missing={})
                for model in data.get("mantle_models", []):
                    model_id = model.get("model_id", "")
                    if model_id:
                        if model_id not in mantle_by_model:
                            mantle_by_model[model_id] = {
                                "regions": set(),
                                "supports_responses_api": False,
                            }
                        mantle_by_model[model_id]["regions"].add(region)
                        # If ANY region reports Responses API support, mark it
                        if model.get("supports_responses_api", False):
                            mantle_by_model[model_id]["supports_responses_api"] = True
            except S3ReadError as e:
                logger.warning(
                    "Failed to read mantle data",
                    extra={"region": region, "error": str(e)},
                )

    # Convert sets to sorted lists
    return {
        mid: {
            "regions": sorted(list(info["regions"])),
            "supports_responses_api": info["supports_responses_api"],
        }
        for mid, info in mantle_by_model.items()
    }


def get_size_category(context_window: int) -> dict:
    """Get size category based on context window."""
    if context_window is None:
        return {"category": "Unknown", "color": "#6B7280", "tier": 0}
    elif context_window >= 128000:
        return {"category": "Large", "color": "#10B981", "tier": 3}
    elif context_window >= 32000:
        return {"category": "Medium", "color": "#3B82F6", "tier": 2}
    else:
        return {"category": "Small", "color": "#F59E0B", "tier": 1}


def get_context_window_from_config(model_id: str) -> dict:
    """
    Get context window specs from config for a model.

    Uses pattern matching to find the best match in context_window_specs.
    Returns dict with context window data or empty dict if not found.
    """
    config = get_config_loader()
    context_specs = config.config.get("model_configuration", {}).get(
        "context_window_specs", {}
    )

    # Remove version suffix for matching (e.g., anthropic.claude-opus-4-6-v1:0 -> anthropic.claude-opus-4-6)
    model_id_clean = model_id.lower()
    # Remove common suffixes
    for suffix in ["-v1:0", "-v1", "-v2:0", "-v2", ":0", ":1"]:
        if model_id_clean.endswith(suffix):
            model_id_clean = model_id_clean[: -len(suffix)]

    # Try exact match first
    if model_id_clean in context_specs:
        return context_specs[model_id_clean]

    # Try prefix matching (longest match wins)
    best_match = None
    best_match_len = 0
    for pattern, specs in context_specs.items():
        if pattern.startswith("_"):  # Skip comment keys
            continue
        if model_id_clean.startswith(pattern) and len(pattern) > best_match_len:
            best_match = specs
            best_match_len = len(pattern)

    return best_match or {}


def build_cross_region_inference(model_id: str, features_by_region: dict) -> dict:
    """Build cross-region inference data for a model.

    Deduplicates profiles by (profile_id, source_region) to avoid duplicates
    when a profile contains multiple model variants.
    """
    profiles = []
    source_regions = set()
    seen_profile_regions = set()  # Track (profile_id, region) pairs to avoid duplicates

    for region, region_profiles in features_by_region.items():
        for profile in region_profiles:
            profile_id = profile.get(
                "inference_profile_id", profile.get("inferenceProfileId")
            )

            # Skip if we've already added this profile for this region
            profile_region_key = (profile_id, region)
            if profile_region_key in seen_profile_regions:
                continue

            # Check if any model in this profile matches
            profile_models = profile.get("models", [])
            matches = False

            # Normalize model_id by stripping version suffix for matching
            # e.g., "anthropic.claude-sonnet-4-6:0" -> "anthropic.claude-sonnet-4-6"
            # e.g., "anthropic.claude-3-5-sonnet-20240620-v1:0" -> "anthropic.claude-3-5-sonnet-20240620-v1"
            model_id_base = model_id.rsplit(":", 1)[0] if ":" in model_id else model_id

            for pm in profile_models:
                # Handle both snake_case and camelCase model ARN
                model_arn = pm.get("model_arn", pm.get("modelArn", ""))
                # Check both full model_id and base (without version suffix)
                if model_id in model_arn or model_id_base in model_arn:
                    matches = True
                    break  # Found a match, no need to check other models

            if matches:
                profiles.append(
                    {
                        "profile_id": profile_id,
                        "profile_name": profile.get(
                            "inference_profile_name",
                            profile.get("inferenceProfileName"),
                        ),
                        "source_region": region,
                        "type": profile.get("type"),
                        "status": profile.get("status", "ACTIVE"),
                        "description": profile.get("description", ""),
                    }
                )
                source_regions.add(region)
                seen_profile_regions.add(profile_region_key)

    return {
        "supported": len(profiles) > 0,
        "source_regions": sorted(list(source_regions)),
        "profiles": profiles,
    }


def _normalize_for_mantle_match(model_id: str) -> str:
    """Normalize a model ID for fuzzy Mantle matching using centralized utility."""
    return get_canonical_model_id(model_id)


def _normalize_model_name(name: str) -> str:
    """Normalize model name for comparison (lowercase, remove hyphens/spaces/underscores)."""
    if not name:
        return ""
    return name.lower().replace("-", "").replace(" ", "").replace("_", "")


def merge_duplicate_models(providers: dict) -> dict:
    """
    Merge models with matching normalized names within each provider.

    This handles cases where the same model appears with different IDs
    from different sources (e.g., Bedrock API vs Mantle API).

    The model with more data (has_pricing, in_region, etc.) is kept as primary,
    and Mantle data is merged from duplicates.

    Args:
        providers: Dict of provider_name -> {"models": {model_id: model_data}}

    Returns:
        The providers dict with duplicates merged
    """
    for provider_name, provider_data in providers.items():
        models = provider_data.get("models", {})
        if not models:
            continue

        # Group models by normalized name
        by_name: dict[str, list[tuple[str, dict]]] = {}
        for model_id, model in models.items():
            norm_name = _normalize_model_name(model.get("model_name", ""))
            if norm_name:
                if norm_name not in by_name:
                    by_name[norm_name] = []
                by_name[norm_name].append((model_id, model))

        # Find and merge duplicates
        to_remove: set[str] = set()
        for norm_name, model_list in by_name.items():
            if len(model_list) <= 1:
                continue

            # Sort by priority: has_pricing > has in_region > has model_arn
            def priority(item: tuple[str, dict]) -> tuple[bool, bool, bool]:
                model_id, model = item
                return (
                    model.get("has_pricing", False),
                    len(model.get("in_region", [])) > 0,
                    bool(model.get("model_arn", "")),
                )

            model_list.sort(key=priority, reverse=True)
            primary_id, primary = model_list[0]

            # Merge data from duplicates into primary
            for dup_id, dup in model_list[1:]:
                # Merge Mantle regions
                primary_mantle = primary.get("availability", {}).get("mantle", {})
                dup_mantle = dup.get("availability", {}).get("mantle", {})

                if dup_mantle.get("supported") and dup_mantle.get("regions"):
                    if not primary_mantle.get("regions"):
                        primary_mantle["regions"] = []
                    # Add unique regions
                    existing = set(primary_mantle.get("regions", []))
                    for region in dup_mantle.get("regions", []):
                        if region not in existing:
                            primary_mantle["regions"].append(region)
                            existing.add(region)
                    primary_mantle["supported"] = True
                    primary.setdefault("availability", {})["mantle"] = primary_mantle

                # Mark duplicate for removal
                to_remove.add(dup_id)
                logger.info(
                    f"Merging duplicate model {dup_id} into {primary_id} "
                    f"(normalized name: {norm_name})"
                )

        # Remove duplicates
        for model_id in to_remove:
            del models[model_id]

    return providers


def has_mantle_pricing(model_id: str, pricing_data: dict) -> bool:
    """
    Check if a model has Mantle-specific pricing in the pricing data.

    Mantle pricing is identified by:
    - pricing_group == "Mantle"
    - dimensions.inference_mode == "mantle"
    - dimensions.source == "mantle"

    Args:
        model_id: The model identifier
        pricing_data: The pricing data structure

    Returns:
        True if Mantle pricing exists for this model
    """
    if not pricing_data:
        return False

    providers = pricing_data.get("providers", {})

    # Try to find the model in pricing data
    for provider_name, provider_data in providers.items():
        if not isinstance(provider_data, dict):
            continue

        for pricing_key, model_pricing in provider_data.items():
            if not isinstance(model_pricing, dict) or "regions" not in model_pricing:
                continue

            # Check if this pricing entry matches our model
            # Use canonical matching for flexibility
            canonical_model = get_canonical_model_id(model_id)
            canonical_pricing = get_canonical_model_id(pricing_key)

            if canonical_model != canonical_pricing:
                continue

            # Check for Mantle pricing in any region
            for region_data in model_pricing.get("regions", {}).values():
                pricing_groups = region_data.get("pricing_groups", {})

                # Check for "Mantle" group
                if "Mantle" in pricing_groups:
                    return True

                # Check dimensions in any group
                for group_entries in pricing_groups.values():
                    for entry in group_entries:
                        dims = entry.get("dimensions", {})
                        if dims.get("inference_mode") == "mantle":
                            return True
                        if dims.get("source") == "mantle":
                            return True

    return False


def build_mantle_inference(
    model_id: str,
    mantle_by_model: dict,
    pricing_data: dict = None,
) -> dict:
    """Build mantle_inference object for a model.

    Uses the centralized model_matcher for fuzzy matching because the Mantle API
    (/v1/models) returns model IDs in a different format than Bedrock's
    ListFoundationModels. Common differences:
    - Missing version suffixes (-v1:0)
    - -instruct vs -v1:0 suffixes
    - Different provider prefixes (moonshotai vs moonshot)
    - Semantic version differences (v3-v1:0 vs v3.1)

    mantle_by_model values are dicts: {"regions": [...], "supports_responses_api": bool}

    Args:
        model_id: The model identifier
        mantle_by_model: Dict of Mantle models with regions and API support
        pricing_data: Optional pricing data to check for Mantle pricing

    Returns a dict with:
    - supported: bool
    - mantle_regions: list of regions
    - total_mantle_regions: int
    - mantle_endpoint_pattern: str
    - matched_mantle_id: str or None (the Mantle model ID that was matched)
    - supports_responses_api: bool
    - has_pricing: bool (indicates Mantle pricing exists)
    """
    # Check for Mantle pricing upfront
    mantle_has_pricing = (
        has_mantle_pricing(model_id, pricing_data) if pricing_data else False
    )

    # 1. Exact match
    mantle_info = mantle_by_model.get(model_id, {})
    regions = mantle_info.get("regions", [])
    if regions:
        return {
            "supported": True,
            "mantle_regions": regions,
            "mantle_endpoint_pattern": "bedrock-mantle.{region}.api.aws",
            "matched_mantle_id": model_id,
            "supports_responses_api": mantle_info.get("supports_responses_api", False),
            "has_pricing": mantle_has_pricing,
        }

    # 2. Fuzzy match using centralized model_matcher
    # Find the best match using calculate_match_score
    best_match_id = None
    best_score = 0.0
    best_info = {}

    for mantle_id, mantle_info in mantle_by_model.items():
        # Skip if there's a semantic conflict (e.g., v3 vs r1)
        if has_semantic_conflict(model_id, mantle_id):
            continue

        score = calculate_match_score(model_id, mantle_id)
        if score > best_score:
            best_score = score
            best_match_id = mantle_id
            best_info = mantle_info

    # Accept match if score is high enough (0.8 threshold)
    if best_match_id and best_score >= 0.8:
        mantle_regions = best_info.get("regions", [])
        supports_responses_api = best_info.get("supports_responses_api", False)
        return {
            "supported": True,
            "mantle_regions": mantle_regions,
            "mantle_endpoint_pattern": "bedrock-mantle.{region}.api.aws",
            "matched_mantle_id": best_match_id,
            "supports_responses_api": supports_responses_api,
            "has_pricing": mantle_has_pricing,
        }

    # No match found
    return {
        "supported": False,
        "mantle_regions": [],
        "mantle_endpoint_pattern": "bedrock-mantle.{region}.api.aws",
        "matched_mantle_id": None,
        "supports_responses_api": False,
        "has_pricing": False,
    }


def build_provisioned_throughput(model_id: str, provisioned_availability: dict) -> dict:
    """Build provisioned throughput availability object for a model."""
    regions = provisioned_availability.get(model_id, [])
    # Also try fuzzy matching similar to find_matching_availability
    if not regions:
        base_id = model_id.split(":")[0]
        for key in provisioned_availability:
            if key.startswith(base_id):
                regions = provisioned_availability[key]
                break
    return {
        "supported": len(regions) > 0,
        "provisioned_regions": sorted(regions) if regions else [],
    }


def build_govcloud_availability(
    model_id: str,
    model_name: str,
    govcloud_availability: dict,
) -> dict:
    """
    Build GovCloud availability data for a model.

    Matches the model to GovCloud availability data from pricing API using
    model name matching (since pricing API uses model names, not IDs).

    Args:
        model_id: The model identifier
        model_name: The model name (used for matching)
        govcloud_availability: Dict mapping model names to GovCloud info:
            {
                "model_name": {
                    "regions": ["us-gov-west-1", ...],
                    "inference_type": "cris" | "in_region"
                }
            }

    Returns:
        Dict with:
        - supported: bool
        - regions: list of GovCloud regions
        - inference_type: "cris" | "in_region" | None
        - source: "pricing_api"
    """
    if not govcloud_availability:
        return {
            "supported": False,
            "regions": [],
            "inference_type": None,
            "source": "pricing_api",
        }

    # Try exact match on model name first
    govcloud_info = govcloud_availability.get(model_name, {})

    # Handle both old format (list) and new format (dict)
    if isinstance(govcloud_info, list):
        # Old format: just a list of regions
        regions = govcloud_info
        inference_type = None
    elif isinstance(govcloud_info, dict):
        # New format: dict with regions and inference_type
        regions = govcloud_info.get("regions", [])
        inference_type = govcloud_info.get("inference_type")
    else:
        regions = []
        inference_type = None

    # If no exact match, try fuzzy matching
    if not regions:
        model_name_lower = model_name.lower() if model_name else ""
        model_id_lower = model_id.lower() if model_id else ""

        for govcloud_model_name, govcloud_data in govcloud_availability.items():
            govcloud_name_lower = govcloud_model_name.lower()

            # Check if model name is contained in GovCloud model name or vice versa
            if model_name_lower and (
                model_name_lower in govcloud_name_lower
                or govcloud_name_lower in model_name_lower
            ):
                # Handle both old and new format
                if isinstance(govcloud_data, list):
                    regions = govcloud_data
                    inference_type = None
                elif isinstance(govcloud_data, dict):
                    regions = govcloud_data.get("regions", [])
                    inference_type = govcloud_data.get("inference_type")
                break

            # Check if model ID contains the GovCloud model name
            if model_id_lower and govcloud_name_lower in model_id_lower:
                if isinstance(govcloud_data, list):
                    regions = govcloud_data
                    inference_type = None
                elif isinstance(govcloud_data, dict):
                    regions = govcloud_data.get("regions", [])
                    inference_type = govcloud_data.get("inference_type")
                break

    return {
        "supported": len(regions) > 0,
        "regions": sorted(regions) if regions else [],
        "inference_type": inference_type,
        "source": "pricing_api",
    }


def _normalize_for_quota_matching(name: str) -> str:
    """
    Normalize a model name or quota model reference for exact matching.
    - Lowercase
    - Replace hyphens/underscores with spaces
    - Strip default version tags (v1, v1.0) — keeps v2, v2.1, etc.
    - Strip 8-digit date codes (e.g. 20240307)
    - Join standalone single-digit pairs: "3 5" -> "3.5"
    - Strip trailing context length qualifiers (200K, 1M Context Length)
    - Collapse whitespace
    """
    n = name.lower().strip()
    # Strip trailing punctuation (AWS quota typos like "Claude Sonnet 4.5.")
    n = re.sub(r"[.;,!]+$", "", n)
    n = n.replace("-", " ").replace("_", " ")
    # Normalize "+" to " plus" (e.g. "Command R+" → "Command R plus")
    n = n.replace("+", " plus")
    # Strip v1/V1/V1.0 (default version, not a distinguishing identifier)
    n = re.sub(r"\bv1(?:\.0)?\b", "", n, flags=re.I)
    # Strip 8-digit date patterns (e.g. 20240307, 20250929)
    n = re.sub(r"\b\d{8}\b", "", n)
    # Join standalone single-digit pairs not adjacent to letters/digits/dots
    # e.g. "3 5" -> "3.5" but NOT "v1 1m" or "3.1 70b"
    n = re.sub(r"(?<![a-zA-Z\d.])(\d)\s+(\d)(?![a-zA-Z\d])", r"\1.\2", n)
    # Strip trailing context length qualifiers (200K, 1M, 256K, 1M Context Length)
    n = re.sub(r"\s+\d+[kKmM](?:\s+context\s+length)?\s*$", "", n, flags=re.I)
    # Collapse whitespace
    n = " ".join(n.split())
    return n


# Mapping of Mantle model ID prefixes to provider display names
# Used for creating stub entries for Mantle-only models
MANTLE_PROVIDER_NAMES = {
    "zai": "Zhipu AI",
    "anthropic": "Anthropic",
    "meta": "Meta",
    "mistral": "Mistral AI",
    "cohere": "Cohere",
    "ai21": "AI21 Labs",
    "amazon": "Amazon",
    "stability": "Stability AI",
    "deepseek": "DeepSeek",
    "minimax": "MiniMax",
    "moonshot": "Moonshot AI",
    "moonshotai": "Moonshot AI",
    "writer": "Writer",
    "luma": "Luma AI",
    "nvidia": "NVIDIA",
    "qwen": "Alibaba Cloud",
}

# Explicit name overrides for Mantle model IDs where generic derivation doesn't work well
# This covers edge cases like version-only names (deepseek.v3.1 → "DeepSeek V3.1")
MANTLE_MODEL_NAME_OVERRIDES = {
    "deepseek.v3.1": "DeepSeek V3.1",
}


# Known acronyms and abbreviations that should be all-uppercase
_UPPERCASE_WORDS = {
    "glm",
    "gpt",
    "llm",
    "ai",
    "xl",
    "sd",
    "sdxl",
    "r1",
    "v1",
    "v2",
    "v3",
    "v4",
}


def _derive_model_name_from_id(mantle_id: str) -> str:
    """
    Derive a human-readable model name from a Mantle model ID.

    Examples:
        zai.glm-4.6 -> GLM 4.6
        anthropic.claude-3-sonnet -> Claude 3 Sonnet
        meta.llama-3-70b-instruct -> Llama 3 70B Instruct
    """
    # Extract the model part after the provider prefix
    if "." in mantle_id:
        model_part = mantle_id.split(".", 1)[1]
    else:
        model_part = mantle_id

    # Replace hyphens with spaces
    name = model_part.replace("-", " ")

    # Title case each word, but handle special cases
    words = name.split()
    result_words = []
    for word in words:
        # Keep version numbers as-is (e.g., "4.6", "3.5")
        if re.match(r"^\d+\.?\d*$", word):
            result_words.append(word)
        # Keep size indicators uppercase (e.g., "70b" -> "70B")
        elif re.match(r"^\d+[bBkKmM]$", word):
            result_words.append(word.upper())
        # Keep known acronyms uppercase
        elif word.lower() in _UPPERCASE_WORDS:
            result_words.append(word.upper())
        else:
            result_words.append(word.title())

    return " ".join(result_words)


def _find_pricing_for_mantle_stub(
    mantle_id: str, model_name: str, pricing_data: dict
) -> Optional[dict]:
    """Search pricing data for a Mantle-only model. Returns pricing_ref dict if found.

    Uses substring matching (for flexibility with -instruct suffixes etc.)
    guarded by has_semantic_conflict to prevent false positives like m2.5 → m2.
    """
    providers = pricing_data.get("providers", {})
    mantle_id_lower = mantle_id.lower()
    model_name_lower = model_name.lower() if model_name else ""

    for prov_name, prov_data in providers.items():
        if not isinstance(prov_data, dict):
            continue
        for model_key, model_data in prov_data.items():
            if not isinstance(model_data, dict) or "regions" not in model_data:
                continue
            key_lower = model_key.lower()
            # Match by model_id or model name (substring matching)
            if (
                mantle_id_lower == key_lower
                or mantle_id_lower in key_lower
                or key_lower in mantle_id_lower
            ):
                # Guard: skip if semantic conflict detected
                if has_semantic_conflict(mantle_id, model_key):
                    continue
                return {
                    "provider": prov_name,
                    "model_key": model_key,
                    "model_name": model_name,
                }
            if model_name_lower and (
                model_name_lower in key_lower or key_lower in model_name_lower
            ):
                if has_semantic_conflict(mantle_id, model_key):
                    continue
                return {
                    "provider": prov_name,
                    "model_key": model_key,
                    "model_name": model_name,
                }
    return None


def create_mantle_only_stub(
    mantle_id: str,
    regions: list,
    collection_timestamp: str,
    supports_responses_api: bool = False,
) -> dict:
    """
    Create a stub model entry for a Mantle-only model (not in Bedrock's ListFoundationModels).

    These models exist in the Mantle API but have no corresponding Bedrock foundation model.
    The stub provides minimal metadata to display the model in the UI.
    """
    # Extract provider using centralized utility from model_matcher
    _, provider_name = get_provider_from_model_id(mantle_id)

    # Derive model name — check explicit overrides first, then generic derivation
    model_name = MANTLE_MODEL_NAME_OVERRIDES.get(
        mantle_id
    ) or _derive_model_name_from_id(mantle_id)

    # Build api_support for Mantle-only model
    api_support = {
        "invoke_model": {
            "supported": False,
            "streaming": False,
            "endpoint": "bedrock-runtime",
        },
        "converse": {
            "supported": False,
            "streaming": False,
            "endpoint": "bedrock-runtime",
            "features": {},
        },
        "chat_completions": {
            "supported": True,
            "endpoints": ["bedrock-runtime", "bedrock-mantle"],
        },
        "responses_api": {
            "supported": supports_responses_api,
            "endpoint": "bedrock-mantle",
        },
        "endpoints_supported": ["bedrock-mantle"],
    }

    # Get documentation URLs from config
    config = get_config_loader()
    default_bedrock_guide = config.get_documentation_url("bedrock_model_ids")
    default_pricing_guide = config.get_documentation_url("bedrock_pricing")

    # Build intermediate data structures for availability
    cross_region_data = {
        "supported": False,
        "source_regions": [],
        "profiles": [],
    }
    batch_inference_data = {
        "supported": False,
        "supported_regions": [],
        "coverage_percentage": 0.0,
        "detection_method": "no_pricing_data",
    }
    provisioned_data = {
        "supported": False,
        "provisioned_regions": [],
    }
    mantle_data = {
        "supported": True,
        "mantle_regions": regions,
        "mantle_endpoint_pattern": "bedrock-mantle.{region}.api.aws",
        "supports_responses_api": supports_responses_api,
        "has_pricing": False,  # Will be updated if pricing is found
    }

    # Build model_pricing structure
    model_pricing = {
        "is_pricing_available": False,
        "pricing_reference_id": mantle_id,
        "pricing_file_reference": {
            "provider": provider_name,
            "model_key": mantle_id,
            "model_name": model_name,
        },
        "pricing_summary": {
            "integration_source": "mantle_only_stub",
            "has_pricing_data": False,
            "integration_timestamp": collection_timestamp,
            "reference_based": False,
        },
    }

    # Build converse_data structure
    converse_data = {
        "context_window": None,
        "max_output_tokens": None,
        "size_category": {"category": "Unknown", "color": "#6B7280", "tier": 0},
        "verified": False,
        "source": "unknown",
        "litellm_verified": False,
        "capabilities_count": 0,
        "use_cases_count": 0,
        "regions_count": 0,
        "has_extended_context": False,
    }

    # Build modalities structure
    modalities = {
        "input_modalities": ["TEXT"],
        "output_modalities": ["TEXT"],
    }

    # Build lifecycle structure
    lifecycle = {"status": "ACTIVE", "release_date": ""}

    # Build documentation links
    docs = {
        "aws_bedrock_guide": default_bedrock_guide,
        "pricing_guide": default_pricing_guide,
    }

    # Determine visibility
    hidden_models = config.config.get("model_configuration", {}).get(
        "hidden_models", []
    )

    return {
        # Visibility flag
        "show_model": mantle_id not in hidden_models,
        # Core identifiers
        "model_id": mantle_id,
        "model_arn": "",
        "model_name": model_name,
        "model_provider": provider_name,
        # Primary regional data (empty for Mantle-only)
        "in_region": [],  # Mantle-only models have no ON_DEMAND availability
        # Model configuration
        "customization": {
            "customization_supported": [],
            "customization_options": {},
        },
        "inference_types_supported": [],
        # Descriptions
        "description": "",
        "short_description": "",
        # Chat features
        "chat_features": {},
        # Consumption options
        "consumption_options": ["mantle"],
        # Collection metadata
        "collection_metadata": {
            "first_discovered_at": collection_timestamp,
            "first_discovered_in_region": regions[0] if regions else "unknown",
            "api_source": "mantle_only",
            "dual_region_collection": False,
            "regions_collected_from": [],
            "phase2_regional_discovery": False,
            "regional_data_source": "mantle_api",
        },
        # Boolean flags
        "has_pricing": False,
        "has_quotas": False,
        # NEW consolidated fields (Phase 3)
        "availability": build_availability(
            regional_availability=[],
            cross_region_data=cross_region_data,
            batch_inference_data=batch_inference_data,
            provisioned_data=provisioned_data,
            mantle_data=mantle_data,
            govcloud_data=None,  # Mantle-only models don't have GovCloud availability
            is_mantle_only=True,
        ),
        "modalities": modalities,
        "capabilities": [],
        "use_cases": [],
        "lifecycle": lifecycle,
        "streaming": True,
        "languages": [],
        "docs": docs,
        "features": {},
        "specs": build_specs(converse_data),
        "pricing": build_pricing_alias(model_pricing),
        "quotas": {},
        "api": api_support,
    }


# Known provider prefixes that appear in quota names but not in model names.
# Ordered longest-first to avoid partial matches.
_PROVIDER_PREFIXES = [
    "anthropic ",
    "ai21 labs ",
    "stability.ai ",
    "stability ai ",
    "mistral ai ",
    "moonshot ai ",
    "writer ai ",
    "luma ai ",
    "twelvelabs ",
    "deepseek ",
    "minimax ",
    "openai ",
    "nvidia ",
    "amazon ",
    "google ",
    "cohere ",
    "meta ",
    "luma ",
    "qwen ",
    "mistral ",  # After 'mistral ai ' — catches "Mistral Mixtral..." in quotas
]


def _strip_provider_prefix(name: str) -> str:
    """Strip a known provider prefix from a normalized (lowercase) name."""
    for prefix in _PROVIDER_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _extract_quota_model_ref(quota_name: str) -> str:
    """
    Extract the model reference string from a quota name.

    Quota names follow patterns like:
      "On-demand model inference requests per minute for Anthropic Claude 3.5 Sonnet"
      "Batch inference job size (in GB) for Claude Sonnet 4.5"
      "(Model customization) ... for a Claude 3 Haiku v1 Fine-tuning job"
      "Model units per provisioned model for the 128k context length variant for Amazon Nova Micro"
      "No-commitment model units for Provisioned Throughput created for base model Amazon Nova 2 Lite V1.0 256K"

    Returns the model name portion, or None if not found.
    """
    name = quota_name.strip()

    # Remove leading category prefix like "(Model customization)"
    name = re.sub(r"^\([^)]+\)\s*", "", name)

    # Remove trailing "(doubled for cross-region calls)" qualifier
    name = re.sub(r"\s*\(doubled\s+for[^)]*\)\s*$", "", name, flags=re.I)

    # Split by "for" and take the LAST segment (the model reference)
    parts = re.split(r"\bfor\s+", name, flags=re.I)
    if len(parts) < 2:
        return None

    ref = parts[-1].strip()

    # Clean up extracted ref:
    # Remove leading articles "a "/"an "
    ref = re.sub(r"^(?:a|an)\s+", "", ref, flags=re.I)
    # Remove "base model"/"custom model" prefix
    ref = re.sub(r"^(?:base|custom)\s+model\s+", "", ref, flags=re.I)
    # Remove trailing job type suffixes (Fine-tuning, Continued Pre-Training, distillation)
    ref = re.sub(
        r"\s+(?:Fine[- ]?tuning|Continued Pre[- ]?Training|distillation)\b.*$",
        "",
        ref,
        flags=re.I,
    )
    # Remove trailing "per month"
    ref = re.sub(r"\s+per\s+month$", "", ref, flags=re.I)

    return ref.strip() if ref.strip() else None


def _build_model_aliases(model_id: str, model_name: str, model_provider: str) -> set:
    """
    Build a set of normalized aliases for a model that quotas might reference.

    Generates aliases from:
    1. model_name (primary)
    2. provider + model_name (for quotas that include provider prefix)
    3. model_name without parenthetical (for Mistral date versions like "(24.07)")
    4. model_id-derived name (catches naming variants like "2407" vs "(24.07)")
    """
    aliases = set()
    if not model_name:
        return aliases

    # Normalize provider name
    prov = (model_provider or "").lower().strip()

    # Alias 1: from model_name
    norm_name = _normalize_for_quota_matching(model_name)
    aliases.add(norm_name)
    # Also without provider prefix (in case model_name includes it, e.g. "DeepSeek-R1")
    aliases.add(_strip_provider_prefix(norm_name))

    # Alias 2: provider + model_name (for quotas like "Anthropic Claude 3.5 Sonnet")
    if prov and not norm_name.startswith(prov):
        aliases.add(_normalize_for_quota_matching(prov + " " + model_name))

    # Alias 3: model_name without parenthetical (e.g. "Mistral Large (24.07)" -> "Mistral Large")
    if "(" in model_name:
        name_no_parens = re.sub(r"\s*\([^)]*\)", "", model_name).strip()
        if name_no_parens:
            np = _normalize_for_quota_matching(name_no_parens)
            aliases.add(np)
            aliases.add(_strip_provider_prefix(np))
            if prov and not np.startswith(prov):
                aliases.add(_normalize_for_quota_matching(prov + " " + name_no_parens))
        # Also: parens removed but content kept (e.g. "Pixtral Large (25.02)" -> "Pixtral Large 25.02")
        name_flat_parens = model_name.replace("(", "").replace(")", "").strip()
        fp = _normalize_for_quota_matching(name_flat_parens)
        aliases.add(fp)
        aliases.add(_strip_provider_prefix(fp))
        if prov and not fp.startswith(prov):
            aliases.add(_normalize_for_quota_matching(prov + " " + name_flat_parens))

    # Alias 5: short name without trailing size+task suffix
    # e.g. "Llama 4 Maverick 17B Instruct" -> "Llama 4 Maverick"
    short_name = re.sub(r"\s+\d+[Bb]\s+(?:Instruct|Chat|IT|PT)\s*$", "", model_name)
    if short_name != model_name:
        sn = _normalize_for_quota_matching(short_name)
        aliases.add(sn)
        aliases.add(_strip_provider_prefix(sn))
        if prov and not sn.startswith(prov):
            aliases.add(_normalize_for_quota_matching(prov + " " + short_name))

    # Alias 6: without trailing version number (e.g. "Stable Image Core 1.0" -> "Stable Image Core")
    short_ver = re.sub(r"\s+\d+\.\d+\s*$", "", model_name)
    if short_ver != model_name:
        sv = _normalize_for_quota_matching(short_ver)
        aliases.add(sv)
        aliases.add(_strip_provider_prefix(sv))
        if prov and not sv.startswith(prov):
            aliases.add(_normalize_for_quota_matching(prov + " " + short_ver))

    # Alias 4: from model_id (catches naming variants)
    if model_id:
        clean_id = model_id.split(":")[0]  # Remove :0, :18k etc.
        # Extract model part after provider prefix (e.g. "anthropic.claude-sonnet-4-5-20250929-v1")
        id_parts = clean_id.split(".", 1)
        model_part = id_parts[1] if len(id_parts) > 1 else clean_id
        # Remove trailing date+v1 or just v1 (default version only; keep v2+ as they distinguish models)
        model_part = re.sub(r"(-\d{8})?-v1$", "", model_part)
        # Remove trailing standalone 8-digit date (for models without version suffix)
        model_part = re.sub(r"-\d{8}$", "", model_part)
        if model_part:
            id_alias = _normalize_for_quota_matching(model_part)
            aliases.add(id_alias)

    # Remove any empty strings
    aliases.discard("")
    return aliases


# Cached quota index: maps normalized model ref -> {region -> [quotas]}
_quota_index = None


def _build_quota_index(quotas_by_region: dict) -> dict:
    """
    Pre-index all quotas by their normalized model reference.
    This enables O(1) lookup per model instead of O(quotas) scanning.
    """
    index = {}
    for region, quotas in quotas_by_region.items():
        for quota in quotas:
            quota_name = quota.get("quotaName", quota.get("quota_name", ""))
            ref = _extract_quota_model_ref(quota_name)
            if not ref:
                continue
            # Normalize and index both with and without provider prefix
            norm = _normalize_for_quota_matching(ref)
            norm_no_prov = _strip_provider_prefix(norm)
            for key in {norm, norm_no_prov}:
                if key:
                    index.setdefault(key, {}).setdefault(region, []).append(quota)
    return index


def build_model_quotas(
    model_id: str, model_name: str, quotas_by_region: dict, model_provider: str = ""
) -> dict:
    """
    Build model-specific quotas by region using exact name matching.

    Uses a pre-built index of quota model references for efficient lookup.
    Matches quota names against model aliases derived from model_name,
    model_provider, and model_id — no hardcoded model lists or keyword matching.
    """
    global _quota_index
    if _quota_index is None:
        _quota_index = _build_quota_index(quotas_by_region)

    aliases = _build_model_aliases(model_id, model_name, model_provider)
    model_quotas = {}
    seen_codes_per_region = {}  # Dedup: same quota found via multiple aliases

    for alias in aliases:
        matched = _quota_index.get(alias, {})
        for region, quotas in matched.items():
            if region not in seen_codes_per_region:
                seen_codes_per_region[region] = set()
            for quota in quotas:
                code = quota.get("quotaCode", quota.get("quota_code", ""))
                if code in seen_codes_per_region[region]:
                    continue
                seen_codes_per_region[region].add(code)
                model_quotas.setdefault(region, []).append(
                    {
                        "quota_code": code,
                        "quota_name": quota.get(
                            "quotaName", quota.get("quota_name", "")
                        ),
                        "quota_arn": quota.get("quotaArn", quota.get("quota_arn", "")),
                        "description": quota.get("description", ""),
                        "quota_applied_at_level": quota.get(
                            "quotaAppliedAtLevel",
                            quota.get("quota_applied_at_level", "ACCOUNT"),
                        ),
                        "value": quota.get("value", 0),
                        "unit": quota.get("unit", "None"),
                        "adjustable": quota.get("adjustable", False),
                        "global_quota": quota.get(
                            "globalQuota", quota.get("global_quota", False)
                        ),
                        "usage_metric": quota.get(
                            "usageMetric", quota.get("usage_metric", {})
                        ),
                        "period": quota.get("period", {}),
                    }
                )

    return model_quotas


def get_consumption_options(
    inference_types: list,
    pricing_data: dict = None,
    pricing_ref: dict = None,
    mantle_supported: bool = False,
) -> list:
    """Determine consumption options from inference types and pricing data.

    'on_demand' is added ONLY when inference_types contains ON_DEMAND.
    Pricing data adds 'batch' and 'provisioned_throughput' but NOT 'on_demand'.
    """
    options = set()

    # Map inference types to consumption options
    type_mapping = {
        "ON_DEMAND": "on_demand",
        "PROVISIONED": "provisioned_throughput",
        "INFERENCE_PROFILE": "cross_region_inference",
    }
    for inf_type in inference_types:
        if inf_type in type_mapping:
            options.add(type_mapping[inf_type])

    # Check pricing data for ADDITIONAL consumption options (batch, provisioned)
    # NOTE: Do NOT add "on_demand" from pricing — that must come from inference_types only
    if pricing_data and pricing_ref:
        provider = pricing_ref.get("provider", "")
        model_key = pricing_ref.get("model_key", "")

        if provider and model_key:
            providers = pricing_data.get("providers", {})
            prov_data = providers.get(provider, {})
            model_pricing = prov_data.get(model_key, {})

            if isinstance(model_pricing, dict) and "regions" in model_pricing:
                # Check first available region for pricing groups
                for region_data in model_pricing.get("regions", {}).values():
                    pricing_groups = region_data.get("pricing_groups", {})

                    # Check for Batch pricing
                    if any(g.startswith("Batch") for g in pricing_groups.keys()):
                        options.add("batch")

                    # Check for Provisioned Throughput pricing
                    if "Provisioned Throughput" in pricing_groups:
                        options.add("provisioned_throughput")

                    # Check for Reserved pricing
                    if any(g.startswith("Reserved") for g in pricing_groups.keys()):
                        options.add("reserved")

                    break  # Only need to check one region

    # Check for Mantle support
    if mantle_supported:
        options.add("mantle")

    # Sort for consistent ordering
    order = [
        "on_demand",
        "batch",
        "cross_region_inference",
        "mantle",
        "provisioned_throughput",
        "reserved",
    ]
    return sorted(
        list(options), key=lambda x: order.index(x) if x in order else len(order)
    )


def check_batch_inference(
    model_id: str,
    pricing_data: dict,
    pricing_ref: dict = None,
    regional_availability: list = None,
) -> dict:
    """Check if batch inference is supported based on pricing data.

    Uses pricing_file_reference.model_key for accurate matching when available.
    Calculates coverage_percentage based on regional_availability.
    """
    supported_regions = []

    # Use pricing reference model_key if available, otherwise fall back to model_id
    lookup_keys = []
    if pricing_ref:
        provider = pricing_ref.get("provider", "")
        model_key = pricing_ref.get("model_key", "")
        if provider and model_key:
            lookup_keys.append((provider, model_key))

    # Also try with the original model_id
    lookup_keys.append((None, model_id))

    providers = pricing_data.get("providers", {})

    for provider_hint, lookup_key in lookup_keys:
        if supported_regions:
            break  # Already found, no need to continue

        for prov_name, prov_data in providers.items():
            if supported_regions:
                break

            # Skip if provider hint doesn't match
            if provider_hint and prov_name.lower() != provider_hint.lower():
                continue

            # Check for provider -> model structure (new schema)
            if isinstance(prov_data, dict):
                # Direct model lookup
                if lookup_key in prov_data:
                    model_data = prov_data[lookup_key]
                    if isinstance(model_data, dict) and "regions" in model_data:
                        for region, region_data in model_data.get(
                            "regions", {}
                        ).items():
                            pricing_groups = region_data.get("pricing_groups", {})
                            # Check for any Batch pricing group
                            if any(
                                g.startswith("Batch") for g in pricing_groups.keys()
                            ):
                                if region not in supported_regions:
                                    supported_regions.append(region)

                # Fuzzy matching as fallback
                if not supported_regions:
                    lookup_lower = lookup_key.lower()
                    for mid, model_data in prov_data.items():
                        if isinstance(model_data, dict) and "regions" in model_data:
                            # Check if lookup_key is contained in mid or vice versa
                            mid_lower = mid.lower()
                            if lookup_lower in mid_lower or mid_lower in lookup_lower:
                                for region, region_data in model_data.get(
                                    "regions", {}
                                ).items():
                                    pricing_groups = region_data.get(
                                        "pricing_groups", {}
                                    )
                                    if any(
                                        g.startswith("Batch")
                                        for g in pricing_groups.keys()
                                    ):
                                        if region not in supported_regions:
                                            supported_regions.append(region)

    # Calculate coverage percentage based on model's regional availability
    total_regions = len(regional_availability) if regional_availability else 0
    batch_region_count = len(supported_regions)
    coverage = (batch_region_count / total_regions * 100) if total_regions > 0 else 0.0
    # Cap at 100% - values > 100% indicate batch pricing in more regions than model availability
    coverage = min(coverage, 100.0)

    return {
        "supported": len(supported_regions) > 0,
        "supported_regions": sorted(supported_regions),
        "coverage_percentage": round(coverage, 1),
        "detection_method": "pricing_data" if supported_regions else "no_pricing_data",
    }


def build_reserved_capacity(
    model_id: str,
    pricing_data: dict,
    pricing_ref: dict = None,
) -> dict:
    """Check if Reserved Capacity pricing exists based on pricing data.

    Scans all regions' pricing_groups for any group starting with "Reserved".
    Extracts commitment terms from group names (e.g., "Reserved 1 Month Geo" → "1_month").

    Uses pricing_file_reference.model_key for accurate matching when available,
    with fuzzy fallback (same pattern as check_batch_inference).
    """
    supported_regions = []
    commitment_terms = set()

    # Use pricing reference model_key if available, otherwise fall back to model_id
    lookup_keys = []
    if pricing_ref:
        provider = pricing_ref.get("provider", "")
        model_key = pricing_ref.get("model_key", "")
        if provider and model_key:
            lookup_keys.append((provider, model_key))

    # Also try with the original model_id
    lookup_keys.append((None, model_id))

    providers = pricing_data.get("providers", {})

    for provider_hint, lookup_key in lookup_keys:
        if supported_regions:
            break  # Already found, no need to continue

        for prov_name, prov_data in providers.items():
            if supported_regions:
                break

            # Skip if provider hint doesn't match
            if provider_hint and prov_name.lower() != provider_hint.lower():
                continue

            if not isinstance(prov_data, dict):
                continue

            # Direct model lookup
            model_data = prov_data.get(lookup_key)

            # Fuzzy matching as fallback
            if not model_data:
                lookup_lower = lookup_key.lower()
                for mid, candidate in prov_data.items():
                    if isinstance(candidate, dict) and "regions" in candidate:
                        mid_lower = mid.lower()
                        if lookup_lower in mid_lower or mid_lower in lookup_lower:
                            model_data = candidate
                            break

            if isinstance(model_data, dict) and "regions" in model_data:
                for region, region_data in model_data.get("regions", {}).items():
                    pricing_groups = region_data.get("pricing_groups", {})
                    reserved_groups = [
                        g for g in pricing_groups.keys() if g.startswith("Reserved")
                    ]
                    if reserved_groups:
                        if region not in supported_regions:
                            supported_regions.append(region)
                        for group_name in reserved_groups:
                            term = _parse_commitment_term(group_name)
                            if term:
                                commitment_terms.add(term)

    return {
        "supported": len(supported_regions) > 0,
        "regions": sorted(supported_regions),
        "commitments": sorted(commitment_terms),
    }


def _parse_commitment_term(group_name: str) -> str:
    """Extract commitment term from a Reserved pricing group name.

    Examples:
        "Reserved 1 Month Geo" → "1_month"
        "Reserved 3 Month Global" → "3_month"
        "Reserved 6 Month Geo" → "6_month"
    """
    match = re.match(r"Reserved\s+(\d+)\s+Month", group_name)
    if match:
        return f"{match.group(1)}_month"
    return ""


def build_api_support(
    model: dict,
    mantle_inference: dict,
    is_mantle_only: bool = False,
) -> dict:
    """Build the unified api_support object for a model.

    Derives API support from existing collected data:
    - InvokeModel: all Bedrock models support it (not Mantle-only)
    - Converse: inferred from chat_features (non-empty = supported)
    - Chat Completions: inferred from mantle_inference.supported
    - Responses API: from mantle_inference.supports_responses_api
    """
    chat_features = model.get("chat_features", {})

    # InvokeModel: all Bedrock models support it, Mantle-only don't
    invoke_supported = not is_mantle_only

    # Converse: inferred from non-empty chat_features
    converse_supported = bool(chat_features) and not is_mantle_only

    # Chat Completions: if model is in Mantle
    chat_completions_supported = mantle_inference.get("supported", False)

    # Responses API: from the probe result
    responses_api_supported = mantle_inference.get("supports_responses_api", False)

    # Determine which endpoints this model is available on
    endpoints = []
    if invoke_supported:
        endpoints.append("bedrock-runtime")
    if chat_completions_supported:
        endpoints.append("bedrock-mantle")

    return {
        "invoke_model": {
            "supported": invoke_supported,
            "streaming": model.get("streaming_supported", False),
            "endpoint": "bedrock-runtime",
        },
        "converse": {
            "supported": converse_supported,
            "streaming": (
                converse_supported
                and (
                    chat_features.get("function_calling_streaming", False)
                    or model.get("streaming_supported", False)
                )
            ),
            "endpoint": "bedrock-runtime",
            "features": (
                {
                    "system_prompts": chat_features.get("system_role", False),
                    "tool_use": chat_features.get("function_calling", False),
                    "streaming_tool_use": chat_features.get(
                        "function_calling_streaming", False
                    ),
                    "vision": bool(chat_features.get("supported_image_types")),
                    "document_chat": chat_features.get("documents", False),
                    "citations": chat_features.get("citations", False),
                    "reasoning": (
                        chat_features.get("reasoning", {}).get("embedded", False)
                        if isinstance(chat_features.get("reasoning"), dict)
                        else False
                    ),
                }
                if converse_supported
                else {}
            ),
        },
        "chat_completions": {
            "supported": chat_completions_supported,
            "endpoints": (
                ["bedrock-runtime", "bedrock-mantle"]
                if chat_completions_supported
                else []
            ),
        },
        "responses_api": {
            "supported": responses_api_supported,
            "endpoint": "bedrock-mantle",
        },
        "endpoints_supported": endpoints,
    }


def build_endpoint_availability(
    regional_availability: list,
    mantle_inference: dict,
    api_support: dict,
) -> dict:
    """Build endpoint_availability showing per-endpoint regional data."""
    runtime_apis = []
    if api_support.get("invoke_model", {}).get("supported"):
        runtime_apis.append("invoke_model")
    if api_support.get("converse", {}).get("supported"):
        runtime_apis.append("converse")
    if api_support.get("chat_completions", {}).get("supported"):
        runtime_apis.append("chat_completions")

    mantle_apis = []
    if api_support.get("chat_completions", {}).get("supported"):
        mantle_apis.append("chat_completions")
    if api_support.get("responses_api", {}).get("supported"):
        mantle_apis.append("responses_api")

    result = {}
    if regional_availability:
        result["bedrock_runtime"] = {
            "regions": regional_availability,
            "apis": runtime_apis,
        }
    if mantle_inference.get("supported"):
        result["bedrock_mantle"] = {
            "regions": mantle_inference.get("mantle_regions", []),
            "apis": mantle_apis,
        }

    return result


# =============================================================================
# Phase 1 JSON Restructure - New consolidated field builders
# =============================================================================


def build_availability(
    regional_availability: list,
    cross_region_data: dict,
    batch_inference_data: dict,
    provisioned_data: dict,
    mantle_data: dict,
    govcloud_data: dict = None,
    is_mantle_only: bool = False,
    reserved_data: dict = None,
) -> dict:
    """
    Build consolidated availability object from component data.

    Args:
        regional_availability: List of on-demand regions
        cross_region_data: Cross-region inference data from build_cross_region_inference()
        batch_inference_data: Batch inference data from check_batch_inference()
        provisioned_data: Provisioned throughput data from build_provisioned_throughput()
        mantle_data: Mantle inference data from build_mantle_inference()
        govcloud_data: GovCloud availability data (optional)
        is_mantle_only: Whether this is a Mantle-only model
        reserved_data: Reserved capacity data from build_reserved_capacity()

    Returns:
        Consolidated availability object with:
        - on_demand: {supported, regions}
        - cross_region: {supported, regions, profiles}
        - batch: {supported, regions}
        - provisioned: {supported, regions}
        - mantle: {supported, regions, only, responses_api}
        - reserved: {supported, regions, commitments}
        - govcloud: {supported, regions, inference_type, source}
    """
    # On-demand availability
    on_demand_regions = regional_availability if regional_availability else []

    # Cross-region inference
    cross_region = {
        "supported": cross_region_data.get("supported", False),
        "regions": cross_region_data.get("source_regions", []),
        "profiles": cross_region_data.get("profiles", []),
    }

    # Batch inference
    batch = {
        "supported": batch_inference_data.get("supported", False),
        "regions": batch_inference_data.get("supported_regions", []),
    }

    # Provisioned throughput
    provisioned = {
        "supported": provisioned_data.get("supported", False),
        "regions": provisioned_data.get("provisioned_regions", []),
    }

    # Mantle inference
    mantle = {
        "supported": mantle_data.get("supported", False),
        "regions": mantle_data.get("mantle_regions", []),
        "only": is_mantle_only,
        "responses_api": mantle_data.get("supports_responses_api", False),
        "has_pricing": mantle_data.get("has_pricing", False),
    }

    # Reserved capacity
    reserved = {
        "supported": reserved_data.get("supported", False) if reserved_data else False,
        "regions": reserved_data.get("regions", []) if reserved_data else [],
        "commitments": reserved_data.get("commitments", []) if reserved_data else [],
    }

    # GovCloud availability
    govcloud_regions = govcloud_data.get("regions", []) if govcloud_data else []
    govcloud_inference_type = (
        govcloud_data.get("inference_type") if govcloud_data else None
    )
    govcloud = {
        "supported": len(govcloud_regions) > 0,
        "regions": govcloud_regions,
        "inference_type": govcloud_inference_type,
        "source": "pricing_api",
    }

    # Determine if In-Region should be hidden (model has both Mantle and In-Region)
    # Mantle is prioritized over In-Region for models with both availability options
    mantle_has_regions = mantle["supported"] and len(mantle["regions"]) > 0
    on_demand_has_regions = len(on_demand_regions) > 0
    hide_in_region = mantle_has_regions and on_demand_has_regions

    return {
        "hide_in_region": hide_in_region,
        "on_demand": {
            "supported": len(on_demand_regions) > 0,
            "regions": on_demand_regions,
        },
        "cross_region": cross_region,
        "batch": batch,
        "provisioned": provisioned,
        "mantle": mantle,
        "reserved": reserved,
        "govcloud": govcloud,
    }


def backfill_availability_from_pricing(
    availability: dict,
    consumption_options: list,
    pricing_data: dict,
    pricing_ref: dict,
    model_id: str,
) -> dict:
    """
    Backfill availability regions from pricing data when API data is missing.

    Some models have pricing data but are not returned by the Bedrock API
    (ListFoundationModels). This function fills in the availability regions
    from pricing data to ensure these models appear in regional availability views.

    The function is conservative - it only backfills when:
    1. The availability regions for a consumption type are empty
    2. The model has that consumption type in consumption_options
    3. The pricing data has regions for that consumption type

    Args:
        availability: The availability dict from build_availability()
        consumption_options: List of consumption options for the model
        pricing_data: Full pricing data dict
        pricing_ref: Pricing reference for this model
        model_id: Model ID for logging

    Returns:
        Updated availability dict with backfilled regions
    """
    if not pricing_data or not pricing_ref:
        return availability

    provider = pricing_ref.get("provider", "")
    model_key = pricing_ref.get("model_key", "")

    if not provider or not model_key:
        return availability

    # Get pricing data for this model
    providers = pricing_data.get("providers", {})
    prov_data = providers.get(provider, {})
    model_pricing = prov_data.get(model_key, {})

    if not isinstance(model_pricing, dict) or "regions" not in model_pricing:
        return availability

    pricing_regions = model_pricing.get("regions", {})
    if not pricing_regions:
        return availability

    # Map consumption options to availability keys and pricing group patterns
    # Each entry: (consumption_option, availability_key, pricing_group_patterns)
    backfill_mappings = [
        ("on_demand", "on_demand", ["On-Demand", "Standard"]),
        ("batch", "batch", ["Batch"]),
        ("provisioned_throughput", "provisioned", ["Provisioned Throughput"]),
    ]

    changes_made = []

    for consumption_opt, avail_key, pricing_patterns in backfill_mappings:
        # Skip if model doesn't have this consumption option
        if consumption_opt not in consumption_options:
            continue

        # Skip if availability already has regions
        avail_section = availability.get(avail_key, {})
        if avail_section.get("regions"):
            continue

        # Find regions from pricing data that have matching pricing groups
        backfill_regions = []
        for region_code, region_data in pricing_regions.items():
            pricing_groups = region_data.get("pricing_groups", {})
            # Check if any pricing group matches our patterns
            has_matching_group = any(
                any(pattern in group_name for pattern in pricing_patterns)
                for group_name in pricing_groups.keys()
            )
            if has_matching_group:
                backfill_regions.append(region_code)

        if backfill_regions:
            # Sort regions for consistency
            backfill_regions = sorted(backfill_regions)

            # Update availability
            if avail_key not in availability:
                availability[avail_key] = {"supported": False, "regions": []}

            availability[avail_key]["regions"] = backfill_regions
            availability[avail_key]["supported"] = True

            changes_made.append(f"{avail_key}={len(backfill_regions)} regions")

    if changes_made:
        logger.info(
            f"Backfilled availability from pricing for {model_id}: {', '.join(changes_made)}"
        )

    return availability


def build_specs(converse_data: dict) -> dict:
    """Build simplified specs object from converse_data."""
    return {
        "context_window": converse_data.get("context_window"),
        "max_output": converse_data.get("max_output_tokens"),
        "extended_context": converse_data.get("extended_context"),
        "size_category": converse_data.get("size_category"),
        "source": converse_data.get("source"),
        "verified": converse_data.get("verified", False),
    }


def build_pricing_alias(model_pricing: dict) -> dict:
    """Build simplified pricing object."""
    return {
        "available": model_pricing.get("is_pricing_available", False),
        "reference": model_pricing.get("pricing_file_reference"),
    }


# =============================================================================
# Sub-functions for transform_model_to_schema() decomposition
# =============================================================================


@tracer.capture_method
def _resolve_context_window(
    model_id: str,
    model: dict,
    token_specs: dict,
    enriched_model: dict,
) -> dict:
    """Resolve context window using 4-tier priority.

    Priority:
    1. Console API metadata (from model-extractor REST call)
    2. Model ID size variant (from model-merger)
    3. profiler-config.json
    4. LiteLLM token_specs (last resort)

    Args:
        model_id: The model identifier
        model: Model data from upstream
        token_specs: Token specifications from LiteLLM
        enriched_model: Enriched model data

    Returns:
        Dictionary with context_window, max_output, source, and extended_* fields
    """
    context_window = None
    max_output = None
    source = None
    extended_context = None
    extended_context_beta = None
    extended_output = None
    extended_output_beta = None

    existing_converse = enriched_model.get(
        "converse_data", model.get("converse_data", {})
    )

    # --- TIER 1: Console API metadata ---
    console_meta = model.get("console_metadata", {})
    if console_meta:
        api_context = console_meta.get("max_context_window")
        if api_context and isinstance(api_context, (int, float)):
            context_window = int(api_context)
            source = "bedrock_console_api"
        api_output = console_meta.get("max_output_tokens")
        if api_output and isinstance(api_output, (int, float)):
            max_output = int(api_output)

    # --- TIER 2: Model ID size variant ---
    if context_window is None:
        variant_cw = model.get("variant_context_window")
        if variant_cw and isinstance(variant_cw, (int, float)):
            context_window = int(variant_cw)
            source = "model_id_variant"

    # --- TIER 3: profiler-config.json ---
    config_specs = get_context_window_from_config(model_id)
    if config_specs:
        config_standard = config_specs.get("standard_context")
        config_extended = config_specs.get("extended_context")

        # If API returned the extended value as context_window, prefer config's standard
        if config_standard and config_extended and context_window == config_extended:
            context_window = config_standard
            source = config_specs.get("source", "config")

        # Use standard_context only if Tiers 1-2 didn't provide context_window
        if context_window is None:
            context_window = config_standard
            source = config_specs.get("source", "config")

        # Use max_output from config if not yet set
        if max_output is None:
            max_output = config_specs.get("max_output")

        # Extended fields: ALWAYS apply from config regardless of tier
        extended_context = config_extended
        extended_context_beta = config_specs.get("extended_context_beta")
        extended_output = config_specs.get("extended_output")
        extended_output_beta = config_specs.get("extended_output_beta")

    # --- TIER 4: LiteLLM token_specs ---
    if context_window is None:
        context_window = token_specs.get("context_window")
    if max_output is None:
        max_output = token_specs.get("max_output_tokens")
    if source is None and token_specs.get("source"):
        source = token_specs.get("source")

    # --- Fallback: existing converse_data ---
    if context_window is None:
        context_window = existing_converse.get("context_window")
    if max_output is None:
        max_output = existing_converse.get("max_output_tokens")
    if source is None:
        source = existing_converse.get("source")

    return {
        "context_window": context_window,
        "max_output": max_output,
        "source": source,
        "extended_context": extended_context,
        "extended_context_beta": extended_context_beta,
        "extended_output": extended_output,
        "extended_output_beta": extended_output_beta,
        "litellm_verified": token_specs.get(
            "litellm_verified", existing_converse.get("litellm_verified", False)
        ),
    }


@tracer.capture_method
def _build_converse_data(
    context_data: dict,
    capabilities: list,
    use_cases: list,
    regional_availability: list,
) -> dict:
    """Build the converse_data structure.

    Args:
        context_data: Output from _resolve_context_window()
        capabilities: Model capabilities list
        use_cases: Model use cases list
        regional_availability: List of available regions

    Returns:
        Complete converse_data dictionary
    """
    context_window = context_data["context_window"]

    converse_data = {
        "context_window": context_window,
        "max_output_tokens": context_data["max_output"],
        "size_category": get_size_category(context_window),
        "verified": context_data["source"] is not None
        and context_data["source"] != "unknown",
        "source": context_data["source"] or "unknown",
        "litellm_verified": context_data["litellm_verified"],
        "capabilities_count": len(capabilities),
        "use_cases_count": len(use_cases),
        "regions_count": len(regional_availability),
    }

    # Add extended context info if available
    if context_data["extended_context"]:
        converse_data["extended_context"] = context_data["extended_context"]
        converse_data["has_extended_context"] = True
        if context_data["extended_context_beta"]:
            converse_data["extended_context_beta"] = context_data[
                "extended_context_beta"
            ]
    else:
        converse_data["has_extended_context"] = False

    # Add extended output info if available
    if context_data["extended_output"]:
        converse_data["extended_output"] = context_data["extended_output"]
        if context_data["extended_output_beta"]:
            converse_data["extended_output_beta"] = context_data["extended_output_beta"]

    return converse_data


@tracer.capture_method
def _merge_lifecycle_data(
    model_lifecycle: dict,
    regional_lifecycle: dict,
    lifecycle_by_model: dict,
    model_id: str,
    model_name: str,
) -> dict:
    """Merge regional lifecycle data into model lifecycle.

    Args:
        model_lifecycle: Base lifecycle from model data
        regional_lifecycle: Regional lifecycle data from lifecycle-collector
        lifecycle_by_model: Legacy lifecycle data by model ID
        model_id: Model identifier
        model_name: Model name (fallback for matching)

    Returns:
        Merged lifecycle dictionary
    """
    # Start with base lifecycle
    result = (
        model_lifecycle.copy()
        if model_lifecycle
        else {"status": "ACTIVE", "release_date": ""}
    )

    # Try to get regional data by model_id first, then by model_name
    regional_data = (regional_lifecycle or {}).get(model_id, {})
    if not regional_data:
        regional_data = (regional_lifecycle or {}).get(model_name, {})

    if regional_data:
        # Build regional lifecycle structure
        regional_status = regional_data.get("regional_status", {})
        status_summary = regional_data.get("status_summary", {})

        # Determine primary status (most restrictive: EOL > LEGACY > ACTIVE)
        if status_summary.get("EOL"):
            primary_status = "EOL"
        elif status_summary.get("LEGACY"):
            primary_status = "LEGACY"
        else:
            primary_status = "ACTIVE"

        # Determine global status
        statuses_present = [s for s, regions in status_summary.items() if regions]
        if len(statuses_present) > 1:
            global_status = "MIXED"
        elif statuses_present:
            global_status = statuses_present[0]
        else:
            global_status = result.get("status", "ACTIVE")

        # Build the new model_lifecycle structure
        result = {
            "status": primary_status,  # Backward compatible - single status
            "global_status": global_status,
            "primary_status": primary_status,
            "regional_status": regional_status,
            "status_summary": status_summary,
            "release_date": result.get("release_date", ""),
        }

        # Add recommended replacement info
        if regional_data.get("recommended_replacement"):
            result["recommended_replacement"] = regional_data["recommended_replacement"]
        if regional_data.get("recommended_model_id"):
            result["recommended_model_id"] = regional_data["recommended_model_id"]

        # Add dates from the first LEGACY or EOL region (for backward compatibility)
        for region, status_data in regional_status.items():
            if status_data.get("status") in ["LEGACY", "EOL"]:
                if status_data.get("legacy_date"):
                    result["legacy_date"] = status_data["legacy_date"]
                if status_data.get("eol_date"):
                    result["eol_date"] = status_data["eol_date"]
                if status_data.get("extended_access_date"):
                    result["extended_access_date"] = status_data["extended_access_date"]
                break

        # Add launch_date from first ACTIVE region (for backward compatibility)
        for region, status_data in regional_status.items():
            if status_data.get("status") == "ACTIVE" and status_data.get("launch_date"):
                result["launch_date"] = status_data["launch_date"]
                break

    # Fallback to old models_by_id if no regional data (backward compatibility)
    elif lifecycle_by_model:
        lifecycle_info = lifecycle_by_model.get(model_id, {})
        if lifecycle_info:
            # Override status from scraped data (active, legacy, eol)
            scraped_status = lifecycle_info.get("lifecycle_status")
            if scraped_status:
                result["status"] = scraped_status.upper()
            # Add EOL date if available
            eol_date = lifecycle_info.get("eol_date")
            if eol_date:
                result["eol_date"] = eol_date
            # Add legacy date if available
            legacy_date = lifecycle_info.get("legacy_date")
            if legacy_date:
                result["legacy_date"] = legacy_date
            # Add recommended replacement if available
            recommended_replacement = lifecycle_info.get("recommended_replacement")
            if recommended_replacement:
                result["recommended_replacement"] = recommended_replacement
            # Add recommended model ID if available
            recommended_model_id = lifecycle_info.get("model_id")
            if recommended_model_id and recommended_model_id != model_id:
                result["recommended_model_id"] = recommended_model_id

    return result


@tracer.capture_method
def _build_model_pricing(
    model: dict,
    pricing_data: dict,
    collection_timestamp: str,
    regional_availability: list,
) -> tuple[dict, dict, bool, Optional[dict]]:
    """Build the model_pricing structure and batch inference data.

    Args:
        model: Model data with pricing info
        pricing_data: Full pricing data for batch inference check
        collection_timestamp: Collection timestamp
        regional_availability: List of available regions

    Returns:
        Tuple of (model_pricing dict, batch_inference dict)
    """
    # Get model pricing from upstream (already in snake_case)
    model_pricing_data = model.get("model_pricing", {})
    has_pricing = model_pricing_data.get(
        "is_pricing_available", model.get("has_pricing", False)
    )
    pricing_ref_id = model_pricing_data.get("pricing_reference_id", "")

    # Use upstream pricing_file_reference if available (from pricing-linker)
    upstream_pricing_ref = model_pricing_data.get("pricing_file_reference")

    # Check batch inference support
    batch_inference = check_batch_inference(
        model.get("model_id", ""),
        pricing_data,
        upstream_pricing_ref,
        regional_availability,
    )

    # Calculate coverage percentage
    if batch_inference.get("supported") and regional_availability:
        batch_regs = len(batch_inference.get("supported_regions", []))
        total_regs = len(regional_availability)
        batch_inference["coverage_percentage"] = (
            round(min(batch_regs / total_regs * 100, 100.0), 1)
            if total_regs > 0
            else 0.0
        )

    # Determine pricing provider and model key
    if upstream_pricing_ref and isinstance(upstream_pricing_ref, dict):
        pricing_provider = upstream_pricing_ref.get(
            "provider", model.get("model_provider", "")
        )
        pricing_model_key = upstream_pricing_ref.get(
            "model_key", pricing_ref_id or model.get("model_id", "")
        )
    else:
        pricing_provider = model.get("model_provider", "")
        pricing_model_key = (
            pricing_ref_id if pricing_ref_id else model.get("model_id", "")
        )

    model_pricing = {
        "is_pricing_available": has_pricing,
        "pricing_reference_id": pricing_ref_id or model.get("model_id", ""),
        "pricing_file_reference": {
            "provider": pricing_provider,
            "model_key": pricing_model_key,
            "model_name": model.get("model_name", ""),
        },
        "pricing_summary": {
            "integration_source": "amazon-bedrock-pricing-collector",
            "has_pricing_data": has_pricing,
            "integration_timestamp": collection_timestamp,
            "reference_based": True,
        },
        # Preserve regions data from pricing-linker (contains pricing_groups per region)
        "regions": model_pricing_data.get("regions", {}),
        "total_regions": model_pricing_data.get("total_regions", 0),
        "confidence": model_pricing_data.get("confidence", 0),
    }

    return model_pricing, batch_inference, has_pricing, upstream_pricing_ref


@tracer.capture_method
def _build_collection_metadata(
    model: dict,
    regional_availability: list,
    collection_timestamp: str,
) -> dict:
    """Build collection metadata structure.

    Args:
        model: Model data
        regional_availability: List of available regions
        collection_timestamp: Collection timestamp

    Returns:
        collection_metadata dictionary
    """
    existing_metadata = model.get("collection_metadata", {})
    return {
        "first_discovered_at": existing_metadata.get(
            "first_discovered_at", collection_timestamp
        ),
        "first_discovered_in_region": existing_metadata.get(
            "first_discovered_in_region",
            regional_availability[0] if regional_availability else "unknown",
        ),
        "api_source": existing_metadata.get("api_source", "list_foundation_models"),
        "dual_region_collection": existing_metadata.get("dual_region_collection", True),
        "regions_collected_from": existing_metadata.get("regions_collected_from", []),
        "phase2_regional_discovery": True,
        "regional_data_source": "api_discovery",
        # extraction_regions: where model was found in ListFoundationModels API (audit/debug only)
        "extraction_regions": model.get("extraction_regions", []),
    }


@tracer.capture_method
def transform_model_to_schema(
    model_id: str,
    model: dict,
    regional_availability: list,
    token_specs: dict,
    quotas_by_region: dict,
    features_by_region: dict,
    enriched_model: dict,
    pricing_data: dict,
    collection_timestamp: str,
    mantle_by_model: dict,
    provisioned_throughput: dict = None,
    lifecycle_by_model: dict = None,
    regional_lifecycle: dict = None,
    govcloud_availability: dict = None,
) -> dict:
    """Merge model data from all sources into final schema.

    Orchestrates sub-functions to build the complete model structure.
    Input model data is already in snake_case from upstream Lambdas.
    """
    # Get enriched data (already in snake_case)
    capabilities = enriched_model.get(
        "model_capabilities", model.get("model_capabilities", [])
    )
    use_cases = enriched_model.get("model_use_cases", model.get("model_use_cases", []))
    doc_links = enriched_model.get(
        "documentation_links", model.get("documentation_links", {})
    )

    # Extract console metadata fields
    console_meta = model.get("console_metadata", {})
    console_languages = console_meta.get("languages", []) if console_meta else []
    console_use_cases = console_meta.get("use_cases", []) if console_meta else []
    console_description = console_meta.get("description", "") if console_meta else ""
    console_short_description = (
        console_meta.get("short_description", "") if console_meta else ""
    )

    # Resolve context window (4-tier priority)
    context_data = _resolve_context_window(model_id, model, token_specs, enriched_model)

    # Build converse data
    converse_data = _build_converse_data(
        context_data, capabilities, use_cases, regional_availability
    )

    # Build cross-region inference
    cross_region = build_cross_region_inference(model_id, features_by_region)

    # Build Mantle inference (with pricing data to determine has_pricing)
    mantle = build_mantle_inference(model_id, mantle_by_model, pricing_data)

    # Build model quotas (using snake_case model_name)
    model_quotas = build_model_quotas(
        model_id,
        model.get("model_name", ""),
        quotas_by_region,
        model_provider=model.get("model_provider", ""),
    )

    # Build model pricing and batch inference data
    # Add model_id to model dict for _build_model_pricing
    model_with_id = {**model, "model_id": model_id}
    model_pricing, batch_inference, has_pricing, upstream_pricing_ref = (
        _build_model_pricing(
            model_with_id, pricing_data, collection_timestamp, regional_availability
        )
    )

    # Build documentation links (pass through all from enricher, with defaults from config)
    config = get_config_loader()
    documentation_links = doc_links.copy() if doc_links else {}
    if "aws_bedrock_guide" not in documentation_links:
        documentation_links["aws_bedrock_guide"] = config.get_documentation_url(
            "bedrock_model_ids"
        )
    if "pricing_guide" not in documentation_links:
        documentation_links["pricing_guide"] = config.get_documentation_url(
            "bedrock_pricing"
        )

    # Get modalities (already in snake_case nested structure)
    model_modalities = model.get("model_modalities", {})
    if not model_modalities:
        model_modalities = {
            "input_modalities": model.get("input_modalities", []),
            "output_modalities": model.get("output_modalities", []),
        }

    # Build collection metadata
    collection_metadata = _build_collection_metadata(
        model, regional_availability, collection_timestamp
    )

    # Merge lifecycle data
    base_lifecycle = model.get("model_lifecycle", {})
    if not base_lifecycle:
        base_lifecycle = {"status": "ACTIVE", "release_date": ""}
    model_lifecycle = _merge_lifecycle_data(
        base_lifecycle,
        regional_lifecycle or {},
        lifecycle_by_model or {},
        model_id,
        model.get("model_name", ""),
    )

    # Get customization (already in snake_case)
    customization = model.get("customization", {})
    if not customization:
        customization = {
            "customization_supported": model.get("customization_supported", []),
            "customization_options": {},
        }

    # Compute consumption options and provisioned throughput independently
    consumption_options = get_consumption_options(
        model.get("inference_types_supported", []),
        pricing_data,
        upstream_pricing_ref,
        mantle_supported=mantle["supported"],
    )
    resolved_provisioned = (
        provisioned_throughput
        if provisioned_throughput
        else {
            "supported": False,
            "provisioned_regions": [],
        }
    )

    # Build reserved capacity data
    reserved_capacity = build_reserved_capacity(
        model_id, pricing_data, upstream_pricing_ref
    )

    # Reconcile consumption_options with actual provisioned throughput data
    if resolved_provisioned.get("supported"):
        if "provisioned_throughput" not in consumption_options:
            consumption_options.append("provisioned_throughput")
    else:
        if "provisioned_throughput" in consumption_options:
            consumption_options.remove("provisioned_throughput")

    # Reconcile consumption_options with actual cross-region inference data
    if cross_region.get("supported"):
        if "cross_region_inference" not in consumption_options:
            consumption_options.append("cross_region_inference")
    else:
        if "cross_region_inference" in consumption_options:
            consumption_options.remove("cross_region_inference")

    # Reconcile consumption_options with actual batch inference data
    if batch_inference.get("supported"):
        if "batch" not in consumption_options:
            consumption_options.append("batch")
    else:
        if "batch" in consumption_options:
            consumption_options.remove("batch")

    # Reconcile consumption_options with actual mantle inference data
    if mantle.get("supported"):
        if "mantle" not in consumption_options:
            consumption_options.append("mantle")
    else:
        if "mantle" in consumption_options:
            consumption_options.remove("mantle")

    # Reconcile consumption_options with actual reserved capacity data
    if reserved_capacity.get("supported"):
        if "reserved" not in consumption_options:
            consumption_options.append("reserved")
    else:
        if "reserved" in consumption_options:
            consumption_options.remove("reserved")

    # Get feature support and chat features from console metadata
    feature_support = console_meta.get("feature_support", {}) if console_meta else {}
    chat_features = console_meta.get("chat_features", {}) if console_meta else {}

    # Build unified API support and endpoint availability
    api_support_model = {
        "chat_features": chat_features,
        "streaming_supported": model.get("streaming_supported", False),
    }
    api_support = build_api_support(api_support_model, mantle, is_mantle_only=False)

    # Build GovCloud availability
    govcloud_data = build_govcloud_availability(
        model_id,
        model.get("model_name", ""),
        govcloud_availability or {},
    )

    # Build consolidated availability object (Phase 3 - new structure only)
    availability = build_availability(
        regional_availability=regional_availability,
        cross_region_data=cross_region,
        batch_inference_data=batch_inference,
        provisioned_data=resolved_provisioned,
        mantle_data=mantle,
        govcloud_data=govcloud_data,
        is_mantle_only=False,
        reserved_data=reserved_capacity,
    )

    # Backfill availability regions from pricing data when API data is missing
    # This handles models that have pricing but aren't returned by ListFoundationModels
    availability = backfill_availability_from_pricing(
        availability=availability,
        consumption_options=consumption_options,
        pricing_data=pricing_data,
        pricing_ref=upstream_pricing_ref,
        model_id=model_id,
    )

    # Determine visibility (config-driven hidden models list)
    hidden_models = config.config.get("model_configuration", {}).get(
        "hidden_models", []
    )
    show_model = model_id not in hidden_models

    # Build the result with new field names only (Phase 3 - old fields removed)
    result = {
        # Visibility flag (frontend uses this to filter display)
        "show_model": show_model,
        # Core identifiers (kept)
        "model_id": model_id,
        "model_arn": model.get("model_arn", ""),
        "model_name": model.get("model_name", ""),
        "model_provider": model.get("model_provider", ""),
        # Primary regional data - use backfilled availability regions if available
        "in_region": availability.get("on_demand", {}).get("regions", []),
        # Model configuration (kept)
        "customization": customization,
        "inference_types_supported": model.get("inference_types_supported", []),
        # Descriptions (kept)
        "description": console_description,
        "short_description": console_short_description,
        # Chat features (kept - used by api_support builder)
        "chat_features": chat_features,
        # Consumption options (kept)
        "consumption_options": consumption_options,
        # Collection metadata (kept)
        "collection_metadata": collection_metadata,
        # Boolean flags (kept - useful for filtering)
        "has_pricing": has_pricing,
        "has_quotas": len(model_quotas) > 0,
        # NEW consolidated fields (Phase 3 - these replace the old fields)
        "availability": availability,
        "modalities": model_modalities,
        "capabilities": capabilities,
        "use_cases": console_use_cases,
        "lifecycle": model_lifecycle,
        "streaming": model.get("streaming_supported", False),
        "languages": console_languages,
        "docs": documentation_links,
        "features": feature_support,
        "specs": build_specs(converse_data),
        "pricing": build_pricing_alias(model_pricing),
        "model_pricing": model_pricing,
        "quotas": model_quotas,
        "api": api_support,
    }

    return result


def find_matching_availability(
    model_id: str,
    model_availability: dict,
    inference_types: list = None,
) -> list:
    """
    Find regional availability for a model, handling ID format differences.

    Model IDs from Bedrock API: anthropic.claude-3-5-sonnet-20241022-v2:0
    Model IDs from Pricing API: anthropic.claude-3-sonnet

    IMPORTANT: If model only supports PROVISIONED inference, don't inherit
    on-demand regions from base model. This prevents provisioned-only models
    like cohere.embed-english-v3:0:512 from incorrectly inheriting regions
    from their base model cohere.embed-english-v3:0.

    Args:
        model_id: The model identifier to look up
        model_availability: Dict mapping model IDs to lists of available regions
        inference_types: List of inference types the model supports (e.g., ["ON_DEMAND", "PROVISIONED"])

    Strategy: Try exact match first, then find the best (longest) match.
    """
    # Check if this is a provisioned-only model using centralized utility
    variant_info = get_model_variant_info(model_id)
    is_provisioned_only = variant_info.get("is_provisioned_only", False) or (
        inference_types is not None and inference_types == ["PROVISIONED"]
    )

    # If provisioned-only, only return exact match - don't inherit from base model
    if is_provisioned_only:
        if model_id in model_availability:
            return model_availability[model_id]
        # For provisioned-only models, don't do fuzzy matching to base model
        # as that would incorrectly inherit on-demand regions
        return []

    # Try exact match first
    if model_id in model_availability:
        return model_availability[model_id]

    # Normalize model_id for matching (remove version suffix like :0, :18k, etc.)
    base_model_id = model_id.split(":")[0] if ":" in model_id else model_id

    # Try matching without version suffix
    if base_model_id in model_availability:
        return model_availability[base_model_id]

    # Find the best (longest) matching pricing key
    # This prevents "claude-3-sonnet" from incorrectly matching "claude-3-5-sonnet-xxx"
    model_id_lower = model_id.lower()
    best_match_key = None
    best_match_length = 0

    for pricing_key in model_availability.keys():
        pricing_key_lower = pricing_key.lower()

        # Check if pricing key is contained in model_id or model_id starts with pricing key
        if pricing_key_lower in model_id_lower or model_id_lower.startswith(
            pricing_key_lower
        ):
            if len(pricing_key) > best_match_length:
                best_match_key = pricing_key
                best_match_length = len(pricing_key)
            continue

        # Also check by removing common prefixes/suffixes and comparing core name
        pricing_parts = (
            pricing_key_lower.replace("anthropic.", "")
            .replace("amazon.", "")
            .replace("meta.", "")
            .replace("mistral.", "")
            .replace("cohere.", "")
            .replace("ai21.", "")
            .replace("stability.", "")
            .replace("nvidia.", "")
            .replace("luma.", "")
        )
        model_parts = (
            model_id_lower.replace("anthropic.", "")
            .replace("amazon.", "")
            .replace("meta.", "")
            .replace("mistral.", "")
            .replace("cohere.", "")
            .replace("ai21.", "")
            .replace("stability.", "")
            .replace("nvidia.", "")
            .replace("luma.", "")
        )

        # Check if core names overlap significantly
        if pricing_parts and model_parts:
            # Remove date/version suffixes from model_parts for comparison
            model_core = re.sub(r"-\d{8}-v\d+.*$", "", model_parts)
            if (
                pricing_parts == model_core
                or pricing_parts in model_core
                or model_core.startswith(pricing_parts)
            ):
                if len(pricing_key) > best_match_length:
                    best_match_key = pricing_key
                    best_match_length = len(pricing_key)

    if best_match_key:
        return model_availability[best_match_key]

    return []


@tracer.capture_method
def build_final_models(
    models_with_pricing: dict,
    regional_availability: dict,
    token_specs: dict,
    quotas_by_region: dict,
    features_by_region: dict,
    enriched_models: dict,
    pricing_data: dict,
    collection_timestamp: str,
    mantle_by_model: dict,
    lifecycle_by_model: dict,
    regional_lifecycle: dict = None,
) -> dict:
    """Build the final comprehensive models structure in expected schema with tracing.

    Also creates stub entries for Mantle-only models (models that exist in the
    Mantle API but not in Bedrock's ListFoundationModels).
    """
    logger.info("Building final models")
    providers = models_with_pricing.get("providers", {})
    enriched_providers = enriched_models.get("providers", {})
    # Upstream uses snake_case: model_availability
    model_availability = regional_availability.get("model_availability", {})
    provisioned_availability_data = regional_availability.get(
        "provisioned_availability", {}
    )
    # Upstream uses snake_case: token_specs
    token_specs_data = token_specs.get("token_specs", {})

    # Extract GovCloud availability from pricing data
    govcloud_availability = pricing_data.get("govcloud_availability", {})

    result_providers = {}
    # Track which Mantle model IDs have been matched to Bedrock models
    matched_mantle_ids = set()

    for provider, provider_data in providers.items():
        result_providers[provider] = {"models": {}}

        for model_id, model in provider_data.get("models", {}).items():
            # Get inference types for this model (needed for availability matching)
            inference_types = model.get("inference_types_supported", [])

            # Get regional availability for this model (with fuzzy matching)
            # Pass inference_types to prevent provisioned-only models from
            # inheriting on-demand regions from base models
            regions = find_matching_availability(
                model_id, model_availability, inference_types
            )

            # Get token specs for this model
            specs = token_specs_data.get(model_id, {})

            # Get enriched data for this model
            enriched = (
                enriched_providers.get(provider, {}).get("models", {}).get(model_id, {})
            )

            # Build provisioned throughput data
            provisioned = build_provisioned_throughput(
                model_id, provisioned_availability_data
            )

            # Build Mantle inference data (needed for tracking matched IDs)
            mantle = build_mantle_inference(model_id, mantle_by_model, pricing_data)

            # Track matched Mantle model ID
            if mantle.get("matched_mantle_id"):
                matched_mantle_ids.add(mantle["matched_mantle_id"])

            # Transform to expected schema
            transformed = transform_model_to_schema(
                model_id=model_id,
                model=model,
                regional_availability=regions,
                token_specs=specs,
                quotas_by_region=quotas_by_region,
                features_by_region=features_by_region,
                enriched_model=enriched,
                pricing_data=pricing_data,
                collection_timestamp=collection_timestamp,
                mantle_by_model=mantle_by_model,
                provisioned_throughput=provisioned,
                lifecycle_by_model=lifecycle_by_model,
                regional_lifecycle=regional_lifecycle,
                govcloud_availability=govcloud_availability,
            )

            result_providers[provider]["models"][model_id] = transformed

    # Identify unmatched Mantle models and create stubs for them
    all_mantle_ids = set(mantle_by_model.keys())
    unmatched_mantle_ids = all_mantle_ids - matched_mantle_ids

    if unmatched_mantle_ids:
        logger.info(
            f"Found {len(unmatched_mantle_ids)} Mantle-only models (not in Bedrock): "
            f"{sorted(unmatched_mantle_ids)}"
        )

        for mantle_id in unmatched_mantle_ids:
            mantle_info = mantle_by_model.get(mantle_id, {})
            regions = mantle_info.get("regions", [])
            supports_responses_api = mantle_info.get("supports_responses_api", False)
            stub = create_mantle_only_stub(
                mantle_id, regions, collection_timestamp, supports_responses_api
            )

            # Try to enrich with pricing data
            pricing_ref = _find_pricing_for_mantle_stub(
                mantle_id, stub["model_name"], pricing_data
            )
            if pricing_ref:
                stub["has_pricing"] = True
                # Update the pricing alias
                stub["pricing"] = {
                    "available": True,
                    "reference": pricing_ref,
                }
                # Update availability.mantle.has_pricing
                stub["availability"]["mantle"]["has_pricing"] = True
                batch = check_batch_inference(mantle_id, pricing_data, pricing_ref)
                if batch.get("supported"):
                    # Update availability.batch
                    stub["availability"]["batch"] = {
                        "supported": True,
                        "regions": batch.get("supported_regions", []),
                    }
                    if "batch" not in stub["consumption_options"]:
                        stub["consumption_options"].append("batch")
                reserved = build_reserved_capacity(mantle_id, pricing_data, pricing_ref)
                if reserved.get("supported"):
                    stub["availability"]["reserved"] = {
                        "supported": True,
                        "regions": reserved.get("regions", []),
                        "commitments": reserved.get("commitments", []),
                    }
                    if "reserved" not in stub["consumption_options"]:
                        stub["consumption_options"].append("reserved")
                logger.info(
                    f"Enriched Mantle-only model {mantle_id} with pricing from "
                    f"{pricing_ref['provider']}/{pricing_ref['model_key']}"
                )

            # Determine provider from the stub
            provider_name = stub.get("model_provider", "Unknown")

            # Ensure provider exists in result_providers
            if provider_name not in result_providers:
                result_providers[provider_name] = {"models": {}}

            # Add the stub model (already has all new fields from create_mantle_only_stub)
            result_providers[provider_name]["models"][mantle_id] = stub
            logger.debug(
                f"Created Mantle-only stub for {mantle_id} under provider {provider_name}"
            )

    # Merge duplicate models by normalized name
    result_providers = merge_duplicate_models(result_providers)

    return result_providers


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    """
    Lambda handler for final aggregation.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "pricingS3Key": "...",
            "modelsS3Key": "...",
            "quotaResults": [...],
            "pricingLinked": {...},
            "regionalAvailability": {...},
            "featureResults": [...],
            "tokenSpecs": {...},
            "enrichedModels": {...}
        }

    Output:
        {
            "status": "SUCCESS",
            "modelsS3Key": "executions/{id}/final/bedrock_models.json",
            "pricingS3Key": "executions/{id}/final/bedrock_pricing.json",
            "totalModels": 108,
            "totalProviders": 17
        }
    """
    logger.info("Starting final aggregation")
    start_time = time.time()
    collection_timestamp = time.strftime(
        "%Y-%m-%dT%H:%M:%S.000000+00:00", time.gmtime()
    )

    # Reset quota index cache for each invocation (Lambda containers may be reused)
    global _quota_index
    _quota_index = None

    # Validate required parameters
    try:
        validate_required_params(event, ["s3Bucket", "executionId"], "FinalAggregator")
    except ValidationError as e:
        return {
            "status": "FAILED",
            "errorType": "ValidationError",
            "errorMessage": str(e),
        }

    s3_bucket = event["s3Bucket"]
    execution_id = parse_execution_id(event["executionId"])
    pricing_s3_key = event.get("pricingS3Key")
    quota_results = event.get("quotaResults", [])
    pricing_linked = event.get("pricingLinked", {})
    regional_availability = event.get("regionalAvailability", {})
    feature_results = event.get("featureResults", [])
    token_specs_result = event.get("tokenSpecs", {})
    enriched_models_result = event.get("enrichedModels", {})
    mantle_results = event.get("mantleResults", [])
    lifecycle_data_result = event.get("lifecycleData", {})
    dry_run = event.get("dryRun", False)

    models_output_key = f"executions/{execution_id}/final/bedrock_models.json"
    pricing_output_key = f"executions/{execution_id}/final/bedrock_pricing.json"

    logger.info(
        "Building final aggregated output", extra={"execution_id": execution_id}
    )

    try:
        s3_client = get_s3_client()

        if not dry_run:
            # Read intermediate data
            models_with_pricing_key = pricing_linked.get("s3Key")
            models_with_pricing = (
                read_from_s3(s3_client, s3_bucket, models_with_pricing_key)
                if models_with_pricing_key
                else {}
            )

            availability_key = regional_availability.get("s3Key")
            availability_data = (
                read_from_s3(s3_client, s3_bucket, availability_key)
                if availability_key
                else {}
            )

            token_specs_key = token_specs_result.get("s3Key")
            token_specs_data = (
                read_from_s3(s3_client, s3_bucket, token_specs_key)
                if token_specs_key
                else {}
            )

            pricing_data = (
                read_from_s3(s3_client, s3_bucket, pricing_s3_key)
                if pricing_s3_key
                else {}
            )

            enriched_models_key = enriched_models_result.get("s3Key")
            enriched_models_data = (
                read_from_s3(s3_client, s3_bucket, enriched_models_key)
                if enriched_models_key
                else {}
            )

            # Read lifecycle data
            lifecycle_s3_key = lifecycle_data_result.get("s3Key")
            lifecycle_data = (
                read_from_s3(s3_client, s3_bucket, lifecycle_s3_key)
                if lifecycle_s3_key
                else {"models_by_id": {}, "regional_lifecycle": {}}
            )
            lifecycle_by_model = lifecycle_data.get("models_by_id", {})
            regional_lifecycle = lifecycle_data.get("regional_lifecycle", {})

            # Aggregate quotas, features, and mantle data
            quotas_by_region = aggregate_quotas(quota_results, s3_client, s3_bucket)
            features_by_region = aggregate_features(
                feature_results, s3_client, s3_bucket
            )
            mantle_by_model = aggregate_mantle(mantle_results, s3_client, s3_bucket)

            # Build final models in expected schema
            final_providers = build_final_models(
                models_with_pricing,
                availability_data,
                token_specs_data,
                quotas_by_region,
                features_by_region,
                enriched_models_data,
                pricing_data,
                collection_timestamp,
                mantle_by_model,
                lifecycle_by_model,
                regional_lifecycle,
            )

            # Calculate statistics
            total_models = sum(
                len(p.get("models", {})) for p in final_providers.values()
            )
            total_providers = len(final_providers)
            total_regions = len(availability_data.get("regions", {}))

            # Count models with pricing and quotas
            models_with_pricing_count = sum(
                1
                for p in final_providers.values()
                for m in p.get("models", {}).values()
                if m.get("has_pricing", False)
            )
            models_with_quotas_count = sum(
                1
                for p in final_providers.values()
                for m in p.get("models", {}).values()
                if m.get("has_quotas", False)
            )
            total_quotas = sum(
                len(quotas)
                for region_quotas in quotas_by_region.values()
                for quotas in (
                    [region_quotas]
                    if isinstance(region_quotas, list)
                    else region_quotas.values()
                )
            )

            # Build final models output in expected schema
            models_output = {
                "metadata": {
                    "collection_timestamp": collection_timestamp,
                    "providers_count": total_providers,
                    "total_models": total_models,
                    "models_with_pricing": models_with_pricing_count,
                    "models_with_quotas": models_with_quotas_count,
                    "regions_covered": total_regions,
                    "total_quotas_available": total_quotas,
                    "collection_method": "comprehensive_structure_with_quota_assignment",
                },
                "providers": final_providers,
            }

            # Write models output
            write_to_s3(s3_client, s3_bucket, models_output_key, models_output)

            # Copy pricing data as-is (pricing-aggregator already formats it)
            # The pricing schema transformation happens in pricing-aggregator
            write_to_s3(s3_client, s3_bucket, pricing_output_key, pricing_data)

        else:
            logger.info("Dry run - skipping final aggregation")
            total_models = 0
            total_providers = 0
            total_regions = 0
            models_with_pricing_count = 0

        duration_ms = int((time.time() - start_time) * 1000)

        # Emit metrics
        metrics.add_metric(
            name="TotalModels", unit=MetricUnit.Count, value=total_models
        )
        metrics.add_metric(
            name="TotalProviders", unit=MetricUnit.Count, value=total_providers
        )
        metrics.add_metric(
            name="ModelsWithPricing",
            unit=MetricUnit.Count,
            value=models_with_pricing_count,
        )
        metrics.add_metric(
            name="DurationMs", unit=MetricUnit.Milliseconds, value=duration_ms
        )

        logger.info(
            "Final aggregation complete",
            extra={
                "total_models": total_models,
                "total_providers": total_providers,
                "models_with_pricing": models_with_pricing_count,
                "duration_ms": duration_ms,
            },
        )

        return {
            "status": "SUCCESS",
            "modelsS3Key": models_output_key,
            "pricingS3Key": pricing_output_key,
            "totalModels": total_models,
            "totalProviders": total_providers,
            "totalRegions": total_regions,
            "durationMs": duration_ms,
        }

    except Exception as e:
        logger.exception("Failed to aggregate", extra={"error_type": type(e).__name__})
        return {
            "status": "FAILED",
            "errorType": type(e).__name__,
            "errorMessage": str(e),
        }


# Force rebuild 1772624008
