"""
Centralized model ID matching utility for the Bedrock Model Profiler.

This module provides functions to normalize, compare, and match model IDs across
different data sources (Bedrock API, Pricing API, Mantle API) that use different
ID formats.

ID Format Examples by Source:
    - Bedrock API: `deepseek.v3-v1:0` (API version suffix)
    - Bedrock API: `deepseek.v3.2:0` (semantic version with API suffix)
    - Pricing API: `deepseek.deepseek-v3-1` (redundant provider prefix)
    - Pricing API: `deepseek.r1` (short form)
    - Mantle API: `deepseek.v3.1` (semantic version only)

Design Principles:
    - Pure functions, no side effects
    - Comprehensive type hints
    - Designed for testability
    - Handle edge cases gracefully

Usage:
    from shared.model_matcher import (
        get_canonical_model_id,
        find_best_match,
        get_model_variant_info,
        calculate_match_score,
        has_semantic_conflict,
    )

    # Normalize IDs for comparison
    canonical = get_canonical_model_id("deepseek.v3-v1:0")  # "deepseek.v3.1"

    # Find best match from candidates
    match, score = find_best_match("deepseek.v3-v1:0", pricing_models)

    # Get variant information
    info = get_model_variant_info("cohere.embed-english-v3:0:512")
"""

from __future__ import annotations

import re
from typing import Any, TypedDict


# =============================================================================
# Type Definitions
# =============================================================================


class ModelVariantInfo(TypedDict):
    """Information about a model variant extracted from its ID."""

    base_id: str
    is_multimodal: bool
    is_provisioned_only: bool
    context_window: int | None
    version: str | None
    api_version: int | None
    has_dimension_suffix: bool


# =============================================================================
# Constants
# =============================================================================

# Suffixes that indicate API versions (not model versions)
API_VERSION_SUFFIXES = ("-v1", "-v2", "-v3", "-v4", "-v5")

# Suffixes that indicate model instance versions
INSTANCE_VERSION_PATTERN = re.compile(r":(\d+)$")

# Context window suffix pattern (e.g., :18k, :200k, :512)
CONTEXT_WINDOW_PATTERN = re.compile(r":(\d+)k?$")

# Dimension suffix pattern for provisioned models (e.g., :0:512, :0:1024)
DIMENSION_SUFFIX_PATTERN = re.compile(r":(\d+):(\d+)$")

# Multimodal suffix
MULTIMODAL_SUFFIX = ":mm"

# Semantic version patterns
# Matches: v3, v3.1, v3.2, V3-1, v3-1, etc.
SEMANTIC_VERSION_PATTERN = re.compile(
    r"[._-]?[vV]?(\d+)(?:[._-](\d+))?(?:[._-](\d+))?$"
)

# API version in model ID (e.g., -v1:0 where -v1 is API version)
API_VERSION_IN_ID_PATTERN = re.compile(r"-v(\d+)(?::\d+)?$")

# Redundant provider prefix pattern (e.g., deepseek.deepseek- -> deepseek.)
REDUNDANT_PROVIDER_PATTERN = re.compile(r"^([a-z]+)\.(\1)[-_]?", re.IGNORECASE)

# Model family identifiers that should not be confused
# Maps normalized family name to canonical form
MODEL_FAMILIES = {
    "v3": "v3",
    "r1": "r1",
    "sonnet": "sonnet",
    "opus": "opus",
    "haiku": "haiku",
    "claude3": "claude-3",
    "claude35": "claude-3.5",
    "claude3.5": "claude-3.5",
    "nova": "nova",
    "titan": "titan",
    "llama": "llama",
    "llama2": "llama-2",
    "llama3": "llama-3",
    "mistral": "mistral",
    "mixtral": "mixtral",
    "command": "command",
    "embed": "embed",
}

# Provider name mappings for normalization
PROVIDER_ALIASES = {
    "anthropic": "anthropic",
    "amazon": "amazon",
    "aws": "amazon",
    "meta": "meta",
    "cohere": "cohere",
    "ai21": "ai21",
    "stability": "stability",
    "mistral": "mistral",
    "deepseek": "deepseek",
    "qwen": "qwen",
    "alibaba": "qwen",
    # Additional providers for Mantle API compatibility
    "moonshotai": "moonshot",
    "moonshot": "moonshot",
    "moonshot-ai": "moonshot",  # Pricing API format
    "kimi-ai": "moonshot",
    "google": "google",
    "nvidia": "nvidia",
    "openai": "openai",
    "zai": "zai",
    "z-ai": "zai",  # Pricing API format
    "z.ai": "zai",  # Pricing API format with dot
    "minimax": "minimax",
    "luma": "luma",
    "writer": "writer",
    "twelvelabs": "twelvelabs",
}


# =============================================================================
# Core Functions
# =============================================================================


def get_canonical_model_id(model_id: str) -> str:
    """
    Normalize any model ID to a canonical form for comparison.

    This function handles various ID formats from different sources:
    - Bedrock API: `deepseek.v3-v1:0` -> `deepseek.v3.1`
    - Pricing API: `deepseek.deepseek-v3-1` -> `deepseek.v3.1`
    - Mantle API: `deepseek.v3.1` -> `deepseek.v3.1`
    - Mantle API: `moonshotai.kimi-k2` -> `moonshot.kimi-k2`

    Normalization steps:
    1. Convert to lowercase
    2. Normalize provider prefix using PROVIDER_ALIASES
    3. Remove instance version suffixes (:0, :1, :2)
    4. Remove context window suffixes (:18k, :200k)
    5. Remove dimension suffixes (:0:512)
    6. Remove multimodal suffix (:mm)
    7. Remove API version suffixes (-v1, -v2, -v3)
    8. Remove redundant provider prefixes (deepseek.deepseek- -> deepseek.)
    9. Normalize semantic versions (v3-1 -> v3.1, -v3-1 -> .v3.1)

    Args:
        model_id: The model identifier to normalize.

    Returns:
        Canonical form of the model ID.

    Examples:
        >>> get_canonical_model_id("deepseek.v3-v1:0")
        'deepseek.v3.1'
        >>> get_canonical_model_id("deepseek.deepseek-v3-1")
        'deepseek.v3.1'
        >>> get_canonical_model_id("deepseek.v3.1")
        'deepseek.v3.1'
        >>> get_canonical_model_id("anthropic.claude-3-5-sonnet-20240620-v1:0")
        'anthropic.claude-3-5-sonnet-20240620'
        >>> get_canonical_model_id("cohere.embed-english-v3:0:512")
        'cohere.embed-english-v3'
        >>> get_canonical_model_id("moonshotai.kimi-k2-thinking")
        'moonshot.kimi-k2-thinking'
    """
    if not model_id:
        return ""

    result = model_id.lower().strip()

    # Step 0: Normalize provider prefix using PROVIDER_ALIASES
    # This ensures moonshotai.model -> moonshot.model before other normalization
    if "." in result:
        # Handle special case: z.ai.model -> check if "z.ai" is a known provider
        parts = result.split(".")
        provider_part = parts[0]
        rest_parts = parts[1:]

        # Check if first two parts form a known provider (e.g., "z.ai")
        if len(parts) >= 2:
            two_part_provider = f"{parts[0]}.{parts[1]}"
            if two_part_provider in PROVIDER_ALIASES:
                provider_part = two_part_provider
                rest_parts = parts[2:]

        canonical_provider = PROVIDER_ALIASES.get(provider_part, provider_part)
        if canonical_provider != provider_part:
            result = canonical_provider + "." + ".".join(rest_parts)

    # Step 1: Remove dimension suffix first (e.g., :0:512 -> :0)
    # This must come before instance version removal
    dim_match = DIMENSION_SUFFIX_PATTERN.search(result)
    if dim_match:
        result = result[: dim_match.start()]

    # Step 2: Remove multimodal suffix
    if result.endswith(MULTIMODAL_SUFFIX.lower()):
        result = result[: -len(MULTIMODAL_SUFFIX)]

    # Step 3: Remove context window suffix (e.g., :18k, :200k)
    # But NOT simple numeric suffixes that are instance versions
    ctx_match = re.search(r":(\d+)k$", result)
    if ctx_match:
        result = result[: ctx_match.start()]

    # Step 4: Remove instance version suffix (:0, :1, :2)
    inst_match = INSTANCE_VERSION_PATTERN.search(result)
    if inst_match:
        result = result[: inst_match.start()]

    # Step 5: Handle API version suffix combined with semantic version
    # e.g., "v3-v1" means model version 3, API version 1 -> normalize to v3.1
    # This is a special case for DeepSeek-style versioning
    # But NOT for date-based IDs like "20240620-v1" (Claude models)
    api_in_id_match = re.search(r"([._-]?v?)(\d+)-v(\d+)$", result)
    if api_in_id_match:
        model_ver = api_in_id_match.group(2)
        # Skip if model_ver looks like a date (6+ digits)
        if len(model_ver) < 6:
            prefix = api_in_id_match.group(1) or "."
            api_ver = api_in_id_match.group(3)
            # Normalize to semantic version format
            if prefix.startswith(("-", "_")):
                prefix = "."
            elif not prefix:
                prefix = "."
            result = result[: api_in_id_match.start()] + f".v{model_ver}.{api_ver}"

    # Step 6: Remove standalone API version suffixes (-v1, -v2, etc.)
    # Only remove if it's clearly an API version suffix, not part of model name
    # API version suffixes typically appear:
    # - After a date (e.g., claude-3-5-sonnet-20240620-v1)
    # - After another version number (e.g., deepseek.v3-v1 - handled above)
    # NOT when it's part of the model name (e.g., embed-english-v3)
    for suffix in API_VERSION_SUFFIXES:
        if result.endswith(suffix):
            prefix_idx = len(result) - len(suffix) - 1
            if prefix_idx < 0:
                break
            # Only remove if preceded by a date (8 digits like 20240620)
            if prefix_idx >= 7:
                potential_date = result[prefix_idx - 7 : prefix_idx + 1]
                if potential_date.isdigit() and len(potential_date) == 8:
                    result = result[: -len(suffix)]
                    break
            # Or if preceded by another version pattern (e.g., -v3-v1)
            # This case is already handled in Step 5
            break

    # Step 7: Remove redundant provider prefix (e.g., deepseek.deepseek-v3 -> deepseek.v3)
    redundant_match = REDUNDANT_PROVIDER_PATTERN.match(result)
    if redundant_match:
        provider = redundant_match.group(1)
        # Remove the redundant prefix, keeping the provider
        result = provider + "." + result[redundant_match.end() :]

    # Step 8: Normalize semantic version separators
    # Convert v3-1 to v3.1, -v3-1 to .v3.1
    result = re.sub(r"([vV])(\d+)-(\d+)$", r"\1\2.\3", result)
    result = re.sub(r"-([vV])(\d+)\.(\d+)$", r".\1\2.\3", result)

    # Step 9: Ensure version prefix is consistent
    # Convert patterns like ".3.1" to ".v3.1" if it looks like a version
    result = re.sub(r"\.(\d+)\.(\d+)$", r".v\1.\2", result)

    # Step 10: Clean up any double dots or trailing separators
    result = re.sub(r"\.+", ".", result)
    result = result.rstrip(".-_")

    return result


def get_model_variant_info(model_id: str) -> ModelVariantInfo:
    """
    Extract variant information from a model ID.

    Analyzes the model ID to determine:
    - Base model ID (without variant suffixes)
    - Whether it's a multimodal variant
    - Whether it's provisioned-only (has dimension suffix)
    - Context window size (if specified)
    - Model version
    - API version

    Args:
        model_id: The model identifier to analyze.

    Returns:
        Dictionary with variant information.

    Examples:
        >>> info = get_model_variant_info("cohere.embed-english-v3:0:512")
        >>> info["base_id"]
        'cohere.embed-english-v3:0'
        >>> info["is_provisioned_only"]
        True
        >>> info["has_dimension_suffix"]
        True

        >>> info = get_model_variant_info("amazon.nova-premier-v1:0:mm")
        >>> info["is_multimodal"]
        True
        >>> info["base_id"]
        'amazon.nova-premier-v1:0'

        >>> info = get_model_variant_info("anthropic.claude-3-5-sonnet-20240620-v1:0:200k")
        >>> info["context_window"]
        200000
        >>> info["base_id"]
        'anthropic.claude-3-5-sonnet-20240620-v1:0'
    """
    if not model_id:
        return ModelVariantInfo(
            base_id="",
            is_multimodal=False,
            is_provisioned_only=False,
            context_window=None,
            version=None,
            api_version=None,
            has_dimension_suffix=False,
        )

    result = model_id
    is_multimodal = False
    is_provisioned_only = False
    context_window: int | None = None
    version: str | None = None
    api_version: int | None = None
    has_dimension_suffix = False

    # Check for dimension suffix (e.g., :0:512) - indicates provisioned-only
    dim_match = DIMENSION_SUFFIX_PATTERN.search(result)
    if dim_match:
        has_dimension_suffix = True
        is_provisioned_only = True
        # The dimension value might indicate embedding dimensions, not context
        result = result[: dim_match.start()]
        # Re-add the instance version if present
        if dim_match.group(1):
            result = result + ":" + dim_match.group(1)

    # Check for multimodal suffix
    if result.lower().endswith(":mm"):
        is_multimodal = True
        result = result[:-3]

    # Check for context window suffix (e.g., :200k, :18k)
    ctx_match = re.search(r":(\d+)k$", result, re.IGNORECASE)
    if ctx_match:
        context_window = int(ctx_match.group(1)) * 1000
        result = result[: ctx_match.start()]

    # Extract API version from suffix (e.g., -v1:0)
    api_match = re.search(r"-v(\d+)(?::\d+)?$", result)
    if api_match:
        api_version = int(api_match.group(1))

    # Extract semantic version
    # Look for patterns like v3.1, v3-1, .v3.1
    ver_match = re.search(r"[._-]v?(\d+)(?:[._-](\d+))?(?::\d+)?$", result)
    if ver_match:
        major = ver_match.group(1)
        minor = ver_match.group(2)
        if minor:
            version = f"{major}.{minor}"
        else:
            version = major

    return ModelVariantInfo(
        base_id=result,
        is_multimodal=is_multimodal,
        is_provisioned_only=is_provisioned_only,
        context_window=context_window,
        version=version,
        api_version=api_version,
        has_dimension_suffix=has_dimension_suffix,
    )


def _extract_provider(model_id: str) -> str:
    """
    Extract and normalize the provider from a model ID.

    Args:
        model_id: The model identifier.

    Returns:
        Normalized provider name, or empty string if not found.

    Examples:
        >>> _extract_provider("anthropic.claude-3-sonnet-v1:0")
        'anthropic'
        >>> _extract_provider("deepseek.deepseek-v3-1")
        'deepseek'
    """
    if not model_id or "." not in model_id:
        return ""

    provider = model_id.split(".")[0].lower()
    return PROVIDER_ALIASES.get(provider, provider)


def _extract_model_family(model_id: str) -> str | None:
    """
    Extract the model family identifier from a model ID.

    Model families are distinct model lines that should not be confused
    (e.g., v3 vs r1, sonnet vs opus).

    Args:
        model_id: The model identifier.

    Returns:
        Model family identifier, or None if not determinable.

    Examples:
        >>> _extract_model_family("deepseek.v3-v1:0")
        'v3'
        >>> _extract_model_family("deepseek.r1")
        'r1'
        >>> _extract_model_family("anthropic.claude-3-sonnet-v1:0")
        'sonnet'
    """
    normalized = model_id.lower()

    # Check for known model families
    for family_key, family_name in MODEL_FAMILIES.items():
        # Look for family as a word boundary match
        pattern = rf"[._-]?{re.escape(family_key)}(?:[._-]|$|\d)"
        if re.search(pattern, normalized):
            return family_name

    return None


def has_semantic_conflict(id1: str, id2: str) -> bool:
    """
    Detect if two model IDs represent semantically different models.

    This function identifies cases where two IDs should NOT be matched
    because they represent fundamentally different models, even if they
    share a provider or have similar naming patterns.

    Conflicts include:
    - Different model families (v3 vs r1, sonnet vs opus)
    - Different major versions (claude-3 vs claude-3.5)
    - Different model types (embed vs generate)

    Args:
        id1: First model identifier.
        id2: Second model identifier.

    Returns:
        True if the IDs represent semantically different models.

    Examples:
        >>> has_semantic_conflict("deepseek.v3-v1:0", "deepseek.r1")
        True
        >>> has_semantic_conflict("claude-3-sonnet", "claude-3-5-sonnet")
        True
        >>> has_semantic_conflict("claude-3-sonnet", "claude-3-opus")
        True
    """
    if not id1 or not id2:
        return False

    norm1 = id1.lower()
    norm2 = id2.lower()

    # Same ID is not a conflict
    if norm1 == norm2:
        return False

    # Check provider mismatch (different providers = conflict)
    provider1 = _extract_provider(id1)
    provider2 = _extract_provider(id2)
    if provider1 and provider2 and provider1 != provider2:
        return True

    # Check model family mismatch
    family1 = _extract_model_family(id1)
    family2 = _extract_model_family(id2)
    if family1 and family2 and family1 != family2:
        return True

    # Check for legacy unversioned Claude pricing keys vs versioned Claude models.
    # Legacy keys like "claude", "claude-instant", "claude-(100k)", "claude-instant-(100k)"
    # should never match versioned models like "claude-3-haiku", "claude-3-5-sonnet", etc.
    versioned_claude_pattern = r"claude[._-]\d+"
    versioned1 = re.search(versioned_claude_pattern, norm1)
    versioned2 = re.search(versioned_claude_pattern, norm2)
    if bool(versioned1) != bool(versioned2):  # one is versioned, the other is not
        # Check the unversioned side is actually a Claude model (not some other model)
        unversioned = norm2 if versioned1 else norm1
        if re.search(r"\.claude(?:-instant)?(?:\(|$|-\(|[^a-z]|$)", unversioned):
            return True

    # Check for Claude version conflicts (3 vs 3.5)
    claude_ver_pattern = r"claude[._-]?(\d+)(?:[._-](\d+))?"
    match1 = re.search(claude_ver_pattern, norm1)
    match2 = re.search(claude_ver_pattern, norm2)
    if match1 and match2:
        ver1 = f"{match1.group(1)}.{match1.group(2) or '0'}"
        ver2 = f"{match2.group(1)}.{match2.group(2) or '0'}"
        if ver1 != ver2:
            return True

    # Check for Claude major version conflicts (3 vs 4)
    # Pattern: claude-opus-4-5 vs claude-3-opus
    claude_major_pattern = r"claude[._-]?(?:opus|sonnet|haiku)?[._-]?(\d+)"
    match1 = re.search(claude_major_pattern, norm1)
    match2 = re.search(claude_major_pattern, norm2)
    if match1 and match2:
        # Extract major version (first digit after claude or variant name)
        major1 = match1.group(1)
        major2 = match2.group(1)
        if major1 != major2:
            return True

    # Also check for claude-X-variant vs claude-variant-X patterns
    # e.g., claude-3-opus vs claude-opus-4
    claude_variant_pattern = r"claude[._-](\d+)[._-](opus|sonnet|haiku)"
    claude_variant_rev_pattern = r"claude[._-](opus|sonnet|haiku)[._-](\d+)"
    match1_std = re.search(claude_variant_pattern, norm1)
    match1_rev = re.search(claude_variant_rev_pattern, norm1)
    match2_std = re.search(claude_variant_pattern, norm2)
    match2_rev = re.search(claude_variant_rev_pattern, norm2)

    # Extract versions from either pattern
    ver1_from_std = match1_std.group(1) if match1_std else None
    ver1_from_rev = match1_rev.group(2) if match1_rev else None
    ver2_from_std = match2_std.group(1) if match2_std else None
    ver2_from_rev = match2_rev.group(2) if match2_rev else None
    ver1 = ver1_from_std or ver1_from_rev
    ver2 = ver2_from_std or ver2_from_rev

    if ver1 and ver2 and ver1 != ver2:
        return True

    # Check for Llama version conflicts
    llama_ver_pattern = r"llama[._-]?(\d+)"
    match1 = re.search(llama_ver_pattern, norm1)
    match2 = re.search(llama_ver_pattern, norm2)
    if match1 and match2:
        if match1.group(1) != match2.group(1):
            return True

    # Check for embed vs non-embed conflict
    is_embed1 = "embed" in norm1
    is_embed2 = "embed" in norm2
    if is_embed1 != is_embed2:
        return True

    # Check for vision vs non-vision conflict
    # "vision" in one ID but not the other = different model line
    # (e.g., palmyra-x5 vs palmyra-vision-7b)
    # Note: VL (vision-language) suffix is handled separately below
    vision_pattern = r"[._-]vision[._-]|[._-]vision$"
    has_vision1 = bool(re.search(vision_pattern, norm1))
    has_vision2 = bool(re.search(vision_pattern, norm2))
    if has_vision1 != has_vision2:
        return True

    # Check for DeepSeek-style version conflicts (v3.1 vs v3.2)
    # This catches cases where the model family is the same but minor version differs
    deepseek_ver_pattern = r"[._-]v?(\d+)[._-](\d+)(?:[._-]|$|:)"
    match1 = re.search(deepseek_ver_pattern, norm1)
    match2 = re.search(deepseek_ver_pattern, norm2)
    if match1 and match2:
        ver1 = f"{match1.group(1)}.{match1.group(2)}"
        ver2 = f"{match2.group(1)}.{match2.group(2)}"
        if ver1 != ver2:
            return True

    # Check for model version conflicts with letter prefix (e.g., m2.1 vs m2.5)
    # This catches MiniMax-style versioning where 'm' prefix is used
    letter_ver_pattern = r"[._-]([a-z])(\d+)\.(\d+)(?:[._-]|$|:)"
    match1 = re.search(letter_ver_pattern, norm1)
    match2 = re.search(letter_ver_pattern, norm2)
    if match1 and match2:
        # Only compare if the letter prefix is the same (e.g., both 'm')
        if match1.group(1) == match2.group(1):
            ver1 = f"{match1.group(2)}.{match1.group(3)}"
            ver2 = f"{match2.group(2)}.{match2.group(3)}"
            if ver1 != ver2:
                return True

    # Check for model size conflicts (e.g., 4b vs 27b, 9b vs 12b, 20b vs 120b)
    # This is critical for preventing wrong matches between different model sizes
    # Pattern matches: 4b, 27b, 70b, 120b, 405b, etc. (with optional hyphen/underscore before)
    size_pattern = r"[._-]?(\d+)[bB](?:[._-]|$|:)"
    sizes1 = re.findall(size_pattern, norm1)
    sizes2 = re.findall(size_pattern, norm2)
    if sizes1 and sizes2:
        # Compare the primary size (usually the first one found)
        # Convert to int for proper numeric comparison
        size1 = int(sizes1[0])
        size2 = int(sizes2[0])
        if size1 != size2:
            return True

    # =========================================================================
    # Extended conflict detection rules
    # Guard: skip all extended rules if canonical forms match (same model,
    # different notation)
    # =========================================================================
    canonical1 = get_canonical_model_id(id1)
    canonical2 = get_canonical_model_id(id2)
    if canonical1 == canonical2:
        return False
    # Also treat separator-only differences as equivalent (m2.1 == m2-1)
    if re.sub(r"[._-]+", "_", canonical1) == re.sub(r"[._-]+", "_", canonical2):
        return False

    # Extract model part (after provider prefix) for some checks
    model_part1 = norm1.split(".", 1)[1] if "." in norm1 else norm1
    model_part2 = norm2.split(".", 1)[1] if "." in norm2 else norm2

    # Check for plus (+) suffix conflict (e.g., command-r vs command-r+)
    has_plus1 = "+" in model_part1 or "-plus" in model_part1
    has_plus2 = "+" in model_part2 or "-plus" in model_part2
    if has_plus1 != has_plus2:
        return True

    # Check for size/variant word conflicts (e.g., nano vs super, large vs lite)
    size_words = {"nano", "mini", "small", "medium", "large", "lite", "super", "ultra", "mega", "tiny"}
    sw1 = {w for w in re.split(r"[._\-:]", norm1) if w in size_words}
    sw2 = {w for w in re.split(r"[._\-:]", norm2) if w in size_words}
    if sw1 and sw2 and sw1 != sw2:
        return True

    # Check for Amazon Nova variant and generation conflicts
    if "nova" in norm1 and "nova" in norm2:
        nova_var1, nova_gen1 = _extract_nova_info(norm1)
        nova_var2, nova_gen2 = _extract_nova_info(norm2)
        # Different variant names = conflict (e.g., nova-micro vs nova-pro)
        if nova_var1 and nova_var2 and nova_var1 != nova_var2:
            return True
        # Different major generation = conflict (e.g., nova-lite vs nova-2-0-lite)
        # Only check when at least one variant is identified (avoids false positives
        # on products like nova-multimodal-embeddings that lack standard variant names)
        if (nova_var1 or nova_var2) and nova_gen1 != nova_gen2:
            return True

    # Check for letter-version asymmetric conflict (e.g., m2.5 vs m2)
    # One ID has letter+major.minor, other has letter+major only
    letter_ver_full = r"[._\-]([a-z])(\d+)[.](\d+)"
    letter_ver_major = r"[._\-]([a-z])(\d+)(?:[._\-]|$|:)"
    m1f = re.search(letter_ver_full, norm1)
    m2f = re.search(letter_ver_full, norm2)
    m1m = re.search(letter_ver_major, norm1)
    m2m = re.search(letter_ver_major, norm2)
    if m1f and not m2f and m2m:
        if m1f.group(1) == m2m.group(1) and m1f.group(2) == m2m.group(2):
            return True
    if m2f and not m1f and m1m:
        if m2f.group(1) == m1m.group(1) and m2f.group(2) == m1m.group(2):
            return True

    # Check for qualifier word conflicts (e.g., gpt-oss-120b vs gpt-oss-safeguard-120b)
    qualifier_words = {"safeguard"}
    words1 = set(re.split(r"[._\-:]", norm1))
    words2 = set(re.split(r"[._\-:]", norm2))
    if (words1 & qualifier_words) != (words2 & qualifier_words):
        return True

    # Check for VL (vision-language) suffix conflict
    vl_pattern = r"[._\-]vl(?:[._\-]|$|:)"
    has_vl1 = bool(re.search(vl_pattern, norm1))
    has_vl2 = bool(re.search(vl_pattern, norm2))
    if has_vl1 != has_vl2:
        return True

    # Check for embed model version/generation conflict (g1 vs v2)
    # Treats gN and vN as interchangeable generation markers for embed models
    if "embed" in norm1 and "embed" in norm2:
        embed_ver_pattern = r"[._\-]([gv])(\d+)(?:[._\-]|$|:)"
        ev1 = re.search(embed_ver_pattern, norm1)
        ev2 = re.search(embed_ver_pattern, norm2)
        if ev1 and ev2 and ev1.group(2) != ev2.group(2):
            return True

    return False


# Nova variant names for conflict detection
_NOVA_VARIANTS = {"lite", "micro", "pro", "premier", "reel", "sonic", "canvas", "omni"}


def _extract_nova_info(s: str) -> tuple:
    """
    Extract (variant_name, major_generation) from a Nova model ID.

    Handles various naming conventions:
    - nova-lite-v1:0 -> (lite, None)
    - nova-2-0-lite -> (lite, 2)
    - nova-2-lite-v1:0 -> (lite, 2)
    - nova-sonic-2-0 -> (sonic, 2)

    Returns:
        (variant, major_gen) where variant is a string or None,
        and major_gen is an int or None.
    """
    idx = s.find("nova")
    if idx < 0:
        return None, None
    rest = s[idx + 4:]
    # Strip instance version (:0, :1) and multimodal suffix (:mm)
    rest = re.sub(r":\d+$", "", rest)
    rest = re.sub(r":mm$", "", rest)
    rest = rest.lstrip(".-_")
    tokens = re.split(r"[._\-]", rest)
    # Remove API version suffix at end (v1, v2)
    if tokens and re.match(r"^v\d+$", tokens[-1]):
        tokens = tokens[:-1]
    variant = None
    gen_numbers = []
    for t in tokens:
        if t in _NOVA_VARIANTS:
            variant = t
        elif t.isdigit():
            gen_numbers.append(int(t))
    major_gen = gen_numbers[0] if gen_numbers else None
    return variant, major_gen


def calculate_match_score(id1: str, id2: str) -> float:
    """
    Calculate a match score between two model IDs.

    The score indicates how likely the two IDs refer to the same model:
    - 1.0: Exact canonical match
    - 0.95+: High-confidence fuzzy match
    - 0.8-0.95: Probable match with some differences
    - 0.0: Semantic conflict (definitely different models)

    Args:
        id1: First model identifier.
        id2: Second model identifier.

    Returns:
        Match score between 0.0 and 1.0.

    Examples:
        >>> calculate_match_score("deepseek.v3-v1:0", "deepseek.v3.1")
        1.0
        >>> calculate_match_score("deepseek.v3-v1:0", "deepseek.deepseek-v3-1")
        1.0
        >>> calculate_match_score("deepseek.v3", "deepseek.r1")
        0.0
    """
    if not id1 or not id2:
        return 0.0

    # Check for semantic conflicts first
    if has_semantic_conflict(id1, id2):
        return 0.0

    # Get canonical forms
    canonical1 = get_canonical_model_id(id1)
    canonical2 = get_canonical_model_id(id2)

    # Exact canonical match
    if canonical1 == canonical2:
        return 1.0

    # Check if one is a prefix of the other (after canonicalization)
    if canonical1.startswith(canonical2) or canonical2.startswith(canonical1):
        # Penalize slightly for length difference
        len_ratio = min(len(canonical1), len(canonical2)) / max(
            len(canonical1), len(canonical2)
        )
        return 0.95 * len_ratio

    # Provider match bonus
    provider1 = _extract_provider(id1)
    provider2 = _extract_provider(id2)
    provider_match = provider1 == provider2 if provider1 and provider2 else False

    # Calculate character-level similarity on canonical forms
    # Using a simple approach: longest common subsequence ratio
    similarity = _calculate_similarity(canonical1, canonical2)

    # Apply provider match bonus
    if provider_match and similarity > 0.5:
        similarity = min(1.0, similarity + 0.1)

    # Threshold: below 0.6 similarity is likely not a match
    if similarity < 0.6:
        return 0.0

    return round(similarity, 3)


def _calculate_similarity(s1: str, s2: str) -> float:
    """
    Calculate string similarity using longest common subsequence ratio.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Similarity ratio between 0.0 and 1.0.
    """
    if not s1 or not s2:
        return 0.0

    if s1 == s2:
        return 1.0

    # Simple LCS-based similarity
    m, n = len(s1), len(s2)

    # Use a space-optimized LCS approach
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev

    lcs_length = prev[n]
    return (2.0 * lcs_length) / (m + n)


def find_best_match(
    model_id: str,
    candidates: dict[str, Any],
    min_score: float = 0.8,
) -> tuple[str | None, float]:
    """
    Find the best matching candidate for a model ID.

    Searches through a dictionary of candidates and returns the key
    with the highest match score above the minimum threshold.

    Args:
        model_id: The model identifier to match.
        candidates: Dictionary where keys are candidate model IDs.
        min_score: Minimum score threshold (default 0.8).

    Returns:
        Tuple of (matched_key, score) or (None, 0.0) if no match found.

    Examples:
        >>> candidates = {
        ...     "deepseek.deepseek-v3-1": {"price": 0.001},
        ...     "deepseek.deepseek-v3-2": {"price": 0.002},
        ...     "deepseek.r1": {"price": 0.003},
        ... }
        >>> match, score = find_best_match("deepseek.v3-v1:0", candidates)
        >>> match
        'deepseek.deepseek-v3-1'
        >>> score >= 0.95
        True

        >>> match, score = find_best_match("unknown.model", candidates)
        >>> match is None
        True
    """
    if not model_id or not candidates:
        return None, 0.0

    best_match: str | None = None
    best_score = 0.0

    for candidate_id in candidates:
        score = calculate_match_score(model_id, candidate_id)
        if score > best_score and score >= min_score:
            best_score = score
            best_match = candidate_id

    return best_match, best_score


def find_all_matches(
    model_id: str,
    candidates: dict[str, Any],
    min_score: float = 0.8,
) -> list[tuple[str, float]]:
    """
    Find all matching candidates for a model ID above the threshold.

    Unlike find_best_match, this returns all candidates that meet
    the minimum score threshold, sorted by score descending.

    Args:
        model_id: The model identifier to match.
        candidates: Dictionary where keys are candidate model IDs.
        min_score: Minimum score threshold (default 0.8).

    Returns:
        List of (matched_key, score) tuples, sorted by score descending.

    Examples:
        >>> candidates = {
        ...     "deepseek.deepseek-v3-1": {},
        ...     "deepseek.v3.1": {},
        ...     "deepseek.r1": {},
        ... }
        >>> matches = find_all_matches("deepseek.v3-v1:0", candidates)
        >>> len(matches) >= 1
        True
    """
    if not model_id or not candidates:
        return []

    matches = []
    for candidate_id in candidates:
        score = calculate_match_score(model_id, candidate_id)
        if score >= min_score:
            matches.append((candidate_id, score))

    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def normalize_provider_prefix(model_id: str) -> str:
    """
    Remove redundant provider prefixes from model IDs.

    Some sources (like Pricing API) use redundant prefixes like
    `deepseek.deepseek-v3` instead of `deepseek.v3`.

    Args:
        model_id: The model identifier.

    Returns:
        Model ID with redundant prefix removed.

    Examples:
        >>> normalize_provider_prefix("deepseek.deepseek-v3-1")
        'deepseek.v3-1'
        >>> normalize_provider_prefix("anthropic.claude-3-sonnet")
        'anthropic.claude-3-sonnet'
    """
    if not model_id:
        return ""

    match = REDUNDANT_PROVIDER_PATTERN.match(model_id.lower())
    if match:
        provider = match.group(1)
        remainder = model_id[match.end() :]
        return f"{provider}.{remainder}"

    return model_id


def is_variant_of(variant_id: str, base_id: str) -> bool:
    """
    Check if variant_id is a variant of base_id.

    A variant is a model ID that extends a base ID with additional
    suffixes like context window (:200k), multimodal (:mm), or
    dimension specifications (:0:512).

    Args:
        variant_id: Potential variant model ID.
        base_id: Base model ID to check against.

    Returns:
        True if variant_id is a variant of base_id.

    Examples:
        >>> is_variant_of("cohere.embed-english-v3:0:512", "cohere.embed-english-v3:0")
        True
        >>> is_variant_of("anthropic.claude-3-5-sonnet-v1:0:200k", "anthropic.claude-3-5-sonnet-v1:0")
        True
        >>> is_variant_of("amazon.nova-premier-v1:0:mm", "amazon.nova-premier-v1:0")
        True
        >>> is_variant_of("deepseek.v3", "deepseek.r1")
        False
    """
    if not variant_id or not base_id:
        return False

    # Get variant info for both
    variant_info = get_model_variant_info(variant_id)
    base_info = get_model_variant_info(base_id)

    # The variant's base_id should match the base model's base_id
    # or the base model itself
    variant_base = variant_info["base_id"].lower()
    base_normalized = base_info["base_id"].lower()

    if variant_base == base_normalized:
        # Check that variant has additional suffixes
        return (
            variant_info["is_multimodal"]
            or variant_info["is_provisioned_only"]
            or variant_info["context_window"] is not None
            or variant_info["has_dimension_suffix"]
        )

    return False


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Type definitions
    "ModelVariantInfo",
    # Core functions
    "get_canonical_model_id",
    "get_model_variant_info",
    "calculate_match_score",
    "find_best_match",
    "find_all_matches",
    "has_semantic_conflict",
    # Utility functions
    "normalize_provider_prefix",
    "is_variant_of",
    # Provider utilities
    "PROVIDER_DISPLAY_NAMES",
    "get_provider_display_name",
    "get_provider_from_model_id",
]


# =============================================================================
# Provider Display Names
# =============================================================================

# Comprehensive mapping of provider prefixes to human-readable display names
# This is the single source of truth for provider name resolution across the codebase
# Used by: final-aggregator (Mantle stubs), pricing-linker, model-merger, etc.
PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    # Major cloud providers
    "amazon": "Amazon",
    "aws": "Amazon",
    "google": "Google",
    "openai": "OpenAI",
    # AI research labs
    "anthropic": "Anthropic",
    "meta": "Meta",
    "deepseek": "DeepSeek",
    "mistral": "Mistral AI",
    "cohere": "Cohere",
    "ai21": "AI21 Labs",
    # Chinese AI companies
    "qwen": "Qwen",
    "alibaba": "Alibaba Cloud",
    "minimax": "MiniMax",
    "moonshot": "Moonshot AI",
    "moonshotai": "Moonshot AI",
    "zai": "Z.AI",
    "zhipu": "Zhipu AI",
    # Image/Video AI companies
    "stability": "Stability AI",
    "luma": "Luma AI",
    "twelvelabs": "TwelveLabs",
    # Other providers
    "nvidia": "NVIDIA",
    "writer": "Writer",
}


def get_provider_display_name(provider_prefix: str) -> str:
    """
    Get the human-readable display name for a provider prefix.

    This function provides a centralized way to resolve provider prefixes
    to their display names, ensuring consistency across the codebase.

    Args:
        provider_prefix: The provider prefix from a model ID (e.g., "google", "openai")

    Returns:
        Human-readable provider name, or title-cased prefix if not found.

    Examples:
        >>> get_provider_display_name("google")
        'Google'
        >>> get_provider_display_name("openai")
        'OpenAI'
        >>> get_provider_display_name("anthropic")
        'Anthropic'
        >>> get_provider_display_name("unknown_provider")
        'Unknown Provider'
    """
    if not provider_prefix:
        return "Unknown"

    normalized = provider_prefix.lower().strip()

    # Check the display names mapping
    if normalized in PROVIDER_DISPLAY_NAMES:
        return PROVIDER_DISPLAY_NAMES[normalized]

    # Fallback: title-case the prefix with proper word separation
    # Convert underscores/hyphens to spaces, then title case
    fallback = normalized.replace("_", " ").replace("-", " ").title()
    return fallback


def get_provider_from_model_id(model_id: str) -> tuple[str, str]:
    """
    Extract provider prefix and display name from a model ID.

    Args:
        model_id: Full model ID (e.g., "google.gemma-3-12b-it")

    Returns:
        Tuple of (normalized_prefix, display_name)

    Examples:
        >>> get_provider_from_model_id("google.gemma-3-12b-it")
        ('google', 'Google')
        >>> get_provider_from_model_id("openai.gpt-oss-120b")
        ('openai', 'OpenAI')
        >>> get_provider_from_model_id("moonshotai.kimi-k2")
        ('moonshot', 'Moonshot AI')
    """
    if not model_id or "." not in model_id:
        return ("unknown", "Unknown")

    prefix = model_id.split(".")[0].lower()

    # Normalize using PROVIDER_ALIASES (e.g., moonshotai -> moonshot)
    normalized = PROVIDER_ALIASES.get(prefix, prefix)

    # Get display name
    display_name = get_provider_display_name(normalized)

    return (normalized, display_name)
