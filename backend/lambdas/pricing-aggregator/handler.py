"""
Pricing Aggregator Lambda

Merges pricing data from all three Bedrock service codes into a unified structure.
Transforms data to match the expected frontend schema with pricing_groups.
"""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict

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
from shared.powertools import logger, tracer, metrics, LambdaContext
from aws_lambda_powertools.metrics import MetricUnit


# =============================================================================
# GovCloud Regions
# =============================================================================

GOVCLOUD_REGIONS = ["us-gov-west-1", "us-gov-east-1"]

# =============================================================================
# Dimension Detection Constants
# =============================================================================

# Inference modes
INFERENCE_MODES = {
    "ON_DEMAND": "on_demand",
    "BATCH": "batch",
    "PROVISIONED": "provisioned",
    "RESERVED": "reserved",
    "MANTLE": "mantle",
    "CUSTOM_MODEL": "custom_model",
}

# Geographic scopes
GEOGRAPHIC_SCOPES = {
    "IN_REGION": "in_region",
    "CRIS_GLOBAL": "cris_global",
    "CRIS_REGIONAL": "cris_regional",
}

# Context types
CONTEXT_TYPES = {
    "STANDARD": "standard",
    "LONG_CONTEXT": "long_context",
}

# Cache types
CACHE_TYPES = {
    "NONE": None,
    "CACHE_READ": "cache_read",
    "CACHE_WRITE": "cache_write",
    "CACHE_WRITE_1H": "cache_write_1h",
}

# Commitment terms (for Reserved pricing)
COMMITMENT_TERMS = {
    "NONE": None,
    "NO_COMMIT": "no_commit",
    "1_MONTH": "1_month",
    "3_MONTH": "3_month",
    "6_MONTH": "6_month",
}

# Tiers
TIERS = {
    "NONE": None,
    "FLEX": "flex",
    "PRIORITY": "priority",
    "STANDARD": "standard",
}


# =============================================================================
# Detection Patterns (regex and keyword-based)
# =============================================================================

# Mantle detection patterns
MANTLE_PATTERNS = [
    re.compile(r"mantle", re.IGNORECASE),
    re.compile(r"openai[-_]?compatible", re.IGNORECASE),
    re.compile(r"chat[-_]?completions", re.IGNORECASE),
]

# CRIS Regional detection patterns (distinct from Global)
CRIS_REGIONAL_PATTERNS = [
    re.compile(r"regional\s*cris", re.IGNORECASE),
    re.compile(r"[-_]geo\b", re.IGNORECASE),  # Matches _geo and -geo suffixes
    re.compile(
        r"cross[-_]?region[-_]?geo", re.IGNORECASE
    ),  # Explicit cross-region-geo pattern
    re.compile(r"\bregional\b(?!.*global)", re.IGNORECASE),
]

# Reserved pricing detection patterns
RESERVED_PATTERNS = [
    re.compile(r"reserved", re.IGNORECASE),
    re.compile(r"_tpm_", re.IGNORECASE),  # Tokens per minute reserved
    re.compile(r"reserved[_-]?(\d+)[_-]?month", re.IGNORECASE),
    re.compile(r"no[_-]?commit", re.IGNORECASE),
]

# Cache type detection patterns
CACHE_PATTERNS = {
    "cache_read": [
        re.compile(r"cache[-_]?read", re.IGNORECASE),
        re.compile(r"cached[-_]?input", re.IGNORECASE),
    ],
    "cache_write": [
        re.compile(r"cache[-_]?write(?![-_]?1h)", re.IGNORECASE),
        re.compile(r"cache[-_]?storage", re.IGNORECASE),
    ],
    "cache_write_1h": [
        re.compile(r"cache[-_]?write[-_]?1h", re.IGNORECASE),
        re.compile(r"1[-_]?hour[-_]?cache", re.IGNORECASE),
    ],
}

# Commitment term detection patterns
COMMITMENT_PATTERNS = {
    "1_month": re.compile(r"1[-_]?month", re.IGNORECASE),
    "3_month": re.compile(r"3[-_]?month", re.IGNORECASE),
    "6_month": re.compile(r"6[-_]?month", re.IGNORECASE),
    "no_commit": re.compile(r"no[-_]?commit", re.IGNORECASE),
}


def detect_mantle_pricing(
    usage_type: str, description: str, inference_type: str = ""
) -> bool:
    """
    Detect if a pricing entry is for Mantle (OpenAI-compatible) endpoint.

    Mantle pricing patterns:
    - "mantle" in usage type or description
    - "openai-compatible" in description or inference type
    - "chat-completions" endpoint references

    Args:
        usage_type: The usagetype field from pricing API
        description: The description field from pricing API
        inference_type: The inferenceType field from pricing API (optional)

    Returns:
        True if this is Mantle pricing
    """
    usage_lower = usage_type.lower() if usage_type else ""
    desc_lower = description.lower() if description else ""
    inference_lower = inference_type.lower() if inference_type else ""

    # Check against Mantle patterns
    for pattern in MANTLE_PATTERNS:
        if (
            pattern.search(usage_lower)
            or pattern.search(desc_lower)
            or pattern.search(inference_lower)
        ):
            return True

    return False


def detect_cris_regional(usage_type: str, description: str) -> bool:
    """
    Detect if a pricing entry is for CRIS Regional (cross-region within geographic area).

    CRIS Regional patterns:
    - "Regional CRIS" in description
    - "_Geo" suffix in usagetype (e.g., USE1_InputTokenCount_Geo)
    - "regional" in usage without "global"
    - "cross-region-geo" in usagetype

    Args:
        usage_type: The usagetype field from pricing API
        description: The description field from pricing API

    Returns:
        True if this is CRIS Regional pricing (not Global)
    """
    usage_lower = usage_type.lower() if usage_type else ""
    desc_lower = description.lower() if description else ""

    # Explicit Global check - if Global, it's not Regional
    if "global" in usage_lower or "global" in desc_lower:
        return False

    # Check against CRIS Regional patterns
    for pattern in CRIS_REGIONAL_PATTERNS:
        if pattern.search(usage_lower) or pattern.search(desc_lower):
            return True

    return False


def detect_reserved_pricing(
    usage_type: str, description: str
) -> tuple[bool, str | None]:
    """
    Detect if a pricing entry is for Reserved capacity and extract commitment term.

    Reserved pricing patterns:
    - "reserved" in usage type or description
    - "_tpm_" (tokens per minute) in usage type
    - "Reserved_1Month", "Reserved_3Month", "Reserved_6Month" patterns
    - "no-commit" for no-commitment reserved

    Exclusions:
    - ProvisionedThroughput entries that don't contain "reserved" are NOT reserved pricing.
      E.g., "USE1-Nova2.0Lite-ProvisionedThroughput-NoCommit-ModelUnits" is Provisioned,
      not Reserved, even though it contains "NoCommit".

    Args:
        usage_type: The usagetype field from pricing API
        description: The description field from pricing API

    Returns:
        Tuple of (is_reserved, commitment_term)
        commitment_term is one of: None, "no_commit", "1_month", "3_month", "6_month"
    """
    usage_lower = usage_type.lower() if usage_type else ""
    desc_lower = description.lower() if description else ""

    # Exclude ProvisionedThroughput entries that don't have "reserved" in them
    # ProvisionedThroughput-NoCommit is Provisioned Throughput pricing, not Reserved pricing
    if "provisionedthroughput" in usage_lower and "reserved" not in usage_lower:
        return False, None

    # Check if this is reserved pricing
    is_reserved = False
    for pattern in RESERVED_PATTERNS:
        if pattern.search(usage_lower) or pattern.search(desc_lower):
            is_reserved = True
            break

    if not is_reserved:
        return False, None

    # Extract commitment term
    commitment = None
    for term, pattern in COMMITMENT_PATTERNS.items():
        if pattern.search(usage_lower) or pattern.search(desc_lower):
            commitment = term
            break

    return True, commitment


def detect_cache_type(usage_type: str, description: str) -> str | None:
    """
    Detect the cache type from usage type and description.

    Cache types:
    - cache_read: Cached input tokens (reading from cache)
    - cache_write: Writing to cache (standard)
    - cache_write_1h: Writing to cache with 1-hour retention

    Args:
        usage_type: The usagetype field from pricing API
        description: The description field from pricing API

    Returns:
        Cache type string or None if not cache pricing
    """
    usage_lower = usage_type.lower() if usage_type else ""
    desc_lower = description.lower() if description else ""

    # Check for cache_write_1h FIRST (most specific)
    # Must check before cache_write to avoid false positives
    for pattern in CACHE_PATTERNS.get("cache_write_1h", []):
        if pattern.search(usage_lower) or pattern.search(desc_lower):
            return "cache_write_1h"

    # Check for cache_write (before cache_read to avoid false positives)
    for pattern in CACHE_PATTERNS.get("cache_write", []):
        if pattern.search(usage_lower) or pattern.search(desc_lower):
            return "cache_write"

    # Check for cache_read
    for pattern in CACHE_PATTERNS.get("cache_read", []):
        if pattern.search(usage_lower) or pattern.search(desc_lower):
            return "cache_read"

    return None


def extract_govcloud_availability(products: list[dict]) -> dict[str, dict]:
    """
    Extract which models are available in GovCloud regions from pricing data.

    The Pricing API returns pricing entries for GovCloud regions (us-gov-west-1,
    us-gov-east-1). This function extracts which models have pricing in these
    regions and determines if they are CRIS or In-Region based on config.

    Args:
        products: List of pricing product entries from the Pricing API

    Returns:
        Dict mapping model names to availability info:
        {
            "Claude 3 Haiku": {
                "regions": ["us-gov-east-1", "us-gov-west-1"],
                "inference_type": "cris"  # or "in_region"
            }
        }
    """
    config_loader = get_config_loader()
    govcloud_models: dict[str, set[str]] = {}

    for product in products:
        attrs = product.get("product", {}).get("attributes", {})
        region = attrs.get("regionCode", "")
        model_name = attrs.get("model", "") or attrs.get("servicename", "")

        if region in GOVCLOUD_REGIONS and model_name:
            # Clean the model name
            cleaned_name = clean_model_name(model_name)
            if cleaned_name and cleaned_name.lower() not in [
                "unknown",
                "unknown model",
            ]:
                if cleaned_name not in govcloud_models:
                    govcloud_models[cleaned_name] = set()
                govcloud_models[cleaned_name].add(region)

    # Build result with inference type
    result = {}
    for model_name, regions in govcloud_models.items():
        # Check if this model should be CRIS in GovCloud
        is_cris = config_loader.is_govcloud_cris_model(model_name)

        result[model_name] = {
            "regions": sorted(list(regions)),
            "inference_type": "cris" if is_cris else "in_region",
        }

    return result


def get_region_locations() -> dict:
    """Get region locations from configuration."""
    return get_config_loader().get_region_locations()


def get_provider_patterns() -> dict:
    """Get provider patterns from configuration."""
    return get_config_loader().get_provider_patterns()


def get_explicit_provider_names() -> dict:
    """Get explicit provider name mappings from configuration."""
    return get_config_loader().get_explicit_provider_names()


def determine_pricing_type(usage_type: str, unit: str, description: str) -> dict:
    """
    Determine the pricing type and unit from usage type, unit, and description.

    Returns:
        {
            'pricing_type': 'token' | 'image' | 'video_second' | 'model_unit' | 'other',
            'unit_label': 'per 1K tokens' | 'per image' | etc.,
            'is_input': True/False/None,
            'is_output': True/False/None,
        }
    """
    usage_lower = usage_type.lower()
    unit_lower = (unit or "").lower()
    desc_lower = (description or "").lower()

    # Determine if input/output
    is_input = "input" in usage_lower or "input" in desc_lower
    is_output = "output" in usage_lower or "output" in desc_lower

    # Check for per-image pricing
    # Patterns: 'per image', 'image', 'images', 'images processed', 'created_image', 'output image'
    is_image_pricing = (
        "per image" in desc_lower
        or unit_lower == "images"
        or unit_lower == "image"  # Support singular form (e.g., Nova Canvas)
        or "images processed" in desc_lower
        or "created_image" in usage_lower
        or "output image" in desc_lower
        or ("stable" in desc_lower and "image" in desc_lower)  # Stability AI pattern
    )

    if is_image_pricing:
        # Image generation models (Canvas, Titan Image Generator, Stability AI, etc.)
        if (
            "t2i" in usage_lower
            or "i2i" in usage_lower
            or "created_image" in usage_lower
            or ("stable" in desc_lower and "image" in desc_lower)
        ):
            return {
                "pricing_type": "image_generation",
                "unit_label": "per image",
                "is_input": None,
                "is_output": None,
            }
        # Image embedding/processing
        return {
            "pricing_type": "image",
            "unit_label": "per image",
            "is_input": is_input or not is_output,
            "is_output": is_output,
        }

    # Check for video generation (I2V = image-to-video, T2V = text-to-video)
    # Patterns: NovaReel-I2V-Medfps-HDRes, NovaReel-T2V-Lowfps-SDRes
    is_video_generation = (
        "i2v" in usage_lower  # image-to-video
        or "t2v" in usage_lower  # text-to-video
        or (
            "video" in usage_lower
            and ("generation" in desc_lower or "generated" in desc_lower)
        )
    )

    if is_video_generation:
        return {
            "pricing_type": "video_generation",
            "unit_label": "per video",
            "is_input": None,
            "is_output": None,
        }

    # Check for video pricing (per second or per frame) - for video processing, not generation
    if "video" in usage_lower and ("second" in unit_lower or "frame" in unit_lower):
        return {
            "pricing_type": "video",
            "unit_label": f"per {unit_lower}",
            "is_input": is_input,
            "is_output": is_output,
        }

    # Check for model units (provisioned throughput)
    if (
        "modelunit" in usage_lower
        or "model-unit" in usage_lower
        or "modelunits" in unit_lower
    ):
        return {
            "pricing_type": "model_unit",
            "unit_label": "per hour",
            "is_input": None,
            "is_output": None,
        }

    # Check for search units (rerank models like Cohere Rerank, Amazon Rerank)
    if (
        "search" in unit_lower
        or "search" in desc_lower
        or "rerank" in usage_lower
        or "rerank" in desc_lower
    ):
        return {
            "pricing_type": "search_unit",
            "unit_label": "per 1K search units",
            "is_input": None,
            "is_output": None,
        }

    # Check for video per-second pricing (Luma AI Ray)
    if ("second" in unit_lower or "per second" in desc_lower) and (
        "video" in desc_lower or "ray" in usage_lower
    ):
        return {
            "pricing_type": "video_second",
            "unit_label": "per second",
            "is_input": None,
            "is_output": None,
        }

    # Check for reserved capacity (TPM-hour pricing)
    # 1P items: unit="1K TPM Hour", dimension contains "tokens-per-minute"
    # 3P Marketplace items: unit="Units", dimension contains "TPM", desc contains "per 1K ... TPM"
    if "tpm" in unit_lower or "tpm" in usage_lower or "tpm" in desc_lower or "tokens-per-minute" in usage_lower:
        return {
            "pricing_type": "reserved_tpm",
            "unit_label": "per 1K TPM Hour",
            "is_input": is_input,
            "is_output": is_output,
        }

    # Check for token-based pricing (most common)
    if (
        "token" in usage_lower
        or "token" in desc_lower
        or "1k token" in desc_lower
        or "1m token" in desc_lower
    ):
        return {
            "pricing_type": "token",
            "unit_label": "per 1K tokens",
            "is_input": is_input,
            "is_output": is_output,
        }

    # Default to token-based for text models
    return {
        "pricing_type": "token",
        "unit_label": "per 1K tokens",
        "is_input": is_input,
        "is_output": is_output,
    }


def determine_pricing_group(
    usage_type: str, inference_type: str, description: str = ""
) -> str:
    """Determine the pricing group based on usage type, inference type, and description.

    This is the legacy function that returns the full group name including
    context and geo modifiers. Kept for backward compatibility.
    """
    usage_lower = usage_type.lower() if usage_type else ""
    inference_lower = inference_type.lower() if inference_type else ""
    description_lower = description.lower() if description else ""

    # Check for Mantle first (highest priority)
    if detect_mantle_pricing(usage_type, description, inference_type):
        return "Mantle"

    # Check for global (cross-region worldwide)
    is_global = "global" in usage_lower or "global" in description_lower

    # Check for geo/regional (cross-region within geographic area) using helper
    is_geo = detect_cris_regional(usage_type, description)

    # Check for Reserved pricing (before provisioned, as reserved is more specific)
    is_reserved, commitment = detect_reserved_pricing(usage_type, description)
    if is_reserved:
        # Build base reserved group name
        if commitment:
            base_group = f"Reserved {commitment.replace('_', ' ').title()}"
        else:
            base_group = "Reserved"
        # Append Global/Geo suffix if applicable
        if is_global:
            return f"{base_group} Global"
        elif is_geo:
            return f"{base_group} Geo"
        return base_group

    # Check for batch
    is_batch = "batch" in usage_lower

    # Check for long context - includes _lctx suffix used in newer AWS format
    is_long_context = (
        "long-context" in usage_lower
        or "long context" in inference_lower
        or "_lctx" in usage_lower  # New AWS format: USE1_InputTokenCount_LCtx
        or "longcontext" in usage_lower
    )

    # Check for provisioned capacity (but not reserved - already handled above)
    is_provisioned = "provisioned" in usage_lower or "provisioned" in inference_lower

    # Check for custom model
    is_custom = "custom" in usage_lower or "fine-tun" in usage_lower

    # Determine group
    if is_custom:
        return "Custom Model"
    elif is_provisioned:
        return "Provisioned Throughput"
    elif is_batch and is_long_context and is_global:
        return "Batch Long Context Global"
    elif is_batch and is_long_context and is_geo:
        return "Batch Long Context Geo"
    elif is_batch and is_long_context:
        return "Batch Long Context"
    elif is_batch and is_global:
        return "Batch Global"
    elif is_batch and is_geo:
        return "Batch Geo"
    elif is_batch:
        return "Batch"
    elif is_long_context and is_global:
        return "On-Demand Long Context Global"
    elif is_long_context and is_geo:
        return "On-Demand Long Context Geo"
    elif is_long_context:
        return "On-Demand Long Context"
    elif is_global:
        return "On-Demand Global"
    elif is_geo:
        return "On-Demand Geo"
    else:
        return "On-Demand"


def determine_pricing_group_with_dimensions(
    usage_type: str, inference_type: str, description: str = ""
) -> dict:
    """
    Determine the pricing group and nested dimensions from usage type, inference type, and description.

    Returns:
        {
            'group': 'On-Demand' | 'Batch' | 'Provisioned Throughput' | 'Reserved' | 'Mantle' | 'Custom Model',
            'dimensions': {
                'inference_mode': 'on_demand' | 'batch' | 'provisioned' | 'reserved' | 'mantle' | 'custom_model',
                'geographic_scope': 'in_region' | 'cris_global' | 'cris_regional',
                'context_type': 'standard' | 'long_context',
                'cache_type': None | 'cache_read' | 'cache_write' | 'cache_write_1h',
                'tier': None | 'flex' | 'priority' | 'standard',
                'commitment': None | 'no_commit' | '1_month' | '3_month' | '6_month',
            }
        }
    """
    usage_lower = usage_type.lower() if usage_type else ""
    inference_lower = inference_type.lower() if inference_type else ""
    description_lower = description.lower() if description else ""

    # Initialize dimensions with defaults
    dimensions = {
        "source": "standard",
        "inference_mode": "on_demand",
        "geographic_scope": "in_region",
        "context_type": "standard",
        "cache_type": None,
        "tier": None,
        "commitment": None,
    }

    # Detect Mantle pricing FIRST (highest priority)
    if detect_mantle_pricing(usage_type, description, inference_type):
        dimensions["inference_mode"] = "mantle"
        dimensions["source"] = "mantle"  # Keep backward compatibility
        return {"group": "Mantle", "dimensions": dimensions}

    # Detect geographic scope (needed for Reserved pricing too)
    if "global" in usage_lower or "global" in description_lower:
        dimensions["geographic_scope"] = "cris_global"
    elif detect_cris_regional(usage_type, description):
        dimensions["geographic_scope"] = "cris_regional"

    # Detect Reserved pricing (before provisioned, as reserved is more specific)
    is_reserved, commitment = detect_reserved_pricing(usage_type, description)
    if is_reserved:
        dimensions["inference_mode"] = "reserved"
        dimensions["commitment"] = commitment
        return {"group": "Reserved", "dimensions": dimensions}

    # Detect cache type
    cache_type = detect_cache_type(usage_type, description)
    if cache_type:
        dimensions["cache_type"] = cache_type

    # Detect tier dimension
    if "flex" in usage_lower:
        dimensions["tier"] = "flex"
    elif "priority" in usage_lower:
        dimensions["tier"] = "priority"

    # Detect context dimension
    if (
        "long-context" in usage_lower
        or "long context" in inference_lower
        or "_lctx" in usage_lower
        or "longcontext" in usage_lower
    ):
        dimensions["context_type"] = "long_context"

    # Determine base group and inference mode
    is_batch = "batch" in usage_lower
    is_provisioned = "provisioned" in usage_lower or "provisioned" in inference_lower
    is_custom = "custom" in usage_lower or "fine-tun" in usage_lower

    if is_custom:
        group = "Custom Model"
        dimensions["inference_mode"] = "custom_model"
    elif is_provisioned:
        group = "Provisioned Throughput"
        dimensions["inference_mode"] = "provisioned"
    elif is_batch:
        group = "Batch"
        dimensions["inference_mode"] = "batch"
    else:
        group = "On-Demand"
        # inference_mode already defaults to "on_demand"

    return {"group": group, "dimensions": dimensions}


def aggregate_dimensions(pricing_entries: list) -> dict:
    """Aggregate available dimensions from all pricing entries.

    Args:
        pricing_entries: List of pricing entry dicts with 'dimensions' field

    Returns:
        {
            'sources': ['standard', 'mantle'],
            'geos': ['global', 'regional'],  # Legacy name for backward compatibility
            'geographic_scopes': ['in_region', 'cris_global', 'cris_regional'],
            'tiers': ['flex', 'priority'],
            'contexts': ['standard', 'long'],  # Legacy name for backward compatibility
            'context_types': ['standard', 'long_context'],
            'inference_modes': ['on_demand', 'batch', 'reserved', 'mantle', ...],
            'commitments': ['no_commit', '1_month', '3_month', '6_month'],
            'cache_types': ['cache_read', 'cache_write', 'cache_write_1h']
        }
    """
    sources = set()
    geos = set()
    geographic_scopes = set()
    tiers = set()
    contexts = set()
    context_types = set()
    inference_modes = set()
    commitments = set()
    cache_types = set()

    for entry in pricing_entries:
        dims = entry.get("dimensions", {})
        if dims.get("source"):
            sources.add(dims["source"])
        # Handle both old "geo" and new "geographic_scope" field names
        if dims.get("geo"):
            geos.add(dims["geo"])
        if dims.get("geographic_scope"):
            geographic_scopes.add(dims["geographic_scope"])
        if dims.get("tier"):
            tiers.add(dims["tier"])
        # Handle both old "context" and new "context_type" field names
        if dims.get("context"):
            contexts.add(dims["context"])
        if dims.get("context_type"):
            context_types.add(dims["context_type"])
        if dims.get("inference_mode"):
            inference_modes.add(dims["inference_mode"])
        if dims.get("commitment"):
            commitments.add(dims["commitment"])
        if dims.get("cache_type"):
            cache_types.add(dims["cache_type"])

    return {
        "sources": sorted(list(sources)) if sources else ["standard"],
        "geos": sorted(list(geos)) if geos else [],  # Legacy
        "geographic_scopes": sorted(list(geographic_scopes))
        if geographic_scopes
        else ["in_region"],
        "tiers": sorted(list(tiers)) if tiers else [],
        "contexts": sorted(list(contexts)) if contexts else ["standard"],  # Legacy
        "context_types": sorted(list(context_types)) if context_types else ["standard"],
        "inference_modes": sorted(list(inference_modes))
        if inference_modes
        else ["on_demand"],
        "commitments": sorted(list(commitments)) if commitments else [],
        "cache_types": sorted(list(cache_types)) if cache_types else [],
    }


def clean_model_name(raw_name: str) -> str:
    """Clean model name by removing AWS-specific suffixes.

    Examples:
        'Stable Diffusion 3 Large v1.0 (Amazon Bedrock Edition)' -> 'Stable Diffusion 3 Large v1.0'
        'Claude 3.5 Sonnet (Amazon Bedrock Edition)' -> 'Claude 3.5 Sonnet'
    """
    if not raw_name or raw_name.lower() in ["unknown", "unknown model"]:
        return raw_name

    cleaned = raw_name.strip()

    # Remove AWS-specific suffixes
    suffixes_to_remove = [
        "(Amazon Bedrock Edition)",
        "(Amazon Bedrock)",
        "Amazon Bedrock Edition",
        "Amazon Bedrock",
    ]

    for suffix in suffixes_to_remove:
        if suffix in cleaned:
            cleaned = cleaned.replace(suffix, "").strip()

    return cleaned if cleaned else raw_name


def extract_from_usagetype(usagetype: str) -> str:
    """Extract model name from usagetype as fallback.

    Patterns like:
    - "USE1-NovaLite-input-tokens" -> "Nova Lite"
    - "APN1-Claude3Sonnet-output" -> "Claude 3 Sonnet"
    """
    if not usagetype:
        return None

    # Remove region prefix (e.g., "USE1-", "APN1-")
    parts = usagetype.split("-")
    if len(parts) < 2:
        return None

    # Skip common non-model parts
    skip_parts = [
        "mp",
        "input",
        "output",
        "tokens",
        "count",
        "units",
        "cache",
        "read",
        "write",
    ]

    for part in parts[1:]:
        if part.lower() in skip_parts:
            continue

        # If part looks like a model name (contains letters and is substantial)
        if len(part) > 3 and any(c.isalpha() for c in part):
            # Try to format it nicely (camelCase -> Title Case)
            formatted = re.sub(r"([a-z])([A-Z])", r"\1 \2", part)
            if len(formatted) > 3:
                return formatted

    return None


def extract_raw_model_name(attributes: dict) -> str:
    """Extract raw model name using multi-strategy approach.

    Priority order:
    1. servicename (for AmazonBedrockFoundationModels)
    2. model (for AmazonBedrock, AmazonBedrockService)
    3. titanModel (special case for Titan models)
    4. Fallback extraction from usagetype
    """
    # Strategy 1: servicename (most common in AmazonBedrockFoundationModels)
    servicename = attributes.get("servicename", "").strip()
    if servicename and servicename not in ["Amazon Bedrock", "Amazon Bedrock Service"]:
        return servicename

    # Strategy 2: model field (most common in AmazonBedrock, AmazonBedrockService)
    model = attributes.get("model", "").strip()
    if model and model.lower() != "unknown":
        return model

    # Strategy 3: titanModel field (special case)
    titan_model = attributes.get("titanModel", "").strip()
    if titan_model:
        return titan_model

    # Strategy 4: Extract from usagetype (fallback)
    usagetype = attributes.get("usagetype", "")
    if usagetype:
        extracted = extract_from_usagetype(usagetype)
        if extracted:
            return extracted

    return "Unknown Model"


def extract_model_info(product: dict) -> dict:
    """Extract model information from a pricing product."""
    attributes = product.get("product", {}).get("attributes", {})
    terms = product.get("terms", {})

    # Extract pricing from OnDemand terms
    price_per_unit = None
    unit = None
    currency = "USD"
    description = ""

    on_demand = terms.get("OnDemand", {})
    for term_key, term_value in on_demand.items():
        price_dimensions = term_value.get("priceDimensions", {})
        for dim_key, dim_value in price_dimensions.items():
            price_per_unit = dim_value.get("pricePerUnit", {}).get("USD")
            unit = dim_value.get("unit")
            description = dim_value.get("description", "")
            break
        break

    # Parse the price
    try:
        price = float(price_per_unit) if price_per_unit else None
    except (ValueError, TypeError):
        price = None

    # Normalize price to per-thousand if needed (some prices are per-million)
    original_price = price
    desc_lower = description.lower()
    # Check for various per-million patterns:
    # - "per 1M" (standard format)
    # - "Million Input Tokens", "Million Response Tokens" (AWS Marketplace format)
    # - "per 1,000,000" or "per million"
    # - "Token Count" (alternate AWS format — same per-million price, different wording)
    is_per_million = (
        "per 1m" in desc_lower
        or "million" in desc_lower
        or "per 1,000,000" in desc_lower
        or "1000000" in desc_lower
        or "token count" in desc_lower
    )
    if price and is_per_million:
        price = price / 1000  # Convert to per-thousand

    # Get model name using multi-strategy extraction
    raw_model_name = extract_raw_model_name(attributes)
    model_name = clean_model_name(raw_model_name)

    return {
        "model": model_name,
        "region": attributes.get("regionCode", "Unknown"),
        "inferenceType": attributes.get("inferenceType", ""),
        "usageType": attributes.get("usagetype", ""),
        "operation": attributes.get("operation", ""),
        "price": price,
        "original_price": original_price,
        "unit": unit,
        "currency": currency,
        "sku": product.get("product", {}).get("sku", ""),
        "description": description,
        "serviceCode": attributes.get("servicecode", "AmazonBedrock"),
        "attributes": attributes,  # Pass all attributes for provider detection fallback
    }


def detect_custom_model_type(description: str, dimension: str) -> str:
    """Detect if this is a Custom Model Import vs Custom Model Training.

    Args:
        description: Price description
        dimension: Price dimension (usagetype)

    Returns:
        'Custom Model Import', 'Custom Model Training', or None
    """
    desc_lower = description.lower()
    dim_lower = dimension.lower()

    # Custom Model Import indicators
    import_indicators = [
        "flan architecture",
        "llama architecture",
        "inference for",
        "storage for",
        "custom model unit per min for inference",
        "custom model unit/month storage",
        "imported model",
        "model import",
    ]

    # Custom Model Training/Customization indicators
    training_indicators = [
        "customization-training",
        "customization-storage",
        "fine",
        "finetun",
        "training",
        "custom training",
        "model customization",
    ]

    # Check for import patterns
    if any(
        indicator in desc_lower or indicator in dim_lower
        for indicator in import_indicators
    ):
        return "Custom Model Import"

    # Check for training/customization patterns
    if any(
        indicator in desc_lower or indicator in dim_lower
        for indicator in training_indicators
    ):
        return "Custom Model Training"

    return None


def normalize_provider_name(provider: str) -> str:
    """Normalize provider name to match model data provider names.

    E.g., 'Mistral' -> 'Mistral AI', 'mistral' -> 'Mistral AI'
    """
    if not provider:
        return provider

    provider_lower = provider.lower().strip()

    # Check explicit mappings first (from config)
    explicit_names = get_explicit_provider_names()
    if provider_lower in explicit_names:
        return explicit_names[provider_lower]

    # Return as-is if no mapping found
    return provider


def infer_provider(model_name: str, attributes: dict = None) -> str:
    """Infer the provider from the model name and attributes.

    Uses multi-strategy approach:
    1. Check explicit 'provider' attribute (normalized to match model data)
    2. Check explicit provider names in model name
    3. Check generic keywords in model name
    4. Fallback: search ALL attributes for provider keywords
    """
    model_lower = model_name.lower()

    # Strategy 1: Check explicit 'provider' attribute (AmazonBedrockService has this)
    if attributes:
        explicit_provider = attributes.get("provider", "").strip()
        if explicit_provider and explicit_provider.lower() != "unknown":
            # Normalize to match model data provider names (e.g., 'Mistral' -> 'Mistral AI')
            return normalize_provider_name(explicit_provider)

    # Get mappings from config
    explicit_names = get_explicit_provider_names()
    provider_patterns = get_provider_patterns()

    # Strategy 2: Check for explicit provider names in model name (high confidence)
    for explicit_name, provider in explicit_names.items():
        if explicit_name in model_lower:
            return provider

    # Strategy 3: Check generic keywords in model name
    for provider, patterns in provider_patterns.items():
        for pattern in patterns:
            if pattern in model_lower:
                return provider

    # Strategy 4: Fallback - search ALL attributes for provider keywords
    if attributes:
        all_text = " ".join(str(v) for v in attributes.values()).lower()
        for provider, patterns in provider_patterns.items():
            for pattern in patterns:
                if pattern in all_text:
                    return provider

    return "Unknown Models"


def normalize_model_id(model_name: str, provider: str) -> str:
    """Normalize model name to a consistent ID format."""
    # Create a provider prefix
    provider_prefix = provider.lower().replace(" ", "-").replace("_", "-")
    if provider_prefix == "unknown-models":
        provider_prefix = "unknown"

    # Clean the model name
    model_clean = model_name.lower().replace(" ", "-").replace(".", "-")

    return f"{provider_prefix}.{model_clean}"


def aggregate_pricing(all_products: list[dict]) -> tuple[dict, dict]:
    """
    Aggregate all pricing products into the expected schema structure.

    Output structure:
    {
        "providers": {
            "provider.model-id": {
                "model_name": "Model Name",
                "model_provider": "Provider",
                "regions": {
                    "us-east-1": {
                        "pricing_groups": {
                            "On-Demand": [...],
                            "Batch": [...]
                        },
                        "total_dimensions": 10,
                        "groups_count": 2,
                        "group_statistics": {...}
                    }
                }
            }
        }
    }
    """
    # Structure: provider_model_id -> region -> pricing_group -> entries
    models_data = defaultdict(
        lambda: {
            "model_name": "",
            "model_provider": "",
            "regions": defaultdict(lambda: {"pricing_groups": defaultdict(list)}),
        }
    )

    group_types_seen = set()
    total_entries = 0

    for product in all_products:
        info = extract_model_info(product)

        model_name = info["model"]
        region = info["region"]

        if (
            model_name == "Unknown"
            or model_name == "Unknown Model"
            or region == "Unknown"
        ):
            continue

        # Check for Custom Model Import/Training first
        custom_model_type = detect_custom_model_type(
            info["description"], info["usageType"]
        )

        # Infer provider with all attributes for fallback detection
        if custom_model_type == "Custom Model Import":
            provider = "Custom Model Import"
        else:
            provider = infer_provider(model_name, info.get("attributes"))

        # Create model ID
        model_id = normalize_model_id(model_name, provider)

        # Determine pricing group (legacy - full group name with modifiers)
        legacy_pricing_group = determine_pricing_group(
            info["usageType"], info["inferenceType"], info["description"]
        )
        group_types_seen.add(legacy_pricing_group)

        # Determine pricing group with dimensions (new - base group + nested dimensions)
        group_info = determine_pricing_group_with_dimensions(
            info["usageType"], info["inferenceType"], info["description"]
        )
        dimensions = group_info["dimensions"]

        # Get location name from config
        region_locations = get_region_locations()
        location = region_locations.get(region, region)

        # Determine pricing type
        pricing_type_info = determine_pricing_type(
            info["usageType"], info["unit"], info["description"]
        )

        # Build pricing entry in expected schema
        pricing_entry = {
            "dimension": info["usageType"],
            "price_per_unit": info["price"],  # Generic price per unit
            "price_per_thousand": info["price"]
            if pricing_type_info["pricing_type"] == "token"
            else None,
            "original_price": info["original_price"],
            "unit": info["unit"] or "tokens",
            "description": info["description"],
            "source_dataset": "aws_pricing_api",
            "model_id": model_id,
            "model_name": model_name,
            "provider": provider,
            "model_provider": provider,
            "location": location,
            "operation": info["operation"],
            "service_code": info["serviceCode"],
            "pricing_type": pricing_type_info["pricing_type"],
            "unit_label": pricing_type_info["unit_label"],
            "is_input": pricing_type_info["is_input"],
            "is_output": pricing_type_info["is_output"],
            # New nested dimensions
            "dimensions": dimensions,
            "pricing_characteristics": {
                "inference_type": "on_demand"
                if "on-demand" in legacy_pricing_group.lower()
                else ("batch" if "batch" in legacy_pricing_group.lower() else "other"),
                "context_type": "long_context"
                if "long context" in legacy_pricing_group.lower()
                else "standard",
                "geographic_scope": "global"
                if "global" in legacy_pricing_group.lower()
                else "regional",
                "cache_type": dimensions.get("cache_type"),
            },
            # Keep legacy group name for backward compatibility
            "pricing_group": legacy_pricing_group,
        }

        models_data[model_id]["model_name"] = model_name
        models_data[model_id]["model_provider"] = provider
        models_data[model_id]["pricing_types"] = models_data[model_id].get(
            "pricing_types", set()
        )
        models_data[model_id]["pricing_types"].add(pricing_type_info["pricing_type"])
        # Use legacy_pricing_group for backward compatibility with existing 10 groups
        models_data[model_id]["regions"][region]["pricing_groups"][
            legacy_pricing_group
        ].append(pricing_entry)
        total_entries += 1

    # Convert to final structure nested by provider: providers -> Provider -> model_id -> data
    # This matches the frontend expected schema
    result = defaultdict(dict)
    total_regions_processed = 0
    total_groups_created = 0

    for model_id, model_data in models_data.items():
        provider = model_data["model_provider"]

        # Convert pricing_types set to list for JSON serialization
        pricing_types_list = sorted(list(model_data.get("pricing_types", set())))

        # Determine primary pricing type for the model
        # Priority: video_generation > image_generation > video_second > video > image > search_unit > token > model_unit
        # Image/video generation models should show per-image/video pricing, not token pricing
        # Token pricing is prioritized over model_unit (provisioned throughput) for card display
        primary_pricing_type = "token"  # default
        for pt in [
            "video_generation",
            "image_generation",
            "video_second",
            "video",
            "image",
            "search_unit",
            "token",
            "model_unit",
        ]:
            if pt in pricing_types_list:
                primary_pricing_type = pt
                break

        # Collect all pricing entries across all regions for dimension aggregation
        all_entries = []
        for region_data in model_data["regions"].values():
            for entries in region_data["pricing_groups"].values():
                all_entries.extend(entries)

        # Aggregate available dimensions and check for mantle pricing
        available_dims = aggregate_dimensions(all_entries)
        has_mantle = "mantle" in available_dims.get("sources", [])

        model_entry = {
            "model_name": model_data["model_name"],
            "model_provider": provider,
            "pricing_types": pricing_types_list,
            "primary_pricing_type": primary_pricing_type,
            "available_dimensions": available_dims,  # NEW: aggregated dimensions
            "has_mantle_pricing": has_mantle,  # NEW: quick check for mantle
            "regions": {},
        }

        for region, region_data in model_data["regions"].items():
            pricing_groups = dict(region_data["pricing_groups"])

            # NOTE: We intentionally do NOT copy Global/Geo entries to On-Demand/Batch
            # when the base group doesn't exist. This preserves the distinction between:
            # - True In-Region pricing (On-Demand, Batch)
            # - CRIS Global pricing (On-Demand Global, Batch Global)
            # - CRIS Geo pricing (On-Demand Geo, Batch Geo)
            # The frontend handles missing groups appropriately.

            total_dimensions = sum(len(entries) for entries in pricing_groups.values())
            groups_count = len(pricing_groups)

            # Calculate group statistics
            group_sizes = {
                group: len(entries) for group, entries in pricing_groups.items()
            }
            largest_groups = sorted(
                group_sizes.items(), key=lambda x: x[1], reverse=True
            )[:5]

            model_entry["regions"][region] = {
                "pricing_groups": pricing_groups,
                "total_dimensions": total_dimensions,
                "groups_count": groups_count,
                "group_statistics": {
                    "total_entries": total_dimensions,
                    "total_groups": groups_count,
                    "group_sizes": group_sizes,
                    "largest_groups": largest_groups,
                    "average_entries_per_group": total_dimensions / groups_count
                    if groups_count > 0
                    else 0,
                },
            }

            total_regions_processed += 1
            total_groups_created += groups_count

        # Nest under provider name
        result[provider][model_id] = model_entry

    metadata_stats = {
        "total_entries": total_entries,
        "total_regions_processed": total_regions_processed,
        "total_groups_created": total_groups_created,
        "group_types_seen": sorted(list(group_types_seen)),
    }

    return result, metadata_stats


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    """
    Lambda handler for pricing aggregation.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "pricingResults": [
                {"status": "SUCCESS", "serviceCode": "AmazonBedrock", "s3Key": "..."},
                ...
            ]
        }

    Output:
        {
            "status": "SUCCESS",
            "s3Key": "executions/{id}/merged/pricing.json",
            "providersCount": 17,
            "totalPricingEntries": 8716
        }
    """
    start_time = time.time()
    collection_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime())

    # Validate required parameters
    try:
        validate_required_params(
            event, ["s3Bucket", "executionId", "pricingResults"], "PricingAggregator"
        )
    except ValidationError as e:
        return {
            "status": "FAILED",
            "errorType": "ValidationError",
            "errorMessage": str(e),
        }

    s3_bucket = event["s3Bucket"]
    execution_id = parse_execution_id(event["executionId"])
    pricing_results = event["pricingResults"]
    dry_run = event.get("dryRun", False)

    output_key = f"executions/{execution_id}/merged/pricing.json"

    logger.info(
        "Starting pricing aggregation", extra={"source_count": len(pricing_results)}
    )

    try:
        s3_client = get_s3_client()

        # Collect all products from successful collectors
        all_products = []
        successful_sources = []

        for item in pricing_results:
            # Handle nested result structure from Map state
            nested_result = item.get("result", {})
            status = item.get("status") or nested_result.get("status")
            s3_key = item.get("s3Key") or nested_result.get("s3Key")
            service_code = item.get("serviceCode")

            if status == "SUCCESS" and s3_key:
                logger.info(
                    "Reading pricing data", extra={"bucket": s3_bucket, "key": s3_key}
                )

                if not dry_run:
                    data = read_from_s3(s3_client, s3_bucket, s3_key)
                    products = data.get("products", [])
                    all_products.extend(products)
                    successful_sources.append(
                        {
                            "service_code": service_code,
                            "s3_key": s3_key,
                            "count": len(products),
                        }
                    )
                    logger.info(
                        "Loaded products",
                        extra={
                            "service_code": service_code,
                            "product_count": len(products),
                        },
                    )
            else:
                logger.warning("Skipping non-successful result", extra={"item": item})

        if dry_run:
            all_products = []

        logger.info(
            "Total products to aggregate", extra={"product_count": len(all_products)}
        )

        # Aggregate pricing data in expected schema
        aggregated, metadata_stats = aggregate_pricing(all_products)

        # Extract GovCloud availability from pricing data
        govcloud_availability = extract_govcloud_availability(all_products)

        # Convert defaultdict to regular dict for JSON serialization
        aggregated = dict(aggregated)

        # Count unique providers (now keys of aggregated since it's nested by provider)
        providers_count = len(aggregated)

        # Build output in expected schema
        output_data = {
            "metadata": {
                "generated_at": collection_timestamp,
                "version": "1.0.0",
                "total_pricing_entries": metadata_stats["total_entries"],
                "data_sources": {
                    "aws_pricing_api": {
                        "success": True,
                        "count": metadata_stats["total_entries"],
                        "error": None,
                    }
                },
                "providers_count": providers_count,
                "total_regions_processed": metadata_stats["total_regions_processed"],
                "total_groups_created": metadata_stats["total_groups_created"],
                "unique_group_types": len(metadata_stats["group_types_seen"]),
                "average_groups_per_region": (
                    metadata_stats["total_groups_created"]
                    / metadata_stats["total_regions_processed"]
                    if metadata_stats["total_regions_processed"] > 0
                    else 0
                ),
                "currency": "USD",
                "pricing_standardization": "Smart conversion applied: per-million to per-thousand when needed, unit extraction from descriptions",
                "structure": "provider > model > region > pricing_groups > dimensions",
                "group_types_available": metadata_stats["group_types_seen"],
                "govcloud_models_count": len(govcloud_availability),
            },
            "providers": aggregated,
            "govcloud_availability": govcloud_availability,
        }

        # Write to S3
        if not dry_run:
            write_to_s3(s3_client, s3_bucket, output_key, output_data)
        else:
            logger.info(
                "Dry run - skipping S3 write",
                extra={"bucket": s3_bucket, "key": output_key},
            )

        duration_ms = int((time.time() - start_time) * 1000)

        # Add metrics
        metrics.add_metric(
            name="PricingEntriesAggregated",
            unit=MetricUnit.Count,
            value=metadata_stats["total_entries"],
        )
        metrics.add_metric(
            name="ProvidersCount", unit=MetricUnit.Count, value=providers_count
        )
        metrics.add_metric(
            name="AggregationDurationMs",
            unit=MetricUnit.Milliseconds,
            value=duration_ms,
        )

        logger.info(
            "Pricing aggregation complete",
            extra={
                "providers_count": providers_count,
                "total_entries": metadata_stats["total_entries"],
                "duration_ms": duration_ms,
            },
        )

        return {
            "status": "SUCCESS",
            "s3Key": output_key,
            "providersCount": providers_count,
            "totalPricingEntries": metadata_stats["total_entries"],
            "durationMs": duration_ms,
        }

    except Exception as e:
        logger.exception("Failed to aggregate pricing", extra={"error": str(e)})
        return {
            "status": "FAILED",
            "errorType": type(e).__name__,
            "errorMessage": str(e),
        }
