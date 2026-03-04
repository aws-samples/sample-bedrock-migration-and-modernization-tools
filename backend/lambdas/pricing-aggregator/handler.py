"""
Pricing Aggregator Lambda

Merges pricing data from all three Bedrock service codes into a unified structure.
Transforms data to match the expected frontend schema with pricing_groups.
"""

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
    usage_lower = usage_type.lower()
    inference_lower = inference_type.lower() if inference_type else ""
    description_lower = description.lower() if description else ""

    # Check for global (cross-region worldwide)
    is_global = "global" in usage_lower or "global" in description_lower

    # Check for geo/regional (cross-region within geographic area)
    # "Regional CRIS" in description, "_Geo" suffix in dimension, or "regional" in usage
    is_geo = (
        "regional" in usage_lower
        or "_geo" in usage_lower
        or "regional cris" in description_lower
    ) and not is_global

    # Check for batch
    is_batch = "batch" in usage_lower

    # Check for long context - includes _lctx suffix used in newer AWS format
    is_long_context = (
        "long-context" in usage_lower
        or "long context" in inference_lower
        or "_lctx" in usage_lower  # New AWS format: USE1_InputTokenCount_LCtx
        or "longcontext" in usage_lower
    )

    # Check for provisioned/reserved capacity
    # Includes Reserved_1Month, Reserved_3Month patterns and _tpm_ (tokens per minute)
    is_provisioned = (
        "provisioned" in usage_lower
        or "provisioned" in inference_lower
        or "reserved" in usage_lower
        or "_tpm_" in usage_lower  # Reserved TPM pricing
    )

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
            'group': 'On-Demand' | 'Batch' | 'Provisioned Throughput' | 'Custom Model',
            'dimensions': {
                'source': 'standard' | 'mantle',
                'geo': None | 'regional' | 'global',
                'tier': None | 'flex' | 'priority',
                'context': 'standard' | 'long'
            }
        }
    """
    usage_lower = usage_type.lower()
    inference_lower = inference_type.lower() if inference_type else ""
    description_lower = description.lower() if description else ""

    # Initialize dimensions
    dimensions = {
        "source": "standard",
        "geo": None,
        "tier": None,
        "context": "standard",
    }

    # Detect Mantle source
    # Mantle pricing typically has "mantle" in usage type or specific patterns
    if "mantle" in usage_lower or "openai-compatible" in inference_lower:
        dimensions["source"] = "mantle"

    # Detect geographic dimension
    if "global" in usage_lower or "global" in description_lower:
        dimensions["geo"] = "global"
    elif (
        "regional" in usage_lower
        or "_geo" in usage_lower
        or "regional cris" in description_lower
    ):
        dimensions["geo"] = "regional"

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
        dimensions["context"] = "long"

    # Determine base group (simplified - no context/geo modifiers)
    is_batch = "batch" in usage_lower
    is_provisioned = (
        "provisioned" in usage_lower
        or "provisioned" in inference_lower
        or "reserved" in usage_lower
        or "_tpm_" in usage_lower
    )
    is_custom = "custom" in usage_lower or "fine-tun" in usage_lower

    if is_custom:
        group = "Custom Model"
    elif is_provisioned:
        group = "Provisioned Throughput"
    elif is_batch:
        group = "Batch"
    else:
        group = "On-Demand"

    return {"group": group, "dimensions": dimensions}


def aggregate_dimensions(pricing_entries: list) -> dict:
    """Aggregate available dimensions from all pricing entries.

    Args:
        pricing_entries: List of pricing entry dicts with 'dimensions' field

    Returns:
        {
            'sources': ['standard', 'mantle'],
            'geos': ['global', 'regional'],
            'tiers': ['flex', 'priority'],
            'contexts': ['standard', 'long']
        }
    """
    sources = set()
    geos = set()
    tiers = set()
    contexts = set()

    for entry in pricing_entries:
        dims = entry.get("dimensions", {})
        if dims.get("source"):
            sources.add(dims["source"])
        if dims.get("geo"):
            geos.add(dims["geo"])
        if dims.get("tier"):
            tiers.add(dims["tier"])
        if dims.get("context"):
            contexts.add(dims["context"])

    return {
        "sources": sorted(list(sources)) if sources else ["standard"],
        "geos": sorted(list(geos)) if geos else [],
        "tiers": sorted(list(tiers)) if tiers else [],
        "contexts": sorted(list(contexts)) if contexts else ["standard"],
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
    is_per_million = (
        "per 1m" in desc_lower
        or "million" in desc_lower
        or "per 1,000,000" in desc_lower
        or "1000000" in desc_lower
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
            },
            "providers": aggregated,
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
