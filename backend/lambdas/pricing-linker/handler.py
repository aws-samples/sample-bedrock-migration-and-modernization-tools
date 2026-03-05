"""
Pricing Linker Lambda - V2 (With PORT Features)

Links pricing data to models, creating price references per model per region.
Works with the correct snake_case schema.

V2 Features (ported from reference implementation):
- Provider-scoped matching: Only matches within same provider
- Conflict detection: Blocks semantic mismatches (haiku/sonnet, 8b/405b)
- Enhanced normalization: Uses centralized model_matcher for edge cases
"""

import time
from typing import Any

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
    get_canonical_model_id,
    calculate_match_score,
    has_semantic_conflict,
)
from shared.powertools import logger, tracer, metrics, LambdaContext
from aws_lambda_powertools.metrics import MetricUnit


def get_provider_aliases() -> dict:
    """Get provider aliases from configuration."""
    config = get_config_loader()
    # Convert list values to sets for efficient lookup
    aliases = config.get_provider_aliases()
    return {k: set(v) for k, v in aliases.items()}


def get_min_confidence_threshold() -> float:
    """Get minimum confidence threshold from configuration."""
    return get_config_loader().get_min_confidence_threshold()


def get_explicit_model_mappings() -> dict:
    """Get explicit model mappings from configuration."""
    config = get_config_loader()
    return config.get_explicit_model_mappings()


def has_on_demand_pricing(pricing_data: dict) -> bool:
    """Check if pricing data has On-Demand pricing in at least one region."""
    if not pricing_data or not isinstance(pricing_data, dict):
        return False

    regions = pricing_data.get("regions", {})
    if not isinstance(regions, dict):
        return False

    for region_data in regions.values():
        if not isinstance(region_data, dict):
            continue
        pricing_groups = region_data.get("pricing_groups", {})
        on_demand = pricing_groups.get("On-Demand", [])
        if on_demand:
            return True
    return False


# =============================================================================
# Direct Match Step (Priority 0 - before fuzzy matching)
# =============================================================================


def try_direct_match(model_id: str, pricing_models: dict) -> tuple[str | None, float]:
    """
    Attempt direct matching before fuzzy matching.

    Direct matching steps:
    1. Exact match on model_id as-is
    2. Exact match without instance version suffix (:0, :1, :2)
    3. Canonical form exact match

    Args:
        model_id: The model identifier to match
        pricing_models: Dict of pricing entries with provider info

    Returns:
        (matched_pricing_key, confidence_score) or (None, 0.0) if no match
    """
    if not model_id or not pricing_models:
        return None, 0.0

    # Step 1: Exact match as-is
    if model_id in pricing_models:
        logger.info(
            "Direct exact match",
            model_id=model_id,
            matched_to=model_id,
            confidence=1.0,
        )
        return model_id, 1.0

    # Step 2: Try without instance version suffix
    # Remove :0, :1, :2 suffixes
    model_id_base = model_id
    if ":" in model_id:
        parts = model_id.rsplit(":", 1)
        if parts[1].isdigit():
            model_id_base = parts[0]
            if model_id_base in pricing_models:
                logger.info(
                    "Direct match without version suffix",
                    model_id=model_id,
                    matched_to=model_id_base,
                    confidence=0.99,
                )
                return model_id_base, 0.99

    # Step 3: Canonical form exact match
    canonical_model = get_canonical_model_id(model_id)
    for pricing_key in pricing_models:
        canonical_pricing = get_canonical_model_id(pricing_key)
        if canonical_model == canonical_pricing:
            logger.info(
                "Canonical exact match",
                model_id=model_id,
                canonical=canonical_model,
                matched_to=pricing_key,
                confidence=0.98,
            )
            return pricing_key, 0.98

    return None, 0.0


# =============================================================================
# PORT Feature 1: Provider-Scoped Matching
# =============================================================================


def extract_provider_from_model_id(model_id: str) -> str:
    """
    Extract provider name from model ID.

    Examples:
        'meta.llama3-8b' -> 'meta'
        'anthropic.claude-3-sonnet' -> 'anthropic'
        'amazon.titan-text-express' -> 'amazon'
    """
    if "." in model_id:
        return model_id.split(".")[0].lower()
    return ""


def providers_match(model_provider: str, pricing_provider: str) -> bool:
    """
    Check if model provider matches pricing provider with alias support.

    Handles variations like:
        - "Stability AI" vs "stability"
        - "AI21 Labs" vs "ai21"
        - "Mistral AI" vs "mistralai"
    """
    if not model_provider or not pricing_provider:
        return False

    model_provider_lower = model_provider.lower().strip()
    pricing_provider_lower = pricing_provider.lower().strip()

    # Direct match
    if model_provider_lower == pricing_provider_lower:
        return True

    # Check alias mappings from config
    provider_aliases = get_provider_aliases()
    for canonical, aliases in provider_aliases.items():
        model_in_aliases = model_provider_lower in aliases or any(
            alias in model_provider_lower or model_provider_lower in alias
            for alias in aliases
        )
        pricing_in_aliases = pricing_provider_lower in aliases or any(
            alias in pricing_provider_lower or pricing_provider_lower in alias
            for alias in aliases
        )
        if model_in_aliases and pricing_in_aliases:
            return True

    return False


# =============================================================================
# PORT Feature 2: Conflict Detection (now using centralized model_matcher)
# =============================================================================

# Note: has_semantic_conflict is now imported from shared.model_matcher


# =============================================================================
# PORT Feature 3: Enhanced Normalization (now using centralized model_matcher)
# =============================================================================


def normalize_model_id(model_id: str, provider: str = "") -> str:
    """
    Normalize model ID for matching using centralized utility.

    This is a wrapper around the centralized get_canonical_model_id() function
    that provides backward compatibility with the existing interface.

    Args:
        model_id: The model identifier to normalize
        provider: Optional provider name (unused, kept for backward compatibility)

    Returns:
        Canonical form of the model ID for matching.
    """
    return get_canonical_model_id(model_id)


def find_best_pricing_match(
    model_id: str, model_name: str, model_provider: str, pricing_models: dict
) -> tuple[str, float]:
    """
    Find the best matching pricing entry for a model.

    Priority order:
    0. Direct exact match (NEW - Step 0)
    1. Explicit mapping (from config) - returns 1.0 confidence
    2. Canonical exact match (NEW - Step 2)
    3. Provider-scoped fuzzy matching with conflict detection

    Features:
        - Direct matching first for performance
        - Provider-scoped matching: Only matches within same provider
        - Conflict detection: Blocks semantic mismatches (using centralized model_matcher)
        - On-Demand prioritization: Prefers entries with On-Demand pricing

    Args:
        model_id: The model identifier
        model_name: Human-readable model name
        model_provider: The model's provider name
        pricing_models: Dict of pricing entries with provider info

    Returns:
        (matched_pricing_key, confidence_score)
    """
    # Priority 0: Try direct match first (NEW)
    direct_match, direct_score = try_direct_match(model_id, pricing_models)
    if direct_match and direct_score >= 0.95:
        return direct_match, direct_score

    # Priority 1: Check explicit mappings
    explicit_mappings = get_explicit_model_mappings()
    if model_id in explicit_mappings:
        mapped_key = explicit_mappings[model_id]
        # Find the mapped key in pricing_models
        for pricing_key, pricing_entry in pricing_models.items():
            canonical_pricing = get_canonical_model_id(pricing_key)
            canonical_mapped = get_canonical_model_id(mapped_key)
            if canonical_pricing == canonical_mapped:
                logger.info(
                    "Explicit mapping match",
                    model_id=model_id,
                    mapped_to=pricing_key,
                    confidence=1.0,
                )
                return pricing_key, 1.0

    # Priority 2-3: Existing fuzzy matching logic
    # Track best matches separately for On-Demand and non-On-Demand
    best_on_demand_match = None
    best_on_demand_score = 0.0
    best_other_match = None
    best_other_score = 0.0

    for pricing_key, pricing_entry in pricing_models.items():
        pricing_data = pricing_entry["data"]
        pricing_provider = pricing_entry["provider"]

        # PORT Feature 1: Provider-scoped matching
        if not providers_match(model_provider, pricing_provider):
            continue

        # PORT Feature 2: Conflict detection using centralized model_matcher
        # The centralized has_semantic_conflict() compares model IDs directly
        if has_semantic_conflict(model_id, pricing_key):
            continue

        # Calculate match score using centralized model_matcher
        # This handles normalization, canonical form comparison, and similarity
        score = calculate_match_score(model_id, pricing_key)

        # Track separately based on whether pricing has On-Demand tier
        if has_on_demand_pricing(pricing_data):
            if score > best_on_demand_score:
                best_on_demand_score = score
                best_on_demand_match = pricing_key
        else:
            if score > best_other_score:
                best_other_score = score
                best_other_match = pricing_key

    # Prefer On-Demand matches if score is reasonable
    if best_on_demand_match and best_on_demand_score >= get_min_confidence_threshold():
        return best_on_demand_match, best_on_demand_score

    # Fall back to other matches if no good On-Demand match
    if best_other_match and best_other_score >= best_on_demand_score:
        return best_other_match, best_other_score

    # Return best On-Demand match even if score is low (as last resort)
    if best_on_demand_match:
        return best_on_demand_match, best_on_demand_score

    return best_other_match, best_other_score if best_other_match else 0.0


def link_pricing_to_models(models_data: dict, pricing_data: dict) -> dict:
    """
    Link pricing information to each model.

    Returns updated models structure with pricing references in correct schema.
    """
    models_with_pricing = 0
    models_without_pricing = 0

    # Flatten pricing models for easier matching, tracking provider for each
    # Structure: { model_key: { 'provider': provider_name, 'data': pricing_data } }
    all_pricing_models = {}
    for provider_name, data in pricing_data.get("providers", {}).items():
        if isinstance(data, dict):
            if "regions" in data:
                # Flat structure: model_id -> {model_name, model_provider, regions}
                all_pricing_models[provider_name] = {
                    "provider": provider_name,
                    "data": data,
                }
            elif "models" in data:
                # Old nested structure: provider -> models -> model_id -> pricing
                for model_key, model_pricing in data.get("models", {}).items():
                    all_pricing_models[model_key] = {
                        "provider": provider_name,
                        "data": model_pricing,
                    }
            else:
                # New nested structure: provider -> model_id -> {model_name, model_provider, regions}
                for model_key, model_pricing in data.items():
                    if isinstance(model_pricing, dict) and "regions" in model_pricing:
                        all_pricing_models[model_key] = {
                            "provider": provider_name,
                            "data": model_pricing,
                        }

    # Process each provider and model
    for provider, provider_data in models_data.get("providers", {}).items():
        for model_id, model in provider_data.get("models", {}).items():
            model_name = model.get("model_name", model_id)
            model_provider = model.get("model_provider", provider)

            # Find matching pricing (now with provider scoping and conflict detection)
            matched_key, confidence = find_best_pricing_match(
                model_id, model_name, model_provider, all_pricing_models
            )

            if matched_key and confidence >= get_min_confidence_threshold():
                pricing_entry = all_pricing_models[matched_key]
                pricing_info = pricing_entry["data"]
                pricing_provider = pricing_entry["provider"]
                pricing_regions = pricing_info.get("regions", {})

                model["model_pricing"] = {
                    "is_pricing_available": True,
                    "pricing_reference_id": matched_key,
                    "pricing_file_reference": {
                        "provider": pricing_provider,
                        "model_key": matched_key,
                    },
                    "confidence": round(confidence, 3),
                    "regions": pricing_regions,
                    "total_regions": len(pricing_regions),
                }
                model["has_pricing"] = True
                models_with_pricing += 1
            else:
                model["model_pricing"] = {
                    "is_pricing_available": False,
                    "pricing_reference_id": None,
                    "pricing_file_reference": None,
                    "confidence": round(confidence, 3) if matched_key else 0,
                    "regions": {},
                    "total_regions": 0,
                }
                model["has_pricing"] = False
                models_without_pricing += 1

    return {
        "models_with_pricing": models_with_pricing,
        "models_without_pricing": models_without_pricing,
        "providers": models_data.get("providers", {}),
    }


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    """
    Lambda handler for pricing linking.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "pricingS3Key": "executions/{id}/merged/pricing.json",
            "modelsS3Key": "executions/{id}/merged/models.json"
        }

    Output:
        {
            "status": "SUCCESS",
            "s3Key": "executions/{id}/intermediate/models-with-pricing.json",
            "modelsWithPricing": 86,
            "modelsWithoutPricing": 22
        }
    """
    start_time = time.time()

    # Validate required parameters
    try:
        validate_required_params(
            event,
            ["s3Bucket", "executionId", "pricingS3Key", "modelsS3Key"],
            "PricingLinker",
        )
    except ValidationError as e:
        return {
            "status": "FAILED",
            "errorType": "ValidationError",
            "errorMessage": str(e),
        }

    s3_bucket = event["s3Bucket"]
    execution_id = parse_execution_id(event["executionId"])
    pricing_s3_key = event["pricingS3Key"]
    models_s3_key = event["modelsS3Key"]
    dry_run = event.get("dryRun", False)

    output_key = f"executions/{execution_id}/intermediate/models-with-pricing.json"

    logger.info("Starting pricing linking", extra={"version": "v2-port-features"})

    try:
        s3_client = get_s3_client()

        if not dry_run:
            # Read pricing and models data
            pricing_data = read_from_s3(s3_client, s3_bucket, pricing_s3_key)
            models_data = read_from_s3(s3_client, s3_bucket, models_s3_key)

            # Link pricing to models
            result = link_pricing_to_models(models_data, pricing_data)

            output_data = {
                "metadata": {
                    "models_with_pricing": result["models_with_pricing"],
                    "models_without_pricing": result["models_without_pricing"],
                    "version": "v2-port-features",
                    "collection_timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                },
                "providers": result["providers"],
            }

            write_to_s3(s3_client, s3_bucket, output_key, output_data)

            models_with_pricing = result["models_with_pricing"]
            models_without_pricing = result["models_without_pricing"]
        else:
            logger.info("Dry run - skipping processing")
            models_with_pricing = 0
            models_without_pricing = 0

        duration_ms = int((time.time() - start_time) * 1000)

        # Emit metrics
        metrics.add_metric(
            name="ModelsLinked", unit=MetricUnit.Count, value=models_with_pricing
        )
        metrics.add_metric(
            name="ModelsUnlinked", unit=MetricUnit.Count, value=models_without_pricing
        )

        logger.info(
            "Pricing linking complete",
            extra={
                "linked_count": models_with_pricing,
                "unlinked_count": models_without_pricing,
                "duration_ms": duration_ms,
            },
        )

        return {
            "status": "SUCCESS",
            "s3Key": output_key,
            "modelsWithPricing": models_with_pricing,
            "modelsWithoutPricing": models_without_pricing,
            "durationMs": duration_ms,
        }

    except Exception as e:
        logger.exception("Failed to link pricing", extra={"error": str(e)})
        return {
            "status": "FAILED",
            "errorType": type(e).__name__,
            "errorMessage": str(e),
        }
