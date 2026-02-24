"""
Model Merger Lambda

Merges and deduplicates models collected from multiple regions.
Extracts context window sizes from model ID variants before deduplication.
Works with the correct snake_case schema.
"""

import logging
import os
import re
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
)

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def get_base_model_id(model_id: str) -> str:
    """
    Extract the base model ID by removing context window suffixes.

    Examples:
        'anthropic.claude-3-5-sonnet-20240620-v1:0:18k' -> 'anthropic.claude-3-5-sonnet-20240620-v1:0'
        'anthropic.claude-3-5-sonnet-20240620-v1:0:200k' -> 'anthropic.claude-3-5-sonnet-20240620-v1:0'
        'anthropic.claude-3-5-sonnet-20240620-v1:0' -> 'anthropic.claude-3-5-sonnet-20240620-v1:0'
    """
    # Pattern matches :NNNk at the end (where N is a digit)
    return re.sub(r":\d+k$", "", model_id)


def parse_variant_size(model_id: str) -> int | None:
    """
    Extract context window size from model ID variant suffix.

    Examples:
        'anthropic.claude-3-5-sonnet-20240620-v1:0:200k' -> 200000
        'anthropic.claude-3-5-sonnet-20240620-v1:0:18k'  -> 18000
        'meta.llama3-70b-instruct-v1:0:51k'              -> 51000

    Returns None if no :NNNk suffix found.
    """
    match = re.search(r":(\d+)k$", model_id)
    if match:
        return int(match.group(1)) * 1000
    return None


def merge_models(all_models: list[dict]) -> dict:
    """
    Merge models from multiple regions, deduplicating by model_id.

    Also deduplicates context window variants (e.g., :18k, :200k, :51k)
    by keeping only the base model.

    Preserves the snake_case schema and merges:
    - regions_available: all regions where model exists
    - inference_types_supported: aggregated across all regions
    - on_demand_regions: only regions where ON_DEMAND is supported

    Returns a provider-grouped structure:
    {
        "providers": {
            "Anthropic": {
                "models": {
                    "anthropic.claude-3-sonnet-v1": { ... }
                }
            }
        }
    }
    """
    # Use dict to deduplicate by model_id
    models_by_id = {}
    # Track max context window from size variants (e.g., :200k, :18k)
    variant_context_windows = {}
    variant_customizations = {}
    variant_inference_types = {}  # Track inference types from variants
    variant_on_demand_regions = {}  # Track ON_DEMAND regions from variants
    variant_only_models = {}  # Track variants whose base model doesn't appear

    for model in all_models:
        model_id = model.get("model_id")
        if not model_id:
            continue

        # Get the region(s) this model entry is from
        model_regions = model.get("regions_available", [])
        model_inference_types = model.get("inference_types_supported", [])
        
        # Determine if this model supports ON_DEMAND in its region(s)
        has_on_demand = "ON_DEMAND" in model_inference_types

        # Get base model ID (remove context window suffixes like :18k, :200k)
        base_model_id = get_base_model_id(model_id)

        # Skip context window variants - but extract size info first
        if model_id != base_model_id:
            size_tokens = parse_variant_size(model_id)
            if size_tokens:
                current_max = variant_context_windows.get(base_model_id, 0)
                variant_context_windows[base_model_id] = max(current_max, size_tokens)

            # Merge customization data from variants
            variant_customs = model.get("customization", {}).get(
                "customization_supported", []
            )
            if variant_customs:
                if base_model_id not in variant_customizations:
                    variant_customizations[base_model_id] = set()
                variant_customizations[base_model_id].update(variant_customs)

            # Merge inference_types_supported from variants
            if model_inference_types:
                if base_model_id not in variant_inference_types:
                    variant_inference_types[base_model_id] = set()
                variant_inference_types[base_model_id].update(model_inference_types)

            # Track ON_DEMAND regions from variants
            if has_on_demand:
                if base_model_id not in variant_on_demand_regions:
                    variant_on_demand_regions[base_model_id] = set()
                variant_on_demand_regions[base_model_id].update(model_regions)

            # Track variant data in case base model doesn't exist
            if base_model_id not in variant_only_models:
                variant_only_models[base_model_id] = model.copy()
                variant_only_models[base_model_id]["model_id"] = base_model_id

            logger.debug(
                f"Skipping context variant: {model_id} (base: {base_model_id})"
            )
            continue

        # Keep first occurrence or merge regions_available
        if model_id not in models_by_id:
            models_by_id[model_id] = model.copy()
            # Ensure regions_available is a list
            if "regions_available" not in models_by_id[model_id]:
                models_by_id[model_id]["regions_available"] = []
            # Initialize on_demand_regions
            if has_on_demand:
                models_by_id[model_id]["on_demand_regions"] = list(model_regions)
            else:
                models_by_id[model_id]["on_demand_regions"] = []
        else:
            # Merge regions_available
            existing_regions = set(models_by_id[model_id].get("regions_available", []))
            new_regions = set(model_regions)
            merged_regions = sorted(list(existing_regions | new_regions))
            models_by_id[model_id]["regions_available"] = merged_regions

            # Merge inference_types_supported (critical: varies by region)
            existing_inference_types = set(
                models_by_id[model_id].get("inference_types_supported", [])
            )
            new_inference_types = set(model_inference_types)
            merged_inference_types = sorted(list(existing_inference_types | new_inference_types))
            models_by_id[model_id]["inference_types_supported"] = merged_inference_types

            # Merge on_demand_regions (only regions where ON_DEMAND is supported)
            if has_on_demand:
                existing_od_regions = set(models_by_id[model_id].get("on_demand_regions", []))
                merged_od_regions = sorted(list(existing_od_regions | new_regions))
                models_by_id[model_id]["on_demand_regions"] = merged_od_regions

            # Update collection_metadata.regions_collected_from
            existing_collected = set(
                models_by_id[model_id]
                .get("collection_metadata", {})
                .get("regions_collected_from", [])
            )
            new_collected = set(
                model.get("collection_metadata", {}).get("regions_collected_from", [])
            )
            merged_collected = sorted(list(existing_collected | new_collected))
            if "collection_metadata" not in models_by_id[model_id]:
                models_by_id[model_id]["collection_metadata"] = {}
            models_by_id[model_id]["collection_metadata"]["regions_collected_from"] = (
                merged_collected
            )

            # Merge console_metadata: keep first non-empty across regions
            existing_console_meta = models_by_id[model_id].get("console_metadata")
            new_console_meta = model.get("console_metadata")
            if not existing_console_meta and new_console_meta:
                models_by_id[model_id]["console_metadata"] = new_console_meta

    # Create base model entries for variant-only models (no base in API)
    for base_id, variant_model in variant_only_models.items():
        if base_id not in models_by_id:
            models_by_id[base_id] = variant_model
            if "regions_available" not in models_by_id[base_id]:
                models_by_id[base_id]["regions_available"] = []
            if "on_demand_regions" not in models_by_id[base_id]:
                models_by_id[base_id]["on_demand_regions"] = []
            logger.info(f"Created base model from variant: {base_id}")

    # Attach variant context windows to base models
    for model_id, max_size in variant_context_windows.items():
        if model_id in models_by_id:
            models_by_id[model_id]["variant_context_window"] = max_size
            logger.info(f"Variant context window for {model_id}: {max_size}")

    # Merge customization data from variants into base models
    for model_id, customs in variant_customizations.items():
        if model_id in models_by_id:
            existing_customs = set(
                models_by_id[model_id]
                .get("customization", {})
                .get("customization_supported", [])
            )
            merged = sorted(list(existing_customs | customs))
            if "customization" not in models_by_id[model_id]:
                models_by_id[model_id]["customization"] = {}
            models_by_id[model_id]["customization"]["customization_supported"] = merged
            logger.info(f"Merged customizations for {model_id}: {merged}")

    # Merge inference_types_supported from variants into base models
    for model_id, inf_types in variant_inference_types.items():
        if model_id in models_by_id:
            existing_inf_types = set(
                models_by_id[model_id].get("inference_types_supported", [])
            )
            merged = sorted(list(existing_inf_types | inf_types))
            models_by_id[model_id]["inference_types_supported"] = merged
            logger.info(f"Merged inference types for {model_id}: {merged}")

    # Merge on_demand_regions from variants into base models
    for model_id, od_regions in variant_on_demand_regions.items():
        if model_id in models_by_id:
            existing_od_regions = set(
                models_by_id[model_id].get("on_demand_regions", [])
            )
            merged = sorted(list(existing_od_regions | od_regions))
            models_by_id[model_id]["on_demand_regions"] = merged
            logger.info(f"Merged on_demand_regions for {model_id}: {merged}")

    # Group by provider
    providers = {}
    for model_id, model in models_by_id.items():
        provider = model.get("model_provider", "Unknown")

        if provider not in providers:
            providers[provider] = {"models": {}}

        providers[provider]["models"][model_id] = model

    return providers


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for model merging.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "modelResults": [
                {"status": "SUCCESS", "region": "us-east-1", "s3Key": "..."},
                ...
            ]
        }

    Output:
        {
            "status": "SUCCESS",
            "s3Key": "executions/{id}/merged/models.json",
            "totalModels": 108,
            "providersCount": 17
        }
    """
    start_time = time.time()

    # Validate required parameters
    try:
        validate_required_params(
            event, ["s3Bucket", "executionId", "modelResults"], "ModelMerger"
        )
    except ValidationError as e:
        return {
            "status": "FAILED",
            "errorType": "ValidationError",
            "errorMessage": str(e),
        }

    s3_bucket = event["s3Bucket"]
    execution_id = parse_execution_id(event["executionId"])
    model_results = event["modelResults"]
    dry_run = event.get("dryRun", False)

    output_key = f"executions/{execution_id}/merged/models.json"

    logger.info(f"Merging models from {len(model_results)} regions")

    try:
        s3_client = get_s3_client()

        # Collect all models from successful extractors
        all_models = []
        regions_processed = []

        for item in model_results:
            # Handle nested result structure from Map state
            # Successful: { region, result: { status, s3Key } }
            # Failed: { status: "FAILED", region, error }
            nested_result = item.get("result", {})
            status = item.get("status") or nested_result.get("status")
            s3_key = item.get("s3Key") or nested_result.get("s3Key")
            region = item.get("region")

            if status == "SUCCESS" and s3_key:
                if not dry_run:
                    data = read_from_s3(s3_client, s3_bucket, s3_key)
                    models = data.get("models", [])
                    all_models.extend(models)
                    regions_processed.append(region)
                    logger.info(f"Loaded {len(models)} models from {region}")
            else:
                logger.warning(f"Skipping non-successful result: {item}")

        # Merge and deduplicate
        providers = merge_models(all_models)

        # Calculate statistics
        providers_count = len(providers)
        total_models = sum(len(p["models"]) for p in providers.values())

        output_data = {
            "metadata": {
                "total_models": total_models,
                "providers_count": providers_count,
                "regions_processed": regions_processed,
                "collection_timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            },
            "providers": providers,
        }

        if not dry_run:
            write_to_s3(s3_client, s3_bucket, output_key, output_data)
        else:
            logger.info(f"Dry run - would write to s3://{s3_bucket}/{output_key}")

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            "status": "SUCCESS",
            "s3Key": output_key,
            "totalModels": total_models,
            "providersCount": providers_count,
            "durationMs": duration_ms,
        }

    except Exception as e:
        logger.error(f"Failed to merge models: {e}", exc_info=True)
        return {
            "status": "FAILED",
            "errorType": type(e).__name__,
            "errorMessage": str(e),
        }
