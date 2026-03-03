"""
Model Merger Lambda

Merges and deduplicates models collected from multiple regions.
Extracts context window sizes from model ID variants before deduplication.
Works with the correct snake_case schema.
"""

import os
import re
import time

from shared import (
    get_s3_client,
    read_from_s3,
    write_to_s3,
    parse_execution_id,
    validate_required_params,
    ValidationError,
    S3ReadError,
)
from shared.model_matcher import get_model_variant_info
from shared.powertools import logger, tracer, metrics, LambdaContext
from aws_lambda_powertools.metrics import MetricUnit


def get_base_model_id(model_id: str) -> str:
    """
    Extract the base model ID by removing context window and variant suffixes.

    Uses the centralized model_matcher utility for consistent behavior across
    all pipeline components.

    Examples:
        'anthropic.claude-3-5-sonnet-20240620-v1:0:18k' -> 'anthropic.claude-3-5-sonnet-20240620-v1:0'
        'amazon.nova-premier-v1:0:mm' -> 'amazon.nova-premier-v1:0'
        'amazon.nova-reel-v1:1' -> 'amazon.nova-reel-v1:0'
        'amazon.titan-embed-image-v1' -> 'amazon.titan-embed-image-v1:0'
    """
    variant_info = get_model_variant_info(model_id)
    base_id = variant_info.get("base_id", model_id)

    # Normalize version suffix (:1, :2, etc.) to :0
    # Only if the model ends with :N where N is a single digit
    base_id = re.sub(r":([1-9])$", ":0", base_id)

    # Add :0 if model doesn't have a version suffix at all
    # Check if model ends with :\d+ pattern
    if not re.search(r":\d+$", base_id):
        base_id = f"{base_id}:0"

    return base_id


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
    - extraction_regions: where model was discovered in API (audit/debug only)
    - inference_types_supported: aggregated across all regions

    Note: Actual ON_DEMAND availability (in_region) is determined by the
    regional-availability Lambda, not here. This merger only tracks where
    models were discovered in the API.

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
    variant_only_models = {}  # Track variants whose base model doesn't appear

    for model in all_models:
        model_id = model.get("model_id")
        if not model_id:
            continue

        # Get the region(s) this model entry is from (extraction regions, not availability)
        model_regions = model.get("extraction_regions", [])
        model_inference_types = model.get("inference_types_supported", [])

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

            # Track variant data in case base model doesn't exist
            if base_model_id not in variant_only_models:
                variant_only_models[base_model_id] = model.copy()
                variant_only_models[base_model_id]["model_id"] = base_model_id

            logger.debug(
                "Skipping context variant",
                extra={"model_id": model_id, "base_model_id": base_model_id},
            )
            continue

        # Keep first occurrence or merge extraction_regions
        if model_id not in models_by_id:
            models_by_id[model_id] = model.copy()
            # Ensure extraction_regions is a list
            if "extraction_regions" not in models_by_id[model_id]:
                models_by_id[model_id]["extraction_regions"] = []
        else:
            # Merge extraction_regions
            existing_regions = set(models_by_id[model_id].get("extraction_regions", []))
            new_regions = set(model_regions)
            merged_regions = sorted(list(existing_regions | new_regions))
            models_by_id[model_id]["extraction_regions"] = merged_regions

            # Merge inference_types_supported (critical: varies by region)
            existing_inference_types = set(
                models_by_id[model_id].get("inference_types_supported", [])
            )
            new_inference_types = set(model_inference_types)
            merged_inference_types = sorted(
                list(existing_inference_types | new_inference_types)
            )
            models_by_id[model_id]["inference_types_supported"] = merged_inference_types

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
            if "extraction_regions" not in models_by_id[base_id]:
                models_by_id[base_id]["extraction_regions"] = []
            logger.info(
                "Created base model from variant", extra={"base_model_id": base_id}
            )

    # Attach variant context windows to base models
    for model_id, max_size in variant_context_windows.items():
        if model_id in models_by_id:
            models_by_id[model_id]["variant_context_window"] = max_size
            logger.info(
                "Variant context window attached",
                extra={"model_id": model_id, "max_size": max_size},
            )

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
            logger.info(
                "Merged customizations",
                extra={"model_id": model_id, "customizations": merged},
            )

    # Merge inference_types_supported from variants into base models
    for model_id, inf_types in variant_inference_types.items():
        if model_id in models_by_id:
            existing_inf_types = set(
                models_by_id[model_id].get("inference_types_supported", [])
            )
            merged = sorted(list(existing_inf_types | inf_types))
            models_by_id[model_id]["inference_types_supported"] = merged
            logger.info(
                "Merged inference types",
                extra={"model_id": model_id, "inference_types": merged},
            )

    # Group by provider
    providers = {}
    for model_id, model in models_by_id.items():
        provider = model.get("model_provider", "Unknown")

        if provider not in providers:
            providers[provider] = {"models": {}}

        providers[provider]["models"][model_id] = model

    return providers


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    """
    Lambda handler for model merging.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "modelResults": [
                {"status": "SUCCESS", "region": "us-east-1", "s3Key": "...", "cacheKey": "..."},
                ...
            ]
        }

    Output:
        {
            "status": "SUCCESS",
            "s3Key": "executions/{id}/merged/models.json",
            "totalModels": 108,
            "providersCount": 17,
            "cacheKeys": {"us-east-1": "executions/{id}/cache/list_foundation_models_us-east-1.json", ...}
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

    logger.info("Starting model merge", extra={"region_count": len(model_results)})

    try:
        s3_client = get_s3_client()

        # Collect all models from successful extractors
        all_models = []
        regions_processed = []
        cache_keys = {}  # Collect cache keys for downstream use

        for item in model_results:
            # Handle nested result structure from Map state
            # Successful: { region, result: { status, s3Key, cacheKey } }
            # Failed: { status: "FAILED", region, error }
            nested_result = item.get("result", {})
            status = item.get("status") or nested_result.get("status")
            s3_key = item.get("s3Key") or nested_result.get("s3Key")
            cache_key = item.get("cacheKey") or nested_result.get("cacheKey")
            region = item.get("region")

            if status == "SUCCESS" and s3_key:
                if not dry_run:
                    data = read_from_s3(s3_client, s3_bucket, s3_key)
                    models = data.get("models", [])
                    all_models.extend(models)
                    regions_processed.append(region)
                    logger.info(
                        "Loaded models from region",
                        extra={"model_count": len(models), "region": region},
                    )
                # Collect cache key if available
                if cache_key and region:
                    cache_keys[region] = cache_key
            else:
                logger.warning("Skipping non-successful result", extra={"item": item})

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
            logger.info(
                "Dry run - skipping S3 write",
                extra={"bucket": s3_bucket, "key": output_key},
            )

        duration_ms = int((time.time() - start_time) * 1000)

        # Add metrics
        metrics.add_metric(
            name="ModelsMerged", unit=MetricUnit.Count, value=total_models
        )
        metrics.add_metric(
            name="ProvidersCount", unit=MetricUnit.Count, value=providers_count
        )
        metrics.add_metric(
            name="MergeDurationMs", unit=MetricUnit.Milliseconds, value=duration_ms
        )

        logger.info(
            "Model merge complete",
            extra={
                "total_models": total_models,
                "providers_count": providers_count,
                "duration_ms": duration_ms,
                "cache_keys_count": len(cache_keys),
            },
        )

        result = {
            "status": "SUCCESS",
            "s3Key": output_key,
            "totalModels": total_models,
            "providersCount": providers_count,
            "durationMs": duration_ms,
        }

        # Include cache keys for downstream use (e.g., regional-availability)
        if cache_keys:
            result["cacheKeys"] = cache_keys

        return result

    except Exception as e:
        logger.exception("Failed to merge models", extra={"error": str(e)})
        return {
            "status": "FAILED",
            "errorType": type(e).__name__,
            "errorMessage": str(e),
        }
