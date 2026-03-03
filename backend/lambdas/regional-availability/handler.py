"""
Regional Availability Lambda

Discovers model availability across all AWS regions using the Bedrock API
(ListFoundationModels) with explicit inference-type filtering.

Why ON_DEMAND filtering?
    An unfiltered ListFoundationModels call returns models across all inference
    types (ON_DEMAND, PROVISIONED, INFERENCE_PROFILE).  This inflates the
    availability map with ~572 false positives — models that exist in the API
    catalogue but cannot actually be invoked on-demand.  Filtering with
    byInferenceType='ON_DEMAND' produces a list that matches 100% with actual
    Converse-API invocability (verified empirically).

Why no pricing data?
    Pricing data was previously unioned into the availability map, but
    investigation showed it adds ~130 phantom model IDs that use pricing-
    specific identifiers (e.g. region-prefixed names) rather than real Bedrock
    model IDs.  These never resolve to invocable models.  Removing the pricing
    union eliminates all false positives with zero loss of genuine coverage.

INFERENCE_PROFILE models are captured separately via the feature-collector
Lambda (ListInferenceProfiles / CRIS), so they are not lost.

Configuration (environment variables):
    AVAILABILITY_MAX_WORKERS: Thread pool size for parallel region queries (default: 10)
    AVAILABILITY_REGION_TIMEOUT: Timeout for region queries in seconds (default: 30)
"""

import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config

from shared import (
    get_s3_client,
    write_to_s3,
    parse_execution_id,
    validate_required_params,
    ValidationError,
    get_cached_models,
    is_cache_valid,
)
from shared.powertools import logger, tracer, metrics, LambdaContext
from aws_lambda_powertools.metrics import MetricUnit

# Configuration with defaults
AVAILABILITY_MAX_WORKERS = int(os.environ.get("AVAILABILITY_MAX_WORKERS", "10"))
AVAILABILITY_REGION_TIMEOUT = int(os.environ.get("AVAILABILITY_REGION_TIMEOUT", "30"))

RETRY_CONFIG = Config(
    retries={"max_attempts": 3, "mode": "adaptive"},
    connect_timeout=5,
    read_timeout=AVAILABILITY_REGION_TIMEOUT,
)


@tracer.capture_method
def _get_models_from_cache(
    s3_client, bucket: str, cache_key: str, region: str
) -> tuple[list[str], list[str], bool]:
    """
    Get model IDs from cached ListFoundationModels response.

    The cache contains unfiltered model summaries, so we need to filter
    by inference type here to match the API behavior.

    Args:
        s3_client: S3 client
        bucket: S3 bucket name
        cache_key: S3 key for cached data
        region: Region name (for logging)

    Returns:
        Tuple of (on_demand_model_ids, provisioned_model_ids, cache_hit)
    """
    cached_data = get_cached_models(s3_client, bucket, cache_key)

    if not cached_data or not is_cache_valid(cached_data):
        logger.debug(
            "Cache miss or invalid",
            extra={"region": region, "cache_key": cache_key},
        )
        return [], [], False

    model_summaries = cached_data.get("model_summaries", [])
    on_demand_models = []
    provisioned_models = []

    for model in model_summaries:
        model_id = model.get("modelId")
        if not model_id:
            continue

        inference_types = model.get("inferenceTypesSupported", [])

        if "ON_DEMAND" in inference_types:
            on_demand_models.append(model_id)
        if "PROVISIONED" in inference_types:
            provisioned_models.append(model_id)

    logger.info(
        "Cache hit",
        extra={
            "region": region,
            "on_demand_count": len(on_demand_models),
            "provisioned_count": len(provisioned_models),
        },
    )

    return on_demand_models, provisioned_models, True


@tracer.capture_method
def _discover_via_api(
    regions: list,
    s3_client=None,
    bucket: str = None,
    cache_keys: dict = None,
) -> tuple:
    """
    Call ListFoundationModels across all regions in parallel, once for
    ON_DEMAND models and once for PROVISIONED models.

    Uses cached data from model-extractor when available to reduce API calls.

    Args:
        regions: List of AWS regions to query
        s3_client: Optional S3 client for reading cache
        bucket: Optional S3 bucket name for cache
        cache_keys: Optional dict mapping region to cache S3 key

    Returns:
        on_demand_availability:   {model_id: set(regions)}
        provisioned_availability: {model_id: set(regions)}
        region_stats:             {region: {on_demand_count, provisioned_count, error, from_cache}}
        cache_hits:               Number of regions served from cache
        api_calls:                Number of regions that required API calls
    """
    on_demand_availability = defaultdict(set)
    provisioned_availability = defaultdict(set)
    region_stats = {}
    cache_keys = cache_keys or {}

    # Separate cached and uncached regions
    cached_regions = [r for r in regions if r in cache_keys]
    uncached_regions = [r for r in regions if r not in cache_keys]

    cache_hits = 0
    api_calls = 0

    logger.info(
        "Processing regions",
        extra={
            "cached_regions": len(cached_regions),
            "uncached_regions": len(uncached_regions),
            "total_regions": len(regions),
        },
    )

    # Process cached regions first (fast, no rate limiting needed)
    if s3_client and bucket:
        for region in cached_regions:
            cache_key = cache_keys[region]
            od_models, prov_models, hit = _get_models_from_cache(
                s3_client, bucket, cache_key, region
            )

            if hit:
                cache_hits += 1
                region_stats[region] = {
                    "on_demand_count": len(od_models),
                    "provisioned_count": len(prov_models),
                    "error": None,
                    "from_cache": True,
                }
                for mid in od_models:
                    on_demand_availability[mid].add(region)
                for mid in prov_models:
                    provisioned_availability[mid].add(region)
            else:
                # Cache miss - add to uncached regions for API call
                uncached_regions.append(region)

    @tracer.capture_method
    def query_region(region: str):
        """Query a single region for both ON_DEMAND and PROVISIONED models."""
        try:
            client = boto3.client("bedrock", region_name=region, config=RETRY_CONFIG)

            # ON_DEMAND: models that can be invoked directly via Converse / InvokeModel
            od_response = client.list_foundation_models(byInferenceType="ON_DEMAND")
            od_models = [
                m["modelId"]
                for m in od_response.get("modelSummaries", [])
                if "modelId" in m
            ]

            # PROVISIONED: models available for Provisioned Throughput
            prov_response = client.list_foundation_models(byInferenceType="PROVISIONED")
            prov_models = [
                m["modelId"]
                for m in prov_response.get("modelSummaries", [])
                if "modelId" in m
            ]

            return region, od_models, prov_models, None
        except Exception as e:
            logger.warning(
                "Failed to query region", extra={"region": region, "error": str(e)}
            )
            return region, [], [], str(e)

    # Process uncached regions with API calls
    if uncached_regions:
        with ThreadPoolExecutor(max_workers=AVAILABILITY_MAX_WORKERS) as executor:
            futures = {executor.submit(query_region, r): r for r in uncached_regions}
            for future in as_completed(futures):
                region, od_models, prov_models, error = future.result()
                api_calls += 1
                region_stats[region] = {
                    "on_demand_count": len(od_models),
                    "provisioned_count": len(prov_models),
                    "error": error,
                    "from_cache": False,
                }
                for mid in od_models:
                    on_demand_availability[mid].add(region)
                for mid in prov_models:
                    provisioned_availability[mid].add(region)

    successful = sum(1 for s in region_stats.values() if s["error"] is None)
    logger.info(
        "Discovery complete",
        extra={
            "on_demand_models": len(on_demand_availability),
            "provisioned_models": len(provisioned_availability),
            "successful_regions": successful,
            "total_regions": len(regions),
            "cache_hits": cache_hits,
            "api_calls": api_calls,
        },
    )

    return (
        on_demand_availability,
        provisioned_availability,
        region_stats,
        cache_hits,
        api_calls,
    )


def _build_availability_output(
    on_demand_availability: dict,
    provisioned_availability: dict,
) -> dict:
    """
    Build the final availability output from API discovery data.

    The primary ``model_availability`` field contains ON_DEMAND models only —
    these are the models that can actually be invoked.  A separate
    ``provisioned_availability`` field captures models available for
    Provisioned Throughput.
    """
    # --- Region summary (based on on-demand models, the primary use-case) ---
    regions_summary = defaultdict(
        lambda: {
            "bedrock_available": True,
            "models_in_region": 0,
            "providers": set(),
        }
    )

    for model_id, regions in on_demand_availability.items():
        provider = model_id.split(".")[0].capitalize() if "." in model_id else "Unknown"
        for region in regions:
            regions_summary[region]["models_in_region"] += 1
            regions_summary[region]["providers"].add(provider)

    result_regions = {}
    for region, data in regions_summary.items():
        result_regions[region] = {
            "bedrock_available": data["bedrock_available"],
            "models_in_region": data["models_in_region"],
            "providers": sorted(list(data["providers"])),
            "model_count": data["models_in_region"],
        }

    # Sort region lists for deterministic output
    model_availability = {
        mid: sorted(list(regs)) for mid, regs in on_demand_availability.items()
    }
    prov_availability = {
        mid: sorted(list(regs)) for mid, regs in provisioned_availability.items()
    }

    return {
        "regions": result_regions,
        "model_availability": model_availability,
        "provisioned_availability": prov_availability,
    }


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    """
    Lambda handler for regional availability computation.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "regions": ["us-east-1", "us-west-2", ...],
            "cacheKeys": {"us-east-1": "executions/.../cache/...", ...},  # optional
            "pricingS3Key": "..."   # accepted for backward compat, ignored
        }

    Output:
        {
            "status": "SUCCESS",
            "s3Key": "executions/{id}/intermediate/regional-availability.json",
            "regionsWithBedrock": 27,
            "cacheHits": 2,
            "apiCalls": 25,
            "cacheHitRate": 7.4
        }
    """
    start_time = time.time()

    # Only s3Bucket and executionId are truly required now.
    # pricingS3Key may still be passed by the state machine — accept but ignore.
    try:
        validate_required_params(
            event, ["s3Bucket", "executionId"], "RegionalAvailability"
        )
    except ValidationError as e:
        return {
            "status": "FAILED",
            "errorType": "ValidationError",
            "errorMessage": str(e),
        }

    s3_bucket = event["s3Bucket"]
    execution_id = parse_execution_id(event["executionId"])
    regions = event.get("regions", [])
    cache_keys = event.get("cacheKeys", {})  # Cache keys from model-extractor
    dry_run = event.get("dryRun", False)

    # Log if pricingS3Key was passed (backward compat — no longer used)
    if "pricingS3Key" in event:
        logger.info(
            "pricingS3Key provided but no longer used",
            extra={
                "note": "pricing data excluded from availability (see module docstring)"
            },
        )

    output_key = f"executions/{execution_id}/intermediate/regional-availability.json"

    logger.info(
        "Starting regional availability check",
        extra={
            "region_count": len(regions),
            "cache_keys_provided": len(cache_keys),
        },
    )

    try:
        s3_client = get_s3_client()

        cache_hits = 0
        api_calls = 0

        if not dry_run:
            # Discover models via filtered ListFoundationModels calls
            # Uses cached data from model-extractor when available
            on_demand = {}
            provisioned = {}
            region_stats = {}
            if regions:
                (
                    on_demand,
                    provisioned,
                    region_stats,
                    cache_hits,
                    api_calls,
                ) = _discover_via_api(
                    regions,
                    s3_client=s3_client,
                    bucket=s3_bucket,
                    cache_keys=cache_keys,
                )

            # Build unified output (on-demand primary, provisioned secondary)
            availability = _build_availability_output(on_demand, provisioned)

            # Calculate cache hit rate
            cache_hit_rate = (
                round(cache_hits / len(regions) * 100, 1) if regions else 0.0
            )

            output_data = {
                "metadata": {
                    "regions_with_bedrock": len(availability["regions"]),
                    "total_models_tracked": len(availability["model_availability"]),
                    "total_provisioned_models": len(
                        availability["provisioned_availability"]
                    ),
                    "api_regions_queried": api_calls,
                    "cache_hits": cache_hits,
                    "cache_hit_rate": cache_hit_rate,
                    "collection_timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "discovery_method": "api_on_demand_filtered_with_cache",
                },
                "region_summary": availability["regions"],
                "model_availability": availability["model_availability"],
                "provisioned_availability": availability["provisioned_availability"],
            }

            write_to_s3(s3_client, s3_bucket, output_key, output_data)
            regions_count = len(availability["regions"])
        else:
            logger.info("Dry run - skipping processing")
            regions_count = 0

        duration_ms = int((time.time() - start_time) * 1000)

        # Emit metrics
        metrics.add_metric(
            name="RegionsChecked", unit=MetricUnit.Count, value=len(regions)
        )
        metrics.add_metric(
            name="RegionsWithBedrock", unit=MetricUnit.Count, value=regions_count
        )
        metrics.add_metric(name="CacheHits", unit=MetricUnit.Count, value=cache_hits)
        metrics.add_metric(name="ApiCalls", unit=MetricUnit.Count, value=api_calls)

        logger.info(
            "Regional availability check complete",
            extra={
                "regions_with_bedrock": regions_count,
                "cache_hits": cache_hits,
                "api_calls": api_calls,
                "duration_ms": duration_ms,
            },
        )

        return {
            "status": "SUCCESS",
            "s3Key": output_key,
            "regionsWithBedrock": regions_count,
            "cacheHits": cache_hits,
            "apiCalls": api_calls,
            "cacheHitRate": round(cache_hits / len(regions) * 100, 1)
            if regions
            else 0.0,
            "durationMs": duration_ms,
        }

    except Exception as e:
        logger.exception("Failed to compute availability", extra={"error": str(e)})
        return {
            "status": "FAILED",
            "errorType": type(e).__name__,
            "errorMessage": str(e),
        }
