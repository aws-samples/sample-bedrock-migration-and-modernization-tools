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
"""

import logging
import os
import time
from typing import Any
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
)

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

RETRY_CONFIG = Config(
    retries={"max_attempts": 3, "mode": "adaptive"},
    connect_timeout=5,
    read_timeout=30,
)


def _discover_via_api(regions: list) -> tuple:
    """
    Call ListFoundationModels across all regions in parallel, once for
    ON_DEMAND models and once for PROVISIONED models.

    Returns:
        on_demand_availability:   {model_id: set(regions)}
        provisioned_availability: {model_id: set(regions)}
        region_stats:             {region: {on_demand_count, provisioned_count, error}}
    """
    on_demand_availability = defaultdict(set)
    provisioned_availability = defaultdict(set)
    region_stats = {}

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
            logger.warning(f"Failed to query region {region}: {e}")
            return region, [], [], str(e)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(query_region, r): r for r in regions}
        for future in as_completed(futures):
            region, od_models, prov_models, error = future.result()
            region_stats[region] = {
                "on_demand_count": len(od_models),
                "provisioned_count": len(prov_models),
                "error": error,
            }
            for mid in od_models:
                on_demand_availability[mid].add(region)
            for mid in prov_models:
                provisioned_availability[mid].add(region)

    successful = sum(1 for s in region_stats.values() if s["error"] is None)
    logger.info(
        f"API discovery: {len(on_demand_availability)} on-demand models, "
        f"{len(provisioned_availability)} provisioned models across "
        f"{successful}/{len(regions)} successful regions"
    )

    return on_demand_availability, provisioned_availability, region_stats


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


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for regional availability computation.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "regions": ["us-east-1", "us-west-2", ...],
            "pricingS3Key": "..."   # accepted for backward compat, ignored
        }

    Output:
        {
            "status": "SUCCESS",
            "s3Key": "executions/{id}/intermediate/regional-availability.json",
            "regionsWithBedrock": 27
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
    dry_run = event.get("dryRun", False)

    # Log if pricingS3Key was passed (backward compat — no longer used)
    if "pricingS3Key" in event:
        logger.info(
            "pricingS3Key provided but no longer used — pricing data "
            "excluded from availability (see module docstring)"
        )

    output_key = f"executions/{execution_id}/intermediate/regional-availability.json"

    logger.info(f"Computing regional availability (API regions: {len(regions)})")

    try:
        s3_client = get_s3_client()

        if not dry_run:
            # Discover models via filtered ListFoundationModels calls
            on_demand = {}
            provisioned = {}
            region_stats = {}
            if regions:
                on_demand, provisioned, region_stats = _discover_via_api(regions)

            # Build unified output (on-demand primary, provisioned secondary)
            availability = _build_availability_output(on_demand, provisioned)

            output_data = {
                "metadata": {
                    "regions_with_bedrock": len(availability["regions"]),
                    "total_models_tracked": len(availability["model_availability"]),
                    "total_provisioned_models": len(
                        availability["provisioned_availability"]
                    ),
                    "api_regions_queried": len(regions),
                    "collection_timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "discovery_method": "api_on_demand_filtered",
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

        return {
            "status": "SUCCESS",
            "s3Key": output_key,
            "regionsWithBedrock": regions_count,
            "durationMs": duration_ms,
        }

    except Exception as e:
        logger.error(f"Failed to compute availability: {e}", exc_info=True)
        return {
            "status": "FAILED",
            "errorType": type(e).__name__,
            "errorMessage": str(e),
        }
