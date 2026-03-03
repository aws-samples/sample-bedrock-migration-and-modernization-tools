"""
Gap Detection Lambda

Analyzes pipeline output to detect gaps in data collection:
- Models without pricing matches
- Low-confidence pricing matches
- New models (delta from previous run)
- Unknown providers not in configuration
- Missing region coverage
- Context window mismatches between sources
- Unknown service codes in pricing data
- Frontend config drift from backend config

Determines if the self-healing agent should be triggered.
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
    get_config_loader,
)
from shared.powertools import logger, tracer, metrics, LambdaContext
from aws_lambda_powertools.metrics import MetricUnit


def analyze_models_data(models_data: dict) -> dict:
    """
    Analyze models data to find gaps.

    Returns dict with:
        - models_without_pricing: list of model IDs
        - low_confidence_matches: list of {model_id, confidence}
        - unknown_providers: set of unknown provider names
        - total_models: int
    """
    config = get_config_loader()
    thresholds = config.get_agent_thresholds()
    low_confidence_threshold = thresholds.get("low_confidence_threshold", 0.6)

    known_providers = set(config.get_provider_patterns().keys())

    models_without_pricing = []
    low_confidence_matches = []
    unknown_providers = set()
    all_models = []
    provider_counts = {}

    for provider, provider_data in models_data.get("providers", {}).items():
        for model_id, model in provider_data.get("models", {}).items():
            all_models.append(model_id)

            # Track provider counts
            model_provider = model.get("model_provider", provider)
            provider_counts[model_provider] = provider_counts.get(model_provider, 0) + 1

            # Check if provider is unknown
            if model_provider and model_provider not in known_providers:
                unknown_providers.add(model_provider)

            # Check pricing status
            has_pricing = model.get("has_pricing", False)
            pricing_info = model.get("model_pricing", {})
            confidence = pricing_info.get("confidence", 0)

            if not has_pricing:
                models_without_pricing.append(
                    {
                        "model_id": model_id,
                        "model_name": model.get("model_name", model_id),
                        "provider": model_provider,
                    }
                )
            elif confidence < low_confidence_threshold:
                low_confidence_matches.append(
                    {
                        "model_id": model_id,
                        "model_name": model.get("model_name", model_id),
                        "provider": model_provider,
                        "confidence": confidence,
                        "pricing_reference_id": pricing_info.get(
                            "pricing_reference_id"
                        ),
                    }
                )

    return {
        "models_without_pricing": models_without_pricing,
        "low_confidence_matches": low_confidence_matches,
        "unknown_providers": list(unknown_providers),
        "total_models": len(all_models),
        "provider_counts": provider_counts,
    }


def detect_new_models(
    current_models: list, previous_models_key: str, s3_client: Any, bucket: str
) -> list:
    """
    Detect new models by comparing with previous run.

    Returns list of new model IDs.
    """
    try:
        previous_data = read_from_s3(
            s3_client, bucket, previous_models_key, default_on_missing={}
        )
        previous_model_ids = set()

        for provider_data in previous_data.get("providers", {}).values():
            for model_id in provider_data.get("models", {}).keys():
                previous_model_ids.add(model_id)

        current_model_ids = set(current_models)
        new_models = list(current_model_ids - previous_model_ids)

        return new_models
    except Exception as e:
        logger.warning("Could not compare with previous run", extra={"error": str(e)})
        return []


def analyze_pricing_coverage(pricing_data: dict, models_data: dict) -> dict:
    """
    Analyze pricing coverage across regions.

    Returns dict with:
        - regions_with_pricing: list of regions
        - regions_missing_pricing: list of regions
        - pricing_providers: set of providers in pricing data
    """
    config = get_config_loader()
    expected_regions = set(config.get_region_list("quota_regions"))

    regions_with_pricing = set()
    pricing_providers = set()

    for provider, provider_data in pricing_data.get("providers", {}).items():
        pricing_providers.add(provider)
        if isinstance(provider_data, dict):
            for model_id, model_data in provider_data.items():
                if isinstance(model_data, dict) and "regions" in model_data:
                    regions_with_pricing.update(model_data["regions"].keys())

    regions_missing = expected_regions - regions_with_pricing

    return {
        "regions_with_pricing": list(regions_with_pricing),
        "regions_missing_pricing": list(regions_missing),
        "pricing_providers": list(pricing_providers),
    }


def detect_context_window_mismatches(models_data: dict, config_dict: dict) -> list:
    """
    Detect models where context window data differs significantly between sources.

    Compares actual context window values from model data against manual overrides
    in the configuration's context_window_specs.

    Args:
        models_data: The aggregated models data from the pipeline
        config_dict: The profiler configuration dictionary

    Returns:
        List of dicts with model_id, actual_value, config_value, variance
    """
    mismatches = []
    context_specs = config_dict.get("model_configuration", {}).get(
        "context_window_specs", {}
    )
    variance_threshold = config_dict.get("gap_detection_config", {}).get(
        "context_window_variance_threshold", 0.1
    )

    for provider, provider_data in models_data.get("providers", {}).items():
        for model_id, model in provider_data.get("models", {}).items():
            # Get context window from actual model data (converse_data)
            actual_ctx = model.get("converse_data", {}).get("context_window")

            # Find matching config spec (partial match on model ID prefix)
            config_ctx = None
            for spec_key, spec_data in context_specs.items():
                if model_id.startswith(spec_key) or spec_key in model_id:
                    config_ctx = spec_data.get("standard_context")
                    break

            # Compare values if both exist
            if actual_ctx and config_ctx:
                max_val = max(actual_ctx, config_ctx)
                if max_val > 0:
                    variance = abs(actual_ctx - config_ctx) / max_val
                    if variance > variance_threshold:
                        mismatches.append(
                            {
                                "model_id": model_id,
                                "actual_value": actual_ctx,
                                "config_value": config_ctx,
                                "variance": round(variance, 3),
                            }
                        )

    return mismatches


def detect_unknown_service_codes(pricing_data: dict, config_dict: dict) -> list:
    """
    Detect service codes in pricing data that aren't in the known list.

    This helps identify when AWS introduces new pricing service codes that
    may require configuration updates.

    Args:
        pricing_data: The aggregated pricing data from the pipeline
        config_dict: The profiler configuration dictionary

    Returns:
        List of unknown service codes found in pricing data
    """
    known_codes = set(config_dict.get("pricing_service_codes", []))
    found_codes = set()

    # Extract service codes from pricing data
    for provider, provider_data in pricing_data.get("providers", {}).items():
        if isinstance(provider_data, dict):
            for model_id, model_data in provider_data.items():
                if isinstance(model_data, dict):
                    service_code = model_data.get("service_code")
                    if service_code:
                        found_codes.add(service_code)

    unknown_codes = list(found_codes - known_codes)
    return unknown_codes


def detect_frontend_config_drift(
    config_dict: dict, frontend_config_key: str, s3_client: Any, bucket: str
) -> dict:
    """
    Detect drift between backend and frontend region/provider configurations.

    Compares the backend profiler-config.json with the frontend-config.json
    stored in S3 to identify mismatches in regions and providers.

    Args:
        config_dict: The backend profiler configuration dictionary
        frontend_config_key: S3 key for the frontend config file
        s3_client: Boto3 S3 client
        bucket: S3 bucket name

    Returns:
        Dict with drift_detected flag and lists of missing/extra items
    """
    try:
        frontend_config = read_from_s3(
            s3_client, bucket, frontend_config_key, default_on_missing={}
        )
    except Exception as e:
        logger.warning(
            "Could not read frontend config for drift detection",
            extra={"error": str(e), "key": frontend_config_key},
        )
        return {"error": "Could not read frontend config", "drift_detected": False}

    # If frontend config is empty (file doesn't exist yet), no drift to detect
    if not frontend_config:
        return {
            "drift_detected": False,
            "note": "Frontend config not yet created",
            "regions_missing_in_frontend": [],
            "regions_extra_in_frontend": [],
            "providers_missing_in_frontend": [],
            "providers_extra_in_frontend": [],
        }

    backend_regions = set(
        config_dict.get("region_configuration", {}).get("region_locations", {}).keys()
    )
    frontend_regions = set(frontend_config.get("regions", {}).keys())

    backend_providers = set(
        config_dict.get("provider_configuration", {}).get("provider_colors", {}).keys()
    )
    frontend_providers = set(frontend_config.get("providers", {}).keys())

    drift_detected = (
        backend_regions != frontend_regions or backend_providers != frontend_providers
    )

    return {
        "drift_detected": drift_detected,
        "regions_missing_in_frontend": list(backend_regions - frontend_regions),
        "regions_extra_in_frontend": list(frontend_regions - backend_regions),
        "providers_missing_in_frontend": list(backend_providers - frontend_providers),
        "providers_extra_in_frontend": list(frontend_providers - backend_providers),
    }


def determine_trigger_decision(analysis: dict) -> dict:
    """
    Determine if the self-healing agent should be triggered based on analysis.

    Returns dict with:
        - should_trigger: bool
        - reasons: list of trigger reasons
        - priority: 'high', 'medium', 'low'
    """
    config = get_config_loader()
    thresholds = config.get_agent_thresholds()

    unmatched_trigger = thresholds.get("unmatched_models_trigger", 5)
    max_low_confidence = thresholds.get("max_low_confidence_matches", 3)
    new_provider_trigger = thresholds.get("new_provider_trigger", True)

    reasons = []
    priority = "low"

    # Check unmatched models
    unmatched_count = len(analysis.get("models_without_pricing", []))
    if unmatched_count >= unmatched_trigger:
        reasons.append(
            f"{unmatched_count} models without pricing (threshold: {unmatched_trigger})"
        )
        priority = "high"

    # Check low confidence matches
    low_confidence_count = len(analysis.get("low_confidence_matches", []))
    if low_confidence_count >= max_low_confidence:
        reasons.append(f"{low_confidence_count} low-confidence matches")
        if priority != "high":
            priority = "medium"

    # Check unknown providers
    unknown_providers = analysis.get("unknown_providers", [])
    if unknown_providers and new_provider_trigger:
        reasons.append(f"Unknown providers detected: {', '.join(unknown_providers)}")
        priority = "high"

    # Check new models
    new_models_count = len(analysis.get("new_models", []))
    if new_models_count > 0:
        reasons.append(f"{new_models_count} new models detected")
        if priority == "low":
            priority = "medium"

    # Check context window mismatches (new gap type)
    context_mismatches = analysis.get("context_window_mismatches", [])
    if len(context_mismatches) > 0:
        reasons.append(f"{len(context_mismatches)} context window mismatches detected")
        if priority == "low":
            priority = "medium"

    # Check unknown service codes (new gap type)
    unknown_codes = analysis.get("unknown_service_codes", [])
    if unknown_codes:
        reasons.append(f"Unknown service codes: {', '.join(unknown_codes)}")
        priority = "high"

    # Check frontend drift (new gap type)
    frontend_drift = analysis.get("frontend_config_drift", {})
    if frontend_drift.get("drift_detected"):
        reasons.append("Frontend config drift detected")
        if priority == "low":
            priority = "medium"

    should_trigger = len(reasons) > 0

    return {"should_trigger": should_trigger, "reasons": reasons, "priority": priority}


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    """
    Lambda handler for gap detection.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "modelsS3Key": "executions/{id}/final/bedrock_models.json",
            "pricingS3Key": "executions/{id}/final/bedrock_pricing.json",
            "previousModelsKey": "latest/bedrock_models.json" (optional)
        }

    Output:
        {
            "status": "SUCCESS",
            "s3Key": "agent/gap-reports/{exec_id}/gap-analysis.json",
            "shouldTriggerAgent": true/false,
            "summary": {
                "modelsWithoutPricing": 12,
                "lowConfidenceMatches": 3,
                "newModelsDetected": 4,
                "unknownProviders": ["newprovider"]
            },
            "priority": "high"/"medium"/"low"
        }
    """
    logger.info("Starting gap detection")
    start_time = time.time()

    # Validate required parameters
    try:
        validate_required_params(event, ["s3Bucket", "executionId"], "GapDetection")
    except ValidationError as e:
        return {
            "status": "FAILED",
            "errorType": "ValidationError",
            "errorMessage": str(e),
        }

    s3_bucket = event["s3Bucket"]
    execution_id = parse_execution_id(event["executionId"])

    # Default paths if not provided
    models_s3_key = event.get(
        "modelsS3Key", f"executions/{execution_id}/final/bedrock_models.json"
    )
    pricing_s3_key = event.get(
        "pricingS3Key", f"executions/{execution_id}/final/bedrock_pricing.json"
    )
    previous_models_key = event.get("previousModelsKey", "latest/bedrock_models.json")

    output_key = f"agent/gap-reports/{execution_id}/gap-analysis.json"

    logger.info("Analyzing gaps", extra={"execution_id": execution_id})

    try:
        s3_client = get_s3_client()

        # Read models and pricing data
        models_data = read_from_s3(
            s3_client, s3_bucket, models_s3_key, default_on_missing={}
        )
        pricing_data = read_from_s3(
            s3_client, s3_bucket, pricing_s3_key, default_on_missing={}
        )

        # Analyze models for gaps
        models_analysis = analyze_models_data(models_data)

        # Get list of all current model IDs
        all_model_ids = []
        for provider_data in models_data.get("providers", {}).values():
            all_model_ids.extend(provider_data.get("models", {}).keys())

        # Detect new models
        new_models = detect_new_models(
            all_model_ids, previous_models_key, s3_client, s3_bucket
        )
        models_analysis["new_models"] = new_models

        # Analyze pricing coverage
        pricing_analysis = analyze_pricing_coverage(pricing_data, models_data)

        # Combine analysis
        full_analysis = {**models_analysis, **pricing_analysis}

        # Get config for new gap detection functions
        config = get_config_loader()
        gap_detection_config = config.config.get("gap_detection_config", {})

        # Detect context window mismatches (new gap type)
        if gap_detection_config.get("enable_context_window_detection", True):
            context_mismatches = detect_context_window_mismatches(
                models_data, config.config
            )
            full_analysis["context_window_mismatches"] = context_mismatches
        else:
            full_analysis["context_window_mismatches"] = []

        # Detect unknown service codes (new gap type)
        if gap_detection_config.get("enable_service_code_detection", True):
            unknown_codes = detect_unknown_service_codes(pricing_data, config.config)
            full_analysis["unknown_service_codes"] = unknown_codes
        else:
            full_analysis["unknown_service_codes"] = []

        # Detect frontend config drift (new gap type)
        if gap_detection_config.get("enable_frontend_drift_detection", True):
            frontend_drift = detect_frontend_config_drift(
                config.config, "config/frontend-config.json", s3_client, s3_bucket
            )
            full_analysis["frontend_config_drift"] = frontend_drift
        else:
            full_analysis["frontend_config_drift"] = {"drift_detected": False}

        # Determine if agent should be triggered
        trigger_decision = determine_trigger_decision(full_analysis)

        gap_count = (
            len(models_analysis["models_without_pricing"])
            + len(models_analysis["low_confidence_matches"])
            + len(new_models)
            + len(models_analysis["unknown_providers"])
            + len(full_analysis.get("context_window_mismatches", []))
            + len(full_analysis.get("unknown_service_codes", []))
            + (
                1
                if full_analysis.get("frontend_config_drift", {}).get("drift_detected")
                else 0
            )
        )
        should_trigger = trigger_decision["should_trigger"]

        # Build output report
        report = {
            "execution_id": execution_id,
            "analysis_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": {
                "total_models": models_analysis["total_models"],
                "models_without_pricing": len(
                    models_analysis["models_without_pricing"]
                ),
                "low_confidence_matches": len(
                    models_analysis["low_confidence_matches"]
                ),
                "new_models_detected": len(new_models),
                "unknown_providers": models_analysis["unknown_providers"],
                "regions_with_pricing": len(pricing_analysis["regions_with_pricing"]),
                "regions_missing_pricing": len(
                    pricing_analysis["regions_missing_pricing"]
                ),
                "context_window_mismatches": len(
                    full_analysis.get("context_window_mismatches", [])
                ),
                "unknown_service_codes": len(
                    full_analysis.get("unknown_service_codes", [])
                ),
                "frontend_config_drift": full_analysis.get(
                    "frontend_config_drift", {}
                ).get("drift_detected", False),
            },
            "trigger_decision": trigger_decision,
            "details": {
                "models_without_pricing": models_analysis["models_without_pricing"],
                "low_confidence_matches": models_analysis["low_confidence_matches"],
                "new_models": new_models,
                "unknown_providers": models_analysis["unknown_providers"],
                "provider_counts": models_analysis["provider_counts"],
                "regions_missing_pricing": pricing_analysis["regions_missing_pricing"],
                "pricing_providers": pricing_analysis["pricing_providers"],
                "context_window_mismatches": full_analysis.get(
                    "context_window_mismatches", []
                ),
                "unknown_service_codes": full_analysis.get("unknown_service_codes", []),
                "frontend_config_drift": full_analysis.get("frontend_config_drift", {}),
            },
            "config_version": get_config_loader().config.get("version", "unknown"),
        }

        # Write report to S3
        write_to_s3(s3_client, s3_bucket, output_key, report)

        duration_ms = int((time.time() - start_time) * 1000)

        # Emit metrics
        metrics.add_metric(name="GapsDetected", unit=MetricUnit.Count, value=gap_count)
        metrics.add_metric(
            name="ShouldTriggerAgent",
            unit=MetricUnit.Count,
            value=1 if should_trigger else 0,
        )
        metrics.add_metric(
            name="DurationMs", unit=MetricUnit.Milliseconds, value=duration_ms
        )

        logger.info(
            "Gap detection complete",
            extra={
                "gaps_detected": gap_count,
                "should_trigger_agent": should_trigger,
                "priority": trigger_decision["priority"],
                "duration_ms": duration_ms,
            },
        )

        return {
            "status": "SUCCESS",
            "s3Key": output_key,
            "shouldTriggerAgent": should_trigger,
            "priority": trigger_decision["priority"],
        }

    except Exception as e:
        logger.exception(
            "Failed to analyze gaps", extra={"error_type": type(e).__name__}
        )
        return {
            "status": "FAILED",
            "errorType": type(e).__name__,
            "errorMessage": str(e),
        }
