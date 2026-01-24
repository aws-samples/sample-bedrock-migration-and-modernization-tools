"""
Gap Detection Lambda

Analyzes pipeline output to detect gaps in data collection:
- Models without pricing matches
- Low-confidence pricing matches
- New models (delta from previous run)
- Unknown providers not in configuration
- Missing region coverage

Determines if the self-healing agent should be triggered.
"""

import logging
import os
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

logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

# Configuration loader
_config_loader = None


def _get_config():
    """Get the configuration loader (lazy initialization)."""
    global _config_loader
    if _config_loader is None:
        _config_loader = get_config_loader()
        _config_loader.load_config()
    return _config_loader


def analyze_models_data(models_data: dict) -> dict:
    """
    Analyze models data to find gaps.

    Returns dict with:
        - models_without_pricing: list of model IDs
        - low_confidence_matches: list of {model_id, confidence}
        - unknown_providers: set of unknown provider names
        - total_models: int
    """
    config = _get_config()
    thresholds = config.get_agent_thresholds()
    low_confidence_threshold = thresholds.get('low_confidence_threshold', 0.6)

    known_providers = set(config.get_provider_patterns().keys())

    models_without_pricing = []
    low_confidence_matches = []
    unknown_providers = set()
    all_models = []
    provider_counts = {}

    for provider, provider_data in models_data.get('providers', {}).items():
        for model_id, model in provider_data.get('models', {}).items():
            all_models.append(model_id)

            # Track provider counts
            model_provider = model.get('model_provider', provider)
            provider_counts[model_provider] = provider_counts.get(model_provider, 0) + 1

            # Check if provider is unknown
            if model_provider and model_provider not in known_providers:
                unknown_providers.add(model_provider)

            # Check pricing status
            has_pricing = model.get('has_pricing', False)
            pricing_info = model.get('model_pricing', {})
            confidence = pricing_info.get('confidence', 0)

            if not has_pricing:
                models_without_pricing.append({
                    'model_id': model_id,
                    'model_name': model.get('model_name', model_id),
                    'provider': model_provider
                })
            elif confidence < low_confidence_threshold:
                low_confidence_matches.append({
                    'model_id': model_id,
                    'model_name': model.get('model_name', model_id),
                    'provider': model_provider,
                    'confidence': confidence,
                    'pricing_reference_id': pricing_info.get('pricing_reference_id')
                })

    return {
        'models_without_pricing': models_without_pricing,
        'low_confidence_matches': low_confidence_matches,
        'unknown_providers': list(unknown_providers),
        'total_models': len(all_models),
        'provider_counts': provider_counts
    }


def detect_new_models(current_models: list, previous_models_key: str, s3_client: Any, bucket: str) -> list:
    """
    Detect new models by comparing with previous run.

    Returns list of new model IDs.
    """
    try:
        previous_data = read_from_s3(s3_client, bucket, previous_models_key, default_on_missing={})
        previous_model_ids = set()

        for provider_data in previous_data.get('providers', {}).values():
            for model_id in provider_data.get('models', {}).keys():
                previous_model_ids.add(model_id)

        current_model_ids = set(current_models)
        new_models = list(current_model_ids - previous_model_ids)

        return new_models
    except Exception as e:
        logger.warning(f"Could not compare with previous run: {e}")
        return []


def analyze_pricing_coverage(pricing_data: dict, models_data: dict) -> dict:
    """
    Analyze pricing coverage across regions.

    Returns dict with:
        - regions_with_pricing: list of regions
        - regions_missing_pricing: list of regions
        - pricing_providers: set of providers in pricing data
    """
    config = _get_config()
    expected_regions = set(config.get_region_list('quota_regions'))

    regions_with_pricing = set()
    pricing_providers = set()

    for provider, provider_data in pricing_data.get('providers', {}).items():
        pricing_providers.add(provider)
        if isinstance(provider_data, dict):
            for model_id, model_data in provider_data.items():
                if isinstance(model_data, dict) and 'regions' in model_data:
                    regions_with_pricing.update(model_data['regions'].keys())

    regions_missing = expected_regions - regions_with_pricing

    return {
        'regions_with_pricing': list(regions_with_pricing),
        'regions_missing_pricing': list(regions_missing),
        'pricing_providers': list(pricing_providers)
    }


def determine_trigger_decision(analysis: dict) -> dict:
    """
    Determine if the self-healing agent should be triggered based on analysis.

    Returns dict with:
        - should_trigger: bool
        - reasons: list of trigger reasons
        - priority: 'high', 'medium', 'low'
    """
    config = _get_config()
    thresholds = config.get_agent_thresholds()

    unmatched_trigger = thresholds.get('unmatched_models_trigger', 5)
    max_low_confidence = thresholds.get('max_low_confidence_matches', 3)
    new_provider_trigger = thresholds.get('new_provider_trigger', True)

    reasons = []
    priority = 'low'

    # Check unmatched models
    unmatched_count = len(analysis.get('models_without_pricing', []))
    if unmatched_count >= unmatched_trigger:
        reasons.append(f"{unmatched_count} models without pricing (threshold: {unmatched_trigger})")
        priority = 'high'

    # Check low confidence matches
    low_confidence_count = len(analysis.get('low_confidence_matches', []))
    if low_confidence_count >= max_low_confidence:
        reasons.append(f"{low_confidence_count} low-confidence matches")
        if priority != 'high':
            priority = 'medium'

    # Check unknown providers
    unknown_providers = analysis.get('unknown_providers', [])
    if unknown_providers and new_provider_trigger:
        reasons.append(f"Unknown providers detected: {', '.join(unknown_providers)}")
        priority = 'high'

    # Check new models
    new_models_count = len(analysis.get('new_models', []))
    if new_models_count > 0:
        reasons.append(f"{new_models_count} new models detected")
        if priority == 'low':
            priority = 'medium'

    should_trigger = len(reasons) > 0

    return {
        'should_trigger': should_trigger,
        'reasons': reasons,
        'priority': priority
    }


def lambda_handler(event: dict, context: Any) -> dict:
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
    start_time = time.time()

    # Validate required parameters
    try:
        validate_required_params(event, ['s3Bucket', 'executionId'], 'GapDetection')
    except ValidationError as e:
        return {
            'status': 'FAILED',
            'errorType': 'ValidationError',
            'errorMessage': str(e)
        }

    s3_bucket = event['s3Bucket']
    execution_id = parse_execution_id(event['executionId'])

    # Default paths if not provided
    models_s3_key = event.get('modelsS3Key', f"executions/{execution_id}/final/bedrock_models.json")
    pricing_s3_key = event.get('pricingS3Key', f"executions/{execution_id}/final/bedrock_pricing.json")
    previous_models_key = event.get('previousModelsKey', 'latest/bedrock_models.json')

    output_key = f"agent/gap-reports/{execution_id}/gap-analysis.json"

    logger.info(f"Analyzing gaps for execution {execution_id}")

    try:
        s3_client = get_s3_client()

        # Read models and pricing data
        models_data = read_from_s3(s3_client, s3_bucket, models_s3_key, default_on_missing={})
        pricing_data = read_from_s3(s3_client, s3_bucket, pricing_s3_key, default_on_missing={})

        # Analyze models for gaps
        models_analysis = analyze_models_data(models_data)

        # Get list of all current model IDs
        all_model_ids = []
        for provider_data in models_data.get('providers', {}).values():
            all_model_ids.extend(provider_data.get('models', {}).keys())

        # Detect new models
        new_models = detect_new_models(all_model_ids, previous_models_key, s3_client, s3_bucket)
        models_analysis['new_models'] = new_models

        # Analyze pricing coverage
        pricing_analysis = analyze_pricing_coverage(pricing_data, models_data)

        # Combine analysis
        full_analysis = {
            **models_analysis,
            **pricing_analysis
        }

        # Determine if agent should be triggered
        trigger_decision = determine_trigger_decision(full_analysis)

        # Build output report
        report = {
            'execution_id': execution_id,
            'analysis_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'summary': {
                'total_models': models_analysis['total_models'],
                'models_without_pricing': len(models_analysis['models_without_pricing']),
                'low_confidence_matches': len(models_analysis['low_confidence_matches']),
                'new_models_detected': len(new_models),
                'unknown_providers': models_analysis['unknown_providers'],
                'regions_with_pricing': len(pricing_analysis['regions_with_pricing']),
                'regions_missing_pricing': len(pricing_analysis['regions_missing_pricing'])
            },
            'trigger_decision': trigger_decision,
            'details': {
                'models_without_pricing': models_analysis['models_without_pricing'],
                'low_confidence_matches': models_analysis['low_confidence_matches'],
                'new_models': new_models,
                'unknown_providers': models_analysis['unknown_providers'],
                'provider_counts': models_analysis['provider_counts'],
                'regions_missing_pricing': pricing_analysis['regions_missing_pricing'],
                'pricing_providers': pricing_analysis['pricing_providers']
            },
            'config_version': _get_config().config.get('version', 'unknown')
        }

        # Write report to S3
        write_to_s3(s3_client, s3_bucket, output_key, report)

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            'status': 'SUCCESS',
            's3Key': output_key,
            'shouldTriggerAgent': trigger_decision['should_trigger'],
            'summary': report['summary'],
            'priority': trigger_decision['priority'],
            'reasons': trigger_decision['reasons'],
            'durationMs': duration_ms
        }

    except Exception as e:
        logger.error(f"Failed to analyze gaps: {e}", exc_info=True)
        return {
            'status': 'FAILED',
            'errorType': type(e).__name__,
            'errorMessage': str(e)
        }
