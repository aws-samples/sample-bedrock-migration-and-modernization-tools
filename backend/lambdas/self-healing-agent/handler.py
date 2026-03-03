"""
Self-Healing Agent Lambda

Uses Bedrock Claude Opus 4.5 to analyze gaps and suggest configuration updates.
Auto-applies safe suggestions and flags high-risk changes for review.

Features:
- Analyzes gap reports from gap-detection Lambda
- Uses Claude to understand patterns and suggest fixes
- Auto-applies safe changes (new provider patterns, aliases, regions)
- Stores suggestions for manual review
"""

import json
import logging
import os
import time
from typing import Any

import boto3

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
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# Configuration loader
_config_loader = None


def _get_config():
    """Get the configuration loader (lazy initialization)."""
    global _config_loader
    if _config_loader is None:
        _config_loader = get_config_loader()
        _config_loader.load_config()
    return _config_loader


def get_bedrock_client():
    """Create Bedrock runtime client."""
    return boto3.client("bedrock-runtime")


def build_analysis_prompt(gap_report: dict, current_config: dict) -> str:
    """
    Build a comprehensive prompt for Claude to analyze all gap types.
    """
    # Extract all gap information
    models_without_pricing = gap_report.get("details", {}).get(
        "models_without_pricing", []
    )
    unknown_providers = gap_report.get("details", {}).get("unknown_providers", [])
    low_confidence_matches = gap_report.get("details", {}).get(
        "low_confidence_matches", []
    )
    new_models = gap_report.get("details", {}).get("new_models", [])
    context_mismatches = gap_report.get("details", {}).get(
        "context_window_mismatches", []
    )
    unknown_service_codes = gap_report.get("details", {}).get(
        "unknown_service_codes", []
    )
    frontend_drift = gap_report.get("details", {}).get("frontend_config_drift", {})

    # Get current configuration sections
    provider_config = current_config.get("provider_configuration", {})
    model_config = current_config.get("model_configuration", {})

    # Truncate context window specs to avoid overly long prompts
    context_specs_str = json.dumps(
        model_config.get("context_window_specs", {}), indent=2
    )
    if len(context_specs_str) > 2000:
        context_specs_str = context_specs_str[:2000] + "\n... (truncated)"

    prompt = f"""You are an expert system analyzing a Bedrock Model Profiler data collection pipeline.
The pipeline collects model information from AWS Bedrock and needs to match models to their pricing data.

## Current Configuration

**Provider Patterns** (keywords used to detect providers):
```json
{json.dumps(provider_config.get("provider_patterns", {}), indent=2)}
```

**Provider Aliases** (name variations for matching):
```json
{json.dumps(provider_config.get("provider_aliases", {}), indent=2)}
```

**Context Window Specs** (manual overrides for context windows):
```json
{context_specs_str}
```

**Known Pricing Service Codes**:
```json
{json.dumps(current_config.get("pricing_service_codes", []), indent=2)}
```

## Gap Report Analysis

### Models Without Pricing ({len(models_without_pricing)} total)
{json.dumps(models_without_pricing[:15], indent=2)}
{"... and more" if len(models_without_pricing) > 15 else ""}

### Unknown Providers ({len(unknown_providers)} total)
{json.dumps(unknown_providers, indent=2)}

### Low Confidence Matches ({len(low_confidence_matches)} total)
{json.dumps(low_confidence_matches[:10], indent=2)}

### New Models Detected ({len(new_models)} total)
{json.dumps(new_models[:15], indent=2)}
{"... and more" if len(new_models) > 15 else ""}

### Context Window Mismatches ({len(context_mismatches)} total)
{json.dumps(context_mismatches[:10], indent=2)}

### Unknown Service Codes
{json.dumps(unknown_service_codes, indent=2)}

### Frontend Config Drift
{json.dumps(frontend_drift, indent=2)}

## Task

Analyze ALL gaps and suggest configuration updates. For each suggestion:
1. Identify the root cause
2. Propose specific configuration changes
3. Assess if the change is safe to auto-apply

## Output Format

Return a JSON object with this exact structure:
```json
{{
  "analysis": {{
    "summary": "Brief summary of findings",
    "root_causes": ["List of identified root causes"],
    "confidence": 0.85
  }},
  "suggestions": [
    {{
      "id": "sugg-001",
      "type": "provider_pattern_addition|provider_alias_addition|context_window_update|service_code_addition|region_addition",
      "priority": "high|medium|low",
      "target_config_path": "provider_configuration.provider_patterns.NVIDIA",
      "description": "Add 'nemotron' pattern for NVIDIA models",
      "current_value": ["nvidia"],
      "suggested_value": ["nvidia", "nemotron"],
      "rationale": "5 NVIDIA Nemotron models failed to match due to missing pattern",
      "affected_models": ["nvidia.nemotron-nano-12b-v2"],
      "confidence": 0.95,
      "auto_apply_safe": true
    }}
  ]
}}
```

**Safety Rules for auto_apply_safe:**
- TRUE for: Adding new patterns, adding aliases, adding regions, adding documentation links, updating context windows (when source is authoritative), adding service codes
- FALSE for: Removing patterns, modifying existing patterns, changing thresholds, any change affecting >20% of models

**Context Window Update Rules:**
- If LiteLLM value is higher than config value, suggest update (models typically get larger context)
- If variance is >50%, mark as requires_review
- Include source attribution in rationale

Return ONLY the JSON object, no other text."""

    return prompt


def invoke_claude(prompt: str, bedrock_client: Any) -> dict:
    """
    Invoke Claude Opus 4.5 via Bedrock to analyze gaps.

    Returns parsed JSON response from Claude.
    """
    config = _get_config()
    model_id = config.get_bedrock_model_id()
    max_tokens = config.get_agent_config().get("max_tokens", 4096)

    logger.info(f"Invoking Bedrock model: {model_id}")

    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,  # Low temperature for consistent, analytical responses
                }
            ),
        )

        response_body = json.loads(response["body"].read())
        content = response_body.get("content", [{}])[0].get("text", "{}")

        # Parse JSON from response
        # Handle case where Claude might wrap in markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response as JSON: {e}")
        return {
            "analysis": {
                "summary": "Failed to parse response",
                "root_causes": ["JSON parsing error"],
                "confidence": 0,
            },
            "suggestions": [],
        }
    except Exception as e:
        logger.error(f"Failed to invoke Bedrock: {e}", exc_info=True)
        raise


def apply_context_window_update(suggestion: dict, current_config: dict) -> tuple:
    """
    Apply a context window update suggestion.

    Returns (success, message)
    """
    target_path = suggestion.get("target_config_path", "")
    suggested_value = suggestion.get("suggested_value")

    if not target_path.startswith("model_configuration.context_window_specs"):
        return False, "Invalid target path for context window update"

    # Extract model key from path
    path_parts = target_path.split(".")
    if len(path_parts) < 3:
        return False, "Invalid path format"

    model_key = path_parts[2]

    # Validate the suggested value structure
    if (
        not isinstance(suggested_value, dict)
        or "standard_context" not in suggested_value
    ):
        return False, "Missing required fields in suggested value"

    # Apply the update
    context_specs = current_config.setdefault("model_configuration", {}).setdefault(
        "context_window_specs", {}
    )

    # Preserve existing fields, update context window
    if model_key in context_specs:
        context_specs[model_key].update(suggested_value)
    else:
        context_specs[model_key] = suggested_value

    # Add source attribution
    context_specs[model_key]["source"] = suggested_value.get("source", "auto_updated")
    context_specs[model_key]["auto_updated_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )

    return True, f"Updated context window for {model_key}"


def apply_service_code_addition(suggestion: dict, current_config: dict) -> tuple:
    """
    Add a new service code to the configuration.

    Returns (success, message)
    """
    suggested_value = suggestion.get("suggested_value")

    if not isinstance(suggested_value, list):
        return False, "Suggested value must be a list of service codes"

    current_codes = current_config.get("pricing_service_codes", [])
    new_codes = [code for code in suggested_value if code not in current_codes]

    if not new_codes:
        return False, "No new service codes to add"

    current_config["pricing_service_codes"] = current_codes + new_codes

    return True, f"Added service codes: {', '.join(new_codes)}"


def count_total_models(config: dict) -> int:
    """Count total models referenced in config."""
    # Approximate count from context window specs
    return len(config.get("model_configuration", {}).get("context_window_specs", {}))


def validate_suggestion(suggestion: dict, current_config: dict) -> tuple:
    """
    Validate a suggestion before applying.

    Returns (is_valid, reason)
    """
    sugg_type = suggestion.get("type", "")
    affected_models = suggestion.get("affected_models", [])

    # Check affected models threshold
    total_models = count_total_models(current_config)
    if total_models > 0:
        affected_ratio = len(affected_models) / total_models
        max_ratio = (
            current_config.get("agent_configuration", {})
            .get("auto_apply_rules", {})
            .get("max_models_affected_for_auto_apply", 0.2)
        )
        if affected_ratio > max_ratio:
            return (
                False,
                f"Affects {affected_ratio:.1%} of models (max: {max_ratio:.1%})",
            )

    # Type-specific validation
    if sugg_type == "context_window_update":
        suggested_value = suggestion.get("suggested_value", {})
        if not isinstance(suggested_value, dict):
            return False, "Invalid context window value format"
        if suggested_value.get("standard_context", 0) <= 0:
            return False, "Invalid context window value"

    if sugg_type == "provider_pattern_addition":
        suggested_value = suggestion.get("suggested_value", [])
        if not isinstance(suggested_value, list) or len(suggested_value) == 0:
            return False, "Invalid pattern list"

    if sugg_type == "service_code_addition":
        suggested_value = suggestion.get("suggested_value", [])
        if not isinstance(suggested_value, list) or len(suggested_value) == 0:
            return False, "Invalid service code list"

    return True, "Valid"


def apply_safe_suggestions(
    suggestions: list, current_config: dict, s3_client: Any, bucket: str
) -> dict:
    """
    Apply suggestions marked as auto_apply_safe to the configuration.
    Enhanced to handle context window updates and service code additions.

    Returns dict with:
        - applied: list of applied suggestion IDs
        - skipped: list of skipped suggestion IDs
        - new_config: updated configuration (if any changes made)
    """
    config = _get_config()
    auto_apply_rules = config.get_agent_config().get("auto_apply_rules", {})
    safe_change_types = set(auto_apply_rules.get("safe_changes", []))

    applied = []
    skipped = []
    config_modified = False
    new_config = current_config.copy()

    for suggestion in suggestions:
        sugg_id = suggestion.get("id", "unknown")
        sugg_type = suggestion.get("type", "")
        auto_safe = suggestion.get("auto_apply_safe", False)

        # Check if this type is safe and marked for auto-apply
        if not auto_safe or sugg_type not in safe_change_types:
            skipped.append(
                {
                    "id": sugg_id,
                    "reason": f"Not safe for auto-apply (type={sugg_type}, auto_safe={auto_safe})",
                }
            )
            continue

        # Validate suggestion before applying
        is_valid, validation_reason = validate_suggestion(suggestion, new_config)
        if not is_valid:
            skipped.append(
                {
                    "id": sugg_id,
                    "reason": f"Validation failed: {validation_reason}",
                }
            )
            continue

        try:
            # Handle different suggestion types
            if sugg_type == "context_window_update":
                success, message = apply_context_window_update(suggestion, new_config)
                if success:
                    config_modified = True
                    applied.append(sugg_id)
                    logger.info(f"Auto-applied {sugg_id}: {message}")
                else:
                    skipped.append({"id": sugg_id, "reason": message})
                continue

            if sugg_type == "service_code_addition":
                success, message = apply_service_code_addition(suggestion, new_config)
                if success:
                    config_modified = True
                    applied.append(sugg_id)
                    logger.info(f"Auto-applied {sugg_id}: {message}")
                else:
                    skipped.append({"id": sugg_id, "reason": message})
                continue

            # Default path-based update for other types
            target_path = suggestion.get("target_config_path", "")
            suggested_value = suggestion.get("suggested_value")

            if not target_path or suggested_value is None:
                skipped.append(
                    {
                        "id": sugg_id,
                        "reason": "Missing target_config_path or suggested_value",
                    }
                )
                continue

            # Navigate to the target path and update
            path_parts = target_path.split(".")
            current = new_config
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            final_key = path_parts[-1]
            current[final_key] = suggested_value
            config_modified = True
            applied.append(sugg_id)

            logger.info(f"Auto-applied suggestion {sugg_id}: {target_path}")

        except Exception as e:
            logger.warning(f"Failed to apply suggestion {sugg_id}: {e}")
            skipped.append({"id": sugg_id, "reason": str(e)})

    # If config was modified, save it
    if config_modified:
        # Update version timestamp
        new_config["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        new_config["version"] = f"{new_config.get('version', '1.0.0')}-auto-updated"

        # Save backup first (before modifying)
        backup_key = f"config/config-history/profiler-config.{time.strftime('%Y%m%d-%H%M%S')}.json"
        write_to_s3(s3_client, bucket, backup_key, current_config)
        logger.info(f"Saved config backup to s3://{bucket}/{backup_key}")

        # Save new config
        config_key = "config/profiler-config.json"
        write_to_s3(s3_client, bucket, config_key, new_config)
        logger.info(f"Saved updated config to s3://{bucket}/{config_key}")

    return {
        "applied": applied,
        "skipped": skipped,
        "config_modified": config_modified,
        "new_config": new_config if config_modified else None,
    }


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for self-healing agent.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "gapReportS3Key": "agent/gap-reports/{exec_id}/gap-analysis.json"
        }

    Output:
        {
            "status": "SUCCESS",
            "s3Key": "agent/suggestions/{exec_id}/suggestions.json",
            "suggestionsCount": 5,
            "highPrioritySuggestions": 2,
            "autoApplied": ["sugg-001", "sugg-002"],
            "configModified": true
        }
    """
    start_time = time.time()

    # Validate required parameters
    try:
        validate_required_params(
            event, ["s3Bucket", "executionId", "gapReportS3Key"], "SelfHealingAgent"
        )
    except ValidationError as e:
        return {
            "status": "FAILED",
            "errorType": "ValidationError",
            "errorMessage": str(e),
        }

    s3_bucket = event["s3Bucket"]
    execution_id = parse_execution_id(event["executionId"])
    gap_report_key = event["gapReportS3Key"]

    output_key = f"agent/suggestions/{execution_id}/suggestions.json"

    logger.info(f"Running self-healing agent for execution {execution_id}")

    try:
        s3_client = get_s3_client()
        bedrock_client = get_bedrock_client()

        # Read gap report
        gap_report = read_from_s3(s3_client, s3_bucket, gap_report_key)

        # Get current config
        config = _get_config()
        current_config = config.config

        # Check if we should proceed based on gap report
        should_trigger = gap_report.get("trigger_decision", {}).get(
            "should_trigger", False
        )
        if not should_trigger:
            logger.info("Gap report indicates no action needed")
            return {
                "status": "SUCCESS",
                "s3Key": None,
                "suggestionsCount": 0,
                "highPrioritySuggestions": 0,
                "autoApplied": [],
                "configModified": False,
                "message": "No gaps detected that require action",
            }

        # Build prompt and invoke Claude
        prompt = build_analysis_prompt(gap_report, current_config)
        claude_response = invoke_claude(prompt, bedrock_client)

        suggestions = claude_response.get("suggestions", [])
        analysis = claude_response.get("analysis", {})

        # Count high priority suggestions
        high_priority_count = sum(1 for s in suggestions if s.get("priority") == "high")

        # Apply safe suggestions
        apply_result = apply_safe_suggestions(
            suggestions, current_config, s3_client, s3_bucket
        )

        # Update suggestions with applied status
        for suggestion in suggestions:
            if suggestion.get("id") in apply_result["applied"]:
                suggestion["status"] = "applied"
            else:
                suggestion["status"] = "pending_review"

        # Build output
        output = {
            "execution_id": execution_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "agent_model": config.get_bedrock_model_id(),
            "gap_report_ref": f"s3://{s3_bucket}/{gap_report_key}",
            "analysis": analysis,
            "suggestions": suggestions,
            "auto_apply_summary": {
                "applied": apply_result["applied"],
                "skipped": apply_result["skipped"],
                "config_modified": apply_result["config_modified"],
            },
            "metadata": {
                "total_suggestions": len(suggestions),
                "high_priority": high_priority_count,
                "auto_applied_count": len(apply_result["applied"]),
                "pending_review_count": len(suggestions) - len(apply_result["applied"]),
            },
        }

        # Write suggestions to S3
        write_to_s3(s3_client, s3_bucket, output_key, output)

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            "status": "SUCCESS",
            "s3Key": output_key,
            "suggestionsCount": len(suggestions),
            "highPrioritySuggestions": high_priority_count,
            "autoApplied": apply_result["applied"],
            "configModified": apply_result["config_modified"],
            "durationMs": duration_ms,
        }

    except Exception as e:
        logger.error(f"Failed to run self-healing agent: {e}", exc_info=True)
        return {
            "status": "FAILED",
            "errorType": type(e).__name__,
            "errorMessage": str(e),
        }
