"""
Model Extractor Lambda

Extracts foundation models from a single AWS region using the Bedrock API.
Also fetches console metadata via direct REST API with SigV4 signing to
extract context window, descriptions, languages, and categories.
Outputs models in the correct snake_case schema matching the original collector.
"""

import json
import logging
import os
import re
import time
from typing import Any
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.exceptions import ClientError

from shared import (
    RETRY_CONFIG,
    write_to_s3,
    parse_execution_id,
    validate_required_params,
    ValidationError,
    S3WriteError,
    get_config_loader,
)

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# Configuration loader - initialized on first use
_config_loader = None


def _get_config():
    """Get the configuration loader (lazy initialization)."""
    global _config_loader
    if _config_loader is None:
        _config_loader = get_config_loader()
        _config_loader.load_config()
    return _config_loader


def get_bedrock_client(region: str):
    """Create Bedrock client for a specific region."""
    return boto3.client("bedrock", region_name=region, config=RETRY_CONFIG)


def get_s3_client():
    return boto3.client("s3", config=RETRY_CONFIG)


def parse_context_window_string(value: str) -> int | None:
    """
    Parse context window strings from consoleIDEMetadata into integers.

    Examples:
        "200K" -> 200000
        "1M (beta)" -> 1000000
        "256K" -> 256000
        "128000" -> 128000
        "1,000,000" -> 1000000
    """
    if not value or not isinstance(value, str):
        return None

    value = value.strip()

    # Try pure numeric (with optional commas)
    clean = value.replace(",", "")
    try:
        return int(clean)
    except ValueError:
        pass

    # Match patterns like "200K", "1M", "1M (beta)", "256K tokens"
    match = re.match(r"^([\d.]+)\s*([KkMm])", value)
    if match:
        num = float(match.group(1))
        unit = match.group(2).upper()
        if unit == "K":
            return int(num * 1000)
        elif unit == "M":
            return int(num * 1000000)

    return None


def parse_use_cases(use_str: str) -> list:
    """
    Parse use case strings from consoleIDEMetadata into clean individual items.

    Handles multiple formats:
    1. Semicolon-separated groups with comma-separated items within each:
       "Complex agentic systems, multi-agent orchestration; visual analysis, document processing"
       -> ["Complex agentic systems", "Multi-agent orchestration", "Visual analysis", ...]

    2. Category names with parenthetical examples (NVIDIA-style):
       "Content Creation (e.g, code snippets, inline docs)Chatbots and AI (e.g, assistants)"
       -> ["Content Creation", "Chatbots and AI"]

    3. Simple comma-separated lists:
       "chat, summarization, translation"
       -> ["Chat", "Summarization", "Translation"]
    """
    if not use_str or not isinstance(use_str, str):
        return []

    use_str = use_str.strip().rstrip(".")

    # Detect parenthetical pattern: "Category (e.g, ...)" or "Category (e.g., ...)"
    has_parenthetical = bool(
        re.search(r"\([^)]*(?:e\.?g\.?|such as|like)[^)]*\)", use_str, re.IGNORECASE)
    )

    if has_parenthetical:
        # Strip all parenthetical content, then split on boundaries
        # First remove parenthetical groups: "(e.g, code snippets, inline docs)"
        cleaned = re.sub(r"\s*\([^)]*\)\s*", " ", use_str)
        # Split on common delimiters: semicolons, or where words run together after stripping
        # e.g. "Content Creation Chatbots and AI" -> need to split on double-space or other cues
        # After removing parens, items are typically separated by commas or semicolons
        if ";" in cleaned:
            items = cleaned.split(";")
        elif "," in cleaned:
            items = cleaned.split(",")
        else:
            # Items may run together after paren removal; split on 2+ spaces
            items = re.split(r"\s{2,}", cleaned)
    elif ";" in use_str:
        # Semicolons present: split on semicolons first, then split each group on commas
        groups = use_str.split(";")
        items = []
        for group in groups:
            items.extend(group.split(","))
    else:
        # Simple comma-separated
        items = use_str.split(",")

    # Clean up each item: strip whitespace, trailing periods, capitalize first letter
    result = []
    seen = set()
    for item in items:
        item = item.strip().rstrip(".")
        if not item or len(item) < 2:
            continue
        # Capitalize first letter
        item = item[0].upper() + item[1:]
        lower = item.lower()
        if lower not in seen:
            seen.add(lower)
            result.append(item)

    return result


def fetch_console_metadata(region: str) -> dict:
    """
    Fetch extended model metadata via direct Bedrock REST API with SigV4 signing.

    Uses the x-console-consumer header to get consoleIDEMetadata which includes
    context windows, descriptions, languages, and categories for ~53 models.

    Returns dict mapping model_id -> metadata dict. Returns empty dict on any error.
    """
    try:
        session = boto3.Session(region_name=region)
        credentials = session.get_credentials()
        if not credentials:
            logger.warning(
                f"No credentials available for console metadata fetch in {region}"
            )
            return {}

        frozen_credentials = credentials.get_frozen_credentials()
        url = f"https://bedrock.{region}.amazonaws.com/foundation-models"

        headers = {
            "Content-Type": "application/json",
            "x-console-consumer": "true",
        }

        request = AWSRequest(method="GET", url=url, headers=headers)
        SigV4Auth(frozen_credentials, "bedrock", region).add_auth(request)

        # Build urllib request with signed headers
        http_request = Request(url, headers=dict(request.headers), method="GET")
        with urlopen(http_request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        model_summaries = data.get("modelSummaries", [])
        metadata_by_id = {}

        for model in model_summaries:
            model_id = model.get("modelId", "")
            if not model_id:
                continue

            meta = {}

            # Parse consoleIDEMetadata (JSON string field)
            console_ide_raw = model.get("consoleIDEMetadata")
            if console_ide_raw and isinstance(console_ide_raw, str):
                try:
                    console_ide = json.loads(console_ide_raw)
                    desc = console_ide.get("description", {})

                    # Context window
                    max_cw_str = desc.get("maxContextWindow")
                    if max_cw_str:
                        parsed = parse_context_window_string(str(max_cw_str))
                        if parsed:
                            meta["max_context_window"] = parsed

                    # Descriptions
                    if desc.get("fullDescription"):
                        meta["description"] = desc["fullDescription"]
                    if desc.get("shortDescription"):
                        meta["short_description"] = desc["shortDescription"]

                    # Languages (comma/and-separated string, e.g. "English, French and Other languages.")
                    lang_str = desc.get("supportedLanguages")
                    if lang_str and isinstance(lang_str, str):
                        # Replace " and " with comma, strip trailing period
                        cleaned = lang_str.rstrip(".").replace(" and ", ", ")
                        meta["languages"] = [
                            l.strip() for l in cleaned.split(",") if l.strip()
                        ]

                    # Use cases (semicolon or comma-separated string, may contain parenthetical examples)
                    use_str = desc.get("supportedUseCases")
                    if use_str and isinstance(use_str, str):
                        meta["use_cases"] = parse_use_cases(use_str)

                except (json.JSONDecodeError, TypeError):
                    pass

            # Also extract from description object (available without console header for some)
            desc_obj = model.get("description", {})
            if isinstance(desc_obj, dict):
                if "max_context_window" not in meta:
                    max_cw_str = desc_obj.get("maxContextWindow")
                    if max_cw_str:
                        parsed = parse_context_window_string(str(max_cw_str))
                        if parsed:
                            meta["max_context_window"] = parsed

            # Extract max output tokens from converse object
            converse = model.get("converse", {})
            if isinstance(converse, dict):
                max_tokens = converse.get("maxTokensMaximum")
                if max_tokens and isinstance(max_tokens, (int, float)):
                    meta["max_output_tokens"] = int(max_tokens)

            if meta:
                metadata_by_id[model_id] = meta

        logger.info(
            f"Fetched console metadata for {len(metadata_by_id)} models from {region}"
        )
        return metadata_by_id

    except (URLError, HTTPError) as e:
        logger.warning(f"Failed to fetch console metadata from {region}: {e}")
        return {}
    except Exception as e:
        logger.warning(f"Unexpected error fetching console metadata from {region}: {e}")
        return {}


def get_documentation_links(model_id: str, provider: str) -> dict:
    """Get documentation links based on provider and model from config."""
    config = _get_config()
    all_docs = config.get_documentation_links()

    # Check for Nova models (Amazon's newer models)
    if "nova" in model_id.lower():
        nova_docs = all_docs.get("nova", all_docs.get("default", {}))
        return nova_docs.copy()

    # Get provider-specific docs or default
    return all_docs.get(provider, all_docs.get("default", {})).copy()


def process_model_data(raw_model: dict, region: str) -> dict:
    """
    Process and structure model data to match the expected schema.

    Converts AWS API response to snake_case schema matching the original collector.
    """
    model_id = raw_model.get("modelId", "")
    provider = raw_model.get("providerName", "")
    collection_timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    return {
        # Core identification (snake_case)
        "model_id": model_id,
        "model_arn": raw_model.get("modelArn", ""),
        "model_name": raw_model.get("modelName", ""),
        "model_provider": provider,
        # Capabilities from API (nested structure)
        "model_modalities": {
            "input_modalities": raw_model.get("inputModalities", []),
            "output_modalities": raw_model.get("outputModalities", []),
        },
        "streaming_supported": raw_model.get("responseStreamingSupported", False),
        "customization": {
            "customization_supported": raw_model.get("customizationsSupported", []),
            "customization_options": {},
        },
        "inference_types_supported": raw_model.get("inferenceTypesSupported", []),
        "model_lifecycle": {
            "status": raw_model.get("modelLifecycle", {}).get("status", "UNKNOWN"),
            "release_date": "",
        },
        # Regional information
        "regions_available": [region],
        # Fields to be enhanced in later phases
        "model_capabilities": [],
        "model_use_cases": [],
        "languages_supported": [],
        "consumption_options": [],
        "cross_region_inference": {},
        "documentation_links": get_documentation_links(model_id, provider),
        "model_pricing": {"is_pricing_available": False},
        "model_service_quotas": {},
        # Collection metadata
        "collection_metadata": {
            "first_discovered_at": collection_timestamp,
            "first_discovered_in_region": region,
            "api_source": "list_foundation_models",
            "dual_region_collection": True,
            "regions_collected_from": [region],
        },
    }


def extract_models(bedrock_client: Any, region: str) -> list[dict]:
    """
    Extract all foundation models from Bedrock API.

    Makes two calls:
    1. Standard boto3 list_foundation_models() for core model data
    2. Direct REST API with x-console-consumer header for extended metadata
       (context windows, descriptions, languages, categories)

    Returns list of model dictionaries with correct snake_case schema.
    """
    models = []

    try:
        response = bedrock_client.list_foundation_models()
        model_summaries = response.get("modelSummaries", [])

        for raw_model in model_summaries:
            processed = process_model_data(raw_model, region)
            models.append(processed)

        logger.info(f"Extracted {len(models)} models from {region}")

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code in ("AccessDeniedException", "UnrecognizedClientException"):
            logger.warning(
                f"Access denied or region not enabled: {region} ({error_code})"
            )
        elif error_code == "InvalidIdentityToken":
            logger.warning(
                f"Invalid token for region {region} - region may require opt-in"
            )
        else:
            logger.error(f"Error listing models in {region}: {e}")

    except Exception as e:
        logger.warning(f"Unexpected error extracting models in {region}: {e}")

    # Fetch console metadata (context windows, descriptions, etc.)
    # This is a separate call that gracefully degrades on failure
    console_metadata = fetch_console_metadata(region)
    if console_metadata:
        attached_count = 0
        for model in models:
            model_id = model.get("model_id", "")
            if model_id in console_metadata:
                model["console_metadata"] = console_metadata[model_id]
                attached_count += 1
        logger.info(
            f"Attached console metadata to {attached_count}/{len(models)} models in {region}"
        )

    return models


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for model extraction.

    Input:
        {
            "region": "us-east-1",
            "s3Bucket": "bucket-name",
            "s3Key": "executions/{id}/models/us-east-1.json"
        }

    Output:
        {
            "status": "SUCCESS",
            "region": "us-east-1",
            "s3Key": "executions/{id}/models/us-east-1.json",
            "modelCount": 108
        }
    """
    start_time = time.time()

    # Validate required parameters
    try:
        validate_required_params(event, ["region"], "ModelExtractor")
    except ValidationError as e:
        return {
            "status": "FAILED",
            "errorType": "ValidationError",
            "errorMessage": str(e),
        }

    region = event["region"]
    s3_bucket = event.get("s3Bucket")
    s3_key = event.get("s3Key", f"test/models/{region}.json")
    dry_run = event.get("dryRun", False)

    logger.info(f"Extracting models from region: {region}")

    try:
        bedrock_client = get_bedrock_client(region)
        models = extract_models(bedrock_client, region)

        output_data = {
            "metadata": {
                "region": region,
                "model_count": len(models),
                "collection_timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            },
            "models": models,
        }

        if not dry_run and s3_bucket:
            s3_client = get_s3_client()
            write_to_s3(s3_client, s3_bucket, s3_key, output_data)
        else:
            logger.info(
                f"Dry run - would write {len(models)} models to s3://{s3_bucket}/{s3_key}"
            )

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            "status": "SUCCESS",
            "region": region,
            "s3Key": s3_key,
            "modelCount": len(models),
            "durationMs": duration_ms,
        }

    except Exception as e:
        logger.error(f"Failed to extract models from {region}: {e}", exc_info=True)
        return {
            "status": "FAILED",
            "region": region,
            "errorType": type(e).__name__,
            "errorMessage": str(e),
            "retryable": "Throttling" in str(e),
        }
