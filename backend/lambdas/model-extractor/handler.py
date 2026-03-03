"""
Model Extractor Lambda

Extracts foundation models from a single AWS region using the Bedrock API.
Also fetches console metadata via direct REST API with SigV4 signing to
extract context window, descriptions, languages, and categories.
Outputs models in the correct snake_case schema matching the original collector.
"""

import json
import os
import re
import time
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
from shared.powertools import logger, tracer, metrics, LambdaContext
from aws_lambda_powertools.metrics import MetricUnit


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


def _parse_model_attributes(raw: str) -> list[str]:
    """Parse model attributes/capabilities from console metadata.

    Handles multiple formats:
    - Comma-separated short tags: "Text generation, Code generation"
    - Semicolon-separated: "Text generation; Code generation"
    - Long descriptions with sentences: "Category: Long description, more text. Another Category: ..."

    Long text (>300 chars) skips comma/semicolon splitting entirely and goes
    straight to sentence parsing, since long text is always a paragraph
    description rather than a list of tags.
    """
    if not raw or not raw.strip():
        return []

    raw = raw.strip()
    is_long_text = len(raw) > 300

    # For short text, try delimiter-based splitting first
    if not is_long_text:
        # Try semicolon split first (most reliable delimiter)
        if ";" in raw:
            attrs = [a.strip() for a in raw.split(";") if a.strip()]
            # If all results are short enough, use them
            if all(len(a) <= 80 for a in attrs):
                return _normalize_capabilities(attrs)

        # Try comma split
        if "," in raw:
            attrs = [a.strip() for a in raw.split(",") if a.strip()]
            # If all results are short (real tags), use them
            if all(len(a) <= 80 for a in attrs):
                return _normalize_capabilities(attrs)

    # Long description format — try splitting on ". " (sentence boundaries)
    # then extract the part before ":" as the capability name
    capabilities = []
    # Split on periods followed by uppercase letter (new sentence/category)
    sentences = re.split(r"\.\s+(?=[A-Z])", raw)
    for sentence in sentences:
        sentence = sentence.strip().rstrip(".")
        if not sentence:
            continue
        # If it has a "Category: description" format, extract the category
        if ":" in sentence:
            category = sentence.split(":", 1)[0].strip()
            # Only use category if it's reasonably short (not a sentence itself)
            if len(category) <= 80 and not any(c in category for c in [".", "!", "?"]):
                capabilities.append(category)
                continue
        # Fallback: use the whole sentence if short enough
        if len(sentence) <= 80:
            capabilities.append(sentence)
        else:
            # Last resort: truncate to first clause
            for delim in [":", ",", " - "]:
                if delim in sentence:
                    first_part = sentence.split(delim, 1)[0].strip()
                    if len(first_part) <= 80:
                        capabilities.append(first_part)
                        break
            else:
                # Just truncate with ellipsis
                capabilities.append(sentence[:77] + "...")

    return _normalize_capabilities(
        capabilities if capabilities else [raw[:80]] if len(raw) > 80 else [raw]
    )


def _normalize_capabilities(capabilities: list[str]) -> list[str]:
    """Normalize and deduplicate a list of capability strings.

    Applies the following transformations to each entry:
    - Strip trailing periods
    - Replace underscores with spaces
    - Normalize unicode dashes (non-breaking hyphen, en-dash, em-dash) to ASCII hyphen
    - Capitalize first letter if lowercase
    - Collapse multiple spaces to a single space
    - Deduplicate case-insensitively (first occurrence wins)
    """
    seen: set[str] = set()
    result: list[str] = []

    for cap in capabilities:
        # Strip trailing periods
        cap = cap.strip().rstrip(".")
        if not cap:
            continue
        # Replace underscores with spaces
        cap = cap.replace("_", " ")
        # Normalize unicode dashes to ASCII hyphen
        cap = cap.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
        # Collapse multiple spaces to single space
        cap = re.sub(r" {2,}", " ", cap)
        # Capitalize first letter if lowercase
        if cap[0].islower():
            cap = cap[0].upper() + cap[1:]

        lower = cap.lower()
        if lower not in seen:
            seen.add(lower)
            result.append(cap)

    return result


@tracer.capture_method
def fetch_console_metadata(region: str) -> dict:
    """
    Fetch extended model metadata via direct Bedrock REST API with SigV4 signing.

    Uses the x-console-consumer header to get consoleIDEMetadata which includes
    context windows, descriptions, languages, categories, feature support,
    capabilities, and more.

    Returns dict mapping model_id -> metadata dict. Returns empty dict on any error.
    """
    try:
        session = boto3.Session(region_name=region)
        credentials = session.get_credentials()
        if not credentials:
            logger.warning(
                "No credentials available for console metadata fetch",
                extra={"region": region},
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

            # Extract top-level fields
            meta["model_family"] = model.get("modelFamily", "")
            meta["guardrails_supported"] = model.get("guardrailsSupported", False)
            meta["batch_supported"] = model.get("batchSupported", {})

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

                    # Languages (comma/and-separated string)
                    lang_str = desc.get("supportedLanguages")
                    if lang_str and isinstance(lang_str, str):
                        cleaned = lang_str.rstrip(".").replace(" and ", ", ")
                        meta["languages"] = [
                            l.strip() for l in cleaned.split(",") if l.strip()
                        ]

                    # Use cases (semicolon or comma-separated string)
                    use_str = desc.get("supportedUseCases")
                    if use_str and isinstance(use_str, str):
                        meta["use_cases"] = parse_use_cases(use_str)

                    # Model attributes (capabilities from AWS)
                    model_attrs = desc.get("modelAttributes")
                    if model_attrs and isinstance(model_attrs, str):
                        attrs = _parse_model_attributes(model_attrs)
                        if attrs:
                            meta["model_attributes"] = attrs

                    # Release date (epoch timestamp)
                    release_date = desc.get("releaseDate")
                    if release_date:
                        meta["release_date"] = release_date

                    # Feature support (agent, knowledgeBase, flows, etc.)
                    feature_support = console_ide.get("featureSupport", {})
                    if feature_support:
                        meta["feature_support"] = {
                            "agent": feature_support.get("agent", {}),
                            "knowledge_base": feature_support.get("knowledgeBase", {}),
                            "flow": feature_support.get("flow", {}),
                            "guardrails": feature_support.get("guardrails", {}),
                            "prompt_caching": feature_support.get(
                                "explicitPromptCaching", {}
                            ),
                            "intelligent_routing": feature_support.get(
                                "intelligentPromptRouting", {}
                            ),
                            "model_evaluation": feature_support.get(
                                "modelEvaluation", {}
                            ),
                            "prompt_management": feature_support.get("prompt", {}),
                            "batch_inference": feature_support.get(
                                "batchInference", {}
                            ),
                            "latency_optimized": feature_support.get(
                                "latencyOptimized", {}
                            ),
                            "system_tools": feature_support.get("systemTool", {}).get(
                                "supportedSystemTools", []
                            ),
                        }

                    # Invoke chat features (function calling, citations, etc.)
                    converse_meta = console_ide.get("converse", {})
                    invoke_features = converse_meta.get("invokeChatFeatures", {})
                    if invoke_features:
                        meta["chat_features"] = {
                            "function_calling": invoke_features.get(
                                "functionToolSupported", False
                            ),
                            "function_calling_streaming": invoke_features.get(
                                "functionToolStreamSupported", False
                            ),
                            "citations": invoke_features.get(
                                "citationsSupported", False
                            ),
                            "documents": invoke_features.get(
                                "documentsSupported", False
                            ),
                            "chat_history": invoke_features.get(
                                "chatHistorySupported", False
                            ),
                            "system_role": invoke_features.get(
                                "systemRoleSupported", False
                            ),
                            "reasoning": invoke_features.get("reasoningSupported", {}),
                            "supported_image_types": invoke_features.get(
                                "userImageTypesSupported", []
                            ),
                            "supported_video_types": invoke_features.get(
                                "userVideoTypesSupported", []
                            ),
                            "supported_audio_types": invoke_features.get(
                                "userAudioTypesSupported", []
                            ),
                            "supported_document_types": invoke_features.get(
                                "userPassthroughDocumentTypesSupported", []
                            ),
                        }

                    # Max tokens from console metadata
                    if converse_meta.get("maxTokensMaximum"):
                        meta["max_output_tokens"] = int(
                            converse_meta["maxTokensMaximum"]
                        )

                except (json.JSONDecodeError, TypeError):
                    pass

            # Fallback: extract from description object if not in consoleIDEMetadata
            desc_obj = model.get("description", {})
            if isinstance(desc_obj, dict):
                if "max_context_window" not in meta:
                    max_cw_str = desc_obj.get("maxContextWindow")
                    if max_cw_str:
                        parsed = parse_context_window_string(str(max_cw_str))
                        if parsed:
                            meta["max_context_window"] = parsed

            # Fallback: extract max output tokens from top-level converse object
            converse = model.get("converse", {})
            if isinstance(converse, dict) and "max_output_tokens" not in meta:
                max_tokens = converse.get("maxTokensMaximum")
                if max_tokens and isinstance(max_tokens, (int, float)):
                    meta["max_output_tokens"] = int(max_tokens)

            if meta:
                metadata_by_id[model_id] = meta

        logger.info(
            "Fetched console metadata",
            extra={"model_count": len(metadata_by_id), "region": region},
        )
        return metadata_by_id

    except (URLError, HTTPError) as e:
        logger.warning(
            "Failed to fetch console metadata",
            extra={"region": region, "error": str(e)},
        )
        return {}
    except Exception as e:
        logger.warning(
            "Unexpected error fetching console metadata",
            extra={"region": region, "error": str(e)},
        )
        return {}


def get_documentation_links(model_id: str, provider: str) -> dict:
    """Get documentation links based on provider and model from config."""
    config = get_config_loader()
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
        "model_family": "",  # Populated from console metadata
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
        # Extraction metadata (where model was discovered in API, not where it can be invoked)
        "extraction_regions": [region],
        # Capabilities and use cases - populated from console metadata
        "model_capabilities": [],  # From modelAttributes
        "model_use_cases": [],  # From supportedUseCases
        "languages_supported": [],  # From supportedLanguages
        # Feature support from API
        "feature_support": {},  # Agent, KB, flows, guardrails, etc.
        "chat_features": {},  # Function calling, citations, etc.
        "guardrails_supported": False,
        "batch_supported": {},
        # Token specs
        "max_context_window": None,
        "max_output_tokens": None,
        # Descriptions
        "description": "",
        "short_description": "",
        # Other fields
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


@tracer.capture_method
def extract_models(
    bedrock_client,
    region: str,
    s3_client=None,
    bucket: str = None,
    execution_id: str = None,
) -> tuple[list[dict], str | None]:
    """
    Extract all foundation models from Bedrock API.

    Makes two calls:
    1. Standard boto3 list_foundation_models() for core model data
    2. Direct REST API with x-console-consumer header for extended metadata
       (context windows, descriptions, languages, capabilities, feature support)

    Also caches the raw API response for reuse by downstream Lambdas (e.g., regional-availability).

    Args:
        bedrock_client: Boto3 Bedrock client
        region: AWS region to extract models from
        s3_client: Optional S3 client for caching
        bucket: Optional S3 bucket for caching
        execution_id: Optional execution ID for cache path

    Returns:
        Tuple of (list of model dictionaries, cache_key or None)
    """
    models = []
    raw_model_summaries = []
    cache_key = None

    try:
        response = bedrock_client.list_foundation_models()
        model_summaries = response.get("modelSummaries", [])
        raw_model_summaries = model_summaries  # Store for caching

        for raw_model in model_summaries:
            processed = process_model_data(raw_model, region)
            models.append(processed)

        logger.info(
            "Extracted models from Bedrock API",
            extra={"model_count": len(models), "region": region},
        )

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code in ("AccessDeniedException", "UnrecognizedClientException"):
            logger.warning(
                "Access denied or region not enabled",
                extra={"region": region, "error_code": error_code},
            )
        elif error_code == "InvalidIdentityToken":
            logger.warning(
                "Invalid token for region - region may require opt-in",
                extra={"region": region},
            )
        else:
            logger.error(
                "Error listing models", extra={"region": region, "error": str(e)}
            )

    except Exception as e:
        logger.warning(
            "Unexpected error extracting models",
            extra={"region": region, "error": str(e)},
        )

    # Fetch console metadata and populate model fields directly
    console_metadata = fetch_console_metadata(region)
    if console_metadata:
        enriched_count = 0
        for model in models:
            model_id = model.get("model_id", "")
            if model_id in console_metadata:
                meta = console_metadata[model_id]
                enriched_count += 1

                # Populate model fields from console metadata
                if meta.get("model_family"):
                    model["model_family"] = meta["model_family"]
                if meta.get("max_context_window"):
                    model["max_context_window"] = meta["max_context_window"]
                if meta.get("max_output_tokens"):
                    model["max_output_tokens"] = meta["max_output_tokens"]
                if meta.get("description"):
                    model["description"] = meta["description"]
                if meta.get("short_description"):
                    model["short_description"] = meta["short_description"]
                if meta.get("languages"):
                    model["languages_supported"] = meta["languages"]
                if meta.get("use_cases"):
                    model["model_use_cases"] = meta["use_cases"]
                if meta.get("model_attributes"):
                    model["model_capabilities"] = meta["model_attributes"]
                if meta.get("release_date"):
                    model["model_lifecycle"]["release_date"] = meta["release_date"]
                if meta.get("feature_support"):
                    model["feature_support"] = meta["feature_support"]
                if meta.get("chat_features"):
                    model["chat_features"] = meta["chat_features"]
                if meta.get("guardrails_supported"):
                    model["guardrails_supported"] = meta["guardrails_supported"]
                if meta.get("batch_supported"):
                    model["batch_supported"] = meta["batch_supported"]

                # Keep raw console_metadata for debugging/reference
                model["console_metadata"] = meta

        logger.info(
            "Enriched models with console metadata",
            extra={
                "enriched_count": enriched_count,
                "total_models": len(models),
                "region": region,
            },
        )

    # Cache raw API response for reuse by regional-availability
    if s3_client and bucket and execution_id and raw_model_summaries:
        cache_key = (
            f"executions/{execution_id}/cache/list_foundation_models_{region}.json"
        )
        try:
            cache_data = {
                "region": region,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model_summaries": raw_model_summaries,
            }
            write_to_s3(s3_client, bucket, cache_key, cache_data)
            logger.info(
                "Cached raw model data for downstream use",
                extra={
                    "region": region,
                    "cache_key": cache_key,
                    "model_count": len(raw_model_summaries),
                },
            )
        except Exception as e:
            logger.warning(
                "Failed to cache model data, continuing without cache",
                extra={"region": region, "error": str(e)},
            )
            cache_key = None

    return models, cache_key


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
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
            "modelCount": 108,
            "cacheKey": "executions/{id}/cache/list_foundation_models_us-east-1.json"
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

    # Extract execution_id from s3Key for caching
    # Format: executions/{execution_id}/models/{region}.json
    execution_id = None
    if s3_key and s3_key.startswith("executions/"):
        parts = s3_key.split("/")
        if len(parts) >= 2:
            execution_id = parts[1]

    logger.info(
        "Starting model extraction",
        extra={"region": region, "execution_id": execution_id},
    )

    try:
        bedrock_client = get_bedrock_client(region)
        s3_client = get_s3_client() if s3_bucket else None

        # Extract models and cache raw API response
        models, cache_key = extract_models(
            bedrock_client,
            region,
            s3_client=s3_client,
            bucket=s3_bucket,
            execution_id=execution_id,
        )

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
            write_to_s3(s3_client, s3_bucket, s3_key, output_data)
        else:
            logger.info(
                "Dry run - skipping S3 write",
                extra={"model_count": len(models), "bucket": s3_bucket, "key": s3_key},
            )

        duration_ms = int((time.time() - start_time) * 1000)

        # Add metrics
        metrics.add_metric(
            name="ModelsExtracted", unit=MetricUnit.Count, value=len(models)
        )
        metrics.add_metric(
            name="ExtractionDurationMs", unit=MetricUnit.Milliseconds, value=duration_ms
        )
        metrics.add_dimension(name="Region", value=region)

        logger.info(
            "Model extraction complete",
            extra={
                "model_count": len(models),
                "region": region,
                "duration_ms": duration_ms,
                "cache_key": cache_key,
            },
        )

        result = {
            "status": "SUCCESS",
            "region": region,
            "s3Key": s3_key,
        }

        # Include cache key if caching was successful (needed by downstream Lambdas)
        if cache_key:
            result["cacheKey"] = cache_key

        return result

    except Exception as e:
        logger.exception(
            "Failed to extract models", extra={"region": region, "error": str(e)}
        )
        return {
            "status": "FAILED",
            "region": region,
            "errorType": type(e).__name__,
            "errorMessage": str(e),
            "retryable": "Throttling" in str(e),
        }
