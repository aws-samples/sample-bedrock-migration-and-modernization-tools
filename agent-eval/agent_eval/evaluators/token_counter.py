# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Token counter for AgentCore traces using Amazon Bedrock CountTokens API.

Extracts model input/output text from normalized trace steps and calls
CountTokens to get exact token counts. Enriches steps with token data
so the cost module can calculate accurate LLM costs.

Usage:
    from agent_eval.evaluators.token_counter import enrich_tokens_from_bedrock
    enriched_run = enrich_tokens_from_bedrock(normalized_run, region="us-east-1")
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Step kinds that represent model invocations
MODEL_KINDS = {"MODEL_INVOKE", "LLM_OUTPUT_CHUNK"}


def _extract_text_from_step(step: Dict[str, Any]) -> Optional[str]:
    """Extract the text content from a step's raw or attributes fields."""
    raw = step.get("raw", {})
    attrs = step.get("attributes", {})

    # Try body.content[].text (OTEL bedrock-runtime event format)
    body = raw.get("body", {})
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return body if body.strip() else None

    if isinstance(body, dict):
        # body.content[].text
        content = body.get("content", [])
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
            if texts:
                return " ".join(texts)
        # body.message.content[].text
        msg = body.get("message", {})
        if isinstance(msg, dict):
            content = msg.get("content", [])
            if isinstance(content, list):
                texts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
                if texts:
                    return " ".join(texts)

    return None


def count_tokens_bedrock(
    text: str,
    model_id: str,
    region: str = "us-east-1",
) -> int:
    """Call Bedrock CountTokens API for exact token count."""
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 required for CountTokens. Install with: pip install boto3")

    client = boto3.client("bedrock-runtime", region_name=region)
    resp = client.count_tokens(
        modelId=model_id,
        input={"converse": {"messages": [{"role": "user", "content": [{"text": text}]}]}},
    )
    return resp.get("inputTokens", resp.get("totalTokens", 0))


def enrich_tokens_from_bedrock(
    normalized_run: Dict[str, Any],
    region: str = "us-east-1",
    default_model_id: str = "anthropic.claude-sonnet-4-20250514-v1:0",
) -> Dict[str, Any]:
    """
    Enrich normalized run steps with exact token counts from Bedrock CountTokens.

    Modifies steps in-place, adding prompt_tokens and completion_tokens to
    attributes of MODEL_INVOKE steps that have extractable text content.

    Args:
        normalized_run: Normalized run dict
        region: AWS region for Bedrock API calls
        default_model_id: Model ID to use if not found in step attributes

    Returns:
        The same normalized_run dict (modified in-place) with token counts added
    """
    enriched = 0
    errors = 0

    for turn in normalized_run.get("turns", []):
        # Collect input/output texts for this turn's model steps
        model_steps = [s for s in turn.get("steps", []) if s.get("kind") in MODEL_KINDS]

        for step in model_steps:
            attrs = step.get("attributes", {})

            # Skip if already has token counts
            if attrs.get("prompt_tokens") or attrs.get("completion_tokens"):
                continue

            text = _extract_text_from_step(step)
            if not text:
                continue

            model_id = (
                attrs.get("model_id")
                or attrs.get("gen_ai.request.model")
                or default_model_id
            )

            try:
                token_count = count_tokens_bedrock(text, model_id, region)
                # Heuristic: if this is an input event (no finish_reason), it's prompt tokens
                # If it has a finish_reason or role=assistant, it's completion tokens
                raw = step.get("raw", {})
                body = raw.get("body", {})
                if isinstance(body, str):
                    try:
                        body = json.loads(body)
                    except (json.JSONDecodeError, TypeError):
                        body = {}

                is_output = (
                    body.get("finish_reason")
                    or body.get("message", {}).get("role") == "assistant"
                )

                if is_output:
                    attrs["completion_tokens"] = token_count
                else:
                    attrs["prompt_tokens"] = token_count

                step["attributes"] = attrs
                enriched += 1
            except Exception as e:
                logger.warning("CountTokens failed for step %s: %s", step.get("name"), e)
                errors += 1

    logger.info("Token enrichment: %d steps enriched, %d errors", enriched, errors)
    return normalized_run
