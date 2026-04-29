# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LiteLLM provider-agnostic judge client implementation.

Uses LiteLLM (https://github.com/BerriAI/litellm) to provide a unified
interface across 100+ LLM providers. Model routing is controlled by the
model_id string in judge config:
  - "bedrock/anthropic.claude-sonnet-4-20250514" → Amazon Bedrock
  - "anthropic/claude-sonnet-4-20250514" → Anthropic direct
  - "gpt-4o" → OpenAI
  - "gemini/gemini-pro" → Google
"""

import json
import time
import logging
from typing import Any, Dict, List, Optional, Union

from agent_eval.judges.judge_client import JudgeClient, JudgeResponse
from agent_eval.judges.exceptions import (
    ValidationResult,
    APIError,
    TimeoutError as JudgeTimeoutError,
)

logger = logging.getLogger(__name__)


class LiteLLMJudgeClient(JudgeClient):
    """Provider-agnostic judge client using LiteLLM."""

    def __init__(
        self,
        judge_id: str,
        model_id: str,
        params: Dict[str, Any],
        timeout_seconds: int = 30,
        **kwargs,
    ):
        super().__init__(judge_id, model_id, params, timeout_seconds)
        try:
            import litellm
            self._litellm = litellm
            litellm.drop_params = True  # ignore unsupported params per provider
        except ImportError:
            raise ImportError(
                "litellm is required for LiteLLMJudgeClient. "
                "Install with: pip install litellm"
            )

    async def execute_judge(
        self,
        prompt: str,
        rubric_id: str,
        scoring_scale: Dict[str, Any],
    ) -> JudgeResponse:
        """Execute judge evaluation via LiteLLM."""
        messages = [{"role": "user", "content": prompt}]
        start = time.time()

        try:
            response = self._litellm.completion(
                model=self.model_id,
                messages=messages,
                temperature=self.params.get("temperature", 0.0),
                max_tokens=self.params.get("max_tokens", 1024),
                timeout=self.timeout_seconds,
            )
        except Exception as e:
            raise APIError(
                f"LiteLLM call failed for judge '{self.judge_id}' "
                f"model '{self.model_id}': {e}"
            ) from e

        latency_ms = (time.time() - start) * 1000
        raw_text = response.choices[0].message.content or ""
        usage = response.get("usage", {}) or {}

        # Parse JSON response
        score, reasoning = self._parse_judge_response(raw_text, scoring_scale)

        return JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_response=raw_text,
            latency_ms=latency_ms,
            metadata={
                "model": self.model_id,
                "provider": "litellm",
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
            },
        )

    async def validate_response(
        self,
        response: Union[str, Dict[str, Any], List[Any]],
        scoring_scale: Dict[str, Any],
    ) -> ValidationResult:
        """Validate judge response against scoring scale."""
        try:
            if isinstance(response, str):
                parsed = json.loads(response)
            else:
                parsed = response

            score = parsed.get("score")
            if score is None:
                return ValidationResult(
                    is_valid=False, error_code="MISSING_SCORE",
                    message="Response missing 'score' field",
                )

            scale_type = scoring_scale.get("type", "numeric")
            if scale_type == "numeric":
                min_val = scoring_scale.get("min", 0)
                max_val = scoring_scale.get("max", 5)
                if not isinstance(score, (int, float)) or not (min_val <= score <= max_val):
                    return ValidationResult(
                        is_valid=False, error_code="SCORE_OUT_OF_RANGE",
                        message=f"Score {score} not in [{min_val}, {max_val}]",
                    )
            elif scale_type == "categorical":
                allowed = scoring_scale.get("values", [])
                if score not in allowed:
                    return ValidationResult(
                        is_valid=False, error_code="INVALID_CATEGORY",
                        message=f"Score '{score}' not in {allowed}",
                    )

            return ValidationResult(is_valid=True)
        except (json.JSONDecodeError, AttributeError) as e:
            return ValidationResult(
                is_valid=False, error_code="PARSE_ERROR",
                message=f"Failed to parse response: {e}",
            )

    def _parse_judge_response(
        self, raw_text: str, scoring_scale: Dict[str, Any]
    ) -> tuple:
        """Extract score and reasoning from raw LLM response."""
        try:
            # Try direct JSON parse
            parsed = json.loads(raw_text.strip())
            return parsed.get("score"), parsed.get("reasoning")
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        import re
        match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", raw_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                return parsed.get("score"), parsed.get("reasoning")
            except json.JSONDecodeError:
                pass

        logger.warning(
            "Could not parse JSON from judge response for %s", self.judge_id
        )
        return None, raw_text
