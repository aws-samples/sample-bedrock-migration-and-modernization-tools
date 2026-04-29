# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Mock judge client for testing and development.

This module provides a mock implementation of JudgeClient that can be used
for testing, development, and demonstrations without requiring actual LLM API calls.
"""

import asyncio
from typing import Any, Dict, Optional

from agent_eval.judges.judge_client import JudgeClient, JudgeResponse
from agent_eval.judges.exceptions import (
    ValidationResult,
    ValidationError,
    APIError,
    TimeoutError as JudgeTimeoutError
)


class MockJudgeClient(JudgeClient):
    """
    Mock judge client for testing and development.
    
    Supports:
    - Deterministic output (configurable scores)
    - Failure simulation
    - Retry scenarios
    - Latency simulation
    """
    
    def __init__(
        self,
        judge_id: str,
        model_id: str = "mock-model",
        params: Optional[Dict[str, Any]] = None,
        timeout_seconds: int = 30,
        # Mock-specific parameters
        deterministic_score: Optional[float] = None,
        should_fail: bool = False,
        failure_mode: str = "api_error",  # "api_error", "timeout", "invalid_transport_payload", "invalid_semantic_response"
        fail_count: int = 0,  # Number of times to fail before succeeding
        latency_ms: float = 100.0
    ):
        """
        Initialize mock judge client.
        
        Args:
            judge_id: Judge identifier
            model_id: Model identifier
            params: Model parameters
            timeout_seconds: Timeout in seconds
            deterministic_score: Fixed score to return (if None, uses rubric-based logic)
            should_fail: Whether to simulate failure
            failure_mode: Type of failure to simulate:
                - "api_error": Simulates API/network failure (raises APIError)
                - "timeout": Simulates timeout (raises TimeoutError)
                - "invalid_transport_payload": Simulates JSON parse failure (raises ValidationError with INVALID_JSON)
                - "invalid_semantic_response": Returns valid JSON but semantically invalid (missing/invalid fields)
            fail_count: Number of failures before success (for retry testing)
            latency_ms: Simulated latency in milliseconds
        """
        super().__init__(judge_id, model_id, params or {}, timeout_seconds)
        self.deterministic_score = deterministic_score
        self.should_fail = should_fail
        self.failure_mode = failure_mode
        self.fail_count = fail_count
        self.latency_ms = latency_ms
        self._call_count = 0
    
    async def execute_judge(
        self,
        prompt: str,
        rubric_id: str,
        scoring_scale: Dict[str, Any]
    ) -> JudgeResponse:
        """
        Execute mock judge evaluation.
        
        Returns deterministic scores or simulates failures based on configuration.
        """
        self._call_count += 1
        
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000.0)
        
        # Simulate failures for retry testing
        if self._call_count <= self.fail_count:
            if self.failure_mode == "timeout":
                raise JudgeTimeoutError(
                    f"Mock timeout on attempt {self._call_count}",
                    timeout_seconds=self.timeout_seconds
                )
            elif self.failure_mode == "api_error":
                raise APIError(f"Mock API error on attempt {self._call_count}")
            elif self.failure_mode == "invalid_transport_payload":
                # Simulate JSON parse failure (like Amazon Bedrock returning malformed JSON)
                raise ValidationError(
                    message=f"Mock JSON parse error on attempt {self._call_count}: invalid json {{",
                    error_code="INVALID_JSON",
                    field="raw_response",
                    expected="valid JSON",
                    actual="invalid json {"
                )
            elif self.failure_mode == "invalid_semantic_response":
                # Return valid JSON but semantically invalid (missing required fields)
                return JudgeResponse(
                    score=None,
                    reasoning=None,
                    raw_response={"incomplete": "response"},  # Valid JSON, but missing score/reasoning
                    latency_ms=self.latency_ms,
                    metadata={"mock": True, "attempt": self._call_count}
                )
            # Backward compatibility: treat "invalid_response" as "invalid_semantic_response"
            elif self.failure_mode == "invalid_response":
                return JudgeResponse(
                    score=None,
                    reasoning=None,
                    raw_response={"incomplete": "response"},
                    latency_ms=self.latency_ms,
                    metadata={"mock": True, "attempt": self._call_count}
                )
        
        # Simulate permanent failure
        if self.should_fail:
            if self.failure_mode == "timeout":
                raise JudgeTimeoutError(
                    "Mock timeout (permanent)",
                    timeout_seconds=self.timeout_seconds
                )
            elif self.failure_mode == "api_error":
                raise APIError("Mock API error (permanent)")
            elif self.failure_mode == "invalid_transport_payload":
                # Simulate JSON parse failure
                raise ValidationError(
                    message="Mock JSON parse error (permanent): invalid json {",
                    error_code="INVALID_JSON",
                    field="raw_response",
                    expected="valid JSON",
                    actual="invalid json {"
                )
            elif self.failure_mode == "invalid_semantic_response":
                # Return valid JSON but semantically invalid
                return JudgeResponse(
                    score=None,
                    reasoning=None,
                    raw_response={"incomplete": "response"},
                    latency_ms=self.latency_ms,
                    metadata={"mock": True}
                )
            # Backward compatibility: treat "invalid_response" as "invalid_semantic_response"
            elif self.failure_mode == "invalid_response":
                return JudgeResponse(
                    score=None,
                    reasoning=None,
                    raw_response={"incomplete": "response"},
                    latency_ms=self.latency_ms,
                    metadata={"mock": True}
                )
        
        # Generate deterministic score
        score = self._generate_score(rubric_id, scoring_scale)
        reasoning = self._generate_reasoning(rubric_id, score)
        
        return JudgeResponse(
            score=score,
            reasoning=reasoning,
            raw_response={"score": score, "reasoning": reasoning},
            latency_ms=self.latency_ms,
            metadata={"mock": True, "rubric_id": rubric_id, "attempt": self._call_count}
        )
    
    async def validate_response(
        self,
        response: Any,
        scoring_scale: Dict[str, Any]
    ) -> ValidationResult:
        """Validate mock judge response."""
        # Simple validation for mock responses
        if isinstance(response, dict) and "score" in response and "reasoning" in response:
            return ValidationResult(is_valid=True)
        else:
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_FORMAT",
                message="Response missing required fields",
                field="score",
                expected="dict with score and reasoning",
                actual=str(type(response))
            )
    
    def _generate_score(self, rubric_id: str, scoring_scale: Dict[str, Any]) -> Any:
        """Generate deterministic score based on rubric and scale."""
        # Use configured deterministic score if provided
        if self.deterministic_score is not None:
            return self.deterministic_score
        
        # Otherwise, use rubric-based logic
        scale_type = scoring_scale.get("type", "numeric")
        
        if scale_type == "numeric":
            # Return mid-range score
            min_val = scoring_scale.get("min", 0)
            max_val = scoring_scale.get("max", 5)
            return (min_val + max_val) / 2
        elif scale_type == "categorical":
            # Return first allowed value
            values = scoring_scale.get("values", ["pass"])
            return values[0]
        else:
            return 3  # Default numeric score
    
    def _generate_reasoning(self, rubric_id: str, score: Any) -> str:
        """Generate deterministic reasoning."""
        return f"Mock evaluation for {rubric_id}. Score: {score}. This is a deterministic mock response."
