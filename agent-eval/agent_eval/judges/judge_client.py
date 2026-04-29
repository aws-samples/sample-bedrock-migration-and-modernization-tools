# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Abstract base class for judge client implementations.

This module defines the interface that all judge clients must implement
to execute rubric evaluations via LLM models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from agent_eval.judges.exceptions import (
    ValidationResult,
    ValidationError,
    APIError,
    TimeoutError as JudgeTimeoutError
)


@dataclass
class JudgeResponse:
    """
    Structured response from a judge execution.
    
    Attributes:
        score: Numeric score (for numeric rubrics) or categorical value
        reasoning: Judge's explanation for the score
        raw_response: Raw response from the judge API (string, dict, or list)
        latency_ms: Execution time in milliseconds
        metadata: Additional metadata (model version, tokens, etc.)
    """
    score: Optional[Any]
    reasoning: Optional[str]
    raw_response: Union[str, Dict[str, Any], List[Any]]  # Flexible to match JobResult
    latency_ms: float
    metadata: Dict[str, Any]


class JudgeClient(ABC):
    """
    Abstract base class for judge client implementations.
    
    All judge clients (Amazon Bedrock, OpenAI, Anthropic, etc.) must extend
    this class and implement the execute_judge method.
    """
    
    def __init__(
        self,
        judge_id: str,
        model_id: str,
        params: Dict[str, Any],
        timeout_seconds: int = 30
    ):
        """
        Initialize judge client.
        
        Args:
            judge_id: Unique identifier for this judge
            model_id: Model identifier (provider-specific)
            params: Model-specific parameters (temperature, max_tokens, etc.)
            timeout_seconds: Maximum execution time per request
        """
        self.judge_id = judge_id
        self.model_id = model_id
        self.params = params
        self.timeout_seconds = timeout_seconds
    
    @abstractmethod
    async def execute_judge(
        self,
        prompt: str,
        rubric_id: str,
        scoring_scale: Dict[str, Any]
    ) -> JudgeResponse:
        """
        Execute judge evaluation for a given prompt.
        
        This method must be implemented by all concrete judge clients.
        It should:
        1. Send the prompt to the judge API
        2. Parse and validate the response
        3. Extract score and reasoning
        4. Return structured JudgeResponse
        
        Args:
            prompt: Evaluation prompt containing evidence and instructions
            rubric_id: Identifier for the rubric being evaluated
            scoring_scale: Scoring scale definition from rubric
            
        Returns:
            JudgeResponse with score, reasoning, and metadata
            
        Raises:
            JudgeTimeoutError: If execution exceeds timeout_seconds
            ValidationError: If response doesn't match expected format
            APIError: If judge API call fails
        """
        pass
    
    @abstractmethod
    async def validate_response(
        self,
        response: Union[str, Dict[str, Any], List[Any]],
        scoring_scale: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate judge response against expected format.
        
        Returns structured validation result with detailed error information
        for better retry decisions and error reporting.
        
        Args:
            response: Raw response from judge API (string, dict, or list)
            scoring_scale: Expected scoring scale from rubric
            
        Returns:
            ValidationResult with:
            - is_valid: True if response is valid
            - error_code: Machine-readable error code (if invalid)
            - message: Human-readable error message (if invalid)
            - field: Field that failed validation (if applicable)
            - expected: Expected value/format (if applicable)
            - actual: Actual value received (if applicable)
        """
        pass
    
    def build_prompt(
        self,
        evidence: Dict[str, Any],
        rubric_description: str,
        scoring_scale: Dict[str, Any],
        rubric_id: str = None,
        turn_id: str = None,
        run_id: str = None
    ) -> str:
        """
        Build evaluation prompt from evidence and rubric.
        
        This default implementation can be overridden by subclasses
        for provider-specific prompt formatting.
        
        CRITICAL REQUIREMENTS:
        - Output MUST be valid JSON only (no prose, no markdown)
        - Output MUST match format: {"score": <value>, "reasoning": "<text>"}
        - For categorical scales, score MUST be one of the allowed values
        - For numeric scales, score MUST be within min/max range
        
        Args:
            evidence: Extracted evidence from NormalizedRun
            rubric_description: Description of what to evaluate
            scoring_scale: Scoring scale definition
            rubric_id: Rubric identifier (for traceability)
            turn_id: Turn identifier (for traceability, if turn-scoped)
            run_id: Run identifier (for traceability)
            
        Returns:
            Formatted prompt string with strict JSON output requirements
        """
        scale_type = scoring_scale.get('type', 'numeric')
        scale_constraints = self._format_scoring_scale_constraints(scoring_scale)
        
        # Build traceability context
        context_parts = []
        if run_id:
            context_parts.append(f"Run ID: {run_id}")
        if turn_id:
            context_parts.append(f"Turn ID: {turn_id}")
        if rubric_id:
            context_parts.append(f"Rubric ID: {rubric_id}")
        context_str = " | ".join(context_parts) if context_parts else "N/A"
        
        prompt_parts = [
            "# Evaluation Task",
            f"\n{rubric_description}\n",
            "\n# Context",
            f"\n{context_str}\n",
            "\n# Scoring Scale",
            f"\n{scale_constraints}\n",
            "\n# Evidence",
            f"\n{self._format_evidence(evidence)}\n",
            "\n# Output Requirements",
            "\nYou MUST respond with ONLY valid JSON in this exact format:",
            '{"score": <value>, "reasoning": "<explanation>"}',
            "\nDo NOT include:",
            "- Markdown formatting (no ```json blocks)",
            "- Additional prose or commentary",
            "- Multiple JSON objects",
            "\nThe score field MUST:",
        ]
        
        if scale_type == 'numeric':
            min_val = scoring_scale.get('min', 0)
            max_val = scoring_scale.get('max', 5)
            prompt_parts.append(f"- Be a number between {min_val} and {max_val} (inclusive)")
        elif scale_type == 'categorical':
            values = scoring_scale.get('values', [])
            values_str = ', '.join(f'"{v}"' for v in values)
            prompt_parts.append(f"- Be exactly one of: {values_str}")
        
        prompt_parts.extend([
            "\nThe reasoning field MUST:",
            "- Explain your score based on the evidence",
            "- Be concise (2-3 sentences)",
            "- Reference specific evidence when possible"
        ])
        
        return "\n".join(prompt_parts)
    
    def _format_scoring_scale_constraints(self, scoring_scale: Dict[str, Any]) -> str:
        """
        Format scoring scale with explicit constraints for prompt.
        
        Includes allowed values/ranges to reduce invalid responses.
        """
        scale_type = scoring_scale.get('type', 'numeric')
        
        if scale_type == 'numeric':
            min_val = scoring_scale.get('min', 0)
            max_val = scoring_scale.get('max', 5)
            return (
                f"Type: Numeric\n"
                f"Range: {min_val} to {max_val} (inclusive)\n"
                f"Interpretation: Higher scores indicate better performance"
            )
        elif scale_type == 'categorical':
            values = scoring_scale.get('values', [])
            values_list = '\n'.join(f"  - {v}" for v in values)
            return (
                f"Type: Categorical\n"
                f"Allowed values:\n{values_list}\n"
                f"You MUST use exactly one of these values"
            )
        else:
            return str(scoring_scale)
    
    def _format_evidence(self, evidence: Dict[str, Any]) -> str:
        """Format evidence for prompt."""
        import json
        return json.dumps(evidence, indent=2)
