# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Job Builder for Agent Traces Evaluator

This module builds JudgeJob queues from rubrics, judges, and NormalizedRun data.
It filters LLM-required rubrics, extracts evidence, and creates job instances.
"""

import hashlib
import json
import warnings
from typing import Dict, Any, List, Optional, Set
from .models import JudgeJob
from .evidence import EvidenceExtractor, EvidenceExtractionError


class JobBuilderError(Exception):
    """Raised when job building fails."""
    pass


class JobBuilder:
    """Builds JudgeJob queue from rubrics, judges, and NormalizedRun."""
    
    # Configuration constants
    MAX_JUDGES = 5
    MAX_REPEATS = 10
    MAX_JOBS_WARNING_THRESHOLD = 1000
    MAX_PAYLOAD_SIZE = 100000  # 100KB
    
    def __init__(self, evidence_budget: int = 10000):
        """
        Initialize job builder.
        
        Args:
            evidence_budget: Maximum characters per rubric payload (default 10,000)
        """
        self.evidence_extractor = EvidenceExtractor(evidence_budget=evidence_budget)
    
    def build_jobs(
        self,
        normalized_run: Dict[str, Any],
        rubrics: List[Any],  # List[Rubric] from rubric_loader
        judges: List[Any],  # List[Judge] from judge_config_schema
        deterministic_metrics: Optional[Dict[str, Any]] = None
    ) -> List[JudgeJob]:
        """
        Build JudgeJob queue.
        
        Job count = (LLM-required rubrics × judges × repeats)
        
        Args:
            normalized_run: The NormalizedRun data dictionary
            rubrics: List of Rubric objects
            judges: List of Judge objects
            deterministic_metrics: Optional deterministic metrics to include in evidence
            
        Returns:
            List of JudgeJob objects ready for execution
            
        Raises:
            JobBuilderError: If job building fails
        """
        # Validate judge count
        if len(judges) > self.MAX_JUDGES:
            raise JobBuilderError(
                f"Too many judges: {len(judges)}. Maximum allowed is {self.MAX_JUDGES}"
            )
        
        # Filter to LLM-required rubrics only
        llm_rubrics = self.filter_llm_rubrics(rubrics)
        
        if not llm_rubrics:
            # No LLM rubrics to evaluate
            return []
        
        if not judges:
            raise JobBuilderError("No judges provided for LLM rubric evaluation")
        
        # Extract run_id for job metadata
        run_id = normalized_run.get("run_id")
        if not run_id:
            raise JobBuilderError("NormalizedRun missing required field 'run_id'")
        
        # Validate turns and detect duplicates
        turns = normalized_run.get("turns", [])
        self._validate_turns(turns)
        
        # Build jobs for each (rubric × judge × repeat) combination
        jobs = []
        
        for rubric in llm_rubrics:
            # Validate rubric scope
            if not hasattr(rubric, "scope") or rubric.scope not in {"turn", "run"}:
                raise JobBuilderError(
                    f"Rubric {getattr(rubric, 'rubric_id', 'unknown')} has invalid scope: "
                    f"{getattr(rubric, 'scope', 'missing')}. Must be 'turn' or 'run'"
                )
            
            for judge in judges:
                # Validate and clamp repeats
                repeats = self._validate_repeats(judge)
                
                for repeat_index in range(repeats):
                    # Determine scope and create jobs accordingly
                    if rubric.scope == "turn":
                        # Create one job per turn
                        for turn in turns:
                            turn_id = turn.get("turn_id")
                            if not turn_id:
                                raise JobBuilderError(
                                    f"Turn missing required 'turn_id' field in run {run_id}"
                                )
                            
                            job = self._create_job(
                                run_id=run_id,
                                turn_id=turn_id,
                                rubric=rubric,
                                judge=judge,
                                repeat_index=repeat_index,
                                normalized_run=normalized_run,
                                deterministic_metrics=deterministic_metrics
                            )
                            jobs.append(job)
                    
                    elif rubric.scope == "run":
                        # Create one job for the entire run
                        job = self._create_job(
                            run_id=run_id,
                            turn_id=None,
                            rubric=rubric,
                            judge=judge,
                            repeat_index=repeat_index,
                            normalized_run=normalized_run,
                            deterministic_metrics=deterministic_metrics
                        )
                        jobs.append(job)
        
        # Warn about combinatorial explosion
        if len(jobs) > self.MAX_JOBS_WARNING_THRESHOLD:
            warnings.warn(
                f"Generated {len(jobs)} jobs (threshold: {self.MAX_JOBS_WARNING_THRESHOLD}). "
                f"Consider reducing rubrics, judges, or repeats to avoid resource exhaustion.",
                UserWarning
            )
        
        return jobs
    
    def _validate_turns(self, turns: List[Dict[str, Any]]) -> None:
        """
        Validate turns have unique turn_ids.
        
        Args:
            turns: List of turn dictionaries
            
        Raises:
            JobBuilderError: If duplicate turn_ids found
        """
        turn_ids: Set[str] = set()
        for idx, turn in enumerate(turns):
            turn_id = turn.get("turn_id")
            if not turn_id:
                raise JobBuilderError(
                    f"Turn at index {idx} missing required 'turn_id' field"
                )
            if turn_id in turn_ids:
                raise JobBuilderError(
                    f"Duplicate turn_id '{turn_id}' found in turns"
                )
            turn_ids.add(turn_id)
    
    def _validate_repeats(self, judge: Any) -> int:
        """
        Validate and clamp judge repeats value.
        
        Args:
            judge: Judge object
            
        Returns:
            Validated repeats value (clamped to [1, MAX_REPEATS])
        """
        repeats = getattr(judge, "repeats", 3)
        
        # Validate type
        if not isinstance(repeats, int):
            warnings.warn(
                f"Judge {getattr(judge, 'judge_id', 'unknown')} has non-integer repeats: {repeats}. "
                f"Using default value 3.",
                UserWarning
            )
            return 3
        
        # Clamp to valid range
        if repeats < 1:
            warnings.warn(
                f"Judge {getattr(judge, 'judge_id', 'unknown')} has repeats < 1: {repeats}. "
                f"Clamping to 1.",
                UserWarning
            )
            return 1
        
        if repeats > self.MAX_REPEATS:
            warnings.warn(
                f"Judge {getattr(judge, 'judge_id', 'unknown')} has repeats > {self.MAX_REPEATS}: {repeats}. "
                f"Clamping to {self.MAX_REPEATS}.",
                UserWarning
            )
            return self.MAX_REPEATS
        
        return repeats
    
    def _create_job(
        self,
        run_id: str,
        turn_id: Optional[str],
        rubric: Any,
        judge: Any,
        repeat_index: int,
        normalized_run: Dict[str, Any],
        deterministic_metrics: Optional[Dict[str, Any]]
    ) -> JudgeJob:
        """
        Create a single JudgeJob.
        
        Args:
            run_id: Run identifier
            turn_id: Turn identifier (None for run-level)
            rubric: Rubric object
            judge: Judge object
            repeat_index: Repeat index (0 to repeats-1)
            normalized_run: The NormalizedRun data
            deterministic_metrics: Optional deterministic metrics
            
        Returns:
            JudgeJob instance
            
        Raises:
            JobBuilderError: If job creation fails
        """
        # Validate required attributes
        rubric_id = self._get_attribute(rubric, "rubric_id", "Rubric")
        judge_id = self._get_attribute(judge, "judge_id", "Judge")
        
        # Generate deterministic job_id
        job_id = self._generate_job_id(
            run_id=run_id,
            turn_id=turn_id,
            rubric_id=rubric_id,
            judge_id=judge_id,
            repeat_index=repeat_index
        )
        
        # Build prompt payload with evidence
        try:
            prompt_payload = self.create_prompt_payload(
                normalized_run=normalized_run,
                rubric=rubric,
                turn_id=turn_id,
                deterministic_metrics=deterministic_metrics
            )
        except EvidenceExtractionError as e:
            raise JobBuilderError(
                f"Evidence extraction failed for rubric {rubric_id} "
                f"(scope={getattr(rubric, 'scope', 'unknown')}, turn_id={turn_id}): {str(e)}"
            )
        
        # Enforce payload size limit
        payload_size = len(json.dumps(prompt_payload, ensure_ascii=False))
        if payload_size > self.MAX_PAYLOAD_SIZE:
            raise JobBuilderError(
                f"Prompt payload for rubric {rubric_id} exceeds size limit: "
                f"{payload_size} bytes > {self.MAX_PAYLOAD_SIZE} bytes. "
                f"Reduce evidence budget or simplify rubric."
            )
        
        return JudgeJob(
            job_id=job_id,
            run_id=run_id,
            turn_id=turn_id,
            rubric_id=rubric_id,
            judge_id=judge_id,
            repeat_index=repeat_index,
            prompt_payload=prompt_payload
        )
    
    def _get_attribute(self, obj: Any, attr: str, obj_type: str) -> Any:
        """
        Safely get attribute from object with clear error message.
        
        Args:
            obj: Object to get attribute from
            attr: Attribute name
            obj_type: Object type name for error message
            
        Returns:
            Attribute value
            
        Raises:
            JobBuilderError: If attribute is missing
        """
        if not hasattr(obj, attr):
            raise JobBuilderError(
                f"{obj_type} object missing required attribute '{attr}'"
            )
        value = getattr(obj, attr)
        if value is None:
            raise JobBuilderError(
                f"{obj_type} attribute '{attr}' is None"
            )
        return value
    
    def _generate_job_id(
        self,
        run_id: str,
        turn_id: Optional[str],
        rubric_id: str,
        judge_id: str,
        repeat_index: int
    ) -> str:
        """
        Generate deterministic unique job identifier using stable hash.
        
        Args:
            run_id: Run identifier
            turn_id: Turn identifier (None for run-level)
            rubric_id: Rubric identifier
            judge_id: Judge identifier
            repeat_index: Repeat index
            
        Returns:
            Deterministic job_id string
        """
        # Create stable hash from job parameters
        turn_part = turn_id if turn_id else "run"
        components = f"{run_id}|{turn_part}|{rubric_id}|{judge_id}|{repeat_index}"
        
        # Use SHA-256 for deterministic hash
        hash_digest = hashlib.sha256(components.encode('utf-8')).hexdigest()[:16]
        
        # Create readable job_id with hash suffix
        return f"{run_id}_{turn_part}_{rubric_id}_{judge_id}_r{repeat_index}_{hash_digest}"
    
    def create_prompt_payload(
        self,
        normalized_run: Dict[str, Any],
        rubric: Any,
        turn_id: Optional[str] = None,
        deterministic_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create prompt payload from NormalizedRun and rubric.
        Uses evidence_selectors to extract relevant data.
        
        Args:
            normalized_run: The NormalizedRun data
            rubric: Rubric object with evidence_selectors
            turn_id: Turn identifier (for turn-scoped rubrics)
            deterministic_metrics: Optional deterministic metrics to include
            
        Returns:
            Dictionary with evidence and instructions for judge
            
        Raises:
            JobBuilderError: If payload creation fails
        """
        # Validate required rubric attributes
        rubric_id = self._get_attribute(rubric, "rubric_id", "Rubric")
        description = self._get_attribute(rubric, "description", "Rubric")
        scoring_scale = self._get_attribute(rubric, "scoring_scale", "Rubric")
        evidence_selectors = self._get_attribute(rubric, "evidence_selectors", "Rubric")
        scope = self._get_attribute(rubric, "scope", "Rubric")
        
        try:
            # Extract evidence using rubric's selectors
            evidence = self.evidence_extractor.extract_evidence(
                normalized_run=normalized_run,
                evidence_selectors=evidence_selectors,
                scope=scope,
                turn_id=turn_id,
                redact_fields=getattr(rubric, "redact_fields", None)
            )
            
            # Build payload
            payload = {
                "rubric_id": rubric_id,
                "rubric_description": description,
                "scoring_scale": scoring_scale,
                "evidence": evidence,
                "run_id": normalized_run.get("run_id"),
            }
            
            # Add turn_id if turn-scoped
            if turn_id:
                payload["turn_id"] = turn_id
            
            # Add evaluation instructions if present
            evaluation_instructions = getattr(rubric, "evaluation_instructions", None)
            if evaluation_instructions:
                payload["evaluation_instructions"] = evaluation_instructions
            
            # Add response format schema hint
            payload["response_format"] = {
                "description": "Return JSON with the following fields",
                "required_fields": {
                    "score": "Numeric score or categorical verdict matching scoring_scale",
                    "reasoning": "Detailed explanation of the score",
                },
                "optional_fields": {
                    "evidence_refs": "References to specific evidence used",
                    "confidence": "Confidence level in the assessment (0.0-1.0)"
                }
            }
            
            # Include subset of deterministic metrics if provided
            if deterministic_metrics:
                # Only include metrics relevant to rubrics (avoid bloat)
                relevant_metrics = self._filter_relevant_metrics(
                    deterministic_metrics, rubric_id
                )
                if relevant_metrics:
                    payload["deterministic_metrics"] = relevant_metrics
            
            return payload
            
        except EvidenceExtractionError:
            # Re-raise with context
            raise
        except Exception as e:
            raise JobBuilderError(
                f"Failed to create prompt payload for rubric {rubric_id}: {str(e)}"
            )
    
    def _filter_relevant_metrics(
        self,
        deterministic_metrics: Dict[str, Any],
        rubric_id: str
    ) -> Dict[str, Any]:
        """
        Filter deterministic metrics to include only relevant subset.
        
        Args:
            deterministic_metrics: Full deterministic metrics (dict or MetricsResult object)
            rubric_id: Rubric identifier
            
        Returns:
            Filtered metrics dictionary
        """
        # Convert MetricsResult to dict if needed
        if hasattr(deterministic_metrics, 'to_dict'):
            metrics_dict = deterministic_metrics.to_dict()
        elif isinstance(deterministic_metrics, dict):
            metrics_dict = deterministic_metrics
        else:
            # Fallback: try to convert using asdict
            from dataclasses import asdict
            try:
                metrics_dict = asdict(deterministic_metrics)
            except TypeError:
                # If all else fails, return empty dict
                return {}
        
        # Define which metrics are relevant for which rubrics
        relevant_keys = {
            "STITCHED_TRACE_SUSPECT": ["stitched_trace_suspect", "confidence_penalty_summary"],
            "LATENCY_REGRESSION_FLAG": ["latency_p50", "latency_p95", "missing_timestamp_rate"],
            "TRACE_COMPLETENESS": ["turn_count", "step_count", "orphan_result_count"],
            "TOOL_CALL_QUALITY": ["tool_call_count", "tool_result_count", "tool_success_rate"],
        }
        
        # Get relevant keys for this rubric (default to common metrics)
        keys_to_include = relevant_keys.get(rubric_id, [
            "turn_count", "step_count", "tool_call_count", 
            "tool_success_rate", "stitched_trace_suspect"
        ])
        
        # Filter metrics
        filtered = {
            key: metrics_dict[key]
            for key in keys_to_include
            if key in metrics_dict
        }
        
        return filtered
    
    def filter_llm_rubrics(self, rubrics: List[Any]) -> List[Any]:
        """
        Return only rubrics where requires_llm_judge=true.
        
        Args:
            rubrics: List of Rubric objects
            
        Returns:
            Filtered list of LLM-required rubrics
        """
        llm_rubrics = []
        for rubric in rubrics:
            requires_llm = getattr(rubric, "requires_llm_judge", False)
            if requires_llm:
                llm_rubrics.append(rubric)
        return llm_rubrics
