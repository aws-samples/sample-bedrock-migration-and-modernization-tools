# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Data Models for Judge Job Execution

This module defines the core data structures for judge job execution:
- JudgeJob: A single unit of work for evaluation
- JobResult: The result of executing a judge job
"""

import json
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timezone


@dataclass
class JudgeJob:
    """
    Represents a single judge evaluation job.
    
    A JudgeJob is created for each (rubric × judge × repeat) combination.
    """
    job_id: str
    run_id: str
    turn_id: Optional[str]  # None for run-level rubrics
    rubric_id: str
    judge_id: str
    repeat_index: int  # 0 to (repeats-1)
    prompt_payload: Dict[str, Any]  # Evidence + instructions for judge
    schema_version: str = "1.0.0"  # For future migrations
    created_at: Optional[str] = None  # ISO 8601 timestamp
    
    def __post_init__(self):
        """Set created_at if not provided."""
        if self.created_at is None:
            self.created_at = self._generate_timestamp()
        
        # Validate payload size
        payload_size = len(json.dumps(self.prompt_payload, ensure_ascii=False))
        if payload_size > 50000:  # 50KB warning threshold
            warnings.warn(
                f"JudgeJob {self.job_id} has large prompt_payload ({payload_size} bytes). "
                f"Consider reducing evidence budget.",
                UserWarning
            )
    
    @staticmethod
    def _generate_timestamp() -> str:
        """Generate ISO 8601 timestamp with millisecond precision."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize JudgeJob to dictionary for logging/debugging.
        
        Returns:
            Dictionary representation of the job
        """
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "rubric_id": self.rubric_id,
            "judge_id": self.judge_id,
            "repeat_index": self.repeat_index,
            "prompt_payload": self.prompt_payload,
            "schema_version": self.schema_version,
            "created_at": self.created_at
        }
    
    def __repr__(self) -> str:
        return (
            f"JudgeJob(job_id={self.job_id}, run_id={self.run_id}, turn_id={self.turn_id}, "
            f"rubric_id={self.rubric_id}, judge_id={self.judge_id}, repeat_index={self.repeat_index})"
        )


@dataclass
class JobResult:
    """
    Represents the result of executing a judge job.
    
    Matches judge_run_record.schema.json format for JSONL serialization.
    """
    job_id: str
    run_id: str
    turn_id: Optional[str]
    rubric_id: str
    judge_id: str
    repeat_index: int
    timestamp: str  # ISO 8601 format with millisecond precision
    status: str  # "success", "failure", "timeout", "invalid_response", "cancelled", "skipped"
    raw_response: Optional[Union[str, Dict[str, Any], List[Any]]] = None
    parsed_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    retry_count: int = 0
    
    # Valid status values
    VALID_STATUSES = {
        "success", "failure", "timeout", "invalid_response", 
        "cancelled", "skipped"
    }
    
    def __post_init__(self):
        """Validate status value and response size."""
        if self.status not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}'. Must be one of {self.VALID_STATUSES}"
            )
        
        # Validate and normalize timestamp format
        self.timestamp = self._normalize_timestamp(self.timestamp)
        
        # Warn about large responses
        if self.raw_response is not None:
            response_size = len(json.dumps(self.raw_response, ensure_ascii=False))
            if response_size > 100000:  # 100KB warning threshold
                warnings.warn(
                    f"JobResult {self.job_id} has large raw_response ({response_size} bytes). "
                    f"Consider storing large responses separately.",
                    UserWarning
                )
    
    @staticmethod
    def _normalize_timestamp(timestamp: str) -> str:
        """
        Normalize timestamp to ISO 8601 with millisecond precision.
        
        Args:
            timestamp: ISO 8601 timestamp string
            
        Returns:
            Normalized timestamp with millisecond precision
        """
        try:
            # Parse and reformat to helps consistent precision
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        except (ValueError, AttributeError):
            # If parsing fails, return as-is (validation happens elsewhere)
            return timestamp
    
    @property
    def score(self) -> Optional[Union[float, str]]:
        """
        Extract score from parsed_response.
        
        Supports multiple response formats:
        - {"score": value}
        - {"verdict": value}
        - {"grade": value}
        
        Returns:
            Numeric score or categorical verdict, or None if not available
        """
        if not self.parsed_response:
            return None
        
        # Try common score field names
        for key in ["score", "verdict", "grade", "rating"]:
            if key in self.parsed_response:
                return self.parsed_response[key]
        
        return None
    
    @property
    def reasoning(self) -> Optional[str]:
        """
        Extract reasoning from parsed_response.
        
        Supports multiple response formats:
        - {"reasoning": value}
        - {"rationale": value}
        - {"explanation": value}
        
        Returns:
            Judge's explanation or None if not available
        """
        if not self.parsed_response:
            return None
        
        # Try common reasoning field names
        for key in ["reasoning", "rationale", "explanation", "justification"]:
            if key in self.parsed_response:
                return self.parsed_response[key]
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert JobResult to dictionary.
        
        Returns:
            Dictionary representation matching judge_run_record.schema.json
        """
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "rubric_id": self.rubric_id,
            "judge_id": self.judge_id,
            "repeat_index": self.repeat_index,
            "timestamp": self.timestamp,
            "raw_response": self.raw_response,
            "parsed_response": self.parsed_response,
            "status": self.status,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "retry_count": self.retry_count
        }
    
    def to_jsonl_line(self) -> str:
        """
        Serialize JobResult as JSONL line for judge_runs.jsonl.
        
        Returns:
            JSON string with newline
        """
        return json.dumps(self.to_dict(), ensure_ascii=False) + "\n"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobResult":
        """
        Create JobResult from dictionary with validation.
        
        Args:
            data: Dictionary matching judge_run_record.schema.json
            
        Returns:
            JobResult instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields
        required_fields = [
            "job_id", "run_id", "rubric_id", "judge_id", 
            "repeat_index", "timestamp", "status"
        ]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in JobResult data: {missing_fields}"
            )
        
        try:
            return cls(
                job_id=data["job_id"],
                run_id=data["run_id"],
                turn_id=data.get("turn_id"),
                rubric_id=data["rubric_id"],
                judge_id=data["judge_id"],
                repeat_index=data["repeat_index"],
                timestamp=data["timestamp"],
                status=data["status"],
                raw_response=data.get("raw_response"),
                parsed_response=data.get("parsed_response"),
                error=data.get("error"),
                latency_ms=data.get("latency_ms"),
                retry_count=data.get("retry_count", 0)
            )
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid JobResult data structure: {str(e)}")
    
    @staticmethod
    def _generate_timestamp() -> str:
        """Generate ISO 8601 timestamp with millisecond precision."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    
    @staticmethod
    def create_success(
        job: JudgeJob,
        raw_response: Union[str, Dict[str, Any], List[Any]],
        parsed_response: Dict[str, Any],
        latency_ms: float,
        retry_count: int = 0
    ) -> "JobResult":
        """
        Create a successful JobResult.
        
        Args:
            job: The JudgeJob that was executed
            raw_response: Raw response from judge LLM
            parsed_response: Parsed response matching judge_response.schema.json
            latency_ms: Execution time in milliseconds
            retry_count: Number of retries attempted
            
        Returns:
            JobResult with status="success"
        """
        return JobResult(
            job_id=job.job_id,
            run_id=job.run_id,
            turn_id=job.turn_id,
            rubric_id=job.rubric_id,
            judge_id=job.judge_id,
            repeat_index=job.repeat_index,
            timestamp=JobResult._generate_timestamp(),
            status="success",
            raw_response=raw_response,
            parsed_response=parsed_response,
            latency_ms=latency_ms,
            retry_count=retry_count
        )
    
    @staticmethod
    def create_failure(
        job: JudgeJob,
        error: str,
        status: str = "failure",
        latency_ms: Optional[float] = None,
        retry_count: int = 0,
        raw_response: Optional[Union[str, Dict[str, Any], List[Any]]] = None
    ) -> "JobResult":
        """
        Create a failed JobResult.
        
        Args:
            job: The JudgeJob that was executed
            error: Error message
            status: Failure status (must be one of: "failure", "timeout", "invalid_response", "cancelled", "skipped")
            latency_ms: Execution time in milliseconds
            retry_count: Number of retries attempted
            raw_response: Raw response if available
            
        Returns:
            JobResult with failure status
            
        Raises:
            ValueError: If status is not a valid failure status
        """
        # Validate status is a failure status
        valid_failure_statuses = {"failure", "timeout", "invalid_response", "cancelled", "skipped"}
        if status not in valid_failure_statuses:
            raise ValueError(
                f"Invalid failure status '{status}'. Must be one of {valid_failure_statuses}"
            )
        
        return JobResult(
            job_id=job.job_id,
            run_id=job.run_id,
            turn_id=job.turn_id,
            rubric_id=job.rubric_id,
            judge_id=job.judge_id,
            repeat_index=job.repeat_index,
            timestamp=JobResult._generate_timestamp(),
            status=status,
            error=error,
            latency_ms=latency_ms,
            retry_count=retry_count,
            raw_response=raw_response
        )
    
    def __repr__(self) -> str:
        return (
            f"JobResult(job_id={self.job_id}, run_id={self.run_id}, turn_id={self.turn_id}, "
            f"status={self.status}, rubric_id={self.rubric_id}, judge_id={self.judge_id})"
        )
