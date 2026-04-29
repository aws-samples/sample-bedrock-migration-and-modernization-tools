# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Worker Pool for executing JudgeJobs with bounded concurrency.

This module implements the WorkerPool class that executes judge jobs
with rate limiting, retry logic, timeout enforcement, and fault tolerance.
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Set

from agent_eval.evaluators.trace_eval.judging.models import JudgeJob, JobResult
from agent_eval.evaluators.trace_eval.judging.rate_limiter import TokenBucketRateLimiter
from agent_eval.evaluators.trace_eval.judging.retry_policy import RetryPolicy, RetryConfig
from agent_eval.judges.judge_client import JudgeClient
from agent_eval.judges.exceptions import (
    ValidationError,
    APIError,
    TimeoutError as JudgeTimeoutError
)

logger = logging.getLogger(__name__)


class ExecutionResult:
    """
    Summary of worker pool execution.
    
    Attributes:
        total_job_count: Total number of jobs
        successful_job_count: Number of successful jobs
        failed_job_count: Number of failed jobs
        skipped_job_count: Number of skipped jobs (resume mode)
        failure_ratio: Ratio of failed to total jobs
        failed_job_ids: List of failed job IDs
        sum_job_latency_ms: Sum of individual job latencies (not wall time)
        wall_time_seconds: Total wall-clock execution time
    """
    
    def __init__(self):
        self.total_job_count = 0
        self.successful_job_count = 0
        self.failed_job_count = 0
        self.skipped_job_count = 0
        self.failure_ratio = 0.0
        self.failed_job_ids: List[str] = []
        self.sum_job_latency_ms = 0.0
        self.wall_time_seconds = 0.0
    
    def to_dict(self) -> Dict:
        """Serialize for results.json."""
        return {
            "total_job_count": self.total_job_count,
            "successful_job_count": self.successful_job_count,
            "failed_job_count": self.failed_job_count,
            "skipped_job_count": self.skipped_job_count,
            "failure_ratio": self.failure_ratio,
            "failed_job_ids": self.failed_job_ids,
            "sum_job_latency_ms": self.sum_job_latency_ms,
            "wall_time_seconds": self.wall_time_seconds
        }


class WorkerPool:
    """
    Worker pool for executing JudgeJobs with bounded concurrency.
    
    Features:
    - Bounded concurrency (default max_concurrency=10)
    - Per-judge rate limiting via TokenBucketRateLimiter
    - Exponential backoff retry via RetryPolicy
    - Timeout enforcement per job
    - Incremental result persistence to judge_runs.jsonl
    - Resume mode (skip completed jobs)
    - Graceful handling of partial failures
    """
    
    def __init__(
        self,
        judge_clients: Dict[str, JudgeClient],
        max_concurrency: int = 10,
        rate_limiter: Optional[TokenBucketRateLimiter] = None,
        retry_policy: Optional[RetryPolicy] = None,
        default_timeout_seconds: int = 30
    ):
        """
        Initialize worker pool.
        
        Args:
            judge_clients: Map of judge_id to JudgeClient instances
            max_concurrency: Maximum concurrent jobs
            rate_limiter: Rate limiter for per-judge throttling
            retry_policy: Retry policy for failed jobs
            default_timeout_seconds: Default timeout per job
            
        Raises:
            ValueError: If max_concurrency <= 0
        """
        if max_concurrency <= 0:
            raise ValueError(
                f"max_concurrency must be > 0, got {max_concurrency}"
            )
        
        self.judge_clients = judge_clients
        self.max_concurrency = max_concurrency
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter()
        self.retry_policy = retry_policy or RetryPolicy()
        self.default_timeout_seconds = default_timeout_seconds
        
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self._output_lock = asyncio.Lock()
    
    async def execute(
        self,
        jobs: List[JudgeJob],
        output_path: str,
        resume: bool = True
    ) -> ExecutionResult:
        """
        Execute all jobs with bounded concurrency.
        
        Args:
            jobs: List of JudgeJob to execute
            output_path: Path to judge_runs.jsonl for incremental writes
            resume: Enable resume mode (skip completed jobs)
            
        Returns:
            ExecutionResult with execution statistics
        """
        result = ExecutionResult()
        result.total_job_count = len(jobs)
        
        # Load completed jobs if resume mode enabled
        completed_job_ids = set()
        skipped_ids = set()
        
        if resume:
            completed_job_ids = self._load_completed_jobs(output_path)
            
            # Calculate actual skipped count (intersection with current jobs)
            current_job_ids = {job.job_id for job in jobs}
            skipped_ids = current_job_ids & completed_job_ids
            result.skipped_job_count = len(skipped_ids)
            
            if skipped_ids:
                logger.info(
                    f"Resume mode: Found {len(skipped_ids)} completed jobs, "
                    f"{len(jobs) - len(skipped_ids)} remaining"
                )
        
        # Filter out completed jobs using the intersection set
        jobs_to_execute = [
            job for job in jobs
            if job.job_id not in skipped_ids
        ]
        
        if not jobs_to_execute:
            logger.info("All jobs already completed")
            result.successful_job_count = result.skipped_job_count
            return result
        
        # Execute jobs concurrently with wall-time tracking
        logger.info(
            f"Executing {len(jobs_to_execute)} jobs with "
            f"max_concurrency={self.max_concurrency}"
        )
        
        start_time = time.time()
        
        # Wrap each job execution to always return JobResult
        tasks = [
            self._execute_job_safe(job, output_path)
            for job in jobs_to_execute
        ]
        
        job_results = await asyncio.gather(*tasks)
        
        result.wall_time_seconds = time.time() - start_time
        
        # Aggregate results
        for job_result in job_results:
            if job_result.status == "success":
                result.successful_job_count += 1
            else:
                result.failed_job_count += 1
                result.failed_job_ids.append(job_result.job_id)
            
            if job_result.latency_ms:
                result.sum_job_latency_ms += job_result.latency_ms
        
        # Add skipped jobs to successful count
        result.successful_job_count += result.skipped_job_count
        
        # Calculate failure ratio
        if result.total_job_count > 0:
            result.failure_ratio = (
                result.failed_job_count / result.total_job_count
            )
        
        logger.info(
            f"Execution complete: {result.successful_job_count} succeeded, "
            f"{result.failed_job_count} failed, "
            f"{result.skipped_job_count} skipped "
            f"(wall time: {result.wall_time_seconds:.2f}s)"
        )
        
        return result
    
    def run(
        self,
        jobs: List[JudgeJob],
        output_path: str,
        resume: bool = True
    ) -> ExecutionResult:
        """
        Synchronous wrapper for execute() - runs async execution in event loop.
        
        This is the main entry point for synchronous callers (like TraceEvaluator).
        
        Args:
            jobs: List of JudgeJob to execute
            output_path: Path to judge_runs.jsonl for incremental writes
            resume: Enable resume mode (skip completed jobs)
            
        Returns:
            ExecutionResult with execution statistics
        """
        return asyncio.run(self.execute(jobs, output_path, resume))
    
    async def _execute_job_safe(
        self,
        job: JudgeJob,
        output_path: str
    ) -> JobResult:
        """
        Safely execute job with semaphore, always returning a JobResult.
        
        This is the only place that writes results to ensure exactly-once semantics.
        
        Args:
            job: JudgeJob to execute
            output_path: Path to judge_runs.jsonl
            
        Returns:
            JobResult (never raises)
        """
        try:
            async with self.semaphore:
                # execute_job never raises, always returns JobResult
                result = await self.execute_job(job, output_path)
            
            # Write result exactly once here
            await self.append_result(result, output_path)
            return result
            
        except Exception as e:
            # Catch any unexpected exceptions (e.g., from append_result)
            logger.error(
                f"Unexpected error in job execution wrapper for {job.job_id}: {e}",
                exc_info=True
            )
            result = JobResult.create_failure(
                job,
                error=f"Unexpected wrapper error: {str(e)}",
                status="failure"
            )
            # Try to write, but don't fail if this also errors
            try:
                await self.append_result(result, output_path)
            except Exception as write_error:
                logger.error(
                    f"Failed to write error result for {job.job_id}: {write_error}"
                )
            return result
    
    async def execute_job(
        self,
        job: JudgeJob,
        output_path: str
    ) -> JobResult:
        """
        Execute single job with retry logic, rate limiting, and timeout.
        
        Always returns JobResult, never raises. Result writing is handled
        by the caller (_execute_job_safe).
        
        Args:
            job: JudgeJob to execute
            output_path: Path to judge_runs.jsonl (for context only)
            
        Returns:
            JobResult with success or failure status
        """
        # Get judge client
        judge_client = self.judge_clients.get(job.judge_id)
        if not judge_client:
            error_msg = f"No judge client found for judge_id: {job.judge_id}"
            logger.error(error_msg)
            return JobResult.create_failure(
                job,
                error=error_msg,
                status="failure"
            )
        
        # Get timeout (from client or default)
        timeout_seconds = getattr(
            judge_client,
            'timeout_seconds',
            self.default_timeout_seconds
        )
        
        # Execute with retry policy, tracking attempt count
        attempt_count = 0
        
        async def execute_with_rate_limit_and_timeout():
            nonlocal attempt_count
            attempt_count += 1
            
            # Apply rate limiting per attempt
            await self.rate_limiter.acquire(job.judge_id)
            
            # Execute with timeout
            return await asyncio.wait_for(
                self._execute_single_attempt(job, judge_client),
                timeout=timeout_seconds
            )
        
        try:
            # Don't pass context as kwarg - retry_policy will use it internally
            result = await self.retry_policy.execute_with_retry(
                execute_with_rate_limit_and_timeout
            )
            
            # Update retry count in result
            if result.status == "success":
                result.retry_count = attempt_count - 1
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(
                f"Job {job.job_id} timed out after {timeout_seconds}s "
                f"(attempt {attempt_count})"
            )
            return JobResult.create_failure(
                job,
                error=f"Timeout after {timeout_seconds}s",
                status="timeout",
                retry_count=attempt_count - 1
            )
        
        except JudgeTimeoutError as e:
            # Handle our custom timeout error
            logger.error(
                f"Job {job.job_id} timed out: {e.message} "
                f"(attempt {attempt_count})"
            )
            return JobResult.create_failure(
                job,
                error=e.message,
                status="timeout",
                retry_count=attempt_count - 1
            )
        
        except ValidationError as e:
            # Validation errors are logged but may be retried
            logger.error(
                f"Job {job.job_id} validation failed: {e.message} "
                f"(code: {e.error_code}, attempt {attempt_count})"
            )
            return JobResult.create_failure(
                job,
                error=f"{e.error_code}: {e.message}",
                status="invalid_response",
                retry_count=attempt_count - 1
            )
        
        except APIError as e:
            # API errors are logged with status code if available
            logger.error(
                f"Job {job.job_id} API error: {e.message} "
                f"(code: {e.error_code}, attempt {attempt_count})"
            )
            return JobResult.create_failure(
                job,
                error=f"{e.error_code}: {e.message}",
                status="failure",
                retry_count=attempt_count - 1
            )
            
        except Exception as e:
            logger.error(
                f"Job {job.job_id} failed after all retries: {e}"
            )
            return JobResult.create_failure(
                job,
                error=str(e),
                status="failure",
                retry_count=attempt_count - 1
            )
    
    async def _execute_single_attempt(
        self,
        job: JudgeJob,
        judge_client: JudgeClient
    ) -> JobResult:
        """
        Execute single attempt of a job.
        
        Args:
            job: JudgeJob to execute
            judge_client: JudgeClient instance
            
        Returns:
            JobResult
            
        Raises:
            Exception: On failure (for retry logic)
        """
        try:
            # Build prompt from payload with traceability context
            prompt = judge_client.build_prompt(
                evidence=job.prompt_payload.get('evidence', {}),
                rubric_description=job.prompt_payload.get('rubric_description', ''),
                scoring_scale=job.prompt_payload.get('scoring_scale', {}),
                rubric_id=job.rubric_id,
                turn_id=job.turn_id,
                run_id=job.run_id
            )
            
            # Execute judge
            judge_response = await judge_client.execute_judge(
                prompt=prompt,
                rubric_id=job.rubric_id,
                scoring_scale=job.prompt_payload.get('scoring_scale', {})
            )
            
            # Validate response before creating success result
            # This ensures consistent validation across all providers (Amazon Bedrock, Mock, etc.)
            await judge_client.validate_response(
                response=judge_response.raw_response,
                scoring_scale=job.prompt_payload.get('scoring_scale', {})
            )
            
            # Create success result
            result = JobResult.create_success(
                job=job,
                raw_response=judge_response.raw_response,
                parsed_response={
                    'score': judge_response.score,
                    'reasoning': judge_response.reasoning
                },
                latency_ms=judge_response.latency_ms,
                retry_count=0  # Updated by retry policy if needed
            )
            
            return result
            
        except asyncio.TimeoutError as e:
            logger.warning(f"Job {job.job_id} timed out: {e}")
            raise
        except Exception as e:
            logger.warning(f"Job {job.job_id} attempt failed: {e}")
            raise
    
    async def append_result(
        self,
        result: JobResult,
        output_path: str
    ) -> None:
        """
        Append result to judge_runs.jsonl incrementally using async I/O.
        
        Thread-safe with async lock to prevent concurrent writes.
        
        Args:
            result: JobResult to append
            output_path: Path to judge_runs.jsonl
        """
        async with self._output_lock:
            try:
                # Ensure output directory exists
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                # Use asyncio.to_thread for non-blocking file I/O
                await asyncio.to_thread(
                    self._write_result_sync,
                    result,
                    output_path
                )
                
            except Exception as e:
                logger.error(f"Failed to append result to {output_path}: {e}")
                raise
    
    @staticmethod
    def _write_result_sync(result: JobResult, output_path: str) -> None:
        """Synchronous file write helper for asyncio.to_thread."""
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(result.to_jsonl_line())
    
    def _load_completed_jobs(self, output_path: str) -> Set[str]:
        """
        Load completed job IDs from existing judge_runs.jsonl.
        
        Handles partial lines at end of file (from crashes mid-write).
        
        Args:
            output_path: Path to judge_runs.jsonl
            
        Returns:
            Set of completed job_ids
        """
        completed_ids = set()
        
        if not os.path.exists(output_path):
            return completed_ids
        
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Process all complete lines
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    job_id = record.get('job_id')
                    if job_id:
                        completed_ids.add(job_id)
                except json.JSONDecodeError as e:
                    # Check if this is the last line (might be partial)
                    if line_num == len(lines):
                        logger.warning(
                            f"Partial/corrupt line at end of {output_path} "
                            f"(line {line_num}): {e}. Skipping."
                        )
                    else:
                        logger.warning(
                            f"Invalid JSON on line {line_num} in {output_path}: {e}"
                        )
                    continue
            
            logger.info(f"Loaded {len(completed_ids)} completed jobs from {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to load completed jobs from {output_path}: {e}")
        
        return completed_ids
