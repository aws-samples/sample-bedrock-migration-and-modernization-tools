# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Deterministic metrics computation for NormalizedRun traces.

This module computes metrics directly from NormalizedRun data without requiring
LLM inference. Metrics include counts, rates, flags, and latency percentiles.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics

from .timestamp_policy import (
    validate_turn_timestamps,
    should_compute_latency_percentiles
)


@dataclass
class MetricsResult:
    """Container for computed deterministic metrics."""
    
    # Basic counts
    turn_count: int
    step_count: int
    tool_call_count: int
    tool_result_count: int
    orphan_result_count: int
    
    # Rates and ratios
    tool_success_rate: float
    missing_timestamp_rate: float
    
    # Flags
    stitched_trace_suspect: bool
    single_turn_fallback_used: bool
    
    # Latency metrics (None if insufficient trusted timestamps)
    latency_p50: Optional[float]
    latency_p95: Optional[float]
    avg_turn_latency_ms: Optional[float]
    
    # Confidence penalty summary
    confidence_penalty_summary: Dict[str, int]
    
    # Additional metadata
    trusted_timestamp_count: int
    untrusted_timestamp_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return asdict(self)


class DeterministicMetrics:
    """Computes deterministic metrics from NormalizedRun data."""
    
    # Tool call kind (normalized by adapter)
    TOOL_CALL_KIND = "TOOL_CALL"
    
    # Tool result kind (normalized by adapter)
    TOOL_RESULT_KIND = "TOOL_RESULT"
    
    # Success status (normalized by adapter)
    SUCCESS_STATUS = "success"
    
    def __init__(self):
        """Initialize metrics computer."""
        pass
    
    def compute(self, normalized_run: Dict[str, Any]) -> MetricsResult:
        """
        Compute all deterministic metrics from NormalizedRun.
        
        Args:
            normalized_run: Validated NormalizedRun dictionary
            
        Returns:
            MetricsResult with all computed metrics
        """
        turns = normalized_run.get("turns", [])
        adapter_stats = normalized_run.get("adapter_stats", {})
        metadata = normalized_run.get("metadata", {})
        
        # Basic counts
        turn_count = len(turns)
        step_count = self._count_total_steps(turns)
        tool_call_count = self._count_tool_calls(turns)
        tool_result_count = self._count_tool_results(turns)
        orphan_result_count = len(adapter_stats.get("orphan_tool_results", []))
        
        # Rates
        tool_success_rate = self._compute_tool_success_rate(turns)
        missing_timestamp_rate = self._compute_missing_timestamp_rate(turns)
        
        # Flags
        stitched_trace_suspect = self._is_stitched_trace_suspect(
            adapter_stats, metadata
        )
        single_turn_fallback_used = self._is_single_turn_fallback_used(
            adapter_stats, metadata
        )
        
        # Latency metrics
        latency_p50, latency_p95, trusted_count, untrusted_count = (
            self._compute_latency_percentiles(turns)
        )
        avg_turn_latency_ms = self._compute_avg_turn_latency(turns)
        
        # Confidence penalty summary
        confidence_penalty_summary = self._compute_confidence_penalty_summary(
            adapter_stats
        )
        
        return MetricsResult(
            turn_count=turn_count,
            step_count=step_count,
            tool_call_count=tool_call_count,
            tool_result_count=tool_result_count,
            orphan_result_count=orphan_result_count,
            tool_success_rate=tool_success_rate,
            missing_timestamp_rate=missing_timestamp_rate,
            stitched_trace_suspect=stitched_trace_suspect,
            single_turn_fallback_used=single_turn_fallback_used,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            avg_turn_latency_ms=avg_turn_latency_ms,
            confidence_penalty_summary=confidence_penalty_summary,
            trusted_timestamp_count=trusted_count,
            untrusted_timestamp_count=untrusted_count
        )
    
    def _count_total_steps(self, turns: List[Dict[str, Any]]) -> int:
        """Count total steps across all turns."""
        total = 0
        for turn in turns:
            steps = turn.get("steps", [])
            total += len(steps)
        return total
    
    def _count_tool_calls(self, turns: List[Dict[str, Any]]) -> int:
        """Count tool call steps across all turns."""
        count = 0
        for turn in turns:
            steps = turn.get("steps", [])
            for step in steps:
                # Use only normalized 'kind' field (adapter guarantees this)
                kind = (step.get("kind") or "").upper()
                if kind == self.TOOL_CALL_KIND:
                    count += 1
        return count
    
    def _count_tool_results(self, turns: List[Dict[str, Any]]) -> int:
        """Count tool result steps across all turns."""
        count = 0
        for turn in turns:
            steps = turn.get("steps", [])
            for step in steps:
                # Use only normalized 'kind' field (adapter guarantees this)
                kind = (step.get("kind") or "").upper()
                if kind == self.TOOL_RESULT_KIND:
                    count += 1
        return count
    
    def _compute_tool_success_rate(self, turns: List[Dict[str, Any]]) -> float:
        """
        Calculate ratio of successful tool calls.
        
        Returns:
            Success rate between 0.0 and 1.0, or 0.0 if no tool calls
        """
        total_tool_calls = 0
        successful_tool_calls = 0
        
        for turn in turns:
            steps = turn.get("steps", [])
            for step in steps:
                # Use only normalized 'kind' field
                kind = (step.get("kind") or "").upper()
                
                # Check if this is a tool call
                if kind == self.TOOL_CALL_KIND:
                    total_tool_calls += 1
                    
                    # Check if successful (normalize status defensively)
                    status = (step.get("status") or "").lower()
                    if status == self.SUCCESS_STATUS:
                        successful_tool_calls += 1
        
        if total_tool_calls == 0:
            return 0.0
        
        return successful_tool_calls / total_tool_calls
    
    def _compute_missing_timestamp_rate(self, turns: List[Dict[str, Any]]) -> float:
        """
        Calculate rate of steps with missing start timestamps.
        
        Note: end_ts is often optional in adapter output, so we only check start_ts.
        Empty strings are treated as missing.
        
        Returns:
            Missing rate between 0.0 and 1.0, or 0.0 if no steps
        """
        total_steps = 0
        missing_timestamp_steps = 0
        
        for turn in turns:
            steps = turn.get("steps", [])
            for step in steps:
                total_steps += 1
                
                start_ts = step.get("start_ts")
                
                # Count as missing if start timestamp is absent or empty string
                if not start_ts:
                    missing_timestamp_steps += 1
        
        if total_steps == 0:
            return 0.0
        
        return missing_timestamp_steps / total_steps
    
    def _is_stitched_trace_suspect(
        self,
        adapter_stats: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Determine if trace is suspect due to stitching issues.
        
        A trace is suspect if:
        1. run_confidence < 0.7, OR
        2. confidence_penalties contain stitching-related penalties
        
        Stitching issues indicate structural problems with trace reconstruction,
        not just low-quality data.
        
        Args:
            adapter_stats: Adapter statistics
            metadata: Run metadata
            
        Returns:
            True if trace is suspect
        """
        # Check run confidence (defensive type handling)
        run_confidence = metadata.get("run_confidence")
        if run_confidence is not None:
            try:
                rc = float(run_confidence)
                if rc < 0.7:
                    return True
            except (TypeError, ValueError):
                # Invalid confidence value - treat as suspect
                return True
        
        # Check for stitching-related confidence penalties
        # These indicate structural issues with trace reconstruction
        confidence_penalties = adapter_stats.get("confidence_penalties", [])
        
        # Stitching-related penalty reasons (match adapter output)
        # Excludes "no_anchor_found" which can occur in normal single-turn traces
        stitching_reasons = {
            "single_turn_fallback",      # Fallback segmentation used
            "missing_grouping_ids",       # No session/request IDs
            "orphan_tool_results"         # Tool results without calls (plural)
        }
        
        for penalty in confidence_penalties:
            reason = penalty.get("reason", "")
            
            # Flag as suspect if stitching-related reason found
            if reason in stitching_reasons:
                return True
        
        return False
    
    def _is_single_turn_fallback_used(
        self,
        adapter_stats: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Determine if single-turn fallback segmentation was used.
        
        Checks both adapter_stats (primary per schema) and metadata (fallback)
        for maximum compatibility.
        
        Args:
            adapter_stats: Adapter statistics
            metadata: Run metadata
            
        Returns:
            True if SINGLE_TURN strategy was used
        """
        # Primary location per schema (required field)
        segmentation_strategy = adapter_stats.get("segmentation_strategy", "")
        
        # Fallback to metadata if not in adapter_stats (defensive)
        if not segmentation_strategy:
            segmentation_strategy = metadata.get("segmentation_strategy_used", "")
        
        # Normalize to uppercase for comparison
        return segmentation_strategy.upper() == "SINGLE_TURN"
    
    def _compute_latency_percentiles(
        self,
        turns: List[Dict[str, Any]]
    ) -> Tuple[Optional[float], Optional[float], int, int]:
        """
        Compute latency p50 and p95 if sufficient trusted timestamps available.
        
        Percentiles are only computed when >= 50% of turns have trusted timestamps.
        
        Args:
            turns: List of turn dictionaries
            
        Returns:
            Tuple of (p50, p95, trusted_count, untrusted_count)
            p50 and p95 are None if insufficient trusted timestamps
        """
        if not turns:
            return None, None, 0, 0
        
        trusted_latencies = []
        trusted_count = 0
        untrusted_count = 0
        
        for turn in turns:
            validation_result = validate_turn_timestamps(turn)
            
            if validation_result.is_trusted and validation_result.duration_ms is not None:
                trusted_latencies.append(validation_result.duration_ms)
                trusted_count += 1
            else:
                untrusted_count += 1
        
        # Check if we have sufficient trusted timestamps
        total_turns = len(turns)
        if not should_compute_latency_percentiles(trusted_count, total_turns):
            return None, None, trusted_count, untrusted_count
        
        # Compute percentiles
        if not trusted_latencies:
            return None, None, trusted_count, untrusted_count
        
        # Sort for percentile calculation
        sorted_latencies = sorted(trusted_latencies)
        
        # Calculate p50 (median) - use statistics.median for consistency
        p50 = statistics.median(sorted_latencies)
        
        # Calculate p95 using nearest-rank method (deterministic)
        p95 = self._nearest_rank_percentile(sorted_latencies, 0.95)
        
        return p50, p95, trusted_count, untrusted_count
    
    def _nearest_rank_percentile(
        self,
        sorted_values: List[float],
        percentile: float
    ) -> float:
        """
        Calculate percentile using nearest-rank method.
        
        This is a deterministic, simple percentile calculation that works
        consistently across Python versions.
        
        Args:
            sorted_values: List of values in ascending order
            percentile: Percentile to calculate (0.0 to 1.0)
            
        Returns:
            Percentile value
        """
        n = len(sorted_values)
        if n == 0:
            return 0.0
        
        # Clamp percentile to valid range [0.0, 1.0]
        percentile = max(0.0, min(1.0, percentile))
        
        # Nearest-rank: ceil(percentile * n)
        # Use manual ceiling to avoid float precision issues
        rank = int(percentile * n + 0.999999999)
        
        # Clamp to valid index range [1, n]
        rank = max(1, min(n, rank))
        
        # Convert to 0-based index
        index = rank - 1
        
        return sorted_values[index]
    
    def _compute_avg_turn_latency(self, turns: List[Dict[str, Any]]) -> Optional[float]:
        """
        Compute average turn latency from total_latency_ms field.
        
        Args:
            turns: List of turn dictionaries
            
        Returns:
            Average latency in milliseconds, or None if no valid latencies
        """
        if not turns:
            return None
        
        valid_latencies = []
        for turn in turns:
            latency = turn.get("total_latency_ms")
            if latency is not None and isinstance(latency, (int, float)) and latency >= 0:
                valid_latencies.append(float(latency))
        
        if not valid_latencies:
            return None
        
        return sum(valid_latencies) / len(valid_latencies)
    
    def _compute_confidence_penalty_summary(
        self,
        adapter_stats: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Aggregate confidence penalties by reason.
        
        Args:
            adapter_stats: Adapter statistics
            
        Returns:
            Dictionary mapping penalty reason to count
        """
        confidence_penalties = adapter_stats.get("confidence_penalties", [])
        
        summary: Dict[str, int] = {}
        for penalty in confidence_penalties:
            reason = penalty.get("reason", "unknown")
            summary[reason] = summary.get(reason, 0) + 1
        
        return summary
