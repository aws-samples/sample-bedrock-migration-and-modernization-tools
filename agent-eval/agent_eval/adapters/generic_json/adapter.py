# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Generic JSON adapter implementation for trace normalization.

This module implements the core adapter logic for transforming Generic JSON
trace files or in-memory trace dicts into the normalized schema format. It uses a multi-stage pipeline:

Stage A - Normalize: Event discovery, field mapping, timestamp parsing, classification
Stage B - Segment: Turn segmentation for multi-turn conversations
Stage C - Derive: Derived field calculation (latency, tool linking, phases)
Stage D - Validate: Schema validation and adapter_stats generation

Note: Classification happens during normalization (Stage A), not as a separate stage.

The adapter uses config-driven field mapping from adapter_config.yaml and
implements graceful degradation with confidence scoring for missing data.

Public API:
    adapt(path, config_path=None) -> Dict[str, Any]
        Main entry point for trace normalization. Accepts a file path or in-memory
        trace dict and transforms it into the normalized schema format.

Constants:
    DEFAULT_CONFIG_PATH: Path
        Default path to adapter_config.yaml, resolved relative to this module.
        This helps the config loads correctly regardless of current working directory.
        
    ADAPTER_VERSION: str
        Version identifier for this adapter implementation (e.g., "1.0.0").
        Included in metadata.adapter_version field of normalized output.

Exception Taxonomy:
    InputError: Raised for file I/O issues or invalid input types
        - File not found
        - Invalid JSON syntax
        - Unreadable input files
        - Invalid input type (not file path or dict)
        
    ValidationError: Raised for schema/data validation issues
        - No events found in trace
        - Schema file cannot be loaded
        - Output fails schema validation
        
    AdaptationError: Raised for internal adapter logic errors
        - Unexpected errors during normalization
        - Configuration errors
        - Internal invariant violations

Example Usage:
    >>> from agent_eval.adapters.generic_json import adapt
    >>> 
    >>> # Basic usage with file path
    >>> result = adapt("trace.json")
    >>> print(f"Run ID: {result['run_id']}")
    >>> print(f"Turns: {len(result['turns'])}")
    >>> print(f"Confidence: {result['metadata']['run_confidence']:.2f}")
    >>> 
    >>> # With in-memory dict (for testing/programmatic use)
    >>> trace_dict = {"events": [...]}
    >>> result = adapt(trace_dict)
    >>> 
    >>> # With custom config
    >>> result = adapt("trace.json", config_path="custom_config.yaml")
    >>> 
    >>> # Access adapter stats for debugging
    >>> stats = result['adapter_stats']
    >>> print(f"Events processed: {stats['total_events_processed']}")
    >>> print(f"Penalties: {len(stats['confidence_penalties'])}")

For detailed configuration options, see adapter_config.yaml documentation.
"""

import json
import hashlib
import re
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import os

from .config_loader import AdapterConfig
from .exceptions import InputError, ValidationError, AdaptationError
from ...utils.validation import (
    parse_timestamp,
    sanitize_latency,
    calculate_confidence_score,
    load_schema,
    validate_against_schema
)

# Default configuration path
from . import DEFAULT_CONFIG_PATH

# Adapter version
ADAPTER_VERSION = "1.0.0"

# Configure logger
logger = logging.getLogger(__name__)


def adapt(path: Union[str, os.PathLike, Dict[str, Any]], config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a Generic JSON trace file or dict and normalize it to the standard schema.
    
    This is the single public entry point for trace normalization.
    
    Args:
        path: Path to the Generic JSON trace file OR a dict containing trace data
        config_path: Optional path to adapter_config.yaml (defaults to DEFAULT_CONFIG_PATH)
        
    Returns:
        Dictionary conforming to normalized schema with:
        - run_id, metadata, adapter_stats, turns[]
        - Each turn has confidence score (0-1)
        - adapter_stats contains confidence_penalties
        
    Raises:
        InputError: If the trace file doesn't exist, contains invalid JSON, is unreadable, or invalid input type
        ValidationError: If no events exist, schema file can't be loaded, or output validation fails
        AdaptationError: If internal adapter logic encounters an unexpected error
        
    Behavior:
        - Graceful degradation: missing fields → null with confidence penalty
        - Config-driven field mapping from adapter_config.yaml
        - Dual latency tracking (normalized_latency_ms and runtime_reported_latency_ms)
        - Multi-turn conversation support with turn stitching
        - Orphan tool results handled with confidence penalties
        - Tool-looking text without markers not misclassified
    
    Example:
        >>> from agent_eval.adapters.generic_json import adapt
        >>> # From file
        >>> result = adapt("trace.json")
        >>> print(result["run_id"])
        'abc-123-def'
        >>> # From dict (for testing/programmatic use)
        >>> result = adapt({"events": [...]})
        >>> print(result["turns"][0]["confidence"])
        0.85
    """
    # Use default config if not provided
    if config_path is None:
        config_path = str(DEFAULT_CONFIG_PATH)
    
    # Load configuration
    config = AdapterConfig(config_path)
    
    # Create normalizer and process trace
    normalizer = _TraceNormalizer(config)
    return normalizer.normalize(path)


class _ConfidenceScorer:
    """
    Calculate confidence scores based on data quality.
    
    This class tracks confidence penalties during trace normalization and
    calculates final confidence scores for turns and runs. Penalties are
    deduplicated by root cause to avoid double-counting the same issue.
    
    Penalty Types (from spec requirement 12.2):
    - missing_timestamp: 0.4 (dedupe per turn)
    - missing_grouping_ids: 0.3 (dedupe per turn)
    - no_anchor_found: 0.3 (dedupe per turn)
    - no_llm_output: 0.2 (dedupe per turn)
    - missing_latency: 0.2 (dedupe per turn, only when timestamps exist but latency can't be computed)
    - single_turn_fallback: 0.25 (dedupe per run)
    - orphan_tool_results: 0.15 (dedupe per turn)
    """
    
    def __init__(self, config: AdapterConfig):
        """
        Initialize confidence scorer with configuration.
        
        Args:
            config: Adapter configuration with penalty weights
        """
        self.config = config
        self.penalties: List[Dict[str, Any]] = []
        self._penalty_config = config.get_confidence_scoring_config()
        
        # Track applied penalties per turn/run with (scope, reason) keys for proper deduplication
        self._applied_penalties_per_turn: Dict[str, set] = {}
        self._applied_penalties_per_run: set = set()  # Set of (scope, reason) tuples
        self._current_turn_id: Optional[str] = None
        
        # Track trusted timestamps for debugging
        self.trusted_timestamps: int = 0
        self.valid_timestamps: int = 0
    
    def set_current_turn(self, turn_id: str) -> None:
        """
        Set the current turn context for penalty tracking.
        
        Args:
            turn_id: Turn identifier for penalty deduplication
        """
        self._current_turn_id = turn_id
        if turn_id not in self._applied_penalties_per_turn:
            self._applied_penalties_per_turn[turn_id] = set()
    
    def add_penalty(self, reason: str, location: str, scope: str = "turn", turn_id: Optional[str] = None) -> None:
        """
        Record a confidence penalty with scope-aware deduplication.
        
        Penalties are deduplicated by (scope, reason) to avoid double-counting
        the same issue while allowing the same reason at different scopes.
        
        Args:
            reason: Penalty reason (e.g., "missing_timestamp")
            location: Where the penalty occurred (e.g., "turn_0.event_3")
            scope: Deduplication scope - "turn" or "run"
            turn_id: Turn identifier (required for turn-scope penalties, optional for run-scope)
            
        Behavior:
            - Run-level penalties: applied once per run regardless of how many turns trigger them
            - Turn-level penalties: applied once per turn, can repeat across different turns
            - Example: missing_timestamp at run-level (all turns) vs turn-level (specific turns)
        """
        # Get penalty value from config
        penalty_value = self._penalty_config["penalties"].get(reason, 0.0)
        
        # If penalty not in config, log warning and skip (strict spec mode)
        if penalty_value == 0.0 and reason not in self._penalty_config["penalties"]:
            # Silently skip undefined penalties (strict spec mode)
            return
        
        # Deduplicate penalties by (scope, reason)
        dedupe_key = (scope, reason)
        
        if scope == "run":
            if dedupe_key in self._applied_penalties_per_run:
                return
            self._applied_penalties_per_run.add(dedupe_key)
        elif scope == "turn":
            # Use provided turn_id or fall back to current turn
            effective_turn_id = turn_id or self._current_turn_id
            if not effective_turn_id:
                # Can't track turn-level penalty without turn_id
                return
            if effective_turn_id not in self._applied_penalties_per_turn:
                self._applied_penalties_per_turn[effective_turn_id] = set()
            if dedupe_key in self._applied_penalties_per_turn[effective_turn_id]:
                return
            self._applied_penalties_per_turn[effective_turn_id].add(dedupe_key)
        
        # Log warning for confidence penalty
        logger.warning(f"Applying confidence penalty: {reason} (penalty={penalty_value:.2f}, location={location}, scope={scope})")
        
        # Record penalty with scope and turn_id for robust filtering
        # FIX Issue #9: Only store schema-compliant fields (reason, penalty, location)
        # Keep scope and turn_id as internal metadata for filtering, but don't expose in final output
        self.penalties.append({
            "reason": reason,
            "penalty": penalty_value,
            "location": location,
            "scope": scope,  # Internal: for filtering
            "turn_id": turn_id or self._current_turn_id  # Internal: for filtering
        })
    
    def calculate_turn_confidence(self, turn_id: str) -> float:
        """
        Calculate final confidence score (0-1) for a turn.
        
        Includes both turn-specific and run-level penalties.
        
        Args:
            turn_id: Turn identifier (e.g., "turn_0")
            
        Returns:
            Confidence score between 0 and 1 (clamped)
        """
        # Get penalties for this turn
        # Include all run-scope penalties + turn-scope penalties for this turn
        turn_penalties = []
        for p in self.penalties:
            scope = p.get("scope", "turn")  # Default to turn for backward compat
            penalty_turn_id = p.get("turn_id")  # Explicit turn_id from penalty record
            
            # Include if: run-scope OR (turn-scope AND matches this turn)
            if scope == "run" or (scope == "turn" and penalty_turn_id == turn_id):
                turn_penalties.append(p)
        
        # Calculate confidence using utility function
        base_score = self._penalty_config["base"]
        score = calculate_confidence_score(turn_penalties, base_score)
        
        # Clamp to [0, 1] range (requirement: "Clamp confidence to [0, 1]")
        return max(0.0, min(1.0, float(score)))
    
    def calculate_run_confidence(self, turns: List[Dict[str, Any]]) -> float:
        """
        Calculate run-level confidence as average of valid turn confidences.
        
        Args:
            turns: List of turn dictionaries with 'confidence' field
            
        Returns:
            Run confidence score between 0 and 1, or 1.0 if no valid turns
            
        Aggregation rule:
            - Average of all turn confidence scores
            - Ignores empty/invalid turns (turns without confidence field)
            - Clamped to [0, 1] range
        """
        valid_confidences = [
            turn["confidence"] 
            for turn in turns 
            if "confidence" in turn and isinstance(turn["confidence"], (int, float))
        ]
        
        if not valid_confidences:
            return 1.0  # No valid turns, default to full confidence
        
        avg_confidence = sum(valid_confidences) / len(valid_confidences)
        return max(0.0, min(1.0, avg_confidence))  # Clamp to [0, 1]
    
    def get_adapter_stats(
        self,
        total_events: int,
        missing_data: int,
        turn_count: int,
        raw_path: Optional[str] = None,
        canonical_sources: Optional[Dict[str, str]] = None,
        orphan_tool_results: Optional[List[Dict[str, Any]]] = None,
        segmentation_strategy: Optional[str] = None,
        mapping_coverage: Optional[float] = None,
        segmentation_strategy_reason: Optional[str] = None,
        events_by_kind: Optional[Dict[str, int]] = None,
        dropped_events_count: int = 0,
        invalid_events_count: int = 0,
        warnings: Optional[List[str]] = None,
        missing_fields_summary: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate adapter_stats object with comprehensive statistics.
        
        Schema compliance: Only includes fields allowed by normalized_run.schema.json
        
        Args:
            total_events: Total number of events processed
            missing_data: Number of events with missing data across ALL field groups
            turn_count: Total number of turns segmented
            raw_path: Which event_path matched in source
            canonical_sources: Which field aliases matched per field (when found)
            orphan_tool_results: Tool results without corresponding calls (with location field)
            segmentation_strategy: Segmentation strategy used (REQUIRED)
            mapping_coverage: Overall field mapping coverage score 0-1 (REQUIRED, number only)
            segmentation_strategy_reason: Why this segmentation strategy was selected
            events_by_kind: Histogram of events by kind (including unknown/missing)
            dropped_events_count: Events that couldn't be processed (malformed)
            invalid_events_count: Events that failed validation
            warnings: List of warnings encountered during processing
            missing_fields_summary: Top 10 missing fields with aliases attempted
            
        Returns:
            adapter_stats dictionary compliant with schema (no forbidden fields)
        """
        # Schema-compliant structure - only allowed fields
        stats = {
            "total_events_processed": total_events,
            "events_with_valid_timestamps": self.trusted_timestamps,  # "valid" here means trusted-and-accepted by adapter policy
            "events_with_missing_data": missing_data,
            "dropped_events_count": dropped_events_count,
            "invalid_events_count": invalid_events_count,
            "turn_count": turn_count,
            # REQUIRED by adapter spec/tests:
            "mapping_coverage": mapping_coverage if mapping_coverage is not None else 0.0,
            "orphan_tool_results": orphan_tool_results or [],
            "segmentation_strategy": segmentation_strategy or "UNKNOWN",
        }
        
        # Clean penalties - strip internal fields (scope, turn_id) before exposing
        clean_penalties = [
            {"reason": p["reason"], "penalty": p["penalty"], "location": p["location"]}
            for p in self.penalties
        ]
        stats["confidence_penalties"] = clean_penalties
        
        # Optional fields allowed by schema
        if events_by_kind:
            stats["events_by_kind"] = events_by_kind
        
        if raw_path:
            stats["raw_path"] = raw_path
        
        if canonical_sources:
            stats["canonical_sources"] = canonical_sources
        
        # Note: orphan_tool_results is now typically included (even if empty list)
        
        if segmentation_strategy_reason:
            stats["segmentation_strategy_reason"] = segmentation_strategy_reason
        
        if warnings:
            stats["warnings"] = warnings
        
        if missing_fields_summary:
            stats["missing_fields_summary"] = missing_fields_summary
        
        return stats


class _TraceNormalizer:
    """
    Internal class for trace normalization logic.
    
    This class implements the multi-stage normalization pipeline:
    
    Stage A - Normalize:
    1. Load and parse JSON
    2. Extract events from configured paths
    3. Normalize fields using alias fallback (includes classification)
    4. Parse timestamps with multiple format support
    
    Stage B - Segment:
    5. Segment events into turns using configured strategies
    6. Order events within turns using timestamps and tie-breakers
    
    Stage C - Derive:
    7. Extract top-level fields (user_query, final_answer, finish_reason)
    8. Link tool calls with results
    9. Calculate latencies (normalized and runtime-reported)
    10. Classify phases and detect attribution
    11. Strip prompt context
    
    Stage D - Validate:
    12. Validate against schema
    13. Generate adapter_stats with warnings and diagnostics
    
    Note: Classification happens during normalization (Stage A) via config.classify_event(),
    not as a separate pipeline stage. Derived fields (attribution, fields_source) are
    computed internally but not exposed in output (schema compliance).
    """
    
    def __init__(self, config: AdapterConfig):
        """
        Initialize trace normalizer with configuration.
        
        Args:
            config: Adapter configuration
        """
        self.config = config
        self.scorer = _ConfidenceScorer(config)
        
        # Statistics tracking (scorer owns timestamp counters)
        self.total_events = 0
        self.missing_data_count = 0  # Comprehensive: tracks missing data across ALL field groups
        self.dropped_events_count = 0  # Events that couldn't be processed (malformed)
        self.invalid_events_count = 0  # Events that failed validation
        self.events_by_kind: Dict[str, int] = {}  # Histogram of events by kind (including unknown)
        self.matched_event_path: Optional[str] = None
        self.canonical_sources: Dict[str, str] = {}  # Fields that were found
        self.missing_fields_attempts: Dict[str, List[str]] = {}  # Track attempted aliases for missing fields
        self.orphan_tool_results: List[Dict[str, Any]] = []
        
        # Field group coverage tracking for mapping_coverage
        self.field_coverage = {
            "ids_found": 0,  # session_id, trace_id, span_id, request_id, turn_id
            "time_found": 0,  # timestamp, end_timestamp
            "tool_found": 0,  # tool_name, tool_run_id, tool_result
            "text_found": 0,  # text, role, user_query, final_answer
            "ids_total": 0,
            "time_total": 0,
            "tool_total": 0,
            "text_total": 0
        }
        
        # Warnings list
        self.warnings: List[str] = []
        
        # Schema validation warning flag
        self._schema_warning_emitted: bool = False
    
    def _track_field_coverage(self, event: Dict[str, Any], normalized: Dict[str, Any]) -> None:
        """
        Track field coverage for mapping_coverage calculation.
        
        This tracks event-level coverage: % of events that had at least one field
        from each field group successfully extracted (not per-field coverage).
        
        Updates field_coverage counters based on which fields were successfully extracted.
        
        Args:
            event: Raw event dictionary
            normalized: Normalized event dictionary
        """
        # Track IDs (session_id, trace_id, span_id, request_id, turn_id)
        # Event-level: did this event have at least one ID field?
        self.field_coverage["ids_total"] += 1
        id_fields = ["session_id", "trace_id", "span_id", "request_id", "turn_id"]
        if any(normalized.get(field) for field in id_fields):
            self.field_coverage["ids_found"] += 1
        
        # Track timestamps (timestamp, end_timestamp)
        # Event-level: did this event have at least one timestamp?
        self.field_coverage["time_total"] += 1
        if normalized.get("start_ts") or normalized.get("end_ts"):
            self.field_coverage["time_found"] += 1
        
        # Track tool fields (tool_name, tool_run_id, tool_result)
        # Event-level: did this event have at least one tool field?
        self.field_coverage["tool_total"] += 1
        tool_fields = ["tool_name", "tool_run_id", "tool_result"]
        if any(normalized.get(field) for field in tool_fields):
            self.field_coverage["tool_found"] += 1
        
        # Track text fields (text, role)
        # Event-level: did this event have at least one text field?
        self.field_coverage["text_total"] += 1
        text_fields = ["text", "role"]
        if any(normalized.get(field) for field in text_fields):
            self.field_coverage["text_found"] += 1
    
    def _has_missing_required_fields(self, normalized: Dict[str, Any], kind: Optional[str]) -> bool:
        """
        Check if event is missing required fields based on its kind.
        
        This is kind-aware: only checks fields that are expected for the given event kind.
        Reduces false positives where events are marked as "missing data" for fields
        they shouldn't have (e.g., USER_INPUT doesn't need tool_name).
        
        Args:
            normalized: Normalized event dictionary
            kind: Event kind (e.g., "USER_INPUT", "TOOL_CALL", etc.)
            
        Returns:
            True if event is missing required fields for its kind, False otherwise
        """
        # Define required fields per kind
        # These are fields that SHOULD be present for each kind
        required_by_kind = {
            "USER_INPUT": ["text"],  # User input should have text
            "TOOL_CALL": ["tool_name"],  # Tool calls should have tool name
            "TOOL_RESULT": ["tool_run_id"],  # Tool results should have run ID (tool_name propagated later)
            "LLM_OUTPUT_CHUNK": ["text"],  # LLM output should have text
            "ASSISTANT_MESSAGE": ["text"],  # Assistant messages should have text
            "FINAL_RESPONSE": ["text"],  # Final responses should have text
            "MODEL_INVOKE": [],  # Model invokes may not have specific required fields
            "EVENT": [],  # Generic events may not have specific required fields
            "UNKNOWN": [],  # Unknown events - can't determine requirements
        }
        
        # Get required fields for this kind (default to empty if kind not recognized)
        required_fields = required_by_kind.get(kind, [])
        
        # Check if any required field is missing
        for field in required_fields:
            if not normalized.get(field):
                return True
        
        # Also check for at least one grouping ID (session_id, trace_id, request_id, turn_id)
        # This is important for all event kinds
        grouping_ids = ["session_id", "trace_id", "request_id", "turn_id"]
        if not any(normalized.get(field) for field in grouping_ids):
            return True
        
        return False
    
    def _calculate_mapping_coverage(self) -> Dict[str, Any]:
        """
        Calculate mapping coverage breakdown.
        
        Coverage is measured at the event level: % of events that had at least
        one field from each field group successfully extracted.
        
        Returns:
            Dictionary with per-field-group event coverage (0..1 ratios) and overall coverage:
            - ids_coverage: ratio of events with at least one ID field (0..1)
            - time_coverage: ratio of events with at least one timestamp (0..1)
            - tool_coverage: ratio of events with at least one tool field (0..1)
            - text_coverage: ratio of events with at least one text field (0..1)
            - overall_mapping_coverage: average of all field group coverages (0..1)
        """
        def safe_ratio(found: int, total: int) -> float:
            return round((found / total) if total > 0 else 0.0, 4)
        
        ids_coverage = safe_ratio(
            self.field_coverage["ids_found"],
            self.field_coverage["ids_total"]
        )
        time_coverage = safe_ratio(
            self.field_coverage["time_found"],
            self.field_coverage["time_total"]
        )
        tool_coverage = safe_ratio(
            self.field_coverage["tool_found"],
            self.field_coverage["tool_total"]
        )
        text_coverage = safe_ratio(
            self.field_coverage["text_found"],
            self.field_coverage["text_total"]
        )
        
        # Overall coverage: average of all field groups
        overall_coverage = round(
            (ids_coverage + time_coverage + tool_coverage + text_coverage) / 4,
            4
        )
        
        return {
            "ids_coverage": ids_coverage,
            "time_coverage": time_coverage,
            "tool_coverage": tool_coverage,
            "text_coverage": text_coverage,
            "overall_mapping_coverage": overall_coverage
        }
    
    def _generate_missing_fields_summary(self) -> Dict[str, List[str]]:
        """
        Generate top 10 missing fields with aliases tried.
        
        Returns:
            Dictionary mapping field names to list of aliases that were attempted
            but didn't find the field. Limited to top 10 most commonly missing fields.
        """
        # Sort by number of attempts (most attempted = most important missing field)
        sorted_missing = sorted(
            self.missing_fields_attempts.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        # Return top 10
        return dict(sorted_missing[:10])
    
    def _flatten_list(self, data: Any) -> List[Any]:
        """
        Recursively flatten nested lists, filtering out None values.
        
        Args:
            data: Data to flatten (can be list, dict, or scalar)
            
        Returns:
            Flattened list of non-list, non-None items
        """
        if not isinstance(data, list):
            return [data] if data is not None else []
        
        result = []
        for item in data:
            if item is None:
                continue
            if isinstance(item, list):
                # Recursively flatten nested lists
                result.extend(self._flatten_list(item))
            else:
                result.append(item)
        return result
    
    def normalize(self, path: Union[str, os.PathLike, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Transform raw Generic JSON into normalized format.
        
        Args:
            path: Path to Generic JSON trace file OR a dict containing trace data
            
        Returns:
            Normalized dict with run_id, metadata, adapter_stats, turns[]
            
        Raises:
            InputError: If trace file doesn't exist, JSON is invalid, or invalid input type
            ValidationError: If no events exist or input is completely unreadable
        """
        logger.info(f"Starting normalization of trace: {path if not isinstance(path, dict) else '<in-memory-trace>'}")
        
        # Load trace data (from file or dict)
        raw_data = self._load_json(path)
        
        # Extract events from configured paths
        events = self._extract_events(raw_data)
        
        # Create trace label for logging (avoid dumping large dicts)
        trace_label = "<in-memory-trace>" if isinstance(path, dict) else str(path)
        
        # Validate that we have events
        if not events:
            logger.error(f"No events found in trace: {trace_label}")
            raise ValidationError(
                f"No events found in trace. "
                f"Checked event paths: {self.config.get_event_paths()}. "
                f"Please ensure the trace contains valid event data."
            )
        
        logger.info(f"Extracted {len(events)} events from trace: {trace_label}")
        
        # Extract run-level fields
        run_id = self._extract_run_id(raw_data, events)
        metadata = self._extract_metadata(raw_data)
        
        # Normalize events (field mapping, timestamp parsing, classification)
        normalized_events = self._normalize_events(events)
        
        # Segment into turns using configured strategies
        turn_groups, strategy_used, strategy_reason = self._segment_into_turns(normalized_events, raw_data)
        
        logger.info(f"Segmentation strategy selected: {strategy_used}")
        logger.debug(f"Segmentation reason: {strategy_reason}")
        
        # Add run-level penalty if SINGLE_TURN fallback was used
        if strategy_used == "SINGLE_TURN":
            logger.warning(f"Using SINGLE_TURN fallback strategy - applying confidence penalty")
            self.scorer.add_penalty("single_turn_fallback", "segmentation", scope="run")
        
        # Build turns from segmented event groups
        turns = []
        for turn_idx, turn_events in enumerate(turn_groups):
            # Order events within turn
            ordered_events = self._order_events_within_turn(turn_events)
            
            # Create turn object (pass strategy_used for top-level field extraction logic)
            turn = self._create_turn_from_events(ordered_events, turn_idx, raw_data, strategy_used)
            turns.append(turn)
        
        logger.info(f"Created {len(turns)} turn(s) from segmented events")
        
        # Store segmentation metadata in metadata (not adapter_stats)
        metadata["segmentation_strategy_used"] = strategy_used
        
        # Calculate run_confidence (average of valid turn confidences)
        run_confidence = self.scorer.calculate_run_confidence(turns)
        metadata["run_confidence"] = run_confidence
        
        logger.info(f"Run confidence score: {run_confidence:.2f}")
        
        # Calculate mapping_coverage
        mapping_coverage = self._calculate_mapping_coverage()
        # Store full breakdown in metadata (object with ids/time/tool/text coverage)
        metadata["mapping_coverage"] = mapping_coverage
        
        # Extract overall score for adapter_stats (number only)
        overall_mapping_coverage = mapping_coverage.get("overall_mapping_coverage", 0.0)
        
        # Add adapter_version and processed_at to metadata (required by schema)
        metadata["adapter_version"] = ADAPTER_VERSION
        metadata["processed_at"] = datetime.now(timezone.utc).isoformat()
        
        # Validate against schema (before generating adapter_stats to capture warnings)
        self._validate_output_structure(normalized_events, turns, run_id, metadata)
        
        # Generate missing fields summary for debugging
        missing_fields_summary = self._generate_missing_fields_summary()
        
        # Generate adapter_stats (scorer uses its own internal counters)
        adapter_stats = self.scorer.get_adapter_stats(
            total_events=self.total_events,
            missing_data=self.missing_data_count,
            turn_count=len(turns),
            raw_path=self.matched_event_path,
            canonical_sources=self.canonical_sources,
            orphan_tool_results=self.orphan_tool_results,          # pass list even if empty
            segmentation_strategy=strategy_used,                   # REQUIRED field
            mapping_coverage=overall_mapping_coverage,             # REQUIRED field (number only)
            segmentation_strategy_reason=strategy_reason if self.config.should_emit_strategy_reason() else None,
            events_by_kind=self.events_by_kind if self.events_by_kind else None,
            dropped_events_count=self.dropped_events_count,
            invalid_events_count=self.invalid_events_count,
            warnings=self.warnings if self.warnings else None,
            missing_fields_summary=missing_fields_summary if missing_fields_summary else None
        )
        
        # Build normalized output - schema compliant top-level structure
        normalized = {
            "run_id": run_id,
            "metadata": metadata,
            "adapter_stats": adapter_stats,
            "turns": turns
        }
        
        # Final schema validation
        self._validate_output(normalized)
        
        logger.info(f"Successfully normalized trace file: {trace_label}")
        logger.debug(f"Adapter stats: {len(adapter_stats.get('confidence_penalties', []))} confidence penalties applied")
        
        return normalized
    
    def _validate_output_structure(
        self,
        events: List[Dict[str, Any]],
        turns: List[Dict[str, Any]],
        run_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Perform cheap invariant checks before full schema validation.
        
        These are fast sanity checks that catch integration mistakes early
        without requiring jsonschema.
        
        Args:
            events: Normalized events
            turns: Turn list
            run_id: Run identifier
            metadata: Metadata dict
        """
        # Check run_id is non-empty string
        if not run_id or not isinstance(run_id, str):
            self.warnings.append(f"Invalid run_id: expected non-empty string, got {type(run_id).__name__}")
        
        # Check turns is a list
        if not isinstance(turns, list):
            self.warnings.append(f"Invalid turns: expected list, got {type(turns).__name__}")
            return
        
        # Check each turn has required fields
        for i, turn in enumerate(turns):
            if not isinstance(turn, dict):
                self.warnings.append(f"Turn {i}: expected dict, got {type(turn).__name__}")
                continue
            
            if "turn_id" not in turn:
                self.warnings.append(f"Turn {i}: missing turn_id")
            
            if "steps" in turn and not isinstance(turn["steps"], list):
                self.warnings.append(f"Turn {i}: steps should be list, got {type(turn['steps']).__name__}")
            
            if "confidence" in turn:
                conf = turn["confidence"]
                if conf is not None:
                    try:
                        conf_float = float(conf)
                        if not (0.0 <= conf_float <= 1.0):
                            self.warnings.append(f"Turn {i}: confidence should be 0..1, got {conf}")
                    except (TypeError, ValueError):
                        self.warnings.append(f"Turn {i}: confidence should be numeric, got {type(conf).__name__}")
    
    def _load_json(self, path: Union[str, os.PathLike, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load and parse JSON file or accept dict directly.
        
        Args:
            path: Path to JSON file OR a dict containing trace data
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            InputError: If file doesn't exist, JSON is invalid, file is unreadable, or invalid input type
        """
        # If already a dict, return it directly (for testing and programmatic use)
        if isinstance(path, dict):
            if not path:
                raise InputError(
                    "Trace data is empty. Please ensure the dict contains valid trace data.",
                    file_path="<in-memory-trace>"
                )
            return path
        
        # Reject non-dict, non-path inputs with clear error
        if not isinstance(path, (str, os.PathLike)):
            raise InputError(
                f"Expected a file path (str/PathLike) or trace dict, got {type(path).__name__}. "
                f"Lists and other types are not supported.",
                file_path="<in-memory-trace>"
            )
        
        # Otherwise, treat as file path
        path_obj = Path(path)
        path_str = str(path)
        
        if not path_obj.exists():
            raise InputError(
                "Trace file not found. Please ensure the file exists.",
                file_path=path_str
            )
        
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise InputError(
                "Failed to parse JSON. Please ensure the file contains valid JSON.",
                file_path=path_str,
                original_error=e
            )
        except OSError as e:
            raise InputError(
                "Failed to read file. Check file permissions.",
                file_path=path_str,
                original_error=e
            )
        
        if data is None:
            raise InputError(
                "Trace file is empty. Please ensure the file contains valid JSON data.",
                file_path=path_str
            )
        
        return data
    
    def _extract_events(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract events from configured event_paths.
        
        Supports:
        - Simple paths: "events", "trace.events"
        - Nested paths with wildcards: "resourceSpans.*.scopeSpans.*.spans"
        
        Warns if multiple paths match (helps debug config changes).
        Filters out malformed events (non-dict) and tracks dropped_events_count.
        
        Args:
            data: Source JSON dictionary
            
        Returns:
            List of valid event dictionaries (malformed events dropped)
        """
        raw_events = []
        event_paths = self.config.get_event_paths()
        matched_paths = []
        
        for path in event_paths:
            extracted = self._extract_from_path(data, path)
            
            # Use "is not None" to handle empty lists correctly
            if extracted is not None:
                # Handle empty list case
                if isinstance(extracted, list):
                    if len(extracted) == 0:
                        # Found but empty - continue to next path
                        continue
                    # Non-empty list found
                    if not raw_events:
                        # First match - use it
                        raw_events.extend(extracted)
                        self.matched_event_path = path
                    else:
                        # Additional match - warn about it
                        matched_paths.append((path, len(extracted)))
                else:
                    # Single item found
                    if not raw_events:
                        raw_events.append(extracted)
                        self.matched_event_path = path
                    else:
                        matched_paths.append((path, 1))
        
        # Warn if multiple paths matched (helps debug config drift)
        if matched_paths:
            alt_paths_str = ", ".join(f"{p} ({n} events)" for p, n in matched_paths)
            self.warnings.append(
                f"Multiple event paths matched. Using '{self.matched_event_path}' ({len(raw_events)} events). "
                f"Alternatives: {alt_paths_str}"
            )
        
        # FIX Issue #3: Filter out malformed events (non-dict) and track dropped_events_count
        valid_events = []
        for idx, event in enumerate(raw_events):
            if not isinstance(event, dict):
                self.dropped_events_count += 1
                if len(self.warnings) < 100:
                    self.warnings.append(
                        f"Dropped malformed event at index {idx}: expected dict, got {type(event).__name__}"
                    )
            else:
                valid_events.append(event)
        
        return valid_events
    
    def _extract_from_path(self, data: Any, path: str) -> Optional[Union[List, Dict]]:
        """
        Extract value from nested path with wildcard support.
        
        Handles nested wildcards with proper flattening at every step.
        Also supports literal dotted keys (e.g., OTEL-style "session.id" as a single key).
        
        Args:
            data: Source data
            path: Dotted path (e.g., "events" or "resourceSpans.*.scopeSpans.*.spans")
            
        Returns:
            Extracted value or None
        """
        # Try literal key first for OTEL-style dotted keys (e.g., {"session.id": "abc"})
        if isinstance(data, dict) and path in data:
            return data[path]
        
        parts = path.split('.')
        current = data
        
        for idx, part in enumerate(parts):
            if current is None:
                return None
            
            if part == '*':
                # Wildcard: traverse all items in list/dict
                if isinstance(current, list):
                    # Collect all non-None items
                    results = []
                    for item in current:
                        if item is not None:
                            results.append(item)
                    current = results if results else None
                elif isinstance(current, dict):
                    # Collect all non-None values
                    results = []
                    for value in current.values():
                        if value is not None:
                            results.append(value)
                    current = results if results else None
                else:
                    return None
                
                # Flatten after wildcard traversal
                if current is not None:
                    current = self._flatten_list(current)
                    if not current:  # Empty after flattening
                        current = None
            else:
                # Regular key access
                if isinstance(current, dict):
                    # Try literal dotted key for remaining path first (e.g., "session.id" in {"session.id": "abc"})
                    remaining_path = '.'.join(parts[idx:])
                    if remaining_path in current:
                        return current[remaining_path]
                    
                    # Fall back to regular key access
                    current = current.get(part)
                elif isinstance(current, list):
                    # Apply key to all items in list
                    results = []
                    for item in current:
                        if isinstance(item, dict):
                            # Try literal dotted key for remaining path first
                            remaining_path = '.'.join(parts[idx:])
                            if remaining_path in item:
                                results.append(item[remaining_path])
                                continue
                            
                            # Fall back to regular key access
                            value = item.get(part)
                            if value is not None:
                                results.append(value)
                    current = results if results else None
                    
                    # Flatten after list key application
                    if current is not None:
                        current = self._flatten_list(current)
                        if not current:  # Empty after flattening
                            current = None
                else:
                    return None
        
        return current
    
    def _extract_run_id(self, data: Dict[str, Any], events: List[Dict[str, Any]]) -> str:
        """
        Extract run_id using config-driven field mapping.
        
        Uses deterministic generation based on trace content if no run_id found.
        
        Args:
            data: Source JSON dictionary
            events: List of events
            
        Returns:
            Run ID string
        """
        # Try direct access first (common case)
        run_id = data.get("run_id")
        
        # If not found, try config-driven field mapping
        if not run_id:
            run_id = self.config.get_field_value_with_fallback(data, "run_id")
        
        # If not found, try first event
        if not run_id and events:
            run_id = self.config.get_field_value_with_fallback(events[0], "run_id")
        
        # If still not found, generate deterministic ID from trace content
        if not run_id:
            # Build stable hash from trace_id/session_id/request_id + first timestamp + event_count + path
            hash_parts = []
            
            # Try to get stable identifiers
            for id_field in ["trace_id", "session_id", "request_id"]:
                value = self.config.get_field_value_with_fallback(data, id_field)
                if not value and events:
                    value = self.config.get_field_value_with_fallback(events[0], id_field)
                if value:
                    hash_parts.append(f"{id_field}:{value}")
            
            # Add first timestamp if available
            if events:
                first_ts = self.config.get_field_value_with_fallback(events[0], "timestamp")
                if first_ts:
                    hash_parts.append(f"ts:{first_ts}")
            
            # typically include event count and matched path to reduce collisions
            hash_parts.append(f"event_count:{len(events)}")
            if self.matched_event_path:
                hash_parts.append(f"matched_path:{self.matched_event_path}")
            
            # Generate deterministic hash
            if hash_parts:
                content = "|".join(hash_parts)
                hash_digest = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
                run_id = f"generated_{hash_digest}"
            else:
                # Last resort: hash bounded canonical subset
                # Use top-level keys + first 3 events + size cap
                canonical_subset = {
                    "keys": sorted(data.keys()),
                    "events_sample": events[:3] if events else [],
                    "event_count": len(events)
                }
                try:
                    content = json.dumps(canonical_subset, sort_keys=True, default=str)
                    # Cap at 10KB to avoid huge traces
                    if len(content) > 10000:
                        content = content[:10000]
                    hash_digest = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
                    run_id = f"generated_{hash_digest}"
                except (TypeError, ValueError):
                    # Fallback to simple hash of keys
                    content = "|".join(sorted(data.keys()))
                    hash_digest = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
                    run_id = f"generated_{hash_digest}"
            
            self.warnings.append(f"No run_id found in trace. Using generated ID: {run_id}")
        
        return str(run_id)
    
    def _extract_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract run-level metadata.
        
        Note: adapter_version and processed_at are set later in normalize()
        after all processing is complete.
        
        Args:
            data: Source JSON dictionary
            
        Returns:
            Metadata dictionary with optional source field
        """
        metadata = {}
        
        # Extract optional source field
        source = self.config.get_field_value_with_fallback(data, "source")
        if source:
            metadata["source"] = str(source)
        
        return metadata
    
    def _normalize_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize events: field mapping, timestamp parsing, classification.
        
        Sorts events deterministically by: trusted timestamps first, then by start_ts_epoch_ms asc, then by source_index asc.
        
        Args:
            events: Raw event list
            
        Returns:
            List of normalized and sorted events with canonical fields
        """
        normalized = []
        
        for idx, event in enumerate(events):
            self.total_events += 1
            
            # FIX Issue #4: Add per-event validation and track invalid_events_count
            try:
                # Normalize single event
                normalized_event = self._normalize_event(event, idx)
                
                # Basic validation: helps required fields exist
                if not normalized_event.get("kind"):
                    self.invalid_events_count += 1
                    if len(self.warnings) < 100:
                        self.warnings.append(f"Event {idx}: no kind assigned (classification failed)")
                
                normalized.append(normalized_event)
            except Exception as e:
                # Event normalization failed - track as invalid
                self.invalid_events_count += 1
                if len(self.warnings) < 100:
                    self.warnings.append(f"Event {idx}: normalization failed: {str(e)[:200]}")
                # Continue processing other events (graceful degradation)
        
        # Sort events deterministically using numeric epoch for reliable ordering
        def sort_key(evt):
            ts_trusted = evt.get("ts_trusted", False)
            start_ts_epoch_ms = evt.get("start_ts_epoch_ms")
            source_index = evt.get("source_index", 0)
            
            # Sort by: ts_trusted (desc), start_ts_epoch_ms (asc, None last), source_index (asc)
            return (
                not ts_trusted,  # False (trusted) sorts before True (untrusted)
                start_ts_epoch_ms if start_ts_epoch_ms is not None else float('inf'),  # None sorts last
                source_index
            )
        
        normalized.sort(key=sort_key)
        
        # Update event_order after sorting
        for order, event in enumerate(normalized):
            event["event_order"] = order
        
        return normalized
    
    def _normalize_event(self, event: Dict[str, Any], source_index: int) -> Dict[str, Any]:
        """
        Normalize a single event with field mapping and timestamp parsing.
        
        Args:
            event: Raw event dictionary
            source_index: Original position in event list
            
        Returns:
            Normalized event dictionary with metadata for penalty tracking
        """
        normalized = {
            "source_index": source_index,
            "event_order": source_index,  # Will be updated during segmentation
            "raw": {},
            "start_ts_epoch_ms": None,  # For numeric sorting
            "end_ts_epoch_ms": None,  # For latency calculation
            "_has_trusted_ts": False,  # Metadata for penalty tracking
            "_has_grouping_id": False  # Metadata for penalty tracking
        }
        
        # Extract and parse start timestamp
        timestamp_value = self.config.get_field_value_with_fallback(event, "timestamp")
        ts_config = self.config.get_timestamp_parse_config()
        
        # Check if this is a UnixNano field (using alias fallback for nested paths)
        field_name = None
        matched_alias = None
        for unix_nano_field in ts_config["unix_nano_fields"]:
            # Use alias fallback to support nested paths like span.startTimeUnixNano
            unix_nano_value = self.config.get_field_value_with_fallback(event, unix_nano_field)
            if unix_nano_value is not None:
                field_name = unix_nano_field
                timestamp_value = unix_nano_value
                matched_alias = unix_nano_field
                break
        
        # Track if this event has any valid/trusted timestamps (event-based counting)
        event_has_valid_ts = False
        event_has_trusted_ts = False
        
        if timestamp_value is not None:
            dt, is_trusted, error = parse_timestamp(
                timestamp_value,
                formats=ts_config["formats"],
                epoch_units=ts_config["epoch_units"],
                infer_epoch_unit_by_magnitude=ts_config["infer_epoch_unit_by_magnitude"],
                min_reasonable_year=ts_config["min_reasonable_year"],
                max_reasonable_year=ts_config["max_reasonable_year"],
                unix_nano_fields=ts_config["unix_nano_fields"],
                field_name=field_name
            )
            
            if dt:
                # helps timezone-aware for epoch conversion
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                
                normalized["start_ts"] = dt.isoformat()
                normalized["start_ts_epoch_ms"] = int(dt.timestamp() * 1000)  # Integer for precision
                normalized["ts_trusted"] = is_trusted
                normalized["_has_trusted_ts"] = is_trusted
                
                # Track for event-based counting
                event_has_valid_ts = True
                if is_trusted:
                    event_has_trusted_ts = True
                
                # Track which alias matched
                if not matched_alias:
                    for alias in self.config.get_field_aliases("timestamp"):
                        if self._extract_from_path(event, alias) is not None:
                            matched_alias = alias
                            break
                if matched_alias and "timestamp" not in self.canonical_sources:
                    self.canonical_sources["timestamp"] = matched_alias
            else:
                normalized["start_ts"] = None
                normalized["ts_trusted"] = False
                # Don't increment missing_data_count here - timestamps are tracked separately
                # Log parse error for debugging (bounded)
                if error and len(self.warnings) < 100:  # Limit warnings
                    self.warnings.append(f"Timestamp parse failed for event {source_index}: {error}")
        else:
            normalized["start_ts"] = None
            normalized["ts_trusted"] = False
            # Don't increment missing_data_count here - timestamps are tracked separately
        
        # Extract and parse end timestamp
        end_timestamp_value = self.config.get_field_value_with_fallback(event, "end_timestamp")
        end_field_name = None
        end_matched_alias = None
        
        # Check for endTimeUnixNano using alias fallback
        for unix_nano_field in ts_config["unix_nano_fields"]:
            if "end" in unix_nano_field.lower():
                unix_nano_value = self.config.get_field_value_with_fallback(event, unix_nano_field)
                if unix_nano_value is not None:
                    end_field_name = unix_nano_field
                    end_timestamp_value = unix_nano_value
                    end_matched_alias = unix_nano_field
                    break
        
        if end_timestamp_value is not None:
            dt_end, is_trusted_end, error_end = parse_timestamp(
                end_timestamp_value,
                formats=ts_config["formats"],
                epoch_units=ts_config["epoch_units"],
                infer_epoch_unit_by_magnitude=ts_config["infer_epoch_unit_by_magnitude"],
                min_reasonable_year=ts_config["min_reasonable_year"],
                max_reasonable_year=ts_config["max_reasonable_year"],
                unix_nano_fields=ts_config["unix_nano_fields"],
                field_name=end_field_name
            )
            
            if dt_end:
                # helps timezone-aware for epoch conversion
                if dt_end.tzinfo is None:
                    dt_end = dt_end.replace(tzinfo=timezone.utc)
                
                normalized["end_ts"] = dt_end.isoformat()
                normalized["end_ts_epoch_ms"] = int(dt_end.timestamp() * 1000)
                normalized["end_ts_trusted"] = is_trusted_end
                
                # Track for event-based counting
                event_has_valid_ts = True
                if is_trusted_end:
                    event_has_trusted_ts = True
                
                # Track which alias matched
                if not end_matched_alias:
                    for alias in self.config.get_field_aliases("end_timestamp"):
                        if self._extract_from_path(event, alias) is not None:
                            end_matched_alias = alias
                            break
                if end_matched_alias and "end_timestamp" not in self.canonical_sources:
                    self.canonical_sources["end_timestamp"] = end_matched_alias
        
        # FIX Issue #1: Event-based counting (count event once if it has any valid/trusted timestamp)
        if event_has_valid_ts:
            self.scorer.valid_timestamps += 1
        if event_has_trusted_ts:
            self.scorer.trusted_timestamps += 1
        
        # Extract identifiers with canonical source tracking
        for field in ["session_id", "trace_id", "span_id", "parent_span_id", "request_id", "turn_id"]:
            value = self.config.get_field_value_with_fallback(event, field)
            if value is not None:
                normalized[field] = str(value)
                if field in ["session_id", "trace_id", "request_id", "turn_id"]:
                    normalized["_has_grouping_id"] = True
                # Track which alias matched (first match only)
                if field not in self.canonical_sources:
                    for alias in self.config.get_field_aliases(field):
                        if self._extract_from_path(event, alias) is not None:
                            self.canonical_sources[field] = alias
                            break
        
        # Extract event typing fields
        for field in ["event_type", "operation"]:
            value = self.config.get_field_value_with_fallback(event, field)
            if value is not None:
                normalized[field] = str(value)
                # Track which alias matched
                if field not in self.canonical_sources:
                    for alias in self.config.get_field_aliases(field):
                        if self._extract_from_path(event, alias) is not None:
                            self.canonical_sources[field] = alias
                            break
        
        # Extract tool fields
        for field in ["tool_name", "tool_run_id", "tool_result", "tool_input", "tool_arguments"]:
            value = self.config.get_field_value_with_fallback(event, field)
            if value is not None:
                normalized[field] = value
                # Track which alias matched
                if field not in self.canonical_sources:
                    for alias in self.config.get_field_aliases(field):
                        if self._extract_from_path(event, alias) is not None:
                            self.canonical_sources[field] = alias
                            break
        
        # Extract model/message fields
        for field in ["role", "span_kind", "model_id", "text"]:
            value = self.config.get_field_value_with_fallback(event, field)
            if value is not None:
                normalized[field] = value
                # Track which alias matched
                if field not in self.canonical_sources:
                    for alias in self.config.get_field_aliases(field):
                        if self._extract_from_path(event, alias) is not None:
                            self.canonical_sources[field] = alias
                            break
        
        # Extract step fields
        status = self.config.get_field_value_with_fallback(event, "status")
        if status:
            normalized["status"] = str(status)
            # Track which alias matched
            if "status" not in self.canonical_sources:
                for alias in self.config.get_field_aliases("status"):
                    if self._extract_from_path(event, alias) is not None:
                        self.canonical_sources["status"] = alias
                        break
        
        latency = self.config.get_field_value_with_fallback(event, "latency_ms")
        if latency is not None:
            sanitized = sanitize_latency(latency)
            if sanitized is not None:
                normalized["latency_ms"] = float(sanitized)
                # Track which alias matched
                if "latency_ms" not in self.canonical_sources:
                    for alias in self.config.get_field_aliases("latency_ms"):
                        if self._extract_from_path(event, alias) is not None:
                            self.canonical_sources["latency_ms"] = alias
                            break
        
        # FIX Issue #2: Increment missing_data_count if this event had ANY missing important fields
        # DEFERRED: Kind-aware missing data check happens after classification
        
        # Classify event using normalized dict (not raw event) - FIX #3 and N1
        # Build a dict with normalized fields taking precedence over raw
        classification_dict = {
            **event,  # Raw event as base
            **normalized  # Normalized fields override (correct precedence)
        }
        kind, rule_id, kind_reason = self.config.classify_event(classification_dict)
        normalized["kind"] = kind
        if rule_id:
            normalized["kind_rule_id"] = rule_id
        if kind_reason:
            normalized["kind_reason"] = kind_reason
        
        # FIX: Kind-aware missing data check (only check fields relevant to this event kind)
        if self._has_missing_required_fields(normalized, kind):
            self.missing_data_count += 1
        
        # FIX Issue #10: Track ALL events by kind, including None/empty/unknown
        kind_key = kind if kind else "UNKNOWN"
        self.events_by_kind[kind_key] = self.events_by_kind.get(kind_key, 0) + 1
        
        # Preserve raw data (byte-safe size check with safe serialization)
        carry_config = self.config.get_carry_fields_config()
        if carry_config["keep_raw_event"]:
            # Limit raw event size (byte-safe, handle non-serializable)
            try:
                raw_json = json.dumps(event, default=str)
                raw_bytes = raw_json.encode('utf-8')
                max_bytes = carry_config["raw_event_max_bytes"]
                if len(raw_bytes) <= max_bytes:
                    # Store serialized version for robustness
                    normalized["raw"] = json.loads(raw_json)
                else:
                    normalized["raw"] = {"truncated": True, "size_bytes": len(raw_bytes)}
            except (TypeError, ValueError) as e:
                normalized["raw"] = {"error": f"Serialization failed: {str(e)[:100]}"}
        
        # Extract attributes (check for None and dict type)
        attributes = {}
        for attr_path in carry_config["attributes_paths"]:
            attr_value = self._extract_from_path(event, attr_path)
            if attr_value is not None and isinstance(attr_value, dict):
                attributes.update(attr_value)
        
        if attributes:
            normalized["attributes"] = attributes
        
        # Track field coverage for mapping_coverage calculation
        self._track_field_coverage(event, normalized)
        
        return normalized
    
    def _segment_into_turns(
        self,
        events: List[Dict[str, Any]],
        raw_data: Dict[str, Any]
    ) -> Tuple[List[List[Dict[str, Any]]], str, Optional[str]]:
        """
        Segment events into turns using configured strategies.
        
        Tries strategies in preference order until one succeeds:
        1. TURN_ID: Explicit turn_id fields
        2. SESSION_PLUS_REQUEST: session_id + request_id combinations
        3. SESSION_PLUS_TRACE_THEN_ANCHOR_SPLIT: Trace-based with anchor splitting
        4. SINGLE_TURN: Fallback - treat all as one turn
        
        Args:
            events: Normalized events
            raw_data: Original JSON data
            
        Returns:
            Tuple of (turn_groups, strategy_used, strategy_reason)
            - turn_groups: List of event lists (one per turn)
            - strategy_used: Name of successful strategy
            - strategy_reason: Optional explanation of why this strategy was chosen
        """
        strategies = self.config.get_segmentation_strategies()
        
        for strategy in strategies:
            if strategy == "TURN_ID":
                turn_groups = self._segment_by_turn_id(events)
                if turn_groups:
                    reason = "Found explicit turn_id fields in events"
                    return turn_groups, strategy, reason
            
            elif strategy == "SESSION_PLUS_REQUEST":
                turn_groups = self._segment_by_session_plus_request(events)
                if turn_groups:
                    reason = "Found session_id + request_id combinations"
                    return turn_groups, strategy, reason
            
            elif strategy == "SESSION_PLUS_TRACE_THEN_ANCHOR_SPLIT":
                turn_groups = self._segment_by_session_trace_anchor(events)
                if turn_groups:
                    reason = "Segmented by session/trace with anchor event splitting"
                    return turn_groups, strategy, reason
            
            elif strategy == "SINGLE_TURN":
                # Fallback typically succeeds
                reason = "No segmentation identifiers found - treating as single turn"
                return [events], strategy, reason
        
        # Should is not expected to reach here if SINGLE_TURN is in strategies
        return [events], "SINGLE_TURN", "Fallback to single turn"
    
    def _segment_by_turn_id(self, events: List[Dict[str, Any]]) -> Optional[List[List[Dict[str, Any]]]]:
        """
        Segment events by explicit turn_id fields.
        
        Looks for turn_id or request_id fields in events and groups by those values.
        Events without turn_id are excluded from segmentation (noise events).
        Fails if more than 50% of events lack turn_id (indicates poor segmentation quality).
        
        Args:
            events: Normalized events
            
        Returns:
            List of event groups (one per turn) or None if no turn_id fields found or quality too low
        """
        turn_id_fields = self.config.get_turn_id_fields()
        
        # Check if any events have turn_id fields
        has_turn_ids = False
        for event in events:
            for field in turn_id_fields:
                if event.get(field):
                    has_turn_ids = True
                    break
            if has_turn_ids:
                break
        
        if not has_turn_ids:
            return None
        
        # Group events by turn_id (try each field in order)
        # FIX: Only include events WITH explicit turn_id to prevent over-splitting
        turn_groups: Dict[str, List[Dict[str, Any]]] = {}
        events_without_turn_id = 0
        
        for event in events:
            # Find first available turn_id field
            turn_id = None
            for field in turn_id_fields:
                turn_id = event.get(field)
                if turn_id:
                    break
            
            # FIX: Skip events without turn_id instead of creating "unknown" group
            # This prevents noise events (internal.metric, debug.log) from creating extra turns
            if not turn_id:
                events_without_turn_id += 1
                continue  # Skip this event - don't add to any turn group
            
            turn_id = str(turn_id)
            
            if turn_id not in turn_groups:
                turn_groups[turn_id] = []
            
            turn_groups[turn_id].append(event)
        
        # Check quality: fail if too many events lack turn_id (indicates poor segmentation quality)
        # Changed from checking "unknown" group to checking skipped events
        events_with_turn_id = len(events) - events_without_turn_id
        if events_with_turn_id == 0:
            return None  # No events with turn_id
        
        skip_ratio = events_without_turn_id / max(1, len(events))
        if skip_ratio > 0.5:
            return None  # Strategy failed - too many events without turn_id
        
        # Convert to list of lists, sorted by earliest event timestamp (not lexicographic)
        result = []
        
        # Sort groups by minimum timestamp within each group
        def get_min_timestamp(group: List[Dict[str, Any]]) -> float:
            """Get earliest timestamp from group, fallback to source_index."""
            min_ts = float('inf')
            min_source_idx = float('inf')
            
            for event in group:
                if event.get('ts_trusted') and event.get('start_ts_epoch_ms') is not None:
                    min_ts = min(min_ts, event['start_ts_epoch_ms'])
                if event.get('source_index') is not None:
                    min_source_idx = min(min_source_idx, event['source_index'])
            
            # Return timestamp if found, otherwise source_index
            return min_ts if min_ts != float('inf') else min_source_idx
        
        sorted_groups = sorted(turn_groups.items(), key=lambda item: get_min_timestamp(item[1]))
        result = [group for _, group in sorted_groups]
        
        return result if result else None
    
    def _segment_by_session_plus_request(self, events: List[Dict[str, Any]]) -> Optional[List[List[Dict[str, Any]]]]:
        """
        Segment events by session_id + request_id combinations.
        
        Groups events that share both session_id and request_id.
        Requires minimum coverage: ≥20% events have both IDs or ≥2 distinct (session, request) keys.
        
        Args:
            events: Normalized events
            
        Returns:
            List of event groups (one per turn) or None if insufficient identifiers
        """
        # Check if we have both session_id and request_id
        has_session = any(event.get("session_id") for event in events)
        has_request = any(event.get("request_id") for event in events)
        
        if not (has_session and has_request):
            return None
        
        # Group by (session_id, request_id) tuple
        turn_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        events_with_both_ids = 0
        
        for event in events:
            session_id = event.get("session_id", "unknown")
            request_id = event.get("request_id", "unknown")
            
            # Track coverage
            if session_id != "unknown" and request_id != "unknown":
                events_with_both_ids += 1
            
            key = (str(session_id), str(request_id))
            
            if key not in turn_groups:
                turn_groups[key] = []
            
            turn_groups[key].append(event)
        
        # Check success criteria: ≥20% coverage OR ≥2 distinct keys
        coverage_ratio = events_with_both_ids / max(1, len(events))
        distinct_keys = len([k for k in turn_groups.keys() if k != ("unknown", "unknown")])
        
        if coverage_ratio < 0.2 and distinct_keys < 2:
            return None  # Strategy failed - insufficient coverage
        
        # Convert to list of lists, sorted by earliest event timestamp (not lexicographic)
        result = []
        
        # Sort groups by minimum timestamp within each group
        def get_min_timestamp(group: List[Dict[str, Any]]) -> float:
            """Get earliest timestamp from group, fallback to source_index."""
            min_ts = float('inf')
            min_source_idx = float('inf')
            
            for event in group:
                if event.get('ts_trusted') and event.get('start_ts_epoch_ms') is not None:
                    min_ts = min(min_ts, event['start_ts_epoch_ms'])
                if event.get('source_index') is not None:
                    min_source_idx = min(min_source_idx, event['source_index'])
            
            # Return timestamp if found, otherwise source_index
            return min_ts if min_ts != float('inf') else min_source_idx
        
        sorted_groups = sorted(turn_groups.items(), key=lambda item: get_min_timestamp(item[1]))
        result = [group for _, group in sorted_groups]
        
        return result if result else None
    
    def _segment_by_session_trace_anchor(self, events: List[Dict[str, Any]]) -> Optional[List[List[Dict[str, Any]]]]:
        """
        Segment events by session/trace ID, then split using anchor events.
        
        This strategy:
        1. Groups events by session_id or trace_id
        2. Within each group, splits on anchor events (USER_INPUT, MODEL_INVOKE)
        3. Applies request_id diagnosis to detect stitched traces
        
        Args:
            events: Normalized events
            
        Returns:
            List of event groups (one per turn) or None if no grouping IDs found
        """
        # Check if we have session_id or trace_id
        has_session = any(event.get("session_id") for event in events)
        has_trace = any(event.get("trace_id") for event in events)
        
        if not (has_session or has_trace):
            return None
        
        # Group by session_id or trace_id (prefer session_id)
        groups: Dict[str, List[Dict[str, Any]]] = {}
        
        for event in events:
            group_id = event.get("session_id") or event.get("trace_id")
            
            if not group_id:
                group_id = "unknown"
            
            group_id = str(group_id)
            
            if group_id not in groups:
                groups[group_id] = []
            
            groups[group_id].append(event)
        
        # Apply request_id diagnosis for stitched trace detection
        diagnosis_config = self.config.get_request_id_diagnosis_config()
        is_stitched = self._diagnose_stitched_trace(events, diagnosis_config)
        
        if is_stitched:
            self.warnings.append("Detected stitched trace: multiple distinct user prompts per request_id")
        
        # Split each group by anchor events
        anchor_events = self.config.get_anchor_events()
        result = []
        
        for group_id, group_events in groups.items():
            # FIX N7: Sort group before splitting to helps correct anchor boundaries
            # Sort by: trusted timestamps first, then by timestamp value, then by source_index
            sorted_group = sorted(
                group_events,
                key=lambda e: (
                    not e.get("ts_trusted", False),  # Trusted first (False sorts before True, so invert with 'not')
                    e.get("start_ts_epoch_ms") if e.get("start_ts_epoch_ms") is not None else float('inf'),
                    e.get("source_index", 0)
                )
            )
            
            # Split by anchor events
            turns = self._split_by_anchors(sorted_group, anchor_events)
            result.extend(turns)
        
        return result if result else None
    
    def _diagnose_stitched_trace(self, events: List[Dict[str, Any]], config: Dict[str, Any]) -> bool:
        """
        Diagnose if trace is stitched (multiple conversations improperly combined).
        
        Checks:
        - distinct_user_prompts_per_request_id_max: Max distinct prompts per request_id
        - request_ids_per_user_prompt_max: Max request_ids per user prompt
        - sample_window_events: Number of events to sample for diagnosis
        
        Args:
            events: Normalized events
            config: Request ID diagnosis configuration
            
        Returns:
            True if trace appears to be stitched
        """
        sample_window = config.get("sample_window_events", 5000)
        max_prompts_per_request = config.get("distinct_user_prompts_per_request_id_max", 1)
        max_requests_per_prompt = config.get("request_ids_per_user_prompt_max", 3)
        
        # Sample events if too many
        sampled_events = events[:sample_window] if len(events) > sample_window else events
        
        # Extract USER_INPUT events with request_id
        user_inputs = [
            e for e in sampled_events
            if e.get("kind") == "USER_INPUT" and e.get("request_id")
        ]
        
        if not user_inputs:
            return False
        
        # Count distinct user prompts per request_id
        prompts_per_request: Dict[str, set] = {}
        requests_per_prompt: Dict[str, set] = {}
        
        for event in user_inputs:
            request_id = str(event.get("request_id"))
            # For USER_INPUT, only use text field (not tool_name)
            prompt = event.get("text", "")
            
            # Skip empty prompts
            if not prompt:
                continue
            
            if request_id not in prompts_per_request:
                prompts_per_request[request_id] = set()
            prompts_per_request[request_id].add(prompt)
            
            if prompt not in requests_per_prompt:
                requests_per_prompt[prompt] = set()
            requests_per_prompt[prompt].add(request_id)
        
        # Check if any request_id has too many distinct prompts
        for request_id, prompts in prompts_per_request.items():
            if len(prompts) > max_prompts_per_request:
                return True
        
        # Check if any prompt has too many request_ids
        for prompt, requests in requests_per_prompt.items():
            if len(requests) > max_requests_per_prompt:
                return True
        
        return False
    
    def _split_by_anchors(self, events: List[Dict[str, Any]], anchor_kinds: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Split events into turns using anchor events.
        
        Anchor events (USER_INPUT, MODEL_INVOKE) mark turn boundaries.
        
        Args:
            events: Events to split
            anchor_kinds: List of event kinds that mark turn boundaries
            
        Returns:
            List of event groups (one per turn)
        """
        if not events:
            return []
        
        turns = []
        current_turn = []
        
        for event in events:
            kind = event.get("kind")
            
            # Check if this is an anchor event
            if kind in anchor_kinds:
                # Start new turn if current turn has events
                if current_turn:
                    turns.append(current_turn)
                    current_turn = []
            
            current_turn.append(event)
        
        # Add final turn
        if current_turn:
            turns.append(current_turn)
        
        return turns
    
    def _order_events_within_turn(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Order events within a turn using timestamps and tie-breaker rules.
        
        Ordering logic:
        1. Events with trusted timestamps: sort by timestamp
        2. Events with same/missing timestamps: apply tie_breaker_order by kind
        3. Maintain source_index for events with invalid timestamps
        4. Assign event_order (sequential integer) for deterministic ordering
        
        Args:
            events: Events in a turn
            
        Returns:
            Ordered events with updated event_order field
        """
        if not events:
            return []
        
        tie_breaker_order = self.config.get_tie_breaker_order()
        
        # Create kind priority map (lower number = higher priority)
        kind_priority = {kind: idx for idx, kind in enumerate(tie_breaker_order)}
        
        def sort_key(evt):
            ts_trusted = evt.get("ts_trusted", False)
            start_ts_epoch_ms = evt.get("start_ts_epoch_ms")
            kind = evt.get("kind", "EVENT")
            source_index = evt.get("source_index", 0)
            
            # Get kind priority (default to end of list)
            kind_prio = kind_priority.get(kind, len(tie_breaker_order))
            
            # Sort by:
            # 1. Trusted timestamps first (False sorts before True)
            # 2. Timestamp value (None sorts last)
            # 3. Kind priority (for tie-breaking)
            # 4. Source index (original order)
            return (
                not ts_trusted,  # Trusted first
                start_ts_epoch_ms if start_ts_epoch_ms is not None else float('inf'),
                kind_prio,
                source_index
            )
        
        # Sort events
        ordered = sorted(events, key=sort_key)
        
        # Assign event_order
        for order, event in enumerate(ordered):
            event["event_order"] = order
        
        return ordered
    
    def _create_turn_from_events(
        self,
        events: List[Dict[str, Any]],
        turn_idx: int,
        raw_data: Dict[str, Any],
        strategy_used: str
    ) -> Dict[str, Any]:
        """
        Create a turn object from a group of events.
        
        Args:
            events: Ordered events for this turn
            turn_idx: Turn index (0-based)
            raw_data: Original JSON data
            strategy_used: Segmentation strategy used (e.g., "SINGLE_TURN", "TURN_ID", etc.)
            
        Returns:
            Turn dictionary with all required fields
        """
        turn_id = f"turn_{turn_idx}"
        self.scorer.set_current_turn(turn_id)
        
        # Check for missing trusted timestamps (once per turn)
        # Check both start and end timestamp trust flags
        has_any_trusted_ts = any(
            evt.get("_has_trusted_ts") or evt.get("end_ts_trusted")
            for evt in events
        )
        if not has_any_trusted_ts:
            self.scorer.add_penalty("missing_timestamp", f"{turn_id}.timestamps", scope="turn", turn_id=turn_id)
        
        # Check for missing grouping IDs (once per turn)
        has_any_grouping_id = any(evt.get("_has_grouping_id") for evt in events)
        
        if not has_any_grouping_id:
            self.scorer.add_penalty("missing_grouping_ids", f"{turn_id}.identifiers", scope="turn", turn_id=turn_id)
        
        # Check for anchor events (no_anchor_found penalty)
        anchor_kinds = self.config.get_anchor_events()
        has_anchor = any(evt.get("kind") in anchor_kinds for evt in events)
        if not has_anchor:
            self.scorer.add_penalty("no_anchor_found", f"{turn_id}.anchors", scope="turn", turn_id=turn_id)
        
        # Stage C: Derive - Extract top-level fields with precedence rules
        # CRITICAL CONTRACT (Option A): Only use top-level extraction for SINGLE_TURN strategy
        # For multi-turn traces, derive all fields from events to avoid smearing root values across turns
        
        # Extract user_query
        user_query = None
        if strategy_used == "SINGLE_TURN":
            # Single-turn: allow top-level dotpath extraction (precedence over events)
            user_query = self._extract_top_level_field(raw_data, "user_query")
        
        # Fallback to first USER_INPUT.text in this turn (typically, for both single and multi-turn)
        if not user_query:
            for evt in events:
                if evt.get("kind") == "USER_INPUT" and evt.get("text"):
                    user_query = evt.get("text")
                    break
        
        # Extract final_answer
        final_answer = None
        if strategy_used == "SINGLE_TURN":
            # Single-turn: allow top-level dotpath extraction (precedence over events)
            final_answer = self._extract_top_level_field(raw_data, "final_answer")
        
        # Fallback to LLM output streaming (typically, for both single and multi-turn)
        if not final_answer:
            final_answer = self._join_llm_output_stream(events)
        
        # Extract finish_reason
        finish_reason = None
        if strategy_used == "SINGLE_TURN":
            # Single-turn: allow top-level dotpath extraction
            finish_reason = self._extract_top_level_field(raw_data, "finish_reason")
        
        # Check for LLM output (stricter heuristic)
        has_llm_output = bool(final_answer) or any(
            evt.get("kind") in ["LLM_OUTPUT_CHUNK", "ASSISTANT_MESSAGE", "FINAL_RESPONSE"] 
            for evt in events
        )
        if not has_llm_output:
            self.scorer.add_penalty("no_llm_output", f"{turn_id}.output", scope="turn", turn_id=turn_id)
        
        # Stage C: Derive - Strip prompt context before creating steps
        events = self._strip_prompt_context(events)
        
        # Stage C: Derive - Link tool calls with results and handle orphans
        events = self._link_tool_calls_and_results(events, turn_id)
        
        # Stage C: Derive - Enrich tool call status from linked results
        events = self._enrich_tool_call_status(events)
        
        # Stage C: Derive - Classify events into phases
        events = self._classify_phases(events)
        
        # Convert events to steps
        steps = []
        for event in events:
            step = self._event_to_step(event)
            steps.append(step)
        
        # Calculate latency (Stage C: Derive)
        normalized_latency = self._calculate_normalized_latency(events)
        runtime_latency = self._extract_runtime_latency(raw_data)
        
        # Apply missing_latency penalty per config policy (on_missing_timestamps: "null_and_penalize")
        # This penalty applies when timestamps are missing OR when they exist but latency can't be computed
        latency_config = self.config.get_latency_config()
        on_missing_policy = latency_config.get("on_missing_timestamps", "null_and_penalize")
        
        # Only apply missing_latency penalty when timestamps exist but latency can't be computed
        # This avoids double-penalizing with missing_timestamp
        if on_missing_policy == "null_and_penalize" and normalized_latency is None:
            # Count both start and end trusted timestamps
            trusted_ts_count = sum(
                1 for evt in events
                if (evt.get("_has_trusted_ts") or evt.get("end_ts_trusted"))
            )
            
            # Only penalize if we have trusted timestamps but still couldn't compute latency
            if trusted_ts_count >= 2:
                self.scorer.add_penalty("missing_latency", f"{turn_id}.latency", scope="turn", turn_id=turn_id)
                self.warnings.append(
                    f"{turn_id}: {trusted_ts_count} trusted timestamps present but latency could not be computed"
                )
        
        # Calculate confidence
        confidence = self.scorer.calculate_turn_confidence(turn_id)
        
        # Extract request_id from first event (schema-required field)
        request_id = None
        for evt in events:
            if evt.get("request_id"):
                request_id = evt.get("request_id")
                break
        
        # Extract timestamp from first trusted timestamp (schema-required field)
        timestamp = None
        for evt in events:
            if evt.get("_has_trusted_ts") and evt.get("start_ts"):
                timestamp = evt.get("start_ts")
                break
        
        # Calculate total_latency_ms (prefers normalized, falls back to runtime)
        total_latency_ms = normalized_latency if normalized_latency is not None else runtime_latency
        
        # Build turn object - schema compliant (no attribution, no fields_source)
        turn = {
            "turn_id": turn_id,
            "request_id": request_id,
            "timestamp": timestamp,
            "user_query": str(user_query) if user_query else None,
            "final_answer": str(final_answer) if final_answer else None,
            "steps": steps,
            "normalized_latency_ms": normalized_latency,
            "runtime_reported_latency_ms": runtime_latency,
            "total_latency_ms": total_latency_ms,
            "confidence": confidence
        }
        
        # Add finish_reason if available
        if finish_reason:
            turn["finish_reason"] = finish_reason
        
        return turn
    
    def _extract_top_level_field(self, data: Dict[str, Any], field_name: str) -> Optional[str]:
        """
        Extract field from top-level JSON using dotted-path syntax.
        
        This implements the required contract for top-level field extraction
        with precedence over event-based extraction.
        
        CRITICAL: This method guarantees dotted-path traversal on the root JSON
        by using the adapter's own _extract_from_path() method, which supports
        dotted paths like "trace.final_answer", "response.user_query", etc.
        
        Args:
            data: Source JSON dictionary (root level)
            field_name: Field to extract (e.g., "final_answer", "user_query", "finish_reason")
            
        Returns:
            Extracted value as string or None
            
        Behavior:
            - If value is a list: returns first non-empty scalar (strict mode for Phase 1)
            - If value is a dict: returns None (can't stringify complex objects)
            - If value is scalar: returns string representation
        """
        # Get configured aliases for this field (e.g., ["final_answer", "trace.final_answer", "response.final_answer"])
        aliases = self.config.get_field_aliases(field_name)
        
        # If no aliases configured, try the field name itself
        if not aliases:
            aliases = [field_name]
        
        # Try each alias in order using dotted-path traversal
        for alias in aliases:
            # Use adapter's own _extract_from_path() which supports dotted paths
            value = self._extract_from_path(data, alias)
            
            if value is None:
                continue
            
            # Handle list values: pick first non-empty scalar (strict mode)
            if isinstance(value, list):
                for item in value:
                    if item and not isinstance(item, (dict, list)):
                        return str(item)
                # All items were empty, dicts, or lists - treat as None
                continue
            
            # Handle dict values: can't stringify complex objects
            if isinstance(value, dict):
                continue
            
            # Scalar value: return as string
            return str(value)
        
        return None
    
    def _detect_attribution(self, events: List[Dict[str, Any]], turn_id: str) -> Dict[str, Any]:
        """
        Detect attribution: tool usage, tool output, and stitched trace suspects.
        
        **NOTE: This method is currently unused and reserved for future implementation.**
        Attribution detection was planned but not yet integrated into the output schema.
        The method remains for potential future use in enhanced trace analysis.
        
        Stage C: Derive implementation with:
        - Tool usage detection: tool_used_if_has_kind=TOOL_CALL
        - Tool output detection by regex patterns
        - Stitched trace detection by question patterns (USER_INPUT events only)
        
        Args:
            events: List of normalized events
            turn_id: Turn identifier
            
        Returns:
            Attribution dictionary with verdicts
        """
        attribution_config = self.config.get_attribution_config()
        verdicts_config = attribution_config.get("verdicts", {})
        stitch_config = attribution_config.get("stitch_suspect", {})
        
        attribution = {}
        
        # Detect tool usage
        tool_used_kind = verdicts_config.get("tool_used_if_has_kind", "TOOL_CALL")
        has_tool_usage = any(event.get("kind") == tool_used_kind for event in events)
        attribution["tool_used"] = has_tool_usage
        
        # Detect tool output by regex patterns
        tool_output_patterns = verdicts_config.get("tool_output_only_if_text_matches_regex", [])
        compiled_patterns = []
        for pattern in tool_output_patterns:
            try:
                compiled_patterns.append(re.compile(pattern))
            except re.error:
                continue
        
        has_tool_output = False
        for event in events:
            text = event.get("text")
            if text:
                for pattern in compiled_patterns:
                    if pattern.search(str(text)):
                        has_tool_output = True
                        break
            if has_tool_output:
                break
        
        attribution["tool_output_detected"] = has_tool_output
        
        # Detect stitched trace suspects (USER_INPUT events only to avoid false positives)
        if stitch_config.get("enabled", True):
            question_pattern = stitch_config.get("question_line_regex")
            suspect_threshold = stitch_config.get("distinct_question_count_suspect_at", 2)
            
            if question_pattern:
                try:
                    compiled_question = re.compile(question_pattern, re.MULTILINE)
                    
                    # Extract distinct questions from USER_INPUT events only
                    questions = set()
                    for event in events:
                        # Only check USER_INPUT events to avoid false positives from assistant outputs
                        if event.get("kind") == "USER_INPUT":
                            text = event.get("text")
                            if text:
                                matches = compiled_question.findall(str(text))
                                questions.update(matches)
                    
                    is_stitched_suspect = len(questions) >= suspect_threshold
                    attribution["stitched_trace_suspect"] = is_stitched_suspect
                    
                    if is_stitched_suspect:
                        logger.warning(f"Detected stitched trace suspect in {turn_id}: {len(questions)} distinct questions found (threshold: {suspect_threshold})")
                        attribution["distinct_question_count"] = len(questions)
                
                except re.error:
                    pass
        
        return attribution
    
    def _classify_phases(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify events into phases based on tool usage patterns.
        
        Phases:
        - PRE_TOOL_GENERATION: Before first tool call
        - TOOL_CALL: Tool execution phase
        - FINAL_GENERATION: After last tool call
        
        Args:
            events: List of normalized events
            
        Returns:
            Events with phase field added
        """
        phase_config = self.config.get_phases_config()
        pre_tool_phase = phase_config.get("pre_tool", "PRE_TOOL_GENERATION")
        tool_phase = phase_config.get("tool_call", "TOOL_CALL")
        post_tool_phase = phase_config.get("post_tool", "FINAL_GENERATION")
        
        # Find first and last tool call indices
        first_tool_idx = None
        last_tool_idx = None
        
        for idx, event in enumerate(events):
            if event.get("kind") == "TOOL_CALL":
                if first_tool_idx is None:
                    first_tool_idx = idx
                last_tool_idx = idx
        
        # Classify phases
        for idx, event in enumerate(events):
            if first_tool_idx is None:
                # No tool calls - all events are in final generation
                event["phase"] = post_tool_phase
            elif idx < first_tool_idx:
                event["phase"] = pre_tool_phase
            elif idx <= last_tool_idx:
                event["phase"] = tool_phase
            else:
                event["phase"] = post_tool_phase
        
        return events
    
    def _link_tool_calls_and_results(self, events: List[Dict[str, Any]], turn_id: str) -> List[Dict[str, Any]]:
        """
        Link tool calls with results and handle orphan tool results.
        
        This implements Stage C: Derive - tool linking logic with:
        - TOOL_RUN_ID strategy: Match by tool_run_id
        - SPAN_PARENT_CHILD strategy: Match by span hierarchy
        - Deduplication within time windows
        - Orphan tool result detection with confidence penalty
        
        IMPORTANT: This method adds linking metadata to events IN-PLACE
        and returns the same list (preserving chronological order).
        
        Args:
            events: List of normalized events
            turn_id: Turn identifier for penalty tracking
            
        Returns:
            Same list of events with tool linking metadata added (order preserved)
        """
        tool_config = self.config.get_tool_linking_config()
        
        # Identify tool runs (kind=TOOL_CALL + tool_name required)
        tool_call_indices = []
        tool_result_indices = []
        
        for idx, event in enumerate(events):
            kind = event.get("kind")
            tool_name = event.get("tool_name")
            
            # Check if this is a valid tool run
            if kind == "TOOL_CALL" and tool_name:
                tool_call_indices.append(idx)
            elif kind == "TOOL_RESULT":
                tool_result_indices.append(idx)
        
        # Deduplicate tool calls within time windows (modifies events in-place)
        dedupe_config = tool_config.get("dedupe", {})
        if dedupe_config.get("enabled", True):
            events = self._deduplicate_tool_calls_inplace(events, tool_call_indices, dedupe_config)
            # Rebuild indices after deduplication
            tool_call_indices = [idx for idx, evt in enumerate(events) if evt.get("kind") == "TOOL_CALL" and evt.get("tool_name")]
            tool_result_indices = [idx for idx, evt in enumerate(events) if evt.get("kind") == "TOOL_RESULT"]
        
        # Link tool calls with results
        link_strategies = tool_config.get("link_results_by", ["TOOL_RUN_ID", "SPAN_PARENT_CHILD"])
        linked_result_indices = set()
        
        for strategy in link_strategies:
            if strategy == "TOOL_RUN_ID":
                linked_result_indices.update(self._link_by_tool_run_id(events, tool_call_indices, tool_result_indices))
            elif strategy == "SPAN_PARENT_CHILD":
                linked_result_indices.update(self._link_by_span_hierarchy(events, tool_call_indices, tool_result_indices))
        
        # Identify orphan tool results with schema-compliant format (location field required)
        orphans = []
        for idx in tool_result_indices:
            if idx not in linked_result_indices:
                result = events[idx]
                # Schema requires "location" field for orphan_tool_results
                # Use turn_id from function parameter and source_index for precise location
                source_idx = result.get('source_index', idx)
                location = f"{turn_id}.step_{source_idx}"
                orphan = {
                    "location": location
                }
                # Add optional fields if available
                if result.get("tool_name"):
                    orphan["tool_name"] = result.get("tool_name")
                if result.get("tool_run_id"):
                    orphan["tool_run_id"] = str(result.get("tool_run_id"))
                orphans.append(orphan)
        
        # Handle orphan tool results with confidence penalty
        if orphans:
            logger.warning(f"Found {len(orphans)} orphan tool result(s) in {turn_id} - tool results without corresponding tool calls")
            self.orphan_tool_results.extend(orphans)
            self.scorer.add_penalty("orphan_tool_results", f"{turn_id}.tool_linking", scope="turn", turn_id=turn_id)
        
        # Return events with linking metadata added (order preserved)
        return events
    def _infer_status_from_result(self, result: Dict[str, Any]) -> str:
        """
        Infer execution status from a tool result event.

        Checks for error signals in strict precedence order to determine
        if a tool execution succeeded or failed.

        Args:
            result: Tool result event dictionary

        Returns:
            Inferred status: "error" if error signals detected, "success" otherwise
        """
        # Check 1: Explicit status field with error values
        status = result.get("status")
        if status:
            status_str = str(status).lower()
            if status_str in ["error", "failed", "failure", "exception", "timeout"]:
                return "error"

        # Check 2: Error field exists and is non-empty
        error_field = result.get("error")
        if error_field:
            # Check if it's a non-empty string or non-empty dict/list
            if isinstance(error_field, str) and error_field.strip():
                return "error"
            elif isinstance(error_field, (dict, list)) and error_field:
                return "error"

        # Check 3: attributes.error exists and is non-empty
        attributes = result.get("attributes", {})
        if isinstance(attributes, dict):
            attr_error = attributes.get("error")
            if attr_error:
                if isinstance(attr_error, str) and attr_error.strip():
                    return "error"
                elif isinstance(attr_error, (dict, list)) and attr_error:
                    return "error"

        # Check 4: raw contains error-shaped fields
        raw = result.get("raw", {})
        if isinstance(raw, dict):
            # Check for common error field names
            error_keys = ["error", "exception", "failure", "error_message", "error_code"]
            for key in error_keys:
                if key in raw and raw[key]:
                    # Non-empty value indicates error
                    if isinstance(raw[key], str) and raw[key].strip():
                        return "error"
                    elif isinstance(raw[key], (dict, list)) and raw[key]:
                        return "error"

        # Check 5: Cautious text heuristics (last resort)
        # Only apply to text field with anchored patterns to avoid false positives
        text = result.get("text", "")
        if isinstance(text, str) and text:
            # Anchored patterns to avoid false positives like "No error found"
            error_patterns = [
                r'\berror\b.*\boccurred\b',  # "error occurred"
                r'\bfailed\b.*\bto\b',        # "failed to"
                r'\bexception\b.*\braised\b', # "exception raised"
                r'\btimeout\b.*\bexceeded\b', # "timeout exceeded"
            ]

            for pattern in error_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return "error"

        # No error signals detected - assume success
        return "success"

    def _enrich_tool_call_status(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich tool call events with inferred status from linked results.

        This implements the status enrichment phase that infers execution outcomes
        from linked tool results. It follows precedence rules:
        - Explicit status (not null, not "unknown") → preserved
        - Linked result exists → infer from result
        - No linked result → remains "unknown"

        ENHANCEMENTS (Gap fixes):
        - Gap 1: Supports both TOOL_RUN_ID and SPAN_PARENT_CHILD linking strategies
          by using _linked_to_call_idx metadata written by the linker
        - Gap 2: Case-insensitive "unknown" check (handles UNKNOWN, Unknown, etc.)
        - Gap 5: Deterministic preference for multiple results (first result, or error if present)

        Args:
            events: List of events with tool linking metadata from _link_tool_calls_and_results()

        Returns:
            Same list of events with enriched status fields (modified in-place)
        """
        # Build bidirectional index using _linked_to_call_idx metadata (Gap 1 fix)
        # This supports both TOOL_RUN_ID and SPAN_PARENT_CHILD linking strategies
        # because the linker already resolved the relationship and wrote _linked_to_call_idx
        call_to_result = {}  # call_idx -> result_event
        
        for idx, event in enumerate(events):
            if event.get("kind") == "TOOL_RESULT":
                # Use the _linked_to_call_idx metadata written by the linker
                call_idx = event.get("_linked_to_call_idx")
                if call_idx is None:
                    continue
                
                # Gap 5 fix: Deterministic preference for multiple results
                if call_idx not in call_to_result:
                    # First result for this call
                    call_to_result[call_idx] = event
                else:
                    # Multiple results - prefer error over success
                    existing = call_to_result[call_idx]
                    new_status = self._infer_status_from_result(event)
                    existing_status = self._infer_status_from_result(existing)
                    if new_status == "error" and existing_status != "error":
                        call_to_result[call_idx] = event
        
        # Enrich tool call events using the link index
        for idx, event in enumerate(events):
            if event.get("kind") == "TOOL_CALL":
                # Check if this tool call needs status enrichment
                current_status = event.get("status")

                # Gap 2 fix: Case-insensitive "unknown" check
                # Only enrich if status is missing or "unknown" (any case)
                if current_status and str(current_status).strip().lower() != "unknown":
                    continue

                # Find linked result using the index (supports all linking strategies)
                linked_result = call_to_result.get(idx)
                if not linked_result:
                    continue

                # Infer status from linked result
                inferred_status = self._infer_status_from_result(linked_result)

                # Update event with inferred status
                event["status"] = inferred_status

                # Add debug metadata
                event["_tool_outcome_inferred"] = True

        return events

    
    def _deduplicate_tool_calls_inplace(
        self,
        events: List[Dict[str, Any]],
        tool_call_indices: List[int],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate tool calls within time windows (in-place, preserves order).
        
        Args:
            events: Full list of events
            tool_call_indices: Indices of tool call events
            config: Deduplication configuration
            
        Returns:
            Events list with duplicates marked for removal
        """
        window_seconds = config.get("window_seconds", 2)
        key_fields = config.get("key_fields", ["tool_run_id", "tool_name"])
        window_ms = window_seconds * 1000
        
        # Group tool calls by key
        groups: Dict[Tuple, List[int]] = {}
        
        for idx in tool_call_indices:
            call = events[idx]
            # Build key from key_fields
            key_values = []
            has_any_key = False
            for field in key_fields:
                value = call.get(field)
                if value:
                    has_any_key = True
                key_values.append(str(value) if value else None)
            
            # If all key fields are None, add source_index as tiebreaker to avoid false dedupe
            if not has_any_key:
                key_values.append(call.get("source_index"))
            
            key = tuple(key_values)
            
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)
        
        # Mark duplicates for removal
        indices_to_remove = set()
        
        for key, indices in groups.items():
            if len(indices) == 1:
                continue
            
            # Sort by timestamp
            sorted_indices = sorted(
                indices,
                key=lambda i: events[i].get("start_ts_epoch_ms") if events[i].get("start_ts_epoch_ms") is not None else float('inf')
            )
            
            # Keep first call and any calls outside time window
            kept_idx = sorted_indices[0]
            
            for idx in sorted_indices[1:]:
                last_ts = events[kept_idx].get("start_ts_epoch_ms")
                curr_ts = events[idx].get("start_ts_epoch_ms")
                
                # Keep if outside window or no timestamps
                if last_ts is None or curr_ts is None or (curr_ts - last_ts) > window_ms:
                    kept_idx = idx
                else:
                    # Within window - prefer richer fields
                    if self._is_richer_event(events[idx], events[kept_idx]):
                        indices_to_remove.add(kept_idx)
                        kept_idx = idx
                    else:
                        indices_to_remove.add(idx)
        
        # Remove duplicates while preserving order
        return [evt for idx, evt in enumerate(events) if idx not in indices_to_remove]
    
    def _is_richer_event(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> bool:
        """
        Check if event1 has richer fields than event2.
        
        Prefers events with more of: tool_arguments, tool_input, attributes.
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            True if event1 is richer
        """
        rich_fields = ["tool_arguments", "tool_input", "attributes"]
        
        score1 = sum(1 for field in rich_fields if event1.get(field))
        score2 = sum(1 for field in rich_fields if event2.get(field))
        
        return score1 > score2
    
    def _link_by_tool_run_id(
        self,
        events: List[Dict[str, Any]],
        tool_call_indices: List[int],
        tool_result_indices: List[int]
    ) -> set:
        """
        Link tool calls with results via tool_run_id.
        
        Args:
            events: Full list of events
            tool_call_indices: Indices of tool call events
            tool_result_indices: Indices of tool result events
            
        Returns:
            Set of linked result indices
        """
        linked = set()
        
        # Build map of tool_run_id to call index
        call_map = {}
        for idx in tool_call_indices:
            call = events[idx]
            tool_run_id = call.get("tool_run_id")
            if tool_run_id:
                call_map[str(tool_run_id)] = idx
        
        # Match results to calls
        for result_idx in tool_result_indices:
            result = events[result_idx]
            tool_run_id = result.get("tool_run_id")
            if tool_run_id and str(tool_run_id) in call_map:
                linked.add(result_idx)
                # Add linking metadata
                result["_linked_by"] = "TOOL_RUN_ID"
                call_idx = call_map[str(tool_run_id)]
                result["_linked_to_call_idx"] = call_idx
                
                # FIX: Propagate tool_name from call to result if result is missing it
                call = events[call_idx]
                if not result.get("tool_name") and call.get("tool_name"):
                    result["tool_name"] = call.get("tool_name")
        
        return linked
    
    def _link_by_span_hierarchy(
        self,
        events: List[Dict[str, Any]],
        tool_call_indices: List[int],
        tool_result_indices: List[int]
    ) -> set:
        """
        Link tool calls with results via span hierarchy (parent_span_id).
        
        Args:
            events: Full list of events
            tool_call_indices: Indices of tool call events
            tool_result_indices: Indices of tool result events
            
        Returns:
            Set of linked result indices
        """
        linked = set()
        
        # Build map of span_id to call index
        call_map = {}
        for idx in tool_call_indices:
            call = events[idx]
            span_id = call.get("span_id")
            if span_id:
                call_map[str(span_id)] = idx
        
        # Match results to calls by parent_span_id
        for result_idx in tool_result_indices:
            result = events[result_idx]
            parent_span_id = result.get("parent_span_id")
            if parent_span_id and str(parent_span_id) in call_map:
                linked.add(result_idx)
                # Add linking metadata
                result["_linked_by"] = "SPAN_PARENT_CHILD"
                call_idx = call_map[str(parent_span_id)]
                result["_linked_to_call_idx"] = call_idx
                
                # FIX: Propagate tool_name from call to result if result is missing it
                call = events[call_idx]
                if not result.get("tool_name") and call.get("tool_name"):
                    result["tool_name"] = call.get("tool_name")
        
        return linked
    
    def _strip_prompt_context(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Strip events with kind PROMPT_CONTEXT and text matching regex patterns.
        
        This implements Stage C: Derive - prompt context stripping logic.
        
        Args:
            events: List of normalized events
            
        Returns:
            Filtered list of events with prompt context removed
        """
        prompt_config = self.config.get_prompt_context_strip_config()
        strip_kinds = prompt_config.get("strip_kinds", ["PROMPT_CONTEXT"])
        strip_patterns = prompt_config.get("strip_text_regex", [])
        
        # Compile strip patterns
        compiled_patterns = []
        for pattern in strip_patterns:
            try:
                compiled_patterns.append(re.compile(pattern))
            except re.error:
                continue
        
        filtered = []
        for event in events:
            # Check if kind should be stripped
            if event.get("kind") in strip_kinds:
                continue
            
            # Check if text matches strip patterns
            text = event.get("text")
            if text:
                should_strip = False
                for pattern in compiled_patterns:
                    if pattern.search(str(text)):
                        should_strip = True
                        break
                
                if should_strip:
                    continue
            
            filtered.append(event)
        
        return filtered
    
    def _join_llm_output_stream(self, events: List[Dict[str, Any]]) -> Optional[str]:
        """
        Join LLM_OUTPUT_CHUNK events into final_answer (fallback if top-level missing).
        
        This implements Stage C: Derive - LLM output streaming logic.
        
        Args:
            events: List of normalized events
            
        Returns:
            Joined LLM output or None
        """
        output_config = self.config.get_output_extraction_config()
        stream_config = output_config.get("assistant_output_stream", {})
        
        include_kinds = stream_config.get("include_kinds", ["LLM_OUTPUT_CHUNK", "ASSISTANT_MESSAGE", "FINAL_RESPONSE"])
        exclude_patterns = stream_config.get("exclude_if_text_matches_regex", [])
        join_with = stream_config.get("join_with", "")  # FIX: Read from stream_config, not parent
        max_chars = output_config.get("max_chars", 200000)
        
        # Compile exclude patterns
        compiled_exclude = []
        for pattern in exclude_patterns:
            try:
                compiled_exclude.append(re.compile(pattern))
            except re.error:
                continue
        
        # Collect chunks
        chunks = []
        total_chars = 0
        
        for event in events:
            kind = event.get("kind")
            text = event.get("text")
            
            # Check if this event should be included
            if kind not in include_kinds or not text:
                continue
            
            # Check if text matches exclude patterns
            should_exclude = False
            for pattern in compiled_exclude:
                if pattern.search(str(text)):
                    should_exclude = True
                    break
            
            if should_exclude:
                continue
            
            # Add chunk
            text_str = str(text)
            if total_chars + len(text_str) > max_chars:
                # Truncate to max_chars
                remaining = max_chars - total_chars
                if remaining > 0:
                    chunks.append(text_str[:remaining])
                break
            
            chunks.append(text_str)
            total_chars += len(text_str)
        
        if not chunks:
            return None
        
        return join_with.join(chunks)
    
    def _strip_internal_fields(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strip internal metadata fields (starting with _) from object.
        
        This prevents internal tracking fields from leaking into output.
        Used in Stage B/C when emitting enriched events.
        
        Args:
            obj: Dictionary potentially containing internal fields
            
        Returns:
            Dictionary with internal fields removed
        """
        return {k: v for k, v in obj.items() if not k.startswith("_")}
    
    def _json_safe(self, value: Any) -> Any:
        """
        helps value is JSON-serializable.
        
        Attempts to serialize and returns the value if successful,
        otherwise converts to string representation.
        
        Args:
            value: Value to make JSON-safe
            
        Returns:
            JSON-serializable value
        """
        if value is None:
            return None
        
        try:
            # Test if serializable
            json.dumps(value, default=str)
            return value
        except (TypeError, ValueError):
            # Fallback to string representation
            return str(value)
    
    def _event_to_step(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert normalized event to step format.
        
        Strips internal metadata fields (_has_*, _*) and helps JSON-safe output.
        Schema compliance: includes event_order, source_index, bounded raw, normalized status.
        
        Args:
            event: Normalized event
            
        Returns:
            Step dictionary with JSON-safe fields compliant with schema
        """
        # Prefer tool_name, then event_type, then operation for better readability
        name = event.get("tool_name") or event.get("event_type") or event.get("operation") or "unknown"
        
        # Normalize status to schema enum: success/error/unknown/skipped
        raw_status = event.get("status")
        if not raw_status:
            status = "unknown"  # Default to unknown when status is missing
        else:
            status_str = str(raw_status).lower()
            
            # Map to success (only with explicit success signals)
            if status_str in ["success", "ok", "completed", "complete", "200", "done"]:
                status = "success"
            # Map to error
            elif status_str in ["error", "failed", "failure", "exception", "timeout", "500", "false"]:
                status = "error"
            # Map to skipped
            elif status_str in ["skipped", "skip", "ignored", "cancelled", "canceled"]:
                status = "skipped"
            # Default to unknown for ambiguous values (safer than assuming success)
            else:
                status = "unknown"
        
        # helps attributes are JSON-safe (dict with primitive values)
        # Also move kind_rule_id and kind_reason to attributes (schema doesn't allow them at top level)
        attributes = event.get("attributes")
        if attributes is not None:
            if isinstance(attributes, dict):
                # Make each attribute value JSON-safe
                attributes = {k: self._json_safe(v) for k, v in attributes.items()}
            else:
                # Not a dict - create new dict
                attributes = {}
        else:
            attributes = {}
        
        # Add kind metadata to attributes if present (schema doesn't allow at step top-level)
        if event.get("kind_rule_id"):
            attributes["_kind_rule_id"] = event.get("kind_rule_id")
        if event.get("kind_reason"):
            attributes["_kind_reason"] = event.get("kind_reason")
        
        # If attributes is empty, set to None
        if not attributes:
            attributes = None
        
        # Get bounded raw event (up to max_bytes limit)
        raw = event.get("raw")
        if raw is not None and isinstance(raw, dict):
            # helps raw is JSON-safe and bounded
            raw = {k: self._json_safe(v) for k, v in raw.items()}
        
        step = {
            "name": name,
            "status": status,
            "type": event.get("event_type"),
            "kind": event.get("kind"),
            "start_ts": event.get("start_ts"),
            "end_ts": event.get("end_ts"),
            "latency_ms": event.get("latency_ms"),
            "span_id": event.get("span_id"),
            "parent_span_id": event.get("parent_span_id"),
            "tool_name": event.get("tool_name"),  # Preserve tool_name for validation
            "tool_run_id": event.get("tool_run_id"),
            "event_order": event.get("event_order"),
            "source_index": event.get("source_index"),
            "attributes": attributes,
            "raw": raw
        }
        
        return step
    
    def _calculate_normalized_latency(self, events: List[Dict[str, Any]]) -> Optional[float]:
        """
        Calculate normalized latency from trusted timestamps using epoch ms.
        
        Stage C: Derive implementation with:
        - Start from first event with kind in: USER_INPUT, MODEL_INVOKE, LLM_OUTPUT_CHUNK, TOOL_CALL
        - End at last event with kind in: LLM_OUTPUT_CHUNK, TOOL_RESULT, EVENT
        - Only uses trusted timestamps
        
        Args:
            events: List of normalized events (should be sorted)
            
        Returns:
            Latency in milliseconds or None
        """
        latency_config = self.config.get_latency_config()
        start_kinds = latency_config.get("normalized_latency_ms", {}).get(
            "start_from_first_kind_in",
            ["USER_INPUT", "MODEL_INVOKE", "LLM_OUTPUT_CHUNK", "TOOL_CALL"]
        )
        end_kinds = latency_config.get("normalized_latency_ms", {}).get(
            "end_at_last_kind_in",
            ["LLM_OUTPUT_CHUNK", "ASSISTANT_MESSAGE", "FINAL_RESPONSE", "TOOL_RESULT", "EVENT"]
        )
        
        # Find first event with start kind and trusted timestamp
        start_ts = None
        for event in events:
            kind = event.get("kind")
            if kind in start_kinds and event.get("ts_trusted") and event.get("start_ts_epoch_ms") is not None:
                start_ts = event["start_ts_epoch_ms"]
                break
        
        # Find last event with end kind and trusted timestamp
        end_ts = None
        for event in reversed(events):
            kind = event.get("kind")
            # Check both start and end timestamps for end calculation
            if kind in end_kinds:
                if event.get("end_ts_trusted") and event.get("end_ts_epoch_ms") is not None:
                    end_ts = event["end_ts_epoch_ms"]
                    break
                elif event.get("ts_trusted") and event.get("start_ts_epoch_ms") is not None:
                    end_ts = event["start_ts_epoch_ms"]
                    break
        
        # Calculate latency if both timestamps found
        if start_ts is not None and end_ts is not None:
            return float(end_ts - start_ts)
        
        # Return None if timestamps missing (caller will apply penalty per policy)
        return None
    
    def _extract_runtime_latency(self, raw_data: Dict[str, Any]) -> Optional[float]:
        """
        Extract runtime-reported latency from configured fields.
        
        Stage C: Derive implementation.
        
        Args:
            raw_data: Original JSON data
            
        Returns:
            Runtime-reported latency in milliseconds or None
        """
        latency_config = self.config.get_latency_config()
        fields = latency_config.get("keep_runtime_reported_latency_ms_fields", ["total_latency_ms", "attributes.latency_ms"])
        
        for field in fields:
            value = self.config.get_field_value_with_fallback(raw_data, field)
            if value is not None:
                sanitized = sanitize_latency(value)
                if sanitized is not None:
                    return float(sanitized)
        
        return None
    
    def _validate_output(self, normalized: Dict[str, Any]) -> None:
        """
        Validate output against normalized schema.
        
        Args:
            normalized: Normalized trace dictionary
            
        Raises:
            ValidationError: If schema file can't be loaded or validation fails
        """
        # Try to load and validate against schema
        schema_path = Path(__file__).parent.parent.parent / "schemas" / "normalized_run.schema.json"
        
        if not schema_path.exists():
            logger.error(f"Schema file not found at: {schema_path}")
            raise ValidationError(
                "Schema file not found. Cannot validate output.",
                schema_path=str(schema_path)
            )
        
        try:
            schema = load_schema(str(schema_path))
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to load schema file: {e}")
            raise ValidationError(
                "Failed to load schema file.",
                schema_path=str(schema_path)
            ) from e
        
        try:
            is_valid, errors = validate_against_schema(normalized, schema)
            
            if not is_valid:
                logger.error(f"Schema validation failed with {len(errors)} error(s): {errors[:3]}")  # Log first 3 errors
                raise ValidationError(
                    "Output does not conform to normalized schema.",
                    validation_errors=errors,
                    schema_path=str(schema_path)
                )
        except ImportError as e:
            # jsonschema library not installed - emit warning but don't fail
            if not self._schema_warning_emitted:
                logger.warning("jsonschema library not installed. Skipping validation.")
                self.warnings.append("jsonschema library not installed. Skipping validation.")
                self._schema_warning_emitted = True
