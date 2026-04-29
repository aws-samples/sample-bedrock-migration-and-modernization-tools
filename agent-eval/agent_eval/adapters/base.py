# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Base adapter interface for trace normalization.

This module defines the Protocol interface that all trace adapters must implement
to helps consistency across different trace format adapters.
"""

from typing import Protocol, Mapping, Any, Optional, Union
import os


class Adapter(Protocol):
    """
    Protocol defining the interface for trace adapters.
    
    All adapters should implement this interface to helps consistency
    across different trace format adapters.
    
    This uses Python's Protocol (PEP 544) for structural subtyping, which means
    any class that implements the required methods will be considered compatible
    with this interface without explicit inheritance.
    """
    
    def adapt(self, path: Union[str, os.PathLike], config_path: Optional[str] = None) -> Mapping[str, Any]:
        """
        Load a trace file and normalize it to the standard schema.
        
        This is the primary interface method that all adapters must implement.
        It takes a trace file path and optional configuration, and returns a
        normalized trace structure conforming to the normalized schema.
        
        Args:
            path: Path to the trace file to be normalized (str or PathLike)
            config_path: Optional path to adapter-specific configuration file.
                        If None, the adapter should use its default configuration.
        
        Returns:
            Mapping conforming to normalized schema with the following structure:
            - run_id (str): Unique identifier for this agent execution
            - metadata (dict): Run-level metadata (required fields):
                - adapter_version (str): Version of the adapter used (required)
                - processed_at (str): ISO 8601 timestamp of processing (required)
                - source (str): Source system identifier (optional)
                - segmentation_strategy_used (str): Strategy used for turn segmentation (optional)
            - adapter_stats (dict): Adapter processing statistics including:
                - total_events_processed (int): Total number of events processed
                - events_with_valid_timestamps (int): Events with valid timestamps
                - events_with_missing_data (int): Events with missing optional data
                - confidence_penalties (list): List of confidence penalty records with:
                    - reason (str): Reason for penalty
                    - penalty (float): Penalty value
                    - location (str): Where the penalty occurred
                - turn_count (int): Total number of turns segmented
                - raw_path (str): Which event_path matched in source
                - canonical_sources (dict): Which field aliases matched per field
                - orphan_tool_results (list): Tool results without corresponding calls
            - turns (list): Array of conversation turns, each containing:
                - turn_id (str): Unique identifier for this turn
                - user_query (str|None): The user query for this turn (nullable)
                - final_answer (str|None): The final response for this turn (nullable)
                - steps (list): Ordered list of execution steps
                - confidence (float): Confidence score (0-1) for data quality
                - normalized_latency_ms (float|None): Latency calculated from timestamps
                - runtime_reported_latency_ms (float|None): Latency reported by source
                - request_id (str|None): Optional request identifier
                - timestamp (str|None): ISO 8601 timestamp of turn start
        
        Raises:
            FileNotFoundError: If the trace file doesn't exist
            ValueError: If JSON is invalid, unreadable, or contains no events
            
        Note:
            For missing optional fields, the adapter should emit null values 
            with confidence penalties instead of raising exceptions. Hard failures
            should only occur for completely unreadable input or when no events exist.
        
        Behavior:
            - Graceful degradation: missing optional fields → null with confidence penalty
            - Config-driven field mapping from adapter configuration
            - Dual latency tracking (normalized and runtime-reported)
            - Multi-turn conversation support with turn segmentation
            - Orphan tool results handled with confidence penalties
            - Tool-looking text without markers not misclassified as tool calls
        
        Example:
            >>> from agent_eval.adapters.generic_json import GenericJsonAdapter
            >>> adapter = GenericJsonAdapter()
            >>> result = adapter.adapt("trace.json")
            >>> print(result["run_id"])
            'abc-123-def'
            >>> print(result["turns"][0]["confidence"])
            0.85
            >>> print(result["metadata"]["segmentation_strategy_used"])
            'TURN_ID'
        """
        ...
