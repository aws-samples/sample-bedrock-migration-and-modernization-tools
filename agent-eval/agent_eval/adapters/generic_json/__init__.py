# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Generic JSON adapter for trace normalization.

This module provides the public API for transforming Generic JSON trace files
into the normalized schema format. It uses a config-driven approach with
adapter_config.yaml for field mappings and classification rules.

Public API:
    adapt(path, config_path=None) -> dict
        Transform a Generic JSON trace file into normalized format.
        
    DEFAULT_CONFIG_PATH: Path
        Default path to adapter_config.yaml (works regardless of CWD).
    
    Exception Classes:
        InputError: File not found, invalid JSON, or unreadable input
        ValidationError: Schema validation failure or no events exist
        AdaptationError: Internal adapter logic errors

Example:
    >>> from agent_eval.adapters.generic_json import adapt
    >>> result = adapt("trace.json")
    >>> print(result["run_id"])
    'abc-123-def'
    >>> print(result["turns"][0]["confidence"])
    0.85
    
    >>> # Error handling
    >>> try:
    ...     result = adapt("missing.json")
    ... except InputError as e:
    ...     print(f"Input error: {e}")
"""

from pathlib import Path

# Default configuration path (works regardless of CWD)
# Uses Path(__file__).with_name() to find adapter_config.yaml in the same directory
DEFAULT_CONFIG_PATH = Path(__file__).with_name("adapter_config.yaml")

# Import the public API function from adapter module
# This will be implemented in subtask 8.2
try:
    from .adapter import adapt
    from .exceptions import InputError, ValidationError, AdaptationError
    __all__ = ["adapt", "DEFAULT_CONFIG_PATH", "InputError", "ValidationError", "AdaptationError"]
except ImportError:
    # adapter.py not yet implemented - export only DEFAULT_CONFIG_PATH
    __all__ = ["DEFAULT_CONFIG_PATH"]
