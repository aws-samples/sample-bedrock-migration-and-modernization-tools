# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Validation utilities for schema compliance checking and confidence scoring.

This module provides shared validation logic for:
- JSON Schema loading and validation
- Timestamp parsing with multiple format support
- Latency sanitization
- Confidence score calculation

All timestamp parsing logic is centralized here to helps consistency
across the adapter implementation.

Key Design Decisions:
1. Timezone Normalization: All timestamps are normalized to UTC-aware datetimes
   to prevent latency calculation errors from mixing naive/aware datetimes.
   This is critical when processing mixed sources (OTEL + app logs).

2. Trust Policy: is_trusted flag indicates reliability for latency calculations:
   - True: Explicit timezone (Z/offset), epoch timestamp, or UnixNano field
   - False: Naive ISO timestamp (treated as UTC but less reliable)

3. Epoch Validation: 
   - Rejects negative values (pre-1970)
   - Rejects zero (often sentinel/invalid)
   - Attempts parsing for small values (< 1 billion) but marks untrusted
   - Heuristic: values >= 1e18 treated as nanoseconds even without field_name

4. Magnitude Inference: Uses improved thresholds (1e11, 1e14) with fallback
   to other units if year bounds are violated.

5. Error Resilience: Schema validation errors are sorted for stable ordering,
   and confidence scoring guards against invalid penalty values.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

try:
    import jsonschema
    from jsonschema import Draft7Validator, ValidationError as JSONSchemaValidationError
except ImportError:
    jsonschema = None
    Draft7Validator = None
    JSONSchemaValidationError = None


def load_schema(schema_path: str) -> dict:
    """
    Load and parse JSON Schema file.
    
    Args:
        schema_path: Path to JSON Schema file
        
    Returns:
        Parsed schema dictionary
        
    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If schema file contains invalid JSON or is empty
    """
    schema_file = Path(schema_path)
    
    if not schema_file.exists():
        raise FileNotFoundError(
            f"Schema file not found: {schema_path}. "
            f"Please ensure the schema file exists."
        )
    
    try:
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON schema from {schema_path}: {e}. "
            f"Please ensure the file contains valid JSON."
        ) from e
    except OSError as e:
        raise ValueError(
            f"Failed to read schema file {schema_path}: {e}. "
            f"Check file permissions."
        ) from e
    
    if not schema:
        raise ValueError(
            f"Schema file is empty: {schema_path}. "
            f"Please ensure the schema file contains valid JSON Schema."
        )
    
    return schema


def validate_against_schema(data: dict, schema: dict) -> Tuple[bool, List[str]]:
    """
    Validate data against JSON Schema.
    
    Args:
        data: Data dictionary to validate
        schema: JSON Schema dictionary
        
    Returns:
        Tuple of (is_valid, error_messages) where:
        - is_valid: True if data conforms to schema, False otherwise
        - error_messages: List of validation error messages (empty if valid)
        
    Raises:
        ImportError: If jsonschema library is not installed
    """
    if jsonschema is None or Draft7Validator is None:
        raise ImportError(
            "jsonschema library is required for schema validation. "
            "Install it with: pip install jsonschema"
        )
    
    validator = Draft7Validator(schema)
    errors = sorted(
        validator.iter_errors(data),
        key=lambda e: ([str(p) for p in e.path], e.message)
    )
    
    if not errors:
        return (True, [])
    
    # Format error messages with field paths and descriptions
    error_messages = []
    for error in errors:
        # Build field path from error.path
        field_path = ".".join(str(p) for p in error.path) if error.path else "root"
        
        # Create descriptive error message
        message = f"Field '{field_path}': {error.message}"
        
        # Add schema path context if available
        if error.schema_path:
            schema_location = ".".join(str(p) for p in error.schema_path)
            message += f" (schema location: {schema_location})"
        
        error_messages.append(message)
    
    return (False, error_messages)


def parse_timestamp(
    value: Any,
    formats: Optional[List[str]] = None,
    epoch_units: Optional[List[str]] = None,
    infer_epoch_unit_by_magnitude: bool = True,
    min_reasonable_year: int = 2000,
    max_reasonable_year: int = 2100,
    unix_nano_fields: Optional[List[str]] = None,
    field_name: Optional[str] = None
) -> Tuple[Optional[datetime], bool, Optional[str]]:
    """
    Parse timestamp from various formats with centralized logic.
    
    This function centralizes ALL timestamp parsing logic for the adapter:
    - ISO 8601 formats (with/without microseconds, with/without timezone)
    - Epoch timestamps (milliseconds, seconds, nanoseconds)
    - OpenTelemetry UnixNano fields
    - Magnitude-based epoch unit inference
    - Year bounds validation (2000-2100)
    - Timezone normalization to UTC-aware datetimes
    
    All timestamps are normalized to UTC-aware datetimes to prevent latency
    calculation errors from mixing naive/aware datetimes (common in OTEL + app logs).
    
    Trust Policy:
    - is_trusted=True: Explicit timezone (Z or offset), epoch timestamp, or UnixNano field
    - is_trusted=False: Naive ISO timestamp without timezone (treated as UTC but less reliable)
    
    Args:
        value: Timestamp value to parse (string, int, float, or None)
        formats: List of strptime format strings for ISO 8601 parsing
        epoch_units: List of supported epoch units (["ms", "s", "ns"])
        infer_epoch_unit_by_magnitude: Whether to infer epoch unit from magnitude
        min_reasonable_year: Minimum valid year (default: 2000)
        max_reasonable_year: Maximum valid year (default: 2100)
        unix_nano_fields: List of field names that contain UnixNano timestamps
        field_name: Optional field name for UnixNano detection
        
    Returns:
        Tuple of (parsed_datetime, is_trusted, error_message) where:
        - parsed_datetime: UTC-aware datetime object, or None if parsing failed
        - is_trusted: True if timestamp is reliable for latency calculation
        - error_message: Error description if parsing failed, None otherwise
    """
    if value is None:
        return (None, False, "Timestamp value is None")
    
    # Default formats if not provided
    if formats is None:
        formats = [
            "%Y-%m-%dT%H:%M:%S.%f%z",      # ISO 8601 with microseconds and offset
            "%Y-%m-%dT%H:%M:%S%z",         # ISO 8601 with offset
            "%Y-%m-%dT%H:%M:%S.%fZ",       # ISO 8601 with microseconds and Z
            "%Y-%m-%dT%H:%M:%SZ",          # ISO 8601 without microseconds
            "%Y-%m-%dT%H:%M:%S.%f",        # ISO 8601 with microseconds, no Z
            "%Y-%m-%dT%H:%M:%S",           # ISO 8601 without microseconds, no Z
            "%Y-%m-%d %H:%M:%S.%f",        # Space-separated with microseconds
            "%Y-%m-%d %H:%M:%S",           # Space-separated without microseconds
        ]
    
    if epoch_units is None:
        epoch_units = ["ms", "s", "ns"]
    
    if unix_nano_fields is None:
        unix_nano_fields = ["startTimeUnixNano", "endTimeUnixNano"]
    
    # Try parsing as string (ISO 8601 formats)
    if isinstance(value, str):
        # Normalize common ISO 8601 variants for better compatibility
        # Some producers emit +00:00 style offsets that older Python %z doesn't accept
        normalized_value = value
        
        # Convert +00:00 style offsets to +0000 for %z compatibility
        # Handles: +00:00, -05:00, +05:30, etc.
        if len(value) > 6 and value[-6] in ('+', '-') and value[-3] == ':':
            normalized_value = value[:-3] + value[-2:]
        
        # Try each format
        for fmt in formats:
            try:
                dt = datetime.strptime(normalized_value, fmt)
                
                # Validate year bounds
                if not (min_reasonable_year <= dt.year <= max_reasonable_year):
                    return (
                        None,
                        False,
                        f"Timestamp year {dt.year} outside reasonable range "
                        f"[{min_reasonable_year}, {max_reasonable_year}]"
                    )
                
                # Timezone normalization: attach UTC timezone for consistency
                # This prevents latency calculation errors from mixing naive/aware datetimes
                
                # If offset-aware (parsed with %z), normalize to UTC and mark trusted
                if dt.tzinfo is not None:
                    dt = dt.astimezone(timezone.utc)
                    return (dt, True, None)
                
                # If literal Z in format (explicit UTC), attach timezone and mark trusted
                if fmt.endswith("Z"):
                    dt = dt.replace(tzinfo=timezone.utc)
                    return (dt, True, None)
                
                # No timezone info: treat as UTC but mark as less trusted
                # This is safer for latency calculations with mixed sources
                dt = dt.replace(tzinfo=timezone.utc)
                return (dt, False, None)
            except ValueError:
                continue
        
        # Try parsing as numeric string (epoch timestamp)
        try:
            numeric_value = float(value)
            return _parse_epoch_timestamp(
                numeric_value,
                epoch_units,
                infer_epoch_unit_by_magnitude,
                min_reasonable_year,
                max_reasonable_year,
                field_name,
                unix_nano_fields
            )
        except (ValueError, TypeError):
            pass
        
        return (None, False, f"Failed to parse timestamp string: {value}")
    
    # Try parsing as numeric (epoch timestamp)
    if isinstance(value, (int, float)):
        return _parse_epoch_timestamp(
            value,
            epoch_units,
            infer_epoch_unit_by_magnitude,
            min_reasonable_year,
            max_reasonable_year,
            field_name,
            unix_nano_fields
        )
    
    return (None, False, f"Unsupported timestamp type: {type(value).__name__}")


def _parse_epoch_timestamp(
    value: Union[int, float],
    epoch_units: List[str],
    infer_by_magnitude: bool,
    min_year: int,
    max_year: int,
    field_name: Optional[str],
    unix_nano_fields: List[str]
) -> Tuple[Optional[datetime], bool, Optional[str]]:
    """
    Parse epoch timestamp with magnitude-based unit inference.
    
    Args:
        value: Numeric timestamp value
        epoch_units: Supported epoch units (["ms", "s", "ns"])
        infer_by_magnitude: Whether to infer unit from magnitude
        min_year: Minimum reasonable year
        max_year: Maximum reasonable year
        field_name: Optional field name for UnixNano detection
        unix_nano_fields: List of field names that contain UnixNano timestamps
        
    Returns:
        Tuple of (datetime, is_trusted, error_message)
    """
    # Reject negative epoch values (pre-1970) - mark as untrusted
    if value < 0:
        return (
            None,
            False,
            f"Negative epoch timestamp not supported: {value}"
        )
    
    # Reject zero explicitly (often a sentinel/invalid value)
    if value == 0:
        return (
            None,
            False,
            "Epoch timestamp is zero (invalid/sentinel)"
        )
    
    # Handle very small values (< 1 billion) that are suspect for modern traces
    # Don't hard-reject; instead try ns/ms/s and mark as untrusted if successful
    # This allows recovery of valid data while signaling lower confidence
    # Order: ns → ms → s (OTEL-ish values most likely, then ms, then s)
    if 0 < value < 1_000_000_000:
        # Too small to be modern seconds; try ns/ms/s explicitly
        for unit in ["ns", "ms", "s"]:
            if unit not in epoch_units:
                continue
            try:
                dt = _convert_epoch_to_datetime(value, unit)
                if min_year <= dt.year <= max_year:
                    # Parsed but suspect (pre-2001 or unusual) -> mark untrusted
                    return (dt, False, None)
            except (ValueError, OSError, OverflowError):
                continue
        
        return (
            None,
            False,
            f"Epoch value {value} too small/out of bounds for all units"
        )
    
    # Check if this is a known UnixNano field (OTEL format)
    if field_name and field_name in unix_nano_fields:
        try:
            # UnixNano: nanoseconds since epoch
            dt = datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc)
            
            # Validate year bounds
            if not (min_year <= dt.year <= max_year):
                return (
                    None,
                    False,
                    f"UnixNano timestamp year {dt.year} outside reasonable range "
                    f"[{min_year}, {max_year}]"
                )
            
            return (dt, True, None)
        except (ValueError, OSError, OverflowError) as e:
            return (None, False, f"Failed to parse UnixNano timestamp: {e}")
    
    # Heuristic: if field_name is unknown but value looks like nanoseconds,
    # treat as ns even without explicit field_name (robust for generic adapter)
    # Guard: year-bounds check protects against ID-like numbers being misinterpreted
    if field_name is None and value >= 1e18 and "ns" in epoch_units:
        try:
            dt = _convert_epoch_to_datetime(value, "ns")
            # Only accept if year is in reasonable range (protects against IDs)
            if min_year <= dt.year <= max_year:
                return (dt, True, None)
        except (ValueError, OSError, OverflowError):
            pass
    
    # Magnitude-based inference
    if infer_by_magnitude:
        # Determine unit by magnitude
        # Typical ranges (updated for better boundaries):
        # - Seconds: ~1.6 billion (2020) to ~4 billion (2100)
        # - Milliseconds: ~1.6 trillion (2020) to ~4 trillion (2100)
        # - Nanoseconds: ~1.6 quintillion (2020) to ~4 quintillion (2100)
        
        # More precise thresholds to avoid misclassification:
        # - If < 1e11 (100 billion): seconds
        # - If < 1e14 (100 trillion): milliseconds
        # - Otherwise: nanoseconds
        
        if value < 1e11:  # Less than 100 billion -> likely seconds
            unit = "s"
        elif value < 1e14:  # Less than 100 trillion -> likely milliseconds
            unit = "ms"
        else:  # Larger -> likely nanoseconds
            unit = "ns"
        
        # Only use inferred unit if it's in the supported list
        if unit not in epoch_units:
            return (
                None,
                False,
                f"Inferred epoch unit '{unit}' not in supported units {epoch_units}"
            )
        
        try:
            dt = _convert_epoch_to_datetime(value, unit)
            
            # Validate year bounds
            if not (min_year <= dt.year <= max_year):
                # Try other units if year is out of bounds
                for alt_unit in epoch_units:
                    if alt_unit == unit:
                        continue
                    try:
                        alt_dt = _convert_epoch_to_datetime(value, alt_unit)
                        if min_year <= alt_dt.year <= max_year:
                            # Found a valid alternative unit
                            # Mark as untrusted since we had to fallback from inferred unit
                            return (alt_dt, False, None)
                    except (ValueError, OSError, OverflowError):
                        continue
                
                return (
                    None,
                    False,
                    f"Epoch timestamp year {dt.year} outside reasonable range "
                    f"[{min_year}, {max_year}] for all units"
                )
            
            return (dt, True, None)
        except (ValueError, OSError, OverflowError) as e:
            return (None, False, f"Failed to parse epoch timestamp: {e}")
    
    # Try each unit explicitly (no inference)
    for unit in epoch_units:
        try:
            dt = _convert_epoch_to_datetime(value, unit)
            
            # Validate year bounds
            if min_year <= dt.year <= max_year:
                return (dt, True, None)
        except (ValueError, OSError, OverflowError):
            continue
    
    return (
        None,
        False,
        f"Failed to parse epoch timestamp with any unit {epoch_units}"
    )


def _convert_epoch_to_datetime(value: Union[int, float], unit: str) -> datetime:
    """
    Convert epoch timestamp to datetime based on unit.
    
    All timestamps are normalized to UTC-aware datetimes to prevent
    latency calculation errors from mixing naive/aware datetimes.
    
    Args:
        value: Numeric timestamp value
        unit: Epoch unit ("s", "ms", or "ns")
        
    Returns:
        UTC-aware datetime object
        
    Raises:
        ValueError: If unit is unsupported or conversion fails
        OSError: If timestamp is out of range for the platform
        OverflowError: If timestamp value is too large
    """
    if unit == "s":
        return datetime.fromtimestamp(value, tz=timezone.utc)
    elif unit == "ms":
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    elif unit == "ns":
        return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc)
    else:
        raise ValueError(f"Unsupported epoch unit: {unit}")


def sanitize_latency(latency: Any) -> Optional[float]:
    """
    Convert latency to float, handling negative values and None.
    
    Negative latency values are converted to zero (with warning logged elsewhere).
    Non-numeric values return None.
    
    Args:
        latency: Latency value (number, string, or None)
        
    Returns:
        Sanitized latency as float, or None if invalid
    """
    if latency is None:
        return None
    
    # Try converting to float
    try:
        latency_float = float(latency)
    except (ValueError, TypeError):
        return None
    
    # Handle negative values (convert to zero)
    if latency_float < 0:
        return 0.0
    
    return latency_float


def calculate_confidence_score(
    penalties: List[Dict[str, Any]],
    base_score: float = 1.0
) -> float:
    """
    Calculate confidence score from penalty list.
    
    Confidence score is calculated by subtracting all penalty values from
    the base score, then clamping to [0, 1] range.
    
    Penalties should be deduplicated by root cause before calling this function
    to avoid double-counting the same issue.
    
    This function guards against invalid penalty values (non-numeric, None, etc.)
    by treating them as 0.0 and continuing.
    
    Args:
        penalties: List of penalty dictionaries with 'penalty' field
        base_score: Starting confidence score (default: 1.0)
        
    Returns:
        Final confidence score clamped to [0, 1]
        
    Examples:
        >>> calculate_confidence_score([])
        1.0
        
        >>> calculate_confidence_score([{"penalty": 0.2}])
        0.8
        
        >>> calculate_confidence_score([{"penalty": 0.3}, {"penalty": 0.4}])
        0.3
        
        >>> calculate_confidence_score([{"penalty": 0.6}, {"penalty": 0.6}])
        0.0
        
        >>> calculate_confidence_score([{"penalty": "invalid"}])
        1.0
    """
    if not penalties:
        return max(0.0, min(1.0, float(base_score)))
    
    # Sum all penalty values, guarding against invalid entries
    # Clamp penalties to non-negative (penalties can't increase confidence)
    total_penalty = 0.0
    for p in penalties:
        try:
            penalty_value = p.get("penalty", 0.0)
            # Clamp to non-negative: penalties can't increase confidence
            total_penalty += max(0.0, float(penalty_value))
        except (TypeError, ValueError, AttributeError):
            # Invalid penalty entry - treat as 0.0 and continue
            continue
    
    # Subtract from base score and clamp to [0, 1]
    try:
        confidence = float(base_score) - total_penalty
    except (TypeError, ValueError):
        # Invalid base_score - default to 1.0
        confidence = 1.0 - total_penalty
    
    return max(0.0, min(1.0, confidence))
