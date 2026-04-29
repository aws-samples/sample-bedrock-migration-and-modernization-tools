# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Timestamp trust policy for latency calculations.

This module defines criteria for trusted timestamps and validates timestamp data
for use in latency metrics computation. Only timestamps meeting strict criteria
are used for latency calculations.
"""

from datetime import datetime, timezone
from typing import Optional, Tuple
import re


# OTEL UnixNano format: integer nanoseconds since Unix epoch
# Example: 1609459200000000000 (2021-01-01 00:00:00 UTC)
OTEL_UNIXNANO_PATTERN = re.compile(r'^\d{19}$')  # 19 digits for nanoseconds

# Maximum allowed duration for a single turn (24 hours in milliseconds)
MAX_TURN_DURATION_MS = 24 * 60 * 60 * 1000


class TimestampValidationResult:
    """Result of timestamp validation."""
    
    def __init__(
        self,
        is_trusted: bool,
        reason: Optional[str] = None,
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
        duration_ms: Optional[float] = None
    ):
        """
        Initialize validation result.
        
        Args:
            is_trusted: Whether timestamps are trusted for latency calculations
            reason: Reason for untrusted status (if applicable)
            start_ts: Start timestamp value
            end_ts: End timestamp value
            duration_ms: Calculated duration in milliseconds (if valid)
        """
        self.is_trusted = is_trusted
        self.reason = reason
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.duration_ms = duration_ms


def _timestamp_to_epoch_ms(timestamp: str) -> Optional[float]:
    """
    Convert timestamp string to epoch milliseconds for comparison.
    
    Handles both OTEL UnixNano and ISO 8601 formats.
    
    Args:
        timestamp: Timestamp string
        
    Returns:
        Epoch milliseconds or None if parsing fails
    """
    if not timestamp:
        return None
    
    # Try OTEL UnixNano first (most precise)
    if is_otel_unixnano_format(timestamp):
        try:
            nanos = int(timestamp)
            return nanos / 1_000_000  # Convert to milliseconds
        except (ValueError, OverflowError):
            return None
    
    # Try ISO 8601
    dt = parse_iso8601_timestamp(timestamp)
    if dt is not None:
        try:
            return dt.timestamp() * 1000  # Convert to milliseconds
        except (ValueError, OverflowError):
            return None
    
    return None


def is_otel_unixnano_format(timestamp: str) -> bool:
    """
    Check if timestamp is in OTEL UnixNano format.
    
    OTEL UnixNano format: 19-digit integer representing nanoseconds since Unix epoch.
    Example: "1609459200000000000" (2021-01-01 00:00:00 UTC)
    
    Args:
        timestamp: Timestamp string to check
        
    Returns:
        True if timestamp matches OTEL UnixNano format
    """
    if not isinstance(timestamp, str):
        return False
    return bool(OTEL_UNIXNANO_PATTERN.match(timestamp))


def parse_iso8601_timestamp(timestamp: str) -> Optional[datetime]:
    """
    Parse ISO 8601 timestamp string to datetime.
    
    Args:
        timestamp: ISO 8601 timestamp string
        
    Returns:
        datetime object or None if parsing fails
    """
    if not isinstance(timestamp, str):
        return None
    
    try:
        # Try parsing with fromisoformat (Python 3.7+)
        # Handle 'Z' suffix by replacing with '+00:00'
        ts = timestamp.replace('Z', '+00:00')
        return datetime.fromisoformat(ts)
    except (ValueError, AttributeError):
        return None


def parse_otel_unixnano(timestamp: str) -> Optional[datetime]:
    """
    Parse OTEL UnixNano timestamp to datetime.
    
    Args:
        timestamp: OTEL UnixNano timestamp string (19 digits)
        
    Returns:
        datetime object or None if parsing fails
    """
    if not is_otel_unixnano_format(timestamp):
        return None
    
    try:
        # Convert nanoseconds to seconds
        nanos = int(timestamp)
        seconds = nanos / 1_000_000_000
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (ValueError, OSError):
        # OSError can occur for timestamps outside valid range
        return None


def calculate_duration_ms(start_ts: str, end_ts: str) -> Optional[float]:
    """
    Calculate duration in milliseconds between two timestamps.
    
    Supports both ISO 8601 and OTEL UnixNano formats.
    
    Args:
        start_ts: Start timestamp
        end_ts: End timestamp
        
    Returns:
        Duration in milliseconds or None if calculation fails
    """
    # Try OTEL UnixNano format first (more precise)
    if is_otel_unixnano_format(start_ts) and is_otel_unixnano_format(end_ts):
        try:
            start_nanos = int(start_ts)
            end_nanos = int(end_ts)
            duration_nanos = end_nanos - start_nanos
            return duration_nanos / 1_000_000  # Convert to milliseconds
        except (ValueError, OverflowError):
            return None
    
    # Fall back to ISO 8601 parsing
    start_dt = parse_iso8601_timestamp(start_ts)
    end_dt = parse_iso8601_timestamp(end_ts)
    
    if start_dt is None or end_dt is None:
        return None
    
    try:
        duration = end_dt - start_dt
        return duration.total_seconds() * 1000  # Convert to milliseconds
    except (ValueError, OverflowError):
        return None


def is_timestamp_trusted(
    start_ts: Optional[str],
    end_ts: Optional[str]
) -> TimestampValidationResult:
    """
    Validate whether timestamps are trusted for latency calculations.
    
    Trusted timestamp criteria:
    1. Both start_time and end_time must be present (not None or empty string)
    2. Timestamps must be in OTEL UnixNano format (19-digit integer nanoseconds)
       OR valid ISO 8601 format
    3. Duration must be non-negative (no clock skew)
    4. Duration must be <= 24 hours (reject unrealistic durations)
    
    Args:
        start_ts: Start timestamp (ISO 8601 or OTEL UnixNano)
        end_ts: End timestamp (ISO 8601 or OTEL UnixNano)
        
    Returns:
        TimestampValidationResult with trust status and details
    """
    # Check for missing timestamps (None or empty string)
    if not start_ts:
        return TimestampValidationResult(
            is_trusted=False,
            reason="missing_start_timestamp",
            start_ts=start_ts,
            end_ts=end_ts
        )
    
    if not end_ts:
        return TimestampValidationResult(
            is_trusted=False,
            reason="missing_end_timestamp",
            start_ts=start_ts,
            end_ts=end_ts
        )
    
    # Check format validity
    start_is_otel = is_otel_unixnano_format(start_ts)
    end_is_otel = is_otel_unixnano_format(end_ts)
    start_is_iso = parse_iso8601_timestamp(start_ts) is not None
    end_is_iso = parse_iso8601_timestamp(end_ts) is not None
    
    if not (start_is_otel or start_is_iso):
        return TimestampValidationResult(
            is_trusted=False,
            reason="invalid_start_timestamp_format",
            start_ts=start_ts,
            end_ts=end_ts
        )
    
    if not (end_is_otel or end_is_iso):
        return TimestampValidationResult(
            is_trusted=False,
            reason="invalid_end_timestamp_format",
            start_ts=start_ts,
            end_ts=end_ts
        )
    
    # Calculate duration
    duration_ms = calculate_duration_ms(start_ts, end_ts)
    
    if duration_ms is None:
        return TimestampValidationResult(
            is_trusted=False,
            reason="duration_calculation_failed",
            start_ts=start_ts,
            end_ts=end_ts
        )
    
    # Check for negative duration (clock skew)
    if duration_ms < 0:
        return TimestampValidationResult(
            is_trusted=False,
            reason="negative_duration",
            start_ts=start_ts,
            end_ts=end_ts,
            duration_ms=duration_ms
        )
    
    # Check for unrealistic duration (>24 hours)
    if duration_ms > MAX_TURN_DURATION_MS:
        return TimestampValidationResult(
            is_trusted=False,
            reason="duration_exceeds_24h",
            start_ts=start_ts,
            end_ts=end_ts,
            duration_ms=duration_ms
        )
    
    # All checks passed
    return TimestampValidationResult(
        is_trusted=True,
        reason=None,
        start_ts=start_ts,
        end_ts=end_ts,
        duration_ms=duration_ms
    )


def validate_turn_timestamps(turn: dict) -> TimestampValidationResult:
    """
    Validate timestamps for a turn.
    
    Checks turn-level pre-calculated latency first, then falls back to
    computing from step boundary timestamps (earliest start to latest end).
    
    Args:
        turn: Turn dictionary from NormalizedRun
        
    Returns:
        TimestampValidationResult for the turn
    """
    # Check if turn has pre-calculated normalized_latency_ms
    normalized_latency = turn.get("normalized_latency_ms")
    if normalized_latency is not None:
        # Validate the pre-calculated latency
        try:
            latency_float = float(normalized_latency)
            
            # Reject negative latencies
            if latency_float < 0:
                return TimestampValidationResult(
                    is_trusted=False,
                    reason="negative_normalized_latency",
                    duration_ms=latency_float
                )
            
            # Reject unrealistic latencies (>24 hours)
            if latency_float > MAX_TURN_DURATION_MS:
                return TimestampValidationResult(
                    is_trusted=False,
                    reason="normalized_latency_exceeds_24h",
                    duration_ms=latency_float
                )
            
            # Trust pre-calculated latency if valid
            return TimestampValidationResult(
                is_trusted=True,
                reason=None,
                duration_ms=latency_float
            )
        except (TypeError, ValueError):
            # Invalid type - fall through to step-based calculation
            pass
    
    # Fall back to calculating from steps
    steps = turn.get("steps", [])
    if not steps:
        return TimestampValidationResult(
            is_trusted=False,
            reason="no_steps_for_timestamp_calculation"
        )
    
    # Find earliest start_ts and latest end_ts across all steps
    # Parse to comparable epoch values to handle mixed formats
    start_epochs = []
    end_epochs = []
    
    for step in steps:
        start_ts = step.get("start_ts")
        end_ts = step.get("end_ts")
        
        # Skip empty strings (treat as missing)
        if start_ts and isinstance(start_ts, str):
            start_epoch = _timestamp_to_epoch_ms(start_ts)
            if start_epoch is not None:
                start_epochs.append((start_epoch, start_ts))
        
        if end_ts and isinstance(end_ts, str):
            end_epoch = _timestamp_to_epoch_ms(end_ts)
            if end_epoch is not None:
                end_epochs.append((end_epoch, end_ts))
    
    if not start_epochs or not end_epochs:
        return TimestampValidationResult(
            is_trusted=False,
            reason="insufficient_step_timestamps"
        )
    
    # Find min/max by epoch value (not lexicographic)
    earliest_start_epoch, earliest_start_ts = min(start_epochs, key=lambda x: x[0])
    latest_end_epoch, latest_end_ts = max(end_epochs, key=lambda x: x[0])
    
    return is_timestamp_trusted(earliest_start_ts, latest_end_ts)


def should_compute_latency_percentiles(
    trusted_count: int,
    total_count: int,
    min_trust_ratio: float = 0.5
) -> bool:
    """
    Determine if latency percentiles should be computed.
    
    Percentiles are only computed when >= 50% of turns have trusted timestamps.
    
    Args:
        trusted_count: Number of turns with trusted timestamps
        total_count: Total number of turns
        min_trust_ratio: Minimum ratio of trusted turns (default: 0.5)
        
    Returns:
        True if percentiles should be computed
    """
    if total_count == 0:
        return False
    
    trust_ratio = trusted_count / total_count
    return trust_ratio >= min_trust_ratio
