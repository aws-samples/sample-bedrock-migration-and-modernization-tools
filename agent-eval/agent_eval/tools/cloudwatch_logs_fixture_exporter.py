#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Generic CloudWatch Logs Fixture Exporter

This is a STANDALONE log fixture generator that exports raw CloudWatch logs
as Generic JSON events. It is NOT part of the AgentCore evidence reconstruction
pipeline (see agentcore_pipeline/ for that).

Purpose:
- Export raw logs from any CloudWatch log group
- Generate test fixtures for adapter development
- Provide generic log enrichment data

This script:
- Does NOT reconstruct turns or sessions
- Does NOT extract final_answer or link tool calls
- Does NOT merge X-Ray spans
- ONLY exports raw log events with basic OTEL field extraction

For AgentCore-specific trace reconstruction, use:
  agent_eval/tools/agentcore_pipeline/export_agentcore_pipeline.py

CRITICAL: This module MUST NOT import any adapter modules to maintain strict separation.

Usage:
    # Export generic log fixtures
    python cloudwatch_logs_fixture_exporter.py \
        --log-group /aws/lambda/my-service \
        --days 7 \
        --output-dir ./outputs/fixtures/
    
    # With log group discovery
    python cloudwatch_logs_fixture_exporter.py \
        --log-group-prefix /aws/lambda/ \
        --days 30 \
        --output-dir ./outputs/fixtures/
    
    # With regex pattern and custom time range
    python cloudwatch_logs_fixture_exporter.py \
        --log-group-pattern ".*agent.*" \
        --start-time "2024-01-01T00:00:00Z" \
        --end-time "2024-01-31T23:59:59Z" \
        --output-dir ./outputs/fixtures/
"""

import argparse
import boto3
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from botocore.exceptions import (
    NoCredentialsError,
    PartialCredentialsError,
    ClientError,
    BotoCoreError
)

# Pagination guardrails
MAX_EVENTS_PER_GROUP = 50000  # Per log group limit
MAX_EVENTS_TOTAL = 200000  # Total across all groups
MAX_BYTES_PER_EVENT = 50000  # 50KB per event raw field
MAX_SLICES_PER_GROUP = 128  # Max recursive slices to prevent query explosion

# Retry configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 32.0


def _generate_export_id(log_groups: List[str], start_time: datetime, end_time: datetime, filter_pattern: Optional[str]) -> str:
    """
    Generate deterministic export ID from query parameters.
    
    This ensures reruns with same parameters produce same filename.
    
    Args:
        log_groups: List of log group names
        start_time: Query start time
        end_time: Query end time
        filter_pattern: Optional filter pattern
        
    Returns:
        Deterministic hash-based export ID
    """
    # Sort log groups for deterministic ordering
    sorted_groups = sorted(log_groups)
    
    # Create hash input
    hash_input = json.dumps({
        "log_groups": sorted_groups,
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "filter": filter_pattern or ""
    }, sort_keys=True)
    
    # Generate short hash
    hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    # Create readable export ID
    timestamp = start_time.strftime("%Y%m%d")
    return f"export_{timestamp}_{hash_digest}"


def _add_jitter(delay: float) -> float:
    """Add jitter to retry delay to avoid thundering herd."""
    return delay * (0.5 + random.random())


def discover_log_groups(
    prefix: Optional[str] = None,
    pattern: Optional[str] = None,
    region: Optional[str] = None,
    profile: Optional[str] = None
) -> List[str]:
    """
    Discover CloudWatch log groups by prefix or regex pattern.
    
    Args:
        prefix: Log group name prefix (e.g., '/aws/lambda/agent')
        pattern: Regex pattern to match log group names
        region: AWS region
        profile: AWS profile name
        
    Returns:
        List of matching log group names
        
    Raises:
        NoCredentialsError: If AWS credentials not configured
        PartialCredentialsError: If AWS credentials are incomplete
        ClientError: If CloudWatch API call fails
    """
    try:
        session_kwargs = {}
        if profile:
            session_kwargs['profile_name'] = profile
        if region:
            session_kwargs['region_name'] = region
        
        session = boto3.Session(**session_kwargs)
        client = session.client('logs')
        
        log_groups = []
        
        # Use prefix for efficient filtering if provided
        kwargs = {}
        if prefix:
            kwargs['logGroupNamePrefix'] = prefix
        
        # Paginate through all log groups
        paginator = client.get_paginator('describe_log_groups')
        for page in paginator.paginate(**kwargs):
            for lg in page.get('logGroups', []):
                log_group_name = lg['logGroupName']
                
                # Apply regex pattern if specified
                if pattern:
                    if re.search(pattern, log_group_name):
                        log_groups.append(log_group_name)
                else:
                    log_groups.append(log_group_name)
        
        return log_groups
        
    except NoCredentialsError:
        raise NoCredentialsError(
            "AWS credentials not found. Please configure credentials using:\n"
            "  - AWS CLI: aws configure\n"
            "  - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY\n"
            "  - IAM role (if running on EC2/Lambda)"
        )
    except PartialCredentialsError as e:
        raise PartialCredentialsError(
            provider=e.provider,
            cred_var=f"Incomplete AWS credentials: {e}. Please provide all required credential components."
        )
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'ExpiredTokenException':
            raise ClientError(
                {
                    'Error': {
                        'Code': 'ExpiredTokenException',
                        'Message': 'AWS credentials have expired. Please refresh your credentials:\n'
                                 '  - For SSO: Run `aws sso login --profile <profile>`\n'
                                 '  - For temporary credentials: Obtain new credentials from your identity provider'
                    }
                },
                operation_name=e.operation_name
            )
        raise


def _event_key(row: List[Dict[str, str]]) -> Tuple[str, str, str, str]:
    """
    Generate deterministic key for deduplication.
    
    Uses timestamp, log stream, ingestion time, and bounded message.
    
    Args:
        row: CloudWatch Logs Insights result row
        
    Returns:
        Tuple of (timestamp, log_stream, ingestion_time, message_prefix)
    """
    vals = {f['field']: f.get('value', '') for f in row}
    return (
        vals.get('@timestamp', ''),
        vals.get('@logStream', ''),
        vals.get('@ingestionTime', ''),
        vals.get('@message', '')[:200]  # Bounded for performance
    )


def _deduplicate_events(events: List[List[Dict[str, str]]]) -> List[List[Dict[str, str]]]:
    """
    Deduplicate events while preserving order.
    
    Uses deterministic key based on CloudWatch fields.
    
    Args:
        events: List of CloudWatch Logs Insights result rows
        
    Returns:
        Deduplicated list preserving original order
    """
    seen = set()
    deduplicated = []
    
    for event in events:
        key = _event_key(event)
        if key not in seen:
            seen.add(key)
            deduplicated.append(event)
    
    return deduplicated


def _query_cloudwatch_with_slicing(
    client,
    log_group_name: str,
    start_time: datetime,
    end_time: datetime,
    filter_pattern: Optional[str],
    min_slice_minutes: int = 5,
    slice_count: int = 0
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Execute CloudWatch Logs Insights query with automatic time-slicing for large result sets.
    
    CloudWatch Logs Insights has a hard limit of 10,000 rows per query.
    This function recursively splits the time window when hitting the limit.
    Deduplicates results to handle boundary overlaps.
    
    Args:
        client: boto3 CloudWatch Logs client
        log_group_name: Log group to query
        start_time: Query start time (timezone-aware)
        end_time: Query end time (timezone-aware)
        filter_pattern: Optional Logs Insights filter expression
        min_slice_minutes: Minimum time slice size (stop splitting below this)
        slice_count: Current slice count (for recursion tracking)
        
    Returns:
        Tuple of (events list, query metadata dict)
    """
    # Validate time range
    if start_time >= end_time:
        raise ValueError(f"start_time ({start_time}) must be before end_time ({end_time})")
    
    # Check slice count guardrail
    if slice_count >= MAX_SLICES_PER_GROUP:
        print(f"  ⚠ Warning: Reached max slices ({MAX_SLICES_PER_GROUP}), returning truncated results")
        return [], {
            "slicing_used": True,
            "slices_created": slice_count,
            "truncated": True,
            "truncation_reason": "max_slices_exceeded"
        }
    
    # Build query string
    query = """
    fields @timestamp, @message, @logStream, @ingestionTime
    | sort @timestamp asc
    """
    
    if filter_pattern:
        query = f"{query}\n| filter {filter_pattern}"
    
    # Track query metadata
    query_metadata = {
        "query_string": query,
        "filter": filter_pattern or None,
        "slicing_used": False,
        "slices_created": 1,
        "min_slice_minutes": min_slice_minutes
    }
    
    # Start query with retry
    query_id = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.start_query(
                logGroupName=log_group_name,
                startTime=int(start_time.timestamp()),
                endTime=int(end_time.timestamp()),
                queryString=query,
                limit=10000  # CloudWatch Logs Insights max per query
            )
            
            query_id = response['queryId']
            break
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            
            if error_code in ('ThrottlingException', 'TooManyRequestsException', 
                            'LimitExceededException', 'ServiceUnavailableException'):
                if attempt < MAX_RETRIES - 1:
                    delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                    jittered_delay = _add_jitter(delay)
                    print(f"  Throttled, retrying in {jittered_delay:.2f}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(jittered_delay)
                    continue
                else:
                    print(f"  ✗ Max retries exceeded for {log_group_name}")
                    raise
            else:
                raise
    
    if query_id is None:
        return [], query_metadata
    
    # Poll for results with timeout
    poll_timeout = 300  # 5 minutes
    poll_start = time.time()
    
    while time.time() - poll_start < poll_timeout:
        try:
            result = client.get_query_results(queryId=query_id)
            status = result['status']
            
            if status == 'Complete':
                results = result.get('results', [])
                
                # Check if we hit the 10k limit and can split further
                if len(results) >= 10000:
                    time_span = end_time - start_time
                    
                    # If window is too small to split, return truncated results
                    if time_span <= timedelta(minutes=min_slice_minutes):
                        print(f"  ⚠ Warning: Hit 10k limit in {time_span.total_seconds()/60:.1f}min window, cannot split further")
                        query_metadata["slicing_used"] = True
                        query_metadata["truncated"] = True
                        query_metadata["truncation_reason"] = "min_slice_size_reached"
                        return results, query_metadata
                    
                    # Split time window and recursively query (with 1ms offset to avoid boundary duplicates)
                    mid_time = start_time + time_span / 2
                    second_start = mid_time + timedelta(milliseconds=1)
                    
                    print(f"  ⚠ Hit 10k limit, splitting window: {start_time.isoformat()} to {end_time.isoformat()}")
                    
                    first_half, first_meta = _query_cloudwatch_with_slicing(
                        client, log_group_name, start_time, mid_time, filter_pattern, min_slice_minutes, slice_count + 1
                    )
                    second_half, second_meta = _query_cloudwatch_with_slicing(
                        client, log_group_name, second_start, end_time, filter_pattern, min_slice_minutes, slice_count + 1
                    )
                    
                    # Merge results and deduplicate
                    combined = first_half + second_half
                    deduplicated = _deduplicate_events(combined)
                    
                    # Merge metadata
                    query_metadata["slicing_used"] = True
                    query_metadata["slices_created"] = first_meta["slices_created"] + second_meta["slices_created"]
                    query_metadata["deduplicated_count"] = len(combined) - len(deduplicated)
                    
                    return deduplicated, query_metadata
                
                return results, query_metadata
                
            elif status in ('Failed', 'Cancelled', 'Timeout'):
                print(f"  ⚠ Warning: Query {status.lower()} for {log_group_name}")
                query_metadata["query_status"] = status.lower()
                return [], query_metadata
            
            time.sleep(1)  # Wait before polling again
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ('ThrottlingException', 'TooManyRequestsException'):
                time.sleep(_add_jitter(2.0))
                continue
            raise
    
    print(f"  ⚠ Warning: Query timeout for {log_group_name}")
    query_metadata["query_status"] = "timeout"
    return [], query_metadata


def _extract_otel_fields(message_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract OTEL-specific fields from parsed message JSON.
    
    Preserves trace context and span hierarchy for AgentCore Observability logs.
    Checks both top-level and nested attributes for better OTEL compatibility.
    
    Args:
        message_data: Parsed message JSON
        
    Returns:
        Dict with extracted OTEL fields
    """
    otel_fields = {}
    
    # Build list of dicts to search (top-level + attributes)
    candidate_dicts = [message_data]
    if isinstance(message_data.get("attributes"), dict):
        candidate_dicts.append(message_data["attributes"])
    if isinstance(message_data.get("resource"), dict) and isinstance(message_data["resource"].get("attributes"), dict):
        candidate_dicts.append(message_data["resource"]["attributes"])
    
    # Extract trace context from all candidate locations
    for data in candidate_dicts:
        if 'trace_id' not in otel_fields:
            for key in ['traceId', 'trace_id', 'TraceId']:
                if key in data:
                    otel_fields['trace_id'] = data[key]
                    break
        
        if 'span_id' not in otel_fields:
            for key in ['spanId', 'span_id', 'SpanId']:
                if key in data:
                    otel_fields['span_id'] = data[key]
                    break
        
        if 'parent_span_id' not in otel_fields:
            for key in ['parentSpanId', 'parent_span_id', 'ParentSpanId']:
                if key in data:
                    otel_fields['parent_span_id'] = data[key]
                    break
        
        # Extract session/request context
        if 'session_id' not in otel_fields:
            for key in ['sessionId', 'session_id', 'SessionId']:
                if key in data:
                    otel_fields['session_id'] = data[key]
                    break
        
        if 'request_id' not in otel_fields:
            for key in ['requestId', 'request_id', 'RequestId']:
                if key in data:
                    otel_fields['request_id'] = data[key]
                    break
        
        # Extract event type/operation
        if 'event_type' not in otel_fields:
            for key in ['eventType', 'event_type', 'type', 'name', 'operation']:
                if key in data:
                    otel_fields['event_type'] = data[key]
                    break
        
        # Extract status
        if 'status' not in otel_fields:
            for key in ['status', 'Status', 'statusCode', 'status_code']:
                if key in data:
                    otel_fields['status'] = data[key]
                    break
        
        # Extract latency if present
        if 'latency_ms' not in otel_fields:
            for key in ['latency_ms', 'latencyMs', 'duration_ms', 'durationMs']:
                if key in data:
                    try:
                        otel_fields['latency_ms'] = float(data[key])
                    except (ValueError, TypeError):
                        pass
                    break
    
    return otel_fields


def _flatten_attributes(data: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """
    Flatten nested dict into dot-notation keys.
    
    Args:
        data: Nested dict
        prefix: Key prefix for recursion
        
    Returns:
        Flattened dict
    """
    flattened = {}
    
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            flattened.update(_flatten_attributes(value, full_key))
        elif isinstance(value, (list, tuple)):
            # Store lists as JSON strings
            flattened[full_key] = json.dumps(value)
        else:
            flattened[full_key] = value
    
    return flattened


def _parse_log_to_generic_json(
    log_entry: Dict[str, Any],
    log_group: str,
    event_index: int
) -> Dict[str, Any]:
    """
    Parse a CloudWatch log entry into Generic JSON event structure.
    
    Extracts event-level fields only - does NOT attempt turn segmentation
    or final_answer extraction (adapter's job).
    
    Args:
        log_entry: Raw CloudWatch Logs Insights result row
        log_group: Log group name
        event_index: Event index within export
        
    Returns:
        Generic JSON event dict with guaranteed minimum fields
    """
    # Initialize event with minimum required fields
    event = {
        "event_index": event_index,
        "event_type": "EVENT",  # Default, may be overridden
        "attributes": {},
        "raw": None
    }
    
    # Extract fields from CloudWatch Logs Insights result format
    raw_message = None
    timestamp = None
    log_stream = None
    ingestion_time = None
    
    for field in log_entry:
        field_name = field.get('field', '')
        field_value = field.get('value', '')
        
        if field_name == '@timestamp':
            timestamp = field_value
        elif field_name == '@message':
            raw_message = field_value
        elif field_name == '@logStream':
            log_stream = field_value
        elif field_name == '@ingestionTime':
            ingestion_time = field_value
        else:
            # Store other fields in attributes
            event['attributes'][field_name] = field_value
    
    # Set timestamp (required field) - use timezone-aware UTC
    event['timestamp'] = timestamp or datetime.now(timezone.utc).isoformat()
    
    # Preserve CloudWatch metadata in attributes
    event['attributes']['log_group'] = log_group
    if log_stream:
        event['attributes']['log_stream'] = log_stream
    if ingestion_time:
        event['attributes']['ingestion_time'] = ingestion_time
    
    # Parse message
    if raw_message:
        # Try to parse as JSON (OTEL/AgentCore format)
        try:
            message_data = json.loads(raw_message)
            
            # Extract OTEL fields if present
            otel_fields = _extract_otel_fields(message_data)
            event.update(otel_fields)
            
            # Flatten and preserve all attributes
            if 'attributes' in message_data and isinstance(message_data['attributes'], dict):
                flattened = _flatten_attributes(message_data['attributes'])
                event['attributes'].update(flattened)
            
            # Preserve body/message content as text
            if 'body' in message_data:
                body = message_data['body']
                if isinstance(body, dict):
                    # Extract text from nested body structures
                    if 'message' in body:
                        event['text'] = str(body['message'])
                    elif 'content' in body:
                        event['text'] = str(body['content'])
                    else:
                        event['text'] = json.dumps(body)
                else:
                    event['text'] = str(body)
            elif 'message' in message_data:
                event['text'] = str(message_data['message'])
            
            # Store bounded raw for debugging
            raw_str = json.dumps(message_data)
            if len(raw_str) <= MAX_BYTES_PER_EVENT:
                event['raw'] = message_data
            else:
                event['raw'] = {"truncated": True, "size_bytes": len(raw_str)}
                event['attributes']['raw_truncated'] = True
            
        except json.JSONDecodeError:
            # Not JSON - store as plain text
            event['text'] = raw_message
            event['event_type'] = "TEXT_LOG"
            
            # Track parse failure for debugging
            event['attributes']['parse_error'] = True
            event['attributes']['parse_error_type'] = "JSONDecodeError"
            
            # Store bounded raw
            if len(raw_message) <= MAX_BYTES_PER_EVENT:
                event['raw'] = {"text": raw_message}
            else:
                event['raw'] = {"truncated": True, "size_bytes": len(raw_message)}
                event['attributes']['raw_truncated'] = True
    
    return event


def _save_export_file(
    export_data: Dict[str, Any],
    output_dir: str,
    export_id: str,
    aws_region: Optional[str] = None
) -> str:
    """
    Save export as JSON file with deterministic filename.
    
    Adds provenance metadata for debugging.
    
    Args:
        export_data: Export data dict (source already set)
        output_dir: Output directory path
        export_id: Deterministic export ID
        aws_region: AWS region (for provenance)
        
    Returns:
        Path to saved file
    """
    # Add optional provenance metadata
    if aws_region:
        export_data['aws_region'] = aws_region
    
    # Add export_id to each event for traceability
    for event in export_data.get('events', []):
        event['export_id'] = export_id
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"{export_id}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    return str(filepath)


def export_cloudwatch_logs(
    log_group_name: Optional[str] = None,
    log_group_prefix: Optional[str] = None,
    log_group_pattern: Optional[str] = None,
    days: Optional[int] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    output_dir: str = "./outputs/exported_traces/",
    filter_pattern: Optional[str] = None,
    region: Optional[str] = None,
    profile: Optional[str] = None
) -> List[str]:
    """
    Export CloudWatch logs as Generic JSON event files.
    
    Args:
        log_group_name: Specific CloudWatch log group name
        log_group_prefix: Log group prefix for discovery
        log_group_pattern: Regex pattern for log group discovery
        days: Number of days to look back (default 90)
        start_time: ISO 8601 start time (overrides days)
        end_time: ISO 8601 end time (overrides days)
        output_dir: Directory to save export files
        filter_pattern: Optional CloudWatch Logs Insights filter
        region: AWS region
        profile: AWS profile name
        
    Returns:
        List of created file paths
        
    Raises:
        NoCredentialsError: If AWS credentials not configured
        PartialCredentialsError: If AWS credentials incomplete
        ClientError: If CloudWatch API call fails
        ValueError: If no log groups found and none specified
    """
    try:
        # Create boto3 session
        session_kwargs = {}
        if profile:
            session_kwargs['profile_name'] = profile
        if region:
            session_kwargs['region_name'] = region
        
        session = boto3.Session(**session_kwargs)
        client = session.client('logs')
        
        # Determine log groups to query
        if log_group_name:
            log_groups = [log_group_name]
        else:
            print(f"Discovering log groups...")
            log_groups = discover_log_groups(
                prefix=log_group_prefix,
                pattern=log_group_pattern,
                region=region,
                profile=profile
            )
            
            if not log_groups:
                raise ValueError(
                    f"No log groups found matching "
                    f"prefix='{log_group_prefix}' pattern='{log_group_pattern}'"
                )
            
            print(f"Found {len(log_groups)} log group(s):")
            for lg in log_groups:
                print(f"  - {lg}")
        
        # Calculate time range (timezone-aware UTC)
        if start_time and end_time:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            
            # Validate time range
            if start_dt >= end_dt:
                raise ValueError(f"start_time must be before end_time (got {start_dt} >= {end_dt})")
        else:
            days = days or 90
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=days)
        
        print(f"\nQuerying logs from {start_dt.isoformat()} to {end_dt.isoformat()}")
        
        # Generate deterministic export ID
        export_id = _generate_export_id(log_groups, start_dt, end_dt, filter_pattern)
        
        # Query each log group
        all_events = []
        total_events = 0
        events_by_log_group = {}
        query_metadata_by_group = {}
        
        for log_group in log_groups:
            print(f"\nProcessing: {log_group}")
            
            try:
                results, query_meta = _query_cloudwatch_with_slicing(
                    client,
                    log_group,
                    start_dt,
                    end_dt,
                    filter_pattern
                )
                
                # Store query metadata
                query_metadata_by_group[log_group] = query_meta
                
                if not results:
                    print(f"  No logs found in time range")
                    events_by_log_group[log_group] = 0
                    continue
                
                # Report slicing if used
                if query_meta.get("slicing_used"):
                    slices = query_meta.get("slices_created", 0)
                    deduped = query_meta.get("deduplicated_count", 0)
                    print(f"  ℹ Time-slicing used: {slices} slices, {deduped} duplicates removed")
                
                # Parse results into Generic JSON events
                log_group_events = []
                for i, log_entry in enumerate(results):
                    event = _parse_log_to_generic_json(log_entry, log_group, total_events + i)
                    log_group_events.append(event)
                    
                    # Check per-group limit
                    if len(log_group_events) >= MAX_EVENTS_PER_GROUP:
                        print(f"  ⚠ Warning: Reached per-group limit ({MAX_EVENTS_PER_GROUP}), truncating")
                        break
                
                all_events.extend(log_group_events)
                total_events += len(log_group_events)
                events_by_log_group[log_group] = len(log_group_events)
                
                print(f"  ✓ Extracted {len(log_group_events)} events")
                
                # Check total limit across all groups
                if total_events >= MAX_EVENTS_TOTAL:
                    print(f"  ⚠ Warning: Reached total events limit ({MAX_EVENTS_TOTAL}), stopping")
                    break
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                print(f"  ✗ Error querying {log_group}: {error_code} - {e}")
                events_by_log_group[log_group] = 0
                query_metadata_by_group[log_group] = {"error": error_code}
                continue
        
        # Re-index events sequentially after all collection and deduplication
        for idx, event in enumerate(all_events):
            event['event_index'] = idx
        
        # Create export file (even if empty)
        export_data = {
            "export_id": export_id,
            "source": "cloudwatch_logs_insights",
            "window": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat()
            },
            "log_groups": log_groups,
            "events_by_log_group": events_by_log_group,
            "query_metadata": query_metadata_by_group,
            "total_events": total_events,
            "events": all_events
        }
        
        filepath = _save_export_file(export_data, output_dir, export_id, region)
        
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Export ID: {export_id}")
        print(f"  Log groups processed: {len(log_groups)}")
        print(f"  Total events: {total_events}")
        print(f"  Output file: {filepath}")
        
        if total_events == 0:
            print(f"\n⚠ Warning: No events found in time range")
        
        return [filepath] if filepath else []
        
    except (NoCredentialsError, PartialCredentialsError) as e:
        print(f"\n✗ AWS Credentials Error:")
        print(f"  {e}")
        raise
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'ExpiredTokenException':
            print(f"\n✗ AWS Credentials Expired:")
            print(f"  {e}")
        else:
            print(f"\n✗ CloudWatch API Error:")
            print(f"  {e}")
        raise


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export CloudWatch logs to Generic JSON event files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export from specific log group (last 30 days)
  python cloudwatch_extractor.py --log-group /aws/lambda/my-agent --days 30

  # Discover log groups by prefix (last 90 days, default)
  python cloudwatch_extractor.py --log-group-prefix /aws/lambda/agent

  # Discover log groups by regex pattern
  python cloudwatch_extractor.py --log-group-pattern ".*agent.*" --days 7

  # Custom time range
  python cloudwatch_extractor.py --log-group /aws/lambda/my-agent \
    --start-time "2024-01-01T00:00:00Z" \
    --end-time "2024-01-31T23:59:59Z"

  # With AWS profile and region
  python cloudwatch_extractor.py --log-group /aws/lambda/my-agent \
    --profile my-profile --region us-west-2
        """
    )
    
    # Log group specification (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--log-group",
        help="Specific CloudWatch log group name"
    )
    group.add_argument(
        "--log-group-prefix",
        help="Log group prefix for discovery (e.g., '/aws/lambda/agent')"
    )
    group.add_argument(
        "--log-group-pattern",
        help="Regex pattern for log group discovery"
    )
    
    # Time range
    parser.add_argument(
        "--days",
        type=int,
        help="Number of days to look back (default: 90)"
    )
    parser.add_argument(
        "--start-time",
        help="ISO 8601 start time (e.g., '2024-01-01T00:00:00Z')"
    )
    parser.add_argument(
        "--end-time",
        help="ISO 8601 end time (e.g., '2024-01-31T23:59:59Z')"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        default="./outputs/exported_traces/",
        help="Output directory for export files (default: ./outputs/exported_traces/)"
    )
    parser.add_argument(
        "--filter",
        dest="filter_pattern",
        help="CloudWatch Logs Insights filter expression (e.g., '@message like /error/' or 'status = 500')"
    )
    
    # AWS configuration
    parser.add_argument(
        "--region",
        help="AWS region (e.g., 'us-east-1')"
    )
    parser.add_argument(
        "--profile",
        help="AWS profile name"
    )
    
    args = parser.parse_args()
    
    try:
        files = export_cloudwatch_logs(
            log_group_name=args.log_group,
            log_group_prefix=args.log_group_prefix,
            log_group_pattern=args.log_group_pattern,
            days=args.days,
            start_time=args.start_time,
            end_time=args.end_time,
            output_dir=args.output_dir,
            filter_pattern=args.filter_pattern,
            region=args.region,
            profile=args.profile
        )
        
        if files:
            print(f"\n✓ Success! Exported to {files[0]}")
            sys.exit(0)
        else:
            print(f"\n⚠ No export files created")
            sys.exit(0)
        
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  - Verify log group prefix/pattern is correct")
        print("  - Check that log groups exist in your AWS account")
        print("  - Try specifying --log-group directly if you know the exact name")
        sys.exit(1)
    except (NoCredentialsError, PartialCredentialsError):
        print("\nTroubleshooting:")
        print("  - Run 'aws configure' to set up credentials")
        print("  - Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        print("  - Or use an IAM role if running on AWS infrastructure")
        print("  - For SSO: Run `aws sso login --profile <profile>`")
        sys.exit(1)
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        print("\nTroubleshooting:")
        if error_code == 'ExpiredTokenException':
            print("  - Refresh your AWS credentials")
            print("  - For SSO: Run `aws sso login --profile <profile>`")
        else:
            print("  - Verify you have CloudWatch Logs read permissions")
            print("  - Check that the log group exists and is accessible")
            print("  - Verify your AWS region is correct")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
