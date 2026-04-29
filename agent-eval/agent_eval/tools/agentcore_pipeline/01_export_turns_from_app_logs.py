#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Export AgentCore session turns from CloudWatch Logs.

This script queries CloudWatch Logs to extract AgentCore session turns and trace information.
It supports two log stream kinds:
  - 'runtime': Direct AgentCore runtime logs (for direct runtime log streams)
  - 'application': Application logs with APPLICATION_LOGS filter (default, backward compatible)

The --log-stream-kind flag allows switching between query templates to support different
log stream sources while maintaining backward compatibility.

Usage:
    # Application logs (default behavior)
    python 01_export_turns_from_app_logs.py --log-group /aws/logs/my-app --session-id abc123
    
    # Runtime logs (direct AgentCore runtime logs)
    python 01_export_turns_from_app_logs.py --log-group /aws/logs/runtime --session-id abc123 --log-stream-kind runtime
"""
import os, json, time, argparse, re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

REGION = os.getenv("AWS_REGION", "us-east-1")
APP_LOG_GROUP = os.getenv("APP_LOG_GROUP")

# Discovery configuration constants
DISCOVERY_KEYWORDS = {
    "agent": 10,
    "bedrock": 5,
    "agentcore": 8,
    "observability": 3
}
MAX_GROUPS_TO_EXPORT = 3


# Custom exceptions
class NoLogGroupsFoundError(Exception):
    """Raised when no log groups match the discovery criteria."""
    pass


class AWSCredentialsError(Exception):
    """Raised when AWS credentials are invalid or missing."""
    pass


logs = boto3.client("logs", region_name=REGION)

def run_logs_insights(query: str, log_groups: list[str], start_time: int, end_time: int, limit: int = 1000, logs_client=None):
    if logs_client is None:
        logs_client = logs
    
    resp = logs_client.start_query(
        logGroupNames=log_groups,
        startTime=start_time,
        endTime=end_time,
        queryString=query,
        limit=limit,
    )
    qid = resp["queryId"]
    while True:
        r = logs_client.get_query_results(queryId=qid)
        if r["status"] in ("Complete", "Failed", "Cancelled"):
            return r
        time.sleep(1)

def rows_to_dicts(results):
    out = []
    for row in results:
        d = {kv["field"]: kv.get("value") for kv in row}
        out.append(d)
    return out


def select_best_log_groups(matched_groups: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Select best log groups deterministically with scoring details.
    
    Args:
        matched_groups: List of log group names that matched search criteria
        
    Returns:
        Tuple of (selected_groups, scoring_details)
    """
    if len(matched_groups) == 1:
        return matched_groups, {"single_match": True}
    
    # Score groups by relevance
    scored_groups = []
    for group in matched_groups:
        score = 0
        matched_keywords = []
        
        # Apply configurable keyword scoring
        for keyword, points in DISCOVERY_KEYWORDS.items():
            if keyword in group.lower():
                score += points
                matched_keywords.append(keyword)
        
        scored_groups.append({
            "group": group,
            "score": score,
            "matched_keywords": matched_keywords
        })
    
    # Sort by score (descending), then alphabetically for determinism
    scored_groups.sort(key=lambda x: (-x["score"], x["group"]))
    
    # Select top N groups
    selected = [item["group"] for item in scored_groups[:MAX_GROUPS_TO_EXPORT]]
    
    scoring_details = {
        "scoring_keywords": DISCOVERY_KEYWORDS,
        "max_groups_to_export": MAX_GROUPS_TO_EXPORT,
        "scored_groups": scored_groups[:10]  # Top 10 for debugging
    }
    
    return selected, scoring_details


def get_selection_reason(matched_groups: List[str], selected_groups: List[str], scoring_details: Dict[str, Any]) -> str:
    """
    Generate human-readable selection reason with scoring explanation.
    
    Args:
        matched_groups: All groups that matched search criteria
        selected_groups: Groups selected for export
        scoring_details: Scoring algorithm details
        
    Returns:
        Descriptive string explaining selection logic
    """
    if len(matched_groups) == 1:
        return "Only one log group matched"
    elif len(selected_groups) == len(matched_groups):
        return f"All {len(matched_groups)} matched groups selected"
    else:
        keywords_str = ", ".join([f"{k}(+{v})" for k, v in DISCOVERY_KEYWORDS.items()])
        return (
            f"Selected top {len(selected_groups)} of {len(matched_groups)} matched groups "
            f"by relevance score (keywords: {keywords_str}), then alphabetically"
        )


def discover_log_groups(
    region: str,
    prefix: Optional[str] = None,
    pattern: Optional[str] = None,
    profile: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Discover CloudWatch log groups and write discovery.json artifact.
    
    Args:
        region: AWS region to search
        prefix: Log group prefix (e.g., "/aws/bedrock/agent")
        pattern: Regex pattern (e.g., ".*agent.*")
        profile: AWS profile name
        output_dir: Directory to write discovery.json
        
    Returns:
        Discovery result dict (also written to discovery.json)
        
    Raises:
        NoLogGroupsFoundError: If no groups match criteria
        AWSCredentialsError: If AWS credentials are invalid
    """
    try:
        # Initialize CloudWatch Logs client
        session = boto3.Session(profile_name=profile, region_name=region)
        logs_client = session.client('logs')
        
        # Discover log groups
        matched_groups = []
        
        if prefix:
            # Prefix-based discovery (efficient)
            paginator = logs_client.get_paginator('describe_log_groups')
            for page in paginator.paginate(logGroupNamePrefix=prefix):
                matched_groups.extend([lg['logGroupName'] for lg in page['logGroups']])
        elif pattern:
            # Pattern-based discovery (requires full scan)
            compiled_pattern = re.compile(pattern)
            paginator = logs_client.get_paginator('describe_log_groups')
            for page in paginator.paginate():
                for lg in page['logGroups']:
                    if compiled_pattern.search(lg['logGroupName']):
                        matched_groups.append(lg['logGroupName'])
        else:
            raise ValueError("Either prefix or pattern must be provided")
        
        if not matched_groups:
            criteria = f"prefix={prefix}" if prefix else f"pattern={pattern}"
            raise NoLogGroupsFoundError(
                f"No log groups found matching {criteria}"
            )
        
        # Deterministic selection
        selected_groups, scoring_details = select_best_log_groups(matched_groups)
        
        # Build discovery result
        discovery_result = {
            "search_criteria": {"prefix": prefix, "pattern": pattern},
            "matched_groups": matched_groups,
            "selected_groups": selected_groups,
            "selection_reason": get_selection_reason(matched_groups, selected_groups, scoring_details),
            "scoring_details": scoring_details,
            "total_matched": len(matched_groups),
            "total_selected": len(selected_groups)
        }
        
        # Write discovery.json artifact
        if output_dir:
            discovery_path = output_dir / "discovery.json"
            with open(discovery_path, 'w') as f:
                json.dump(discovery_result, f, indent=2)
        
        return discovery_result
        
    except (NoCredentialsError, ClientError) as e:
        raise AWSCredentialsError(
            f"AWS credentials are invalid or missing: {str(e)}"
        ) from e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-group", help="Exact log group name")
    ap.add_argument("--log-group-prefix", help="Log group prefix for discovery")
    ap.add_argument("--log-group-pattern", help="Regex pattern for discovery")
    ap.add_argument("--session-id", help="AgentCore session_id (run_id). If omitted, gets all sessions.")
    ap.add_argument("--minutes", type=int, default=180, help="Lookback minutes window")
    ap.add_argument("--out", default="session_turns.json")
    ap.add_argument("--output-dir", type=Path, help="Output directory for discovery.json artifact")
    ap.add_argument("--region", default=REGION, help="AWS region")
    ap.add_argument("--profile", help="AWS profile name")
    ap.add_argument(
        "--log-stream-kind",
        choices=["runtime", "application", "otel"],
        default="application",
        help="Log stream kind: 'runtime' for direct AgentCore runtime logs, 'application' for APPLICATION_LOGS filter, 'otel' for OTEL stream (default: application)"
    )
    args = ap.parse_args()

    # Validate: exactly one of log-group, log-group-prefix, or log-group-pattern
    discovery_args = [args.log_group, args.log_group_prefix, args.log_group_pattern]
    if sum(x is not None for x in discovery_args) != 1:
        ap.error("Exactly one of --log-group, --log-group-prefix, or --log-group-pattern required")

    # Discover log groups if needed
    if args.log_group:
        log_groups = [args.log_group]
        # No discovery.json written
    else:
        discovery_result = discover_log_groups(
            region=args.region,
            prefix=args.log_group_prefix,
            pattern=args.log_group_pattern,
            profile=args.profile,
            output_dir=args.output_dir
        )
        log_groups = discovery_result["selected_groups"]
        
        print(f"Discovery matched {discovery_result['total_matched']} log groups")
        print(f"Selected {discovery_result['total_selected']} groups: {log_groups}")
        print(f"Selection reason: {discovery_result['selection_reason']}")

    end_ts = int(time.time())
    start_ts = end_ts - args.minutes * 60

    # Build filter - either specific session or all sessions
    session_filter = f'| filter session_id = "{args.session_id}"' if args.session_id else ""

    # Define query templates based on log stream kind
    if args.log_stream_kind == "runtime":
        # Query template for direct AgentCore runtime logs
        # These logs contain runtime-level trace information without APPLICATION_LOGS filter
        q = f"""
        fields
          @timestamp as ts,
          session_id,
          trace_id,
          request_id,
          span_id,
          body.request_payload.prompt as user_query
        {session_filter}
        | sort ts asc
        """
    elif args.log_stream_kind == "application":
        # Query template for application logs (existing behavior)
        # Uses APPLICATION_LOGS filter for application-level logs
        q = f"""
        fields
          @timestamp as ts,
          session_id,
          trace_id,
          request_id,
          span_id,
          body.request_payload.prompt as user_query
        | filter @log like /APPLICATION_LOGS/
        {session_filter}
        | sort ts asc
        """
    elif args.log_stream_kind == "otel":
        # Query template for OTEL stream (otel-rt-logs)
        # OTEL logs have different field structure than APPLICATION_LOGS
        # Filter to otel-rt-logs stream and extract OTEL-specific fields
        # Only include events with valid traceId (filters out non-trace OTEL events)
        session_filter_otel = f'| filter attributes.session.id = "{args.session_id}"' if args.session_id else ""
        q = f"""
        fields
          @timestamp as ts,
          timeUnixNano as ts_nano,
          traceId as trace_id,
          spanId as span_id,
          attributes.session.id as session_id,
          body.input as body_input,
          body.output as body_output,
          body
        | filter @logStream = "otel-rt-logs"
        | filter ispresent(traceId) and traceId != ""
        {session_filter_otel}
        | sort @timestamp asc
        """
    else:
        # This should is not expected to happen due to argparse choices validation
        raise ValueError(f"Invalid log-stream-kind: {args.log_stream_kind}")

    # Export from all selected log groups
    all_turns = []
    for log_group in log_groups:
        print(f"Exporting from log group: {log_group}")
        
        # Initialize logs client with proper region/profile
        session = boto3.Session(profile_name=args.profile, region_name=args.region)
        logs_client = session.client('logs')
        
        r = run_logs_insights(q, [log_group], start_ts, end_ts, logs_client=logs_client)
        if r["status"] != "Complete":
            print(f"Warning: Logs Insights failed for {log_group}: {r['status']}")
            continue

        turns = rows_to_dicts(r.get("results", []))
        all_turns.extend(turns)

    # Basic normalization
    normalized = []
    for t in all_turns:
        trace_id = t.get("trace_id") or t.get("traceId")
        if not trace_id:
            continue
        
        # Handle different timestamp formats
        timestamp = t.get("ts")
        if not timestamp and t.get("ts_nano"):
            # Convert nanoseconds to ISO format
            # CloudWatch Insights returns timeUnixNano as string, not number
            try:
                ts_nano_str = t.get("ts_nano")
                if ts_nano_str:
                    ts_nano = int(ts_nano_str)
                    from datetime import datetime
                    timestamp = datetime.utcfromtimestamp(ts_nano / 1e9).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                else:
                    timestamp = None
            except (ValueError, TypeError) as e:
                print(f"Warning: Failed to convert timestamp {t.get('ts_nano')}: {e}")
                timestamp = None
        
        # Extract user query with fallback logic for OTEL format
        user_query = ""
        if t.get("user_query"):
            # Direct field from runtime/application logs
            user_query = t.get("user_query")
        elif t.get("body_input"):
            # OTEL format: body.input contains nested JSON
            try:
                body_input = t.get("body_input")
                if isinstance(body_input, str):
                    body_input = json.loads(body_input)
                
                if isinstance(body_input, dict):
                    # Extract from messages array
                    messages = body_input.get("messages", [])
                    if messages and isinstance(messages, list):
                        # Get first user message
                        for msg in messages:
                            if msg.get("role") == "user":
                                content = msg.get("content", {})
                                if isinstance(content, dict):
                                    content_str = content.get("content", "")
                                    if isinstance(content_str, str):
                                        # Content is a JSON array string
                                        try:
                                            content_arr = json.loads(content_str)
                                            if isinstance(content_arr, list) and len(content_arr) > 0:
                                                user_query = content_arr[0].get("text", "")
                                                break
                                        except:
                                            user_query = content_str
                                            break
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                # If parsing fails, leave user_query empty
                pass
        
        normalized.append({
            "timestamp": timestamp,
            "session_id": t.get("session_id"),
            "trace_id": trace_id,
            "request_id": t.get("request_id"),
            "span_id": t.get("span_id"),
            "user_query": user_query or "",
        })

    out = {
        "run_id": args.session_id if args.session_id else "ALL_SESSIONS",
        "window": {"start_epoch": start_ts, "end_epoch": end_ts},
        "turns": normalized
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(normalized)} turns to {args.out}")

if __name__ == "__main__":
    main()
