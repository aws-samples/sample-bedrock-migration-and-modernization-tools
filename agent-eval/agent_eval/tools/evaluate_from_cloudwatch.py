#!/usr/bin/env python3
"""
CloudWatch → Evaluation Pipeline

Single command to pull agent traces from CloudWatch, convert to evaluation
format, and run the evaluation framework.

Usage:
    python -m agent_eval.tools.evaluate_from_cloudwatch \
        --log-group /aws/bedrock/agent-eval-test \
        --judge-config judges.yaml \
        --output-dir ./output

    # With custom rubrics and time range
    python -m agent_eval.tools.evaluate_from_cloudwatch \
        --log-group /aws/bedrock/agent-eval-test \
        --judge-config judges.real.single.yaml \
        --rubrics agent_rubrics.yaml \
        --minutes 60 \
        --output-dir ./output
"""

import argparse
import json
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

try:
    import boto3
except ImportError:
    print("Error: boto3 required. Install with: pip install boto3")
    sys.exit(1)


def extract_traces_from_cloudwatch(log_group: str, region: str, minutes: int) -> list[dict]:
    """Pull model invocation logs and reconstruct conversation traces."""
    client = boto3.client('logs', region_name=region)

    end_time = int(time.time() * 1000)
    start_time = int((time.time() - minutes * 60) * 1000)

    # Get all log events
    streams = client.describe_log_streams(
        logGroupName=log_group, orderBy="LastEventTime", descending=True, limit=10
    )

    events = []
    for stream in streams.get('logStreams', []):
        response = client.get_log_events(
            logGroupName=log_group,
            logStreamName=stream['logStreamName'],
            startTime=start_time,
            endTime=end_time,
            startFromHead=True,
        )
        for e in response.get('events', []):
            try:
                parsed = json.loads(e['message'])
                parsed['_cw_timestamp'] = e['timestamp']
                events.append(parsed)
            except json.JSONDecodeError:
                continue

    if not events:
        return []

    # Group events into conversation turns by extracting user messages and responses
    trace_events = []
    for event in events:
        ts = event.get('timestamp', '')
        request_id = event.get('requestId', '')
        model_id = event.get('modelId', '')
        input_body = (event.get('input') or {}).get('inputBodyJson', {})
        output_body = (event.get('output') or {}).get('outputBodyJson', {})

        messages = input_body.get('messages', [])
        user_msg = None
        for m in messages:
            if m.get('role') == 'user':
                content = m.get('content', '')
                if isinstance(content, list):
                    content = ' '.join(c.get('text', '') for c in content if isinstance(c, dict))
                elif isinstance(content, str):
                    pass
                user_msg = content

        assistant_msg = None
        if output_body:
            out_content = output_body.get('content', [])
            if isinstance(out_content, list):
                assistant_msg = ' '.join(c.get('text', '') for c in out_content if isinstance(c, dict) and 'text' in c)

        if user_msg:
            trace_events.append({
                'timestamp': ts,
                'request_id': request_id,
                'model_id': model_id,
                'user_query': user_msg[-500:],  # Last user message (truncate context)
                'assistant_response': assistant_msg,
            })

    # Build trace JSON in evaluator format
    session_id = f"cloudwatch-{log_group.replace('/', '-')}-{int(time.time())}"
    eval_events = []
    turn_counter = 0

    for te in trace_events:
        if te.get('user_query'):
            turn_counter += 1
            turn_id = f"turn-{turn_counter}"

            eval_events.append({
                "timestamp": te['timestamp'],
                "type": "user_message",
                "turn_id": turn_id,
                "content": te['user_query'],
            })

            eval_events.append({
                "timestamp": te['timestamp'],
                "type": "model_output",
                "turn_id": turn_id,
                "content": te.get('assistant_response') or '(no response captured)',
            })

    return [{
        "session_id": session_id,
        "trace_id": f"trace-{session_id}",
        "events": eval_events,
    }]


def main():
    parser = argparse.ArgumentParser(description="CloudWatch → Evaluation Pipeline")
    parser.add_argument("--log-group", required=True, help="CloudWatch log group name")
    parser.add_argument("--judge-config", required=True, help="Path to judges.yaml")
    parser.add_argument("--rubrics", help="Path to rubrics.yaml (optional)")
    parser.add_argument("--output-dir", default="./cloudwatch_eval_output", help="Output directory")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--minutes", type=int, default=60, help="Minutes of logs to pull (default: 60)")
    parser.add_argument("--export-only", action="store_true", help="Export traces without evaluating")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract traces
    print("=" * 60)
    print("STEP 1: Extracting traces from CloudWatch")
    print("=" * 60)
    print(f"Log group: {args.log_group}")
    print(f"Time range: last {args.minutes} minutes")

    traces = extract_traces_from_cloudwatch(args.log_group, args.region, args.minutes)

    if not traces:
        print("✗ No traces found in the specified time range")
        return 1

    total_events = sum(len(t.get('events', [])) for t in traces)
    print(f"✓ Extracted {len(traces)} trace(s) with {total_events} events")

    # Save extracted traces
    for i, trace in enumerate(traces):
        trace_path = output_path / f"extracted_trace_{i}.json"
        trace_path.write_text(json.dumps(trace, indent=2))
        print(f"  Saved: {trace_path}")

    if args.export_only:
        print("\n✓ Export complete (--export-only mode)")
        return 0

    # Step 2: Evaluate
    print("\n" + "=" * 60)
    print("STEP 2: Evaluating traces")
    print("=" * 60)

    results = []
    for i, trace in enumerate(traces):
        trace_path = output_path / f"extracted_trace_{i}.json"
        eval_output = output_path / f"eval_{i}"
        eval_output.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i+1}/{len(traces)}] Evaluating: {trace_path.name}")

        cmd = [
            sys.executable, "-m", "agent_eval.cli",
            "--input", str(trace_path),
            "--judge-config", args.judge_config,
            "--output-dir", str(eval_output),
        ]
        if args.rubrics:
            cmd.extend(["--rubrics", args.rubrics])
        if args.verbose:
            cmd.append("--verbose")

        result = subprocess.run(cmd, capture_output=not args.verbose)  # nosec B603 — list-form call with argparse-validated CLI args
        success = result.returncode == 0
        results.append({"trace": str(trace_path), "output": str(eval_output), "success": success})
        print(f"  {'✓' if success else '✗'} Output: {eval_output}")

    # Summary
    succeeded = sum(1 for r in results if r["success"])
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Traces extracted: {len(traces)}")
    print(f"  Evaluations succeeded: {succeeded}/{len(results)}")
    print(f"  Output: {output_path}")

    summary = {
        "log_group": args.log_group,
        "region": args.region,
        "minutes": args.minutes,
        "traces_extracted": len(traces),
        "evaluations": results,
    }
    (output_path / "cloudwatch_eval_summary.json").write_text(json.dumps(summary, indent=2))

    if args.verbose:
        print(f"\nView results: streamlit run ui/app.py")
        print(f"  → Point to: {output_path}")

    return 0 if succeeded == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
