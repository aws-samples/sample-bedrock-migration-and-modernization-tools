# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Amazon Bedrock Invocation Log Extraction

Utilities for extracting and parsing Amazon Bedrock invocation logs
from CloudWatch Logs or S3.

Amazon Bedrock invocation logging captures every model request/response when enabled.
Logs can be sent to CloudWatch Logs, S3, or both.

Prerequisites:
    - Invocation logging must be enabled via the Amazon Bedrock console or API
    - IAM permissions for CloudWatch Logs read access or S3 read access

Actual log schema (verified from CloudWatch):
    {
        "schemaType": "ModelInvocationLog",
        "schemaVersion": "1.0",
        "timestamp": "2026-03-27T04:37:00Z",
        "accountId": "123456789012",
        "region": "us-east-1",
        "requestId": "uuid",
        "operation": "Converse",
        "modelId": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "identity": { "arn": "arn:aws:iam::..." },
        "input": {
            "inputContentType": "application/json",
            "inputTokenCount": 150,
            "inputBodyJson": { "messages": [...], "system": [...], "inferenceConfig": {...} }
        },
        "output": {
            "outputContentType": "application/json",
            "outputTokenCount": 45,
            "outputBodyJson": {
                "output": { "message": { "role": "assistant", "content": [...] } },
                "stopReason": "end_turn",
                "metrics": { "latencyMs": 2500 },
                "usage": { "inputTokens": 150, "outputTokens": 45 }
            }
        }
    }

Usage:
    from src.invocation_logs import (
        check_logging_config,
        print_logging_status,
        extract_logs_from_cloudwatch,
        extract_logs_from_s3,
        sample_logs,
    )
"""

import gzip
import json
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import boto3


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Amazon BedrockInvocationLog:
    """Parsed Amazon Bedrock invocation log entry."""
    request_id: str
    timestamp: str
    model_id: str
    operation: str
    region: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    input_body: Dict[str, Any]
    output_body: Dict[str, Any]
    status: str = "Success"
    error_code: Optional[str] = None

    @property
    def prompt(self) -> str:
        """Extract the user prompt from the input body."""
        messages = self.input_body.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            return block["text"]
                elif isinstance(content, str):
                    return content
        return ""

    @property
    def system_prompt(self) -> str:
        """Extract the system prompt from the input body."""
        system = self.input_body.get("system", [])
        if isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and "text" in block:
                    return block["text"]
        return ""

    @property
    def response(self) -> str:
        """Extract the model response from the output body."""
        output = self.output_body.get("output", {})
        if isinstance(output, dict):
            message = output.get("message", {})
            content = message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        return block["text"]
        return ""


def _parse_log_record(record: Dict[str, Any]) -> Amazon BedrockInvocationLog:
    """Parse a raw log record dict into a Amazon BedrockInvocationLog."""
    inp = record.get("input", {})
    out = record.get("output", {})
    output_body = out.get("outputBodyJson", {})

    # Latency is inside output.outputBodyJson.metrics.latencyMs
    latency_ms = output_body.get("metrics", {}).get("latencyMs", 0)

    return Amazon BedrockInvocationLog(
        request_id=record.get("requestId", ""),
        timestamp=record.get("timestamp", ""),
        model_id=record.get("modelId", ""),
        operation=record.get("operation", ""),
        region=record.get("region", ""),
        input_tokens=inp.get("inputTokenCount", 0),
        output_tokens=out.get("outputTokenCount", 0),
        latency_ms=float(latency_ms),
        input_body=inp.get("inputBodyJson", {}),
        output_body=output_body,
        status="Error" if record.get("errorCode") else "Success",
        error_code=record.get("errorCode"),
    )


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def check_logging_config(region: str = "us-east-1") -> Dict[str, Any]:
    """
    Check the current Amazon Bedrock invocation logging configuration.

    Returns:
        Dict with logging configuration, or empty dict if not configured.
    """
    client = boto3.client("bedrock", region_name=region)
    try:
        response = client.get_model_invocation_logging_configuration()
        return response.get("loggingConfig", {})
    except Exception as e:
        print(f"Error checking logging config: {e}")
        return {}


def print_logging_status(region: str = "us-east-1") -> None:
    """Print a human-readable summary of the logging configuration."""
    config = check_logging_config(region)

    if not config:
        print("Invocation logging: NOT CONFIGURED")
        return

    print("Invocation logging: ENABLED")
    s3_config = config.get("s3Config", {})
    if s3_config:
        print(f"  S3 bucket:          {s3_config.get('bucketName', 'N/A')}")
        print(f"  S3 prefix:          {s3_config.get('keyPrefix', 'N/A')}")

    cw_config = config.get("cloudWatchConfig", {})
    if cw_config:
        print(f"  CloudWatch group:   {cw_config.get('logGroupName', 'N/A')}")


# ============================================================================
# CLOUDWATCH LOG EXTRACTION
# ============================================================================

def extract_logs_from_cloudwatch(
    log_group: str = "/aws/bedrock/model-invocation-logs",
    log_stream: str = "aws/bedrock/modelinvocations",
    model_id_filter: Optional[str] = None,
    max_records: int = 100,
    hours_back: Optional[int] = None,
    region: str = "us-east-1",
) -> List[Amazon BedrockInvocationLog]:
    """
    Extract and parse Amazon Bedrock invocation logs from CloudWatch Logs.

    Args:
        log_group: CloudWatch log group name
        log_stream: CloudWatch log stream name
        model_id_filter: Optional model ID to filter by (substring match)
        max_records: Maximum number of records to return
        hours_back: If set, only return logs from the last N hours
        region: AWS region

    Returns:
        List of parsed Amazon BedrockInvocationLog entries
    """
    logs_client = boto3.client("logs", region_name=region)

    kwargs = {
        "logGroupName": log_group,
        "logStreamName": log_stream,
        "limit": min(max_records * 2, 10000),  # over-fetch to account for filtering
        "startFromHead": False,  # most recent first
    }

    if hours_back:
        kwargs["startTime"] = int(
            (datetime.now(timezone.utc) - timedelta(hours=hours_back)).timestamp() * 1000
        )
        kwargs["endTime"] = int(datetime.now(timezone.utc).timestamp() * 1000)

    logs: List[Amazon BedrockInvocationLog] = []
    next_token = None

    while len(logs) < max_records:
        if next_token:
            kwargs["nextToken"] = next_token

        response = logs_client.get_log_events(**kwargs)
        events = response.get("events", [])

        if not events:
            break

        for event in events:
            if len(logs) >= max_records:
                break

            try:
                record = json.loads(event["message"])
            except json.JSONDecodeError:
                continue

            # Apply model filter
            if model_id_filter and model_id_filter not in record.get("modelId", ""):
                continue

            logs.append(_parse_log_record(record))

        # Check for pagination
        new_token = response.get("nextForwardToken") or response.get("nextBackwardToken")
        if new_token == next_token:
            break  # no more pages
        next_token = new_token

    print(f"Extracted {len(logs)} invocation logs from CloudWatch")
    return logs


# ============================================================================
# S3 LOG EXTRACTION
# ============================================================================

def list_log_files(
    s3_bucket: str,
    s3_prefix: str = "bedrock-invocation-logs/",
    hours_back: int = 24,
    region: str = "us-east-1",
) -> List[str]:
    """List available invocation log files in S3."""
    s3 = boto3.client("s3", region_name=region)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)

    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            if obj["LastModified"].replace(tzinfo=None) >= cutoff.replace(tzinfo=None):
                keys.append(obj["Key"])

    return keys


def extract_logs_from_s3(
    s3_bucket: str,
    s3_prefix: str = "bedrock-invocation-logs/",
    hours_back: int = 24,
    model_id_filter: Optional[str] = None,
    max_records: int = 1000,
    region: str = "us-east-1",
) -> List[Amazon BedrockInvocationLog]:
    """
    Extract and parse Amazon Bedrock invocation logs from S3.

    Args:
        s3_bucket: S3 bucket where logs are stored
        s3_prefix: Key prefix within the bucket
        hours_back: How many hours back to extract
        model_id_filter: Optional model ID to filter by (substring match)
        max_records: Maximum number of records to return
        region: AWS region

    Returns:
        List of parsed Amazon BedrockInvocationLog entries
    """
    s3 = boto3.client("s3", region_name=region)
    keys = list_log_files(s3_bucket, s3_prefix, hours_back, region)

    print(f"Found {len(keys)} log files in s3://{s3_bucket}/{s3_prefix}")

    logs: List[Amazon BedrockInvocationLog] = []

    for key in keys:
        if len(logs) >= max_records:
            break

        try:
            response = s3.get_object(Bucket=s3_bucket, Key=key)
            body = response["Body"].read()

            if key.endswith(".gz"):
                body = gzip.decompress(body)

            content = body.decode("utf-8")

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                data = [json.loads(line) for line in content.strip().split("\n") if line.strip()]

            if isinstance(data, dict):
                data = [data]

            for record in data:
                if model_id_filter and model_id_filter not in record.get("modelId", ""):
                    continue

                logs.append(_parse_log_record(record))

                if len(logs) >= max_records:
                    break

        except Exception as e:
            print(f"  Error reading {key}: {e}")
            continue

    print(f"Extracted {len(logs)} invocation logs from S3")
    return logs


# ============================================================================
# SAMPLING
# ============================================================================

def sample_logs(
    logs: List[Amazon BedrockInvocationLog],
    n: int = 10,
    seed: int = 42,
) -> List[Amazon BedrockInvocationLog]:
    """
    Randomly sample n logs from the extracted records.

    Args:
        logs: List of Amazon BedrockInvocationLog entries
        n: Number of samples to select
        seed: Random seed for reproducibility

    Returns:
        List of sampled Amazon BedrockInvocationLog entries
    """
    random.seed(seed)
    n = min(n, len(logs))
    sampled = random.sample(logs, n)
    print(f"Sampled {len(sampled)} entries from {len(logs)} logs")
    return sampled
