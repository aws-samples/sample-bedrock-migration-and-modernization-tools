# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fetch AgentCore cost data from CloudWatch Metrics API.

Two metric namespaces provide exact cost data:
  - AWS/Bedrock: InputTokenCount, OutputTokenCount, Invocations, InvocationLatency (by ModelId)
  - bedrock-agentcore: gen_ai.client.token.usage (by model + token type), http.server.duration

Prerequisites:
  - Valid AWS credentials with cloudwatch:GetMetricData and cloudwatch:ListMetrics
  - Agent must have been invoked (metrics appear after first invocation)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CloudWatchMetricsResult:
    """Aggregated metrics fetched from CloudWatch."""
    input_tokens: int = 0
    output_tokens: int = 0
    invocations: int = 0
    avg_latency_ms: float = 0.0
    total_duration_ms: float = 0.0
    model_id: Optional[str] = None
    token_breakdown_by_model: Dict[str, Dict[str, int]] = field(default_factory=dict)
    source: str = "cloudwatch_metrics"


def fetch_metrics_from_cloudwatch(
    region: str,
    model_id: Optional[str] = None,
    hours_back: int = 1,
) -> CloudWatchMetricsResult:
    """
    Fetch exact token counts and latency from CloudWatch Metrics API.

    Queries AWS/Bedrock namespace for InputTokenCount, OutputTokenCount,
    Invocations, and InvocationLatency. Optionally filters by ModelId.

    Args:
        region: AWS region
        model_id: Optional Bedrock model ID to filter by
        hours_back: How many hours back to query (default 1)

    Returns:
        CloudWatchMetricsResult with aggregated metrics
    """
    import boto3

    cw = boto3.client("cloudwatch", region_name=region)
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours_back)

    dims = [{"Name": "ModelId", "Value": model_id}] if model_id else []

    queries = [
        ("input_tokens", "InputTokenCount", "Sum"),
        ("output_tokens", "OutputTokenCount", "Sum"),
        ("invocations", "Invocations", "Sum"),
        ("latency", "InvocationLatency", "Average"),
    ]

    metric_data_queries = [
        {
            "Id": qid,
            "MetricStat": {
                "Metric": {
                    "Namespace": "AWS/Bedrock",
                    "MetricName": metric_name,
                    **({"Dimensions": dims} if dims else {}),
                },
                "Period": hours_back * 3600,
                "Stat": stat,
            },
        }
        for qid, metric_name, stat in queries
    ]

    resp = cw.get_metric_data(
        MetricDataQueries=metric_data_queries,
        StartTime=start,
        EndTime=now,
    )

    result = CloudWatchMetricsResult(model_id=model_id)
    for r in resp.get("MetricDataResults", []):
        total = sum(r.get("Values", []))
        if r["Id"] == "input_tokens":
            result.input_tokens = int(total)
        elif r["Id"] == "output_tokens":
            result.output_tokens = int(total)
        elif r["Id"] == "invocations":
            result.invocations = int(total)
        elif r["Id"] == "latency":
            result.avg_latency_ms = total

    result.total_duration_ms = result.avg_latency_ms * result.invocations

    # Discover per-model breakdown if no model_id filter
    if not model_id:
        result.token_breakdown_by_model = _fetch_per_model_breakdown(cw, start, now, hours_back)

    logger.info(
        "Fetched from CloudWatch: %d input tokens, %d output tokens, %d invocations",
        result.input_tokens, result.output_tokens, result.invocations,
    )
    return result


def _fetch_per_model_breakdown(
    cw, start: datetime, end: datetime, hours_back: int,
) -> Dict[str, Dict[str, int]]:
    """Discover all models and get per-model token counts."""
    breakdown = {}
    resp = cw.list_metrics(Namespace="AWS/Bedrock", MetricName="InputTokenCount")
    model_ids = set()
    for m in resp.get("Metrics", []):
        for d in m.get("Dimensions", []):
            if d["Name"] == "ModelId":
                model_ids.add(d["Value"])

    if not model_ids:
        return breakdown

    # Query all models in one batch
    queries = []
    for i, mid in enumerate(model_ids):
        safe_id = f"m{i}"
        queries.append({
            "Id": f"in_{safe_id}",
            "MetricStat": {
                "Metric": {
                    "Namespace": "AWS/Bedrock",
                    "MetricName": "InputTokenCount",
                    "Dimensions": [{"Name": "ModelId", "Value": mid}],
                },
                "Period": hours_back * 3600,
                "Stat": "Sum",
            },
            "Label": mid,
        })
        queries.append({
            "Id": f"out_{safe_id}",
            "MetricStat": {
                "Metric": {
                    "Namespace": "AWS/Bedrock",
                    "MetricName": "OutputTokenCount",
                    "Dimensions": [{"Name": "ModelId", "Value": mid}],
                },
                "Period": hours_back * 3600,
                "Stat": "Sum",
            },
            "Label": mid,
        })

    resp = cw.get_metric_data(
        MetricDataQueries=queries, StartTime=start, EndTime=end,
    )

    # Parse results
    model_list = list(model_ids)
    for r in resp.get("MetricDataResults", []):
        idx = int(r["Id"].split("_m")[1])
        mid = model_list[idx]
        total = int(sum(r.get("Values", [])))
        entry = breakdown.setdefault(mid, {"input": 0, "output": 0})
        if r["Id"].startswith("in_"):
            entry["input"] = total
        else:
            entry["output"] = total

    return breakdown


# Keep backward-compatible aliases for the upload path
def parse_uploaded_usage_file(content: str) -> List[Any]:
    """Parse uploaded usage file. Kept for backward compatibility."""
    from agent_eval.evaluators.agentcore_cost import parse_usage_logs
    import json
    content = content.strip()
    if content.startswith("["):
        try:
            arr = json.loads(content)
            lines = "\n".join(json.dumps(r) for r in arr)
            return parse_usage_logs(lines)
        except (json.JSONDecodeError, TypeError):
            pass
    return parse_usage_logs(content)
