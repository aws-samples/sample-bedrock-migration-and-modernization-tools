# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Lambda handler for CloudWatch metric queries.

Proxies CloudWatch GetMetricData calls scoped to ISVBedrock/Gateway namespace.

Environment variables:
    TENANTS_TABLE - DynamoDB table for profile records
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone

import boto3

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared import dynamo_utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TENANTS_TABLE = os.environ.get("TENANTS_TABLE", "")
METRIC_NAMESPACE = "ISVBedrock/Gateway"


def handler(event, context):
    """Main Lambda handler - routes based on path."""
    http_method = event.get("httpMethod", "")
    path = event.get("path", "")

    try:
        if http_method == "GET" and "/metrics/query" in path:
            return _query_metrics(event)
        elif http_method == "GET" and "/metrics/list" in path:
            return _list_metrics(event)

        return _response(404, {"error": "Not found"})

    except Exception as exc:
        logger.exception("Unhandled error in metrics handler")
        return _response(500, {"error": str(exc)})


def _query_metrics(event):
    """GET /metrics/query - Query CloudWatch metrics.

    Query params:
        tenant_id (required): Profile to scope metrics to
        metric_name (required): e.g. InputTokensCost, OutputTokensCost, InvocationLatencyMs
        stat (optional): Sum, Average, Maximum, Minimum, p50, p90, p99 (default: Sum)
        period (optional): seconds (default: 300)
        hours (optional): lookback window in hours (default: 24)
        group_by (optional): dimension to group by (TenantId, ModelId, InferenceProfile)
    """
    params = event.get("queryStringParameters") or {}
    profile_id = params.get("tenant_id", "").strip()
    metric_name = params.get("metric_name", "").strip()
    stat = params.get("stat", "Sum").strip()
    period = int(params.get("period", "300"))
    hours = int(params.get("hours", "24"))

    if not profile_id or not metric_name:
        return _response(400, {"error": "tenant_id and metric_name are required"})

    # Validate profile exists
    profile = dynamo_utils.get_profile(TENANTS_TABLE, profile_id)
    if not profile:
        return _response(404, {"error": f"Profile {profile_id} not found"})

    region = profile.get("region", os.environ.get("AWS_REGION", "us-east-1"))
    cloudwatch = boto3.client("cloudwatch", region_name=region)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours)

    try:
        response = cloudwatch.get_metric_data(
            MetricDataQueries=[
                {
                    "Id": "m0",
                    "MetricStat": {
                        "Metric": {
                            "Namespace": METRIC_NAMESPACE,
                            "MetricName": metric_name,
                            "Dimensions": [
                                {"Name": "TenantId", "Value": profile_id},
                            ],
                        },
                        "Period": period,
                        "Stat": stat,
                    },
                    "ReturnData": True,
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
        )
    except Exception as exc:
        logger.error("Failed to query CloudWatch metrics: %s", exc)
        return _response(502, {"error": f"Failed to query metrics: {exc}"})

    # Transform response
    results = response.get("MetricDataResults", [])
    datapoints = []
    if results:
        timestamps = results[0].get("Timestamps", [])
        values = results[0].get("Values", [])
        for ts, val in zip(timestamps, values):
            datapoints.append({
                "timestamp": ts.isoformat(),
                "value": val,
            })
        # Sort by timestamp ascending
        datapoints.sort(key=lambda d: d["timestamp"])

    return _response(200, {
        "profile_id": profile_id,
        "metric_name": metric_name,
        "stat": stat,
        "period": period,
        "datapoints": datapoints,
        "count": len(datapoints),
    })


def _list_metrics(event):
    """GET /metrics/list - List available metrics for a profile."""
    params = event.get("queryStringParameters") or {}
    profile_id = params.get("tenant_id", "").strip()

    if not profile_id:
        return _response(400, {"error": "tenant_id is required"})

    profile = dynamo_utils.get_profile(TENANTS_TABLE, profile_id)
    if not profile:
        return _response(404, {"error": f"Profile {profile_id} not found"})

    region = profile.get("region", os.environ.get("AWS_REGION", "us-east-1"))
    cloudwatch = boto3.client("cloudwatch", region_name=region)

    try:
        response = cloudwatch.list_metrics(
            Namespace=METRIC_NAMESPACE,
            Dimensions=[
                {"Name": "TenantId", "Value": profile_id},
            ],
        )
    except Exception as exc:
        logger.error("Failed to list metrics: %s", exc)
        return _response(502, {"error": f"Failed to list metrics: {exc}"})

    metrics = []
    for m in response.get("Metrics", []):
        metrics.append({
            "metric_name": m.get("MetricName", ""),
            "dimensions": {
                d["Name"]: d["Value"] for d in m.get("Dimensions", [])
            },
        })

    return _response(200, {
        "profile_id": profile_id,
        "metrics": metrics,
        "count": len(metrics),
    })


def _response(status_code: int, body: dict) -> dict:
    """Build API Gateway proxy response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "GET,OPTIONS",
        },
        "body": json.dumps(body, default=str),
    }
