# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Lambda handler for Cost Explorer operations.

Routes based on httpMethod + resource path from API Gateway proxy integration.

Environment variables:
    TENANTS_TABLE - DynamoDB table for profile records
"""

import json
import logging
import os

import boto3

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TENANTS_TABLE = os.environ.get("TENANTS_TABLE", "")


def handler(event, context):
    """Main Lambda handler - routes to appropriate operation."""
    http_method = event.get("httpMethod", "")
    path = event.get("path", "")

    try:
        if "/profile-costs" in path and http_method == "GET":
            return _get_profile_costs(event)

        return _response(404, {"error": "Not found"})

    except Exception as exc:
        logger.exception("Unhandled error in cost_explorer handler")
        return _response(500, {"error": str(exc)})


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def _get_profile_costs(event):
    """GET /cost-explorer/profile-costs - Get cost data grouped by profile."""
    params = event.get("queryStringParameters") or {}
    start_date = params.get("start_date")  # Required YYYY-MM-DD
    end_date = params.get("end_date")      # Required YYYY-MM-DD
    granularity = params.get("granularity", "DAILY")  # DAILY or MONTHLY
    profile_id = params.get("tenant_id")   # Optional

    if not start_date or not end_date:
        return _response(400, {"error": "start_date and end_date are required (YYYY-MM-DD)"})

    ce = boto3.client("ce", region_name="us-east-1")  # CE only works in us-east-1

    filter_expr = {"Tags": {"Key": "managed_by", "Values": ["isv-bedrock-observability"]}}
    if profile_id:
        filter_expr = {"And": [
            {"Tags": {"Key": "managed_by", "Values": ["isv-bedrock-observability"]}},
            {"Tags": {"Key": "tenant_id", "Values": [profile_id]}},
        ]}

    response = ce.get_cost_and_usage(
        TimePeriod={"Start": start_date, "End": end_date},
        Granularity=granularity,
        Metrics=["UnblendedCost", "UsageQuantity"],
        GroupBy=[{"Type": "TAG", "Key": "tenant_id"}],
        Filter=filter_expr,
    )

    # Transform into a clean response
    costs = []
    for period in response.get("ResultsByTime", []):
        for group in period.get("Groups", []):
            tag_value = group["Keys"][0].replace("tenant_id$", "")
            costs.append({
                "tenant_id": tag_value,
                "period_start": period["TimePeriod"]["Start"],
                "period_end": period["TimePeriod"]["End"],
                "cost": float(group["Metrics"]["UnblendedCost"]["Amount"]),
                "usage_quantity": float(group["Metrics"]["UsageQuantity"]["Amount"]),
                "currency": group["Metrics"]["UnblendedCost"]["Unit"],
            })

    return _response(200, {"costs": costs, "count": len(costs)})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_body(event) -> dict:
    """Parse JSON body from API Gateway event."""
    body = event.get("body", "")
    if not body:
        return {}
    if isinstance(body, str):
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {}
    return body


def _response(status_code: int, body: dict) -> dict:
    """Build API Gateway proxy response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
        },
        "body": json.dumps(body, default=str),
    }
