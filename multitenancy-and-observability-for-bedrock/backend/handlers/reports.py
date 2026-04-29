# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Lambda handler for report generation (placeholder).

Will be fully implemented in Phase 5.

Environment variables:
    TENANTS_TABLE - DynamoDB table for profile records
"""

import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event, context):
    """Main Lambda handler - placeholder for report operations."""
    http_method = event.get("httpMethod", "")
    path = event.get("path", "")

    return _response(501, {
        "error": "Reports are not yet implemented",
        "message": "Report generation will be available in a future release.",
    })


def _response(status_code: int, body: dict) -> dict:
    """Build API Gateway proxy response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        },
        "body": json.dumps(body, default=str),
    }
