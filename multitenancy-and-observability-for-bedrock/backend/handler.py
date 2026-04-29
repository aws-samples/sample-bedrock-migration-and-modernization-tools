# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Backend Lambda entry point — routes to the appropriate handler based on path."""

import json
import logging
import os
import sys

# helps shared modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from handlers import profiles, discovery, dashboards, alerts, cost_explorer, metrics, reports

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """Route API Gateway events to the correct handler module."""
    path = event.get("path", "")
    logger.info("Backend router: %s %s", event.get("httpMethod", ""), path)

    if "/reports" in path:
        return reports.handler(event, context)
    elif "/metrics" in path:
        return metrics.handler(event, context)
    elif "/cost-explorer" in path:
        return cost_explorer.handler(event, context)
    elif "/dashboards" in path:
        return dashboards.handler(event, context)
    elif "/alerts" in path:
        return alerts.handler(event, context)
    elif "/discovery" in path:
        return discovery.handler(event, context)
    elif "/profiles" in path:
        return profiles.handler(event, context)
    else:
        return {
            "statusCode": 404,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"No handler for path: {path}"}),
        }
