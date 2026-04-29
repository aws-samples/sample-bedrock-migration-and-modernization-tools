# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Lambda handler for alert CRUD operations.

Routes based on httpMethod + resource path from API Gateway proxy integration.

Environment variables:
    ALERTS_TABLE  - DynamoDB table for alert records
    TENANTS_TABLE - DynamoDB table for profile records
"""

import json
import logging
import os
import uuid
import datetime

import boto3

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared import dynamo_utils
from shared.tag_utils import parse_tag_filters, get_profile_ids_for_tags

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ALERTS_TABLE = os.environ.get("ALERTS_TABLE", "")
TENANTS_TABLE = os.environ.get("TENANTS_TABLE", "")

METRIC_NAMESPACE = "ISVBedrock/Gateway"

# Valid metric names that can be used in alerts
VALID_METRIC_NAMES = {
    "InputTokensCost",
    "OutputTokensCost",
    "InputTokens",
    "OutputTokens",
    "InvocationSuccess",
    "InvocationFailure",
    "InvocationLatencyMs",
}

# Valid threshold modes
VALID_THRESHOLD_MODES = {"absolute", "percentage_increase", "percentage_decrease"}

# Valid comparison operators
VALID_COMPARISONS = {
    "GreaterThanOrEqualToThreshold",
    "LessThanOrEqualToThreshold",
    "GreaterThanThreshold",
    "LessThanThreshold",
}

# Valid action types
VALID_ACTIONS = {"notify", "throttle", "suspend"}

# Default periods per alert type (legacy compatibility + new defaults)
_DEFAULT_PERIODS = {
    "cost_threshold": 300,
    "tokens_per_minute": 60,
    "requests_per_minute": 60,
}

# Legacy metric definitions per alert type (kept for backward compatibility)
_ALERT_METRICS = {
    "cost_threshold": {
        "metrics": [
            {"name": "InputTokensCost", "id": "itc"},
            {"name": "OutputTokensCost", "id": "otc"},
        ],
        "expression_id": "totalCost",
        "expression": "itc + otc",
    },
    "tokens_per_minute": {
        "metrics": [
            {"name": "InputTokens", "id": "it"},
            {"name": "OutputTokens", "id": "ot"},
        ],
        "expression_id": "totalTokens",
        "expression": "it + ot",
    },
    "requests_per_minute": {
        "metrics": [
            {"name": "InvocationSuccess", "id": "is1"},
            {"name": "InvocationFailure", "id": "if1"},
        ],
        "expression_id": "totalRequests",
        "expression": "is1 + if1",
    },
    "metric_threshold": {
        "metrics": [
            {"name": "InputTokensCost", "id": "itc"},
        ],
        "expression_id": "metricVal",
        "expression": "itc",
    },
    "anomaly_detection": {
        "metrics": [
            {"name": "InputTokensCost", "id": "itc"},
        ],
        "expression_id": "metricVal",
        "expression": "itc",
    },
    "trend_slope": {
        "metrics": [
            {"name": "InputTokensCost", "id": "itc"},
        ],
        "expression_id": "metricVal",
        "expression": "itc",
    },
    "arima_deviation": {
        "metrics": [
            {"name": "InputTokensCost", "id": "itc"},
        ],
        "expression_id": "metricVal",
        "expression": "itc",
    },
}


def handler(event, context):
    """Main Lambda handler - routes to appropriate operation."""
    http_method = event.get("httpMethod", "")
    path = event.get("path", "")
    path_params = event.get("pathParameters") or {}
    alert_id = path_params.get("alert_id")
    if not alert_id:
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[0] == "alerts":
            alert_id = parts[1]

    try:
        if alert_id:
            if http_method == "GET":
                return _get_alert(alert_id)
            elif http_method == "PUT":
                return _update_alert(alert_id, event)
            elif http_method == "DELETE":
                return _delete_alert(alert_id)
        else:
            if http_method == "POST":
                return _create_alert(event)
            elif http_method == "GET":
                return _list_alerts(event)

        return _response(404, {"error": "Not found"})

    except Exception as exc:
        logger.exception("Unhandled error in alerts handler")
        return _response(500, {"error": str(exc)})


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def _create_alert(event):
    """POST /alerts - Create a CloudWatch alarm for a profile.

    Accepts the enhanced alert body with:
      - alert_name (required): User-defined name
      - tenant_id (required): Tenant to scope the alert to
      - alert_type (required): One of cost_threshold, tokens_per_minute, requests_per_minute
      - metric_name (optional): Specific metric to alert on; if not provided,
        falls back to the legacy composite expression for the alert_type
      - threshold_mode: absolute | percentage_increase | percentage_decrease
      - threshold_value (required): Numeric threshold
      - comparison: CloudWatch comparison operator
      - action: notify | throttle | suspend
      - action_config: Config for the action (email, auto_recover_minutes)
      - period: Evaluation period in seconds
      - dashboard_id / widget_id: Optional link to a dashboard widget
    """
    body = _parse_body(event)
    if not body:
        return _response(400, {"error": "Invalid or missing request body"})

    # --- Required fields ---
    alert_name = body.get("alert_name", "").strip()
    tenant_id = body.get("tenant_id", "").strip()
    alert_type = body.get("alert_type", "").strip()
    threshold_value = body.get("threshold_value") or body.get("threshold")

    if not alert_name or not tenant_id or not alert_type or threshold_value is None:
        return _response(400, {
            "error": "alert_name, tenant_id, alert_type, and threshold_value are required"
        })

    if alert_type not in _ALERT_METRICS:
        return _response(400, {
            "error": f"Invalid alert_type. Must be one of: {', '.join(_ALERT_METRICS.keys())}"
        })

    threshold_value = float(threshold_value)

    # --- Optional fields with defaults ---
    metric_name = body.get("metric_name", "").strip() or None
    if metric_name and metric_name not in VALID_METRIC_NAMES:
        return _response(400, {
            "error": f"Invalid metric_name. Must be one of: {', '.join(sorted(VALID_METRIC_NAMES))}"
        })

    threshold_mode = body.get("threshold_mode", "absolute").strip()
    if threshold_mode not in VALID_THRESHOLD_MODES:
        return _response(400, {
            "error": f"Invalid threshold_mode. Must be one of: {', '.join(VALID_THRESHOLD_MODES)}"
        })

    comparison = body.get("comparison", "GreaterThanOrEqualToThreshold").strip()
    if comparison not in VALID_COMPARISONS:
        return _response(400, {
            "error": f"Invalid comparison. Must be one of: {', '.join(VALID_COMPARISONS)}"
        })

    action = body.get("action", "notify").strip()
    if action not in VALID_ACTIONS:
        return _response(400, {
            "error": f"Invalid action. Must be one of: {', '.join(VALID_ACTIONS)}"
        })

    action_config = body.get("action_config") or {}
    email = action_config.get("email", body.get("email", "")).strip()
    auto_recover_minutes = action_config.get("auto_recover_minutes")

    period = body.get("period")
    if period is None:
        period = _DEFAULT_PERIODS.get(alert_type, 300)
    else:
        period = int(period)

    dashboard_id = body.get("dashboard_id") or None
    widget_id = body.get("widget_id") or None
    tag_dimensions = body.get("tag_dimensions") or []

    # 1. Validate profile exists
    profile = dynamo_utils.get_profile(TENANTS_TABLE, tenant_id)
    if not profile:
        return _response(404, {"error": f"Profile {tenant_id} not found"})

    profile_name = profile.get("tenant_name", "unknown")
    region = profile.get("region", os.environ.get("AWS_REGION", "us-east-1"))
    alert_id = str(uuid.uuid4())

    # 2. Create or reuse SNS topic
    sns = boto3.client("sns", region_name=region)
    topic_name = f"isv-obs-alerts-{tenant_id[:8]}"
    try:
        topic_response = sns.create_topic(Name=topic_name)
        topic_arn = topic_response["TopicArn"]
    except Exception as exc:
        logger.error("Failed to create SNS topic %s: %s", topic_name, exc)
        return _response(500, {"error": f"Failed to create SNS topic: {exc}"})

    # 3. If email provided, subscribe to SNS topic
    subscription_arn = None
    if email:
        try:
            sub_response = sns.subscribe(
                TopicArn=topic_arn,
                Protocol="email",
                Endpoint=email,
            )
            subscription_arn = sub_response.get("SubscriptionArn")
        except Exception as exc:
            logger.warning("Failed to subscribe %s to topic %s: %s", email, topic_name, exc)

    # 4. Build and create CloudWatch alarm
    alarm_name = f"isv-obs-{alert_type.replace('_', '-')}-{tenant_id[:8]}-{alert_id[:8]}"
    cloudwatch = boto3.client("cloudwatch", region_name=region)

    try:
        alarm_kwargs = _build_alarm_kwargs(
            alarm_name=alarm_name,
            alert_type=alert_type,
            tenant_id=tenant_id,
            profile_name=profile_name,
            topic_arn=topic_arn,
            threshold_value=threshold_value,
            threshold_mode=threshold_mode,
            comparison=comparison,
            period=period,
            metric_name=metric_name,
            alert_name=alert_name,
            tag_dimensions=tag_dimensions,
            profile_tags=profile.get("tags", {}),
        )
        cloudwatch.put_metric_alarm(**alarm_kwargs)
    except Exception as exc:
        logger.error("Failed to create CloudWatch alarm %s: %s", alarm_name, exc)
        return _response(500, {"error": f"Failed to create CloudWatch alarm: {exc}"})

    # 5. Store in DynamoDB
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record = {
        "alert_id": alert_id,
        "alert_name": alert_name,
        "tenant_id": tenant_id,
        "alert_type": alert_type,
        "metric_name": metric_name or "",
        "threshold_value": threshold_value,
        "threshold_mode": threshold_mode,
        "comparison": comparison,
        "action": action,
        "action_config": action_config,
        "period": period,
        "email": email,
        "alarm_name": alarm_name,
        "topic_arn": topic_arn,
        "subscription_arn": subscription_arn or "",
        "region": region,
        "dashboard_id": dashboard_id or "",
        "widget_id": widget_id or "",
        "tag_dimensions": tag_dimensions,
        "action_executed": False,
        "created_at": now,
        "updated_at": now,
    }

    # For throttle/suspend actions, store pending_action for lazy evaluation
    if action in ("throttle", "suspend"):
        record["pending_action"] = action
        if auto_recover_minutes:
            record["auto_recover_minutes"] = int(auto_recover_minutes)

    dynamo_utils.put_alert(ALERTS_TABLE, record)
    logger.info("Created alert %s (alarm: %s) for profile %s", alert_id, alarm_name, tenant_id)

    return _response(201, record)


def _list_alerts(event):
    """GET /alerts - List alerts, optionally filtered by tenant_id.

    Enriches each alert with live CloudWatch alarm state.
    For alerts with action=throttle or action=suspend, if the alarm state is
    ALARM and the profile is still active, the action is executed (lazy evaluation).

    NOTE: A proper production implementation would use EventBridge + Step Functions
    to react to alarm state changes in real time (Phase 4). This lazy evaluation
    approach helps correctness when the API is polled but may have latency.
    """
    params = event.get("queryStringParameters") or {}
    tenant_id = params.get("tenant_id")

    # Parse tag filters
    tag_filters_raw = params.get("tag_filters", "")
    tag_filters = parse_tag_filters(tag_filters_raw)

    if tenant_id:
        alerts = dynamo_utils.list_alerts_by_tenant(
            ALERTS_TABLE, "tenant_id_index", tenant_id,
        )
    else:
        table = dynamo_utils._table(ALERTS_TABLE)
        response = table.scan(Limit=100)
        alerts = [dynamo_utils._deserialize_record(item) for item in response.get("Items", [])]

    # If tag filters provided, resolve matching profile IDs and filter alerts
    if tag_filters:
        matching_profile_ids = get_profile_ids_for_tags(TENANTS_TABLE, tag_filters)
        alerts = [a for a in alerts if a.get("tenant_id") in matching_profile_ids]

    # Enrich with live alarm state
    if alerts:
        alarm_names = [a["alarm_name"] for a in alerts if a.get("alarm_name")]
        if alarm_names:
            try:
                region = alerts[0].get("region", os.environ.get("AWS_REGION", "us-east-1"))
                cloudwatch = boto3.client("cloudwatch", region_name=region)
                cw_response = cloudwatch.describe_alarms(AlarmNames=alarm_names)
                state_map = {
                    a["AlarmName"]: a["StateValue"]
                    for a in cw_response.get("MetricAlarms", [])
                }
                for alert in alerts:
                    alarm_name = alert.get("alarm_name", "")
                    alert["state"] = state_map.get(alarm_name, "UNKNOWN")
            except Exception as exc:
                logger.warning("Failed to describe alarms: %s", exc)
                for alert in alerts:
                    alert["state"] = "UNKNOWN"

        # Execute pending actions for throttle/suspend alerts in ALARM state
        _execute_pending_actions(alerts)

    return _response(200, {
        "alerts": alerts,
        "count": len(alerts),
    })


def _get_alert(alert_id):
    """GET /alerts/{alert_id} - Get a single alert.

    Enriches the alert with live alarm state. For throttle/suspend alerts
    in ALARM state, executes the action if not already executed.
    """
    alert = dynamo_utils.get_alert(ALERTS_TABLE, alert_id)
    if not alert:
        return _response(404, {"error": f"Alert {alert_id} not found"})

    # Enrich with live state
    alarm_name = alert.get("alarm_name")
    if alarm_name:
        try:
            region = alert.get("region", os.environ.get("AWS_REGION", "us-east-1"))
            cloudwatch = boto3.client("cloudwatch", region_name=region)
            cw_response = cloudwatch.describe_alarms(AlarmNames=[alarm_name])
            alarms = cw_response.get("MetricAlarms", [])
            if alarms:
                alert["state"] = alarms[0]["StateValue"]
            else:
                alert["state"] = "UNKNOWN"
        except Exception as exc:
            logger.warning("Failed to describe alarm %s: %s", alarm_name, exc)
            alert["state"] = "UNKNOWN"

    # Execute pending action if applicable
    _execute_pending_actions([alert])

    return _response(200, alert)


def _update_alert(alert_id, event):
    """PUT /alerts/{alert_id} - Update alert settings and re-create CW alarm."""
    body = _parse_body(event)
    if not body:
        return _response(400, {"error": "Invalid or missing request body"})

    alert = dynamo_utils.get_alert(ALERTS_TABLE, alert_id)
    if not alert:
        return _response(404, {"error": f"Alert {alert_id} not found"})

    # Update allowed fields
    if "alert_name" in body:
        alert["alert_name"] = body["alert_name"].strip()
    if "threshold_value" in body:
        alert["threshold_value"] = float(body["threshold_value"])
    # Support legacy "threshold" field
    if "threshold" in body and "threshold_value" not in body:
        alert["threshold_value"] = float(body["threshold"])
    if "threshold_mode" in body:
        mode = body["threshold_mode"].strip()
        if mode not in VALID_THRESHOLD_MODES:
            return _response(400, {"error": f"Invalid threshold_mode: {mode}"})
        alert["threshold_mode"] = mode
    if "comparison" in body:
        comp = body["comparison"].strip()
        if comp not in VALID_COMPARISONS:
            return _response(400, {"error": f"Invalid comparison: {comp}"})
        alert["comparison"] = comp
    if "period" in body:
        alert["period"] = int(body["period"])
    if "metric_name" in body:
        mn = body["metric_name"].strip()
        if mn and mn not in VALID_METRIC_NAMES:
            return _response(400, {"error": f"Invalid metric_name: {mn}"})
        alert["metric_name"] = mn
    if "action" in body:
        act = body["action"].strip()
        if act not in VALID_ACTIONS:
            return _response(400, {"error": f"Invalid action: {act}"})
        alert["action"] = act
        # Reset action_executed when action changes
        alert["action_executed"] = False
        if act in ("throttle", "suspend"):
            alert["pending_action"] = act
        elif "pending_action" in alert:
            del alert["pending_action"]
    if "action_config" in body:
        alert["action_config"] = body["action_config"]
    if "dashboard_id" in body:
        alert["dashboard_id"] = body["dashboard_id"] or ""
    if "widget_id" in body:
        alert["widget_id"] = body["widget_id"] or ""

    tenant_id = alert["tenant_id"]
    profile = dynamo_utils.get_profile(TENANTS_TABLE, tenant_id)
    profile_name = profile.get("tenant_name", "unknown") if profile else "unknown"
    region = alert.get("region", os.environ.get("AWS_REGION", "us-east-1"))

    # Handle email subscription update
    new_email = None
    if "action_config" in body and "email" in body["action_config"]:
        new_email = body["action_config"]["email"].strip()
    elif "email" in body:
        new_email = body["email"].strip()

    if new_email is not None and new_email != alert.get("email", ""):
        alert["email"] = new_email
        if alert["email"] and alert.get("topic_arn"):
            try:
                sns = boto3.client("sns", region_name=region)
                sub_response = sns.subscribe(
                    TopicArn=alert["topic_arn"],
                    Protocol="email",
                    Endpoint=alert["email"],
                )
                alert["subscription_arn"] = sub_response.get("SubscriptionArn", "")
            except Exception as exc:
                logger.warning("Failed to subscribe %s: %s", alert["email"], exc)

    # Re-create CloudWatch alarm with updated settings
    try:
        cloudwatch = boto3.client("cloudwatch", region_name=region)
        alarm_kwargs = _build_alarm_kwargs(
            alarm_name=alert["alarm_name"],
            alert_type=alert["alert_type"],
            tenant_id=tenant_id,
            profile_name=profile_name,
            topic_arn=alert.get("topic_arn", ""),
            threshold_value=alert.get("threshold_value", alert.get("threshold", 0)),
            threshold_mode=alert.get("threshold_mode", "absolute"),
            comparison=alert.get("comparison", "GreaterThanOrEqualToThreshold"),
            period=alert.get("period", 300),
            metric_name=alert.get("metric_name") or None,
            alert_name=alert.get("alert_name", ""),
        )
        cloudwatch.put_metric_alarm(**alarm_kwargs)
    except Exception as exc:
        logger.error("Failed to update CloudWatch alarm %s: %s", alert["alarm_name"], exc)
        return _response(500, {"error": f"Failed to update CloudWatch alarm: {exc}"})

    alert["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    dynamo_utils.put_alert(ALERTS_TABLE, alert)
    logger.info("Updated alert %s", alert_id)

    return _response(200, alert)


def _delete_alert(alert_id):
    """DELETE /alerts/{alert_id} - Delete CW alarm, optionally SNS topic, and DynamoDB record."""
    alert = dynamo_utils.get_alert(ALERTS_TABLE, alert_id)
    if not alert:
        return _response(404, {"error": f"Alert {alert_id} not found"})

    region = alert.get("region", os.environ.get("AWS_REGION", "us-east-1"))

    # Delete CloudWatch alarm
    alarm_name = alert.get("alarm_name")
    if alarm_name:
        try:
            cloudwatch = boto3.client("cloudwatch", region_name=region)
            cloudwatch.delete_alarms(AlarmNames=[alarm_name])
            logger.info("Deleted CloudWatch alarm %s", alarm_name)
        except Exception as exc:
            logger.warning("Failed to delete CloudWatch alarm %s: %s", alarm_name, exc)

    # Check if there are other alerts for the same profile before deleting the SNS topic
    topic_arn = alert.get("topic_arn")
    if topic_arn:
        tenant_id = alert["tenant_id"]
        other_alerts = dynamo_utils.list_alerts_by_tenant(
            ALERTS_TABLE, "tenant_id_index", tenant_id,
        )
        # Only delete SNS topic if this is the last alert for the profile
        remaining = [a for a in other_alerts if a.get("alert_id") != alert_id]
        if not remaining:
            try:
                sns = boto3.client("sns", region_name=region)
                sns.delete_topic(TopicArn=topic_arn)
                logger.info("Deleted SNS topic %s", topic_arn)
            except Exception as exc:
                logger.warning("Failed to delete SNS topic %s: %s", topic_arn, exc)

    # Delete DynamoDB record
    dynamo_utils.delete_alert(ALERTS_TABLE, alert_id)
    logger.info("Deleted alert %s", alert_id)

    return _response(200, {"message": f"Alert {alert_id} deleted"})


# ---------------------------------------------------------------------------
# Action execution (lazy evaluation)
# ---------------------------------------------------------------------------

def _execute_pending_actions(alerts: list) -> None:
    """For alerts with action=throttle/suspend in ALARM state, execute the action.

    This implements the lazy evaluation pattern: when an alert with a
    throttle or suspend action is in ALARM state and the action has not
    yet been executed, update the profile status accordingly.

    NOTE: A proper production implementation would use EventBridge + Step Functions
    to react to CloudWatch alarm state changes in real time. This lazy approach
    works when the API is actively polled but introduces latency between when
    the alarm fires and when the action takes effect.
    """
    for alert in alerts:
        action = alert.get("action", "notify")
        state = alert.get("state", "UNKNOWN")
        action_executed = alert.get("action_executed", False)

        if action not in ("throttle", "suspend"):
            continue

        if state == "ALARM" and not action_executed:
            # Execute the action
            profile_id = alert.get("tenant_id", "")
            if not profile_id:
                continue

            try:
                profile = dynamo_utils.get_profile(TENANTS_TABLE, profile_id)
                if not profile:
                    logger.warning(
                        "Profile %s not found for alert %s action execution",
                        profile_id, alert.get("alert_id"),
                    )
                    continue

                current_status = profile.get("status", "active")
                target_status = "throttled" if action == "throttle" else "suspended"

                if current_status == "active":
                    dynamo_utils.update_profile_status(
                        TENANTS_TABLE, profile_id, target_status,
                    )
                    logger.info(
                        "Executed %s action for alert %s: profile %s status changed to %s",
                        action, alert.get("alert_id"), profile_id, target_status,
                    )

                # Mark action as executed
                alert["action_executed"] = True
                alert["action_executed_at"] = (
                    datetime.datetime.now(datetime.timezone.utc).isoformat()
                )
                dynamo_utils.put_alert(ALERTS_TABLE, alert)

            except Exception as exc:
                logger.error(
                    "Failed to execute %s action for alert %s: %s",
                    action, alert.get("alert_id"), exc,
                )

        elif state == "OK" and action_executed:
            # If alarm recovered and auto_recover is set, restore profile status
            auto_recover = alert.get("auto_recover_minutes") or (
                alert.get("action_config", {}).get("auto_recover_minutes")
            )
            if auto_recover:
                profile_id = alert.get("tenant_id", "")
                try:
                    dynamo_utils.update_profile_status(
                        TENANTS_TABLE, profile_id, "active",
                    )
                    alert["action_executed"] = False
                    alert.pop("action_executed_at", None)
                    dynamo_utils.put_alert(ALERTS_TABLE, alert)
                    logger.info(
                        "Auto-recovered profile %s to active for alert %s",
                        profile_id, alert.get("alert_id"),
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to auto-recover profile %s for alert %s: %s",
                        profile_id, alert.get("alert_id"), exc,
                    )


# ---------------------------------------------------------------------------
# Alarm builder
# ---------------------------------------------------------------------------

def _build_alarm_kwargs(
    alarm_name: str,
    alert_type: str,
    tenant_id: str,
    profile_name: str,
    topic_arn: str,
    threshold_value: float,
    threshold_mode: str = "absolute",
    comparison: str = "GreaterThanOrEqualToThreshold",
    period: int = 300,
    metric_name: str | None = None,
    alert_name: str = "",
    tag_dimensions: list | None = None,
    profile_tags: dict | None = None,
) -> dict:
    """Build kwargs for cloudwatch.put_metric_alarm().

    Supports three threshold modes:
      - absolute: Simple threshold comparison (existing behavior).
      - percentage_increase: Alert if metric increases by X% compared to
        previous period. Uses two MetricStat queries (current and previous)
        and a math expression to compute percentage change.
      - percentage_decrease: Alert if metric decreases by X% compared to
        previous period.
    """
    description = alert_name or f"{alert_type} alarm for profile {profile_name}"
    alarm_actions = [topic_arn] if topic_arn else []

    if threshold_mode == "absolute":
        metrics = _build_absolute_metrics(
            alert_type, tenant_id, period, metric_name,
            tag_dimensions=tag_dimensions, profile_tags=profile_tags,
        )
        return {
            "AlarmName": alarm_name,
            "AlarmDescription": description,
            "ActionsEnabled": True,
            "AlarmActions": alarm_actions,
            "Metrics": metrics,
            "ComparisonOperator": comparison,
            "Threshold": threshold_value,
            "EvaluationPeriods": 1,
            "TreatMissingData": "notBreaching",
        }

    elif threshold_mode in ("percentage_increase", "percentage_decrease"):
        metrics = _build_percentage_metrics(
            alert_type, tenant_id, period, threshold_mode, metric_name,
            tag_dimensions=tag_dimensions, profile_tags=profile_tags,
        )
        # For percentage mode, the math expression already computes the
        # percentage change. We compare against the threshold_value directly.
        # percentage_increase: pctChange > threshold means "increased by more than X%"
        # percentage_decrease: pctChange < -threshold means "decreased by more than X%"
        if threshold_mode == "percentage_increase":
            effective_comparison = comparison
            effective_threshold = threshold_value
        else:
            # For decrease, we negate: if user says "alert if decreases by 20%",
            # the expression yields a negative value, so we compare < -threshold
            effective_comparison = _invert_comparison(comparison)
            effective_threshold = -threshold_value

        return {
            "AlarmName": alarm_name,
            "AlarmDescription": description,
            "ActionsEnabled": True,
            "AlarmActions": alarm_actions,
            "Metrics": metrics,
            "ComparisonOperator": effective_comparison,
            "Threshold": effective_threshold,
            "EvaluationPeriods": 1,
            "TreatMissingData": "notBreaching",
        }

    # Fallback to absolute
    metrics = _build_absolute_metrics(
        alert_type, tenant_id, period, metric_name,
        tag_dimensions=tag_dimensions, profile_tags=profile_tags,
    )
    return {
        "AlarmName": alarm_name,
        "AlarmDescription": description,
        "ActionsEnabled": True,
        "AlarmActions": alarm_actions,
        "Metrics": metrics,
        "ComparisonOperator": comparison,
        "Threshold": threshold_value,
        "EvaluationPeriods": 1,
        "TreatMissingData": "notBreaching",
    }


def _build_tag_dims(tag_dimensions: list | None, profile_tags: dict | None) -> list:
    """Build extra CloudWatch Dimensions from tag_dimensions and profile tags."""
    dims = []
    for tag_dim in (tag_dimensions or []):
        tag_key = tag_dim.replace("Tag_", "", 1)
        tag_value = (profile_tags or {}).get(tag_key, "")
        if isinstance(tag_value, list):
            tag_value = " / ".join(tag_value)
        if tag_value:
            dims.append({"Name": tag_dim, "Value": tag_value})
    return dims


def _build_absolute_metrics(
    alert_type: str,
    tenant_id: str,
    period: int,
    metric_name: str | None = None,
    tag_dimensions: list | None = None,
    profile_tags: dict | None = None,
) -> list:
    """Build metric queries for absolute threshold mode.

    If metric_name is provided, creates a single metric query for that metric.
    Otherwise, falls back to the legacy composite expression for the alert_type.
    """
    extra_dims = _build_tag_dims(tag_dimensions, profile_tags)

    if metric_name:
        # Single metric query
        return [
            {
                "Id": "m0",
                "MetricStat": {
                    "Metric": {
                        "Namespace": METRIC_NAMESPACE,
                        "MetricName": metric_name,
                        "Dimensions": [
                            {"Name": "TenantId", "Value": tenant_id},
                        ] + extra_dims,
                    },
                    "Period": period,
                    "Stat": "Sum",
                },
                "ReturnData": True,
            }
        ]

    # Legacy composite expression
    config = _ALERT_METRICS[alert_type]
    metrics = []
    for m in config["metrics"]:
        metrics.append({
            "Id": m["id"],
            "MetricStat": {
                "Metric": {
                    "Namespace": METRIC_NAMESPACE,
                    "MetricName": m["name"],
                    "Dimensions": [
                        {"Name": "TenantId", "Value": tenant_id},
                    ] + extra_dims,
                },
                "Period": period,
                "Stat": "Sum",
            },
            "ReturnData": False,
        })
    metrics.append({
        "Id": config["expression_id"],
        "Expression": config["expression"],
        "ReturnData": True,
    })
    return metrics


def _build_percentage_metrics(
    alert_type: str,
    tenant_id: str,
    period: int,
    threshold_mode: str,
    metric_name: str | None = None,
    tag_dimensions: list | None = None,
    profile_tags: dict | None = None,
) -> list:
    """Build metric queries for percentage-based threshold modes.

    Uses two evaluation windows:
      - current: the current period
      - previous: the preceding period (same duration, offset by one period)

    Then a math expression computes:
      pctChange = 100 * (current - previous) / previous

    CloudWatch does not natively support period offsets in alarms, so we use
    two metric queries with different periods. The "current" query uses the
    standard period, and we compute the percentage change using a RATE-style
    approach with metric math.

    For simplicity, we use a single metric with period = period, and compute
    the rate of change using RATE(). RATE() returns per-second change, so we
    convert to percentage.

    Actually, the most reliable approach for CloudWatch alarms is to use
    two MetricStat entries. We define:
      - m_cur: current period metric
      - An expression that computes percentage change using RATE and period

    Simplified approach: use RATE(METRICS()) which gives per-second change,
    multiply by period and divide by metric value to get percentage.
    """
    if metric_name:
        target_metric = metric_name
    else:
        # Use the first metric from the alert type config
        config = _ALERT_METRICS[alert_type]
        target_metric = config["metrics"][0]["name"]

    # We use the metric directly and compute percentage change using RATE()
    # RATE(m_cur) gives the per-second rate of change
    # Percentage change ~ RATE(m_cur) * period / m_cur * 100
    extra_dims = _build_tag_dims(tag_dimensions, profile_tags)
    metrics = [
        {
            "Id": "m_cur",
            "MetricStat": {
                "Metric": {
                    "Namespace": METRIC_NAMESPACE,
                    "MetricName": target_metric,
                    "Dimensions": [
                        {"Name": "TenantId", "Value": tenant_id},
                    ] + extra_dims,
                },
                "Period": period,
                "Stat": "Sum",
            },
            "ReturnData": False,
        },
        {
            "Id": "pctChange",
            "Expression": f"100 * RATE(m_cur) * {period} / m_cur",
            "Label": "Percentage Change",
            "ReturnData": True,
        },
    ]

    return metrics


def _invert_comparison(comparison: str) -> str:
    """Invert a CloudWatch comparison operator for percentage_decrease mode."""
    inversion_map = {
        "GreaterThanOrEqualToThreshold": "LessThanOrEqualToThreshold",
        "LessThanOrEqualToThreshold": "GreaterThanOrEqualToThreshold",
        "GreaterThanThreshold": "LessThanThreshold",
        "LessThanThreshold": "GreaterThanThreshold",
    }
    return inversion_map.get(comparison, comparison)


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
