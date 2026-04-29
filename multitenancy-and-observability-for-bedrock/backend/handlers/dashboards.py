# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Lambda handler for dashboard CRUD operations.

Routes based on httpMethod + resource path from API Gateway proxy integration.

Environment variables:
    DASHBOARDS_TABLE        - DynamoDB table for dashboard records
    TENANTS_TABLE           - DynamoDB table for profile records
    STORAGE_BUCKET - S3 bucket containing seed/dashboard_templates.json
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
from shared.dashboard_builder import (
    build_dashboard_body,
    merge_widget_overrides,
    sanitize_dashboard_name,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DASHBOARDS_TABLE = os.environ.get("DASHBOARDS_TABLE", "")
TENANTS_TABLE = os.environ.get("TENANTS_TABLE", "")
STORAGE_BUCKET = os.environ.get("STORAGE_BUCKET", "")


def handler(event, context):
    """Main Lambda handler - routes to appropriate operation."""
    http_method = event.get("httpMethod", "")
    path = event.get("path", "")
    path_params = event.get("pathParameters") or {}
    dashboard_id = path_params.get("dashboard_id")
    if not dashboard_id:
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[0] == "dashboards":
            dashboard_id = parts[1]

    try:
        # Check for /dashboards/{dashboard_id}/widgets sub-resource
        if dashboard_id and path.endswith("/widgets"):
            if http_method == "GET":
                return _get_dashboard_widgets(dashboard_id)

        if dashboard_id:
            if http_method == "GET":
                return _get_dashboard(dashboard_id)
            elif http_method == "PUT":
                return _update_dashboard(dashboard_id, event)
            elif http_method == "DELETE":
                return _delete_dashboard(dashboard_id)
        else:
            if http_method == "POST":
                return _create_dashboard(event)
            elif http_method == "GET":
                return _list_dashboards(event)

        return _response(404, {"error": "Not found"})

    except Exception as exc:
        logger.exception("Unhandled error in dashboards handler")
        return _response(500, {"error": str(exc)})


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def _create_dashboard(event):
    """POST /dashboards - Create a CloudWatch dashboard from a template.

    Accepts an optional ``widget_overrides`` dict keyed by widget_id to
    customise individual widget settings (stat, period, analysis, etc.)
    before building the CloudWatch dashboard.
    """
    body = _parse_body(event)
    if not body:
        return _response(400, {"error": "Invalid or missing request body"})

    # Support both single profile_id and multi profile_ids
    profile_ids = body.get("tenant_ids", [])
    single_profile_id = body.get("tenant_id", "").strip()
    if not profile_ids and single_profile_id:
        profile_ids = [single_profile_id]
    template_id = body.get("template_id", "").strip()
    dashboard_name = body.get("dashboard_name", "").strip()
    widget_overrides = body.get("widget_overrides") or {}
    tag_dimensions = body.get("tag_dimensions") or []
    widget_ids = body.get("widget_ids") or []

    if not profile_ids or not template_id:
        return _response(400, {"error": "tenant_id (or tenant_ids) and template_id are required"})

    # 1. Validate all profiles exist
    profiles_list = []
    for tid in profile_ids:
        profile = dynamo_utils.get_profile(TENANTS_TABLE, tid)
        if not profile:
            return _response(404, {"error": f"Profile {tid} not found"})
        profiles_list.append(profile)

    primary_profile = profiles_list[0]

    # 2. Load template(s) from S3
    if template_id == "custom":
        if not widget_ids:
            return _response(400, {"error": "widget_ids required for custom template"})
        template = _build_custom_template(widget_ids)
        if template is None:
            return _response(500, {"error": "Failed to load templates from S3"})
    else:
        template = _load_template(template_id)
        if template is None:
            return _response(404, {"error": f"Template {template_id} not found"})

    # 3. Resolve widgets (apply overrides)
    resolved_widgets = merge_widget_overrides(
        template.get("widgets", []), widget_overrides,
    )

    # 4. Substitute capacity limit placeholder in widget expressions/slices
    capacity_limit = primary_profile.get("capacity_limit", 1000)
    _substitute_capacity_limit(resolved_widgets, capacity_limit)

    # 5. Build CloudWatch dashboard body (multi-profile aware)
    region = primary_profile.get("region", os.environ.get("AWS_REGION", "us-east-1"))
    template_for_build = dict(template)
    template_for_build["widgets"] = resolved_widgets
    dashboard_body = build_dashboard_body(
        template_for_build, primary_profile, region, tenants=profiles_list,
        tag_dimensions=tag_dimensions,
    )

    # 5. Generate CW dashboard name
    if len(profiles_list) > 1:
        name_parts = "-".join(t.get("tenant_name", "t")[:10] for t in profiles_list[:3])
        if len(profiles_list) > 3:
            name_parts += f"-plus{len(profiles_list) - 3}"
        cw_name = sanitize_dashboard_name(f"isv-obs-{name_parts}-{template_id}")
    else:
        profile_name = primary_profile.get("tenant_name", "unknown")
        cw_name = sanitize_dashboard_name(f"isv-obs-{profile_name}-{template_id}")

    # 6. Create CloudWatch dashboard
    try:
        cloudwatch = boto3.client("cloudwatch", region_name=region)
        cloudwatch.put_dashboard(
            DashboardName=cw_name,
            DashboardBody=dashboard_body,
        )
    except Exception as exc:
        logger.error("Failed to create CloudWatch dashboard: %s", exc)
        return _response(500, {"error": f"Failed to create CloudWatch dashboard: {exc}"})

    # 7. Store in DynamoDB
    dashboard_id = str(uuid.uuid4())
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    console_url = (
        f"https://{region}.console.aws.amazon.com/cloudwatch/home"
        f"?region={region}#dashboards/dashboard/{cw_name}"
    )

    record = {
        "dashboard_id": dashboard_id,
        "tenant_id": profile_ids[0],  # primary profile for GSI
        "tenant_ids": profile_ids,
        "template_id": template_id,
        "dashboard_name": dashboard_name or template.get("name", template_id),
        "cw_dashboard_name": cw_name,
        "region": region,
        "console_url": console_url,
        "widgets": resolved_widgets,
        "widget_overrides": widget_overrides,
        "widget_ids": widget_ids,
        "tag_dimensions": tag_dimensions,
        "created_at": now,
        "updated_at": now,
    }

    dynamo_utils.put_dashboard(DASHBOARDS_TABLE, record)
    logger.info("Created dashboard %s (CW: %s) for profiles %s", dashboard_id, cw_name, profile_ids)

    return _response(201, record)


def _update_dashboard(dashboard_id, event):
    """PUT /dashboards/{dashboard_id} - Update dashboard name and/or widget settings.

    Re-builds the CloudWatch dashboard body and calls put_dashboard again.
    Persists the new widget_overrides in the DynamoDB record.
    """
    body = _parse_body(event)
    if not body:
        return _response(400, {"error": "Invalid or missing request body"})

    dashboard = dynamo_utils.get_dashboard(DASHBOARDS_TABLE, dashboard_id)
    if not dashboard:
        return _response(404, {"error": f"Dashboard {dashboard_id} not found"})

    # Update dashboard name if provided
    new_name = body.get("dashboard_name")
    if new_name is not None:
        dashboard["dashboard_name"] = new_name.strip()

    # Merge new widget overrides with existing ones
    new_overrides = body.get("widget_overrides") or {}
    existing_overrides = dashboard.get("widget_overrides") or {}
    for widget_id, overrides in new_overrides.items():
        if widget_id not in existing_overrides:
            existing_overrides[widget_id] = {}
        existing_overrides[widget_id].update(overrides)
    dashboard["widget_overrides"] = existing_overrides

    # Update tag_dimensions if provided
    new_tag_dimensions = body.get("tag_dimensions")
    if new_tag_dimensions is not None:
        dashboard["tag_dimensions"] = new_tag_dimensions

    # Re-load template to rebuild the dashboard
    template_id = dashboard.get("template_id", "")
    if template_id == "custom":
        stored_widget_ids = dashboard.get("widget_ids", [])
        template = _build_custom_template(stored_widget_ids)
        if template is None:
            return _response(500, {"error": "Failed to load templates from S3 during update"})
    else:
        template = _load_template(template_id)
        if template is None:
            return _response(500, {"error": f"Template {template_id} not found during update"})

    # Re-resolve widgets
    resolved_widgets = merge_widget_overrides(
        template.get("widgets", []), existing_overrides,
    )
    dashboard["widgets"] = resolved_widgets

    # Rebuild CloudWatch dashboard (multi-profile aware)
    profile_ids = dashboard.get("tenant_ids", [dashboard.get("tenant_id", "")])
    profiles_list = []
    for tid in profile_ids:
        t = dynamo_utils.get_profile(TENANTS_TABLE, tid)
        if t:
            profiles_list.append(t)
    if not profiles_list:
        return _response(404, {"error": "No valid profiles found for this dashboard"})

    # Substitute capacity limit placeholder
    capacity_limit = profiles_list[0].get("capacity_limit", 1000)
    _substitute_capacity_limit(resolved_widgets, capacity_limit)

    region = dashboard.get("region", os.environ.get("AWS_REGION", "us-east-1"))
    template_for_build = dict(template)
    template_for_build["widgets"] = resolved_widgets
    dashboard_body = build_dashboard_body(
        template_for_build, profiles_list[0], region, tenants=profiles_list,
        tag_dimensions=dashboard.get("tag_dimensions") or [],
    )

    cw_name = dashboard.get("cw_dashboard_name", "")
    try:
        cloudwatch = boto3.client("cloudwatch", region_name=region)
        cloudwatch.put_dashboard(
            DashboardName=cw_name,
            DashboardBody=dashboard_body,
        )
    except Exception as exc:
        logger.error("Failed to update CloudWatch dashboard %s: %s", cw_name, exc)
        return _response(500, {"error": f"Failed to update CloudWatch dashboard: {exc}"})

    dashboard["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    dynamo_utils.put_dashboard(DASHBOARDS_TABLE, dashboard)
    logger.info("Updated dashboard %s (CW: %s)", dashboard_id, cw_name)

    return _response(200, dashboard)


def _list_dashboards(event):
    """GET /dashboards - List dashboards, optionally filtered by profile_id or tags."""
    params = event.get("queryStringParameters") or {}
    profile_id = params.get("tenant_id")

    # Parse tag filters
    tag_filters_raw = params.get("tag_filters", "")
    tag_filters = parse_tag_filters(tag_filters_raw)

    if profile_id:
        dashboards = dynamo_utils.list_dashboards_by_tenant(
            DASHBOARDS_TABLE, "tenant_id_index", profile_id,
        )
    else:
        table = dynamo_utils._table(DASHBOARDS_TABLE)
        response = table.scan(Limit=100)
        dashboards = [dynamo_utils._deserialize_record(item) for item in response.get("Items", [])]

    # If tag filters provided, resolve matching profile IDs and filter dashboards
    if tag_filters:
        matching_profile_ids = get_profile_ids_for_tags(TENANTS_TABLE, tag_filters)
        dashboards = [
            d for d in dashboards
            if d.get("tenant_id") in matching_profile_ids
            or any(tid in matching_profile_ids for tid in d.get("tenant_ids", []))
        ]

    return _response(200, {
        "dashboards": dashboards,
        "count": len(dashboards),
    })


def _get_dashboard(dashboard_id):
    """GET /dashboards/{dashboard_id} - Get a single dashboard.

    Returns the full dashboard record INCLUDING the resolved widgets array
    so the frontend can display current widget settings.
    """
    dashboard = dynamo_utils.get_dashboard(DASHBOARDS_TABLE, dashboard_id)
    if not dashboard:
        return _response(404, {"error": f"Dashboard {dashboard_id} not found"})
    return _response(200, dashboard)


def _get_dashboard_widgets(dashboard_id):
    """GET /dashboards/{dashboard_id}/widgets - Return the list of widgets for a dashboard.

    Used by the alert creation form to let users pick a dashboard widget to alert on.
    """
    dashboard = dynamo_utils.get_dashboard(DASHBOARDS_TABLE, dashboard_id)
    if not dashboard:
        return _response(404, {"error": f"Dashboard {dashboard_id} not found"})

    widgets = dashboard.get("widgets", [])
    # Return a simplified view for widget selection
    widget_summaries = []
    for w in widgets:
        widget_summaries.append({
            "widget_id": w.get("widget_id", ""),
            "title": w.get("title", ""),
            "type": w.get("type", ""),
            "metrics": w.get("metrics", []),
            "stat": w.get("stat", "Sum"),
            "period": w.get("period", 300),
        })

    return _response(200, {
        "dashboard_id": dashboard_id,
        "widgets": widget_summaries,
        "count": len(widget_summaries),
    })


def _delete_dashboard(dashboard_id):
    """DELETE /dashboards/{dashboard_id} - Delete CW dashboard and DynamoDB record."""
    dashboard = dynamo_utils.get_dashboard(DASHBOARDS_TABLE, dashboard_id)
    if not dashboard:
        return _response(404, {"error": f"Dashboard {dashboard_id} not found"})

    # Delete CloudWatch dashboard
    cw_name = dashboard.get("cw_dashboard_name")
    if cw_name:
        try:
            region = dashboard.get("region", os.environ.get("AWS_REGION", "us-east-1"))
            cloudwatch = boto3.client("cloudwatch", region_name=region)
            cloudwatch.delete_dashboards(DashboardNames=[cw_name])
            logger.info("Deleted CloudWatch dashboard %s", cw_name)
        except Exception as exc:
            logger.warning("Failed to delete CloudWatch dashboard %s: %s", cw_name, exc)

    # Delete DynamoDB record
    dynamo_utils.delete_dashboard(DASHBOARDS_TABLE, dashboard_id)
    logger.info("Deleted dashboard %s", dashboard_id)

    return _response(200, {"message": f"Dashboard {dashboard_id} deleted"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_template(template_id: str) -> dict | None:
    """Load a dashboard template by ID from S3.

    Returns the template dict, or None if not found.
    """
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(
            Bucket=STORAGE_BUCKET,
            Key="seed/dashboard_templates.json",
        )
        templates_data = json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as exc:
        logger.error("Failed to load dashboard templates from S3: %s", exc)
        return None

    templates = templates_data.get("templates", [])
    for t in templates:
        if t.get("id") == template_id:
            return t
    return None


def _load_all_templates() -> list | None:
    """Load all dashboard templates from S3.

    Returns list of template dicts, or None on failure.
    """
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(
            Bucket=STORAGE_BUCKET,
            Key="seed/dashboard_templates.json",
        )
        templates_data = json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as exc:
        logger.error("Failed to load dashboard templates from S3: %s", exc)
        return None

    return templates_data.get("templates", [])


def _build_custom_template(widget_ids: list[str]) -> dict | None:
    """Build a synthetic template containing only the requested widgets.

    Loads ALL templates from S3, merges their widget lists, filters to the
    requested widget_ids, and auto-assigns grid positions.
    """
    all_templates = _load_all_templates()
    if all_templates is None:
        return None

    # Collect all widgets across all templates, dedup by widget_id
    all_widgets: dict[str, dict] = {}
    for t in all_templates:
        for w in t.get("widgets", []):
            wid = w.get("widget_id", "")
            if wid and wid not in all_widgets:
                all_widgets[wid] = w

    # Filter to only requested widget_ids (preserve order)
    filtered = []
    for wid in widget_ids:
        if wid in all_widgets:
            filtered.append(dict(all_widgets[wid]))

    if not filtered:
        return {"id": "custom", "name": "Custom Dashboard", "widgets": []}

    # Auto-assign grid positions (2 columns, 12-unit wide grid)
    col_width = 12
    row_height = 6
    for i, w in enumerate(filtered):
        col = (i % 2) * (col_width // 2)
        row = (i // 2) * row_height
        w["position"] = {"x": col, "y": row, "w": col_width // 2, "h": row_height}

    return {"id": "custom", "name": "Custom Dashboard", "widgets": filtered}


def _substitute_capacity_limit(widgets: list, capacity_limit: int) -> None:
    """Replace __CAPACITY_LIMIT__ placeholder in widget slices and titles.

    Modifies the widgets list in place.
    """
    cap_str = str(capacity_limit)
    for w in widgets:
        # Substitute in slices
        slices = w.get("slices")
        if slices:
            for s in slices:
                if "__CAPACITY_LIMIT__" in s.get("expression", ""):
                    s["expression"] = s["expression"].replace("__CAPACITY_LIMIT__", cap_str)
        # Substitute in title
        title = w.get("title", "")
        if "__CAPACITY_LIMIT__" in title:
            w["title"] = title.replace("__CAPACITY_LIMIT__", cap_str)
        # Substitute in single expression field
        expr = w.get("expression", "") or ""
        if "__CAPACITY_LIMIT__" in expr:
            w["expression"] = expr.replace("__CAPACITY_LIMIT__", cap_str)


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
