# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Pure logic for converting a dashboard template + profile context into CloudWatch dashboard JSON.

No AWS SDK calls -- this module is purely functional so it can be unit-tested
without mocking boto3.
"""

import copy
import json
from typing import Any, Optional

METRIC_NAMESPACE = "ISVBedrock/Gateway"

# Maps template widget types to CloudWatch dashboard widget properties.
_VIEW_MAP = {
    "timeseries": {"type": "metric", "view": "timeSeries"},
    "bar": {"type": "metric", "view": "bar"},
    "number": {"type": "metric", "view": "singleValue"},
    "single_value": {"type": "metric", "view": "singleValue"},
    "pie": {"type": "metric", "view": "pie"},
}


def merge_widget_overrides(template_widgets: list, overrides: dict) -> list:
    """Apply per-widget overrides to template widgets.

    Args:
        template_widgets: List of widget dicts from the template.
        overrides: Dict keyed by widget_id, values are dicts of fields to override.
            Example: {"cost_timeseries": {"stat": "Average", "period": 3600, "analysis": "trend"}}

    Returns:
        A new list of widget dicts with overrides merged in.
        Original template_widgets are not mutated.
    """
    if not overrides:
        return copy.deepcopy(template_widgets)

    result = []
    for widget in template_widgets:
        widget_copy = copy.deepcopy(widget)
        widget_id = widget_copy.get("widget_id", "")
        if widget_id and widget_id in overrides:
            widget_override = overrides[widget_id]
            for key, value in widget_override.items():
                widget_copy[key] = value
        result.append(widget_copy)
    return result


def build_dashboard_body(
    template: dict,
    profile: dict,
    region: str,
    widget_overrides: Optional[dict] = None,
    tenants: Optional[list[dict]] = None,
    tag_dimensions: Optional[list[str]] = None,
) -> str:
    """Convert a dashboard template into a CloudWatch dashboard JSON string.

    Args:
        template: A single template dict from dashboard_templates.json.
        profile: The primary profile record (used for single-profile dashboards).
        region: AWS region for each widget.
        widget_overrides: Optional dict keyed by widget_id with per-widget
            setting overrides (stat, period, analysis, etc.).
        tenants: Optional list of profile dicts for multi-profile dashboards.
            When provided, each widget renders one metric line per profile,
            allowing side-by-side comparison.

    Returns:
        A JSON string suitable for cloudwatch.put_dashboard(DashboardBody=...).
    """
    # Build the list of {tenant_id, tenant_name} for metric rendering
    if tenants and len(tenants) > 1:
        profile_list = [{"tenant_id": t["tenant_id"], "tenant_name": t.get("tenant_name", t["tenant_id"][:8])} for t in tenants]
    else:
        profile_list = [{"tenant_id": profile["tenant_id"], "tenant_name": profile.get("tenant_name", "")}]

    # Apply widget overrides before building
    widgets = template.get("widgets", [])
    if widget_overrides:
        widgets = merge_widget_overrides(widgets, widget_overrides)

    # Build full profile records list for tag dimension lookups
    full_profiles = tenants if tenants and len(tenants) > 1 else [profile]

    cw_widgets: list[dict[str, Any]] = []
    for widget in widgets:
        cw_widget = _build_widget(widget, profile_list, region, tag_dimensions=tag_dimensions, full_tenants=full_profiles)
        cw_widgets.append(cw_widget)

    # No variable selectors — profile/tag filtering happens in the UI
    # at dashboard creation time, not inside the CW dashboard.
    dashboard_body: dict[str, Any] = {"widgets": cw_widgets}

    return json.dumps(dashboard_body)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_widget(
    widget: dict,
    tenant_list: list[dict],
    region: str,
    tag_dimensions: Optional[list[str]] = None,
    full_tenants: Optional[list[dict]] = None,
) -> dict:
    """Build a single CloudWatch dashboard widget from a template widget.

    Args:
        widget: Template widget definition.
        tenant_list: List of {"tenant_id": ..., "tenant_name": ...} dicts.
            For multi-profile dashboards, each metric is rendered once per profile.
        region: AWS region.
        tag_dimensions: Optional list of tag dimension names (e.g. ["Tag_Environment"]).
        full_tenants: Optional list of full profile records (with tags) for tag dimension values.
    """
    widget_type = widget.get("type", "timeseries")
    view_props = _VIEW_MAP.get(widget_type, _VIEW_MAP["timeseries"])

    pos = widget.get("position", {})
    position = {
        "x": pos.get("x", 0),
        "y": pos.get("y", 0),
        "width": pos.get("w", 12),
        "height": pos.get("h", 6),
    }

    metrics = widget.get("metrics", [])
    dimensions = widget.get("dimensions", ["InferenceProfile"])
    stat = widget.get("stat", "Sum")
    period = widget.get("period", 300)
    use_search = widget.get("use_search", False)
    stacked = widget.get("stacked", False)
    expression = widget.get("expression")
    metric_ids = widget.get("metric_ids", {})
    analysis = widget.get("analysis")

    slices = widget.get("slices")

    properties: dict[str, Any] = {
        "view": view_props["view"],
        "region": region,
        "title": widget.get("title", ""),
        "period": period,
        "stat": stat,
        "stacked": stacked,
    }

    # Pie charts should use the full time range, not just the latest period
    if widget_type == "pie":
        properties["setPeriodToTimeRange"] = True

    if slices and metric_ids:
        # Multi-slice mode: hidden base metrics + multiple visible expressions
        properties["metrics"] = _build_sliced_metrics(
            metrics, dimensions, stat, period, tenant_list, metric_ids, slices,
            tag_dimensions=tag_dimensions, full_tenants=full_tenants,
        )
    elif use_search:
        properties["metrics"] = _build_search_metrics(
            metrics, dimensions, stat, period, expression,
            tag_dimensions=tag_dimensions,
        )
    elif expression and metric_ids:
        properties["metrics"] = _build_expression_metrics(
            metrics, dimensions, stat, period, tenant_list, metric_ids, expression,
            tag_dimensions=tag_dimensions, full_tenants=full_tenants,
        )
    else:
        properties["metrics"] = _build_explicit_metrics(
            metrics, dimensions, stat, period, tenant_list,
            tag_dimensions=tag_dimensions, full_tenants=full_tenants,
        )

    # Apply analysis overlays (trend, anomaly_detection, arima)
    if analysis and analysis != "none":
        _apply_analysis_overlay(properties, analysis, metrics, stat, period)

    return {
        "type": view_props["type"],
        "x": position["x"],
        "y": position["y"],
        "width": position["width"],
        "height": position["height"],
        "properties": properties,
    }


def _apply_analysis_overlay(
    properties: dict,
    analysis: str,
    metrics: list[str],
    stat: str,
    period: int,
) -> None:
    """Add analysis overlay expressions to a widget's metrics.

    Modifies properties["metrics"] in place by appending metric math expressions.

    Supported analysis types:
        - "trend": Adds a LINEAR() regression line overlay for each metric.
        - "anomaly_detection": Adds ANOMALY_DETECTION_BAND() around each metric
            with a standard band width of 2.
        - "arima": Uses CloudWatch's ANOMALY_DETECTION_BAND() with a wider band
            width (3) -- CW's built-in ML model captures seasonal patterns
            internally. Also adds a simple moving average (SMA) expression
            for visual comparison.

    Args:
        properties: The widget properties dict (mutated in place).
        analysis: One of "trend", "anomaly_detection", "arima".
        metrics: List of metric names from the widget.
        stat: The stat used for the metrics.
        period: The period in seconds.
        tenant_id: Tenant ID for dimension scoping.
    """
    existing_metrics = properties.get("metrics", [])

    # We need metric IDs to reference in math expressions.
    # First, ensure each base metric has an ID assigned.
    # For explicit metrics (list-of-lists format), we need to add IDs.
    _ensure_metric_ids(existing_metrics, metrics)

    if analysis == "trend":
        # Add a LINEAR() regression line for each metric
        for i, metric_name in enumerate(metrics):
            mid = f"m{i}"
            expr_id = f"trend_{mid}"
            existing_metrics.append([{
                "expression": f"LINEAR({mid})",
                "label": f"{metric_name} Trend",
                "id": expr_id,
            }])

    elif analysis == "anomaly_detection":
        # Add ANOMALY_DETECTION_BAND() around each metric (band width = 2)
        for i, metric_name in enumerate(metrics):
            mid = f"m{i}"
            expr_id = f"ad_{mid}"
            existing_metrics.append([{
                "expression": f"ANOMALY_DETECTION_BAND({mid}, 2)",
                "label": f"{metric_name} Anomaly Band",
                "id": expr_id,
            }])

    elif analysis == "arima":
        # ARIMA / seasonality analysis:
        # CloudWatch's ANOMALY_DETECTION_BAND() uses built-in ML that captures
        # seasonal patterns automatically. We use a wider band width (3) to
        # reflect the broader uncertainty inherent in seasonal forecasting.
        # We also add a simple moving average (SMA) as a complementary overlay
        # so users can visually compare smoothed trends against the anomaly band.
        for i, metric_name in enumerate(metrics):
            mid = f"m{i}"
            ad_id = f"arima_ad_{mid}"
            sma_id = f"arima_sma_{mid}"
            # Anomaly band with wider width for seasonal pattern capture
            existing_metrics.append([{
                "expression": f"ANOMALY_DETECTION_BAND({mid}, 3)",
                "label": f"{metric_name} Seasonal Band",
                "id": ad_id,
            }])
            # Simple moving average: average over 5 data points for smoothing
            existing_metrics.append([{
                "expression": f"AVG(METRICS(\"{mid}\"))",
                "label": f"{metric_name} Moving Avg",
                "id": sma_id,
            }])

    properties["metrics"] = existing_metrics


def _ensure_metric_ids(existing_metrics: list, metric_names: list[str]) -> None:
    """Ensure each base metric entry in existing_metrics has an 'id' field.

    For explicit metric entries (list format), appends or updates the trailing
    options dict to include an id like 'm0', 'm1', etc.

    For expression-based entries (dict-in-list), skips them.

    Modifies existing_metrics in place.
    """
    metric_idx = 0
    for entry in existing_metrics:
        if not isinstance(entry, list) or len(entry) == 0:
            continue
        # Expression entries are [{expression: ..., ...}]
        if isinstance(entry[0], dict) and "expression" in entry[0]:
            continue
        # Explicit metric entry: [namespace, name, dim1, val1, ..., {options}]
        mid = f"m{metric_idx}"
        metric_idx += 1
        # Check if last element is already an options dict
        if isinstance(entry[-1], dict):
            entry[-1]["id"] = mid
        else:
            entry.append({"id": mid})


def _build_search_metrics(
    metrics: list[str],
    dimensions: list[str],
    stat: str,
    period: int,
    expression: str | None = None,
    tag_dimensions: Optional[list[str]] = None,
) -> list:
    """Build SEARCH expression metrics (used for cross-profile views)."""
    result: list = []
    # Always include base dimensions — CloudWatch requires exact dimension matching.
    # Metrics are emitted with at least (TenantId, InferenceProfile), so SEARCH
    # must include both to find them.
    base_dims = {"TenantId", "InferenceProfile"}
    all_dims = sorted(base_dims | set(dimensions)) + (tag_dimensions or [])
    dim_str = ",".join(all_dims)

    for metric_name in metrics:
        search_expr = (
            f"SEARCH('{{{METRIC_NAMESPACE},{dim_str}}} "
            f"MetricName=\"{metric_name}\"', '{stat}', {period})"
        )
        result.append([{"expression": search_expr, "label": metric_name}])

    if expression:
        result.append([{"expression": expression, "label": "Result"}])

    return result


def _build_explicit_metrics(
    metrics: list[str],
    dimensions: list[str],
    stat: str,
    period: int,
    tenant_list: list[dict],
    tag_dimensions: Optional[list[str]] = None,
    full_tenants: Optional[list[dict]] = None,
) -> list:
    """Build explicit metric references, one line per profile per metric.

    For multi-profile dashboards this produces e.g.:
        InputTokensCost (Profile A), InputTokensCost (Profile B), ...
    so they appear as separate lines on the same chart.
    """
    # Build a lookup from tenant_id to full profile record for tag values
    profile_by_id: dict = {}
    if full_tenants:
        for ft in full_tenants:
            profile_by_id[ft["tenant_id"]] = ft

    result: list = []
    multi = len(tenant_list) > 1
    for t in tenant_list:
        tid = t["tenant_id"]
        tname = t.get("tenant_name", tid[:8])
        for metric_name in metrics:
            metric_ref: list[Any] = [METRIC_NAMESPACE, metric_name]
            metric_ref.extend(["TenantId", tid])
            for dim in dimensions:
                if dim == "TenantId":
                    continue
                if dim == "InferenceProfile":
                    # Use actual profile ID instead of variable reference
                    profile_record = profile_by_id.get(tid, {})
                    ip_id = profile_record.get("inference_profile_id", "")
                    metric_ref.extend(["InferenceProfile", ip_id])
                else:
                    metric_ref.extend([dim, ""])
            label = f"{metric_name} ({tname})" if multi else metric_name
            metric_ref.append({"stat": stat, "period": period, "label": label})
            result.append(metric_ref)
    return result


def _build_expression_metrics(
    metrics: list[str],
    dimensions: list[str],
    stat: str,
    period: int,
    tenant_list: list[dict],
    metric_ids: dict[str, str],
    expression: str,
    tag_dimensions: Optional[list[str]] = None,
    full_tenants: Optional[list[dict]] = None,
) -> list:
    """Build metrics with a math expression (e.g. error rate percentage).

    For multi-profile: creates base metrics per profile with suffixed IDs,
    then one expression per profile.
    """
    # Build a lookup from tenant_id to full profile record for tag values
    profile_by_id: dict = {}
    if full_tenants:
        for ft in full_tenants:
            profile_by_id[ft["tenant_id"]] = ft

    result: list = []
    multi = len(tenant_list) > 1

    for ti, t in enumerate(tenant_list):
        tid = t["tenant_id"]
        tname = t.get("tenant_name", tid[:8])
        suffix = f"_t{ti}" if multi else ""

        for metric_name in metrics:
            mid = metric_ids.get(metric_name, metric_name[:2].lower()) + suffix
            metric_ref: list[Any] = [METRIC_NAMESPACE, metric_name]
            metric_ref.extend(["TenantId", tid])
            for dim in dimensions:
                if dim == "TenantId":
                    continue
                if dim == "InferenceProfile":
                    profile_record = profile_by_id.get(tid, {})
                    ip_id = profile_record.get("inference_profile_id", "")
                    metric_ref.extend(["InferenceProfile", ip_id])
                else:
                    metric_ref.extend([dim, ""])
            metric_ref.append({"stat": stat, "period": period, "id": mid, "visible": False})
            result.append(metric_ref)

        # Build expression with profile-suffixed IDs
        expr = expression
        for metric_name, base_mid in metric_ids.items():
            expr = expr.replace(base_mid, base_mid + suffix)
        label = f"Result ({tname})" if multi else "Result"
        result.append([{"expression": expr, "label": label, "id": f"expr{ti}"}])

    return result


def _build_sliced_metrics(
    metrics: list[str],
    dimensions: list[str],
    stat: str,
    period: int,
    tenant_list: list[dict],
    metric_ids: dict[str, str],
    slices: list[dict],
    tag_dimensions: Optional[list[str]] = None,
    full_tenants: Optional[list[dict]] = None,
) -> list:
    """Build metrics with multiple expression slices (used for pie charts).

    Creates hidden base metrics with IDs, then adds one visible expression
    per slice entry. Each slice has an 'expression' and 'label'.

    For multi-profile dashboards, creates one set of slices per profile.
    """
    profile_by_id: dict = {}
    if full_tenants:
        for ft in full_tenants:
            profile_by_id[ft["tenant_id"]] = ft

    result: list = []
    multi = len(tenant_list) > 1

    for ti, t in enumerate(tenant_list):
        tid = t["tenant_id"]
        tname = t.get("tenant_name", tid[:8])
        suffix = f"_t{ti}" if multi else ""

        # Hidden base metrics
        for metric_name in metrics:
            mid = metric_ids.get(metric_name, metric_name[:2].lower()) + suffix
            metric_ref: list[Any] = [METRIC_NAMESPACE, metric_name]
            metric_ref.extend(["TenantId", tid])
            for dim in dimensions:
                if dim == "TenantId":
                    continue
                if dim == "InferenceProfile":
                    profile_record = profile_by_id.get(tid, {})
                    ip_id = profile_record.get("inference_profile_id", "")
                    metric_ref.extend(["InferenceProfile", ip_id])
                else:
                    metric_ref.extend([dim, ""])
            metric_ref.append({"stat": stat, "period": period, "id": mid, "visible": False})
            result.append(metric_ref)

        # Visible expression slices
        for si, s in enumerate(slices):
            expr = s["expression"]
            label = s.get("label", f"Slice {si}")
            # Suffix metric IDs for multi-profile
            for metric_name, base_mid in metric_ids.items():
                expr = expr.replace(base_mid, base_mid + suffix)
            if multi:
                label = f"{label} ({tname})"
            result.append([{"expression": expr, "label": label, "id": f"s{si}_t{ti}"}])

    return result


def _build_variables(
    tag_dimensions: Optional[list[str]] = None,
    full_profiles: Optional[list[dict]] = None,
) -> list[dict]:
    """Build variable selectors for the dashboard (InferenceProfile + tag dimensions).

    SEARCH expressions must include all base dimensions (TenantId, InferenceProfile)
    because CloudWatch requires exact dimension matching.
    """
    # Build the InferenceProfile variable selector.
    # If we have profile records, use static values with friendly labels.
    # Otherwise fall back to dynamic SEARCH-based population.
    if full_profiles:
        values = []
        for p in full_profiles:
            pid = p.get("inference_profile_id", "")
            name = p.get("tenant_name", pid[:8])
            if pid:
                values.append({"value": pid, "label": name})
        profile_var: dict[str, Any] = {
            "type": "property",
            "property": "InferenceProfile",
            "inputType": "select",
            "id": "profileVariable",
            "label": "Inference Profile",
            "visible": True,
            "values": [v for v in values] if values else [{"value": "", "label": "No profiles"}],
            "defaultValue": values[0]["value"] if values else "",
        }
    else:
        profile_var = {
            "type": "property",
            "property": "InferenceProfile",
            "inputType": "select",
            "id": "profileVariable",
            "label": "Inference Profile",
            "visible": True,
            "search": (
                f"{{{METRIC_NAMESPACE},TenantId,InferenceProfile}} "
                f"MetricName=\"InputTokens\""
            ),
            "populateFrom": "InferenceProfile",
            "defaultValue": "__FIRST",
        }

    return [profile_var]


def sanitize_dashboard_name(name: str) -> str:
    """Sanitize a string for use as a CloudWatch dashboard name.

    CW dashboard names allow alphanumeric, hyphens, and underscores.
    """
    safe = ""
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            safe += ch
        else:
            safe += "-"
    return safe
