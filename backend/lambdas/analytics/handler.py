"""
Analytics Lambda handler for Bedrock Model Profiler.

Routes:
  POST /events    — Record anonymous usage events (public)
  GET  /dashboard — Return aggregated dashboard data (admin only)
"""

import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

# Environment
TABLE_NAME = os.environ.get("ANALYTICS_TABLE", "bedrock-profiler-analytics-dev")
ADMIN_GROUP = os.environ.get("ADMIN_GROUP", "admins")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
USER_POOL_ID = os.environ.get("USER_POOL_ID", "")
COGNITO_REGION = os.environ.get("COGNITO_REGION", "us-east-1")

# DynamoDB resource
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(TABLE_NAME)

# Cognito client (lazy init to avoid cold start penalty when not needed)
_cognito_client = None


def _get_cognito_client():
    global _cognito_client
    if _cognito_client is None:
        _cognito_client = boto3.client("cognito-idp", region_name=COGNITO_REGION)
    return _cognito_client


# Valid event types
VALID_EVENT_TYPES = {
    "page_view",
    "section_change",
    "model_detail_open",
    "comparison_add",
    "comparison_remove",
    "comparison_clear",
    "favorite_toggle",
    "search_filter",
    "region_availability_view",
}

# TTL: 7 days for session buckets
SESSION_BUCKET_TTL_DAYS = 7


def lambda_handler(event, context):
    """Route dispatcher."""
    route_key = event.get("routeKey", "")

    if route_key == "POST /events":
        return handle_post_events(event)
    elif route_key == "GET /dashboard":
        return handle_get_dashboard(event)
    else:
        return response(404, {"error": "Not found"})


# ─── POST /events ──────────────────────────────────────────────────────────


def handle_post_events(event):
    """Record anonymous usage events."""
    try:
        body = json.loads(event.get("body", "{}"))
    except (json.JSONDecodeError, TypeError):
        return response(400, {"error": "Invalid JSON body"})

    events = body.get("events", [])
    auid = body.get("auid", "")
    country = body.get("country", "Unknown")
    region = body.get("region", "")

    if not events or not auid:
        return response(400, {"error": "Missing required fields: events, auid"})

    if len(events) > 50:
        return response(400, {"error": "Maximum 50 events per batch"})

    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    ts = int(now.timestamp() * 1000)

    # Counters for daily aggregate + session bucket
    view_count = 0
    event_count = 0
    section_counts = {}
    feature_counts = {"modelDetails": 0, "comparisons": 0, "favorites": 0}
    model_counts = {}
    compared_models = {}
    favorited_models = {}
    provider_comparisons = {}
    provider_favorites = {}

    # Count events for aggregates (no raw event writes)
    for evt in events:
        evt_type = evt.get("type", "")
        if evt_type not in VALID_EVENT_TYPES:
            continue

        event_count += 1
        section = evt.get("section", "")
        meta = evt.get("meta", {})
        model_id = meta.get("modelId", "")
        provider = meta.get("provider", "")

        if evt_type in ("page_view", "section_change"):
            view_count += 1
        if section:
            section_counts[section] = section_counts.get(section, 0) + 1
        if evt_type == "model_detail_open":
            feature_counts["modelDetails"] += 1
            if model_id:
                model_counts[model_id] = model_counts.get(model_id, 0) + 1
        elif evt_type == "comparison_add":
            feature_counts["comparisons"] += 1
            if model_id:
                compared_models[model_id] = compared_models.get(model_id, 0) + 1
            if provider:
                provider_comparisons[provider] = (
                    provider_comparisons.get(provider, 0) + 1
                )
        elif evt_type == "comparison_remove":
            # Don't count removes as comparisons — track separately (BUG 7 fix)
            pass
        elif evt_type == "comparison_clear":
            # Don't count clears as comparisons (BUG 7 fix)
            pass
        elif evt_type == "favorite_toggle":
            feature_counts["favorites"] += 1
            if model_id:
                favorited_models[model_id] = favorited_models.get(model_id, 0) + 1
            if provider:
                provider_favorites[provider] = provider_favorites.get(provider, 0) + 1

    # Update daily aggregate (unchanged — efficient single UpdateItem)
    _update_daily_aggregate(
        today,
        view_count,
        section_counts,
        feature_counts,
        model_counts,
        compared_models,
        favorited_models,
        provider_comparisons,
        provider_favorites,
        country,
        region,
    )

    # Upsert session bucket (replaces raw event writes)
    _upsert_session_bucket(
        now,
        auid,
        view_count,
        event_count,
        section_counts,
        feature_counts,
        country,
    )

    _upsert_user(auid, today, country, region)

    return response(200, {"status": "ok", "recorded": event_count})


# ─── Cognito Data ──────────────────────────────────────────────────────────


def _get_cognito_summary(start_date, end_date):
    """Query Cognito summary records from DynamoDB cache."""
    items = []
    kwargs = {
        "KeyConditionExpression": Key("PK").eq("COGNITO#SUMMARY")
        & Key("SK").between(start_date, end_date)
    }
    while True:
        result = table.query(**kwargs)
        items.extend(result.get("Items", []))
        if "LastEvaluatedKey" not in result:
            break
        kwargs["ExclusiveStartKey"] = result["LastEvaluatedKey"]
    return items


def _get_cognito_user_count():
    """Get estimated total users from Cognito (instant, 1 API call)."""
    if not USER_POOL_ID:
        return 0
    try:
        client = _get_cognito_client()
        resp = client.describe_user_pool(UserPoolId=USER_POOL_ID)
        return resp["UserPool"].get("EstimatedNumberOfUsers", 0)
    except Exception as e:
        print(f"Error getting Cognito user count: {e}")
        return 0


def _get_cognito_users():
    """Query all COGNITO#USERS records from DynamoDB.

    Returns list of user dicts with dateCreated, status, enabled, country.
    """
    items = []
    kwargs = {
        "KeyConditionExpression": Key("PK").eq("COGNITO#USERS"),
    }
    while True:
        result = table.query(**kwargs)
        items.extend(result.get("Items", []))
        if "LastEvaluatedKey" not in result:
            break
        kwargs["ExclusiveStartKey"] = result["LastEvaluatedKey"]
    return items


def _build_cognito_summary(cognito_items, start_date=None, end_date=None):
    """Build Cognito summary by querying COGNITO#USERS records directly.

    Instead of summing daily snapshots (which double-counts across days),
    query all user records and compute period-based metrics from dateCreated:
      - totalRegistered: all confirmed+enabled users
      - newUsersInPeriod: users whose dateCreated falls within [start_date, end_date]
      - returningUsersInPeriod: totalRegistered - newUsersInPeriod
      - uniqueUsers: totalRegistered (all registered users have logged in)
      - avgDailyUsers: uniqueUsers / days_in_period
      - usersByCountry: count of USERS per country (not views)
      - dailyBreakdown: from COGNITO#SUMMARY daily snapshots (for charts)
    """
    # Query all user records for accurate period-based computation
    user_records = _get_cognito_users()

    total_registered = 0
    new_in_period = 0
    all_countries = {}

    for user in user_records:
        if not user.get("enabled", False):
            continue
        if user.get("status") not in ("CONFIRMED", "EXTERNAL_PROVIDER"):
            continue

        total_registered += 1

        # Get dateCreated ISO string (written by cognito-sync)
        date_created = user.get("dateCreated", "")
        if date_created and start_date and end_date:
            # Compare date portion (YYYY-MM-DD) of the ISO timestamp
            created_date = date_created[:10]
            if start_date <= created_date <= end_date:
                new_in_period += 1
            # Users created after end_date are not counted

        # Country — count of USERS, not views
        country = user.get("country", "")
        if country:
            all_countries[country] = all_countries.get(country, 0) + 1

    # Returning = all registered users who existed before the period
    returning_in_period = total_registered - new_in_period

    # All registered users have logged in at least once (OIDC registration)
    unique_users = total_registered

    # Compute days in period for avg daily users
    days_in_period = 1
    if start_date and end_date:
        try:
            d_start = datetime.strptime(start_date, "%Y-%m-%d")
            d_end = datetime.strptime(end_date, "%Y-%m-%d")
            days_in_period = max((d_end - d_start).days + 1, 1)
        except (ValueError, TypeError):
            days_in_period = 1

    avg_daily_users = round(unique_users / days_in_period, 1)

    # Build daily breakdown from COGNITO#SUMMARY snapshots (for charts)
    daily = []
    for item in sorted(cognito_items, key=lambda x: x["SK"]):
        date = item["SK"]
        daily.append(
            {
                "date": date,
                "totalUsers": int(item.get("totalUsers", 0)),
                "newUsers": int(item.get("newUsersToday", 0)),
                "returningUsers": int(item.get("returningUsersToday", 0)),
            }
        )

    return {
        "totalRegistered": total_registered,
        "newUsersInPeriod": new_in_period,
        "returningUsersInPeriod": returning_in_period,
        "uniqueUsers": unique_users,
        "avgDailyUsers": avg_daily_users,
        "usersByCountry": [
            {"id": k, "count": v}
            for k, v in sorted(all_countries.items(), key=lambda x: x[1], reverse=True)
        ],
        "dailyBreakdown": daily,
    }


# ─── GET /dashboard ────────────────────────────────────────────────────────


def handle_get_dashboard(event):
    """Return aggregated dashboard data with previous period comparison."""
    if not _is_admin(event):
        return response(403, {"error": "Forbidden"})

    params = event.get("queryStringParameters") or {}
    now = datetime.now(timezone.utc)

    # Parse date range: support start/end or days
    if "start" in params and "end" in params:
        start_date = params["start"]
        end_date = params["end"]
        days = (
            datetime.strptime(end_date, "%Y-%m-%d")
            - datetime.strptime(start_date, "%Y-%m-%d")
        ).days + 1
    else:
        days = min(int(params.get("days", "30")), 365)
        end_date = now.strftime("%Y-%m-%d")
        start_date = (now - timedelta(days=days - 1)).strftime("%Y-%m-%d")

    # Previous period (same length, immediately before)
    prev_end = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    prev_start = (
        datetime.strptime(prev_end, "%Y-%m-%d") - timedelta(days=days - 1)
    ).strftime("%Y-%m-%d")

    # Query current and previous periods
    current_items = _query_aggregates(start_date, end_date)
    prev_items = _query_aggregates(prev_start, prev_end)

    # Build summaries
    summary, time_series, all_countries, all_regions = _build_summary(
        current_items, now, start_date
    )
    prev_summary, _, _, _ = _build_summary(prev_items, now, prev_start)

    # Hourly series for today
    today_str = now.strftime("%Y-%m-%d")
    hourly_series = []
    if start_date <= today_str <= end_date:
        hourly_series = _get_hourly_series(today_str)

    # Query Cognito summary cache + user records for period-based metrics
    cognito_items = _get_cognito_summary(start_date, end_date)
    cognito_summary = _build_cognito_summary(cognito_items, start_date, end_date)

    # Get instant user count
    cognito_user_count = _get_cognito_user_count()

    return response(
        200,
        {
            "summary": summary,
            "previousPeriod": prev_summary,
            "timeSeries": time_series,
            "hourlySeries": hourly_series,
            "countries": list(all_countries),
            "regions": list(all_regions),
            "period": {"start": start_date, "end": end_date, "days": days},
            "cognito": {
                "loggedInUsers": cognito_user_count,
                **cognito_summary,
            },
        },
    )


def _query_aggregates(start_date, end_date):
    """Query AGG#daily records for a date range with pagination."""
    items = []
    kwargs = {
        "KeyConditionExpression": Key("PK").eq("AGG#daily")
        & Key("SK").between(start_date, end_date)
    }
    while True:
        result = table.query(**kwargs)
        items.extend(result.get("Items", []))
        if "LastEvaluatedKey" not in result:
            break
        kwargs["ExclusiveStartKey"] = result["LastEvaluatedKey"]
    return items


def _count_returning_users(unique_user_ids, start_date):
    """Count users whose firstSeen is before the period start (BUG 2 fix).

    For multi-day periods, a user who is "new" on day 1 but returns on day 5
    should be counted as returning. The old `unique - new` approach double-
    subtracted these users. This queries USER# records to check firstSeen.

    Optimization: For large user sets (>500), try Cognito summary cache first
    to avoid expensive batch_get_item calls.
    """
    if not unique_user_ids:
        return 0

    # For large sets, try Cognito summary cache first
    if len(unique_user_ids) > 500:
        try:
            result = table.get_item(
                Key={"PK": "COGNITO#SUMMARY", "SK": start_date},
                ProjectionExpression="returningUsersToday",
            )
            cached = result.get("Item", {}).get("returningUsersToday")
            if cached is not None:
                return int(cached)
        except Exception:
            pass  # Fall through to batch_get_item

    # Fallback: batch get (works well for <500 users)
    returning = 0
    user_list = list(unique_user_ids)
    # DynamoDB batch_get_item limit is 100 keys per call
    for i in range(0, len(user_list), 100):
        batch = user_list[i : i + 100]
        keys = [{"PK": f"USER#{uid}", "SK": "META"} for uid in batch]
        try:
            resp = dynamodb.meta.client.batch_get_item(
                RequestItems={
                    TABLE_NAME: {
                        "Keys": keys,
                        "ProjectionExpression": "firstSeen",
                    }
                }
            )
            for item in resp.get("Responses", {}).get(TABLE_NAME, []):
                if item.get("firstSeen", "") < start_date:
                    returning += 1
        except Exception as e:
            print(f"Error in batch_get_item for returning users: {e}")
    return returning


def _build_summary(items, now, start_date=None):
    """Build summary, time_series, countries, regions from aggregate items."""
    time_series = []
    total_views = 0
    all_unique_users = set()
    all_new_users = set()
    all_countries = set()
    all_regions = set()
    total_sections = {}
    total_features = {"modelDetails": 0, "comparisons": 0, "favorites": 0}
    total_model_counts = {}
    total_compared_models = {}
    total_favorited_models = {}
    total_provider_comparisons = {}
    total_provider_favorites = {}
    total_country_views = {}
    total_region_views = {}

    for item in items:
        date = item["SK"]
        views = int(item.get("views", 0))
        unique = _to_set(item.get("uniqueUsers", set()))
        new_users = _to_set(item.get("newUsers", set()))
        sections = _to_dict(item.get("sections", {}))
        features = _to_dict(item.get("features", {}))
        countries = _to_set(item.get("countries", set()))
        regions = _to_set(item.get("regions", set()))
        models = _to_dict(item.get("topModels", {}))
        compared = _to_dict(item.get("comparedModels", {}))
        favorited = _to_dict(item.get("favoritedModels", {}))
        prov_comp = _to_dict(item.get("providerComparisons", {}))
        prov_fav = _to_dict(item.get("providerFavorites", {}))

        total_views += views
        all_unique_users.update(unique)
        all_new_users.update(new_users)
        all_countries.update(countries)
        all_regions.update(regions)

        for k, v in sections.items():
            total_sections[k] = total_sections.get(k, 0) + int(v)
        for k, v in features.items():
            total_features[k] = total_features.get(k, 0) + int(v)
        for k, v in models.items():
            total_model_counts[k] = total_model_counts.get(k, 0) + int(v)
        for k, v in compared.items():
            total_compared_models[k] = total_compared_models.get(k, 0) + int(v)
        for k, v in favorited.items():
            total_favorited_models[k] = total_favorited_models.get(k, 0) + int(v)
        for k, v in prov_comp.items():
            total_provider_comparisons[k] = total_provider_comparisons.get(k, 0) + int(
                v
            )
        for k, v in prov_fav.items():
            total_provider_favorites[k] = total_provider_favorites.get(k, 0) + int(v)

        # BUG 1 & 8 fix: Accumulate actual view counts from countryViews/regionViews maps
        # Fall back to old countries/regions Sets for historical days that lack the new maps
        country_views = _to_dict(item.get("countryViews", {}))
        if country_views:
            for k, v in country_views.items():
                total_country_views[k] = total_country_views.get(k, 0) + int(v)
        else:
            # Backward compat: old daily aggregates only have a countries Set (no view counts)
            # Count each country as 1 view per day it appeared — the old behavior
            for c in countries:
                total_country_views[c] = total_country_views.get(c, 0) + 1

        region_views = _to_dict(item.get("regionViews", {}))
        if region_views:
            for k, v in region_views.items():
                total_region_views[k] = total_region_views.get(k, 0) + int(v)
        else:
            # Backward compat: old daily aggregates only have a regions Set (no view counts)
            for r in regions:
                total_region_views[r] = total_region_views.get(r, 0) + 1

        time_series.append(
            {
                "date": date,
                "views": views,
                "uniqueUsers": len(unique),
                "newUsers": len(new_users),
                "returningUsers": max(0, len(unique) - len(new_users)),
                "sections": sections,
                "detailOpens": int(features.get("modelDetails", 0)),
                "comparisonAdds": int(features.get("comparisons", 0)),
                "favoriteToggles": int(features.get("favorites", 0)),
                "countries": list(countries),
                "regions": list(regions),
            }
        )

    time_series.sort(key=lambda x: x["date"])

    def _top(d, n=20):
        return [
            {"id": k, "count": v}
            for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]
        ]

    today_str = now.strftime("%Y-%m-%d")
    today_data = next((ts for ts in time_series if ts["date"] == today_str), None)

    # BUG 2 fix: Use firstSeen-based calculation for multi-day returning users
    if start_date and all_unique_users:
        returning_users = _count_returning_users(all_unique_users, start_date)
    else:
        # Fallback for backward compat (e.g. previous period summary)
        returning_users = max(0, len(all_unique_users) - len(all_new_users))

    # Comparison winner: model that appears most frequently in comparisons
    comparison_winner = None
    if total_compared_models:
        winner_id = max(total_compared_models, key=total_compared_models.get)
        comparison_winner = {
            "modelId": winner_id,
            "comparisons": total_compared_models[winner_id],
            "totalComparisons": sum(total_compared_models.values()),
        }

    # Average daily unique users (only days with activity)
    daily_user_counts = [
        ts["uniqueUsers"] for ts in time_series if ts["uniqueUsers"] > 0
    ]
    avg_daily_users = round(sum(daily_user_counts) / max(len(daily_user_counts), 1), 1)

    # Engagement metrics
    views_per_user = round(total_views / max(len(all_unique_users), 1), 1)
    most_active_section = (
        max(total_sections, key=total_sections.get) if total_sections else None
    )

    summary = {
        "totalViews": total_views,
        "uniqueUsers": len(all_unique_users),
        "newUsers": len(all_new_users),
        "returningUsers": returning_users,
        "activeToday": today_data["uniqueUsers"] if today_data else 0,
        "avgDailyViews": round(total_views / max(len(time_series), 1), 1),
        "sectionUsage": total_sections,
        "featureUsage": total_features,
        "topModels": _top(total_model_counts),
        "topComparedModels": _top(total_compared_models),
        "topFavoritedModels": _top(total_favorited_models),
        "providerComparisons": _top(total_provider_comparisons),
        "providerFavorites": _top(total_provider_favorites),
        # BUG 1 fix: Use actual view counts from countryViews map (not day-presence)
        "countryCounts": [
            {"id": k, "count": v}
            for k, v in sorted(
                total_country_views.items(), key=lambda x: x[1], reverse=True
            )
        ],
        # BUG 8 fix: Add regionCounts with actual view counts
        "regionCounts": [
            {"id": k, "count": v}
            for k, v in sorted(
                total_region_views.items(), key=lambda x: x[1], reverse=True
            )
        ],
        "totalRegions": len(all_regions),
        # Task 10: Comparison winner, engagement, and audience metrics
        "comparisonWinner": comparison_winner,
        "avgDailyUsers": avg_daily_users,
        "viewsPerUser": views_per_user,
        "mostActiveSection": most_active_section,
    }

    return summary, time_series, all_countries, all_regions


def _get_hourly_series(today_str):
    """Query session buckets for today and aggregate by hour."""
    hourly = {
        f"{h:02d}:00": {"hour": f"{h:02d}:00", "views": 0, "events": 0, "users": set()}
        for h in range(24)
    }

    kwargs = {
        "KeyConditionExpression": Key("PK").eq(f"SESSION#{today_str}"),
    }
    while True:
        result = table.query(**kwargs)
        for item in result.get("Items", []):
            # SK format: "HH:MM#auid" — extract hour from first 2 chars
            sk = item.get("SK", "")
            try:
                hour = int(sk[:2])
            except (ValueError, IndexError):
                continue
            hour_key = f"{hour:02d}:00"
            if hour_key in hourly:
                hourly[hour_key]["views"] += int(item.get("views", 0))
                hourly[hour_key]["events"] += int(item.get("events", 0))
                item_auid = item.get("auid", "")
                if item_auid:
                    hourly[hour_key]["users"].add(item_auid)
        if "LastEvaluatedKey" not in result:
            break
        kwargs["ExclusiveStartKey"] = result["LastEvaluatedKey"]

    # Convert sets to counts
    series = []
    for h in range(24):
        key = f"{h:02d}:00"
        entry = hourly[key]
        series.append(
            {
                "hour": key,
                "views": entry["views"],
                "events": entry["events"],
                "uniqueUsers": len(entry["users"]),
            }
        )
    return series


# ─── Session Bucket ────────────────────────────────────────────────────────


def _upsert_session_bucket(now, auid, views, events, sections, features, country):
    """Upsert a 5-minute session bucket item (replaces raw event writes).

    Key: SESSION#{YYYY-MM-DD} / {HH:MM}#{auid}
    Multiple flushes within the same 5-min window update the same item.
    """
    today = now.strftime("%Y-%m-%d")
    bucket = f"{now.hour:02d}:{(now.minute // 5) * 5:02d}"
    ttl = int((now + timedelta(days=SESSION_BUCKET_TTL_DAYS)).timestamp())

    try:
        # Core counters + metadata
        update_parts_set = ["#aid = :auid", "#c = :country", "#lt = :ts", "#ttl = :ttl"]
        add_parts = []
        expr_names = {
            "#aid": "auid",
            "#c": "country",
            "#lt": "lastTs",
            "#ttl": "ttl",
        }
        expr_values = {
            ":auid": auid,
            ":country": country,
            ":ts": int(now.timestamp() * 1000),
            ":ttl": ttl,
        }

        if views > 0:
            add_parts.append("#v :views")
            expr_names["#v"] = "views"
            expr_values[":views"] = views

        if events > 0:
            add_parts.append("#e :events")
            expr_names["#e"] = "events"
            expr_values[":events"] = events

        update_expr = "SET " + ", ".join(update_parts_set)
        if add_parts:
            update_expr += " ADD " + ", ".join(add_parts)

        table.update_item(
            Key={"PK": f"SESSION#{today}", "SK": f"{bucket}#{auid}"},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
        )

        # Increment section/feature map counters
        for key, count in sections.items():
            _increment_session_map(today, bucket, auid, "sections", key, count)
        for key, count in features.items():
            if count > 0:
                _increment_session_map(today, bucket, auid, "features", key, count)

    except Exception as e:
        print(f"Error upserting session bucket: {e}")


def _increment_session_map(today, bucket, auid, map_attr, key, count):
    """Atomically increment a counter in a session bucket map attribute."""
    if not key or count <= 0:
        return
    session_key = {"PK": f"SESSION#{today}", "SK": f"{bucket}#{auid}"}
    try:
        table.update_item(
            Key=session_key,
            UpdateExpression="ADD #m.#k :val",
            ExpressionAttributeNames={"#m": map_attr, "#k": key},
            ExpressionAttributeValues={":val": count},
        )
    except Exception:
        try:
            table.update_item(
                Key=session_key,
                UpdateExpression="SET #m = if_not_exists(#m, :empty)",
                ExpressionAttributeNames={"#m": map_attr},
                ExpressionAttributeValues={":empty": {}},
            )
            table.update_item(
                Key=session_key,
                UpdateExpression="ADD #m.#k :val",
                ExpressionAttributeNames={"#m": map_attr, "#k": key},
                ExpressionAttributeValues={":val": count},
            )
        except Exception as e:
            print(f"Error updating session {map_attr}.{key}: {e}")


# ─── Aggregate Updates ─────────────────────────────────────────────────────


def _increment_map_counter(today, map_attr, key, count):
    """Atomically increment a counter inside a DynamoDB map attribute."""
    if not key or count <= 0:
        return
    try:
        table.update_item(
            Key={"PK": "AGG#daily", "SK": today},
            UpdateExpression="ADD #m.#k :val",
            ExpressionAttributeNames={"#m": map_attr, "#k": key},
            ExpressionAttributeValues={":val": count},
        )
    except Exception:
        try:
            table.update_item(
                Key={"PK": "AGG#daily", "SK": today},
                UpdateExpression="SET #m = if_not_exists(#m, :empty)",
                ExpressionAttributeNames={"#m": map_attr},
                ExpressionAttributeValues={":empty": {}},
            )
            table.update_item(
                Key={"PK": "AGG#daily", "SK": today},
                UpdateExpression="ADD #m.#k :val",
                ExpressionAttributeNames={"#m": map_attr, "#k": key},
                ExpressionAttributeValues={":val": count},
            )
        except Exception as e:
            print(f"Error updating {map_attr}.{key}: {e}")


def _update_daily_aggregate(
    today,
    views,
    sections,
    features,
    model_counts,
    compared_models,
    favorited_models,
    provider_comparisons,
    provider_favorites,
    country,
    region="",
):
    """Atomically update daily aggregate counters."""
    set_parts = ["#ts = :now"]
    add_parts = []
    expr_names = {"#ts": "updatedAt"}
    expr_values = {":now": int(time.time() * 1000)}

    if views > 0:
        add_parts.append("#v :views")
        expr_names["#v"] = "views"
        expr_values[":views"] = views

    if country and country != "Unknown":
        add_parts.append("#c :country")
        expr_names["#c"] = "countries"
        expr_values[":country"] = {country}

    if region:
        add_parts.append("#r :region")
        expr_names["#r"] = "regions"
        expr_values[":region"] = {region}

    update_expr = "SET " + ", ".join(set_parts)
    if add_parts:
        update_expr += " ADD " + ", ".join(add_parts)

    try:
        table.update_item(
            Key={"PK": "AGG#daily", "SK": today},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
        )
    except Exception as e:
        print(f"Error updating daily aggregate: {e}")

    for key, count in sections.items():
        _increment_map_counter(today, "sections", key, count)
    for key, count in features.items():
        _increment_map_counter(today, "features", key, count)
    for key, count in model_counts.items():
        _increment_map_counter(today, "topModels", key, count)
    for key, count in compared_models.items():
        _increment_map_counter(today, "comparedModels", key, count)
    for key, count in favorited_models.items():
        _increment_map_counter(today, "favoritedModels", key, count)
    for key, count in provider_comparisons.items():
        _increment_map_counter(today, "providerComparisons", key, count)
    for key, count in provider_favorites.items():
        _increment_map_counter(today, "providerFavorites", key, count)

    # BUG 1 fix: Increment countryViews map counter (actual view counts, not day-presence)
    if country and country != "Unknown" and views > 0:
        _increment_map_counter(today, "countryViews", country, views)

    # BUG 8 fix: Increment regionViews map counter (actual view counts, not day-presence)
    if region and views > 0:
        _increment_map_counter(today, "regionViews", region, views)


def _upsert_user(auid, today, country, region=""):
    """Create or update anonymous user tracking record."""
    try:
        update_expr = (
            "SET firstSeen = if_not_exists(firstSeen, :today), "
            "lastSeen = :today, "
            "country = :country"
        )
        expr_values = {":today": today, ":country": country, ":one": 1}
        if region:
            update_expr += ", #r = :region"
        update_expr += " ADD visitCount :one"

        expr_names = {}
        if region:
            expr_names["#r"] = "region"
            expr_values[":region"] = region

        kwargs = {
            "Key": {"PK": f"USER#{auid}", "SK": "META"},
            "UpdateExpression": update_expr,
            "ExpressionAttributeValues": expr_values,
        }
        if expr_names:
            kwargs["ExpressionAttributeNames"] = expr_names

        table.update_item(**kwargs)

        result = table.get_item(
            Key={"PK": f"USER#{auid}", "SK": "META"},
            ProjectionExpression="firstSeen",
        )
        is_new = result.get("Item", {}).get("firstSeen") == today

        update_expr = "ADD uniqueUsers :auid_set"
        expr_values = {":auid_set": {auid}}
        if is_new:
            update_expr += ", newUsers :auid_set"

        table.update_item(
            Key={"PK": "AGG#daily", "SK": today},
            UpdateExpression=update_expr,
            ExpressionAttributeValues=expr_values,
        )
    except Exception as e:
        print(f"Error upserting user: {e}")


# ─── Helpers ───────────────────────────────────────────────────────────────


def _is_admin(event):
    """Check if the caller belongs to the admin Cognito group."""
    try:
        claims = (
            event.get("requestContext", {})
            .get("authorizer", {})
            .get("jwt", {})
            .get("claims", {})
        )
        groups_claim = claims.get("cognito:groups", "")
        if isinstance(groups_claim, list):
            groups = groups_claim
        elif isinstance(groups_claim, str):
            stripped = groups_claim.strip("[] ")
            if not stripped:
                groups = []
            else:
                try:
                    groups = json.loads(groups_claim)
                except (json.JSONDecodeError, ValueError):
                    groups = [
                        g.strip().strip('"')
                        for g in stripped.replace(",", " ").split()
                        if g.strip()
                    ]
        else:
            groups = []
        return ADMIN_GROUP in groups
    except (AttributeError, TypeError):
        return False


def _to_set(val):
    """Convert DynamoDB set/list to Python set."""
    if isinstance(val, set):
        return val
    if isinstance(val, list):
        return set(val)
    return set()


def _to_dict(val):
    """Convert DynamoDB map to Python dict with int values."""
    if not isinstance(val, dict):
        return {}
    return {k: int(v) for k, v in val.items()}


def response(status_code, body):
    """Build API Gateway response with CORS headers."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": ALLOWED_ORIGINS,
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        },
        "body": json.dumps(body, default=_json_serializer),
    }


def _json_serializer(obj):
    """Handle Decimal and set serialization for JSON."""
    if isinstance(obj, Decimal):
        return int(obj) if obj == int(obj) else float(obj)
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type {type(obj)} is not JSON serializable")
