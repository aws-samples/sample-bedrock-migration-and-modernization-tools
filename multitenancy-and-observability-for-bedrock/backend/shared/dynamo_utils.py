# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""DynamoDB helper functions for profile, dashboard, alert, and pricing access patterns.

All functions accept table name strings and create Table references internally.
"""

import logging
from decimal import Decimal
from typing import Optional

import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger(__name__)

_dynamodb = None


def _get_dynamodb():
    """Lazy-init DynamoDB resource (reused across invocations)."""
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource("dynamodb")
    return _dynamodb


def _table(table_name: str):
    """Get a DynamoDB Table resource by name."""
    return _get_dynamodb().Table(table_name)


def _now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Profile operations
# ---------------------------------------------------------------------------

def get_profile(table_name: str, tenant_id: str) -> Optional[dict]:
    """Get a single profile by tenant_id."""
    table = _table(table_name)
    response = table.get_item(Key={"tenant_id": tenant_id})
    item = response.get("Item")
    if not item:
        return None
    return _deserialize_profile(item)


def list_profiles(table_name: str, limit: int = 100, last_key: Optional[dict] = None) -> dict:
    """Paginated scan of all profiles.

    Returns:
        dict with keys: tenants (list), last_key (dict or None)
    """
    table = _table(table_name)
    kwargs = {"Limit": limit}
    if last_key:
        kwargs["ExclusiveStartKey"] = last_key

    response = table.scan(**kwargs)
    profiles = [_deserialize_profile(item) for item in response.get("Items", [])]

    return {
        "profiles": profiles,
        "last_key": response.get("LastEvaluatedKey"),
    }


def put_profile(table_name: str, profile: dict) -> None:
    """Create or update a profile. Expects a dict with at least tenant_id."""
    table = _table(table_name)
    now = _now_iso()
    item = dict(profile)

    if not item.get("created_at"):
        item["created_at"] = now
    item["updated_at"] = now

    # Convert floats to Decimal for DynamoDB
    item = _serialize_for_dynamo(item)
    table.put_item(Item=item)


def update_profile_status(table_name: str, tenant_id: str, status: str) -> dict:
    """Update only the status and updated_at fields for a profile."""
    table = _table(table_name)
    now = _now_iso()

    response = table.update_item(
        Key={"tenant_id": tenant_id},
        UpdateExpression="SET #s = :status, updated_at = :updated_at",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":status": status,
            ":updated_at": now,
        },
        ReturnValues="ALL_NEW",
    )
    return _deserialize_profile(response.get("Attributes", {}))


def delete_profile(table_name: str, tenant_id: str) -> None:
    """Delete a profile by tenant_id."""
    table = _table(table_name)
    table.delete_item(Key={"tenant_id": tenant_id})


# ---------------------------------------------------------------------------
# Dashboard operations
# ---------------------------------------------------------------------------

def put_dashboard(table_name: str, dashboard: dict) -> None:
    """Create or update a dashboard record."""
    table = _table(table_name)
    now = _now_iso()
    item = dict(dashboard)

    if not item.get("created_at"):
        item["created_at"] = now
    item["updated_at"] = now

    item = _serialize_for_dynamo(item)
    table.put_item(Item=item)


def get_dashboard(table_name: str, dashboard_id: str) -> Optional[dict]:
    """Get a single dashboard by dashboard_id."""
    table = _table(table_name)
    response = table.get_item(Key={"dashboard_id": dashboard_id})
    item = response.get("Item")
    if not item:
        return None
    return _deserialize_record(item)


def list_dashboards_by_tenant(table_name: str, index_name: str, tenant_id: str) -> list:
    """Query dashboards by tenant_id using a GSI."""
    table = _table(table_name)
    response = table.query(
        IndexName=index_name,
        KeyConditionExpression=Key("tenant_id").eq(tenant_id),
    )
    return [_deserialize_record(item) for item in response.get("Items", [])]


def delete_dashboard(table_name: str, dashboard_id: str) -> None:
    """Delete a dashboard by dashboard_id."""
    table = _table(table_name)
    table.delete_item(Key={"dashboard_id": dashboard_id})


# ---------------------------------------------------------------------------
# Alert operations
# ---------------------------------------------------------------------------

def put_alert(table_name: str, alert: dict) -> None:
    """Create or update an alert record."""
    table = _table(table_name)
    now = _now_iso()
    item = dict(alert)

    if not item.get("created_at"):
        item["created_at"] = now
    item["updated_at"] = now

    item = _serialize_for_dynamo(item)
    table.put_item(Item=item)


def get_alert(table_name: str, alert_id: str) -> Optional[dict]:
    """Get a single alert by alert_id."""
    table = _table(table_name)
    response = table.get_item(Key={"alert_id": alert_id})
    item = response.get("Item")
    if not item:
        return None
    return _deserialize_record(item)


def list_alerts_by_tenant(table_name: str, index_name: str, tenant_id: str) -> list:
    """Query alerts by tenant_id using a GSI."""
    table = _table(table_name)
    response = table.query(
        IndexName=index_name,
        KeyConditionExpression=Key("tenant_id").eq(tenant_id),
    )
    return [_deserialize_record(item) for item in response.get("Items", [])]


def delete_alert(table_name: str, alert_id: str) -> None:
    """Delete an alert by alert_id."""
    table = _table(table_name)
    table.delete_item(Key={"alert_id": alert_id})


# ---------------------------------------------------------------------------
# Pricing operations
# ---------------------------------------------------------------------------

def get_pricing(table_name: str, cache_key: str) -> Optional[dict]:
    """Get a pricing cache entry by cache_key."""
    table = _table(table_name)
    response = table.get_item(Key={"region#model_id": cache_key})
    item = response.get("Item")
    if not item:
        return None
    return _deserialize_pricing(item)


def put_pricing(table_name: str, pricing_entry: dict) -> None:
    """Put a pricing entry with TTL."""
    table = _table(table_name)
    item = _serialize_for_dynamo(pricing_entry)
    table.put_item(Item=item)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_for_dynamo(data: dict) -> dict:
    """Convert Python types to DynamoDB-compatible types."""
    result = {}
    for k, v in data.items():
        if isinstance(v, float):
            result[k] = Decimal(str(v))
        elif isinstance(v, dict):
            result[k] = _serialize_for_dynamo(v)
        elif isinstance(v, list):
            result[k] = [Decimal(str(i)) if isinstance(i, float) else i for i in v]
        else:
            result[k] = v
    return result


def _deserialize_profile(item: dict) -> dict:
    """Convert DynamoDB item back to a plain dict with Python types."""
    result = {}
    for k, v in item.items():
        if isinstance(v, Decimal):
            result[k] = float(v) if "." in str(v) else int(v)
        else:
            result[k] = v
    return result


def _deserialize_record(item: dict) -> dict:
    """Generic deserializer: convert DynamoDB item back to a plain dict."""
    result = {}
    for k, v in item.items():
        if isinstance(v, Decimal):
            result[k] = float(v) if "." in str(v) else int(v)
        elif isinstance(v, dict):
            result[k] = _deserialize_record(v)
        else:
            result[k] = v
    return result


def _deserialize_pricing(item: dict) -> dict:
    """Convert DynamoDB pricing item back to plain dict."""
    result = {}
    for k, v in item.items():
        if isinstance(v, Decimal):
            result[k] = float(v)
        else:
            result[k] = v
    return result
