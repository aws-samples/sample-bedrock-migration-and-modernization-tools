# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Utilities for Cost Allocation Tag integration and tag-based filtering."""

import logging
import os

logger = logging.getLogger(__name__)

PREDEFINED_TAG_CATEGORIES = ("Tenant", "Environment", "Region", "User", "Model", "Application")

TAG_DIMENSION_PREFIX = "Tag_"


def validate_tags(tags, tenant_name: str) -> dict:
    """Validate and normalize tags against the predefined category schema.

    - Auto-fills the ``Tenant`` tag from *tenant_name* when missing or empty.
    - Only includes predefined categories that have non-empty values.
    - Allows custom tag keys alongside predefined categories.
    - Raises ``ValueError`` for empty custom tag keys or values that exceed
      CloudWatch dimension limits (256 chars).

    Returns a new dict with the validated tags.
    """
    if not isinstance(tags, dict):
        tags = {}

    validated: dict = {}

    # Include predefined categories only when a value is provided
    for cat in PREDEFINED_TAG_CATEGORIES:
        val = tags.get(cat, "")
        if isinstance(val, list):
            val = " / ".join(str(v) for v in val)
        val = str(val).strip()
        if val:
            validated[cat] = val

    # Auto-fill Tenant if empty
    if "Tenant" not in validated:
        validated["Tenant"] = tenant_name

    # Add custom (non-predefined) keys
    for k, v in tags.items():
        if k in PREDEFINED_TAG_CATEGORIES:
            continue
        k = k.strip()
        if not k:
            raise ValueError("Custom tag key must not be empty")
        if isinstance(v, list):
            v = " / ".join(str(x) for x in v)
        v = str(v).strip()
        if len(v) > 256:
            raise ValueError(f"Tag value for '{k}' exceeds 256 characters")
        if v:
            validated[k] = v

    return validated


def tags_to_cw_dimensions(tags: dict) -> list[dict]:
    """Convert a profile's tags dict to CloudWatch dimension entries.

    Returns a list of ``{"Name": "Tag_<Key>", "Value": "<value>"}`` dicts,
    skipping tags with empty values.
    """
    dims: list[dict] = []
    for k, v in (tags or {}).items():
        val = " / ".join(v) if isinstance(v, list) else str(v)
        if val:
            dims.append({"Name": f"{TAG_DIMENSION_PREFIX}{k}", "Value": val})
    return dims


def parse_tag_filters(raw: str) -> dict:
    """Parse 'Environment:Dev,team:platform' into {'Environment': 'Dev', 'team': 'platform'}."""
    if not raw or not raw.strip():
        return {}
    result = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        key, value = pair.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            result[key] = value
    return result


def filter_by_tags(items: list, tag_filters: dict) -> list:
    """Filter a list of dicts by matching tags. Each item must have a 'tags' dict field."""
    if not tag_filters:
        return items
    filtered = []
    for item in items:
        item_tags = item.get("tags") or {}
        if all(item_tags.get(k) == v for k, v in tag_filters.items()):
            filtered.append(item)
    return filtered


def get_profile_ids_for_tags(tenants_table: str, tag_filters: dict) -> set:
    """Scan profiles table and return profile IDs whose tags match all filters."""
    if not tag_filters:
        return set()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from shared import dynamo_utils

    result = dynamo_utils.list_profiles(tenants_table, limit=1000)
    matching_ids = set()
    for profile in result.get("profiles", []):
        profile_tags = profile.get("tags") or {}
        if all(profile_tags.get(k) == v for k, v in tag_filters.items()):
            matching_ids.add(profile.get("tenant_id"))
    return matching_ids
