"""
Lifecycle Collector Lambda

Scrapes model lifecycle data from AWS Bedrock documentation.
Source: https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html

Includes TTL-based caching to reduce external scraping.
"""

import os
import re
import time
from datetime import datetime
from typing import Any, Callable, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from shared import (
    get_s3_client,
    read_from_s3,
    write_to_s3,
    get_config_loader,
)
from shared.powertools import logger, tracer, metrics, LambdaContext
from aws_lambda_powertools.metrics import MetricUnit

# Configuration
DATA_BUCKET = os.environ.get("DATA_BUCKET")
REQUEST_TIMEOUT = 30  # seconds

# Cache configuration
LIFECYCLE_CACHE_KEY = "cache/lifecycle_data.json"
DEFAULT_CACHE_TTL_HOURS = 24


def get_lifecycle_url() -> str:
    """Get model lifecycle documentation URL from config."""
    config = get_config_loader()
    return config.get_documentation_url("bedrock_model_lifecycle")


def get_cached_or_fetch(
    s3_client: Any,
    bucket: str,
    cache_key: str,
    fetch_fn: Callable[[], dict],
    source_url: str,
    ttl_hours: int = DEFAULT_CACHE_TTL_HOURS,
) -> Tuple[dict, bool]:
    """
    Get data from cache if valid, otherwise fetch and cache.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        cache_key: S3 key for cache file
        fetch_fn: Function to call if cache miss (returns data dict)
        source_url: URL of the data source (for metadata)
        ttl_hours: Cache TTL in hours (default 24)

    Returns:
        Tuple of (data, from_cache)
    """
    try:
        cached = read_from_s3(s3_client, bucket, cache_key, default_on_missing=None)
        if cached:
            cached_at_str = cached.get("cached_at")
            if cached_at_str:
                cached_at = datetime.strptime(cached_at_str, "%Y-%m-%dT%H:%M:%SZ")
                age_hours = (datetime.utcnow() - cached_at).total_seconds() / 3600
                cache_ttl = cached.get("ttl_hours", ttl_hours)

                if age_hours < cache_ttl:
                    logger.info(
                        "Cache hit - using cached data",
                        extra={
                            "cache_key": cache_key,
                            "age_hours": round(age_hours, 1),
                            "ttl_hours": cache_ttl,
                        },
                    )
                    return cached.get("data", {}), True
                else:
                    logger.info(
                        "Cache expired - fetching fresh data",
                        extra={
                            "cache_key": cache_key,
                            "age_hours": round(age_hours, 1),
                            "ttl_hours": cache_ttl,
                        },
                    )
    except Exception as e:
        logger.warning(
            "Cache read failed, fetching fresh data",
            extra={"cache_key": cache_key, "error": str(e)},
        )

    # Cache miss or expired - fetch fresh data
    logger.info("Fetching fresh data from source", extra={"source_url": source_url})
    data = fetch_fn()

    # Update cache
    cache_data = {
        "cached_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ttl_hours": ttl_hours,
        "source_url": source_url,
        "data": data,
    }
    try:
        write_to_s3(s3_client, bucket, cache_key, cache_data)
        logger.info("Cache updated", extra={"cache_key": cache_key})
    except Exception as e:
        logger.warning(
            "Cache write failed - continuing without caching",
            extra={"cache_key": cache_key, "error": str(e)},
        )

    return data, False


# Regex pattern for AWS region codes
# Matches: us-east-1, us-west-2, eu-central-1, ap-northeast-1, us-gov-west-1, etc.
REGION_PATTERN = re.compile(r"([a-z]{2}(?:-gov)?-[a-z]+-\d)")


def parse_regions_from_text(text: str) -> list[str]:
    """Extract AWS region codes from any text.

    Args:
        text: Any text that may contain region codes

    Returns:
        List of unique region codes found, in order of appearance

    Examples:
        >>> parse_regions_from_text("us-east-1, us-west-2 Regions")
        ['us-east-1', 'us-west-2']
        >>> parse_regions_from_text("Available in us-gov-west-1")
        ['us-gov-west-1']
    """
    if not text:
        return []

    matches = REGION_PATTERN.findall(text)
    # Preserve order while removing duplicates
    seen = set()
    unique_regions = []
    for region in matches:
        if region not in seen:
            seen.add(region)
            unique_regions.append(region)
    return unique_regions


def parse_regions_from_cell(cell) -> list[str]:
    """Extract region codes from a table cell, handling list items.

    Some cells contain itemized lists with one region per item.
    This function extracts all regions from all list items.

    Args:
        cell: BeautifulSoup cell element

    Returns:
        List of unique region codes found
    """
    regions = []

    # Check for itemized list
    ul = cell.find("ul", class_="itemizedlist")
    if ul:
        for li in ul.find_all("li"):
            li_text = li.get_text(strip=True)
            regions.extend(parse_regions_from_text(li_text))
    else:
        # Fall back to full cell text
        cell_text = cell.get_text(strip=True)
        regions.extend(parse_regions_from_text(cell_text))

    # Remove duplicates while preserving order
    seen = set()
    unique_regions = []
    for region in regions:
        if region not in seen:
            seen.add(region)
            unique_regions.append(region)
    return unique_regions


def parse_date_with_regions(text: str) -> dict:
    """Parse a date string that may contain embedded region information.

    AWS lifecycle docs embed regions in date strings like:
    "August 25, 2025 (us-east-1, us-east-2, us-west-2 Regions)"

    Args:
        text: Date string potentially containing region info

    Returns:
        Dictionary with:
        - date: The base date string (without region text)
        - regions: List of region codes found
        - all_regions: True if text indicates "all regions"

    Examples:
        >>> parse_date_with_regions("August 25, 2025 (us-east-1, us-east-2 Regions)")
        {'date': 'August 25, 2025', 'regions': ['us-east-1', 'us-east-2'], 'all_regions': False}
        >>> parse_date_with_regions("No sooner than 9/23/2025")
        {'date': 'No sooner than 9/23/2025', 'regions': [], 'all_regions': False}
    """
    if not text:
        return {"date": "", "regions": [], "all_regions": False}

    text = text.strip()

    # Check for "all regions" indicator
    all_regions = bool(re.search(r"\ball\s+regions?\b", text, re.IGNORECASE))

    # Extract regions from the text
    regions = parse_regions_from_text(text)

    # Extract the base date by removing the parenthetical region info
    # Pattern matches: (region-1, region-2, ... Regions) or (all Regions)
    # Also matches: (us-east-1 and us-west-2) without "Regions" suffix
    date_text = re.sub(r"\s*\([^)]*(?:region|Region)[^)]*\)", "", text).strip()

    # Remove parenthetical content that contains region codes
    # This handles cases like "(us-east-1 and us-west-2)"
    if regions:
        date_text = re.sub(r"\s*\([^)]*\)", "", date_text).strip()

    # Also remove trailing "Regions" if present without parentheses
    date_text = re.sub(r"\s+Regions?\s*$", "", date_text, flags=re.IGNORECASE).strip()

    return {
        "date": date_text,
        "regions": regions,
        "all_regions": all_regions,
    }


def get_first_list_item_or_text(cell) -> str:
    """Extract text from a cell, handling itemized lists.

    Some cells in AWS docs contain itemized lists (one item per region).
    This function extracts just the first item if it's a list,
    otherwise returns the full cell text.
    """
    # Check for itemized list
    ul = cell.find("ul", class_="itemizedlist")
    if ul:
        first_li = ul.find("li")
        if first_li:
            return first_li.get_text(strip=True)
    # Fall back to full text
    return cell.get_text(strip=True)


def get_all_list_items(cell) -> list[str]:
    """Extract all list items from a cell.

    Args:
        cell: BeautifulSoup cell element

    Returns:
        List of text from each list item, or single-item list with cell text
    """
    ul = cell.find("ul", class_="itemizedlist")
    if ul:
        items = []
        for li in ul.find_all("li"):
            items.append(li.get_text(strip=True))
        return items if items else [cell.get_text(strip=True)]
    return [cell.get_text(strip=True)]


@tracer.capture_method
def fetch_lifecycle_page() -> str:
    """Fetch the HTML content from the AWS lifecycle documentation page."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BedrockProfiler/1.0)",
        "Accept": "text/html,application/xhtml+xml",
    }

    lifecycle_url = get_lifecycle_url()
    response = requests.get(lifecycle_url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def get_table_headers(table) -> list[str]:
    """Extract header names from a table."""
    header_row = table.find("tr")
    if header_row:
        return [
            th.get_text(strip=True).lower() for th in header_row.find_all(["th", "td"])
        ]
    return []


def parse_active_table(table) -> list[dict]:
    """Parse the Active models table.

    Columns: Provider, Model name, Model ID, Regions supported, Launch date,
    EOL date, Input modalities, Output modalities

    Enhanced to extract:
    - active_regions: List of region codes where model is active
    - launch_dates_by_region: Dict mapping region to launch date
    """
    models = []
    all_rows = table.find_all("tr")

    # Skip header row
    for row in all_rows[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) < 4:
            continue

        def get_cell_text(idx: int) -> str:
            if idx < len(cells):
                return cells[idx].get_text(strip=True)
            return ""

        # Parse regions from column 4 (index 3)
        active_regions = parse_regions_from_cell(cells[3]) if len(cells) > 3 else []

        # Parse launch dates - may have multiple dates per region
        launch_dates_by_region = {}
        if len(cells) > 4:
            launch_items = get_all_list_items(cells[4])
            for item in launch_items:
                # Extract regions and date from each item
                parsed = parse_date_with_regions(item)
                item_regions = parsed["regions"]
                item_date = parsed["date"]

                if item_regions:
                    # Map each region to its launch date
                    for region in item_regions:
                        if region not in launch_dates_by_region:
                            launch_dates_by_region[region] = item_date
                elif item_date and active_regions:
                    # No specific regions mentioned - apply to all active regions
                    for region in active_regions:
                        if region not in launch_dates_by_region:
                            launch_dates_by_region[region] = item_date

        # Parse EOL date with regions
        eol_parsed = {"date": None, "regions": [], "all_regions": False}
        if len(cells) > 5:
            eol_parsed = parse_date_with_regions(get_cell_text(5))

        model_data = {
            "provider": get_cell_text(0),
            "model_name": get_cell_text(1),
            "model_id": get_cell_text(2),
            "regions": get_cell_text(
                3
            ),  # Keep original text for backward compatibility
            "active_regions": active_regions,  # NEW: parsed region list
            "launch_date": get_first_list_item_or_text(cells[4])
            if len(cells) > 4
            else None,
            "launch_dates_by_region": launch_dates_by_region,  # NEW: region -> date mapping
            "eol_date": eol_parsed["date"]
            if eol_parsed["date"]
            else (get_cell_text(5) if len(cells) > 5 else None),
            "eol_regions": eol_parsed["regions"],  # NEW: regions for EOL date
            "eol_all_regions": eol_parsed["all_regions"],  # NEW: applies to all regions
            "input_modalities": get_cell_text(6) if len(cells) > 6 else None,
            "output_modalities": get_cell_text(7) if len(cells) > 7 else None,
            "lifecycle_status": "active",
        }

        if model_data["model_id"]:
            models.append(model_data)

    return models


def parse_legacy_table(table) -> list[dict]:
    """Parse the Legacy models table.

    Columns: Model version, Legacy date, Public extended access date, EOL date,
    Recommended model version replacement, Recommended model ID

    Note: The "Recommended model ID" column contains the ID of the REPLACEMENT model,
    not the legacy model itself. The legacy model is identified by model_name (Model version).

    Enhanced to extract:
    - legacy_regions: List of region codes where model is legacy
    - eol_regions: List of region codes for EOL date
    - extended_access_regions: List of region codes for extended access date

    Handles continuation rows: Some models have multiple rows with different regional
    dates. Continuation rows have fewer columns (typically 3) and contain date/region
    info that should be associated with the previous model. The first cell of a
    continuation row contains the legacy date (not a model name).
    """
    models = []
    all_rows = table.find_all("tr")

    # Track the current model name for continuation rows
    current_model_name = None
    current_recommended_replacement = None
    current_recommended_model_id = None

    # Skip header row
    for row in all_rows[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) < 3:
            continue

        def get_cell_text(idx: int) -> str:
            if idx < len(cells):
                return cells[idx].get_text(strip=True)
            return ""

        # Detect if this is a continuation row
        # Continuation rows have fewer cells (typically 3) and the first cell
        # contains date/region info instead of a model name
        first_cell_text = get_cell_text(0)
        first_cell_has_regions = bool(parse_regions_from_text(first_cell_text))
        is_continuation_row = len(cells) <= 4 and first_cell_has_regions

        if is_continuation_row:
            # This is a continuation row - use the previous model name
            # The columns are shifted: cell 0 = legacy date, cell 1 = extended access, cell 2 = EOL
            if not current_model_name:
                continue  # Skip if we haven't seen a model yet

            # Parse dates from shifted columns
            legacy_parsed = parse_date_with_regions(get_cell_text(0))
            extended_parsed = parse_date_with_regions(get_cell_text(1))
            eol_parsed = parse_date_with_regions(get_cell_text(2))

            model_data = {
                "model_name": current_model_name,
                "legacy_date": legacy_parsed["date"] or get_cell_text(0),
                "legacy_regions": legacy_parsed["regions"],
                "legacy_all_regions": legacy_parsed["all_regions"],
                "extended_access_date": extended_parsed["date"],
                "extended_access_regions": extended_parsed["regions"],
                "extended_access_all_regions": extended_parsed["all_regions"],
                "eol_date": eol_parsed["date"] or get_cell_text(2),
                "eol_regions": eol_parsed["regions"],
                "eol_all_regions": eol_parsed["all_regions"],
                "recommended_replacement": current_recommended_replacement,
                "recommended_model_id": current_recommended_model_id,
                "lifecycle_status": "legacy",
                "provider": None,
                "regions": None,
                "active_regions": [],
                "launch_date": None,
                "launch_dates_by_region": {},
                "input_modalities": None,
                "output_modalities": None,
            }
        else:
            # This is a regular row with a model name
            row_model_name = first_cell_text

            if row_model_name:
                current_model_name = row_model_name
                # Capture replacement info from rows with model names
                current_recommended_replacement = (
                    get_cell_text(4) if len(cells) > 5 else get_cell_text(3)
                )
                current_recommended_model_id = (
                    get_cell_text(5) if len(cells) > 5 else get_cell_text(4)
                )
            else:
                # Empty model name and not a continuation row - skip
                continue

            # Parse legacy date with regions
            legacy_parsed = parse_date_with_regions(get_cell_text(1))

            # Parse extended access date with regions (only with 6+ columns)
            extended_parsed = {"date": None, "regions": [], "all_regions": False}
            if len(cells) > 5:
                extended_parsed = parse_date_with_regions(get_cell_text(2))

            # Parse EOL date with regions
            eol_idx = 3 if len(cells) > 5 else 2
            eol_parsed = parse_date_with_regions(get_cell_text(eol_idx))

            model_data = {
                "model_name": current_model_name,
                "legacy_date": legacy_parsed["date"] or get_cell_text(1),
                "legacy_regions": legacy_parsed["regions"],
                "legacy_all_regions": legacy_parsed["all_regions"],
                "extended_access_date": extended_parsed["date"]
                if len(cells) > 5
                else None,
                "extended_access_regions": extended_parsed["regions"],
                "extended_access_all_regions": extended_parsed["all_regions"],
                "eol_date": eol_parsed["date"] or get_cell_text(eol_idx),
                "eol_regions": eol_parsed["regions"],
                "eol_all_regions": eol_parsed["all_regions"],
                "recommended_replacement": current_recommended_replacement,
                "recommended_model_id": current_recommended_model_id,
                "lifecycle_status": "legacy",
                "provider": None,
                "regions": None,
                "active_regions": [],
                "launch_date": None,
                "launch_dates_by_region": {},
                "input_modalities": None,
                "output_modalities": None,
            }

        # Each row (including continuation rows) creates a separate entry
        # The build_regional_lifecycle function will merge them by model_name
        models.append(model_data)

    return models


def parse_eol_table(table) -> list[dict]:
    """Parse the EOL (End-of-Life) models table.

    Columns: Model version, Legacy date, EOL date, Recommended model version
    replacement, Recommended model ID

    Note: The "Recommended model ID" column contains the ID of the REPLACEMENT
    model, not the EOL model itself. The EOL model is identified by model_name
    (Model version).

    Enhanced to extract:
    - legacy_regions: List of region codes for legacy date
    - eol_regions: List of region codes for EOL date

    Handles continuation rows: Some models may have multiple rows with different
    regional dates. Continuation rows have fewer columns and the first cell
    contains date/region info instead of a model name.
    """
    models = []
    all_rows = table.find_all("tr")

    # Track the current model name for continuation rows
    current_model_name = None
    current_recommended_replacement = None
    current_recommended_model_id = None

    # Skip header row
    for row in all_rows[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue

        def get_cell_text(idx: int) -> str:
            if idx < len(cells):
                return cells[idx].get_text(strip=True)
            return ""

        # Detect if this is a continuation row
        # Continuation rows have fewer cells and the first cell contains
        # date/region info instead of a model name
        first_cell_text = get_cell_text(0)
        first_cell_has_regions = bool(parse_regions_from_text(first_cell_text))
        is_continuation_row = len(cells) <= 3 and first_cell_has_regions

        if is_continuation_row:
            # This is a continuation row - use the previous model name
            # The columns are shifted: cell 0 = legacy date, cell 1 = EOL date
            if not current_model_name:
                continue  # Skip if we haven't seen a model yet

            # Parse dates from shifted columns
            legacy_parsed = parse_date_with_regions(get_cell_text(0))
            eol_parsed = parse_date_with_regions(get_cell_text(1))

            model_data = {
                "model_name": current_model_name,
                "legacy_date": legacy_parsed["date"] or get_cell_text(0),
                "legacy_regions": legacy_parsed["regions"],
                "legacy_all_regions": legacy_parsed["all_regions"],
                "eol_date": eol_parsed["date"] or get_cell_text(1),
                "eol_regions": eol_parsed["regions"],
                "eol_all_regions": eol_parsed["all_regions"],
                "recommended_replacement": current_recommended_replacement,
                "recommended_model_id": current_recommended_model_id,
                "lifecycle_status": "eol",
                "provider": None,
                "regions": None,
                "active_regions": [],
                "launch_date": None,
                "launch_dates_by_region": {},
                "input_modalities": None,
                "output_modalities": None,
                "extended_access_date": None,
                "extended_access_regions": [],
                "extended_access_all_regions": False,
            }
        else:
            # This is a regular row with a model name
            row_model_name = first_cell_text

            if row_model_name:
                current_model_name = row_model_name
                # Capture replacement info from rows with model names
                current_recommended_replacement = get_cell_text(3)
                current_recommended_model_id = get_cell_text(4)
            else:
                # Empty model name and not a continuation row - skip
                continue

            # Parse legacy date with regions
            legacy_parsed = parse_date_with_regions(get_cell_text(1))

            # Parse EOL date with regions
            eol_parsed = parse_date_with_regions(get_cell_text(2))

            model_data = {
                "model_name": current_model_name,
                "legacy_date": legacy_parsed["date"] or get_cell_text(1),
                "legacy_regions": legacy_parsed["regions"],
                "legacy_all_regions": legacy_parsed["all_regions"],
                "eol_date": eol_parsed["date"] or get_cell_text(2),
                "eol_regions": eol_parsed["regions"],
                "eol_all_regions": eol_parsed["all_regions"],
                "recommended_replacement": current_recommended_replacement,
                "recommended_model_id": current_recommended_model_id,
                "lifecycle_status": "eol",
                "provider": None,
                "regions": None,
                "active_regions": [],
                "launch_date": None,
                "launch_dates_by_region": {},
                "input_modalities": None,
                "output_modalities": None,
                "extended_access_date": None,
                "extended_access_regions": [],
                "extended_access_all_regions": False,
            }

        # Each row (including continuation rows) creates a separate entry
        # The build_regional_lifecycle function will merge them by model_name
        models.append(model_data)

    return models

    return models


def parse_lifecycle_table(table, status: str) -> list[dict]:
    """Parse a lifecycle table and extract model information.

    Dispatches to the appropriate parser based on table status/structure.

    Active table columns: Provider, Model name, Model ID, Regions, Launch date,
        EOL date, Input modalities, Output modalities
    Legacy table columns: Model version, Legacy date, Public extended access date,
        EOL date, Recommended model version replacement, Recommended model ID
    EOL table columns: Model version, Legacy date, EOL date,
        Recommended model version replacement, Recommended model ID
    """
    headers = get_table_headers(table)

    # Detect table type by headers
    if "provider" in headers or "model id" in headers:
        return parse_active_table(table)
    elif "public extended access date" in headers:
        return parse_legacy_table(table)
    elif "recommended model id" in headers:
        return parse_eol_table(table)
    else:
        # Fallback: use status to determine parser
        if status == "active":
            return parse_active_table(table)
        elif status == "legacy":
            return parse_legacy_table(table)
        elif status == "eol":
            return parse_eol_table(table)
        else:
            logger.warning(
                f"Unknown table structure for status {status}, headers: {headers}"
            )
            return []


def build_regional_lifecycle(
    active_models: list[dict],
    legacy_models: list[dict],
    eol_models: list[dict],
) -> dict:
    """Build a per-model regional lifecycle structure.

    Combines data from Active, Legacy, and EOL tables to create a comprehensive
    view of each model's lifecycle status across different regions.

    Args:
        active_models: Models from the Active table
        legacy_models: Models from the Legacy table
        eol_models: Models from the EOL table

    Returns:
        Dictionary keyed by model identifier (model_id or model_name) with:
        - model_id: The model ID (if available)
        - model_name: The model name
        - regional_status: Dict mapping region to status details
        - status_summary: Dict mapping status to list of regions
        - recommended_replacement: Replacement model name (if applicable)
        - recommended_model_id: Replacement model ID (if applicable)
    """
    regional_lifecycle = {}

    # Helper to get or create model entry
    def get_or_create_entry(key: str, model_id: Optional[str], model_name: str) -> dict:
        if key not in regional_lifecycle:
            regional_lifecycle[key] = {
                "model_id": model_id,
                "model_name": model_name,
                "regional_status": {},
                "status_summary": {
                    "ACTIVE": [],
                    "LEGACY": [],
                    "EOL": [],
                },
                "recommended_replacement": None,
                "recommended_model_id": None,
            }
        return regional_lifecycle[key]

    # Process Active models
    for model in active_models:
        model_id = model.get("model_id")
        model_name = model.get("model_name", "")

        if not model_id:
            continue

        # Use model_id as the key for active models
        entry = get_or_create_entry(model_id, model_id, model_name)

        # Get active regions
        active_regions = model.get("active_regions", [])
        launch_dates_by_region = model.get("launch_dates_by_region", {})
        eol_date = model.get("eol_date")
        eol_regions = model.get("eol_regions", [])
        eol_all_regions = model.get("eol_all_regions", False)

        # Add regional status for each active region
        for region in active_regions:
            if region not in entry["regional_status"]:
                entry["regional_status"][region] = {
                    "status": "ACTIVE",
                }

            region_entry = entry["regional_status"][region]
            region_entry["status"] = "ACTIVE"

            # Add launch date if available
            if region in launch_dates_by_region:
                region_entry["launch_date"] = launch_dates_by_region[region]

            # Add EOL date if applicable to this region
            if eol_date:
                if eol_all_regions or not eol_regions or region in eol_regions:
                    region_entry["eol_date"] = eol_date

            # Track in summary
            if region not in entry["status_summary"]["ACTIVE"]:
                entry["status_summary"]["ACTIVE"].append(region)

    # Process Legacy models
    for model in legacy_models:
        model_name = model.get("model_name", "")

        if not model_name:
            continue

        # Use model_name as the key for legacy models (they don't have their own model_id)
        entry = get_or_create_entry(model_name, None, model_name)

        # Update replacement info
        entry["recommended_replacement"] = model.get("recommended_replacement")
        entry["recommended_model_id"] = model.get("recommended_model_id")

        # Get legacy-specific data
        legacy_date = model.get("legacy_date")
        legacy_regions = model.get("legacy_regions", [])
        legacy_all_regions = model.get("legacy_all_regions", False)

        extended_access_date = model.get("extended_access_date")
        extended_access_regions = model.get("extended_access_regions", [])
        extended_access_all_regions = model.get("extended_access_all_regions", False)

        eol_date = model.get("eol_date")
        eol_regions = model.get("eol_regions", [])
        eol_all_regions = model.get("eol_all_regions", False)

        # Determine which regions are legacy
        # If specific regions are mentioned, use those
        # Otherwise, if "all regions" is indicated, we'll mark it but can't enumerate
        if legacy_regions:
            regions_to_process = legacy_regions
        elif legacy_all_regions:
            # "All regions" - we can't enumerate, but we can note it
            # Use existing regions from the entry if any, or create a placeholder
            regions_to_process = list(entry["regional_status"].keys()) or ["all"]
        else:
            # No specific regions - apply to any existing regions or create placeholder
            regions_to_process = list(entry["regional_status"].keys()) or ["all"]

        for region in regions_to_process:
            if region not in entry["regional_status"]:
                entry["regional_status"][region] = {}

            region_entry = entry["regional_status"][region]

            # Set status to LEGACY (overrides ACTIVE if model appears in both tables)
            region_entry["status"] = "LEGACY"

            # Add legacy date
            if legacy_date:
                region_entry["legacy_date"] = legacy_date

            # Add extended access date if applicable
            # If extended_access_regions is empty but legacy_regions is not,
            # only apply to the legacy_regions (same row's regions)
            if extended_access_date:
                should_apply = False
                if extended_access_all_regions:
                    should_apply = True
                elif extended_access_regions:
                    # Specific regions listed in extended access date
                    should_apply = region in extended_access_regions
                elif legacy_regions:
                    # No specific extended regions, but we have legacy regions
                    # Only apply to the same row's regions
                    should_apply = region in legacy_regions
                else:
                    # No specific regions anywhere - apply to all
                    should_apply = True

                if should_apply:
                    region_entry["extended_access_date"] = extended_access_date

            # Add EOL date if applicable
            # Same logic as extended_access_date
            if eol_date:
                should_apply = False
                if eol_all_regions:
                    should_apply = True
                elif eol_regions:
                    # Specific regions listed in EOL date
                    should_apply = region in eol_regions
                elif legacy_regions:
                    # No specific EOL regions, but we have legacy regions
                    # Only apply to the same row's regions
                    should_apply = region in legacy_regions
                else:
                    # No specific regions anywhere - apply to all
                    should_apply = True

                if should_apply:
                    region_entry["eol_date"] = eol_date

            # Update summary - move from ACTIVE to LEGACY if needed
            if region in entry["status_summary"]["ACTIVE"]:
                entry["status_summary"]["ACTIVE"].remove(region)
            if region not in entry["status_summary"]["LEGACY"]:
                entry["status_summary"]["LEGACY"].append(region)

    # Process EOL models
    for model in eol_models:
        model_name = model.get("model_name", "")

        if not model_name:
            continue

        # Use model_name as the key
        entry = get_or_create_entry(model_name, None, model_name)

        # Update replacement info
        entry["recommended_replacement"] = model.get("recommended_replacement")
        entry["recommended_model_id"] = model.get("recommended_model_id")

        # Get EOL-specific data
        legacy_date = model.get("legacy_date")
        legacy_regions = model.get("legacy_regions", [])
        legacy_all_regions = model.get("legacy_all_regions", False)

        eol_date = model.get("eol_date")
        eol_regions = model.get("eol_regions", [])
        eol_all_regions = model.get("eol_all_regions", False)

        # Determine which regions are EOL
        if eol_regions:
            regions_to_process = eol_regions
        elif eol_all_regions:
            regions_to_process = list(entry["regional_status"].keys()) or ["all"]
        else:
            regions_to_process = list(entry["regional_status"].keys()) or ["all"]

        for region in regions_to_process:
            if region not in entry["regional_status"]:
                entry["regional_status"][region] = {}

            region_entry = entry["regional_status"][region]

            # Set status to EOL (overrides ACTIVE and LEGACY)
            region_entry["status"] = "EOL"

            # Add legacy date
            if legacy_date:
                region_entry["legacy_date"] = legacy_date

            # Add EOL date
            if eol_date:
                region_entry["eol_date"] = eol_date

            # Update summary - move from other statuses to EOL
            if region in entry["status_summary"]["ACTIVE"]:
                entry["status_summary"]["ACTIVE"].remove(region)
            if region in entry["status_summary"]["LEGACY"]:
                entry["status_summary"]["LEGACY"].remove(region)
            if region not in entry["status_summary"]["EOL"]:
                entry["status_summary"]["EOL"].append(region)

    # Now handle models that appear in BOTH Active and Legacy tables
    # (like Claude 3.5 Sonnet v1 which is ACTIVE in some regions, LEGACY in others)
    # We need to cross-reference by model_name

    # Build a lookup from model_name to model_id from active models
    name_to_id = {}
    for model in active_models:
        model_name = model.get("model_name", "")
        model_id = model.get("model_id")
        if model_name and model_id:
            name_to_id[model_name] = model_id

    # Merge entries where a legacy model_name matches an active model
    # Use a set to track which entries have been merged (avoid duplicates from
    # continuation rows that create multiple entries for the same model_name)
    entries_to_merge = set()
    for model in legacy_models + eol_models:
        model_name = model.get("model_name", "")
        if model_name in name_to_id:
            model_id = name_to_id[model_name]
            if model_name in regional_lifecycle and model_id in regional_lifecycle:
                entries_to_merge.add((model_name, model_id))

    for model_name, model_id in entries_to_merge:
        name_entry = regional_lifecycle[model_name]
        id_entry = regional_lifecycle[model_id]

        # Merge regional_status from name_entry into id_entry
        for region, status_data in name_entry["regional_status"].items():
            if region not in id_entry["regional_status"]:
                id_entry["regional_status"][region] = status_data
            else:
                # Merge the data, preferring legacy/eol status over active
                existing = id_entry["regional_status"][region]
                if status_data.get("status") in ["LEGACY", "EOL"]:
                    existing["status"] = status_data["status"]
                # Add any dates from the legacy/eol entry
                for key in ["legacy_date", "extended_access_date", "eol_date"]:
                    if key in status_data and status_data[key]:
                        existing[key] = status_data[key]

        # Update status_summary
        for status in ["ACTIVE", "LEGACY", "EOL"]:
            for region in name_entry["status_summary"][status]:
                if region not in id_entry["status_summary"][status]:
                    # Remove from other statuses first
                    for other_status in ["ACTIVE", "LEGACY", "EOL"]:
                        if (
                            other_status != status
                            and region in id_entry["status_summary"][other_status]
                        ):
                            id_entry["status_summary"][other_status].remove(region)
                    id_entry["status_summary"][status].append(region)

        # Copy replacement info
        if name_entry.get("recommended_replacement"):
            id_entry["recommended_replacement"] = name_entry["recommended_replacement"]
        if name_entry.get("recommended_model_id"):
            id_entry["recommended_model_id"] = name_entry["recommended_model_id"]

        # Update model_id in the id_entry if not set
        if not id_entry.get("model_id"):
            id_entry["model_id"] = model_id

        # Remove the duplicate name entry
        del regional_lifecycle[model_name]

    # Sort region lists in status_summary for consistency
    for entry in regional_lifecycle.values():
        for status in entry["status_summary"]:
            entry["status_summary"][status].sort()

    return regional_lifecycle


def scrape_lifecycle_data() -> dict:
    """Scrape and parse all lifecycle tables from the AWS documentation.

    Returns:
        Dictionary containing:
        - models: List of all model records
        - models_by_id: Lookup dictionary keyed by model_id (for Active models)
        - models_by_name: Lookup dictionary keyed by model_name (for Legacy/EOL models)
        - regional_lifecycle: Per-model regional status structure
        - status_counts: Count of models per status
        - total_models: Total number of models found
    """
    html_content = fetch_lifecycle_page()
    soup = BeautifulSoup(html_content, "lxml")

    # Find all tables with class 'table-contents' inside 'table-container'
    tables = soup.select(".table-container .table-contents table")

    all_models = []
    status_counts = {"active": 0, "legacy": 0, "eol": 0}

    # The page has 3 sections: Active, Legacy, EOL
    # Tables appear in order on the page
    status_mapping = ["active", "legacy", "eol"]

    if len(tables) < 3:
        logger.warning(
            "Unexpected table count - page structure may have changed",
            extra={"expected": 3, "found": len(tables)},
        )

    # Store models by status for regional_lifecycle building
    active_models = []
    legacy_models = []
    eol_models = []

    for idx, table in enumerate(tables[:3]):  # Only process first 3 tables
        status = status_mapping[idx] if idx < len(status_mapping) else "unknown"
        models = parse_lifecycle_table(table, status)
        all_models.extend(models)
        status_counts[status] = len(models)
        logger.info(
            "Parsed lifecycle models", extra={"status": status, "count": len(models)}
        )

        # Store by status
        if status == "active":
            active_models = models
        elif status == "legacy":
            legacy_models = models
        elif status == "eol":
            eol_models = models

    # Create lookup by model_id for Active models (they have actual model_id)
    models_by_id = {}
    # Create lookup by model_name for Legacy/EOL models (they don't have their own model_id)
    models_by_name = {}

    for model in all_models:
        lifecycle_status = model.get("lifecycle_status", "")

        if lifecycle_status == "active":
            # Active models have their own model_id
            model_id = model.get("model_id")
            if model_id:
                models_by_id[model_id] = model
        else:
            # Legacy and EOL models are identified by model_name
            # (their "recommended_model_id" field is the REPLACEMENT, not themselves)
            model_name = model.get("model_name")
            if model_name:
                models_by_name[model_name] = model

    # Build regional lifecycle structure
    regional_lifecycle = build_regional_lifecycle(
        active_models, legacy_models, eol_models
    )

    return {
        "models": all_models,
        "models_by_id": models_by_id,
        "models_by_name": models_by_name,
        "regional_lifecycle": regional_lifecycle,
        "status_counts": status_counts,
        "total_models": len(all_models),
    }


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    """
    Lambda handler for lifecycle data collection.

    Input:
        {
            "s3Bucket": "bucket-name",
            "s3Key": "executions/{id}/lifecycle/lifecycle.json",
            "dryRun": false  // Optional: skip S3 write for testing
        }

    Output:
        {
            "status": "SUCCESS",
            "s3Key": "executions/{id}/lifecycle/lifecycle.json",
            "recordCount": 150,
            "statusCounts": {"active": 100, "legacy": 30, "eol": 20},
            "regionalLifecycleCount": 150,
            "fromCache": true,
            "durationMs": 2500
        }
    """
    start_time = time.time()

    # Extract parameters
    s3_bucket = event.get("s3Bucket", DATA_BUCKET)
    s3_key = event.get("s3Key", "test/lifecycle.json")
    dry_run = event.get("dryRun", False)

    logger.info(
        "Starting lifecycle collection", extra={"bucket": s3_bucket, "dry_run": dry_run}
    )

    try:
        s3_client = get_s3_client()
        source_url = get_lifecycle_url()

        # Scrape lifecycle data with caching
        lifecycle_data, from_cache = get_cached_or_fetch(
            s3_client=s3_client,
            bucket=s3_bucket,
            cache_key=LIFECYCLE_CACHE_KEY,
            fetch_fn=scrape_lifecycle_data,
            source_url=source_url,
            ttl_hours=DEFAULT_CACHE_TTL_HOURS,
        )

        # Structure the output
        output_data = {
            "metadata": {
                "source_url": source_url,
                "record_count": lifecycle_data["total_models"],
                "status_counts": lifecycle_data["status_counts"],
                "regional_lifecycle_count": len(lifecycle_data["regional_lifecycle"]),
                "from_cache": from_cache,
                "collection_timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            },
            "models": lifecycle_data["models"],
            "models_by_id": lifecycle_data["models_by_id"],
            "models_by_name": lifecycle_data["models_by_name"],
            "regional_lifecycle": lifecycle_data["regional_lifecycle"],
        }

        # Write to S3 (skip in dry run mode)
        if not dry_run and s3_bucket:
            write_to_s3(s3_client, s3_bucket, s3_key, output_data)
        else:
            logger.info(
                "Dry run mode - skipping S3 write",
                extra={"bucket": s3_bucket, "key": s3_key},
            )

        duration_ms = int((time.time() - start_time) * 1000)

        # Emit metrics
        metrics.add_metric(
            name="LifecycleModelsCollected",
            unit=MetricUnit.Count,
            value=lifecycle_data["total_models"],
        )

        logger.info(
            "Lifecycle collection complete",
            extra={
                "record_count": lifecycle_data["total_models"],
                "status_counts": lifecycle_data["status_counts"],
                "from_cache": from_cache,
                "duration_ms": duration_ms,
            },
        )

        return {
            "status": "SUCCESS",
            "s3Key": s3_key,
            "recordCount": lifecycle_data["total_models"],
            "statusCounts": lifecycle_data["status_counts"],
            "regionalLifecycleCount": len(lifecycle_data["regional_lifecycle"]),
            "fromCache": from_cache,
            "durationMs": duration_ms,
            "dryRun": dry_run,
        }

    except requests.RequestException as e:
        logger.exception("Failed to fetch lifecycle page", extra={"error": str(e)})
        return {
            "status": "FAILED",
            "errorType": "RequestError",
            "errorMessage": str(e),
            "retryable": True,
        }
    except Exception as e:
        logger.exception("Failed to collect lifecycle data", extra={"error": str(e)})
        return {
            "status": "FAILED",
            "errorType": type(e).__name__,
            "errorMessage": str(e),
            "retryable": False,
        }
