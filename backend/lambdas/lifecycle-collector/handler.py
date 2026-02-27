"""
Lifecycle Collector Lambda

Scrapes model lifecycle data from AWS Bedrock documentation.
Source: https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html
"""

import logging
import os
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

from shared import (
    RETRY_CONFIG,
    get_s3_client,
    write_to_s3,
    validate_required_params,
    ValidationError,
)

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# Configuration
DATA_BUCKET = os.environ.get("DATA_BUCKET")
LIFECYCLE_URL = (
    "https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html"
)
REQUEST_TIMEOUT = 30  # seconds


def fetch_lifecycle_page() -> str:
    """Fetch the HTML content from the AWS lifecycle documentation page."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BedrockProfiler/1.0)",
        "Accept": "text/html,application/xhtml+xml",
    }

    response = requests.get(LIFECYCLE_URL, headers=headers, timeout=REQUEST_TIMEOUT)
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

    Columns: Provider, Model name, Model ID, Regions supported, Launch date, EOL date, Input modalities, Output modalities
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

        model_data = {
            "provider": get_cell_text(0),
            "model_name": get_cell_text(1),
            "model_id": get_cell_text(2),
            "regions": get_cell_text(3),
            "launch_date": get_cell_text(4) if len(cells) > 4 else None,
            "eol_date": get_cell_text(5) if len(cells) > 5 else None,
            "input_modalities": get_cell_text(6) if len(cells) > 6 else None,
            "output_modalities": get_cell_text(7) if len(cells) > 7 else None,
            "lifecycle_status": "active",
        }

        if model_data["model_id"]:
            models.append(model_data)

    return models


def parse_legacy_table(table) -> list[dict]:
    """Parse the Legacy models table.

    Columns: Model version, Legacy date, Public extended access date, EOL date, Recommended model version replacement, Recommended model ID
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

        # Legacy table has different structure
        model_data = {
            "model_name": get_cell_text(0),  # Model version
            "legacy_date": get_cell_text(1),
            "extended_access_date": get_cell_text(2)
            if len(cells) > 5
            else None,  # Only in legacy table
            "eol_date": get_cell_text(3) if len(cells) > 5 else get_cell_text(2),
            "recommended_replacement": get_cell_text(4)
            if len(cells) > 5
            else get_cell_text(3),
            "model_id": get_cell_text(5)
            if len(cells) > 5
            else get_cell_text(4),  # Recommended model ID
            "lifecycle_status": "legacy",
            # Fields not available in legacy table
            "provider": None,
            "regions": None,
            "launch_date": None,
            "input_modalities": None,
            "output_modalities": None,
        }

        # Use model_name as identifier if model_id is empty
        if model_data["model_id"] or model_data["model_name"]:
            models.append(model_data)

    return models


def parse_eol_table(table) -> list[dict]:
    """Parse the EOL (End-of-Life) models table.

    Columns: Model version, Legacy date, EOL date, Recommended model version replacement, Recommended model ID
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

        # EOL table structure
        model_data = {
            "model_name": get_cell_text(0),  # Model version
            "legacy_date": get_cell_text(1),
            "eol_date": get_cell_text(2),
            "recommended_replacement": get_cell_text(3),
            "model_id": get_cell_text(4),  # Recommended model ID
            "lifecycle_status": "eol",
            # Fields not available in EOL table
            "provider": None,
            "regions": None,
            "launch_date": None,
            "input_modalities": None,
            "output_modalities": None,
            "extended_access_date": None,
        }

        # Use model_name as identifier if model_id is empty
        if model_data["model_id"] or model_data["model_name"]:
            models.append(model_data)

    return models


def parse_lifecycle_table(table, status: str) -> list[dict]:
    """Parse a lifecycle table and extract model information.

    Dispatches to the appropriate parser based on table status/structure.

    Active table columns: Provider, Model name, Model ID, Regions, Launch date, EOL date, Input modalities, Output modalities
    Legacy table columns: Model version, Legacy date, Public extended access date, EOL date, Recommended model version replacement, Recommended model ID
    EOL table columns: Model version, Legacy date, EOL date, Recommended model version replacement, Recommended model ID
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


def scrape_lifecycle_data() -> dict:
    """Scrape and parse all lifecycle tables from the AWS documentation.

    Returns:
        Dictionary containing:
        - models: List of all model records
        - models_by_id: Lookup dictionary keyed by model_id
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
            f"Expected 3 tables but found {len(tables)}. Page structure may have changed."
        )

    for idx, table in enumerate(tables[:3]):  # Only process first 3 tables
        status = status_mapping[idx] if idx < len(status_mapping) else "unknown"
        models = parse_lifecycle_table(table, status)
        all_models.extend(models)
        status_counts[status] = len(models)
        logger.info(f"Parsed {len(models)} {status} models")

    # Create lookup by model_id for easy merging
    models_by_id = {}
    for model in all_models:
        model_id = model["model_id"]
        if model_id:
            models_by_id[model_id] = model

    return {
        "models": all_models,
        "models_by_id": models_by_id,
        "status_counts": status_counts,
        "total_models": len(all_models),
    }


def lambda_handler(event: dict, context: Any) -> dict:
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
            "durationMs": 2500
        }
    """
    start_time = time.time()

    # Extract parameters
    s3_bucket = event.get("s3Bucket", DATA_BUCKET)
    s3_key = event.get("s3Key", "test/lifecycle.json")
    dry_run = event.get("dryRun", False)

    logger.info(f"Starting lifecycle collection: bucket={s3_bucket}, dryRun={dry_run}")

    try:
        # Scrape lifecycle data
        lifecycle_data = scrape_lifecycle_data()

        # Structure the output
        output_data = {
            "metadata": {
                "source_url": LIFECYCLE_URL,
                "record_count": lifecycle_data["total_models"],
                "status_counts": lifecycle_data["status_counts"],
                "collection_timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            },
            "models": lifecycle_data["models"],
            "models_by_id": lifecycle_data["models_by_id"],
        }

        # Write to S3 (skip in dry run mode)
        if not dry_run and s3_bucket:
            s3_client = get_s3_client()
            write_to_s3(s3_client, s3_bucket, s3_key, output_data)
        else:
            logger.info(
                f"Dry run mode - skipping S3 write. Would write to s3://{s3_bucket}/{s3_key}"
            )

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            "status": "SUCCESS",
            "s3Key": s3_key,
            "recordCount": lifecycle_data["total_models"],
            "statusCounts": lifecycle_data["status_counts"],
            "durationMs": duration_ms,
            "dryRun": dry_run,
        }

    except requests.RequestException as e:
        logger.error(f"Failed to fetch lifecycle page: {e}", exc_info=True)
        return {
            "status": "FAILED",
            "errorType": "RequestError",
            "errorMessage": str(e),
            "retryable": True,
        }
    except Exception as e:
        logger.error(f"Failed to collect lifecycle data: {e}", exc_info=True)
        return {
            "status": "FAILED",
            "errorType": type(e).__name__,
            "errorMessage": str(e),
            "retryable": False,
        }
