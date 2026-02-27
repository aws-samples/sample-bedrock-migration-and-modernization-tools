#!/usr/bin/env python3
"""
Local test runner for lifecycle-collector Lambda.

Usage:
    python test_lifecycle_collector_local.py [--output-dir DIR]

This script tests the lifecycle scraping logic locally without S3.
"""

import argparse
import json
import logging
import os
import sys
import time

# Add lambdas directory to path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "lambdas", "lifecycle-collector")
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test lifecycle collector locally")
    parser.add_argument(
        "--output-dir", default="./output", help="Output directory for JSON files"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Import the scraper function (after path setup)
    try:
        from handler import scrape_lifecycle_data, LIFECYCLE_URL
    except ImportError as e:
        # If shared module import fails, we need to mock it
        logger.warning(f"Import error (expected if shared module not available): {e}")
        logger.info("Attempting direct import of scraping functions...")

        # Direct import of just the scraping logic
        import requests
        from bs4 import BeautifulSoup

        LIFECYCLE_URL = (
            "https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html"
        )
        REQUEST_TIMEOUT = 30

        def fetch_lifecycle_page() -> str:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; BedrockProfiler/1.0)",
                "Accept": "text/html,application/xhtml+xml",
            }
            response = requests.get(
                LIFECYCLE_URL, headers=headers, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.text

        def get_table_headers(table) -> list:
            """Extract header names from a table."""
            header_row = table.find("tr")
            if header_row:
                return [
                    th.get_text(strip=True).lower()
                    for th in header_row.find_all(["th", "td"])
                ]
            return []

        def parse_active_table(table) -> list:
            """Parse the Active models table."""
            models = []
            all_rows = table.find_all("tr")

            for row in all_rows[1:]:  # Skip header
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

        def parse_legacy_table(table) -> list:
            """Parse the Legacy models table."""
            models = []
            all_rows = table.find_all("tr")

            for row in all_rows[1:]:  # Skip header
                cells = row.find_all(["td", "th"])
                if len(cells) < 4:
                    continue

                def get_cell_text(idx: int) -> str:
                    if idx < len(cells):
                        return cells[idx].get_text(strip=True)
                    return ""

                model_data = {
                    "model_name": get_cell_text(0),
                    "legacy_date": get_cell_text(1),
                    "extended_access_date": get_cell_text(2)
                    if len(cells) > 5
                    else None,
                    "eol_date": get_cell_text(3)
                    if len(cells) > 5
                    else get_cell_text(2),
                    "recommended_replacement": get_cell_text(4)
                    if len(cells) > 5
                    else get_cell_text(3),
                    "model_id": get_cell_text(5)
                    if len(cells) > 5
                    else get_cell_text(4),
                    "lifecycle_status": "legacy",
                    "provider": None,
                    "regions": None,
                    "launch_date": None,
                    "input_modalities": None,
                    "output_modalities": None,
                }

                if model_data["model_id"] or model_data["model_name"]:
                    models.append(model_data)

            return models

        def parse_eol_table(table) -> list:
            """Parse the EOL models table."""
            models = []
            all_rows = table.find_all("tr")

            for row in all_rows[1:]:  # Skip header
                cells = row.find_all(["td", "th"])
                if len(cells) < 4:
                    continue

                def get_cell_text(idx: int) -> str:
                    if idx < len(cells):
                        return cells[idx].get_text(strip=True)
                    return ""

                model_data = {
                    "model_name": get_cell_text(0),
                    "legacy_date": get_cell_text(1),
                    "eol_date": get_cell_text(2),
                    "recommended_replacement": get_cell_text(3),
                    "model_id": get_cell_text(4),
                    "lifecycle_status": "eol",
                    "provider": None,
                    "regions": None,
                    "launch_date": None,
                    "input_modalities": None,
                    "output_modalities": None,
                    "extended_access_date": None,
                }

                if model_data["model_id"] or model_data["model_name"]:
                    models.append(model_data)

            return models

        def parse_lifecycle_table(table, status: str) -> list:
            """Parse a lifecycle table based on its structure."""
            headers = get_table_headers(table)

            if "provider" in headers or "model id" in headers:
                return parse_active_table(table)
            elif "public extended access date" in headers:
                return parse_legacy_table(table)
            elif "recommended model id" in headers:
                return parse_eol_table(table)
            else:
                if status == "active":
                    return parse_active_table(table)
                elif status == "legacy":
                    return parse_legacy_table(table)
                elif status == "eol":
                    return parse_eol_table(table)
                else:
                    logger.warning(f"Unknown table structure for status {status}")
                    return []

        def scrape_lifecycle_data() -> dict:
            html_content = fetch_lifecycle_page()
            soup = BeautifulSoup(html_content, "lxml")
            tables = soup.select(".table-container .table-contents table")

            all_models = []
            status_counts = {"active": 0, "legacy": 0, "eol": 0}
            status_mapping = ["active", "legacy", "eol"]

            for idx, table in enumerate(tables[:3]):
                status = status_mapping[idx] if idx < len(status_mapping) else "unknown"
                models = parse_lifecycle_table(table, status)
                all_models.extend(models)
                status_counts[status] = len(models)
                logger.info(f"Parsed {len(models)} {status} models")

            models_by_id = {}
            for model in all_models:
                model_id = model.get("model_id")
                if model_id:
                    models_by_id[model_id] = model

            return {
                "models": all_models,
                "models_by_id": models_by_id,
                "status_counts": status_counts,
                "total_models": len(all_models),
            }

    # Run the scraper
    start_time = time.time()

    try:
        logger.info(f"Fetching lifecycle data from: {LIFECYCLE_URL}")
        data = scrape_lifecycle_data()

        duration_ms = int((time.time() - start_time) * 1000)

        # Structure output
        output_data = {
            "metadata": {
                "source_url": LIFECYCLE_URL,
                "record_count": data["total_models"],
                "status_counts": data["status_counts"],
                "collection_timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "duration_ms": duration_ms,
            },
            "models": data["models"],
            "models_by_id": data["models_by_id"],
        }

        # Write to file
        output_file = os.path.join(args.output_dir, "lifecycle.json")
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"Output written to: {output_file}")
        logger.info(f"Total models: {data['total_models']}")
        logger.info(f"Status counts: {data['status_counts']}")
        logger.info(f"Duration: {duration_ms}ms")

        # Print sample models
        print("\n" + "=" * 60)
        print("SAMPLE MODELS BY STATUS:")
        print("=" * 60)

        for status in ["active", "legacy", "eol"]:
            status_models = [
                m for m in data["models"] if m["lifecycle_status"] == status
            ]
            print(f"\n{status.upper()} ({len(status_models)} models):")
            for model in status_models[:3]:
                model_id = model.get("model_id", "N/A")
                model_name = model.get("model_name", "Unknown")
                provider = model.get("provider", "N/A")
                print(f"  - {model_id}: {model_name} (Provider: {provider})")

        return {
            "status": "SUCCESS",
            "recordCount": data["total_models"],
            "statusCounts": data["status_counts"],
            "durationMs": duration_ms,
            "outputFile": output_file,
        }

    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        return {
            "status": "FAILED",
            "errorType": type(e).__name__,
            "errorMessage": str(e),
        }


if __name__ == "__main__":
    result = main()
    print("\n" + "=" * 60)
    print("RESULT:")
    print(json.dumps(result, indent=2))
