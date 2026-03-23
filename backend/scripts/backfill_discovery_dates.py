"""
One-time backfill script: scan all historical executions to find the earliest
first_discovered_at per model and patch latest/bedrock_models.json.

Usage:
    python backfill_discovery_dates.py --bucket bedrock-profiler-data-390445053194-prod [--dry-run]
"""

import argparse
import json
import boto3
from datetime import datetime, timezone


def list_execution_model_keys(s3_client, bucket):
    """List all executions/*/final/bedrock_models.json keys, sorted oldest first."""
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix="executions/", Delimiter="/"):
        for prefix in page.get("CommonPrefixes", []):
            exec_prefix = prefix["Prefix"]
            key = f"{exec_prefix}final/bedrock_models.json"
            keys.append(key)
    return sorted(keys)


def load_json(s3_client, bucket, key):
    """Load a JSON file from S3, return None if not found."""
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        print(f"  Warning: could not read {key}: {e}")
        return None


def extract_discovery_dates(data):
    """Extract model_id -> {first_discovered_at, first_discovered_in_region} from a bedrock_models.json."""
    result = {}
    for provider_data in data.get("providers", {}).values():
        for model_id, model in provider_data.get("models", {}).items():
            metadata = model.get("collection_metadata", {})
            discovered_at = metadata.get("first_discovered_at")
            if discovered_at:
                result[model_id] = {
                    "first_discovered_at": discovered_at,
                    "first_discovered_in_region": metadata.get("first_discovered_in_region", "unknown"),
                }
    return result


def parse_timestamp(ts):
    """Parse a timestamp string in various formats, return UTC-naive datetime for comparison."""
    if not ts:
        return None
    # Try ISO format first (e.g. "2026-03-04T13:10:03.000000+00:00")
    try:
        dt = datetime.fromisoformat(ts.rstrip("Z"))
        # Strip timezone info for consistent comparison (all timestamps are UTC)
        return dt.replace(tzinfo=None)
    except (ValueError, AttributeError):
        pass
    # Handle "2026-03-04 13:09:18 UTC" format from model-extractor
    try:
        return datetime.strptime(ts.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Backfill first_discovered_at from historical executions")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    args = parser.parse_args()

    s3_client = boto3.client("s3", region_name=args.region)

    # Step 1: Find all execution keys
    print(f"Scanning executions in s3://{args.bucket}/executions/...")
    exec_keys = list_execution_model_keys(s3_client, args.bucket)
    print(f"Found {len(exec_keys)} execution outputs")

    # Step 2: Scan all executions to find earliest discovery per model
    earliest = {}  # model_id -> {first_discovered_at, first_discovered_in_region}

    for i, key in enumerate(exec_keys):
        print(f"  [{i+1}/{len(exec_keys)}] Reading {key}...")
        data = load_json(s3_client, args.bucket, key)
        if not data:
            continue

        dates = extract_discovery_dates(data)
        for model_id, info in dates.items():
            ts = parse_timestamp(info["first_discovered_at"])
            if ts is None:
                continue
            existing = earliest.get(model_id)
            if existing is None:
                earliest[model_id] = info
            else:
                existing_ts = parse_timestamp(existing["first_discovered_at"])
                if existing_ts and ts < existing_ts:
                    earliest[model_id] = info

    print(f"\nFound earliest discovery dates for {len(earliest)} models")

    # Step 3: Load current latest/bedrock_models.json
    latest_key = "latest/bedrock_models.json"
    print(f"\nLoading {latest_key}...")
    latest = load_json(s3_client, args.bucket, latest_key)
    if not latest:
        print("ERROR: Could not load latest/bedrock_models.json")
        return

    # Step 4: Patch discovery dates
    updated_count = 0
    new_count = 0
    for provider_name, provider_data in latest.get("providers", {}).items():
        for model_id, model in provider_data.get("models", {}).items():
            if model_id not in earliest:
                continue
            metadata = model.setdefault("collection_metadata", {})
            current_ts = parse_timestamp(metadata.get("first_discovered_at", ""))
            earliest_ts = parse_timestamp(earliest[model_id]["first_discovered_at"])

            if earliest_ts is None:
                continue

            if current_ts is None or earliest_ts < current_ts:
                old_val = metadata.get("first_discovered_at", "(none)")
                metadata["first_discovered_at"] = earliest[model_id]["first_discovered_at"]
                metadata["first_discovered_in_region"] = earliest[model_id]["first_discovered_in_region"]
                updated_count += 1
                print(f"  {model_id}: {old_val} -> {earliest[model_id]['first_discovered_at']}")

    print(f"\nUpdated {updated_count} models with earlier discovery dates")

    # Step 5: Write back
    if args.dry_run:
        print("\n[DRY RUN] Would write updated latest/bedrock_models.json")
    else:
        print(f"\nWriting updated {latest_key}...")
        s3_client.put_object(
            Bucket=args.bucket,
            Key=latest_key,
            Body=json.dumps(latest, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        print("Done!")


if __name__ == "__main__":
    main()
