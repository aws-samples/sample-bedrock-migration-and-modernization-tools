#!/usr/bin/env python3
"""
On-Demand Availability Verification Script

Cross-checks on-demand regional availability by comparing:
  Phase 1: Unfiltered vs ON_DEMAND-filtered ListFoundationModels across all regions
  Phase 2: Pricing data regions vs ON_DEMAND API results
  Phase 3: Converse API spot-checks for actual invocability

Usage:
    python verify_ondemand_availability.py --all
    python verify_ondemand_availability.py --phase 1
    python verify_ondemand_availability.py --phase 2
    python verify_ondemand_availability.py --phase 3
    python verify_ondemand_availability.py --phase 1 --phase 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RETRY_CONFIG = Config(
    retries={"max_attempts": 3, "mode": "adaptive"},
    connect_timeout=5,
    read_timeout=30,
)

CONVERSE_SPOT_CHECK_MODELS = [
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "amazon.titan-text-express-v1",
    "mistral.mistral-7b-instruct-v0:2",
    "amazon.nova-lite-v1:0",
]

# Embedding models that don't support Converse
EMBEDDING_KEYWORDS = ["embed", "embedding", "rerank"]

MAX_WORKERS_PER_PHASE = 10


def _log(msg: str) -> None:
    """Print progress to stderr."""
    print(msg, file=sys.stderr, flush=True)


# ===================================================================
# Region discovery (shared by all phases)
# ===================================================================


def discover_bedrock_regions() -> list[str]:
    """
    Dynamically discover AWS regions where the Bedrock control-plane is
    reachable.  Uses EC2 DescribeRegions then probes each with a
    lightweight ListFoundationModels(maxResults=1) call.
    """
    _log("[discovery] Fetching enabled regions via EC2 DescribeRegions ...")
    ec2 = boto3.client("ec2", region_name="us-east-1", config=RETRY_CONFIG)
    try:
        resp = ec2.describe_regions(
            AllRegions=False,
            Filters=[
                {"Name": "opt-in-status", "Values": ["opt-in-not-required", "opted-in"]}
            ],
        )
        all_regions = sorted(r["RegionName"] for r in resp.get("Regions", []))
    except ClientError:
        all_regions = [
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "eu-central-1",
            "eu-north-1",
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "sa-east-1",
        ]

    _log(f"[discovery] Probing {len(all_regions)} regions for Bedrock endpoint ...")

    def _probe(region: str) -> tuple[str, bool]:
        try:
            client = boto3.client("bedrock", region_name=region, config=RETRY_CONFIG)
            client.list_foundation_models()
            return region, True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code == "AccessDeniedException":
                return region, True  # endpoint exists; IAM blocked
            return region, False
        except (EndpointConnectionError, Exception):
            return region, False

    bedrock_regions: list[str] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PER_PHASE) as pool:
        futures = {pool.submit(_probe, r): r for r in all_regions}
        for fut in as_completed(futures):
            region, ok = fut.result()
            if ok:
                bedrock_regions.append(region)

    bedrock_regions.sort()
    _log(f"[discovery] Found {len(bedrock_regions)} Bedrock regions")
    return bedrock_regions


# ===================================================================
# Phase 1: ListFoundationModels comparison
# ===================================================================


def _query_region_both(region: str) -> dict:
    """
    For a single region call ListFoundationModels twice:
      1. unfiltered
      2. filtered by ON_DEMAND
    Return a dict with counts and model sets.
    """
    result: dict[str, Any] = {
        "region": region,
        "error": None,
        "unfiltered_ids": set(),
        "ondemand_ids": set(),
        "unfiltered_details": {},  # model_id -> inferenceTypesSupported
        "ondemand_details": {},
    }
    try:
        client = boto3.client("bedrock", region_name=region, config=RETRY_CONFIG)

        # 1. Unfiltered
        resp_all = client.list_foundation_models()
        for m in resp_all.get("modelSummaries", []):
            mid = m.get("modelId", "")
            if mid:
                result["unfiltered_ids"].add(mid)
                result["unfiltered_details"][mid] = m.get("inferenceTypesSupported", [])

        # 2. ON_DEMAND filtered
        resp_od = client.list_foundation_models(byInferenceType="ON_DEMAND")
        for m in resp_od.get("modelSummaries", []):
            mid = m.get("modelId", "")
            if mid:
                result["ondemand_ids"].add(mid)
                result["ondemand_details"][mid] = m.get("inferenceTypesSupported", [])

    except Exception as exc:
        result["error"] = str(exc)

    return result


def run_phase1(regions: list[str]) -> dict:
    """
    Phase 1: compare unfiltered vs ON_DEMAND-filtered ListFoundationModels
    for every Bedrock region.

    Returns:
        {
            "region_results": { region: {total, ondemand, delta_count, delta_models} },
            "global_delta": { model_id: {regions, inference_types} },
            "unfiltered_map": { model_id: set(regions) },
            "ondemand_map":  { model_id: set(regions) },
        }
    """
    _log(f"\n{'=' * 70}")
    _log("PHASE 1: ListFoundationModels — unfiltered vs ON_DEMAND")
    _log(f"{'=' * 70}")
    _log(f"Querying {len(regions)} regions (2 calls each) ...")

    region_results: dict[str, dict] = {}
    global_delta: dict[str, dict] = defaultdict(
        lambda: {"regions": set(), "inference_types": set()}
    )
    unfiltered_map: dict[str, set] = defaultdict(set)
    ondemand_map: dict[str, set] = defaultdict(set)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PER_PHASE) as pool:
        futures = {pool.submit(_query_region_both, r): r for r in regions}
        done_count = 0
        for fut in as_completed(futures):
            done_count += 1
            data = fut.result()
            region = data["region"]

            if data["error"]:
                _log(
                    f"  [{done_count}/{len(regions)}] {region}: ERROR — {data['error']}"
                )
                region_results[region] = {
                    "total": 0,
                    "ondemand": 0,
                    "delta_count": 0,
                    "delta_models": {},
                    "error": data["error"],
                }
                continue

            delta_ids = data["unfiltered_ids"] - data["ondemand_ids"]
            delta_models: dict[str, list[str]] = {}
            for mid in delta_ids:
                itypes = data["unfiltered_details"].get(mid, [])
                delta_models[mid] = itypes
                global_delta[mid]["regions"].add(region)
                global_delta[mid]["inference_types"].update(itypes)

            for mid in data["unfiltered_ids"]:
                unfiltered_map[mid].add(region)
            for mid in data["ondemand_ids"]:
                ondemand_map[mid].add(region)

            region_results[region] = {
                "total": len(data["unfiltered_ids"]),
                "ondemand": len(data["ondemand_ids"]),
                "delta_count": len(delta_ids),
                "delta_models": delta_models,
                "error": None,
            }
            _log(
                f"  [{done_count}/{len(regions)}] {region}: "
                f"total={len(data['unfiltered_ids'])}, "
                f"on-demand={len(data['ondemand_ids'])}, "
                f"delta={len(delta_ids)}"
            )

    return {
        "region_results": region_results,
        "global_delta": {
            mid: {
                "regions": sorted(v["regions"]),
                "inference_types": sorted(v["inference_types"]),
            }
            for mid, v in global_delta.items()
        },
        "unfiltered_map": {mid: regs for mid, regs in unfiltered_map.items()},
        "ondemand_map": {mid: regs for mid, regs in ondemand_map.items()},
    }


def print_phase1(phase1: dict) -> None:
    """Print the Phase 1 report section."""
    rr = phase1["region_results"]
    gd = phase1["global_delta"]

    print("\n" + "=" * 70)
    print("PHASE 1 REPORT: ListFoundationModels — Unfiltered vs ON_DEMAND")
    print("=" * 70)

    # 1. Region summary table
    print(f"\n{'Region':<25} {'Total':>7} {'OnDemand':>9} {'Delta':>7} {'Error'}")
    print("-" * 70)
    for region in sorted(rr.keys()):
        r = rr[region]
        err = r.get("error") or ""
        print(
            f"{region:<25} {r['total']:>7} {r['ondemand']:>9} {r['delta_count']:>7} {err[:30]}"
        )

    totals_total = sum(r["total"] for r in rr.values())
    totals_od = sum(r["ondemand"] for r in rr.values())
    totals_delta = sum(r["delta_count"] for r in rr.values())
    print("-" * 70)
    print(f"{'TOTALS':<25} {totals_total:>7} {totals_od:>9} {totals_delta:>7}")

    # 2. Delta analysis (top 20)
    print(f"\n{'─' * 70}")
    print("DELTA ANALYSIS: Models in unfiltered but NOT in ON_DEMAND filter")
    print(f"{'─' * 70}")

    if not gd:
        print("  (none — every model appears in ON_DEMAND filter)")
    else:
        sorted_delta = sorted(
            gd.items(), key=lambda x: len(x[1]["regions"]), reverse=True
        )[:20]
        print(f"\nShowing top {len(sorted_delta)} of {len(gd)} unique delta models:\n")
        print(f"  {'Model ID':<55} {'#Regions':>8}  InferenceTypes")
        print(f"  {'-' * 55} {'-' * 8}  {'-' * 25}")
        for mid, info in sorted_delta:
            itypes_str = (
                ", ".join(info["inference_types"])
                if info["inference_types"]
                else "(empty)"
            )
            print(f"  {mid:<55} {len(info['regions']):>8}  {itypes_str}")
            if len(info["regions"]) <= 5:
                print(f"  {'':55} {'':>8}  regions: {', '.join(info['regions'])}")

    # Inference type breakdown
    itype_counts: dict[str, int] = defaultdict(int)
    for info in gd.values():
        key = (
            ", ".join(sorted(info["inference_types"]))
            if info["inference_types"]
            else "(none)"
        )
        itype_counts[key] += 1

    if itype_counts:
        print(f"\n  Inference type distribution among delta models:")
        for itype, count in sorted(itype_counts.items(), key=lambda x: -x[1]):
            print(f"    {itype:<40} {count:>5} models")


# ===================================================================
# Phase 2: Pricing region analysis
# ===================================================================


def _discover_s3_bucket() -> str:
    """Discover the profiler data bucket: bedrock-profiler-data-{account_id}-prod."""
    sts = boto3.client("sts", config=RETRY_CONFIG)
    account_id = sts.get_caller_identity()["Account"]
    return f"bedrock-profiler-data-{account_id}-prod"


def _read_pricing_from_s3(bucket: str) -> dict:
    """Read latest/bedrock_pricing.json from S3."""
    s3 = boto3.client("s3", config=RETRY_CONFIG)
    resp = s3.get_object(Bucket=bucket, Key="latest/bedrock_pricing.json")
    return json.loads(resp["Body"].read().decode("utf-8"))


def _extract_from_pricing(pricing_data: dict) -> dict[str, set[str]]:
    """
    Extract {model_id: set(regions)} from the pricing structure.
    Mirrors _extract_from_pricing in the regional-availability handler.
    """
    model_availability: dict[str, set[str]] = defaultdict(set)

    providers_data = pricing_data.get("providers", {})
    for provider_name, provider_models in providers_data.items():
        if not isinstance(provider_models, dict):
            continue
        for model_id, model_data in provider_models.items():
            if not isinstance(model_data, dict):
                continue
            model_regions = model_data.get("regions", {})
            for region in model_regions.keys():
                model_availability[model_id].add(region)

    return model_availability


def run_phase2(ondemand_map: dict[str, set[str]] | None) -> dict:
    """
    Phase 2: compare pricing data regions with Phase 1 ON_DEMAND results.

    If Phase 1 wasn't run (ondemand_map is None), runs a lightweight Phase 1
    just for ON_DEMAND to get the comparison data.

    Returns:
        {
            "bucket": str,
            "pricing_model_count": int,
            "pricing_region_count": int,
            "false_positives": { model_id: [regions] },
        }
    """
    _log(f"\n{'=' * 70}")
    _log("PHASE 2: Pricing Region Analysis")
    _log(f"{'=' * 70}")

    # If we don't have Phase 1 data, we need to collect ON_DEMAND data
    if ondemand_map is None:
        _log("[phase2] No Phase 1 data; running ON_DEMAND collection ...")
        regions = discover_bedrock_regions()
        p1 = run_phase1(regions)
        ondemand_map = p1["ondemand_map"]

    # Read pricing data from S3
    bucket = _discover_s3_bucket()
    _log(f"[phase2] Reading pricing from s3://{bucket}/latest/bedrock_pricing.json ...")
    pricing_data = _read_pricing_from_s3(bucket)
    pricing_map = _extract_from_pricing(pricing_data)

    pricing_all_regions: set[str] = set()
    for regs in pricing_map.values():
        pricing_all_regions.update(regs)

    _log(
        f"[phase2] Pricing covers {len(pricing_map)} model IDs across {len(pricing_all_regions)} regions"
    )

    # Find false positives: pricing claims a region but ON_DEMAND doesn't list the model
    false_positives: dict[str, list[str]] = {}
    for model_id, pricing_regions in pricing_map.items():
        od_regions = ondemand_map.get(model_id, set())
        fp_regions = sorted(pricing_regions - od_regions)
        if fp_regions:
            false_positives[model_id] = fp_regions

    _log(f"[phase2] Found {len(false_positives)} models with pricing false positives")

    return {
        "bucket": bucket,
        "pricing_model_count": len(pricing_map),
        "pricing_region_count": len(pricing_all_regions),
        "false_positives": false_positives,
        "pricing_map": pricing_map,
    }


def print_phase2(phase2: dict) -> None:
    """Print the Phase 2 report section."""
    fp = phase2["false_positives"]

    print("\n" + "=" * 70)
    print("PHASE 2 REPORT: Pricing False Positives")
    print("=" * 70)

    print(f"\n  S3 bucket:        {phase2['bucket']}")
    print(f"  Pricing models:   {phase2['pricing_model_count']}")
    print(f"  Pricing regions:  {phase2['pricing_region_count']}")
    print(f"  False positives:  {len(fp)} models\n")

    if not fp:
        print("  (none — pricing data is consistent with ON_DEMAND filter)")
        return

    # Sort by number of false-positive regions descending
    sorted_fp = sorted(fp.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"  {'Model ID':<55} {'#FP Regions':>11}  Regions")
    print(f"  {'-' * 55} {'-' * 11}  {'-' * 30}")
    for mid, regions in sorted_fp[:40]:
        regions_str = ", ".join(regions[:6])
        if len(regions) > 6:
            regions_str += f" (+{len(regions) - 6} more)"
        print(f"  {mid:<55} {len(regions):>11}  {regions_str}")

    if len(sorted_fp) > 40:
        print(f"\n  ... and {len(sorted_fp) - 40} more models (truncated)")

    # Aggregate: which regions appear most in false positives?
    region_fp_count: dict[str, int] = defaultdict(int)
    for regions in fp.values():
        for r in regions:
            region_fp_count[r] += 1

    print(f"\n  False positives by region:")
    for r, cnt in sorted(region_fp_count.items(), key=lambda x: -x[1])[:15]:
        print(f"    {r:<25} {cnt:>5} models")


# ===================================================================
# Phase 3: Converse API spot-check
# ===================================================================


def _is_embedding_model(model_id: str) -> bool:
    """Check if a model is an embedding/rerank model (doesn't support Converse)."""
    lower = model_id.lower()
    return any(kw in lower for kw in EMBEDDING_KEYWORDS)


def _try_converse(region: str, model_id: str) -> dict:
    """
    Attempt a minimal Converse API call. Returns a result dict with the outcome.
    """
    result = {
        "region": region,
        "model_id": model_id,
        "status": "UNKNOWN",
        "error_code": None,
        "message": None,
    }

    try:
        client = boto3.client(
            "bedrock-runtime", region_name=region, config=RETRY_CONFIG
        )
        resp = client.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "Hi"}],
                }
            ],
            inferenceConfig={"maxTokens": 1},
        )
        result["status"] = "SUCCESS"
        result["message"] = f"stopReason={resp.get('stopReason', '?')}"

    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        msg = exc.response.get("Error", {}).get("Message", "")[:120]
        result["error_code"] = code
        result["message"] = msg

        if code == "AccessDeniedException":
            result["status"] = "ACCESS_DENIED"
        elif code == "ModelNotReadyException":
            result["status"] = "NOT_READY"
        elif code == "ValidationException":
            result["status"] = "VALIDATION_ERROR"
        elif code == "ResourceNotFoundException":
            result["status"] = "NOT_FOUND"
        elif code == "ThrottlingException":
            result["status"] = "THROTTLED"
        elif code == "ModelTimeoutException":
            result["status"] = "TIMEOUT"
        else:
            result["status"] = f"CLIENT_ERROR:{code}"

    except EndpointConnectionError:
        result["status"] = "NO_ENDPOINT"
        result["message"] = "bedrock-runtime endpoint not reachable"

    except Exception as exc:
        result["status"] = f"EXCEPTION:{type(exc).__name__}"
        result["message"] = str(exc)[:120]

    return result


def run_phase3(
    unfiltered_map: dict[str, set[str]] | None,
    ondemand_map: dict[str, set[str]] | None,
) -> dict:
    """
    Phase 3: Converse API spot-check for representative models.

    Returns:
        {
            "results": { model_id: { region: {status, ...} } },
            "summary": { model_id: {tested, success, access_denied, ...} },
        }
    """
    _log(f"\n{'=' * 70}")
    _log("PHASE 3: Converse API Spot-Check")
    _log(f"{'=' * 70}")

    # If we don't have Phase 1 data, collect it
    if unfiltered_map is None or ondemand_map is None:
        _log("[phase3] No Phase 1 data; running collection ...")
        regions = discover_bedrock_regions()
        p1 = run_phase1(regions)
        unfiltered_map = p1["unfiltered_map"]
        ondemand_map = p1["ondemand_map"]

    # Build task list: for each spot-check model, find regions where it appears unfiltered
    tasks: list[tuple[str, str]] = []
    for model_id in CONVERSE_SPOT_CHECK_MODELS:
        if _is_embedding_model(model_id):
            _log(f"  Skipping embedding model: {model_id}")
            continue
        model_regions = sorted(unfiltered_map.get(model_id, set()))
        if not model_regions:
            _log(f"  Model not found in any region (unfiltered): {model_id}")
            continue
        for region in model_regions:
            tasks.append((model_id, region))

    _log(
        f"[phase3] {len(tasks)} Converse calls across {len(CONVERSE_SPOT_CHECK_MODELS)} models ..."
    )

    results: dict[str, dict[str, dict]] = defaultdict(dict)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PER_PHASE) as pool:
        futures = {pool.submit(_try_converse, r, mid): (mid, r) for mid, r in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            mid, reg = res["model_id"], res["region"]
            results[mid][reg] = res
            if done % 20 == 0 or done == len(tasks):
                _log(f"  [{done}/{len(tasks)}] completed")

    # Build summary per model
    summary: dict[str, dict] = {}
    for model_id in CONVERSE_SPOT_CHECK_MODELS:
        model_res = results.get(model_id, {})
        counts: dict[str, int] = defaultdict(int)
        for region_res in model_res.values():
            counts[region_res["status"]] += 1
        counts["tested"] = len(model_res)

        od_regions = ondemand_map.get(model_id, set())
        uf_regions = unfiltered_map.get(model_id, set())
        counts["in_ondemand_filter"] = len(od_regions)
        counts["in_unfiltered"] = len(uf_regions)
        summary[model_id] = dict(counts)

    return {
        "results": {mid: dict(regs) for mid, regs in results.items()},
        "summary": summary,
        "ondemand_map": ondemand_map,
        "unfiltered_map": unfiltered_map,
    }


def print_phase3(phase3: dict) -> None:
    """Print the Phase 3 report section."""
    results = phase3["results"]
    summary = phase3["summary"]
    ondemand_map = phase3.get("ondemand_map", {})

    print("\n" + "=" * 70)
    print("PHASE 3 REPORT: Converse API Spot-Check")
    print("=" * 70)

    for model_id in CONVERSE_SPOT_CHECK_MODELS:
        s = summary.get(model_id, {})
        print(f"\n  Model: {model_id}")
        print(f"    Unfiltered regions:  {s.get('in_unfiltered', 0)}")
        print(f"    ON_DEMAND regions:   {s.get('in_ondemand_filter', 0)}")
        print(f"    Tested:              {s.get('tested', 0)}")

        # Status breakdown
        skip_keys = {"tested", "in_ondemand_filter", "in_unfiltered"}
        status_items = {k: v for k, v in s.items() if k not in skip_keys and v > 0}
        if status_items:
            for status, cnt in sorted(status_items.items(), key=lambda x: -x[1]):
                print(f"      {status:<25} {cnt:>4}")

    # Per-model × region matrix
    print(f"\n{'─' * 70}")
    print("CONVERSE RESULT MATRIX (model × region)")
    print(f"{'─' * 70}")

    for model_id in CONVERSE_SPOT_CHECK_MODELS:
        model_res = results.get(model_id, {})
        if not model_res:
            print(f"\n  {model_id}: (no regions tested)")
            continue

        od_regions = ondemand_map.get(model_id, set())
        print(f"\n  {model_id}:")
        print(f"    {'Region':<25} {'Converse':<20} {'InOD?':>5}  Detail")
        print(f"    {'-' * 25} {'-' * 20} {'-' * 5}  {'-' * 35}")

        for region in sorted(model_res.keys()):
            r = model_res[region]
            in_od = "YES" if region in od_regions else "NO"
            detail = (r.get("message") or "")[:35]
            print(f"    {region:<25} {r['status']:<20} {in_od:>5}  {detail}")


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify on-demand model availability across Bedrock regions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        type=int,
        action="append",
        choices=[1, 2, 3],
        help="Run a specific phase (can be repeated). Default: all phases.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Run all phases (default when no --phase given).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Also dump raw results as JSON to stdout after the report.",
    )
    args = parser.parse_args()

    phases_to_run: set[int] = set()
    if args.all or args.phase is None:
        phases_to_run = {1, 2, 3}
    else:
        phases_to_run = set(args.phase)

    start = time.time()
    _log(f"Phases to run: {sorted(phases_to_run)}")

    # Verify credentials
    try:
        sts = boto3.client("sts", config=RETRY_CONFIG)
        identity = sts.get_caller_identity()
        _log(f"AWS Account: {identity['Account']}, Identity: {identity['Arn']}")
    except Exception as exc:
        _log(f"FATAL: Cannot verify AWS credentials — {exc}")
        sys.exit(1)

    # Discover regions (shared across phases)
    regions: list[str] | None = None
    phase1_data: dict | None = None
    phase2_data: dict | None = None
    phase3_data: dict | None = None

    # ── Phase 1 ──────────────────────────────────────────────────────
    if 1 in phases_to_run:
        regions = discover_bedrock_regions()
        phase1_data = run_phase1(regions)
        print_phase1(phase1_data)

    # ── Phase 2 ──────────────────────────────────────────────────────
    if 2 in phases_to_run:
        ondemand_map = None
        if phase1_data is not None:
            ondemand_map = phase1_data["ondemand_map"]
        phase2_data = run_phase2(ondemand_map)
        print_phase2(phase2_data)

    # ── Phase 3 ──────────────────────────────────────────────────────
    if 3 in phases_to_run:
        unfiltered_map = None
        ondemand_map = None
        if phase1_data is not None:
            unfiltered_map = phase1_data["unfiltered_map"]
            ondemand_map = phase1_data["ondemand_map"]
        phase3_data = run_phase3(unfiltered_map, ondemand_map)
        print_phase3(phase3_data)

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"COMPLETED in {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Optional JSON dump
    if args.json:

        def _sets_to_lists(obj: Any) -> Any:
            """Recursively convert sets to sorted lists for JSON serialisation."""
            if isinstance(obj, set):
                return sorted(obj)
            if isinstance(obj, dict):
                return {k: _sets_to_lists(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sets_to_lists(i) for i in obj]
            return obj

        raw: dict[str, Any] = {}
        if phase1_data:
            raw["phase1"] = _sets_to_lists(phase1_data)
        if phase2_data:
            raw["phase2"] = _sets_to_lists(phase2_data)
        if phase3_data:
            raw["phase3"] = _sets_to_lists(phase3_data)

        print("\n--- RAW JSON ---")
        print(json.dumps(raw, indent=2, default=str))


if __name__ == "__main__":
    main()
