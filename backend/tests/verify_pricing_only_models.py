#!/usr/bin/env python3
"""
Pricing-Only Model Verification Script

Determines whether any models that appear in pricing data but NOT in the
ON_DEMAND-filtered ListFoundationModels API are genuinely invocable.

This answers the question: "If we stop using pricing data to discover
on-demand models, do we lose any models that actually work?"

Steps:
  1. Collect ON_DEMAND-filtered and unfiltered ListFoundationModels across all regions
  2. Read pricing data from S3 and extract model×region pairs
  3. Identify pricing-only models (in pricing but not in ON_DEMAND filter)
  4. Attempt Converse API calls to verify actual invocability
  5. Report findings

Usage:
    python verify_pricing_only_models.py
    python verify_pricing_only_models.py --json
    python verify_pricing_only_models.py --max-models 30
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

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

S3_BUCKET = "bedrock-profiler-data-390445053194-prod"
S3_KEY = "latest/bedrock_pricing.json"

MAX_WORKERS = 10
DEFAULT_MAX_MODELS = 50

# Pricing model IDs starting with these prefixes are non-model services
UNKNOWN_PREFIXES = ("unknown.",)

# Embedding models don't support Converse
EMBEDDING_KEYWORDS = ["embed", "embedding", "rerank"]

# Version suffix patterns to strip for fuzzy matching
VERSION_SUFFIX_RE = re.compile(r"(-v\d+)?:\d+$")
DATE_SUFFIX_RE = re.compile(r"-\d{8}(-v\d+)?(:\d+)?$")


def _log(msg: str) -> None:
    """Print progress to stderr."""
    print(msg, file=sys.stderr, flush=True)


# ===================================================================
# Step 1: Collect data
# ===================================================================


def discover_bedrock_regions() -> List[str]:
    """
    Dynamically discover AWS regions where the Bedrock control-plane is
    reachable. Uses EC2 DescribeRegions then probes each with a
    lightweight ListFoundationModels call.
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

    def _probe(region: str) -> Tuple[str, bool]:
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

    bedrock_regions = []  # type: List[str]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_probe, r): r for r in all_regions}
        for fut in as_completed(futures):
            region, ok = fut.result()
            if ok:
                bedrock_regions.append(region)

    bedrock_regions.sort()
    _log(f"[discovery] Found {len(bedrock_regions)} Bedrock regions")
    return bedrock_regions


def _query_region(region: str) -> Dict[str, Any]:
    """
    For a single region, call ListFoundationModels twice:
      1. unfiltered (all models)
      2. filtered by ON_DEMAND
    """
    result = {
        "region": region,
        "error": None,
        "unfiltered_ids": set(),  # type: Set[str]
        "ondemand_ids": set(),  # type: Set[str]
    }
    try:
        client = boto3.client("bedrock", region_name=region, config=RETRY_CONFIG)

        resp_all = client.list_foundation_models()
        for m in resp_all.get("modelSummaries", []):
            mid = m.get("modelId", "")
            if mid:
                result["unfiltered_ids"].add(mid)

        resp_od = client.list_foundation_models(byInferenceType="ON_DEMAND")
        for m in resp_od.get("modelSummaries", []):
            mid = m.get("modelId", "")
            if mid:
                result["ondemand_ids"].add(mid)

    except Exception as exc:
        result["error"] = str(exc)

    return result


def collect_bedrock_data(
    regions: List[str],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Query all regions for unfiltered and ON_DEMAND-filtered model lists.

    Returns:
        (unfiltered_map, ondemand_map) where each is {model_id: set(regions)}
    """
    _log(f"\n[step1] Querying {len(regions)} regions (2 API calls each) ...")

    unfiltered_map = defaultdict(set)  # type: Dict[str, Set[str]]
    ondemand_map = defaultdict(set)  # type: Dict[str, Set[str]]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_query_region, r): r for r in regions}
        done_count = 0
        for fut in as_completed(futures):
            done_count += 1
            data = fut.result()
            region = data["region"]

            if data["error"]:
                _log(
                    f"  [{done_count}/{len(regions)}] {region}: ERROR — {data['error']}"
                )
                continue

            for mid in data["unfiltered_ids"]:
                unfiltered_map[mid].add(region)
            for mid in data["ondemand_ids"]:
                ondemand_map[mid].add(region)

            _log(
                f"  [{done_count}/{len(regions)}] {region}: "
                f"unfiltered={len(data['unfiltered_ids'])}, "
                f"on-demand={len(data['ondemand_ids'])}"
            )

    _log(
        f"[step1] Total unique models: "
        f"unfiltered={len(unfiltered_map)}, on-demand={len(ondemand_map)}"
    )
    return dict(unfiltered_map), dict(ondemand_map)


def read_pricing_data() -> Dict[str, Any]:
    """Read pricing data from S3."""
    _log(f"[step1] Reading pricing from s3://{S3_BUCKET}/{S3_KEY} ...")
    s3 = boto3.client("s3", config=RETRY_CONFIG)
    resp = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
    data = json.loads(resp["Body"].read().decode("utf-8"))
    _log(f"[step1] Pricing data loaded successfully")
    return data


def extract_pricing_pairs(pricing_data: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Extract {model_id: set(regions)} from pricing structure.
    Iterates providers → model_id → regions.
    """
    model_regions = defaultdict(set)  # type: Dict[str, Set[str]]

    providers_data = pricing_data.get("providers", {})
    for provider_name, provider_models in providers_data.items():
        if not isinstance(provider_models, dict):
            continue
        for model_id, model_data in provider_models.items():
            if not isinstance(model_data, dict):
                continue
            regions = model_data.get("regions", {})
            for region in regions.keys():
                model_regions[model_id].add(region)

    _log(f"[step1] Pricing data contains {len(model_regions)} model IDs")
    return dict(model_regions)


# ===================================================================
# Step 2: Find pricing-only models
# ===================================================================


def _is_unknown_prefix(model_id: str) -> bool:
    """Check if model ID starts with a non-model prefix like 'unknown.'."""
    return any(model_id.startswith(p) for p in UNKNOWN_PREFIXES)


def _is_embedding_model(model_id: str) -> bool:
    """Check if a model is an embedding/rerank model."""
    lower = model_id.lower()
    return any(kw in lower for kw in EMBEDDING_KEYWORDS)


def _normalize_model_id(model_id: str) -> str:
    """
    Strip version suffixes for fuzzy matching.
    E.g., 'anthropic.claude-sonnet-4-5-20250929-v1:0' → 'anthropic.claude-sonnet-4-5'
    """
    # Strip trailing ':N' or '-vN:N'
    normalized = VERSION_SUFFIX_RE.sub("", model_id)
    # Strip date suffix like '-20250929'
    normalized = DATE_SUFFIX_RE.sub("", normalized)
    return normalized


def _fuzzy_match(pricing_id: str, all_bedrock_ids: Set[str]) -> Optional[str]:
    """
    Try to fuzzy-match a pricing model ID to a real Bedrock model ID.

    Strategies:
      1. pricing_id is a substring of a Bedrock ID
      2. A Bedrock ID is a substring of pricing_id
      3. Normalized forms match
    """
    pricing_norm = _normalize_model_id(pricing_id)

    best_match = None  # type: Optional[str]
    best_score = 0

    for bedrock_id in all_bedrock_ids:
        # Exact substring: pricing ID is prefix/substring of bedrock ID
        if pricing_id in bedrock_id:
            # Prefer longest pricing_id match (most specific)
            score = len(pricing_id)
            if score > best_score:
                best_match = bedrock_id
                best_score = score
            continue

        # Reverse: bedrock ID is substring of pricing ID (less likely but check)
        if bedrock_id in pricing_id:
            score = len(bedrock_id)
            if score > best_score:
                best_match = bedrock_id
                best_score = score
            continue

        # Normalized match
        bedrock_norm = _normalize_model_id(bedrock_id)
        if pricing_norm == bedrock_norm:
            # Prefer this as it's a strong match
            score = len(pricing_norm) + 100
            if score > best_score:
                best_match = bedrock_id
                best_score = score
            continue

        # Normalized substring
        if pricing_norm in bedrock_norm or bedrock_norm in pricing_norm:
            score = min(len(pricing_norm), len(bedrock_norm))
            if score > best_score:
                best_match = bedrock_id
                best_score = score

    return best_match


# Classification of a pricing model×region pair
CATEGORY_IN_ONDEMAND = "IN_ONDEMAND"  # Already in ON_DEMAND filter (skip)
CATEGORY_IN_UNFILTERED = "IN_UNFILTERED_ONLY"  # Known Bedrock model but not ON_DEMAND
CATEGORY_PRICING_ONLY = "PRICING_ONLY"  # Not in any Bedrock list
CATEGORY_UNKNOWN = "UNKNOWN_PREFIX"  # Starts with 'unknown.' etc.


def classify_pricing_models(
    pricing_map: Dict[str, Set[str]],
    ondemand_map: Dict[str, Set[str]],
    unfiltered_map: Dict[str, Set[str]],
) -> Dict[str, Dict[str, Any]]:
    """
    Classify each pricing model ID into categories.

    Returns:
        {
            model_id: {
                "category": str,
                "pricing_regions": set(str),
                "ondemand_regions": set(str),    # regions in ON_DEMAND filter
                "unfiltered_regions": set(str),  # regions in unfiltered list
                "excess_regions": set(str),      # pricing regions NOT in ON_DEMAND
                "fuzzy_match": Optional[str],    # matched Bedrock ID if pricing-only
                "is_embedding": bool,
            }
        }
    """
    all_bedrock_ids = set(unfiltered_map.keys())
    classified = {}  # type: Dict[str, Dict[str, Any]]

    for model_id, pricing_regions in pricing_map.items():
        od_regions = ondemand_map.get(model_id, set())
        uf_regions = unfiltered_map.get(model_id, set())

        # Regions in pricing but not in ON_DEMAND filter
        excess_regions = pricing_regions - od_regions

        info = {
            "pricing_regions": pricing_regions,
            "ondemand_regions": od_regions,
            "unfiltered_regions": uf_regions,
            "excess_regions": excess_regions,
            "fuzzy_match": None,
            "is_embedding": _is_embedding_model(model_id),
        }  # type: Dict[str, Any]

        if not excess_regions:
            # All pricing regions are covered by ON_DEMAND filter → skip
            info["category"] = CATEGORY_IN_ONDEMAND
        elif _is_unknown_prefix(model_id):
            info["category"] = CATEGORY_UNKNOWN
        elif model_id in unfiltered_map:
            # Known in Bedrock (unfiltered) but not in ON_DEMAND for some regions
            info["category"] = CATEGORY_IN_UNFILTERED
        else:
            # Not in any Bedrock list — try fuzzy match
            info["category"] = CATEGORY_PRICING_ONLY
            info["fuzzy_match"] = _fuzzy_match(model_id, all_bedrock_ids)

        classified[model_id] = info

    return classified


# ===================================================================
# Step 3: Converse API verification
# ===================================================================


def _try_converse(region: str, model_id: str) -> Dict[str, Any]:
    """
    Attempt a minimal Converse API call. Returns a result dict with the outcome.
    """
    result = {
        "region": region,
        "model_id": model_id,
        "status": "UNKNOWN",
        "error_code": None,
        "message": None,
    }  # type: Dict[str, Any]

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
        result["message"] = "stopReason=%s" % resp.get("stopReason", "?")

    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        msg = exc.response.get("Error", {}).get("Message", "")[:150]
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
            result["status"] = "CLIENT_ERROR:%s" % code

    except EndpointConnectionError:
        result["status"] = "NO_ENDPOINT"
        result["message"] = "bedrock-runtime endpoint not reachable"

    except Exception as exc:
        result["status"] = "EXCEPTION:%s" % type(exc).__name__
        result["message"] = str(exc)[:150]

    return result


def run_converse_checks(
    classified: Dict[str, Dict[str, Any]],
    max_models: int,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    For pricing-only and unfiltered-only models, attempt Converse API calls
    on their excess regions (pricing regions not covered by ON_DEMAND).

    We test:
      - The pricing model ID directly
      - If that fails and a fuzzy match exists, the matched Bedrock model ID

    Returns:
        {
            model_id: {
                "pricing_id_results": {region: result_dict},
                "fuzzy_id_results": {region: result_dict},  # only if fuzzy match tried
            }
        }
    """
    # Collect testable models: PRICING_ONLY and IN_UNFILTERED_ONLY with excess regions
    testable = {}  # type: Dict[str, Dict[str, Any]]
    for model_id, info in classified.items():
        if info["category"] in (CATEGORY_IN_ONDEMAND, CATEGORY_UNKNOWN):
            continue
        if not info["excess_regions"]:
            continue
        if info["is_embedding"]:
            continue
        testable[model_id] = info

    _log(f"\n[step3] {len(testable)} models eligible for Converse testing")

    if not testable:
        _log("[step3] Nothing to test — all pricing models are in ON_DEMAND filter")
        return {}

    # Sort by number of excess regions (descending) and take top N
    sorted_models = sorted(
        testable.items(),
        key=lambda x: len(x[1]["excess_regions"]),
        reverse=True,
    )

    if len(sorted_models) > max_models:
        _log(
            f"[step3] Sampling top {max_models} models by excess region count "
            f"(out of {len(sorted_models)})"
        )
        sorted_models = sorted_models[:max_models]
    else:
        _log(f"[step3] Testing all {len(sorted_models)} eligible models")

    # Build task list: (model_id, region, is_fuzzy_attempt)
    tasks = []  # type: List[Tuple[str, str, str, bool]]
    for model_id, info in sorted_models:
        for region in sorted(info["excess_regions"]):
            # First attempt: use the pricing model ID
            tasks.append((model_id, region, model_id, False))

    total_tasks = len(tasks)
    _log(f"[step3] {total_tasks} Converse calls to make (pricing IDs first)")

    # Phase 1: Try with pricing model IDs
    results = defaultdict(lambda: {"pricing_id_results": {}, "fuzzy_id_results": {}})  # type: Dict[str, Dict[str, Any]]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for model_id, region, call_id, _ in tasks:
            fut = pool.submit(_try_converse, region, call_id)
            futures[fut] = (model_id, region, call_id)

        done = 0
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            model_id, region, call_id = futures[fut]
            results[model_id]["pricing_id_results"][region] = res

            if done % 25 == 0 or done == total_tasks:
                _log(f"  [{done}/{total_tasks}] pricing ID Converse calls completed")

    # Phase 2: For pricing IDs that failed (NOT_FOUND or VALIDATION_ERROR),
    # try the fuzzy-matched Bedrock ID if available
    fuzzy_tasks = []  # type: List[Tuple[str, str, str]]
    for model_id, info in sorted_models:
        fuzzy_id = info.get("fuzzy_match")
        if not fuzzy_id:
            continue
        if _is_embedding_model(fuzzy_id):
            continue
        for region in sorted(info["excess_regions"]):
            pricing_result = results[model_id]["pricing_id_results"].get(region, {})
            status = pricing_result.get("status", "")
            if status in ("NOT_FOUND", "VALIDATION_ERROR", "UNKNOWN"):
                fuzzy_tasks.append((model_id, region, fuzzy_id))

    if fuzzy_tasks:
        _log(f"[step3] {len(fuzzy_tasks)} fuzzy-match Converse calls to make")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures2 = {}
            for model_id, region, fuzzy_id in fuzzy_tasks:
                fut = pool.submit(_try_converse, region, fuzzy_id)
                futures2[fut] = (model_id, region, fuzzy_id)

            done2 = 0
            for fut in as_completed(futures2):
                done2 += 1
                res = fut.result()
                model_id, region, fuzzy_id = futures2[fut]
                results[model_id]["fuzzy_id_results"][region] = res

                if done2 % 25 == 0 or done2 == len(fuzzy_tasks):
                    _log(
                        f"  [{done2}/{len(fuzzy_tasks)}] fuzzy-match "
                        f"Converse calls completed"
                    )
    else:
        _log("[step3] No fuzzy-match retries needed")

    return dict(results)


# ===================================================================
# Step 4: Report
# ===================================================================


def print_report(
    classified: Dict[str, Dict[str, Any]],
    converse_results: Dict[str, Dict[str, Any]],
    max_models: int,
) -> None:
    """Print the structured report to stdout."""

    # ── Summary ──────────────────────────────────────────────────────
    cat_counts = defaultdict(int)  # type: Dict[str, int]
    cat_pair_counts = defaultdict(int)  # type: Dict[str, int]

    for model_id, info in classified.items():
        cat = info["category"]
        cat_counts[cat] += 1
        cat_pair_counts[cat] += len(info.get("excess_regions", set()))

    total_pricing_models = len(classified)
    in_ondemand = cat_counts.get(CATEGORY_IN_ONDEMAND, 0)
    in_unfiltered = cat_counts.get(CATEGORY_IN_UNFILTERED, 0)
    pricing_only = cat_counts.get(CATEGORY_PRICING_ONLY, 0)
    unknown_count = cat_counts.get(CATEGORY_UNKNOWN, 0)

    print("=" * 80)
    print("PRICING-ONLY MODEL VERIFICATION REPORT")
    print("=" * 80)

    print("\n── SUMMARY ─────────────────────────────────────────────────")
    print(f"  Total pricing model IDs:                     {total_pricing_models:>6}")
    print(
        f"  Fully covered by ON_DEMAND filter:           {in_ondemand:>6}  (no excess regions)"
    )
    print(
        f"  In unfiltered list, NOT in ON_DEMAND filter: {in_unfiltered:>6}  "
        f"({cat_pair_counts.get(CATEGORY_IN_UNFILTERED, 0)} excess model×region pairs)"
    )
    print(
        f"  Pricing-only (not in any Bedrock list):      {pricing_only:>6}  "
        f"({cat_pair_counts.get(CATEGORY_PRICING_ONLY, 0)} excess model×region pairs)"
    )
    print(
        f"  Unknown/non-model (unknown.* prefix):        {unknown_count:>6}  (filtered out)"
    )

    # ── Fuzzy match summary ──────────────────────────────────────────
    pricing_only_models = {
        mid: info
        for mid, info in classified.items()
        if info["category"] == CATEGORY_PRICING_ONLY
    }
    matched = {
        mid: info
        for mid, info in pricing_only_models.items()
        if info.get("fuzzy_match")
    }
    unmatched = {
        mid: info
        for mid, info in pricing_only_models.items()
        if not info.get("fuzzy_match")
    }

    print(f"\n── PRICING-ONLY FUZZY MATCHING ─────────────────────────────")
    print(f"  Fuzzy-matched to a Bedrock model ID:  {len(matched):>6}")
    print(f"  No match found:                       {len(unmatched):>6}")

    if matched:
        print(f"\n  Matches:")
        for mid in sorted(matched.keys()):
            fm = matched[mid]["fuzzy_match"]
            n_excess = len(matched[mid]["excess_regions"])
            print(f"    {mid:<50} → {fm}  ({n_excess} regions)")

    if unmatched:
        print(f"\n  Unmatched pricing-only IDs:")
        for mid in sorted(unmatched.keys()):
            n_excess = len(unmatched[mid]["excess_regions"])
            is_emb = " [EMBEDDING]" if unmatched[mid]["is_embedding"] else ""
            print(f"    {mid:<50} ({n_excess} regions){is_emb}")

    # ── IN_UNFILTERED_ONLY detail ────────────────────────────────────
    unfiltered_only = {
        mid: info
        for mid, info in classified.items()
        if info["category"] == CATEGORY_IN_UNFILTERED
    }
    if unfiltered_only:
        print(f"\n── IN-UNFILTERED-BUT-NOT-ON_DEMAND MODELS ─────────────────")
        print(f"  These models exist in the Bedrock unfiltered list but are NOT")
        print(
            f"  returned by byInferenceType='ON_DEMAND'. Pricing claims extra regions.\n"
        )
        sorted_uf = sorted(
            unfiltered_only.items(),
            key=lambda x: len(x[1]["excess_regions"]),
            reverse=True,
        )
        print(f"  {'Model ID':<55} {'Excess':>6} {'OD':>4} {'UF':>4} {'Pricing':>7}")
        print(f"  {'-' * 55} {'-' * 6} {'-' * 4} {'-' * 4} {'-' * 7}")
        for mid, info in sorted_uf[:30]:
            print(
                f"  {mid:<55} "
                f"{len(info['excess_regions']):>6} "
                f"{len(info['ondemand_regions']):>4} "
                f"{len(info['unfiltered_regions']):>4} "
                f"{len(info['pricing_regions']):>7}"
            )
        if len(sorted_uf) > 30:
            print(f"  ... and {len(sorted_uf) - 30} more")

    # ── Unknown prefix detail ────────────────────────────────────────
    unknowns = {
        mid: info
        for mid, info in classified.items()
        if info["category"] == CATEGORY_UNKNOWN
    }
    if unknowns:
        print(f"\n── FILTERED OUT: unknown.* PREFIX MODELS ───────────────────")
        for mid in sorted(unknowns.keys()):
            n_regions = len(unknowns[mid]["pricing_regions"])
            print(f"    {mid:<50} ({n_regions} pricing regions)")

    # ── Converse results ─────────────────────────────────────────────
    print(f"\n── CONVERSE API VERIFICATION RESULTS ───────────────────────")

    if not converse_results:
        print("  (No models required Converse verification)")
    else:
        # Aggregate status counts
        all_statuses = defaultdict(int)  # type: Dict[str, int]
        total_calls = 0
        for model_id, result_data in converse_results.items():
            for region, res in result_data.get("pricing_id_results", {}).items():
                all_statuses[res["status"]] += 1
                total_calls += 1
            for region, res in result_data.get("fuzzy_id_results", {}).items():
                all_statuses["FUZZY:" + res["status"]] += 1
                total_calls += 1

        print(f"\n  Total Converse calls made: {total_calls}")
        print(f"\n  Status distribution (pricing ID attempts):")
        pricing_statuses = defaultdict(int)  # type: Dict[str, int]
        for model_id, result_data in converse_results.items():
            for region, res in result_data.get("pricing_id_results", {}).items():
                pricing_statuses[res["status"]] += 1
        for status, count in sorted(pricing_statuses.items(), key=lambda x: -x[1]):
            print(f"    {status:<30} {count:>5}")

        fuzzy_statuses = defaultdict(int)  # type: Dict[str, int]
        for model_id, result_data in converse_results.items():
            for region, res in result_data.get("fuzzy_id_results", {}).items():
                fuzzy_statuses[res["status"]] += 1
        if fuzzy_statuses:
            print(f"\n  Status distribution (fuzzy-match ID attempts):")
            for status, count in sorted(fuzzy_statuses.items(), key=lambda x: -x[1]):
                print(f"    {status:<30} {count:>5}")

        # Per-model detail
        print(f"\n  Per-model Converse results:")
        print(f"  {'─' * 75}")

        for model_id in sorted(converse_results.keys()):
            result_data = converse_results[model_id]
            info = classified.get(model_id, {})
            category = info.get("category", "?")
            fuzzy = info.get("fuzzy_match", None)

            print(f"\n  {model_id}")
            print(f"    Category: {category}")
            if fuzzy:
                print(f"    Fuzzy match: {fuzzy}")

            pricing_res = result_data.get("pricing_id_results", {})
            fuzzy_res = result_data.get("fuzzy_id_results", {})

            if pricing_res:
                print(
                    f"    {'Region':<25} {'PricingID Status':<22} {'FuzzyID Status':<22} Detail"
                )
                print(f"    {'-' * 25} {'-' * 22} {'-' * 22} {'-' * 30}")
                for region in sorted(pricing_res.keys()):
                    p_res = pricing_res[region]
                    f_res = fuzzy_res.get(region, {})
                    f_status = f_res.get("status", "-") if f_res else "-"
                    detail = (p_res.get("message") or "")[:30]
                    if f_res and f_res.get("status") == "SUCCESS":
                        detail = "FUZZY SUCCESS: " + (f_res.get("message") or "")[:15]
                    print(
                        f"    {region:<25} {p_res['status']:<22} {f_status:<22} {detail}"
                    )

    # ── KEY FINDING ──────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("KEY FINDING")
    print("=" * 80)

    # Find models where Converse SUCCEEDS but ON_DEMAND filter did NOT include them
    genuine_losses = []  # type: List[Tuple[str, str, str]]
    for model_id, result_data in converse_results.items():
        info = classified.get(model_id, {})
        od_regions = info.get("ondemand_regions", set())

        # Check pricing ID successes
        for region, res in result_data.get("pricing_id_results", {}).items():
            if res["status"] == "SUCCESS" and region not in od_regions:
                genuine_losses.append((model_id, region, "pricing_id"))

        # Check fuzzy ID successes
        for region, res in result_data.get("fuzzy_id_results", {}).items():
            if res["status"] == "SUCCESS" and region not in od_regions:
                fuzzy_id = info.get("fuzzy_match", model_id)
                genuine_losses.append((model_id, region, "fuzzy:%s" % fuzzy_id))

    # Also flag ACCESS_DENIED as potentially invocable (model exists, just IAM)
    access_denied_cases = []  # type: List[Tuple[str, str, str]]
    for model_id, result_data in converse_results.items():
        info = classified.get(model_id, {})
        od_regions = info.get("ondemand_regions", set())

        for region, res in result_data.get("pricing_id_results", {}).items():
            if res["status"] == "ACCESS_DENIED" and region not in od_regions:
                access_denied_cases.append((model_id, region, "pricing_id"))

        for region, res in result_data.get("fuzzy_id_results", {}).items():
            if res["status"] == "ACCESS_DENIED" and region not in od_regions:
                fuzzy_id = info.get("fuzzy_match", model_id)
                access_denied_cases.append((model_id, region, "fuzzy:%s" % fuzzy_id))

    if genuine_losses:
        print(
            f"\n  ⚠ GENUINE LOSSES DETECTED: {len(genuine_losses)} model×region pairs"
        )
        print(f"  These models are INVOCABLE via Converse but NOT in ON_DEMAND filter!")
        print(f"  Removing pricing union WOULD lose these:\n")
        print(f"    {'Model ID':<50} {'Region':<25} Via")
        print(f"    {'-' * 50} {'-' * 25} {'-' * 20}")
        for model_id, region, via in genuine_losses:
            print(f"    {model_id:<50} {region:<25} {via}")
    else:
        print(f"\n  ✓ NO GENUINE LOSSES FOUND")
        print(f"  None of the pricing-only models are invocable via Converse API.")
        print(
            f"  Removing the pricing union will NOT lose any working on-demand models."
        )

    if access_denied_cases:
        print(
            f"\n  ⚠ NOTE: {len(access_denied_cases)} model×region pairs returned ACCESS_DENIED"
        )
        print(f"  These models MAY be invocable but this account lacks permission.")
        print(f"  Consider testing with broader IAM permissions if concerned.\n")
        for model_id, region, via in access_denied_cases[:20]:
            print(f"    {model_id:<50} {region:<25} {via}")
        if len(access_denied_cases) > 20:
            print(f"    ... and {len(access_denied_cases) - 20} more")

    # Count models NOT tested (if we sampled)
    not_tested = {
        mid
        for mid, info in classified.items()
        if info["category"] in (CATEGORY_PRICING_ONLY, CATEGORY_IN_UNFILTERED)
        and info["excess_regions"]
        and not info["is_embedding"]
        and mid not in converse_results
    }
    if not_tested:
        print(
            f"\n  NOTE: {len(not_tested)} models were NOT tested (exceeded "
            f"--max-models {max_models}). Run with --max-models {len(not_tested) + max_models} "
            f"to test all."
        )

    # Embedding models note
    embedding_skipped = {
        mid
        for mid, info in classified.items()
        if info["category"] in (CATEGORY_PRICING_ONLY, CATEGORY_IN_UNFILTERED)
        and info["excess_regions"]
        and info["is_embedding"]
    }
    if embedding_skipped:
        print(
            f"\n  NOTE: {len(embedding_skipped)} embedding/rerank models were skipped "
            f"(Converse API not applicable)."
        )


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify whether pricing-only models are genuinely invocable",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Dump raw results as JSON after the report.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=DEFAULT_MAX_MODELS,
        help="Max number of models to test via Converse (default: %d). "
        "Top N by excess region count are selected." % DEFAULT_MAX_MODELS,
    )
    args = parser.parse_args()

    start = time.time()

    # Verify credentials
    try:
        sts = boto3.client("sts", config=RETRY_CONFIG)
        identity = sts.get_caller_identity()
        _log(f"AWS Account: {identity['Account']}, Identity: {identity['Arn']}")
    except Exception as exc:
        _log("FATAL: Cannot verify AWS credentials — %s" % exc)
        sys.exit(1)

    # ── Step 1: Collect data ─────────────────────────────────────────
    _log("\n" + "=" * 70)
    _log("STEP 1: Collect Data")
    _log("=" * 70)

    regions = discover_bedrock_regions()
    unfiltered_map, ondemand_map = collect_bedrock_data(regions)
    pricing_data = read_pricing_data()
    pricing_map = extract_pricing_pairs(pricing_data)

    # ── Step 2: Classify pricing models ──────────────────────────────
    _log("\n" + "=" * 70)
    _log("STEP 2: Classify Pricing Models")
    _log("=" * 70)

    classified = classify_pricing_models(pricing_map, ondemand_map, unfiltered_map)

    # Log category counts
    cat_counts = defaultdict(int)  # type: Dict[str, int]
    for info in classified.values():
        cat_counts[info["category"]] += 1
    for cat, count in sorted(cat_counts.items()):
        _log(f"  {cat}: {count} models")

    # ── Step 3: Converse verification ────────────────────────────────
    _log("\n" + "=" * 70)
    _log("STEP 3: Converse API Verification")
    _log("=" * 70)

    converse_results = run_converse_checks(classified, args.max_models)

    # ── Step 4: Report ───────────────────────────────────────────────
    elapsed = time.time() - start
    _log(f"\n[done] Data collection and testing completed in {elapsed:.1f}s")

    print_report(classified, converse_results, args.max_models)

    print(f"\nCompleted in {elapsed:.1f}s")

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

        raw = {
            "classified": _sets_to_lists(classified),
            "converse_results": _sets_to_lists(converse_results),
        }  # type: Dict[str, Any]

        print("\n--- RAW JSON ---")
        print(json.dumps(raw, indent=2, default=str))


if __name__ == "__main__":
    main()
