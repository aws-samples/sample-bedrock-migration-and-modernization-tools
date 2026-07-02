#!/usr/bin/env python3
"""
Build config/models_profiles.jsonl (anchor-merge strategy).

Why this exists
---------------
The other generators don't cover Mantle (OpenAI-compatible) models such as
``openai.gpt-5.5`` / ``openai.gpt-5.4``: those are NOT returned by
``bedrock:ListFoundationModels`` or the scraped pricing CSV, and the AWS Price
List API publishes them inconsistently (units differ between models, and brand-new
models lag — e.g. gpt-5.x has GovCloud rows but no commercial rows yet).

Strategy (additive, anchor = golden truth)
------------------------------------------
1. Load the existing ``models_profiles.jsonl`` as the GOLDEN ANCHOR. Every anchor
   line is kept **verbatim** (trusted model_id / region / prices / tiers) — this
   script never rewrites or re-prices an existing entry.
2. Discover currently-available Mantle models per region via the Mantle
   ``/v1/models`` endpoint.
3. For any discovered model NOT already represented in the anchor, build a profile
   priced from the AWS Price List API (unit-normalized -> per-1M, per region, per
   tier). Where the Price List has no row yet, fall back to MANUAL_PRICE_OVERRIDES.
4. Append the new entries and write the merged catalog.

Run locally (needs AWS creds with bedrock + pricing access), then push the file as
a separate job (e.g. ``aws s3 cp config/models_profiles.jsonl s3://$S3_BUCKET/config/``).

Usage
-----
    python scripts/build_models_profiles.py                       # write in place
    python scripts/build_models_profiles.py --dry-run             # report only
    python scripts/build_models_profiles.py --output /tmp/new.jsonl
    python scripts/build_models_profiles.py --regions us-east-1,us-east-2,us-west-2
    python scripts/build_models_profiles.py --include-dated       # keep dated snapshots
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

# Make src/ importable so we can reuse the Mantle helpers.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from mantle_client import list_mantle_models, fetch_mantle_pricing  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_models_profiles")

ANCHOR_DEFAULT = PROJECT_ROOT / "config" / "models_profiles.jsonl"

# Regions to probe for currently-available Mantle models. Extend as availability
# grows; the script silently skips a region where Mantle isn't reachable.
DEFAULT_REGIONS = ["us-east-1", "us-east-2", "us-west-2"]

# Price List tier name -> catalog tier name (anchor uses default/flex/priority).
# 'batch' is intentionally dropped to match the existing catalog shape.
PRICE_TIER_MAP = {"standard": "default", "flex": "flex", "priority": "priority"}
KEEP_TIERS = ["default", "flex", "priority"]

# Commercial $/1M (input, output). Source: AWS News blog "Get started with OpenAI
# GPT-5.5, GPT-5.4 models, and Codex on Amazon Bedrock" (Bedrock rates match OpenAI
# direct rates). Used ONLY when the Price List API has no row for (model, region)
# yet — the Price List takes precedence once AWS publishes commercial rows.
MANUAL_PRICE_OVERRIDES = {
    "openai.gpt-5.5": {"input": 5.0, "cached_input": 0.5, "output": 30.0},
    "openai.gpt-5.4": {"input": 2.5, "cached_input": 0.25, "output": 15.0},
}

# Offline seed: Mantle availability captured from LIVE queries during development
# (2026-06, regions where /v1/models listed each model). Used by --offline so the
# catalog can be regenerated without AWS access. A live run supersedes this and may
# find additional regions/models. Prices for these come from MANUAL_PRICE_OVERRIDES
# (the Price List API has no commercial rows for them yet).
OFFLINE_SEED_AVAILABILITY = {
    "openai.gpt-5.5": ["us-east-1", "us-east-2"],
    "openai.gpt-5.4": ["us-east-1", "us-east-2", "us-west-2"],
}

# Dated snapshot aliases like ``openai.gpt-5.5-2026-04-23`` — skipped by default
# (the undated id is the rolling "latest" pointer the catalog should list).
DATED_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")

# Cross-region / partition prefixes to strip when comparing model ids for dedup.
_PREFIX_RE = re.compile(r"^(global|apac|us-gov|us|eu|ap|ca|sa|me|af|il|mx)\.")
_VERSION_RE = re.compile(r"(-\d+)?:\d+$")  # trailing "-1:0" or ":0"


def norm_model(model_id: str) -> str:
    """Normalize a model id for cross-source dedup.

    Strips the ``bedrock/``(+``converse/``) routing prefix, any cross-region
    prefix (us./eu./global./...), and a trailing version suffix (``-1:0`` / ``:0``).
    e.g. ``bedrock/openai.gpt-oss-120b-1:0`` and Mantle ``openai.gpt-oss-120b``
    both normalize to ``openai.gpt-oss-120b``.
    """
    s = model_id.strip().lower()
    for p in ("bedrock/converse/", "bedrock/", "converse/"):
        if s.startswith(p):
            s = s[len(p):]
            break
    s = _PREFIX_RE.sub("", s)
    s = _VERSION_RE.sub("", s)
    return s


def load_anchor(path: Path) -> tuple[list[str], set[tuple[str, str]]]:
    """Return (verbatim_lines, index) where index = {(norm_model, region)}."""
    lines: list[str] = []
    index: set[tuple[str, str]] = set()
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        lines.append(raw)
        try:
            d = json.loads(raw)
            index.add((norm_model(d["model_id"]), d.get("region", "")))
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("anchor: could not index line (%s): %s", e, raw[:80])
    return lines, index


def build_price_index() -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Return price_idx[model][region][tier][io] = price_per_1m (Price List API)."""
    rows = fetch_mantle_pricing()  # already unit-normalized to per-1M
    idx: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        tier = PRICE_TIER_MAP.get(r["tier"])
        if tier is None or r["price_per_1m"] is None:
            continue  # drop batch / missing prices
        idx[r["model"]][r["region"]][tier][r["io"]] = r["price_per_1m"]
    return idx


def discover_mantle(regions: list[str], include_dated: bool) -> dict[str, set[str]]:
    """Return {mantle_model_id: {region, ...}} of currently-available models."""
    found: dict[str, set[str]] = defaultdict(set)
    for region in regions:
        try:
            models = list_mantle_models(region)
        except Exception as e:  # noqa: BLE001 - region may not have Mantle
            log.warning("Mantle not reachable in %s: %s", region, e)
            continue
        for m in models:
            mid = m["model_id"]
            if not include_dated and DATED_RE.search(mid):
                continue
            found[mid].add(region)
        log.info("  %s: %d Mantle models", region, len(models))
    return found


def build_new_entry(mid: str, region: str, price_idx: dict) -> dict | None:
    """Build a catalog entry for a Mantle model+region, or None if unpriceable."""
    tiers = dict(price_idx.get(mid, {}).get(region, {}))  # {tier: {io: price}}
    # Fall back to manual override only when the Price List has nothing.
    if not tiers and mid in MANUAL_PRICE_OVERRIDES:
        tiers = {"default": dict(MANUAL_PRICE_OVERRIDES[mid])}

    default = tiers.get("default", {})
    if "input" not in default or "output" not in default:
        return None  # no usable default price -> skip (caller logs)

    service_tiers = [
        t for t in KEEP_TIERS
        if t in tiers and "input" in tiers[t] and "output" in tiers[t]
    ]
    entry = {
        "model_id": f"bedrock/{mid}",
        "region": region,
        "input_token_cost": default["input"],
        "output_token_cost": default["output"],
        "service_tiers": service_tiers or ["default"],
        "tier_pricing": {t: tiers[t] for t in (service_tiers or ["default"])},
        # Mantle marker: the engine routes these via the OpenAI-compatible Responses
        # API (bedrock-mantle.<region>.api.aws/openai/v1) with the BEDROCK_MANTLE_API_KEY,
        # NOT Converse. Threaded into scenarios so benchmark()/run_inference() pick the
        # right path. mantle_region is the region whose Mantle endpoint to call.
        "endpoint": "bedrock_mantle",
        "mantle_region": region,
    }
    # Cached-input pricing isn't in the legacy schema; include it when known so it's
    # available to consumers that support prompt caching (ignored by ones that don't).
    if "cached_input" in default:
        entry["cached_input_token_cost"] = default["cached_input"]
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor", type=Path, default=ANCHOR_DEFAULT,
                    help="existing golden models_profiles.jsonl (kept verbatim)")
    ap.add_argument("--output", type=Path, default=ANCHOR_DEFAULT,
                    help="where to write the merged catalog (default: in place)")
    ap.add_argument("--regions", default=",".join(DEFAULT_REGIONS),
                    help="comma-separated regions to probe for Mantle models")
    ap.add_argument("--include-dated", action="store_true",
                    help="also include dated snapshot ids (e.g. gpt-5.5-2026-04-23)")
    ap.add_argument("--offline", action="store_true",
                    help="no AWS calls: use the embedded OFFLINE_SEED_AVAILABILITY + "
                         "MANUAL_PRICE_OVERRIDES instead of live Mantle/Price List queries")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]

    if not args.anchor.exists():
        log.error("Anchor not found: %s", args.anchor)
        return 2

    log.info("Loading anchor: %s", args.anchor)
    anchor_lines, anchor_idx = load_anchor(args.anchor)
    log.info("  %d anchor entries (kept verbatim)", len(anchor_lines))

    if args.offline:
        log.info("OFFLINE mode: using embedded seed + manual overrides (no AWS calls)")
        price_idx = {}  # forces build_new_entry to use MANUAL_PRICE_OVERRIDES
        discovered = {mid: set(regs) for mid, regs in OFFLINE_SEED_AVAILABILITY.items()}
    else:
        log.info("Fetching Mantle pricing from Price List API (this scans all Bedrock products)...")
        price_idx = build_price_index()
        log.info("  priced %d Mantle models from Price List API", len(price_idx))

        log.info("Discovering available Mantle models in: %s", ", ".join(regions))
        discovered = discover_mantle(regions, args.include_dated)

    new_entries: list[dict] = []
    skipped_priced: list[tuple[str, str]] = []
    already: set[str] = set()
    for mid in sorted(discovered):
        for region in sorted(discovered[mid]):
            if (norm_model(f"bedrock/{mid}"), region) in anchor_idx:
                already.add(mid)
                continue
            entry = build_new_entry(mid, region, price_idx)
            if entry is None:
                skipped_priced.append((mid, region))
                continue
            new_entries.append(entry)

    # ---- Summary ----
    log.info("\n=== Summary ===")
    log.info("anchor kept           : %d", len(anchor_lines))
    log.info("new entries added     : %d", len(new_entries))
    if already:
        log.info("already in anchor     : %d models (%s)",
                 len(already), ", ".join(sorted(already)))
    if new_entries:
        added_models = sorted({e["model_id"] for e in new_entries})
        log.info("new models            : %s", ", ".join(added_models))
        for e in new_entries:
            src = "price-list" if (e["model_id"].replace("bedrock/", "") in price_idx
                                   and e["region"] in price_idx[e["model_id"].replace("bedrock/", "")]) else "override"
            log.info("  + %-26s %-12s in=$%s/M out=$%s/M tiers=%s [%s]",
                     e["model_id"], e["region"], e["input_token_cost"],
                     e["output_token_cost"], ",".join(e["service_tiers"]), src)
    if skipped_priced:
        log.warning("skipped (no price)    : %d", len(skipped_priced))
        for mid, region in skipped_priced:
            log.warning("  - %-26s %-12s (no Price List row and no override)", mid, region)

    if args.dry_run:
        log.info("\n[dry-run] not writing. Would write %d total entries to %s",
                 len(anchor_lines) + len(new_entries), args.output)
        return 0

    out_lines = anchor_lines + [json.dumps(e) for e in new_entries]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_lines) + "\n")
    log.info("\nWrote %d entries to %s", len(out_lines), args.output)
    log.info("Next: push it -> aws s3 cp %s s3://$S3_BUCKET/config/models_profiles.jsonl", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
