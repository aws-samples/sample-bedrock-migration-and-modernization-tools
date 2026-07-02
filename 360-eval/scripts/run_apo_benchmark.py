#!/usr/bin/env python3
"""
Benchmark ORIGINAL vs APO-OPTIMIZED prompts side-by-side and generate the HTML report.

Reuses the optimized prompts from a prior `test_apo.py` run (its
apo/optimized_dataset_*.csv), so it skips the ~22-min APO job and only runs the
benchmark + scoring + report. Each dataset row is run twice on the target model:
once with the original prompt, once with the optimized prompt (model_id suffixed
`_Prompt_Optimized`) — which is exactly how `evaluate_both` surfaces them in a report.

Requires AWS creds with Bedrock access (target + judge models invoked live, SigV4).

Usage:
  AWS_PROFILE=prod AWS_REGION=us-east-1 python scripts/run_apo_benchmark.py
  python scripts/run_apo_benchmark.py --apo-dir outputs/apo-test-20260615-003003 --limit 5
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TARGET = "bedrock/us.amazon.nova-2-lite-v1:0"
JUDGE = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def find_optimized_dataset(apo_dir: Path) -> str:
    cands = glob.glob(str(apo_dir / "apo" / "optimized_dataset_*.csv"))
    if not cands:
        raise SystemExit(f"No apo/optimized_dataset_*.csv under {apo_dir}")
    return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apo-dir", type=Path, help="prior test_apo output dir (default: latest outputs/apo-test-*)")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--judge", default=JUDGE)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--limit", type=int, default=5, help="dataset rows (each run twice: original + optimized)")
    args = ap.parse_args()
    os.environ.setdefault("AWS_REGION", args.region)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from benchmarks_run import execute_benchmark
    from visualization.html_report import create_html_report

    apo_dir = args.apo_dir
    if not apo_dir:
        dirs = sorted(glob.glob(str(PROJECT_ROOT / "outputs" / "apo-test-*")))
        if not dirs:
            raise SystemExit("No outputs/apo-test-* dir found; run test_apo.py first or pass --apo-dir")
        apo_dir = Path(dirs[-1])
    ds = find_optimized_dataset(apo_dir)
    rows = list(csv.DictReader(open(ds, encoding="utf-8")))[: args.limit]
    print(f"Reusing optimized prompts from: {ds}  ({len(rows)} rows)")

    # Nova Lite 2 per-1M token costs (from the catalog).
    IN_COST, OUT_COST = 0.3, 2.5

    def scn(prompt, golden, label=None):
        s = {
            "region": args.region,
            "prompt": prompt,
            "task_types": "function-calling",
            "task_criteria": "model emits the correct tool/function call",
            "golden_answer": golden,
            "configured_output_tokens_for_request": 512,
            "model_id": args.target,
            "input_token_cost": IN_COST,
            "output_token_cost": OUT_COST,
            "TEMPERATURE": 0.2,
            "target_rpm": None,
        }
        if label:
            s["prompt_optimization_label"] = label
        return s

    scenarios = []
    for r in rows:
        golden = r.get("golden_answer", "")
        scenarios.append(scn(r["original_prompt"], golden, None))
        scenarios.append(scn(r["optimized_prompt"], golden, "Prompt_Optimized"))

    cfg = {
        "invocations_per_scenario": 1,
        "experiment_counts": 1,
        "parallel_calls": 4,
        "sleep_between_invocations": 0,
        "experiment_wait_time": 0,
        "user_defined_metrics": "",
        "yard_stick": 3,
        "judge_models": [{
            "model_id": args.judge, "region": args.region,
            "input_token_cost": 3.0, "output_token_cost": 15.0,
        }],
        "eval_id": "apo-bench-" + datetime.now().strftime("%Y%m%d-%H%M%S"),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs" / ("apo-bench-" + ts)
    unprocessed_dir = out_dir / "unprocessed"
    os.makedirs(unprocessed_dir, exist_ok=True)

    print(f"Benchmarking {len(scenarios)} scenarios ({len(rows)} original + {len(rows)} optimized) "
          f"on {args.target}, judged by {args.judge} ...")
    t0 = time.time()
    results, _, _ = execute_benchmark(scenarios, cfg, str(unprocessed_dir),
                                      yard_stick=3, latency_only_mode=False, stream_evaluation=True)
    print(f"benchmark done in {int(time.time()-t0)}s; {len(results)} result records")
    if not results:
        print("FAIL: no result records")
        return 1

    df = pd.DataFrame(results)
    df["run_count"] = 1
    df["timestamp"] = pd.Timestamp.now()
    df["run_start_time"] = ts
    out_csv = out_dir / f"invocations_1_{ts}_{uuid.uuid4().hex[:12]}_apo-bench.csv"
    df.to_csv(out_csv, index=False)

    # ---- Quick side-by-side summary (from in-memory records) ----
    from collections import defaultdict
    agg = defaultdict(lambda: {"n": 0, "pass": 0, "scores": []})
    for rec in results:
        mid = rec.get("model_id", "?")
        perf = rec.get("performance_metrics") or {}
        a = agg[mid]
        a["n"] += 1
        if perf.get("judge_success") is True:
            a["pass"] += 1
        for k, v in (perf.get("judge_scores") or {}).items():
            if isinstance(v, (int, float)):
                a["scores"].append(v)

    print("\n" + "=" * 72)
    print("ORIGINAL vs OPTIMIZED — Nova Lite 2 (judge: Sonnet 4.5)")
    print("=" * 72)
    for mid in sorted(agg):
        a = agg[mid]
        pass_rate = 100.0 * a["pass"] / a["n"] if a["n"] else 0
        avg_score = sum(a["scores"]) / len(a["scores"]) if a["scores"] else float("nan")
        tag = "OPTIMIZED" if mid.endswith("_Prompt_Optimized") else "ORIGINAL "
        print(f"  [{tag}] {mid}")
        print(f"            n={a['n']}  pass_rate={pass_rate:.0f}%  avg_judge_score={avg_score:.2f}")

    report = create_html_report(output_dir=out_dir, timestamp=ts)
    print("\nresults CSV :", out_csv)
    print("HTML report :", report)
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
