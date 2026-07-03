#!/usr/bin/env python3
"""
End-to-end test of the 360-eval APO (Advanced Prompt Optimization) implementation.

Drives the real engine path — `benchmarks_run._run_apo_phase` -> `apo_client`
(build record -> create_advanced_prompt_optimization_job -> poll -> download ->
parse -> inject optimized prompts) — against live Bedrock APO.

Defaults (override via flags/env):
  dataset : assets/test-datasets/sample-benchmark-prompts-function-calling-v2.csv
  target  : bedrock/us.amazon.nova-2-lite-v1:0      (Nova Lite 2)
  judge   : bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0  (Sonnet 4.5, LLM-as-Judge)
  mode    : evaluate_both   (keeps original + adds a *_Prompt_Optimized variant)

Requirements:
  - AWS credentials with Bedrock APO + S3 access (e.g. AWS_PROFILE=prod).
  - Account model access to the target + judge models in the region.
  - An S3 bucket for the APO job input/output (S3_BUCKET env or --bucket).

WARNING: this submits a REAL APO job. Wall-clock is typically 20-50 minutes; the
script blocks until the job reaches a terminal state.

Usage:
  AWS_PROFILE=prod AWS_REGION=us-east-1 \
    python scripts/test_apo.py
  python scripts/test_apo.py --bucket 360eval-data-381437929339-us-east-1 --region us-east-1
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_DATASET = PROJECT_ROOT / "assets" / "test-datasets" / "sample-benchmark-prompts-function-calling-v2.csv"
DEFAULT_TARGET = "bedrock/us.amazon.nova-2-lite-v1:0"
DEFAULT_JUDGE = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Function-calling LLM-as-Judge rubric. Uses the APO Responses placeholders
# {prompt}/{prediction}/{gold}; no literal braces so nothing needs escaping.
LLMJ_RUBRIC = (
    "You are grading whether the target model produced the correct tool/function "
    "call(s) for the user's request.\n\n"
    "User request:\n{prompt}\n\n"
    "Model response (candidate tool calls):\n{prediction}\n\n"
    "Reference (golden) tool calls:\n{gold}\n\n"
    "Grade as the fraction of these checks passed:\n"
    "1. The tool/function name(s) match the reference.\n"
    "2. Completeness - every parameter present in the reference is present (order does NOT matter).\n"
    "3. Argument values match the reference (semantically-equivalent values count as correct).\n"
    "4. No extra or hallucinated tool calls / parameters absent from the reference.\n\n"
    "Scoring: 1.0 = all checks pass; 0.0 = wrong tool or no valid call; a fraction in between.\n\n"
    "Output exactly two lines:\n"
    "Score: <float between 0 and 1>\n"
    "Reason: one sentence naming the function and any missing, wrong, or extra parameters."
)


def load_rows(dataset: Path, region: str, limit: int | None):
    """Read the CSV into engine-shaped `raw` rows: {prompt, golden_answer, region}."""
    rows = []
    with dataset.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if "text_prompt" not in cols or "golden_answer" not in cols:
            raise SystemExit(f"Dataset must have text_prompt + golden_answer columns; got {cols}")
        for r in reader:
            rows.append({
                "prompt": r["text_prompt"],
                "golden_answer": r["golden_answer"],
                "region": region,
            })
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--target", default=DEFAULT_TARGET, help="target model id")
    ap.add_argument("--judge", default=DEFAULT_JUDGE, help="LLM-as-Judge model id")
    ap.add_argument("--mode", default="evaluate_both", choices=["evaluate_both", "optimize_only"])
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--bucket", default=os.environ.get("S3_BUCKET", "360eval-data-381437929339-us-east-1"))
    ap.add_argument("--limit", type=int, default=20, help="max dataset rows to load (APO samples <=5 internally)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build + print the APO record (system-prompt extraction + LLMJ record) "
                         "WITHOUT submitting a job — cheap sanity check of record formatting")
    args = ap.parse_args()

    os.environ.setdefault("AWS_REGION", args.region)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("test_apo")

    # Import after sys.path + AWS_REGION are set (apo_client reads AWS_REGION at import).
    from benchmarks_run import _run_apo_phase

    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    raw = load_rows(args.dataset, args.region, args.limit)
    scenarios = [{**r, "model_id": args.target} for r in raw]
    eval_id = f"apo-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = str(PROJECT_ROOT / "outputs" / eval_id)
    os.makedirs(output_dir, exist_ok=True)

    cfg = {
        "prompt_optimization_mode": args.mode,
        "apo_evaluator": "llmj",
        "apo_llmj_rubric": LLMJ_RUBRIC,
        "apo_llmj_judge_model": args.judge,
        "apo_steering_criteria": [],
        "apo_bucket": args.bucket,
        "eval_id": eval_id,
    }

    print("=" * 78)
    print(f"APO end-to-end test")
    print(f"  dataset : {args.dataset.name} ({len(raw)} rows; APO samples <=5)")
    print(f"  target  : {args.target}")
    print(f"  judge   : {args.judge}  (LLM-as-Judge)")
    print(f"  mode    : {args.mode}")
    print(f"  region  : {args.region}   bucket: {args.bucket}")
    print(f"  eval_id : {eval_id}")
    print(f"  output  : {output_dir}")
    print("=" * 78)

    if args.dry_run:
        # Mirror the engine's record-building (extract system prompt -> build LLMJ record)
        # without submitting. Surfaces template/format problems cheaply.
        from utils import extract_system_prompt_hybrid
        from apo_client import build_apo_record_llmj
        from benchmarks_run import _clean_bedrock_model_id

        sample_rows = raw[:5]
        system_prompt, variable_parts = extract_system_prompt_hybrid(
            [r["prompt"] for r in sample_rows], min_len=20,
            fallback_model_id="us.amazon.nova-lite-v1:0", region=args.region,
        )
        if not system_prompt:
            print("DRY-RUN FAIL: could not extract a shared system prompt from the samples.")
            return 1
        apo_sample_rows = [
            {"variable_part": v, "golden": sample_rows[i].get("golden_answer", "")}
            for i, v in enumerate(variable_parts)
        ]
        record = build_apo_record_llmj(
            f"360eval-{eval_id[:24]}", system_prompt, apo_sample_rows,
            rubric=LLMJ_RUBRIC, judge_model_id=_clean_bedrock_model_id(args.judge),
        )
        print("DRY-RUN — APO record that WOULD be submitted:")
        print(f"  version            : {record.get('version')}")
        print(f"  templateId         : {record.get('templateId')}")
        print(f"  promptTemplate     : {len(record.get('promptTemplate',''))} chars; "
              f"ends with '{{{{input}}}}': {record.get('promptTemplate','').rstrip().endswith('{{input}}')}")
        print(f"    template preview : {record.get('promptTemplate','')[:200]!r}")
        print(f"  evaluationSamples  : {len(record.get('evaluationSamples', []))}")
        print(f"  customLLMJConfig   : modelId={record['customLLMJConfig']['customLLMJModelId']}, "
              f"rubric={len(record['customLLMJConfig']['customLLMJPrompt'])} chars")
        print(f"  first sample shape : {json.dumps(record['evaluationSamples'][0])[:200]!r}")
        print("\nDRY-RUN OK (record built; not submitted).")
        return 0

    print("Submitting a REAL APO job — this can take 20-50 minutes. Polling...\n")

    t0 = time.time()
    try:
        optimized = _run_apo_phase(scenarios, raw, cfg, output_dir, eval_id)
    except Exception as e:
        log.exception("APO phase raised")
        print(f"\nFAIL: _run_apo_phase raised: {type(e).__name__}: {e}")
        return 1
    elapsed = int(time.time() - t0)

    # ---- Inspect outcome ----
    print("\n" + "=" * 78)
    print(f"APO phase returned after {elapsed}s")
    print(f"  scenarios in : {len(scenarios)}   scenarios out: {len(optimized)}")
    opt_variants = [s for s in optimized if s.get("prompt_optimization_label") == "Prompt_Optimized"]
    print(f"  *_Prompt_Optimized variants produced: {len(opt_variants)}")

    log_path = Path(output_dir) / "apo" / "apo_optimization_log.json"
    result = "UNKNOWN"
    if log_path.exists():
        apo_log = json.loads(log_path.read_text())
        print(f"\n  evaluator={apo_log.get('evaluator')}  apply_mode={apo_log.get('apply_mode')}")
        for model_id, m in apo_log.get("models", {}).items():
            print(f"\n  model: {model_id}")
            print(f"    status               : {m.get('status')}")
            print(f"    job_arn              : {m.get('job_arn')}")
            print(f"    optimized_present    : {m.get('optimized_template_present')}")
            print(f"    original_score       : {m.get('original_score')}")
            print(f"    optimized_score      : {m.get('optimized_score')}")
            if m.get("error"):
                print(f"    error                : {m.get('error')}")
            tpl = Path(output_dir) / "apo" / f"optimized_template_{model_id.replace('/', '_').replace(':', '_').replace('.', '_')}.txt"
            if tpl.exists():
                preview = tpl.read_text(encoding="utf-8")[:500]
                print(f"    optimized_template   : {tpl}\n      preview: {preview!r}")
        any_ok = any(m.get("optimized_template_present") for m in apo_log.get("models", {}).values())
        result = "PASS" if any_ok else "FAIL (no optimized template produced)"
    else:
        result = "FAIL (no apo_optimization_log.json written — APO likely skipped before submitting)"

    print("\n" + "=" * 78)
    print(f"RESULT: {result}")
    print("=" * 78)
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
