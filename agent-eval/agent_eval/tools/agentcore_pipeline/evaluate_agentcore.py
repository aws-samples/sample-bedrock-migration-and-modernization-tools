#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
AgentCore → Evaluation Bridge

End-to-end pipeline that:
1. Exports traces from AgentCore (via existing 3-stage pipeline)
2. Feeds normalized output into the evaluation framework
3. Produces evaluation artifacts viewable in the dashboard

This bridges the gap between the AgentCore extraction pipeline
and the evaluation framework — the missing integration piece.

Usage:
    # Full pipeline: extract from AgentCore + evaluate
    python -m agent_eval.tools.agentcore_pipeline.evaluate_agentcore \
        --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/my-agent:v1 \
        --judge-config judges.yaml \
        --rubrics rubrics.yaml \
        --output-dir ./output

    # Evaluate previously exported AgentCore traces
    python -m agent_eval.tools.agentcore_pipeline.evaluate_agentcore \
        --agentcore-export-dir ./exports/agentcore_exports/run-abc123/merged \
        --judge-config judges.yaml \
        --rubrics rubrics.yaml \
        --output-dir ./output

    # Export only (skip evaluation)
    python -m agent_eval.tools.agentcore_pipeline.evaluate_agentcore \
        --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/my-agent:v1 \
        --export-only \
        --output-dir ./output
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from typing import Optional


def find_normalized_runs(directory: Path) -> list[Path]:
    """Find all normalized run JSON files in a directory."""
    patterns = ["normalized_run.*.json", "*.json"]
    for pattern in patterns:
        files = sorted(directory.glob(pattern))
        if files:
            return files
    return []


def run_agentcore_export(
    arn: Optional[str] = None,
    days: int = 7,
    region: str = "us-east-1",
    output_dir: str = "./agentcore_export",
    extra_args: list[str] = None,
) -> tuple[int, Path]:
    """
    Run the AgentCore extraction pipeline.

    Returns (exit_code, merged_dir_path).
    """
    import importlib

    if arn:
        mod = importlib.import_module("agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn")
        argv = [
            "--agent-runtime-arn", arn,
            "--days", str(days),
            "--region", region,
            "--output-root", output_dir,
        ]
    else:
        mod = importlib.import_module("agent_eval.tools.agentcore_pipeline.export_agentcore_pipeline")
        argv = [
            "export-agentcore",
            "--days", str(days),
            "--region", region,
            "--output-root", output_dir,
        ]

    if extra_args:
        argv.extend(extra_args)

    print(f"Running AgentCore export: {' '.join(argv)}")
    saved_argv = sys.argv
    returncode = 0
    try:
        sys.argv = ["script"] + argv
        mod.main()
    except SystemExit as e:
        returncode = e.code if e.code else 0
    except Exception as e:
        print(f"AgentCore export failed: {e}")
        returncode = 1
    finally:
        sys.argv = saved_argv

    # Find the merged output directory
    export_root = Path(output_dir) / "agentcore_exports"
    merged_dirs = sorted(export_root.glob("*/merged"), key=lambda p: p.stat().st_mtime, reverse=True)
    merged_dir = merged_dirs[0] if merged_dirs else export_root

    return returncode, merged_dir


def run_evaluation(
    input_path: str,
    judge_config_path: str,
    output_dir: str,
    rubrics_path: Optional[str] = None,
    verbose: bool = False,
) -> int:
    """Run the evaluation pipeline on a single normalized run."""
    from agent_eval.cli import main as eval_main

    argv = [
        "--input", input_path,
        "--input-is-normalized",
        "--judge-config", judge_config_path,
        "--output-dir", output_dir,
    ]
    if rubrics_path:
        argv.extend(["--rubrics", rubrics_path])
    if verbose:
        argv.append("--verbose")

    try:
        return eval_main(argv) or 0
    except SystemExit as e:
        return e.code if e.code else 0


def main():
    parser = argparse.ArgumentParser(
        description="AgentCore → Evaluation Bridge: extract traces and evaluate them"
    )

    # Source selection (one of these required)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--agent-runtime-arn",
        help="AgentCore runtime ARN to extract traces from",
    )
    source.add_argument(
        "--agentcore-export-dir",
        help="Path to previously exported AgentCore merged directory",
    )

    # Export options
    parser.add_argument("--days", type=int, default=7, help="Days of traces to export (default: 7)")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--export-only", action="store_true", help="Export traces without evaluating")

    # Evaluation options
    parser.add_argument("--judge-config", help="Path to judges.yaml (required unless --export-only)")
    parser.add_argument("--rubrics", help="Path to rubrics.yaml (optional, merges with defaults)")
    parser.add_argument("--output-dir", default="./agentcore_eval_output", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not args.export_only and not args.judge_config:
        parser.error("--judge-config is required unless --export-only is set")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Get normalized runs
    if args.agent_runtime_arn:
        print("=" * 60)
        print("STEP 1: Extracting traces from AgentCore")
        print("=" * 60)

        exit_code, merged_dir = run_agentcore_export(
            arn=args.agent_runtime_arn,
            days=args.days,
            region=args.region,
            output_dir=str(output_path / "export"),
        )

        if exit_code != 0:
            print(f"\n✗ AgentCore export failed (exit code: {exit_code})")
            return exit_code

        print(f"✓ Export complete: {merged_dir}")
    else:
        merged_dir = Path(args.agentcore_export_dir)
        if not merged_dir.exists():
            print(f"✗ Export directory not found: {merged_dir}")
            return 1
        print(f"Using existing export: {merged_dir}")

    # Find normalized runs
    normalized_files = find_normalized_runs(merged_dir)
    if not normalized_files:
        print(f"✗ No normalized run files found in {merged_dir}")
        return 1

    print(f"Found {len(normalized_files)} normalized run(s)")

    if args.export_only:
        print("\n✓ Export complete (--export-only mode)")
        for f in normalized_files:
            print(f"  {f}")
        return 0

    # Step 2: Evaluate each normalized run
    print("\n" + "=" * 60)
    print("STEP 2: Evaluating traces")
    print("=" * 60)

    results = []
    for i, nf in enumerate(normalized_files):
        run_output = output_path / "evaluations" / nf.stem
        run_output.mkdir(parents=True, exist_ok=True)
        print(f"\n[{i+1}/{len(normalized_files)}] Evaluating: {nf.name}")

        exit_code = run_evaluation(
            input_path=str(nf),
            judge_config_path=args.judge_config,
            output_dir=str(run_output),
            rubrics_path=args.rubrics,
            verbose=args.verbose,
        )

        results.append({
            "file": str(nf),
            "output_dir": str(run_output),
            "exit_code": exit_code,
            "success": exit_code == 0,
        })

        status = "✓" if exit_code == 0 else "✗"
        print(f"  {status} Output: {run_output}")

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    succeeded = sum(1 for r in results if r["success"])
    failed = len(results) - succeeded
    print(f"  Evaluated: {len(results)} run(s)")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")
    print(f"  Output:    {output_path}")

    # Write summary
    summary_path = output_path / "agentcore_eval_summary.json"
    summary_path.write_text(json.dumps({
        "source": args.agent_runtime_arn or args.agentcore_export_dir,
        "total_runs": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }, indent=2))
    print(f"  Summary:   {summary_path}")

    if args.verbose:
        print(f"\nView results in dashboard:")
        print(f"  streamlit run ui/app.py")
        print(f"  → Point to: {output_path / 'evaluations'}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
