#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
AgentCore Evidence Reconstruction Pipeline Orchestrator

This script orchestrates the 3-stage AgentCore evidence reconstruction pipeline.
CloudWatch is just the storage backend - this pipeline reconstructs the evidence.

CLI Contract:
  Command: export-agentcore
  Exit Codes:
    0 - Success (all quality gates passed)
    2 - Validation error (bad args/time window)
    3 - AWS access error (credentials/permissions)
    4 - Quality gate failed (missing user_query/traces beyond threshold, no evidence found)
    5 - Runtime error (timeout, stage failure, unexpected exception)

Output Structure:
  <output-root>/agentcore_exports/<run-id>/
    raw/                    - Intermediate outputs from scripts 1-3
    merged/                 - Final normalized_run.json
    reports/                - summary.json, summary.txt
    manifest.json           - Reproducibility metadata

Usage:
    python -m agent_eval.tools.agentcore_pipeline.export_agentcore_pipeline export-agentcore \\
        --days 30 --region us-west-2 --output-root ./outputs
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Exit codes
EXIT_SUCCESS = 0
EXIT_VALIDATION_ERROR = 2
EXIT_AWS_ERROR = 3
EXIT_QUALITY_GATE_FAILED = 4
EXIT_RUNTIME_ERROR = 5


def setup_logging(log_level: str, quiet: bool) -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger("export_agentcore")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, log_level.upper()))
    handler = logging.StreamHandler(sys.stdout)
    if quiet:
        handler.setLevel(logging.WARNING)
    else:
        handler.setLevel(getattr(logging, log_level.upper()))
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def generate_run_id(
    start_time: datetime,
    end_time: datetime,
    region: Optional[str],
    profile: Optional[str],
    app_inputs: Dict[str, Any],
    trace_inputs: Dict[str, Any]
) -> str:
    """Generate deterministic run ID from inputs (discovery removed)."""
    hash_input = json.dumps({
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "region": region or "default",
        "profile": profile or "default",
        "app": app_inputs,
        "trace": trace_inputs
    }, sort_keys=True)
    hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
    date_str = start_time.strftime("%Y%m%d")
    end_str = end_time.strftime("%Y%m%d")
    return f"ac_{date_str}_{end_str}_{hash_digest}"


def validate_time_window(
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    days: Optional[int]
) -> Tuple[datetime, datetime]:
    """Validate and resolve time window."""
    if start_time and end_time:
        if start_time >= end_time:
            raise ValueError(f"start_time must be before end_time: {start_time} >= {end_time}")
        return start_time, end_time
    elif days:
        if days <= 0:
            raise ValueError(f"days must be positive: {days}")
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)
        return start_dt, end_dt
    else:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=7)
        return start_dt, end_dt


def setup_output_structure(
    output_root: Path,
    run_id: str,
    overwrite: bool
) -> Dict[str, Path]:
    """Create output directory structure."""
    base_dir = output_root / "agentcore_exports" / run_id
    if base_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists: {base_dir}\n"
                f"Use --overwrite to allow overwriting, or specify a different --run-id"
            )
        else:
            import shutil
            shutil.rmtree(base_dir)
    paths = {
        "base": base_dir,
        "raw": base_dir / "raw",
        "merged": base_dir / "merged",
        "reports": base_dir / "reports"
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_script_command(
    script_name: str,
    script_dir: Path,
    args: Dict[str, Any]
) -> List[str]:
    """Build command for running a pipeline script."""
    script_path = script_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    cmd = [sys.executable, str(script_path)]
    for key, value in args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif isinstance(value, list):
            for item in value:
                cmd.extend([f"--{key}", str(item)])
        else:
            cmd.extend([f"--{key}", str(value)])
    return cmd


def run_script(
    script_name: str,
    stage_num: int,
    stage_name: str,
    cmd: List[str],
    cwd: Optional[Path],
    logger: logging.Logger,
    print_commands: bool,
    timeout: Optional[int] = 3600
) -> Dict[str, Any]:
    """Run a pipeline script and capture results."""
    import importlib
    import io
    from contextlib import redirect_stdout, redirect_stderr

    logger.info(f"Stage {stage_num}: {stage_name}")
    if print_commands:
        logger.info(f"Command: {' '.join(cmd)}")
        if cwd:
            logger.info(f"CWD: {cwd}")

    # cmd is [sys.executable, script_path, --arg1, val1, ...]
    # Extract script path and argv from the command list
    script_path = cmd[1] if len(cmd) > 1 else ""
    argv = cmd[2:]

    # Derive module name from script path for importlib
    module_name = Path(script_path).stem
    package = "agent_eval.tools.agentcore_pipeline"
    full_module = f"{package}.{module_name}"

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    saved_argv = sys.argv
    saved_cwd = os.getcwd()

    try:
        if cwd:
            os.chdir(str(cwd))
        sys.argv = ["script"] + argv
        mod = importlib.import_module(full_module)
        # Reload in case it was already imported with different state
        importlib.reload(mod)
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            mod.main()
        return {
            "status": "success",
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "returncode": 0
        }
    except SystemExit as e:
        rc = e.code if e.code else 0
        stdout_text = stdout_buf.getvalue()
        stderr_text = stderr_buf.getvalue()
        if rc != 0:
            logger.error(f"Stage {stage_num} failed with exit code {rc}")
            if stdout_text:
                logger.error(f"Stdout: {stdout_text[:500]}")
            if stderr_text:
                logger.error(f"Stderr: {stderr_text[:500]}")
            combined_output = (stdout_text + stderr_text).lower()
            is_aws_error = any(marker in combined_output for marker in [
                "nocredentialserror", "partialcredentialserror",
                "expiredtokenexception", "invalidclienttokenid",
                "signaturedoesnotmatch", "accessdenied",
                "unrecognizedclientexception", "accessdeniedexception",
                "notauthorizedexception", "requestexpired",
                "tokenrefreshrequired", "expiredtoken"
            ])
            return {
                "status": "failed",
                "stdout": stdout_text,
                "stderr": stderr_text,
                "returncode": rc,
                "is_aws_error": is_aws_error
            }
        return {
            "status": "success",
            "stdout": stdout_text,
            "stderr": stderr_text,
            "returncode": 0
        }
    except Exception as e:
        logger.error(f"Stage {stage_num} failed: {e}")
        return {
            "status": "failed",
            "stdout": stdout_buf.getvalue(),
            "stderr": str(e),
            "returncode": 1,
            "is_aws_error": False
        }
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)


def compute_coverage_stats(
    raw_dir: Path,
    merged_dir: Path,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Compute coverage and join statistics from pipeline outputs."""
    stats = {
        "turns_indexed": 0,
        "traces_exported": 0,
        "merged_turns": 0,
        "joins": {
            "by_trace_id": 0,
            "by_request_id": 0,
            "by_session_time": 0,
            "unmatched": 0
        },
        "user_query": {
            "turns_with_user_query": 0,
            "turns_missing_user_query": 0
        },
        "traces": {
            "turns_with_steps": 0,
            "turns_missing_steps": 0,
            "turns_missing_trace_artifacts": 0
        }
    }
    turn_files = list(raw_dir.glob("*turn*.json"))
    for turn_file in turn_files:
        try:
            with open(turn_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    stats["turns_indexed"] += len(data)
                    for turn in data:
                        if turn.get("user_query"):
                            stats["user_query"]["turns_with_user_query"] += 1
                        else:
                            stats["user_query"]["turns_missing_user_query"] += 1
                elif isinstance(data, dict) and "turns" in data:
                    turns = data["turns"]
                    stats["turns_indexed"] += len(turns)
                    for turn in turns:
                        if turn.get("user_query"):
                            stats["user_query"]["turns_with_user_query"] += 1
                        else:
                            stats["user_query"]["turns_missing_user_query"] += 1
        except Exception as e:
            logger.warning(f"Could not read {turn_file.name}: {e}")
    trace_files = list(raw_dir.glob("*trace*.json"))
    for trace_file in trace_files:
        try:
            with open(trace_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and ("trace_id" in item or "spans" in item):
                            stats["traces_exported"] += 1
                elif isinstance(data, dict):
                    if "traces" in data:
                        stats["traces_exported"] += len(data["traces"])
                    elif "trace_id" in data or "spans" in data:
                        stats["traces_exported"] += 1
        except Exception as e:
            logger.warning(f"Could not read {trace_file.name}: {e}")
    normalized_files = list(merged_dir.glob("*.json"))
    for norm_file in normalized_files:
        try:
            with open(norm_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    turns = data.get("turns", [])
                    stats["merged_turns"] += len(turns)
                    for turn in turns:
                        steps = (turn.get("steps") or 
                                turn.get("steps_runtime") or 
                                turn.get("steps_runtime_xray") or [])
                        if steps and len(steps) > 0:
                            stats["traces"]["turns_with_steps"] += 1
                        else:
                            stats["traces"]["turns_missing_steps"] += 1
                        has_trace_artifacts = (
                            (steps and len(steps) > 0) or
                            turn.get("total_latency_ms") is not None or
                            turn.get("trace_joined") is True
                        )
                        if not has_trace_artifacts:
                            stats["traces"]["turns_missing_trace_artifacts"] += 1
                        turn_meta = turn.get("metadata", {})
                        trace_joined = turn.get("trace_joined")
                        join_reason = turn.get("join_reason") or turn_meta.get("join_reason")
                        if trace_joined is False or (join_reason and "fail" in str(join_reason).lower()):
                            stats["joins"]["unmatched"] += 1
                        elif turn.get("trace_id") or turn_meta.get("trace_id"):
                            if has_trace_artifacts:
                                stats["joins"]["by_trace_id"] += 1
                            else:
                                stats["joins"]["unmatched"] += 1
                        elif turn.get("request_id") or turn_meta.get("request_id"):
                            if has_trace_artifacts:
                                stats["joins"]["by_request_id"] += 1
                            else:
                                stats["joins"]["unmatched"] += 1
                        elif turn.get("session_id") or turn_meta.get("session_id"):
                            if has_trace_artifacts:
                                stats["joins"]["by_session_time"] += 1
                            else:
                                stats["joins"]["unmatched"] += 1
                        else:
                            stats["joins"]["unmatched"] += 1
        except Exception as e:
            logger.warning(f"Could not read {norm_file.name}: {e}")
    return stats


def check_quality_gates(
    stats: Dict[str, Any],
    max_missing_user_query: int,
    max_missing_trace_artifacts: int,
    fail_on_missing_user_query: bool,
    fail_on_missing_traces: bool,
    logger: logging.Logger
) -> Tuple[bool, List[str]]:
    """Check quality gates and return (passed, failures)."""
    failures = []
    missing_user_query = stats["user_query"]["turns_missing_user_query"]
    missing_trace_artifacts = stats["traces"]["turns_missing_trace_artifacts"]
    if fail_on_missing_user_query and missing_user_query > max_missing_user_query:
        failures.append(
            f"Missing user_query: {missing_user_query} turns "
            f"(threshold: {max_missing_user_query})"
        )
    if fail_on_missing_traces and missing_trace_artifacts > max_missing_trace_artifacts:
        failures.append(
            f"Missing trace artifacts: {missing_trace_artifacts} turns "
            f"(threshold: {max_missing_trace_artifacts})"
        )
    passed = len(failures) == 0
    if not passed:
        logger.error("Quality gates failed:")
        for failure in failures:
            logger.error(f"  - {failure}")
    return passed, failures


def run_stage1(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    script_dir: Path,
    logger: logging.Logger
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Run Stage 1 and read discovery.json if it exists.

    Args:
        args: Parsed command-line arguments
        paths: Output directory paths
        script_dir: Directory containing pipeline scripts
        logger: Logger instance

    Returns:
        Tuple of (stage1_result, discovery_result)
        - stage1_result: Script execution result dict
        - discovery_result: Discovery result from discovery.json (or None)
    """
    logger.info("="*60)
    logger.info("Stage 1: Extract turns from CloudWatch app logs")
    logger.info("="*60)

    # Build Script 1 arguments
    script1_args = {
        "output-dir": str(paths["raw"]),
        "region": args.region,
        "profile": args.profile,
        "days": args.days if not args.start_time else None,
        "start-time": args.start_time,
        "end-time": args.end_time
    }

    # Track if discovery was requested
    discovery_requested = False

    # Add log group arguments
    if args.app_log_group:
        script1_args["log-group"] = args.app_log_group
    elif args.app_log_group_prefix:
        script1_args["log-group-prefix"] = args.app_log_group_prefix
        discovery_requested = True
    elif args.app_log_group_pattern:
        script1_args["log-group-pattern"] = args.app_log_group_pattern
        discovery_requested = True

    if args.max_turns:
        script1_args["max-turns"] = args.max_turns

    # Run Script 1
    script1_cmd = build_script_command("01_export_turns_from_app_logs.py", script_dir, script1_args)
    stage1_result = run_script(
        "01_export_turns_from_app_logs.py",
        1,
        "Extract turns from app logs",
        script1_cmd,
        script_dir,
        logger,
        args.print_commands
    )

    # If Stage 1 failed, return early with no discovery result
    if stage1_result["status"] != "success":
        return stage1_result, None

    # Read discovery.json if it exists (from raw/ directory)
    discovery_path = paths["raw"] / "discovery.json"
    discovery_result = None

    if discovery_path.exists():
        try:
            with open(discovery_path, 'r') as f:
                raw_discovery = json.load(f)

            # Validate discovery.json schema
            required_keys = [
                "search_criteria",
                "matched_groups",
                "selected_groups",
                "selection_reason",
                "total_matched",
                "total_selected"
            ]
            missing_keys = [k for k in required_keys if k not in raw_discovery]

            if missing_keys:
                logger.warning(
                    f"discovery.json missing required keys: {missing_keys}; "
                    f"ignoring discovery metadata"
                )
                discovery_result = None
            else:
                # Extract only schema-defined fields (filter out extra junk)
                discovery_result = {
                    "search_criteria": raw_discovery["search_criteria"],
                    "matched_groups": raw_discovery["matched_groups"],
                    "selected_groups": raw_discovery["selected_groups"],
                    "selection_reason": raw_discovery["selection_reason"],
                    "total_matched": raw_discovery["total_matched"],
                    "total_selected": raw_discovery["total_selected"]
                }

                # Include optional scoring_details if present
                if "scoring_details" in raw_discovery:
                    discovery_result["scoring_details"] = raw_discovery["scoring_details"]

                logger.info(f"Discovery result loaded: {discovery_result['total_selected']} groups selected")
                logger.info(f"Selected groups: {discovery_result['selected_groups']}")
                logger.info(f"Selection reason: {discovery_result['selection_reason']}")

        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse discovery.json: {e}")
            discovery_result = None
        except Exception as e:
            logger.warning(f"Error reading discovery.json: {e}")
            discovery_result = None
    else:
        # Discovery was requested but discovery.json not found
        if discovery_requested:
            logger.warning(
                "Discovery requested (--log-group-prefix or --log-group-pattern) "
                "but discovery.json not found; proceeding without discovery metadata"
            )

    return stage1_result, discovery_result


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception:
        return "error_computing_hash"


def write_manifest(
    manifest_path: Path,
    run_id: str,
    args: argparse.Namespace,
    window: Dict[str, str],
    script_commands: List[Dict[str, Any]],
    output_files: Dict[str, List[str]],
    discovery_result: Optional[Dict[str, Any]] = None
):
    """
    Write manifest.json for reproducibility.
    
    Args:
        manifest_path: Path to write manifest.json
        run_id: Run identifier
        args: Parsed command-line arguments
        window: Time window dict with start/end
        script_commands: List of script execution results
        output_files: Dict of output files by stage
        discovery_result: Optional discovery result from discovery.json
    """
    artifact_hashes = {}
    for stage, files in output_files.items():
        for file_path_str in files:
            file_path = Path(file_path_str)
            if file_path.exists():
                artifact_hashes[file_path_str] = compute_file_hash(file_path)
    
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cli_args": vars(args),
        "window": window,
        "trace_provider": args.trace_provider,
        "script_commands": script_commands,
        "output_files": output_files,
        "artifact_hashes": artifact_hashes,
        "tool_version": {
            "note": "Add git SHA here if needed"
        }
    }
    
    # Add discovery section if discovery was used
    if discovery_result is not None:
        manifest["discovery"] = discovery_result
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def write_summary_json(
    summary_path: Path,
    run_id: str,
    window: Dict[str, str],
    aws_config: Dict[str, Optional[str]],
    inputs: Dict[str, Any],
    stats: Dict[str, Any],
    quality_gates_passed: bool,
    quality_gate_failures: List[str],
    output_paths: Dict[str, Any]
):
    """Write machine-readable summary.json."""
    summary = {
        "run_id": run_id,
        "window": window,
        "aws": aws_config,
        "inputs": inputs,
        "counts": {
            "turns_indexed": stats["turns_indexed"],
            "traces_exported": stats["traces_exported"],
            "merged_turns": stats["merged_turns"]
        },
        "joins": stats["joins"],
        "user_query": stats["user_query"],
        "traces": stats["traces"],
        "quality_gates": {
            "passed": quality_gates_passed,
            "failures": quality_gate_failures
        },
        "outputs": output_paths
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


def write_summary_txt(
    summary_path: Path,
    run_id: str,
    window: Dict[str, str],
    stats: Dict[str, Any],
    quality_gates_passed: bool,
    quality_gate_failures: List[str]
):
    """Write human-readable summary.txt."""
    lines = [
        "="*60,
        "AgentCore Pipeline Summary",
        "="*60,
        f"Run ID: {run_id}",
        f"Window: {window['start']} to {window['end']}",
        "",
        "Counts:",
        f"  Turns indexed: {stats['turns_indexed']}",
        f"  Traces exported: {stats['traces_exported']}",
        f"  Merged turns: {stats['merged_turns']}",
        "",
        "Joins:",
        f"  By trace_id: {stats['joins']['by_trace_id']}",
        f"  By request_id: {stats['joins']['by_request_id']}",
        f"  By session_time: {stats['joins']['by_session_time']}",
        f"  Unmatched: {stats['joins']['unmatched']}",
        "",
        "User Query:",
        f"  With user_query: {stats['user_query']['turns_with_user_query']}",
        f"  Missing user_query: {stats['user_query']['turns_missing_user_query']}",
        "",
        "Traces:",
        f"  With steps: {stats['traces']['turns_with_steps']}",
        f"  Missing steps: {stats['traces']['turns_missing_steps']}",
        "",
        "Quality Gates:",
        f"  Status: {'PASSED' if quality_gates_passed else 'FAILED'}",
    ]
    if quality_gate_failures:
        lines.append("  Failures:")
        for failure in quality_gate_failures:
            lines.append(f"    - {failure}")
    lines.append("="*60)
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))


def print_final_summary(
    run_id: str,
    window: Dict[str, str],
    stats: Dict[str, Any],
    quality_gates_passed: bool,
    output_paths: Dict[str, Any],
    logger: logging.Logger
):
    """Print final summary to console."""
    logger.info("="*60)
    logger.info("Pipeline Complete")
    logger.info("="*60)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Window: {window['start']} to {window['end']}")
    logger.info("")
    logger.info(f"Turns indexed: {stats['turns_indexed']}")
    logger.info(f"Traces exported: {stats['traces_exported']}")
    logger.info(f"Merged turns: {stats['merged_turns']}")
    logger.info("")
    logger.info("Joins:")
    logger.info(f"  By trace_id: {stats['joins']['by_trace_id']}")
    logger.info(f"  By request_id: {stats['joins']['by_request_id']}")
    logger.info(f"  Unmatched: {stats['joins']['unmatched']}")
    logger.info("")
    logger.info("User Query:")
    logger.info(f"  With: {stats['user_query']['turns_with_user_query']}")
    logger.info(f"  Missing: {stats['user_query']['turns_missing_user_query']}")
    logger.info("")
    logger.info("Output Paths:")
    for key, value in output_paths.items():
        if isinstance(value, list):
            logger.info(f"  {key}: {len(value)} files")
        else:
            logger.info(f"  {key}: {value}")
    logger.info("")
    logger.info(f"Quality Gates: {'PASSED' if quality_gates_passed else 'FAILED'}")
    logger.info("="*60)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AgentCore Evidence Reconstruction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Relative time window
  python -m agent_eval.tools.agentcore_pipeline.export_agentcore_pipeline export-agentcore \\
      --days 30 --region us-west-2 --output-root ./outputs

  # Absolute time window
  python -m agent_eval.tools.agentcore_pipeline.export_agentcore_pipeline export-agentcore \\
      --start-time "2026-02-01T00:00:00Z" --end-time "2026-02-26T23:59:59Z" \\
      --region us-west-2 --output-root ./outputs
        """
    )
    parser.add_argument(
        "command",
        choices=["export-agentcore"],
        help="Command to run"
    )
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument(
        "--days",
        type=int,
        dest="days",
        help="Number of days to look back (default: 7)"
    )
    time_group.add_argument(
        "--start-time",
        dest="start_time",
        help="ISO 8601 start time (requires --end-time)"
    )
    parser.add_argument(
        "--end-time",
        dest="end_time",
        help="ISO 8601 end time (requires --start-time)"
    )
    parser.add_argument("--region", help="AWS region")
    parser.add_argument("--profile", help="AWS profile name")
    app_group = parser.add_mutually_exclusive_group()
    app_group.add_argument("--app-log-group", help="App log group name")
    app_group.add_argument("--app-log-group-prefix", help="App log group prefix")
    app_group.add_argument("--app-log-group-pattern", help="App log group regex pattern")
    parser.add_argument(
        "--trace-provider",
        choices=["xray", "cloudwatch_logs", "auto"],
        default="auto",
        help="Trace provider (default: auto)"
    )
    trace_group = parser.add_mutually_exclusive_group()
    trace_group.add_argument("--trace-log-group", help="Trace log group name")
    trace_group.add_argument("--trace-log-group-prefix", help="Trace log group prefix")
    trace_group.add_argument("--trace-log-group-pattern", help="Trace log group regex pattern")
    parser.add_argument(
        "--output-root",
        default="./outputs",
        help="Base output directory (default: ./outputs)"
    )
    parser.add_argument("--run-id", help="Custom run ID (default: auto-generated)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing run directory"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    parser.add_argument("--max-turns", type=int, help="Max turns to export")
    parser.add_argument("--max-traces", type=int, help="Max traces to export")
    parser.add_argument(
        "--no-fail-on-missing-user-query",
        dest="fail_on_missing_user_query",
        action="store_false",
        default=True,
        help="Don't fail if user_query missing"
    )
    parser.add_argument(
        "--no-fail-on-missing-traces",
        dest="fail_on_missing_traces",
        action="store_false",
        default=True,
        help="Don't fail if traces missing"
    )
    parser.add_argument(
        "--max-missing-user-query",
        type=int,
        default=0,
        help="Max missing user_query threshold (default: 0)"
    )
    parser.add_argument(
        "--max-missing-trace-artifacts",
        type=int,
        default=0,
        help="Max missing trace artifacts threshold (default: 0)"
    )
    parser.add_argument(
        "--join-strategy",
        choices=["trace_id", "request_id", "session_time", "auto"],
        default="auto",
        help="Join strategy (default: auto)"
    )
    parser.add_argument(
        "--no-emit-summary-json",
        dest="emit_summary_json",
        action="store_false",
        default=True,
        help="Don't emit summary.json"
    )
    parser.add_argument(
        "--log-level",
        choices=["INFO", "DEBUG", "WARNING"],
        default="INFO",
        help="Log level (default: INFO)"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--print-commands", action="store_true", help="Print script commands")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    logger = setup_logging(args.log_level, args.quiet)
    try:
        start_time = None
        end_time = None
        if args.days and (args.start_time or args.end_time):
            logger.error("Cannot use --days with --start-time or --end-time")
            return EXIT_VALIDATION_ERROR
        if args.start_time or args.end_time:
            if not (args.start_time and args.end_time):
                logger.error("--start-time and --end-time must be used together")
                return EXIT_VALIDATION_ERROR
            try:
                start_time = datetime.fromisoformat(args.start_time.replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(args.end_time.replace('Z', '+00:00'))
            except ValueError as e:
                logger.error(f"Invalid time format: {e}")
                return EXIT_VALIDATION_ERROR
        try:
            start_time, end_time = validate_time_window(start_time, end_time, args.days)
        except ValueError as e:
            logger.error(f"Time window validation failed: {e}")
            return EXIT_VALIDATION_ERROR
        window = {
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        }
        if args.trace_provider == "cloudwatch_logs":
            logger.error("CloudWatch Logs trace provider not yet implemented")
            logger.error("Use --trace-provider xray or --trace-provider auto (defaults to X-Ray)")
            return EXIT_VALIDATION_ERROR
        if args.trace_provider == "auto":
            logger.info("Trace provider set to 'auto', defaulting to X-Ray")
            args.trace_provider = "xray"
        if args.trace_log_group or args.trace_log_group_prefix or args.trace_log_group_pattern:
            logger.warning("Trace log group flags are not yet used by the pipeline scripts")
            logger.warning("Trace extraction currently uses X-Ray API, not CloudWatch Logs")
        app_inputs = {
            "log_group": args.app_log_group,
            "log_group_prefix": args.app_log_group_prefix,
            "log_group_pattern": args.app_log_group_pattern
        }
        trace_inputs = {
            "provider": args.trace_provider,
            "log_group": args.trace_log_group,
            "log_group_prefix": args.trace_log_group_prefix,
            "log_group_pattern": args.trace_log_group_pattern
        }
        if args.run_id:
            run_id = args.run_id
        else:
            run_id = generate_run_id(
                start_time, end_time, args.region, args.profile,
                app_inputs, trace_inputs
            )
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Window: {window['start']} to {window['end']}")
        try:
            output_root = Path(args.output_root)
            paths = setup_output_structure(output_root, run_id, args.overwrite)
        except FileExistsError as e:
            logger.error(str(e))
            return EXIT_VALIDATION_ERROR
        logger.info(f"Output directory: {paths['base']}")
        if args.dry_run:
            logger.info("="*60)
            logger.info("DRY RUN MODE - No scripts will be executed")
            logger.info("="*60)
            logger.info(f"Run ID: {run_id}")
            logger.info(f"Output directory: {paths['base']}")
            logger.info(f"Window: {window['start']} to {window['end']}")
            logger.info("")
            logger.info("App inputs:")
            for k, v in app_inputs.items():
                if v:
                    logger.info(f"  {k}: {v}")
            logger.info("")
            logger.info("Trace inputs:")
            for k, v in trace_inputs.items():
                if v:
                    logger.info(f"  {k}: {v}")
            logger.info("")
            logger.info("Planned script commands:")
            script_dir = Path(__file__).parent
            script1_args = {
                "output-dir": str(paths["raw"]),
                "region": args.region,
                "profile": args.profile
            }
            if args.days:
                script1_args["days"] = args.days
            else:
                script1_args["start-time"] = args.start_time
                script1_args["end-time"] = args.end_time
            if args.app_log_group:
                script1_args["log-group"] = args.app_log_group
            elif args.app_log_group_prefix:
                script1_args["log-group-prefix"] = args.app_log_group_prefix
            elif args.app_log_group_pattern:
                script1_args["log-group-pattern"] = args.app_log_group_pattern
            script1_cmd = build_script_command("01_export_turns_from_app_logs.py", script_dir, script1_args)
            logger.info(f"Stage 1: {' '.join(script1_cmd)}")
            script2_cmd = build_script_command("02_build_session_trace_index.py", script_dir, {
                "input-dir": str(paths["raw"]),
                "output-dir": str(paths["raw"])
            })
            logger.info(f"Stage 2: {' '.join(script2_cmd)}")
            script3_cmd = build_script_command("03_add_xray_steps_and_latency.py", script_dir, {
                "input-dir": str(paths["raw"]),
                "output-dir": str(paths["merged"]),
                "join-strategy": args.join_strategy
            })
            logger.info(f"Stage 3: {' '.join(script3_cmd)}")
            logger.info("="*60)
            return EXIT_SUCCESS
        script_dir = Path(__file__).parent
        script_commands = []
        
        # Run Stage 1 and capture discovery result
        script1_result, discovery_result = run_stage1(
            args, paths, script_dir, logger
        )
        script_commands.append(script1_result)
        
        if script1_result["status"] != "success":
            if script1_result.get("timed_out"):
                logger.error("Stage 1 timed out")
                return EXIT_RUNTIME_ERROR
            if script1_result.get("is_aws_error"):
                logger.error("AWS credentials or permissions error")
                return EXIT_AWS_ERROR
            else:
                logger.error("Stage 1 failed")
                return EXIT_RUNTIME_ERROR
        logger.info("="*60)
        logger.info("Stage 2: Build session trace index")
        logger.info("="*60)
        script2_args = {
            "input-dir": str(paths["raw"]),
            "output-dir": str(paths["raw"]),
            "region": args.region,
            "profile": args.profile
        }
        if args.max_traces:
            script2_args["max-traces"] = args.max_traces
        script2_cmd = build_script_command("02_build_session_trace_index.py", script_dir, script2_args)
        script2_result = run_script(
            "02_build_session_trace_index.py",
            2,
            "Build trace index",
            script2_cmd,
            script_dir,
            logger,
            args.print_commands
        )
        script_commands.append(script2_result)
        if script2_result["status"] != "success":
            if script2_result.get("timed_out"):
                logger.error("Stage 2 timed out")
                return EXIT_RUNTIME_ERROR
            if script2_result.get("is_aws_error"):
                logger.error("AWS credentials or permissions error")
                return EXIT_AWS_ERROR
            logger.error("Stage 2 failed")
            return EXIT_RUNTIME_ERROR
        logger.info("="*60)
        logger.info("Stage 3: Merge X-Ray spans and calculate latency")
        logger.info("="*60)
        script3_args = {
            "input-dir": str(paths["raw"]),
            "output-dir": str(paths["merged"]),
            "region": args.region,
            "profile": args.profile,
            "join-strategy": args.join_strategy
        }
        script3_cmd = build_script_command("03_add_xray_steps_and_latency.py", script_dir, script3_args)
        script3_result = run_script(
            "03_add_xray_steps_and_latency.py",
            3,
            "Merge X-Ray spans",
            script3_cmd,
            script_dir,
            logger,
            args.print_commands
        )
        script_commands.append(script3_result)
        if script3_result["status"] != "success":
            if script3_result.get("timed_out"):
                logger.error("Stage 3 timed out")
                return EXIT_RUNTIME_ERROR
            if script3_result.get("is_aws_error"):
                logger.error("AWS credentials or permissions error")
                return EXIT_AWS_ERROR
            logger.error("Stage 3 failed")
            return EXIT_RUNTIME_ERROR
        logger.info("="*60)
        logger.info("Computing coverage statistics")
        logger.info("="*60)
        stats = compute_coverage_stats(paths["raw"], paths["merged"], logger)
        if stats["turns_indexed"] == 0:
            logger.error("Capability check failed: No turns indexed")
            logger.error("Pipeline may have failed or no data in time window")
            return EXIT_QUALITY_GATE_FAILED
        if stats["traces_exported"] == 0:
            logger.error("Capability check failed: No traces exported")
            logger.error("Trace provider may be misconfigured or no trace data available")
            return EXIT_QUALITY_GATE_FAILED
        quality_gates_passed, quality_gate_failures = check_quality_gates(
            stats,
            args.max_missing_user_query,
            args.max_missing_trace_artifacts,
            args.fail_on_missing_user_query,
            args.fail_on_missing_traces,
            logger
        )
        actual_turn_files = list(paths["raw"].glob("*turn*.json"))
        actual_trace_files = list(paths["raw"].glob("*trace*.json"))
        actual_normalized_files = list(paths["merged"].glob("*.json"))
        output_files = {
            "stage1_turns": [str(f) for f in actual_turn_files],
            "stage2_traces": [str(f) for f in actual_trace_files],
            "stage3_normalized": [str(f) for f in actual_normalized_files]
        }
        output_paths = {
            "turn_index": [str(f) for f in actual_turn_files],
            "trace_export": [str(f) for f in actual_trace_files],
            "normalized_run": [str(f) for f in actual_normalized_files],
            "manifest": str(paths["base"] / "manifest.json"),
            "summary_json": str(paths["reports"] / "summary.json"),
            "summary_txt": str(paths["reports"] / "summary.txt")
        }
        write_manifest(
            paths["base"] / "manifest.json",
            run_id,
            args,
            window,
            script_commands,
            output_files,
            discovery_result
        )
        if args.emit_summary_json:
            write_summary_json(
                paths["reports"] / "summary.json",
                run_id,
                window,
                {"region": args.region, "profile": args.profile},
                {"app": app_inputs, "trace": trace_inputs},
                stats,
                quality_gates_passed,
                quality_gate_failures,
                output_paths
            )
        write_summary_txt(
            paths["reports"] / "summary.txt",
            run_id,
            window,
            stats,
            quality_gates_passed,
            quality_gate_failures
        )
        print_final_summary(run_id, window, stats, quality_gates_passed, output_paths, logger)
        if quality_gates_passed:
            logger.info("✓ All quality gates passed")
            return EXIT_SUCCESS
        else:
            logger.error("✗ Quality gates failed")
            return EXIT_QUALITY_GATE_FAILED
    except KeyboardInterrupt:
        logger.error("Interrupted by user")
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
