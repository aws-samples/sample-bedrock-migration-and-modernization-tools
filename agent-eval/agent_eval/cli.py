# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""CLI entry point for agentic-eval."""

import argparse
import json
import os
import sys
from pathlib import Path


# Exit codes
EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_CONFIG_ERROR = 3
EXIT_VALIDATION_ERROR = 4


def validate_cli_args(args: argparse.Namespace) -> None:
    """
    Validate CLI arguments before execution.
    
    Args:
        args: Parsed CLI arguments
        
    Raises:
        SystemExit: With appropriate exit code if validation fails
    """
    errors = []
    exit_code = EXIT_USAGE_ERROR  # Default, may be overridden
    
    # Validate --input exists and is readable
    if not os.path.exists(args.input):
        errors.append(f"Input file not found: {args.input}")
        exit_code = EXIT_VALIDATION_ERROR
    elif not os.path.isfile(args.input):
        errors.append(f"Input path is not a file: {args.input}")
        exit_code = EXIT_VALIDATION_ERROR
    elif not os.access(args.input, os.R_OK):
        errors.append(f"Input file not readable: {args.input}")
        exit_code = EXIT_VALIDATION_ERROR
    # Note: We don't validate JSON structure here even with --input-is-normalized
    # The runner will validate against normalized_run.schema.json (the real validation)
    # CLI only checks file existence and readability
    
    # Validate --judge-config exists and is readable (config error)
    if not os.path.exists(args.judge_config):
        errors.append(f"Judge config file not found: {args.judge_config}")
        exit_code = EXIT_CONFIG_ERROR
    elif not os.path.isfile(args.judge_config):
        errors.append(f"Judge config path is not a file: {args.judge_config}")
        exit_code = EXIT_CONFIG_ERROR
    elif not os.access(args.judge_config, os.R_OK):
        errors.append(f"Judge config file not readable: {args.judge_config}")
        exit_code = EXIT_CONFIG_ERROR
    
    # Validate --rubrics exists if provided (config error)
    if args.rubrics:
        if not os.path.exists(args.rubrics):
            errors.append(f"Rubrics file not found: {args.rubrics}")
            exit_code = EXIT_CONFIG_ERROR
        elif not os.path.isfile(args.rubrics):
            errors.append(f"Rubrics path is not a file: {args.rubrics}")
            exit_code = EXIT_CONFIG_ERROR
        elif not os.access(args.rubrics, os.R_OK):
            errors.append(f"Rubrics file not readable: {args.rubrics}")
            exit_code = EXIT_CONFIG_ERROR
    
    # Validate --output-dir is writable (usage error)
    output_path = Path(args.output_dir).resolve()
    if output_path.exists():
        if not output_path.is_dir():
            errors.append(f"Output path exists but is not a directory: {args.output_dir}")
            exit_code = EXIT_USAGE_ERROR
        else:
            # Do a touch test to verify we can actually write files
            try:
                test_file = output_path / f".write_test_{os.getpid()}"
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                errors.append(f"Output directory not writable (touch test failed): {args.output_dir} - {e}")
                exit_code = EXIT_USAGE_ERROR
    else:
        # Check if parent directory exists and is writable + executable
        parent = output_path.parent
        if not parent.exists():
            errors.append(f"Parent directory does not exist: {parent}")
            exit_code = EXIT_USAGE_ERROR
        elif not os.access(str(parent), os.W_OK):
            errors.append(f"Cannot create output directory (parent not writable): {parent}")
            exit_code = EXIT_USAGE_ERROR
        elif not os.access(str(parent), os.X_OK):
            errors.append(f"Cannot create output directory (parent not executable/searchable): {parent}")
            exit_code = EXIT_USAGE_ERROR
    
    if errors:
        print("CLI validation errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(exit_code)
    
    # Update args with resolved absolute path for consistency
    args.output_dir = str(output_path)


def trace_eval_cli(argv=None):
    """
    Thin CLI entrypoint for trace evaluation.
    
    ARCHITECTURE: Handles adapter logic at CLI layer.
    TraceEvaluator only accepts normalized input.
    
    Responsibilities:
    - Parse CLI arguments
    - Run adapter if input is not normalized
    - Validate required arguments (preflight only)
    - Delegate to TraceEvaluator.run()
    - Handle exit codes from runner (preserves runner's canonical exit codes)
    
    Args:
        argv: Command-line arguments (defaults to sys.argv[1:] if None)
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        prog="trace-eval",
        description="Evaluate normalized agent traces using rubric-driven multi-judge system",
        epilog="Part of the agent-evaluation framework"
    )
    
    # Required arguments
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Path to input file (raw trace or NormalizedRun JSON). "
             "If raw trace, adapter will run automatically unless --input-is-normalized is set."
    )
    parser.add_argument(
        "--judge-config",
        required=True,
        metavar="PATH",
        help="Path to judges.yaml configuration file (1-5 judges required)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory for output files (trace_eval.json, judge_runs.jsonl, results.json)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--rubrics",
        metavar="PATH",
        help="Path to user rubrics.yaml (optional, merges with default rubrics by rubric_id)"
    )
    parser.add_argument(
        "--input-is-normalized",
        action="store_true",
        help="Skip adapter processing and treat input as pre-normalized NormalizedRun JSON"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed progress information"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with full stack traces on errors"
    )
    
    args = parser.parse_args(argv)
    
    # Validate arguments early (modifies args.output_dir to absolute path)
    validate_cli_args(args)
    
    try:
        from agent_eval.evaluators.trace_eval.runner import TraceEvaluator
        from pathlib import Path
        import tempfile
        
        # Handle adapter logic at CLI layer
        if args.input_is_normalized:
            # Input is already normalized, use directly
            normalized_input_path = args.input
        else:
            # Run adapter to normalize input
            from agent_eval.adapters.generic_json.adapter import adapt
            
            if args.verbose:
                print(f"Running adapter on {args.input}...")
            
            normalized_data = adapt(args.input)
            
            # Write normalized data to temp file for evaluator
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use run_id from normalized data for filename
            run_id = normalized_data.get("run_id", "unknown")
            normalized_input_path = str(output_dir / f"normalized_input.{run_id}.json")
            
            with open(normalized_input_path, 'w') as f:
                json.dump(normalized_data, f, ensure_ascii=False, indent=2)
            
            if args.verbose:
                print(f"✓ Adapter completed, wrote {normalized_input_path}")
        
        # Create evaluator with normalized input
        evaluator = TraceEvaluator(
            input_path=normalized_input_path,
            judge_config_path=args.judge_config,
            output_dir=args.output_dir,
            rubrics_path=args.rubrics,
            verbose=args.verbose,
            debug=args.debug
        )
        
        exit_code = evaluator.run()
        
        # Ensure we return an integer exit code
        if exit_code is None:
            # Treat None as success (backward compatibility)
            return EXIT_SUCCESS
        return int(exit_code)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
        
    except Exception as e:
        # Only catch truly unexpected failures outside runner
        # Runner owns evaluation-layer classification and returns its exit code
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        return EXIT_RUNTIME_ERROR


def eval_pipeline_cli(argv=None):
    """
    CLI entrypoint for full evaluation pipeline.
    
    Automatically detects input format (raw vs normalized) and runs
    the complete pipeline: adapter (if needed) → evaluator → results.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv[1:] if None)
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        prog="eval-pipeline",
        description="Run complete evaluation pipeline with automatic format detection",
        epilog="Part of the agent-evaluation framework"
    )
    
    # Required arguments
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Path to input file (raw trace or NormalizedRun JSON). "
             "Format will be auto-detected."
    )
    parser.add_argument(
        "--judge-config",
        required=True,
        metavar="PATH",
        help="Path to judges.yaml configuration file (1-5 judges required)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory for output files (normalized_run.json, trace_eval.json, judge_runs.jsonl, results.json)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--adapter-config",
        metavar="PATH",
        help="Path to adapter configuration (optional, for raw trace processing)"
    )
    parser.add_argument(
        "--rubrics",
        metavar="PATH",
        help="Path to user rubrics.yaml (optional, merges with default rubrics by rubric_id)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed progress information"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with full stack traces on errors"
    )
    
    args = parser.parse_args(argv)
    
    # Validate arguments (reuse existing validation logic)
    validate_cli_args(args)
    
    try:
        from agent_eval.pipeline import run_pipeline
        
        results = run_pipeline(
            input_path=args.input,
            judge_config_path=args.judge_config,
            output_dir=args.output_dir,
            adapter_config_path=args.adapter_config,
            rubrics_path=args.rubrics,
            verbose=args.verbose,
            debug=args.debug
        )
        
        # Return exit code from evaluation
        return results.get("evaluation_exit_code", EXIT_RUNTIME_ERROR)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
        
    except Exception as e:
        print(f"Pipeline error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        return EXIT_RUNTIME_ERROR


def main(argv=None):
    """
    Main CLI entry point with subcommand support.
    
    Currently delegates directly to trace_eval_cli.
    Future: add subcommands for adapter, results-merge, etc.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv[1:] if None)
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    # For now, directly run trace evaluation
    # Future: add argparse subparsers for multiple commands
    return trace_eval_cli(argv)


if __name__ == "__main__":
    sys.exit(main())
