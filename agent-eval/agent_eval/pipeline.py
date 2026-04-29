# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Pipeline orchestrator for end-to-end trace evaluation.

This module provides a unified pipeline that:
1. Detects if input is raw or normalized format using schema validation
2. Runs Generic_JSON_Adapter if needed
3. Validates adapter output against NormalizedRun schema
4. Persists normalized artifact with safe filenames
5. Calls TraceEvaluator with normalized input
6. Returns evaluation results

ARCHITECTURE: Pipeline is the single source of truth for:
- Input format detection (via schema validation)
- Adapter invocation and output validation
- Normalized artifact persistence
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional


class PipelineError(Exception):
    """Pipeline execution error."""
    pass


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """
    Sanitize text for use in filenames.
    
    Replaces unsafe characters with underscores and truncates to max_length.
    Returns "run" as fallback if result is empty.
    
    Args:
        text: Text to sanitize
        max_length: Maximum length of sanitized text
        
    Returns:
        Sanitized text safe for use in filenames (never empty)
    """
    # Replace unsafe characters with underscores
    safe_text = re.sub(r'[^A-Za-z0-9._-]', '_', text)
    # Strip leading/trailing dots and underscores to avoid hidden files
    safe_text = safe_text.strip('._')
    # Truncate to max length
    safe_text = safe_text[:max_length]
    # Return fallback if empty
    return safe_text or "run"


def detect_input_format(input_path: str) -> str:
    """
    Detect if input is raw or normalized format using schema validation.
    
    Uses InputValidator to perform proper schema validation instead of
    heuristic field checking. This ensures we correctly identify normalized
    inputs and don't accidentally treat malformed normalized files as raw.
    
    Args:
        input_path: Path to input file
        
    Returns:
        "normalized" if input passes NormalizedRun schema validation, "raw" otherwise
        
    Raises:
        PipelineError: If critical errors occur (missing schema, import failures, etc.)
    """
    try:
        from agent_eval.evaluators.trace_eval.input_validator import InputValidator, ValidationError as InputValidationError
        
        # Load JSON data
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Attempt schema validation
        validator = InputValidator()
        validator.validate(data)
        
        # If validation passes, it's normalized
        return "normalized"
        
    except InputValidationError:
        # Schema validation failed - this is expected for raw input
        return "raw"
    except json.JSONDecodeError:
        # Invalid JSON - treat as raw (adapter will handle the error)
        return "raw"
    except (ImportError, FileNotFoundError, AttributeError) as e:
        # Critical errors that should not be hidden:
        # - ImportError: validator module or dependencies missing
        # - FileNotFoundError: schema file missing
        # - AttributeError: validator API changed
        raise PipelineError(
            f"Critical error during input format detection: {e}. "
            f"This indicates a setup or configuration problem, not a raw input file."
        ) from e
    except Exception as e:
        # Unexpected errors should also surface
        raise PipelineError(
            f"Unexpected error during input format detection: {e}"
        ) from e


def run_pipeline(
    input_path: str,
    judge_config_path: str,
    output_dir: str,
    adapter_config_path: Optional[str] = None,
    rubrics_path: Optional[str] = None,
    verbose: bool = False,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline.
    
    ARCHITECTURE: Pipeline is the single source of truth for:
    - Input format detection (via schema validation)
    - Adapter invocation and output validation
    - Normalized artifact persistence
    
    TraceEvaluator only accepts pre-validated normalized input.
    
    Args:
        input_path: Path to raw trace or NormalizedRun file
        judge_config_path: Path to judges.yaml configuration
        output_dir: Directory for output files
        adapter_config_path: Optional path to adapter configuration
        rubrics_path: Optional path to user rubrics.yaml
        verbose: Enable verbose output
        debug: Enable debug mode
        
    Returns:
        Pipeline execution results with:
        - input_format: "raw" or "normalized"
        - normalized_path: Path to normalized artifact
        - evaluation_exit_code: Exit code from TraceEvaluator
        - output_dir: Output directory path
        
    Raises:
        PipelineError: If pipeline execution fails
    """
    try:
        from agent_eval.evaluators.trace_eval.runner import TraceEvaluator
        from agent_eval.adapters.generic_json.adapter import adapt
        from agent_eval.evaluators.trace_eval.input_validator import InputValidator, ValidationError
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print("=" * 60)
            print("EVALUATION PIPELINE")
            print("=" * 60)
        
        # Step 1: Detect input format using schema validation
        input_format = detect_input_format(input_path)
        
        if verbose:
            print(f"\nStep 1: Detected input format: {input_format}")
        
        # Step 2: Normalize input if needed
        if input_format == "raw":
            if verbose:
                print(f"\nStep 2: Running Generic_JSON_Adapter on {input_path}")
            
            # Run adapter (verified signature: adapt(path, config_path=None))
            normalized_data = adapt(
                path=input_path,
                config_path=adapter_config_path
            )
            
            # Validate adapter output against NormalizedRun schema
            if verbose:
                print("  Validating adapter output against schema...")
            
            validator = InputValidator()
            try:
                validated_data = validator.validate(normalized_data)
            except ValidationError as e:
                # ValidationError has .message attribute
                error_msg = e.message if hasattr(e, 'message') else str(e)
                raise PipelineError(
                    f"Adapter output failed schema validation: {error_msg}"
                ) from e
            
            if verbose:
                print("  ✓ Adapter output validated")
            
            # Persist normalized artifact with safe filename
            run_id = validated_data.get("run_id", "unknown")
            safe_run_id = sanitize_filename(run_id)
            normalized_path = output_path / f"normalized_run.{safe_run_id}.json"
            
            with open(normalized_path, 'w', encoding='utf-8') as f:
                json.dump(validated_data, f, ensure_ascii=False, indent=2)
            
            if verbose:
                print(f"✓ Wrote normalized artifact: {normalized_path}")
            
            # Use normalized artifact as input for evaluator
            evaluator_input = str(normalized_path)
            
        else:
            if verbose:
                print(f"\nStep 2: Input already normalized, copying to output directory")
            
            # Load and validate the normalized input
            with open(input_path, 'r') as f:
                normalized_data = json.load(f)
            
            # Validate to ensure it's truly normalized
            validator = InputValidator()
            try:
                validated_data = validator.validate(normalized_data)
            except ValidationError as e:
                error_msg = e.message if hasattr(e, 'message') else str(e)
                raise PipelineError(
                    f"Input claimed to be normalized but failed validation: {error_msg}"
                ) from e
            
            # Pipeline is single source of truth for normalized artifact persistence
            # Always write canonical normalized file to output_dir
            run_id = validated_data.get("run_id", "unknown")
            safe_run_id = sanitize_filename(run_id)
            normalized_path = output_path / f"normalized_run.{safe_run_id}.json"
            
            with open(normalized_path, 'w', encoding='utf-8') as f:
                json.dump(validated_data, f, ensure_ascii=False, indent=2)
            
            if verbose:
                print(f"✓ Wrote canonical normalized artifact: {normalized_path}")
            
            # Use canonical normalized artifact as input for evaluator
            evaluator_input = str(normalized_path)
        
        # Step 3: Run TraceEvaluator in normalized-only mode
        # Pipeline owns adaptation - runner only accepts normalized input
        if verbose:
            print(f"\nStep 3: Running TraceEvaluator (normalized-only mode)")
        
        evaluator = TraceEvaluator(
            input_path=evaluator_input,
            judge_config_path=judge_config_path,
            output_dir=output_dir,
            rubrics_path=rubrics_path,
            verbose=verbose,
            debug=debug
        )
        
        exit_code = evaluator.run()
        
        # Step 4: Return results
        results = {
            "input_format": input_format,
            "normalized_path": str(normalized_path),
            "evaluation_exit_code": exit_code,
            "output_dir": output_dir,
            "success": exit_code == 0
        }
        
        if verbose:
            print("\n" + "=" * 60)
            if results["success"]:
                print("✓ PIPELINE COMPLETE")
            else:
                print(f"✗ PIPELINE FAILED (exit code: {exit_code})")
            print(f"  Output directory: {output_dir}")
            print("=" * 60)
        
        return results
        
    except PipelineError:
        # Re-raise PipelineError without wrapping to avoid noisy nested messages
        raise
    except Exception as e:
        raise PipelineError(f"Pipeline execution failed: {e}") from e
