# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Output Writer for Trace Evaluation Results

This module writes canonical output files for trace evaluation:
- trace_eval.json: Detailed evaluation results with metrics and rubric scores
- results.json: Summary with fingerprinting and execution statistics
- Validates JSON output and computes file hashes for integrity
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import asdict, is_dataclass
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

from .deterministic_metrics import MetricsResult
from .judging.aggregator import CrossJudgeResult


class OutputWriter:
    """
    Writes canonical output files for trace evaluation.
    
    Produces three output files:
    1. trace_eval.json - Detailed evaluation results
    2. judge_runs.jsonl - Raw per-job records (written incrementally by WorkerPool)
    3. results.json - Summary with fingerprinting and execution stats
    """
    
    FORMAT_VERSION = "1.0.0"
    
    def __init__(self, output_dir: str):
        """
        Initialize output writer.
        
        Args:
            output_dir: Directory where output files will be written
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _canonicalize_for_json(self, obj: Any) -> Any:
        """
        Deep-convert object to JSON-serializable primitives.
        
        Handles dataclasses, enums, Path, sets, Decimal, datetime, bytes, etc.
        
        Args:
            obj: Object to canonicalize
            
        Returns:
            JSON-serializable version of obj
        """
        # Handle None
        if obj is None:
            return None
        
        # Handle dataclasses
        if is_dataclass(obj) and not isinstance(obj, type):
            return self._canonicalize_for_json(asdict(obj))
        
        # Handle enums
        if isinstance(obj, Enum):
            return obj.value
        
        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)
        
        # Handle datetime/date
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        # Handle Decimal
        if isinstance(obj, Decimal):
            return float(obj)
        
        # Handle bytes
        if isinstance(obj, bytes):
            return obj.hex()
        
        # Handle sets
        if isinstance(obj, set):
            return sorted(self._canonicalize_for_json(item) for item in obj)
        
        # Handle dictionaries
        if isinstance(obj, dict):
            return {
                str(k): self._canonicalize_for_json(v) 
                for k, v in obj.items()
            }
        
        # Handle lists/tuples
        if isinstance(obj, (list, tuple)):
            return [self._canonicalize_for_json(item) for item in obj]
        
        # Handle numpy types if numpy is available
        try:
            import numpy as np
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        
        # Primitives (str, int, float, bool)
        return obj
    
    def write_trace_eval(
        self,
        run_id: str,
        deterministic_metrics: MetricsResult,
        rubric_results: List[Dict[str, Any]],
        judge_summary: Dict[str, Any]
    ) -> str:
        """
        Write trace_eval.json with canonical structure.
        
        Args:
            run_id: Run identifier from NormalizedRun
            deterministic_metrics: Computed deterministic metrics
            rubric_results: List of rubric evaluation results
            judge_summary: Summary of judge execution statistics
            
        Returns:
            Path to written trace_eval.json file
            
        Raises:
            IOError: If file cannot be written
            ValueError: If output validation fails
        """
        output_path = self.output_dir / "trace_eval.json"
        
        # Build output structure matching trace_eval_output.schema.json
        output_data = {
            "format_version": self.FORMAT_VERSION,
            "run_id": run_id,
            "deterministic_metrics": deterministic_metrics.to_dict(),
            "rubric_results": rubric_results,
            "judge_summary": judge_summary
        }
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Validate output
        if not self.validate_json_output(str(output_path)):
            raise ValueError(f"Generated trace_eval.json is not valid JSON: {output_path}")
        
        return str(output_path)
    
    def write_results_json(
        self,
        run_id: str,
        rubrics_config: Dict[str, Any],
        judge_config: Dict[str, Any],
        input_data: Dict[str, Any],
        deterministic_metrics: MetricsResult,
        rubric_results: Dict[str, Any],
        judge_disagreements: List[Dict[str, Any]],
        artifact_paths: Dict[str, str],
        execution_stats: Dict[str, Any]
    ) -> str:
        """
        Write results.json with stable canonical structure.
        
        Args:
            run_id: Run identifier from NormalizedRun
            rubrics_config: Rubrics configuration for hashing
            judge_config: Judge configuration for hashing
            input_data: Input NormalizedRun for hashing
            deterministic_metrics: Full metrics object
            rubric_results: Per-rubric, per-turn structure with aggregated scores
            judge_disagreements: List of high-risk disagreements
            artifact_paths: Paths to judge_runs.jsonl and trace_eval.json
            execution_stats: Execution statistics (total_jobs, completed_jobs, etc.)
            
        Returns:
            Path to written results.json file
            
        Raises:
            IOError: If file cannot be written
            ValueError: If output validation fails
        """
        output_path = self.output_dir / "results.json"
        
        # Compute fingerprints for cross-run comparison
        rubrics_hash = self.compute_config_hash(rubrics_config)
        judge_config_hash = self.compute_config_hash(judge_config)
        input_hash = self.compute_config_hash(input_data)
        
        # Compute artifact hashes for integrity verification
        artifact_hashes = {}
        for artifact_name, artifact_path in artifact_paths.items():
            try:
                artifact_hashes[artifact_name] = self.compute_file_hash(artifact_path)
            except (FileNotFoundError, IOError) as e:
                # Artifact may not exist yet (e.g., trace_eval.json written after results.json)
                artifact_hashes[artifact_name] = None
        
        # Build output structure matching results.schema.json
        output_data = {
            "format_version": self.FORMAT_VERSION,
            "run_id": run_id,
            "rubrics_hash": rubrics_hash,
            "judge_config_hash": judge_config_hash,
            "input_hash": input_hash,
            "deterministic_metrics": deterministic_metrics.to_dict(),
            "rubric_results": rubric_results,
            "judge_disagreements": judge_disagreements,
            "artifact_paths": artifact_paths,
            "artifact_hashes": artifact_hashes,
            "execution_stats": execution_stats
        }
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Validate output
        if not self.validate_json_output(str(output_path)):
            raise ValueError(f"Generated results.json is not valid JSON: {output_path}")
        
        return str(output_path)
    
    def compute_file_hash(self, file_path: str) -> str:
        """
        Compute SHA-256 hash of a file.
        
        Args:
            file_path: Path to file to hash
            
        Returns:
            Hexadecimal SHA-256 hash string
            
        Raises:
            FileNotFoundError: If file does not exist
            IOError: If file cannot be read
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def compute_config_hash(self, config_data: Dict[str, Any]) -> str:
        """
        Compute SHA-256 hash of configuration data.
        
        Canonicalizes non-JSON-native types before hashing to ensure stability.
        
        Args:
            config_data: Configuration dictionary to hash
            
        Returns:
            Hexadecimal SHA-256 hash string
        """
        # Canonicalize to JSON-serializable primitives
        canonical_data = self._canonicalize_for_json(config_data)
        
        # Serialize config to canonical JSON (sorted keys, no whitespace)
        config_json = json.dumps(canonical_data, sort_keys=True, separators=(',', ':'))
        
        # Compute SHA-256 hash
        sha256_hash = hashlib.sha256(config_json.encode('utf-8'))
        
        return sha256_hash.hexdigest()
    
    def validate_json_output(self, file_path: str) -> bool:
        """
        Validate that output file is valid JSON.
        
        Args:
            file_path: Path to JSON file to validate
            
        Returns:
            True if file is valid JSON, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
            return True
        except (json.JSONDecodeError, IOError):
            return False
    
    def validate_against_schema(
        self, 
        file_path: str, 
        schema_path: str
    ) -> tuple[bool, Optional[str]]:
        """
        Validate JSON output against JSON Schema.
        
        Optional validation for production use. Requires jsonschema package.
        
        Args:
            file_path: Path to JSON file to validate
            schema_path: Path to JSON Schema file
            
        Returns:
            Tuple of (is_valid, error_message)
            
        Note:
            Returns (True, None) if jsonschema package not available
        """
        try:
            import jsonschema
            from jsonschema import validate, ValidationError
        except ImportError:
            # jsonschema not available, skip validation
            return (True, None)
        
        try:
            # Load data and schema
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            
            # Validate
            validate(instance=data, schema=schema)
            return (True, None)
            
        except ValidationError as e:
            return (False, f"Schema validation failed: {e.message}")
        except (IOError, json.JSONDecodeError) as e:
            return (False, f"Failed to load file or schema: {str(e)}")
    
    def build_rubric_results_for_trace_eval(
        self,
        cross_judge_results: List[CrossJudgeResult]
    ) -> List[Dict[str, Any]]:
        """
        Build rubric_results array for trace_eval.json.
        
        Args:
            cross_judge_results: List of cross-judge aggregation results
            
        Returns:
            List of rubric result dictionaries matching schema
        """
        rubric_results = []
        
        for cross_result in cross_judge_results:
            result_dict = {
                "rubric_id": cross_result.rubric_id,
                "scope": "turn" if cross_result.turn_id else "run",
                "turn_id": cross_result.turn_id,
                "within_judge_results": [
                    wj_result.to_dict() 
                    for wj_result in cross_result.individual_judge_results
                ],
                "cross_judge_result": {
                    "weighted_vote": cross_result.weighted_vote,
                    "weighted_average": cross_result.weighted_average,
                    "disagreement_signal": cross_result.disagreement_signal,
                    "high_risk_flag": cross_result.high_risk_flag,
                    "judge_count": cross_result.judge_count,
                    "scoring_type": cross_result.scoring_type,
                    "mixed_types_error": cross_result.mixed_types_error,
                    "scale_warning": cross_result.scale_warning
                }
            }
            
            rubric_results.append(result_dict)
        
        return rubric_results
    
    def build_rubric_results_for_results_json(
        self,
        cross_judge_results: List[CrossJudgeResult]
    ) -> Dict[str, Any]:
        """
        Build rubric_results object for results.json.
        
        Per-rubric, per-turn structure with aggregated scores.
        
        Args:
            cross_judge_results: List of cross-judge aggregation results
            
        Returns:
            Dictionary mapping rubric_id to turn results
        """
        rubric_results = {}
        
        for cross_result in cross_judge_results:
            rubric_id = cross_result.rubric_id
            
            if rubric_id not in rubric_results:
                rubric_results[rubric_id] = {
                    "scope": "turn" if cross_result.turn_id else "run",
                    "turns": {}
                }
            
            # Use turn_id or "run" for run-level rubrics
            turn_key = cross_result.turn_id if cross_result.turn_id else "run"
            
            rubric_results[rubric_id]["turns"][turn_key] = {
                "cross_judge_score": (
                    cross_result.weighted_vote 
                    if cross_result.scoring_type == "categorical" 
                    else cross_result.weighted_average
                ),
                "disagreement_signal": cross_result.disagreement_signal,
                "high_risk_flag": cross_result.high_risk_flag,
                "judge_count": cross_result.judge_count,
                "scoring_type": cross_result.scoring_type,
                "mixed_types_error": cross_result.mixed_types_error,
                "scale_warning": cross_result.scale_warning
            }
        
        return rubric_results
    
    def extract_judge_disagreements(
        self,
        cross_judge_results: List[CrossJudgeResult],
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract high-risk disagreements from cross-judge results.
        
        Args:
            cross_judge_results: List of cross-judge aggregation results
            threshold: Optional disagreement threshold override. If None, uses high_risk_flag from results.
            
        Returns:
            List of disagreement records with rubric_id, turn_id, disagreement_score
        """
        disagreements = []
        
        for cross_result in cross_judge_results:
            # If threshold provided, compute high-risk based on it
            # Otherwise, trust the high_risk_flag from aggregation
            is_high_risk = (
                cross_result.disagreement_signal > threshold 
                if threshold is not None 
                else cross_result.high_risk_flag
            )
            
            if is_high_risk:
                disagreement = {
                    "rubric_id": cross_result.rubric_id,
                    "turn_id": cross_result.turn_id,
                    "disagreement_score": cross_result.disagreement_signal
                }
                disagreements.append(disagreement)
        
        return disagreements
    
    def build_judge_summary(
        self,
        total_jobs: int,
        successful_jobs: int,
        failed_jobs: int,
        judge_count: int
    ) -> Dict[str, Any]:
        """
        Build judge_summary object for trace_eval.json.
        
        Args:
            total_jobs: Total number of judge jobs
            successful_jobs: Number of successful jobs
            failed_jobs: Number of failed jobs
            judge_count: Number of judges used
            
        Returns:
            Judge summary dictionary
        """
        return {
            "total_jobs": total_jobs,
            "successful_jobs": successful_jobs,
            "failed_jobs": failed_jobs,
            "judge_count": judge_count
        }
    
    def build_execution_stats(
        self,
        total_jobs: int,
        completed_jobs: int,
        failed_jobs: int,
        failed_job_ids: List[str],
        duration_seconds: float
    ) -> Dict[str, Any]:
        """
        Build execution_stats object for results.json.
        
        Args:
            total_jobs: Total number of jobs
            completed_jobs: Number of completed jobs
            failed_jobs: Number of failed jobs
            failed_job_ids: List of failed job IDs
            duration_seconds: Total execution duration in seconds
            
        Returns:
            Execution stats dictionary
        """
        failure_ratio = failed_jobs / total_jobs if total_jobs > 0 else 0.0
        
        return {
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "failure_ratio": failure_ratio,
            "failed_job_ids": failed_job_ids,
            "duration_seconds": duration_seconds
        }
