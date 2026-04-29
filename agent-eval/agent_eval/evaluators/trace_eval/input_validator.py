# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Input validation for NormalizedRun files.

This module validates NormalizedRun files against the normalized_run.schema.json
schema and extracts required fields for evaluation.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

try:
    from jsonschema import ValidationError as JsonSchemaValidationError, FormatChecker
except ImportError:
    raise ImportError(
        "jsonschema is required for input validation. "
        "Install it with: pip install jsonschema"
    )


# PII patterns for redaction in error messages
PII_PATTERNS = [
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL]'),
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN]'),
    (re.compile(r'\b\d{16}\b'), '[CARD]'),
    (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), '[PHONE]'),
    (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), '[IP]'),
    # JWT tokens (xxx.yyy.zzz format)
    (re.compile(r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b'), '[JWT]'),
    # AWS access keys (AKIA for standard, ASIA for STS)
    (re.compile(r'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b'), '[AWS_KEY]'),
    # Bearer tokens (any non-whitespace after Bearer)
    (re.compile(r'\bBearer\s+\S+'), '[BEARER_TOKEN]'),
    (re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', re.IGNORECASE), '[UUID]'),
]


class ValidationError(Exception):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


@dataclass
class ExtractedFields:
    """Container for extracted fields from NormalizedRun."""
    run_id: str
    turns: List[Dict[str, Any]]
    adapter_stats: Dict[str, Any]
    metadata: Dict[str, Any]


class InputValidator:
    """Validates NormalizedRun files against schema and extracts required fields."""
    
    # Default limits for input validation
    DEFAULT_MAX_FILE_SIZE_MB = 100
    DEFAULT_MAX_TURNS = 10000
    DEFAULT_MAX_STEPS_PER_TURN = 1000
    DEFAULT_MAX_TOTAL_STEPS = 100000  # Across entire run
    
    def __init__(
        self,
        schema_path: Optional[str] = None,
        enable_format_checking: bool = True,
        max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_steps_per_turn: int = DEFAULT_MAX_STEPS_PER_TURN,
        max_total_steps: int = DEFAULT_MAX_TOTAL_STEPS
    ):
        """
        Initialize validator with schema.
        
        Args:
            schema_path: Path to normalized_run.schema.json. If None, uses default location.
            enable_format_checking: Enable strict format validation (date-time, uri, etc.)
            max_file_size_mb: Maximum file size in MB (default: 100)
            max_turns: Maximum number of turns allowed (default: 10000)
            max_steps_per_turn: Maximum steps per turn (default: 1000)
            max_total_steps: Maximum total steps across entire run (default: 100000)
        """
        # Validate constructor arguments
        if max_file_size_mb <= 0:
            raise ValueError(f"max_file_size_mb must be positive, got {max_file_size_mb}")
        if max_turns <= 0:
            raise ValueError(f"max_turns must be positive, got {max_turns}")
        if max_steps_per_turn <= 0:
            raise ValueError(f"max_steps_per_turn must be positive, got {max_steps_per_turn}")
        if max_total_steps <= 0:
            raise ValueError(f"max_total_steps must be positive, got {max_total_steps}")
        
        if schema_path is None:
            # Default to schema in agent_eval/schemas/
            default_path = Path(__file__).parent.parent.parent / "schemas" / "normalized_run.schema.json"
            schema_path = str(default_path)
        
        self.schema_path = Path(schema_path)
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
        
        # Check for external $ref (not supported)
        self._check_external_refs(self.schema)
        
        # Determine validator class based on $schema field (or default to Draft7)
        schema_version = self.schema.get("$schema", "")
        try:
            if "draft-07" in schema_version or not schema_version:
                from jsonschema.validators import Draft7Validator as ValidatorClass
            elif "2019-09" in schema_version:
                from jsonschema.validators import Draft201909Validator as ValidatorClass
            elif "2020-12" in schema_version:
                from jsonschema.validators import Draft202012Validator as ValidatorClass
            else:
                # Default to Draft7 for unknown versions
                from jsonschema.validators import Draft7Validator as ValidatorClass
        except ImportError as e:
            raise ImportError(
                f"Required jsonschema validator not available for schema version '{schema_version}'. "
                f"Ensure jsonschema is up to date: {e}"
            )
        
        # Create validator instance (reusable, better performance)
        format_checker = FormatChecker() if enable_format_checking else None
        self.validator = ValidatorClass(
            schema=self.schema,
            format_checker=format_checker
        )
        
        # Note: External $ref resolution across files is not supported.
        # Schema must be self-contained or use bundled definitions.
        # If your schema uses $ref to external files, pre-bundle them into a single schema.
        
        # Store limits
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.max_turns = max_turns
        self.max_steps_per_turn = max_steps_per_turn
        self.max_total_steps = max_total_steps
    
    def _check_external_refs(self, schema: Dict[str, Any], path: str = "") -> None:
        """
        Check for external $ref that would fail at runtime.
        
        Args:
            schema: Schema object to check
            path: Current path in schema (for error messages)
            
        Raises:
            ValidationError: If external $ref detected
        """
        if isinstance(schema, dict):
            if "$ref" in schema:
                ref = schema["$ref"]
                # Allow internal refs (#/...) and JSON Schema meta-schema refs (http://json-schema.org/...)
                if not ref.startswith("#") and not ref.startswith("http://json-schema.org/") and not ref.startswith("https://json-schema.org/"):
                    raise ValidationError(
                        f"External $ref not supported at {path or 'root'}: {ref}. "
                        "Schema must be self-contained or use bundled definitions (#/definitions/...).",
                        details={"ref": ref, "path": path}
                    )
            
            # Recursively check nested objects
            for key, value in schema.items():
                self._check_external_refs(value, f"{path}.{key}" if path else key)
        elif isinstance(schema, list):
            for i, item in enumerate(schema):
                self._check_external_refs(item, f"{path}[{i}]")
    
    def _redact_pii(self, text: str) -> str:
        """
        Redact potential PII from error messages.
        
        Args:
            text: Text that may contain PII
            
        Returns:
            Text with PII patterns replaced
        """
        for pattern, replacement in PII_PATTERNS:
            text = pattern.sub(replacement, text)
        return text
    
    def validate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input against normalized_run.schema.json.
        
        Args:
            input_data: Raw JSON data to validate
            
        Returns:
            Validated NormalizedRun data
            
        Raises:
            ValidationError: With descriptive error message if validation fails
        """
        # Type guard: ensure input is a dict
        if not isinstance(input_data, dict):
            raise ValidationError(
                f"Input must be a JSON object (dict), got {type(input_data).__name__}",
                details={"input_type": type(input_data).__name__}
            )
        
        # Check size limits
        self._check_size_limits(input_data)
        
        # Collect all validation errors using iter_errors for better diagnostics
        # Sort errors for deterministic output (convert paths to strings for safe comparison)
        errors = sorted(
            self.validator.iter_errors(input_data),
            key=lambda e: (tuple(map(str, e.path)), tuple(map(str, e.schema_path)))
        )
        
        if errors:
            # Take the first error for the main message
            first_error = errors[0]
            error_path = " -> ".join(str(p) for p in first_error.path) if first_error.path else "root"
            
            # Include failing value snippet (redacted if contains PII)
            failing_value = str(first_error.instance)[:100] if first_error.instance is not None else "null"
            if len(str(first_error.instance)) > 100:
                failing_value += "..."
            failing_value = self._redact_pii(failing_value)
            
            # Redact PII from error message as well
            error_message = self._redact_pii(first_error.message)
            
            # Build detailed error info
            error_details = {
                "path": list(first_error.path),
                "validator": first_error.validator,
                "validator_value": first_error.validator_value,
                "schema_path": list(first_error.schema_path),
                "failing_value": failing_value,
                "total_errors": len(errors)
            }
            
            # Include additional errors if multiple (with PII redaction)
            if len(errors) > 1:
                error_details["additional_errors"] = [
                    {
                        "path": " -> ".join(str(p) for p in e.path) if e.path else "root",
                        "message": self._redact_pii(e.message)
                    }
                    for e in errors[1:6]  # Limit to 5 additional errors
                ]
            
            raise ValidationError(
                f"Schema validation failed at {error_path}: {error_message}",
                details=error_details
            )
        
        return input_data
    
    def _check_size_limits(self, input_data: Dict[str, Any]) -> None:
        """
        Check input against size limits.
        
        Args:
            input_data: Input data to check
            
        Raises:
            ValidationError: If limits are exceeded
        """
        turns = input_data.get("turns", [])
        
        # Type guard: ensure turns is a list
        if not isinstance(turns, list):
            raise ValidationError(
                f"Field 'turns' must be a list, got {type(turns).__name__}",
                details={"turns_type": type(turns).__name__}
            )
        
        if len(turns) > self.max_turns:
            raise ValidationError(
                f"Too many turns: {len(turns)} exceeds limit of {self.max_turns}",
                details={"turn_count": len(turns), "max_turns": self.max_turns}
            )
        
        total_steps = 0
        for i, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue  # Schema validation will catch this
            
            steps = turn.get("steps", [])
            if not isinstance(steps, list):
                continue  # Schema validation will catch this
            
            step_count = len(steps)
            total_steps += step_count
            
            if step_count > self.max_steps_per_turn:
                raise ValidationError(
                    f"Too many steps in turn {i}: {step_count} exceeds limit of {self.max_steps_per_turn}",
                    details={
                        "turn_index": i,
                        "turn_id": turn.get("turn_id"),
                        "step_count": step_count,
                        "max_steps": self.max_steps_per_turn
                    }
                )
        
        # Check total steps across entire run
        if total_steps > self.max_total_steps:
            raise ValidationError(
                f"Too many total steps: {total_steps} exceeds limit of {self.max_total_steps}",
                details={
                    "total_steps": total_steps,
                    "max_total_steps": self.max_total_steps,
                    "turn_count": len(turns)
                }
            )
    
    def extract_fields(self, normalized_run: Dict[str, Any]) -> ExtractedFields:
        """
        Extract required fields from validated NormalizedRun.
        
        Args:
            normalized_run: Validated NormalizedRun data
            
        Returns:
            ExtractedFields containing run_id, turns, adapter_stats, metadata
            
        Raises:
            ValidationError: If required fields are missing
        """
        try:
            run_id = normalized_run["run_id"]
            turns = normalized_run["turns"]
            adapter_stats = normalized_run["adapter_stats"]
            metadata = normalized_run["metadata"]  # Required by schema
            
            return ExtractedFields(
                run_id=run_id,
                turns=turns,
                adapter_stats=adapter_stats,
                metadata=metadata
            )
        except KeyError as e:
            raise ValidationError(
                f"Missing required field: {e}",
                details={"missing_field": str(e)}
            )
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a NormalizedRun file.
        
        Args:
            file_path: Path to JSON file containing NormalizedRun
            
        Returns:
            Validated NormalizedRun data
            
        Raises:
            ValidationError: If file cannot be read or validation fails
            FileNotFoundError: If file does not exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        # Check file size before loading
        file_size = path.stat().st_size
        if file_size > self.max_file_size_bytes:
            raise ValidationError(
                f"File too large: {file_size / (1024*1024):.2f}MB exceeds limit of {self.max_file_size_bytes / (1024*1024):.0f}MB",
                details={
                    "file_size_bytes": file_size,
                    "max_size_bytes": self.max_file_size_bytes
                }
            )
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValidationError(
                f"Invalid JSON in input file: {e}",
                details={"file": file_path, "error": str(e)}
            )
        except UnicodeDecodeError as e:
            raise ValidationError(
                f"File encoding error (expected UTF-8): {e}",
                details={"file": file_path, "error": str(e)}
            )
        
        return self.validate(input_data)
