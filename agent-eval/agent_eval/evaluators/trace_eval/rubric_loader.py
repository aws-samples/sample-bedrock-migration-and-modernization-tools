# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Rubric Loader for Agent Traces Evaluator

This module handles loading, merging, and validating rubric configurations.
It supports both default rubrics (shipped with the system) and user-provided
rubric overrides.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import yaml

from .selector_validation import validate_turn_scoped_selectors, SelectorValidationError


class RubricValidationError(Exception):
    """Raised when rubric validation fails."""
    pass


class Rubric:
    """Represents a single evaluation rubric."""
    
    def __init__(self, data: Dict[str, Any]):
        self.rubric_id: str = data.get("rubric_id", "")
        self.description: str = data.get("description", "")
        self.weight: float = data.get("weight", 1.0)
        self.severity: str = data.get("severity", "medium")
        self.enabled: bool = data.get("enabled", True)
        self.scoring_scale: Dict[str, Any] = data.get("scoring_scale", {})
        self.aggregation_type: str = data.get("aggregation_type", "")
        self.run_aggregation_policy: str = data.get("run_aggregation_policy", "standard")
        self.evaluation_granularity: Optional[str] = data.get("evaluation_granularity")
        self.requires_llm_judge: bool = data.get("requires_llm_judge", False)
        self.evaluation_instructions: Optional[str] = data.get("evaluation_instructions")
        self.evidence_selectors: List[str] = data.get("evidence_selectors", [])
        self.evidence_budget: int = data.get("evidence_budget", 10000)  # Will be overridden by loader
        self.scope: str = data.get("scope", "")
        self.scope_behavior: Optional[str] = data.get("scope_behavior")
        self.deterministic_source: Optional[str] = data.get("deterministic_source")
        self.redact_fields: Optional[List[str]] = data.get("redact_fields")
        self.sample_config: Optional[Dict[str, Any]] = data.get("sample_config")  # For sample_turns behavior
        self._raw_data = data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rubric to dictionary representation."""
        return self._raw_data
    
    def __repr__(self) -> str:
        return f"Rubric(rubric_id={self.rubric_id}, scope={self.scope}, requires_llm_judge={self.requires_llm_judge}, enabled={self.enabled})"


class RubricLoader:
    """Loads and validates rubric configurations."""
    
    # Class constants for validation
    DEFAULT_EVIDENCE_BUDGET = 10000  # Fallback if not in YAML
    DEFAULT_TURN_SELECTOR_MODE = "narrow_to_turn"  # Fallback if not specified
    VALID_SEVERITIES = ["low", "medium", "high", "critical"]
    VALID_SCOPE_BEHAVIORS = ["per_turn", "aggregate_all_turns", "sample_turns"]
    VALID_RUN_AGGREGATION_POLICIES = ["standard", "max_severity_escalation"]
    VALID_AGGREGATION_TYPES = ["median", "mean", "majority_vote", "deterministic"]
    VALID_TURN_SELECTOR_MODES = ["narrow_to_turn", "full_context"]
    VALID_EVALUATION_GRANULARITIES = ["per_turn_then_run", "run_only"]
    
    def __init__(self):
        self.default_rubrics_path = self._get_default_rubrics_path()
        self.file_default_evidence_budget = self.DEFAULT_EVIDENCE_BUDGET
        self.default_version = None  # Will be set when loading defaults
        self.turn_selector_mode = self.DEFAULT_TURN_SELECTOR_MODE  # Fallback, overridden by YAML
    
    def _get_default_rubrics_path(self) -> Path:
        """Get the path to default_rubrics.yaml using explicit package resource location."""
        # Path is relative to this file's location
        current_dir = Path(__file__).parent
        yaml_path = current_dir / "default_rubrics.yaml"
        
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Default rubrics file not found at expected location: {yaml_path}"
            )
        
        return yaml_path
    
    def _check_duplicate_rubric_ids(self, rubrics: List[Dict[str, Any]], source: str) -> None:
        """Check for duplicate rubric IDs in a single file."""
        seen_ids: Set[str] = set()
        for rubric_data in rubrics:
            rubric_id = rubric_data.get("rubric_id", "")
            if rubric_id in seen_ids:
                raise RubricValidationError(
                    f"Duplicate rubric_id '{rubric_id}' found in {source}"
                )
            seen_ids.add(rubric_id)
    
    def _validate_schema_block(self, schema_data: Any, source: str) -> None:
        """Validate the schema block if present in YAML."""
        if not isinstance(schema_data, dict):
            raise RubricValidationError(
                f"schema block in {source} must be a dictionary, got {type(schema_data)}"
            )
        
        # Schema block contains field definitions with type/values/min/max
        # Validate each field definition has proper structure
        for field_name, field_def in schema_data.items():
            if not isinstance(field_def, dict):
                raise RubricValidationError(
                    f"schema.{field_name} in {source} must be a dictionary, got {type(field_def)}"
                )
            
            if "type" not in field_def:
                raise RubricValidationError(
                    f"schema.{field_name} in {source} must contain 'type' field"
                )
            
            field_type = field_def["type"]
            if field_type not in ["enum", "numeric", "string"]:
                raise RubricValidationError(
                    f"schema.{field_name}.type in {source} must be 'enum', 'numeric', or 'string', got '{field_type}'"
                )
            
            # Validate enum fields have values
            if field_type == "enum" and "values" not in field_def:
                raise RubricValidationError(
                    f"schema.{field_name} in {source} with type='enum' must contain 'values' field"
                )
            
            # Validate numeric fields have min/max
            if field_type == "numeric":
                if "min" not in field_def or "max" not in field_def:
                    raise RubricValidationError(
                        f"schema.{field_name} in {source} with type='numeric' must contain 'min' and 'max' fields"
                    )
    
    def _validate_version_compatibility(self, user_version: str, source: str) -> None:
        """Validate user rubrics version is compatible with default version."""
        if not self.default_version:
            return  # No default loaded yet
        
        # Parse versions (assuming semantic versioning: major.minor.patch)
        try:
            default_parts = self.default_version.split('.')
            user_parts = user_version.split('.')
            
            default_major = int(default_parts[0])
            default_minor = int(default_parts[1]) if len(default_parts) > 1 else 0
            
            user_major = int(user_parts[0])
            user_minor = int(user_parts[1]) if len(user_parts) > 1 else 0
            
            # Reject if major version mismatch
            if default_major != user_major:
                raise RubricValidationError(
                    f"Version mismatch in {source}: user version {user_version} incompatible with default version {self.default_version} (major version differs)"
                )
            
            # Warn if minor version mismatch (but allow)
            if default_minor != user_minor:
                import warnings
                warnings.warn(
                    f"Version mismatch in {source}: user version {user_version} differs from default version {self.default_version} (minor version differs)",
                    UserWarning
                )
        except (ValueError, IndexError) as e:
            raise RubricValidationError(
                f"Invalid version format in {source}: {user_version} (expected semantic versioning like '1.0.0')"
            )
    
    def load_default_rubrics(self) -> List[Rubric]:
        """
        Load 8 default rubrics from default_rubrics.yaml.
        
        Returns:
            List of Rubric objects
            
        Raises:
            FileNotFoundError: If default_rubrics.yaml is missing
            RubricValidationError: If default rubrics are invalid
        """
        with open(self.default_rubrics_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if not data or "rubrics" not in data:
            raise RubricValidationError(
                "Default rubrics file must contain 'rubrics' key"
            )
        
        # Validate version field
        if "version" not in data:
            raise RubricValidationError(
                "Default rubrics file must contain 'version' field"
            )
        
        self.default_version = data["version"]
        
        # Validate and read default_evidence_budget field
        if "default_evidence_budget" not in data:
            raise RubricValidationError(
                "Default rubrics file must contain 'default_evidence_budget' field"
            )
        
        # Validate default_evidence_budget type and value
        if not isinstance(data["default_evidence_budget"], int) or data["default_evidence_budget"] <= 0:
            raise RubricValidationError(
                f"default_evidence_budget must be a positive integer, got {data['default_evidence_budget']}"
            )
        
        self.file_default_evidence_budget = data["default_evidence_budget"]
        
        # Validate turn_selector_mode field
        if "turn_selector_mode" not in data:
            raise RubricValidationError(
                "Default rubrics file must contain 'turn_selector_mode' field"
            )
        
        if data["turn_selector_mode"] not in self.VALID_TURN_SELECTOR_MODES:
            raise RubricValidationError(
                f"turn_selector_mode must be one of {self.VALID_TURN_SELECTOR_MODES}, got '{data['turn_selector_mode']}'"
            )
        
        # Store turn_selector_mode for use by job_builder
        self.turn_selector_mode = data["turn_selector_mode"]
        
        # Validate schema block if present
        if "schema" in data:
            self._validate_schema_block(data["schema"], "default_rubrics.yaml")
        
        # Check for duplicate rubric IDs
        self._check_duplicate_rubric_ids(data["rubrics"], "default_rubrics.yaml")
        
        # Validate exactly 8 default rubrics
        if len(data["rubrics"]) != 8:
            raise RubricValidationError(
                f"Default rubrics file must contain exactly 8 rubrics, found {len(data['rubrics'])}"
            )
        
        rubrics = []
        for rubric_data in data["rubrics"]:
            # Apply file's default_evidence_budget if not specified
            if "evidence_budget" not in rubric_data:
                rubric_data["evidence_budget"] = self.file_default_evidence_budget
            
            rubric = Rubric(rubric_data)
            self.validate_rubric(rubric)
            rubrics.append(rubric)
        
        return rubrics
    
    def load_user_rubrics(self, path: str) -> List[Rubric]:
        """
        Load user-provided rubrics from file.
        
        Args:
            path: Path to user rubrics YAML file
            
        Returns:
            List of Rubric objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            RubricValidationError: If rubrics are invalid
        """
        user_path = Path(path)
        if not user_path.exists():
            raise FileNotFoundError(f"User rubrics file not found: {path}")
        
        with open(user_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if not data or "rubrics" not in data:
            raise RubricValidationError(
                "User rubrics file must contain 'rubrics' key"
            )
        
        # Validate required top-level fields for user rubrics
        if "version" not in data:
            raise RubricValidationError(
                f"User rubrics file {path} must contain 'version' field"
            )
        
        # Validate version compatibility
        self._validate_version_compatibility(data["version"], f"user rubrics file {path}")
        
        # Validate turn_selector_mode if present (optional for user rubrics)
        if "turn_selector_mode" in data:
            if data["turn_selector_mode"] not in self.VALID_TURN_SELECTOR_MODES:
                raise RubricValidationError(
                    f"turn_selector_mode in {path} must be one of {self.VALID_TURN_SELECTOR_MODES}, got '{data['turn_selector_mode']}'"
                )
            # Apply user turn_selector_mode override
            self.turn_selector_mode = data["turn_selector_mode"]
        
        # Validate schema block if present
        if "schema" in data:
            self._validate_schema_block(data["schema"], f"user rubrics file {path}")
        
        # User rubrics can optionally override default_evidence_budget
        user_default_budget = data.get("default_evidence_budget", self.file_default_evidence_budget)
        
        # Validate user_default_budget if provided
        if "default_evidence_budget" in data:
            if not isinstance(data["default_evidence_budget"], int) or data["default_evidence_budget"] <= 0:
                raise RubricValidationError(
                    f"default_evidence_budget in {path} must be a positive integer, got {data['default_evidence_budget']}"
                )
        
        # Check for duplicate rubric IDs
        self._check_duplicate_rubric_ids(data["rubrics"], f"user rubrics file {path}")
        
        rubrics = []
        for rubric_data in data["rubrics"]:
            # Apply user file's default_evidence_budget if not specified
            if "evidence_budget" not in rubric_data:
                rubric_data["evidence_budget"] = user_default_budget
            
            rubric = Rubric(rubric_data)
            self.validate_rubric(rubric)
            rubrics.append(rubric)
        
        return rubrics
    
    def merge_rubrics(
        self, 
        default: List[Rubric], 
        user: List[Rubric]
    ) -> List[Rubric]:
        """
        Merge rubrics by rubric_id.
        User rubrics override defaults with same ID.
        Only enabled rubrics are included in the result.
        
        Args:
            default: List of default rubrics
            user: List of user rubrics
            
        Returns:
            Merged list of enabled rubrics only (stable order preserved)
        """
        # Create a dictionary of default rubrics by ID (compute once)
        default_ids_set = {rubric.rubric_id for rubric in default}
        merged_dict: Dict[str, Rubric] = {
            rubric.rubric_id: rubric for rubric in default if rubric.enabled
        }
        
        # Track user rubric order for stable output
        user_rubric_order = []
        
        # Override with user rubrics (same ID) or add new ones
        for user_rubric in user:
            if user_rubric.enabled:
                merged_dict[user_rubric.rubric_id] = user_rubric
                if user_rubric.rubric_id not in default_ids_set:
                    user_rubric_order.append(user_rubric.rubric_id)
            elif user_rubric.rubric_id in merged_dict:
                # User explicitly disabled a default rubric
                del merged_dict[user_rubric.rubric_id]
        
        # Return as list, maintaining stable order (defaults first, then new user rubrics in file order)
        result = []
        
        # Add all default rubric IDs in original order (with user overrides if present)
        for rubric in default:
            if rubric.rubric_id in merged_dict:
                result.append(merged_dict[rubric.rubric_id])
        
        # Add any new user rubrics not in defaults (in user file order)
        for rubric_id in user_rubric_order:
            if rubric_id in merged_dict:
                result.append(merged_dict[rubric_id])
        
        # Validate no duplicate rubric_ids in final merged result
        final_ids = [r.rubric_id for r in result]
        if len(final_ids) != len(set(final_ids)):
            raise RubricValidationError(
                "Duplicate rubric_ids found in merged rubrics after combining default and user rubrics"
            )
        
        return result
    
    def validate_rubric(self, rubric: Rubric) -> None:
        """
        Validate rubric structure.
        
        Args:
            rubric: Rubric to validate
            
        Raises:
            RubricValidationError: If rubric is invalid
        """
        # Validate required fields
        if not rubric.rubric_id:
            raise RubricValidationError("Rubric must have rubric_id")
        
        if not rubric.description:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} must have description"
            )
        
        # Validate enabled is bool
        if not isinstance(rubric.enabled, bool):
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} enabled must be a boolean, got {type(rubric.enabled).__name__}"
            )
        
        # Validate requires_llm_judge is bool
        if not isinstance(rubric.requires_llm_judge, bool):
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} requires_llm_judge must be a boolean, got {type(rubric.requires_llm_judge).__name__}"
            )
        
        if not rubric.scoring_scale:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} must have scoring_scale"
            )
        
        if not rubric.evidence_selectors:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} must have evidence_selectors"
            )
        
        # Validate evidence_selectors are non-empty strings
        for i, selector in enumerate(rubric.evidence_selectors):
            if not isinstance(selector, str) or not selector.strip():
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} evidence_selector at index {i} must be a non-empty string"
                )
        
        # Validate turn-scoped selectors don't use run-level paths
        # This must happen after evidence_selectors validation and before scope validation
        try:
            validate_turn_scoped_selectors(
                rubric_id=rubric.rubric_id,
                scope=rubric.scope,
                turn_selector_mode=self.turn_selector_mode,
                evidence_selectors=rubric.evidence_selectors
            )
        except SelectorValidationError as e:
            # Re-raise as RubricValidationError to maintain consistent error type
            raise RubricValidationError(str(e)) from e
        
        if not rubric.scope:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} must have scope"
            )
        
        # Validate scope value
        if rubric.scope not in ["turn", "run"]:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} scope must be 'turn' or 'run', got '{rubric.scope}'"
            )
        
        # Validate scope_behavior is required
        if not rubric.scope_behavior:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} must have scope_behavior"
            )
        
        if rubric.scope_behavior not in self.VALID_SCOPE_BEHAVIORS:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} scope_behavior must be one of {self.VALID_SCOPE_BEHAVIORS}, got '{rubric.scope_behavior}'"
            )
        
        # Validate scope_behavior validity cross-checked with scope
        if rubric.scope == "turn" and rubric.scope_behavior not in ["per_turn"]:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} with scope='turn' must use scope_behavior='per_turn', got '{rubric.scope_behavior}'"
            )
        
        if rubric.scope == "run" and rubric.scope_behavior not in ["aggregate_all_turns", "sample_turns"]:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} with scope='run' must use scope_behavior='aggregate_all_turns' or 'sample_turns', got '{rubric.scope_behavior}'"
            )
        
        # Validate sample_turns scope_behavior has required parameters
        if rubric.scope_behavior == "sample_turns":
            # Check for sample configuration in raw data
            if "sample_config" not in rubric._raw_data:
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} with scope_behavior='sample_turns' must specify 'sample_config' with 'sample_n' and 'strategy'"
                )
            
            sample_config = rubric._raw_data["sample_config"]
            if not isinstance(sample_config, dict):
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} sample_config must be a dictionary"
                )
            
            if "sample_n" not in sample_config or "strategy" not in sample_config:
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} sample_config must contain 'sample_n' and 'strategy' fields"
                )
            
            if not isinstance(sample_config["sample_n"], int) or sample_config["sample_n"] <= 0:
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} sample_config.sample_n must be a positive integer, got {sample_config['sample_n']}"
                )
            
            valid_strategies = ["random", "first_n", "last_n", "evenly_spaced"]
            if sample_config["strategy"] not in valid_strategies:
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} sample_config.strategy must be one of {valid_strategies}, got '{sample_config['strategy']}'"
                )
            
            # Store sample_config on rubric for job_builder to use
            rubric.sample_config = sample_config
        
        if rubric.run_aggregation_policy not in self.VALID_RUN_AGGREGATION_POLICIES:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} run_aggregation_policy must be one of {self.VALID_RUN_AGGREGATION_POLICIES}, got '{rubric.run_aggregation_policy}'"
            )
        
        # Validate aggregation_type
        if rubric.aggregation_type not in self.VALID_AGGREGATION_TYPES:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} aggregation_type must be one of {self.VALID_AGGREGATION_TYPES}, got '{rubric.aggregation_type}'"
            )
        
        # Validate evaluation_granularity if present
        if rubric.evaluation_granularity:
            if rubric.evaluation_granularity not in self.VALID_EVALUATION_GRANULARITIES:
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} evaluation_granularity must be one of {self.VALID_EVALUATION_GRANULARITIES}, got '{rubric.evaluation_granularity}'"
                )
        
        # Validate weight with bounds
        if rubric.weight < 0.0 or rubric.weight > 5.0:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} weight must be between 0.0 and 5.0, got {rubric.weight}"
            )
        
        # Validate severity with enum constraint
        if rubric.severity not in self.VALID_SEVERITIES:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} severity must be one of {self.VALID_SEVERITIES}, got '{rubric.severity}'"
            )
        
        # Validate evidence_budget with fallback to default
        if rubric.evidence_budget <= 0:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} evidence_budget must be positive, got {rubric.evidence_budget}"
            )
        
        # Validate scoring_scale type
        scale_type = rubric.scoring_scale.get("type")
        if scale_type not in ["numeric", "categorical"]:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} scoring_scale type must be 'numeric' or 'categorical', got '{scale_type}'"
            )
        
        # Validate scoring_scale structure based on type
        if scale_type == "numeric":
            if "min" not in rubric.scoring_scale or "max" not in rubric.scoring_scale:
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} with numeric scoring_scale must have 'min' and 'max'"
                )
            # Validate min < max
            if rubric.scoring_scale["min"] >= rubric.scoring_scale["max"]:
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} scoring_scale min must be less than max, got min={rubric.scoring_scale['min']}, max={rubric.scoring_scale['max']}"
                )
        elif scale_type == "categorical":
            if "values" not in rubric.scoring_scale:
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} with categorical scoring_scale must have 'values'"
                )
            if not isinstance(rubric.scoring_scale["values"], list):
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} categorical values must be a list"
                )
            # Validate values non-empty and unique
            if not rubric.scoring_scale["values"]:
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} categorical values must not be empty"
                )
            if len(rubric.scoring_scale["values"]) != len(set(rubric.scoring_scale["values"])):
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} categorical values must be unique"
                )
        
        # Validate aggregation_type matches scoring_scale type (except deterministic)
        if rubric.aggregation_type != "deterministic":
            if scale_type == "numeric" and rubric.aggregation_type not in ["median", "mean"]:
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} with numeric scoring_scale must use 'median' or 'mean' aggregation, got '{rubric.aggregation_type}'"
                )
            
            if scale_type == "categorical" and rubric.aggregation_type != "majority_vote":
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} with categorical scoring_scale must use 'majority_vote' aggregation, got '{rubric.aggregation_type}'"
                )
        
        # Validate deterministic rubric consistency
        if not rubric.requires_llm_judge and rubric.aggregation_type != "deterministic":
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} with requires_llm_judge=false must use aggregation_type='deterministic', got '{rubric.aggregation_type}'"
            )
        
        # Validate reverse: deterministic aggregation requires no LLM judge
        if rubric.aggregation_type == "deterministic" and rubric.requires_llm_judge:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} with aggregation_type='deterministic' must have requires_llm_judge=false (deterministic means no judge)"
            )
        
        # Validate deterministic rubrics have deterministic_source
        if not rubric.requires_llm_judge and not rubric.deterministic_source:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} with requires_llm_judge=false must specify deterministic_source"
            )
        
        # Validate LLM rubrics don't have deterministic_source (misleading)
        if rubric.requires_llm_judge and rubric.deterministic_source:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} with requires_llm_judge=true must not specify deterministic_source (LLM rubrics don't use deterministic sources)"
            )
        
        # Validate deterministic_source format (non-empty string, dotted path)
        if rubric.deterministic_source:
            if not isinstance(rubric.deterministic_source, str) or not rubric.deterministic_source.strip():
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} deterministic_source must be a non-empty string"
                )
            # Validate dotted path format (e.g., "module.function" or "class.method")
            if "." not in rubric.deterministic_source:
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} deterministic_source must be a dotted path (e.g., 'module.function'), got '{rubric.deterministic_source}'"
                )
        
        # Validate deterministic rubrics don't have evaluation_instructions (optional enforcement)
        if not rubric.requires_llm_judge and rubric.evaluation_instructions:
            # Allow but note: deterministic rubrics can have instructions for documentation
            pass
        
        # Validate LLM rubrics have evaluation_instructions
        if rubric.requires_llm_judge and not rubric.evaluation_instructions:
            raise RubricValidationError(
                f"Rubric {rubric.rubric_id} with requires_llm_judge=true must have evaluation_instructions"
            )
        
        # Validate max_severity_escalation only for categorical rubrics
        if rubric.run_aggregation_policy == "max_severity_escalation":
            if scale_type != "categorical":
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} with run_aggregation_policy='max_severity_escalation' must have categorical scoring_scale, got '{scale_type}'"
                )
        
        # Validate redact_fields is a list of strings if present
        if rubric.redact_fields is not None:
            if not isinstance(rubric.redact_fields, list):
                raise RubricValidationError(
                    f"Rubric {rubric.rubric_id} redact_fields must be a list, got {type(rubric.redact_fields)}"
                )
            for i, field in enumerate(rubric.redact_fields):
                if not isinstance(field, str) or not field.strip():
                    raise RubricValidationError(
                        f"Rubric {rubric.rubric_id} redact_fields at index {i} must be a non-empty string"
                    )
