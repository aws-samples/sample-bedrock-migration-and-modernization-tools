# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Pydantic schema validation for adapter_config.yaml.

This module provides fail-fast validation for the adapter configuration,
ensuring all required fields are present and properly structured before
the adapter attempts to process any traces.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import re


class TimestampParseConfig(BaseModel):
    """Configuration for timestamp parsing."""
    epoch_units: List[str] = Field(default_factory=lambda: ["ms", "s", "ns"])
    infer_epoch_unit_by_magnitude: bool = True
    min_reasonable_year: int = 2000
    max_reasonable_year: int = 2100
    unix_nano_fields: List[str] = Field(default_factory=list)
    formats: List[str] = Field(default_factory=list)

    @field_validator('epoch_units')
    @classmethod
    def validate_epoch_units(cls, v):
        valid_units = {"ms", "s", "ns"}
        for unit in v:
            if unit not in valid_units:
                raise ValueError(f"Invalid epoch unit '{unit}'. Must be one of: {valid_units}")
        return v


class CarryFieldsConfig(BaseModel):
    """Configuration for raw data preservation."""
    attributes_paths: List[str] = Field(default_factory=list)
    keep_raw_event: bool = True
    raw_event_max_bytes: int = 50000


class NormalizeConfig(BaseModel):
    """Stage A: Normalize configuration."""
    event_paths: List[str] = Field(min_length=1)
    field_aliases: Dict[str, List[str]] = Field(default_factory=dict)
    timestamp_parse: TimestampParseConfig
    carry_fields: CarryFieldsConfig

    @field_validator('event_paths')
    @classmethod
    def validate_event_paths(cls, v):
        if not v:
            raise ValueError("event_paths must contain at least one path")
        return v

    @field_validator('field_aliases')
    @classmethod
    def validate_field_aliases(cls, v):
        # Verify we have at least some common field aliases
        required_aliases = {'timestamp', 'event_type', 'tool_name'}
        missing = required_aliases - set(v.keys())
        if missing:
            # Warning only - not a hard error
            pass
        return v


class ClassificationCondition(BaseModel):
    """A single classification condition."""
    field: str
    regex: Optional[str] = None
    exists: Optional[bool] = None
    equals: Optional[Any] = None

    @model_validator(mode='after')
    def validate_condition(self):
        # Count how many condition types are specified
        condition_types = sum([
            self.regex is not None,
            self.exists is not None,
            self.equals is not None
        ])
        
        if condition_types == 0:
            raise ValueError(f"Condition for field '{self.field}' must have either 'regex', 'exists', or 'equals'")
        if condition_types > 1:
            raise ValueError(f"Condition for field '{self.field}' can only have one of 'regex', 'exists', or 'equals'")
        
        # Validate regex pattern if present
        if self.regex is not None:
            try:
                re.compile(self.regex)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern for field '{self.field}': {e}")
        
        return self


class ClassificationRule(BaseModel):
    """A classification rule for event kinds."""
    id: str
    kind: str
    all: Optional[List[ClassificationCondition]] = None
    any: Optional[List[ClassificationCondition]] = None

    @model_validator(mode='after')
    def validate_rule(self):
        if self.all is None and self.any is None:
            raise ValueError(f"Rule '{self.id}' must have either 'all' or 'any' conditions")
        return self


class ClassifyConfig(BaseModel):
    """Stage A: Classify configuration."""
    rule_order_policy: Literal["first_match_wins"] = "first_match_wins"
    rules: List[ClassificationRule] = Field(min_length=1)
    default_kind: str = "EVENT"

    @field_validator('rules')
    @classmethod
    def validate_rules(cls, v):
        if not v:
            raise ValueError("classify.rules must contain at least one rule")
        
        # Check for duplicate rule IDs
        rule_ids = [rule.id for rule in v]
        duplicates = [rid for rid in rule_ids if rule_ids.count(rid) > 1]
        if duplicates:
            raise ValueError(f"Duplicate rule IDs found: {set(duplicates)}")
        
        return v


class RequestIdDiagnosisConfig(BaseModel):
    """Configuration for stitched trace detection."""
    distinct_user_prompts_per_request_id_max: int = 1
    request_ids_per_user_prompt_max: int = 3
    sample_window_events: int = 5000


class SegmentConfig(BaseModel):
    """Stage B: Segment configuration."""
    strategy_preference: List[str] = Field(min_length=1)
    turn_id_fields: List[str] = Field(default_factory=list)
    request_id_diagnosis: RequestIdDiagnosisConfig
    anchor_events_in_order: List[str] = Field(default_factory=list)
    tie_breaker_order: List[str] = Field(default_factory=list)
    emit_strategy_reason: bool = True
    min_events_per_turn: int = 1

    @field_validator('strategy_preference')
    @classmethod
    def validate_strategies(cls, v):
        valid_strategies = {
            "TURN_ID",
            "SESSION_PLUS_REQUEST",
            "SESSION_PLUS_TRACE_THEN_ANCHOR_SPLIT",
            "SINGLE_TURN"
        }
        for strategy in v:
            if strategy not in valid_strategies:
                raise ValueError(
                    f"Invalid segmentation strategy '{strategy}'. "
                    f"Must be one of: {valid_strategies}"
                )
        return v


class PhasesConfig(BaseModel):
    """Phase classification configuration."""
    pre_tool: str = "PRE_TOOL_GENERATION"
    tool_call: str = "TOOL_CALL"
    post_tool: str = "FINAL_GENERATION"


class PromptContextStripConfig(BaseModel):
    """Configuration for prompt context stripping."""
    strip_kinds: List[str] = Field(default_factory=list)
    strip_text_regex: List[str] = Field(default_factory=list)

    @field_validator('strip_text_regex')
    @classmethod
    def validate_regex_patterns(cls, v):
        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern in strip_text_regex: {e}")
        return v


class AssistantOutputStreamConfig(BaseModel):
    """Configuration for LLM output streaming."""
    include_kinds: List[str] = Field(default_factory=list)
    exclude_if_text_matches_regex: List[str] = Field(default_factory=list)

    @field_validator('exclude_if_text_matches_regex')
    @classmethod
    def validate_regex_patterns(cls, v):
        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern in exclude_if_text_matches_regex: {e}")
        return v


class OutputExtractionConfig(BaseModel):
    """Configuration for top-level field extraction."""
    top_level_path_syntax: str = "dot"
    top_level_dotpath_required: bool = True
    top_level_fields: Dict[str, List[str]] = Field(default_factory=dict)
    assistant_output_stream: AssistantOutputStreamConfig
    join_with: str = ""
    max_chars: int = 200000


class ToolRunExistsOnlyIfConfig(BaseModel):
    """Configuration for tool run detection."""
    kind_in: List[str] = Field(default_factory=list)
    tool_name_required: bool = True


class DedupeConfig(BaseModel):
    """Configuration for tool deduplication."""
    enabled: bool = True
    window_seconds: int = 2
    key_fields: List[str] = Field(default_factory=list)
    prefer_richer_fields: Optional[List[str]] = None


class ToolLinkingConfig(BaseModel):
    """Configuration for tool linking."""
    tool_run_exists_only_if: ToolRunExistsOnlyIfConfig
    tool_run_id_fields: List[str] = Field(default_factory=list)
    link_results_by: List[str] = Field(default_factory=list)
    dedupe: DedupeConfig


class NormalizedLatencyConfig(BaseModel):
    """Configuration for normalized latency calculation."""
    start_from_first_kind_in: List[str] = Field(default_factory=list)
    end_at_last_kind_in: List[str] = Field(default_factory=list)


class LatencyConfig(BaseModel):
    """Configuration for latency calculation."""
    normalized_latency_ms: NormalizedLatencyConfig
    keep_runtime_reported_latency_ms_fields: List[str] = Field(default_factory=list)
    on_missing_timestamps: str = "null_and_penalize"


class VerdictsConfig(BaseModel):
    """Configuration for attribution verdicts."""
    tool_used_if_has_kind: str = "TOOL_CALL"
    tool_output_only_if_text_matches_regex: List[str] = Field(default_factory=list)

    @field_validator('tool_output_only_if_text_matches_regex')
    @classmethod
    def validate_regex_patterns(cls, v):
        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern in tool_output_only_if_text_matches_regex: {e}")
        return v


class StitchSuspectConfig(BaseModel):
    """Configuration for stitched trace detection."""
    enabled: bool = True
    question_line_regex: str
    distinct_question_count_suspect_at: int = 2

    @field_validator('question_line_regex')
    @classmethod
    def validate_regex_pattern(cls, v):
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern in question_line_regex: {e}")
        return v


class AttributionConfig(BaseModel):
    """Configuration for attribution and stitched trace detection."""
    verdicts: VerdictsConfig
    stitch_suspect: StitchSuspectConfig


class DeriveConfig(BaseModel):
    """Stage C: Derive configuration."""
    phases: PhasesConfig
    prompt_context_strip: PromptContextStripConfig
    output_extraction: OutputExtractionConfig
    tool_linking: ToolLinkingConfig
    latency: LatencyConfig
    attribution: AttributionConfig


class ScoringConfig(BaseModel):
    """Confidence scoring configuration."""
    base: float = 1.0
    penalties: Dict[str, float] = Field(default_factory=dict)

    @field_validator('base')
    @classmethod
    def validate_base(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"base confidence score must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator('penalties')
    @classmethod
    def validate_penalties(cls, v):
        for key, penalty in v.items():
            if not 0.0 <= penalty <= 1.0:
                raise ValueError(
                    f"Penalty '{key}' must be between 0.0 and 1.0, got {penalty}"
                )
        return v


class ConfidenceConfig(BaseModel):
    """Confidence scoring configuration."""
    scoring: ScoringConfig
    emit_fields: List[str] = Field(default_factory=list)


class StatsConfig(BaseModel):
    """Statistics configuration."""
    emit_adapter_stats: bool = True
    max_error_examples: int = 20


class AdapterConfig(BaseModel):
    """Complete adapter configuration schema."""
    version: int
    adapter_name: str
    normalize: NormalizeConfig
    classify: ClassifyConfig
    segment: SegmentConfig
    derive: DeriveConfig
    confidence: ConfidenceConfig
    stats: StatsConfig

    @field_validator('version')
    @classmethod
    def validate_version(cls, v):
        if v != 1:
            raise ValueError(f"Unsupported config version: {v}. Expected version 1.")
        return v

    @model_validator(mode='after')
    def validate_config(self):
        """Cross-field validation and warnings for unknown keys."""
        # Verify comprehensive field aliases (50+ mappings)
        total_aliases = sum(len(aliases) for aliases in self.normalize.field_aliases.values())
        if total_aliases < 50:
            # This is a warning, not an error - log it but don't fail
            import warnings
            warnings.warn(
                f"Config has only {total_aliases} field aliases. "
                f"Recommended: 50+ for comprehensive source format support."
            )
        
        return self


def validate_config(config_dict: Dict[str, Any]) -> AdapterConfig:
    """
    Validate adapter configuration dictionary against schema.
    
    Args:
        config_dict: Configuration dictionary loaded from YAML
        
    Returns:
        Validated AdapterConfig instance
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Handle None or empty config
    if config_dict is None:
        raise ValueError(
            "Configuration is empty or None. "
            "The YAML file may be empty or not properly synced. "
            "Please ensure adapter_config.yaml contains valid configuration."
        )
    
    if not isinstance(config_dict, dict):
        raise ValueError(
            f"Configuration must be a dictionary, got {type(config_dict).__name__}"
        )
    
    # Check for unknown top-level keys (config drift detection)
    known_keys = {
        'version', 'adapter_name', 'normalize', 'classify', 
        'segment', 'derive', 'confidence', 'stats'
    }
    unknown_keys = set(config_dict.keys()) - known_keys
    if unknown_keys:
        import warnings
        warnings.warn(
            f"Unknown configuration keys detected (possible config drift): {unknown_keys}. "
            f"These keys will be ignored."
        )
    
    # Validate using Pydantic
    return AdapterConfig(**config_dict)
