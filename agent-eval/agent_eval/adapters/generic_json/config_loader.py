# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Configuration loader for Generic JSON adapter.

This module provides the AdapterConfig class that loads and parses adapter_config.yaml,
compiles regex patterns at load time for fail-fast validation, caches compiled patterns
for performance, and pre-resolves alias accessors for speed optimization.
"""

import re
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Pattern, Callable, Tuple
import yaml

from .config_schema import validate_config, AdapterConfig as AdapterConfigSchema


class AdapterConfig:
    """
    Configuration loader and accessor for adapter_config.yaml.
    
    This class:
    - Loads and validates YAML configuration using Pydantic
    - Compiles all regex patterns at load time for fail-fast validation
    - Caches compiled regex patterns for performance
    - Pre-resolves alias accessors (dotpath getters) for speed optimization
    - Emits warnings for unknown config keys (config drift detection)
    
    Usage:
        config = AdapterConfig("path/to/adapter_config.yaml")
        
        # Access configuration sections
        event_paths = config.get_event_paths()
        field_aliases = config.get_field_aliases("timestamp")
        
        # Classify events
        kind, rule_id = config.classify_event(event_dict)
        
        # Get segmentation strategies
        strategies = config.get_segmentation_strategies()
    """
    
    def __init__(self, config_path: str):
        """
        Load and parse adapter configuration.
        
        Args:
            config_path: Path to adapter_config.yaml file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config validation fails or YAML parsing fails
        """
        self.config_path = Path(config_path)
        self._raw_config: Dict[str, Any] = {}
        self._validated_config: Optional[AdapterConfigSchema] = None  # Assigned in _load_config()
        
        # Cached compiled regex patterns
        self._compiled_regexes: Dict[str, Pattern] = {}
        
        # Pre-resolved alias accessors (dotpath getters)
        self._alias_accessors: Dict[str, List[Callable[[Dict[str, Any]], Any]]] = {}
        
        # Load and validate configuration (assigns self._validated_config)
        self._load_config()
        
        self._compile_regex_patterns()
        self._resolve_alias_accessors()
    
    def _load_config(self) -> None:
        """
        Load YAML configuration and validate structure.
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config validation fails, YAML is empty, or file is unreadable
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}. "
                f"Please ensure adapter_config.yaml exists."
            )
        
        # Load YAML with UTF-8 encoding
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(
                f"Failed to parse YAML configuration from {self.config_path}: {e}"
            ) from e
        except OSError as e:
            raise ValueError(
                f"Failed to read configuration file {self.config_path}: {e}. "
                f"Check file permissions."
            ) from e
        
        # Handle empty YAML
        if self._raw_config is None:
            raise ValueError(
                f"Configuration file is empty: {self.config_path}. "
                f"Please ensure adapter_config.yaml contains valid configuration."
            )
        
        # Warn about unknown top-level keys (config drift detection)
        self._warn_unknown_keys()
        
        # Validate using Pydantic schema
        try:
            self._validated_config = validate_config(self._raw_config)
        except ValueError as e:
            raise ValueError(
                f"Configuration validation failed for {self.config_path}: {e}"
            ) from e
        
        # Type safety: ensure validation succeeded
        if self._validated_config is None:
            raise ValueError(
                f"Configuration validation returned None for {self.config_path}. "
                f"This indicates a bug in the validation logic."
            )
    
    def _warn_unknown_keys(self) -> None:
        """
        Emit warnings for unknown configuration keys (config drift detection).
        
        This helps detect when the YAML file has keys that aren't in the schema,
        which could indicate outdated configuration or typos.
        """
        if not isinstance(self._raw_config, dict):
            return
        
        known_keys = {
            'version', 'adapter_name', 'normalize', 'classify',
            'segment', 'derive', 'confidence', 'stats'
        }
        unknown_keys = set(self._raw_config.keys()) - known_keys
        
        if unknown_keys:
            warnings.warn(
                f"Unknown configuration keys detected in {self.config_path} "
                f"(possible config drift): {unknown_keys}. "
                f"These keys will be ignored.",
                UserWarning
            )
    
    def _compile_regex_patterns(self) -> None:
        """
        Compile all regex patterns at load time for fail-fast validation.
        
        This method compiles regex patterns from:
        - Classification rules (classify.rules[].any/all[].regex)
        - Prompt context stripping (derive.prompt_context_strip.strip_text_regex)
        - Tool output detection (derive.attribution.verdicts.tool_output_only_if_text_matches_regex)
        - Stitched trace detection (derive.attribution.stitch_suspect.question_line_regex)
        - Output stream exclusion (derive.output_extraction.assistant_output_stream.exclude_if_text_matches_regex)
        
        Regex Compile Flags:
        - Most patterns: No flags (default behavior)
        - question_line_regex: re.MULTILINE (for line-based matching with ^ and $)
        
        Raises:
            ValueError: If any regex pattern is invalid
        """
        # Verify rule order policy is supported
        if self._validated_config.classify.rule_order_policy != "first_match_wins":
            raise ValueError(
                f"Unsupported rule_order_policy: {self._validated_config.classify.rule_order_policy}. "
                f"Only 'first_match_wins' is currently supported."
            )
        
        # Compile classification rule patterns
        for rule in self._validated_config.classify.rules:
            # Compile 'all' conditions
            if rule.all:
                for idx, condition in enumerate(rule.all):
                    if condition.regex:
                        # Include index to avoid key collisions
                        key = f"classify.rule.{rule.id}.all.{condition.field}.{idx}"
                        try:
                            self._compiled_regexes[key] = re.compile(condition.regex)
                        except re.error as e:
                            raise ValueError(
                                f"Invalid regex in classification rule '{rule.id}' "
                                f"for field '{condition.field}' (condition {idx}): {e}"
                            ) from e
            
            # Compile 'any' conditions
            if rule.any:
                for idx, condition in enumerate(rule.any):
                    if condition.regex:
                        # Include index to avoid key collisions
                        key = f"classify.rule.{rule.id}.any.{condition.field}.{idx}"
                        try:
                            self._compiled_regexes[key] = re.compile(condition.regex)
                        except re.error as e:
                            raise ValueError(
                                f"Invalid regex in classification rule '{rule.id}' "
                                f"for field '{condition.field}' (condition {idx}): {e}"
                            ) from e
        
        # Compile prompt context strip patterns
        for i, pattern in enumerate(self._validated_config.derive.prompt_context_strip.strip_text_regex or []):
            key = f"derive.prompt_context_strip.{i}"
            try:
                self._compiled_regexes[key] = re.compile(pattern)
            except re.error as e:
                raise ValueError(
                    f"Invalid regex in prompt_context_strip pattern {i}: {e}"
                ) from e
        
        # Compile tool output detection patterns
        for i, pattern in enumerate(self._validated_config.derive.attribution.verdicts.tool_output_only_if_text_matches_regex or []):
            key = f"derive.attribution.tool_output.{i}"
            try:
                self._compiled_regexes[key] = re.compile(pattern)
            except re.error as e:
                raise ValueError(
                    f"Invalid regex in tool_output_only_if_text_matches_regex pattern {i}: {e}"
                ) from e
        
        # Compile stitched trace detection pattern (use MULTILINE for line-based matching)
        pattern = self._validated_config.derive.attribution.stitch_suspect.question_line_regex
        try:
            self._compiled_regexes["derive.attribution.stitch_suspect"] = re.compile(pattern, re.MULTILINE)
        except re.error as e:
            raise ValueError(
                f"Invalid regex in stitch_suspect.question_line_regex: {e}"
            ) from e
        
        # Compile output stream exclusion patterns
        for i, pattern in enumerate(self._validated_config.derive.output_extraction.assistant_output_stream.exclude_if_text_matches_regex or []):
            key = f"derive.output_extraction.exclude.{i}"
            try:
                self._compiled_regexes[key] = re.compile(pattern)
            except re.error as e:
                raise ValueError(
                    f"Invalid regex in assistant_output_stream.exclude_if_text_matches_regex pattern {i}: {e}"
                ) from e
    
    def _resolve_alias_accessors(self) -> None:
        """
        Pre-resolve alias accessors (dotpath getters) for speed optimization.
        
        Creates callable functions for each field alias that can extract values
        from nested dictionaries using dotted-path notation (e.g., "attributes.timestamp").
        
        Note: This implementation only supports simple dotted paths for nested dicts.
        Wildcard patterns (e.g., "resourceSpans.*.scopeSpans") are NOT supported here
        and must be handled separately by the adapter's event discovery logic.
        """
        for field_name, aliases in self._validated_config.normalize.field_aliases.items():
            accessors = []
            for alias in aliases:
                # Validate that alias is a simple dotpath (no wildcards or array notation)
                if '*' in alias or '[' in alias or ']' in alias or '..' in alias:
                    raise ValueError(
                        f"Field alias '{alias}' for field '{field_name}' contains invalid characters. "
                        f"Only simple dotted paths are supported in field_aliases (no *, [, ], or ..). "
                        f"Use event_paths for wildcard patterns."
                    )
                
                # Create a dotpath accessor function
                accessor = self._create_dotpath_accessor(alias)
                accessors.append(accessor)
            self._alias_accessors[field_name] = accessors
    
    def _create_dotpath_accessor(self, dotpath: str) -> Callable[[Dict[str, Any]], Any]:
        """
        Create a callable accessor function for a dotted-path field.
        
        Note: This accessor only supports nested dictionaries. It does not traverse
        lists or handle wildcard patterns (e.g., "resourceSpans.*.scopeSpans").
        For list traversal, the adapter will need to implement separate logic.
        
        Args:
            dotpath: Dotted path like "attributes.timestamp" or "span.start_time"
            
        Returns:
            Callable that takes a dict and returns the value at the dotpath, or None
        """
        # Cache parts as tuple for micro-optimization (immutable, faster lookup)
        parts = tuple(dotpath.split('.'))
        
        def accessor(data: Dict[str, Any]) -> Any:
            """Extract value from nested dict using dotpath."""
            current = data
            for part in parts:
                if not isinstance(current, dict):
                    return None
                current = current.get(part)
                if current is None:
                    return None
            return current
        
        return accessor
    
    # -------------------------------------------------------------------------
    # Public API - Normalize Section
    # -------------------------------------------------------------------------
    
    def get_event_paths(self) -> List[str]:
        """
        Get list of event paths to search for events in source JSON.
        
        Returns:
            List of event paths (e.g., ["events", "trace.events", "spans"])
        """
        return self._validated_config.normalize.event_paths
    
    def get_field_aliases(self, field_name: str) -> List[str]:
        """
        Get list of field aliases for a target normalized field.
        
        Args:
            field_name: Target field name (e.g., "timestamp", "tool_name")
            
        Returns:
            List of source field aliases to try in order, or empty list if not found
        """
        return self._validated_config.normalize.field_aliases.get(field_name, [])
    
    def get_field_value_with_fallback(self, data: Dict[str, Any], field_name: str) -> Any:
        """
        Extract field value using alias fallback chain.
        
        Tries each alias in order until a value is found.
        Uses pre-resolved dotpath accessors for performance.
        
        Args:
            data: Source event dictionary
            field_name: Target field name (e.g., "timestamp", "tool_name")
            
        Returns:
            First non-None value found, or None if all aliases fail
        """
        accessors = self._alias_accessors.get(field_name, [])
        for accessor in accessors:
            value = accessor(data)
            if value is not None:
                return value
        return None
    
    def get_timestamp_parse_config(self) -> Dict[str, Any]:
        """
        Get timestamp parsing configuration.
        
        Returns:
            Dictionary with:
            - epoch_units: List of supported epoch units (["ms", "s", "ns"])
            - infer_epoch_unit_by_magnitude: Whether to infer unit from magnitude
            - min_reasonable_year: Minimum valid year (2000)
            - max_reasonable_year: Maximum valid year (2100)
            - unix_nano_fields: List of OTEL UnixNano field names
            - formats: List of strptime format strings
            
        Note:
            Callers should treat returned dictionaries as read-only. Modifying
            the returned structure may lead to undefined behavior.
        """
        ts_config = self._validated_config.normalize.timestamp_parse
        return {
            "epoch_units": ts_config.epoch_units,
            "infer_epoch_unit_by_magnitude": ts_config.infer_epoch_unit_by_magnitude,
            "min_reasonable_year": ts_config.min_reasonable_year,
            "max_reasonable_year": ts_config.max_reasonable_year,
            "unix_nano_fields": ts_config.unix_nano_fields,
            "formats": ts_config.formats
        }
    
    def get_carry_fields_config(self) -> Dict[str, Any]:
        """
        Get raw data preservation configuration.
        
        Returns:
            Dictionary with:
            - attributes_paths: List of paths to extract attributes from
            - keep_raw_event: Whether to preserve raw event data
            - raw_event_max_bytes: Maximum bytes for raw event data
        """
        carry_config = self._validated_config.normalize.carry_fields
        return {
            "attributes_paths": carry_config.attributes_paths,
            "keep_raw_event": carry_config.keep_raw_event,
            "raw_event_max_bytes": carry_config.raw_event_max_bytes
        }
    
    # -------------------------------------------------------------------------
    # Public API - Classify Section
    # -------------------------------------------------------------------------
    
    def classify_event(self, event: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Classify event into a kind based on classification rules.
        
        Applies rules in order with first_match_wins policy.
        
        Args:
            event: Event dictionary to classify
            
        Returns:
            Tuple of (kind, rule_id, kind_reason) where:
            - kind: Event kind (e.g., "USER_INPUT", "TOOL_CALL", "EVENT")
            - rule_id: ID of matching rule, or None if default kind used
            - kind_reason: Human-readable explanation of why this kind was assigned
        """
        for rule in self._validated_config.classify.rules:
            matched, match_detail = self._rule_matches(rule, event)
            if matched:
                # Build kind_reason from match_detail
                kind_reason = self._build_kind_reason(rule, match_detail)
                return (rule.kind, rule.id, kind_reason)
        
        # No rule matched, use default kind
        return (self._validated_config.classify.default_kind, None, "No classification rule matched; using default kind")
    
    def _build_kind_reason(self, rule, match_detail: Dict[str, Any]) -> str:
        """
        Build a human-readable kind_reason from match details.
        
        Args:
            rule: ClassificationRule that matched
            match_detail: Dictionary with matched condition details
            
        Returns:
            Human-readable reason string
        """
        parts = [f"Matched rule '{rule.id}'"]
        
        if match_detail.get("matched_conditions"):
            for cond in match_detail["matched_conditions"][:3]:  # Limit to first 3 for brevity
                field = cond.get("field", "?")
                op = cond.get("op", "?")
                
                if op == "regex":
                    pattern = cond.get("pattern", "?")
                    parts.append(f"{field} matches regex '{pattern[:50]}'")
                elif op == "exists":
                    exists_val = cond.get("exists_value", True)
                    parts.append(f"{field} {'exists' if exists_val else 'does not exist'}")
                elif op == "equals":
                    equals_val = cond.get("equals_value", "?")
                    parts.append(f"{field} equals '{equals_val}'")
        
        return "; ".join(parts)
    
    def _rule_matches(self, rule, event: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a classification rule matches an event.
        
        Args:
            rule: ClassificationRule from config
            event: Event dictionary
            
        Returns:
            Tuple of (matched, match_detail) where:
            - matched: True if rule matches, False otherwise
            - match_detail: Dictionary with details about which conditions matched
        """
        match_detail = {
            "rule_id": rule.id,
            "kind": rule.kind,
            "matched_conditions": []
        }
        
        # Check 'all' conditions (all must match)
        if rule.all:
            for idx, condition in enumerate(rule.all):
                matched, witness = self._condition_matches_with_witness(condition, event, rule.id, idx, 'all')
                if not matched:
                    return (False, {})
                match_detail["matched_conditions"].append(witness)
        
        # Check 'any' conditions (at least one must match)
        if rule.any:
            any_matched = False
            for idx, condition in enumerate(rule.any):
                matched, witness = self._condition_matches_with_witness(condition, event, rule.id, idx, 'any')
                if matched:
                    any_matched = True
                    match_detail["matched_conditions"].append(witness)
                    break  # First match wins for 'any'
            if not any_matched:
                return (False, {})
        
        return (True, match_detail)
    
    def _condition_matches_with_witness(
        self, 
        condition, 
        event: Dict[str, Any], 
        rule_id: str, 
        condition_idx: int, 
        condition_type: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a single condition matches an event and return match witness.
        
        Args:
            condition: ClassificationCondition from config
            event: Event dictionary
            rule_id: Rule ID for regex lookup
            condition_idx: Index of condition in the rule's condition list
            condition_type: Either 'all' or 'any'
            
        Returns:
            Tuple of (matched, witness) where:
            - matched: True if condition matches, False otherwise
            - witness: Dictionary with match details (field, op, pattern/value, etc.)
        """
        # Get field value using alias fallback
        value = self.get_field_value_with_fallback(event, condition.field)
        
        # Truncate value for preview (avoid huge strings in metadata)
        value_preview = str(value)[:200] if value is not None else None
        
        witness = {
            "combinator": condition_type,
            "field": condition.field,
            "value_preview": value_preview
        }
        
        # Check 'exists' condition
        if condition.exists is not None:
            matched = (value is not None) == condition.exists
            witness["op"] = "exists"
            witness["exists_value"] = condition.exists
            return (matched, witness)
        
        # Check 'equals' condition
        if condition.equals is not None:
            if value is None:
                witness["op"] = "equals"
                witness["equals_value"] = condition.equals
                return (False, witness)
            
            matched = str(value) == str(condition.equals)
            witness["op"] = "equals"
            witness["equals_value"] = condition.equals
            return (matched, witness)
        
        # Check 'regex' condition
        if condition.regex is not None:
            if value is None:
                witness["op"] = "regex"
                witness["pattern"] = condition.regex
                return (False, witness)
            
            # Get compiled regex pattern using the exact key format from _compile_regex_patterns
            regex_key = f"classify.rule.{rule_id}.{condition_type}.{condition.field}.{condition_idx}"
            
            pattern = self._compiled_regexes.get(regex_key)
            if pattern is None:
                # This should never happen if _compile_regex_patterns worked correctly
                # Use RuntimeError to distinguish from user/config errors
                raise RuntimeError(
                    f"Regex pattern not found in cache for rule '{rule_id}', "
                    f"condition type '{condition_type}', field '{condition.field}', index {condition_idx}. "
                    f"This indicates a bug in the config loader."
                )
            
            # Match against string value
            matched = bool(pattern.search(str(value)))
            witness["op"] = "regex"
            witness["pattern"] = condition.regex
            return (matched, witness)
        
        # Should never reach here due to Pydantic validation
        return (False, witness)
    
    def get_default_kind(self) -> str:
        """
        Get default event kind for unmatched events.
        
        Returns:
            Default kind (typically "EVENT")
        """
        return self._validated_config.classify.default_kind
    
    # -------------------------------------------------------------------------
    # Public API - Segment Section
    # -------------------------------------------------------------------------
    
    def get_segmentation_strategies(self) -> List[str]:
        """
        Get list of segmentation strategies in preference order.
        
        Returns:
            List of strategy names (e.g., ["TURN_ID", "SESSION_PLUS_REQUEST", ...])
        """
        return self._validated_config.segment.strategy_preference
    
    def get_turn_id_fields(self) -> List[str]:
        """
        Get list of field names to use for turn ID extraction.
        
        Returns:
            List of field names (e.g., ["turn_id", "request_id"])
        """
        return self._validated_config.segment.turn_id_fields
    
    def get_request_id_diagnosis_config(self) -> Dict[str, Any]:
        """
        Get configuration for stitched trace detection.
        
        Returns:
            Dictionary with diagnosis thresholds
        """
        diag = self._validated_config.segment.request_id_diagnosis
        return {
            "distinct_user_prompts_per_request_id_max": diag.distinct_user_prompts_per_request_id_max,
            "request_ids_per_user_prompt_max": diag.request_ids_per_user_prompt_max,
            "sample_window_events": diag.sample_window_events
        }
    
    def get_anchor_events(self) -> List[str]:
        """
        Get list of anchor event kinds for turn boundaries.
        
        Returns:
            List of event kinds (e.g., ["USER_INPUT", "MODEL_INVOKE"])
        """
        return self._validated_config.segment.anchor_events_in_order
    
    def get_tie_breaker_order(self) -> List[str]:
        """
        Get tie-breaker order for events with same/missing timestamps.
        
        Returns:
            List of event kinds in priority order
        """
        return self._validated_config.segment.tie_breaker_order
    
    def should_emit_strategy_reason(self) -> bool:
        """
        Check if segmentation strategy reason should be emitted.
        
        Returns:
            True if strategy reason should be included in output
        """
        return self._validated_config.segment.emit_strategy_reason
    
    def should_emit_debug_fields(self) -> bool:
        """
        Check if debug fields (like fields_source) should be emitted.
        
        Returns:
            True if debug fields should be included in output (default: False for production)
        """
        # Default to False for production - can be made configurable later
        return getattr(self._validated_config, 'emit_debug_fields', False)
    
    def get_min_events_per_turn(self) -> int:
        """
        Get minimum number of events required per turn.
        
        Returns:
            Minimum events per turn (typically 1)
        """
        return self._validated_config.segment.min_events_per_turn
    
    # -------------------------------------------------------------------------
    # Public API - Derive Section
    # -------------------------------------------------------------------------
    
    def get_phases_config(self) -> Dict[str, str]:
        """
        Get phase classification configuration.
        
        Returns:
            Dictionary mapping phase names to phase identifiers
        """
        phases = self._validated_config.derive.phases
        return {
            "pre_tool": phases.pre_tool,
            "tool_call": phases.tool_call,
            "post_tool": phases.post_tool
        }
    
    def get_prompt_context_strip_config(self) -> Dict[str, Any]:
        """
        Get prompt context stripping configuration.
        
        Returns:
            Dictionary with:
            - strip_kinds: List of event kinds to strip
            - strip_text_patterns: List of compiled regex patterns
        """
        strip_config = self._validated_config.derive.prompt_context_strip
        
        # Get compiled patterns (guard against None)
        patterns = []
        for i in range(len(strip_config.strip_text_regex or [])):
            key = f"derive.prompt_context_strip.{i}"
            pattern = self._compiled_regexes.get(key)
            if pattern:
                patterns.append(pattern)
        
        return {
            "strip_kinds": strip_config.strip_kinds,
            "strip_text_patterns": patterns
        }
    
    def get_output_extraction_config(self) -> Dict[str, Any]:
        """
        Get top-level output extraction configuration.
        
        Returns:
            Dictionary with output extraction settings
        """
        output_config = self._validated_config.derive.output_extraction
        
        # Get compiled exclusion patterns (guard against None)
        exclusion_patterns = []
        for i in range(len(output_config.assistant_output_stream.exclude_if_text_matches_regex or [])):
            key = f"derive.output_extraction.exclude.{i}"
            pattern = self._compiled_regexes.get(key)
            if pattern:
                exclusion_patterns.append(pattern)
        
        return {
            "top_level_path_syntax": output_config.top_level_path_syntax,
            "top_level_dotpath_required": output_config.top_level_dotpath_required,
            "top_level_fields": output_config.top_level_fields,
            "assistant_output_stream": {
                "include_kinds": output_config.assistant_output_stream.include_kinds,
                "exclusion_patterns": exclusion_patterns
            },
            "join_with": output_config.join_with,
            "max_chars": output_config.max_chars
        }
    
    def get_tool_linking_config(self) -> Dict[str, Any]:
        """
        Get tool linking configuration.
        
        Returns:
            Dictionary with tool linking settings
        """
        tool_config = self._validated_config.derive.tool_linking
        return {
            "tool_run_exists_only_if": {
                "kind_in": tool_config.tool_run_exists_only_if.kind_in,
                "tool_name_required": tool_config.tool_run_exists_only_if.tool_name_required
            },
            "tool_run_id_fields": tool_config.tool_run_id_fields,
            "link_results_by": tool_config.link_results_by,
            "dedupe": {
                "enabled": tool_config.dedupe.enabled,
                "window_seconds": tool_config.dedupe.window_seconds,
                "key_fields": tool_config.dedupe.key_fields,
                "prefer_richer_fields": tool_config.dedupe.prefer_richer_fields
            }
        }
    
    def get_latency_config(self) -> Dict[str, Any]:
        """
        Get latency calculation configuration.
        
        Returns:
            Dictionary with latency calculation settings
        """
        latency_config = self._validated_config.derive.latency
        return {
            "normalized_latency_ms": {
                "start_from_first_kind_in": latency_config.normalized_latency_ms.start_from_first_kind_in,
                "end_at_last_kind_in": latency_config.normalized_latency_ms.end_at_last_kind_in
            },
            "keep_runtime_reported_latency_ms_fields": latency_config.keep_runtime_reported_latency_ms_fields,
            "on_missing_timestamps": latency_config.on_missing_timestamps
        }
    
    def get_attribution_config(self) -> Dict[str, Any]:
        """
        Get attribution and stitched trace detection configuration.
        
        Returns:
            Dictionary with attribution settings
        """
        attr_config = self._validated_config.derive.attribution
        
        # Get compiled tool output patterns (guard against None)
        tool_output_patterns = []
        for i in range(len(attr_config.verdicts.tool_output_only_if_text_matches_regex or [])):
            key = f"derive.attribution.tool_output.{i}"
            pattern = self._compiled_regexes.get(key)
            if pattern:
                tool_output_patterns.append(pattern)
        
        # Get compiled stitch suspect pattern
        stitch_pattern = self._compiled_regexes.get("derive.attribution.stitch_suspect")
        
        return {
            "verdicts": {
                "tool_used_if_has_kind": attr_config.verdicts.tool_used_if_has_kind,
                "tool_output_patterns": tool_output_patterns
            },
            "stitch_suspect": {
                "enabled": attr_config.stitch_suspect.enabled,
                "question_line_pattern": stitch_pattern,
                "distinct_question_count_suspect_at": attr_config.stitch_suspect.distinct_question_count_suspect_at
            }
        }
    
    # -------------------------------------------------------------------------
    # Public API - Confidence Section
    # -------------------------------------------------------------------------
    
    def get_confidence_scoring_config(self) -> Dict[str, Any]:
        """
        Get confidence scoring configuration.
        
        Returns:
            Dictionary with:
            - base: Base confidence score (1.0)
            - penalties: Dictionary of penalty names to penalty values
        """
        scoring = self._validated_config.confidence.scoring
        return {
            "base": scoring.base,
            "penalties": scoring.penalties
        }
    
    def get_confidence_emit_fields(self) -> List[str]:
        """
        Get list of confidence fields to emit in output.
        
        Returns:
            List of field names (e.g., ["run_confidence", "turn_confidence"])
        """
        return self._validated_config.confidence.emit_fields
    
    # -------------------------------------------------------------------------
    # Public API - Stats Section
    # -------------------------------------------------------------------------
    
    def should_emit_adapter_stats(self) -> bool:
        """
        Check if adapter_stats should be emitted in output.
        
        Returns:
            True if adapter_stats should be included
        """
        return self._validated_config.stats.emit_adapter_stats
    
    def get_max_error_examples(self) -> int:
        """
        Get maximum number of error examples to include in adapter_stats.
        
        Returns:
            Maximum error examples (typically 20)
        """
        return self._validated_config.stats.max_error_examples
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def get_version(self) -> int:
        """
        Get configuration version.
        
        Returns:
            Configuration version number
        """
        return self._validated_config.version
    
    def get_adapter_name(self) -> str:
        """
        Get adapter name.
        
        Returns:
            Adapter name (e.g., "generic-json-adapter")
        """
        return self._validated_config.adapter_name
