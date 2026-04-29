# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Judge Configuration Schema and Loader

This module provides validation and loading for judge configuration files.
Judges are LLM models configured to evaluate rubrics requiring LLM judgment.

Requirements:
- Judge count must be between 1 and 5 inclusive
- Each judge must define: judge_id, provider, model_id, params
- Default repeats=3 if omitted
- Optional fields: concurrency, rate_limit, retry_policy, timeout_seconds
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml


class ValidationError(Exception):
    """Raised when judge configuration validation fails."""
    pass


class ConfigurationError(Exception):
    """Raised when judge configuration is invalid or malformed."""
    pass


@dataclass
class Judge:
    """
    Represents a single judge configuration.
    
    A judge is an LLM model that evaluates rubrics requiring LLM judgment.
    """
    judge_id: str
    provider: str  # e.g., "openai", "anthropic", "bedrock"
    model_id: str
    params: Dict[str, Any]  # Model-specific parameters
    repeats: int = 3
    concurrency: Optional[int] = None
    rate_limit: Optional[int] = None
    retry_policy: Optional[Dict[str, Any]] = None
    timeout_seconds: int = 30
    
    # Provider-specific optional settings
    region_name: Optional[str] = None  # AWS region for Amazon Bedrock
    streaming: bool = False  # Enable streaming responses (Amazon Bedrock)
    use_converse_api: bool = True  # Use Converse API when available (Amazon Bedrock)
    
    def __post_init__(self):
        """Validate judge fields after initialization."""
        if not self.judge_id:
            raise ValidationError("judge_id is required")
        if not self.provider:
            raise ValidationError("provider is required")
        if not self.model_id:
            raise ValidationError("model_id is required")
        if self.params is None:
            raise ValidationError("params is required")
        if self.repeats < 1:
            raise ValidationError(f"repeats must be >= 1, got {self.repeats}")
        if self.timeout_seconds < 1:
            raise ValidationError(f"timeout_seconds must be >= 1, got {self.timeout_seconds}")


@dataclass
class JudgeConfig:
    """
    Represents the complete judge configuration.
    
    Contains 1-5 judges for evaluation.
    """
    judges: List[Judge] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate judge configuration after initialization."""
        if not self.judges:
            raise ValidationError("At least 1 judge required")
        if len(self.judges) > 5:
            raise ValidationError("Maximum 5 judges allowed")


class JudgeConfigLoader:
    """
    Loads and validates judge configuration from YAML files.
    
    Enforces:
    - Judge count between 1 and 5 inclusive
    - Required fields per judge
    - Default values for optional fields
    """
    
    def load(self, path: str) -> JudgeConfig:
        """
        Load judge configuration from YAML file.
        
        Args:
            path: Path to judges.yaml file
            
        Returns:
            Validated JudgeConfig object
            
        Raises:
            ConfigurationError: If validation fails (wraps ValidationError for consistency)
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigurationError(f"Judge configuration file not found: {path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in judge configuration: {e}")
        
        if not data:
            raise ConfigurationError("Judge configuration file is empty")
        
        if 'judges' not in data:
            raise ConfigurationError("Judge configuration must contain 'judges' key")
        
        judges_data = data['judges']
        if not isinstance(judges_data, list):
            raise ConfigurationError("'judges' must be a list")
        
        # Validate judge count before processing
        try:
            self.validate_judge_count(judges_data)
        except ValidationError as e:
            # Wrap ValidationError as ConfigurationError for consistency
            raise ConfigurationError(str(e))
        
        # Parse and validate each judge
        judges = []
        seen_ids = set()
        for idx, judge_data in enumerate(judges_data):
            try:
                judge = self._parse_judge(judge_data)
                
                # Check for duplicate judge_id
                if judge.judge_id in seen_ids:
                    raise ConfigurationError(f"Duplicate judge_id: {judge.judge_id}")
                seen_ids.add(judge.judge_id)
                
                judges.append(judge)
            except (ValidationError, ConfigurationError) as e:
                # Include judge_id in error message if available
                judge_id = judge_data.get('judge_id', '<unknown>') if isinstance(judge_data, dict) else '<unknown>'
                raise ConfigurationError(f"Error in judge {idx} (judge_id={judge_id}): {e}")
        
        # Create config (validation happens in __post_init__)
        try:
            return JudgeConfig(judges=judges)
        except ValidationError as e:
            # Wrap ValidationError as ConfigurationError for consistency
            raise ConfigurationError(str(e))
    
    def validate_judge_count(self, judges: List[Any]) -> None:
        """
        Validate judge count is between 1 and 5.
        
        Args:
            judges: List of judge configurations
            
        Raises:
            ValidationError: With specific error message for violations
        """
        count = len(judges)
        if count < 1:
            raise ValidationError("At least 1 judge required")
        if count > 5:
            raise ValidationError("Maximum 5 judges allowed")
    
    def _parse_judge(self, judge_data: Dict[str, Any]) -> Judge:
        """
        Parse a single judge configuration and apply defaults.
        
        Args:
            judge_data: Raw judge configuration dict
            
        Returns:
            Judge object with defaults applied
            
        Raises:
            ConfigurationError: If required fields missing or invalid types
        """
        if not isinstance(judge_data, dict):
            raise ConfigurationError("Each judge must be a dictionary")
        
        # Extract required fields
        judge_id = judge_data.get('judge_id')
        provider = judge_data.get('provider')
        model_id = judge_data.get('model_id')
        params = judge_data.get('params')
        
        if not judge_id:
            raise ConfigurationError("judge_id is required")
        if not provider:
            raise ConfigurationError("provider is required")
        if not model_id or not isinstance(model_id, str) or not model_id.strip():
            raise ConfigurationError("model_id must be a non-empty string")
        if params is None:
            raise ConfigurationError("params is required")
        
        # Validate params type (must be dict)
        if not isinstance(params, dict):
            raise ConfigurationError("params must be a dictionary")
        
        # Apply defaults for optional fields
        repeats = judge_data.get('repeats', 3)
        concurrency = judge_data.get('concurrency')
        rate_limit = judge_data.get('rate_limit')
        retry_policy = judge_data.get('retry_policy')
        timeout_seconds = judge_data.get('timeout_seconds', 30)
        
        # Provider-specific optional settings
        region_name = judge_data.get('region_name')
        streaming = judge_data.get('streaming', False)
        use_converse_api = judge_data.get('use_converse_api', True)
        
        # Validate retry_policy structure if provided
        if retry_policy is not None:
            if not isinstance(retry_policy, dict):
                raise ConfigurationError("retry_policy must be a dictionary")
            
            # Validate expected keys and types
            if 'max_retries' in retry_policy:
                max_retries = retry_policy['max_retries']
                if not isinstance(max_retries, int):
                    raise ConfigurationError(f"retry_policy.max_retries must be an integer, got {type(max_retries).__name__}")
                if max_retries < 0:
                    raise ConfigurationError(f"retry_policy.max_retries must be >= 0, got {max_retries}")
            
            if 'backoff_multiplier' in retry_policy:
                backoff_multiplier = retry_policy['backoff_multiplier']
                if not isinstance(backoff_multiplier, (int, float)):
                    raise ConfigurationError(f"retry_policy.backoff_multiplier must be a number, got {type(backoff_multiplier).__name__}")
                if backoff_multiplier < 1:
                    raise ConfigurationError(f"retry_policy.backoff_multiplier must be >= 1, got {backoff_multiplier}")
        
        # Validate numeric field types and bounds
        if not isinstance(repeats, int):
            raise ConfigurationError(f"repeats must be an integer, got {type(repeats).__name__}")
        if repeats < 1:
            raise ConfigurationError(f"repeats must be >= 1, got {repeats}")
        
        if not isinstance(timeout_seconds, int):
            raise ConfigurationError(f"timeout_seconds must be an integer, got {type(timeout_seconds).__name__}")
        if timeout_seconds < 1:
            raise ConfigurationError(f"timeout_seconds must be >= 1, got {timeout_seconds}")
        
        if concurrency is not None:
            if not isinstance(concurrency, int):
                raise ConfigurationError(f"concurrency must be an integer, got {type(concurrency).__name__}")
            if concurrency < 1:
                raise ConfigurationError(f"concurrency must be >= 1, got {concurrency}")
        
        if rate_limit is not None:
            if not isinstance(rate_limit, int):
                raise ConfigurationError(f"rate_limit must be an integer, got {type(rate_limit).__name__}")
            if rate_limit < 1:
                raise ConfigurationError(f"rate_limit must be >= 1, got {rate_limit}")
        
        return Judge(
            judge_id=judge_id,
            provider=provider,
            model_id=model_id,
            params=params,
            repeats=repeats,
            concurrency=concurrency,
            rate_limit=rate_limit,
            retry_policy=retry_policy,
            timeout_seconds=timeout_seconds,
            region_name=region_name,
            streaming=streaming,
            use_converse_api=use_converse_api
        )
    
    def apply_defaults(self, judge: Judge) -> Judge:
        """
        Apply default values to a judge configuration.
        
        This method is provided for backward compatibility. Defaults are already
        applied during parsing in _parse_judge(), so this method is typically
        not needed.
        
        Note: This method does NOT mutate the input judge. It returns the same
        judge object since defaults are already applied.
        
        Args:
            judge: Judge object
            
        Returns:
            Judge object (same instance, defaults already applied)
        """
        # Defaults already applied in _parse_judge
        # This method exists for API compatibility but does not mutate
        return judge
