# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Judge client factory for building judge client instances from configuration.

This module provides a centralized factory for creating judge clients,
keeping the runner orchestration-focused and separating provider selection logic.
"""

from typing import Dict, Any


class JudgeClientFactoryError(Exception):
    """Raised when judge client instantiation fails."""
    pass


# Judge client registry for extensibility
JUDGE_CLIENT_FACTORIES = {}


def register_judge_client(provider: str, factory_class):
    """
    Register a judge client factory for a provider.
    
    Args:
        provider: Provider name (e.g., 'bedrock', 'mock')
        factory_class: Judge client class to instantiate
    """
    JUDGE_CLIENT_FACTORIES[provider] = factory_class


def _initialize_default_factories():
    """Initialize default judge client factories."""
    try:
        from agent_eval.providers.bedrock_client import BedrockJudgeClient
        register_judge_client("bedrock", BedrockJudgeClient)
    except ImportError:
        pass  # Amazon Bedrock client not available
    
    try:
        from agent_eval.providers.litellm_client import LiteLLMJudgeClient
        register_judge_client("litellm", LiteLLMJudgeClient)
    except ImportError:
        pass  # LiteLLM client not available
    
    try:
        from agent_eval.judges.mock_client import MockJudgeClient
        register_judge_client("mock", MockJudgeClient)
    except ImportError:
        pass  # Mock client not available


# Initialize default factories on module load
_initialize_default_factories()


def build_judge_clients(judge_config) -> Dict[str, Any]:
    """
    Build judge client instances from judge configuration.
    
    Uses a registry pattern for extensibility. Supports:
    - bedrock: Amazon Bedrock judge client
    - mock: Mock judge client for testing/development
    
    Args:
        judge_config: JudgeConfig object with judges list
        
    Returns:
        Dictionary mapping judge_id to JudgeClient instance
        
    Raises:
        JudgeClientFactoryError: If judge client instantiation fails
    """
    try:
        judge_clients = {}
        
        # Access judges from JudgeConfig object
        judges = judge_config.judges
        
        for judge in judges:
            judge_id = judge.judge_id
            provider = judge.provider
            model_id = judge.model_id
            params = judge.params
            timeout_seconds = judge.timeout_seconds
            
            # Get judge client factory
            client_factory = JUDGE_CLIENT_FACTORIES.get(provider)
            if not client_factory:
                supported = ", ".join(JUDGE_CLIENT_FACTORIES.keys())
                raise JudgeClientFactoryError(
                    f"Unsupported provider '{provider}' for judge_id='{judge_id}'. "
                    f"Supported providers: {supported}"
                )
            
            # Build kwargs with required parameters
            client_kwargs = {
                "judge_id": judge_id,
                "model_id": model_id,
                "params": params,
                "timeout_seconds": timeout_seconds
            }
            
            # Add provider-specific optional parameters if present
            # These are passed through to the client constructor
            if provider == "bedrock":
                # Amazon Bedrock-specific parameters
                if hasattr(judge, 'region_name') and judge.region_name is not None:
                    client_kwargs["region_name"] = judge.region_name
                if hasattr(judge, 'streaming'):
                    client_kwargs["streaming"] = judge.streaming
                if hasattr(judge, 'use_converse_api'):
                    client_kwargs["use_converse_api"] = judge.use_converse_api
            # Mock client doesn't need provider-specific parameters
            
            # Instantiate judge client
            try:
                judge_client = client_factory(**client_kwargs)
            except Exception as e:
                raise JudgeClientFactoryError(
                    f"Failed to instantiate {provider} judge client for "
                    f"judge_id='{judge_id}', model_id='{model_id}': {e}"
                ) from e
            
            judge_clients[judge_id] = judge_client
        
        return judge_clients
        
    except JudgeClientFactoryError:
        raise
    except Exception as e:
        raise JudgeClientFactoryError(f"Failed to build judge clients: {e}") from e
