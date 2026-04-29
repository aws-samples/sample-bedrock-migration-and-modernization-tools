# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Provider implementations for judge clients.

This module contains concrete implementations of JudgeClient for
various LLM providers (Amazon Bedrock, OpenAI, Anthropic, etc.).
"""

from agent_eval.providers.bedrock_client import BedrockJudgeClient

__all__ = ['Amazon BedrockJudgeClient']
