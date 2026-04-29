# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Shared judge primitives and interfaces.

This module contains reusable judge building blocks:
- JudgeConfigLoader: Load and validate judge configuration (1-5 judges)
- JudgeClient: Abstract interface for judge execution
- Judge data models and exceptions
- Configuration schemas
- Client factory: Build judge client instances from configuration

Evaluator-specific orchestration (job building, queue running, aggregation)
lives in agent_eval.evaluators.trace_eval.judging.
"""

from agent_eval.judges.client_factory import (
    build_judge_clients,
    register_judge_client,
    JudgeClientFactoryError
)

__all__ = ["build_judge_clients", "register_judge_client", "JudgeClientFactoryError"]
