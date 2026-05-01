# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Cost insights module for agent evaluation.

Extracts token usage from normalized traces and calculates costs using
LiteLLM's pricing data or user-provided pricing overrides.

Usage:
    from agent_eval.evaluators.cost_insights import compute_cost_summary
    summary = compute_cost_summary(normalized_run)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# OTEL / trace attribute keys where token counts may appear
TOKEN_FIELD_ALIASES = {
    "input": [
        "prompt_tokens", "input_tokens", "inputTokens",
        "gen_ai.usage.prompt_tokens", "llm.usage.prompt_tokens",
    ],
    "output": [
        "completion_tokens", "output_tokens", "outputTokens",
        "gen_ai.usage.completion_tokens", "llm.usage.completion_tokens",
    ],
}

MODEL_ID_ALIASES = [
    "model_id", "modelId", "model", "llm.request.model",
    "gen_ai.request.model", "bedrock.model_id",
]


@dataclass
class StepUsage:
    """Token usage for a single step."""
    turn_id: str
    step_name: str
    model_id: Optional[str] = None
    agent_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0


@dataclass
class TurnCost:
    """Aggregated cost for a single turn."""
    turn_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    model_calls: int = 0


@dataclass
class CostSummary:
    """Complete cost summary for a run."""
    run_id: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    cost_per_turn: List[TurnCost] = field(default_factory=list)
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    tokens_by_model: Dict[str, Dict[str, int]] = field(default_factory=dict)
    cost_by_agent: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    step_details: List[StepUsage] = field(default_factory=list)
    pricing_source: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "run_id": self.run_id,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "cost_per_turn": [
                {
                    "turn_id": t.turn_id,
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                    "total_cost_usd": round(t.total_cost, 6),
                    "model_calls": t.model_calls,
                }
                for t in self.cost_per_turn
            ],
            "cost_by_model": {
                k: round(v, 6) for k, v in self.cost_by_model.items()
            },
            "tokens_by_model": self.tokens_by_model,
            "cost_by_agent": {
                agent_id: {
                    "total_cost_usd": round(info.get("total_cost", 0), 6),
                    "input_tokens": info.get("input_tokens", 0),
                    "output_tokens": info.get("output_tokens", 0),
                    "model_calls": info.get("model_calls", 0),
                }
                for agent_id, info in self.cost_by_agent.items()
            },
            "pricing_source": self.pricing_source,
        }


def _extract_field(attrs: Dict[str, Any], aliases: List[str]) -> Optional[Any]:
    """Extract a field value trying multiple alias keys."""
    for alias in aliases:
        if "." in alias:
            # Handle nested keys like "gen_ai.usage.prompt_tokens"
            parts = alias.split(".")
            val = attrs
            for p in parts:
                if isinstance(val, dict):
                    val = val.get(p)
                else:
                    val = None
                    break
            if val is not None:
                return val
        elif alias in attrs:
            return attrs[alias]
    return None


def _get_model_pricing(model_id: str, pricing_overrides: Optional[Dict] = None) -> tuple:
    """
    Get per-token pricing (input_cost_per_token, output_cost_per_token).

    Priority:
    1. User-provided pricing overrides
    2. LiteLLM's built-in model cost data
    3. Zero (unknown pricing)
    """
    if pricing_overrides and model_id in pricing_overrides:
        p = pricing_overrides[model_id]
        return p.get("input_cost_per_token", 0.0), p.get("output_cost_per_token", 0.0)

    # Try LiteLLM pricing
    try:
        import litellm
        cost_map = litellm.get_model_cost_map(url="")
        # Try direct lookup, then with invoke prefix for Bedrock models
        for key in [model_id, f"bedrock/invoke/{model_id.split('/', 1)[-1]}" if "/" in model_id else model_id]:
            cost_info = cost_map.get(key, {})
            if cost_info:
                return (
                    cost_info.get("input_cost_per_token", 0.0),
                    cost_info.get("output_cost_per_token", 0.0),
                )
    except Exception:
        pass

    return 0.0, 0.0


def extract_usage_from_steps(
    turns: List[Dict[str, Any]],
) -> List[StepUsage]:
    """Extract token usage from normalized run steps."""
    usages = []
    for turn in turns:
        turn_id = turn.get("turn_id", "unknown")
        for step in turn.get("steps", []):
            attrs = step.get("attributes") or {}
            raw = step.get("raw") or {}
            # Merge attrs and raw for broader field search
            merged = {**raw, **attrs}

            input_tok = _extract_field(merged, TOKEN_FIELD_ALIASES["input"])
            output_tok = _extract_field(merged, TOKEN_FIELD_ALIASES["output"])

            if input_tok is None and output_tok is None:
                continue

            model_id = _extract_field(merged, MODEL_ID_ALIASES)
            input_tok = int(input_tok or 0)
            output_tok = int(output_tok or 0)

            agent_id = step.get("agent_id") or _extract_field(merged, [
                "agent_id", "agentId", "agent_name", "agentName",
                "aws.local.service", "service.name",
            ])

            usages.append(StepUsage(
                turn_id=turn_id,
                step_name=step.get("name", "unknown"),
                model_id=model_id,
                agent_id=agent_id if isinstance(agent_id, str) else None,
                input_tokens=input_tok,
                output_tokens=output_tok,
                total_tokens=input_tok + output_tok,
            ))
    return usages


def compute_cost_summary(
    normalized_run: Dict[str, Any],
    pricing_overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> CostSummary:
    """
    Compute cost summary from a normalized run.

    Args:
        normalized_run: Normalized run dict (matching normalized_run.schema.json)
        pricing_overrides: Optional dict mapping model_id to
            {"input_cost_per_token": float, "output_cost_per_token": float}

    Returns:
        CostSummary with per-turn, per-model, and total cost breakdowns
    """
    run_id = normalized_run.get("run_id", "unknown")
    turns = normalized_run.get("turns", [])
    step_usages = extract_usage_from_steps(turns)

    if not step_usages:
        return CostSummary(run_id=run_id, pricing_source="no_usage_data")

    # Apply pricing to each step
    pricing_source = "litellm"
    if pricing_overrides:
        pricing_source = "user_overrides"

    for su in step_usages:
        if su.model_id:
            in_price, out_price = _get_model_pricing(su.model_id, pricing_overrides)
            su.input_cost = su.input_tokens * in_price
            su.output_cost = su.output_tokens * out_price
            su.total_cost = su.input_cost + su.output_cost

    # Aggregate per turn
    turn_map: Dict[str, TurnCost] = {}
    for su in step_usages:
        tc = turn_map.setdefault(su.turn_id, TurnCost(turn_id=su.turn_id))
        tc.input_tokens += su.input_tokens
        tc.output_tokens += su.output_tokens
        tc.total_tokens += su.total_tokens
        tc.total_cost += su.total_cost
        tc.model_calls += 1

    # Aggregate per model
    cost_by_model: Dict[str, float] = {}
    tokens_by_model: Dict[str, Dict[str, int]] = {}
    for su in step_usages:
        mid = su.model_id or "unknown"
        cost_by_model[mid] = cost_by_model.get(mid, 0.0) + su.total_cost
        tb = tokens_by_model.setdefault(mid, {"input": 0, "output": 0})
        tb["input"] += su.input_tokens
        tb["output"] += su.output_tokens

    # Aggregate per agent
    cost_by_agent: Dict[str, Dict[str, Any]] = {}
    for su in step_usages:
        if su.agent_id:
            ag = cost_by_agent.setdefault(su.agent_id, {"total_cost": 0.0, "input_tokens": 0, "output_tokens": 0, "model_calls": 0})
            ag["total_cost"] += su.total_cost
            ag["input_tokens"] += su.input_tokens
            ag["output_tokens"] += su.output_tokens
            ag["model_calls"] += 1

    # Totals
    total_in = sum(su.input_tokens for su in step_usages)
    total_out = sum(su.output_tokens for su in step_usages)

    return CostSummary(
        run_id=run_id,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        total_tokens=total_in + total_out,
        total_cost=sum(su.total_cost for su in step_usages),
        cost_per_turn=list(turn_map.values()),
        cost_by_model=cost_by_model,
        tokens_by_model=tokens_by_model,
        cost_by_agent=cost_by_agent,
        step_details=step_usages,
        pricing_source=pricing_source,
    )
