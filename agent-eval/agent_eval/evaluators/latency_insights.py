# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Latency insights module for agent evaluation.

Extracts TTFT, per-category latency breakdown, and identifies slowest
steps from normalized traces.

Usage:
    from agent_eval.evaluators.latency_insights import compute_latency_summary
    summary = compute_latency_summary(normalized_run)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Step kinds that map to each latency category
LLM_KINDS = {"MODEL_INVOKE", "LLM_OUTPUT_CHUNK"}
TOOL_KINDS = {"TOOL_CALL", "TOOL_RESULT"}


@dataclass
class TurnLatency:
    """Latency breakdown for a single turn."""
    turn_id: str
    total_ms: float = 0.0
    llm_ms: float = 0.0
    tool_ms: float = 0.0
    overhead_ms: float = 0.0
    ttft_ms: Optional[float] = None
    step_count: int = 0


@dataclass
class SlowestStep:
    """A slow step for optimization insights."""
    turn_id: str
    name: str
    kind: Optional[str]
    latency_ms: float


@dataclass
class LatencySummary:
    """Complete latency summary for a run."""
    run_id: str
    total_ms: float = 0.0
    avg_turn_ms: float = 0.0
    ttft_ms: Optional[float] = None
    total_llm_ms: float = 0.0
    total_tool_ms: float = 0.0
    total_overhead_ms: float = 0.0
    tokens_per_second: Optional[float] = None
    per_turn: List[TurnLatency] = field(default_factory=list)
    slowest_steps: List[SlowestStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_ms": round(self.total_ms, 1),
            "avg_turn_ms": round(self.avg_turn_ms, 1),
            "ttft_ms": round(self.ttft_ms, 1) if self.ttft_ms is not None else None,
            "total_llm_ms": round(self.total_llm_ms, 1),
            "total_tool_ms": round(self.total_tool_ms, 1),
            "total_overhead_ms": round(self.total_overhead_ms, 1),
            "tokens_per_second": round(self.tokens_per_second, 1) if self.tokens_per_second is not None else None,
            "per_turn": [
                {
                    "turn_id": t.turn_id,
                    "total_ms": round(t.total_ms, 1),
                    "llm_ms": round(t.llm_ms, 1),
                    "tool_ms": round(t.tool_ms, 1),
                    "overhead_ms": round(t.overhead_ms, 1),
                    "ttft_ms": round(t.ttft_ms, 1) if t.ttft_ms is not None else None,
                }
                for t in self.per_turn
            ],
            "slowest_steps": [
                {
                    "turn_id": s.turn_id,
                    "name": s.name,
                    "kind": s.kind,
                    "latency_ms": round(s.latency_ms, 1),
                }
                for s in self.slowest_steps
            ],
        }


def _step_latency(step: Dict[str, Any]) -> float:
    """Get latency for a step, preferring explicit field over timestamp delta."""
    lat = step.get("latency_ms")
    if lat is not None:
        return float(lat)
    start = step.get("start_ts")
    end = step.get("end_ts")
    if start and end:
        from datetime import datetime, timezone
        try:
            t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(end.replace("Z", "+00:00"))
            return max((t1 - t0).total_seconds() * 1000, 0)
        except (ValueError, TypeError):
            pass
    return 0.0


def _step_kind(step: Dict[str, Any]) -> Optional[str]:
    """Get the kind of a step."""
    return step.get("kind") or step.get("type")


def compute_latency_summary(normalized_run: Dict[str, Any]) -> LatencySummary:
    """Compute latency summary from a normalized run."""
    run_id = normalized_run.get("run_id", "unknown")
    turns = normalized_run.get("turns", [])

    if not turns:
        return LatencySummary(run_id=run_id)

    all_steps: List[SlowestStep] = []
    turn_latencies: List[TurnLatency] = []
    total_output_tokens = 0
    total_llm_ms = 0.0

    for turn in turns:
        turn_id = turn.get("turn_id", "unknown")
        steps = turn.get("steps", [])
        total_ms = turn.get("total_latency_ms") or turn.get("normalized_latency_ms") or 0.0

        llm_ms = 0.0
        tool_ms = 0.0
        ttft_ms = None
        first_step_ts = None

        for step in steps:
            lat = _step_latency(step)
            kind = _step_kind(step)

            if kind in LLM_KINDS:
                llm_ms += lat
                # TTFT: time from turn start to first LLM response
                if ttft_ms is None and step.get("start_ts"):
                    if first_step_ts is None:
                        # Use the first step's start as turn start
                        for s in steps:
                            if s.get("start_ts"):
                                first_step_ts = s["start_ts"]
                                break
                    if first_step_ts:
                        from datetime import datetime
                        try:
                            t0 = datetime.fromisoformat(first_step_ts.replace("Z", "+00:00"))
                            t1 = datetime.fromisoformat(step["start_ts"].replace("Z", "+00:00"))
                            ttft_ms = max((t1 - t0).total_seconds() * 1000, 0)
                        except (ValueError, TypeError):
                            pass
                # Track output tokens for tokens/sec
                attrs = step.get("attributes") or {}
                for key in ["completion_tokens", "output_tokens", "outputTokens"]:
                    if key in attrs:
                        total_output_tokens += int(attrs[key])
                        break
            elif kind in TOOL_KINDS:
                tool_ms += lat

            if lat > 0:
                all_steps.append(SlowestStep(
                    turn_id=turn_id, name=step.get("name", "unknown"),
                    kind=kind, latency_ms=lat,
                ))

        # If total_ms wasn't set, sum from steps
        if total_ms == 0 and (llm_ms + tool_ms) > 0:
            total_ms = llm_ms + tool_ms

        overhead_ms = max(total_ms - llm_ms - tool_ms, 0)

        turn_latencies.append(TurnLatency(
            turn_id=turn_id, total_ms=total_ms, llm_ms=llm_ms,
            tool_ms=tool_ms, overhead_ms=overhead_ms, ttft_ms=ttft_ms,
            step_count=len(steps),
        ))
        total_llm_ms += llm_ms

    # Aggregate
    total_ms = sum(t.total_ms for t in turn_latencies)
    avg_turn_ms = total_ms / len(turn_latencies) if turn_latencies else 0
    first_ttft = next((t.ttft_ms for t in turn_latencies if t.ttft_ms is not None), None)

    # Tokens per second (output tokens / LLM time)
    tps = None
    if total_llm_ms > 0 and total_output_tokens > 0:
        tps = total_output_tokens / (total_llm_ms / 1000)

    # Top 5 slowest steps
    slowest = sorted(all_steps, key=lambda s: s.latency_ms, reverse=True)[:5]

    return LatencySummary(
        run_id=run_id,
        total_ms=total_ms,
        avg_turn_ms=avg_turn_ms,
        ttft_ms=first_ttft,
        total_llm_ms=sum(t.llm_ms for t in turn_latencies),
        total_tool_ms=sum(t.tool_ms for t in turn_latencies),
        total_overhead_ms=sum(t.overhead_ms for t in turn_latencies),
        tokens_per_second=tps,
        per_turn=turn_latencies,
        slowest_steps=slowest,
    )
