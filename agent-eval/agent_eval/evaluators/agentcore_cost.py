# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AgentCore cost insights module for agent evaluation.

Calculates Amazon Bedrock AgentCore runtime costs (compute, memory,
AgentCore Memory service) from trace-derived estimates, CloudWatch metrics,
or uploaded usage logs.

Usage:
    from agent_eval.evaluators.agentcore_cost import compute_agentcore_cost_summary
    summary = compute_agentcore_cost_summary(normalized_run)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Published AgentCore pricing (USD) — https://aws.amazon.com/bedrock/agentcore/pricing/
AGENTCORE_PRICING = {
    "vcpu_per_hour": 0.0895,
    "memory_gb_per_hour": 0.00945,
    "memory_event_per_1k": 0.25,
    "memory_storage_per_1k_per_month": 0.75,
    "memory_retrieval_per_1k": 0.50,
}

# Preset resource configurations
RESOURCE_CONFIGS = {
    "Small (1 vCPU / 512 MB)": {"vcpu": 1.0, "memory_gb": 0.5},
    "Medium (2 vCPU / 1 GB)": {"vcpu": 2.0, "memory_gb": 1.0},
    "Large (4 vCPU / 2 GB)": {"vcpu": 4.0, "memory_gb": 2.0},
    "X-Large (8 vCPU / 4 GB)": {"vcpu": 8.0, "memory_gb": 4.0},
}

DEFAULT_CONFIG_LABEL = "Small (1 vCPU / 512 MB)"

# Span operation names that indicate AgentCore Memory usage
# Supports both API-level names and SDK-level function names from real traces
MEMORY_EVENT_OPS = {"CreateEvent", "create_event"}
MEMORY_RETRIEVAL_OPS = {"RetrieveMemoryRecords", "retrieve_memories"}

# Scope names that indicate memory operations (from real AgentCore OTEL traces)
MEMORY_SCOPES = {"bedrock_agentcore.memory.client", "bedrock_agentcore.memory.integrations.strands.session_manager"}


@dataclass
class ResourceConfig:
    """AgentCore resource configuration."""
    vcpu: float = 1.0
    memory_gb: float = 0.5
    label: str = DEFAULT_CONFIG_LABEL


@dataclass
class UsageLogRecord:
    """Parsed usage log record (from uploaded files or legacy vended logs)."""
    session_id: str
    elapsed_time_seconds: float
    vcpu_hours_used: float
    gb_hours_used: float
    resource_arn: str = ""
    agent_name: str = ""
    event_timestamp: str = ""


@dataclass
class MemoryServiceCost:
    """AgentCore Memory service cost breakdown."""
    event_count: int = 0
    retrieval_count: int = 0
    event_cost: float = 0.0
    retrieval_cost: float = 0.0
    total_cost: float = 0.0


@dataclass
class AgentCoreCostSummary:
    """Complete AgentCore cost summary for a run."""
    compute_cost: float = 0.0
    memory_cost: float = 0.0
    agentcore_memory_cost: float = 0.0
    total_runtime_cost: float = 0.0
    session_duration_seconds: float = 0.0
    vcpu_hours_used: float = 0.0
    gb_hours_used: float = 0.0
    memory_service: Optional[MemoryServiceCost] = None
    resource_config: Optional[ResourceConfig] = None
    pricing_source: str = "estimated"  # "exact" | "estimated"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compute_cost_usd": round(self.compute_cost, 6),
            "memory_cost_usd": round(self.memory_cost, 6),
            "agentcore_memory_cost_usd": round(self.agentcore_memory_cost, 6),
            "total_runtime_cost_usd": round(self.total_runtime_cost, 6),
            "session_duration_seconds": round(self.session_duration_seconds, 3),
            "vcpu_hours_used": round(self.vcpu_hours_used, 8),
            "gb_hours_used": round(self.gb_hours_used, 8),
            "memory_service": {
                "event_count": self.memory_service.event_count,
                "retrieval_count": self.memory_service.retrieval_count,
                "event_cost_usd": round(self.memory_service.event_cost, 6),
                "retrieval_cost_usd": round(self.memory_service.retrieval_cost, 6),
                "total_cost_usd": round(self.memory_service.total_cost, 6),
            } if self.memory_service else None,
            "resource_config": {
                "vcpu": self.resource_config.vcpu,
                "memory_gb": self.resource_config.memory_gb,
                "label": self.resource_config.label,
            } if self.resource_config else None,
            "pricing_source": self.pricing_source,
        }


def parse_usage_logs(content: str) -> List[UsageLogRecord]:
    """Parse usage log content (JSON or JSONL)."""
    records = []
    for line in content.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            records.append(UsageLogRecord(
                session_id=rec.get("session.id", rec.get("session_id", "")),
                elapsed_time_seconds=float(rec.get("elapsed_time_seconds", 0)),
                vcpu_hours_used=float(rec.get("agent.runtime.vcpu.hours.used", 0)),
                gb_hours_used=float(rec.get("agent.runtime.memory.gb_hours.used", 0)),
                resource_arn=rec.get("resource_arn", ""),
                agent_name=rec.get("agent.name", rec.get("agent_name", "")),
                event_timestamp=rec.get("event_timestamp", ""),
            ))
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("Skipping malformed usage log line: %s", e)
    return records


def _compute_from_usage_logs(
    usage_logs: List[UsageLogRecord],
) -> tuple:
    """Compute cost from uploaded usage log data. Returns (vcpu_hrs, gb_hrs, duration_s)."""
    total_vcpu = sum(r.vcpu_hours_used for r in usage_logs)
    total_gb = sum(r.gb_hours_used for r in usage_logs)
    total_duration = sum(r.elapsed_time_seconds for r in usage_logs)
    return total_vcpu, total_gb, total_duration


def _compute_from_trace(
    normalized_run: Dict[str, Any],
    config: ResourceConfig,
) -> tuple:
    """Estimate cost from trace span timestamps. Returns (vcpu_hrs, gb_hrs, duration_s)."""
    turns = normalized_run.get("turns", [])
    if not turns:
        return 0.0, 0.0, 0.0

    # Find earliest start and latest end across all steps
    min_start = float("inf")
    max_end = 0.0
    for turn in turns:
        for step in turn.get("steps", []):
            start = step.get("start_time_ms") or step.get("start_time", 0)
            end = step.get("end_time_ms") or step.get("end_time", 0)
            if start and start < min_start:
                min_start = start
            if end and end > max_end:
                max_end = end

    if min_start == float("inf") or max_end == 0.0:
        return 0.0, 0.0, 0.0

    duration_s = (max_end - min_start) / 1000.0  # ms → seconds
    duration_hrs = duration_s / 3600.0
    vcpu_hrs = duration_hrs * config.vcpu
    gb_hrs = duration_hrs * config.memory_gb
    return vcpu_hrs, gb_hrs, duration_s


def extract_memory_operations(normalized_run: Dict[str, Any]) -> MemoryServiceCost:
    """Extract AgentCore Memory service operations from trace spans."""
    event_count = 0
    retrieval_count = 0

    for turn in normalized_run.get("turns", []):
        for step in turn.get("steps", []):
            name = (step.get("name") or "").strip()
            attrs = step.get("attributes") or {}
            raw = step.get("raw") or {}
            merged = {**raw, **attrs}

            # Check multiple attribute paths for operation name
            op = (
                merged.get("aws.operation.name")
                or merged.get("code.function.name")
                or merged.get("operation.name")
                or name
            )

            # Check scope for memory-specific scopes
            scope = merged.get("scope", merged.get("scope.name", ""))

            if op in MEMORY_EVENT_OPS:
                event_count += 1
            elif op in MEMORY_RETRIEVAL_OPS:
                retrieval_count += 1

    p = AGENTCORE_PRICING
    event_cost = (event_count / 1000.0) * p["memory_event_per_1k"]
    retrieval_cost = (retrieval_count / 1000.0) * p["memory_retrieval_per_1k"]

    return MemoryServiceCost(
        event_count=event_count,
        retrieval_count=retrieval_count,
        event_cost=event_cost,
        retrieval_cost=retrieval_cost,
        total_cost=event_cost + retrieval_cost,
    )


def compute_agentcore_cost_summary(
    normalized_run: Dict[str, Any],
    resource_config: Optional[ResourceConfig] = None,
    usage_logs: Optional[List[UsageLogRecord]] = None,
) -> AgentCoreCostSummary:
    """
    Compute AgentCore runtime cost summary.

    Args:
        normalized_run: Normalized run dict
        resource_config: Resource tier (defaults to Small 1 vCPU / 512 MB)
        usage_logs: Optional uploaded usage logs for resource consumption data

    Returns:
        AgentCoreCostSummary with compute, memory, and service cost breakdowns
    """
    config = resource_config or ResourceConfig()
    p = AGENTCORE_PRICING

    if usage_logs:
        vcpu_hrs, gb_hrs, duration_s = _compute_from_usage_logs(usage_logs)
        pricing_source = "exact"
    else:
        vcpu_hrs, gb_hrs, duration_s = _compute_from_trace(normalized_run, config)
        pricing_source = "estimated"

    compute_cost = vcpu_hrs * p["vcpu_per_hour"]
    memory_cost = gb_hrs * p["memory_gb_per_hour"]
    mem_svc = extract_memory_operations(normalized_run)

    return AgentCoreCostSummary(
        compute_cost=compute_cost,
        memory_cost=memory_cost,
        agentcore_memory_cost=mem_svc.total_cost,
        total_runtime_cost=compute_cost + memory_cost + mem_svc.total_cost,
        session_duration_seconds=duration_s,
        vcpu_hours_used=vcpu_hrs,
        gb_hours_used=gb_hrs,
        memory_service=mem_svc,
        resource_config=config,
        pricing_source=pricing_source,
    )
