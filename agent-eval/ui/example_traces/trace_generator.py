# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Trace Generator — Converts agent sessions into evaluation trace format.

Usage:
    from neo_trace_generator import TraceBuilder

    tb = TraceBuilder("agent-session-001")
    tb.user_message("What's happening with Acme Corp?")
    tb.tool_call("search_accounts", {"queryTerm": "Acme Corp"}, tool_run_id="t1")
    tb.tool_result("t1", '{"name": "Acme Corp", "id": "001xxx"}')
    tb.model_output("Acme Corp is a strategic account with $2.1M ARR...")
    tb.save("traces/acme-research.json")
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


class TraceBuilder:
    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"agent-{uuid.uuid4().hex[:8]}"
        self.trace_id = f"trace-{uuid.uuid4().hex[:12]}"
        self.events: list[dict] = []
        self._turn = 0
        self._tool_counter = 0
        self._clock = datetime.now(timezone.utc)

    def _tick(self, seconds: float = 1.0) -> str:
        self._clock += timedelta(seconds=seconds)
        return self._clock.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def _turn_id(self) -> str:
        return f"turn-{self._turn}"

    def user_message(self, content: str, delay: float = 2.0):
        self._turn += 1
        self.events.append({
            "timestamp": self._tick(delay),
            "type": "user_message",
            "turn_id": self._turn_id(),
            "content": content,
        })
        return self

    def tool_call(self, tool_name: str, arguments: dict, tool_run_id: str = None, delay: float = 0.5):
        self._tool_counter += 1
        tid = tool_run_id or f"tool-{self._tool_counter:03d}"
        self.events.append({
            "timestamp": self._tick(delay),
            "type": "tool_call",
            "turn_id": self._turn_id(),
            "tool_name": tool_name,
            "tool_run_id": tid,
            "tool_arguments": json.dumps(arguments),
        })
        return tid

    def tool_result(self, tool_run_id: str, result: str, status: str = "success", delay: float = 1.5):
        self.events.append({
            "timestamp": self._tick(delay),
            "type": "tool_result",
            "turn_id": self._turn_id(),
            "tool_run_id": tool_run_id,
            "tool_result": result,
            "status": status,
        })
        return self

    def model_output(self, content: str, delay: float = 2.0):
        self.events.append({
            "timestamp": self._tick(delay),
            "type": "model_output",
            "turn_id": self._turn_id(),
            "content": content,
        })
        return self

    def subagent_call(self, agent_name: str, query: str, delay: float = 0.5):
        tid = self.tool_call("use_subagent", {"agent_name": agent_name, "query": query}, delay=delay)
        return tid

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "events": self.events,
        }

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return str(p)
