#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
03_add_xray_steps_and_latency_v4.py

Merges an "index" turns file (Script 1) with a "detail" file (Script 2),
adding:
- final_answer
- finish_reason
- total_latency_ms
- steps_runtime (normalized, from actual spans/events only)
- steps_runtime_xray (xray-style view derived from steps_runtime)
- tool_runs (derived from steps_runtime)
- attribution (derived from tool_runs + overlap)

Key changes vs prior versions:
✅ DO NOT fabricate steps from user_query.
✅ turn_refs are treated as references and resolved to span objects when possible.
✅ Build span lookup from multiple possible keys (spans/subsegments/segments).
✅ If no resolvable evidence => steps_runtime stays empty (not wrong).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Helpers
# -------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts or not isinstance(ts, str):
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None

def _first_str(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _first_dict(d: Dict[str, Any], keys: List[str]) -> Optional[Dict[str, Any]]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, dict):
            return v
    return None

def _as_dict(v: Any) -> Optional[Dict[str, Any]]:
    return v if isinstance(v, dict) else None

def _canon_tool_run_id(step: Dict[str, Any]) -> Optional[str]:
    return _first_str(step, ["tool_use_id", "toolUseId", "tool_call_id", "toolCallId"])

def _looks_like_boilerplate(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    # Filter common prompt boilerplate / memory dumps
    lowers = t.lower()
    if "you are a helpful customer support agent" in lowers:
        return True
    if "these are user preferences" in lowers or "these are user facts" in lowers:
        return True
    if "<guidelines>" in lowers and "</guidelines>" in lowers:
        return True
    return False

def _looks_like_retrieval_results(text: str) -> bool:
    """
    Detect if text looks like tool output (retrieval results, KB chunks, etc.)
    even if no INVOCATION event was logged.
    """
    if not text:
        return False
    t = text.strip()
    lowers = t.lower()
    
    # Strong signals: structured retrieval output
    if "retrieved" in lowers and "document id:" in lowers:
        return True
    if "retrieved" in lowers and "score:" in lowers:
        return True
    if "document id: s3://" in lowers or "document id: arn:" in lowers:
        return True
    if lowers.startswith("retrieved") and ("results" in lowers or "documents" in lowers):
        return True
    
    # KB/metadata patterns
    if "metadata:" in lowers and "score:" in lowers:
        return True
    if "knowledge base" in lowers and "document" in lowers and "score:" in lowers:
        return True
    
    return False

def _json_preview(v: Any, max_len: int = 500) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v[:max_len] + ("…" if len(v) > max_len else "")
    if isinstance(v, (dict, list)):
        try:
            s = json.dumps(v, ensure_ascii=False)
            return s[:max_len] + ("…" if len(s) > max_len else "")
        except Exception:
            return None
    return str(v)[:max_len]

_RE_RETRIEVAL_RESULTS = re.compile(
    r"(?is)\bRetrieved\s+\d+\s+results\b.*\bScore:\s*[\d.]+\b.*\bDocument ID:\b"
)

_RE_TOOL_RESULT_JSON = re.compile(r'(?s)^\s*\{\s*"statusCode"\s*:\s*\d+\s*,\s*"body"\s*:\s*".*"\s*\}\s*$')

def _norm_whitespace(s: str) -> str:
    """Normalize whitespace for robust matching."""
    return re.sub(r"\s+", " ", (s or "").strip())

def looks_like_retrieval_results_text(text: Optional[str]) -> bool:
    if not text or not isinstance(text, str):
        return False
    return _RE_RETRIEVAL_RESULTS.search(text) is not None

def looks_like_tool_result_payload(text: Optional[str]) -> bool:
    """
    Detects tool-return payloads that some runtimes log as an LLM chunk:
    {"statusCode":200,"body":"..."}
    """
    if not text or not isinstance(text, str):
        return False
    return _RE_TOOL_RESULT_JSON.match(text) is not None

def extract_first_question_line(text: Optional[str]) -> Optional[str]:
    """
    Heuristic: many runtimes echo the user query or a prior query.
    Grab the first line ending with '?' (short-ish) if present.
    """
    if not text or not isinstance(text, str):
        return None
    for line in text.splitlines():
        s = line.strip()
        if s.endswith("?") and 3 <= len(s) <= 300:
            return s
    return None

def trace_stitch_suspect(index_user_query: Optional[str], steps: List[Dict[str, Any]]) -> bool:
    """
    Mark suspect if steps contain a different question echoed than the index turn's user_query.
    This catches situations where the trace contains a prior turn's question.
    """
    if not index_user_query:
        return False
    want = index_user_query.strip()
    for s in steps or []:
        if not isinstance(s, dict):
            continue
        txt = s.get("text")
        q = extract_first_question_line(txt)
        if q and q.strip() != want:
            return True
    return False

def count_distinct_questions(steps: List[Dict[str, Any]]) -> int:
    qs = set()
    for s in steps or []:
        if not isinstance(s, dict):
            continue
        q = extract_first_question_line(s.get("text"))
        if q:
            qs.add(q.strip())
    return len(qs)

def result_before_invocation_same_ts(steps: List[Dict[str, Any]]) -> bool:
    by_ts = {}
    for s in steps or []:
        ts = s.get("timestamp")
        if not ts:
            continue
        by_ts.setdefault(ts, []).append(s)
    
    # Sort timestamps for deterministic iteration
    for ts in sorted(by_ts.keys()):
        group = by_ts[ts]
        saw_result = any(
            (("LLM" in (g.get("type") or "").upper()) and looks_like_tool_result_payload(g.get("text")))
            for g in group if isinstance(g, dict)
        )
        saw_inv = any(
            ("INVOCATION" in (g.get("type") or "").upper())
            for g in group if isinstance(g, dict)
        )
        if saw_result and saw_inv:
            # If your raw order is preserved, this detects the case you pasted.
            first_result_idx = next((i for i, g in enumerate(group)
                                     if isinstance(g, dict) and ("LLM" in (g.get("type") or "").upper()) and looks_like_tool_result_payload(g.get("text"))), None)
            first_inv_idx = next((i for i, g in enumerate(group)
                                  if isinstance(g, dict) and ("INVOCATION" in (g.get("type") or "").upper())) , None)
            if first_result_idx is not None and first_inv_idx is not None and first_result_idx < first_inv_idx:
                return True
    return False

# -------------------------
# Phase classification (deterministic)
# -------------------------

def is_prompt_context_chunk(s: Dict[str, Any]) -> bool:
    """
    Detect prompt assembly boilerplate that should be excluded from analysis.
    Expanded to catch all common prompt scaffolding patterns.
    
    Strips retrieval dumps regardless of event type for consistent cleaning.
    """
    t = s.get("text") or ""
    
    # Retrieval results dump (not assistant generation) - check first, regardless of type
    if looks_like_retrieval_results_text(t):
        return True
    
    # Type-specific checks only for LLM_OUTPUT_CHUNK
    if (s.get("type") or "").upper() != "LLM_OUTPUT_CHUNK":
        return False
    
    # System prompt / guidelines
    if "You are a helpful customer support agent" in t:
        return True
    if "<guidelines>" in t and "</guidelines>" in t:
        return True
    
    # User preferences / facts injection
    if "These are user preferences:" in t:
        return True
    if "These are user facts:" in t:
        return True
    
    # Very short chunks that are just whitespace or newlines
    if len(t.strip()) < 3:
        return True
    
    return False

def find_turn_anchor_idx(steps: List[Dict[str, Any]], user_query: str, prefix_len: int = 80) -> Optional[int]:
    """
    Find the LAST LLM_OUTPUT_CHUNK that contains the current turn's user_query.
    This is the anchor point where the actual turn begins.
    Using 'last' ensures we cut at the correct point even if earlier queries appear.
    
    Uses normalized whitespace matching on first N chars for robustness against
    truncation, chunking, and whitespace variations.
    
    Tries both prefix and mid-slice to handle cases where runtime includes
    only a truncated middle of the query.
    """
    if not user_query:
        return None
    
    uq = _norm_whitespace(user_query)
    needles = [uq[:prefix_len]]
    
    # Add mid-slice as fallback for truncated queries
    if len(uq) > prefix_len + 40:
        mid = uq[len(uq)//2 : len(uq)//2 + prefix_len]
        needles.append(mid)
    
    # Filter out empty needles
    needles = [n for n in needles if n]
    
    if not needles:
        return None
    
    anchor = None
    for i, s in enumerate(steps or []):
        if not isinstance(s, dict):
            continue
        if (s.get("type") or "").upper() == "LLM_OUTPUT_CHUNK":
            text = _norm_whitespace(s.get("text") or "")
            if any(needle in text for needle in needles):
                anchor = i
    return anchor

def crop_to_user_query_anchor(steps: List[Dict[str, Any]], user_query: str) -> List[Dict[str, Any]]:
    """
    Crop steps to start from the anchor point (last occurrence of user_query).
    This removes prior turn contamination.
    """
    anchor = find_turn_anchor_idx(steps, user_query)
    if anchor is not None:
        return steps[anchor:]
    return steps

def strip_prompt_context(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove prompt assembly boilerplate chunks.
    """
    return [s for s in steps if not is_prompt_context_chunk(s)]

def dedupe_tool_invocations(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dedupe INVOCATION events by tool_use_id/tool_call_id.
    Prevents cross-turn tool contamination.
    Keep only the FIRST occurrence of each tool_use_id.
    Handles both "INVOCATION" and "TOOL_CALL" type patterns.
    """
    seen = set()
    deduped = []
    for s in steps or []:
        if not isinstance(s, dict):
            deduped.append(s)
            continue
        
        stype = (s.get("type") or "").upper()
        tid = s.get("tool_use_id") or s.get("tool_call_id")
        
        # Check for both INVOCATION and TOOL patterns
        is_tool = ("INVOCATION" in stype) or ("TOOL" in stype)
        
        if is_tool and tid:
            if tid in seen:
                # Skip duplicate - this is replay/noise from trace stitching
                continue
            seen.add(tid)
        
        deduped.append(s)
    return deduped

def compute_normalized_latency(steps: List[Dict[str, Any]]) -> Optional[float]:
    """
    Compute latency from actual generation steps (not trace envelope).
    Returns milliseconds between first and last step timestamp.
    """
    if not steps:
        return None
    
    timestamps = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        ts_str = s.get("timestamp")
        if ts_str:
            ts = parse_ts(ts_str)
            if ts:
                timestamps.append(ts)
    
    if len(timestamps) < 2:
        return None
    
    timestamps.sort()
    delta = timestamps[-1] - timestamps[0]
    return delta.total_seconds() * 1000  # convert to milliseconds

def compute_post_tool_latency(steps: List[Dict[str, Any]]) -> Optional[float]:
    """
    Compute latency from first TOOL_CALL to last step.
    Measures time spent in tool execution + final generation.
    Falls back to type-based detection if phase is not available.
    """
    if not steps:
        return None
    
    timestamps = []
    started = False
    
    for s in steps:
        if not isinstance(s, dict):
            continue
        
        # Try phase first, fallback to type detection
        phase = s.get("phase")
        stype = (s.get("type") or "").upper()
        
        if not started:
            # Check phase or type for tool invocation
            if phase == "TOOL_CALL" or ("INVOCATION" in stype) or ("TOOL" in stype):
                started = True
        
        if started:
            ts_str = s.get("timestamp")
            if ts_str:
                ts = parse_ts(ts_str)
                if ts:
                    timestamps.append(ts)
    
    if len(timestamps) < 2:
        return None
    
    timestamps.sort()
    delta = timestamps[-1] - timestamps[0]
    return delta.total_seconds() * 1000

def classify_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deterministic phase classification for observability:
    - PRE_TOOL_GENERATION: assistant thinking/planning before first tool call
    - TOOL_CALL: actual tool invocations
    - FINAL_GENERATION: everything after first tool call (actual model generation)
    
    Note: This is for tagging only. Filtering is done by strip_prompt_context().
    Pre-tool generation (assistant planning) should NOT be filtered out.
    """
    seen_tool = False
    out = []
    
    for s in steps or []:
        if not isinstance(s, dict):
            out.append(s)
            continue
        
        # Make a copy to avoid mutating original
        s_copy = dict(s)
        t = (s_copy.get("type") or "").upper()
        
        if ("INVOCATION" in t) or ("TOOL" in t):
            seen_tool = True
            s_copy["phase"] = "TOOL_CALL"
        elif not seen_tool:
            s_copy["phase"] = "PRE_TOOL_GENERATION"
        else:
            s_copy["phase"] = "FINAL_GENERATION"
        
        out.append(s_copy)
    
    return out

def build_xray_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Tag steps with phases for observability.
    Does NOT filter - all content after strip_prompt_context() is kept.
    """
    return classify_steps(steps)

def build_steps_runtime_clean(steps: List[Dict[str, Any]], user_query: str) -> List[Dict[str, Any]]:
    """
    Build clean steps for a turn by:
    1. Stripping prompt context boilerplate (guidelines, preferences, retrieval dumps)
    2. Deduping tool invocations (prevents cross-turn tool ID reuse)
    3. Classifying phases (PROMPT_CONTEXT / TOOL_CALL / FINAL_GENERATION)
    4. Filtering to relevant phases only (excludes PROMPT_CONTEXT)
    
    Note: Cropping to user_query anchor should be done BEFORE calling this function.
    This ensures clean, deterministic steps for evaluation and X-Ray views.
    """
    steps = strip_prompt_context(steps)
    steps = dedupe_tool_invocations(steps)
    steps = build_xray_steps(steps)
    return steps

def extract_context_injections(steps: List[Dict[str, Any]], max_events: int = 10) -> List[Dict[str, Any]]:
    """
    Extract context injections (facts, preferences, policies) that were
    injected into the prompt but are not assistant-generated output.
    
    Should be called on cropped steps (after anchor) but before stripping,
    so injection chunks are still available.
    
    Only captures injections from first N events to avoid over-capture
    from multi-turn traces or quoted context.
    """
    injections = []
    
    for i, s in enumerate(steps or []):
        # Only check first N events to avoid over-capture
        if i >= max_events:
            break
        
        if not isinstance(s, dict):
            continue
        
        text = s.get("text") or ""
        if not text or len(text) < 40:  # Skip very short chunks
            continue
        
        # Detect injection patterns
        if "These are user preferences:" in text:
            injections.append({
                "type": "USER_PREFERENCES",
                "timestamp": s.get("timestamp"),
                "content_preview": text[:500],
            })
        elif "These are user facts:" in text:
            injections.append({
                "type": "USER_FACTS",
                "timestamp": s.get("timestamp"),
                "content_preview": text[:500],
            })
        elif "You are a helpful customer support agent" in text:
            injections.append({
                "type": "SYSTEM_PROMPT",
                "timestamp": s.get("timestamp"),
                "content_preview": text[:500],
            })
    
    return injections

# -------------------------
# Span extraction + normalization
# -------------------------

def normalize_step_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    """
    Script 2 already emits normalized step events (LLM_OUTPUT_CHUNK / INVOCATION).
    We should NOT run span_to_step() here. We only:
    - normalize field names
    - coalesce tool ids
    - coalesce tool args/input
    - keep text as-is
    """
    out = {
        "timestamp": _first_str(ev, ["timestamp", "ts", "time"]),
        "type": _first_str(ev, ["type", "event_type", "eventType"]) or "EVENT",
        "tool_name": _first_str(ev, ["tool_name", "toolName"]),
        "tool_arguments": _as_dict(ev.get("tool_arguments")) or _as_dict(ev.get("toolArguments")),
        "tool_input": _as_dict(ev.get("tool_input")) or _as_dict(ev.get("toolInput")),
        "tool_call_id": _first_str(ev, ["tool_call_id", "toolCallId"]),
        "tool_use_id": _first_str(ev, ["tool_use_id", "toolUseId"]),
        "text": _first_str(ev, ["text", "message", "content"]),
    }
    
    # coalesce tool ids if only one present
    if not out["tool_use_id"] and out["tool_call_id"]:
        out["tool_use_id"] = out["tool_call_id"]
    if not out["tool_call_id"] and out["tool_use_id"]:
        out["tool_call_id"] = out["tool_use_id"]
    
    # coalesce tool args/input
    if out["tool_arguments"] is None and out["tool_input"] is not None:
        out["tool_arguments"] = out["tool_input"]
    if out["tool_input"] is None and out["tool_arguments"] is not None:
        out["tool_input"] = out["tool_arguments"]
    
    return out

def _span_id(span: Dict[str, Any]) -> Optional[str]:
    # common ids in trace/span data
    return _first_str(span, ["span_id", "id", "subsegment_id", "segment_id", "SpanId", "spanId"])

def _span_ts(span: Dict[str, Any]) -> Optional[str]:
    return _first_str(span, ["timestamp", "start_time", "startTime", "ts", "time"])

def _infer_type_from_span(span: Dict[str, Any]) -> str:
    """
    Infer a stable step type. We prefer tool-ish signals.
    """
    # explicit type
    t = _first_str(span, ["type", "event_type", "eventType", "kind", "op"])
    if t:
        return t

    # tool-ish keys
    if _first_str(span, ["tool_name", "toolName", "tool", "functionName", "function", "operation", "target"]):
        return "INVOCATION"

    # llm-ish keys
    if any(k in span for k in ["model", "completion", "prompt_tokens", "completion_tokens", "finish_reason"]):
        return "LLM"

    # xray style "name"/"namespace"
    name = _first_str(span, ["name", "span_name", "spanName"])
    namespace = _first_str(span, ["namespace"])
    if namespace:
        return f"XRAY:{namespace}"
    if name:
        return "SPAN"

    return "EVENT"

def span_to_step(span: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a raw span/subsegment into our normalized step schema.
    NOTE: This should only be called for real span objects (not user_query).
    """
    typ = _infer_type_from_span(span)
    tool_name = _first_str(span, ["tool_name", "toolName", "tool", "functionName", "function", "operation", "target"])

    tool_args = _first_dict(span, ["tool_arguments", "toolArguments", "arguments", "args", "parameters"])
    tool_inp = _first_dict(span, ["tool_input", "toolInput", "input", "request", "payload"])

    # If span has richer I/O but not dict-shaped, keep a preview in text
    text = _first_str(span, ["text", "message", "content"])
    if not text:
        # try output-ish keys
        outv = None
        for k in ["output", "response", "result", "completion", "summary"]:
            if k in span:
                outv = span.get(k)
                break
        text = _json_preview(outv)

    return {
        "timestamp": _span_ts(span),
        "type": typ,
        "tool_name": tool_name,
        "tool_arguments": tool_args,
        "tool_input": tool_inp,
        "tool_call_id": _first_str(span, ["tool_call_id", "toolCallId", "call_id", "callId"]),
        "tool_use_id": _first_str(span, ["tool_use_id", "toolUseId", "use_id", "useId"]),
        "text": text,
    }

def build_span_index(detail: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build a global lookup of spans by id from multiple possible locations.
    We intentionally scan broad structures because Script 2 outputs vary.
    """
    index: Dict[str, Dict[str, Any]] = {}

    def collect_from(obj: Any) -> None:
        if isinstance(obj, dict):
            # common span arrays
            for k in ["spans", "subsegments", "segments", "trace_spans", "span_events"]:
                arr = obj.get(k)
                if isinstance(arr, list):
                    for it in arr:
                        if isinstance(it, dict):
                            sid = _span_id(it)
                            if sid and sid not in index:
                                index[sid] = it

            # recurse
            for v in obj.values():
                collect_from(v)
        elif isinstance(obj, list):
            for it in obj:
                collect_from(it)

    collect_from(detail)
    return index

def resolve_turn_refs_to_steps(turn_refs: Any, span_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    turn_refs can be:
      - list of strings (span ids)
      - list of dicts (may contain span_id/id, or may already be span-ish)
    We try:
      1) if dict looks like a full span => use it
      2) else if dict has id/span_id => deref from span_index
      3) else if string => deref
    Anything we can't resolve => ignore (do NOT fabricate).
    """
    if not isinstance(turn_refs, list):
        return []

    steps: List[Dict[str, Any]] = []

    for ref in turn_refs:
        span_obj: Optional[Dict[str, Any]] = None

        if isinstance(ref, dict):
            # if it already looks like a span (has name/namespace/duration/etc.), accept it
            if any(k in ref for k in ["name", "namespace", "subsegments", "annotations", "metadata", "duration_ms"]):
                span_obj = ref
            else:
                rid = _span_id(ref)
                if rid and rid in span_index:
                    span_obj = span_index[rid]
        elif isinstance(ref, str):
            rid = ref.strip()
            if rid and rid in span_index:
                span_obj = span_index[rid]

        if span_obj:
            step = span_to_step(span_obj)
            # keep only meaningful steps: must have something besides timestamp
            if step.get("type") or step.get("tool_name") or step.get("text"):
                steps.append(step)

    return steps

# -------------------------
# Data models
# -------------------------

@dataclass
class DetailTurn:
    session_id: Optional[str]
    trace_id: Optional[str]
    request_id: Optional[str]
    span_id: Optional[str]
    timestamp: Optional[str]
    user_query: Optional[str]
    final_answer: Optional[str]
    finish_reason: Optional[str]
    total_latency_ms: Optional[float]
    steps_runtime: List[Dict[str, Any]]
    trace_start_ts: Optional[str]
    trace_end_ts: Optional[str]
    _raw: Dict[str, Any]


def _coalesce_session_id(candidate: Optional[str], parent_session_id: Optional[str]) -> Optional[str]:
    if not candidate or candidate == "ALL_SESSIONS":
        return parent_session_id
    return candidate

def _coalesce_trace_id(candidate: Optional[str], parent_trace_id: Optional[str]) -> Optional[str]:
    if not candidate:
        return parent_trace_id
    return candidate

# -------------------------
# Flatten detail
# -------------------------

def flatten_detail(detail: Dict[str, Any], span_index: Dict[str, Dict[str, Any]], debug: bool = False) -> Tuple[List[DetailTurn], Dict[str, Any]]:
    stats = {
        "detail_candidates_found": 0,
        "detail_norm_records": 0,
        "flatten_paths_used": [],
        "trace_steps_aggregated": 0,
        "traces_backfilled_with_steps": 0,
        "span_index_size": len(span_index),
    }

    turns_raw: List[Tuple[Dict[str, Any], Optional[str], Optional[str]]] = []

    def add_turns(container_turns: Any, parent_session_id: Optional[str], parent_trace_id: Optional[str], path_label: str) -> None:
        if isinstance(container_turns, list):
            if path_label not in stats["flatten_paths_used"]:
                stats["flatten_paths_used"].append(path_label)
            for t in container_turns:
                if isinstance(t, dict):
                    turns_raw.append((t, parent_session_id, parent_trace_id))

    # A) top-level turns
    add_turns(detail.get("turns"), None, None, "root.turns")
    add_turns(detail.get("turns_merged_normalized"), None, None, "root.turns_merged_normalized")

    sessions = detail.get("sessions")
    if isinstance(sessions, list):
        for s in sessions:
            if not isinstance(s, dict):
                continue
            parent_session_id = s.get("session_id")

            add_turns(s.get("turns"), parent_session_id, None, "sessions[].turns")
            add_turns(s.get("turns_merged_normalized"), parent_session_id, None, "sessions[].turns_merged_normalized")

            traces = s.get("traces")
            if isinstance(traces, list):
                for tr in traces:
                    if not isinstance(tr, dict):
                        continue
                    parent_trace_id = tr.get("trace_id")

                    # Primary
                    add_turns(tr.get("turns_enriched"), parent_session_id, parent_trace_id, "sessions[].traces[].turns_enriched")
                    # Older/optional
                    add_turns(tr.get("turns"), parent_session_id, parent_trace_id, "sessions[].traces[].turns")
                    add_turns(tr.get("turns_merged_normalized"), parent_session_id, parent_trace_id, "sessions[].traces[].turns_merged_normalized")

                    # Trace-level fallback: synth record (no fabricated steps)
                    if isinstance(tr.get("trace_enrichment"), dict):
                        if "sessions[].traces[].trace_enrichment" not in stats["flatten_paths_used"]:
                            stats["flatten_paths_used"].append("sessions[].traces[].trace_enrichment")
                        enr = tr["trace_enrichment"]
                        synth = {
                            "session_id": parent_session_id,
                            "trace_id": parent_trace_id,
                            "final_answer": enr.get("final_answer"),
                            "finish_reason": enr.get("finish_reason"),
                            "total_latency_ms": enr.get("total_latency_ms"),
                            "trace_start_ts": enr.get("start_ts"),
                            "trace_end_ts": enr.get("end_ts"),
                        }
                        turns_raw.append((synth, parent_session_id, parent_trace_id))

                    # NEW: collect trace-level steps from turn_refs by resolving to spans
                    # We attach these steps later (backfill) when a turn record has no steps_runtime.
                    if "sessions[].traces[].turn_refs" not in stats["flatten_paths_used"] and tr.get("turn_refs") is not None:
                        stats["flatten_paths_used"].append("sessions[].traces[].turn_refs")

    stats["detail_candidates_found"] = len(turns_raw)

    # Build trace->steps map by resolving turn_refs for each trace
    trace_steps: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    if isinstance(sessions, list):
        for s in sessions:
            if not isinstance(s, dict):
                continue
            sid = s.get("session_id")
            traces = s.get("traces")
            if not (sid and isinstance(traces, list)):
                continue
            for tr in traces:
                if not isinstance(tr, dict):
                    continue
                tid = tr.get("trace_id")
                if not tid:
                    continue
                
                # CRITICAL FIX: Extract steps_runtime from trace_enrichment
                te = tr.get("trace_enrichment")
                if isinstance(te, dict) and isinstance(te.get("steps_runtime"), list):
                    # Use steps from trace_enrichment (from Script 2)
                    # IMPORTANT: These are already normalized events, NOT spans
                    raw_steps = [x for x in te["steps_runtime"] if isinstance(x, dict)]
                    steps_norm = [normalize_step_event(rs) for rs in raw_steps]
                    steps_norm = dedupe_invocations(steps_norm)
                    if steps_norm:
                        trace_steps[(sid, tid)] = steps_norm
                        stats["trace_steps_aggregated"] += len(steps_norm)
                
                # Fallback: resolve from turn_refs if no trace_enrichment steps
                elif tr.get("turn_refs") is not None:
                    refs = tr.get("turn_refs")
                    steps = resolve_turn_refs_to_steps(refs, span_index)
                    if steps:
                        trace_steps[(sid, tid)] = steps
                        stats["trace_steps_aggregated"] += len(steps)

    detail_turns: List[DetailTurn] = []

    for t, parent_session_id, parent_trace_id in turns_raw:
        sid = _coalesce_session_id(t.get("session_id"), parent_session_id)
        tid = _coalesce_trace_id(t.get("trace_id"), parent_trace_id)

        tl = t.get("total_latency_ms")
        total_latency_ms: Optional[float] = None
        if tl is not None and tl != "" and tl != "null":
            try:
                total_latency_ms = float(tl)
            except Exception:
                total_latency_ms = None

        # IMPORTANT: Only accept steps_runtime if it looks real. Otherwise backfill from resolved trace steps.
        steps_runtime: List[Dict[str, Any]] = []
        if isinstance(t.get("steps_runtime"), list):
            # Script 2 already emits normalized events - use normalize_step_event, NOT span_to_step
            raw_steps = [x for x in t["steps_runtime"] if isinstance(x, dict)]
            for rs in raw_steps:
                # accept if it has any step-like keys (NOT user_query)
                if any(k in rs for k in ["tool_name", "toolName", "tool", "function", "functionName", "name", "namespace", "type", "event_type", "eventType"]):
                    steps_runtime.append(normalize_step_event(rs))
            # if nothing usable, treat as empty so we can backfill
            if not steps_runtime:
                steps_runtime = []

        # Backfill from trace_steps only if we have session+trace and no steps yet
        if not steps_runtime and sid and tid and (sid, tid) in trace_steps:
            steps_runtime = trace_steps[(sid, tid)]
            stats["traces_backfilled_with_steps"] += 1

        dt = DetailTurn(
            session_id=sid,
            trace_id=tid,
            request_id=t.get("request_id"),
            span_id=t.get("span_id"),
            timestamp=t.get("timestamp"),
            user_query=t.get("user_query"),
            final_answer=t.get("final_answer"),
            finish_reason=t.get("finish_reason"),
            total_latency_ms=total_latency_ms,
            steps_runtime=steps_runtime,
            trace_start_ts=t.get("trace_start_ts") or t.get("start_ts"),
            trace_end_ts=t.get("trace_end_ts") or t.get("end_ts"),
            _raw=t,
        )
        detail_turns.append(dt)

    stats["detail_norm_records"] = len(detail_turns)

    if debug:
        print("DEBUG: span_index_size:", stats["span_index_size"])
        print("DEBUG: trace_steps_aggregated:", stats["trace_steps_aggregated"])
        # show one example of a step if available
        for d in detail_turns:
            if d.steps_runtime:
                print("DEBUG: sample step:", d.steps_runtime[0])
                break

    return detail_turns, stats

# -------------------------
# Matching & merge
# -------------------------

def build_detail_indexes(detail_turns: List[DetailTurn]) -> Dict[str, Dict[Any, DetailTurn]]:
    def score(d: DetailTurn) -> int:
        return (10 if d.steps_runtime else 0) + (5 if d.final_answer else 0) + (3 if d.total_latency_ms is not None else 0)

    idx_session_trace: Dict[Tuple[str, str], DetailTurn] = {}
    idx_session_request: Dict[Tuple[str, str], DetailTurn] = {}
    idx_session_span: Dict[Tuple[str, str], DetailTurn] = {}

    for d in detail_turns:
        if d.session_id and d.trace_id:
            key = (d.session_id, d.trace_id)
            if key not in idx_session_trace or score(d) > score(idx_session_trace[key]):
                idx_session_trace[key] = d
        if d.session_id and d.request_id:
            key = (d.session_id, d.request_id)
            if key not in idx_session_request or score(d) > score(idx_session_request[key]):
                idx_session_request[key] = d
        if d.session_id and d.span_id:
            key = (d.session_id, d.span_id)
            if key not in idx_session_span or score(d) > score(idx_session_span[key]):
                idx_session_span[key] = d

    return {
        "session+trace": idx_session_trace,
        "session+request": idx_session_request,
        "session+span": idx_session_span,
    }

def dedupe_invocations(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Script 2 may emit two INVOCATION events for the same tool call:
    - one with tool_input + tool_use_id
    - one with tool_arguments + tool_call_id
    We dedupe by canonical tool_run_id + tool_name + timestamp.
    We keep the richer record (prefers tool_arguments, then tool_input).
    
    Handles both INVOCATION and TOOL* type events (TOOL_CALL, TOOL_INVOCATION, etc.)
    """
    out: List[Dict[str, Any]] = []
    seen: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    
    for s in steps:
        if not isinstance(s, dict):
            continue
        stype = (s.get("type") or "").upper()
        # Treat both INVOCATION and TOOL* events as tool events
        is_tool = ("INVOCATION" in stype) or ("TOOL" in stype)
        if not is_tool:
            out.append(s)
            continue
        
        run_id = _canon_tool_run_id(s) or ""
        tname = s.get("tool_name") or ""
        ts = s.get("timestamp") or ""
        key = (run_id, tname, ts)
        
        prev = seen.get(key)
        if prev is None:
            seen[key] = s
        else:
            # prefer record with tool_arguments > tool_input
            prev_score = 1 if prev.get("tool_arguments") else (0 if prev.get("tool_input") else -1)
            cur_score = 1 if s.get("tool_arguments") else (0 if s.get("tool_input") else -1)
            if cur_score > prev_score:
                seen[key] = s
    
    # rebuild: combine non-tool events with deduped tool events, then sort by timestamp
    invocs = list(seen.values())
    invocs.sort(key=lambda x: (x.get("timestamp") or "", x.get("tool_name") or "", _canon_tool_run_id(x) or ""))
    
    # Filter out ALL tool events (not just INVOCATION) to prevent double-inclusion
    combined = [s for s in out if not (("INVOCATION" in (s.get("type") or "").upper()) or ("TOOL" in (s.get("type") or "").upper()))] + invocs
    
    # Sort by timestamp with deterministic secondary keys for stable ordering
    combined.sort(key=lambda x: (
        x.get("timestamp") or "",
        x.get("tool_name") or "",
        _canon_tool_run_id(x) or "",
        (x.get("type") or "").upper()
    ))
    return combined

def build_steps_runtime_xray(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build an X-Ray-style timeline from already-clean steps.
    
    IMPORTANT: Input steps are expected to be already cleaned by build_steps_runtime_clean():
    - Cropped to turn anchor
    - Prompt context stripped
    - Tools deduped
    - Phases tagged
    
    This function only does projection into X-Ray event format (type classification).
    Does NOT re-filter or re-classify.
    """
    out: List[Dict[str, Any]] = []
    
    for s in steps or []:
        if not isinstance(s, dict):
            continue
        
        stype = (s.get("type") or "").upper()
        ts = s.get("timestamp")
        txt = s.get("text")
        tool_name = s.get("tool_name")
        tool_run_id = s.get("tool_use_id") or s.get("tool_call_id")
        phase = s.get("phase")
        
        if "INVOCATION" in stype or "TOOL" in stype:
            out.append({
                "timestamp": ts,
                "type": "TOOL_CALL",
                "tool_name": tool_name,
                "tool_run_id": tool_run_id,
                "input": s.get("tool_input") or s.get("tool_arguments"),
                "phase": phase,
            })
            continue
        
        if "LLM" in stype:
            if looks_like_tool_result_payload(txt):
                out.append({
                    "timestamp": ts,
                    "type": "TOOL_RESULT_PAYLOAD",
                    "text_preview": _json_preview(txt, max_len=900),
                    "phase": phase,
                })
            elif looks_like_retrieval_results_text(txt):
                out.append({
                    "timestamp": ts,
                    "type": "RETRIEVAL_RESULTS_TEXT",
                    "text_preview": _json_preview(txt, max_len=900),
                    "phase": phase,
                })
            else:
                out.append({
                    "timestamp": ts,
                    "type": "LLM_GENERATION",
                    "text_preview": _json_preview(txt, max_len=900),
                    "phase": phase,
                })
            continue
        
        # fallback
        out.append({
            "timestamp": ts,
            "type": s.get("type") or "EVENT",
            "text_preview": _json_preview(txt, max_len=900),
            "phase": phase,
        })
    
    return out

def build_tool_runs(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tool_runs: List[Dict[str, Any]] = []
    for s in steps or []:
        if not isinstance(s, dict):
            continue
        stype = (s.get("type") or "").upper()
        tname = s.get("tool_name")
        if not tname:
            continue
        if "INVOCATION" in stype or "TOOL" in stype:
            tool_runs.append({
                "tool_name": tname,
                "timestamp": s.get("timestamp"),
                "tool_run_id": s.get("tool_use_id") or s.get("tool_call_id"),
                "tool_arguments": s.get("tool_arguments"),
                "tool_input": s.get("tool_input"),
                "tool_call_id": s.get("tool_call_id"),
                "tool_use_id": s.get("tool_use_id"),
            })
    return tool_runs

def merge_turn(index_turn: Dict[str, Any], detail: DetailTurn) -> Dict[str, Any]:
    out = dict(index_turn)

    if detail.final_answer is not None:
        out["final_answer"] = detail.final_answer
    if detail.finish_reason is not None:
        out["finish_reason"] = detail.finish_reason
    if detail.total_latency_ms is not None:
        latency = detail.total_latency_ms
        if abs(latency - int(latency)) < 1e-9:
            out["total_latency_ms"] = int(latency)
        else:
            out["total_latency_ms"] = latency

    # Steps
    if detail.steps_runtime:
        out["steps_runtime"] = detail.steps_runtime  # preserve raw for debugging
        
        # Extract context injections from cropped steps (before stripping)
        uq = out.get("user_query") or ""
        anchor = find_turn_anchor_idx(detail.steps_runtime, uq)
        
        # Safety: if anchor not found and we have a user_query, avoid whole-trace contamination
        if uq and anchor is None:
            # Anchor detection failed - don't use entire trace for either extraction
            anchor_failed = True
            raw_cropped = []
        else:
            anchor_failed = False
            raw_cropped = detail.steps_runtime[anchor:] if anchor is not None else detail.steps_runtime
        
        if anchor_failed:
            out["context_injections"] = []
            out["steps_runtime_clean"] = []
        else:
            out["context_injections"] = extract_context_injections(raw_cropped)
            # Build clean steps from already-cropped data
            out["steps_runtime_clean"] = build_steps_runtime_clean(raw_cropped, uq)
        
        out["steps_runtime_xray"] = build_steps_runtime_xray(out["steps_runtime_clean"])
        out["tool_runs"] = build_tool_runs(out["steps_runtime_clean"])
        
        # Compute granular latency metrics
        normalized_latency = compute_normalized_latency(out["steps_runtime_clean"])
        if normalized_latency is not None:
            out["normalized_latency_ms"] = int(normalized_latency) if abs(normalized_latency - int(normalized_latency)) < 1e-9 else normalized_latency
        
        post_tool_latency = compute_post_tool_latency(out["steps_runtime_clean"])
        if post_tool_latency is not None:
            out["post_tool_latency_ms"] = int(post_tool_latency) if abs(post_tool_latency - int(post_tool_latency)) < 1e-9 else post_tool_latency
    else:
        out["steps_runtime_clean"] = []
        out["steps_runtime_xray"] = []
        out["tool_runs"] = []
        out["context_injections"] = []

    if detail.trace_start_ts:
        out["trace_start_ts"] = detail.trace_start_ts
    if detail.trace_end_ts:
        out["trace_end_ts"] = detail.trace_end_ts

    # --- Attribution (4-state) ---
    steps = out.get("steps_runtime_clean", [])  # use clean steps for attribution
    has_invocation = any(
        isinstance(s, dict)
        and (s.get("tool_name") is not None)
        and ("INVOCATION" in (s.get("type") or "").upper() or "TOOL" in (s.get("type") or "").upper())
        for s in steps
    )
    
    has_toolish_text = any(
        isinstance(s, dict)
        and ("LLM" in (s.get("type") or "").upper())
        and (looks_like_retrieval_results_text(s.get("text")) or looks_like_tool_result_payload(s.get("text")))
        for s in steps
    )
    
    stitch_suspect = trace_stitch_suspect(out.get("user_query"), steps)
    distinct_q = count_distinct_questions(steps)
    multi_query_trace = distinct_q >= 2
    order_anomaly = result_before_invocation_same_ts(steps)
    
    if has_invocation:
        verdict = "TOOL_USED"
    elif has_toolish_text:
        verdict = "TOOL_OUTPUT_ONLY"   # tool-ish evidence, but no invocation event captured
    elif stitch_suspect:
        verdict = "TRACE_STITCH_SUSPECT"
    else:
        verdict = "NO_TOOL_EVIDENCE"
    
    # Surface stitch suspicion and multi-query even when tools are used
    if stitch_suspect:
        verdict = verdict + "+TRACE_STITCH_SUSPECT"
    if multi_query_trace:
        verdict = verdict + "+MULTI_QUERY_TRACE"
    
    out["attribution"] = {
        "used_tool": bool(out.get("tool_runs")),   # strict: only true when we have real tool runs
        "verdict": verdict,
        "has_invocation": has_invocation,
        "has_toolish_text": has_toolish_text,
        "trace_stitch_suspect": stitch_suspect,
        "multi_query_trace": multi_query_trace,
        "distinct_question_count": distinct_q,
        "order_anomaly": order_anomaly,
    }

    return out

# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="session_turns.json from Script 1")
    ap.add_argument("--detail", required=True, help="session_enriched_runtime.json from Script 2")
    ap.add_argument("--out", required=True, help="output file")
    ap.add_argument("--debug", action="store_true", help="print debug info")
    ap.add_argument("--ts-window-seconds", type=int, default=0, help="optional timestamp fallback window; 0 disables")
    ap.add_argument("--min-spans", type=int, default=1, help="Require at least N detail records per (session_id, trace_id)")
    args = ap.parse_args()

    index = load_json(args.index)
    detail = load_json(args.detail)

    index_turns = index.get("turns") or []
    if not isinstance(index_turns, list):
        raise ValueError("Index file must contain 'turns' list")

    span_index = build_span_index(detail)

    detail_turns, flat_stats = flatten_detail(detail, span_index, debug=args.debug)

    # min-spans gating
    counts: Dict[Tuple[str, str], int] = {}
    for d in detail_turns:
        if d.session_id and d.trace_id:
            counts[(d.session_id, d.trace_id)] = counts.get((d.session_id, d.trace_id), 0) + 1
    allowed = {k for k, c in counts.items() if c >= args.min_spans} if args.min_spans > 1 else None

    idx_maps = build_detail_indexes(detail_turns)

    matched_by_strategy = {"session+trace": 0, "session+request": 0, "session+span": 0, "timestamp-fallback": 0, "none": 0}
    unmatched = 0
    blocked_by_min_spans = 0

    detail_ts_list: List[Tuple[datetime, DetailTurn]] = []
    if args.ts_window_seconds > 0:
        for d in detail_turns:
            dt = parse_ts(d.timestamp)
            if dt:
                detail_ts_list.append((dt, d))
        detail_ts_list.sort(key=lambda x: x[0])

    merged: List[Dict[str, Any]] = []

    for t in index_turns:
        if not isinstance(t, dict):
            merged.append(t)
            matched_by_strategy["none"] += 1
            continue

        sid = t.get("session_id")
        tid = t.get("trace_id")
        rid = t.get("request_id")
        spid = t.get("span_id")

        if allowed is not None and sid and tid and (sid, tid) not in allowed:
            merged.append(t)
            blocked_by_min_spans += 1
            matched_by_strategy["none"] += 1
            continue

        found: Optional[DetailTurn] = None
        strategy = "none"

        if sid and tid and (sid, tid) in idx_maps["session+trace"]:
            found = idx_maps["session+trace"][(sid, tid)]
            strategy = "session+trace"
        elif sid and rid and (sid, rid) in idx_maps["session+request"]:
            found = idx_maps["session+request"][(sid, rid)]
            strategy = "session+request"
        elif sid and spid and (sid, spid) in idx_maps["session+span"]:
            found = idx_maps["session+span"][(sid, spid)]
            strategy = "session+span"
        elif args.ts_window_seconds > 0:
            it = parse_ts(t.get("timestamp"))
            if it and detail_ts_list:
                best: Optional[DetailTurn] = None
                best_delta = None
                for dts, d in detail_ts_list:
                    delta = abs((dts - it).total_seconds())
                    if delta <= args.ts_window_seconds and (best_delta is None or delta < best_delta):
                        best = d
                        best_delta = delta
                if best:
                    found = best
                    strategy = "timestamp-fallback"

        if found:
            merged.append(merge_turn(t, found))
            matched_by_strategy[strategy] += 1
        else:
            merged.append(t)
            unmatched += 1
            matched_by_strategy["none"] += 1

    out_obj: Dict[str, Any] = {
        "run_id": index.get("run_id"),
        "window": index.get("window"),
        "turns": index_turns,
        "turns_merged_normalized": merged,
        "merge_stats": {
            "index_turns": len(index_turns),
            "detail_candidates_found": flat_stats["detail_candidates_found"],
            "detail_norm_records": flat_stats["detail_norm_records"],
            "flatten_paths_used": flat_stats["flatten_paths_used"],
            "span_index_size": flat_stats["span_index_size"],
            "trace_steps_aggregated": flat_stats["trace_steps_aggregated"],
            "traces_backfilled_with_steps": flat_stats["traces_backfilled_with_steps"],
            "unmatched_index_turns": unmatched,
            "blocked_by_min_spans": blocked_by_min_spans,
            "matched_by_strategy": matched_by_strategy,
            "ts_fallback_enabled": args.ts_window_seconds > 0,
            "ts_window_seconds": args.ts_window_seconds,
            "min_spans": args.min_spans,
        }
    }

    dump_json(args.out, out_obj)
    print(f"Wrote: {args.out}")

    if args.debug:
        print("DEBUG: merge_stats:", out_obj["merge_stats"])

if __name__ == "__main__":
    main()