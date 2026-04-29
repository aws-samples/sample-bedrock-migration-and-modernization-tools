#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
02_build_session_trace_index.py  (Script 2 - OTEL robust)

Build session -> trace index from Script 1 output (turns),
and hydrate per-trace enrichment from OTEL runtime logs via CloudWatch Logs Insights.

Key improvement:
- We query @message and parse JSON in Python (most reliable).
- We extract content/tool calls from multiple possible field paths.

Enrichment per traceId:
- final_answer: prefer body.message.content[0].text, else last text chunk found
- finish_reason: body.finish_reason (or common variants)
- total_latency_ms: max(@timestamp) - min(@timestamp) using OTEL events
- steps_runtime: ordered stream built from:
    - body.content[].toolUse OR tool_use variants
    - body.message.content[] (sometimes toolUse/text sits here)
    - body.tool_calls[] OR toolCalls/toolCalls variants
    - text chunks from either content list

Usage:
  python3 02_build_session_trace_index.py \
    --turns session_turns.json \
    --otel-log-group "/aws/bedrock-agentcore/runtimes/<runtime>-DEFAULT" \
    --region us-east-1 \
    --out session_enriched_runtime.json
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3


# -----------------------------
# IO helpers
# -----------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def _kv_row(row: List[Dict[str, str]]) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for c in row:
        if isinstance(c, dict) and "field" in c:
            kv[c["field"]] = c.get("value", "")
    return kv


# -----------------------------
# Time parsing
# -----------------------------

def _parse_cw_ts_to_epoch_ms(ts_value: str) -> Optional[int]:
    """
    CW Insights @timestamp usually like:
      "2026-02-15 17:54:06.077"
    Sometimes ISO with Z.
    """
    if not ts_value:
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(ts_value, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            pass

    try:
        dt = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None

def _epoch_ms_to_iso(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# -----------------------------
# Safe JSON access
# -----------------------------

def _safe_get(d: Any, *path: Any) -> Any:
    cur = d
    for p in path:
        if cur is None:
            return None
        if isinstance(p, int):
            if not isinstance(cur, list) or p < 0 or p >= len(cur):
                return None
            cur = cur[p]
        else:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(p)
    return cur

def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []

def _as_dict(x: Any) -> Optional[Dict[str, Any]]:
    return x if isinstance(x, dict) else None

def _parse_message_json(msg: str) -> Optional[Dict[str, Any]]:
    if not msg or not isinstance(msg, str):
        return None
    try:
        return json.loads(msg)
    except Exception:
        return None


# -----------------------------
# Step extraction
# -----------------------------

def _extract_content_lists(event: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """
    OTEL event shapes vary. Return candidate "content lists" that may contain:
      - {"text": "..."}
      - {"toolUse": {...}} (or tool_use variants)
    """
    body = _safe_get(event, "body") or {}
    candidates: List[Any] = []

    # Common locations
    candidates.append(_safe_get(body, "content"))
    candidates.append(_safe_get(body, "message", "content"))
    candidates.append(_safe_get(body, "message", "content", 0, "content"))  # defensive
    candidates.append(_safe_get(event, "message", "content"))              # defensive
    candidates.append(_safe_get(event, "content"))                         # defensive

    out: List[List[Dict[str, Any]]] = []
    for c in candidates:
        if isinstance(c, list) and c and all(isinstance(i, dict) for i in c):
            out.append(c)  # list of dict items
    return out

def _extract_tool_calls(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Tool calls may appear in:
      body.tool_calls, body.toolCalls, body.toolCalls (camel), etc.
    """
    body = _safe_get(event, "body") or {}

    for key in ("tool_calls", "toolCalls", "toolcalls", "toolCallsV2"):
        tc = _safe_get(body, key)
        if isinstance(tc, list) and tc and all(isinstance(i, dict) for i in tc):
            return tc

    # Defensive: sometimes top-level
    for key in ("tool_calls", "toolCalls"):
        tc = _safe_get(event, key)
        if isinstance(tc, list) and tc and all(isinstance(i, dict) for i in tc):
            return tc

    return []

def _extract_finish_reason(event: Dict[str, Any]) -> Optional[str]:
    body = _safe_get(event, "body") or {}
    for key in ("finish_reason", "finishReason", "stop_reason", "stopReason"):
        v = _safe_get(body, key)
        if isinstance(v, str) and v.strip():
            return v
    return None

def _extract_preferred_final_answer(event: Dict[str, Any]) -> Optional[str]:
    """
    Prefer: body.message.content[0].text if present.
    """
    body = _safe_get(event, "body") or {}
    v = _safe_get(body, "message", "content", 0, "text")
    if isinstance(v, str) and v.strip():
        return v
    return None

def _append_steps_from_content_list(steps: List[Dict[str, Any]], ts_iso: str, content: List[Dict[str, Any]]) -> Optional[str]:
    """
    Adds steps and returns last text chunk seen (for fallback final_answer).
    """
    last_text: Optional[str] = None

    for item in content:
        if not isinstance(item, dict):
            continue

        # text chunk
        text = item.get("text")
        if isinstance(text, str) and text:
            steps.append({
                "timestamp": ts_iso,
                "type": "LLM_OUTPUT_CHUNK",
                "text": text,
            })
            last_text = text

        # toolUse variants
        tool_use = item.get("toolUse") or item.get("tool_use") or item.get("tooluse")
        if isinstance(tool_use, dict):
            steps.append({
                "timestamp": ts_iso,
                "type": "INVOCATION",
                "tool_name": tool_use.get("name"),
                "tool_input": tool_use.get("input"),
                "tool_use_id": tool_use.get("toolUseId") or tool_use.get("tool_use_id") or tool_use.get("id"),
            })

    return last_text

def _append_steps_from_tool_calls(steps: List[Dict[str, Any]], ts_iso: str, tool_calls: List[Dict[str, Any]]) -> None:
    for tc in tool_calls:
        fn = tc.get("function")
        if isinstance(fn, dict):
            steps.append({
                "timestamp": ts_iso,
                "type": "INVOCATION",
                "tool_name": fn.get("name"),
                "tool_arguments": fn.get("arguments"),
                "tool_call_id": tc.get("id"),
            })


# -----------------------------
# CloudWatch Logs Insights runner
# -----------------------------

def _run_insights_query(
    logs_client,
    log_group: str,
    start_epoch_s: int,
    end_epoch_s: int,
    query: str,
    poll_seconds: float = 0.8,
    timeout_seconds: int = 120,
    limit: int = 10000,
) -> List[List[Dict[str, str]]]:
    resp = logs_client.start_query(
        logGroupName=log_group,
        startTime=int(start_epoch_s),
        endTime=int(end_epoch_s),
        queryString=query.strip(),
        limit=limit,
    )
    qid = resp["queryId"]

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        r = logs_client.get_query_results(queryId=qid)
        status = r.get("status")
        if status in ("Complete", "Failed", "Cancelled", "Timeout"):
            return r.get("results", []) or []
        time.sleep(poll_seconds)

    return []


def hydrate_from_otel(
    logs_client,
    log_group: str,
    start_epoch_s: int,
    end_epoch_s: int,
    log_stream_filter: str,
    debug_sample_traces: int = 0,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Query OTEL runtime logs and build enrichment for each traceId by parsing @message JSON.
    """
    query = rf"""
fields @timestamp, traceId, spanId, @message
| filter ispresent(traceId)
| filter @logStream like /{log_stream_filter}/
| sort @timestamp asc
| limit 10000
"""

    rows = _run_insights_query(
        logs_client=logs_client,
        log_group=log_group,
        start_epoch_s=start_epoch_s,
        end_epoch_s=end_epoch_s,
        query=query,
        timeout_seconds=180,
        limit=10000,
    )

    by_trace: Dict[str, Dict[str, Any]] = {}
    distinct_traces = set()

    # optional debug: capture a few traceIds and print where tool signals appear
    debug_seen = set()

    for row in rows:
        kv = _kv_row(row)
        trace_id = kv.get("traceId")
        if not trace_id:
            continue
        distinct_traces.add(trace_id)

        ts_raw = kv.get("@timestamp") or kv.get("timestamp")
        ts_ms = _parse_cw_ts_to_epoch_ms(ts_raw) if ts_raw else None
        if ts_ms is None:
            continue
        ts_iso = _epoch_ms_to_iso(ts_ms)

        event = _parse_message_json(kv.get("@message", ""))
        if event is None:
            continue

        tr = by_trace.setdefault(trace_id, {
            "start_ms": ts_ms,
            "end_ms": ts_ms,
            "finish_reason": None,
            "preferred_answer": None,
            "last_text_chunk": None,
            "steps_runtime": [],
        })

        tr["start_ms"] = min(tr["start_ms"], ts_ms)
        tr["end_ms"] = max(tr["end_ms"], ts_ms)

        # finish_reason (latest non-empty)
        fr = _extract_finish_reason(event)
        if fr:
            tr["finish_reason"] = fr

        # preferred final answer path (latest wins)
        pa = _extract_preferred_final_answer(event)
        if pa:
            tr["preferred_answer"] = pa

        # Steps: scan ALL candidate content lists
        content_lists = _extract_content_lists(event)
        for content in content_lists:
            last = _append_steps_from_content_list(tr["steps_runtime"], ts_iso, content)
            if last:
                tr["last_text_chunk"] = last

        # Steps: tool_calls array
        tool_calls = _extract_tool_calls(event)
        if tool_calls:
            _append_steps_from_tool_calls(tr["steps_runtime"], ts_iso, tool_calls)

        # Debug print a couple traces if requested
        if debug_sample_traces > 0 and trace_id not in debug_seen:
            # mark as "debugged" only when we see any tool signal or any content list
            has_any = bool(content_lists) or bool(tool_calls)
            if has_any:
                debug_seen.add(trace_id)
                print("DEBUG trace:", trace_id)
                print("  content_lists:", [len(c) for c in content_lists])
                print("  tool_calls:", len(tool_calls))
                print("  finish_reason:", tr.get("finish_reason"))
                if tr["steps_runtime"]:
                    print("  first_step:", tr["steps_runtime"][0])
                    print("  last_step:", tr["steps_runtime"][-1])
                if len(debug_seen) >= debug_sample_traces:
                    debug_sample_traces = 0  # stop further debug prints

    enrich_by_trace: Dict[str, Dict[str, Any]] = {}

    for tid, tr in by_trace.items():
        final_answer = tr.get("preferred_answer") or tr.get("last_text_chunk")
        has_answer = bool(final_answer and isinstance(final_answer, str) and final_answer.strip())

        enrich_by_trace[tid] = {
            "start_ts": _epoch_ms_to_iso(tr["start_ms"]),
            "end_ts": _epoch_ms_to_iso(tr["end_ms"]),
            "total_latency_ms": int(tr["end_ms"] - tr["start_ms"]),
            "final_answer": final_answer if has_answer else None,
            "finish_reason": tr.get("finish_reason"),
            "steps_runtime": tr.get("steps_runtime") or [],
            "answer_events": 1 if has_answer else 0,
            "has_answer": has_answer,
            "answer_source": "body.message.content[0].text (preferred) else last body.*.content[].text chunk",
        }

    stats = {
        "logs_distinct_traces": len(distinct_traces),
        "events_scanned": len(rows),
        "log_stream_filter": log_stream_filter,
    }
    return enrich_by_trace, stats


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True, help="Script 1 output JSON")
    ap.add_argument("--spans", required=False, help="Raw spans JSONL (optional)")
    ap.add_argument("--otel-log-group", required=True,
                    help="OTEL runtime log group, e.g. /aws/bedrock-agentcore/runtimes/<runtime-name>-DEFAULT")
    ap.add_argument("--log-stream-filter", required=False, default="otel-rt-logs",
                    help="Log stream name substring for OTEL runtime logs (default: otel-rt-logs)")
    ap.add_argument("--region", required=False, default="us-east-1")
    ap.add_argument("--pad-seconds", required=False, type=int, default=7200,
                    help="Pad the Script1 window on both sides (default 7200 = 2h)")
    ap.add_argument("--debug-sample-traces", required=False, type=int, default=0,
                    help="Print debug info for up to N traces where tool/content fields are detected")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tdata = load_json(args.turns)
    turns = tdata.get("turns", [])

    sessions: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(lambda: {
        "turn_refs": [],
        "request_ids": set(),
        "span_ids": set(),
    }))

    indexed_trace_ids: set = set()

    for i, t in enumerate(turns):
        sid = t.get("session_id")
        tid = t.get("trace_id")
        if not sid or not tid:
            continue
        indexed_trace_ids.add(tid)

        sessions[sid][tid]["turn_refs"].append({
            "turn_id": i,
            "timestamp": t.get("timestamp"),
            "request_id": t.get("request_id"),
            "span_id": t.get("span_id"),
            "user_query": t.get("user_query"),
        })
        if t.get("request_id"):
            sessions[sid][tid]["request_ids"].add(t["request_id"])
        if t.get("span_id"):
            sessions[sid][tid]["span_ids"].add(t["span_id"])

    # optional span counts
    span_counts: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(lambda: {
        "total": 0,
        "by_name": Counter(),
        "by_type": Counter(),
    }))

    if args.spans:
        spans = load_jsonl(args.spans)
        for s in spans:
            sid = s.get("session_id")
            tid = s.get("trace_id")
            if not sid or not tid:
                continue
            span_counts[sid][tid]["total"] += 1
            name = s.get("name") or s.get("span_name") or "UNKNOWN"
            span_counts[sid][tid]["by_name"][name] += 1
            stype = s.get("type") or s.get("span_kind") or "UNKNOWN"
            span_counts[sid][tid]["by_type"][stype] += 1

    enrich_stats: Dict[str, Any] = {
        "otel_log_group": args.otel_log_group,
        "indexed_trace_count": len(indexed_trace_ids),
        "logs_distinct_traces": 0,
        "indexed_traces_matched_in_logs": 0,
        "indexed_traces_with_answer": 0,
        "indexed_traces_missing_answer": 0,
        "pad_seconds": args.pad_seconds,
        "region": args.region,
        "events_scanned": 0,
        "log_stream_filter": args.log_stream_filter,
    }

    logs = boto3.client("logs", region_name=args.region)
    window = tdata.get("window") or {}
    start_epoch = int(window.get("start_epoch"))
    end_epoch = int(window.get("end_epoch"))

    enrich_by_trace, log_stats = hydrate_from_otel(
        logs_client=logs,
        log_group=args.otel_log_group,
        start_epoch_s=start_epoch - args.pad_seconds,
        end_epoch_s=end_epoch + args.pad_seconds,
        log_stream_filter=args.log_stream_filter,
        debug_sample_traces=args.debug_sample_traces,
    )

    enrich_stats["logs_distinct_traces"] = log_stats.get("logs_distinct_traces", 0)
    enrich_stats["events_scanned"] = log_stats.get("events_scanned", 0)

    matched = 0
    with_answer = 0
    missing_answer = 0

    for tid in indexed_trace_ids:
        if tid in enrich_by_trace:
            matched += 1
            if enrich_by_trace[tid].get("has_answer"):
                with_answer += 1
            else:
                missing_answer += 1
        else:
            missing_answer += 1

    enrich_stats["indexed_traces_matched_in_logs"] = matched
    enrich_stats["indexed_traces_with_answer"] = with_answer
    enrich_stats["indexed_traces_missing_answer"] = missing_answer

    # Build sessions output
    sessions_out = []
    for sid, traces in sessions.items():
        trace_list = []
        for tid, info in traces.items():
            sc = span_counts[sid].get(tid)
            tr_enrich = enrich_by_trace.get(tid)

            turns_enriched: List[Dict[str, Any]] = []
            for tr in sorted(info["turn_refs"], key=lambda x: x.get("turn_id", 0)):
                enriched_turn = {
                    "session_id": sid,
                    "trace_id": tid,
                    "request_id": tr.get("request_id"),
                    "span_id": tr.get("span_id"),
                    "timestamp": tr.get("timestamp"),
                    "user_query": tr.get("user_query"),
                }
                if tr_enrich:
                    enriched_turn.update({
                        "final_answer": tr_enrich.get("final_answer"),
                        "finish_reason": tr_enrich.get("finish_reason"),
                        "total_latency_ms": tr_enrich.get("total_latency_ms"),
                        "trace_start_ts": tr_enrich.get("start_ts"),
                        "trace_end_ts": tr_enrich.get("end_ts"),
                        "answer_events": tr_enrich.get("answer_events"),
                        "has_answer": tr_enrich.get("has_answer"),
                        "answer_source": tr_enrich.get("answer_source"),
                    })
                turns_enriched.append(enriched_turn)

            trace_obj: Dict[str, Any] = {
                "trace_id": tid,
                "turn_refs": sorted(info["turn_refs"], key=lambda x: x.get("turn_id", 0)),
                "turns_enriched": turns_enriched,
                "request_ids": sorted(list(info["request_ids"])),
                "span_ids": sorted(list(info["span_ids"])),
                "span_counts": {
                    "total": (sc["total"] if sc else None),
                    "by_name": (dict(sc["by_name"]) if sc else {}),
                    "by_type": (dict(sc["by_type"]) if sc else {}),
                },
            }

            if tr_enrich:
                trace_obj["trace_enrichment"] = {
                    "start_ts": tr_enrich.get("start_ts"),
                    "end_ts": tr_enrich.get("end_ts"),
                    "total_latency_ms": tr_enrich.get("total_latency_ms"),
                    "final_answer": tr_enrich.get("final_answer"),
                    "finish_reason": tr_enrich.get("finish_reason"),
                    "answer_events": tr_enrich.get("answer_events"),
                    "has_answer": tr_enrich.get("has_answer"),
                    "answer_source": tr_enrich.get("answer_source"),
                    "steps_runtime": tr_enrich.get("steps_runtime") or [],
                }

            trace_list.append(trace_obj)

        sessions_out.append({
            "session_id": sid,
            "trace_count": len(trace_list),
            "traces": trace_list,
        })

    out = {
        "run_id": tdata.get("run_id"),
        "window": tdata.get("window"),
        "sessions": sessions_out,
        "enrich_stats": enrich_stats,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote: {args.out}")
    print("Enrich stats:", json.dumps(enrich_stats, indent=2))


if __name__ == "__main__":
    main()
