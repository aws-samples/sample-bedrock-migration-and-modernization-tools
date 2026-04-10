"""
Agent Evaluation Dashboard v2 — Production UI

Features:
- Traffic light summary (instant pass/fail)
- Turn timeline with latency bars
- Radar chart for rubric scores
- Run comparison mode
- Progressive drill-down
- Dark mode ready
"""

import json
import io
import math
import tempfile
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime

import streamlit as st
import pandas as pd
import altair as alt
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_jsonl(path: Path) -> list[dict]:
    records = []
    try:
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return records


def find_output_dirs(base: Path) -> list[Path]:
    candidates = []
    for p in sorted(base.rglob("trace_eval.json")):
        candidates.append(p.parent)
    return candidates


def load_run(directory: Path) -> dict:
    """Load all artifacts from an output directory into a single dict."""
    trace_eval = load_json(directory / "trace_eval.json")
    judge_runs = load_jsonl(directory / "judge_runs.jsonl")
    norm_files = list(directory.glob("normalized_run.*.json"))
    normalized = load_json(norm_files[0]) if norm_files else None
    return {
        "dir": directory,
        "trace_eval": trace_eval,
        "judge_runs": judge_runs,
        "normalized": normalized,
        "run_id": (trace_eval or {}).get("run_id", (normalized or {}).get("run_id", directory.name)),
    }


def rubric_score(rr: dict) -> Optional[float]:
    """Extract the primary score from a rubric result."""
    cross = rr.get("cross_judge_result", {})
    return cross.get("weighted_average")


def score_to_status(score: Optional[float], max_score: float = 5.0) -> str:
    if score is None:
        return "unknown"
    ratio = score / max_score
    if ratio >= 0.8:
        return "pass"
    if ratio >= 0.6:
        return "warn"
    return "fail"


STATUS_CONFIG = {
    "pass": {"emoji": "🟢", "color": "#2ecc71", "label": "Good"},
    "warn": {"emoji": "🟡", "color": "#f39c12", "label": "Needs Review"},
    "fail": {"emoji": "🔴", "color": "#e74c3c", "label": "Failing"},
    "unknown": {"emoji": "⚪", "color": "#95a5a6", "label": "No Data"},
}


def generate_pdf_report(run: dict) -> bytes:
    """Generate a PDF-style markdown report as downloadable text."""
    te = run.get("trace_eval") or {}
    dm = te.get("deterministic_metrics", {})
    rubric_results = te.get("rubric_results", [])
    js = te.get("judge_summary", {})

    lines = [
        f"# Agent Evaluation Report",
        f"**Run ID:** {run.get('run_id', 'unknown')}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Deterministic Metrics",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Turns | {dm.get('turn_count', '—')} |",
        f"| Steps | {dm.get('step_count', '—')} |",
        f"| Tool Calls | {dm.get('tool_call_count', '—')} |",
        f"| Tool Success Rate | {dm.get('tool_success_rate', 0):.0%} |",
        f"| Latency p50 | {dm.get('latency_p50', '—')} ms |",
        f"| Latency p95 | {dm.get('latency_p95', '—')} ms |",
        "",
        "## Rubric Scores",
        "| Rubric | Scope | Score | High Risk |",
        "|--------|-------|-------|-----------|",
    ]

    seen = {}
    for rr in rubric_results:
        rid = rr.get("rubric_id", "?")
        cross = rr.get("cross_judge_result", {})
        s = rubric_score(rr)
        v = cross.get("weighted_vote")
        risk = "🚨" if cross.get("high_risk_flag") else ""
        display = f"{s:.1f}/5" if s is not None else (v or "—")
        if rid not in seen:
            seen[rid] = True
            lines.append(f"| {rid} | {rr.get('scope', '—')} | {display} | {risk} |")

    lines.extend([
        "",
        "## Judge Summary",
        f"- Total jobs: {js.get('total_jobs', '—')}",
        f"- Succeeded: {js.get('successful_jobs', '—')}",
        f"- Failed: {js.get('failed_jobs', '—')}",
    ])

    return "\n".join(lines).encode("utf-8")


def run_inline_evaluation(trace_data: dict, app_dir: Path) -> Path | None:
    """Run evaluation on uploaded trace data, return output dir or None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "uploaded_trace.json"
        input_path.write_text(json.dumps(trace_data))
        output_dir = app_dir / "upload_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        mock_config = app_dir.parent / "test-fixtures" / "judges.mock.yaml"
        if not mock_config.exists():
            return None

        cmd = [
            "python3", "-m", "agent_eval.cli",
            "--input", str(input_path),
            "--judge-config", str(mock_config),
            "--output-dir", str(output_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, cwd=str(app_dir.parent))
        return output_dir if result.returncode == 0 else None


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Agent Eval Dashboard", page_icon="🔍", layout="wide")

# Read query params for shareable URLs
query_params = st.query_params
url_run_id = query_params.get("run", None)
url_tab = query_params.get("tab", None)

# Dark-mode-friendly custom CSS + keyboard nav
st.markdown("""
<style>
    /* Traffic light badges */
    .traffic-light { display: inline-flex; align-items: center; gap: 6px;
        padding: 8px 16px; border-radius: 10px; font-weight: 600; font-size: 0.95rem;
        margin-bottom: 4px; }
    .tl-pass { background: rgba(46,204,113,0.15); color: #2ecc71; border: 1px solid rgba(46,204,113,0.25); }
    .tl-warn { background: rgba(243,156,18,0.15); color: #f39c12; border: 1px solid rgba(243,156,18,0.25); }
    .tl-fail { background: rgba(231,76,60,0.15); color: #e74c3c; border: 1px solid rgba(231,76,60,0.25); }
    .tl-unknown { background: rgba(149,165,166,0.15); color: #95a5a6; border: 1px solid rgba(149,165,166,0.25); }
    .score-big { font-size: 1.8rem; font-weight: 700; }
    .drill-hint { font-size: 0.8rem; color: #888; margin-top: 2px; }
    .embed-card { border: 1px solid #333; border-radius: 10px; padding: 16px;
        margin: 8px 0; font-family: monospace; font-size: 0.85rem;
        background: rgba(0,0,0,0.03); }
    /* Section headers */
    .section-header { font-size: 1.1rem; font-weight: 600; margin: 1rem 0 0.5rem 0; }
    /* Metric cards */
    [data-testid="stMetric"] { background: rgba(255,255,255,0.03); border-radius: 8px;
        padding: 12px; border: 1px solid rgba(255,255,255,0.06); }
    /* Sidebar styling */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 { font-size: 1.2rem; }
    /* Expander styling */
    [data-testid="stExpander"] { border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — load runs
# ---------------------------------------------------------------------------

_app_dir = Path(__file__).resolve().parent
_default_sample = str(_app_dir / "sample_output")

st.sidebar.header("📂 Load Evaluation")

# Sidebar info section (360-eval pattern)
with st.sidebar:
    st.markdown("""
    ### Agent Eval Dashboard

    Visualize and analyze agent evaluation results:
    - 📊 Deterministic metrics & rubric scores
    - 🔄 Turn-by-turn trace exploration
    - 📋 Rubric scorecard with drill-down
    - ⚖️ Judge reasoning detail
    - 📈 Cross-run trend analysis
    - 🛠️ Visual rubric builder

    [View on GitHub](https://github.com/aws-samples/sample-bedrock-migration-and-modernization-tools/tree/main/agent-eval)
    """)
    st.divider()

# Drag-and-drop trace upload
st.sidebar.markdown("#### Quick Evaluate")
uploaded_file = st.sidebar.file_uploader(
    "Drop a trace JSON to evaluate",
    type=["json"],
    help="Upload a raw trace or NormalizedRun JSON file to evaluate it instantly with mock judges.",
)

if uploaded_file is not None:
    try:
        trace_data = json.loads(uploaded_file.read())
        with st.sidebar:
            with st.spinner("Evaluating uploaded trace..."):
                result_dir = run_inline_evaluation(trace_data, _app_dir)
        if result_dir:
            st.sidebar.success("✓ Evaluation complete!")
            st.sidebar.caption(f"Results in: {result_dir}")
            # Auto-switch to the upload output
            path_input = st.sidebar.text_input("Output directory path", value=str(result_dir))
        else:
            st.sidebar.error("Evaluation failed. Check trace format.")
            path_input = st.sidebar.text_input("Output directory path", value=_default_sample)
    except json.JSONDecodeError:
        st.sidebar.error("Invalid JSON file.")
        path_input = st.sidebar.text_input("Output directory path", value=_default_sample)
else:
    st.sidebar.markdown("---")
    path_input = st.sidebar.text_input("Output directory path", value=_default_sample)
base_dir = Path(path_input).expanduser().resolve() if path_input else None

if base_dir and base_dir.is_file():
    base_dir = base_dir.parent

# Discover runs
runs: list[dict] = []
if base_dir and base_dir.is_dir():
    sub_dirs = find_output_dirs(base_dir)
    if sub_dirs:
        for d in sub_dirs:
            runs.append(load_run(d))
    elif (base_dir / "trace_eval.json").exists() or list(base_dir.glob("normalized_run.*.json")):
        runs.append(load_run(base_dir))

if not runs:
    st.title("🔍 Agent Evaluation Dashboard")
    st.caption("Offline trace-based evaluation for AI agents")
    st.divider()
    st.info(
        "👈 Enter the path to an evaluation output directory in the sidebar.\n\n"
        "```\npython -m agent_eval.cli --input trace.json "
        "--judge-config judges.yaml --rubrics rubrics.yaml --output-dir ./output\n```"
    )
    with st.expander("ℹ️ Getting Started"):
        st.markdown("""
        **1.** Run an evaluation using the CLI to generate output artifacts
        **2.** Point this dashboard at the output directory
        **3.** Explore metrics, rubric scores, and judge reasoning

        The dashboard auto-discovers all evaluation runs in the directory tree.
        """)
    st.stop()

# Mode selection
compare_mode = len(runs) > 1 and st.sidebar.checkbox("⚖️ Compare runs", value=False)

if compare_mode:
    labels = [r["run_id"] for r in runs]
    sel_a = st.sidebar.selectbox("Run A", range(len(runs)), format_func=lambda i: labels[i])
    sel_b = st.sidebar.selectbox("Run B", range(len(runs)), index=min(1, len(runs)-1), format_func=lambda i: labels[i])
    selected_runs = [runs[sel_a], runs[sel_b]]
else:
    if len(runs) == 1:
        selected_runs = [runs[0]]
    else:
        labels = [r["run_id"] for r in runs]
        # Auto-select from URL param if present
        default_idx = 0
        if url_run_id:
            for i, lbl in enumerate(labels):
                if lbl == url_run_id:
                    default_idx = i
                    break
        sel = st.sidebar.selectbox("Select run", range(len(runs)), index=default_idx, format_func=lambda i: labels[i])
        selected_runs = [runs[sel]]

run = selected_runs[0]

# Update URL with current run for shareable links
st.query_params["run"] = run["run_id"]

# PDF export button
if run.get("trace_eval"):
    st.sidebar.markdown("---")
    pdf_data = generate_pdf_report(run)
    st.sidebar.download_button(
        label="📄 Export Report",
        data=pdf_data,
        file_name=f"eval_report_{run['run_id']}.md",
        mime="text/markdown",
        help="Download evaluation scorecard as a markdown report",
    )

# Embeddable summary card
def generate_embed_card(r: dict) -> str:
    te = r.get("trace_eval") or {}
    dm = te.get("deterministic_metrics", {})
    rr_list = te.get("rubric_results", [])
    scores = [rubric_score(rr) for rr in rr_list if rubric_score(rr) is not None]
    avg = sum(scores) / len(scores) if scores else None
    status = score_to_status(avg)
    cfg = STATUS_CONFIG[status]
    tsr = dm.get("tool_success_rate")
    return (
        f"{cfg['emoji']} Eval: {r['run_id']}\n"
        f"Score: {avg:.1f}/5 | Turns: {dm.get('turn_count', '?')} | "
        f"Tools: {dm.get('tool_call_count', '?')} ({tsr:.0%} success)\n"
        f"p50: {dm.get('latency_p50', '?')}ms | p95: {dm.get('latency_p95', '?')}ms"
    ) if avg is not None else f"⚪ Eval: {r['run_id']} — no scores available"

if run.get("trace_eval"):
    with st.sidebar.expander("📋 Embed Card (Slack/Email)"):
        card_text = generate_embed_card(run)
        st.code(card_text, language=None)
        st.caption("Copy and paste into Slack or email")

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("🔍 Agent Evaluation Dashboard")
st.caption("Offline trace-based evaluation for AI agents — analyze quality without re-running the agent")

# ===================================================================
# COMPARE MODE
# ===================================================================

if compare_mode and len(selected_runs) == 2:
    run_a, run_b = selected_runs

    st.header(f"⚖️ Comparing: `{run_a['run_id']}` vs `{run_b['run_id']}`")

    # --- Metric comparison ---
    def _metrics(r):
        te = r.get("trace_eval") or {}
        return te.get("deterministic_metrics", {})

    ma, mb = _metrics(run_a), _metrics(run_b)

    metric_keys = [
        ("turn_count", "Turns"), ("step_count", "Steps"), ("tool_call_count", "Tool Calls"),
        ("tool_success_rate", "Tool Success"), ("latency_p50", "Latency p50"),
        ("latency_p95", "Latency p95"), ("orphan_result_count", "Orphan Results"),
    ]

    cols = st.columns(len(metric_keys))
    for col, (key, label) in zip(cols, metric_keys):
        va, vb = ma.get(key), mb.get(key)
        if va is not None and vb is not None:
            delta = vb - va
            fmt = lambda v: f"{v:.0%}" if key == "tool_success_rate" else (f"{v:.0f}ms" if "latency" in key else str(v))
            delta_str = f"{delta:+.0f}" if delta != 0 else "same"
            # For latency and orphans, lower is better
            invert = key in ("latency_p50", "latency_p95", "orphan_result_count")
            col.metric(label, fmt(vb), delta_str, delta_color="inverse" if invert else "normal")
        else:
            col.metric(label, "—")

    # --- Rubric comparison ---
    st.markdown("#### ⚖️ Rubric Score Comparison")

    def _rubric_map(r):
        te = r.get("trace_eval") or {}
        out = {}
        for rr in te.get("rubric_results", []):
            rid = rr.get("rubric_id", "?")
            s = rubric_score(rr)
            if rid not in out or (s is not None and (out[rid] is None or s < out[rid])):
                out[rid] = s
        return out

    rmap_a, rmap_b = _rubric_map(run_a), _rubric_map(run_b)
    all_rubrics = sorted(set(list(rmap_a.keys()) + list(rmap_b.keys())))

    if all_rubrics:
        rows = []
        for rid in all_rubrics:
            sa, sb = rmap_a.get(rid), rmap_b.get(rid)
            delta = (sb - sa) if sa is not None and sb is not None else None
            rows.append({
                "Rubric": rid,
                f"Run A ({run_a['run_id']})": f"{sa:.1f}" if sa is not None else "—",
                f"Run B ({run_b['run_id']})": f"{sb:.1f}" if sb is not None else "—",
                "Delta": f"{delta:+.1f}" if delta is not None else "—",
                "Status": "🟢 Better" if delta and delta > 0 else ("🔴 Worse" if delta and delta < 0 else "⚪ Same"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.stop()

# ===================================================================
# SINGLE RUN MODE
# ===================================================================

trace_eval = run["trace_eval"]
normalized = run["normalized"]
judge_runs = run["judge_runs"]

# ---------------------------------------------------------------------------
# Traffic Light Summary (top of page)
# ---------------------------------------------------------------------------

if trace_eval:
    rubric_results = trace_eval.get("rubric_results", [])
    dm = trace_eval.get("deterministic_metrics", {})

    # Compute overall status
    scores = [rubric_score(rr) for rr in rubric_results]
    valid_scores = [s for s in scores if s is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
    overall_status = score_to_status(avg_score)
    cfg = STATUS_CONFIG[overall_status]

    # Traffic light bar
    st.markdown(
        f'<div class="traffic-light tl-{overall_status}">'
        f'{cfg["emoji"]} Overall: {cfg["label"]}'
        f'{"  —  Avg score: " + f"{avg_score:.1f}/5" if avg_score else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.caption(f"Run: `{run['run_id']}`")

    # Per-rubric traffic lights — aggregate by rubric_id, show worst score
    if rubric_results:
        # Group by rubric_id: keep min numeric score (worst case) or first vote
        rubric_agg: dict[str, dict] = {}
        for rr in rubric_results:
            rid = rr.get("rubric_id", "?")
            s = rubric_score(rr)
            cross = rr.get("cross_judge_result", {})
            vote = cross.get("weighted_vote")
            if rid not in rubric_agg:
                rubric_agg[rid] = {"score": s, "vote": vote, "turns": 1}
            else:
                rubric_agg[rid]["turns"] += 1
                prev = rubric_agg[rid]["score"]
                if s is not None and (prev is None or s < prev):
                    rubric_agg[rid]["score"] = s

        unique_rubrics = list(rubric_agg.keys())
        cols = st.columns(min(len(unique_rubrics), 6))
        for i, rid in enumerate(unique_rubrics):
            col = cols[i % len(cols)]
            agg = rubric_agg[rid]
            s = agg["score"]
            turn_count = agg["turns"]
            turn_hint = f" <span style='font-size:0.7em;color:#888'>(worst of {turn_count} turns)</span>" if turn_count > 1 else ""
            if s is not None:
                status = score_to_status(s)
                scfg = STATUS_CONFIG[status]
                col.markdown(f"{scfg['emoji']} **{rid}**<br><span class='score-big'>{s:.1f}</span>/5{turn_hint}", unsafe_allow_html=True)
            elif agg["vote"]:
                vote_status = "pass" if agg["vote"] == "pass" else "fail"
                vcfg = STATUS_CONFIG[vote_status]
                col.markdown(f"{vcfg['emoji']} **{rid}**<br><span class='score-big'>{agg['vote'].upper()}</span>", unsafe_allow_html=True)

    st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_turns, tab_rubrics, tab_judges, tab_trends, tab_builder = st.tabs(
    ["📊 Metrics", "🔄 Turns", "📋 Rubrics", "⚖️ Judges", "📈 Trends", "🛠️ Rubric Builder"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Metrics
# ---------------------------------------------------------------------------

with tab_overview:
    if trace_eval:
        dm = trace_eval.get("deterministic_metrics", {})

        st.markdown("#### 📊 Deterministic Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Turns", dm.get("turn_count", "—"))
        col2.metric("Steps", dm.get("step_count", "—"))
        col3.metric("Tool Calls", dm.get("tool_call_count", "—"))
        tsr = dm.get("tool_success_rate")
        col4.metric("Tool Success Rate", f"{tsr:.0%}" if tsr is not None else "—")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Latency p50", f"{dm['latency_p50']:.0f}ms" if dm.get("latency_p50") else "—")
        col6.metric("Latency p95", f"{dm['latency_p95']:.0f}ms" if dm.get("latency_p95") else "—")
        col7.metric("Orphan Results", dm.get("orphan_result_count", "—"))
        col8.metric("Missing Timestamps", f"{dm.get('missing_timestamp_rate', 0):.0%}")

        flags = []
        if dm.get("stitched_trace_suspect"):
            flags.append("⚠️ Stitched trace suspect")
        if dm.get("single_turn_fallback_used"):
            flags.append("⚠️ Single-turn fallback used")
        if flags:
            st.warning(" | ".join(flags))

        # Radar chart for rubric scores
        rubric_results = trace_eval.get("rubric_results", [])
        numeric_rubrics = [(rr.get("rubric_id"), rubric_score(rr)) for rr in rubric_results if rubric_score(rr) is not None]

        if numeric_rubrics:
            st.divider()
            st.markdown("#### 📋 Rubric Radar")
            radar_df = pd.DataFrame(numeric_rubrics, columns=["rubric", "score"])
            radar_df["color"] = radar_df["score"].apply(
                lambda v: "#2ecc71" if v >= 4 else ("#f39c12" if v >= 3 else "#e74c3c")
            )
            chart = alt.Chart(radar_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("rubric:N", sort=None, title=None),
                y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 5]), title="Score"),
                color=alt.Color("color:N", scale=None),
                tooltip=["rubric", "score"],
            ).properties(height=250)
            st.altair_chart(chart, use_container_width=True)

        # Judge summary
        js = trace_eval.get("judge_summary", {})
        if js:
            with st.expander("Judge Execution Summary"):
                jc1, jc2, jc3, jc4 = st.columns(4)
                jc1.metric("Total Jobs", js.get("total_jobs", "—"))
                jc2.metric("Successful", js.get("successful_jobs", "—"))
                jc3.metric("Failed", js.get("failed_jobs", "—"))
                jc4.metric("Judges Used", js.get("judge_count", "—"))

    if normalized:
        stats = normalized.get("adapter_stats", {})
        if stats:
            with st.expander("Adapter Statistics"):
                ac1, ac2, ac3 = st.columns(3)
                ac1.metric("Events Processed", stats.get("total_events_processed", "—"))
                ac2.metric("Turn Count", stats.get("turn_count", "—"))
                ac3.metric("Mapping Coverage", f"{stats.get('mapping_coverage', 0):.0%}")
                st.caption(f"Segmentation: {stats.get('segmentation_strategy', '—')}")

# ---------------------------------------------------------------------------
# Tab 2 — Turn Explorer with Timeline
# ---------------------------------------------------------------------------

with tab_turns:
    if not normalized:
        st.warning("No normalized run file found.")
    else:
        turns = normalized.get("turns", [])
        if not turns:
            st.info("No turns in this trace.")
        else:
            # Turn timeline overview
            st.markdown("#### ⏱️ Turn Timeline")
            timeline_data = []
            for i, t in enumerate(turns):
                lat = t.get("total_latency_ms") or 0
                conf = t.get("confidence", 0)
                n_steps = len(t.get("steps", []))
                query_preview = (t.get("user_query") or "")[:50]
                timeline_data.append({
                    "Turn": f"T{i+1}",
                    "Latency (ms)": lat,
                    "Confidence": conf,
                    "Steps": n_steps,
                    "Query": query_preview,
                })

            tl_df = pd.DataFrame(timeline_data)

            # Latency bar chart
            tl_df["color"] = tl_df["Latency (ms)"].apply(
                lambda v: "#e74c3c" if v > 5000 else ("#f39c12" if v > 2000 else "#2ecc71")
            )
            lat_chart = alt.Chart(tl_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("Turn:N", sort=None),
                y=alt.Y("Latency (ms):Q", title="Latency (ms)"),
                color=alt.Color("color:N", scale=None),
                tooltip=["Turn", "Latency (ms)", "Steps", "Query"],
            ).properties(height=200)
            st.altair_chart(lat_chart, use_container_width=True)

            # Turn detail drill-down
            st.divider()
            st.markdown("#### 🔎 Turn Detail")
            turn_labels = [f"Turn {i+1}: {(t.get('user_query') or '(no query)')[:60]}" for i, t in enumerate(turns)]

            # Navigation with Prev/Next buttons
            if "turn_idx" not in st.session_state:
                st.session_state.turn_idx = 0
            # Clamp to valid range
            st.session_state.turn_idx = max(0, min(st.session_state.turn_idx, len(turns) - 1))

            nav_col1, nav_col2, nav_col3 = st.columns([1, 6, 1])
            with nav_col1:
                if st.button("◀ Prev", disabled=st.session_state.turn_idx <= 0, use_container_width=True):
                    st.session_state.turn_idx -= 1
                    st.rerun()
            with nav_col3:
                if st.button("Next ▶", disabled=st.session_state.turn_idx >= len(turns) - 1, use_container_width=True):
                    st.session_state.turn_idx += 1
                    st.rerun()
            with nav_col2:
                selected_idx = st.selectbox(
                    "Select turn", range(len(turns)),
                    index=st.session_state.turn_idx,
                    format_func=lambda i: turn_labels[i],
                )
                if selected_idx != st.session_state.turn_idx:
                    st.session_state.turn_idx = selected_idx
                    st.rerun()

            turn = turns[selected_idx]

            # --- Executive Summary Card ---
            turn_id_str = f"turn_{selected_idx}"
            turn_rubric_scores = {}
            if trace_eval:
                for rr in trace_eval.get("rubric_results", []):
                    if rr.get("turn_id") == turn_id_str:
                        rid = rr.get("rubric_id", "?")
                        s = rubric_score(rr)
                        if s is not None:
                            turn_rubric_scores[rid] = s

            steps = turn.get("steps", [])
            tool_calls = [s for s in steps if s.get("kind") == "TOOL_CALL" or s.get("type") == "tool_call"]
            tool_names = [s.get("name", "unknown") for s in tool_calls]
            errors = [s for s in steps if s.get("status") == "error"]
            answer = turn.get("final_answer") or ""
            answer_preview = (answer[:200] + "…") if len(answer) > 200 else answer
            query = turn.get("user_query") or "(no query)"

            # Determine overall turn health
            red_flags = []
            if turn_rubric_scores:
                for rid, s in turn_rubric_scores.items():
                    if s <= 2.0:
                        red_flags.append(f"{rid}: {s:.0f}/5")
                avg = sum(turn_rubric_scores.values()) / len(turn_rubric_scores)
                turn_status = score_to_status(avg)
            else:
                avg = None
                turn_status = "unknown"
            tcfg = STATUS_CONFIG[turn_status]

            with st.container():
                st.markdown(
                    f"### {tcfg['emoji']} Turn {selected_idx + 1} Summary"
                    + (f" — Avg: **{avg:.1f}/5**" if avg else "")
                )

                sc1, sc2, sc3 = st.columns(3)
                sc1.markdown(f"**Asked:** {query[:80]}{'…' if len(query) > 80 else ''}")
                sc2.markdown(f"**Tools:** {', '.join(tool_names) if tool_names else 'None ⚠️'}")
                sc3.markdown(f"**Errors:** {len(errors)}" + (" 🚨" if errors else " ✅"))

                if answer_preview:
                    st.markdown(f"**Answer:** {answer_preview}")

                # Turn-level rubric lights
                if turn_rubric_scores:
                    light_cols = st.columns(min(len(turn_rubric_scores), 6))
                    for i, (rid, s) in enumerate(turn_rubric_scores.items()):
                        col = light_cols[i % len(light_cols)]
                        status = score_to_status(s)
                        scfg = STATUS_CONFIG[status]
                        col.markdown(f"{scfg['emoji']} {rid}<br>**{s:.0f}**/5", unsafe_allow_html=True)

                if red_flags:
                    st.error(f"🚨 **Red flags:** {' · '.join(red_flags)}")

                st.divider()

            # Detailed view (collapsible)
            with st.expander("🔍 Detailed Turn Breakdown", expanded=False):
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("Confidence", f"{turn.get('confidence', 0):.2f}")
                lat = turn.get("total_latency_ms")
                tc2.metric("Latency", f"{lat:.0f}ms" if lat else "—")
                tc3.metric("Steps", len(steps))

                # User query
                st.markdown("**💬 User Query**")
                st.info(turn.get("user_query") or "_No query captured_")

                # Steps
                if steps:
                    st.markdown("**🔧 Steps**")
                    for i, step in enumerate(steps):
                        kind = step.get("kind") or step.get("type") or "STEP"
                        name = step.get("name", "unnamed")
                        status = step.get("status", "unknown")
                        latency = step.get("latency_ms")
                        icon = {"success": "✅", "error": "❌", "unknown": "⚪", "skipped": "⏭️"}.get(status, "⚪")
                        lat_str = f" ({latency:.0f}ms)" if latency else ""

                        st.markdown(f"{icon} **{i+1}. [{kind}] {name}{lat_str}**")
                        st.json({k: v for k, v in step.items() if v is not None and k != "raw"})
                        if step.get("raw"):
                            st.caption("Raw source:")
                            st.json(step["raw"])

                # Final answer
                st.markdown("**🎯 Final Answer**")
                st.success(turn.get("final_answer") or "_No answer captured_")

# ---------------------------------------------------------------------------
# Tab 3 — Rubric Scorecard (progressive drill-down)
# ---------------------------------------------------------------------------

with tab_rubrics:
    if not trace_eval:
        st.warning("No trace_eval.json found.")
    else:
        rubric_results = trace_eval.get("rubric_results", [])
        if not rubric_results:
            st.info("No rubric results.")
        else:
            # Rubric selector for drill-down
            rubric_ids = [rr.get("rubric_id", "?") for rr in rubric_results]
            unique_ids = list(dict.fromkeys(rubric_ids))  # preserve order, dedupe

            selected_rubric = st.selectbox("Select rubric to inspect", ["All Rubrics"] + unique_ids)

            if selected_rubric == "All Rubrics":
                # Summary table
                rows = []
                for rr in rubric_results:
                    rid = rr.get("rubric_id", "?")
                    scope = rr.get("scope", "—")
                    turn_id = rr.get("turn_id", "—")
                    cross = rr.get("cross_judge_result", {})
                    s = rubric_score(rr)
                    vote = cross.get("weighted_vote")
                    dis = cross.get("disagreement_signal", 0)
                    risk = cross.get("high_risk_flag", False)
                    status = score_to_status(s)
                    rows.append({
                        "Status": STATUS_CONFIG[status]["emoji"],
                        "Rubric": rid,
                        "Scope": scope,
                        "Turn": turn_id if turn_id else "—",
                        "Score": f"{s:.1f}" if s is not None else (vote or "—"),
                        "Disagreement": f"{dis:.2f}",
                        "High Risk": "🚨" if risk else "",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption("👆 Select a rubric above to drill into judge details")
            else:
                # Drill-down into specific rubric
                matching = [rr for rr in rubric_results if rr.get("rubric_id") == selected_rubric]
                for rr in matching:
                    turn_id = rr.get("turn_id")
                    cross = rr.get("cross_judge_result", {})
                    s = rubric_score(rr)
                    status = score_to_status(s)
                    cfg = STATUS_CONFIG[status]

                    header = f"{cfg['emoji']} **{selected_rubric}**"
                    if turn_id:
                        header += f" — Turn: `{turn_id}`"
                    st.markdown(header)

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Score", f"{s:.1f}/5" if s is not None else (cross.get("weighted_vote") or "—"))
                    mc2.metric("Disagreement", f"{cross.get('disagreement_signal', 0):.2f}")
                    mc3.metric("High Risk", "Yes 🚨" if cross.get("high_risk_flag") else "No")

                    # Per-judge breakdown
                    within = rr.get("within_judge_results", [])
                    if within:
                        st.markdown("**Per-judge breakdown:**")
                        judge_rows = []
                        for wj in within:
                            judge_rows.append({
                                "Judge": wj.get("judge_id", "?"),
                                "Median": wj.get("median"),
                                "Mean": f"{wj['mean']:.2f}" if wj.get("mean") is not None else "—",
                                "Variance": f"{wj.get('variance', 0):.3f}",
                                "Samples": wj.get("sample_size", 0),
                                "Vote": wj.get("majority_vote") or "—",
                            })
                        st.dataframe(pd.DataFrame(judge_rows), use_container_width=True, hide_index=True)

                    # Show judge reasoning from JSONL
                    relevant_runs = [
                        jr for jr in judge_runs
                        if jr.get("rubric_id") == selected_rubric
                        and (turn_id is None or jr.get("turn_id") == turn_id)
                    ]
                    if relevant_runs:
                        st.markdown("**Judge reasoning:**")
                        for jr in relevant_runs:
                            reasoning = jr.get("reasoning", "")
                            judge_id = jr.get("judge_id", "?")
                            if reasoning:
                                st.markdown(f"*{judge_id}:* {reasoning}")

                    st.divider()

# ---------------------------------------------------------------------------
# Tab 4 — Judge Detail
# ---------------------------------------------------------------------------

with tab_judges:
    if not judge_runs:
        st.warning("No judge_runs.jsonl found.")
    else:
        st.caption(f"{len(judge_runs)} judge run records")

        rubric_ids = sorted(set(r.get("rubric_id", "?") for r in judge_runs))
        judge_ids = sorted(set(r.get("judge_id", "?") for r in judge_runs))

        fc1, fc2 = st.columns(2)
        filter_rubric = fc1.multiselect("Filter by rubric", rubric_ids, default=rubric_ids)
        filter_judge = fc2.multiselect("Filter by judge", judge_ids, default=judge_ids)

        filtered = [
            r for r in judge_runs
            if r.get("rubric_id") in filter_rubric and r.get("judge_id") in filter_judge
        ]

        for record in filtered:
            rid = record.get("rubric_id", "?")
            jid = record.get("judge_id", "?")
            score = record.get("score") or record.get("category") or "—"
            reasoning = record.get("reasoning", "")

            with st.expander(f"[{rid}] Judge: {jid} → Score: {score}"):
                if reasoning:
                    st.markdown(reasoning)
                st.json({k: v for k, v in record.items() if k != "reasoning" and v is not None})

# ---------------------------------------------------------------------------
# Tab 5 — Trends (quality over time across multiple runs)
# ---------------------------------------------------------------------------

with tab_trends:
    st.header("Quality Trends")

    if len(runs) < 2:
        st.info(
            "Trends require multiple evaluation runs in the same directory.\n\n"
            "Point to a parent directory containing multiple output folders to see quality over time."
        )
    else:
        # Build trend data from all discovered runs
        trend_rows = []
        for r in runs:
            te = r.get("trace_eval") or {}
            dm = te.get("deterministic_metrics", {})
            rubric_results = te.get("rubric_results", [])

            # Compute average rubric score for this run
            scores = [rubric_score(rr) for rr in rubric_results if rubric_score(rr) is not None]
            avg = sum(scores) / len(scores) if scores else None

            # Get timestamp from normalized run metadata
            norm = r.get("normalized") or {}
            processed_at = (norm.get("metadata") or {}).get("processed_at", "")

            trend_rows.append({
                "Run": r["run_id"],
                "Avg Score": round(avg, 2) if avg is not None else None,
                "Turns": dm.get("turn_count", 0),
                "Tool Success": dm.get("tool_success_rate"),
                "Latency p50 (ms)": dm.get("latency_p50"),
                "Latency p95 (ms)": dm.get("latency_p95"),
                "Processed": processed_at[:19] if processed_at else "—",
            })

        trend_df = pd.DataFrame(trend_rows)

        # Score trend chart
        score_df = trend_df[trend_df["Avg Score"].notna()].copy()
        if not score_df.empty:
            st.markdown("#### 📊 Average Rubric Score")
            score_df["color"] = score_df["Avg Score"].apply(
                lambda v: "#2ecc71" if v >= 4 else ("#f39c12" if v >= 3 else "#e74c3c")
            )
            score_chart = alt.Chart(score_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("Run:N", sort=None),
                y=alt.Y("Avg Score:Q", scale=alt.Scale(domain=[0, 5])),
                color=alt.Color("color:N", scale=None),
                tooltip=["Run", "Avg Score", "Turns"],
            ).properties(height=250)
            st.altair_chart(score_chart, use_container_width=True)

        # Latency trend
        lat_df = trend_df[trend_df["Latency p50 (ms)"].notna()].copy()
        if not lat_df.empty:
            st.markdown("#### ⏱️ Latency Trend")
            lat_melted = lat_df.melt(
                id_vars=["Run"],
                value_vars=["Latency p50 (ms)", "Latency p95 (ms)"],
                var_name="Metric",
                value_name="ms",
            )
            lat_chart = alt.Chart(lat_melted).mark_line(point=True).encode(
                x=alt.X("Run:N", sort=None),
                y=alt.Y("ms:Q", title="Latency (ms)"),
                color="Metric:N",
                tooltip=["Run", "Metric", "ms"],
            ).properties(height=250)
            st.altair_chart(lat_chart, use_container_width=True)

        # Per-rubric trend across runs
        st.divider()
        st.markdown("#### 🗺️ Per-Rubric Scores Across Runs")
        rubric_trend_rows = []
        for r in runs:
            te = r.get("trace_eval") or {}
            seen_rubrics = {}
            for rr in te.get("rubric_results", []):
                rid = rr.get("rubric_id", "?")
                s = rubric_score(rr)
                if s is not None and rid not in seen_rubrics:
                    seen_rubrics[rid] = s
            for rid, s in seen_rubrics.items():
                rubric_trend_rows.append({"Run": r["run_id"], "Rubric": rid, "Score": s})

        if rubric_trend_rows:
            rt_df = pd.DataFrame(rubric_trend_rows)
            rubric_heatmap = alt.Chart(rt_df).mark_rect().encode(
                x=alt.X("Run:N", sort=None),
                y=alt.Y("Rubric:N", sort=None),
                color=alt.Color("Score:Q", scale=alt.Scale(domain=[1, 5], scheme="redyellowgreen")),
                tooltip=["Run", "Rubric", "Score"],
            ).properties(height=max(200, len(rt_df["Rubric"].unique()) * 30))
            st.altair_chart(rubric_heatmap, use_container_width=True)

        # Summary table
        st.divider()
        st.markdown("#### 📋 Run Summary Table")
        st.dataframe(trend_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 6 — Rubric Builder
# ---------------------------------------------------------------------------

with tab_builder:

    st.markdown("#### 🛠️ Rubric Builder")
    st.caption("Create custom rubrics visually, preview the YAML, and export a ready-to-use rubrics file.")

    # Session state for rubric list
    if "custom_rubrics" not in st.session_state:
        st.session_state.custom_rubrics = []

    # --- Form ---
    with st.form("rubric_form", clear_on_submit=True):
        st.markdown("**Define a new rubric**")

        fc1, fc2 = st.columns(2)
        rubric_id = fc1.text_input("Rubric ID", placeholder="e.g. MY_CUSTOM_CHECK", help="Uppercase with underscores")
        severity = fc2.selectbox("Severity", ["critical", "high", "medium"])

        description = st.text_input("Description", placeholder="What does this rubric measure?")

        fc3, fc4, fc5 = st.columns(3)
        weight = fc3.number_input("Weight", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
        scope = fc4.selectbox("Scope", ["turn", "run"], help="turn = evaluated per turn, run = evaluated once for the whole session")
        scoring_type = fc5.selectbox("Scoring Type", ["numeric", "categorical"])

        evaluation_instructions = st.text_area(
            "Evaluation Instructions",
            height=180,
            placeholder="Score 1-5 based on ...:\n5 = Excellent — ...\n4 = Good — ...\n3 = Adequate — ...\n2 = Poor — ...\n1 = Failing — ...",
            help="Detailed scoring guide the LLM judge will follow",
        )

        st.markdown("**Evidence Selectors** — what context does the judge see?")
        ev_col1, ev_col2 = st.columns(2)
        ev_query = ev_col1.checkbox("User query", value=True)
        ev_tool_calls = ev_col1.checkbox("Tool calls", value=False)
        ev_tool_results = ev_col2.checkbox("Tool results", value=True)
        ev_final_answer = ev_col2.checkbox("Final answer", value=True)

        submitted = st.form_submit_button("➕ Add Rubric", use_container_width=True)

    if submitted:
        errors = []
        if not rubric_id.strip():
            errors.append("Rubric ID is required")
        if not description.strip():
            errors.append("Description is required")
        if not evaluation_instructions.strip():
            errors.append("Evaluation instructions are required")
        if any(r["rubric_id"] == rubric_id.strip().upper() for r in st.session_state.custom_rubrics):
            errors.append(f"Rubric ID '{rubric_id.strip().upper()}' already exists")

        if errors:
            for e in errors:
                st.error(e)
        else:
            evidence = []
            if ev_query:
                evidence.append("$.user_query")
            if ev_tool_calls:
                evidence.append("$.steps[?(@.kind=='TOOL_CALL')]")
            if ev_tool_results:
                evidence.append("$.steps[?(@.kind=='TOOL_RESULT')]")
            if ev_final_answer:
                evidence.append("$.final_answer")

            rubric = {
                "rubric_id": rubric_id.strip().upper(),
                "description": description.strip(),
                "weight": float(weight),
                "severity": severity,
                "enabled": True,
                "scoring_scale": {"type": scoring_type, "min": 1, "max": 5},
                "aggregation_type": "median",
                "run_aggregation_policy": "standard",
                "requires_llm_judge": True,
                "evaluation_instructions": evaluation_instructions.strip(),
                "evidence_selectors": evidence,
                "scope": scope,
                "scope_behavior": "per_turn" if scope == "turn" else "aggregate_all_turns",
                "evidence_budget": 10000,
            }
            st.session_state.custom_rubrics.append(rubric)
            st.success(f"Added **{rubric['rubric_id']}**")
            st.rerun()

    # --- Current rubrics list ---
    if st.session_state.custom_rubrics:
        st.markdown("---")
        st.markdown(f"#### 📦 Your Rubrics ({len(st.session_state.custom_rubrics)})")

        for i, r in enumerate(st.session_state.custom_rubrics):
            sev_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡"}[r["severity"]]
            with st.expander(f"{sev_icon} **{r['rubric_id']}** — {r['description']}", expanded=False):
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Weight", f"{r['weight']:.1f}")
                mc2.metric("Severity", r["severity"])
                mc3.metric("Scope", r["scope"])
                mc4.metric("Evidence", f"{len(r['evidence_selectors'])} selectors")
                st.markdown("**Instructions:**")
                st.text(r["evaluation_instructions"])
                if st.button(f"🗑️ Remove", key=f"rm_{i}"):
                    st.session_state.custom_rubrics.pop(i)
                    st.rerun()

        # --- YAML Preview & Export ---
        st.markdown("---")
        st.markdown("#### 📄 YAML Preview")

        yaml_doc = {
            "version": "1.0.0",
            "default_evidence_budget": 10000,
            "rubrics": st.session_state.custom_rubrics,
        }
        yaml_str = yaml.dump(yaml_doc, default_flow_style=False, sort_keys=False, allow_unicode=True)
        st.code(yaml_str, language="yaml")

        dl_col1, dl_col2 = st.columns(2)
        dl_col1.download_button(
            "⬇️ Download rubrics.yaml",
            data=yaml_str,
            file_name="rubrics.yaml",
            mime="text/yaml",
            use_container_width=True,
        )
        if dl_col2.button("🗑️ Clear All", use_container_width=True):
            st.session_state.custom_rubrics = []
            st.rerun()
    else:
        st.info("No rubrics yet. Use the form above to create your first rubric, or import an existing file below.")

    # --- Import existing rubrics ---
    st.markdown("---")
    st.divider()
    st.markdown("#### 📂 Import Existing Rubrics")
    uploaded = st.file_uploader("Upload a rubrics YAML file", type=["yaml", "yml"], key="rubric_upload")
    if uploaded:
        try:
            parsed = yaml.safe_load(uploaded.read())
            rubrics_list = parsed.get("rubrics", []) if isinstance(parsed, dict) else []
            if not rubrics_list:
                st.error("No rubrics found in the uploaded file. Expected a 'rubrics' key with a list.")
            else:
                st.success(f"Found **{len(rubrics_list)}** rubrics in file")
                # Preview what will be imported
                preview_ids = [r.get("rubric_id", "?") for r in rubrics_list]
                existing_ids = {r["rubric_id"] for r in st.session_state.custom_rubrics}
                new_ids = [rid for rid in preview_ids if rid not in existing_ids]
                dupe_ids = [rid for rid in preview_ids if rid in existing_ids]

                if dupe_ids:
                    st.warning(f"Duplicates (will be skipped): {', '.join(dupe_ids)}")
                if new_ids:
                    st.info(f"Will import: {', '.join(new_ids)}")

                imp_col1, imp_col2 = st.columns(2)
                if imp_col1.button("✅ Import new rubrics", use_container_width=True):
                    added = 0
                    for r in rubrics_list:
                        if r.get("rubric_id") not in existing_ids:
                            st.session_state.custom_rubrics.append(r)
                            added += 1
                    st.success(f"Imported {added} rubrics")
                    st.rerun()
                if imp_col2.button("🔄 Replace all with imported", use_container_width=True):
                    st.session_state.custom_rubrics = rubrics_list
                    st.success(f"Replaced with {len(rubrics_list)} rubrics")
                    st.rerun()
        except yaml.YAMLError as e:
            st.error(f"Invalid YAML: {e}")
