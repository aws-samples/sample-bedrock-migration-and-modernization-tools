# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Agent Evaluation Dashboard — 360-Eval Style Layout

Sidebar radio navigation with workflow-ordered pages:
- Setup: Rubric Builder, Judge Config, Trace Upload
- Run: Config summary, trigger evaluation, monitor progress
- Results: Metrics, Turns, Rubric Scorecard, Judge Detail
- Trends: Cross-run quality analysis
"""

import json
import io
import math
import sys
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime


from dataclasses import dataclass


@dataclass
class _CliResult:
    """Lightweight result object for CLI invocations."""
    returncode: int
    stdout: str = ""
    stderr: str = ""


def _run_eval_cli(args: list[str], cwd: str, timeout: int = 120) -> _CliResult:
    """Run agent_eval CLI by directly invoking its main function."""
    import os
    saved_cwd = os.getcwd()
    try:
        os.chdir(cwd)
        from agent_eval.cli import main as eval_main
        rc = eval_main(args) or 0
        return _CliResult(returncode=rc)
    except SystemExit as e:
        rc = e.code if e.code else 0
        return _CliResult(returncode=rc)
    except Exception as e:
        return _CliResult(returncode=1, stderr=str(e))
    finally:
        os.chdir(saved_cwd)

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
    trace_eval = load_json(directory / "trace_eval.json")
    judge_runs = load_jsonl(directory / "judge_runs.jsonl")
    results = load_json(directory / "results.json")
    norm_files = list(directory.glob("normalized_run.*.json"))
    normalized = load_json(norm_files[0]) if norm_files else None
    return {
        "dir": directory,
        "trace_eval": trace_eval,
        "judge_runs": judge_runs,
        "results": results,
        "normalized": normalized,
        "run_id": (trace_eval or {}).get("run_id", (normalized or {}).get("run_id", directory.name)),
    }


def rubric_score(rr: dict) -> Optional[float]:
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
    te = run.get("trace_eval") or {}
    dm = te.get("deterministic_metrics", {})
    rubric_results = te.get("rubric_results", [])
    scores = [rubric_score(rr) for rr in rubric_results if rubric_score(rr) is not None]
    avg = sum(scores) / len(scores) if scores else None
    lines = [
        f"# Evaluation Report: {run['run_id']}",
        f"Generated: {datetime.now().isoformat()[:19]}",
        "",
        "## Deterministic Metrics",
        f"- Turns: {dm.get('turn_count', '—')}",
        f"- Steps: {dm.get('step_count', '—')}",
        f"- Tool Calls: {dm.get('tool_call_count', '—')}",
        f"- Tool Success Rate: {dm.get('tool_success_rate', 0):.0%}",
        f"- Latency p50: {dm.get('latency_p50', '—')}ms",
        f"- Latency p95: {dm.get('latency_p95', '—')}ms",
        "",
        "## Rubric Scores",
    ]
    for rr in rubric_results:
        rid = rr.get("rubric_id", "?")
        s = rubric_score(rr)
        status = score_to_status(s)
        emoji = STATUS_CONFIG[status]["emoji"]
        lines.append(f"- {emoji} {rid}: {s:.1f}/5" if s is not None else f"- ⚪ {rid}: N/A")
    if avg is not None:
        lines.extend(["", f"**Overall Average: {avg:.1f}/5**"])
    return "\n".join(lines).encode("utf-8")


def run_inline_evaluation(trace_data: dict, app_dir: Path) -> Optional[Path]:
    try:
        tmp = Path(tempfile.mkdtemp(prefix="eval_upload_"))
        trace_path = tmp / "uploaded_trace.json"
        trace_path.write_text(json.dumps(trace_data, indent=2))
        output_dir = tmp / "output"
        mock_judges = app_dir.parent / "test-fixtures" / "judges.mock.yaml"
        default_rubrics = app_dir.parent / "test-fixtures" / "rubrics.default.yaml"
        cli_args = [
            "--input", str(trace_path),
            "--judge-config", str(mock_judges),
            "--output-dir", str(output_dir),
        ]
        if default_rubrics.exists():
            cli_args.extend(["--rubrics", str(default_rubrics)])
        result = _run_eval_cli(cli_args, cwd=str(app_dir.parent), timeout=120)
        if result.returncode == 0 and output_dir.exists():
            return output_dir
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Agent Eval Dashboard", page_icon="🔍", layout="wide")

query_params = st.query_params
url_run_id = query_params.get("run", None)

st.markdown("""
<style>
    .traffic-light { display: inline-flex; align-items: center; gap: 6px;
        padding: 8px 16px; border-radius: 10px; font-weight: 600; font-size: 0.95rem;
        margin-bottom: 4px; }
    .tl-pass { background: rgba(46,204,113,0.15); color: #2ecc71; border: 1px solid rgba(46,204,113,0.25); }
    .tl-warn { background: rgba(243,156,18,0.15); color: #f39c12; border: 1px solid rgba(243,156,18,0.25); }
    .tl-fail { background: rgba(231,76,60,0.15); color: #e74c3c; border: 1px solid rgba(231,76,60,0.25); }
    .tl-unknown { background: rgba(149,165,166,0.15); color: #95a5a6; border: 1px solid rgba(149,165,166,0.25); }
    .score-big { font-size: 1.8rem; font-weight: 700; }
    [data-testid="stMetric"] { background: rgba(255,255,255,0.03); border-radius: 8px;
        padding: 12px; border: 1px solid rgba(255,255,255,0.06); }
    [data-testid="stExpander"] { border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — Navigation & Run Loading
# ---------------------------------------------------------------------------

_app_dir = Path(__file__).resolve().parent
_default_sample = str(_app_dir / "sample_output")

with st.sidebar:
    st.markdown("### 🔍 Agent Eval Dashboard")
    st.markdown("""
    Offline trace-based evaluation for AI agents.
    - Configure rubrics & judges
    - Upload and evaluate traces
    - Analyze results & trends

    [View on GitHub](https://github.com/aws-samples/sample-bedrock-migration-and-modernization-tools/tree/main/agent-eval)
    """)
    st.divider()

    active_page = st.radio("Navigation", ["Setup", "Run", "Results", "Trends"], key="nav_radio")
    st.divider()

# Context-sensitive sidebar: run loading only for Results/Trends
runs: list[dict] = []
run = None
trace_eval = None
normalized = None
judge_runs = []
compare_mode = False
selected_runs = []

if active_page in ("Results", "Trends"):
    with st.sidebar:
        st.markdown("#### 📂 Load Evaluation")

        # Drag-and-drop trace upload
        uploaded_file = st.file_uploader(
            "Drop a trace JSON to evaluate",
            type=["json"],
            help="Upload a raw trace or NormalizedRun JSON to evaluate instantly with mock judges.",
        )

        if uploaded_file is not None:
            try:
                trace_data = json.loads(uploaded_file.read())
                with st.spinner("Evaluating uploaded trace..."):
                    result_dir = run_inline_evaluation(trace_data, _app_dir)
                if result_dir:
                    st.success("✓ Evaluation complete!")
                    st.caption(f"Results in: {result_dir}")
                    path_input = st.text_input("Output directory path", value=str(result_dir))
                else:
                    st.error("Evaluation failed. Check trace format.")
                    path_input = st.text_input("Output directory path", value=_default_sample)
            except json.JSONDecodeError:
                st.error("Invalid JSON file.")
                path_input = st.text_input("Output directory path", value=_default_sample)
        else:
            path_input = st.text_input("Output directory path", value=_default_sample)

        base_dir = Path(path_input).expanduser().resolve() if path_input else None
        if base_dir and base_dir.is_file():
            base_dir = base_dir.parent

        if base_dir and base_dir.is_dir():
            sub_dirs = find_output_dirs(base_dir)
            if sub_dirs:
                for d in sub_dirs:
                    runs.append(load_run(d))
            elif (base_dir / "trace_eval.json").exists() or list(base_dir.glob("normalized_run.*.json")):
                runs.append(load_run(base_dir))

        if runs:
            compare_mode = len(runs) > 1 and st.checkbox("⚖️ Compare runs", value=False)

            if compare_mode:
                labels = [r["run_id"] for r in runs]
                sel_a = st.selectbox("Run A", range(len(runs)), format_func=lambda i: labels[i])
                sel_b = st.selectbox("Run B", range(len(runs)), index=min(1, len(runs)-1), format_func=lambda i: labels[i])
                selected_runs = [runs[sel_a], runs[sel_b]]
            else:
                if len(runs) == 1:
                    selected_runs = [runs[0]]
                else:
                    labels = [r["run_id"] for r in runs]
                    default_idx = 0
                    if url_run_id:
                        for i, lbl in enumerate(labels):
                            if lbl == url_run_id:
                                default_idx = i
                                break
                    sel = st.selectbox("Select run", range(len(runs)), index=default_idx, format_func=lambda i: labels[i])
                    selected_runs = [runs[sel]]

            run = selected_runs[0]
            st.query_params["run"] = run["run_id"]
            trace_eval = run.get("trace_eval")
            normalized = run.get("normalized")
            judge_runs = run.get("judge_runs", [])

            # PDF export
            if trace_eval:
                st.divider()
                pdf_data = generate_pdf_report(run)
                st.download_button(
                    label="📄 Export Report",
                    data=pdf_data,
                    file_name=f"eval_report_{run['run_id']}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            # Embed card
            if trace_eval:
                with st.expander("📋 Embed Card (Slack/Email)"):
                    te_card = trace_eval
                    dm_card = te_card.get("deterministic_metrics", {})
                    rr_card = te_card.get("rubric_results", [])
                    scores_card = [rubric_score(rr) for rr in rr_card if rubric_score(rr) is not None]
                    avg_card = sum(scores_card) / len(scores_card) if scores_card else None
                    if avg_card is not None:
                        status_card = score_to_status(avg_card)
                        cfg_card = STATUS_CONFIG[status_card]
                        tsr_card = dm_card.get("tool_success_rate", 0)
                        card_text = (
                            f"{cfg_card['emoji']} Eval: {run['run_id']}\n"
                            f"Score: {avg_card:.1f}/5 | Turns: {dm_card.get('turn_count', '?')} | "
                            f"Tools: {dm_card.get('tool_call_count', '?')} ({tsr_card:.0%} success)\n"
                            f"p50: {dm_card.get('latency_p50', '?')}ms | p95: {dm_card.get('latency_p95', '?')}ms"
                        )
                    else:
                        card_text = f"⚪ Eval: {run['run_id']} — no scores available"
                    st.code(card_text, language=None)
                    st.caption("Copy and paste into Slack or email")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔍 Agent Evaluation Dashboard")
st.caption("Offline trace-based evaluation for AI agents — analyze quality without re-running the agent")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: SETUP
# ═══════════════════════════════════════════════════════════════════════════

JUDGE_PRESETS = {
    "Claude Sonnet (Bedrock)": {"provider": "bedrock", "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0", "temperature": 0.0, "max_tokens": 2048, "timeout_seconds": 60},
    "Claude Opus (Bedrock)": {"provider": "bedrock", "model_id": "anthropic.claude-opus-4-20250514-v1:0", "temperature": 0.0, "max_tokens": 2048, "timeout_seconds": 60},
    "Nova Pro (Bedrock)": {"provider": "bedrock", "model_id": "amazon.nova-pro-v1:0", "temperature": 0.0, "max_tokens": 2048, "timeout_seconds": 60},
    "Mock Judge": {"provider": "mock", "model_id": "mock-model-v1", "temperature": 0.0, "max_tokens": 1024, "timeout_seconds": 30},
}

if active_page == "Setup":
    rubric_tab, judge_tab, trace_tab = st.tabs(["🛠️ Rubric Configuration", "⚖️ Judge Configuration", "📎 Trace Upload"])

    # ── Rubric Configuration ──
    with rubric_tab:
        st.subheader("Rubric Configuration")
        st.caption("Create custom rubrics visually, preview the YAML, and export a ready-to-use rubrics file.")

        if "custom_rubrics" not in st.session_state:
            st.session_state.custom_rubrics = []

        build_col, preview_col = st.columns([3, 2])

        with build_col:
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
                    "Evaluation Instructions", height=140,
                    placeholder="Score 1-5 based on ...:\n5 = Excellent — ...\n4 = Good — ...\n3 = Adequate — ...\n2 = Poor — ...\n1 = Failing — ...",
                    help="Detailed scoring guide the LLM judge will follow",
                )
                st.markdown("**Evidence Selectors** — what context does the judge see?")
                ev_col1, ev_col2 = st.columns(2)
                ev_query = ev_col1.checkbox("User query", value=True)
                ev_tool_calls = ev_col1.checkbox("Tool calls", value=False)
                ev_tool_results = ev_col2.checkbox("Tool results", value=True)
                ev_final_answer = ev_col2.checkbox("Final answer", value=True)
                submitted_rubric = st.form_submit_button("➕ Add Rubric", use_container_width=True)

            if submitted_rubric:
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
                    if ev_query: evidence.append("$.user_query")
                    if ev_tool_calls: evidence.append("$.steps[?(@.kind=='TOOL_CALL')]")
                    if ev_tool_results: evidence.append("$.steps[?(@.kind=='TOOL_RESULT')]")
                    if ev_final_answer: evidence.append("$.final_answer")
                    rubric = {
                        "rubric_id": rubric_id.strip().upper(), "description": description.strip(),
                        "weight": float(weight), "severity": severity, "enabled": True,
                        "scoring_scale": {"type": scoring_type, "min": 1, "max": 5},
                        "aggregation_type": "median", "run_aggregation_policy": "standard",
                        "requires_llm_judge": True, "evaluation_instructions": evaluation_instructions.strip(),
                        "evidence_selectors": evidence, "scope": scope,
                        "scope_behavior": "per_turn" if scope == "turn" else "aggregate_all_turns",
                        "evidence_budget": 10000,
                    }
                    st.session_state.custom_rubrics.append(rubric)
                    st.success(f"Added **{rubric['rubric_id']}**")
                    st.rerun()

            st.divider()
            st.markdown("**📂 Import Existing Rubrics**")
            uploaded_rubrics = st.file_uploader("Upload a rubrics YAML file", type=["yaml", "yml"], key="rubric_upload")
            if uploaded_rubrics:
                try:
                    parsed = yaml.safe_load(uploaded_rubrics.read())
                    rubrics_list = parsed.get("rubrics", []) if isinstance(parsed, dict) else []
                    if not rubrics_list:
                        st.error("No rubrics found. Expected a 'rubrics' key with a list.")
                    else:
                        st.success(f"Found **{len(rubrics_list)}** rubrics in file")
                        existing_ids = {r["rubric_id"] for r in st.session_state.custom_rubrics}
                        preview_ids = [r.get("rubric_id", "?") for r in rubrics_list]
                        new_ids = [rid for rid in preview_ids if rid not in existing_ids]
                        dupe_ids = [rid for rid in preview_ids if rid in existing_ids]
                        if dupe_ids: st.warning(f"Duplicates (will be skipped): {', '.join(dupe_ids)}")
                        if new_ids: st.info(f"Will import: {', '.join(new_ids)}")
                        imp_c1, imp_c2 = st.columns(2)
                        if imp_c1.button("✅ Import new rubrics", use_container_width=True):
                            added = sum(1 for r in rubrics_list if r.get("rubric_id") not in existing_ids and (st.session_state.custom_rubrics.append(r) or True))
                            st.success(f"Imported {added} rubrics"); st.rerun()
                        if imp_c2.button("🔄 Replace all with imported", use_container_width=True):
                            st.session_state.custom_rubrics = rubrics_list
                            st.success(f"Replaced with {len(rubrics_list)} rubrics"); st.rerun()
                except yaml.YAMLError as e:
                    st.error(f"Invalid YAML: {e}")

        with preview_col:
            if st.session_state.custom_rubrics:
                st.markdown(f"**📦 Your Rubrics ({len(st.session_state.custom_rubrics)})**")
                for i, r in enumerate(st.session_state.custom_rubrics):
                    sev_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡"}[r["severity"]]
                    with st.expander(f"{sev_icon} **{r['rubric_id']}** — {r['description']}", expanded=False):
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Weight", f"{r['weight']:.1f}")
                        mc2.metric("Severity", r["severity"])
                        mc3.metric("Scope", r.get("scope", "turn"))
                        mc4.metric("Evidence", f"{len(r.get('evidence_selectors', []))} selectors")
                        st.markdown("**Instructions:**")
                        st.text(r["evaluation_instructions"])
                        if st.button(f"🗑️ Remove", key=f"rm_r_{i}"):
                            st.session_state.custom_rubrics.pop(i); st.rerun()

                st.divider()
                st.markdown("**📄 YAML Preview**")
                yaml_doc = {"version": "1.0.0", "default_evidence_budget": 10000, "rubrics": st.session_state.custom_rubrics}
                yaml_str = yaml.dump(yaml_doc, default_flow_style=False, sort_keys=False, allow_unicode=True)
                st.code(yaml_str, language="yaml")
                dl_c1, dl_c2 = st.columns(2)
                dl_c1.download_button("⬇️ Download rubrics.yaml", data=yaml_str, file_name="rubrics.yaml", mime="text/yaml", use_container_width=True)
                if dl_c2.button("🗑️ Clear All", key="clear_rubrics", use_container_width=True):
                    st.session_state.custom_rubrics = []; st.rerun()
            else:
                st.info("No rubrics yet. Use the form to create your first rubric, or import an existing file.")

    # ── Judge Configuration ──
    with judge_tab:
        st.subheader("Judge Configuration")
        st.caption("Configure LLM judges visually, preview the YAML, and export a ready-to-use judges file.")

        if "custom_judges" not in st.session_state:
            st.session_state.custom_judges = []

        build_col_j, preview_col_j = st.columns([3, 2])

        with build_col_j:
            preset_choice = st.selectbox("Quick-fill from preset", ["(none)"] + list(JUDGE_PRESETS.keys()))
            preset = JUDGE_PRESETS.get(preset_choice, {})

            with st.form("judge_form", clear_on_submit=True):
                st.markdown("**Define a new judge**")
                jc1, jc2 = st.columns(2)
                judge_id = jc1.text_input("Judge ID", placeholder="e.g. claude_sonnet_4")
                provider = jc2.selectbox("Provider", ["bedrock", "litellm", "mock"], index=["bedrock", "litellm", "mock"].index(preset.get("provider", "bedrock")))
                model_id = st.text_input("Model ID", value=preset.get("model_id", ""), placeholder="e.g. us.anthropic.claude-sonnet-4-20250514-v1:0")
                st.markdown("**Parameters**")
                pc1, pc2 = st.columns(2)
                temperature = pc1.slider("Temperature", 0.0, 1.0, preset.get("temperature", 0.0), 0.1)
                max_tokens = pc2.number_input("Max Tokens", min_value=256, max_value=8192, value=preset.get("max_tokens", 2048), step=256)
                st.markdown("**Execution Settings**")
                ec1, ec2, ec3 = st.columns(3)
                repeats = ec1.number_input("Repeats", min_value=1, max_value=10, value=3)
                timeout_seconds = ec2.number_input("Timeout (s)", min_value=1, max_value=300, value=preset.get("timeout_seconds", 30))
                region_name = ec3.text_input("AWS Region (optional)", placeholder="e.g. us-east-1")
                st.markdown("**Advanced (optional)**")
                ac1, ac2 = st.columns(2)
                concurrency = ac1.number_input("Concurrency", min_value=0, max_value=50, value=0, help="0 = use default")
                rate_limit = ac2.number_input("Rate Limit (req/s)", min_value=0, max_value=100, value=0, help="0 = no limit")
                bc1, bc2 = st.columns(2)
                streaming = bc1.checkbox("Streaming", value=False)
                use_converse_api = bc2.checkbox("Use Converse API", value=True)
                submitted_judge = st.form_submit_button("➕ Add Judge", use_container_width=True)

            if submitted_judge:
                errors = []
                if not judge_id.strip(): errors.append("Judge ID is required")
                if not model_id.strip(): errors.append("Model ID is required")
                if len(st.session_state.custom_judges) >= 5: errors.append("Maximum 5 judges allowed")
                if any(j["judge_id"] == judge_id.strip() for j in st.session_state.custom_judges):
                    errors.append(f"Judge ID '{judge_id.strip()}' already exists")
                if errors:
                    for e in errors: st.error(e)
                else:
                    judge = {
                        "judge_id": judge_id.strip(), "provider": provider, "model_id": model_id.strip(),
                        "params": {"temperature": temperature, "max_tokens": int(max_tokens)},
                        "repeats": int(repeats), "timeout_seconds": int(timeout_seconds),
                    }
                    if region_name.strip(): judge["region_name"] = region_name.strip()
                    if concurrency > 0: judge["concurrency"] = int(concurrency)
                    if rate_limit > 0: judge["rate_limit"] = int(rate_limit)
                    if streaming: judge["streaming"] = True
                    if not use_converse_api: judge["use_converse_api"] = False
                    st.session_state.custom_judges.append(judge)
                    st.success(f"Added judge **{judge['judge_id']}**"); st.rerun()

            st.divider()
            st.markdown("**📂 Import Existing Judges**")
            uploaded_judges = st.file_uploader("Upload a judges YAML file", type=["yaml", "yml"], key="judge_upload")
            if uploaded_judges:
                try:
                    parsed = yaml.safe_load(uploaded_judges.read())
                    judges_list = parsed.get("judges", []) if isinstance(parsed, dict) else []
                    if not judges_list:
                        st.error("No judges found. Expected a 'judges' key with a list.")
                    else:
                        st.success(f"Found **{len(judges_list)}** judges in file")
                        existing_ids = {j["judge_id"] for j in st.session_state.custom_judges}
                        new_ids = [j.get("judge_id", "?") for j in judges_list if j.get("judge_id") not in existing_ids]
                        dupe_ids = [j.get("judge_id", "?") for j in judges_list if j.get("judge_id") in existing_ids]
                        if dupe_ids: st.warning(f"Duplicates (will be skipped): {', '.join(dupe_ids)}")
                        if new_ids: st.info(f"Will import: {', '.join(new_ids)}")
                        total_after = len(existing_ids) + len(new_ids)
                        if total_after > 5 and len(judges_list) > 5:
                            st.error("Import would exceed the 5-judge maximum.")
                        else:
                            imp_c1, imp_c2 = st.columns(2)
                            if imp_c1.button("✅ Import new judges", use_container_width=True):
                                added = 0
                                for j in judges_list:
                                    if j.get("judge_id") not in existing_ids and len(st.session_state.custom_judges) < 5:
                                        st.session_state.custom_judges.append(j); added += 1
                                st.success(f"Imported {added} judges"); st.rerun()
                            if imp_c2.button("🔄 Replace all with imported", use_container_width=True):
                                if len(judges_list) > 5: st.error("Cannot import more than 5 judges.")
                                else:
                                    st.session_state.custom_judges = judges_list
                                    st.success(f"Replaced with {len(judges_list)} judges"); st.rerun()
                except yaml.YAMLError as e:
                    st.error(f"Invalid YAML: {e}")

        with preview_col_j:
            if st.session_state.custom_judges:
                st.markdown(f"**📦 Your Judges ({len(st.session_state.custom_judges)}/5)**")
                for i, j in enumerate(st.session_state.custom_judges):
                    prov_icon = "🟣" if j["provider"] == "bedrock" else "🟢" if j["provider"] == "litellm" else "⚪"
                    with st.expander(f"{prov_icon} **{j['judge_id']}** — {j['provider']} / {j['model_id']}", expanded=False):
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Repeats", j["repeats"])
                        mc2.metric("Timeout", f"{j['timeout_seconds']}s")
                        mc3.metric("Temperature", j["params"]["temperature"])
                        mc4.metric("Max Tokens", j["params"]["max_tokens"])
                        extras = []
                        if j.get("region_name"): extras.append(f"Region: {j['region_name']}")
                        if j.get("concurrency"): extras.append(f"Concurrency: {j['concurrency']}")
                        if j.get("rate_limit"): extras.append(f"Rate limit: {j['rate_limit']}/s")
                        if j.get("streaming"): extras.append("Streaming: on")
                        if j.get("use_converse_api") is False: extras.append("Converse API: off")
                        if extras: st.caption(" · ".join(extras))
                        if st.button("🗑️ Remove", key=f"rm_j_{i}"):
                            st.session_state.custom_judges.pop(i); st.rerun()

                st.divider()
                st.markdown("**📄 YAML Preview**")
                yaml_j = yaml.dump({"judges": st.session_state.custom_judges}, default_flow_style=False, sort_keys=False, allow_unicode=True)
                st.code(yaml_j, language="yaml")
                dl_c1, dl_c2 = st.columns(2)
                dl_c1.download_button("⬇️ Download judges.yaml", data=yaml_j, file_name="judges.yaml", mime="text/yaml", use_container_width=True)
                if dl_c2.button("🗑️ Clear All Judges", use_container_width=True):
                    st.session_state.custom_judges = []; st.rerun()
            else:
                st.info("No judges yet. Use the form to create your first judge, or import an existing file.")

    # ── Trace Upload ──
    with trace_tab:
        st.subheader("Trace Upload")
        st.caption("Upload agent execution traces for evaluation")
        st.file_uploader("Upload trace JSON files", type=["json"], accept_multiple_files=True, key="trace_upload_setup")
        st.info("**Supported formats:** Generic JSON, AgentCore export, CloudWatch export")
        st.text_input("Or enter path to trace directory", placeholder="./traces/", key="trace_dir_setup")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: RUN
# ═══════════════════════════════════════════════════════════════════════════

elif active_page == "Run":
    st.subheader("Run Evaluation")
    st.caption("Review your configuration and trigger an evaluation")

    n_rubrics = len(st.session_state.get("custom_rubrics", []))
    n_judges = len(st.session_state.get("custom_judges", []))

    # Check for uploaded traces
    uploaded_traces = st.session_state.get("trace_upload_setup", None)
    trace_dir_path = st.session_state.get("trace_dir_setup", "")
    has_traces = (uploaded_traces and len(uploaded_traces) > 0) or (trace_dir_path and Path(trace_dir_path).expanduser().exists())

    c1, c2, c3 = st.columns(3)
    c1.metric("Rubrics Loaded", n_rubrics)
    c2.metric("Judges Configured", n_judges)
    c3.metric("Traces", "✓" if has_traces else "0")

    missing = []
    if n_judges == 0: missing.append("judges")
    if not has_traces: missing.append("traces")
    if missing:
        st.warning(f"Please configure {' and '.join(missing)} in the **Setup** page before running.")

    st.divider()

    with st.expander("📋 Configuration Summary", expanded=True):
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**Rubrics:**")
            if n_rubrics:
                for r in st.session_state.custom_rubrics:
                    sev_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡"}[r["severity"]]
                    st.markdown(f"- {sev_icon} {r['rubric_id']} (weight: {r['weight']:.1f})")
            else:
                st.caption("None — will use default rubrics")
        with sc2:
            st.markdown("**Judges:**")
            if n_judges:
                for j in st.session_state.custom_judges:
                    st.markdown(f"- 🟣 {j['judge_id']} ({j['provider']} / {j['model_id'][:40]}...)")
            else:
                st.caption("None configured")

    st.divider()
    output_dir = st.text_input("Output directory", value="./output", key="run_output_dir")
    run_clicked = st.button("▶️ Run Evaluation", use_container_width=True, type="primary", disabled=(n_judges == 0 or not has_traces))

    # ── Execution logic ──
    if run_clicked:
        try:
            run_tmp = Path(tempfile.mkdtemp(prefix="eval_run_"))
            cli_base = _app_dir.parent  # agent-eval root

            # Write judges.yaml
            judges_path = run_tmp / "judges.yaml"
            judges_path.write_text(yaml.dump({"judges": st.session_state.custom_judges}, default_flow_style=False, sort_keys=False))

            # Write rubrics.yaml if any
            rubrics_path = None
            if n_rubrics > 0:
                rubrics_path = run_tmp / "rubrics.yaml"
                rubrics_path.write_text(yaml.dump(
                    {"version": "1.0.0", "default_evidence_budget": 10000, "rubrics": st.session_state.custom_rubrics},
                    default_flow_style=False, sort_keys=False,
                ))

            # Resolve trace input
            if trace_dir_path and Path(trace_dir_path).expanduser().exists():
                trace_files = list(Path(trace_dir_path).expanduser().glob("*.json"))
            elif uploaded_traces:
                trace_files = []
                for uf in uploaded_traces:
                    tp = run_tmp / uf.name
                    tp.write_bytes(uf.getvalue())
                    trace_files.append(tp)
            else:
                trace_files = []

            if not trace_files:
                st.error("No trace files found.")
            else:
                out_base = Path(output_dir).expanduser().resolve()
                progress_bar = st.progress(0, text="Starting evaluation...")
                status_area = st.empty()
                total = len(trace_files)
                succeeded, failed = 0, 0

                for idx, trace_file in enumerate(trace_files):
                    trace_name = trace_file.stem
                    trace_out = out_base / trace_name
                    status_area.info(f"Evaluating **{trace_name}** ({idx+1}/{total})...")

                    cli_args = [
                        "--input", str(trace_file),
                        "--judge-config", str(judges_path),
                        "--output-dir", str(trace_out),
                        "--verbose",
                    ]
                    if rubrics_path:
                        cli_args.extend(["--rubrics", str(rubrics_path)])

                    result = _run_eval_cli(cli_args, cwd=str(cli_base), timeout=300)

                    if result.returncode == 0:
                        succeeded += 1
                    else:
                        failed += 1
                        with st.expander(f"❌ {trace_name} — exit code {result.returncode}"):
                            if result.stderr: st.code(result.stderr, language="text")

                    progress_bar.progress((idx + 1) / total, text=f"Completed {idx+1}/{total}")

                progress_bar.progress(1.0, text="Done!")
                if failed == 0:
                    status_area.success(f"✅ All {succeeded} traces evaluated successfully! Output: `{out_base}`")
                else:
                    status_area.warning(f"Completed: {succeeded} succeeded, {failed} failed. Output: `{out_base}`")

                # Auto-navigate to Results
                st.info("👉 Switch to **Results** in the sidebar and point to the output directory to view results.")

        except TimeoutError:
            st.error("Evaluation timed out (300s limit). Try fewer traces or simpler judge configs.")
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

    # ── Previous run results ──
    if "eval_last_output" in st.session_state:
        st.divider()
        st.caption(f"Last output: `{st.session_state.eval_last_output}`")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS
# ═══════════════════════════════════════════════════════════════════════════

elif active_page == "Results":
    if not runs:
        st.info("👈 Enter the path to an evaluation output directory in the sidebar.\n\n"
                "```\npython -m agent_eval.cli --input trace.json --judge-config judges.yaml --rubrics rubrics.yaml --output-dir ./output\n```")
        with st.expander("ℹ️ Getting Started"):
            st.markdown("**1.** Run an evaluation using the CLI to generate output artifacts\n"
                        "**2.** Point this dashboard at the output directory\n"
                        "**3.** Explore metrics, rubric scores, and judge reasoning")
        st.stop()

    # ── Compare Mode ──
    if compare_mode and len(selected_runs) == 2:
        run_a, run_b = selected_runs
        st.header(f"⚖️ Comparing: `{run_a['run_id']}` vs `{run_b['run_id']}`")

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
                invert = key in ("latency_p50", "latency_p95", "orphan_result_count")
                col.metric(label, fmt(vb), delta_str, delta_color="inverse" if invert else "normal")
            else:
                col.metric(label, "—")

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

    # ── Single Run: Traffic Light Summary ──
    if trace_eval:
        rubric_results = trace_eval.get("rubric_results", [])
        dm = trace_eval.get("deterministic_metrics", {})
        scores = [rubric_score(rr) for rr in rubric_results]
        valid_scores = [s for s in scores if s is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
        overall_status = score_to_status(avg_score)
        cfg = STATUS_CONFIG[overall_status]

        st.markdown(
            f'<div class="traffic-light tl-{overall_status}">'
            f'{cfg["emoji"]} Overall: {cfg["label"]}'
            f'{"  —  Avg score: " + f"{avg_score:.1f}/5" if avg_score else ""}'
            f'</div>', unsafe_allow_html=True,
        )
        st.caption(f"Run: `{run['run_id']}`")

        if rubric_results:
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

    # ── Results Sub-tabs ──
    metrics_tab, turns_tab, rubrics_tab, judges_tab, cost_tab = st.tabs(
        ["📊 Metrics", "🔄 Turns", "📋 Rubric Scorecard", "⚖️ Judge Detail", "💰 Cost"]
    )

    # ── Metrics ──
    with metrics_tab:
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

            # Rubric radar
            rubric_results = trace_eval.get("rubric_results", [])
            if rubric_results:
                seen = {}
                for rr in rubric_results:
                    rid = rr.get("rubric_id", "?")
                    s = rubric_score(rr)
                    if s is not None and rid not in seen:
                        seen[rid] = s
                if seen:
                    st.markdown("#### 📋 Rubric Radar")
                    radar_df = pd.DataFrame([{"Rubric": k, "Score": v} for k, v in seen.items()])
                    radar_df["color"] = radar_df["Score"].apply(
                        lambda v: "#2ecc71" if v >= 4 else ("#f39c12" if v >= 3 else "#e74c3c")
                    )
                    radar_chart = alt.Chart(radar_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                        x=alt.X("Rubric:N", sort=None),
                        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 5])),
                        color=alt.Color("color:N", scale=None),
                        tooltip=["Rubric", "Score"],
                    ).properties(height=250)
                    st.altair_chart(radar_chart, use_container_width=True)

            # Judge summary
            js = trace_eval.get("judge_summary", {})
            if js:
                with st.expander("Judge Execution Summary"):
                    jc1, jc2, jc3, jc4 = st.columns(4)
                    jc1.metric("Total Jobs", js.get("total_jobs", "—"))
                    jc2.metric("Succeeded", js.get("succeeded", "—"))
                    jc3.metric("Failed", js.get("failed", "—"))
                    jc4.metric("Judges Used", js.get("judge_count", "—"))
        else:
            st.warning("No trace_eval.json found.")

    # ── Turns ──
    with turns_tab:
        if not normalized:
            st.warning("No normalized run file found.")
        else:
            turns = normalized.get("turns", [])
            if not turns:
                st.info("No turns in this trace.")
            else:
                st.markdown("#### ⏱️ Turn Timeline")
                timeline_data = []
                for i, t in enumerate(turns):
                    lat = t.get("total_latency_ms") or 0
                    n_steps = len(t.get("steps", []))
                    query_preview = (t.get("user_query") or "")[:50]
                    timeline_data.append({"Turn": f"T{i+1}", "Latency (ms)": lat, "Steps": n_steps, "Query": query_preview})

                tl_df = pd.DataFrame(timeline_data)
                tl_df["color"] = tl_df["Latency (ms)"].apply(
                    lambda v: "#e74c3c" if v > 5000 else ("#f39c12" if v > 2000 else "#2ecc71")
                )
                lat_chart = alt.Chart(tl_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("Turn:N", sort=None), y=alt.Y("Latency (ms):Q"),
                    color=alt.Color("color:N", scale=None), tooltip=["Turn", "Latency (ms)", "Steps", "Query"],
                ).properties(height=200)
                st.altair_chart(lat_chart, use_container_width=True)

                st.divider()
                st.markdown("#### 🔎 Turn Detail")
                turn_labels = [f"Turn {i+1}: {(t.get('user_query') or '(no query)')[:60]}" for i, t in enumerate(turns)]

                if "turn_idx" not in st.session_state:
                    st.session_state.turn_idx = 0
                st.session_state.turn_idx = max(0, min(st.session_state.turn_idx, len(turns) - 1))

                nav_col1, nav_col2, nav_col3 = st.columns([1, 6, 1])
                with nav_col1:
                    if st.button("◀ Prev", disabled=st.session_state.turn_idx <= 0, use_container_width=True):
                        st.session_state.turn_idx -= 1; st.rerun()
                with nav_col3:
                    if st.button("Next ▶", disabled=st.session_state.turn_idx >= len(turns) - 1, use_container_width=True):
                        st.session_state.turn_idx += 1; st.rerun()
                with nav_col2:
                    selected_idx = st.selectbox("Select turn", range(len(turns)), index=st.session_state.turn_idx, format_func=lambda i: turn_labels[i])
                    if selected_idx != st.session_state.turn_idx:
                        st.session_state.turn_idx = selected_idx; st.rerun()

                turn = turns[selected_idx]
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

                red_flags = []
                if turn_rubric_scores:
                    for rid, s in turn_rubric_scores.items():
                        if s <= 2.0: red_flags.append(f"{rid}: {s:.0f}/5")
                    avg = sum(turn_rubric_scores.values()) / len(turn_rubric_scores)
                    turn_status = score_to_status(avg)
                else:
                    avg = None
                    turn_status = "unknown"
                tcfg = STATUS_CONFIG[turn_status]

                st.markdown(f"### {tcfg['emoji']} Turn {selected_idx + 1} Summary" + (f" — Avg: **{avg:.1f}/5**" if avg else ""))
                sc1, sc2, sc3 = st.columns(3)
                sc1.markdown(f"**Asked:** {query[:80]}{'…' if len(query) > 80 else ''}")
                sc2.markdown(f"**Tools:** {', '.join(tool_names) if tool_names else 'None ⚠️'}")
                sc3.markdown(f"**Errors:** {len(errors)}" + (" 🚨" if errors else " ✅"))
                if answer_preview:
                    st.markdown(f"**Answer:** {answer_preview}")

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

                with st.expander("🔍 Detailed Turn Breakdown", expanded=False):
                    tc1, tc2, tc3 = st.columns(3)
                    tc1.metric("Confidence", f"{turn.get('confidence', 0):.2f}")
                    lat = turn.get("total_latency_ms")
                    tc2.metric("Latency", f"{lat:.0f}ms" if lat else "—")
                    tc3.metric("Steps", len(steps))
                    st.markdown("**💬 User Query**")
                    st.info(turn.get("user_query") or "_No query captured_")
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
                    st.markdown("**🎯 Final Answer**")
                    st.success(turn.get("final_answer") or "_No answer captured_")

    # ── Rubric Scorecard ──
    with rubrics_tab:
        if not trace_eval:
            st.warning("No trace_eval.json found.")
        else:
            rubric_results = trace_eval.get("rubric_results", [])
            if not rubric_results:
                st.info("No rubric results.")
            else:
                rubric_ids = [rr.get("rubric_id", "?") for rr in rubric_results]
                unique_ids = list(dict.fromkeys(rubric_ids))
                selected_rubric = st.selectbox("Select rubric to inspect", ["All Rubrics"] + unique_ids)

                if selected_rubric == "All Rubrics":
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
                            "Status": STATUS_CONFIG[status]["emoji"], "Rubric": rid, "Scope": scope,
                            "Turn": turn_id if turn_id else "—",
                            "Score": f"{s:.1f}" if s is not None else (vote or "—"),
                            "Disagreement": f"{dis:.2f}", "High Risk": "🚨" if risk else "",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    st.caption("👆 Select a rubric above to drill into judge details")
                else:
                    matching = [rr for rr in rubric_results if rr.get("rubric_id") == selected_rubric]
                    for rr in matching:
                        turn_id = rr.get("turn_id")
                        cross = rr.get("cross_judge_result", {})
                        s = rubric_score(rr)
                        status = score_to_status(s)
                        cfg = STATUS_CONFIG[status]
                        header = f"{cfg['emoji']} **{selected_rubric}**"
                        if turn_id: header += f" — Turn: `{turn_id}`"
                        st.markdown(header)
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Score", f"{s:.1f}/5" if s is not None else (cross.get("weighted_vote") or "—"))
                        mc2.metric("Disagreement", f"{cross.get('disagreement_signal', 0):.2f}")
                        mc3.metric("High Risk", "Yes 🚨" if cross.get("high_risk_flag") else "No")

                        within = rr.get("within_judge_results", [])
                        if within:
                            st.markdown("**Per-judge breakdown:**")
                            judge_rows = []
                            for wj in within:
                                judge_rows.append({
                                    "Judge": wj.get("judge_id", "?"), "Median": wj.get("median"),
                                    "Mean": f"{wj['mean']:.2f}" if wj.get("mean") is not None else "—",
                                    "Variance": f"{wj.get('variance', 0):.3f}",
                                    "Samples": wj.get("sample_size", 0), "Vote": wj.get("majority_vote") or "—",
                                })
                            st.dataframe(pd.DataFrame(judge_rows), use_container_width=True, hide_index=True)

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

    # ── Judge Detail ──
    with judges_tab:
        if not judge_runs:
            st.warning("No judge_runs.jsonl found.")
        else:
            st.caption(f"{len(judge_runs)} judge run records")
            rubric_ids = sorted(set(r.get("rubric_id", "?") for r in judge_runs))
            judge_ids = sorted(set(r.get("judge_id", "?") for r in judge_runs))
            fc1, fc2 = st.columns(2)
            filter_rubric = fc1.multiselect("Filter by rubric", rubric_ids, default=rubric_ids)
            filter_judge = fc2.multiselect("Filter by judge", judge_ids, default=judge_ids)
            filtered = [r for r in judge_runs if r.get("rubric_id") in filter_rubric and r.get("judge_id") in filter_judge]
            for record in filtered:
                rid = record.get("rubric_id", "?")
                jid = record.get("judge_id", "?")
                score = record.get("score") or record.get("category") or "—"
                reasoning = record.get("reasoning", "")
                with st.expander(f"[{rid}] Judge: {jid} → Score: {score}"):
                    if reasoning: st.markdown(reasoning)
                    st.json({k: v for k, v in record.items() if k != "reasoning" and v is not None})

    # ── Cost Insights ──
    with cost_tab:
        run_results = run.get("results") if run else None
        cost_data = run_results.get("cost_insights") if run_results else None
        if not cost_data:
            st.info("No cost data available. Cost insights require traces with token usage data (e.g., `prompt_tokens`, `completion_tokens` in step attributes).")
        else:
            st.markdown("#### 💰 Cost Summary")
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Total Cost", f"${cost_data.get('total_cost_usd', 0):.4f}")
            cc2.metric("Input Tokens", f"{cost_data.get('total_input_tokens', 0):,}")
            cc3.metric("Output Tokens", f"{cost_data.get('total_output_tokens', 0):,}")
            cc4.metric("Total Tokens", f"{cost_data.get('total_tokens', 0):,}")

            st.caption(f"Pricing source: {cost_data.get('pricing_source', 'unknown')}")

            # Cost by model
            cost_by_model = cost_data.get("cost_by_model", {})
            tokens_by_model = cost_data.get("tokens_by_model", {})
            if cost_by_model:
                st.markdown("#### 📊 Cost by Model")
                model_rows = []
                for model_id, cost in sorted(cost_by_model.items(), key=lambda x: -x[1]):
                    toks = tokens_by_model.get(model_id, {})
                    model_rows.append({
                        "Model": model_id,
                        "Input Tokens": toks.get("input", 0),
                        "Output Tokens": toks.get("output", 0),
                        "Cost (USD)": f"${cost:.6f}",
                    })
                st.dataframe(model_rows, use_container_width=True, hide_index=True)

            # Cost per turn
            cost_per_turn = cost_data.get("cost_per_turn", [])
            if cost_per_turn:
                st.markdown("#### 🔄 Cost per Turn")
                turn_rows = []
                for t in cost_per_turn:
                    turn_rows.append({
                        "Turn": t["turn_id"],
                        "Input Tokens": t.get("input_tokens", 0),
                        "Output Tokens": t.get("output_tokens", 0),
                        "Model Calls": t.get("model_calls", 0),
                        "Cost (USD)": f"${t.get('total_cost_usd', 0):.6f}",
                    })
                st.dataframe(turn_rows, use_container_width=True, hide_index=True)

                # Bar chart of cost per turn
                import pandas as pd
                chart_df = pd.DataFrame([
                    {"Turn": t["turn_id"], "Cost (USD)": t.get("total_cost_usd", 0)}
                    for t in cost_per_turn
                ])
                if not chart_df.empty and chart_df["Cost (USD)"].sum() > 0:
                    st.bar_chart(chart_df.set_index("Turn"))

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: TRENDS
# ═══════════════════════════════════════════════════════════════════════════

elif active_page == "Trends":
    if not runs:
        st.info("👈 Enter the path to an evaluation output directory in the sidebar to view trends.")
        st.stop()

    st.subheader("📈 Quality Trends Across Runs")

    if len(runs) < 2:
        st.info("Trends require multiple evaluation runs in the same directory.\n\n"
                "Point to a parent directory containing multiple output folders to see quality over time.")
    else:
        trend_rows = []
        for r in runs:
            te = r.get("trace_eval") or {}
            dm = te.get("deterministic_metrics", {})
            rubric_results = te.get("rubric_results", [])
            scores = [rubric_score(rr) for rr in rubric_results if rubric_score(rr) is not None]
            avg = sum(scores) / len(scores) if scores else None
            norm = r.get("normalized") or {}
            processed_at = (norm.get("metadata") or {}).get("processed_at", "")
            trend_rows.append({
                "Run": r["run_id"], "Avg Score": round(avg, 2) if avg is not None else None,
                "Turns": dm.get("turn_count", 0), "Tool Success": dm.get("tool_success_rate"),
                "Latency p50 (ms)": dm.get("latency_p50"), "Latency p95 (ms)": dm.get("latency_p95"),
                "Processed": processed_at[:19] if processed_at else "—",
            })

        trend_df = pd.DataFrame(trend_rows)

        score_df = trend_df[trend_df["Avg Score"].notna()].copy()
        if not score_df.empty:
            st.markdown("#### 📊 Average Rubric Score")
            score_df["color"] = score_df["Avg Score"].apply(
                lambda v: "#2ecc71" if v >= 4 else ("#f39c12" if v >= 3 else "#e74c3c")
            )
            score_chart = alt.Chart(score_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("Run:N", sort=None), y=alt.Y("Avg Score:Q", scale=alt.Scale(domain=[0, 5])),
                color=alt.Color("color:N", scale=None), tooltip=["Run", "Avg Score", "Turns"],
            ).properties(height=250)
            st.altair_chart(score_chart, use_container_width=True)

        lat_df = trend_df[trend_df["Latency p50 (ms)"].notna()].copy()
        if not lat_df.empty:
            st.markdown("#### ⏱️ Latency Trend")
            lat_melted = lat_df.melt(id_vars=["Run"], value_vars=["Latency p50 (ms)", "Latency p95 (ms)"], var_name="Metric", value_name="ms")
            lat_chart = alt.Chart(lat_melted).mark_line(point=True).encode(
                x=alt.X("Run:N", sort=None), y=alt.Y("ms:Q", title="Latency (ms)"),
                color="Metric:N", tooltip=["Run", "Metric", "ms"],
            ).properties(height=250)
            st.altair_chart(lat_chart, use_container_width=True)

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
                x=alt.X("Run:N", sort=None), y=alt.Y("Rubric:N", sort=None),
                color=alt.Color("Score:Q", scale=alt.Scale(domain=[1, 5], scheme="redyellowgreen")),
                tooltip=["Run", "Rubric", "Score"],
            ).properties(height=max(200, len(rt_df["Rubric"].unique()) * 30))
            st.altair_chart(rubric_heatmap, use_container_width=True)

        st.divider()
        st.markdown("#### 📋 Run Summary Table")
        st.dataframe(trend_df, use_container_width=True, hide_index=True)
