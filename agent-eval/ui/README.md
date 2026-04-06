# Agent Eval Dashboard

Streamlit UI for visualizing agent evaluation results.

## Quick Start

```bash
pip install streamlit
cd agent-eval
streamlit run ui/app.py
```

## Views

### 📊 Run Overview
- Deterministic metrics at a glance (turns, steps, tool calls, latency)
- Tool success rate, orphan results, timestamp coverage
- Quality flags (stitched trace suspect, single-turn fallback)
- Judge execution summary
- Adapter statistics

### 🔄 Turn Explorer
- Step-by-step walkthrough of each conversation turn
- User query → tool calls → model reasoning → final answer
- Expandable step detail with raw source events
- Latency per step

### 📋 Rubric Scorecard
- Cross-judge aggregated scores per rubric
- Disagreement signal and high-risk flags
- Per-judge breakdown (median, mean, variance, sample size)

### ⚖️ Judge Detail
- Raw judge run records from `judge_runs.jsonl`
- Filter by rubric and judge
- Full reasoning text and scoring detail

## Input

Point the dashboard at any evaluation output directory containing:
- `trace_eval.json` — evaluation metrics and rubric results
- `normalized_run.*.json` — normalized trace data
- `judge_runs.jsonl` — raw judge responses
- `results.json` (optional) — aggregated results
