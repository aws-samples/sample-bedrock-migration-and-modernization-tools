# Agent Eval Dashboard

Streamlit UI for configuring, running, and visualizing agent evaluations.

## Quick Start

```bash
pip install streamlit altair pandas pyyaml
cd agent-eval
streamlit run ui/app.py
```

## Navigation

The dashboard uses sidebar radio navigation with four workflow-ordered pages:

### 📥 Setup

Configure your evaluation before running.

**🛠️ Rubric Configuration**
- Visual rubric builder: ID, severity, weight, scope, scoring type, evaluation instructions, evidence selectors
- Side-by-side form + live preview with YAML export
- Import existing `rubrics.yaml` (merge or replace)

**⚖️ Judge Configuration**
- Visual judge builder: ID, provider, model ID, temperature, max tokens, repeats, timeout, region
- Provider options: Amazon Bedrock (🟣), LiteLLM (🟢 — 100+ providers), Mock (⚪)
- Preset selector: Claude Sonnet, Claude Opus, Nova Pro, Mock Judge
- 5-judge maximum enforced with duplicate detection
- Import existing `judges.yaml` (merge or replace)

**📎 Trace Upload**
- Upload trace JSON files or point to a directory
- Supports: Generic JSON, AgentCore export, CloudWatch export

### ▶️ Run

- Configuration summary with validation warnings
- Triggers evaluation via CLI (`python -m agent_eval.cli`)
- Progress bar with per-trace status and error reporting

### 📤 Results

View evaluation results with traffic light summary and four sub-tabs:

**📊 Metrics** — Deterministic metrics (turns, steps, tool calls, latency, success rate), rubric radar chart, judge execution summary

**🔄 Turns** — Turn timeline with latency bars, turn detail with Prev/Next navigation, per-turn rubric scores, red flag detection, detailed step breakdown

**📋 Rubric Scorecard** — All rubrics summary table, drill-down into per-judge breakdown and reasoning

**⚖️ Judge Detail** — Raw judge run records, filter by rubric and judge, full reasoning text

**💰 Cost** — Token usage and cost breakdown from trace data. Shows total cost, input/output tokens, cost-by-model table, cost-per-turn table with bar chart. Pricing resolved via LiteLLM or user overrides. Requires traces with token data (`prompt_tokens`, `completion_tokens` in step attributes).

**⏱️ Latency** — Performance analysis with time-to-first-token (TTFT), latency breakdown by category (🧠 LLM vs 🔧 Tool vs ⚙️ Overhead), per-turn latency table with stacked bar chart, tokens/sec throughput, and top 5 slowest steps for optimization targeting.

**Additional features:**
- Run comparison mode (metric deltas + rubric score table)
- Drag-and-drop trace upload with inline mock evaluation
- PDF/markdown report export
- Embeddable Slack/email summary card
- Shareable URLs via query params

### 📈 Trends

Cross-run quality analysis (requires multiple runs in the same directory):
- Average rubric score bar chart
- Latency p50/p95 trend lines
- Per-rubric heatmap across runs
- Run summary table

## Input

Point the Results page sidebar at any evaluation output directory containing:
- `trace_eval.json` — evaluation metrics and rubric results
- `normalized_run.*.json` — normalized trace data
- `judge_runs.jsonl` — raw judge responses
- `results.json` (optional) — aggregated results
