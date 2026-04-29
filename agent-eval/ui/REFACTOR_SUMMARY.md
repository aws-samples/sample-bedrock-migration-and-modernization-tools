# Agent Eval Dashboard — UI Refactor Summary

**Branch:** `feature/eval-dashboard-and-sa-rubrics`
**File changed:** `agent-eval/ui/app.py` (934 insertions, 884 deletions)
**Date:** 2026-04-17

---

## What Changed

### 1. Layout: Flat Tabs → 360-Eval Style Sidebar Navigation

**Before:** 7 flat tabs in a single horizontal bar
```
📊 Metrics | 🔄 Turns | 📋 Rubrics | ⚖️ Judges | 📈 Trends | 🛠️ Rubric Builder | ⚖️ Judge Config
```
- Input (configuration) and Output (results) mixed together
- Tab bar wraps/truncates on smaller screens
- No workflow guidance for new users

**After:** Sidebar radio navigation with 4 workflow-ordered pages
```
Sidebar: Setup → Run → Results → Trends
```
- **Setup** page: sub-tabs for Rubric Config | Judge Config | Trace Upload
- **Run** page: config summary, validation, trigger evaluation
- **Results** page: sub-tabs for Metrics | Turns | Rubric Scorecard | Judge Detail
- **Trends** page: cross-run quality analysis

This mirrors the [360-eval dashboard](https://360-eval.wwso.aws.dev/) pattern used in the same repo.

### 2. New Feature: Judge Configuration Builder

Added a full visual judge configuration UI (previously only rubric builder existed):

- **Visual form:** judge_id, provider, model_id, temperature, max_tokens, repeats, timeout, region, concurrency, rate_limit, streaming, Converse API
- **Preset selector:** one-click fill for Claude Sonnet, Claude Opus, Nova Pro, Mock Judge
- **5-judge maximum** enforced (matching `judge_config_schema.py` validation)
- **Duplicate ID detection**
- **Expandable cards** per judge with metrics display
- **YAML preview & export:** live rendering + download `judges.yaml`
- **Import existing `judges.yaml`:** merge or replace modes with duplicate detection

### 3. New Feature: Wired Run Page

The Run page now triggers real evaluations end-to-end:

1. Writes rubrics + judges from Setup to temp YAML files
2. Resolves traces from file upload or directory path
3. Calls `python -m agent_eval.cli` for each trace via subprocess
4. Shows progress bar with per-trace status
5. Displays errors inline with expandable stderr output
6. Points user to Results page on completion

### 4. Setup Page: Side-by-Side Form + Preview Layout

Each Setup sub-tab uses a two-column layout:
- **Left (60%):** Configuration form + import
- **Right (40%):** Live preview of configured items + YAML preview + export

### 5. Context-Sensitive Sidebar

The sidebar content adapts based on the active page:
- **Setup/Run:** Only navigation shown
- **Results/Trends:** Run loading, comparison mode, PDF export, embed card

---

## What's Preserved (No Functionality Lost)

All existing features carry over unchanged:

- ✅ Traffic light summary (pass/warn/fail per rubric)
- ✅ Deterministic metrics (turns, steps, tool calls, latency, success rate)
- ✅ Rubric radar chart
- ✅ Turn explorer with Prev/Next navigation
- ✅ Turn-level rubric scores and red flag detection
- ✅ Rubric scorecard with progressive drill-down
- ✅ Judge detail with multi-select filtering
- ✅ Run comparison mode (metric deltas + rubric score table)
- ✅ Trends: score bars, latency trends, rubric heatmap, summary table
- ✅ Drag-and-drop trace upload with inline mock evaluation
- ✅ PDF/markdown report export
- ✅ Embeddable Slack/email summary card
- ✅ Shareable URLs via query params
- ✅ Rubric Builder (visual create + YAML import/export)
- ✅ Custom CSS (traffic light badges, metric cards, dark mode)

---

## File Size

| Version | Lines |
|---------|-------|
| Before (7 flat tabs) | 1,362 |
| After (sidebar nav + judge builder + wired run) | 1,198 |

Net reduction of 164 lines despite adding two new features (judge builder + wired run page).

---

## How to Test

```bash
cd agent-eval
streamlit run ui/app.py
```

1. **Setup → Rubric Configuration:** Create a rubric or import `test-fixtures/rubrics.default.yaml`
2. **Setup → Judge Configuration:** Select "Mock Judge" preset, click Add
3. **Setup → Trace Upload:** Enter `ui/example_traces/` as the trace directory
4. **Run:** Review config summary, click ▶️ Run Evaluation
5. **Results:** Point sidebar to the output directory, explore all sub-tabs
6. **Trends:** Point to a directory with multiple run outputs (e.g., `ui/real_bedrock_output*` parent)
