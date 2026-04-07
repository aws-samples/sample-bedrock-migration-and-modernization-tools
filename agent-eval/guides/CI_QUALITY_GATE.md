# CI/CD Quality Gate for Agent Evaluation

## Overview

A GitHub Actions workflow that automatically evaluates agent traces on every PR, enforces quality thresholds, and posts a scorecard comment — so your team catches regressions before they merge.

## How It Works

When a PR touches any file in `agent-eval/`, the workflow runs four jobs:

```
1. Unit Tests        → pytest on agent_eval/tests/
2. Smoke Test        → Evaluates baseline traces with mock judges
3. Quality Gate      → Checks scores against thresholds (blocks merge if failed)
4. PR Comment        → Posts a scorecard summary on the PR
```

## What the PR Comment Looks Like

Every PR gets an auto-posted scorecard:

```
## 🔍 Agent Eval Scorecard
**Status:** ✅ Passed

| Metric             | Value |
|--------------------|-------|
| Turns              | 1     |
| Tool Calls         | 1     |
| Tool Success Rate  | 100%  |
| Latency p50        | 5800 ms |
| Avg Rubric Score   | 3.0/5 |
| Judge Jobs         | 16 (16 passed) |
```

With expandable rubric-level detail.

## Quality Thresholds

The quality gate enforces these minimums:

| Check | Threshold | What it catches |
|-------|-----------|-----------------|
| Tool success rate | ≥ 80% | Broken tool integrations |
| Orphan tool results | 0 | Tool call/result linking bugs |
| Avg rubric score | ≥ 2.5/5 | Overall quality regression |

If any threshold fails, the PR is blocked from merging.

## Customizing Thresholds

Edit the threshold values in `.github/workflows/agent-eval-ci.yml` under the `quality-gate` job:

```python
# In the "Check quality thresholds" step:
if tsr < 0.8:          # Tool success rate minimum
if orphans > 0:        # Orphan results maximum
if avg < 2.5:          # Average rubric score minimum
```

## Adding Your Own Traces to CI

To add new baseline traces that CI evaluates:

1. Add your trace JSON to `agent-eval/test-fixtures/baseline/` with a `good_`, `bad_`, or `partial_` prefix
2. Update `test-fixtures/baseline/manifest.yaml` with the trace metadata
3. The smoke test automatically picks up all `good_*.json` files

## Running Locally

You can run the same checks locally before pushing:

```bash
cd agent-eval

# Unit tests
pytest agent_eval/tests/ -v

# Smoke test (same as CI)
./scripts/smoke_test_raw_traces.sh

# Full pre-push gate
./scripts/pre_push_check.sh
```

## Workflow Triggers

| Trigger | When |
|---------|------|
| Pull request | Any PR touching `agent-eval/` files |
| Push to main | Direct pushes to main touching `agent-eval/` |
| Manual | Via "Run workflow" button in GitHub Actions tab |

## Artifacts

The smoke test uploads evaluation output as GitHub Actions artifacts (retained for 14 days). Download them from the Actions tab to inspect `trace_eval.json`, `judge_runs.jsonl`, and normalized runs.
