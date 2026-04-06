# AgentCore → Evaluation Bridge

## Overview

`evaluate_agentcore.py` connects the existing AgentCore trace extraction pipeline to the evaluation framework, enabling single-command evaluation of agents running on Amazon Bedrock AgentCore.

## Usage

### Evaluate previously exported AgentCore traces

If you've already run the 3-stage extraction pipeline and have normalized traces in a `merged/` directory:

```bash
python agent_eval/tools/agentcore_pipeline/evaluate_agentcore.py \
  --agentcore-export-dir ./exports/agentcore_exports/run-abc123/merged \
  --judge-config test-fixtures/judges.mock.yaml \
  --rubrics ui/example_traces/agent_rubrics.yaml \
  --output-dir ./output
```

### Full pipeline: extract from AgentCore + evaluate

If you have AWS credentials and an AgentCore runtime ARN:

```bash
python agent_eval/tools/agentcore_pipeline/evaluate_agentcore.py \
  --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/my-agent:v1 \
  --judge-config judges.yaml \
  --rubrics ui/example_traces/agent_rubrics.yaml \
  --output-dir ./output
```

### Export only (skip evaluation)

```bash
python agent_eval/tools/agentcore_pipeline/evaluate_agentcore.py \
  --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/my-agent:v1 \
  --export-only \
  --output-dir ./output
```

## Options

| Flag | Required | Description |
|------|----------|-------------|
| `--agent-runtime-arn` | One of these | AgentCore runtime ARN to extract traces from |
| `--agentcore-export-dir` | required | Path to previously exported merged directory |
| `--judge-config` | Yes (unless `--export-only`) | Path to judges.yaml (mock or real Bedrock) |
| `--rubrics` | No | Custom rubrics.yaml (merges with defaults) |
| `--output-dir` | No | Output directory (default: `./agentcore_eval_output`) |
| `--days` | No | Days of traces to export (default: 7) |
| `--region` | No | AWS region (default: us-east-1) |
| `--export-only` | No | Export traces without evaluating |
| `--verbose` | No | Verbose output |

## Pipeline Flow

```
AgentCore Runtime ARN
        ↓
┌─────────────────────────┐
│ Stage 1: Export turns    │  (APPLICATION_LOGS → session/trace IDs)
│ Stage 2: Build index    │  (OTEL spans → conversation data)
│ Stage 3: Merge X-Ray    │  (X-Ray traces → steps + latency)
└────────────┬────────────┘
             ↓
     normalized_run.json
             ↓
┌─────────────────────────┐
│ Evaluation Pipeline      │
│ • Deterministic metrics  │
│ • Rubric-based scoring   │
│ • Judge execution        │
└────────────┬────────────┘
             ↓
    Evaluation Artifacts
    ├── trace_eval.json
    ├── judge_runs.jsonl
    └── results.json
             ↓
    View in Dashboard
    (streamlit run ui/app.py)
```

## Output

```
<output-dir>/
├── agentcore_eval_summary.json    # Overall summary (runs evaluated, pass/fail)
├── export/                        # Raw AgentCore extraction output (if using ARN)
└── evaluations/
    └── normalized_run.<run-id>/
        ├── trace_eval.json        # Metrics + rubric scores
        ├── judge_runs.jsonl       # Raw judge responses
        ├── results.json           # Aggregated results
        └── normalized_run.*.json  # Normalized trace
```

## Testing Without AWS Credentials

A simulated AgentCore export is included for testing:

```bash
python agent_eval/tools/agentcore_pipeline/evaluate_agentcore.py \
  --agentcore-export-dir ui/agentcore_simulated_export/merged \
  --judge-config test-fixtures/judges.mock.yaml \
  --output-dir ./test-output \
  --verbose
```

This uses a 3-turn customer support agent trace (knowledge base lookup, order management, warranty replacement) with mock judges.

## AWS Permissions

When using `--agent-runtime-arn`, you need:

```json
{
  "Effect": "Allow",
  "Action": [
    "logs:DescribeLogGroups",
    "logs:StartQuery",
    "logs:GetQueryResults",
    "xray:BatchGetTraces"
  ],
  "Resource": "*"
}
```
