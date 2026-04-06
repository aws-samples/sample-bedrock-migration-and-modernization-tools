# Agent Evaluation — Example Traces & Rubrics

## Overview

Example traces, custom rubrics, and a trace generator for evaluating AI agents using the Agent Evaluation Framework.

## Trace Generator

`trace_generator.py` — A reusable Python class for converting agent sessions into the evaluator's trace format.

```python
from trace_generator import TraceBuilder

tb = TraceBuilder("my-session")
tb.user_message("What's the status of order #12345?")
t1 = tb.tool_call("order_lookup", {"order_id": "12345"})
tb.tool_result(t1, '{"status": "shipped"}')
tb.model_output("Order #12345 has been shipped.")
tb.save("traces/my-trace.json")
```

Supports: `user_message`, `tool_call`, `tool_result`, `model_output`, `subagent_call`.

## Example Traces (6 scenarios)

| Trace | Scenario | Turns | Tool Calls | Tests |
|-------|----------|-------|------------|-------|
| `coding-session.json` | Multi-turn coding assistance | 6 | 9 | Tool selection, accuracy, coherence |
| `multi-tool-research.json` | Account research with multiple APIs | 2 | 4 | Data accuracy, multi-tool chaining |
| `subagent-routing.json` | Request delegation to specialist agent | 1 | 1 | Subagent delegation, routing |
| `parallel-subagent.json` | Parallel subagent execution | 1 | 2 | Parallel delegation, context merging |
| `error-recovery.json` | Permission denied + retry flow | 1 | 4 | Error recovery, graceful degradation |
| `hallucination-risk.json` | Ungrounded answer + correction | 2 | 1 | Hallucination detection, data accuracy |

## Agent Rubrics (10 custom)

`agent_rubrics.yaml` — Domain-specific evaluation criteria for AI agents.

| Rubric | Severity | Weight | What it measures |
|--------|----------|--------|------------------|
| TOOL_SELECTION | high | 1.5 | Did the agent pick the right tool? |
| ANSWER_ACCURACY | critical | 2.0 | Is the answer factually grounded in tool results? |
| ROUTING_QUALITY | high | 1.5 | Was the request routed to the right subagent? |
| CONCISENESS | medium | 0.8 | Appropriately concise without losing info? |
| CONVERSATION_COHERENCE | medium | 1.0 | Context retention across turns |
| ERROR_RECOVERY | high | 1.2 | Graceful handling when tools fail |
| HALLUCINATION_DETECTION | critical | 2.0 | Avoids stating ungrounded facts |
| DATA_ACCURACY | critical | 2.0 | Numeric data matches tool results exactly |
| SUBAGENT_DELEGATION | high | 1.5 | Correct delegation in multi-agent systems |
| ACTION_ITEM_EXTRACTION | medium | 1.0 | Surfaces follow-ups with owner and timeline |

## Running Evaluations

```bash
# With mock judges (no AWS credentials needed)
python -m agent_eval.cli \
  --input ui/example_traces/hallucination-risk.json \
  --judge-config test-fixtures/judges.mock.yaml \
  --rubrics ui/example_traces/agent_rubrics.yaml \
  --output-dir ./output

# With real Bedrock judges
python -m agent_eval.cli \
  --input ui/example_traces/multi-tool-research.json \
  --judge-config test-fixtures/judges.real.single.yaml \
  --rubrics ui/example_traces/agent_rubrics.yaml \
  --output-dir ./output
```

## Viewing Results

```bash
streamlit run ui/app.py
```

## Key Findings

The hallucination trace demonstrates the rubrics' value:
- **Turn 1** (ungrounded answer): HALLUCINATION_DETECTION=1/5, DATA_ACCURACY=1/5, TOOL_SELECTION=1/5
- **Turn 2** (tool-grounded answer): All rubrics score 5/5

This shows the rubrics can differentiate between grounded and ungrounded agent behavior at the turn level.
