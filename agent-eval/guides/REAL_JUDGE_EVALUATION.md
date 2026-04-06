# Real Bedrock Judge Evaluation Results

## Overview

Benchmark results from evaluating 6 diverse agent traces using Claude Sonnet as a real LLM judge via Amazon Bedrock. These results validate that the evaluation framework produces meaningful, differentiated scores across different agent behaviors.

## Configuration

- **Judge**: Claude Sonnet (`us.anthropic.claude-sonnet-4-20250514-v1:0`)
- **Temperature**: 0 (deterministic)
- **Rubrics**: 18 total (8 default + 10 custom agent rubrics)
- **Total judge jobs**: 522 across all traces

## Results Summary

| Trace | Avg Score | Turns | Tools | Tool Success | Key Finding |
|-------|-----------|-------|-------|-------------|-------------|
| 🟢 parallel-subagent | 4.6/5 | 1 | 2 | 100% | Best performer — correct parallel delegation |
| 🟢 multi-tool-research | 4.2/5 | 2 | 4 | 100% | Strong tool usage and data accuracy |
| 🟢 error-recovery | 4.2/5 | 1 | 4 | 75% | Perfect error recovery (5/5) after permission denied |
| 🟢 subagent-routing | 4.1/5 | 1 | 1 | 100% | Good delegation to specialist agent |
| 🟡 coding-session | 3.4/5 | 6 | 9 | 100% | Longest trace, data accuracy lower on condensed results |
| 🔴 hallucination-risk | 2.7/5 | 2 | 1 | 100% | Correctly caught ungrounded answer in Turn 1 |

**Overall average: 3.9/5**

## Detailed Results

### hallucination-risk (2.7/5) — Validates Hallucination Detection

The most important test case. Turn 0 answers a factual question without any tool call; Turn 1 correctly searches and reports no results.

| Rubric | Turn 0 (ungrounded) | Turn 1 (grounded) |
|--------|---------------------|-------------------|
| HALLUCINATION_DETECTION | 🔴 1/5 | 🟢 5/5 |
| DATA_ACCURACY | 🔴 1/5 | 🟢 5/5 |
| ANSWER_ACCURACY | 🔴 1/5 | 🟢 5/5 |
| TOOL_SELECTION | 🔴 1/5 | 🔴 1/5 |
| TOOL_GROUNDEDNESS | 🟡 3/5 | 🔴 1/5 |
| CONCISENESS | 🟢 5/5 | 🟢 5/5 |
| ERROR_RECOVERY | 🟡 3/5 | 🟢 5/5 |

**Takeaway**: The rubrics correctly identify that a concise, confident-sounding answer can still be critically flawed if it's not grounded in tool results.

### parallel-subagent (4.6/5) — Best Overall

Agent correctly delegates to two specialist subagents in parallel and merges results.

| Rubric | Score |
|--------|-------|
| SUBAGENT_DELEGATION | 🟢 5/5 |
| ROUTING_QUALITY | 🟢 5/5 |
| ANSWER_ACCURACY | 🟢 5/5 |
| DATA_ACCURACY | 🟢 5/5 |
| HALLUCINATION_DETECTION | 🟢 5/5 |
| TOOL_CONSISTENCY | 🟢 5/5 |
| CONCISENESS | 🟢 5/5 |

### error-recovery (4.2/5) — Validates Graceful Degradation

Agent encounters permission denied, requests access, retries, and succeeds.

| Rubric | Score |
|--------|-------|
| ERROR_RECOVERY | 🟢 5/5 |
| ANSWER_ACCURACY | 🟢 5/5 |
| DATA_ACCURACY | 🟢 5/5 |
| HALLUCINATION_DETECTION | 🟢 5/5 |
| TOOL_SELECTION | 🟢 5/5 |
| TOOL_CONSISTENCY | 🟢 5/5 |

### coding-session (3.4/5) — Longest Trace, Mixed Results

6-turn session with 9 tool calls. Strong on tool selection and coherence, weaker on data accuracy (condensed tool results) and action item extraction.

| Rubric | Avg Score |
|--------|-----------|
| CONVERSATION_COHERENCE | 🟢 5.0/5 |
| CONCISENESS | 🟢 4.8/5 |
| ERROR_RECOVERY | 🟢 4.7/5 |
| TOOL_SELECTION | 🟢 4.3/5 |
| TOOL_GROUNDEDNESS | 🟢 4.3/5 |
| TOOL_CONSISTENCY | 🟢 4.3/5 |
| ROUTING_QUALITY | 🟢 4.0/5 |
| ANSWER_ACCURACY | 🟡 3.7/5 |
| HALLUCINATION_DETECTION | 🟡 3.0/5 |
| TOOL_CALL_QUALITY | 🟡 3.0/5 |
| TOOL_CHAINING | 🔴 2.5/5 |
| DATA_ACCURACY | 🔴 2.3/5 |
| ACTION_ITEM_EXTRACTION | 🔴 1.8/5 |
| SUBAGENT_DELEGATION | 🔴 1.3/5 |

## How to Reproduce

```bash
cd agent-eval

# Run on a single trace
python -m agent_eval.cli \
  --input ui/example_traces/hallucination-risk.json \
  --judge-config test-fixtures/judges.real.single.yaml \
  --rubrics ui/example_traces/agent_rubrics.yaml \
  --output-dir ./output \
  --verbose

# View in dashboard
streamlit run ui/app.py
```

Requires AWS credentials with `bedrock:InvokeModel` permission and Claude Sonnet model access enabled in your region.

## Key Takeaways

1. **Rubrics differentiate meaningfully** — scores range from 1/5 to 5/5 within the same trace
2. **Hallucination detection works** — ungrounded answers score 1/5 on critical rubrics while grounded answers score 5/5
3. **Error recovery is measurable** — the framework correctly rewards graceful degradation
4. **Longer traces surface more issues** — the 6-turn coding session revealed weaknesses not visible in single-turn traces
5. **Real judges add value over mocks** — mock judges return uniform 3.0/5; real judges provide actionable differentiation
