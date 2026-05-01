# AgentCore Cost Insights

Calculate the per-invocation cost of running agents on Amazon Bedrock AgentCore, combining LLM inference cost with runtime compute, memory, and AgentCore Memory service costs.

> **Setup Guide**: See [OBSERVABILITY_SETUP.md](OBSERVABILITY_SETUP.md) for what to enable at each level to get exact vs estimated cost and latency data.

## Overview

The 💰 Cost tab provides a layered cost breakdown:

| Section | What it measures | Accuracy |
|---------|-----------------|----------|
| **🧾 Total Evaluated Run Cost** | Combined cost across all agents | Aggregate of below |
| **🤖 LLM Inference Cost** | Token-based model invocation cost | 🟢 Exact with CloudWatch Metrics, 🟡 estimated from trace |
| **🤖 Cost by Agent** | Per-agent LLM cost attribution | 🟢 Exact with X-Ray traces |
| **📡 CloudWatch Metrics** | Exact token counts from AWS | 🟢 Exact |
| **⚙️ AgentCore Compute** | vCPU and memory runtime cost | 🟡 Always estimated from duration × config |
| **🧠 Memory Service** | AgentCore Memory operation cost | 🟢 Exact from operation counts |

## Data Source Options (Setup → ☁️ AgentCore Config)

| Option | What it does | Best for |
|--------|-------------|----------|
| **📊 Estimate from trace data** | Derives cost from trace timestamps and step attributes | Quick analysis, no AWS access needed |
| **🔗 Fetch from CloudWatch Metrics** | Queries `AWS/Bedrock` for exact token counts and invocation data | Production cost validation |
| **📎 Upload usage logs** | Parse APPLICATION_LOGS for session correlation | Offline analysis |

## Resource Configuration

Select the resource tier matching your AgentCore agent deployment:

| Preset | vCPU | Memory |
|--------|------|--------|
| Small (default) | 1 | 512 MB |
| Medium | 2 | 1 GB |
| Large | 4 | 2 GB |
| X-Large | 8 | 4 GB |
| Custom | user-defined | user-defined |

Affects the compute cost estimate. When exact usage data is available, actual consumption values are used.

## Per-Agent Cost Attribution

For multi-agent systems, the tool breaks down LLM cost by agent. Each step in the normalized trace carries an optional `agent_id` field, populated from:

- X-Ray segment names (e.g., `SupervisorAgent_SupervisorAgent.DEFAULT`)
- `aws.local.service` annotation in X-Ray spans
- `agent_id` / `agent_name` fields in step attributes

The **🤖 Cost by Agent** table shows token usage and cost per agent, enabling you to identify which agents are the cost drivers.

**Requires**: X-Ray trace delivery enabled (Level 3 in [OBSERVABILITY_SETUP.md](OBSERVABILITY_SETUP.md)).

## AgentCore Memory Service Cost

Detected automatically from trace span operation names:

| Operation | Pricing | Detection |
|-----------|---------|-----------|
| Short-term memory events | $0.25 / 1,000 events | `CreateEvent` or `create_event` spans |
| Long-term memory retrieval | $0.50 / 1,000 retrievals | `RetrieveMemoryRecords` or `retrieve_memories` spans |

## Pricing Reference

| Component | Rate | Unit |
|-----------|------|------|
| vCPU | $0.0895 | per vCPU-hour |
| Memory | $0.00945 | per GB-hour |
| Memory events | $0.25 | per 1,000 events |
| Memory retrieval | $0.50 | per 1,000 retrievals |
| LLM inference | Varies by model | per 1,000 tokens |

Source: [Amazon Bedrock AgentCore pricing](https://aws.amazon.com/bedrock/agentcore/pricing/)

## Limitations

- **AgentCore compute**: Always estimated from session duration × resource config. Per-invocation vCPU/memory metering is not available from CloudWatch.
- **Pricing constants**: Hardcoded in `agentcore_cost.py`. Update `AGENTCORE_PRICING` if rates change.
- **Memory storage**: Monthly storage cost ($0.75/1,000 records/month) not included in per-invocation cost.
- **Built-in tools**: Browser and Code Interpreter costs not yet tracked.

## Demo Data

| File | Description |
|------|-------------|
| `test-fixtures/baseline/agentcore_cost_demo.json` | Single-agent trace with memory operations |
| `ui/agentcore_demo_output/` | Single-agent dashboard output |
| `ui/multi_agent_demo_output/` | Multi-agent (SupervisorAgent → MathAgent) dashboard output with per-agent cost |
