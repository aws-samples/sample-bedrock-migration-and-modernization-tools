# Amazon Bedrock AgentCore Conversation Extractor

Extract complete conversation data from Amazon Bedrock AgentCore by merging APPLICATION_LOGS with GenAI Observability OTEL spans.

## 🚀 NEW: ARN-based Wrapper (Recommended)

**Simplify trace extraction with a single command!** The new wrapper automates ARN parsing, log group discovery, and pipeline orchestration.

### Quick Start

```bash
python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \
  --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1
```

**Benefits:**
- ✅ No manual log group discovery
- ✅ Single command execution
- ✅ Automatic error handling
- ✅ Discovery artifact for reproducibility

**See [WRAPPER_README.md](./WRAPPER_README.md) for complete documentation.**

---

## Manual 3-Script Approach (Legacy)

The original 3-script approach is still available for advanced use cases. Continue reading for manual execution instructions.

## 🔍 How to Find Tool Calls Programmatically

Tool calls and Knowledge Base operations are embedded in OTEL span messages. Here's how to extract them:

### Tool Call Flow in Spans

Tool calls appear in **OUTPUT** of one span, and tool results appear in **INPUT** of the next span:

```
Span N (OUTPUT):
  └─ toolUse: { name: "current_time", toolUseId: "tooluse_xyz", input: {} }

Span N+1 (INPUT):
  └─ toolResult: { toolUseId: "tooluse_xyz", status: "success", content: "2026-02-11T23:02:17Z" }
```

### Query Pattern

Use **CloudWatch Logs Insights API** to query OTEL spans:

```python
import boto3

client = boto3.client('logs', region_name='us-east-1')

# CloudWatch Logs Insights query
query = f'''
fields @timestamp, @message
| filter traceId = "{trace_id}"
| filter scope.name = "strands.telemetry.tracer"
| sort @timestamp asc
'''

# Execute query
response = client.start_query(
    logGroupName='/aws/bedrock-agentcore/runtimes/customersupportagent-crGhIpFJYP-DEFAULT',
    startTime=int(start_time.timestamp()),
    endTime=int(end_time.timestamp()),
    queryString=query
)

# Wait for results
query_id = response['queryId']
result = client.get_query_results(queryId=query_id)

# Parse messages to extract tool calls
for record in result['results']:
    message = json.loads(record['@message'])
    body = message['body']
    
    # Tool calls in OUTPUT
    if 'output' in body:
        for msg in body['output']['messages']:
            content = json.loads(msg['content']['content'])
            for item in content:
                if 'toolUse' in item:
                    tool_name = item['toolUse']['name']
                    tool_input = item['toolUse']['input']
    
    # Tool results in INPUT
    if 'input' in body:
        for msg in body['input']['messages']:
            content = json.loads(msg['content']['content'])
            for item in content:
                if 'toolResult' in item:
                    result = item['toolResult']['content']
```

### Example: `current_time` Tool

**Question**: "What time is it now?"

**Tool Call** (found in span OUTPUT):
```json
{
  "toolUse": {
    "name": "current_time",
    "toolUseId": "tooluse_8naYbQI24FLbKg86DLHmiV",
    "input": {}
  }
}
```

**Tool Result** (found in next span INPUT):
```json
{
  "toolResult": {
    "toolUseId": "tooluse_8naYbQI24FLbKg86DLHmiV",
    "status": "success",
    "content": [{"text": "2026-02-11T23:02:17.740309+00:00"}]
  }
}
```

### Knowledge Base Queries

Knowledge Base retrievals appear as implicit tool calls. Look for:
- Agent reasoning: "Let me search..." or "I'll retrieve..."
- Followed by detailed information in the response
- No explicit `toolUse` but content from KB in output

**See**: `guides/PROGRAMMATIC_TRACE_ACCESS.md` for complete implementation details.

---

## Architecture for Session JSON Output

### Data Sources (What to Use for Each Field)

| Field | Primary Source | Details |
|-------|---------------|---------|
| **run_id (session_id)** | ✅ Application Logs | `/aws/vendedlogs/bedrock-agentcore/runtime/APPLICATION_LOGS/...` |
| **timestamp** | ✅ Application Logs | `@timestamp` / `event_timestamp` for the user's turn |
| | ✅ Traces / Runtime Logs | For step timing (span start/end) |
| **user_query** | ✅ Application Logs | `body.request_payload.prompt` |
| **final_answer** | ✅ Runtime OTEL Log | `body.message.content.0.text` |
| | ✅ X-Ray Trace Segments | If answer is attached in span attributes (depends on instrumentation) |
| **total_latency_ms** | ✅ Best: Trace Spans | Compute from trace start → trace end |
| | Fallback: Runtime OTEL Logs | Min/max timestamps per traceId |
| **steps** | ✅ Best: CloudWatch Traces | X-Ray `batch_get_traces` and parse spans/subsegments |
| (AGENT_OPERATION, LLM_CALL, INVOCATION) | Fallback: Runtime OTEL Logs | If they include span events for each step (often incomplete vs trace service) |

### Join Keys

- **Session timeline**: `session_id` - Groups all turns in a conversation
- **Turn key**: `trace_id` - One per question/turn (e.g., 5 questions = 5 trace_ids)
- **Optional**: `request_id` and `span_id` - Help with debugging and deduplication

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER CONVERSATION                            │
│  Session: abc-123 (5 questions across multiple turns)          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   Turn 1: trace_id = xyz-1   │
              │   Turn 2: trace_id = xyz-2   │
              │   Turn 3: trace_id = xyz-3   │
              │   Turn 4: trace_id = xyz-4   │
              │   Turn 5: trace_id = xyz-5   │
              └──────────────┬───────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ APPLICATION    │  │ RUNTIME OTEL   │  │ TRACE SPANS    │
│ LOGS           │  │ LOGS           │  │ (X-Ray/CW)     │
├────────────────┤  ├────────────────┤  ├────────────────┤
│ ✅ session_id  │  │ ✅ trace_id    │  │ ✅ trace_id    │
│ ✅ trace_id    │  │ ✅ final_answer│  │ ✅ steps       │
│ ✅ timestamp   │  │    (body.msg)  │  │ ✅ latency     │
│ ✅ user_query  │  │ ✅ timestamps  │  │ ✅ operations  │
│    (prompt)    │  │                │  │ ✅ span timing │
└────────┬───────┘  └────────┬───────┘  └────────┬───────┘
         │                   │                   │
         │    JOIN BY trace_id (per turn)        │
         └───────────────────┼───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   MERGED SESSION JSON        │
              │                              │
              │  run_id: abc-123             │
              │  turns: [                    │
              │    {                         │
              │      trace_id: xyz-1,        │
              │      timestamp: ...,         │
              │      user_query: "...",      │
              │      final_answer: "...",    │
              │      latency_ms: 1234,       │
              │      steps: [...]            │
              │    },                        │
              │    { trace_id: xyz-2, ... }, │
              │    ...                       │
              │  ]                           │
              └──────────────────────────────┘
```

### Key Insights

1. **Application Logs are reliable** for session_id, prompt, trace_id, and timestamp
2. **Runtime OTEL logs** contain the actual model output (response_payload is null in Application Logs)
3. **Trace spans** provide the most complete operational steps and accurate latency
4. **trace_id is the join key** across all three sources for each turn
5. **session_id groups turns** into complete conversations

## How It Works - Detailed Explanation

### The Problem

When you invoke an AgentCore agent, the conversation data is split across multiple log sources:

1. **APPLICATION_LOGS** - Has session_id, prompt, trace_id, and timestamp (but response_payload is null)
2. **OTEL Runtime Logs** - Contains the actual model output (body.message.content[].text) keyed by the same traceId
3. **Trace Spans** - Contains detailed operational steps

### The Solution

The clean solution is to treat `trace_id` as the join key and build your JSON by querying:
- **(A) Application logs** - Get session_id, trace_id, event_timestamp, and prompt
- **(B) OTEL runtime logs** - Get the final answer using the trace_id
- **(C) Trace spans** - Get detailed steps/operations using the trace_id

Then merge all three in code.

### Understanding Session ID vs Trace ID

- **session_id (run_id)** - The conversation/thread identifier across multiple turns
- **trace_id** - Per-request/per-turn identifier (you'll have multiple traceIds inside one session)

### Pipeline Flow

1. Pull the last N prompts from Application logs → get `{session_id, trace_id, event_timestamp, prompt}`
2. For those traceIds, pull final answer from OTEL runtime logs
3. For those traceIds, pull steps/spans from CloudWatch Traces (Transaction Search / X-Ray backend) or from the aws/spans log group
4. Output JSON grouped by session_id

### Step-by-Step Data Flow

#### Step 1: Extract from APPLICATION_LOGS

**What we query:**
```
Log Group: /aws/vendedlogs/bedrock-agentcore/runtime/APPLICATION_LOGS/customersupportagent-crGhIpFJYP
```

**What we get:**
```json
{
  "session_id": "44b744ad-db29-418d-9e57-fd1107face44",
  "trace_id": "698cef883d471fd90a7d19ba655b5089",
  "event_timestamp": 1770844040972,
  "body": {
    "request_payload": {
      "prompt": "What are the warranty support guidelines?",
      "actor_id": "DEFAULT"
    },
    "response_payload": null  ← ALWAYS NULL!
  }
}
```

**Key takeaways:**
- ✅ We get: session_id, trace_id, timestamp, user's question
- ❌ We DON'T get: agent's response (always null)
- 🔑 We extract the trace_id to use in Step 2

#### Step 2: Query OTEL Spans using Trace IDs

**What we query:**
```
Log Group: /aws/bedrock-agentcore/runtimes/customersupportagent-crGhIpFJYP-DEFAULT
Filter: traceId = "698d0a700e10d77545948f32197c8f8d"
```

**What we get:**
```json
{
  "traceId": "698d0a700e10d77545948f32197c8f8d",
  "spanId": "49f98b7287d03240",
  "timeUnixNano": 1770850932888604639,
  "body": {
    "input": {
      "messages": [
        {
          "role": "user",
          "content": {
            "content": "[{\"text\": \"What are the warranty support guidelines?\"}]"
          }
        }
      ]
    },
    "output": {
      "messages": [
        {
          "role": "assistant",
          "content": {
            "message": "I'd be happy to help you with warranty information..."
          }
        }
      ]
    }
  }
}
```

**Key takeaways:**
- ✅ We get: BOTH input and output (complete conversation!)
- ✅ We get: Multiple spans showing operational steps
- 🔑 We match by trace_id from Step 1
- 📝 This is the SAME data you see in GenAI Observability console

#### Step 3: Merge the Data

**Matching logic:**
```
APPLICATION_LOG.trace_id == OTEL_SPAN.traceId
Example: "698d0a700e10d77545948f32197c8f8d"
```

**What we create:**
Complete conversation records with all 6 parameters plus detailed operational steps.

### Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                              │
│                                                                 │
│  "What are the warranty support guidelines?"                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AgentCore Runtime                            │
│                                                                 │
│  Generates:                                                     │
│  - session_id: 44b744ad-db29-418d-9e57-fd1107face44            │
│  - trace_id: 698cef883d471fd90a7d19ba655b5089                  │
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
               │                              │
               ▼                              ▼
┌──────────────────────────┐    ┌────────────────────────────────┐
│   APPLICATION_LOGS       │    │      OTEL Spans                │
│                          │    │                                │
│  ✅ session_id           │    │  ✅ traceId (matches!)         │
│  ✅ trace_id             │    │  ✅ body.input (user query)    │
│  ✅ timestamp            │    │  ✅ body.output (response)     │
│  ✅ request_payload      │    │  ✅ Complete conversation!     │
│  ❌ response_payload:null│    │  ✅ Operational steps          │
└──────────────┬───────────┘    └────────────┬───────────────────┘
               │                              │
               │    Match by trace_id         │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │   MERGED CONVERSATION        │
               │                              │
               │  All 6 Parameters:           │
               │  ✓ run_id                    │
               │  ✓ timestamp                 │
               │  ✓ user_query                │
               │  ✓ final_answer              │
               │  ✓ total_latency_ms          │
               │  ✓ steps (detailed)          │
               └──────────────────────────────┘
```

---

## Quick Start

### Three-Script Pipeline

```bash
# Script 1: Extract turns from Application Logs
export APP_LOG_GROUP="/aws/vendedlogs/bedrock-agentcore/runtime/APPLICATION_LOGS/customersupportagent-crGhIpFJYP"
python3 01_export_turns_from_app_logs.py --minutes 300 --out session_turns.json

# Script 2: Enrich with runtime OTEL data
export RUNTIME_LOG_GROUP="/aws/bedrock-agentcore/runtimes/customersupportagent-crGhIpFJYP-DEFAULT"
python3 02_enrich_from_runtime_otel_logs.py --turns session_turns.json --out session_enriched_runtime.json

# Script 3: Add X-Ray trace data and merge
python3 03_add_xray_steps_and_latency.py --index session_turns.json --detail session_enriched_runtime.json --out session_final.json
```

---

## Requirements

- Python 3.7+
- boto3
- AWS credentials configured with access to:
  - CloudWatch Logs
  - Amazon Bedrock AgentCore APPLICATION_LOGS
  - Amazon Bedrock AgentCore Runtime logs
  - X-Ray traces

---

## Data Sources

### APPLICATION_LOGS
- **Location**: `/aws/vendedlogs/bedrock-agentcore/runtime/APPLICATION_LOGS/customersupportagent-crGhIpFJYP`
- **Contains**: Request payloads, trace IDs, session IDs, timestamps
- **Missing**: Response payloads (always null)

### OTEL Spans
- **Location**: `/aws/bedrock-agentcore/runtimes/customersupportagent-crGhIpFJYP-DEFAULT`
- **Contains**: Complete conversation data with input/output in `body.input` and `body.output`
- **Contains**: Detailed operational steps (agent operations, LLM calls, tool calls)
- **Format**: OpenTelemetry structured logs

---

## Key Insights

### Why Two Sources?

1. **APPLICATION_LOGS** = Operational metadata
   - Fast to query
   - Has trace IDs for linking
   - Missing actual conversation content

2. **OTEL Spans** = Complete conversation data
   - Has full input/output
   - Has detailed operational steps
   - Requires trace IDs to find
   - This is what GenAI Observability console shows

### Why Trace IDs?

Trace IDs are the **linking key** between the two sources:
- Generated by AgentCore for each invocation
- Stored in both APPLICATION_LOGS and OTEL spans
- Allows us to match request metadata with conversation content

### Why Not Just Use OTEL Spans?

You could, but:
- Harder to query (need to know trace IDs first)
- APPLICATION_LOGS provides better session grouping
- Combining both gives you the complete picture

---

## Common Questions

**Q: Why is response_payload typically null in APPLICATION_LOGS?**  
A: By design. AgentCore doesn't log responses in APPLICATION_LOGS. You must use OTEL spans for responses.

**Q: Where does GenAI Observability console get its data?**  
A: From the same OTEL spans we're querying! The console just has a UI to browse them.

**Q: Can I get data without trace IDs?**  
A: Not easily. Trace IDs are the key to finding the right spans. That's why we start with APPLICATION_LOGS.

---

## Troubleshooting

### No spans found
- Verify the runtime ID is correct
- Check that observability is enabled for the agent
- helps the time range covers when conversations occurred

### Incomplete conversations (missing user_query or final_answer)
- The script searches based on the time window specified
- Increase the lookback window if needed
- Some traces may not have complete conversation data

### AWS Permissions
Required IAM permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [
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
  ]
}
```

---

## Summary

**Input:** Runtime ID and time window
**Process:** 
1. Get trace IDs from APPLICATION_LOGS
2. Find matching OTEL spans with conversation data
3. Merge by trace_id

**Output:** Complete conversation records with all 6 parameters including detailed operational steps

**Key Concept:** Trace IDs are the bridge between operational logs and conversation content.

**Relationship Mapping:** Use `trace_id` to group steps by conversation turn - all steps with the same trace_id processed the same question.
