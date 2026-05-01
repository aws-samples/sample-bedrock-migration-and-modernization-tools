# Observability Setup Guide

What to enable for cost and latency insights in the Agent Evaluation Dashboard.

## Quick Reference — What You Get at Each Level

| Setup Level | What to enable | Cost data | Latency data | Effort |
|---|---|---|---|---|
| **Level 0: None** | Nothing — just load a trace | 🟡 Estimated LLM cost from trace tokens · 🟡 Estimated compute from duration | 🟡 Per-turn totals from trace timestamps | Zero |
| **Level 1: OTEL instrumentation** | ADOT SDK in agent code | 🟢 Exact LLM tokens in trace spans · 🟡 Estimated compute | 🟢 Per-step LLM/tool/overhead breakdown | Add 2 lines to requirements.txt |
| **Level 2: CloudWatch Metrics** | Level 1 + Transaction Search | 🟢 Exact tokens via CloudWatch API · 🟢 Exact memory service ops | 🟢 Per-step breakdown + TTFT + tokens/sec | One-time account setup |
| **Level 3: X-Ray Traces** | Level 2 + trace delivery | All of Level 2 + 🟢 Per-agent cost attribution | 🟢 Real per-step durations from spans | Enable trace delivery per agent |
| **Level 4: Application Logs** | Level 3 + log delivery | All of Level 3 + session/request correlation | All of Level 3 + request-level detail | Enable log delivery per agent |

Most users should target **Level 2** for accurate cost data or **Level 3** for multi-agent visibility.

---

## Level 0: Trace-Only (No Setup Required)

Load any normalized trace file into the dashboard. The tool derives what it can from timestamps and step attributes.

**What works:**
- LLM cost — if trace steps contain `prompt_tokens` / `completion_tokens` in attributes
- Compute cost — estimated from first-to-last span duration × resource config
- Memory service cost — if trace contains `CreateEvent` / `RetrieveMemoryRecords` step names
- Latency — per-turn totals from step timestamps

**What's missing:**
- No CloudWatch validation of token counts
- No per-step latency breakdown (LLM vs tool vs overhead)
- No per-agent attribution
- Compute cost is a rough estimate

---

## Level 1: OTEL Instrumentation

Add the AWS Distro for OpenTelemetry (ADOT) SDK to your agent code. This is already included in agents created with the AgentCore CLI.

### Setup

Add to your `requirements.txt`:
```
aws-opentelemetry-distro>=0.10.0
boto3
```

Run your agent with auto-instrumentation:
```bash
opentelemetry-instrument python main.py
```

For containerized agents:
```dockerfile
CMD ["opentelemetry-instrument", "python", "main.py"]
```

### What this enables
- `gen_ai.client.token.usage` metrics with input/output token counts per model
- `gen_ai.client.operation.duration` for LLM call timing
- `strands.tool.duration` for tool call timing (Strands framework)
- Span-level timestamps for per-step latency breakdown

### Accuracy impact

| Component | Before | After |
|---|---|---|
| LLM token counts | From trace attributes (if present) | Emitted by OTEL instrumentation |
| Per-step latency | Turn-level only | LLM vs tool vs overhead breakdown |

---

## Level 2: CloudWatch Metrics (Recommended)

Enable CloudWatch Transaction Search so OTEL metrics and Bedrock invocation metrics are queryable via the CloudWatch Metrics API.

### Setup

**Option A: Console**
1. Open [CloudWatch console](https://console.aws.amazon.com/cloudwatch)
2. Navigate to **Application Signals (APM)** → **Transaction search**
3. Choose **Enable Transaction Search**
4. Select the checkbox to ingest spans as structured logs
5. Choose **Save**

**Option B: CLI**
```bash
# 1. Add resource policy for X-Ray → CloudWatch Logs
aws logs put-resource-policy \
  --policy-name TransactionSearchXRayAccess \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Sid": "TransactionSearchXRayAccess",
      "Effect": "Allow",
      "Principal": {"Service": "xray.amazonaws.com"},
      "Action": "logs:PutLogEvents",
      "Resource": [
        "arn:aws:logs:REGION:ACCOUNT_ID:log-group:aws/spans:*",
        "arn:aws:logs:REGION:ACCOUNT_ID:log-group:/aws/application-signals/data:*"
      ],
      "Condition": {
        "ArnLike": {"aws:SourceArn": "arn:aws:xray:REGION:ACCOUNT_ID:*"},
        "StringEquals": {"aws:SourceAccount": "ACCOUNT_ID"}
      }
    }]
  }'

# 2. Set trace destination to CloudWatch Logs
aws xray update-trace-segment-destination --destination CloudWatchLogs
```

### What this enables
- **`AWS/Bedrock` namespace**: `InputTokenCount`, `OutputTokenCount`, `Invocations`, `InvocationLatency` per ModelId
- **`bedrock-agentcore` namespace**: `gen_ai.client.token.usage`, `http.server.duration`, `strands.event_loop.*`

### Using in the dashboard
1. Go to **Setup → ☁️ AgentCore Config**
2. Select **🔗 Fetch from CloudWatch Metrics (exact)**
3. Enter your AWS region and optionally filter by model ID
4. Click **🔗 Fetch Metrics**

### IAM permissions required
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "cloudwatch:GetMetricData",
      "cloudwatch:ListMetrics"
    ],
    "Resource": "*"
  }]
}
```

### Accuracy impact

| Component | Before | After |
|---|---|---|
| LLM token counts | From trace attributes | 🟢 Exact from `AWS/Bedrock` CloudWatch metrics |
| LLM token cost | Tokens × LiteLLM pricing | 🟢 Exact tokens × pricing |
| Invocation count | Inferred from trace turns | 🟢 Exact from CloudWatch |
| AgentCore compute | 🟡 Still estimated from trace duration | 🟡 Still estimated (no per-invocation metering available) |

---

## Level 3: X-Ray Trace Delivery

Enable trace delivery from your AgentCore runtime to X-Ray. This provides real per-step durations and agent identity for multi-agent cost attribution.

### Setup

**Option A: Console**
1. Open [AgentCore console → Agent Runtime](https://console.aws.amazon.com/bedrock-agentcore/agents)
2. Select your agent
3. In the **Tracing** pane, choose **Edit** → toggle to **Enable** → **Save**

**Option B: CloudWatch Logs Delivery API**
```bash
RUNTIME_ARN="arn:aws:bedrock-agentcore:REGION:ACCOUNT_ID:runtime/YOUR_RUNTIME_ID"

# 1. Create trace delivery source
aws logs put-delivery-source \
  --name "my-agent-traces-source" \
  --log-type "TRACES" \
  --resource-arn "$RUNTIME_ARN"

# 2. Create X-Ray delivery destination
aws logs put-delivery-destination \
  --name "my-agent-traces-dest" \
  --delivery-destination-type "XRAY"

# 3. Connect them
aws logs create-delivery \
  --delivery-source-name "my-agent-traces-source" \
  --delivery-destination-arn "arn:aws:logs:REGION:ACCOUNT_ID:delivery-destination:my-agent-traces-dest"
```

### What this enables
- X-Ray segments with real durations for `ConverseStream`, `CountTokens`, memory operations
- Agent identity via segment name (e.g., `SupervisorAgent_SupervisorAgent.DEFAULT`)
- Parent-child span relationships showing which agent called which

### Accuracy impact

| Component | Before | After |
|---|---|---|
| Per-step latency | Estimated from trace timestamps | 🟢 Real durations from X-Ray spans |
| Per-agent cost | Not available | 🟢 Cost attributed by agent via `agent_id` from X-Ray segment names |
| Tokens/sec | Estimated | 🟢 Exact tokens ÷ real LLM duration |
| Slowest steps | From trace order | 🟢 Ranked by real X-Ray span duration |

---

## Level 4: Application Log Delivery

Enable APPLICATION_LOGS delivery for per-invocation request/response correlation and detailed session tracking.

### Setup

**Option A: Console**
1. Open [AgentCore console → Agent Runtime](https://console.aws.amazon.com/bedrock-agentcore/agents)
2. Select your agent
3. In the **Log delivery** pane, choose **Add** → **CloudWatch Logs**
4. Set **Log type** to `APPLICATION_LOGS`
5. Choose **Add**

**Option B: CloudWatch Logs Delivery API**
```bash
RUNTIME_ARN="arn:aws:bedrock-agentcore:REGION:ACCOUNT_ID:runtime/YOUR_RUNTIME_ID"
LOG_GROUP="/aws/vendedlogs/bedrock-agentcore/runtimes/YOUR_RUNTIME_ID"

# 1. Create log group
aws logs create-log-group --log-group-name "$LOG_GROUP"

# 2. Create delivery source
aws logs put-delivery-source \
  --name "my-agent-logs-source" \
  --log-type "APPLICATION_LOGS" \
  --resource-arn "$RUNTIME_ARN"

# 3. Create delivery destination
aws logs put-delivery-destination \
  --name "my-agent-logs-dest" \
  --delivery-destination-type "CWL" \
  --delivery-destination-configuration "{\"destinationResourceArn\": \"arn:aws:logs:REGION:ACCOUNT_ID:log-group:$LOG_GROUP\"}"

# 4. Connect them
aws logs create-delivery \
  --delivery-source-name "my-agent-logs-source" \
  --delivery-destination-arn "arn:aws:logs:REGION:ACCOUNT_ID:delivery-destination:my-agent-logs-dest"
```

### What this enables
- Per-invocation records with `request_id`, `session_id`, `trace_id`, `span_id`
- Request/response payloads
- Correlation between invocations and X-Ray traces

### IAM permissions required (for fetching logs)
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "logs:FilterLogEvents",
      "logs:GetLogEvents",
      "logs:DescribeLogGroups"
    ],
    "Resource": "arn:aws:logs:*:*:log-group:/aws/vendedlogs/bedrock-agentcore/*"
  }]
}
```

---

## Cost Accuracy Summary

| Cost Component | Level 0 | Level 1 | Level 2 | Level 3+ |
|---|---|---|---|---|
| **LLM token counts** | 🟡 From trace attrs | 🟡 From OTEL spans | 🟢 Exact (CloudWatch) | 🟢 Exact |
| **LLM token cost** | 🟡 Tokens × LiteLLM | 🟡 Tokens × LiteLLM | 🟢 Exact tokens × pricing | 🟢 Exact |
| **AgentCore compute** | 🟡 Duration × config | 🟡 Duration × config | 🟡 Duration × config | 🟡 Duration × config |
| **Memory service** | 🟢 Op counts × rates | 🟢 Op counts × rates | 🟢 Op counts × rates | 🟢 Op counts × rates |
| **Per-agent breakdown** | ❌ | ❌ | ❌ | 🟢 From X-Ray segments |

> **Note**: AgentCore compute cost (vCPU-hours, GB-hours) is always estimated from session duration × resource configuration. Per-invocation resource metering is not currently available as a CloudWatch metric or vended log field.

## Latency Accuracy Summary

| Latency Component | Level 0 | Level 1 | Level 2 | Level 3+ |
|---|---|---|---|---|
| **Total session latency** | 🟡 From trace timestamps | 🟢 From OTEL spans | 🟢 From OTEL spans | 🟢 From X-Ray |
| **Per-turn latency** | 🟡 From trace timestamps | 🟢 From OTEL spans | 🟢 From OTEL spans | 🟢 From X-Ray |
| **LLM vs Tool vs Overhead** | ❌ | 🟢 From step kind classification | 🟢 From step kind | 🟢 Real durations |
| **TTFT** | ❌ | 🟢 First LLM response time | 🟢 First LLM response | 🟢 Real timing |
| **Tokens/sec** | ❌ | 🟡 Estimated | 🟢 Exact tokens ÷ duration | 🟢 Exact |
| **Slowest steps** | ❌ | 🟡 From trace order | 🟡 From trace order | 🟢 Real X-Ray durations |

## Pricing Reference

| Component | Rate | Unit | Source |
|---|---|---|---|
| vCPU | $0.0895 | per vCPU-hour | [AgentCore pricing](https://aws.amazon.com/bedrock/agentcore/pricing/) |
| Memory | $0.00945 | per GB-hour | [AgentCore pricing](https://aws.amazon.com/bedrock/agentcore/pricing/) |
| Memory events | $0.25 | per 1,000 events | [AgentCore pricing](https://aws.amazon.com/bedrock/agentcore/pricing/) |
| Memory retrieval | $0.50 | per 1,000 retrievals | [AgentCore pricing](https://aws.amazon.com/bedrock/agentcore/pricing/) |
| LLM inference | Varies by model | per 1,000 tokens | [Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) |

## Known Limitations

- **AgentCore compute metering**: No per-invocation vCPU/memory consumption metric exists. Compute cost is always estimated from session duration.
- **Pricing updates**: Pricing constants are hardcoded in `agentcore_cost.py`. Update `AGENTCORE_PRICING` if rates change.
- **Memory storage**: Monthly storage cost ($0.75/1,000 records/month) is not included in per-invocation cost.
- **Browser and Code Interpreter**: AgentCore built-in tool costs are not yet tracked.
- **CloudWatch metric delay**: Custom OTEL metrics may take 1–5 minutes to appear in CloudWatch after invocation.
