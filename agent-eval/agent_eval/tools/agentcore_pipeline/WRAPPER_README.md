# AgentCore ARN-based Trace Extraction Wrapper

A thin wrapper orchestrator that simplifies extracting AgentCore traces from CloudWatch. Users provide only an AgentCore Runtime ARN and optional parameters, and the system handles all discovery and orchestration automatically.

## Overview

This wrapper automates the complete trace extraction workflow:
1. **Parse ARN** - Validates and extracts components from AgentCore Runtime ARN
2. **Fetch Runtime Metadata** - Calls bedrock-agentcore-control API to get authoritative runtime ID
3. **Discover Log Groups** - Automatically finds runtime and OTEL log groups in CloudWatch
4. **Execute Pipeline** - Orchestrates the 3-script pipeline to extract and merge traces
5. **Write Outputs** - Creates organized output directory with all artifacts

## Quick Start

### Basic Usage

Extract traces using only an ARN:

```bash
python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \
  --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1 \
  --region us-east-1
```

### With Specific Session

Filter traces for a specific session:

```bash
python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \
  --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1 \
  --session-id 44b744ad-db29-418d-9e57-fd1107face44 \
  --minutes 180
```

### With Custom Time Window

Adjust the lookback window:

```bash
python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \
  --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1 \
  --minutes 300 \
  --pad-seconds 7200
```

### With AWS Profile

Use a specific AWS profile:

```bash
python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \
  --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1 \
  --profile myprofile \
  --debug
```

## CLI Reference

### Required Arguments

- `--agent-runtime-arn` - AgentCore Runtime ARN
  - Format: `arn:aws:bedrock-agentcore:<region>:<account-id>:<resource-id>`
  - Example: `arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1`

### Optional Arguments

- `--region` - AWS region (default: `us-east-1`)
- `--profile` - AWS profile name for credential selection
- `--session-id` - Specific session ID to extract (extracts all sessions if not specified)
- `--minutes` - Lookback window in minutes for CloudWatch queries (default: `180`)
- `--output-dir` - Output directory path (default: `./out/run-<timestamp>`)
- `--pad-seconds` - Time padding in seconds for CloudWatch queries (default: `7200`)
- `--log-stream-kind` - Log stream kind for Script 1 (default: `runtime`, choices: `runtime`/`application`)
- `--debug` - Enable debug output in scripts

### Escape Hatch Arguments

Use these when automatic log group discovery fails or you have custom log group names:

- `--runtime-log-group` - Override runtime log group discovery with explicit log group name
- `--otel-log-group` - Override OTEL log group discovery with explicit log group name

**Note:** Both escape hatch arguments must be used together. When both are provided, the wrapper skips the GetAgentRuntime API call and log group discovery entirely.

Example with escape hatches:

```bash
python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \
  --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1 \
  --runtime-log-group /custom/runtime/logs \
  --otel-log-group /custom/otel/logs
```

## Output Files

The wrapper produces 4 output files in the specified output directory:

1. **discovery.json** - ARN parsing and log group discovery results
   - Contains parsed ARN components
   - Runtime metadata from GetAgentRuntime API
   - Discovered log group names
   - Configuration used for execution

2. **01_session_turns.json** - Script 1 output
   - Extracted turns from runtime logs
   - Contains run_id, window, and turns array

3. **02_session_enriched_runtime.json** - Script 2 output
   - OTEL enrichment data
   - Contains sessions and enrich_stats

4. **03_turns_merged_normalized.json** - Script 3 final output
   - Merged and normalized traces
   - Contains turns_merged_normalized and merge_stats

## Programmatic API

### Basic Usage

```python
from agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn import (
    parse_agentcore_arn,
    get_runtime_metadata,
    discover_log_groups,
    execute_pipeline,
    PipelineConfig
)

# Parse ARN
arn = parse_agentcore_arn(
    "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1"
)

# Get runtime metadata
runtime = get_runtime_metadata(arn, region="us-east-1")

# Discover log groups
log_groups = discover_log_groups(
    runtime.agent_runtime_id,
    region="us-east-1"
)

# Execute pipeline
config = PipelineConfig(
    arn=arn,
    runtime=runtime,
    log_groups=log_groups,
    region="us-east-1",
    profile=None,
    session_id=None,
    minutes=180,
    output_dir="./out/run-001",
    pad_seconds=7200,
    debug=False,
    log_stream_kind="runtime"
)

result = execute_pipeline(config)

if result.success:
    print(f"Pipeline completed successfully!")
    print(f"Final output: {result.script3_output}")
else:
    print(f"Pipeline failed: {result.error}")
```

### With Error Handling

```python
from agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn import (
    parse_agentcore_arn,
    get_runtime_metadata,
    discover_log_groups,
    RuntimeNotFoundError,
    LogGroupNotFoundError
)

try:
    arn = parse_agentcore_arn(arn_string)
    runtime = get_runtime_metadata(arn, region="us-east-1")
    log_groups = discover_log_groups(runtime.agent_runtime_id, region="us-east-1")
except ValueError as e:
    print(f"Invalid ARN: {e}")
except RuntimeNotFoundError as e:
    print(f"Runtime not found: {e}")
except LogGroupNotFoundError as e:
    print(f"Log groups not found: {e}")
except RuntimeError as e:
    print(f"AWS credentials error: {e}")
```

### Using Escape Hatches

```python
# Bypass log group discovery with explicit log group names
log_groups = discover_log_groups(
    agent_runtime_id="runtime-123",
    region="us-east-1",
    runtime_override="/custom/runtime/logs",
    otel_override="/custom/otel/logs"
)
# Returns immediately without API calls
```

## Error Handling

### Invalid ARN Format

**Symptom:** `ValueError: Invalid ARN prefix`

**Solution:** Verify ARN format matches:
```
arn:aws:bedrock-agentcore:<region>:<account-id>:<resource-id>
```

### Runtime Not Found

**Symptom:** `RuntimeNotFoundError: Runtime not found`

**Solution:**
- Verify ARN is correct
- Check runtime exists in specified region
- helps you have permission to access the runtime

### Log Groups Not Found

**Symptom:** `LogGroupNotFoundError: Runtime logs not found`

**Solution:**
- Verify runtime has logging enabled in AgentCore configuration
- Check runtime has been invoked at least once
- Use escape hatch arguments if using custom log group names:
  ```bash
  --runtime-log-group /custom/runtime/logs \
  --otel-log-group /custom/otel/logs
  ```

### AWS Credentials Missing

**Symptom:** `RuntimeError: AWS credentials not found`

**Solution:** Configure credentials using one of:
1. AWS CLI: `aws configure --profile <profile>`
2. Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
3. AWS credentials file: `~/.aws/credentials`
4. IAM role (if running on EC2/Lambda)

### Script Execution Failure

**Symptom:** `ScriptExecutionError: Script 1 failed`

**Solution:**
- Check error output for specific script error
- Verify CloudWatch Logs permissions
- Try increasing `--minutes` if no data found
- Enable `--debug` for detailed output

### Empty Turns Array

**Symptom:** `ValueError: Script 1 output field 'turns' is empty`

**Solution:**
- Increase `--minutes` to expand time window
- Verify `--session-id` is correct (if specified)
- Check runtime has activity in the specified time window
- Try different `--log-stream-kind` (runtime vs application)

## Troubleshooting

### When to Use Escape Hatches

Use `--runtime-log-group` and `--otel-log-group` when:
- Automatic log group discovery fails
- Using custom log group names
- Log groups don't follow standard naming pattern
- Debugging log group discovery issues
- You already know the exact log group names

### Debugging Tips

1. **Enable debug mode:**
   ```bash
   --debug
   ```

2. **Check discovery.json:**
   - Verify ARN parsing is correct
   - Check runtime metadata
   - Confirm log group names

3. **Verify IAM permissions:**
   - `bedrock-agentcore-control:GetAgentRuntime`
   - `logs:DescribeLogGroups`
   - `logs:StartQuery`
   - `logs:GetQueryResults`

4. **Test with escape hatches:**
   - Bypass discovery to isolate issues
   - Verify log group names manually in CloudWatch console

### Common Issues

**Issue:** Pipeline hangs during CloudWatch query

**Solution:** CloudWatch Insights queries can take time. Wait for completion or check CloudWatch console for query status.

---

**Issue:** "No data found" in output

**Solution:** 
- Increase `--minutes` value
- Verify runtime has been invoked in time window
- Check `--session-id` is correct

---

**Issue:** Permission denied errors

**Solution:** Verify IAM permissions listed above. Test with AWS CLI:
```bash
aws bedrock-agentcore-control get-agent-runtime --agent-runtime-arn <arn>
aws logs describe-log-groups --log-group-name-prefix /aws/bedrock-agentcore/
```

## Required IAM Permissions

Minimum IAM policy required:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore-control:GetAgentRuntime",
        "logs:DescribeLogGroups",
        "logs:StartQuery",
        "logs:GetQueryResults"
      ],
      "Resource": "*"
    }
  ]
}
```

## Implementation Notes

### MVP Approach: Subprocess Orchestration

**Current Implementation:** The wrapper uses `subprocess.run` to execute the 3 existing scripts. This is an MVP approach chosen for:
- Rapid development
- No modifications to existing scripts
- Clear separation of concerns

**Limitations:**
- Subprocess overhead
- Limited error context from scripts
- No shared state between scripts

**Future Refactoring:** Consider direct module imports for:
- Better error handling
- Improved performance
- Shared configuration
- Unified logging

This is marked for future enhancement but does not impact functionality.

### Log Stream Kind

The `--log-stream-kind` argument controls which query template Script 1 uses:
- `runtime` (default): Direct AgentCore runtime logs
- `application`: APPLICATION_LOGS filter (legacy)

Default changed to `runtime` for better compatibility with direct ARN-based extraction.

## Comparison: Manual vs Wrapper Approach

### Manual Approach (3 separate commands)

```bash
# Step 1: Manually find log groups in CloudWatch console
# Step 2: Run Script 1
python -m agent_eval.tools.agentcore_pipeline.01_export_turns_from_app_logs \
  --log-group /aws/bedrock-agentcore/runtimes/abc-123-endpoint/runtime-logs \
  --minutes 180 --output 01_turns.json

# Step 3: Run Script 2
python -m agent_eval.tools.agentcore_pipeline.02_build_session_trace_index \
  --turns 01_turns.json \
  --otel-log-group /aws/bedrock-agentcore/runtimes/abc-123-endpoint/otel-rt-logs \
  --output 02_enriched.json

# Step 4: Run Script 3
python -m agent_eval.tools.agentcore_pipeline.03_add_xray_steps_and_latency \
  --index 01_turns.json --detail 02_enriched.json --output 03_merged.json
```

### Wrapper Approach (1 command)

```bash
python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \
  --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc-def-123:v1
```

**Benefits:**
- No manual log group discovery
- Single command execution
- Automatic error handling
- Discovery artifact for reproducibility
- Consistent output organization

## Migration Guide

### For Existing Users

If you're currently using the 3-script approach manually:

1. **Find your ARN:** Get the AgentCore Runtime ARN from the AWS console or API
2. **Run wrapper:** Use the wrapper with your ARN
3. **Compare outputs:** The final output format is identical

### Backward Compatibility

The wrapper does not modify the existing 3 scripts. You can continue using them manually if needed. The wrapper simply orchestrates their execution.

## Examples

### Extract Last 3 Hours of Traces

```bash
python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \
  --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc:v1 \
  --minutes 180
```

### Extract Specific Session from Last 24 Hours

```bash
python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \
  --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc:v1 \
  --session-id 44b744ad-db29-418d-9e57-fd1107face44 \
  --minutes 1440
```

### Extract with Custom Output Directory

```bash
python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \
  --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc:v1 \
  --output-dir ./traces/production/2024-01-15
```

### Extract Using Different AWS Profile

```bash
python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \
  --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc:v1 \
  --profile production-readonly
```

### Extract with Escape Hatches (Custom Log Groups)

```bash
python -m agent_eval.tools.agentcore_pipeline.run_from_agentcore_arn \
  --agent-runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:agent/abc:v1 \
  --runtime-log-group /custom/agentcore/runtime-logs \
  --otel-log-group /custom/agentcore/otel-logs
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review error messages for specific guidance
3. Enable `--debug` for detailed output
4. Check discovery.json for configuration details
