# Sample Trace Fixtures

This directory contains sample trace files for testing the Generic JSON adapter. Each trace is designed to test specific edge cases and functionality.

## Trace Files

### 1. `trace_single_turn_success.json`
**Purpose**: Test basic single-turn conversation with successful tool call

**Features**:
- Single turn with explicit turn markers
- Complete tool call lifecycle (invocation → result)
- All events have valid timestamps
- Successful status for all steps
- Includes span hierarchy (parent_span_id)
- **Turn end marker**: Explicit `turn_end` event with `finish_reason`
- **Message linking**: `message_id` and `conversation_id` for robust user→model→assistant linking
- **Tool attributes**: `provider`, `tool_latency_ms`, `status_code`, `latency_ms` in tool events
- **Conversation window**: Top-level `conversation_window_start` and `conversation_window_end`
- Top-level fields: run_id, final_answer, user_query, total_latency_ms

**Expected Behavior**:
- Should normalize successfully with 100% confidence
- Should calculate normalized_latency_ms from timestamps
- Should preserve runtime_reported_latency_ms
- Should link tool call with tool result via tool_run_id
- Should extract final_answer from turn_end event attributes (unambiguous vs streaming chunks)
- Should use tool_latency_ms when timestamps are unreliable

---

### 2. `trace_multi_turn.json`
**Purpose**: Test multi-turn conversation with explicit turn_ids

**Features**:
- Three distinct turns with explicit turn_id and request_id fields
- Turn 1: Simple Q&A without tools
- Turn 2: Q&A with tool call (search_api)
- Turn 3: Simple acknowledgment
- All events properly tagged with turn_id
- **Turn end markers**: Each turn has explicit `turn_end` event with `finish_reason` and `final_answer` in attributes
- **Complete lifecycle**: Turn 3 includes `model_invoke` event (not just output) to avoid "missing model call" penalties
- **Message linking**: `message_id` and `conversation_id` on all events for robust linking when turn_id absent
- **Tool attributes**: Tool events include `provider`, `tool_latency_ms`, `status_code`, `latency_ms`, `result_count`
- **Per-turn latency**: Each `turn_end` event includes `turn_latency_ms` for cross-check with timestamp-based calculation
- **Conversation window**: Top-level attributes include `conversation_window_start`, `conversation_window_end`, `total_conversation_latency_ms`, `turn_count`

**Expected Behavior**:
- Should segment into 3 separate turns using TURN_ID strategy
- Each turn should have its own confidence score
- Should calculate per-turn latencies and validate against turn_latency_ms
- Should handle tool linking within turn 2
- Should extract final_answer from turn_end attributes (unambiguous)
- Should not penalize turn 3 for missing model invoke (now present)

---

### 3. `trace_stitched.json`
**Purpose**: Test stitched multi-turn trace detection (suspect detection) and recovery

**Features**:
- Three distinct user questions with different content
- All events share the same request_id (improperly stitched)
- **Recovery mechanisms**: Each Q&A has unique `turn_id` and `message_id` for segmentation recovery
- **Turn end markers**: Each turn has explicit `turn_end` event with `finish_reason` and `final_answer` in attributes
- **Model call linking**: `model_call_id` on model_invoke and output chunks for robust linking (doesn't rely on parent_span_id)
- **Per-turn windows**: Each turn has `window_start` and `window_end` in attributes, plus `turn_latency_ms`
- **Model latency**: `model_latency_ms` in model_invoke attributes to avoid relying purely on timestamps
- **Stitch diagnostics**: Top-level `stitch_diagnostics` with deterministic fields:
  - `distinct_user_prompts`: 3 (triggers suspect detection)
  - `shared_request_id`: "req-shared-001" (the reused ID)
  - `request_id_reuse_detected`: true
  - `distinct_turn_ids`: array of turn IDs
  - `distinct_message_ids`: count of unique message IDs
  - `segmentation_hint`: guidance for adapter
- **Conversation window**: Top-level `conversation_window_start`, `conversation_window_end`, `total_conversation_latency_ms`

**Expected Behavior**:
- Should detect multiple distinct questions per request_id using stitch_diagnostics
- Should apply confidence penalty for stitched trace suspect (deterministic calculation)
- Should segment into 3 separate turns using turn_id (preferred) or message_id (fallback)
- Should NOT rely solely on request_id for segmentation (it's unreliable)
- Should extract final_answer from turn_end attributes per turn (unambiguous)
- Should link output chunks to model_invoke via model_call_id
- Should use turn_latency_ms for cross-check when timestamps are skewed
- Should emit segmentation_strategy_reason explaining detection and recovery

---

### 4. `trace_with_failure.json`
**Purpose**: Test handling of failed steps

**Features**:
- Tool call that fails with error status
- Error details in tool_result
- Error attributes (error_type, retry_after, status_code)
- Agent gracefully handles failure in final_answer
- **Turn end marker**: `turn_end` event with `finish_reason: "error"` and `error_occurred: true`
- **Message linking**: `message_id` and `conversation_id` for complete lifecycle tracking
- **Tool attributes**: Both tool_invocation and tool_result include `provider`, `tool_latency_ms`, `status_code`, `latency_ms`
- **Conversation window**: Top-level window markers for temporal bounds

**Expected Behavior**:
- Should preserve error status in normalized output
- Should include error attributes
- Should calculate latency including failed step
- Should not penalize confidence for legitimate errors (error status is valid)
- Should extract final_answer from turn_end attributes
- Should recognize finish_reason: "error" as valid completion

---

### 5. `trace_minimal_fields.json`
**Purpose**: Test minimal required fields with missing optionals and robust fallbacks

**Features**:
- Only essential fields present for basic trace processing
- No tool calls, no span hierarchy, no complex metadata
- **Deterministic event IDs**: Each event has unique `event_id` for stable linking and deduplication
- **Role on all events**: Both user and assistant events include `role` field for classification without relying solely on event_type
- **Dual timestamp formats**: Both ISO 8601 string (`timestamp`) and epoch milliseconds (`timestamp_ms`) for flexible parsing
- **Segmentation fields**: Includes `trace_id`, `request_id`, `turn_id` (even when null) to support segmentation beyond timestamps
- **Sequence numbers**: Monotonic `sequence` integers (1, 2, ...) as tie-breaker when timestamps collide or are missing
- **UTC timestamps**: All timestamps use Z suffix for unambiguous UTC
- **Canonical field naming**: Uses `event_type` as primary field (adapter can map to `kind` via config)
- No turn_id, request_id values (set to null explicitly)
- No final_answer or user_query at top level

**Expected Behavior**:
- Should normalize successfully with confidence penalties for missing optionals
- Should use SINGLE_TURN fallback strategy (no turn_id available)
- Should extract user_query and final_answer from event text
- Should handle missing optional fields gracefully (null values)
- Should use event_id for deterministic deduplication
- Should classify events using both event_type and role
- Should parse timestamps from either ISO 8601 or epoch_ms format
- Should use sequence numbers for ordering when timestamps are identical/missing
- Should recognize null request_id/turn_id as explicit absence (not missing field)

---

### 6. `trace_orphan_tool_results.json`
**Purpose**: Test handling of orphan tool results (results without corresponding calls)

**Features**:
- Two tool_result events without matching tool_call events
- First orphan: no tool_run_id and no matching invocation
- Second orphan: has tool_run_id but no matching call
- **Turn end marker**: Explicit `turn_end` event with `orphan_tool_results_detected` count
- **Message linking**: `message_id`, `conversation_id`, and `model_call_id` for complete lifecycle
- **Tool attributes**: Tool results include `status_code`, `provider`, `latency_ms`, `result_count`
- **Model latency**: `model_latency_ms` in model_invoke for fallback when timestamps unreliable
- **Orphan diagnostics**: Top-level `orphan_diagnostics` with deterministic fields:
  - `orphan_tool_results_count`: 2
  - `orphan_details`: array with tool_name, tool_run_id (if present), and reason
- **Conversation window**: Top-level window markers for temporal bounds

**Expected Behavior**:
- Should handle orphan results without crashing
- Should apply confidence penalty for orphan results (deterministic calculation from orphan_diagnostics)
- Should store orphan_tool_results in adapter_stats with details
- Should not fabricate tool calls to match orphans
- Should extract final_answer from turn_end attributes
- Should use orphan_diagnostics for deterministic penalty calculation

---

### 7. `trace_otel_format.json`
**Purpose**: Test OpenTelemetry OTLP/JSON canonical format with proper semantic conventions

**Features**:
- **Canonical OTLP/JSON structure**: OpenTelemetry resourceSpans format
- **Array-based attributes**: `resource.attributes` and `span.attributes` use canonical array of `{key, value}` objects (not dicts)
- **Proper span kinds**: Uses real OTEL span kinds (1=INTERNAL, 3=CLIENT) without redundant custom `span_kind` attribute
- **Root span handling**: Root span (user_input) omits `parentSpanId` entirely (not empty string)
- **Tool result as event**: Tool result is modeled as an event on the tool span (not a separate child span) for easier linking/deduplication
- **Span events**: Uses OTEL span events for tool_result and final_answer (attached to parent spans)
- **Post-tool span**: Explicit "post_tool_processing" span fills timestamp gap between tool end and LLM output start for clear latency attribution
- **Stable linking fields**: `request_id`, `turn_id`, `conversation_id` in resource.attributes for segmentation without relying on time
- **Semantic attributes**: Dual attribute naming:
  - Custom: `model_id`, `tool_name`, `tool_run_id` (existing adapter mappings)
  - OTEL-standard: `llm.model_id`, `tool.name`, `tool.run_id` (vendor-agnostic, no regex needed)
- **Final answer placement**: `final_answer` attached as event on model_invoke span (in addition to top-level field)
- **UnixNano timestamps**: startTimeUnixNano, endTimeUnixNano for nanosecond precision
- **Nested structure**: Tests event_paths: "resourceSpans.*.scopeSpans.*.spans"
- **Message linking**: `message_id` and `model_call_id` in span attributes

**Expected Behavior**:
- Should discover events via OTEL event_paths with wildcard traversal
- Should parse array-based attributes correctly (key/value pairs)
- Should parse UnixNano timestamps correctly (magnitude inference)
- Should extract attributes from nested structure
- Should preserve span hierarchy via traceId/spanId/parentSpanId
- Should handle omitted parentSpanId on root span (not treat empty string as parent)
- Should extract tool result from span events (not separate span)
- Should link tool invocation and result via single span + event
- Should recognize both custom and OTEL-standard attribute names
- Should extract final_answer from span event (in addition to top-level)
- Should attribute latency correctly with post_tool_processing span
- Should use resource.attributes for segmentation (request_id, turn_id)

---

### 8. `trace_missing_timestamps.json`
**Purpose**: Test handling of missing and invalid timestamps

**Features**:
- Event 1: No timestamp field at all
- Event 2: Null timestamp
- Event 3: Invalid timestamp format string
- Event 4: Valid ISO 8601 timestamp
- Event 5: Uses 'ts' field instead of 'timestamp' (tests field alias)

**Expected Behavior**:
- Should handle missing timestamps gracefully
- Should apply confidence penalty for missing/invalid timestamps
- Should use event_order and source_index for deterministic ordering
- Should mark timestamps as untrusted (ts_trusted=false)
- Should set latency_ms to null when timestamps missing

---

### 9. `trace_duplicate_tool_calls.json`
**Purpose**: Test tool call deduplication within time windows

**Features**:
- Two identical tool calls within 2 second window (should dedupe)
- Second duplicate has extra attributes (prefer richer)
- Third tool call outside window (should NOT dedupe)
- Tests deduplication by key_fields (tool_run_id, tool_name)

**Expected Behavior**:
- Should deduplicate first two tool calls (within window)
- Should prefer duplicate with richer attributes
- Should keep third tool call (outside window)
- Should link all tool results correctly

---

### 10. `trace_bad_epoch_units.json`
**Purpose**: Test epoch timestamp magnitude inference and year bounds validation

**Features**:
- Event 1: Ambiguous epoch (1705315800) - could be seconds or milliseconds
- Event 2: Epoch in milliseconds (1705315800100)
- Event 3: Epoch in nanoseconds (1705315800200000000)
- Event 4: Epoch in milliseconds (1705315800400)
- Event 5: Far future timestamp (9999999999 = year 2286, outside bounds)
- Event 6: Boundary timestamp (946684800 = Jan 1, 2000, exactly at min_reasonable_year)

**Expected Behavior**:
- Should infer epoch units by magnitude (seconds vs ms vs ns)
- Should validate timestamps against year bounds (2000-2100)
- Should reject/flag timestamps outside reasonable range
- Should handle boundary cases correctly
- Should apply confidence penalty for invalid timestamps

---

## Testing Guidelines

### Unit Tests
Each trace should be loaded and normalized in unit tests:
```python
def test_trace_single_turn_success():
    result = adapt("examples/sample_traces/trace_single_turn_success.json")
    assert result is not None
    assert validates_against_schema(result)
    assert result['turns'][0]['confidence'] == 1.0
```

### Property Tests
Use these traces as examples in property-based tests:
```python
@given(sample_trace_path())
def test_all_samples_normalize_successfully(trace_path):
    result = adapt(trace_path)
    assert validates_against_schema(result)
```

### Integration Tests
Test complete pipeline with all samples:
```python
def test_all_sample_traces():
    for trace_file in Path("examples/sample_traces").glob("trace_*.json"):
        result = adapt(str(trace_file))
        assert result is not None
        # Verify expected behavior per trace
```

## Adding New Traces

When adding new sample traces:

1. Follow naming convention: `trace_<description>.json`
2. Include descriptive attributes explaining edge cases
3. Document expected behavior in this README
4. Add corresponding test cases
5. helps trace tests specific edge case or requirement

## Requirements Coverage

These traces validate the following requirements:

- **Req 9.4**: Multi-turn conversation support (trace_multi_turn.json)
- **Req 9.5**: Tool calls and success status (trace_single_turn_success.json)
- **Req 9.6**: Failed steps (trace_with_failure.json)
- **Req 9.7**: Missing optional fields (trace_minimal_fields.json)
- **Req 9.8**: Stitched traces (trace_stitched.json)
- **Req 9.9**: Orphan tool results (trace_orphan_tool_results.json)
- **Req 9.10**: Tool-looking text without markers (tested via classification rules)
- **Req 2.8**: OTEL format support (trace_otel_format.json)
- **Req 2.11**: Missing/invalid timestamps (trace_missing_timestamps.json)
- **Req 8.7**: Tool deduplication (trace_duplicate_tool_calls.json)
- **Req 2.9**: Epoch unit inference and year bounds (trace_bad_epoch_units.json)
