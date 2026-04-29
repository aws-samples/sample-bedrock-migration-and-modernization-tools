# Trace Evaluator Architecture

## Module Organization

The trace evaluator follows a clean separation of concerns:

### `agent_eval/schemas/` ✅ EXISTS
**JSON schemas for validation and contracts**

```
schemas/
├── normalized_run.schema.json       ✅ EXISTS - Input contract
├── rubric.schema.json               ✅ EXISTS - Rubric definition
├── judge_config.schema.json         ✅ EXISTS - Judge configuration (1-5 judges)
├── judge_response.schema.json       ✅ EXISTS - Judge response format
├── judge_run_record.schema.json     ✅ EXISTS - JSONL record structure
├── trace_eval_output.schema.json    ✅ EXISTS - trace_eval.json output
└── results.schema.json              ✅ EXISTS - results.json output
```

**Why critical**: Validator, judge response parsing, JSONL writer, and output writer all enforce these schemas.

### `agent_eval/evaluators/trace_eval/`
**Evaluator-specific orchestration and logic**

```
trace_eval/
├── __init__.py                      ✅ EXISTS
├── ARCHITECTURE.md                  ✅ EXISTS - This file
├── default_rubrics.yaml             ⚠️  TODO - 8 built-in rubrics (TOOL_GROUNDEDNESS, etc.)
├── input_validator.py               ⚠️  TODO - NormalizedRun schema validation
├── deterministic_metrics.py         ⚠️  TODO - Metrics computed without LLM
├── timestamp_policy.py              ⚠️  TODO - Timestamp trust rules for latency p50/p95
├── rubric_loader.py                 ⚠️  TODO - Load and merge rubrics
├── runner.py                        ⚠️  TODO - Main orchestration (TraceEvaluator)
├── output_writer.py                 ⚠️  TODO - Write canonical output files
└── judging/                         ✅ EXISTS (directory)
    ├── __init__.py                  ✅ EXISTS
    ├── models.py                    ⚠️  TODO - JudgeJob, JobResult data models
    ├── job_builder.py               ⚠️  TODO - Build JudgeJobs from rubrics
    ├── queue_runner.py              ⚠️  TODO - Worker Pool execution
    ├── aggregator.py                ⚠️  TODO - Within-judge and cross-judge aggregation
    ├── evidence.py                  ⚠️  TODO - Evidence extraction (jsonpath + scope + budget + redaction)
    ├── rate_limiter.py              ⚠️  TODO - Token bucket rate limiting per judge
    └── retry_policy.py              ⚠️  TODO - Exponential backoff retry logic
```

**Critical gap - evidence.py**: Applies evidence_selectors (jsonpath), enforces scope (turn vs run), enforces evidence budget (10k chars max), optionally redacts sensitive fields. This is the single biggest "you'll regret not having it" gap.

### `agent_eval/judges/` ✅ EXISTS (directory)
**Shared judge primitives (reusable across evaluators)**

```
judges/
├── __init__.py                      ✅ EXISTS
├── judge_config_loader.py           ⚠️  TODO - Load and validate judge configuration (1-5 judges)
├── judge_client.py                  ⚠️  TODO - Abstract JudgeClient interface
├── models.py                        ⚠️  TODO - Judge, JudgeConfig data models
└── exceptions.py                    ⚠️  TODO - JudgeExecutionError, ValidationError
```

### `agent_eval/providers/`
**Runtime provider implementations (Bedrock-first)**

```
providers/
├── __init__.py                      ⚠️  TODO - Provider registry (resolve provider: bedrock|openai|anthropic)
└── bedrock_client.py                ⚠️  TODO - Amazon Bedrock judge implementation (PRIORITY)
```

**Note**: Start with Amazon Bedrock only. Add OpenAI/Anthropic clients later after Stage 3 tests pass.

### `agent_eval/evaluators/results/` ✅ EXISTS (directory)
**Results aggregation and composition (optional - may merge into trace_eval)**

```
results/
├── __init__.py                      ✅ EXISTS
├── models.py                        ⚠️  TODO - Result dataclasses (if keeping separate)
└── writer.py                        ⚠️  TODO - Hashing + schema validation + artifact paths
```

**Decision needed**: Keep separate or merge into `trace_eval/output_writer.py`. If keeping separate, needs real content.

### `agent_eval/cli.py` ✅ EXISTS
**Thin CLI entrypoint**

Responsibilities:
- Parse CLI arguments
- Validate required arguments
- Delegate to `TraceEvaluator.run()`
- Handle exit codes and top-level errors

**Does NOT contain orchestration logic** - that lives in `runner.py`.

## Design Principles

1. **Separation of Concerns**: Evaluator orchestration is separate from reusable judge primitives
2. **No Circular Dependencies**: Evaluator depends on judges/providers, not vice versa
3. **Thin CLI**: CLI is just argument parsing + delegation
4. **Reusable Building Blocks**: `judges/` and `providers/` can be used by other evaluators
5. **Clear Ownership**: Each module has a single responsibility
6. **Schema-First**: All I/O validated against JSON schemas in `schemas/`
7. **Evidence Safety**: Evidence extraction enforces scope, budget (10k chars), and PII redaction

## Critical Implementation Gaps (Priority Order)

### 1. Evidence Extraction (`judging/evidence.py`) - HIGHEST PRIORITY
**Why critical**: Without this, you can't safely build prompts for judges.

Must implement:
- Apply `evidence_selectors` using jsonpath-ng
- Enforce scope (turn-scoped selectors cannot pull entire run data)
- Enforce evidence budget (max 10,000 chars per rubric payload)
- Truncate with warning when evidence exceeds budget
- Basic PII redaction (strip sensitive fields from tool payloads)

### 2. Default Rubrics (`default_rubrics.yaml`) - REQUIRED FOR TASK 2
**Why critical**: System ships with 8 built-in rubrics with stable IDs.

Must include:
- TOOL_GROUNDEDNESS
- TOOL_CONSISTENCY
- TOOL_CALL_QUALITY
- TOOL_CHAINING
- TRACE_COMPLETENESS
- SAFETY_PII
- LATENCY_REGRESSION_FLAG
- STITCHED_TRACE_SUSPECT

Each with: rubric_id, description, scoring_scale (type + aggregation_type), requires_llm_judge, evidence_selectors, scope

### 3. Timestamp Trust Policy (`timestamp_policy.py`) - REQUIRED FOR DETERMINISTIC METRICS
**Why critical**: Latency p50/p95 need explicit trust rules so tests can pin behavior.

Must implement:
- Trusted timestamp criteria: OTEL UnixNano format with both start_time and end_time
- Clock skew detection: reject negative duration or >24h duration
- Missing end timestamp handling: mark as untrusted, exclude from latency calculations
- Set latency_p50/p95 to None when <50% of turns have trusted timestamps

### 4. Judge Run Record Models (`judging/models.py`) - REQUIRED FOR JSONL STABILITY
**Why critical**: Keeps judge_runs.jsonl records consistent and testable.

Must implement:
- JudgeJob dataclass (job_id, run_id, turn_id, rubric_id, judge_id, repeat_index, prompt_payload)
- JobResult dataclass (matches judge_run_record.schema.json)
- to_jsonl_line() method for serialization

### 5. Rate Limiting & Retry Policy (`judging/rate_limiter.py`, `judging/retry_policy.py`)
**Why critical**: Prevents overwhelming judge APIs and handles transient failures.

Must implement:
- Token bucket rate limiter (per-judge)
- Exponential backoff retry (1s, 2s, 4s, max 3 retries)
- Timeout enforcement (per-call timeout using timeout_seconds config)

### 6. Provider Registry (`providers/__init__.py`)
**Why critical**: Evaluator needs to resolve `provider: bedrock|openai|anthropic` from judge config.

Must implement:
- Registry mapping provider string to client class
- Start with Amazon Bedrock only, add others later

### 7. Results Module Decision
**Why critical**: Avoid dead code or unclear ownership.

Options:
- **Option A**: Keep `evaluators/results/` with models.py + writer.py for hashing/validation
- **Option B**: Merge into `trace_eval/output_writer.py` and remove `results/` package

Recommendation: Option B (merge) unless you plan multiple evaluators sharing result composition logic.

## Evaluation Flow

```
CLI (cli.py)
  ↓
TraceEvaluator (runner.py)
  ↓
├─→ InputValidator (input_validator.py)
├─→ RubricLoader (rubric_loader.py)
├─→ DeterministicMetrics (deterministic_metrics.py)
├─→ JobBuilder (judging/job_builder.py)
│     ↓
│   Uses: JudgeConfigLoader (judges/judge_config_loader.py)
│
├─→ WorkerPool (judging/queue_runner.py)
│     ↓
│   Uses: JudgeClient (judges/judge_client.py)
│         ↓
│       Implemented by: BedrockClient (providers/bedrock_client.py)
│
├─→ Aggregator (judging/aggregator.py)
└─→ OutputWriter (output_writer.py)
```

## Key Interfaces

### JudgeClient (judges/judge_client.py)
```python
class JudgeClient(ABC):
    @abstractmethod
    def execute_judge(self, job: JudgeJob) -> JobResult:
        """Execute a single judge job and return result."""
        pass
```

### TraceEvaluator (runner.py)
```python
class TraceEvaluator:
    def __init__(self, input_path, judge_config_path, output_dir, ...):
        """Initialize evaluator with configuration."""
        pass
    
    def run(self) -> int:
        """
        Run complete evaluation pipeline:
        1. Validate input
        2. Compute deterministic metrics
        3. Build JudgeJob queue
        4. Execute jobs via Worker Pool
        5. Aggregate results
        6. Write output files
        
        Returns exit code (0 for success).
        """
        pass
```

## Implementation Status Summary

### ✅ Complete (Task 1)
- Module structure created
- All 7 JSON schemas created and validated
- CLI entrypoint added (thin, delegates to runner)
- Import sanity tests passing (22 tests)

### ⚠️ TODO (Tasks 2-20)
- Default rubrics YAML (8 rubrics)
- All evaluator components (validator, metrics, loader, runner, writer)
- All judging components (job builder, queue runner, aggregator, evidence extractor)
- All judge primitives (config loader, client interface, models, exceptions)
- Amazon Bedrock provider implementation
- Rate limiting and retry policy
- Timestamp trust policy
- All unit, integration, property, and golden tests

### 🔴 Critical Path (Must implement first)
1. Evidence extraction (`judging/evidence.py`)
2. Default rubrics (`default_rubrics.yaml`)
3. Timestamp trust policy (`timestamp_policy.py`)
4. Judge run record models (`judging/models.py`)
5. Rate limiter and retry policy

### 📋 Next Steps
1. Implement Task 2: Default rubrics system
2. Implement Task 3: Input validation
3. Implement Task 4: Judge configuration system
4. Continue through tasks sequentially, running tests at each checkpoint

## Testing Strategy

### Test Organization (`agent_eval/tests/`)

```
tests/
├── __init__.py                                  ✅ EXISTS
├── test_module_imports.py                       ✅ EXISTS - Module import sanity (22 tests passing)
├── test_trace_eval_input_validator.py           ⚠️  TODO - Input validation tests
├── test_trace_eval_deterministic_metrics.py     ⚠️  TODO - Deterministic metrics tests
├── test_trace_eval_timestamp_policy.py          ⚠️  TODO - Timestamp trust policy tests
├── test_trace_eval_rubric_loader.py             ⚠️  TODO - Rubric loading and merging tests
├── test_trace_eval_job_builder.py               ⚠️  TODO - JudgeJob building tests (CRITICAL)
├── test_trace_eval_queue_runner.py              ⚠️  TODO - Worker Pool execution tests (CRITICAL)
├── test_trace_eval_aggregator.py                ⚠️  TODO - Aggregation tests (CRITICAL)
├── test_trace_eval_evidence.py                  ⚠️  TODO - Evidence extraction tests (CRITICAL)
├── test_trace_eval_evidence_budget.py           ⚠️  TODO - Evidence budget enforcement tests
├── test_trace_eval_outputs_schema.py            ⚠️  TODO - Output schema validation tests (CRITICAL)
├── test_trace_eval_rate_limiter.py              ⚠️  TODO - Rate limiting tests
├── test_trace_eval_retry_policy.py              ⚠️  TODO - Retry policy tests
├── test_trace_eval_integration.py               ⚠️  TODO - End-to-end integration tests with mock judges
└── test_trace_eval_golden.py                    ⚠️  TODO - Golden tests for deterministic outputs
```

### Test Types

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test end-to-end flow with mock judges
- **Property tests**: Verify correctness properties with hypothesis (34 properties defined in design doc)
- **Golden tests**: Ensure deterministic outputs remain stable

### Critical Test Requirements

1. **test_trace_eval_job_builder.py**: Verify job count formula (LLM rubrics × judges × repeats)
2. **test_trace_eval_queue_runner.py**: Verify bounded concurrency, retry, fault tolerance
3. **test_trace_eval_aggregator.py**: Verify within-judge and cross-judge aggregation correctness
4. **test_trace_eval_evidence.py**: Verify evidence selector extraction, scope enforcement, budget enforcement
5. **test_trace_eval_outputs_schema.py**: Verify all output files match schemas and contain required fields

See `agent_eval/tests/test_module_imports.py` for module import sanity tests (22 tests passing).
