# Running Production Gate Tests

Quick reference guide for executing the adapter production gate tests.

## Quick Start

```bash
cd agent-eval
python -m pytest agent_eval/tests/test_production_gates.py -v
```

## Test Commands

### Run All Tests
```bash
python -m pytest agent_eval/tests/test_production_gates.py -v
```

### Run Specific Test
```bash
python -m pytest agent_eval/tests/test_production_gates.py::TestProductionGates::test_case_01_single_turn_clean -v
```

### Run with Coverage
```bash
python -m pytest agent_eval/tests/test_production_gates.py --cov=agent_eval.adapters.generic_json --cov-report=html
```

### Run with Detailed Output
```bash
python -m pytest agent_eval/tests/test_production_gates.py -vv -s
```

### Run Only Failed Tests
```bash
python -m pytest agent_eval/tests/test_production_gates.py --lf
```

## Test Categories

### Happy Path (Cases 1-3)
```bash
python -m pytest agent_eval/tests/test_production_gates.py -k "case_01 or case_02 or case_03" -v
```

### Error Handling (Cases 4-5)
```bash
python -m pytest agent_eval/tests/test_production_gates.py -k "case_04 or case_05" -v
```

### Resilience (Cases 6-9)
```bash
python -m pytest agent_eval/tests/test_production_gates.py -k "case_06 or case_07 or case_08 or case_09" -v
```

### Tool Handling (Cases 10-13)
```bash
python -m pytest agent_eval/tests/test_production_gates.py -k "case_10 or case_11 or case_12 or case_13" -v
```

### Edge Cases (Cases 14-15)
```bash
python -m pytest agent_eval/tests/test_production_gates.py -k "case_14 or case_15" -v
```

## Expected Results

All 15 tests should pass:
- **13 tests** should pass normally (expected pass)
- **2 tests** should pass by catching expected exceptions (cases 4 and 5)

## Test Fixtures

Test fixtures are located in:
```
agent-eval/test-fixtures/production-gates/
├── case_01_single_turn_clean.json
├── case_02_multi_turn_clean.json
├── case_03_in_memory_dict.json (uses case_01)
├── case_05_missing_event_path.json
├── case_06_malformed_events.json
├── case_07_dirty_timestamps.json
├── case_08_missing_grouping_ids.json
├── case_09_turn_segmentation_noise.json
├── case_10_tool_success_inference.json
├── case_11_tool_failure_inference.json
├── case_12_span_parent_linking.json
├── case_13_orphan_tool_result.json
├── case_14_prompt_contamination.json
└── case_15_large_dirty_trace.json
```

## Troubleshooting

### Test Failures

If tests fail, check:
1. Adapter configuration: `agent-eval/agent_eval/adapters/generic_json/adapter_config.yaml`
2. Schema file: `agent-eval/agent_eval/schemas/normalized_run.schema.json`
3. Test fixtures: helps JSON files are valid

### Common Issues

**Issue**: `InputError: File not found`
- **Solution**: helps you're running from the `agent-eval` directory

**Issue**: `ValidationError: Schema file not found`
- **Solution**: Check that `agent_eval/schemas/normalized_run.schema.json` exists

**Issue**: `ImportError: No module named 'agent_eval'`
- **Solution**: Install the package: `pip install -e .`

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run Production Gate Tests
  run: |
    cd agent-eval
    python -m pytest agent_eval/tests/test_production_gates.py -v --junitxml=test-results.xml
```

### Pre-commit Hook
```bash
#!/bin/bash
cd agent-eval
python -m pytest agent_eval/tests/test_production_gates.py -q
```

## Related Documentation

- **Test Strategy**: `agent-eval/guides/ADAPTER_PRODUCTION_TESTING_STRATEGY.md`
- **Test Results**: `agent-eval/guides/PRODUCTION_GATE_TEST_RESULTS.md`
- **Adapter Documentation**: `agent-eval/agent_eval/adapters/generic_json/README.md`
