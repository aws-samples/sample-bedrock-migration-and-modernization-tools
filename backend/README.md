# Bedrock Model Profiler — Backend

Serverless data pipeline using AWS Step Functions to collect, enrich, and aggregate Amazon Bedrock model data from 10+ sources. Runs twice daily at 6 AM and 6 PM UTC. Features inter-Lambda caching (~97% cache hit rate), self-healing with Claude Opus 4.5, and centralized model ID matching.

## Architecture

The backend orchestrates data collection through 4 distinct phases:

```
Phase 0: Initialization
  DiscoverRegions → InitializeExecution → ConfigSync

Phase 1: Parallel Collection (Wave 1)
  ├── pricing-collector (x3 service codes) → pricing-aggregator
  ├── model-extractor (x27 regions, caches to S3) → model-merger
  └── quota-collector (xN regions)

Phase 2: Parallel Enrichment (Wave 2)
  ├── pricing-linker (fuzzy matching via model_matcher.py)
  ├── regional-availability (reads cache from model-extractor)
  ├── feature-collector (xN, reads cache from region-discovery)
  ├── token-specs-collector (LiteLLM, 24h TTL cache)
  ├── mantle-collector (xN regions)
  └── lifecycle-collector (AWS docs scraping, 24h TTL cache)

Phase 3: Aggregation & Intelligence
  final-aggregator → gap-detection → [self-healing-agent] → copy-to-latest
```

## Lambda Functions

The pipeline consists of 17 Lambda functions with carefully tuned concurrency and resource limits:

| Function | Timeout | Memory | Concurrency | Description |
|----------|---------|--------|-------------|-------------|
| region-discovery | 120s | 256MB | 1 | Discover Bedrock regions + cache inference profiles |
| config-sync | 60s | 256MB | 1 | Sync frontend config from backend profiler-config.json |
| pricing-collector | 300s | 512MB | 3 (Map) | Collect pricing per service code from AWS Pricing API |
| pricing-aggregator | 120s | 1GB | 1 | Merge pricing from 3 service codes |
| model-extractor | 60s | 256MB | 10 (Map) | Extract models from all regions + Bedrock Console REST API |
| model-merger | 60s | 512MB | 1 | Deduplicate and merge models from all regions |
| quota-collector | 60s | 256MB | 10 (Map) | Collect Service Quotas per region |
| pricing-linker | 120s | 1GB | 1 | Fuzzy-match models to pricing (confidence threshold 0.7) |
| regional-availability | 60s | 512MB | 1 | Compute ON_DEMAND + PROVISIONED availability (from cache) |
| feature-collector | 60s | 256MB | 10 (Map) | Collect inference profiles per region (from cache) |
| token-specs-collector | 120s | 512MB | 1 | Fetch token specs from LiteLLM GitHub (24h TTL) |
| mantle-collector | 120s | 256MB | 10 (Map) | Collect Mantle models + probe Responses API |
| lifecycle-collector | 90s | 256MB | 1 | Scrape AWS docs for model lifecycle (24h TTL) |
| final-aggregator | 180s | 2GB | 1 | Merge all enrichments into final schema |
| gap-detection | 120s | 512MB | 1 | Detect 7 gap types (pricing, provider, context, etc.) |
| self-healing-agent | 300s | 1GB | 1 | Claude Opus 4.5 auto-suggests and applies safe config fixes |
| copy-to-latest | 60s | 256MB | 1 | Copy final files to latest/ for CloudFront |

## Shared Layer

All Lambda functions import from a common Python layer (`layers/common/python/shared/`):

| Module | Purpose |
|--------|---------|
| model_matcher.py | Model ID normalization, fuzzy matching, semantic conflict detection, variant analysis |
| cache_utils.py | S3-based inter-Lambda caching (get_cached_models, is_cache_valid, build_cache_key) |
| config_loader.py | Externalized config from profiler-config.json (S3 with embedded defaults fallback) |
| s3_utils.py | S3 read/write operations with JSON handling |
| validation.py | Input validation and custom exceptions |
| execution.py | Step Functions execution ID parsing |
| powertools.py | AWS Lambda Powertools wrappers (logger, tracer, metrics) |
| exceptions.py | Structured exception hierarchy |
| types.py | Type definitions (ModelVariantInfo, etc.) |

### Lambda Handler Pattern

All Lambda functions follow a consistent structure:

```python
import boto3
from shared import s3_utils, config, validation

def lambda_handler(event, context):
    # Validate input
    validation.require_params(event, ['s3Bucket', 's3Key'])

    # Process
    result = process_data(event)

    # Write results to S3
    s3_utils.write_json(bucket, key, result)

    # Return structured response
    return {"status": "SUCCESS", "s3Key": key, "recordCount": len(result)}
```

## Caching Architecture

The pipeline achieves ~97% cache hit rate through intelligent S3-based caching:

- **Model data**: model-extractor → S3 cache → regional-availability (per-execution)
- **Inference profiles**: region-discovery → S3 cache → feature-collector (per-execution)
- **LiteLLM data**: token-specs-collector → S3 cache → self (24h TTL)
- **Lifecycle data**: lifecycle-collector → S3 cache → self (24h TTL)

**Result**: ~480 → ~29 API calls per execution

Cache utilities in `shared/cache_utils.py` provide:
- `get_cached_models(bucket, execution_id, region)` — Retrieve cached model data
- `is_cache_valid(last_modified, ttl_hours)` — Check TTL-based cache validity
- `build_cache_key(execution_id, resource_type, region)` — Construct consistent S3 keys

## Self-Healing

The pipeline includes autonomous gap detection and remediation:

### Gap Detection

`gap-detection` Lambda identifies 7 gap types:
- Models without pricing
- Low-confidence pricing matches
- Unknown provider patterns
- New models not in config
- Context window mismatches
- Unknown pricing service codes
- Frontend config drift

### Self-Healing Agent

`self-healing-agent` Lambda invokes Claude Opus 4.5 to:
1. Analyze gap detection report
2. Generate safe configuration changes (provider patterns, context windows, service codes)
3. Validate against safety thresholds
4. Create config backup
5. Apply changes to `profiler-config.json`

The agent only auto-applies changes that meet safety criteria. All others are logged for manual review.

## Configuration

External configuration in `config/profiler-config.json` (auto-versioned by self-healing agent):

```json
{
  "version": "1.2.3",
  "external_urls": {...},
  "provider_configuration": {
    "patterns": [...],
    "fallback_provider": "AWS"
  },
  "region_configuration": {
    "bedrock_regions": [...],
    "priority_regions": [...]
  },
  "model_configuration": {
    "inference_type_patterns": {...},
    "context_window_overrides": {...}
  },
  "matching_configuration": {
    "confidence_threshold": 0.7,
    "fuzzy_match_cutoff": 0.8
  },
  "agent_configuration": {
    "auto_apply_threshold": 0.9,
    "max_changes_per_run": 10
  },
  "pricing_service_codes": [...],
  "gap_detection_config": {...}
}
```

## S3 Output Structure

```
s3://bedrock-profiler-data-{account}-{env}/
├── executions/{id}/
│   ├── pricing/          # Raw pricing per service code
│   ├── models/           # Raw models per region
│   ├── quotas/           # Raw quotas per region
│   ├── features/         # Inference profiles per region
│   ├── mantle/           # Mantle models per region
│   ├── cache/            # Inter-Lambda cache (model data, profiles)
│   ├── merged/           # Aggregated pricing + models
│   ├── intermediate/     # Enrichment outputs
│   ├── enriched/         # Feature-enriched data
│   └── final/            # Final bedrock_models.json + bedrock_pricing.json
├── latest/               # Current production data (served by CloudFront)
├── cache/                # TTL-based caches (LiteLLM, lifecycle)
├── config/               # profiler-config.json + history
└── agent/                # Gap reports + self-healing suggestions
```

## Testing

The backend includes ~150 tests covering all Lambda functions and shared utilities.

### Run All Tests

```bash
cd backend/tests
python -m pytest
```

### Local Workflow Test

```bash
cd backend/tests
python test_workflow_local.py
```

### Production Validation

```bash
cd backend/tests
./run_production_tests.sh
```

## Deployment

### Prerequisites

- AWS SAM CLI
- Python 3.11
- AWS credentials with permissions for:
  - Bedrock (ListFoundationModels, ListInferenceProfiles)
  - Pricing API (GetProducts)
  - Service Quotas (ListServiceQuotas)
  - S3, Lambda, Step Functions, IAM, CloudWatch

### Build and Deploy

```bash
cd infra
sam build -t backend-template.yaml
sam deploy --stack-name bedrock-profiler-dev --capabilities CAPABILITY_NAMED_IAM --resolve-s3
```

### Deploy to Production

```bash
cd infra
sam build -t backend-template.yaml
sam deploy --stack-name bedrock-profiler-prod --capabilities CAPABILITY_NAMED_IAM --resolve-s3 \
  --parameter-overrides "Environment=\"prod\" ScheduleEnabled=\"true\"" \
  --region us-east-1 --no-confirm-changeset
```

### SAM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Environment | dev | Environment name (dev/staging/prod) |
| ScheduleEnabled | true | Enable scheduled execution (6 AM and 6 PM UTC) |
| CloudFrontDistributionArn | - | CloudFront distribution for cache invalidation |
| ExistingDataBucket | - | Use existing S3 bucket (optional) |
| LogLevel | INFO | Lambda logging level (DEBUG/INFO/WARNING/ERROR) |
| AvailabilityMaxWorkers | 10 | Max concurrent workers for availability computation |
| QuotaBatchSize | 50 | Batch size for quota API calls |
| CognitoRegion | us-east-1 | Region for Cognito user pool |

### Manual Execution

```bash
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:REGION:ACCOUNT:stateMachine:bedrock-profiler-workflow-dev \
  --input '{}'
```

## Monitoring

### CloudWatch Logs

- State machine: `/aws/stepfunctions/bedrock-profiler-{env}`
- Lambda functions: `/aws/lambda/bedrock-profiler-*-{env}`

### Step Functions Console

View execution graph, input/output for each state, and error details in the AWS Step Functions console.

### Gap Detection Reports

Self-healing reports are written to `s3://{bucket}/agent/{execution-id}/`:
- `gap-report.json` — Detected gaps
- `self-healing-suggestions.json` — Proposed fixes
- `applied-changes.json` — Auto-applied changes

## Error Handling

- **Retryable errors** (throttling, timeouts): Automatic retry with exponential backoff (3 attempts)
- **Failed regions**: Logged but workflow continues (graceful degradation)
- **Critical failures**: Pricing aggregation or final aggregation failures stop the workflow
- **Lambda failures**: Structured error responses with detailed error messages

## Cost

Per execution (~51+ Lambda invocations):
- Lambda: ~$0.01-0.02
- Step Functions: ~$0.0003 (state transitions)
- S3: ~$0.001 (PUT requests + storage)
- Bedrock (self-healing): ~$0.003-0.01 (conditional)

**Estimated monthly cost**: ~$1.00-2.00 (twice-daily execution)

## Performance

| Metric | Value |
|--------|-------|
| Total Duration | 2-3 minutes |
| Max Parallelism | 27 concurrent Lambdas |
| API Calls per Execution | ~29 (with caching) |
| Cache Hit Rate | ~97% |
| Data Freshness | Updated twice daily (6 AM and 6 PM UTC) |

## AWS Services Used

- **Bedrock**: ListFoundationModels, ListInferenceProfiles
- **Pricing API**: GetProducts (us-east-1 only)
- **Service Quotas**: ListServiceQuotas
- **Step Functions**: Workflow orchestration
- **Lambda**: Serverless compute
- **S3**: Data storage and caching
- **CloudWatch**: Logging and monitoring
- **EventBridge**: Scheduled execution
- **IAM**: Permission management
