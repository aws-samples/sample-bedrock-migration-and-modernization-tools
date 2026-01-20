# Bedrock Model Profiler - Step Functions Workflow

Serverless workflow for collecting and aggregating Amazon Bedrock model data using AWS Step Functions with maximum parallelization.

## Architecture

```
                                    START
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐             ┌───────────────┐             ┌───────────────┐
│  PRICING MAP  │             │  MODELS MAP   │             │  QUOTAS MAP   │
│  (3 parallel) │             │  (2 parallel) │             │ (20 parallel) │
└───────┬───────┘             └───────┬───────┘             └───────┬───────┘
        │                             │                             │
        ▼                             ▼                             │
┌───────────────┐             ┌───────────────┐                     │
│ AGG PRICING   │             │ MERGE MODELS  │                     │
└───────┬───────┘             └───────┬───────┘                     │
        │                             │                             │
        └──────────────┬──────────────┘                             │
                       │                                            │
                       ▼                                            │
        ┌──────────────────────────────────────┐                    │
        │         PARALLEL ENRICHMENT          │                    │
        │  ┌────────┬────────┬──────────────┐  │                    │
        │  │  LINK  │REGIONAL│   FEATURES   │  │                    │
        │  │PRICING │ AVAIL  │ (20 parallel)│  │                    │
        │  ├────────┴────────┴──────────────┤  │                    │
        │  │         TOKEN SPECS            │  │                    │
        │  └────────────────────────────────┘  │                    │
        └──────────────────┬───────────────────┘                    │
                           │                                        │
                           └────────────────────┬───────────────────┘
                                                │
                                                ▼
                                    ┌───────────────────┐
                                    │ FINAL AGGREGATION │
                                    └─────────┬─────────┘
                                              │
                                              ▼
                                           SUCCESS
```

## Performance

| Metric | Sequential (Current) | Parallel (Step Functions) |
|--------|---------------------|---------------------------|
| Total Duration | 6-7 min | 2-3 min |
| Max Parallelism | 10 threads | 25 Lambdas |
| Retry Granularity | Entire phase | Single region |

## Project Structure

```
bedrock-profiler-stepfunctions/
├── statemachine/
│   └── bedrock-profiler.asl.json    # Step Functions definition (ASL)
├── lambdas/
│   ├── README.md                     # Lambda function interfaces
│   ├── pricing-collector/            # Collect pricing per service code
│   ├── pricing-aggregator/           # Merge pricing data
│   ├── model-extractor/              # Extract models per region
│   ├── model-merger/                 # Merge models from regions
│   ├── quota-collector/              # Collect quotas per region
│   ├── pricing-linker/               # Link pricing to models
│   ├── regional-availability/        # Compute availability map
│   ├── feature-collector/            # Collect inference profiles
│   ├── token-specs-collector/        # Fetch from LiteLLM
│   ├── final-aggregator/             # Merge all data
│   └── copy-to-latest/               # Copy to latest/ prefix
└── infrastructure/
    └── template.yaml                 # SAM template
```

## Prerequisites

- AWS SAM CLI
- Python 3.11
- AWS credentials with permissions for:
  - Bedrock (ListFoundationModels, ListInferenceProfiles)
  - Pricing API (GetProducts)
  - Service Quotas (ListServiceQuotas)
  - S3, Lambda, Step Functions, IAM, CloudWatch

## Deployment

### 1. Build

```bash
cd bedrock-profiler-stepfunctions
sam build -t infrastructure/template.yaml
```

### 2. Deploy

```bash
sam deploy \
  --guided \
  --stack-name bedrock-profiler-dev \
  --capabilities CAPABILITY_NAMED_IAM
```

### 3. Manual Execution

```bash
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:REGION:ACCOUNT:stateMachine:bedrock-profiler-workflow-dev \
  --input '{}'
```

## Configuration

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Environment` | dev | Environment name (dev/staging/prod) |
| `ScheduleEnabled` | true | Enable daily 6 AM UTC execution |

### Region Lists

Edit `statemachine/bedrock-profiler.asl.json` to modify:

- `modelRegions`: Regions for model extraction (default: us-east-1, us-west-2)
- `quotaRegions`: Regions for quota collection (20 regions)
- `featureRegions`: Regions for inference profile collection (20 regions)

## S3 Output Structure

```
s3://bedrock-profiler-data-{account}-{env}/
├── executions/{execution-id}/
│   ├── pricing/
│   │   ├── AmazonBedrock.json
│   │   ├── AmazonBedrockService.json
│   │   └── AmazonBedrockFoundationModels.json
│   ├── models/
│   │   ├── us-east-1.json
│   │   └── us-west-2.json
│   ├── quotas/
│   │   └── {region}.json (x20)
│   ├── features/
│   │   └── {region}.json (x20)
│   ├── merged/
│   │   ├── pricing.json
│   │   └── models.json
│   ├── intermediate/
│   │   ├── models-with-pricing.json
│   │   ├── regional-availability.json
│   │   └── token-specs.json
│   └── final/
│       ├── bedrock_models.json
│       └── bedrock_pricing.json
└── latest/
    ├── bedrock_models.json      # Always points to latest successful run
    └── bedrock_pricing.json
```

## Monitoring

### CloudWatch Logs

- State machine: `/aws/stepfunctions/bedrock-profiler-{env}`
- Lambda functions: `/aws/lambda/bedrock-profiler-*-{env}`

### Step Functions Console

View execution graph, input/output for each state, and error details in the AWS Step Functions console.

## Error Handling

- **Retryable errors** (throttling, timeouts): Automatic retry with exponential backoff
- **Failed regions**: Logged but workflow continues (graceful degradation)
- **Critical failures**: Pricing aggregation or final aggregation failures stop the workflow

## Cost Estimate

Per execution (~51 Lambda invocations):
- Lambda: ~$0.01-0.02
- Step Functions: ~$0.0003 (state transitions)
- S3: ~$0.001 (PUT requests + storage)

**Daily cost**: ~$0.50-1.00/month

## Migration from Python Scripts

The existing Python collectors in `2025.09 - Amazon Bedrock Model Choice/` are being migrated to Lambda functions. Each phase of the original collector maps to one or more Lambda functions:

| Original Phase | Lambda Function(s) |
|----------------|-------------------|
| Pricing Collector | pricing-collector (x3), pricing-aggregator |
| Phase 1: Models | model-extractor (x2), model-merger |
| Phase 2: Link Pricing | pricing-linker |
| Phase 3: Regional Availability | regional-availability |
| Phase 4: Features | feature-collector (x20) |
| Phase 4.5: Token Specs | token-specs-collector |
| Phase 5: Quotas | quota-collector (x20) |
| Final Output | final-aggregator, copy-to-latest |
