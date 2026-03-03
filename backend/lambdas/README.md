# Lambda Function Interfaces

This document defines the input/output contracts for each Lambda function in the Bedrock Profiler Step Functions workflow.

## Overview

| Function | Purpose | Timeout | Memory | Concurrency |
|----------|---------|---------|--------|-------------|
| `pricing-collector` | Collect pricing for one service code | 5 min | 512MB | 3 |
| `pricing-aggregator` | Merge pricing from all service codes | 2 min | 1GB | 1 |
| `model-extractor` | List foundation models from one region | 1 min | 256MB | 10 |
| `model-merger` | Merge models from all regions | 1 min | 512MB | 1 |
| `quota-collector` | Collect service quotas from one region | 1 min | 256MB | N (all discovered) |
| `pricing-linker` | Link pricing data to models | 2 min | 1GB | 1 |
| `regional-availability` | Compute regional availability map | 1 min | 512MB | 1 |
| `feature-collector` | Collect inference profiles from one region | 1 min | 256MB | 10 |
| `token-specs-collector` | Fetch token specs from LiteLLM | 2 min | 512MB | 1 |
| `lifecycle-collector` | Scrape lifecycle status from AWS docs | 1.5 min | 256MB | 1 |
| `final-aggregator` | Merge all data into final output | 3 min | 2GB | 1 |
| `copy-to-latest` | Copy final output to latest/ prefix | 1 min | 256MB | 1 |

---

## Function Contracts

### 1. pricing-collector

Collects pricing data from AWS Pricing API for a single service code.

**Input:**
```json
{
  "serviceCode": "AmazonBedrock",
  "s3Bucket": "bedrock-profiler-data",
  "s3Key": "executions/{execution-id}/pricing/AmazonBedrock.json"
}
```

**Output:**
```json
{
  "status": "SUCCESS",
  "serviceCode": "AmazonBedrock",
  "s3Key": "executions/{execution-id}/pricing/AmazonBedrock.json",
  "recordCount": 1250,
  "durationMs": 45000
}
```

**AWS APIs Called:**
- `pricing:GetProducts` (us-east-1 only)

**Environment Variables:**
- `PRICING_API_REGION`: us-east-1

---

### 2. pricing-aggregator

Merges pricing data from all three service codes into a single structured JSON.

**Input:**
```json
{
  "s3Bucket": "bedrock-profiler-data",
  "executionId": "arn:aws:states:...",
  "pricingResults": [
    {"status": "SUCCESS", "serviceCode": "AmazonBedrock", "s3Key": "..."},
    {"status": "SUCCESS", "serviceCode": "AmazonBedrockService", "s3Key": "..."},
    {"status": "SUCCESS", "serviceCode": "AmazonBedrockFoundationModels", "s3Key": "..."}
  ]
}
```

**Output:**
```json
{
  "status": "SUCCESS",
  "s3Key": "executions/{execution-id}/merged/pricing.json",
  "providersCount": 17,
  "totalPricingEntries": 3500,
  "durationMs": 8000
}
```

---

### 3. model-extractor

Lists foundation models from a single AWS region using Bedrock API.

**Input:**
```json
{
  "region": "us-east-1",
  "s3Bucket": "bedrock-profiler-data",
  "s3Key": "executions/{execution-id}/models/us-east-1.json"
}
```

**Output:**
```json
{
  "status": "SUCCESS",
  "region": "us-east-1",
  "s3Key": "executions/{execution-id}/models/us-east-1.json",
  "modelCount": 108,
  "cacheKey": "executions/{execution-id}/cache/list_foundation_models_us-east-1.json",
  "durationMs": 2500
}
```

**AWS APIs Called:**
- `bedrock:ListFoundationModels`

---

### 4. model-merger

Merges and deduplicates models from multiple regions.

**Input:**
```json
{
  "s3Bucket": "bedrock-profiler-data",
  "executionId": "arn:aws:states:...",
  "modelResults": [
    {"status": "SUCCESS", "region": "us-east-1", "s3Key": "..."},
    {"status": "SUCCESS", "region": "us-west-2", "s3Key": "..."}
  ]
}
```

**Output:**
```json
{
  "status": "SUCCESS",
  "s3Key": "executions/{execution-id}/merged/models.json",
  "totalModels": 108,
  "providersCount": 17,
  "durationMs": 3000
}
```

---

### 5. quota-collector

Collects Bedrock service quotas from a single region.

**Input:**
```json
{
  "region": "us-east-1",
  "s3Bucket": "bedrock-profiler-data",
  "s3Key": "executions/{execution-id}/quotas/us-east-1.json"
}
```

**Output:**
```json
{
  "status": "SUCCESS",
  "region": "us-east-1",
  "s3Key": "executions/{execution-id}/quotas/us-east-1.json",
  "quotaCount": 45,
  "durationMs": 1500
}
```

**AWS APIs Called:**
- `service-quotas:ListServiceQuotas` (ServiceCode: bedrock)

---

### 6. pricing-linker

Links pricing data to models, creating price references per model per region.

**Input:**
```json
{
  "s3Bucket": "bedrock-profiler-data",
  "executionId": "arn:aws:states:...",
  "pricingS3Key": "executions/{execution-id}/merged/pricing.json",
  "modelsS3Key": "executions/{execution-id}/merged/models.json"
}
```

**Output:**
```json
{
  "status": "SUCCESS",
  "s3Key": "executions/{execution-id}/intermediate/models-with-pricing.json",
  "modelsWithPricing": 86,
  "modelsWithoutPricing": 22,
  "durationMs": 5000
}
```

---

### 7. regional-availability

Computes regional availability map from pricing data.

**Input:**
```json
{
  "s3Bucket": "bedrock-profiler-data",
  "executionId": "arn:aws:states:...",
  "regions": ["us-east-1", "us-west-2", "eu-west-1", "..."],
  "cacheKeys": {
    "us-east-1": "executions/{id}/cache/list_foundation_models_us-east-1.json",
    "us-west-2": "executions/{id}/cache/list_foundation_models_us-west-2.json"
  },
  "pricingS3Key": "executions/{execution-id}/merged/pricing.json"
}
```

**Output:**
```json
{
  "status": "SUCCESS",
  "s3Key": "executions/{execution-id}/intermediate/regional-availability.json",
  "regionsWithBedrock": 27,
  "cacheHits": 27,
  "apiCalls": 0,
  "cacheHitRate": 100.0,
  "durationMs": 2000
}
```

---

### 8. feature-collector

Collects inference profiles and enhanced features from a single region.

**Input:**
```json
{
  "region": "us-east-1",
  "s3Bucket": "bedrock-profiler-data",
  "s3Key": "executions/{execution-id}/features/us-east-1.json",
  "inferenceProfileCacheKeys": {
    "us-east-1": "executions/{id}/cache/inference_profiles_us-east-1.json",
    "us-west-2": "executions/{id}/cache/inference_profiles_us-west-2.json"
  }
}
```

**Output:**
```json
{
  "status": "SUCCESS",
  "region": "us-east-1",
  "s3Key": "executions/{execution-id}/features/us-east-1.json",
  "inferenceProfileCount": 12,
  "fromCache": true,
  "durationMs": 500
}
```

**AWS APIs Called:**
- `bedrock:ListInferenceProfiles`

---

### 9. token-specs-collector

Fetches token specifications (context window, max output) from LiteLLM.

**Input:**
```json
{
  "s3Bucket": "bedrock-profiler-data",
  "executionId": "arn:aws:states:...",
  "modelsS3Key": "executions/{execution-id}/merged/models.json"
}
```

**Output:**
```json
{
  "status": "SUCCESS",
  "s3Key": "executions/{execution-id}/intermediate/token-specs.json",
  "modelsWithSpecs": 104,
  "modelsWithoutSpecs": 4,
  "source": "litellm",
  "fromCache": false,
  "durationMs": 8000
}
```

**External APIs Called:**
- LiteLLM model database (HTTPS)

---

### 10. lifecycle-collector

Scrapes AWS documentation for model lifecycle status (active, legacy, EOL).

**Input:**
```json
{
  "s3Bucket": "bedrock-profiler-data",
  "executionId": "arn:aws:states:...",
  "s3Key": "executions/{execution-id}/lifecycle/lifecycle.json"
}
```

**Output:**
```json
{
  "status": "SUCCESS",
  "s3Key": "executions/{execution-id}/lifecycle/lifecycle.json",
  "activeModels": 95,
  "legacyModels": 8,
  "eolModels": 5,
  "fromCache": false,
  "durationMs": 3000
}
```

**External APIs Called:**
- AWS Documentation (HTTPS scraping)

---

### 11. final-aggregator

Merges all collected data into the final comprehensive JSON output.

**Input:**
```json
{
  "s3Bucket": "bedrock-profiler-data",
  "executionId": "arn:aws:states:...",
  "pricingS3Key": "executions/{execution-id}/merged/pricing.json",
  "modelsS3Key": "executions/{execution-id}/merged/models.json",
  "quotaResults": [...],
  "pricingLinked": {...},
  "regionalAvailability": {...},
  "featureResults": [...],
  "tokenSpecs": {...}
}
```

**Output:**
```json
{
  "status": "SUCCESS",
  "modelsS3Key": "executions/{execution-id}/final/bedrock_models.json",
  "pricingS3Key": "executions/{execution-id}/final/bedrock_pricing.json",
  "totalModels": 108,
  "totalProviders": 17,
  "totalRegions": 20,
  "modelsWithPricing": 86,
  "modelsWithQuotas": 108,
  "durationMs": 12000
}
```

---

### 12. copy-to-latest

Copies final outputs to the `latest/` prefix for easy access.

**Input:**
```json
{
  "s3Bucket": "bedrock-profiler-data",
  "executionId": "arn:aws:states:...",
  "finalResult": {
    "modelsS3Key": "executions/{execution-id}/final/bedrock_models.json",
    "pricingS3Key": "executions/{execution-id}/final/bedrock_pricing.json"
  }
}
```

**Output:**
```json
{
  "status": "SUCCESS",
  "latestModelsKey": "latest/bedrock_models.json",
  "latestPricingKey": "latest/bedrock_pricing.json",
  "durationMs": 3000
}
```

**AWS APIs Called:**
- `s3:CopyObject`

---

## Error Handling

All functions follow a consistent error response format:

```json
{
  "status": "FAILED",
  "errorType": "ServiceException",
  "errorMessage": "Access denied to pricing API",
  "region": "us-east-1",
  "retryable": true
}
```

### Retryable Errors
- `ThrottlingException`
- `ProvisionedThroughputExceededException`
- `ServiceUnavailableException`
- Network timeouts

### Non-Retryable Errors
- `AccessDeniedException`
- `ValidationException`
- `InvalidParameterException`

---

## IAM Permissions Required

### Per-Function Permissions

| Function | Permissions |
|----------|-------------|
| pricing-collector | `pricing:GetProducts` |
| model-extractor | `bedrock:ListFoundationModels` |
| quota-collector | `service-quotas:ListServiceQuotas` |
| feature-collector | `bedrock:ListInferenceProfiles` |
| All functions | `s3:GetObject`, `s3:PutObject` on bucket |
| copy-to-latest | `s3:CopyObject` |
