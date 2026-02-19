# Bedrock Model Profiler - Backend Architecture

> Comprehensive technical reference for all backend Lambda functions, the Step Functions state machine, shared layer, and profiler configuration.

---

## Table of Contents

1. [State Machine Workflow](#1-state-machine-workflow)
2. [Shared Lambda Layer](#2-shared-lambda-layer)
3. [Lambda Functions](#3-lambda-functions)
   - [Region Discovery](#31-region-discovery)
   - [Pricing Collector](#32-pricing-collector)
   - [Pricing Aggregator](#33-pricing-aggregator)
   - [Model Extractor](#34-model-extractor)
   - [Model Merger](#35-model-merger)
   - [Quota Collector](#36-quota-collector)
   - [Pricing Linker](#37-pricing-linker)
   - [Regional Availability](#38-regional-availability)
   - [Feature Collector](#39-feature-collector)
   - [Token Specs Collector](#310-token-specs-collector)
   - [Model Enricher](#311-model-enricher)
   - [Mantle Collector](#312-mantle-collector)
   - [Final Aggregator](#313-final-aggregator)
   - [Gap Detection](#314-gap-detection)
   - [Self-Healing Agent](#315-self-healing-agent)
   - [Copy to Latest](#316-copy-to-latest)
4. [Profiler Configuration](#4-profiler-configuration)
5. [SAM Template / IAM Permissions](#5-sam-template--iam-permissions)
6. [S3 Key Map](#6-s3-key-map)

---

## 1. State Machine Workflow

**Definition file:** `backend/statemachine/bedrock-profiler.asl.json` (687 lines)

The workflow is a multi-wave Step Functions state machine that collects, enriches, and aggregates Bedrock model data. It runs daily at 6 AM UTC via EventBridge Scheduler.

### Execution Flow (numbered)

1. **DiscoverRegions** - Calls `RegionDiscoveryFunction` with `{}` input. Discovers which AWS regions have Bedrock inference profiles. On failure, catches all errors and continues with the failed result (which includes fallback regions).
2. **InitializeExecution** - Pass state. Sets up execution context from `$$.Execution.Id`, `$$.Execution.StartTime`, and region discovery output. Assembles `config` object with S3 bucket, service codes, and region lists.
3. **Wave1_ParallelCollection** - Parallel state with 3 branches:
   - **Branch 0: Pricing** - Map over `pricingServiceCodes` (MaxConcurrency=3) -> `PricingCollectorFunction` -> `PricingAggregatorFunction`
   - **Branch 1: Models** - Map over `modelRegions` (MaxConcurrency=2) -> `ModelExtractorFunction` -> `ModelMergerFunction`
   - **Branch 2: Quotas** - Map over `quotaRegions` (MaxConcurrency=10) -> `QuotaCollectorFunction`
4. **PrepareWave2** - Pass state. Restructures Wave 1 results from array indices to named keys: `pricingAggregated` (index 0), `modelsMerged` (index 1), `quotaResults` (index 2).
5. **Wave2_EnrichmentProcessing** - Parallel state with 6 branches:
   - **Branch 0:** `PricingLinkerFunction` - Links pricing to models
   - **Branch 1:** `RegionalAvailabilityFunction` - Discovers model availability per region
   - **Branch 2:** Map over `featureRegions` (MaxConcurrency=10) -> `FeatureCollectorFunction`
   - **Branch 3:** `TokenSpecsCollectorFunction` - Fetches token specs from LiteLLM
   - **Branch 4:** `ModelEnricherFunction` - Enriches with capabilities/use cases
   - **Branch 5:** Map over `featureRegions` (MaxConcurrency=10) -> `MantleCollectorFunction`
6. **PrepareAggregation** - Pass state. Restructures Wave 2 results from array indices to named keys.
7. **FinalAggregation** - Calls `FinalAggregatorFunction`. Merges all data sources into final JSON outputs.
8. **GapDetection** - Calls `GapDetectionFunction`. Analyzes pipeline output for data gaps. On failure, catches and skips to CopyToLatest.
9. **CheckShouldTriggerAgent** - Choice state. If `$.gapDetection.shouldTriggerAgent == true`, go to InvokeSelfHealingAgent; otherwise go to CopyToLatest.
10. **InvokeSelfHealingAgent** - Calls `SelfHealingAgentFunction`. Uses Bedrock Claude to analyze gaps and suggest config updates. On failure, catches and continues to CopyToLatest.
11. **CopyToLatest** - Calls `CopyToLatestFunction`. Copies final outputs to `latest/` prefix.
12. **ExecutionSucceeded** - Terminal Succeed state.

### Hardcoded Values in ASL

| Value | Location (line) | Description |
|-------|-----------------|-------------|
| `["AmazonBedrock", "AmazonBedrockService", "AmazonBedrockFoundationModels"]` | L30-34 | Pricing service codes |
| `["us-east-1", "us-west-2"]` | L35-38 | Model extraction regions |
| 16 quota regions | L39-56 | Quota collection regions |
| `MaxConcurrency: 3` | L75 | Pricing collector parallelism |
| `MaxConcurrency: 2` | L160 | Model extractor parallelism |
| `MaxConcurrency: 10` | L239 | Quota collector parallelism |
| `MaxConcurrency: 10` | L369, L479 | Feature/Mantle collector parallelism |
| `TimeoutSeconds: 300` | L114 | Pricing collector timeout |
| `TimeoutSeconds: 120` | L148 | Pricing aggregator timeout |
| `TimeoutSeconds: 60` | Various | Model extractor, quota, feature timeouts |
| `TimeoutSeconds: 180` | L583 | Final aggregator timeout |
| `TimeoutSeconds: 300` | L656 | Self-healing agent timeout |
| `"latest/bedrock_models.json"` | L597 | Previous models key for gap detection |

### Retry Configuration Patterns

- **Lambda/Throttle errors:** IntervalSeconds=5, MaxAttempts=3, BackoffRate=2
- **Aggressive throttle retry:** IntervalSeconds=10, MaxAttempts=5, BackoffRate=2 (pricing collector)
- **General retries:** IntervalSeconds=3, MaxAttempts=2, BackoffRate=2

---

## 2. Shared Lambda Layer

**Path:** `backend/layers/common/python/shared/`

All Lambda functions import from this shared layer. It provides:

### `config.py`

```python
RETRY_CONFIG = Config(
    retries={'max_attempts': 3, 'mode': 'adaptive'},
    connect_timeout=10,
    read_timeout=30
)
```

Global boto3 retry configuration used by all Lambdas.

### `s3_utils.py`

- **`get_s3_client()`** - Returns boto3 S3 client with `RETRY_CONFIG`.
- **`read_from_s3(s3_client, bucket, key, default_on_missing=None)`** - Reads JSON from S3. Raises `S3ReadError` on failure. If `default_on_missing` is provided, returns that value instead of raising when object is missing.
- **`write_to_s3(s3_client, bucket, key, data, content_type='application/json')`** - Writes JSON to S3 with `indent=2`. Raises `S3WriteError` on failure.

Custom exceptions: `S3ReadError`, `S3WriteError` - both carry `bucket`, `key`, and `original_error` attributes.

### `execution.py`

- **`parse_execution_id(execution_id_or_arn)`** - Extracts execution ID from a Step Functions ARN (takes the last segment after `:`). Passes through plain IDs unchanged.

### `validation.py`

- **`validate_required_params(event, required_params, handler_name)`** - Checks that all required keys exist in the event dict. Raises `ValidationError` with `missing_params` list.
- **`build_error_response(error, retryable=False)`** - Builds standardized `{status: FAILED, errorType, errorMessage, retryable}` dict.

### `config_loader.py`

- **`ConfigLoader`** class - Loads configuration from S3 (`config/profiler-config.json`) with fallback to embedded `DEFAULT_CONFIG`. Provides typed accessors for every config section:
  - `get_provider_aliases()`, `get_provider_patterns()`, `get_explicit_provider_names()`
  - `get_region_list(type)`, `get_region_locations()`, `get_region_coordinates()`
  - `get_model_families()`, `get_claude_variants()`, `get_nova_variants()`, `get_llama_sizes()`
  - `get_min_confidence_threshold()`, `get_agent_thresholds()`, `get_bedrock_model_id()`
- **`get_config_loader()`** - Global singleton factory.

The S3 bucket for config is resolved from env vars: `CONFIG_BUCKET` > `DATA_BUCKET` > `S3_BUCKET`.

---

## 3. Lambda Functions

### 3.1 Region Discovery

**File:** `backend/lambdas/region-discovery/handler.py` (145 lines)

**Purpose:** Dynamically discovers all AWS regions where Bedrock inference profiles are available, replacing hardcoded region lists.

**AWS APIs Used:**
- `ec2.describe_regions(AllRegions=False, Filters=[{opt-in-status: [opt-in-not-required, opted-in]}])`
- `bedrock.list_inference_profiles(maxResults=1)` (per region, to test availability)

**Input:** `{}` (no parameters required)

**Output:**
```json
{
  "status": "SUCCESS",
  "featureRegions": ["us-east-1", "us-west-2", ...],
  "totalRegions": 27,
  "allEnabledRegions": 33,
  "discoveryTimestamp": "2024-01-01T00:00:00Z"
}
```

**Key Logic:**
- Uses `ThreadPoolExecutor(max_workers=20)` to check all regions in parallel (line 77)
- If `bedrock.list_inference_profiles` returns `AccessDeniedException`, the region is counted as available (Bedrock exists but no access)
- Always returns `featureRegions` even on failure (fallback list of 16 hardcoded regions)

**Hardcoded Values:**
| Value | Line | Description |
|-------|------|-------------|
| `max_workers=20` | 77 | Thread pool size for parallel region checks |
| `maxResults=1` | 52 | Minimal query to test Bedrock availability |
| 16 fallback regions | 38-43, 136-141 | Used when EC2 DescribeRegions or Bedrock checks fail |

**Dependencies:** None (first step in pipeline)

**IAM Permissions:** `ec2:DescribeRegions`, `bedrock:ListInferenceProfiles`

---

### 3.2 Pricing Collector

**File:** `backend/lambdas/pricing-collector/handler.py` (283 lines)

**Purpose:** Collects pricing data from the AWS Pricing API for a single Bedrock service code. Also fetches from the AWS Bulk Pricing API for additional coverage (e.g., Stability AI models not in GetProducts).

**AWS APIs Used:**
- `pricing.get_products(ServiceCode=X, MaxResults=100, FormatVersion='aws_v1')`
- HTTP GET to `https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/{service_code}/current/{region}/index.json`

**Input:**
```json
{
  "serviceCode": "AmazonBedrock",
  "s3Bucket": "bucket-name",
  "s3Key": "executions/{id}/pricing/AmazonBedrock.json"
}
```

**Output S3 Key:** `executions/{id}/pricing/{serviceCode}.json`

**Output Data Structure:**
```json
{
  "metadata": { "serviceCode", "recordCount", "getProductsCount", "bulkApiCount", "collectionTimestamp", "pricingRegion" },
  "products": [ { "product": { "sku", "attributes" }, "terms": { "OnDemand": {...} } } ]
}
```

**Key Logic:**
- Paginates through `get_products` with a safety limit of `max_batches = 100` (line 126)
- Pauses 0.5s every 10 batches to avoid throttling (line 162)
- Merges Bulk API products with GetProducts, deduplicating by SKU (lines 233-238)
- Bulk API products are tagged with `"source": "bulk_pricing_api"` (line 104)

**Hardcoded Values:**
| Value | Line | Description |
|-------|------|-------------|
| `us-east-1` | 33 | Pricing API region (env var `PRICING_API_REGION`) |
| `MaxResults=100` | 135 | GetProducts page size |
| `max_batches=100` | 126 | Safety limit for pagination loops |
| `0.5` (sleep) | 162 | Throttle pause every 10 batches |
| `2` (sleep) | 168 | Retry delay on throttling |
| Bulk pricing URL template | 38 | `https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/...` |

**Dependencies:** None (Wave 1)

**IAM Permissions:** `pricing:GetProducts`, `pricing:DescribeServices`, `s3:PutObject`

---

### 3.3 Pricing Aggregator

**File:** `backend/lambdas/pricing-aggregator/handler.py` (827 lines)

**Purpose:** Merges pricing data from all three Bedrock service codes into a unified, provider-grouped structure with pricing groups (On-Demand, Batch, Long Context, etc.).

**AWS APIs Used:** None directly (reads from S3, writes to S3)

**Input:**
```json
{
  "s3Bucket": "bucket-name",
  "executionId": "exec-123",
  "pricingResults": [
    {"status": "SUCCESS", "serviceCode": "AmazonBedrock", "s3Key": "..."}
  ]
}
```

**Output S3 Key:** `executions/{id}/merged/pricing.json`

**Output Data Structure:**
```json
{
  "metadata": { "generated_at", "version", "total_pricing_entries", "providers_count", ... },
  "providers": {
    "Anthropic": {
      "anthropic.claude-3-5-sonnet": {
        "model_name": "Claude 3.5 Sonnet",
        "model_provider": "Anthropic",
        "pricing_types": ["token"],
        "primary_pricing_type": "token",
        "regions": {
          "us-east-1": {
            "pricing_groups": {
              "On-Demand": [...],
              "Batch": [...]
            },
            "total_dimensions": 10,
            "groups_count": 2
          }
        }
      }
    }
  }
}
```

**Key Logic / Algorithms:**

1. **Model name extraction** (`extract_raw_model_name`, line 289): 4-strategy approach:
   - `servicename` attribute (for AmazonBedrockFoundationModels)
   - `model` attribute (for AmazonBedrock/Service)
   - `titanModel` attribute (special case)
   - Fallback: extract from `usagetype` using camelCase splitting

2. **Provider inference** (`infer_provider`, line 443): 4-strategy approach:
   - Explicit `provider` attribute (normalized via `explicit_provider_names` from config)
   - Explicit provider names in model name
   - Generic keyword patterns from config `provider_patterns`
   - Fallback: search ALL attributes for provider keywords

3. **Pricing group determination** (`determine_pricing_group`, line 176): Classifies usage types into groups:
   - On-Demand, On-Demand Global, On-Demand Long Context, On-Demand Long Context Global
   - Batch, Batch Global, Batch Long Context, Batch Long Context Global
   - Provisioned Throughput, Custom Model

4. **Pricing type classification** (`determine_pricing_type`, line 57): Classifies as `token`, `image`, `image_generation`, `video_generation`, `video`, `video_second`, `model_unit`, `search_unit`

5. **Price normalization** (line 353-364): Converts per-million prices to per-thousand by detecting patterns like "per 1M", "Million Input Tokens", etc.

6. **Global-to-OnDemand fallback** (lines 645-649): If `On-Demand` group doesn't exist but `On-Demand Global` does, copies it as `On-Demand` for frontend compatibility. Same for Batch.

**Hardcoded Values:**
| Value | Line | Description |
|-------|------|-------------|
| Model name suffixes to clean | 243-248 | `(Amazon Bedrock Edition)`, `(Amazon Bedrock)` |
| Skip parts for usagetype parsing | 273 | `['mp', 'input', 'output', 'tokens', 'count', 'units', 'cache', 'read', 'write']` |
| Pricing type priority order | 627 | `video_generation > image_generation > ... > token > model_unit` |
| Custom Model Import indicators | 401-405 | `flan architecture`, `llama architecture`, etc. |
| Custom Model Training indicators | 409-411 | `customization-training`, `fine`, `finetun`, etc. |

**Dependencies:** Reads output from Pricing Collector (Wave 1)

**IAM Permissions:** `s3:GetObject`, `s3:PutObject`

---

### 3.4 Model Extractor

**File:** `backend/lambdas/model-extractor/handler.py` (480 lines)

**Purpose:** Extracts foundation models from a single AWS region using the Bedrock API. Also fetches console metadata via direct REST API with SigV4 signing to extract context windows, descriptions, languages, and categories.

**AWS APIs Used:**
- `bedrock.list_foundation_models()` (standard boto3 call)
- `GET https://bedrock.{region}.amazonaws.com/foundation-models` (direct REST with SigV4 and `x-console-consumer: true` header)

**Input:**
```json
{
  "region": "us-east-1",
  "s3Bucket": "bucket-name",
  "s3Key": "executions/{id}/models/us-east-1.json"
}
```

**Output S3 Key:** `executions/{id}/models/{region}.json`

**Output Data Structure:**
```json
{
  "metadata": { "region", "model_count", "collection_timestamp" },
  "models": [
    {
      "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
      "model_arn": "...",
      "model_name": "Claude 3.5 Sonnet",
      "model_provider": "Anthropic",
      "model_modalities": { "input_modalities": ["TEXT"], "output_modalities": ["TEXT"] },
      "streaming_supported": true,
      "inference_types_supported": ["ON_DEMAND"],
      "model_lifecycle": { "status": "ACTIVE" },
      "regions_available": ["us-east-1"],
      "documentation_links": {...},
      "console_metadata": {
        "max_context_window": 200000,
        "max_output_tokens": 8192,
        "description": "...",
        "languages": ["English", "French"],
        "use_cases": ["Complex analysis", "Code generation"]
      }
    }
  ]
}
```

**Key Logic:**

1. **Console metadata fetch** (`fetch_console_metadata`, line 163): Makes a SigV4-signed GET request to the Bedrock REST API with `x-console-consumer: true` header. Parses `consoleIDEMetadata` JSON field from each model summary to extract:
   - `maxContextWindow` (parsed via `parse_context_window_string`)
   - `fullDescription`, `shortDescription`
   - `supportedLanguages` (comma/and-separated string)
   - `supportedUseCases` (semicolon/comma-separated)
   - `maxTokensMaximum` from `converse` object

2. **Context window string parsing** (`parse_context_window_string`, line 59):
   ```python
   # "200K" -> 200000, "1M (beta)" -> 1000000, "1,000,000" -> 1000000
   match = re.match(r"^([\d.]+)\s*([KkMm])", value)
   if unit == "K": return int(num * 1000)
   elif unit == "M": return int(num * 1000000)
   ```

3. **Use case parsing** (`parse_use_cases`, line 95): Handles multiple formats:
   - Semicolon-separated groups with comma-separated items
   - Category names with parenthetical examples (NVIDIA-style)
   - Simple comma-separated lists
   - Deduplicates case-insensitively

**Hardcoded Values:**
| Value | Line | Description |
|-------|------|-------------|
| `x-console-consumer: true` | 187 | Header to get extended model metadata |
| `timeout=30` | 194 | Console metadata HTTP timeout |
| `min item length: 2` | 151 | Minimum use case string length |

**Dependencies:** Uses `get_config_loader()` for documentation links

**IAM Permissions:** `bedrock:ListFoundationModels`, `s3:PutObject`

---

### 3.5 Model Merger

**File:** `backend/lambdas/model-merger/handler.py` (259 lines)

**Purpose:** Merges and deduplicates models collected from multiple regions. Extracts context window sizes from model ID variants (e.g., `:200k`, `:18k`) before deduplication.

**AWS APIs Used:** None (S3 only)

**Input:**
```json
{
  "s3Bucket": "bucket-name",
  "executionId": "exec-123",
  "modelResults": [
    {"status": "SUCCESS", "region": "us-east-1", "s3Key": "..."}
  ]
}
```

**Output S3 Key:** `executions/{id}/merged/models.json`

**Output Data Structure:**
```json
{
  "metadata": { "total_models", "providers_count", "regions_processed", "collection_timestamp" },
  "providers": {
    "Anthropic": {
      "models": {
        "anthropic.claude-3-5-sonnet-20240620-v1:0": { ...model_data... }
      }
    }
  }
}
```

**Key Logic:**

1. **Context window variant deduplication** (lines 29-56): Strips `:NNNk` suffixes (e.g., `:200k`, `:18k`) to get the base model ID, then tracks the maximum context window from all variants:
   ```python
   def get_base_model_id(model_id: str) -> str:
       return re.sub(r':\d+k$', '', model_id)

   def parse_variant_size(model_id: str) -> int | None:
       match = re.search(r':(\d+)k$', model_id)
       if match: return int(match.group(1)) * 1000
   ```

2. **Region merging** (lines 107-124): For duplicate model IDs across regions, merges `regions_available` and `collection_metadata.regions_collected_from` as sorted union sets. Keeps first non-empty `console_metadata`.

3. **Variant context windows** (lines 132-136): Attaches `variant_context_window` to base models (the max size from any `:NNNk` variant).

**Hardcoded Values:** None significant.

**Dependencies:** Reads Model Extractor outputs

**IAM Permissions:** `s3:GetObject`, `s3:PutObject`

---

### 3.6 Quota Collector

**File:** `backend/lambdas/quota-collector/handler.py` (177 lines)

**Purpose:** Collects Bedrock service quotas from a single AWS region using the Service Quotas API.

**AWS APIs Used:**
- `service-quotas.list_service_quotas(ServiceCode='bedrock', MaxResults=100)`

**Input:**
```json
{
  "region": "us-east-1",
  "s3Bucket": "bucket-name",
  "s3Key": "executions/{id}/quotas/us-east-1.json"
}
```

**Output S3 Key:** `executions/{id}/quotas/{region}.json`

**Output Data Structure:**
```json
{
  "metadata": { "region", "quotaCount", "serviceCode", "collectionTimestamp" },
  "quotas": [
    {
      "quotaCode": "L-XXXXXX",
      "quotaName": "On-demand model inference requests per minute for Claude 3.5 Sonnet",
      "quotaArn": "arn:...",
      "value": 100,
      "unit": "None",
      "adjustable": true,
      "globalQuota": false,
      "usageMetric": {...},
      "period": {...},
      "region": "us-east-1"
    }
  ]
}
```

**Hardcoded Values:**
| Value | Line | Description |
|-------|------|-------------|
| `SERVICE_CODE = 'bedrock'` | 27 | Service Quotas service code |
| `MaxResults=100` | 52 | Page size |

**Dependencies:** None (Wave 1)

**IAM Permissions:** `servicequotas:ListServiceQuotas`, `servicequotas:GetServiceQuota`, `s3:PutObject`

---

### 3.7 Pricing Linker

**File:** `backend/lambdas/pricing-linker/handler.py` (535 lines)

**Purpose:** Links pricing data to models using fuzzy matching with provider-scoped matching, conflict detection, and enhanced normalization (V2 PORT features).

**AWS APIs Used:** None (S3 only)

**Input:**
```json
{
  "s3Bucket": "bucket-name",
  "executionId": "exec-123",
  "pricingS3Key": "executions/{id}/merged/pricing.json",
  "modelsS3Key": "executions/{id}/merged/models.json"
}
```

**Output S3 Key:** `executions/{id}/intermediate/models-with-pricing.json`

**Key Logic / Algorithms:**

1. **Provider-scoped matching** (`providers_match`, line 102): Only matches models within the same provider using alias lookups from config. Prevents cross-provider mismatches.

2. **Semantic conflict detection** (`has_semantic_conflict`, line 142): Blocks matches between:
   - Claude variants (haiku/sonnet/opus)
   - Nova variants (micro/lite/pro/premier/canvas/reel/sonic)
   - Llama size mismatches (8b vs 70b vs 405b)
   - Model type conflicts (embedding vs generation, rerank vs chat)
   - General size mismatches: blocks if difference > 30% of max size (line 225)
   ```python
   if max_size > 0 and abs(model_size - pricing_size) > max_size * 0.3:
       return True  # Conflict detected
   ```

3. **Enhanced normalization** (`normalize_model_id`, line 235): Provider-specific rules:
   - Qwen: removes `instruct` variations
   - DeepSeek: normalizes version formats (`V3` -> `3`)
   - Cohere: removes `model` keyword
   - Stability: normalizes `stable-diffusion` -> `sd`
   - Strips all separators for fuzzy comparison

4. **Match scoring** (`find_best_pricing_match`, line 274):
   - Exact match (after normalization): score = 1.0
   - Prefix match: score = 0.95
   - SequenceMatcher similarity: variable score
   - Prefers On-Demand pricing entries over non-On-Demand

**Hardcoded Values:**
| Value | Line | Description |
|-------|------|-------------|
| Suffixes to remove | 248 | `['-it', '-instruct', '-chat', '-v1', '-v2', '-v3', ':0', ':1', ':2']` |
| Exact match score | 329 | `1.0` |
| Prefix match score | 334 | `0.95` |
| Size variance threshold | 225 | `0.3` (30%) |

**Dependencies:** Reads Pricing Aggregator and Model Merger outputs

**IAM Permissions:** `s3:GetObject`, `s3:PutObject`

---

### 3.8 Regional Availability

**File:** `backend/lambdas/regional-availability/handler.py` (273 lines)

**Purpose:** Discovers model availability across all AWS regions using `ListFoundationModels` with explicit inference-type filtering (`ON_DEMAND` and `PROVISIONED`).

**AWS APIs Used:**
- `bedrock.list_foundation_models(byInferenceType='ON_DEMAND')` (per region)
- `bedrock.list_foundation_models(byInferenceType='PROVISIONED')` (per region)

**Input:**
```json
{
  "s3Bucket": "bucket-name",
  "executionId": "exec-123",
  "regions": ["us-east-1", "us-west-2", ...],
  "pricingS3Key": "..."  // accepted for backward compat, IGNORED
}
```

**Output S3 Key:** `executions/{id}/intermediate/regional-availability.json`

**Output Data Structure:**
```json
{
  "metadata": { "regions_with_bedrock", "total_models_tracked", "total_provisioned_models", "discovery_method": "api_on_demand_filtered" },
  "region_summary": {
    "us-east-1": { "bedrock_available": true, "models_in_region": 150, "providers": ["Anthropic", "Meta"], "model_count": 150 }
  },
  "model_availability": { "anthropic.claude-3-5-sonnet-v1:0": ["us-east-1", "us-west-2", ...] },
  "provisioned_availability": { "anthropic.claude-3-5-sonnet-v1:0": ["us-east-1"] }
}
```

**Key Logic:**
- Uses `ThreadPoolExecutor(max_workers=10)` for parallel region queries (line 94)
- Makes TWO API calls per region (ON_DEMAND + PROVISIONED)
- **Pricing data is intentionally excluded** from availability (see module docstring lines 14-23) because it adds ~130 phantom model IDs

**Hardcoded Values:**
| Value | Line | Description |
|-------|------|-------------|
| `max_workers=10` | 94 | Thread pool size |
| Retry config | 47-51 | 3 retries, adaptive mode, 5s connect, 30s read |

**Dependencies:** Reads dynamically discovered regions from Wave 1

**IAM Permissions:** `bedrock:ListFoundationModels`, `s3:GetObject`, `s3:PutObject`

---

### 3.9 Feature Collector

**File:** `backend/lambdas/feature-collector/handler.py` (159 lines)

**Purpose:** Collects inference profiles from a single region using the Bedrock API. These represent Cross-Region Inference (CRIS) capabilities.

**AWS APIs Used:**
- `bedrock.list_inference_profiles()` (paginated)

**Input:**
```json
{
  "region": "us-east-1",
  "s3Bucket": "bucket-name",
  "s3Key": "executions/{id}/features/us-east-1.json"
}
```

**Output S3 Key:** `executions/{id}/features/{region}.json`

**Output Data Structure:**
```json
{
  "metadata": { "region", "inferenceProfileCount", "collectionTimestamp" },
  "inferenceProfiles": [
    {
      "inferenceProfileId": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
      "inferenceProfileArn": "arn:...",
      "inferenceProfileName": "US Anthropic Claude 3.5 Sonnet",
      "description": "...",
      "status": "ACTIVE",
      "type": "SYSTEM_DEFINED",
      "models": [{"modelArn": "arn:..."}],
      "region": "us-east-1"
    }
  ]
}
```

**Dependencies:** None (Wave 2, uses dynamically discovered regions)

**IAM Permissions:** `bedrock:ListInferenceProfiles`, `s3:PutObject`

---

### 3.10 Token Specs Collector

**File:** `backend/lambdas/token-specs-collector/handler.py` (223 lines)

**Purpose:** Fetches token specifications (context window, max output tokens) from the LiteLLM open-source model database on GitHub.

**External APIs Used:**
- HTTP GET `https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json`

**Input:**
```json
{
  "s3Bucket": "bucket-name",
  "executionId": "exec-123",
  "modelsS3Key": "executions/{id}/merged/models.json"
}
```

**Output S3 Key:** `executions/{id}/intermediate/token-specs.json`

**Output Data Structure:**
```json
{
  "metadata": { "models_with_specs", "models_without_specs", "litellm_models_available", "source": "litellm" },
  "token_specs": {
    "anthropic.claude-3-5-sonnet-v1:0": {
      "context_window": 200000,
      "max_output_tokens": 8192,
      "source": "litellm",
      "litellm_verified": true
    }
  }
}
```

**Key Logic:**
1. **Bedrock filtering** (line 70): Filters LiteLLM's full model database to only entries with `bedrock` in the key or `litellm_provider`.
2. **Matching** (line 89): Tries exact match on normalized model ID first, then partial substring matching. Normalizes by stripping `bedrock/` prefix from LiteLLM keys.

**Hardcoded Values:**
| Value | Line | Description |
|-------|------|-------------|
| LiteLLM URL | 29 | `https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json` |
| User-Agent | 57 | `BedrockProfiler/1.0` |
| `timeout=30` | 61 | HTTP timeout |

**Dependencies:** Reads Model Merger output

**IAM Permissions:** `s3:GetObject`, `s3:PutObject`

---

### 3.11 Model Enricher

**File:** `backend/lambdas/model-enricher/handler.py` (264 lines)

**Purpose:** Enriches models with capabilities, use cases, and documentation links derived from model metadata (modalities, provider, model name).

**AWS APIs Used:** None (S3 only)

**Input:**
```json
{
  "s3Bucket": "bucket-name",
  "executionId": "exec-123",
  "modelsS3Key": "executions/{id}/merged/models.json"
}
```

**Output S3 Key:** `executions/{id}/enriched/models.json`

**Key Logic:**

1. **Capability extraction** (`extract_capabilities`, line 43): Derives capabilities from modalities:
   - TEXT in + TEXT out -> `text_generation`, `chat`
   - IMAGE in -> `vision`, `image_understanding`, `multimodal`
   - IMAGE out -> `image_generation`, `text_to_image`
   - EMBEDDING out -> `embeddings`, `semantic_search`
   - Model ID keywords: `code/codestral/devstral` -> `code_generation`
   - Keywords: `claude/sonnet/opus/nova/llama/mistral/command` -> `reasoning`
   - Keywords: `claude/nova/llama-3/mistral/command-r` -> `function_calling`

2. **Use case mapping** (`extract_use_cases`, line 101): Maps capabilities to use cases via a static dictionary (e.g., `chat` -> `conversational_ai`, `customer_support`, `virtual_assistants`).

3. **Documentation links** (`get_documentation_links`, line 128): Loads from config. Special handling for Nova models (uses `nova` key).

**Hardcoded Values:**
| Value | Line | Description |
|-------|------|-------------|
| Capability keyword lists | 83-92 | Model ID substrings that trigger specific capabilities |
| Capability-to-use-case map | 105-118 | Static mapping dictionary |

**Dependencies:** Reads Model Merger output, uses config for documentation links

**IAM Permissions:** `s3:GetObject`, `s3:PutObject`

---

### 3.12 Mantle Collector

**File:** `backend/lambdas/mantle-collector/handler.py` (242 lines)

**Purpose:** Collects model lists from the Mantle API endpoint for a single region using SigV4-signed HTTP requests.

**External APIs Used:**
- `GET https://bedrock-mantle.{region}.api.aws/v1/models` (SigV4 signed with `bedrock` service)

**Input:**
```json
{
  "region": "us-east-1",
  "s3Bucket": "bucket-name",
  "s3Key": "executions/{id}/mantle/{region}.json"
}
```

**Output S3 Key:** `executions/{id}/mantle/{region}.json`

**Output Data Structure:**
```json
{
  "metadata": { "region", "mantle_model_count", "collection_timestamp", "endpoint" },
  "mantle_models": [
    { "model_id": "anthropic.claude-3-5-sonnet", "model_name": "claude-3-5-sonnet", "provider": "anthropic", "region": "us-east-1" }
  ]
}
```

**Key Logic:**
- Creates SigV4-signed request with `bedrock` as the service name (line 64)
- Explicitly sets `Host` header before signing to prevent SigV4 mismatch (line 59)
- Handles both `{"data": [...]}` and flat array responses (lines 92-97)
- Model name is derived from model ID: `m.get("id", "").split(".")[-1]` (line 105)
- Uses module-level `_boto3_session` for credential reuse across invocations (line 34)

**Hardcoded Values:**
| Value | Line | Description |
|-------|------|-------------|
| `MANTLE_ENDPOINT_PATTERN` | 30 | `bedrock-mantle.{region}.api.aws` |
| `REQUEST_TIMEOUT_SECONDS` | 31 | `10` seconds |

**Dependencies:** None (Wave 2, uses dynamically discovered regions)

**IAM Permissions:** `bedrock-mantle:*`, `s3:PutObject`

---

### 3.13 Final Aggregator

**File:** `backend/lambdas/final-aggregator/handler.py` (1349 lines)

**Purpose:** The largest and most complex Lambda. Merges ALL collected data into the final comprehensive JSON outputs (`bedrock_models.json` and `bedrock_pricing.json`).

**AWS APIs Used:** None (S3 only)

**Input:** Takes all Wave 1 and Wave 2 results:
```json
{
  "s3Bucket": "...", "executionId": "...",
  "pricingS3Key": "...", "modelsS3Key": "...",
  "quotaResults": [...], "pricingLinked": {...},
  "regionalAvailability": {...}, "featureResults": [...],
  "tokenSpecs": {...}, "enrichedModels": {...}, "mantleResults": [...]
}
```

**Output S3 Keys:**
- `executions/{id}/final/bedrock_models.json`
- `executions/{id}/final/bedrock_pricing.json`

**Key Logic / Algorithms:**

1. **Context window 4-tier priority** (lines 729-795):
   - **Tier 1:** Console API metadata (`console_metadata.max_context_window`)
   - **Tier 2:** Model ID size variants (`variant_context_window` from model-merger)
   - **Tier 3:** `profiler-config.json` `context_window_specs` (with prefix matching)
   - **Tier 4:** LiteLLM token specs (last resort)
   - Special logic: if API returns extended context (e.g., 1M) but config has standard (200K), prefers config standard and stores extended separately.

2. **Quota matching** (`build_model_quotas`, line 467): Uses an inverted index (`_build_quota_index`) for O(1) lookup:
   - Pre-indexes all quotas by their normalized model reference
   - Builds multiple aliases per model from `_build_model_aliases` (line 360):
     - model_name, provider + model_name, name without parenthetical, model_id-derived alias, short name without size suffix, name without trailing version
   - Quota name parsing (`_extract_quota_model_ref`, line 314): Extracts model reference from patterns like "On-demand model inference requests per minute for Anthropic Claude 3.5 Sonnet"
   - Normalization (`_normalize_for_quota_matching`, line 249): Strips punctuation, dates, default version tags, context length qualifiers, collapses whitespace

3. **Provider prefix stripping** (`_PROVIDER_PREFIXES`, line 282): 18 known provider prefixes, ordered longest-first:
   ```python
   _PROVIDER_PREFIXES = [
       "anthropic ", "ai21 labs ", "stability.ai ", "stability ai ",
       "mistral ai ", "moonshot ai ", "writer ai ", "luma ai ",
       "twelvelabs ", "deepseek ", "minimax ", "openai ",
       "nvidia ", "amazon ", "google ", "cohere ", "meta ",
       "luma ", "qwen ", "mistral ",
   ]
   ```

4. **Cross-region inference** (`build_cross_region_inference`, line 165): Matches inference profiles to models by checking if model_id appears in any profile's model ARN. Deduplicates by `(profile_id, source_region)`.

5. **Batch inference detection** (`check_batch_inference`, line 596): Checks pricing data for any pricing group starting with "Batch". Calculates coverage percentage relative to the model's regional availability, capped at 100%.

6. **Consumption options** (`get_consumption_options`, line 523): Derives from inference types and pricing data:
   - `ON_DEMAND` -> `on_demand`
   - `PROVISIONED` -> `provisioned_throughput`
   - `INFERENCE_PROFILE` -> `cross_region_inference`
   - Checks pricing for `Batch` and `Provisioned Throughput` groups
   - Adds `mantle` if Mantle is supported

7. **Size categorization** (`get_size_category`, line 117):
   - `>= 128K` -> Large (green)
   - `>= 32K` -> Medium (blue)
   - `< 32K` -> Small (amber)
   - `None` -> Unknown (gray)

**Hardcoded Values:**
| Value | Line | Description |
|-------|------|-------------|
| Size thresholds | 121-126 | 128000 (Large), 32000 (Medium) |
| Size colors | 122-126 | `#10B981` (green), `#3B82F6` (blue), `#F59E0B` (amber), `#6B7280` (gray) |
| Provider prefixes | 282-303 | 18 known provider name prefixes |
| Default doc links | 910-915 | `model-ids-arns.html`, `bedrock/pricing/` |
| Mantle endpoint pattern | 228 | `bedrock-mantle.{region}.api.aws` |
| Consumption option order | 584-590 | `on_demand, batch, cross_region_inference, mantle, provisioned_throughput` |
| Version tag suffixes to strip | 144 | `["-v1:0", "-v1", "-v2:0", "-v2", ":0", ":1"]` |

**Dependencies:** ALL upstream Lambda outputs

**IAM Permissions:** `s3:GetObject`, `s3:PutObject`

---

### 3.14 Gap Detection

**File:** `backend/lambdas/gap-detection/handler.py` (347 lines)

**Purpose:** Analyzes pipeline output to detect gaps in data collection and determines if the self-healing agent should be triggered.

**AWS APIs Used:** None (S3 only)

**Input:**
```json
{
  "s3Bucket": "...", "executionId": "...",
  "modelsS3Key": "executions/{id}/final/bedrock_models.json",
  "pricingS3Key": "executions/{id}/final/bedrock_pricing.json",
  "previousModelsKey": "latest/bedrock_models.json"
}
```

**Output S3 Key:** `agent/gap-reports/{execution_id}/gap-analysis.json`

**Output:**
```json
{
  "status": "SUCCESS",
  "s3Key": "agent/gap-reports/{exec_id}/gap-analysis.json",
  "shouldTriggerAgent": true,
  "summary": { "modelsWithoutPricing": 12, "lowConfidenceMatches": 3, "newModelsDetected": 4, "unknownProviders": [] },
  "priority": "high"
}
```

**Key Logic:**

1. **Model analysis** (`analyze_models_data`, line 46):
   - Identifies models without pricing
   - Finds low-confidence pricing matches (below `low_confidence_threshold` from config, default 0.6)
   - Detects unknown providers (not in config's `provider_patterns`)

2. **New model detection** (`detect_new_models`, line 109): Compares current model IDs with previous `latest/bedrock_models.json`

3. **Trigger decision** (`determine_trigger_decision`, line 163):
   - Unmatched models >= 5 -> HIGH priority
   - Low confidence matches >= 3 -> MEDIUM priority
   - Unknown providers detected -> HIGH priority
   - New models detected -> MEDIUM priority

**Hardcoded Values (from config):**
| Value | Config Key | Default | Description |
|-------|-----------|---------|-------------|
| `unmatched_models_trigger` | `agent_configuration.thresholds` | 5 | Min unmatched to trigger agent |
| `low_confidence_threshold` | `agent_configuration.thresholds` | 0.6 | Below this = low confidence |
| `max_low_confidence_matches` | `agent_configuration.thresholds` | 3 | Min low-conf to trigger |
| `new_provider_trigger` | `agent_configuration.thresholds` | true | Trigger on unknown providers |

**Dependencies:** Reads Final Aggregator outputs and previous `latest/bedrock_models.json`

**IAM Permissions:** `s3:GetObject`, `s3:PutObject`

---

### 3.15 Self-Healing Agent

**File:** `backend/lambdas/self-healing-agent/handler.py` (418 lines)

**Purpose:** Uses Bedrock Claude Opus 4.5 to analyze gap reports and suggest configuration updates. Auto-applies safe suggestions and flags high-risk changes for review.

**AWS APIs Used:**
- `bedrock-runtime.invoke_model(modelId='us.anthropic.claude-opus-4-5-20251101-v1:0', ...)`

**Input:**
```json
{
  "s3Bucket": "...", "executionId": "...",
  "gapReportS3Key": "agent/gap-reports/{exec_id}/gap-analysis.json"
}
```

**Output S3 Key:** `agent/suggestions/{execution_id}/suggestions.json`

**Key Logic:**

1. **Prompt construction** (`build_analysis_prompt`, line 54): Builds a detailed prompt including:
   - Current provider patterns and aliases from config
   - Up to 20 models without pricing
   - Up to 10 low-confidence matches
   - Up to 20 new models
   - Unknown providers

2. **Claude invocation** (`invoke_claude`, line 144):
   - Model: `us.anthropic.claude-opus-4-5-20251101-v1:0` (from config)
   - Temperature: `0.2` (low for analytical responses)
   - Max tokens: `4096` (from config)
   - Handles markdown code block wrapping in response

3. **Auto-apply logic** (`apply_safe_suggestions`, line 201):
   - **Safe changes** (auto-applied): `provider_pattern_addition`, `provider_alias_addition`, `region_addition`, `documentation_link_addition`
   - **Requires review**: `provider_pattern_removal`, `provider_pattern_modification`, `threshold_change`
   - Max affected models for auto-apply: 20% (`max_models_affected_for_auto_apply: 0.2`)
   - Saves updated config to `config/profiler-config.json` and backs up to `config/config-history/profiler-config.{timestamp}.json`

**Hardcoded Values:**
| Value | Line | Description |
|-------|------|-------------|
| `temperature: 0.2` | 171 | Low temperature for analytical responses |
| `anthropic_version: bedrock-2023-05-31` | 163 | Anthropic API version |
| Config S3 key | 273 | `config/profiler-config.json` |
| Backup key pattern | 278 | `config/config-history/profiler-config.{timestamp}.json` |
| Prompt limits | 87, 94, 97 | First 20 unmatched models, 10 low-conf, 20 new models |

**Dependencies:** Reads Gap Detection output, current config from S3

**IAM Permissions:** `s3:GetObject`, `s3:PutObject`, `bedrock:InvokeModel` (Claude models)

---

### 3.16 Copy to Latest

**File:** `backend/lambdas/copy-to-latest/handler.py` (207 lines)

**Purpose:** Copies final outputs from the execution path to the `latest/` prefix for frontend consumption. Also stamps `date_added` on new models and preserves it for existing ones.

**AWS APIs Used:**
- `s3.copy_object(...)` with `MetadataDirective='REPLACE'`
- `s3.get_object(...)`, `s3.put_object(...)`

**Input:**
```json
{
  "s3Bucket": "...", "executionId": "...",
  "finalResult": {
    "modelsS3Key": "executions/{id}/final/bedrock_models.json",
    "pricingS3Key": "executions/{id}/final/bedrock_pricing.json"
  }
}
```

**Output S3 Keys (destinations):**
- `latest/bedrock_models.json`
- `latest/bedrock_pricing.json`
- `latest/manifest.json`

**Key Logic:**

1. **date_added stamping** (`stamp_date_added`, line 54):
   - Reads previous `latest/bedrock_models.json` to get existing `date_added` values
   - For existing models: preserves their original `date_added`
   - For new models: stamps today's date (`YYYY-MM-DD`)
   - Writes updated data back to the source key before the copy

2. **Manifest creation** (line 177): Creates `latest/manifest.json` with `lastUpdated`, `executionId`, and file references.

**Hardcoded Values:**
| Value | Line | Description |
|-------|------|-------------|
| `latest/bedrock_models.json` | 149 | Target models key |
| `latest/bedrock_pricing.json` | 150 | Target pricing key |
| `latest/manifest.json` | 184 | Manifest key |
| Content-Type | 39, 104, 186 | `application/json` |

**Dependencies:** Reads Final Aggregator outputs and previous latest data

**IAM Permissions:** `s3:GetObject`, `s3:PutObject`, `s3:CopyObject`

---

## 4. Profiler Configuration

**File:** `backend/config/profiler-config.json` (1131 lines)

This JSON file is the central configuration for the entire pipeline. It is loaded from S3 at runtime by the `ConfigLoader` class, with embedded defaults as fallback.

### Structure

```
{
  "version": "1.0.0-auto-updated-auto-updated",
  "last_updated": "2026-02-04T16:29:46Z",

  "provider_configuration": {
    "provider_aliases":          // Canonical name -> [alias1, alias2, ...] for pricing matching
    "provider_patterns":         // Provider -> [keywords] for model name detection
    "explicit_provider_names":   // lowercase -> Display Name mapping
    "provider_colors":           // Provider -> hex color (for frontend)
    "documentation_links":       // Provider -> {aws_bedrock_guide, pricing_guide, provider_guide}
  },

  "region_configuration": {
    "model_regions":             // Regions for ListFoundationModels (default: us-east-1, us-west-2)
    "quota_regions":             // Regions for Service Quotas collection (16 regions)
    "feature_regions":           // Regions for inference profiles / availability (27 regions)
    "region_locations":          // region-code -> "Human Name" (34 regions)
    "region_coordinates":        // region-code -> {lat, lng, name, geo} (34 regions)
    "aws_regions":               // [{value, label, geo}] for frontend dropdowns (34 regions)
    "geo_region_options":        // [{value, label}] for geo filter (8 options)
    "geo_prefix_map":            // geo-code -> region prefix (US->us-, EU->eu-, etc.)
  },

  "model_configuration": {
    "model_families":            // ["claude", "titan", "nova", ...] (9 families)
    "model_variants":            // ["haiku", "sonnet", "opus", ...] (17 variants)
    "claude_variants":           // ["haiku", "sonnet", "opus"] for conflict detection
    "nova_variants":             // ["micro", "lite", "pro", "premier", "canvas", "reel", "sonic"]
    "llama_sizes":               // ["8b", "70b", "405b", "11b", "90b", "1b", "3b"]
    "capability_keywords":       // ["code", "codestral", "devstral"]
    "reasoning_models":          // Model ID fragments that have reasoning capability
    "function_calling_models":   // Model ID fragments that support function calling
    "context_window_thresholds": // {small: 32000, medium: 128000, large: 500000}
    "context_window_specs":      // model-prefix -> {standard_context, extended_context, max_output, ...}
                                 // 20+ entries for Anthropic, Amazon, Meta, Mistral, AI21, Cohere, DeepSeek
  },

  "matching_configuration": {
    "min_confidence_threshold":  // 0.7 - minimum score for pricing match acceptance
    "size_variance_threshold":   // 0.3 - max allowed size difference ratio
    "suffixes_to_remove":        // ["-it", "-instruct", "-chat", "-v1", ...] for normalization
    "type_conflicts":            // [{group1: [...], group2: [...]}] for semantic conflict detection
  },

  "agent_configuration": {
    "bedrock_model_id":          // "us.anthropic.claude-opus-4-5-20251101-v1:0"
    "max_tokens":                // 4096
    "thresholds": {
      "unmatched_models_trigger":    // 5 - trigger agent if >= this many unmatched
      "low_confidence_threshold":    // 0.6 - below this = low confidence
      "new_provider_trigger":        // true - trigger on unknown providers
      "max_low_confidence_matches":  // 3 - trigger if >= this many low-conf
    },
    "auto_apply_rules": {
      "safe_changes":                // ["provider_pattern_addition", "provider_alias_addition", ...]
      "requires_review":             // ["provider_pattern_removal", "threshold_change", ...]
      "max_models_affected_for_auto_apply": // 0.2 (20%)
    }
  },

  "pricing_service_codes": ["AmazonBedrock", "AmazonBedrockService", "AmazonBedrockFoundationModels"]
}
```

### Context Window Specs (Notable Entries)

| Model Prefix | Standard | Extended | Max Output | Source |
|-------------|----------|----------|------------|--------|
| `anthropic.claude-opus-4-6` | 200K | 1M | 128K | anthropic_docs |
| `anthropic.claude-sonnet-4-5` | 200K | 1M | 64K | anthropic_docs |
| `anthropic.claude-sonnet-4` | 200K | 1M | 64K | anthropic_docs |
| `anthropic.claude-3-7-sonnet` | 200K | - | 64K (ext: 128K) | anthropic_docs |
| `anthropic.claude-3-5-sonnet` | 200K | - | 8192 | anthropic_docs |
| `amazon.nova-premier` | 300K | 1M | 5000 | aws_api |
| `amazon.nova-pro` | 300K | - | 5000 | aws_api |
| `amazon.nova-micro` | 128K | - | 5000 | aws_api |
| `ai21.jamba-1-5` | 256K | - | 4096 | litellm |
| `meta.llama3-1/2/3` | 128K | - | 2048 | litellm |

---

## 5. SAM Template / IAM Permissions

**File:** `infra/backend-template.yaml` (639 lines)

### Global Settings

- **Runtime:** Python 3.11
- **Default Timeout:** 60s
- **Default Memory:** 256 MB
- **Layer:** `SharedUtilsLayer` (from `backend/layers/common/`)
- **Environment Variables:** `ENVIRONMENT`, `DATA_BUCKET`, `LOG_LEVEL=INFO`

### Per-Lambda Resources

| Lambda | Timeout | Memory | IAM Permissions |
|--------|---------|--------|----------------|
| PricingCollector | 300s | 512 MB | `pricing:GetProducts`, `pricing:DescribeServices`, `s3:PutObject` |
| PricingAggregator | 120s | 1024 MB | `s3:GetObject`, `s3:PutObject` |
| ModelExtractor | 60s | 256 MB | `bedrock:ListFoundationModels`, `s3:PutObject` |
| ModelMerger | 60s | 512 MB | `s3:GetObject`, `s3:PutObject` |
| QuotaCollector | 60s | 256 MB | `servicequotas:ListServiceQuotas`, `servicequotas:GetServiceQuota`, `s3:PutObject` |
| PricingLinker | 120s | 1024 MB | `s3:GetObject`, `s3:PutObject` |
| RegionalAvailability | 60s | 512 MB | `bedrock:ListFoundationModels`, `s3:GetObject`, `s3:PutObject` |
| FeatureCollector | 60s | 256 MB | `bedrock:ListInferenceProfiles`, `s3:PutObject` |
| RegionDiscovery | 120s | 256 MB | `ec2:DescribeRegions`, `bedrock:ListInferenceProfiles` |
| TokenSpecsCollector | 120s | 512 MB | `s3:GetObject`, `s3:PutObject` |
| ModelEnricher | 120s | 512 MB | `s3:GetObject`, `s3:PutObject` |
| MantleCollector | 120s | 256 MB | `bedrock-mantle:*`, `s3:PutObject` |
| FinalAggregator | 180s | 2048 MB | `s3:GetObject`, `s3:PutObject` |
| GapDetection | 120s | 512 MB | `s3:GetObject`, `s3:PutObject` |
| SelfHealingAgent | 300s | 1024 MB | `s3:GetObject`, `s3:PutObject`, `bedrock:InvokeModel` (Claude models) |
| CopyToLatest | 60s | 256 MB | `s3:GetObject`, `s3:PutObject`, `s3:CopyObject` |

### Additional Infrastructure

- **S3 Bucket** (`DataBucket`): Versioned, 30-day lifecycle for `executions/`, 7-day noncurrent version expiration. Public access fully blocked.
- **CloudFront Bucket Policy**: Grants `s3:GetObject` on `latest/*` to CloudFront service principal (conditional on distribution ARN).
- **EventBridge Schedule**: `cron(0 6 * * ? *)` (6 AM UTC daily) with 30-minute flexible window.
- **Step Functions Logging**: ALL level with execution data, CloudWatch Logs with 30-day retention.
- **X-Ray Tracing**: Enabled on the state machine.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Environment` | `dev` | Allowed: dev, staging, prod |
| `ScheduleEnabled` | `true` | Enable/disable daily schedule |
| `CloudFrontDistributionArn` | `''` | Optional CloudFront ARN for bucket policy |
| `ExistingDataBucket` | `''` | Use existing bucket instead of creating new |

---

## 6. S3 Key Map

All data flows through S3. Here is the complete key structure:

```
s3://{bucket}/
  executions/{execution-id}/
    pricing/
      AmazonBedrock.json              # Pricing Collector output
      AmazonBedrockService.json
      AmazonBedrockFoundationModels.json
    models/
      us-east-1.json                  # Model Extractor output
      us-west-2.json
    quotas/
      us-east-1.json                  # Quota Collector output
      us-west-2.json
      ... (16 regions)
    merged/
      models.json                     # Model Merger output
      pricing.json                    # Pricing Aggregator output
    intermediate/
      models-with-pricing.json        # Pricing Linker output
      regional-availability.json      # Regional Availability output
      token-specs.json                # Token Specs Collector output
    enriched/
      models.json                     # Model Enricher output
    features/
      us-east-1.json                  # Feature Collector output
      ... (27 regions)
    mantle/
      us-east-1.json                  # Mantle Collector output
      ... (27 regions)
    final/
      bedrock_models.json             # Final Aggregator output (models)
      bedrock_pricing.json            # Final Aggregator output (pricing)

  agent/
    gap-reports/{execution-id}/
      gap-analysis.json               # Gap Detection output
    suggestions/{execution-id}/
      suggestions.json                # Self-Healing Agent output

  config/
    profiler-config.json              # Runtime configuration
    config-history/
      profiler-config.{timestamp}.json  # Config backups from self-healing

  latest/
    bedrock_models.json               # Copy to Latest (served via CloudFront)
    bedrock_pricing.json              # Copy to Latest (served via CloudFront)
    manifest.json                     # Execution metadata
```

---

## Data Flow Summary

```
                    DiscoverRegions
                          |
                   InitializeExecution
                          |
          +---------------+---------------+
          |               |               |
    [Pricing x3]    [Models x2]     [Quotas x16]
          |               |               |
  PricingAggregator  ModelMerger          |
          |               |               |
          +-------+-------+-------+-------+
                  |               |
         PrepareWave2      (quotas passed through)
                  |
    +------+------+------+------+------+
    |      |      |      |      |      |
  Pricing Regional Feature Token  Model  Mantle
  Linker  Avail.  Coll.  Specs  Enrich  Coll.
    |      |      |      |      |      |
    +------+------+------+------+------+
                  |
          FinalAggregator
                  |
           GapDetection
                  |
       [if gaps] SelfHealingAgent
                  |
           CopyToLatest
                   |
           ExecutionSucceeded
```

---

## 7. Hardcoded Values Inventory

Comprehensive table of every hardcoded value across all Lambdas and the state machine.

### State Machine (`backend/statemachine/bedrock-profiler.asl.json`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| bedrock-profiler.asl.json | 30-33 | `["AmazonBedrock", "AmazonBedrockService", "AmazonBedrockFoundationModels"]` | Pricing service codes | No (stable AWS service codes) |
| bedrock-profiler.asl.json | 35-37 | `["us-east-1", "us-west-2"]` | Model extraction regions | Yes — should use region discovery |
| bedrock-profiler.asl.json | 39-56 | 16 hardcoded quota regions | Quota collection regions | Yes — should use region discovery |
| bedrock-profiler.asl.json | 75 | `MaxConcurrency: 3` | Pricing collector parallelism | Maybe — depends on API limits |
| bedrock-profiler.asl.json | 160 | `MaxConcurrency: 2` | Model extractor parallelism | Maybe — conservative limit |
| bedrock-profiler.asl.json | 239 | `MaxConcurrency: 10` | Quota collector parallelism | Maybe — higher OK for quotas |
| bedrock-profiler.asl.json | 369, 479 | `MaxConcurrency: 10` | Feature/Mantle collector parallelism | Maybe |
| bedrock-profiler.asl.json | 114 | `TimeoutSeconds: 300` | Pricing collector timeout | No (appropriate for API pagination) |
| bedrock-profiler.asl.json | 148 | `TimeoutSeconds: 120` | Pricing aggregator timeout | No |
| bedrock-profiler.asl.json | Various | `TimeoutSeconds: 60` | Model extractor, quota, feature timeouts | No |
| bedrock-profiler.asl.json | 583 | `TimeoutSeconds: 180` | Final aggregator timeout | No |
| bedrock-profiler.asl.json | 656 | `TimeoutSeconds: 300` | Self-healing agent timeout | No |
| bedrock-profiler.asl.json | 597 | `"latest/bedrock_models.json"` | Previous models key for gap detection | No (convention) |
| bedrock-profiler.asl.json | Various | `IntervalSeconds=5, MaxAttempts=3, BackoffRate=2` | Retry configuration for Lambda errors | No |
| bedrock-profiler.asl.json | Various | `IntervalSeconds=10, MaxAttempts=5, BackoffRate=2` | Aggressive throttle retry for pricing | No |

### Region Discovery (`backend/lambdas/region-discovery/handler.py`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| handler.py | 23 | `region_name='us-east-1'` | EC2 client region for DescribeRegions | No (global API, any region works) |
| handler.py | 52 | `maxResults=1` | Minimal query to test Bedrock availability | No (intentionally minimal) |
| handler.py | 77 | `max_workers=20` | Thread pool size for parallel region checks | Maybe — could be config |
| handler.py | 38-43 | 16 fallback regions (first occurrence) | Fallback when EC2 DescribeRegions fails | Yes — should be in config |
| handler.py | 136-141 | 16 fallback regions (second occurrence, identical) | Fallback when entire discovery fails | Yes — duplicated, should be config |

### Pricing Collector (`backend/lambdas/pricing-collector/handler.py`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| handler.py | 33 | `'us-east-1'` (env var default) | Pricing API region | No (Pricing API only available in us-east-1/ap-south-1) |
| handler.py | 38 | `"https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/{service_code}/current/{region}/index.json"` | Bulk pricing API URL template | No (AWS public endpoint) |
| handler.py | 69 | `timeout=60` | Bulk pricing HTTP timeout | Maybe — could be config |
| handler.py | 126 | `max_batches=100` | Safety limit for pagination loops | Maybe |
| handler.py | 134 | `MaxResults=100` | GetProducts page size | No (API maximum) |
| handler.py | 135 | `'aws_v1'` | FormatVersion for GetProducts | No (required format) |
| handler.py | 162 | `time.sleep(0.5)` | Throttle pause every 10 batches | Maybe — could be config |
| handler.py | 168 | `time.sleep(2)` | Retry delay on throttling | Maybe |

### Pricing Aggregator (`backend/lambdas/pricing-aggregator/handler.py`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| handler.py | 86 | `'stable' in desc_lower and 'image' in desc_lower` | Stability AI image detection heuristic | Yes — fragile to new providers |
| handler.py | 243-248 | `['(Amazon Bedrock Edition)', '(Amazon Bedrock)', ...]` | Suffixes to clean from model names | Yes — new suffixes may appear |
| handler.py | 273 | `['mp', 'input', 'output', 'tokens', 'count', 'units', 'cache', 'read', 'write']` | Skip parts for usagetype parsing | Maybe |
| handler.py | 280 | `len(part) > 3` | Minimum part length for model name extraction | No (reasonable heuristic) |
| handler.py | 300-301 | `'Amazon Bedrock'`, `'Amazon Bedrock Service'` | Service names to skip in model name extraction | No (AWS-specific) |
| handler.py | 331 | `'USD'` | Default currency | No (Pricing API always USD) |
| handler.py | 357-361 | Per-million detection: `'per 1m'`, `'million'`, `'per 1,000,000'`, `'1000000'` | Price normalization patterns | Maybe — new patterns may appear |
| handler.py | 401-405 | `['flan architecture', 'llama architecture', 'inference for', ...]` | Custom Model Import indicators | Yes — new architectures may be added |
| handler.py | 408-411 | `['customization-training', 'fine', 'finetun', 'training', ...]` | Custom Model Training indicators | Maybe |
| handler.py | 626-627 | `['video_generation', 'image_generation', 'video_second', ...]` | Pricing type priority order | Maybe — new types could appear |
| handler.py | 779 | `'1.0.0'` | Output version string | No (metadata) |

### Model Extractor (`backend/lambdas/model-extractor/handler.py`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| handler.py | 182 | `f"https://bedrock.{region}.amazonaws.com/foundation-models"` | Console metadata REST API URL | No (AWS endpoint pattern) |
| handler.py | 186 | `"x-console-consumer": "true"` | Header to get extended model metadata | No (required for console data) |
| handler.py | 194 | `timeout=30` | Console metadata HTTP timeout | Maybe |
| handler.py | 83 | `r"^([\d.]+)\s*([KkMm])"` | Context window string parsing regex | No (standard K/M pattern) |
| handler.py | 151 | `len(item) < 2` | Minimum use case string length | No |

### Model Merger (`backend/lambdas/model-merger/handler.py`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| handler.py | 39 | `r':\d+k$'` | Context window variant suffix regex | No (established Bedrock pattern) |
| handler.py | 53 | `r':(\d+)k$'` | Variant size extraction regex | No |

### Quota Collector (`backend/lambdas/quota-collector/handler.py`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| handler.py | 27 | `SERVICE_CODE = 'bedrock'` | Service Quotas service code | No (AWS service code) |
| handler.py | 52 | `MaxResults=100` | Page size | No (API limit) |

### Pricing Linker (`backend/lambdas/pricing-linker/handler.py`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| handler.py | 248 | `['-it', '-instruct', '-chat', '-v1', '-v2', '-v3', ':0', ':1', ':2']` | Suffixes to remove during normalization | Yes — loaded from config but also hardcoded |
| handler.py | 225 | `0.3` (30% variance) | Size variance threshold for conflict detection | Yes — should use config value |
| handler.py | 329 | `1.0` | Exact match score | No |
| handler.py | 334 | `0.95` | Prefix match score | No |
| handler.py | 200-203 | `[['embed', 'embedding'], ['generator', ...]], [['rerank'], ...], ...` | Type conflict groups | Yes — should use config `type_conflicts` |

### Regional Availability (`backend/lambdas/regional-availability/handler.py`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| handler.py | 47-51 | `Config(retries={'max_attempts': 3, 'mode': 'adaptive'}, connect_timeout=5, read_timeout=30)` | Local RETRY_CONFIG (duplicates shared layer) | Yes — should use shared RETRY_CONFIG |
| handler.py | 94 | `max_workers=10` | Thread pool size | Maybe |

### Feature Collector (`backend/lambdas/feature-collector/handler.py`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| (none significant) | — | — | All values are either from API or config | — |

### Token Specs Collector (`backend/lambdas/token-specs-collector/handler.py`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| handler.py | 29 | `'https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json'` | LiteLLM data source URL | Yes — URL may change, should be config |
| handler.py | 57 | `'BedrockProfiler/1.0'` | User-Agent header | No (cosmetic) |
| handler.py | 61 | `timeout=30` | HTTP timeout | Maybe |
| handler.py | 22-26 | `Config(retries={'max_attempts': 3, ...}, connect_timeout=10, read_timeout=30)` | Local RETRY_CONFIG (duplicates shared layer) | Yes — should use shared RETRY_CONFIG |

### Model Enricher (`backend/lambdas/model-enricher/handler.py`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| handler.py | 83 | `['code', 'codestral', 'devstral']` | Code capability keywords | Yes — new code models may appear |
| handler.py | 87 | `['claude', 'sonnet', 'opus', 'nova', 'llama', 'mistral', 'command']` | Reasoning capability keywords | Yes — new reasoning models will appear |
| handler.py | 91 | `['claude', 'nova', 'llama-3', 'mistral', 'command-r']` | Function calling keywords | Yes — new models will support FC |
| handler.py | 105-118 | Entire `capability_to_use_cases` dict | Static mapping of capability -> use cases | Yes — should be in config |

### Mantle Collector (`backend/lambdas/mantle-collector/handler.py`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| handler.py | 30 | `"bedrock-mantle.{region}.api.aws"` | Mantle endpoint pattern | No (AWS endpoint convention) |
| handler.py | 31 | `REQUEST_TIMEOUT_SECONDS = 10` | HTTP timeout | Maybe |
| handler.py | 64 | `"bedrock"` (SigV4 service name) | SigV4 signing service | No |

### Final Aggregator (`backend/lambdas/final-aggregator/handler.py`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| handler.py | 121-126 | `128000` (Large), `32000` (Medium) | Context window size category thresholds | Yes — should use config `context_window_thresholds` |
| handler.py | 120-126 | `#10B981`, `#3B82F6`, `#F59E0B`, `#6B7280` | Size category colors | Yes — frontend concern, should be config |
| handler.py | 144 | `["-v1:0", "-v1", "-v2:0", "-v2", ":0", ":1"]` | Version suffixes to strip for config matching | Maybe — duplicates pricing-linker logic |
| handler.py | 228 | `"bedrock-mantle.{region}.api.aws"` | Mantle endpoint pattern (duplicated from mantle-collector) | Yes — duplicated |
| handler.py | 282-303 | 20 provider prefixes list (`_PROVIDER_PREFIXES`) | Known provider name prefixes for quota matching | Yes — should be config-driven |
| handler.py | 584-590 | `['on_demand', 'batch', 'cross_region_inference', 'mantle', 'provisioned_throughput']` | Consumption option ordering | Maybe |
| handler.py | 910-915 | `model-ids-arns.html`, `bedrock/pricing/` | Default documentation link URLs | No (stable AWS docs) |
| handler.py | 1044-1064 | 9 hardcoded provider prefixes (`anthropic.`, `amazon.`, etc.) | Provider prefix stripping for availability matching | Yes — should be config-driven |
| handler.py | 537-541 | `{"ON_DEMAND": "on_demand", "PROVISIONED": "provisioned_throughput", "INFERENCE_PROFILE": "cross_region_inference"}` | Inference type to consumption option mapping | No (AWS API enum values) |

### Gap Detection (`backend/lambdas/gap-detection/handler.py`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| handler.py | 58 | `0.6` (default) | Low confidence threshold fallback | No (loaded from config) |
| handler.py | 175 | `5` (default) | Unmatched models trigger fallback | No (loaded from config) |
| handler.py | 176 | `3` (default) | Max low confidence matches fallback | No (loaded from config) |
| handler.py | 262 | `'latest/bedrock_models.json'` | Default previous models key | No (convention) |

### Self-Healing Agent (`backend/lambdas/self-healing-agent/handler.py`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| handler.py | 170 | `0.2` | Claude temperature | Maybe — could be config |
| handler.py | 162 | `'bedrock-2023-05-31'` | Anthropic API version string | No (required by API) |
| handler.py | 273 | `'config/profiler-config.json'` | Config S3 key | No (must match ConfigLoader) |
| handler.py | 278 | `f"config/config-history/profiler-config.{timestamp}.json"` | Config backup key pattern | No (convention) |
| handler.py | 87, 94, 97 | `[:20]`, `[:10]`, `[:20]` | Prompt truncation limits (first N items) | Maybe |

### Copy to Latest (`backend/lambdas/copy-to-latest/handler.py`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| handler.py | 149 | `"latest/bedrock_models.json"` | Target models key | No (CloudFront convention) |
| handler.py | 150 | `"latest/bedrock_pricing.json"` | Target pricing key | No (CloudFront convention) |
| handler.py | 184 | `"latest/manifest.json"` | Manifest key | No |
| handler.py | 39, 104, 186 | `"application/json"` | Content-Type for S3 objects | No |

### Analytics (`backend/lambdas/analytics/handler.py`)

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| handler.py | 20 | `'bedrock-profiler-analytics-dev'` | Default DynamoDB table name | No (env var override) |
| handler.py | 21 | `'admins'` | Default admin Cognito group | No (env var override) |
| handler.py | 29-33 | `VALID_EVENT_TYPES` set (7 event types) | Valid analytics event types | Maybe — new events may be added |
| handler.py | 36 | `SESSION_BUCKET_TTL_DAYS = 7` | TTL for session data | Maybe |
| handler.py | 68 | `len(events) > 50` | Max events per batch | No (protection limit) |
| handler.py | 155 | `min(..., 365)` | Max days for dashboard query | No (protection limit) |

### Shared Layer

| File | Line | Value | Purpose | Should Be Dynamic? |
|------|------|-------|---------|---------------------|
| config.py | 3-6 | `Config(retries={'max_attempts': 3, 'mode': 'adaptive'}, connect_timeout=10, read_timeout=30)` | Global RETRY_CONFIG | Maybe |
| config_loader.py | 157 | `"config/profiler-config.json"` | Default config S3 key | No (must be stable) |
| config_loader.py | 168 | `CONFIG_BUCKET > DATA_BUCKET > S3_BUCKET` | Env var resolution order | No |
| config_loader.py | 18-142 | Entire `DEFAULT_CONFIG` dict | Embedded fallback config | No (necessary fallback) |
| config_loader.py | 342 | `'us.anthropic.claude-opus-4-5-20251101-v1:0'` | Fallback agent model ID | Yes — will change with model releases |

---

## 8. Redundancies & Simplification Opportunities

### 8.1 Duplicate RETRY_CONFIG Definitions

Three separate `RETRY_CONFIG` definitions exist:

1. **Shared layer** (`backend/layers/common/python/shared/config.py`): The canonical definition — `max_attempts=3, adaptive, connect_timeout=10, read_timeout=30`
2. **Regional Availability** (`backend/lambdas/regional-availability/handler.py:47-51`): Local copy — `max_attempts=3, adaptive, connect_timeout=5, read_timeout=30` (different connect timeout)
3. **Token Specs Collector** (`backend/lambdas/token-specs-collector/handler.py:22-26`): Local copy — identical to shared layer

**Fix:** Remove local `RETRY_CONFIG` from regional-availability and token-specs-collector. Import from shared layer. If regional-availability genuinely needs `connect_timeout=5`, create a named variant in the shared layer.

### 8.2 Duplicate S3 read/write Implementations

`token-specs-collector/handler.py:36-49` defines its own `read_from_s3` and `write_to_s3` functions that duplicate the shared layer's `s3_utils.py`. The local versions lack error handling (no `S3ReadError`/`S3WriteError`, no `default_on_missing` support).

**Fix:** Replace with `from shared import read_from_s3, write_to_s3`.

### 8.3 Duplicate Config Loader Initialization Pattern

Seven Lambdas repeat the exact same lazy-init pattern for `_config_loader`:

```python
_config_loader = None
def _get_config():
    global _config_loader
    if _config_loader is None:
        _config_loader = get_config_loader()
        _config_loader.load_config()
    return _config_loader
```

Found in: `pricing-aggregator/handler.py:30-39`, `model-extractor/handler.py:37-47`, `pricing-linker/handler.py:34-44`, `model-enricher/handler.py:29-38`, `final-aggregator/handler.py:29-38`, `gap-detection/handler.py:34-43`, `self-healing-agent/handler.py:37-46`.

**Fix:** Move `load_config()` call into `get_config_loader()` itself (or make `ConfigLoader.config` property auto-load, which it already does at `config_loader.py:214-216`). Then Lambdas can just call `get_config_loader()` directly without the wrapper.

### 8.4 Duplicate Documentation Link Logic

Documentation link resolution is implemented twice:

1. `model-extractor/handler.py:277-288` (`get_documentation_links`)
2. `model-enricher/handler.py:128-143` (`get_documentation_links`)

Both have identical logic: check for "nova" in model_id, then fall back to provider docs from config.

**Fix:** Move `get_documentation_links(model_id, provider)` into the shared layer or into `ConfigLoader`.

### 8.5 Duplicate Execution ID Parsing

`token-specs-collector/handler.py:159-160` manually parses execution IDs:
```python
if ':' in execution_id:
    execution_id = execution_id.split(':')[-1]
```

This duplicates `shared/execution.py`'s `parse_execution_id()` function.

**Fix:** Use `from shared import parse_execution_id`.

### 8.6 Duplicate Provider Prefix Lists

Provider prefixes for stripping/matching are hardcoded in multiple places:

1. `final-aggregator/handler.py:282-303` — `_PROVIDER_PREFIXES` list (20 entries)
2. `final-aggregator/handler.py:1044-1064` — `find_matching_availability` has 9 hardcoded `.replace("anthropic.", "")...` chains
3. `pricing-aggregator/handler.py:443-484` — `infer_provider` uses config patterns but also hardcodes fallback logic

**Fix:** Consolidate provider prefix lists into `profiler-config.json` and provide a `get_provider_prefixes()` accessor in `ConfigLoader`.

### 8.7 Mantle Endpoint Pattern Duplication

The Mantle endpoint pattern `"bedrock-mantle.{region}.api.aws"` is hardcoded in:

1. `mantle-collector/handler.py:30`
2. `final-aggregator/handler.py:228`

**Fix:** Define once in config or shared constants.

### 8.8 Overly Complex Pricing Aggregator `aggregate_pricing` Function

`pricing-aggregator/handler.py:500-684` (`aggregate_pricing`) is a 184-line function that:
- Extracts model info
- Infers providers
- Determines pricing groups and types
- Normalizes prices
- Structures output by provider/region
- Applies Global-to-OnDemand fallback
- Calculates group statistics

This is a God function. It should be decomposed into: (a) extraction, (b) grouping, (c) statistics computation.

### 8.9 Final Aggregator as a Monolith

`final-aggregator/handler.py` at 1,349 lines is the largest Lambda by far. It handles:
- Quota aggregation and indexing (lines 41-520)
- Cross-region inference building (lines 165-218)
- Mantle inference building (lines 221-229)
- Provisioned throughput building (lines 232-246)
- Quota name parsing and matching (lines 249-520)
- Consumption option determination (lines 523-593)
- Batch inference detection (lines 596-684)
- Context window resolution (4-tier priority, lines 714-796)
- Model transformation (lines 687-1002)
- Availability matching (lines 1005-1083)
- Final assembly (lines 1086-1349)

**Fix:** Extract quota matching, availability matching, and context window resolution into separate modules in the shared layer.

### 8.10 Unnecessary Data Transformations

1. **camelCase to snake_case round-trip:** Feature collector outputs `inferenceProfiles` (camelCase), then `final-aggregator/handler.py:78-79` handles both `inference_profiles` and `inferenceProfiles`. If feature-collector outputted snake_case consistently, this dual handling would be unnecessary.

2. **Pricing data pass-through:** `final-aggregator/handler.py:1322` copies pricing data as-is to the final output. The pricing aggregator already formats it correctly. This is a no-op copy that could be eliminated by having copy-to-latest directly copy from `merged/pricing.json`.

### 8.11 Duplicate Error Handling Boilerplate

Every Lambda handler has the same error handling structure:
```python
except Exception as e:
    logger.error(f"Failed to ...: {e}", exc_info=True)
    return {"status": "FAILED", "errorType": type(e).__name__, "errorMessage": str(e)}
```

**Fix:** Create a `@lambda_error_handler` decorator in the shared layer that wraps the handler and standardizes error responses.

---

## 9. Enhancement Opportunities

### 9.1 Missing Data We Could Collect

| Data Point | API/Source | Current Status |
|------------|-----------|---------------|
| **Model deprecation dates** | `ListFoundationModels` → `modelLifecycle.deprecationDate` | Not collected — only `status` is captured |
| **Guardrail compatibility** | `ListGuardrails` + `GetGuardrail` | Not collected — would show which models support guardrails |
| **Custom model base models** | `ListCustomModels` → `baseModelId` | Not collected — could show fine-tuning relationships |
| **Model evaluation results** | `ListEvaluationJobs` | Not collected — could show benchmark data |
| **Prompt caching support** | Converse API metadata / Pricing API `cache` usage types | Partially collected via pricing but not surfaced per-model |
| **Latency benchmarks** | InvokeModel/Converse with timing | Not collected — would require active invocation |
| **Model throughput limits** | Service Quotas → TPM quotas | Collected but not cross-referenced with model capacity info |
| **Knowledge base compatibility** | `ListFoundationModels` → integration capabilities | Not explicitly tracked |

### 9.2 Unused API Capabilities

1. **`ListFoundationModels` parameters not used:**
   - `byCustomizationType` — could filter to only fine-tunable models
   - `byOutputModality` — could pre-filter by modality instead of post-filtering
   - `byProvider` — could target specific provider updates

2. **`GetFoundationModel` API:** Not used at all. Could provide individual model details with richer metadata than the list endpoint.

3. **`ListInferenceProfiles` parameters:**
   - `typeEquals` — could filter SYSTEM_DEFINED vs APPLICATION profiles
   - `maxResults` — could optimize pagination

4. **Bulk Pricing API regions:** Only `us-east-1` is queried. `ap-south-1` also offers bulk pricing and might have different regional pricing data.

### 9.3 Performance Improvements

1. **Parallel S3 reads in final-aggregator:** Currently reads 6+ S3 objects sequentially (`models_with_pricing`, `availability_data`, `token_specs_data`, `pricing_data`, `enriched_models_data`, plus per-region quotas/features/mantle). Using `ThreadPoolExecutor` for concurrent reads could cut aggregation time by 50-70%.

2. **Pre-computed quota index caching:** `final-aggregator/handler.py:443-478` builds a quota index on every invocation. Since Lambda containers are reused, the index could be cached with a TTL check (quota data changes daily, not per-invocation).

3. **Batch S3 writes:** Multiple small S3 writes (quotas per region, features per region, mantle per region) could be batched using S3 multi-part upload or written as a single aggregate file per wave.

4. **Region discovery caching:** `region-discovery/handler.py` makes N API calls (one per region). Results could be cached in S3 with a TTL (regions don't change frequently) and only refreshed weekly.

5. **LiteLLM data caching:** `token-specs-collector/handler.py` fetches a ~2MB JSON file from GitHub on every execution. This could be cached in S3 and only refreshed if the ETag changes.

6. **Pricing collector deduplication:** Pricing data from 3 service codes has significant overlap. The deduplication by SKU in `pricing-collector/handler.py:233-238` could be moved to the aggregator to avoid downloading duplicate data.

### 9.4 Reliability Improvements

1. **No circuit breaker for external APIs:** The LiteLLM GitHub fetch (`token-specs-collector/handler.py:61`) and Mantle endpoint (`mantle-collector/handler.py:73`) have no circuit breaker. If GitHub is down, every execution wastes time and retries. A circuit breaker pattern (stored in DynamoDB or S3) that skips external calls after N consecutive failures would improve resilience.

2. **No partial retry for Map states:** If 1 of 16 quota regions fails, the entire quota collection is treated as partial success. The state machine could re-run only the failed regions.

3. **Self-healing agent has no rollback:** `self-healing-agent/handler.py:267-279` auto-applies config changes and creates a backup, but there's no mechanism to detect if the change caused a regression in the next run and automatically revert.

4. **No data validation between waves:** The pipeline trusts upstream outputs without validation. A corrupt pricing file from Wave 1 would cascade into bad pricing-linked data in Wave 2 and incorrect final output. Adding checksum/schema validation between waves would catch issues early.

5. **No dead-letter queue for failed executions:** Failed Step Functions executions are logged but not queued for manual review or automatic retry.

### 9.5 Data Quality Improvements

1. **Confidence score transparency:** The pricing-linker assigns confidence scores but they're not exposed in the frontend. Models with 0.7-0.8 confidence may have incorrect pricing displayed with no warning.

2. **Stale data detection:** No mechanism to detect if a model's pricing has been unchanged for an unusual number of runs (possible API failure silently returning cached data).

3. **Cross-validation:** Token specs from LiteLLM could be cross-validated against console metadata and config values. Disagreements should be flagged, not silently overridden by the priority system.

4. **Model ID normalization inconsistency:** Different Lambdas normalize model IDs differently — `pricing-linker` strips separators and suffixes, `final-aggregator` strips version suffixes, `model-merger` strips `:Nk` suffixes. A single canonical normalization function in the shared layer would prevent matching inconsistencies.

5. **Provider name canonicalization:** Provider names flow through multiple stages (Pricing API → pricing-aggregator → pricing-linker → final-aggregator) and may be normalized differently at each stage. A single `canonicalize_provider(name)` function in the shared layer would ensure consistency.

---

## 10. Backlog — Self-Healing Mechanism

### 10.1 Current State

The self-healing mechanism is implemented across two Lambdas:

**Gap Detection** (`backend/lambdas/gap-detection/handler.py`, 347 lines):
- Reads the final `bedrock_models.json` and `bedrock_pricing.json` outputs
- Compares current model IDs against the previous `latest/bedrock_models.json` to detect new models
- Identifies models without pricing matches, low-confidence matches (below 0.6), and unknown providers (not in config's `provider_patterns`)
- Applies configurable thresholds to decide whether to trigger the self-healing agent:
  - >= 5 unmatched models → HIGH priority
  - >= 3 low-confidence matches → MEDIUM priority
  - Unknown providers detected → HIGH priority
  - New models detected → MEDIUM priority
- Writes a detailed gap analysis report to `agent/gap-reports/{execution_id}/gap-analysis.json`
- Returns `shouldTriggerAgent: true/false` to the state machine

**Self-Healing Agent** (`backend/lambdas/self-healing-agent/handler.py`, 418 lines):
- Reads the gap analysis report and the current `profiler-config.json`
- Constructs a detailed prompt for Claude Opus 4.5 including: current provider patterns/aliases, up to 20 unmatched models, up to 10 low-confidence matches, up to 20 new models, and unknown providers
- Invokes `us.anthropic.claude-opus-4-5-20251101-v1:0` via Bedrock with temperature 0.2
- Parses Claude's structured JSON response containing analysis and suggestions
- Auto-applies "safe" changes: `provider_pattern_addition`, `provider_alias_addition`, `region_addition`, `documentation_link_addition`
- Flags for review: `provider_pattern_removal`, `provider_pattern_modification`, `threshold_change`
- Has a safety cap: won't auto-apply changes affecting >20% of models
- Backs up the original config to `config/config-history/` before applying changes
- Writes suggestions to `agent/suggestions/{execution_id}/suggestions.json`

### 10.2 The Problem

When new providers or models appear in Amazon Bedrock, the profiler may silently fail to handle them correctly:

1. **Pricing format divergence:** New providers may use different pricing attribute naming conventions (e.g., a new `servicename` format, new `usagetype` patterns, per-character pricing instead of per-token). The pricing-aggregator's `extract_raw_model_name` (4-strategy approach) and `determine_pricing_type` would fail to classify them correctly.

2. **Model ID format evolution:** The fuzzy matching in pricing-linker relies on patterns like `provider.model-name-version:N`. If a new provider uses a different ID convention (e.g., no dots, UUID-based IDs), the `normalize_model_id` function would produce garbage output, and `SequenceMatcher` similarity would return low scores.

3. **New consumption types:** When `INFERENCE_PROFILE` was first introduced, the `get_consumption_options` mapping (`final-aggregator/handler.py:537-541`) had to be manually updated. Future types (e.g., `SERVERLESS`, `DEDICATED`) would require the same manual intervention.

4. **Provider detection gaps:** The `infer_provider` function in pricing-aggregator searches model names and attributes for known provider keywords. A completely new provider with no matching keywords would be classified as "Unknown Models" with incorrect grouping.

5. **Missing documentation links:** New providers have no documentation links configured, so models show generic fallback URLs that may not be helpful.

6. **Quota name format changes:** The quota matching in final-aggregator parses quota names like "On-demand model inference requests per minute for Anthropic Claude 3.5 Sonnet". If AWS changes the naming format, the regex-based `_extract_quota_model_ref` function would fail to extract model references.

7. **Context window specs:** New models don't have entries in `profiler-config.json`'s `context_window_specs`, falling back to LiteLLM data (which may also not have the model yet), resulting in `null` context windows.

### 10.3 Future Vision

A GenAI-powered self-healing system that can autonomously detect, diagnose, and fix data gaps:

**Architecture:**

```
Daily Execution
       |
  [Normal Pipeline Waves 1-2]
       |
  FinalAggregator
       |
  GapDetection ──── Produces gap-analysis.json
       |
  [shouldTriggerAgent?]
       |
  SelfHealingAgent
       |
  ┌────┴────┐
  │ Analyze │── Reads: gap report, current config, project docs, Lambda source
  │  (GenAI)│   (context: README, BACKEND_ARCHITECTURE.md, pricing-linker logic)
  └────┬────┘
       │
  ┌────┴─────────┐
  │ Generate Fix │── Produces: config patches, matching rule updates, code patches
  └────┬─────────┘
       │
  ┌────┴──────────────┐
  │ Validate & Apply  │── Safe changes: auto-apply with rollback capability
  │                    │── Risky changes: create PR or flag for review
  └────┬──────────────┘
       │
  CopyToLatest (with patched config active)
```

The agent would have access to:
- The full `profiler-config.json` and its schema
- Gap analysis data (unmatched models, new providers, low-confidence matches)
- The backend architecture documentation (this document)
- Source code of relevant Lambda handlers (pricing-aggregator patterns, pricing-linker normalization)
- History of previous self-healing actions and their outcomes

### 10.4 Key Scenarios It Should Handle

1. **New provider appears in Bedrock (e.g., "Reka AI"):** Agent should detect unknown provider models, analyze model ID patterns, add `provider_patterns` entries (e.g., `"Reka": ["reka"]`), add `provider_aliases` entries, add `explicit_provider_names` mapping, and add placeholder `documentation_links`.

2. **Existing provider adds a new model family (e.g., Anthropic ships "Claude Code"):** Agent should detect the unmatched models, verify the provider is known, analyze why pricing-linker failed (e.g., "code" keyword triggers false `code_generation` capability but doesn't match pricing), and suggest normalization rule updates.

3. **AWS changes pricing attribute format:** Agent should detect a sudden spike in "Unknown Models" in pricing-aggregator output, analyze the raw pricing products to identify the new format, and suggest updates to `extract_raw_model_name` strategies or `determine_pricing_type` patterns.

4. **New consumption type appears (e.g., `DEDICATED_THROUGHPUT`):** Agent should detect models with unknown inference types, update the `type_mapping` in `get_consumption_options`, and suggest frontend UI changes.

5. **Quota naming format changes:** Agent should detect a drop in quota match rate, compare current quota names against expected patterns, and suggest updates to `_extract_quota_model_ref` regex patterns or `_PROVIDER_PREFIXES` list.

6. **LiteLLM data source goes stale or changes schema:** Agent should detect that token-specs matching rate dropped significantly, analyze the LiteLLM data format for schema changes, and suggest parser updates or fallback data sources.

7. **Context window specs need updating:** Agent should detect models with `null` context windows, cross-reference with console API data and LiteLLM data, and suggest new `context_window_specs` entries in `profiler-config.json`.

### 10.5 Architecture Sketch

```
┌─────────────────────────────────────────────────────────────┐
│                    Step Functions Workflow                    │
│                                                              │
│  [Wave1] ──> [Wave2] ──> FinalAggregator ──> GapDetection  │
│                                                    │         │
│                                          shouldTriggerAgent? │
│                                                    │         │
│                                          ┌─────────▼───────┐ │
│                                          │  SelfHealingAgent│ │
│                                          └─────────┬───────┘ │
│                                                    │         │
│                                          ┌─────────▼───────┐ │
│                                          │  CopyToLatest   │ │
│                                          └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘

SelfHealingAgent internals:
┌─────────────────────────────────────────────────────────┐
│                                                          │
│  1. Load context:                                        │
│     - gap-analysis.json                                  │
│     - profiler-config.json                               │
│     - BACKEND_ARCHITECTURE.md (from S3 or embedded)      │
│     - Previous healing actions history                   │
│                                                          │
│  2. Construct analysis prompt with all context            │
│                                                          │
│  3. Invoke Claude (Bedrock)                              │
│     - Structured output: analysis + suggestions          │
│     - Tool use: validate_config, test_matching_rule      │
│                                                          │
│  4. Classify suggestions:                                │
│     ┌──────────────┐  ┌──────────────┐                   │
│     │ Safe (auto)  │  │ Risky (review)│                  │
│     │ - Add pattern│  │ - Modify rule │                  │
│     │ - Add alias  │  │ - Remove rule │                  │
│     │ - Add region │  │ - Change logic│                  │
│     └──────┬───────┘  └──────┬───────┘                   │
│            │                 │                            │
│     Apply to config   Write to suggestions.json          │
│     + create backup   (for human review)                 │
│                                                          │
│  5. Validate applied changes:                            │
│     - Re-run matching logic with new config              │
│     - Verify improvement (more matches, higher scores)   │
│     - If regression detected: rollback from backup       │
│                                                          │
│  6. Record outcome in healing-history.json               │
│     - What was detected, what was applied, result        │
│     - Used for learning in future invocations            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 10.6 Risks and Safeguards

| Risk | Mitigation |
|------|-----------|
| **Hallucinated config changes** — Claude suggests invalid config paths or values that break the pipeline | Schema validation before applying any change; test the change against a sample of models before committing |
| **Cascading failures** — A bad auto-applied config change causes the next day's run to produce worse results | Automatic rollback: if the next run's gap report is worse than the previous one, revert to the backup config |
| **Cost runaway** — Claude invocations on every daily run when gaps are persistent | Cooldown period: don't re-trigger if the same gaps were already analyzed in the last N runs; exponential backoff on repeated failures |
| **Over-broad patterns** — Adding a pattern like `"r1"` for DeepSeek could match unrelated models | Affected-models validation: the agent must verify that a new pattern only matches the intended models, not existing ones from other providers |
| **Config drift** — Repeated auto-updates make the config difficult to understand or review | Config history with diffs: every auto-update includes a human-readable summary of what changed and why; periodic config review alerts |
| **Prompt injection via model names** — Malicious model names could inject instructions into the Claude prompt | Sanitize all model names/IDs before including in prompts; use structured tool-use instead of free-form JSON generation |
| **Silent degradation** — The agent "fixes" issues by lowering thresholds rather than addressing root causes | Threshold changes require human review (already in `requires_review` list); log confidence score trends over time |
| **Race condition** — Concurrent executions both modify config simultaneously | Use S3 conditional writes (If-Match ETag) for config updates; state machine ensures sequential execution via single EventBridge schedule |

### 10.7 Status

**Status: BACKLOG — do NOT implement yet.**

The current implementation (gap-detection + self-healing-agent) provides a foundation with:
- Gap detection and analysis (fully functional)
- Claude-based suggestion generation (functional but lightly tested)
- Safe auto-apply with backup (implemented)
- State machine integration (implemented)

Exploration has started but the system is **unverified in production**. Key gaps before production use:
- No regression detection / automatic rollback
- No healing history tracking for learning
- No validation of applied changes before committing
- No integration testing with intentionally broken config
- Claude prompt needs refinement based on real-world gap patterns
- No alerting/notification when changes are auto-applied
- Documentation context loading not yet implemented (currently only config context)

