# Bedrock Model Profiler - Data Sources Documentation

> **Version:** 1.0  
> **Last Updated:** March 2026  
> **Maintainer:** AWS Bedrock Profiler Team

## Executive Summary

The Bedrock Model Profiler aggregates data from **7 distinct data sources** to provide comprehensive information about Amazon Bedrock foundation models. The system collects:

| Data Category | Source Type | Data Points |
|--------------|-------------|-------------|
| **Pricing** | AWS Pricing API + Bulk Pricing API | ~3,500 pricing entries |
| **Models** | Bedrock ListFoundationModels API | ~108 models |
| **Console Metadata** | Bedrock REST API (x-console-consumer) | Context windows, capabilities, features |
| **Service Quotas** | Service Quotas API | ~45 quotas per region |
| **Inference Profiles** | Bedrock ListInferenceProfiles API | Cross-region inference profiles |
| **Token Specs** | LiteLLM GitHub Repository | Context windows, max output tokens |
| **Lifecycle Data** | AWS Documentation (Web Scraping) | Active/Legacy/EOL status |
| **Mantle Models** | Bedrock Mantle API | OpenAI-compatible model availability |
| **Regional Availability** | Bedrock API (filtered) | ON_DEMAND + PROVISIONED availability |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          STEP FUNCTIONS WORKFLOW (Daily @ 6 AM UTC)                 │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                    ┌───────────────┐
                                    │DiscoverRegions│
                                    │   Lambda      │
                                    └───────┬───────┘
                                            │
                                            ▼
                              ┌─────────────────────────┐
                              │   InitializeExecution   │
                              │   (Pass State)          │
                              └───────────┬─────────────┘
                                          │
                                          ▼
                              ┌─────────────────────────┐
                              │      ConfigSync         │
                              │   (Frontend Config)     │
                              └───────────┬─────────────┘
                                          │
           ┌──────────────────────────────┼──────────────────────────────┐
           │                              │                              │
           ▼                              ▼                              ▼
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│  WAVE 1 - PARALLEL  │      │  WAVE 1 - PARALLEL  │      │  WAVE 1 - PARALLEL  │
├─────────────────────┤      ├─────────────────────┤      ├─────────────────────┤
│ ┌─────────────────┐ │      │ ┌─────────────────┐ │      │ ┌─────────────────┐ │
│ │pricing-collector│ │      │ │ model-extractor │ │      │ │ quota-collector │ │
│ │   (x3 parallel) │ │      │ │  (x27 regions)  │ │      │ │   (xN regions)  │ │
│ └────────┬────────┘ │      │ └────────┬────────┘ │      │ └────────┬────────┘ │
│          ▼          │      │          ▼          │      │          │          │
│ ┌─────────────────┐ │      │ ┌─────────────────┐ │      │          │          │
│ │pricing-aggregator│ │      │ │  model-merger   │ │      │          │          │
│ └────────┬────────┘ │      │ └────────┬────────┘ │      │          │          │
└──────────┼──────────┘      └──────────┼──────────┘      └──────────┼──────────┘
           │                            │                            │
           └────────────────────────────┼────────────────────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │   PrepareWave2    │
                              └─────────┬─────────┘
                                        │
    ┌───────────────┬───────────────┬───┴────┬───────────────┬───────────────┐
    │               │               │        │               │               │
    ▼               ▼               ▼        ▼               ▼               ▼
┌────────┐   ┌──────────┐   ┌──────────┐ ┌────────┐   ┌──────────┐   ┌──────────┐
│pricing-│   │regional- │   │feature-  │ │token-  │   │ mantle-  │   │lifecycle-│
│linker  │   │availability│  │collector │ │specs-  │   │collector │   │collector │
│        │   │          │   │(xN rgns) │ │collector│   │(xN rgns) │   │          │
└───┬────┘   └────┬─────┘   └────┬─────┘ └───┬────┘   └────┬─────┘   └────┬─────┘
    │             │              │           │             │              │
    └─────────────┴──────────────┴───────────┴─────────────┴──────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │  PrepareAggregation│
                              └─────────┬─────────┘
                                        │
                              ┌─────────▼─────────┐
                              │ final-aggregator  │
                              └─────────┬─────────┘
                                        │
                              ┌─────────▼─────────┐
                              │   gap-detection   │
                              └─────────┬─────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │ shouldTriggerAgent == true? │
                         └──────────────┬──────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │self-healing-agent │ (conditional)
                              └─────────┬─────────┘
                                        │
                              ┌─────────▼─────────┐
                              │  copy-to-latest   │
                              └─────────┬─────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │   S3: latest/     │
                              │ bedrock_models.json│
                              │ bedrock_pricing.json│
                              └─────────┬─────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │    CloudFront     │
                              │     Frontend      │
                              └───────────────────┘
```

---

### Caching Architecture

The pipeline uses S3-based caching to minimize redundant API calls:

| Cache Type | Producer | Consumer | TTL | Location |
|------------|----------|----------|-----|----------|
| **Model Data** | `model-extractor` | `regional-availability` | Per-execution | `cache/list_foundation_models_{region}.json` |
| **Inference Profiles** | `region-discovery` | `feature-collector` | Per-execution | `cache/inference_profiles_{region}.json` |
| **LiteLLM Data** | `token-specs-collector` | Self | 24 hours | `cache/litellm_model_prices.json` |
| **Lifecycle Data** | `lifecycle-collector` | Self | 24 hours | `cache/lifecycle_data.json` |

---

## AWS APIs Section

### 1. AWS Pricing API

**Lambda:** `pricing-collector`

#### GetProducts API
```python
# boto3 method
pricing_client = boto3.client('pricing', region_name='us-east-1')
response = pricing_client.get_products(
    ServiceCode='AmazonBedrock',  # or AmazonBedrockService, AmazonBedrockFoundationModels
    MaxResults=100,
    FormatVersion='aws_v1'
)
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ServiceCode` | `AmazonBedrock`, `AmazonBedrockService`, `AmazonBedrockFoundationModels` | Three service codes collected in parallel |
| `MaxResults` | 100 | Pagination limit |
| `FormatVersion` | `aws_v1` | Response format version |

**Fields Extracted:**
```json
{
  "product": {
    "sku": "...",
    "attributes": {
      "servicename": "Claude 3.5 Sonnet",
      "model": "...",
      "regionCode": "us-east-1",
      "usagetype": "USE1-Claude3Sonnet-input-tokens",
      "inferenceType": "on-demand",
      "operation": "InvokeModel",
      "provider": "Anthropic"
    }
  },
  "terms": {
    "OnDemand": {
      "priceDimensions": {
        "pricePerUnit": { "USD": "0.003" },
        "unit": "1K tokens",
        "description": "..."
      }
    }
  }
}
```

**Transformation Logic:**
- Prices per-million are converted to per-thousand (÷1000)
- Model names cleaned: Remove "(Amazon Bedrock Edition)" suffix
- Provider inferred from model name patterns
- Pricing groups determined: On-Demand, Batch, Provisioned, Long Context, Global

#### Bulk Pricing API (Public HTTPS)
```python
# URL pattern
url = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/{service_code}/current/{region}/index.json"

# Example
url = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrockFoundationModels/current/us-east-1/index.json"
```

**Purpose:** Captures models not in GetProducts API (e.g., Stability AI models)

---

### 2. Bedrock ListFoundationModels API

**Lambdas:** `model-extractor`, `regional-availability`

```python
# boto3 method
bedrock_client = boto3.client('bedrock', region_name='us-east-1')
response = bedrock_client.list_foundation_models()

# For regional availability (filtered)
response = bedrock_client.list_foundation_models(byInferenceType='ON_DEMAND')
response = bedrock_client.list_foundation_models(byInferenceType='PROVISIONED')
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `region_name` | All 27 Bedrock regions (for extraction and availability) | All-region collection for models and availability |
| `byInferenceType` | `ON_DEMAND`, `PROVISIONED` | Filters for regional availability |

**Fields Extracted from modelSummaries:**
```json
{
  "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
  "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
  "modelName": "Claude 3 Sonnet",
  "providerName": "Anthropic",
  "inputModalities": ["TEXT", "IMAGE"],
  "outputModalities": ["TEXT"],
  "responseStreamingSupported": true,
  "customizationsSupported": ["FINE_TUNING"],
  "inferenceTypesSupported": ["ON_DEMAND", "PROVISIONED"],
  "modelLifecycle": { "status": "ACTIVE" }
}
```

**Transformation Logic:**
- Convert camelCase to snake_case (e.g., `modelId` → `model_id`)
- Build nested `model_modalities` structure
- Extract `extraction_regions` for audit tracking

---

### 3. Bedrock Console REST API (SigV4 Signed)

**Lambda:** `model-extractor`

```python
# Direct REST API call with SigV4 signing
url = f"https://bedrock.{region}.amazonaws.com/foundation-models"

headers = {
    "Content-Type": "application/json",
    "x-console-consumer": "true"  # KEY: Enables extended metadata
}

# SigV4 signing
request = AWSRequest(method="GET", url=url, headers=headers)
SigV4Auth(credentials, "bedrock", region).add_auth(request)
```

**Fields Extracted from consoleIDEMetadata:**
```json
{
  "modelFamily": "Claude 3",
  "guardrailsSupported": true,
  "batchSupported": {},
  "consoleIDEMetadata": {
    "description": {
      "maxContextWindow": "200K",
      "fullDescription": "...",
      "shortDescription": "...",
      "supportedLanguages": "English, French, German, Spanish...",
      "supportedUseCases": "Complex agentic systems; visual analysis...",
      "modelAttributes": "Text generation, Code generation",
      "releaseDate": 1709251200000
    },
    "featureSupport": {
      "agent": {},
      "knowledgeBase": {},
      "flow": {},
      "guardrails": {},
      "explicitPromptCaching": {},
      "intelligentPromptRouting": {},
      "modelEvaluation": {},
      "prompt": {},
      "batchInference": {},
      "latencyOptimized": {}
    },
    "converse": {
      "maxTokensMaximum": 4096,
      "invokeChatFeatures": {
        "functionToolSupported": true,
        "functionToolStreamSupported": true,
        "citationsSupported": true,
        "documentsSupported": true,
        "chatHistorySupported": true,
        "systemRoleSupported": true,
        "reasoningSupported": {},
        "userImageTypesSupported": ["png", "jpeg", "gif", "webp"],
        "userVideoTypesSupported": [],
        "userAudioTypesSupported": [],
        "userPassthroughDocumentTypesSupported": ["pdf", "txt", "docx"]
      }
    }
  }
}
```

**Transformation Logic:**
- Parse context window strings: "200K" → 200000, "1M (beta)" → 1000000
- Parse use cases: semicolon/comma-separated → array
- Parse model attributes: handle long descriptions, extract categories
- Map `featureSupport` to `feature_support` (snake_case)
- Map `invokeChatFeatures` to `chat_features`

---

### 4. Service Quotas API

**Lambda:** `quota-collector`

```python
# boto3 method
quotas_client = boto3.client('service-quotas', region_name='us-east-1')
response = quotas_client.list_service_quotas(
    ServiceCode='bedrock',
    MaxResults=100
)
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ServiceCode` | `bedrock` | Service code for Bedrock |
| `MaxResults` | 100 | Pagination limit |
| Regions | All discovered regions | Collected in parallel (dynamically discovered) |

**Fields Extracted:**
```json
{
  "QuotaCode": "L-XXXXXXXX",
  "QuotaName": "On-demand model inference requests per minute for Anthropic Claude 3.5 Sonnet",
  "QuotaArn": "arn:aws:servicequotas:us-east-1::bedrock/L-XXXXXXXX",
  "Value": 1000,
  "Unit": "None",
  "Adjustable": true,
  "GlobalQuota": false,
  "UsageMetric": {},
  "Period": {}
}
```

**Transformation Logic:**
- Normalize to snake_case
- Add `region` field to each quota
- Match quotas to models via quota name parsing

---

### 5. Bedrock ListInferenceProfiles API

**Lambda:** `feature-collector`

```python
# boto3 method with pagination
bedrock_client = boto3.client('bedrock', region_name='us-east-1')
paginator = bedrock_client.get_paginator('list_inference_profiles')

for page in paginator.paginate():
    for profile in page['inferenceProfileSummaries']:
        # Process profile
```

**Fields Extracted:**
```json
{
  "inferenceProfileId": "us.anthropic.claude-3-sonnet",
  "inferenceProfileArn": "arn:aws:bedrock:us-east-1:...",
  "inferenceProfileName": "US Anthropic Claude 3 Sonnet",
  "description": "...",
  "status": "ACTIVE",
  "type": "SYSTEM_DEFINED",
  "models": [
    {
      "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
    }
  ]
}
```

**Transformation Logic:**
- Normalize to snake_case
- Add `region` field
- Build `cross_region_inference` structure per model

---

### 6. Bedrock Mantle API (SigV4 Signed)

**Lambda:** `mantle-collector`

```python
# Endpoint pattern
host = f"bedrock-mantle.{region}.api.aws"
url = f"https://{host}/v1/models"

# SigV4 signing
headers = {"Content-Type": "application/json", "Host": host}
aws_request = AWSRequest(method="GET", url=url, headers=headers)
SigV4Auth(credentials, "bedrock", region).add_auth(aws_request)
```

**Response Format:**
```json
{
  "data": [
    {
      "id": "anthropic.claude-3-sonnet-20240229-v1:0",
      "owned_by": "anthropic"
    }
  ]
}
```

**Responses API Probe:**
```python
# POST /v1/responses to detect Responses API support
url = f"https://{host}/v1/responses"
body = {"model": model_id}  # Empty input - free probe

# Results:
# - HTTP 200 + error.code "invalid_prompt" = SUPPORTED
# - HTTP 400 + error.code "validation_error" = NOT SUPPORTED
```

**Transformation Logic:**
- Extract `model_id`, `model_name`, `provider`, `region`
- Add `supports_responses_api` boolean flag

---

## Web Scraping Sources

### 1. LiteLLM Model Database

**Lambda:** `token-specs-collector`

```python
# Source URL
url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

# HTTP request
request = Request(url, headers={
    'User-Agent': 'BedrockProfiler/1.0',
    'Cache-Control': 'no-cache, no-store',
    'Pragma': 'no-cache'
})
```

**Data Structure:**
```json
{
  "bedrock/anthropic.claude-3-sonnet-20240229-v1:0": {
    "max_input_tokens": 200000,
    "max_output_tokens": 4096,
    "litellm_provider": "bedrock"
  }
}
```

**Fields Extracted:**
- `context_window` (from `max_input_tokens` or `max_tokens`)
- `max_output_tokens`

**Filtering Applied:**
- Only models with `'bedrock'` in key or `litellm_provider`

**Transformation Logic:**
- Normalize model keys (remove `bedrock/` prefix)
- Fuzzy match to Bedrock model IDs
- Output in snake_case schema

---

### 2. AWS Model Lifecycle Documentation

**Lambda:** `lifecycle-collector`

```python
# Source URL
url = "https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html"

# HTTP request
response = requests.get(url, headers={
    'User-Agent': 'Mozilla/5.0 (compatible; BedrockProfiler/1.0)',
    'Accept': 'text/html,application/xhtml+xml'
}, timeout=30)

# HTML parsing with BeautifulSoup + lxml
soup = BeautifulSoup(html_content, 'lxml')
tables = soup.select('.table-container .table-contents table')
```

**Tables Scraped:**

| Table | Columns | Status |
|-------|---------|--------|
| Active Models | Provider, Model name, Model ID, Regions, Launch date, EOL date, Input modalities, Output modalities | `active` |
| Legacy Models | Model version, Legacy date, Public extended access date, EOL date, Recommended replacement, Recommended model ID | `legacy` |
| EOL Models | Model version, Legacy date, EOL date, Recommended replacement, Recommended model ID | `eol` |

**Fields Extracted:**
```json
{
  "model_name": "Claude 3 Sonnet",
  "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
  "provider": "Anthropic",
  "lifecycle_status": "active",
  "launch_date": "2024-03-04",
  "eol_date": null,
  "legacy_date": null,
  "extended_access_date": null,
  "recommended_replacement": null,
  "recommended_model_id": null
}
```

**Transformation Logic:**
- Active models: indexed by `model_id`
- Legacy/EOL models: indexed by `model_name` (they don't have their own `model_id`)
- `recommended_model_id` contains the REPLACEMENT model ID, not the legacy model's ID

---

## S3 Data Flow

### Intermediate Files (per execution)

| S3 Key Pattern | Lambda Producer | Description |
|----------------|-----------------|-------------|
| `executions/{id}/pricing/AmazonBedrock.json` | pricing-collector | Raw pricing for service code |
| `executions/{id}/pricing/AmazonBedrockService.json` | pricing-collector | Raw pricing for service code |
| `executions/{id}/pricing/AmazonBedrockFoundationModels.json` | pricing-collector | Raw pricing for service code |
| `executions/{id}/merged/pricing.json` | pricing-aggregator | Aggregated pricing data |
| `executions/{id}/models/{region}.json` | model-extractor | Models from region |
| `executions/{id}/cache/list_foundation_models_{region}.json` | model-extractor | Cached API response for reuse |
| `executions/{id}/merged/models.json` | model-merger | Deduplicated models |
| `executions/{id}/quotas/{region}.json` | quota-collector | Quotas from region |
| `executions/{id}/features/{region}.json` | feature-collector | Inference profiles from region |
| `executions/{id}/mantle/{region}.json` | mantle-collector | Mantle models from region |
| `executions/{id}/lifecycle/lifecycle.json` | lifecycle-collector | Lifecycle data from docs |
| `executions/{id}/intermediate/models-with-pricing.json` | pricing-linker | Models with pricing references |
| `executions/{id}/intermediate/regional-availability.json` | regional-availability | ON_DEMAND + PROVISIONED regions |
| `executions/{id}/intermediate/token-specs.json` | token-specs-collector | LiteLLM token specs |
| `executions/{id}/final/bedrock_models.json` | final-aggregator | Final models output |
| `executions/{id}/final/bedrock_pricing.json` | final-aggregator | Final pricing output |
| `executions/{id}/gap-detection/gap-report.json` | gap-detection | Gap analysis report |
| `config/frontend-config.json` | config-sync | Frontend configuration |
| `agent/gap-reports/{id}/gap-analysis.json` | gap-detection | Detailed gap analysis |
| `agent/suggestions/{id}/suggestions.json` | self-healing-agent | AI-generated suggestions |

### Final Output Files

| S3 Key | Description | Size (approx) |
|--------|-------------|---------------|
| `latest/bedrock_models.json` | Complete model catalog with all metadata | ~2-3 MB |
| `latest/bedrock_pricing.json` | Pricing data by provider/model/region | ~1-2 MB |

---

## Lambda-by-Lambda Reference

### Pre-Wave Lambdas (Initialization)

| Lambda | AWS APIs | Purpose | Timeout | Memory |
|--------|----------|---------|---------|--------|
| `region-discovery` | `bedrock:ListInferenceProfiles` | Dynamically discover regions with Bedrock | 30 sec | 256 MB |
| `config-sync` | S3 read/write | Sync frontend config from backend | 30 sec | 256 MB |

### Wave 1 Lambdas (Parallel Collection)

| Lambda | AWS APIs | Web Sources | Timeout | Memory | Concurrency |
|--------|----------|-------------|---------|--------|-------------|
| `pricing-collector` | `pricing:GetProducts` | Bulk Pricing API (HTTPS) | 5 min | 512 MB | 3 |
| `pricing-aggregator` | S3 read/write | - | 2 min | 1 GB | 1 |
| `model-extractor` | `bedrock:ListFoundationModels`, Bedrock REST API (SigV4) | - | 1 min | 256 MB | 10 |
| `model-merger` | S3 read/write | - | 1 min | 512 MB | 1 |
| `quota-collector` | `service-quotas:ListServiceQuotas` | - | 1 min | 256 MB | 10 |

### Wave 2 Lambdas (Enrichment)

| Lambda | AWS APIs | Web Sources | Timeout | Memory | Concurrency |
|--------|----------|-------------|---------|--------|-------------|
| `pricing-linker` | S3 read/write | - | 2 min | 1 GB | 1 |
| `regional-availability` | `bedrock:ListFoundationModels` (filtered) | - | 5 min | 512 MB | 1 |
| `feature-collector` | `bedrock:ListInferenceProfiles` | - | 1 min | 256 MB | 10 |
| `token-specs-collector` | S3 read/write | LiteLLM GitHub | 2 min | 512 MB | 1 |
| `mantle-collector` | Mantle REST API (SigV4) | - | 2 min | 256 MB | 10 |
| `lifecycle-collector` | S3 write | AWS Docs (scraping) | 1.5 min | 256 MB | 1 |

### Wave 3 Lambdas (Aggregation)

| Lambda | AWS APIs | Timeout | Memory | Description |
|--------|----------|---------|--------|-------------|
| `final-aggregator` | S3 read/write | 3 min | 2 GB | Merge all data into final schema |
| `gap-detection` | S3 read/write | 2 min | 512 MB | Analyze gaps, detect regressions |
| `self-healing-agent` | `bedrock:InvokeModel` (Claude) | 5 min | 512 MB | AI-powered config suggestions |
| `copy-to-latest` | `s3:CopyObject` | 1 min | 256 MB | Copy final files to latest/ |

---

## Configuration Sources

### profiler-config.json Structure

Located at: `backend/config/profiler-config.json`

```json
{
  "version": "1.0.0",
  "last_updated": "2026-02-04T16:29:46Z",
  
  "provider_configuration": {
    "provider_aliases": {
      "mistral ai": ["mistral", "mistralai", "mistral ai"],
      "stability ai": ["stability", "stabilityai", "stable diffusion"]
    },
    "provider_patterns": {
      "Anthropic": ["claude", "anthropic", "sonnet", "haiku", "opus"],
      "Meta": ["llama", "mllama"],
      "Amazon": ["titan", "nova", "rerank"]
    },
    "explicit_provider_names": {
      "mistral": "Mistral AI",
      "ai21": "AI21 Labs"
    },
    "provider_colors": {
      "Amazon": "#FF9900",
      "Anthropic": "#D4A27F"
    }
  },
  
  "region_configuration": {
    "region_locations": {
      "us-east-1": "US East (N. Virginia)",
      "eu-west-1": "Europe (Ireland)"
    }
  },
  
  "model_configuration": {
    "context_window_specs": {
      "anthropic.claude-opus-4": {
        "standard_context": 200000,
        "extended_context": 1000000,
        "max_output": 16384,
        "extended_output": 65536,
        "extended_output_beta": true,
        "source": "config"
      }
    }
  },
  
  "documentation_links": {
    "anthropic": {
      "provider_docs": "https://docs.anthropic.com/",
      "aws_bedrock_guide": "https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html"
    }
  },
  
  "matching_configuration": {
    "min_confidence_threshold": 0.7
  }
}
```

### Configuration Usage by Lambda

| Lambda | Configuration Used |
|--------|-------------------|
| `pricing-aggregator` | `provider_patterns`, `explicit_provider_names`, `region_locations` |
| `pricing-linker` | `provider_aliases`, `min_confidence_threshold` |
| `model-extractor` | `documentation_links` |
| `final-aggregator` | `context_window_specs` |

---

## Summary Statistics

### Data Volume (Typical Execution)

| Metric | Value |
|--------|-------|
| **Total Models** | ~108 |
| **Total Providers** | ~17 |
| **Pricing Entries** | ~3,500 |
| **Regions Covered** | ~27 |
| **Quotas per Region** | ~45 |
| **Inference Profiles** | ~50+ |
| **Mantle Models** | ~30+ |

### Execution Timing

| Phase | Duration |
|-------|----------|
| Wave 1 (Pricing + Models + Quotas) | ~3-5 min |
| Wave 2 (Enrichment) | ~2-3 min |
| Final Aggregation | ~1-2 min |
| **Total Execution** | ~8-12 min |

### API Call Counts (per execution)

| API | Approximate Calls |
|-----|-------------------|
| `pricing:GetProducts` | 300+ (paginated, 3 service codes) |
| `bedrock:ListFoundationModels` | 27 (model extraction, cached for availability) |
| `service-quotas:ListServiceQuotas` | N (one per discovered region) |
| `bedrock:ListInferenceProfiles` | 27 (one per feature region) |
| Mantle API | 27 (one per region) + probes |
| LiteLLM (HTTPS) | 1 (cached for 24h) |
| AWS Docs (HTTPS) | 1 (cached for 24h) |

---

## Error Handling

All Lambdas implement consistent error handling:

```json
{
  "status": "FAILED",
  "errorType": "ThrottlingException",
  "errorMessage": "Rate exceeded",
  "retryable": true
}
```

### Retryable Errors (auto-retry with backoff)
- `ThrottlingException`
- `ProvisionedThroughputExceededException`
- `ServiceUnavailableException`
- Network timeouts
- HTTP 5xx errors

### Non-Retryable Errors
- `AccessDeniedException`
- `ValidationException`
- `InvalidParameterException`
- HTTP 4xx errors (except throttling)

### Retry Configuration (Step Functions)
```json
{
  "IntervalSeconds": 5,
  "MaxAttempts": 3,
  "BackoffRate": 2
}
```

---

## IAM Permissions Summary

| Lambda | Required Permissions |
|--------|---------------------|
| `pricing-collector` | `pricing:GetProducts` |
| `model-extractor` | `bedrock:ListFoundationModels` |
| `quota-collector` | `service-quotas:ListServiceQuotas` |
| `feature-collector` | `bedrock:ListInferenceProfiles` |
| `regional-availability` | `bedrock:ListFoundationModels` |
| `mantle-collector` | `bedrock:*` (for SigV4 to Mantle endpoint) |
| `self-healing-agent` | `bedrock:InvokeModel` |
| All Lambdas | `s3:GetObject`, `s3:PutObject` |
| `copy-to-latest` | `s3:CopyObject` |

---

## ConfigSync Lambda

**Purpose:** Syncs frontend-relevant configuration from the backend `profiler-config.json` and generates optimized files for the frontend to consume.

**Workflow Position:** Runs immediately after `InitializeExecution`, before Wave 1 parallel collection.

### Input/Output

**Input:**
```json
{
  "s3Bucket": "bucket-name",
  "executionId": "exec-123",
  "generateJs": false
}
```

**Output Files:**

| S3 Key | Description |
|--------|-------------|
| `config/frontend-config.json` | JSON config with regions, providers, colors |
| `config/generated-constants.js` | Optional JS module for direct import |

### Frontend Config Structure

```json
{
  "version": "1.0.0",
  "generated_at": "2026-03-03T12:00:00Z",
  "source": "profiler-config.json",
  "regions": {
    "us-east-1": {
      "label": "N. Virginia",
      "fullName": "US East (N. Virginia)",
      "geo": "US",
      "lat": 38.9519,
      "lng": -77.448
    }
  },
  "aws_regions": [...],
  "geo_region_options": [...],
  "providers": {
    "colors": { "Amazon": "#FF9900", ... },
    "documentation": { ... }
  },
  "model_config": {
    "context_thresholds": { "small": 32000, "medium": 128000, "large": 500000 }
  }
}
```

### Benefits

1. **Single Source of Truth** - Region/provider config defined once in backend
2. **Automatic Updates** - Frontend config regenerated each pipeline run
3. **Gap Detection** - Drift between frontend and backend configs detected

---

## Caching System

The pipeline implements a caching layer to reduce redundant API calls between Lambdas.

### Model Extraction Cache

**Producer:** `model-extractor`
**Consumer:** `regional-availability`

When `model-extractor` calls `ListFoundationModels`, it caches the raw API response:

```
executions/{execution-id}/cache/list_foundation_models_{region}.json
```

**Cache Structure:**
```json
{
  "region": "us-east-1",
  "timestamp": "2026-03-03T12:00:00Z",
  "model_summaries": [
    {
      "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
      "inferenceTypesSupported": ["ON_DEMAND", "PROVISIONED"],
      ...
    }
  ]
}
```

### Cache Flow

```
┌─────────────────┐     cache write      ┌─────────────────────────────────┐
│ model-extractor │ ──────────────────►  │ S3: cache/list_foundation_      │
│ (us-east-1)     │                      │     models_us-east-1.json       │
└─────────────────┘                      └─────────────────────────────────┘
                                                        │
                                                        │ cache read
                                                        ▼
                                         ┌─────────────────────────────────┐
                                         │   regional-availability         │
                                         │   (skips API call for cached    │
                                         │    regions)                     │
                                         └─────────────────────────────────┘
```

### Cache Utilities

Located in `backend/layers/common/python/shared/cache_utils.py`:

| Function | Description |
|----------|-------------|
| `get_cached_models(s3_client, bucket, key)` | Read cached data from S3 |
| `is_cache_valid(data, max_age_seconds=3600)` | Check if cache is still fresh |
| `build_cache_key(exec_id, region, type)` | Generate standardized cache key |

### Cache Keys Flow in State Machine

```json
{
  "PrepareWave2": {
    "Parameters": {
      "modelCacheKeys.$": "$.wave1Results[1].modelsMerged.cacheKeys"
    }
  },
  "ComputeRegionalAvailability": {
    "Parameters": {
      "cacheKeys.$": "$.modelCacheKeys"
    }
  }
}
```

### Cache Performance Impact

| Metric | Without Cache | With Cache |
|--------|---------------|------------|
| Regional availability API calls | ~54 (27 regions × 2 filters) | ~50 (2 cached × 2 filters saved) |
| Potential savings with all regions cached | N/A | Up to 93% reduction |

---

## Self-Healing Enhancement System

The self-healing system has been enhanced to detect and auto-fix additional gap types beyond basic pricing mismatches.

### Enhanced Gap Detection

**Lambda:** `gap-detection`

| Gap Type | Detection Method | Trigger Threshold |
|----------|-----------------|-------------------|
| Models without pricing | Missing `has_pricing` flag | ≥5 models |
| Low-confidence matches | `confidence < 0.6` | ≥3 matches |
| Unknown providers | Not in `provider_patterns` | Any detected |
| New models | Delta from previous run | Any detected |
| Context window mismatches | Config vs. API variance >10% | Any detected |
| Unknown service codes | Not in `pricing_service_codes` | Any detected |
| Frontend config drift | Backend ≠ frontend regions/providers | Any drift |

### Gap Report Structure

```json
{
  "summary": {
    "total_models": 108,
    "models_without_pricing": 3,
    "low_confidence_matches": 2,
    "new_models_detected": 5,
    "unknown_providers": ["NewProvider"],
    "context_window_mismatches": 1,
    "unknown_service_codes": [],
    "frontend_config_drift": false
  },
  "trigger_decision": {
    "should_trigger": true,
    "reasons": ["5 new models detected", "Unknown providers: NewProvider"],
    "priority": "high"
  },
  "details": {
    "context_window_mismatches": [
      {
        "model_id": "anthropic.claude-3-opus-20240229-v1:0",
        "actual_value": 200000,
        "config_value": 180000,
        "variance": 0.11
      }
    ],
    "frontend_config_drift": {
      "drift_detected": false,
      "regions_missing_in_frontend": [],
      "providers_missing_in_frontend": []
    }
  }
}
```

### GenAI Auto-Update Capabilities

**Lambda:** `self-healing-agent`
**Model:** Claude Opus 4.5 via Bedrock

The self-healing agent can automatically update:

| Update Type | Auto-Apply Safe | Example |
|-------------|-----------------|---------|
| `provider_pattern_addition` | ✅ Yes | Add `"nemotron"` to NVIDIA patterns |
| `provider_alias_addition` | ✅ Yes | Add `"deepseek-ai"` alias |
| `context_window_update` | ✅ Yes | Update Claude context to 200K |
| `service_code_addition` | ✅ Yes | Add new Bedrock pricing service code |
| `region_addition` | ✅ Yes | Add new region coordinates |
| `documentation_link_addition` | ✅ Yes | Add provider doc links |
| `provider_pattern_removal` | ❌ Review | Could break existing matches |
| `threshold_change` | ❌ Review | Affects system behavior |

### Auto-Apply Rules

Configured in `profiler-config.json`:

```json
{
  "agent_configuration": {
    "auto_apply_rules": {
      "safe_changes": [
        "provider_pattern_addition",
        "provider_alias_addition",
        "context_window_update",
        "service_code_addition"
      ],
      "requires_review": [
        "provider_pattern_removal",
        "threshold_change"
      ],
      "max_models_affected_for_auto_apply": 0.2
    }
  }
}
```

### Validation Thresholds

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_models_affected_for_auto_apply` | 20% | Prevents mass changes |
| `context_window_variance_threshold` | 10% | Flags significant mismatches |
| `low_confidence_threshold` | 0.6 | Minimum match confidence |

### Config History

Auto-applied changes create backups:

```
config/config-history/profiler-config.{timestamp}.json
```

This enables rollback if auto-updates cause issues.
