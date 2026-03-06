# Bedrock Model Profiler - Data Sources

> **Version:** 2.0  
> **Last Updated:** March 2026  
> **Maintainer:** AWS Bedrock Profiler Team

This document details all data sources used by the Bedrock Model Profiler, including reliability assessments, fallback mechanisms, and known issues.

---

## Table of Contents

1. [Data Sources Overview](#data-sources-overview)
2. [AWS APIs](#aws-apis)
   - [Bedrock ListFoundationModels](#1-bedrock-listfoundationmodels-api)
   - [Bedrock Console REST API](#2-bedrock-console-rest-api)
   - [AWS Pricing API](#3-aws-pricing-api)
   - [Bedrock ListInferenceProfiles](#4-bedrock-listinferenceprofiles-api)
   - [Service Quotas API](#5-service-quotas-api)
   - [Bedrock Mantle API](#6-bedrock-mantle-api)
3. [External Sources](#external-sources)
   - [LiteLLM Model Database](#7-litellm-model-database)
   - [AWS Lifecycle Documentation](#8-aws-lifecycle-documentation)
4. [Data Reliability Matrix](#data-reliability-matrix)
5. [Fallback Mechanisms](#fallback-mechanisms)
6. [Known Issues](#known-issues)

---

## Data Sources Overview

The profiler aggregates data from **8 distinct sources**:

| # | Source | Type | Reliability | Update Frequency |
|---|--------|------|-------------|------------------|
| 1 | Bedrock ListFoundationModels | AWS API | High | Real-time |
| 2 | Bedrock Console REST API | AWS API (SigV4) | High | Real-time |
| 3 | AWS Pricing API | AWS API | High | Near real-time |
| 4 | Bedrock ListInferenceProfiles | AWS API | High | Real-time |
| 5 | Service Quotas API | AWS API | High | Real-time |
| 6 | Bedrock Mantle API | AWS API (SigV4) | Medium | Real-time |
| 7 | LiteLLM Model Database | External (GitHub) | Medium | Community-maintained |
| 8 | AWS Lifecycle Documentation | Web Scraping | Low-Medium | Documentation updates |

---

## AWS APIs

### 1. Bedrock ListFoundationModels API

**Lambda:** `model-extractor`  
**Reliability:** **HIGH**  
**Regions:** 27 Bedrock-enabled regions

#### API Call

```python
bedrock_client = boto3.client('bedrock', region_name='us-east-1')
response = bedrock_client.list_foundation_models()
```

#### Fields Extracted

| Field | Type | Description | Reliability |
|-------|------|-------------|-------------|
| `modelId` | string | Unique identifier | **HIGH** - Primary key |
| `modelArn` | string | Full ARN | **HIGH** |
| `modelName` | string | Display name | **HIGH** |
| `providerName` | string | Provider name | **HIGH** |
| `inputModalities` | array | Input types (TEXT, IMAGE, etc.) | **HIGH** |
| `outputModalities` | array | Output types | **HIGH** |
| `responseStreamingSupported` | boolean | Streaming capability | **HIGH** |
| `customizationsSupported` | array | Fine-tuning options | **HIGH** |
| `inferenceTypesSupported` | array | ON_DEMAND, PROVISIONED | **HIGH** |
| `modelLifecycle.status` | string | ACTIVE, LEGACY, EOL | **UNRELIABLE** |

#### Reliability Notes

| Aspect | Assessment |
|--------|------------|
| **Core fields** | Highly reliable - authoritative source |
| **`modelLifecycle.status`** | **UNRELIABLE** - Often returns ACTIVE even for legacy models |
| **Availability** | Indicates API support, not actual regional availability |

> **Warning**: The `modelLifecycle.status` field from this API frequently shows ACTIVE for models that are actually LEGACY or EOL. Always cross-reference with the lifecycle-collector data from AWS documentation.

---

### 2. Bedrock Console REST API

**Lambda:** `model-extractor`  
**Reliability:** **HIGH** (for metadata that exists)  
**Authentication:** SigV4 signed requests with `x-console-consumer: true` header

#### API Call

```python
url = f"https://bedrock.{region}.amazonaws.com/foundation-models"
headers = {
    "Content-Type": "application/json",
    "x-console-consumer": "true"  # Enables extended metadata
}
# SigV4 signing required
```

#### Extended Metadata Fields

| Field | Location | Description | Reliability |
|-------|----------|-------------|-------------|
| `maxContextWindow` | `consoleIDEMetadata.description` | Context window (e.g., "200K") | **HIGH** |
| `maxTokensMaximum` | `consoleIDEMetadata.converse` | Max output tokens | **HIGH** |
| `supportedUseCases` | `consoleIDEMetadata.description` | Use case list | **MEDIUM** |
| `supportedLanguages` | `consoleIDEMetadata.description` | Language support | **MEDIUM** |
| `modelAttributes` | `consoleIDEMetadata.description` | Capabilities | **MEDIUM** |
| `featureSupport` | `consoleIDEMetadata.featureSupport` | Agent, KB, Flow support | **HIGH** |
| `invokeChatFeatures` | `consoleIDEMetadata.converse` | Tool use, vision, etc. | **HIGH** |
| `releaseDate` | `consoleIDEMetadata.description` | Launch timestamp | **MEDIUM** |

#### Feature Support Object

```json
{
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
  }
}
```

**Note:** An empty object `{}` indicates the feature IS supported. Absence of the key or `null` indicates NOT supported.

#### Chat Features (Converse API)

```json
{
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
```

#### Reliability Notes

- **Primary source** for context windows and chat features
- Not all models have `consoleIDEMetadata` populated
- Fallback to LiteLLM when console data is missing

---

### 3. AWS Pricing API

**Lambda:** `pricing-collector`, `pricing-aggregator`  
**Reliability:** **HIGH**  
**Region:** us-east-1 only (global pricing endpoint)

#### Service Codes

| Service Code | Content |
|-------------|---------|
| `AmazonBedrock` | Base model pricing |
| `AmazonBedrockService` | Additional service pricing |
| `AmazonBedrockFoundationModels` | Foundation model specific pricing |

#### API Call

```python
pricing_client = boto3.client('pricing', region_name='us-east-1')
response = pricing_client.get_products(
    ServiceCode='AmazonBedrock',
    MaxResults=100,
    FormatVersion='aws_v1'
)
```

#### Pricing Entry Fields

| Field | Type | Description | Reliability |
|-------|------|-------------|-------------|
| `sku` | string | Unique pricing identifier | **HIGH** |
| `usagetype` | string | Usage type pattern | **HIGH** |
| `inferenceType` | string | on-demand, batch, etc. | **HIGH** |
| `regionCode` | string | AWS region | **HIGH** |
| `pricePerUnit.USD` | number | Price in USD | **HIGH** |
| `unit` | string | Pricing unit (1K tokens, etc.) | **HIGH** |

#### Usage Type Patterns

| Pattern | Meaning | Example |
|---------|---------|---------|
| `USE1-Claude3Sonnet-input-tokens` | Region + Model + Direction | us-east-1 input |
| `*-output-tokens` | Output pricing | |
| `*-Batch-*` | Batch inference | 50% discount typically |
| `*-GlobalInference-*` | CRIS Global pricing | |
| `*-GeoInference-*` | CRIS Geo pricing | |
| `*-LongContext-*` | Extended context | >128K tokens |

#### Reliability Notes

- **Highly reliable** for pricing data
- Some models missing from Pricing API (use Bulk Pricing API fallback)
- Model name formats differ from Bedrock API (requires fuzzy matching)

---

### 4. Bedrock ListInferenceProfiles API

**Lambda:** `feature-collector`  
**Reliability:** **HIGH**

#### API Call

```python
bedrock_client = boto3.client('bedrock', region_name='us-east-1')
paginator = bedrock_client.get_paginator('list_inference_profiles')
for page in paginator.paginate():
    for profile in page['inferenceProfileSummaries']:
        # Process profile
```

#### Fields Extracted

| Field | Type | Description | Reliability |
|-------|------|-------------|-------------|
| `inferenceProfileId` | string | Profile ID (e.g., `us.anthropic.claude-3-sonnet`) | **HIGH** |
| `inferenceProfileName` | string | Display name | **HIGH** |
| `type` | string | SYSTEM_DEFINED or APPLICATION | **HIGH** |
| `status` | string | ACTIVE | **HIGH** |
| `models` | array | List of model ARNs | **HIGH** |

#### Profile ID Prefixes

| Prefix | Scope | Description |
|--------|-------|-------------|
| `us.` | US regions | North America |
| `eu.` | EU regions | Europe |
| `apac.` | APAC regions | Asia Pacific |

---

### 5. Service Quotas API

**Lambda:** `quota-collector`  
**Reliability:** **HIGH**

#### API Call

```python
quotas_client = boto3.client('service-quotas', region_name='us-east-1')
response = quotas_client.list_service_quotas(
    ServiceCode='bedrock',
    MaxResults=100
)
```

#### Fields Extracted

| Field | Type | Description | Reliability |
|-------|------|-------------|-------------|
| `QuotaCode` | string | Quota identifier (L-XXXXXXXX) | **HIGH** |
| `QuotaName` | string | Human-readable name | **HIGH** |
| `Value` | number | Default quota value | **HIGH** |
| `Adjustable` | boolean | Can be increased | **HIGH** |
| `Unit` | string | Unit of measurement | **HIGH** |

#### Quota Name Patterns

Quota names follow patterns that link to specific models:

- `On-demand model inference requests per minute for Anthropic Claude 3.5 Sonnet`
- `Provisioned throughput model units for Amazon Titan`

---

### 6. Bedrock Mantle API

**Lambda:** `mantle-collector`  
**Reliability:** **MEDIUM**  
**Authentication:** SigV4 signed requests

#### API Call

```python
host = f"bedrock-mantle.{region}.api.aws"
url = f"https://{host}/v1/models"

# SigV4 signing required
aws_request = AWSRequest(method="GET", url=url, headers={"Host": host})
SigV4Auth(credentials, "bedrock", region).add_auth(aws_request)
```

#### Response Format

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

#### Responses API Detection

```python
# POST /v1/responses to detect Responses API support
url = f"https://{host}/v1/responses"
body = {"model": model_id}

# Result interpretation:
# - HTTP 200 + error.code "invalid_prompt" = SUPPORTED
# - HTTP 400 + error.code "validation_error" = NOT SUPPORTED
```

#### Reliability Notes

- **Medium reliability** - API is undocumented
- Model ID format sometimes differs from Bedrock API
- Not all regions have Mantle enabled
- Useful for detecting OpenAI-compatible endpoint availability

---

## External Sources

### 7. LiteLLM Model Database

**Lambda:** `token-specs-collector`  
**Reliability:** **MEDIUM**  
**URL:** `https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json`

#### Data Structure

```json
{
  "bedrock/anthropic.claude-3-sonnet-20240229-v1:0": {
    "max_input_tokens": 200000,
    "max_output_tokens": 4096,
    "litellm_provider": "bedrock"
  }
}
```

#### Fields Used

| Field | Mapped To | Fallback Priority |
|-------|-----------|-------------------|
| `max_input_tokens` | `specs.context_window` | 3rd (after Console API, config) |
| `max_output_tokens` | `specs.max_output` | 3rd |

#### Reliability Notes

| Aspect | Assessment |
|--------|------------|
| **Availability** | Generally available but external dependency |
| **Accuracy** | Community-maintained, may lag behind |
| **Coverage** | Good coverage for popular models |
| **Freshness** | Updates depend on community contributions |

> **Warning**: LiteLLM is a **fallback source**. Prefer Console API or profiler-config.json for authoritative values.

#### Fallback URL

```
https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/model_prices_and_context_window_backup.json
```

---

### 8. AWS Lifecycle Documentation

**Lambda:** `lifecycle-collector`  
**Reliability:** **LOW-MEDIUM**  
**URL:** `https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html`

#### Tables Scraped

| Table | Status | Fields |
|-------|--------|--------|
| Active Models | ACTIVE | Provider, Model name, Model ID, Regions, Launch date, EOL date |
| Legacy Models | LEGACY | Model version, Legacy date, Extended access date, EOL date, Replacement |
| EOL Models | EOL | Model version, Legacy date, EOL date, Replacement |

#### Parsing Challenges

| Challenge | Solution |
|-----------|----------|
| **HTML structure changes** | Defensive parsing with fallbacks |
| **Region data in date fields** | Regex extraction (e.g., "March 2026 (us-east-1, us-west-2)") |
| **Inconsistent formatting** | Multiple parsing patterns |
| **Missing model IDs** | Match by model name |

#### Reliability Notes

| Aspect | Assessment |
|--------|------------|
| **HTML structure** | Can break if AWS changes page layout |
| **Data freshness** | Updates when AWS updates docs (not real-time) |
| **Region information** | Often embedded in date text, requires parsing |
| **Model matching** | Legacy/EOL tables use model names, not IDs |

> **Warning**: This is the most fragile data source. The lifecycle-collector has extensive error handling and fallbacks, but documentation format changes can cause parsing failures.

---

## Data Reliability Matrix

### By Field Category

| Category | Primary Source | Fallback | Reliability |
|----------|---------------|----------|-------------|
| **Model Identity** | Bedrock API | - | **HIGH** |
| **Context Window** | Console API | Config → LiteLLM | **HIGH** |
| **Max Output** | Console API | Config → LiteLLM | **HIGH** |
| **Pricing** | Pricing API | Bulk Pricing API | **HIGH** |
| **Availability (On-Demand)** | Bedrock API per region | - | **HIGH** |
| **Availability (CRIS)** | ListInferenceProfiles | - | **HIGH** |
| **Availability (Mantle)** | Mantle API | - | **MEDIUM** |
| **Lifecycle Status** | Lifecycle Docs | Bedrock API (unreliable) | **MEDIUM** |
| **Feature Support** | Console API | - | **HIGH** |
| **Quotas** | Service Quotas API | - | **HIGH** |

### By Data Source

| Source | Overall Reliability | Known Issues |
|--------|-------------------|--------------|
| Bedrock ListFoundationModels | **HIGH** | `modelLifecycle.status` unreliable |
| Bedrock Console REST API | **HIGH** | Not all models have metadata |
| AWS Pricing API | **HIGH** | Model name format differences |
| ListInferenceProfiles | **HIGH** | - |
| Service Quotas API | **HIGH** | - |
| Bedrock Mantle API | **MEDIUM** | Undocumented, format differences |
| LiteLLM | **MEDIUM** | Community-maintained, may lag |
| Lifecycle Documentation | **LOW-MEDIUM** | HTML scraping fragility |

---

## Fallback Mechanisms

### Context Window Fallback Chain

```
1. Console API (bedrock_console_api)
   └── Parsed from consoleIDEMetadata.description.maxContextWindow
       ↓ if missing
2. profiler-config.json (config)
   └── Manual overrides in context_window_specs
       ↓ if missing
3. LiteLLM (litellm)
   └── max_input_tokens from external database
       ↓ if missing
4. Default (null)
```

### Lifecycle Status Fallback Chain

```
1. Lifecycle Documentation (lifecycle-collector)
   └── Scraped from AWS docs, most authoritative
       ↓ if missing
2. Bedrock API (model_lifecycle.status)
   └── From ListFoundationModels - UNRELIABLE
       ↓ if missing
3. Default to "ACTIVE"
```

### Pricing Match Fallback Chain

```
1. Explicit Mapping (config)
   └── matching_configuration.explicit_model_mappings
       ↓ if not mapped
2. Exact Match
   └── Model ID matches pricing key exactly
       ↓ if no match
3. Fuzzy Match (>0.7 confidence)
   └── Normalized string similarity
       ↓ if below threshold
4. No Pricing (has_pricing: false)
```

---

## Known Issues

### 1. Model Lifecycle Status Unreliability

**Issue:** Bedrock API's `modelLifecycle.status` often shows ACTIVE for models that are actually LEGACY or EOL.

**Impact:** Models may display incorrect lifecycle status.

**Mitigation:** 
- Lifecycle-collector scrapes AWS documentation for authoritative status
- Documentation data takes precedence over API data

### 2. Pricing API Model Name Mismatch

**Issue:** Pricing API uses different model name formats than Bedrock API.

**Examples:**
- Bedrock: `anthropic.claude-3-sonnet-20240229-v1:0`
- Pricing: `anthropic.claude-3-sonnet`

**Mitigation:**
- Fuzzy matching with semantic conflict detection
- Explicit mappings in config for known problem cases

### 3. Mantle vs Bedrock Model ID Differences

**Issue:** Mantle API sometimes uses different model ID formats.

**Example:**
- Bedrock: `deepseek.v3-v1:0`
- Mantle: `deepseek.v3.1`

**Mitigation:**
- Multiple normalization patterns in final-aggregator
- Creates stub entries for Mantle-only models

### 4. Documentation Scraping Fragility

**Issue:** Lifecycle collector can break if AWS changes HTML structure.

**Mitigation:**
- 24-hour cache reduces API calls
- Graceful degradation to Bedrock API status
- Monitoring alerts for parsing failures

### 5. LiteLLM Data Staleness

**Issue:** Community-maintained data may not reflect latest AWS updates.

**Mitigation:**
- Console API is preferred source
- Config overrides for known discrepancies
- Self-healing agent can update config

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | System architecture overview |
| [DATA-SCHEMA.md](./DATA-SCHEMA.md) | Model JSON schema reference |
| [PRICING-SCHEMA.md](./PRICING-SCHEMA.md) | Pricing JSON schema reference |
| [model-matching-issues.md](./model-matching-issues.md) | Known matching issues and resolutions |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-03-06 | Reorganized with reliability focus, added fallback chains |
| 2026-03-05 | Added reliability matrix and known issues |
| 2026-03-03 | Initial documentation |
