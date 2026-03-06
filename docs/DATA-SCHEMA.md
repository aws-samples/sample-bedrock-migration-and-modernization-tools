# Bedrock Model Profiler - Data Schema

> **Version:** 2.0  
> **Last Updated:** March 2026  
> **Output File:** `latest/bedrock_models.json`

This document provides a complete reference for the model data schema produced by the Bedrock Model Profiler.

---

## Table of Contents

1. [Overview](#overview)
2. [Top-Level Structure](#top-level-structure)
3. [Model Object](#model-object)
4. [Field Reference](#field-reference)
   - [Core Identifiers](#core-identifiers)
   - [Availability Object](#availability-object)
   - [Specs Object](#specs-object)
   - [Modalities Object](#modalities-object)
   - [Lifecycle Object](#lifecycle-object)
   - [Pricing Reference](#pricing-reference)
   - [API Support Object](#api-support-object)
   - [Chat Features Object](#chat-features-object)
   - [Features Object](#features-object)
   - [Quotas Object](#quotas-object)
5. [Model Type Examples](#model-type-examples)
6. [Lambda Field Contributions](#lambda-field-contributions)

---

## Overview

The `bedrock_models.json` file is the primary data output containing all model information.

| Property | Value |
|----------|-------|
| **File Size** | ~2-3 MB |
| **Update Frequency** | Daily at 6 AM UTC |
| **Structure** | Provider-first hierarchy |
| **Total Models** | ~120+ |
| **Total Providers** | ~18 |

---

## Top-Level Structure

```json
{
  "metadata": {
    "generated_at": "2026-03-06T06:15:32Z",
    "execution_id": "arn:aws:states:us-east-1:123456789:execution:...",
    "version": "1.0.0",
    "total_models": 123,
    "total_providers": 18,
    "total_regions": 33,
    "models_with_pricing": 108,
    "models_with_quotas": 120,
    "collection_stats": {
      "pricing_regions": 33,
      "quota_regions": 16,
      "feature_regions": 27,
      "mantle_regions": 27
    }
  },
  "providers": {
    "Anthropic": {
      "models": {
        "anthropic.claude-3-5-sonnet-20241022-v2:0": { /* Model Object */ }
      }
    },
    "Amazon": { /* ... */ },
    "Meta": { /* ... */ }
  }
}
```

### Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `generated_at` | string | ISO 8601 timestamp of generation |
| `execution_id` | string | Step Functions execution ARN |
| `version` | string | Schema version |
| `total_models` | number | Total model count |
| `total_providers` | number | Unique provider count |
| `total_regions` | number | Regions with model availability |
| `models_with_pricing` | number | Models with linked pricing data |
| `models_with_quotas` | number | Models with quota information |
| `collection_stats` | object | Per-data-type region counts |

---

## Model Object

Each model is a complete object with the following structure:

```json
{
  "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
  "model_arn": "arn:aws:bedrock:us-east-1::foundation-model/...",
  "model_name": "Claude 3.5 Sonnet v2",
  "model_provider": "Anthropic",
  
  "in_region": ["us-east-1", "us-west-2", "eu-west-1"],
  
  "customization": { /* ... */ },
  "inference_types_supported": ["ON_DEMAND", "PROVISIONED"],
  
  "description": "Full description...",
  "short_description": "Brief description...",
  
  "availability": { /* Availability Object */ },
  "modalities": { /* Modalities Object */ },
  "capabilities": ["chat", "function_calling", "image_understanding"],
  "use_cases": ["Complex agentic systems", "Visual analysis"],
  "lifecycle": { /* Lifecycle Object */ },
  "streaming": true,
  "languages": ["English", "French", "German"],
  "docs": { /* Documentation Links */ },
  "features": { /* Feature Support Object */ },
  "specs": { /* Specs Object */ },
  "pricing": { /* Pricing Reference */ },
  "model_pricing": { /* Full Pricing Details */ },
  "quotas": { /* Quotas by Region */ },
  "api": { /* API Support Object */ },
  "chat_features": { /* Chat Features Object */ },
  "consumption_options": ["on_demand", "batch", "cross_region_inference"],
  "collection_metadata": { /* Internal tracking */ },
  
  "has_pricing": true,
  "has_quotas": true
}
```

---

## Field Reference

### Core Identifiers

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `model_id` | string | Unique model identifier | model-extractor |
| `model_arn` | string | Full ARN of the model | model-extractor |
| `model_name` | string | Human-readable display name | model-extractor |
| `model_provider` | string | Provider name (normalized) | model-extractor |
| `in_region` | string[] | Regions with ON_DEMAND availability | regional-availability |

---

### Availability Object

The `availability` object consolidates all consumption types:

```json
{
  "availability": {
    "on_demand": {
      "supported": true,
      "regions": ["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-1"]
    },
    "cross_region": {
      "supported": true,
      "regions": ["us-east-1", "us-west-2"],
      "profiles": [
        {
          "profile_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
          "profile_name": "US Anthropic Claude 3.5 Sonnet v2",
          "source_region": "us-east-1",
          "type": "SYSTEM_DEFINED",
          "status": "ACTIVE",
          "description": "Cross-region inference profile"
        }
      ]
    },
    "batch": {
      "supported": true,
      "regions": ["us-east-1", "us-west-2"]
    },
    "provisioned": {
      "supported": true,
      "regions": ["us-east-1", "us-west-2", "eu-west-1"]
    },
    "mantle": {
      "supported": true,
      "regions": ["us-east-1", "us-west-2"],
      "only": false,
      "responses_api": true,
      "has_pricing": true
    }
  }
}
```

| Sub-field | Type | Description |
|-----------|------|-------------|
| `on_demand.supported` | boolean | ON_DEMAND inference available |
| `on_demand.regions` | string[] | Regions with on-demand support |
| `cross_region.supported` | boolean | CRIS available |
| `cross_region.regions` | string[] | Source regions for CRIS |
| `cross_region.profiles` | array | Inference profile details |
| `batch.supported` | boolean | Batch inference available |
| `batch.regions` | string[] | Regions with batch support |
| `provisioned.supported` | boolean | Provisioned throughput available |
| `provisioned.regions` | string[] | Regions with provisioned support |
| `mantle.supported` | boolean | Mantle (OpenAI-compatible) available |
| `mantle.regions` | string[] | Regions with Mantle support |
| `mantle.only` | boolean | True if ONLY available via Mantle |
| `mantle.responses_api` | boolean | Supports Responses API |
| `mantle.has_pricing` | boolean | Has Mantle-specific pricing |

---

### Specs Object

```json
{
  "specs": {
    "context_window": 200000,
    "max_output": 8192,
    "extended_context": 1000000,
    "size_category": {
      "category": "Large",
      "color": "#10B981",
      "tier": 3
    },
    "source": "bedrock_console_api",
    "verified": true
  }
}
```

| Field | Type | Description | Priority Source |
|-------|------|-------------|-----------------|
| `context_window` | number | Standard context window (tokens) | Console API > Config > LiteLLM |
| `max_output` | number | Maximum output tokens | Console API > Config > LiteLLM |
| `extended_context` | number | Extended context (if available) | profiler-config.json |
| `size_category` | object | Visual categorization | Computed |
| `source` | string | Data source identifier | Auto-set |
| `verified` | boolean | Whether data is verified | Auto-set |

**Size Categories:**

| Category | Threshold | Color |
|----------|-----------|-------|
| Large | ≥128K tokens | Green (#10B981) |
| Medium | ≥32K tokens | Blue (#3B82F6) |
| Small | <32K tokens | Orange (#F59E0B) |

**Source Values:**

| Value | Description |
|-------|-------------|
| `bedrock_console_api` | From Bedrock Console REST API |
| `config` | Manual override in profiler-config.json |
| `litellm` | From LiteLLM external database |
| `model_id_variant` | Extracted from model ID |

---

### Modalities Object

```json
{
  "modalities": {
    "input_modalities": ["TEXT", "IMAGE"],
    "output_modalities": ["TEXT"]
  }
}
```

| Value | Description | Model Types |
|-------|-------------|-------------|
| `TEXT` | Text input/output | Most models |
| `IMAGE` | Image input/output | Vision models, image generators |
| `VIDEO` | Video input/output | Nova Reel, Luma Ray |
| `AUDIO` | Audio input/output | Nova Sonic, Voxtral |
| `EMBEDDING` | Embedding output | Titan Embeddings, Nova Embeddings |

---

### Lifecycle Object

```json
{
  "lifecycle": {
    "status": "ACTIVE",
    "global_status": "ACTIVE",
    "primary_status": "ACTIVE",
    "regional_status": {
      "us-east-1": {
        "status": "ACTIVE",
        "launch_date": "2024-10-22"
      }
    },
    "status_summary": {
      "ACTIVE": ["us-east-1", "us-west-2", "eu-west-1"],
      "LEGACY": [],
      "EOL": []
    },
    "release_date": "2024-10-22",
    "launch_date": "2024-10-22",
    "eol_date": null,
    "legacy_date": null,
    "recommended_replacement": null,
    "recommended_model_id": null
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Current status: ACTIVE, LEGACY, EOL |
| `global_status` | string | Overall status (ACTIVE, LEGACY, EOL, MIXED) |
| `primary_status` | string | Most restrictive status |
| `regional_status` | object | Per-region status details |
| `status_summary` | object | Lists of regions by status |
| `release_date` | string | Initial release date |
| `eol_date` | string | End-of-life date (if scheduled) |
| `legacy_date` | string | Legacy transition date |
| `recommended_replacement` | string | Suggested replacement model name |
| `recommended_model_id` | string | Suggested replacement model ID |

> **Note:** The `global_status` will be "MIXED" when a model has different lifecycle statuses across regions (e.g., LEGACY in us-east-1 but ACTIVE in ap-northeast-1).

---

### Pricing Reference

```json
{
  "pricing": {
    "available": true,
    "reference": {
      "provider": "Anthropic",
      "model_key": "anthropic.claude-3-5-sonnet-20241022-v2:0",
      "model_name": "Claude 3.5 Sonnet v2"
    }
  }
}
```

This is a lightweight reference pointing to `bedrock_pricing.json`. For full pricing details, use `model_pricing`.

---

### API Support Object

```json
{
  "api": {
    "invoke_model": {
      "supported": true,
      "streaming": true,
      "endpoint": "bedrock-runtime"
    },
    "converse": {
      "supported": true,
      "streaming": true,
      "endpoint": "bedrock-runtime",
      "features": {
        "system_prompts": true,
        "tool_use": true,
        "streaming_tool_use": true,
        "vision": true,
        "document_chat": true,
        "citations": false,
        "reasoning": false
      }
    },
    "chat_completions": {
      "supported": true,
      "endpoints": ["bedrock-runtime", "bedrock-mantle"]
    },
    "responses_api": {
      "supported": true,
      "endpoint": "bedrock-mantle"
    },
    "endpoints_supported": ["bedrock-runtime", "bedrock-mantle"]
  }
}
```

| API | Description | Endpoint |
|-----|-------------|----------|
| `invoke_model` | Native InvokeModel API | bedrock-runtime |
| `converse` | Converse API with features | bedrock-runtime |
| `chat_completions` | OpenAI-compatible API | bedrock-mantle |
| `responses_api` | Responses API (agent-style) | bedrock-mantle |

---

### Chat Features Object

```json
{
  "chat_features": {
    "function_calling": true,
    "function_calling_streaming": true,
    "citations": false,
    "documents": true,
    "chat_history": true,
    "system_role": true,
    "reasoning": {
      "embedded": false,
      "budget_control": false
    },
    "supported_image_types": ["png", "jpeg", "gif", "webp"],
    "supported_video_types": [],
    "supported_audio_types": [],
    "supported_document_types": ["pdf", "txt", "docx"]
  }
}
```

---

### Features Object (Bedrock Feature Support)

```json
{
  "features": {
    "agent": { "supported": true },
    "knowledgeBase": { "supported": true },
    "flow": { "supported": true },
    "guardrails": { "supported": true },
    "explicitPromptCaching": { "supported": true },
    "intelligentPromptRouting": { "supported": true },
    "modelEvaluation": { "supported": true },
    "prompt": { "supported": true },
    "batchInference": { "supported": true },
    "latencyOptimized": { "supported": false }
  }
}
```

---

### Quotas Object

```json
{
  "quotas": {
    "us-east-1": [
      {
        "quota_code": "L-XXXXXXXX",
        "quota_name": "On-demand model inference requests per minute for Anthropic Claude 3.5 Sonnet",
        "quota_arn": "arn:aws:servicequotas:us-east-1::bedrock/L-XXXXXXXX",
        "description": "Maximum number of inference requests per minute",
        "quota_applied_at_level": "ACCOUNT",
        "value": 1000,
        "unit": "None",
        "adjustable": true,
        "global_quota": false
      }
    ]
  }
}
```

---

## Model Type Examples

### Text Generation Model (Claude)

```json
{
  "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
  "model_name": "Claude 3.5 Sonnet v2",
  "model_provider": "Anthropic",
  "modalities": {
    "input_modalities": ["TEXT", "IMAGE"],
    "output_modalities": ["TEXT"]
  },
  "capabilities": ["chat", "function_calling", "image_understanding", "code_generation"],
  "specs": {
    "context_window": 200000,
    "max_output": 8192
  },
  "streaming": true
}
```

### Image Generation Model (Nova Canvas)

```json
{
  "model_id": "amazon.nova-canvas-v1:0",
  "model_name": "Nova Canvas",
  "model_provider": "Amazon",
  "modalities": {
    "input_modalities": ["TEXT", "IMAGE"],
    "output_modalities": ["IMAGE"]
  },
  "capabilities": ["image_generation", "image_editing"],
  "specs": {
    "context_window": null,
    "max_output": null
  },
  "streaming": false
}
```

### Video Generation Model (Nova Reel)

```json
{
  "model_id": "amazon.nova-reel-v1:0",
  "model_name": "Nova Reel",
  "model_provider": "Amazon",
  "modalities": {
    "input_modalities": ["TEXT", "IMAGE"],
    "output_modalities": ["VIDEO"]
  },
  "capabilities": ["video_generation"],
  "streaming": false
}
```

### Embedding Model (Titan Embeddings)

```json
{
  "model_id": "amazon.titan-embed-text-v2:0",
  "model_name": "Titan Text Embeddings V2",
  "model_provider": "Amazon",
  "modalities": {
    "input_modalities": ["TEXT"],
    "output_modalities": ["EMBEDDING"]
  },
  "capabilities": ["embedding"],
  "specs": {
    "context_window": 8192,
    "max_output": 1024
  },
  "streaming": false
}
```

### Rerank Model (Amazon Rerank)

```json
{
  "model_id": "amazon.rerank-v1:0",
  "model_name": "Amazon Rerank",
  "model_provider": "Amazon",
  "modalities": {
    "input_modalities": ["TEXT"],
    "output_modalities": ["TEXT"]
  },
  "capabilities": ["rerank"],
  "streaming": false
}
```

### Mantle-Only Model

```json
{
  "model_id": "openai.gpt-4o",
  "model_name": "GPT-4o",
  "model_provider": "OpenAI",
  "availability": {
    "on_demand": {
      "supported": false,
      "regions": []
    },
    "mantle": {
      "supported": true,
      "regions": ["us-east-1", "us-west-2"],
      "only": true,
      "responses_api": true
    }
  }
}
```

### CRIS-Only Model (No On-Demand)

Some models are only available via Cross-Region Inference Service profiles:

```json
{
  "model_id": "anthropic.claude-opus-4-5-20251101-v1:0",
  "availability": {
    "on_demand": {
      "supported": false,
      "regions": []
    },
    "cross_region": {
      "supported": true,
      "regions": ["us-east-1"],
      "profiles": [
        {
          "profile_id": "us.anthropic.claude-opus-4-5-20251101-v1:0",
          "profile_name": "US Claude Opus 4.5",
          "type": "SYSTEM_DEFINED"
        }
      ]
    }
  }
}
```

---

## Lambda Field Contributions

This table shows which Lambda function contributes which fields:

| Lambda | Fields Contributed |
|--------|-------------------|
| **model-extractor** | `model_id`, `model_arn`, `model_name`, `model_provider`, `inference_types_supported`, `customization`, `modalities`, `streaming`, `description`, `short_description`, `use_cases`, `languages`, `capabilities`, `chat_features`, `features` |
| **model-merger** | Deduplication, `extraction_regions` |
| **pricing-collector** | Raw pricing entries |
| **pricing-aggregator** | Structured pricing by provider/model/region |
| **pricing-linker** | `pricing`, `model_pricing`, `has_pricing` |
| **quota-collector** | `quotas` per region |
| **regional-availability** | `in_region`, `availability.on_demand`, `availability.provisioned` |
| **feature-collector** | `availability.cross_region`, `consumption_options` |
| **token-specs-collector** | `specs.context_window`, `specs.max_output` (fallback) |
| **mantle-collector** | `availability.mantle`, `api.chat_completions`, `api.responses_api` |
| **lifecycle-collector** | `lifecycle` |
| **final-aggregator** | Final merge, `specs.size_category`, derived fields |

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | System architecture overview |
| [DATA-SOURCES.md](./DATA-SOURCES.md) | Data sources and reliability |
| [PRICING-SCHEMA.md](./PRICING-SCHEMA.md) | Pricing JSON schema reference |

---

## Changelog

| Date | Version | Change |
|------|---------|--------|
| 2026-03-06 | 2.0 | Reorganized into dedicated schema document |
| 2026-03-05 | 1.0 | Initial comprehensive documentation |
