# Bedrock Model Profiler - JSON Structure Reference

> **Version:** 1.0  
> **Last Updated:** March 2026  
> **Status:** Production

This document provides a complete reference for the JSON output files produced by the Bedrock Model Profiler pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [bedrock_models.json](#bedrock_modelsjson)
   - [Metadata Object](#metadata-object)
   - [Providers Object](#providers-object)
   - [Model Object](#model-object)
   - [Field Reference](#field-reference)
3. [bedrock_pricing.json](#bedrock_pricingjson)
   - [Pricing Metadata](#pricing-metadata)
   - [Pricing Structure](#pricing-structure)
   - [Pricing Groups](#pricing-groups)
   - [Pricing Dimensions](#pricing-dimensions)
4. [Lambda Contributions](#lambda-contributions)
5. [Schema Examples](#schema-examples)

---

## Overview

The pipeline produces two main JSON files:

| File | Description | Size | Update Frequency |
|------|-------------|------|------------------|
| `latest/bedrock_models.json` | Complete model catalog with all metadata | ~2-3 MB | Daily |
| `latest/bedrock_pricing.json` | Pricing data by provider/model/region | ~1-2 MB | Daily |

Both files follow a **provider-first hierarchy** for efficient lookups:

```
{
  "metadata": { ... },
  "providers": {
    "Anthropic": { ... },
    "Amazon": { ... },
    "Meta": { ... }
  }
}
```

---

## bedrock_models.json

### Top-Level Structure

```json
{
  "metadata": {
    "generated_at": "2026-03-05T06:15:32Z",
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
    "<provider_name>": {
      "models": {
        "<model_id>": { /* Model Object */ }
      }
    }
  }
}
```

### Metadata Object

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

### Providers Object

The `providers` object is keyed by canonical provider name (e.g., "Anthropic", "Amazon", "Meta"):

```json
{
  "providers": {
    "Anthropic": {
      "models": {
        "anthropic.claude-3-5-sonnet-20241022-v2:0": { /* Model */ },
        "anthropic.claude-3-opus-20240229-v1:0": { /* Model */ }
      }
    },
    "Amazon": {
      "models": {
        "amazon.nova-pro-v1:0": { /* Model */ },
        "amazon.titan-text-express-v1": { /* Model */ }
      }
    }
  }
}
```

### Model Object

Each model contains these top-level fields:

```json
{
  "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
  "model_arn": "arn:aws:bedrock:us-east-1::foundation-model/...",
  "model_name": "Claude 3.5 Sonnet v2",
  "model_provider": "Anthropic",
  
  "in_region": ["us-east-1", "us-west-2", "eu-west-1"],
  
  "customization": { /* Customization Object */ },
  "inference_types_supported": ["ON_DEMAND", "PROVISIONED"],
  
  "description": "Full description from console metadata...",
  "short_description": "Brief description...",
  
  "chat_features": { /* Chat Features Object */ },
  "consumption_options": ["on_demand", "batch", "cross_region_inference", "mantle"],
  
  "collection_metadata": { /* Collection Metadata Object */ },
  
  "has_pricing": true,
  "has_quotas": true,
  
  "availability": { /* Availability Object */ },
  "modalities": { /* Modalities Object */ },
  "capabilities": ["chat", "function_calling", "image_understanding"],
  "use_cases": ["Complex agentic systems", "Visual analysis", "Code generation"],
  "lifecycle": { /* Lifecycle Object */ },
  "streaming": true,
  "languages": ["English", "French", "German", "Spanish"],
  "docs": { /* Documentation Links Object */ },
  "features": { /* Feature Support Object */ },
  "specs": { /* Specs Object */ },
  "pricing": { /* Pricing Reference Object */ },
  "model_pricing": { /* Full Pricing Details */ },
  "quotas": { /* Quotas by Region */ },
  "api": { /* API Support Object */ }
}
```

---

### Field Reference

#### Core Identifiers

| Field | Type | Description | Source Lambda |
|-------|------|-------------|---------------|
| `model_id` | string | Unique model identifier | model-extractor |
| `model_arn` | string | Full ARN of the model | model-extractor |
| `model_name` | string | Human-readable display name | model-extractor |
| `model_provider` | string | Provider name (normalized) | model-extractor |
| `in_region` | string[] | Regions with ON_DEMAND availability | regional-availability |

#### Availability Object

The `availability` object consolidates all consumption options:

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
    },
    "reserved": {
      "supported": true,
      "regions": ["us-east-1", "us-west-2", "eu-west-1"],
      "commitments": ["1_month", "3_month"]
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
| `reserved.supported` | boolean | Reserved Capacity available |
| `reserved.regions` | string[] | Regions with Reserved Capacity pricing |
| `reserved.commitments` | string[] | Commitment terms (e.g., "1_month", "3_month") |

#### Specs Object

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
| `size_category` | object | Visual categorization | Computed from context_window |
| `source` | string | Data source identifier | Auto-set |
| `verified` | boolean | Whether data is verified | Auto-set |

**Context Window Priority:**
1. Bedrock Console API metadata (`bedrock_console_api`)
2. Model ID variant extraction (`model_id_variant`)
3. `profiler-config.json` manual overrides (`config`)
4. LiteLLM external data (`litellm`)

**Size Categories:**
- **Large**: ≥128K tokens (green)
- **Medium**: ≥32K tokens (blue)
- **Small**: <32K tokens (orange)

#### Modalities Object

```json
{
  "modalities": {
    "input_modalities": ["TEXT", "IMAGE"],
    "output_modalities": ["TEXT"]
  }
}
```

| Value | Description |
|-------|-------------|
| `TEXT` | Text input/output |
| `IMAGE` | Image input/output |
| `VIDEO` | Video input/output |
| `AUDIO` | Audio input/output |
| `EMBEDDING` | Embedding output |

#### Lifecycle Object

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
| `global_status` | string | Overall status across regions (ACTIVE, LEGACY, EOL, MIXED) |
| `primary_status` | string | Most restrictive status |
| `regional_status` | object | Per-region status details |
| `status_summary` | object | Lists of regions by status |
| `release_date` | string | Initial release date |
| `eol_date` | string | End-of-life date (if scheduled) |
| `legacy_date` | string | Legacy transition date |
| `recommended_replacement` | string | Suggested replacement model name |
| `recommended_model_id` | string | Suggested replacement model ID |

#### Pricing Reference Object

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

#### API Support Object

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

#### Chat Features Object

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

#### Features Object (Bedrock Feature Support)

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

#### Quotas Object

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
        "global_quota": false,
        "usage_metric": {},
        "period": {}
      }
    ],
    "us-west-2": [ /* ... */ ]
  }
}
```

#### Consumption Options

| Option | Description |
|--------|-------------|
| `on_demand` | Pay-per-use inference |
| `batch` | Asynchronous batch inference |
| `cross_region_inference` | Cross-region inference (CRIS) |
| `provisioned_throughput` | Provisioned throughput capacity |
| `reserved` | Reserved Capacity (commitment-based) |
| `mantle` | OpenAI-compatible endpoint |

#### Collection Metadata

```json
{
  "collection_metadata": {
    "first_discovered_at": "2026-03-05T06:00:00Z",
    "first_discovered_in_region": "us-east-1",
    "api_source": "list_foundation_models",
    "dual_region_collection": true,
    "regions_collected_from": ["us-east-1", "us-west-2"],
    "phase2_regional_discovery": true,
    "regional_data_source": "api_discovery",
    "extraction_regions": ["us-east-1", "us-west-2"]
  }
}
```

---

## bedrock_pricing.json

### Pricing Structure

```json
{
  "metadata": {
    "generated_at": "2026-03-05T06:15:32Z",
    "version": "1.0.0",
    "total_pricing_entries": 3500,
    "total_providers": 18,
    "group_types_available": [
      "On-Demand",
      "On-Demand Global",
      "On-Demand Long Context",
      "Batch",
      "Batch Global",
      "Provisioned Throughput",
      "Custom Model",
      "Mantle"
    ],
    "service_codes_collected": [
      "AmazonBedrock",
      "AmazonBedrockService",
      "AmazonBedrockFoundationModels"
    ]
  },
  "providers": {
    "<provider_name>": {
      "<model_key>": {
        "model_name": "Claude 3.5 Sonnet",
        "model_provider": "Anthropic",
        "pricing_types": ["on_demand", "batch", "provisioned"],
        "available_dimensions": {
          "sources": ["standard", "mantle"],
          "geos": ["regional"],
          "tiers": ["flex", "priority"],
          "contexts": ["standard", "long"]
        },
        "has_mantle_pricing": true,
        "regions": {
          "us-east-1": {
            "pricing_groups": { /* ... */ }
          }
        }
      }
    }
  }
}
```

### Pricing Groups

The `pricing_groups` object categorizes prices by consumption type:

```json
{
  "pricing_groups": {
    "On-Demand": [
      {
        "dimension": "input_tokens",
        "price_per_unit": 0.003,
        "price_per_thousand": 0.003,
        "unit": "1K tokens",
        "description": "USD 0.003 per 1,000 input tokens",
        "sku": "ABC123DEF456",
        "dimensions": {
          "source": "standard",
          "geo": null,
          "tier": null,
          "context": "standard"
        }
      },
      {
        "dimension": "output_tokens",
        "price_per_unit": 0.015,
        "price_per_thousand": 0.015,
        "unit": "1K tokens",
        "description": "USD 0.015 per 1,000 output tokens",
        "sku": "XYZ789GHI012",
        "dimensions": {
          "source": "standard",
          "geo": null,
          "tier": null,
          "context": "standard"
        }
      }
    ],
    "On-Demand Long Context": [ /* Extended context pricing */ ],
    "Batch": [ /* 50% of on-demand */ ],
    "Provisioned Throughput": [ /* Per model unit */ ],
    "Mantle": [ /* OpenAI-compatible pricing */ ]
  }
}
```

### Pricing Groups Explained

| Group | Description | Typical Discount |
|-------|-------------|------------------|
| **On-Demand** | Standard in-region pay-per-use | Baseline |
| **On-Demand Global** | CRIS Global pricing (same from any source) | 0-10% premium |
| **On-Demand Geo** | CRIS Geo pricing (varies by source region) | 0-10% premium |
| **On-Demand Long Context** | Extended context (>128K tokens) | ~25% premium |
| **Batch** | Asynchronous batch processing | ~50% discount |
| **Batch Global/Geo** | CRIS batch pricing | ~50% discount + geo |
| **Batch Long Context** | Batch + extended context | ~25% discount |
| **Provisioned Throughput** | Reserved capacity per model unit | Volume discount |
| **Reserved X Month Global/Geo** | Commitment-based (1-3 month) | ~30-50% discount |
| **Custom Model** | Fine-tuned model inference | Varies |
| **Mantle** | OpenAI-compatible endpoint pricing | Baseline |

### Pricing Dimensions

Each pricing entry includes a `dimensions` object for granular filtering:

```json
{
  "dimensions": {
    "source": "standard",
    "geo": null,
    "tier": null,
    "context": "standard"
  }
}
```

| Dimension | Values | Description |
|-----------|--------|-------------|
| `source` | `standard`, `mantle` | Pricing source - standard Bedrock or Mantle |
| `geo` | `null`, `regional`, `global` | Geographic scope for CRIS |
| `tier` | `null`, `flex`, `priority` | Service tier |
| `context` | `standard`, `long` | Context window type |

### Model-Level Dimension Summary

Each model includes an `available_dimensions` summary:

```json
{
  "available_dimensions": {
    "sources": ["standard", "mantle"],
    "geos": ["regional"],
    "tiers": ["flex", "priority"],
    "contexts": ["standard", "long"]
  },
  "has_mantle_pricing": true
}
```

---

## Lambda Contributions

This table shows which Lambda function contributes which data to the final output:

| Lambda | Output Location | Contributes To | Key Fields |
|--------|-----------------|----------------|------------|
| **region-discovery** | State machine | Config | Active regions list |
| **config-sync** | `config/frontend-config.json` | Frontend | Provider colors, region metadata |
| **model-extractor** | `executions/{id}/models/{region}.json` | `bedrock_models.json` | `model_id`, `model_name`, `model_arn`, `model_provider`, `inference_types_supported`, `customization`, `console_metadata` |
| **model-merger** | `executions/{id}/merged/models.json` | `bedrock_models.json` | Deduplicated model list, `extraction_regions` |
| **pricing-collector** | `executions/{id}/pricing/{code}.json` | `bedrock_pricing.json` | Raw pricing entries |
| **pricing-aggregator** | `executions/{id}/merged/pricing.json` | `bedrock_pricing.json` | Structured pricing by provider/model/region |
| **pricing-linker** | `executions/{id}/intermediate/models-with-pricing.json` | `bedrock_models.json` | `pricing`, `model_pricing`, `has_pricing` |
| **quota-collector** | `executions/{id}/quotas/{region}.json` | `bedrock_models.json` | `quotas` |
| **regional-availability** | `executions/{id}/intermediate/regional-availability.json` | `bedrock_models.json` | `in_region`, `availability.on_demand`, `availability.provisioned` |
| **feature-collector** | `executions/{id}/features/{region}.json` | `bedrock_models.json` | `availability.cross_region`, `consumption_options` |
| **token-specs-collector** | `executions/{id}/intermediate/token-specs.json` | `bedrock_models.json` | `specs.context_window`, `specs.max_output` (fallback) |
| **mantle-collector** | `executions/{id}/mantle/{region}.json` | `bedrock_models.json` | `availability.mantle`, `api.chat_completions`, `api.responses_api` |
| **lifecycle-collector** | `executions/{id}/lifecycle/lifecycle.json` | `bedrock_models.json` | `lifecycle` |
| **final-aggregator** | `executions/{id}/final/*.json` | Both | Final merge of all data |
| **gap-detection** | `agent/gap-reports/{id}/gap-analysis.json` | Triggers self-healing | Gap analysis report |
| **self-healing-agent** | `config/profiler-config.json` | Config updates | Auto-applied fixes |
| **copy-to-latest** | `latest/*.json` | Production files | Copies final to latest |

---

## Schema Examples

### Complete Model Example

```json
{
  "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
  "model_arn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0",
  "model_name": "Claude 3.5 Sonnet v2",
  "model_provider": "Anthropic",
  
  "in_region": ["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-1"],
  
  "customization": {
    "customization_supported": ["FINE_TUNING"],
    "customization_options": {}
  },
  "inference_types_supported": ["ON_DEMAND", "PROVISIONED"],
  
  "description": "Claude 3.5 Sonnet is Anthropic's most intelligent model...",
  "short_description": "Most intelligent Claude model for complex tasks",
  
  "chat_features": {
    "function_calling": true,
    "function_calling_streaming": true,
    "citations": false,
    "documents": true,
    "chat_history": true,
    "system_role": true,
    "reasoning": { "embedded": false, "budget_control": false },
    "supported_image_types": ["png", "jpeg", "gif", "webp"],
    "supported_video_types": [],
    "supported_audio_types": [],
    "supported_document_types": ["pdf", "txt", "docx"]
  },
  
  "consumption_options": ["on_demand", "batch", "cross_region_inference", "mantle", "provisioned_throughput", "reserved"],
  
  "collection_metadata": {
    "first_discovered_at": "2024-10-22T00:00:00Z",
    "first_discovered_in_region": "us-east-1",
    "api_source": "list_foundation_models",
    "dual_region_collection": true,
    "regions_collected_from": ["us-east-1", "us-west-2"],
    "phase2_regional_discovery": true,
    "regional_data_source": "api_discovery",
    "extraction_regions": ["us-east-1", "us-west-2"]
  },
  
  "has_pricing": true,
  "has_quotas": true,
  
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
          "description": ""
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
    },
    "reserved": {
      "supported": true,
      "regions": ["us-east-1", "us-west-2", "eu-west-1"],
      "commitments": ["1_month", "3_month"]
    }
  },

  "modalities": {
    "input_modalities": ["TEXT", "IMAGE"],
    "output_modalities": ["TEXT"]
  },
  
  "capabilities": ["chat", "function_calling", "image_understanding", "code_generation"],
  "use_cases": ["Complex agentic systems", "Visual analysis", "Code generation"],
  
  "lifecycle": {
    "status": "ACTIVE",
    "global_status": "ACTIVE",
    "primary_status": "ACTIVE",
    "regional_status": {
      "us-east-1": { "status": "ACTIVE", "launch_date": "2024-10-22" }
    },
    "status_summary": {
      "ACTIVE": ["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-1"],
      "LEGACY": [],
      "EOL": []
    },
    "release_date": "2024-10-22"
  },
  
  "streaming": true,
  "languages": ["English", "French", "German", "Spanish", "Italian", "Portuguese"],
  
  "docs": {
    "aws_bedrock_guide": "https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html",
    "pricing_guide": "https://aws.amazon.com/bedrock/pricing/",
    "provider_guide": "https://docs.anthropic.com/en/docs/intro-to-claude"
  },
  
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
  },
  
  "specs": {
    "context_window": 200000,
    "max_output": 8192,
    "extended_context": null,
    "size_category": {
      "category": "Large",
      "color": "#10B981",
      "tier": 3
    },
    "source": "bedrock_console_api",
    "verified": true
  },
  
  "pricing": {
    "available": true,
    "reference": {
      "provider": "Anthropic",
      "model_key": "anthropic.claude-3-5-sonnet-20241022-v2:0",
      "model_name": "Claude 3.5 Sonnet v2"
    }
  },
  
  "model_pricing": {
    "is_pricing_available": true,
    "pricing_reference_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "pricing_file_reference": {
      "provider": "Anthropic",
      "model_key": "anthropic.claude-3-5-sonnet-20241022-v2:0",
      "model_name": "Claude 3.5 Sonnet v2"
    },
    "pricing_summary": {
      "integration_source": "amazon-bedrock-pricing-collector",
      "has_pricing_data": true,
      "integration_timestamp": "2026-03-05T06:15:32Z",
      "reference_based": true
    },
    "regions": {},
    "total_regions": 4,
    "confidence": 1.0
  },
  
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
        "global_quota": false,
        "usage_metric": {},
        "period": {}
      }
    ]
  },
  
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

### Complete Pricing Example

```json
{
  "providers": {
    "Anthropic": {
      "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "model_name": "Claude 3.5 Sonnet v2",
        "model_provider": "Anthropic",
        "pricing_types": ["on_demand", "batch", "provisioned"],
        "available_dimensions": {
          "sources": ["standard", "mantle"],
          "geos": [],
          "tiers": [],
          "contexts": ["standard"]
        },
        "has_mantle_pricing": true,
        "regions": {
          "us-east-1": {
            "pricing_groups": {
              "On-Demand": [
                {
                  "dimension": "input_tokens",
                  "price_per_unit": 0.003,
                  "price_per_thousand": 0.003,
                  "unit": "1K tokens",
                  "description": "USD 0.003 per 1,000 input tokens",
                  "sku": "ABC123DEF456",
                  "dimensions": {
                    "source": "standard",
                    "geo": null,
                    "tier": null,
                    "context": "standard"
                  }
                },
                {
                  "dimension": "output_tokens",
                  "price_per_unit": 0.015,
                  "price_per_thousand": 0.015,
                  "unit": "1K tokens",
                  "description": "USD 0.015 per 1,000 output tokens",
                  "sku": "XYZ789GHI012",
                  "dimensions": {
                    "source": "standard",
                    "geo": null,
                    "tier": null,
                    "context": "standard"
                  }
                }
              ],
              "Batch": [
                {
                  "dimension": "input_tokens",
                  "price_per_unit": 0.0015,
                  "price_per_thousand": 0.0015,
                  "unit": "1K tokens",
                  "description": "USD 0.0015 per 1,000 input tokens (batch)",
                  "sku": "BAT123ABC456",
                  "dimensions": {
                    "source": "standard",
                    "geo": null,
                    "tier": null,
                    "context": "standard"
                  }
                }
              ],
              "Mantle": [
                {
                  "dimension": "input_tokens",
                  "price_per_unit": 0.003,
                  "price_per_thousand": 0.003,
                  "unit": "1K tokens",
                  "description": "USD 0.003 per 1,000 input tokens (Mantle)",
                  "sku": "MNT123XYZ789",
                  "dimensions": {
                    "source": "mantle",
                    "geo": null,
                    "tier": null,
                    "context": "standard"
                  }
                }
              ]
            }
          }
        }
      }
    }
  }
}
```

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | System architecture overview |
| [DATA_SOURCES.md](./DATA_SOURCES.md) | API documentation and data sources |
| [backend/lambdas/README.md](../backend/lambdas/README.md) | Lambda contracts and interfaces |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-03-05 | Initial comprehensive documentation |
