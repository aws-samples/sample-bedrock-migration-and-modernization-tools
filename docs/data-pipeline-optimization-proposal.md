# Data Pipeline Optimization Proposal

## Executive Summary

This document analyzes the current data pipeline output structure for the Amazon Bedrock Model Profiler and proposes optimizations to improve consistency between frontend and backend, reduce redundancy, and simplify data access patterns.

---

## Table of Contents

1. [Current Architecture Overview](#current-architecture-overview)
2. [Data Flow Diagram](#data-flow-diagram)
3. [Current Data Structures](#current-data-structures)
4. [Identified Problems](#identified-problems)
5. [Proposed Solutions](#proposed-solutions)
   - [Option A: Region-Centric Unified Structure](#option-a-region-centric-unified-structure)
   - [Option B: Hybrid with Summary Arrays](#option-b-hybrid-with-summary-arrays)
   - [Option C: Minimal Changes - Add Computed Views](#option-c-minimal-changes---add-computed-views)
6. [Comparison Matrix](#comparison-matrix)
7. [Migration Strategy](#migration-strategy)
8. [Recommendation](#recommendation)

---

## Current Architecture Overview

### Pipeline Output Files

| File | Description | Size (approx) |
|------|-------------|---------------|
| `latest/bedrock_models.json` | Model metadata, capabilities, availability | ~128 models |
| `latest/bedrock_pricing.json` | Detailed per-region pricing | ~14,678 entries |

### Intermediate Files

| File | Producer Lambda | Purpose |
|------|-----------------|---------|
| `merged/models.json` | `model-merger` | Combined model data from us-east-1 + us-west-2 |
| `intermediate/regional-availability.json` | `regional-availability` | Per-region model discovery |
| `intermediate/features.json` | `feature-collector` | Cross-region inference profiles |
| `intermediate/token-specs.json` | `token-specs-collector` | Context window & output limits |
| `quotas/{region}.json` | `quota-collector` | Service quotas per region |
| `mantle/{region}.json` | `mantle-collector` | Mantle API availability per region |
| `pricing/*.json` | `pricing-collector` | Raw pricing from AWS Pricing API |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Step Functions Workflow                              │
│                         (Daily at 6 AM UTC)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│   pricing-    │           │    model-     │           │    quota-     │
│   collector   │           │   extractor   │           │   collector   │
│  (3 parallel) │           │  (2 regions)  │           │ (20 regions)  │
└───────┬───────┘           └───────┬───────┘           └───────┬───────┘
        │                           │                           │
        ▼                           ▼                           │
┌───────────────┐           ┌───────────────┐                   │
│   pricing-    │           │    model-     │                   │
│  aggregator   │           │    merger     │                   │
└───────┬───────┘           └───────┬───────┘                   │
        │                           │                           │
        │    ┌──────────────────────┼───────────────────────┐   │
        │    │                      │                       │   │
        │    ▼                      ▼                       ▼   │
        │  ┌─────────┐    ┌──────────────────┐    ┌───────────┐ │
        │  │regional-│    │    feature-      │    │  mantle-  │ │
        │  │availab. │    │    collector     │    │ collector │ │
        │  └────┬────┘    └────────┬─────────┘    └─────┬─────┘ │
        │       │                  │                    │       │
        │       └──────────────────┼────────────────────┘       │
        │                          │                            │
        ▼                          ▼                            ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           final-aggregator                                 │
│  • Merges all data sources                                                │
│  • Links pricing to models                                                │
│  • Matches quotas to models                                               │
│  • Builds availability objects                                            │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       copy-to-latest          │
                    │  • bedrock_models.json        │
                    │  • bedrock_pricing.json       │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │         CloudFront            │
                    │    /latest/bedrock_*.json     │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │          Frontend             │
                    │  • useModels.js hook          │
                    │  • Fetches both files         │
                    │  • Joins pricing to models    │
                    └───────────────────────────────┘
```

---

## Current Data Structures

### Model Object (bedrock_models.json)

```json
{
  "model_id": "anthropic.claude-sonnet-4",
  "model_arn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4",
  "model_name": "Claude Sonnet 4",
  "model_provider": "Anthropic",

  "model_modalities": {
    "input_modalities": ["TEXT", "IMAGE"],
    "output_modalities": ["TEXT"]
  },

  "streaming_supported": true,

  "customization": {
    "customization_supported": ["FINE_TUNING"],
    "customization_options": {}
  },

  "inference_types_supported": ["ON_DEMAND", "PROVISIONED"],

  "in_region": ["us-east-1", "us-west-2", "eu-west-1", ...],

  "model_lifecycle": {
    "status": "ACTIVE",
    "release_date": 1715644800.0
  },

  "model_capabilities": ["Text generation", "Code generation", ...],
  "model_use_cases": ["Chatbots", "Content creation", ...],
  "languages_supported": ["English", "French", ...],

  "description": "...",
  "short_description": "...",

  "feature_support": {
    "agent": { "isStreamingSupported": true, "isSupported": true },
    "knowledge_base": { "isExternalSourcesSupported": true, "isParsingSupported": true, "isSupported": true },
    "flow": { "isSupported": true },
    "guardrails": { "isSupported": true },
    "prompt_caching": { "isSupported": true },
    "intelligent_routing": { "isSupported": false },
    "model_evaluation": { "isSupported": true },
    "prompt_management": { "isSupported": true },
    "batch_inference": { "baseModelSupported": true, "crossRegionSupported": true, "customModelSupported": false, "tokenizerSupported": false },
    "latency_optimized": { "isSupported": true },
    "system_tools": ["computer_use", "text_editor"]
  },

  "chat_features": {
    "function_calling": true,
    "function_calling_streaming": true,
    "citations": true,
    "documents": true,
    "chat_history": true,
    "system_role": true,
    "reasoning": { "enabled": true, "budget_tokens": true },
    "supported_image_types": ["png", "jpeg", "gif", "webp"],
    "supported_video_types": [],
    "supported_audio_types": [],
    "supported_document_types": ["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"]
  },

  "consumption_options": ["on_demand", "batch", "provisioned_throughput", "cross_region_inference", "mantle"],

  "cross_region_inference": {
    "supported": true,
    "profiles_count": 5,
    "source_regions": ["us-east-1", "us-west-2", "eu-west-1", ...],
    "profiles": [
      {
        "profile_id": "us.anthropic.claude-sonnet-4",
        "profile_name": "US Claude Sonnet 4",
        "source_region": "us-east-1",
        "target_regions": ["us-east-1", "us-west-2"]
      },
      ...
    ]
  },

  "is_mantle": true,
  "mantle_only": false,
  "mantle_inference": {
    "supported": true,
    "mantle_regions": ["us-east-1", "us-west-2", "eu-west-1"],
    "total_mantle_regions": 3,
    "mantle_endpoint_pattern": "bedrock-mantle.{region}.api.aws",
    "matched_mantle_id": "anthropic.claude-sonnet-4",
    "supports_responses_api": true
  },

  "provisioned_throughput": {
    "supported": true,
    "provisioned_regions": ["us-east-1", "us-west-2"],
    "total_provisioned_regions": 2
  },

  "documentation_links": {
    "aws_bedrock_guide": "https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html",
    "pricing_guide": "https://aws.amazon.com/bedrock/pricing/"
  },

  "model_pricing": {
    "is_pricing_available": true,
    "pricing_reference_id": "anthropic.claude-sonnet-4",
    "pricing_file_reference": {
      "provider": "Anthropic",
      "model_key": "anthropic.claude-sonnet-4",
      "model_name": "Claude Sonnet 4"
    },
    "pricing_summary": {
      "integration_source": "amazon-bedrock-pricing-collector",
      "has_pricing_data": true,
      "integration_timestamp": "2026-02-25T19:13:09Z",
      "reference_based": true
    }
  },

  "model_service_quotas": {
    "us-east-1": [
      {
        "quota_code": "L-ABC123",
        "quota_name": "On-demand model inference tokens per minute for Anthropic Claude Sonnet 4",
        "quota_arn": "arn:aws:servicequotas:us-east-1:...:bedrock/L-ABC123",
        "description": "",
        "quota_applied_at_level": "ACCOUNT",
        "value": 400000.0,
        "unit": "None",
        "adjustable": true,
        "global_quota": false,
        "usage_metric": {},
        "period": {}
      },
      ...
    ],
    "eu-west-1": [...]
  },

  "collection_metadata": {
    "first_discovered_at": "2026-02-25 19:12:42 UTC",
    "first_discovered_in_region": "us-east-1",
    "api_source": "list_foundation_models",
    "dual_region_collection": true,
    "regions_collected_from": ["us-east-1", "us-west-2"],
    "phase2_regional_discovery": true,
    "regional_data_source": "api_discovery",
    "extraction_regions": ["us-east-1", "us-west-2"]
  },

  "regional_availability_source": "api_discovery",
  "total_in_region": 15,

  "batch_inference_supported": {
    "supported": true,
    "supported_regions": ["us-east-1", "us-west-2", "eu-west-1", ...],
    "coverage_percentage": 100.0,
    "detection_method": "pricing_data"
  },

  "converse_data": {
    "context_window": 200000,
    "max_output_tokens": 64000,
    "size_category": { "category": "Large", "color": "#10B981", "tier": 3 },
    "verified": true,
    "source": "bedrock_console_api",
    "litellm_verified": true,
    "capabilities_count": 8,
    "use_cases_count": 6,
    "regions_count": 15,
    "has_extended_context": false
  },

  "has_pricing": true,
  "has_quotas": true,

  "api_support": {
    "invoke_model": { "supported": true, "streaming": true, "endpoint": "bedrock-runtime" },
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
        "citations": true,
        "reasoning": true
      }
    },
    "chat_completions": { "supported": true, "endpoints": ["bedrock-runtime", "bedrock-mantle"] },
    "responses_api": { "supported": true, "endpoint": "bedrock-mantle" },
    "endpoints_supported": ["bedrock-runtime", "bedrock-mantle"]
  },

  "endpoint_availability": {
    "bedrock_runtime": {
      "regions": ["us-east-1", "us-west-2", "eu-west-1", ...],
      "apis": ["invoke_model", "converse", "chat_completions"]
    },
    "bedrock_mantle": {
      "regions": ["us-east-1", "us-west-2", "eu-west-1"],
      "apis": ["chat_completions", "responses_api"]
    }
  }
}
```

### Pricing Object (bedrock_pricing.json)

```json
{
  "metadata": {
    "providersCount": 22,
    "totalProducts": 14678,
    "group_types_seen": ["Batch", "On-Demand", "Provisioned Throughput", ...]
  },
  "providers": {
    "Anthropic": {
      "anthropic.claude-sonnet-4": {
        "model_name": "Claude Sonnet 4",
        "model_provider": "Anthropic",
        "pricing_types": ["token"],
        "primary_pricing_type": "token",
        "regions": {
          "us-east-1": {
            "pricing_groups": {
              "On-Demand": [
                {
                  "dimension": "USE1-Claude4Sonnet-input-tokens",
                  "price_per_unit": 0.003,
                  "price_per_thousand": 0.003,
                  "unit": "1K tokens",
                  "description": "$0.003 per 1K token for Claude4Sonnet-input-tokens in US East (N. Virginia)",
                  "is_input": true,
                  "is_output": false,
                  "pricing_characteristics": {
                    "inference_type": "on_demand",
                    "context_type": "standard",
                    "geographic_scope": "regional"
                  }
                },
                ...
              ],
              "On-Demand Long Context": [...],
              "Batch": [...],
              "Provisioned Throughput": [...]
            }
          },
          "eu-west-1": {...}
        }
      }
    }
  }
}
```

### Quota Object (per region)

```json
{
  "quota_code": "L-ABC123",
  "quota_name": "On-demand model inference tokens per minute for Anthropic Claude Sonnet 4",
  "quota_arn": "arn:aws:servicequotas:us-east-1:169497827606:bedrock/L-ABC123",
  "description": "",
  "quota_applied_at_level": "ACCOUNT",
  "value": 400000.0,
  "unit": "None",
  "adjustable": true,
  "global_quota": false,
  "usage_metric": {},
  "period": {}
}
```

---

## Identified Problems

### 1. Availability Data Fragmentation

**Problem:** Model availability information is scattered across 7+ different locations in the model object.

| Data | Location | Format |
|------|----------|--------|
| On-demand regions | `in_region[]` | Array of region codes |
| CRIS source regions | `cross_region_inference.source_regions[]` | Array of region codes |
| CRIS profiles | `cross_region_inference.profiles[]` | Array of profile objects |
| Mantle regions | `mantle_inference.mantle_regions[]` | Array of region codes |
| Batch regions | `batch_inference_supported.supported_regions[]` | Array of region codes |
| Provisioned regions | `provisioned_throughput.provisioned_regions[]` | Array of region codes |
| Endpoint availability | `endpoint_availability.*.regions[]` | Nested arrays |
| Quotas | `model_service_quotas[region][]` | Object keyed by region |

**Frontend Impact:** Components like `RegionalAvailability.jsx` must reconstruct per-region availability from multiple arrays:

```javascript
// Current: Must check 5 different places
function getRegionAvailability(model, regionCode) {
  const inRegionList = model.in_region || []
  const crisRegions = model.cross_region_inference?.source_regions || []
  const mantleRegions = model.mantle_inference?.mantle_regions || []

  const onDemand = inRegionList.includes(regionCode)
  const cris = crisRegions.includes(regionCode)
  const mantle = mantleRegions.includes(regionCode)

  return { available: onDemand || cris || mantle, onDemand, cris, mantle }
}
```

### 2. Duplicate/Redundant Data

| Field | Duplicates | Notes |
|-------|------------|-------|
| `total_in_region` | `in_region.length` | Computed value stored |
| `cross_region_inference.profiles_count` | `profiles.length` | Computed value stored |
| `converse_data.capabilities_count` | `model_capabilities.length` | Computed value stored |
| `converse_data.regions_count` | `in_region.length` | Same as total_in_region |
| `mantle_inference.total_mantle_regions` | `mantle_regions.length` | Computed value stored |
| `provisioned_throughput.total_provisioned_regions` | `provisioned_regions.length` | Computed value stored |

### 3. Feature/Capability Fragmentation

Features are spread across 5 different objects:

```
model
├── streaming_supported (boolean)
├── feature_support
│   ├── agent.isSupported
│   ├── knowledge_base.isSupported
│   ├── guardrails.isSupported
│   ├── prompt_caching.isSupported
│   ├── batch_inference.baseModelSupported
│   └── ... (10 properties)
├── chat_features
│   ├── function_calling
│   ├── citations
│   ├── reasoning.enabled
│   └── ... (12 properties)
├── api_support
│   ├── converse.features.*
│   └── responses_api.supported
└── customization
    └── customization_supported[]
```

**Frontend Impact:** Deep property access patterns throughout components:

```javascript
// Current: Deep nested access
const hasAgent = model.feature_support?.agent?.isSupported
const hasKnowledgeBase = model.feature_support?.knowledge_base?.isSupported
const hasFunctionCalling = model.chat_features?.function_calling
const hasReasoning = model.chat_features?.reasoning?.enabled
```

### 4. Two-File Fetch with Complex Join

**Problem:** Frontend must fetch both `bedrock_models.json` and `bedrock_pricing.json`, then join them using complex matching logic.

```javascript
// useModels.js - 140+ lines of pricing matching logic
function getModelPricing(model, pricingData) {
  // First try using pricing_file_reference (preferred method)
  const pricingRef = model.model_pricing?.pricing_file_reference
  if (pricingRef?.provider && pricingRef?.model_key) {
    const providerData = pricingData.providers[pricingRef.provider]
    if (providerData?.[pricingRef.model_key]) {
      return providerData[pricingRef.model_key]
    }
  }
  // Fallback: try matching by model_id directly
  // ... more complex matching logic
}
```

**Impact:**
- Two HTTP requests on page load
- Complex client-side join logic
- Increased bundle size from pricing parsing utilities

### 5. Verbose Quota Structure

**Problem:** Quota objects contain many fields never used by the frontend.

```json
{
  "quota_code": "L-ABC123",           // ✅ Used (for links)
  "quota_name": "...",                // ✅ Used (display)
  "quota_arn": "arn:aws:...",         // ❌ Never used
  "description": "",                  // ❌ Always empty
  "quota_applied_at_level": "ACCOUNT",// ❌ Never used
  "value": 400000.0,                  // ✅ Used
  "unit": "None",                     // ❌ Never used (always "None")
  "adjustable": true,                 // ✅ Used
  "global_quota": false,              // ❌ Never used
  "usage_metric": {},                 // ❌ Always empty
  "period": {}                        // ❌ Always empty
}
```

### 6. Inconsistent Naming Conventions

| Pattern | Examples |
|---------|----------|
| snake_case | `model_id`, `in_region`, `cross_region_inference` |
| camelCase | `isSupported`, `isStreamingSupported`, `baseModelSupported` |
| Mixed | `pricing_file_reference` (snake) contains `model_key` (snake) |

---

## Proposed Solutions

### Option A: Region-Centric Unified Structure

**Philosophy:** Reorganize all data around regions as the primary key. All availability, quotas, and access types are unified under a single `availability` object keyed by region.

#### Proposed Model Structure

```json
{
  "model_id": "anthropic.claude-sonnet-4",
  "model_name": "Claude Sonnet 4",
  "provider": "Anthropic",
  "status": "ACTIVE",
  "release_date": "2025-05-14",

  "specs": {
    "context_window": 200000,
    "max_output_tokens": 64000,
    "extended_context": null,
    "streaming": true
  },

  "modalities": {
    "input": ["TEXT", "IMAGE"],
    "output": ["TEXT"]
  },

  "availability": {
    "us-east-1": {
      "access": {
        "on_demand": true,
        "cris": ["us", "global"],
        "mantle": true,
        "batch": true,
        "provisioned": true
      },
      "quotas": {
        "on_demand": {
          "tpm": { "value": 400000, "adjustable": true, "code": "L-ABC123" },
          "rpm": { "value": 4000, "adjustable": true, "code": "L-DEF456" }
        },
        "cross_region": {
          "tpm": { "value": 2000000, "adjustable": true, "code": "L-GHI789" },
          "rpm": { "value": 10000, "adjustable": true, "code": "L-JKL012" }
        },
        "batch": {
          "job_size_gb": { "value": 200, "adjustable": true, "code": "L-MNO345" },
          "records_per_file": { "value": 50000, "adjustable": true, "code": "L-PQR678" }
        }
      },
      "pricing": {
        "on_demand": {
          "input_1k": 0.003,
          "output_1k": 0.015,
          "cache_read_1k": 0.0003,
          "cache_write_1k": 0.00375
        },
        "batch": {
          "input_1k": 0.0015,
          "output_1k": 0.0075
        }
      }
    },
    "eu-west-1": {
      "access": {
        "on_demand": true,
        "cris": ["eu"],
        "mantle": false,
        "batch": true,
        "provisioned": false
      },
      "quotas": {
        "on_demand": {
          "tpm": { "value": 200000, "adjustable": true, "code": "L-STU901" }
        }
      },
      "pricing": {
        "on_demand": {
          "input_1k": 0.003,
          "output_1k": 0.015
        }
      }
    }
  },

  "capabilities": {
    "agent": true,
    "knowledge_base": true,
    "kb_parsing": true,
    "kb_external_sources": false,
    "guardrails": true,
    "function_calling": true,
    "function_calling_streaming": true,
    "citations": true,
    "documents": true,
    "reasoning": true,
    "reasoning_budget": true,
    "prompt_caching": true,
    "batch_inference": true,
    "latency_optimized": true,
    "fine_tuning": true,
    "continued_pretraining": false,
    "system_tools": ["computer_use", "text_editor"]
  },

  "apis": {
    "invoke_model": true,
    "converse": true,
    "chat_completions": true,
    "responses_api": true
  },

  "media_support": {
    "image_input": ["png", "jpeg", "gif", "webp"],
    "video_input": [],
    "audio_input": [],
    "document_input": ["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"]
  },

  "metadata": {
    "description": "...",
    "short_description": "...",
    "capabilities_list": ["Text generation", "Code generation", ...],
    "use_cases": ["Chatbots", "Content creation", ...],
    "languages": ["English", "French", ...],
    "documentation_url": "https://docs.aws.amazon.com/bedrock/..."
  }
}
```

#### Benefits

| Benefit | Description |
|---------|-------------|
| **Single region lookup** | `model.availability["us-east-1"]` gives all info for a region |
| **No array scanning** | No need to check `includes()` on 5 arrays |
| **Unified quotas** | Quotas are with their region, pre-categorized |
| **Inline pricing** | No second file fetch needed |
| **Flat capabilities** | `model.capabilities.agent` vs `model.feature_support?.agent?.isSupported` |
| **Consistent naming** | All snake_case |

#### Drawbacks

| Drawback | Description |
|----------|-------------|
| **Breaking change** | Requires full frontend rewrite |
| **Larger per-model payload** | Pricing data embedded (but no second fetch) |
| **Migration complexity** | Significant backend + frontend changes |

#### Frontend Usage Example

```javascript
// New: Simple, direct access
function getRegionAvailability(model, regionCode) {
  const region = model.availability?.[regionCode]
  if (!region) return { available: false }

  const { access } = region
  return {
    available: access.on_demand || access.cris?.length > 0 || access.mantle,
    onDemand: access.on_demand,
    cris: access.cris || [],
    mantle: access.mantle,
    batch: access.batch,
    provisioned: access.provisioned
  }
}

// Get quota directly
const tpm = model.availability?.["us-east-1"]?.quotas?.on_demand?.tpm?.value

// Check capability
const hasAgent = model.capabilities?.agent
```

---

### Option B: Hybrid with Summary Arrays

**Philosophy:** Keep backward-compatible array structures but add pre-computed summary objects. Allows gradual migration.

#### Proposed Model Structure

```json
{
  "model_id": "anthropic.claude-sonnet-4",
  "model_name": "Claude Sonnet 4",
  "model_provider": "Anthropic",

  "specs": {
    "context_window": 200000,
    "max_output_tokens": 64000,
    "streaming": true
  },

  "model_modalities": {
    "input_modalities": ["TEXT", "IMAGE"],
    "output_modalities": ["TEXT"]
  },

  "model_lifecycle": {
    "status": "ACTIVE",
    "release_date": "2025-05-14"
  },

  "availability_summary": {
    "total_regions": 15,
    "on_demand_regions": ["us-east-1", "us-west-2", "eu-west-1", ...],
    "cris_regions": ["us-east-1", "us-west-2", "eu-west-1", ...],
    "mantle_regions": ["us-east-1", "us-west-2", "eu-west-1"],
    "batch_regions": ["us-east-1", "us-west-2", "eu-west-1", ...],
    "provisioned_regions": ["us-east-1", "us-west-2"],
    "quota_regions": ["us-east-1", "us-west-2", "eu-west-1", ...]
  },

  "availability_by_region": {
    "us-east-1": {
      "on_demand": true,
      "cris": ["us", "global"],
      "mantle": true,
      "batch": true,
      "provisioned": true,
      "has_quotas": true
    },
    "eu-west-1": {
      "on_demand": true,
      "cris": ["eu"],
      "mantle": false,
      "batch": true,
      "provisioned": false,
      "has_quotas": true
    }
  },

  "capabilities_flat": {
    "agent": true,
    "knowledge_base": true,
    "guardrails": true,
    "function_calling": true,
    "citations": true,
    "reasoning": true,
    "prompt_caching": true,
    "batch_inference": true
  },

  "pricing_summary": {
    "type": "token",
    "default_region": "us-east-1",
    "input_1k": 0.003,
    "output_1k": 0.015,
    "cache_read_1k": 0.0003,
    "cache_write_1k": 0.00375
  },

  "quotas_summary": {
    "total_quotas": 45,
    "adjustable_quotas": 12,
    "categories": ["on_demand", "cross_region", "batch"],
    "regions_with_quotas": 8
  },

  "quotas_by_region": {
    "us-east-1": {
      "on_demand": {
        "tpm": { "value": 400000, "adjustable": true, "code": "L-ABC123" },
        "rpm": { "value": 4000, "adjustable": true, "code": "L-DEF456" }
      },
      "cross_region": {
        "tpm": { "value": 2000000, "adjustable": true, "code": "L-GHI789" }
      }
    }
  },

  "in_region": ["us-east-1", "us-west-2", ...],
  "cross_region_inference": {
    "supported": true,
    "source_regions": ["us-east-1", ...],
    "profiles": [...]
  },
  "mantle_inference": {
    "supported": true,
    "mantle_regions": ["us-east-1", ...]
  },
  "provisioned_throughput": {
    "supported": true,
    "provisioned_regions": ["us-east-1", ...]
  },
  "batch_inference_supported": {
    "supported": true,
    "supported_regions": ["us-east-1", ...]
  },

  "feature_support": { ... },
  "chat_features": { ... },
  "model_service_quotas": { ... }
}
```

#### Benefits

| Benefit | Description |
|---------|-------------|
| **Backward compatible** | Old fields still present, frontend can migrate gradually |
| **New computed views** | `availability_by_region`, `capabilities_flat`, `quotas_by_region` |
| **Inline pricing summary** | No second fetch for basic pricing display |
| **Gradual deprecation** | Mark old fields as deprecated, remove in v2 |

#### Drawbacks

| Drawback | Description |
|----------|-------------|
| **Larger payload** | Duplicate data during transition |
| **Maintenance burden** | Must keep old + new in sync |
| **Partial improvement** | Still have some redundancy |

#### Frontend Migration Path

```javascript
// Phase 1: Use new fields where available, fallback to old
function getRegionAvailability(model, regionCode) {
  // New way (if available)
  if (model.availability_by_region?.[regionCode]) {
    const r = model.availability_by_region[regionCode]
    return {
      available: r.on_demand || r.cris?.length > 0 || r.mantle,
      onDemand: r.on_demand,
      cris: r.cris || [],
      mantle: r.mantle
    }
  }

  // Fallback to old way
  const inRegionList = model.in_region || []
  const crisRegions = model.cross_region_inference?.source_regions || []
  const mantleRegions = model.mantle_inference?.mantle_regions || []
  return {
    available: inRegionList.includes(regionCode) || crisRegions.includes(regionCode) || mantleRegions.includes(regionCode),
    onDemand: inRegionList.includes(regionCode),
    cris: crisRegions.includes(regionCode),
    mantle: mantleRegions.includes(regionCode)
  }
}

// Phase 2: Remove fallback, use only new fields
// Phase 3: Backend removes deprecated fields
```

---

### Option C: Minimal Changes - Add Computed Views

**Philosophy:** Keep existing structure exactly as-is, but add new computed summary fields. Lowest risk, smallest change.

#### Additions Only (No Removals)

```json
{
  "computed": {
    "availability_matrix": {
      "us-east-1": ["on_demand", "cris_us", "cris_global", "mantle", "batch", "provisioned"],
      "eu-west-1": ["on_demand", "cris_eu", "batch"],
      "ap-northeast-1": ["on_demand", "batch"]
    },

    "pricing_display": {
      "type": "token",
      "input_1k": 0.003,
      "output_1k": 0.015,
      "region": "us-east-1"
    },

    "capabilities_booleans": {
      "agent": true,
      "knowledge_base": true,
      "guardrails": true,
      "function_calling": true,
      "citations": true,
      "reasoning": true,
      "prompt_caching": true,
      "batch": true
    },

    "quotas_condensed": {
      "us-east-1": {
        "on_demand_tpm": 400000,
        "on_demand_rpm": 4000,
        "cris_tpm": 2000000,
        "adjustable_count": 4
      }
    },

    "totals": {
      "regions": 15,
      "on_demand_regions": 15,
      "cris_regions": 12,
      "mantle_regions": 3,
      "quota_regions": 8,
      "total_quotas": 45
    }
  },

  "in_region": [...],
  "cross_region_inference": {...},
  "mantle_inference": {...},
  "model_service_quotas": {...}
}
```

#### Benefits

| Benefit | Description |
|---------|-------------|
| **Zero breaking changes** | All existing code continues to work |
| **Immediate benefit** | New `computed` block available immediately |
| **Easy rollback** | Just remove `computed` block if issues |
| **Gradual adoption** | Frontend updates one component at a time |

#### Drawbacks

| Drawback | Description |
|----------|-------------|
| **Larger payload** | Adds ~20% more data |
| **Still redundant** | Original structure unchanged |
| **Tech debt** | Eventually need full cleanup |

---

## Comparison Matrix

| Criteria | Option A | Option B | Option C |
|----------|----------|----------|----------|
| **Breaking Changes** | High | Low | None |
| **Migration Effort** | High | Medium | Low |
| **Payload Size** | Smaller | Larger (temp) | Larger |
| **Code Simplicity** | Best | Good | Same |
| **Maintenance** | Lowest | Medium | Highest |
| **Time to Implement** | 2-3 weeks | 1-2 weeks | 2-3 days |
| **Risk** | Higher | Medium | Lowest |
| **Long-term Value** | Highest | High | Low |

### Payload Size Comparison (Estimated)

| Version | Single Model Size | Full File Size |
|---------|-------------------|----------------|
| Current | ~15 KB | ~1.9 MB |
| Option A | ~12 KB | ~1.5 MB |
| Option B | ~18 KB | ~2.3 MB |
| Option C | ~18 KB | ~2.3 MB |

---

## Migration Strategy

### For Option A (Recommended for New Projects)

```
Phase 1: Backend (Week 1)
├── Update final-aggregator to produce new structure
├── Output to new file: bedrock_models_v2.json
├── Keep v1 file for backward compatibility
└── Deploy and validate

Phase 2: Frontend Migration (Week 2)
├── Update useModels.js to fetch v2
├── Update RegionalAvailability.jsx
├── Update ModelCard.jsx
├── Update ModelCardExpanded.jsx
├── Update comparison components
└── Remove pricing file fetch

Phase 3: Cleanup (Week 3)
├── Remove v1 file generation
├── Remove old frontend code paths
├── Update documentation
└── Performance testing
```

### For Option B (Recommended for Gradual Migration)

```
Phase 1: Backend Additions (Week 1)
├── Add availability_by_region
├── Add capabilities_flat
├── Add pricing_summary
├── Add quotas_by_region
└── Deploy (no breaking changes)

Phase 2: Frontend Migration (Weeks 2-4)
├── Update components one at a time
├── Use feature flags for rollout
├── A/B test new vs old code paths
└── Monitor performance

Phase 3: Deprecation (Week 5+)
├── Mark old fields as deprecated
├── Remove old frontend code
├── Plan backend field removal
└── v2 release removes deprecated fields
```

### For Option C (Recommended for Quick Wins)

```
Phase 1: Add computed block (Day 1-2)
├── Update final-aggregator
├── Add computed summary fields
└── Deploy

Phase 2: Frontend updates (Day 3-5)
├── Use computed.pricing_display for cards
├── Use computed.availability_matrix for tables
├── Use computed.capabilities_booleans for filters
└── No other changes needed
```

---

## Recommendation

### Short-term (Next Sprint): Option C

Add the `computed` block to get immediate benefits:
- Inline pricing summary removes second file fetch
- Pre-computed availability matrix simplifies RegionalAvailability component
- Flattened capabilities simplify filter logic

### Medium-term (Next Quarter): Option B

Migrate to the hybrid structure:
- Add `availability_by_region` and `quotas_by_region`
- Deprecate redundant fields
- Update frontend components incrementally

### Long-term (v2 Release): Option A

For a clean v2 release:
- Full region-centric structure
- Remove all deprecated fields
- Simplified frontend codebase
- Smaller payload size

---

## Appendix: Field Mapping Reference

### Current → Option A Mapping

| Current Field | Option A Location |
|---------------|-------------------|
| `in_region[]` | `availability[region].access.on_demand` |
| `cross_region_inference.source_regions[]` | `availability[region].access.cris` |
| `cross_region_inference.profiles[]` | Removed (derived from cris array) |
| `mantle_inference.mantle_regions[]` | `availability[region].access.mantle` |
| `provisioned_throughput.provisioned_regions[]` | `availability[region].access.provisioned` |
| `batch_inference_supported.supported_regions[]` | `availability[region].access.batch` |
| `model_service_quotas[region][]` | `availability[region].quotas` |
| `feature_support.agent.isSupported` | `capabilities.agent` |
| `chat_features.function_calling` | `capabilities.function_calling` |
| `converse_data.context_window` | `specs.context_window` |
| `model_pricing.pricing_file_reference` | `availability[region].pricing` |

### Quota Category Parsing

| Quota Name Pattern | Category | Metric |
|-------------------|----------|--------|
| "On-demand model inference tokens per minute" | `on_demand` | `tpm` |
| "On-demand model inference requests per minute" | `on_demand` | `rpm` |
| "Cross-region model inference tokens per minute" | `cross_region` | `tpm` |
| "Cross-region model inference requests per minute" | `cross_region` | `rpm` |
| "Global cross-region model inference" | `cross_region_global` | varies |
| "Batch inference job size" | `batch` | `job_size_gb` |
| "Records per input file per batch inference job" | `batch` | `records_per_file` |
| "Model invocation max tokens per day" | `daily` | `tpd` |

---

*Document Version: 1.0*
*Created: 2026-02-26*
*Author: Data Pipeline Analysis*
