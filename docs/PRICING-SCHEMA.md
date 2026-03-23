# Bedrock Model Profiler - Pricing Schema

> **Version:** 2.0  
> **Last Updated:** March 2026  
> **Output File:** `latest/bedrock_pricing.json`

This document provides a complete reference for the pricing data schema produced by the Bedrock Model Profiler.

---

## Table of Contents

1. [Overview](#overview)
2. [Top-Level Structure](#top-level-structure)
3. [Pricing Groups](#pricing-groups)
4. [Pricing Dimensions](#pricing-dimensions)
5. [Pricing Entry Structure](#pricing-entry-structure)
6. [Pricing Types by Model Category](#pricing-types-by-model-category)
7. [Examples](#examples)

---

## Overview

The `bedrock_pricing.json` file contains all pricing information organized by provider and model.

| Property | Value |
|----------|-------|
| **File Size** | ~1-2 MB |
| **Update Frequency** | Twice daily (6 AM and 6 PM UTC) |
| **Structure** | Provider → Model → Region → Pricing Groups |
| **Pricing Entries** | ~3,500+ |
| **Service Codes** | AmazonBedrock, AmazonBedrockService, AmazonBedrockFoundationModels |

---

## Top-Level Structure

```json
{
  "metadata": {
    "generated_at": "2026-03-06T06:15:32Z",
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
    "Anthropic": {
      "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "model_name": "Claude 3.5 Sonnet v2",
        "model_provider": "Anthropic",
        "pricing_types": ["on_demand", "batch", "provisioned"],
        "available_dimensions": { /* ... */ },
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

### Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `generated_at` | string | ISO 8601 timestamp |
| `version` | string | Schema version |
| `total_pricing_entries` | number | Total pricing dimension count |
| `total_providers` | number | Unique provider count |
| `group_types_available` | array | All pricing group types present |
| `service_codes_collected` | array | AWS pricing service codes used |

---

## Pricing Groups

Pricing is organized into groups based on consumption type:

### Standard Groups

| Group | Description | Typical Use |
|-------|-------------|-------------|
| **On-Demand** | Standard pay-per-use | Default pricing |
| **Batch** | Asynchronous batch processing | ~50% discount |
| **Provisioned Throughput** | Reserved capacity | Per model unit |
| **Custom Model** | Fine-tuned models | Custom training |

### Cross-Region Inference (CRIS) Groups

| Group | Description | Pricing Behavior |
|-------|-------------|------------------|
| **On-Demand Global** | Global CRIS pricing | Same from any source region |
| **On-Demand Geo** | Geographic CRIS pricing | Varies by source region |
| **Batch Global** | Batch + Global CRIS | Combined discounts |
| **Batch Geo** | Batch + Geo CRIS | Combined discounts |

### Extended Context Groups

| Group | Description | Applies To |
|-------|-------------|------------|
| **On-Demand Long Context** | Extended context window | >128K tokens |
| **On-Demand Long Context Global** | Extended + Global CRIS | Combined |
| **On-Demand Long Context Geo** | Extended + Geo CRIS | Combined |
| **Batch Long Context** | Batch + Extended context | Combined |

### Reserved Pricing Groups

| Group | Description | Commitment |
|-------|-------------|------------|
| **Reserved 1 Month Global** | 1-month commitment | ~30% discount |
| **Reserved 3 Month Global** | 3-month commitment | ~50% discount |
| **Reserved 1 Month Geo** | 1-month + Geo CRIS | Combined |
| **Reserved 3 Month Geo** | 3-month + Geo CRIS | Combined |

### Special Groups

| Group | Description |
|-------|-------------|
| **Mantle** | OpenAI-compatible endpoint pricing |

---

## Pricing Dimensions

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

### Dimension Definitions

| Dimension | Values | Description |
|-----------|--------|-------------|
| `source` | `standard`, `mantle` | Pricing source |
| `geo` | `null`, `regional`, `global` | Geographic scope |
| `tier` | `null`, `flex`, `priority` | Service tier |
| `context` | `standard`, `long` | Context window type |

### Model-Level Dimension Summary

Each model includes an `available_dimensions` summary:

```json
{
  "available_dimensions": {
    "sources": ["standard", "mantle"],
    "geos": ["regional", "global"],
    "tiers": [],
    "contexts": ["standard", "long"]
  },
  "has_mantle_pricing": true
}
```

---

## Pricing Entry Structure

Each pricing entry follows this structure:

```json
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
}
```

### Entry Fields

| Field | Type | Description |
|-------|------|-------------|
| `dimension` | string | Pricing dimension (input_tokens, output_tokens, etc.) |
| `price_per_unit` | number | Price in USD |
| `price_per_thousand` | number | Price per 1K units (for tokens) |
| `unit` | string | Unit of measurement |
| `description` | string | Human-readable description |
| `sku` | string | AWS pricing SKU |
| `dimensions` | object | Dimension filters |

### Common Dimensions

| Dimension | Description | Model Types |
|-----------|-------------|-------------|
| `input_tokens` | Input token pricing | Text models |
| `output_tokens` | Output token pricing | Text models |
| `image` | Image generation/processing | Image models |
| `video_second` | Per-second video pricing | Video models |
| `search_unit` | Search/rerank pricing | Rerank models |
| `embedding` | Embedding generation | Embedding models |
| `model_unit` | Provisioned throughput | Provisioned models |

---

## Pricing Types by Model Category

### Text Generation Models

```json
{
  "pricing_groups": {
    "On-Demand": [
      {
        "dimension": "input_tokens",
        "price_per_unit": 0.003,
        "unit": "1K tokens"
      },
      {
        "dimension": "output_tokens",
        "price_per_unit": 0.015,
        "unit": "1K tokens"
      }
    ],
    "Batch": [
      {
        "dimension": "input_tokens",
        "price_per_unit": 0.0015,
        "unit": "1K tokens"
      },
      {
        "dimension": "output_tokens",
        "price_per_unit": 0.0075,
        "unit": "1K tokens"
      }
    ]
  }
}
```

### Image Generation Models

```json
{
  "pricing_groups": {
    "On-Demand": [
      {
        "dimension": "image",
        "price_per_unit": 0.04,
        "unit": "image"
      }
    ]
  }
}
```

Common image pricing dimensions:

| Dimension | Description | Example Models |
|-----------|-------------|----------------|
| `image` | Standard image generation | Nova Canvas |
| `image_512` | 512x512 resolution | Stability SD |
| `image_1024` | 1024x1024 resolution | Stability SD |
| `background_removal` | Background removal | Stability |
| `upscale` | Image upscaling | Stability |

### Video Generation Models

```json
{
  "pricing_groups": {
    "On-Demand": [
      {
        "dimension": "video_second",
        "price_per_unit": 0.12,
        "unit": "second"
      }
    ]
  }
}
```

### Embedding Models

```json
{
  "pricing_groups": {
    "On-Demand": [
      {
        "dimension": "input_tokens",
        "price_per_unit": 0.0001,
        "unit": "1K tokens"
      }
    ]
  }
}
```

### Rerank Models

```json
{
  "pricing_groups": {
    "On-Demand": [
      {
        "dimension": "search_unit",
        "price_per_unit": 0.001,
        "unit": "search unit"
      }
    ]
  }
}
```

---

## Examples

### Complete Model Pricing (Claude 3.5 Sonnet)

```json
{
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
```

### CRIS Pricing (Cross-Region Inference)

```json
{
  "pricing_groups": {
    "On-Demand Global": [
      {
        "dimension": "input_tokens",
        "price_per_unit": 0.0033,
        "unit": "1K tokens",
        "description": "Global cross-region inference",
        "dimensions": {
          "source": "standard",
          "geo": "global",
          "tier": null,
          "context": "standard"
        }
      }
    ],
    "On-Demand Geo": [
      {
        "dimension": "input_tokens",
        "price_per_unit": 0.0031,
        "unit": "1K tokens",
        "description": "Regional cross-region inference",
        "dimensions": {
          "source": "standard",
          "geo": "regional",
          "tier": null,
          "context": "standard"
        }
      }
    ]
  }
}
```

### Long Context Pricing

```json
{
  "pricing_groups": {
    "On-Demand Long Context": [
      {
        "dimension": "input_tokens",
        "price_per_unit": 0.00375,
        "unit": "1K tokens",
        "description": "Extended context (>128K tokens)",
        "dimensions": {
          "source": "standard",
          "geo": null,
          "tier": null,
          "context": "long"
        }
      }
    ]
  }
}
```

### Provisioned Throughput Pricing

```json
{
  "pricing_groups": {
    "Provisioned Throughput": [
      {
        "dimension": "model_unit",
        "price_per_unit": 37.80,
        "unit": "model unit hour",
        "description": "Per model unit per hour"
      }
    ]
  }
}
```

---

## Pricing Calculation Guide

### Token-Based Models

```
Cost = (input_tokens / 1000) × input_price + (output_tokens / 1000) × output_price
```

**Example:** Claude 3.5 Sonnet with 10K input, 2K output
```
Cost = (10000/1000) × $0.003 + (2000/1000) × $0.015
     = $0.03 + $0.03
     = $0.06
```

### Image Generation Models

```
Cost = number_of_images × price_per_image
```

### Video Generation Models

```
Cost = video_duration_seconds × price_per_second
```

### Batch Inference Discount

Batch pricing is typically **50% of on-demand**:
```
Batch Cost = On-Demand Cost × 0.5
```

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | System architecture overview |
| [DATA-SOURCES.md](./DATA-SOURCES.md) | Data sources and reliability |
| [DATA-SCHEMA.md](./DATA-SCHEMA.md) | Model JSON schema reference |

---

## Changelog

| Date | Version | Change |
|------|---------|--------|
| 2026-03-06 | 2.0 | Separated into dedicated pricing schema document |
| 2026-03-05 | 1.0 | Initial documentation |
