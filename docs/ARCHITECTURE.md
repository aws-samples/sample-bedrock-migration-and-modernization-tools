# Bedrock Model Profiler - Architecture

> **Version:** 2.1  
> **Last Updated:** March 2026  
> **Status:** Production  
> **Live URL:** https://d3oem6l61p8j11.cloudfront.net

This document is the **single source of truth** for understanding the Bedrock Model Profiler architecture.

---

## Table of Contents

1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Data Flow](#data-flow)
4. [AWS Services Used](#aws-services-used)
5. [Project Structure](#project-structure)
6. [Lambda Functions](#lambda-functions)
7. [Caching System](#caching-system)
8. [Self-Healing System](#self-healing-system)
9. [Configuration](#configuration)
10. [Deployment](#deployment)

---

## Overview

### Purpose

The **Amazon Bedrock Model Profiler** is a full-stack serverless tool for exploring, analyzing, and comparing Amazon Bedrock foundation models. It provides:

- **Comprehensive Model Catalog**: 100+ models from 17+ providers
- **Real-time Pricing Data**: Pricing across 30+ regions with multiple consumption options
- **Regional Availability Maps**: Visual representation of model availability by region
- **Model Comparison**: Side-by-side comparison of model specifications
- **Self-Healing Pipeline**: AI-powered automatic configuration updates

### Key Features

| Feature | Description |
|---------|-------------|
| **Self-Healing Pipeline** | Claude-powered gap detection and automatic config fixes |
| **17 Lambda Functions** | Modular data collection and processing |
| **Inter-Lambda Caching** | ~97% cache hit rate, ~29 API calls per execution |
| **Daily Updates** | Automated data refresh at 6 AM UTC |
| **Multi-Source Aggregation** | 7+ data sources combined into unified JSON |

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BEDROCK MODEL PROFILER                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐                                                          │
│  │  EventBridge  │─────────────────────┐                                    │
│  │ (Daily 6 AM)  │                     │                                    │
│  └───────────────┘                     │                                    │
│                                        ▼                                    │
│                               ┌──────────────────┐                          │
│                               │  Step Functions  │                          │
│                               │   (Orchestrator) │                          │
│                               └────────┬─────────┘                          │
│                                        │                                    │
│         ┌──────────────────────────────┼──────────────────────────────┐     │
│         │                              │                              │     │
│         ▼                              ▼                              ▼     │
│  ┌──────────────┐           ┌──────────────────┐           ┌────────────┐   │
│  │   PRICING    │           │     MODELS       │           │   QUOTAS   │   │
│  │ • collector  │           │ • extractor      │           │ • collector│   │
│  │ • aggregator │           │ • merger         │           │            │   │
│  └──────┬───────┘           └────────┬─────────┘           └─────┬──────┘   │
│         │                            │                           │          │
│         └────────────────────────────┼───────────────────────────┘          │
│                                      ▼                                      │
│         ┌────────────────────────────────────────────────────────┐          │
│         │                  ENRICHMENT PHASE                       │          │
│         │  pricing-linker │ regional-availability │ features     │          │
│         │  token-specs    │ mantle-collector     │ lifecycle    │          │
│         └────────────────────────────┬───────────────────────────┘          │
│                                      │                                      │
│                                      ▼                                      │
│                          ┌────────────────────┐                             │
│                          │  final-aggregator  │                             │
│                          └─────────┬──────────┘                             │
│                                    │                                        │
│                                    ▼                                        │
│                          ┌────────────────────┐                             │
│                          │   gap-detection    │────┐                        │
│                          └─────────┬──────────┘    │ (if gaps)              │
│                                    │               ▼                        │
│                                    │    ┌────────────────────┐              │
│                                    │    │ self-healing-agent │              │
│                                    │    │   (Claude Opus)    │              │
│                                    │    └─────────┬──────────┘              │
│                                    │              │                         │
│                                    ▼◀─────────────┘                         │
│                          ┌────────────────────┐                             │
│                          │   copy-to-latest   │                             │
│                          └─────────┬──────────┘                             │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────┐           ┌──────────────┐           ┌──────────────┐     │
│  │   S3 Bucket  │◀──────────│   CloudFront │──────────▶│    React     │     │
│  │   (Data)     │           │    (CDN)     │           │   Frontend   │     │
│  └──────────────┘           └──────────────┘           └──────────────┘     │
│                                                                             │
│  ┌──────────────┐                                                           │
│  │   Cognito    │───────────────────────────────────────────────────────────┤
│  │   (Auth)     │    User Groups: beta-access-users, operators, admins      │
│  └──────────────┘                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Inventory

| Component Type | Count | Purpose |
|----------------|-------|---------|
| Lambda Functions | 17 | Data collection, processing, intelligence |
| Step Functions | 1 | Workflow orchestration |
| S3 Buckets | 1 | Data storage + frontend hosting |
| CloudFront Distributions | 1 | CDN delivery |
| EventBridge Rules | 1 | Daily trigger |
| Lambda Layers | 1 | Shared utilities |
| Cognito User Pool | 1 | Authentication |

---

## Data Flow

### Complete Pipeline Flow

```
PHASE 0: INITIALIZATION
  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │ DiscoverRegions  │ ──► │ InitializeExec   │ ──► │   ConfigSync     │
  │ (dynamic regions)│     │ (set up paths)   │     │ (frontend config)│
  └──────────────────┘     └──────────────────┘     └──────────────────┘

PHASE 1: PARALLEL COLLECTION (Wave 1)
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                                                                             │
  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
  │  │ pricing-collector│   │ model-extractor │    │ quota-collector │         │
  │  │   (×3 parallel)  │   │  (×27 regions)  │    │  (×N regions)   │         │
  │  └────────┬─────────┘   └────────┬────────┘    └────────┬────────┘         │
  │           │                      │                      │                   │
  │           ▼                      ▼                      │                   │
  │  ┌─────────────────┐    ┌─────────────────┐             │                   │
  │  │pricing-aggregator│   │  model-merger   │             │                   │
  │  │ + cache write    │   │ + cache keys    │             │                   │
  │  └────────┬─────────┘   └────────┬────────┘             │                   │
  │           │                      │                      │                   │
  └───────────┼──────────────────────┼──────────────────────┼───────────────────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     ▼

PHASE 2: ENRICHMENT (Wave 2)
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  ┌─────────────┐  ┌───────────────┐  ┌─────────────┐  ┌─────────────────┐  │
  │  │pricing-linker│  │regional-avail │  │feature-coll │  │token-specs-coll │  │
  │  │             │  │(uses cache)   │  │ (×N regions)│  │(LiteLLM fetch)  │  │
  │  └─────────────┘  └───────────────┘  └─────────────┘  └─────────────────┘  │
  │                                                                             │
  │  ┌─────────────────┐  ┌─────────────────┐                                  │
  │  │ mantle-collector│  │lifecycle-collect│                                  │
  │  │  (×N regions)   │  │  (web scrape)   │                                  │
  │  └─────────────────┘  └─────────────────┘                                  │
  └─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼

PHASE 3: AGGREGATION & INTELLIGENCE
  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │ final-aggregator │ ──► │  gap-detection   │ ──► │self-healing-agent│
  │ (merge all data) │     │ (analyze gaps)   │     │  (if triggered)  │
  └──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                             │
                                     ┌───────────────────────┘
                                     ▼
PHASE 4: PUBLICATION
  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │  copy-to-latest  │ ──► │    CloudFront    │ ──► │  React Frontend  │
  │  (S3 copy)       │     │   (cache/serve)  │     │   (display)      │
  └──────────────────┘     └──────────────────┘     └──────────────────┘
```

### Data Transformations

| Stage | Input | Transformation | Output |
|-------|-------|----------------|--------|
| `pricing-collector` | Pricing API pages | Extract & normalize | Raw pricing JSON |
| `pricing-aggregator` | 3 pricing files | Dedupe, categorize | Merged pricing |
| `model-extractor` | Bedrock API | Add console metadata | Enriched models |
| `model-merger` | 27 region files | Dedupe, merge metadata | Unified model list |
| `pricing-linker` | Models + Pricing | Fuzzy match (>0.7) | Models with pricing refs |
| `regional-availability` | 27 regions API | Filter ON_DEMAND/PROVISIONED | Availability map |
| `final-aggregator` | All enrichments | Merge into final schema | Production-ready JSON |

### S3 File Structure

```
bucket/
├── latest/                              # Production files (served via CloudFront)
│   ├── bedrock_models.json             # Complete model catalog (~2-3 MB)
│   └── bedrock_pricing.json            # Pricing by provider/model (~1-2 MB)
│
├── config/                              # Configuration files
│   ├── profiler-config.json            # Backend configuration
│   ├── frontend-config.json            # Generated frontend config
│   └── config-history/                 # Config backups from auto-updates
│       └── profiler-config.{timestamp}.json
│
├── executions/{execution-id}/           # Per-execution data (retained for debugging)
│   ├── pricing/                         # Raw pricing data
│   │   ├── AmazonBedrock.json
│   │   ├── AmazonBedrockService.json
│   │   └── AmazonBedrockFoundationModels.json
│   ├── models/                          # Per-region model data
│   │   ├── us-east-1.json
│   │   └── us-west-2.json
│   ├── cache/                           # Cached API responses for reuse
│   │   ├── list_foundation_models_us-east-1.json
│   │   └── inference_profiles_us-east-1.json
│   ├── quotas/{region}.json             # Service quotas
│   ├── features/{region}.json           # Inference profiles
│   ├── mantle/{region}.json             # Mantle models
│   ├── lifecycle/lifecycle.json         # Model lifecycle data
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
│
└── agent/                               # Self-healing agent data
    ├── gap-reports/{execution-id}/
    │   └── gap-analysis.json
    └── suggestions/{execution-id}/
        └── suggestions.json
```

---

## AWS Services Used

### Service Summary

| Service | Purpose | Key APIs Used |
|---------|---------|---------------|
| **Cognito** | User authentication (OIDC) | User groups: beta-access-users, operators, admins |
| **Bedrock** | Model data | ListFoundationModels, ListInferenceProfiles, InvokeModel |
| **Bedrock Console REST API** | Extended metadata | SigV4-signed requests with x-console-consumer header |
| **Pricing API** | Pricing data | GetProducts (3 service codes) |
| **Service Quotas** | Regional quotas | ListServiceQuotas (27+ regions) |
| **Step Functions** | Orchestration | 4-phase workflow |
| **S3** | Storage | Data, frontend hosting, caching, config |
| **CloudFront** | CDN | OAC distribution |
| **EventBridge** | Scheduling | Daily trigger at 6 AM UTC |
| **Lambda Powertools** | Observability | Structured logging, tracing, metrics |

### API Rate Limits & Mitigations

| API | Rate Limit | Mitigation Strategy |
|-----|------------|---------------------|
| Pricing API | 10 TPS | Adaptive retry, exponential backoff |
| Bedrock ListFoundationModels | 10 TPS | Caching between Lambdas |
| Service Quotas | 5 TPS | Max concurrency = 10, adaptive retry |
| Bedrock InvokeModel | Model-dependent | Only called when gaps detected |

---

## Project Structure

```
bedrock-model-profiler/
├── frontend/                    # React 18 + Vite + Tailwind CSS v4 + Radix UI
│   ├── src/
│   │   ├── components/         # UI components
│   │   │   ├── ui/            # Radix UI primitives
│   │   │   ├── models/        # Model display components
│   │   │   └── comparison/    # Comparison feature
│   │   ├── stores/            # Zustand state management
│   │   ├── hooks/             # Custom React hooks
│   │   ├── auth/              # Cognito OIDC integration
│   │   └── config/            # App configuration
│   ├── public/
│   └── scripts/               # Deployment scripts
│
├── backend/
│   ├── lambdas/               # 17 Python Lambda functions
│   │   ├── region-discovery/
│   │   ├── config-sync/
│   │   ├── pricing-collector/
│   │   ├── pricing-aggregator/
│   │   ├── model-extractor/
│   │   ├── model-merger/
│   │   ├── quota-collector/
│   │   ├── pricing-linker/
│   │   ├── regional-availability/
│   │   ├── feature-collector/
│   │   ├── token-specs-collector/
│   │   ├── mantle-collector/
│   │   ├── lifecycle-collector/
│   │   ├── final-aggregator/
│   │   ├── gap-detection/
│   │   ├── self-healing-agent/
│   │   └── copy-to-latest/
│   ├── layers/
│   │   └── common/            # Shared utilities
│   │       └── python/shared/
│   │           ├── s3_utils.py
│   │           ├── config_loader.py
│   │           ├── model_matcher.py
│   │           ├── cache_utils.py
│   │           └── validation.py
│   ├── statemachine/          # Step Functions ASL definition
│   ├── config/                # profiler-config.json
│   └── tests/                 # ~150 pytest tests
│
├── infra/                     # SAM templates (CloudFormation)
│   ├── backend-template.yaml
│   ├── frontend-template.yaml
│   └── analytics-template.yaml
│
└── docs/                      # Documentation
    ├── ARCHITECTURE.md        # This file
    ├── DATA-SOURCES.md        # API documentation
    ├── DATA-SCHEMA.md         # Model JSON schema
    └── PRICING-SCHEMA.md      # Pricing JSON schema
```

### Technology Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18, Vite, Tailwind CSS v4, Radix UI, Zustand |
| **Backend** | Python 3.12, AWS Lambda, Step Functions |
| **Infrastructure** | AWS SAM, CloudFormation |
| **Data Storage** | S3 (JSON files) |
| **CDN** | CloudFront |
| **Auth** | Cognito OIDC |
| **AI/ML** | Bedrock Claude Opus 4.5 (self-healing) |
| **Observability** | Lambda Powertools, CloudWatch |

---

## Lambda Functions

### Function Overview

| Function | Purpose | Timeout | Memory | Concurrency |
|----------|---------|---------|--------|-------------|
| `region-discovery` | Discover active Bedrock regions | 30s | 256MB | 1 |
| `config-sync` | Sync frontend config | 30s | 256MB | 1 |
| `pricing-collector` | Collect pricing per service code | 5min | 512MB | 3 |
| `pricing-aggregator` | Merge all pricing data | 2min | 1GB | 1 |
| `model-extractor` | Extract models per region | 1min | 256MB | 10 |
| `model-merger` | Merge models from all regions | 1min | 512MB | 1 |
| `quota-collector` | Collect quotas per region | 1min | 256MB | 10 |
| `pricing-linker` | Link pricing to models | 2min | 1GB | 1 |
| `regional-availability` | Compute availability map | 5min | 512MB | 1 |
| `feature-collector` | Collect inference profiles | 1min | 256MB | 10 |
| `token-specs-collector` | Fetch LiteLLM specs | 2min | 512MB | 1 |
| `mantle-collector` | Collect Mantle models | 2min | 256MB | 10 |
| `lifecycle-collector` | Scrape lifecycle data | 1.5min | 256MB | 1 |
| `final-aggregator` | Merge all data | 3min | 2GB | 1 |
| `gap-detection` | Analyze data gaps | 2min | 512MB | 1 |
| `self-healing-agent` | AI config updates | 5min | 512MB | 1 |
| `copy-to-latest` | Copy to production | 1min | 256MB | 1 |

### Lambda Data Flow Matrix

| Lambda | Produces | Consumes |
|--------|----------|----------|
| `region-discovery` | Active regions list, inference profile cache | - |
| `config-sync` | frontend-config.json | profiler-config.json |
| `pricing-collector` | Raw pricing per service code | - |
| `pricing-aggregator` | Merged pricing JSON | Raw pricing files |
| `model-extractor` | Models per region, API cache | - |
| `model-merger` | Merged models JSON, cache keys | Model files per region |
| `quota-collector` | Quotas per region | - |
| `pricing-linker` | Models with pricing refs | Merged pricing, merged models |
| `regional-availability` | Availability map | Model cache (from extractor) |
| `feature-collector` | Inference profiles | Profile cache (from discovery) |
| `token-specs-collector` | Token specs | LiteLLM external data |
| `mantle-collector` | Mantle model list | - |
| `lifecycle-collector` | Lifecycle data | AWS docs (web scrape) |
| `final-aggregator` | Final JSONs | All intermediate files |
| `gap-detection` | Gap report | Final JSONs, previous latest |
| `self-healing-agent` | Config suggestions | Gap report, config |
| `copy-to-latest` | latest/ files | Final JSONs |

---

## Caching System

The pipeline implements a caching layer to reduce redundant API calls between Lambdas.

### Cache Types

| Cache Type | Producer | Consumer | TTL | Location |
|------------|----------|----------|-----|----------|
| **Model Data** | `model-extractor` | `regional-availability` | Per-execution | `cache/list_foundation_models_{region}.json` |
| **Inference Profiles** | `region-discovery` | `feature-collector` | Per-execution | `cache/inference_profiles_{region}.json` |
| **LiteLLM Data** | `token-specs-collector` | Self | 24 hours | `cache/litellm_model_prices.json` |
| **Lifecycle Data** | `lifecycle-collector` | Self | 24 hours | `cache/lifecycle_data.json` |

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

### Cache Performance Impact

| Metric | Without Cache | With Cache |
|--------|---------------|------------|
| Model API calls | ~54 (27 regions × 2 filters) | ~27 + cache reads |
| Inference profile calls | 27 | 0 (100% cache hits) |
| Total API calls | ~480 | ~29 |
| **Reduction** | - | **~94%** |

---

## Self-Healing System

### Architecture

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│final-aggregator│────▶│  gap-detection │────▶│  Gaps Found?   │
└────────────────┘     └────────────────┘     └───────┬────────┘
                                                      │
                              ┌────────────────────────┴───────────────┐
                              │                                        │
                              ▼                                        ▼
                      ┌────────────────┐                    ┌────────────────┐
                      │ self-healing   │                    │  copy-to-latest│
                      │    agent       │───────────────────▶│                │
                      │ (Claude Opus)  │                    └────────────────┘
                      └───────┬────────┘
                              │
                              ▼
                      ┌────────────────┐
                      │profiler-config │
                      │   (updated)    │
                      └────────────────┘
```

### Gap Detection Types

| Gap Type | Detection Logic | Trigger Threshold |
|----------|-----------------|-------------------|
| **Models without pricing** | `has_pricing == false` | ≥5 models |
| **Low confidence matches** | `confidence < 0.6` | ≥3 matches |
| **Unknown providers** | Not in `provider_patterns` | Any detected |
| **New models** | Delta from `latest/bedrock_models.json` | Any detected |
| **Context window mismatch** | Config vs API variance > 10% | Any detected |
| **Unknown service codes** | Not in `pricing_service_codes` | Any detected |
| **Frontend config drift** | Backend regions ≠ frontend regions | Any drift |

### Auto-Update Rules

| Change Type | Auto-Apply Safe | Example |
|-------------|-----------------|---------|
| `provider_pattern_addition` | Yes | Add `"nemotron"` to NVIDIA patterns |
| `context_window_update` | Yes | Update Claude context to 200K |
| `service_code_addition` | Yes | Add new pricing service code |
| `region_addition` | Yes | Add new region coordinates |
| `documentation_link_addition` | Yes | Add provider doc links |
| `provider_pattern_removal` | **No** | Requires review |
| `threshold_change` | **No** | Requires review |

---

## Configuration

### profiler-config.json Overview

Located at: `backend/config/profiler-config.json`

| Section | Purpose |
|---------|---------|
| `external_urls` | API endpoints, documentation links |
| `provider_configuration` | Provider aliases, patterns, colors, docs |
| `region_configuration` | Region lists, coordinates, metadata |
| `model_configuration` | Model families, variants, context specs, hidden models |
| `matching_configuration` | Fuzzy matching thresholds, explicit mappings |
| `agent_configuration` | Self-healing agent settings |
| `pricing_service_codes` | AWS pricing API service codes |
| `gap_detection_config` | Gap detection thresholds |

### Key Configuration Example

```json
{
  "provider_configuration": {
    "provider_patterns": {
      "Anthropic": ["claude", "sonnet", "haiku", "opus"],
      "Amazon": ["titan", "nova", "rerank"],
      "Meta": ["llama", "mllama"]
    }
  },
  "matching_configuration": {
    "explicit_model_mappings": {
      "anthropic.claude-opus-4-5-20251101-v1:0": "anthropic.claude-opus-4-5"
    },
    "min_confidence_threshold": 0.7
  },
  "model_configuration": {
    "hidden_models": ["zai.glm-5"]
  },
  "agent_configuration": {
    "bedrock_model_id": "us.anthropic.claude-opus-4-5-20251101-v1:0"
  }
}
```

### Hiding Models from the Frontend

To hide a model from the frontend without removing it from the data pipeline, add its `model_id` to the `hidden_models` array in `profiler-config.json`:

```json
"model_configuration": {
  "hidden_models": ["zai.glm-5", "another.model-id"]
}
```

The `final-aggregator` Lambda sets `show_model: false` on these models. The frontend's `flattenModels()` in `useModels.js` filters them out. The model data is still collected and stored — it just won't appear in any frontend view.

---

## Deployment

### Stack Names

| Environment | Stack Name | S3 Bucket Pattern |
|-------------|------------|-------------------|
| Development | `bedrock-profiler-dev` | `bedrock-profiler-dev-*` |
| Production | `bedrock-profiler-prod` | `bedrock-profiler-prod-*` |

### Deployment Commands

**Full Backend Deployment:**
```bash
cd infra
sam build -t backend-template.yaml
sam deploy --stack-name bedrock-profiler-dev \
  --capabilities CAPABILITY_NAMED_IAM \
  --resolve-s3
```

**Frontend Deployment:**
```bash
cd frontend
npm run build
./scripts/deploy.sh  # Syncs to S3 + CloudFront invalidation
```

**Infrastructure Setup (First Time):**
```bash
./setup-infrastructure.sh
```

### Execution Timing

| Phase | Duration | Notes |
|-------|----------|-------|
| Phase 0: Initialization | ~10s | Region discovery + config sync |
| Phase 1: Wave 1 Collection | 3-5 min | Parallel (pricing slowest) |
| Phase 2: Wave 2 Enrichment | 2-3 min | Parallel (regional-avail slowest) |
| Phase 3: Aggregation | 1-2 min | Serial |
| Phase 4: Publication | ~10s | S3 copy |
| **Total** | **8-12 min** | |

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [DATA-SOURCES.md](./DATA-SOURCES.md) | Data sources, APIs, and reliability |
| [DATA-SCHEMA.md](./DATA-SCHEMA.md) | Model JSON schema reference |
| [PRICING-SCHEMA.md](./PRICING-SCHEMA.md) | Pricing JSON schema reference |
| [backend/lambdas/README.md](../backend/lambdas/README.md) | Lambda contracts and interfaces |
| [CLAUDE.md](../CLAUDE.md) | Development guide for Claude Code |

---

## Changelog

| Date | Version | Change |
|------|---------|--------|
| 2026-03-06 | 2.1 | Reorganized documentation, added schema references |
| 2026-03-05 | 2.0 | Major update with comprehensive restructure |
| 2026-03-03 | 1.1 | Added caching system documentation |
