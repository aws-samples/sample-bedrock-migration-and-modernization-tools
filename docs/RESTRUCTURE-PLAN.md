# JSON Restructure Plan - Comprehensive Refactoring

**Created**: 2026-03-03
**Status**: PLANNING - Awaiting Approval
**Estimated Effort**: 10-14 days

---

## Executive Summary

This document outlines a major restructuring of the Bedrock Model Profiler's JSON output to:
1. **Consolidate scattered region data** into a single `availability` object
2. **Normalize pricing to per-million** (remove confusing per-thousand storage)
3. **Simplify field names** (remove `model_` prefixes, shorten names)
4. **Remove redundant fields** (duplicate booleans, derivable values)
5. **Make structure human-readable** and easier to understand

---

## Current Problems

### 1. Region Data Scattered Across 8+ Fields
```
Current:
├── in_region: ["us-east-1", ...]
├── regions_available: ["us-east-1", ...]  ← DUPLICATE!
├── cross_region_inference.source_regions: [...]
├── batch_inference_supported.supported_regions: [...]
├── provisioned_throughput.provisioned_regions: [...]
├── mantle_inference.mantle_regions: [...]
├── endpoint_availability.bedrock_runtime.regions: [...]
└── endpoint_availability.bedrock_mantle.regions: [...]
```

### 2. Pricing Confusion
- Backend stores: **per-thousand** (`price_per_thousand: 0.003`)
- Frontend converts to: **per-million** (`* 1000`)
- Display shows: **per-million** (`$3.00 / 1M tokens`)

### 3. Verbose Field Names
- `model_modalities` → should be `modalities`
- `model_capabilities` → should be `capabilities`
- `model_service_quotas` → should be `quotas`
- `streaming_supported` → should be `streaming`

### 4. Redundant Boolean Flags
- `has_pricing` duplicates `model_pricing.is_pricing_available`
- `has_quotas` can be computed from `quotas` object
- `is_mantle` duplicates `mantle_inference.supported`

---

## Proposed New Structure

### Before vs After Comparison

```
BEFORE (35 fields, scattered)          AFTER (25 fields, organized)
─────────────────────────────          ────────────────────────────
model_id                         →     model_id
model_arn                        →     model_arn  
model_name                       →     model_name
model_provider                   →     model_provider

model_modalities                 →     modalities
  input_modalities               →       input
  output_modalities              →       output

streaming_supported              →     streaming
model_capabilities               →     capabilities
model_use_cases                  →     use_cases
languages_supported              →     languages

customization                    →     customization (unchanged)
inference_types_supported        →     inference_types

model_lifecycle                  →     lifecycle (unchanged)

in_region                        ─┐
regions_available                ─┤    availability
cross_region_inference           ─┤      on_demand
batch_inference_supported        ─┼→       supported: boolean
provisioned_throughput           ─┤        regions: []
mantle_inference                 ─┤      cross_region
is_mantle                        ─┤        supported: boolean
mantle_only                      ─┘        regions: []
                                          profiles: []
                                        batch
                                          supported: boolean
                                          regions: []
                                        provisioned
                                          supported: boolean
                                          regions: []
                                        mantle
                                          supported: boolean
                                          regions: []
                                          only: boolean
                                          responses_api: boolean

consumption_options              →     (derived from availability.*.supported)

converse_data                    →     specs
  context_window                 →       context_window
  max_output_tokens              →       max_output
  extended_context               →       extended_context
  has_extended_context           →       (derived)
  ...                            →       (simplified)

model_pricing                    →     pricing
  is_pricing_available           →       available: boolean
  pricing_file_reference         →       reference: {...}
has_pricing                      ─┘     (REMOVED - use pricing.available)

model_service_quotas             →     quotas
has_quotas                       ─┘     (REMOVED - derive from quotas)

api_support                      →     api (simplified)
endpoint_availability            ─┘     (REMOVED - redundant)

feature_support                  →     features
chat_features                    →     chat_features

documentation_links              →     docs
description                      →     description
short_description                →     short_description

collection_metadata              →     _metadata (internal)
regional_availability_source     ─┘     (REMOVED - move to _metadata)
date_added                       →     date_added
```

### New `availability` Object (Consolidated)

```json
{
  "availability": {
    "on_demand": {
      "supported": true,
      "regions": ["us-east-1", "us-west-2", "eu-west-1"]
    },
    "cross_region": {
      "supported": true,
      "regions": ["us-east-1", "eu-west-1"],
      "profiles": [
        {
          "profile_id": "us.anthropic.claude-3-sonnet",
          "profile_name": "US Claude 3 Sonnet",
          "type": "SYSTEM_DEFINED",
          "status": "ACTIVE"
        }
      ]
    },
    "batch": {
      "supported": true,
      "regions": ["us-east-1", "us-west-2"]
    },
    "provisioned": {
      "supported": false,
      "regions": []
    },
    "mantle": {
      "supported": false,
      "regions": [],
      "only": false,
      "responses_api": false
    }
  }
}
```

### New `pricing` Object (Per-Million)

```json
{
  "pricing": {
    "available": true,
    "reference": {
      "provider": "Anthropic",
      "model_key": "anthropic.claude-3-sonnet",
      "model_name": "Claude 3 Sonnet"
    }
  }
}
```

### New `specs` Object (Simplified)

```json
{
  "specs": {
    "context_window": 200000,
    "max_output": 4096,
    "extended_context": 1000000,
    "size_category": "large",
    "source": "litellm",
    "verified": true
  }
}
```

---

## Pricing Normalization Change

### Current Flow (Confusing)
```
AWS API → per-million (some)
    ↓
pricing-aggregator → CONVERTS TO per-thousand
    ↓
bedrock_pricing.json → stores per-thousand
    ↓
useModels.js → CONVERTS TO per-million (* 1000)
    ↓
UI → displays per-million
```

### New Flow (Clear)
```
AWS API → per-million (some)
    ↓
pricing-aggregator → NORMALIZES TO per-million
    ↓
bedrock_pricing.json → stores per-million
    ↓
useModels.js → NO CONVERSION NEEDED
    ↓
UI → displays per-million
```

### Code Change Required

**File**: `backend/lambdas/pricing-aggregator/handler.py`

```python
# BEFORE (line ~380):
if price and is_per_million:
    price = price / 1000  # Convert to per-thousand

# AFTER:
if price and not is_per_million:
    price = price * 1000  # Convert to per-million
```

**File**: `frontend/src/hooks/useModels.js`

```javascript
// BEFORE (line ~430):
if (inputPrice !== null) inputPrice = inputPrice * 1000
if (outputPrice !== null) outputPrice = outputPrice * 1000

// AFTER:
// REMOVE these lines - prices already in per-million
```

---

## Migration Strategy

### Phase 1: Backend Dual-Output (No Breaking Changes)

1. Update `final-aggregator` to output BOTH old and new field names
2. Update `pricing-aggregator` to output BOTH `price_per_thousand` and `price_per_million`
3. Deploy backend
4. Verify frontend still works with old fields

### Phase 2: Frontend Migration

1. Update `useModels.js` to use new field names
2. Update all components to use new paths
3. Remove pricing conversion code
4. Test thoroughly

### Phase 3: Backend Cleanup

1. Remove old field names from `final-aggregator`
2. Remove `price_per_thousand` from pricing
3. Deploy final version

---

## Files to Modify

### Backend (5 files)

| File | Changes | Risk |
|------|---------|------|
| `final-aggregator/handler.py` | Restructure output, add `availability` | **HIGH** |
| `pricing-aggregator/handler.py` | Change to per-million | **MEDIUM** |
| `model-merger/handler.py` | Update field names | **LOW** |
| `regional-availability/handler.py` | Update output structure | **MEDIUM** |
| `pricing-linker/handler.py` | Update pricing reference | **LOW** |

### Frontend (10+ files)

| File | Changes | Risk |
|------|---------|------|
| `useModels.js` | Update field mappings, remove conversion | **HIGH** |
| `ModelCard.jsx` | Update field paths | **HIGH** |
| `ModelCardExpanded.jsx` | Update 358+ field accesses | **HIGH** |
| `ModelExplorer.jsx` | Update filter logic | **MEDIUM** |
| `ModelComparison.jsx` | Update comparison logic | **MEDIUM** |
| `RegionalAvailability.jsx` | Update region access | **MEDIUM** |
| `OverviewTab.jsx` | Update spec access | **MEDIUM** |
| `PricingTab.jsx` | Update pricing access | **MEDIUM** |
| `TechSpecsTab.jsx` | Update spec access | **MEDIUM** |
| `filters.js` | Update filter functions | **MEDIUM** |
| `constants.js` | Update field references | **LOW** |

---

## Field Migration Map

```javascript
const MIGRATION_MAP = {
  // AVAILABILITY (consolidated)
  'in_region': 'availability.on_demand.regions',
  'regions_available': null, // REMOVED
  'total_regions_available': null, // REMOVED (derive from availability)
  'cross_region_inference.supported': 'availability.cross_region.supported',
  'cross_region_inference.source_regions': 'availability.cross_region.regions',
  'cross_region_inference.profiles': 'availability.cross_region.profiles',
  'batch_inference_supported.supported': 'availability.batch.supported',
  'batch_inference_supported.supported_regions': 'availability.batch.regions',
  'provisioned_throughput.supported': 'availability.provisioned.supported',
  'provisioned_throughput.provisioned_regions': 'availability.provisioned.regions',
  'mantle_inference.supported': 'availability.mantle.supported',
  'mantle_inference.mantle_regions': 'availability.mantle.regions',
  'mantle_inference.supports_responses_api': 'availability.mantle.responses_api',
  'is_mantle': null, // REMOVED (use availability.mantle.supported)
  'mantle_only': 'availability.mantle.only',
  
  // SPECS (renamed)
  'converse_data.context_window': 'specs.context_window',
  'converse_data.max_output_tokens': 'specs.max_output',
  'converse_data.extended_context': 'specs.extended_context',
  'converse_data.has_extended_context': null, // REMOVED (derive)
  'converse_data.size_category': 'specs.size_category',
  'converse_data.source': 'specs.source',
  'converse_data.verified': 'specs.verified',
  
  // PRICING (renamed)
  'model_pricing.is_pricing_available': 'pricing.available',
  'model_pricing.pricing_file_reference': 'pricing.reference',
  'has_pricing': null, // REMOVED
  
  // QUOTAS (renamed)
  'model_service_quotas': 'quotas',
  'has_quotas': null, // REMOVED
  
  // MODALITIES (renamed)
  'model_modalities.input_modalities': 'modalities.input',
  'model_modalities.output_modalities': 'modalities.output',
  
  // SIMPLE RENAMES
  'model_capabilities': 'capabilities',
  'model_use_cases': 'use_cases',
  'model_lifecycle': 'lifecycle',
  'streaming_supported': 'streaming',
  'languages_supported': 'languages',
  'documentation_links': 'docs',
  'api_support': 'api',
  'feature_support': 'features',
  'collection_metadata': '_metadata',
  
  // REMOVED
  'endpoint_availability': null,
  'regional_availability_source': null,
};
```

---

## Testing Checklist

### Backend Tests
- [ ] `final-aggregator` outputs correct structure
- [ ] Pricing is in per-million format
- [ ] All regions consolidated in `availability`
- [ ] Old fields still present (Phase 1)
- [ ] Workflow completes successfully

### Frontend Tests
- [ ] Model Explorer loads and displays models
- [ ] Model cards show correct data
- [ ] Model details show all information
- [ ] Regional availability page works
- [ ] Comparison page works
- [ ] Filters work correctly
- [ ] Pricing displays correctly (per-million)
- [ ] CRIS profiles display correctly

### Integration Tests
- [ ] Full workflow → frontend displays correctly
- [ ] CloudFront cache invalidation
- [ ] No console errors

---

## Rollback Plan

If issues are found:

1. **Phase 1 rollback**: Frontend uses old fields, no changes needed
2. **Phase 2 rollback**: Revert frontend to use old field names
3. **Phase 3 rollback**: Re-add old fields to backend

---

## Approval Required

Before proceeding, please confirm:

1. ✅ Agree with the new `availability` structure?
2. ✅ Agree with pricing normalization to per-million?
3. ✅ Agree with field name simplifications?
4. ✅ Agree with the phased migration approach?
5. ✅ Ready to proceed with Phase 1?

---

## Next Steps After Approval

1. Create a new git branch: `feature/json-restructure`
2. Implement Phase 1 (backend dual-output)
3. Deploy and verify
4. Implement Phase 2 (frontend migration)
5. Deploy and verify
6. Implement Phase 3 (cleanup)
7. Final deployment
