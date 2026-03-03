# JSON Data Optimization - Future Work

**Created**: 2026-03-03
**Status**: Pending
**Priority**: Medium-High

## Overview

Analysis of the generated JSON files revealed significant optimization opportunities that could reduce total file size by ~41% (from 26.64 MB to ~15.8 MB).

## Current State

| File | Size |
|------|------|
| `bedrock_models.json` | 6.15 MB |
| `bedrock_pricing.json` | 20.49 MB |
| **Total** | **26.64 MB** |

---

## Critical Issues to Address

### 🔴 PRIORITY 1: Externalize Quotas (High Impact, High Effort)

**Problem**: Quota data accounts for 53% of models.json (~3.26 MB)
- 8,974 quota entries stored redundantly
- Same quota repeated for every region a model is available in
- Each model with 31 regions × 15 quotas = 465 duplicate entries

**Solution**: Create separate `bedrock_quotas.json`:
```json
{
  "quota_definitions": {
    "L-186C5310": {
      "name": "Model invocation max tokens per day for {model_name}",
      "unit": "None",
      "adjustable": false
    }
  },
  "model_quotas": {
    "anthropic.claude-sonnet-4-6:0": {
      "quota_codes": ["L-186C5310", "L-1750D6FF"],
      "regional_values": {
        "us-east-1": {"L-186C5310": 144000000000.0}
      }
    }
  }
}
```

**Estimated Savings**: ~3 MB (50% of models.json)

**Files to Modify**:
- `backend/lambdas/final-aggregator/handler.py` - Change quota output structure
- `frontend/src/hooks/useModels.js` - Load quotas separately
- `frontend/src/components/models/ModelCardExpanded.jsx` - Update quota rendering

---

### 🔴 PRIORITY 2: Flatten Pricing Dimensions (High Impact, Medium Effort)

**Problem**: 55% of pricing dimension fields are redundant
- 14,726 pricing dimensions with 20 fields each
- 11 fields are redundant or derivable from parent context

**Redundant Fields to Remove**:
| Field | Reason |
|-------|--------|
| `price_per_thousand` | Same as `price_per_unit` |
| `original_price` | Same as `price_per_unit` |
| `model_provider` | Same as `provider` |
| `source_dataset` | Always "aws_pricing_api" |
| `service_code` | Always "AmazonBedrock" |
| `model_id` | Derivable from parent |
| `model_name` | Derivable from parent |
| `provider` | Derivable from parent |
| `location` | Derivable from region key |
| `pricing_group` | Derivable from parent key |
| `pricing_type` | Derivable from parent |
| `description` | Can be generated client-side |
| `unit_label` | Derivable from `unit` |

**Keep Only**:
```json
{
  "dimension": "USE1-Claude2.0-input-tokens",
  "price": 0.008,
  "unit": "1K tokens",
  "is_input": true,
  "is_output": false,
  "characteristics": {"type": "on_demand", "scope": "regional"}
}
```

**Estimated Savings**: ~7.4 MB (66% reduction in pricing.json)

**Files to Modify**:
- `backend/lambdas/pricing-aggregator/handler.py` - Flatten output structure
- `frontend/src/components/models/ModelCardExpanded.jsx` - Update pricing rendering
- `frontend/src/components/comparison/tabs/PricingTab.jsx` - Update comparison logic

---

### 🟡 PRIORITY 3: Consolidate Region Fields (Low Impact, Low Effort)

**Problem**: Region availability scattered across multiple fields
- `in_region`
- `cross_region_inference.source_regions`
- `batch_inference_supported.supported_regions`
- `mantle_inference.mantle_regions`
- `endpoint_availability`

**Solution**: Single `availability` object:
```json
{
  "availability": {
    "on_demand": ["us-east-1", "us-west-2"],
    "batch": ["us-east-1"],
    "cross_region": ["us-east-1", "eu-west-1"],
    "mantle": [],
    "provisioned": ["us-west-2"]
  }
}
```

**Estimated Savings**: ~0.3 MB

---

### 🟡 PRIORITY 4: Remove Duplicate Booleans (Low Impact, Low Effort)

**Problem**: Duplicate boolean flags
- `has_pricing` duplicates `model_pricing.is_pricing_available`
- `is_mantle` duplicates `mantle_inference.supported`

**Solution**: Remove redundant top-level booleans, use nested values

**Estimated Savings**: ~0.2 MB

---

## Implementation Plan

### Phase 1: Quick Wins (Priority 3 & 4)
- Estimated time: 2-4 hours
- Low risk, immediate benefits
- ~0.5 MB savings

### Phase 2: Pricing Optimization (Priority 2)
- Estimated time: 4-8 hours
- Medium risk, requires frontend updates
- ~7.4 MB savings

### Phase 3: Quota Externalization (Priority 1)
- Estimated time: 8-16 hours
- Higher risk, requires new file and loading logic
- ~3 MB savings

---

## What's Working Well (Don't Change)

✅ Consistent `snake_case` naming throughout final output
✅ Clear provider → model hierarchy
✅ Good separation between models.json and pricing.json
✅ Comprehensive metadata in both files
✅ `pricing_file_reference` links models to pricing correctly
✅ CRIS profiles properly populated (609 total profiles)
✅ Consumption options correctly reconciled

---

## Estimated Total Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total Size | 26.64 MB | ~15.8 MB | 41% reduction |
| Load Time | ~2-3s | ~1-1.5s | 50% faster |
| Memory Usage | High | Medium | Significant |

---

## Notes

- Frontend must handle missing fields gracefully after optimization
- Consider gzip compression at CloudFront level (already enabled?)
- May want to implement lazy loading for quotas/pricing details
- Consider pagination for very large datasets
