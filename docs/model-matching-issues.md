# Model Matching & Duplication Issues

> **Date**: 2026-03-04
> **Status**: Analysis Complete, Claude 4.x Resolution Implemented
> **Priority**: Medium-High

## Executive Summary

The Bedrock Model Profiler pipeline has several fuzzy matching and model ID normalization issues that cause:
1. **Duplicate models** appearing in the frontend (same model, different IDs)
2. **Incorrect pricing matches** (V3.2 matched to R1 pricing)
3. **Mantle-only stubs** created for models that exist in Bedrock (matching failure)
4. **Availability inheritance** giving on-demand regions to provisioned-only variants

---

## Table of Contents

- [Issue 1: Mantle vs Bedrock Model ID Format Mismatch](#issue-1-mantle-vs-bedrock-model-id-format-mismatch)
- [Issue 2: Pricing API Model ID Format Mismatch](#issue-2-pricing-api-model-id-format-mismatch)
- [Issue 3: Bedrock API Returns Multiple Variants](#issue-3-bedrock-api-returns-multiple-variants)
- [Issue 4: Availability Inheritance via Fuzzy Matching](#issue-4-availability-inheritance-via-fuzzy-matching)
- [Issue 5: Claude 4.x Matching to Claude 3.x (RESOLVED)](#issue-5-claude-4x-matching-to-claude-3x-resolved)
- [Fuzzy Matching Locations](#fuzzy-matching-locations)
- [Recommended Actions](#recommended-actions)
- [Adding New Explicit Mappings](#adding-new-explicit-mappings)
- [Files to Modify](#files-to-modify)

---

## Issue 1: Mantle vs Bedrock Model ID Format Mismatch

### Symptom
`deepseek.v3-v1:0` and `deepseek.v3.1` appear as **separate models** in the UI, when they represent the same DeepSeek V3.1 model.

### Root Cause
The Bedrock API and Mantle API use different ID formats for the same model:

| Source | Model ID | Format |
|--------|----------|--------|
| Bedrock API | `deepseek.v3-v1:0` | API version suffix (`-v1:0`) |
| Mantle API | `deepseek.v3.1` | Semantic version (`.1`) |

### Why Matching Fails

Location: `backend/lambdas/final-aggregator/handler.py:233-345`

```
Normalization steps:
  Bedrock: deepseek.v3-v1:0 → strip ":0" → strip "-v1" → "deepseek.v3"
  Mantle:  deepseek.v3.1    → (no suffix to strip)     → "deepseek.v3.1"

Match attempts:
  ✗ Exact: "deepseek.v3-v1:0" ≠ "deepseek.v3.1"
  ✗ Normalized: "deepseek.v3" ≠ "deepseek.v3.1"
  ✗ Prefix: "deepseek.v3.1".startswith("deepseek.v3") = True
            BUT next char is "." which is excluded from separator list
  ✗ Provider-agnostic: "v3" ≠ "v3.1"

Result: No match → deepseek.v3.1 becomes a "Mantle-only stub"
```

The prefix match logic at line 251-256 explicitly excludes `.` as a separator to prevent false positives like `v3` matching `v3.1`. This is correct in general but breaks this specific case.

### Affected Models
Any model where:
- Mantle uses `.X` semantic versioning
- Bedrock uses `-vX:0` API versioning

---

## Issue 2: Pricing API Model ID Format Mismatch

### Symptom
`deepseek.v3.2` incorrectly matches to `deepseek.r1` pricing instead of the correct `deepseek.deepseek-v3-2`.

### Root Cause
AWS Pricing API uses a redundant provider prefix format:

| Source | Model ID |
|--------|----------|
| Bedrock API | `deepseek.v3.2` |
| Pricing API | `deepseek.deepseek-v3-2` |

### Why Wrong Match Wins

Location: `backend/lambdas/pricing-linker/handler.py:262-426`

```
Normalization:
  Bedrock: deepseek.v3.2          → "deepseek32"
  Pricing: deepseek.deepseek-v3-2 → "deepseekdeepseek32" (extra "deepseek")
  Pricing: deepseek.r1            → "deepseekr1"

Similarity scores (SequenceMatcher):
  "deepseek32" vs "deepseekr1"           = 0.800 ← WRONG MATCH WINS
  "deepseek32" vs "deepseekdeepseek32"   = 0.714 ← Correct match loses

The shorter wrong match has higher character similarity than the
longer correct match with redundant prefix.
```

### Current Match Results

| Model ID | Matched To | Correct? |
|----------|------------|----------|
| `deepseek.v3.2` | `deepseek.r1` | **NO** |
| `deepseek.r1-v1:0` | `deepseek.r1` | YES |
| `deepseek.v3-v1:0` | `deepseek.deepseek-v3-1` | YES |

### Affected Models
Any model where Pricing API has redundant provider prefix in the model key.

---

## Issue 3: Bedrock API Returns Multiple Variants

### Symptom
Same model appears multiple times in the UI with the same `model_name` but different `model_id`.

### Examples

| Model Name | Variants in Final Output | Difference |
|------------|--------------------------|------------|
| Nova Premier | `amazon.nova-premier-v1:0`, `amazon.nova-premier-v1:0:mm` | `:mm` = multimodal |
| Nova Reel | `amazon.nova-reel-v1:0`, `amazon.nova-reel-v1:1` | Minor version |
| Titan Embed Image | `amazon.titan-embed-image-v1`, `amazon.titan-embed-image-v1:0` | API entry points |
| Titan Embed Text | `amazon.titan-embed-text-v1`, `amazon.titan-embed-text-v1:2` | Revision |
| Claude v2 | `anthropic.claude-v2:0`, `anthropic.claude-v2:1` | v2.0 vs v2.1 |
| Embed English | `cohere.embed-english-v3`, `cohere.embed-english-v3:0:512` | Fixed dimension |
| Embed Multilingual | `cohere.embed-multilingual-v3`, `cohere.embed-multilingual-v3:0:512` | Fixed dimension |

### Root Cause
Bedrock's `ListFoundationModels` API returns these as separate entries with different capabilities:

```
From Bedrock API (us-east-1):
  cohere.embed-english-v3        → inference_types: [ON_DEMAND]
  cohere.embed-english-v3:0:512  → inference_types: [PROVISIONED]
```

### Current Pipeline Behavior

| Stage | Behavior |
|-------|----------|
| **model-merger** | Filters context-window variants (`:8k`, `:200k`) but keeps version/capability variants |
| **Frontend** | Uses full `model_id` as React key, strips trailing `:N` for display |
| **Result** | Multiple cards with same/similar display name |

### Variant Suffixes Explained

| Suffix Pattern | Meaning | Example |
|----------------|---------|---------|
| `:0`, `:1`, `:2` | Minor version/revision | `claude-v2:1` = v2.1 |
| `:8k`, `:200k` | Context window (filtered) | Provisioned throughput sizing |
| `:0:512` | Version + dimension | Fixed 512-dim embedding output |
| `:mm` | Multimodal variant | Video input support |

---

## Issue 4: Availability Inheritance via Fuzzy Matching

### Symptom
`cohere.embed-english-v3:0:512` shows 12 regions in `in_region` but only supports PROVISIONED inference (not on-demand).

### Root Cause

Location: `backend/lambdas/final-aggregator/handler.py:1551-1629`

```python
def find_matching_availability(model_id, model_availability):
    # Try exact match first
    if model_id in model_availability:
        return model_availability[model_id]

    # Fallback: strip version suffix
    base_model_id = model_id.split(":")[0]  # cohere.embed-english-v3
    if base_model_id in model_availability:
        return model_availability[base_model_id]  # ← Inherits base model's regions
```

### Data Flow

```
regional-availability output:
  model_availability:
    cohere.embed-english-v3 → 12 regions (ON_DEMAND)
  provisioned_availability:
    cohere.embed-english-v3:0:512 → 6 regions (PROVISIONED only)

final-aggregator processing:
  For cohere.embed-english-v3:0:512:
    → No exact match in model_availability
    → base_model_id = "cohere.embed-english-v3"
    → Finds base model → Returns 12 regions
    → Result: in_region = 12 (INCORRECT - should be 0 for on-demand)
```

### Impact
Models that only support provisioned throughput appear to have on-demand availability in regions where they cannot actually be invoked on-demand.

---

## Fuzzy Matching Locations

| Lambda | Purpose | Matching Type | Location |
|--------|---------|---------------|----------|
| `model-merger` | Merge regions | **Exact** (strips `:NNNk` only) | `handler.py:29-39` |
| `regional-availability` | Region discovery | **Exact** | N/A |
| `pricing-linker` | Model → Pricing | **Fuzzy** (SequenceMatcher) | `handler.py:320-426` |
| `final-aggregator` | Model → Mantle | **Fuzzy** (normalization) | `handler.py:233-345` |
| `final-aggregator` | Model → Availability | **Fuzzy** (base ID fallback) | `handler.py:1551-1629` |
| `final-aggregator` | Model → Quotas | **Fuzzy** (heavy normalization) | `handler.py:367-395` |

---

## Recommended Actions

### Priority 1: Quick Wins

#### 1.1 Fix DeepSeek Mantle Matching
Add normalization rule to treat `-v1` as equivalent to `.1`:

```python
# In _normalize_for_mantle_match()
normalized = re.sub(r'-v(\d+)$', r'.\1', normalized)
```

#### 1.2 Fix Pricing API Redundant Prefix
Add deduplication for `provider.provider-model` pattern:

```python
# In normalize_model_id()
normalized = re.sub(r'^(\w+)\.\1-', r'\1.', normalized)
```

#### 1.3 Fix Availability Inheritance
Don't inherit on-demand regions for provisioned-only models:

```python
# In find_matching_availability() or transform_model_to_schema()
if 'ON_DEMAND' not in model.get('inference_types_supported', []):
    regional_availability = []  # Don't inherit base model's on-demand regions
```

### Priority 2: Medium Effort

#### 2.1 Differentiate Model Names for Variants
Append distinguishing suffix to `model_name`:

| Current | Proposed |
|---------|----------|
| Embed English | Embed English (512-dim) for `:0:512` |
| Claude | Claude v2.1 for `:1` variant |
| Nova Premier | Nova Premier (Multimodal) for `:mm` |

#### 2.2 Add Variant Grouping in Frontend
Group models by base ID in the UI, show variants as expandable sub-entries.

### Priority 3: Architectural

#### 3.1 Consolidate Fuzzy Matching Logic
Create shared utility `backend/layers/common/python/shared/model_matcher.py`:
- Single source of truth for normalization rules
- Consistent matching behavior across all lambdas
- Easier to test and maintain

#### 3.2 Add Model ID Aliasing Table
Explicit mappings for known format differences in config:

```json
{
  "model_id_aliases": {
    "deepseek.v3-v1:0": ["deepseek.v3.1"],
    "deepseek.v3.2": ["deepseek.deepseek-v3-2"]
  }
}
```

#### 3.3 Variant Filtering Option
Add configuration to collapse variants at model-merger level:

```json
{
  "variant_handling": {
    "collapse_minor_versions": true,
    "collapse_dimension_variants": true,
    "keep_multimodal_variants": true
  }
}
```

---

## Files to Modify

| Issue | File | Lines/Section |
|-------|------|---------------|
| Mantle matching | `backend/lambdas/final-aggregator/handler.py` | 233-345 (`_normalize_for_mantle_match`, `build_mantle_inference`) |
| Pricing matching | `backend/lambdas/pricing-linker/handler.py` | 262-317 (`normalize_model_id`), 320-426 (`find_best_pricing_match`) |
| Availability inheritance | `backend/lambdas/final-aggregator/handler.py` | 1551-1629 (`find_matching_availability`) |
| Variant filtering | `backend/lambdas/model-merger/handler.py` | 105-135 (variant handling logic) |
| Frontend display | `frontend/src/components/models/ModelCard.jsx` | 125-136 (`getDisplayModelId`) |
| Configuration | `backend/config/profiler-config.json` | `provider_aliases`, `matching_configuration` |

---

## Testing Recommendations

1. **Unit tests** for each normalization function with edge cases
2. **Integration test** comparing expected vs actual matches for known problematic models
3. **Regression test** to ensure fixes don't break existing correct matches
4. **Visual review** of frontend after changes to verify duplicate reduction

---

## Appendix: Affected Model Examples

### DeepSeek (Mantle + Pricing issues)
- `deepseek.v3-v1:0` / `deepseek.v3.1` - Same model, different format
- `deepseek.v3.2` - Incorrectly matched to R1 pricing

### Amazon (Multiple variants)
- `amazon.nova-premier-v1:0` / `amazon.nova-premier-v1:0:mm`
- `amazon.nova-reel-v1:0` / `amazon.nova-reel-v1:1`
- `amazon.titan-embed-image-v1` / `amazon.titan-embed-image-v1:0`
- `amazon.titan-embed-text-v1` / `amazon.titan-embed-text-v1:2`

### Anthropic (Version variants)
- `anthropic.claude-v2:0` / `anthropic.claude-v2:1`

### Cohere (Dimension variants + availability inheritance)
- `cohere.embed-english-v3` / `cohere.embed-english-v3:0:512`
- `cohere.embed-multilingual-v3` / `cohere.embed-multilingual-v3:0:512`

### Anthropic Claude 4.x (RESOLVED)
- `anthropic.claude-opus-4-5-20251101-v1:0` - Was incorrectly matching to Claude 3 Opus
- `anthropic.claude-sonnet-4-5-20251022-v2:0` - Was incorrectly matching to Claude 3.5 Sonnet
- See [Issue 5](#issue-5-claude-4x-matching-to-claude-3x-resolved) for resolution details

---

## Issue 5: Claude 4.x Matching to Claude 3.x (RESOLVED)

### Problem Description

Claude Opus 4.5 (`anthropic.claude-opus-4-5-20251101-v1:0`) was incorrectly matching to Claude 3 Opus (`anthropic.claude-3-opus`) pricing instead of its correct pricing key.

**Root Cause:**
1. The `get_canonical_model_id()` function normalizes model IDs by stripping version suffixes
2. After normalization, "claude-opus-4-5" and "claude-3-opus" have similar fuzzy match scores
3. The `has_semantic_conflict()` function didn't detect Claude 3 vs Claude 4 as a conflict

### Solution

#### 1. Explicit Model Mappings

Added explicit mappings in `profiler-config.json`:

```json
{
  "matching_configuration": {
    "explicit_model_mappings": {
      "anthropic.claude-opus-4-5-20251101-v1:0": "anthropic.claude-opus-4-5",
      "anthropic.claude-opus-4-6-20260115-v1:0": "anthropic.claude-opus-4-6",
      "anthropic.claude-sonnet-4-5-20251022-v2:0": "anthropic.claude-sonnet-4-5",
      "anthropic.claude-sonnet-4-20250514-v1:0": "anthropic.claude-sonnet-4",
      "anthropic.claude-haiku-4-5-20251022-v1:0": "anthropic.claude-haiku-4-5"
    }
  }
}
```

#### 2. Enhanced Conflict Detection

Updated `has_semantic_conflict()` in `model_matcher.py` to detect:
- Claude major version differences (3 vs 4)
- Both naming patterns: `claude-3-opus` and `claude-opus-4`

The function now includes patterns for:
```python
# Check for Claude major version conflicts (3 vs 4)
# Pattern: claude-opus-4-5 vs claude-3-opus
claude_major_pattern = r"claude[._-]?(?:opus|sonnet|haiku)?[._-]?(\d+)"

# Also check for claude-X-variant vs claude-variant-X patterns
# e.g., claude-3-opus vs claude-opus-4
claude_variant_pattern = r"claude[._-](\d+)[._-](opus|sonnet|haiku)"
claude_variant_rev_pattern = r"claude[._-](opus|sonnet|haiku)[._-](\d+)"
```

#### 3. Priority Order

The pricing-linker now uses this priority:
1. **Explicit mapping** (confidence 1.0) - from config
2. **Semantic conflict check** - blocks mismatches
3. **Fuzzy matching** - fallback for unknown models

### Verification

To verify correct matching:

```bash
# Check pricing-linker logs for Claude models
aws logs filter-log-events \
  --log-group-name /aws/lambda/bedrock-profiler-pricing-linker \
  --filter-pattern "claude-opus-4" \
  --start-time $(date -d '1 hour ago' +%s000)
```

Expected log output:
```
Explicit mapping match model_id=anthropic.claude-opus-4-5-20251101-v1:0 mapped_to=anthropic.claude-opus-4-5 confidence=1.0
```

---

## Adding New Explicit Mappings

When a new model has matching issues:

1. Identify the model ID from Bedrock API
2. Identify the correct pricing key from Pricing API
3. Add mapping to `profiler-config.json`:

```json
{
  "matching_configuration": {
    "explicit_model_mappings": {
      "new.model-id-v1:0": "new.pricing-key"
    }
  }
}
```

4. Deploy config update (triggers self-healing agent)

---

## Related Files

- `backend/config/profiler-config.json` - Explicit mappings
- `backend/layers/common/python/shared/model_matcher.py` - Conflict detection
- `backend/lambdas/pricing-linker/handler.py` - Matching logic
