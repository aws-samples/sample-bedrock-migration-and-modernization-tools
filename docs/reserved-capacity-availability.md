# Reserved Capacity Availability - Gap & Proposed Fix

**Created**: 2026-03-06
**Status**: IMPLEMENTED
**Priority**: Medium

---

## The Problem

The `availability` object in `bedrock_models.json` has a `provisioned` section but **no `reserved` section**. These are two distinct capacity types:

| Type | API Source | Models | Era |
|------|-----------|--------|-----|
| Provisioned Throughput | `ListFoundationModels(byInferenceType="PROVISIONED")` | 20 older models (Titan, Llama 2, Claude 3) | Legacy |
| Reserved Capacity | **None exists** | 6+ newer models (Claude 4.x, Nova, Llama 4) | Current |

AWS is transitioning newer models from Provisioned Throughput to Reserved Capacity, but there is **no Bedrock API** to query Reserved availability (`byInferenceType="RESERVED"` does not exist).

### What we checked (all 11 sources)

- `ListFoundationModels(byInferenceType="PROVISIONED")` — only legacy Provisioned Throughput
- `ListFoundationModels` — no `RESERVED` inference type option
- `ListInferenceProfiles` — only CRIS profiles
- Service Quotas — no Reserved-specific quotas
- Feature Collector, Mantle, Token Specs, Lifecycle, Console API — none track Reserved
- **Pricing API — ONLY source** where Reserved entries appear

### Zero overlap between the two sets

```
Provisioned Throughput (20 models):
  amazon.titan-*, meta.llama2-*, anthropic.claude-3-haiku/sonnet/opus, etc.

Reserved Capacity (6+ models):
  anthropic.claude-sonnet-4-5, anthropic.claude-sonnet-4-6,
  amazon.nova-pro/lite/micro, meta.llama4-scout, etc.

Overlap: NONE
```

---

## The Fix

Derive `availability.reserved` from pricing data in `final-aggregator/handler.py`.

### Logic

If a model has pricing entries with `pricing_group` containing "Reserved" (e.g., "Reserved 1 Month", "Reserved 3 Month Global"), then:
1. `availability.reserved.supported = true`
2. `availability.reserved.regions` = regions where those Reserved pricing entries exist

### Where to implement

- **File**: `backend/lambdas/final-aggregator/handler.py`
- **Pattern**: Similar to existing `build_provisioned_throughput()` function (~line 499)
- **New function**: `build_reserved_capacity(model_id, pricing_data)` that scans pricing entries for Reserved groups
- **Schema change**: Add `availability.reserved` to `transform_model_to_schema()`

### Proposed schema addition

```json
{
  "availability": {
    "on_demand": { ... },
    "cross_region": { ... },
    "batch": { ... },
    "provisioned": { ... },
    "mantle": { ... },
    "reserved": {
      "supported": true,
      "regions": ["us-east-1", "us-west-2", "eu-west-1"],
      "commitments": ["1_month", "3_month", "6_month"]
    }
  }
}
```

### Frontend impact

- Update `useModels.js` to read `availability.reserved`
- Add Reserved Capacity indicator to model cards/details
- Update `docs/JSON-STRUCTURE.md` schema docs

---

## GovCloud & Isolated Partitions

### The Problem

The pipeline dynamically discovers regions via `region-discovery` Lambda (`ec2.describe_regions()` + `list_inference_profiles`), but this only finds **commercial partition** regions. GovCloud (`us-gov-west-1`, `us-gov-east-1`), China (`cn-north-1`, `cn-northwest-1`), and ISO/ISOB regions are separate AWS partitions with separate credentials.

### Tested (2026-03-06)

All API calls from commercial account credentials fail:

| API | us-gov-west-1 | us-gov-east-1 |
|-----|---------------|---------------|
| `ListFoundationModels` | `UnrecognizedClientException` | `UnrecognizedClientException` |
| `ListInferenceProfiles` | `UnrecognizedClientException` | `UnrecognizedClientException` |

Commercial credentials are **not valid** in GovCloud. The endpoints resolve but auth is rejected.

### What we DO have

The **Pricing API** (called from `us-east-1`) returns **global data across all partitions**, including GovCloud. Models like Nova Pro, Nova Lite, and Llama 3 8B already have pricing entries with `regionCode: "us-gov-west-1"`.

### Data availability per partition

| Data Type | Commercial | GovCloud | China | ISO/ISOB |
|-----------|-----------|----------|-------|----------|
| Availability (ListFoundationModels) | Yes | No (needs GovCloud creds) | No | No |
| Quotas (ServiceQuotas) | Yes | No | No | No |
| Features (ListInferenceProfiles) | Yes | No | No | No |
| Pricing | Yes | **Yes** (via global Pricing API) | Unknown | No |

### Proposed Fix

Same approach as Reserved Capacity — derive GovCloud availability from pricing data in `final-aggregator`:

1. Scan pricing entries for GovCloud region codes (`us-gov-west-1`, `us-gov-east-1`)
2. Add those regions to `availability.on_demand.regions` with a `source: "pricing"` flag to distinguish from API-confirmed availability
3. No changes needed to `region-discovery` or collection Lambdas

**Alternative**: If GovCloud credentials become available in the future, add `us-gov-west-1` and `us-gov-east-1` to a `govcloud_regions` config list and make cross-partition API calls with separate credentials.

---

## References

- Pricing classification: `backend/lambdas/pricing-aggregator/handler.py` → `detect_reserved_pricing()` (line ~203)
- Current provisioned builder: `backend/lambdas/final-aggregator/handler.py` → `build_provisioned_throughput()` (line ~499)
- Schema docs: `docs/JSON-STRUCTURE.md`
