# Known Bugs

## Backend: `availability.batch.supported` does not distinguish in-region vs CRIS batch

**Status:** Open — frontend workaround in place
**Component:** `backend/lambdas/final-aggregator/handler.py` → `check_batch_inference()` (line ~1288)

### Problem

`model.availability.batch.supported` is a single boolean flag set to `true` if **any** pricing group starting with `"Batch"` exists for the model. This includes:

- `"Batch"` → in-region batch
- `"Batch Global"` → CRIS global batch
- `"Batch Geo"` → CRIS geographic batch
- `"Batch Long Context"`, `"Batch Long Context Global"`, etc.

There is no distinction between in-region batch and CRIS batch in the availability data model. A model with only in-region batch pricing would incorrectly show batch as available under the CRIS section.

### Current frontend workaround

In `AvailabilitySummary` (`frontend/src/components/models/ModelCardExpanded.jsx`), we derive `hasCrisBatch` by scanning `fullPricing.regions.{region}.pricing_groups` for any group matching `Batch Global` or `Batch Geo`. This is the same approach the Pricing tab uses.

### Proposed backend fix

In `build_availability()` and/or `check_batch_inference()`, separate the batch data into:

```python
"batch": {
    "supported": True,            # any batch
    "regions": [...],             # in-region batch regions
    "cris_supported": True,       # CRIS-specific batch
    "cris_regions": [...]         # CRIS batch source regions
}
```

Or split into `availability.batch` (in-region) and `availability.cris_batch` (cross-region).

This would allow the frontend to use the availability data directly instead of scanning pricing groups.
