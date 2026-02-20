# Schema Analysis - bedrock_models.json

Analysis performed: 2026-02-19

## Summary

- **Total fields**: 124
- **Used by frontend**: ~46
- **Potentially removable**: ~22 fields (~70-100 KB, 2.4% reduction)

---

## Fields NOT USED BY FRONTEND (but valuable API data)

| Field | Source | Recommendation |
|-------|--------|----------------|
| `feature_support.*` | API | **Keep** - valuable for future features |
| `chat_features.*` | API | **Keep** - indicates Converse API support |
| `short_description` | API | **Keep** - could use in UI |
| `model_lifecycle.release_date` | API | **Keep** - could use in UI |

### feature_support structure
```json
{
  "agent": { "isSupported": true, "isStreamingSupported": true },
  "knowledge_base": { "isSupported": true, "isExternalSourcesSupported": true, "isParsingSupported": false },
  "flow": { "isSupported": false },
  "guardrails": { "isSupported": true },
  "prompt_caching": { "isSupported": true },
  "intelligent_routing": { "isSupported": false },
  "model_evaluation": { "isSupported": true },
  "prompt_management": { "isSupported": true },
  "batch_inference": { "baseModelSupported": true, "crossRegionSupported": false },
  "latency_optimized": { "isSupported": false },
  "system_tools": [{ "name": "web_search" }]
}
```

### chat_features structure
```json
{
  "function_calling": true,
  "function_calling_streaming": true,
  "citations": true,
  "documents": true,
  "chat_history": true,
  "system_role": true,
  "reasoning": { "embedded": false },
  "supported_image_types": ["jpeg", "png", "gif", "webp"],
  "supported_video_types": [],
  "supported_audio_types": [],
  "supported_document_types": ["pdf"]
}
```

---

## REDUNDANT Fields (can be computed client-side)

| Field | Reason | Approx Size |
|-------|--------|-------------|
| `has_pricing` | = `model_pricing.is_pricing_available` | ~1 KB |
| `has_quotas` | = `Object.keys(model_service_quotas).length > 0` | ~1 KB |
| `total_regions_available` | = `regions_available.length` | ~1 KB |
| `mantle_inference.total_mantle_regions` | = `mantle_regions.length` | ~1 KB |
| `provisioned_throughput.total_provisioned_regions` | = `provisioned_regions.length` | ~1 KB |
| `converse_data.capabilities_count` | = `model_capabilities.length` | ~1 KB |
| `converse_data.use_cases_count` | = `model_use_cases.length` | ~1 KB |
| `converse_data.regions_count` | = `regions_available.length` | ~1 KB |
| `converse_data.size_category` | Can compute from `context_window` | 6 KB |
| `model_pricing.pricing_reference_id` | = `pricing_file_reference.model_key` | ~2 KB |

**Total**: ~15 KB

---

## INTERNAL METADATA (should remove from output)

| Field | Purpose | Size |
|-------|---------|------|
| `collection_metadata.first_discovered_at` | Internal tracking | - |
| `collection_metadata.first_discovered_in_region` | Internal tracking | - |
| `collection_metadata.api_source` | Internal tracking | - |
| `collection_metadata.dual_region_collection` | Internal tracking | - |
| `collection_metadata.regions_collected_from` | Internal tracking | - |
| `collection_metadata.phase2_regional_discovery` | Internal tracking | - |
| `collection_metadata.regional_data_source` | Internal tracking | - |
| `regional_availability_source` | Internal tracking | ~1 KB |
| `model_pricing.pricing_summary.integration_source` | Internal tracking | - |
| `model_pricing.pricing_summary.integration_timestamp` | Internal tracking | - |
| `model_pricing.pricing_summary.reference_based` | Internal tracking | - |
| `model_pricing.pricing_summary.has_pricing_data` | Duplicate of is_pricing_available | - |
| `converse_data.source` | Internal tracking | ~1 KB |
| `converse_data.verified` | Internal flag | ~1 KB |
| `converse_data.litellm_verified` | Internal flag | ~1 KB |
| `batch_inference_supported.detection_method` | Internal tracking | ~1 KB |

**Total collection_metadata**: 34 KB
**Total pricing_summary**: 18 KB
**Total other internal**: ~5 KB

---

## INVENTED/HARDCODED Data (from config, not API)

| Field | Source | Recommendation |
|-------|--------|----------------|
| `documentation_links` | `profiler-config.json` | **Keep** - useful links |
| `model_arn` | Constructed from model_id | **Remove** - can derive client-side |

---

## Size Analysis by Field

| Field | Total Size | Per Model |
|-------|-----------|-----------|
| `collection_metadata` | 34.0 KB | ~293 bytes |
| `feature_support` | 28.4 KB | ~244 bytes |
| `description` | 39.3 KB | ~339 bytes |
| `model_pricing.pricing_summary` | 18.4 KB | ~158 bytes |
| `short_description` | 14.4 KB | ~124 bytes |
| `chat_features` | 11.9 KB | ~103 bytes |
| `model_arn` | 8.7 KB | ~75 bytes |
| `converse_data.size_category` | 6.2 KB | ~53 bytes |

---

## Recommendations

### Phase 1: Quick wins (minimal code changes)
1. Remove `collection_metadata` entirely
2. Remove `model_pricing.pricing_summary`
3. Remove `regional_availability_source`
4. Remove `model_arn`

**Estimated savings**: ~55 KB (1.2%)

### Phase 2: Redundant field cleanup
1. Remove `has_pricing`, `has_quotas`
2. Remove `total_*` count fields
3. Remove `converse_data.*_count` fields
4. Remove `converse_data.size_category`

**Estimated savings**: ~15 KB (0.3%)

### Phase 3: Internal flags cleanup
1. Remove `converse_data.source`, `verified`, `litellm_verified`
2. Remove `batch_inference_supported.detection_method`

**Estimated savings**: ~5 KB

---

## Fields to KEEP

These fields are from the API and valuable even if not currently used:

- `feature_support` - Bedrock feature compatibility (agents, KB, guardrails, etc.)
- `chat_features` - Converse API capabilities (function calling, citations, etc.)
- `short_description` - Concise model description
- `description` - Full model description
- `model_lifecycle.release_date` - Model release timestamp
- `languages_supported` - Supported languages list
