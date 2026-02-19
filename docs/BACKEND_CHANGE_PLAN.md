# Backend Change Plan

> **Generated**: 2026-02-19
> **Based on**: Deep Code Audit (Task 01), Comprehensive Architecture Report (Task 02), Hard-Coded Values Risk Report (Task 03), Redundancy & Enhancement Analysis (Task 04)
> **Scope**: All 17 Lambda handlers, shared layer, ASL workflow, profiler config, SAM templates

---

## Executive Summary

The Bedrock Model Profiler backend is a functional, daily-running pipeline of 17 Lambda functions orchestrated by Step Functions. It collects, enriches, and aggregates Amazon Bedrock model data from multiple AWS APIs and external sources. The pipeline works — but it has accumulated technical debt that creates **silent failure risks** and **scalability bottlenecks**.

The most urgent issues are: (1) **4 critical hard-coded values** — external API URLs and fallback region lists that will silently produce incomplete data when third-party services change or AWS adds regions; (2) **a 1,365-line monolith** (final-aggregator) that handles 11 responsibilities and is the single biggest maintainability risk; and (3) **config values that exist in `profiler-config.json` but are ignored by code** that hard-codes its own copies — the lowest-effort, highest-impact fix available.

This plan organizes improvements into 5 tracks (hard-coded cleanup, redundancy elimination, scalability, documentation, self-healing backlog) and 4 phases (quick wins through backlog). The guiding principle is the user's stated goal: **clean the code, make it scalable, avoid hard-coded values, and trust the information retrieved from APIs**.

---

## Guiding Principles

1. **Risk reduction first** — fix things that can break silently before cosmetic improvements
2. **Quick wins before large refactors** — wire up existing config before building new infrastructure
3. **No breaking changes to pipeline output format** — `latest/bedrock_models.json` and `latest/bedrock_pricing.json` schemas must remain backward-compatible
4. **Trust retrieved data** — prefer API-sourced information over hard-coded overrides; hard-code only as fallback
5. **Self-healing agent remains backlog** — document and gate, don't implement or expand
6. **Deploy safely** — changes must be deployable without disrupting the daily 6 AM UTC execution

---

## Change Tracks

### Track 1: Hard-Coded Value Externalization (Risk Avoidance)

This track addresses the 18 values recommended for externalization in the Hard-Coded Values Risk Report, plus wiring up 6 config sections that already exist in `profiler-config.json` but are ignored by code.

#### T1.1 — Wire Up Existing Config Sections (Code Ignores Config)

**Priority**: P0 | **Effort**: S (2-3 hours) | **Risk**: Low | **Phase**: 1 (Quick Win)

The S3 `profiler-config.json` already contains sections that the code hard-codes instead of reading. This is the **single highest-impact, lowest-effort improvement** — the config infrastructure already exists.

| Config Section | Exists in S3 Config | Code That Ignores It | Fix |
|----------------|:-------------------:|----------------------|-----|
| `capability_keywords` | Yes | `model-enricher/handler.py:83-92` hard-codes `['code', 'codestral', ...]` | Replace hard-coded lists with `config.get('capability_keywords')` |
| `reasoning_models` | Yes | `model-enricher/handler.py:87` hard-codes `['claude', 'sonnet', ...]` | Replace with `config.get('reasoning_models')` |
| `function_calling_models` | Yes | `model-enricher/handler.py:91` hard-codes `['claude', 'nova', ...]` | Replace with `config.get('function_calling_models')` |
| `context_window_thresholds` | Yes | `final-aggregator/handler.py:121-126` hard-codes `128000/32000` | Replace with `config.get('context_window_thresholds')` |
| `suffixes_to_remove` | Yes | `pricing-linker/handler.py:248` hard-codes `['-it', '-instruct', ...]` | Replace with `config.get_suffixes_to_remove()` (accessor exists) |
| `type_conflicts` | Yes (accessor exists) | `pricing-linker/handler.py:200-204` hard-codes conflict groups | Replace with `config.get_type_conflicts()` (accessor exists) |

**Files to modify**: `model-enricher/handler.py`, `final-aggregator/handler.py`, `pricing-linker/handler.py`

**Migration approach**: Each Lambda already initializes `_config_loader`. Add calls to existing config accessors (or add simple accessors where missing). Keep the hard-coded values as inline fallback defaults in case config loading fails.

**Rollback plan**: Revert the Lambda code; config values remain in S3 unused (no harm).

**Success metric**: Zero hard-coded capability keywords, context window thresholds, or suffix lists in Lambda code. All values sourced from `profiler-config.json` at runtime.

---

#### T1.2 — Externalize Critical External API URLs

**Priority**: P0 | **Effort**: S (1-2 hours) | **Risk**: Low | **Phase**: 1 (Quick Win)

These 4 values are the highest-risk hard-coded items. A third-party change breaks data collection silently.

| # | Value | Current Location | Target Config Path | Risk if Unchanged |
|---|-------|-----------------|-------------------|-------------------|
| 1 | LiteLLM URL `https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json` | `token-specs-collector/handler.py:29` | `external_apis.litellm_url` | **CRITICAL** — BerriAI renames repo, all token specs lost silently |
| 2 | Mantle endpoint pattern `bedrock-mantle.{region}.api.aws` | `mantle-collector/handler.py:30` + `final-aggregator/handler.py:228` | `external_apis.mantle_endpoint_pattern` | **CRITICAL** — AWS changes hostname, all Mantle data lost; duplicated in 2 files |
| 3 | Bulk Pricing URL template | `pricing-collector/handler.py:38` | `external_apis.bulk_pricing_url_template` | **HIGH** — versioned URL (`/offers/v1.0/`) could change |
| 4 | Fallback region list (16 regions, duplicated) | `region-discovery/handler.py:38-43, 136-141` | `region_configuration.fallback_regions` | **HIGH** — AWS adds ~2-4 regions/year; stale fallback is invisible |

**Migration approach**:
1. Add an `external_apis` section to `profiler-config.json` with the 3 URL values
2. Add a `fallback_regions` key under `region_configuration`
3. Add `ConfigLoader` accessors: `get_litellm_url()`, `get_mantle_endpoint_pattern()`, `get_bulk_pricing_url_template()`, `get_fallback_regions()`
4. Update each Lambda to read from config, with the current hard-coded value as the fallback default in the accessor
5. Deduplicate: `region-discovery` defines the fallback list once as a module constant, then replace with config; `final-aggregator` removes its Mantle endpoint copy and reads from config

**Rollback plan**: Config accessors have fallback defaults matching current hard-coded values. If config loading fails, behavior is identical to today.

**Success metric**: All 4 values configurable at runtime without code deployment. Mantle endpoint defined in exactly 1 place. Fallback region list defined in exactly 1 place.

---

#### T1.3 — Externalize High-Priority Values

**Priority**: P1 | **Effort**: S (1-2 hours) | **Risk**: Low | **Phase**: 2

| # | Value | Current Location | Target Config Path |
|---|-------|-----------------|-------------------|
| 1 | `_PROVIDER_PREFIXES` (20 entries) | `final-aggregator/handler.py:282-303` | `provider_configuration.provider_prefixes` |
| 2 | `anthropic_version: 'bedrock-2023-05-31'` | `self-healing-agent/handler.py:162` | `agent_configuration.anthropic_version` |
| 3 | S3 latest paths (`latest/bedrock_models.json`, etc.) | `copy-to-latest/handler.py:149-150`, `gap-detection/handler.py:262` | `s3_paths.latest_models`, `s3_paths.latest_pricing` |
| 4 | Capability-to-use-case mapping (13 entries) | `model-enricher/handler.py:105-119` | `model_configuration.capability_use_case_mapping` |

**Migration approach**: Same pattern as T1.2 — add config sections, add accessors with fallback defaults, update Lambdas.

**Rollback plan**: Fallback defaults in accessors match current hard-coded values.

**Success metric**: All 4 value groups configurable at runtime.

---

#### T1.4 — Externalize Medium-Priority Values

**Priority**: P2 | **Effort**: S (1-2 hours) | **Risk**: Low | **Phase**: 2

| # | Value | Current Location | Target Config Path |
|---|-------|-----------------|-------------------|
| 1 | `temperature: 0.2` | `self-healing-agent/handler.py:170` | `agent_configuration.temperature` |
| 2 | `User-Agent: BedrockProfiler/1.0` | `token-specs-collector/handler.py:57` | `external_apis.user_agent` |
| 3 | `max_batches=100` | `pricing-collector/handler.py:126` | `pricing_configuration.max_batches` |
| 4 | Pricing type detection patterns | `pricing-aggregator/handler.py:69-173` | `pricing_configuration.pricing_type_patterns` |
| 5 | Pricing group detection patterns | `pricing-aggregator/handler.py:176-227` | `pricing_configuration.pricing_group_patterns` |

**Note**: Items 4-5 are the most complex — the pricing classification logic is imperative (if/elif chains). Converting to declarative config requires careful testing. Consider implementing as a Phase 3 item if the declarative approach proves complex.

**Success metric**: All 5 value groups configurable at runtime.

---

#### T1.5 — Add Monitoring for Keep-As-Is Values

**Priority**: P2 | **Effort**: M (3-4 hours) | **Risk**: Low | **Phase**: 2

These values should stay in code but need operational monitoring:

| Value | Monitoring Action | Implementation |
|-------|-------------------|----------------|
| `x-console-consumer: true` header | CloudWatch alarm when console metadata count < 30 models | Add custom metric in `model-extractor` |
| `FormatVersion='aws_v1'` | CloudWatch alarm on Pricing API errors | Add custom metric in `pricing-collector` |
| Pricing type/group detection defaults | Log metric when default type is used | Add counter in `pricing-aggregator` |
| Fallback region list usage | CloudWatch metric when fallback is triggered | Add custom metric in `region-discovery` |
| `token-specs-collector` empty results | CloudWatch alarm when `modelsWithSpecs == 0` | Add alarm on existing output metric |
| ASL timeout for final-aggregator | Alert at 80% of `TimeoutSeconds` | CloudWatch alarm on execution duration |

**Success metric**: CloudWatch dashboard with 6+ pipeline health metrics; at least 2 alarms configured for critical silent failures.

---

### Track 2: Redundancy Elimination (Maintainability)

This track addresses the 15 redundancies identified in Section 8 of the architecture report, ordered by priority.

#### T2.1 — Token-Specs-Collector Shared Layer Adoption

**Priority**: P1 | **Effort**: S (1 hour) | **Risk**: Low | **Phase**: 1 (Quick Win)

**What**: Refactor `token-specs-collector/handler.py` to use the shared layer instead of its own implementations.

**Addresses**: Redundancies 8.1 (duplicate RETRY_CONFIG), 8.2 (duplicate S3 read/write), 8.5 (duplicate execution ID parsing), 8.15 (duplicate get_s3_client)

**Files affected**:
- `backend/lambdas/token-specs-collector/handler.py` — Remove lines 22-50 (local RETRY_CONFIG, get_s3_client, read_from_s3, write_to_s3), add shared layer imports, replace `event['s3Bucket']` with `validate_required_params`

**Changes**:
1. Replace `from botocore.config import Config` + local RETRY_CONFIG with `from shared import RETRY_CONFIG`
2. Replace local `get_s3_client()`, `read_from_s3()`, `write_to_s3()` with `from shared import get_s3_client, read_from_s3, write_to_s3`
3. Replace inline execution ID parsing (L159-160) with `from shared import parse_execution_id`
4. Replace direct `event['s3Bucket']` access with `from shared import validate_required_params, ValidationError`
5. Add proper error handling with `S3ReadError`, `S3WriteError`

**Expected benefit**: ~30 lines removed; consistent error handling; future shared layer improvements automatically apply.

**Risk**: Low — the shared layer functions are well-tested and used by 13 other Lambdas.

**Success metric**: `token-specs-collector` has zero local S3/retry implementations; imports exclusively from shared layer.

---

#### T2.2 — Regional-Availability RETRY_CONFIG Cleanup

**Priority**: P1 | **Effort**: S (15 min) | **Risk**: Low | **Phase**: 1 (Quick Win)

**What**: Remove the duplicate `RETRY_CONFIG` in `regional-availability/handler.py:47-51` and use the shared layer's config.

**Addresses**: Redundancy 8.1

**Files affected**: `backend/lambdas/regional-availability/handler.py`

**Changes**: Remove local `RETRY_CONFIG` definition (lines 47-51). Import `RETRY_CONFIG` from shared layer. If the lower `connect_timeout=5` is intentionally different for parallel region probing, create a `RETRY_CONFIG_FAST` variant in the shared layer with a docstring explaining the use case.

**Decision needed**: Is `connect_timeout=5` intentional? If yes, add `RETRY_CONFIG_FAST` to shared layer. If no, just import the standard config.

**Expected benefit**: One fewer inconsistent timeout configuration; clear documentation of intent.

**Success metric**: Zero local RETRY_CONFIG definitions outside the shared layer (except intentional named variants).

---

#### T2.3 — Provider Prefix List Consolidation

**Priority**: P1 | **Effort**: M (2-3 hours) | **Risk**: Medium | **Phase**: 2

**What**: Consolidate the 3 separate provider prefix/pattern lists into a single config-driven source.

**Addresses**: Redundancy 8.6

**Files affected**:
- `backend/config/profiler-config.json` — Add `provider_configuration.provider_prefixes` section
- `backend/layers/common/python/shared/config_loader.py` — Add `get_provider_prefixes()` accessor
- `backend/lambdas/final-aggregator/handler.py` — Replace `_PROVIDER_PREFIXES` (L282-303) with config call; replace 9 hard-coded `.replace()` calls (L1061-1080) with config-driven loop
- `backend/lambdas/pricing-aggregator/handler.py` — Verify `infer_provider` uses config patterns consistently

**Current state (out of sync)**:
- `final-aggregator` `_PROVIDER_PREFIXES`: 20 entries
- `final-aggregator` `find_matching_availability` `.replace()` chain: 9 entries
- These are already out of sync — availability matching silently fails for 11 providers

**Migration approach**:
1. Add `provider_prefixes` to config with the full 20-entry list
2. Add `get_provider_prefixes()` accessor
3. Replace `_PROVIDER_PREFIXES` with `config.get_provider_prefixes()`
4. Replace the `.replace()` chain with a loop over the same config list
5. Test with current pipeline output to verify no regression

**Risk**: Medium — changing matching logic could affect quota/availability linkage. Requires full pipeline test.

**Success metric**: Provider prefix list defined in exactly 1 place (config). Adding a new provider requires only a config update, not a code change.

---

#### T2.4 — Final Aggregator Decomposition

**Priority**: P1 | **Effort**: XL (8-16 hours) | **Risk**: High | **Phase**: 3

**What**: Extract 3 major subsystems from the 1,365-line `final-aggregator/handler.py` into shared layer modules.

**Addresses**: Redundancy 8.9 (monolith), Over-complexity 8.16.1

**Current state**: 1,365 lines, 11 responsibilities, the single biggest maintainability risk in the backend.

**Decomposition strategy**:

| New Module | Extracted From | Lines | Responsibilities |
|-----------|---------------|-------|-----------------|
| `shared/quota_matching.py` | `final-aggregator:249-520` | ~270 | `_normalize_for_quota_matching`, `_extract_quota_model_ref`, `_build_model_aliases`, `_build_quota_index`, `build_model_quotas` |
| `shared/context_window.py` | `final-aggregator:714-796` | ~80 | `resolve_context_window` with 4-tier priority system |
| `shared/availability_matching.py` | `final-aggregator:1005-1099` | ~95 | `find_matching_availability` with provider prefix stripping |

**Result**: Handler reduced from 1,365 to ~920 lines (orchestration + model transformation). Each extracted module is independently testable.

**Migration approach**:
1. Create the 3 new shared layer modules with the extracted functions (no logic changes)
2. Update `final-aggregator/handler.py` to import from the new modules
3. Run full pipeline and compare output byte-for-byte with pre-refactor output
4. Add unit tests for each extracted module

**Risk**: High — this is the most complex refactor. The internal state flow between functions is tightly coupled. Requires careful extraction to avoid breaking data dependencies. Must be tested with a full pipeline run.

**Rollback plan**: Revert to the monolithic handler. The shared layer modules can be deleted without affecting other Lambdas.

**Success metric**: `final-aggregator/handler.py` < 1,000 lines. Quota matching, context window resolution, and availability matching each have their own module with unit tests.

---

#### T2.5 — Pricing Aggregator Function Decomposition

**Priority**: P2 | **Effort**: L (4-6 hours) | **Risk**: Medium | **Phase**: 3

**What**: Decompose the 184-line `aggregate_pricing` God function into focused sub-functions.

**Addresses**: Redundancy 8.8, Over-complexity 8.16.2

**Files affected**: `backend/lambdas/pricing-aggregator/handler.py`

**Decomposition**:
1. `extract_model_info(product)` — model name extraction + provider inference (current steps 1-2)
2. `classify_and_normalize(product, model_info)` — pricing group/type classification + price normalization (steps 3-4)
3. `structure_by_provider(classified_products)` — output structuring (step 5)
4. `apply_fallbacks_and_stats(structured_data)` — Global-to-OnDemand fallback + statistics (steps 6-7)

**Success metric**: No single function > 60 lines in pricing-aggregator. Each sub-function independently testable.

---

#### T2.6 — Analytics Lambda Minimal Shared Layer Adoption

**Priority**: P1 | **Effort**: S (30 min) | **Risk**: Low | **Phase**: 1 (Quick Win)

**What**: Replace `print()` with `logger` in the analytics Lambda.

**Addresses**: Redundancy 8.12 (partial — full shared layer adoption is lower priority)

**Files affected**: `backend/lambdas/analytics/handler.py`

**Changes**: Replace 5 `print()` calls (lines 399, 429, 460, 501, 564) with `logger.error()` / `logger.warning()`. Add `from shared import get_logger` or define a local logger with `logging.getLogger()`.

**Note**: Full shared layer adoption (validation, error responses, RETRY_CONFIG) is lower priority because the analytics Lambda is architecturally different (API Gateway + DynamoDB). The `print()` → `logger` fix is the highest-value minimal change.

**Success metric**: Zero `print()` calls in analytics Lambda. All errors visible in CloudWatch with proper log levels.

---

#### T2.7 — Delete Legacy Pricing-Linker Artifacts

**Priority**: P2 | **Effort**: S (15 min) | **Risk**: Low | **Phase**: 1 (Quick Win)

**What**: Remove 987 lines of dead code from the pricing-linker directory.

**Addresses**: Redundancy 8.13

**Files to delete**:
- `backend/lambdas/pricing-linker/handler_v1.py` (301 lines) — original implementation, not referenced by any SAM template or ASL
- `backend/lambdas/pricing-linker/compare_implementations.py` (686 lines) — CLI comparison tool with its own drifting `V2_PROVIDER_ALIASES`

**Note**: If the comparison script is still useful for development, move it to a `tools/` or `scripts/` directory outside the Lambda deployment package.

**Success metric**: `pricing-linker/` directory contains only `handler.py` and `requirements.txt`. Lambda deployment package is ~30KB smaller.

---

#### T2.8 — Duplicate Documentation Link Logic

**Priority**: P2 | **Effort**: S (30 min) | **Risk**: Low | **Phase**: 2

**What**: Consolidate documentation link resolution into a single function.

**Addresses**: Redundancy 8.4

**Files affected**:
- `backend/layers/common/python/shared/config_loader.py` — Add `get_documentation_links(model_id, provider)` method
- `backend/lambdas/model-extractor/handler.py` — Remove local `get_documentation_links` (L277-288), use config method
- `backend/lambdas/model-enricher/handler.py` — Remove local `get_documentation_links` (L128-143), use config method

**Success metric**: Documentation link resolution defined in exactly 1 place.

---

#### T2.9 — Mantle Endpoint Deduplication

**Priority**: P3 | **Effort**: S (15 min) | **Risk**: Low | **Phase**: 2

**What**: Remove the duplicate Mantle endpoint pattern from `final-aggregator`.

**Addresses**: Redundancy 8.7

**Note**: This is automatically resolved by T1.2 (externalizing Mantle endpoint to config). Both `mantle-collector` and `final-aggregator` will read from the same config key.

---

### Track 3: Scalability Improvements

This track focuses on changes that improve the pipeline's ability to handle growth: more AWS regions, more model providers, more pricing service codes, and larger data volumes.

#### T3.1 — Config Validation on Startup

**Priority**: P1 | **Effort**: M (3-4 hours) | **Risk**: Low | **Phase**: 2

**What**: Add a config validation step that runs when `ConfigLoader` loads config, catching issues before they cause silent failures.

**Files affected**:
- `backend/layers/common/python/shared/config_loader.py` — Add `validate_config()` method

**Validations to implement**:
1. Required sections exist: `provider_configuration`, `region_configuration`, `model_configuration`, `matching_configuration`
2. Required keys within sections (e.g., `provider_aliases` is a non-empty dict)
3. Region lists are non-empty
4. External API URLs are valid URL patterns (if externalized per T1.2)
5. Numeric thresholds are within reasonable ranges (e.g., `min_confidence_threshold` between 0 and 1)
6. Version string format check (detect runaway `-auto-updated` appending)

**Behavior on validation failure**: Log warnings but continue with defaults. Do NOT fail the Lambda — validation is advisory, not blocking.

**Success metric**: Config validation runs on every Lambda cold start. Validation warnings appear in CloudWatch logs when config is malformed.

---

#### T3.2 — Dynamic Region Handling in ASL

**Priority**: P2 | **Effort**: M (2-3 hours) | **Risk**: Medium | **Phase**: 2

**What**: Replace hard-coded region lists in the ASL workflow with dynamic region discovery output.

**Current state**: The ASL workflow hard-codes:
- `["us-east-1", "us-west-2"]` for model extraction regions (L35-38)
- 16 quota regions (L39-56)

**Target state**: Use `DiscoverRegions` output to drive all region-dependent Map states.

**Files affected**:
- `backend/statemachine/bedrock-profiler.asl.json` — Replace hard-coded region arrays with references to `$.regionDiscovery.featureRegions` (or a subset)
- May need to add region categorization to `region-discovery` output (model regions vs quota regions vs feature regions)

**Risk**: Medium — changing the ASL affects the entire pipeline execution. Requires careful testing with `sam local` before deployment.

**Success metric**: Zero hard-coded region lists in the ASL workflow. Adding a new AWS region requires zero code or ASL changes.

---

#### T3.3 — Declarative Pricing Classification

**Priority**: P2 | **Effort**: L (4-6 hours) | **Risk**: Medium | **Phase**: 3

**What**: Convert the imperative pricing type/group classification logic in `pricing-aggregator` to a declarative, config-driven rule system.

**Current state**: `determine_pricing_type` (L57) and `determine_pricing_group` (L176) use if/elif chains with hard-coded string patterns. Adding a new pricing type requires modifying code.

**Target state**: Classification rules defined in `profiler-config.json`:
```json
{
  "pricing_configuration": {
    "type_rules": [
      {"pattern": "image", "fields": ["usage_type", "description"], "type": "image"},
      {"pattern": "video", "fields": ["usage_type"], "type": "video_generation"}
    ],
    "group_rules": [
      {"pattern": "batch", "fields": ["usage_type"], "group": "Batch"},
      {"pattern": "global", "fields": ["usage_type"], "group_suffix": "Global"}
    ]
  }
}
```

**Files affected**:
- `backend/config/profiler-config.json` — Add `pricing_configuration` section
- `backend/layers/common/python/shared/config_loader.py` — Add accessors
- `backend/lambdas/pricing-aggregator/handler.py` — Replace if/elif chains with rule engine

**Success metric**: Adding a new pricing type requires only a config update. Zero if/elif chains for pricing classification.

---

#### T3.4 — Parallel S3 Reads in Final Aggregator

**Priority**: P1 | **Effort**: M (2-3 hours) | **Risk**: Low | **Phase**: 2

**What**: Use `ThreadPoolExecutor` for concurrent S3 reads in the final aggregator.

**Addresses**: Enhancement 9.3.1

**Current state**: Final aggregator reads 6+ S3 objects sequentially. With growing data (more regions, more models), this becomes a bottleneck.

**Files affected**: `backend/lambdas/final-aggregator/handler.py`

**Changes**: Wrap the sequential S3 reads (models_with_pricing, availability_data, token_specs_data, pricing_data, enriched_models_data, plus per-region quotas/features/mantle) in a `ThreadPoolExecutor` with `max_workers=6`.

**Expected benefit**: 50-70% faster aggregation phase.

**Success metric**: Final aggregator S3 read phase completes in < 5 seconds (currently ~10-15 seconds).

---

#### T3.5 — LiteLLM Data Caching with ETag

**Priority**: P1 | **Effort**: S (1-2 hours) | **Risk**: Low | **Phase**: 2

**What**: Cache the LiteLLM model database in S3 and only re-download when the ETag changes.

**Addresses**: Enhancement 9.3.5

**Current state**: `token-specs-collector` downloads a ~2MB JSON file from GitHub on every execution. This is an external dependency on every daily run.

**Files affected**: `backend/lambdas/token-specs-collector/handler.py`

**Changes**:
1. Before fetching, check S3 for a cached copy with stored ETag
2. Send `If-None-Match` header with the stored ETag
3. If GitHub returns 304 (Not Modified), use the cached copy
4. If GitHub returns 200, update the cache and ETag in S3

**Expected benefit**: Eliminates external dependency on most runs. Faster execution. Resilient to GitHub outages (uses cached data).

**Success metric**: LiteLLM data fetched from GitHub only when content changes (typically weekly, not daily).

---

#### T3.6 — Separate Frontend Config from Backend Config

**Priority**: P3 | **Effort**: M (3-4 hours) | **Risk**: Medium | **Phase**: 3

**What**: Split `profiler-config.json` into backend-only and frontend-only sections.

**Addresses**: Over-complexity 8.16.3

**Current state**: 39% of `profiler-config.json` (436 of 1,131 lines) is frontend-only data (`provider_colors`, `region_coordinates`, `aws_regions`, `geo_region_options`, `geo_prefix_map`). The self-healing agent can modify this file, risking frontend breakage from a malformed agent update.

**Target state**: Two config files:
- `config/profiler-config.json` — Backend pipeline config (provider patterns, matching rules, agent config)
- `config/frontend-config.json` — Frontend display config (colors, coordinates, region labels)

**Risk**: Medium — requires updating the frontend to read from a different config path, and updating `copy-to-latest` to copy the frontend config.

**Success metric**: Self-healing agent cannot modify frontend display data. Each config file has a clear, single purpose.

---

### Track 4: Documentation & Consistency Cleanup

#### T4.1 — Update CLAUDE.md Lambda Count

**Priority**: P1 | **Effort**: S (5 min) | **Risk**: None | **Phase**: 1 (Quick Win)

**What**: Update `CLAUDE.md` to reflect 17 Lambdas instead of 11.

**Files affected**: `CLAUDE.md`

**Success metric**: CLAUDE.md accurately states 17 Lambda functions.

---

#### T4.2 — Add Missing `requirements.txt` Files

**Priority**: P3 | **Effort**: S (10 min) | **Risk**: None | **Phase**: 1 (Quick Win)

**What**: Add `requirements.txt` files to the 4 Lambdas that lack them.

**Addresses**: Redundancy 8.14

**Files to create**:
- `backend/lambdas/gap-detection/requirements.txt`
- `backend/lambdas/mantle-collector/requirements.txt`
- `backend/lambdas/region-discovery/requirements.txt`
- `backend/lambdas/self-healing-agent/requirements.txt`

**Content**: `# No external dependencies — uses Lambda runtime boto3 + shared layer`

**Success metric**: All 17 Lambda directories have a `requirements.txt` file.

---

#### T4.3 — Update `backend/lambdas/README.md` to Cover All 17 Lambdas

**Priority**: P2 | **Effort**: M (1-2 hours) | **Risk**: None | **Phase**: 2

**What**: The README currently documents 11 Lambdas. Add the missing 6: region-discovery, model-enricher, mantle-collector, gap-detection, self-healing-agent, analytics.

**Files affected**: `backend/lambdas/README.md`

**Success metric**: README documents all 17 Lambda input/output contracts.

---

#### T4.4 — Consistent Error Handling Decorator

**Priority**: P3 | **Effort**: M (2-3 hours) | **Risk**: Low | **Phase**: 3

**What**: Create a `@lambda_error_handler` decorator in the shared layer to replace the duplicated try/except boilerplate in all 16 pipeline Lambdas.

**Addresses**: Redundancy 8.11

**Files affected**:
- `backend/layers/common/python/shared/` — Add decorator
- All 16 pipeline Lambda handlers — Replace try/except with decorator

**Success metric**: Error response format defined in exactly 1 place. Adding a field (e.g., `timestamp`) to error responses requires changing 1 file.

---

### Track 5: Self-Healing Agent Roadmap (Backlog)

> **STATUS: BACKLOG — do NOT implement now.**
>
> The self-healing agent is a future capability. This section documents its current state, what works, what's missing, and a phased roadmap for making it production-ready. It is included here for planning purposes only.

#### 5.1 Current Implementation State

The self-healing mechanism spans 2 Lambdas totaling 765 lines:

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Gap Detection | `backend/lambdas/gap-detection/handler.py` | 347 | **Functional** — runs daily, produces gap reports |
| Self-Healing Agent | `backend/lambdas/self-healing-agent/handler.py` | 418 | **Partially functional** — invokes Claude, applies changes, but unverified |

**Evidence of prior activity**: The `profiler-config.json` version string is `"1.0.0-auto-updated-auto-updated"`, confirming the agent has run at least twice and auto-applied changes. Each run appends `-auto-updated` to the version (L270: `f"{version}-auto-updated"`). Last config update: `2026-02-04T16:29:46Z`.

**What works today**:
- Gap detection correctly identifies models without pricing, low-confidence matches, unknown providers, and new models
- Configurable thresholds for trigger decisions (from `agent_configuration.thresholds`)
- Claude Opus 4.5 invocation with structured prompt construction
- Safe/risky change classification (`safe_changes` vs `requires_review` lists)
- Auto-apply with config backup to `config/config-history/`
- State machine integration (conditional trigger via `CheckShouldTriggerAgent` choice state)

**What's missing (critical gaps)**:
1. **No regression detection** — If an auto-applied change makes the next run worse, there's no mechanism to detect or revert
2. **No validation of applied changes** — Changes are applied without testing them against sample data
3. **No healing history** — No record of what was changed, why, and whether it helped
4. **No rollback mechanism** — Backups exist but there's no automated rollback trigger
5. **No alerting** — Auto-applied changes happen silently; no notification to operators
6. **No feature flag** — The agent runs unconditionally when gaps are detected; cannot be disabled without code change
7. **Version string accumulation** — Each run appends `-auto-updated`, creating an ever-growing version string with no reset mechanism
8. **No integration testing** — The agent has never been tested with intentionally broken config
9. **Prompt refinement needed** — The Claude prompt needs tuning based on real-world gap patterns

#### 5.2 Immediate Recommendations (Before Backlog Work)

These are **gating changes** that should be applied before any further self-healing development:

| # | Change | Effort | Purpose |
|---|--------|--------|---------|
| 1 | Add a feature flag (`agent_configuration.enabled: false`) | S (30 min) | Disable auto-apply without code deployment |
| 2 | Fix version string accumulation | S (15 min) | Replace append logic with a proper version bump or timestamp |
| 3 | Add SNS notification on auto-apply | S (1 hour) | Operators know when config changes |

**Note**: These are small, safe changes that reduce risk from the existing implementation. They are NOT "implementing the self-healing agent" — they are gating an already-running system.

#### 5.3 Phased Roadmap to Production-Ready

| Phase | Focus | Effort | Prerequisites |
|-------|-------|--------|---------------|
| **Phase A: Gate & Monitor** | Feature flag, alerting, version fix | S (2 hours) | None |
| **Phase B: Validate & Rollback** | Pre-apply validation, automatic rollback on regression | L (8-12 hours) | Phase A |
| **Phase C: History & Learning** | Healing history tracking, outcome measurement, prompt refinement | L (8-12 hours) | Phase B |
| **Phase D: Expand Scope** | Code patch suggestions (not just config), documentation context loading | XL (16+ hours) | Phase C |

**Phase A — Gate & Monitor** (Effort: S):
- Add `agent_configuration.enabled` feature flag (default: `false`)
- Add SNS notification when auto-apply occurs
- Fix version string to use timestamp instead of append
- Add CloudWatch metric for agent invocation count and outcome

**Phase B — Validate & Rollback** (Effort: L):
- Before applying config changes, run a dry-run of pricing-linker with the new config against current data
- Compare match rates: if new config produces fewer matches or lower average confidence, reject the change
- Implement automatic rollback: if the next pipeline run's gap report is worse than the previous one, revert to the backup config
- Add S3 conditional writes (If-Match ETag) to prevent concurrent config modifications

**Phase C — History & Learning** (Effort: L):
- Create `agent/healing-history.json` tracking: what was detected, what was suggested, what was applied, what was the outcome
- Use history in future prompts: "In the past, adding pattern X for provider Y improved match rate by Z%"
- Add cooldown: don't re-analyze the same gaps within N runs
- Refine Claude prompt based on real-world patterns

**Phase D — Expand Scope** (Effort: XL):
- Load `BACKEND_ARCHITECTURE.md` as context for the agent
- Enable code patch suggestions (e.g., new normalization rules for pricing-linker)
- Add tool-use capabilities for the agent (validate_config, test_matching_rule)
- Consider multi-step reasoning for complex fixes

---

## Prioritized Roadmap

### Phase 1: Quick Wins (< 1 day total)

| # | Item | Track | Effort | Risk | Benefit |
|---|------|-------|--------|------|---------|
| 1 | T1.1 — Wire up existing config sections (6 values) | Track 1 | 2-3h | Low | **High** — eliminates 6 hard-coded values with zero config changes |
| 2 | T1.2 — Externalize critical API URLs (4 values) | Track 1 | 1-2h | Low | **High** — eliminates 4 critical/high risk values |
| 3 | T2.1 — Token-specs-collector shared layer adoption | Track 2 | 1h | Low | **High** — eliminates 4 redundancies in 1 Lambda |
| 4 | T2.2 — Regional-availability RETRY_CONFIG cleanup | Track 2 | 15min | Low | **Medium** — eliminates inconsistent timeout |
| 5 | T2.6 — Analytics Lambda `print()` → `logger` | Track 2 | 30min | Low | **Medium** — consistent logging |
| 6 | T2.7 — Delete legacy pricing-linker artifacts | Track 2 | 15min | Low | **Medium** — removes 987 lines of dead code |
| 7 | T4.1 — Update CLAUDE.md Lambda count | Track 4 | 5min | None | **Low** — documentation accuracy |
| 8 | T4.2 — Add missing `requirements.txt` files | Track 4 | 10min | None | **Low** — consistency |

**Total Phase 1 effort**: ~6-8 hours
**Total Phase 1 risk**: Low
**Total Phase 1 benefit**: 10 hard-coded values eliminated, 5 redundancies fixed, 987 dead lines removed

### Phase 2: Medium Effort (1-3 days total)

| # | Item | Track | Effort | Risk | Benefit |
|---|------|-------|--------|------|---------|
| 9 | T1.3 — Externalize high-priority values (4 groups) | Track 1 | 1-2h | Low | **High** — provider prefixes, API version, S3 paths configurable |
| 10 | T1.4 — Externalize medium-priority values (5 groups) | Track 1 | 1-2h | Low | **Medium** — temperature, user-agent, pricing patterns configurable |
| 11 | T1.5 — Add monitoring for keep-as-is values | Track 1 | 3-4h | Low | **High** — silent failures become visible |
| 12 | T2.3 — Provider prefix list consolidation | Track 2 | 2-3h | Medium | **High** — fixes out-of-sync lists, new providers need only config update |
| 13 | T2.8 — Duplicate documentation link logic | Track 2 | 30min | Low | **Low** — single source of truth |
| 14 | T2.9 — Mantle endpoint deduplication | Track 2 | 15min | Low | **Low** — resolved by T1.2 |
| 15 | T3.1 — Config validation on startup | Track 3 | 3-4h | Low | **High** — catches config issues before silent failures |
| 16 | T3.2 — Dynamic region handling in ASL | Track 3 | 2-3h | Medium | **High** — zero-change region scaling |
| 17 | T3.4 — Parallel S3 reads in final aggregator | Track 3 | 2-3h | Low | **Medium** — 50-70% faster aggregation |
| 18 | T3.5 — LiteLLM data caching with ETag | Track 3 | 1-2h | Low | **Medium** — eliminates daily external dependency |
| 19 | T4.3 — Update README to cover all 17 Lambdas | Track 4 | 1-2h | None | **Low** — documentation completeness |

**Total Phase 2 effort**: ~18-28 hours (2-3 days)
**Total Phase 2 risk**: Low-Medium
**Total Phase 2 benefit**: All hard-coded values externalized, monitoring in place, config validation, dynamic regions, performance improvements

### Phase 3: Larger Refactors (3-5 days total)

| # | Item | Track | Effort | Risk | Benefit |
|---|------|-------|--------|------|---------|
| 20 | T2.4 — Final aggregator decomposition | Track 2 | 8-16h | High | **High** — monolith → testable modules |
| 21 | T2.5 — Pricing aggregator function decomposition | Track 2 | 4-6h | Medium | **Medium** — God function → focused functions |
| 22 | T3.3 — Declarative pricing classification | Track 3 | 4-6h | Medium | **Medium** — new pricing types via config |
| 23 | T3.6 — Separate frontend/backend config | Track 3 | 3-4h | Medium | **Medium** — reduces self-healing agent risk |
| 24 | T4.4 — Consistent error handling decorator | Track 4 | 2-3h | Low | **Low** — DRY error handling |

**Total Phase 3 effort**: ~21-35 hours (3-5 days)
**Total Phase 3 risk**: Medium-High
**Total Phase 3 benefit**: Major maintainability improvements, testable architecture

### Phase 4: Backlog Items

| # | Item | Track | Effort | Risk | Benefit |
|---|------|-------|--------|------|---------|
| 25 | Self-healing Phase A — Gate & Monitor | Track 5 | 2h | Low | **Medium** — safety controls |
| 26 | Self-healing Phase B — Validate & Rollback | Track 5 | 8-12h | Medium | **High** — safe auto-healing |
| 27 | Self-healing Phase C — History & Learning | Track 5 | 8-12h | Medium | **Medium** — intelligent healing |
| 28 | Self-healing Phase D — Expand Scope | Track 5 | 16+h | High | **High** — autonomous maintenance |
| 29 | Enhancement 9.1 — Collect deprecation dates | Track 3 | S | Low | **Medium** — new data point |
| 30 | Enhancement 9.1 — Collect prompt caching support | Track 3 | S | Low | **Medium** — new data point |
| 31 | Enhancement 9.4 — Circuit breaker for external APIs | Track 3 | M | Low | **Medium** — resilience |
| 32 | Enhancement 9.4 — Inter-wave data validation | Track 3 | M | Low | **High** — catches corrupt data early |
| 33 | Enhancement 9.5 — Unified model ID normalization | Track 3 | M | Medium | **High** — prevents matching bugs |
| 34 | Enhancement 9.5 — Unified provider canonicalization | Track 3 | M | Medium | **High** — consistent provider names |
| 35 | Enhancement 9.6 — Structured logging with correlation IDs | Track 4 | M | Low | **Medium** — easier debugging |
| 36 | Enhancement 9.6 — CloudWatch custom metrics dashboard | Track 4 | M | Low | **Medium** — pipeline health visibility |

---

## Risk Register

Risks of **making** the changes (not risks of the current state):

| Risk | Likelihood | Impact | Mitigation | Affected Items |
|------|-----------|--------|------------|----------------|
| **Config loading failure breaks Lambdas** | Low | High | All config accessors have fallback defaults matching current hard-coded values. If S3 config is unavailable, behavior is identical to today. | T1.1, T1.2, T1.3, T1.4 |
| **Final aggregator decomposition introduces matching regression** | Medium | High | Byte-for-byte comparison of pipeline output before and after refactor. Run full pipeline in staging before production. | T2.4 |
| **Provider prefix consolidation changes matching behavior** | Medium | Medium | Compare quota match rates and availability match rates before/after. Run with both old and new logic in parallel for 1 week. | T2.3 |
| **ASL region changes cause Map state failures** | Low | High | Test with `sam local` first. Deploy to staging. Keep hard-coded fallback in ASL as a commented-out backup. | T3.2 |
| **Pricing classification rule engine has edge cases** | Medium | Medium | Run new rule engine against historical pricing data (last 30 days). Compare classification results with current if/elif output. | T3.3 |
| **Deployment during daily 6 AM UTC run** | Low | Medium | Deploy changes outside the 5:30-7:00 AM UTC window. Use SAM `--no-execute-changeset` to preview before applying. | All |
| **Self-healing agent feature flag breaks existing behavior** | Low | Low | Feature flag defaults to current behavior (enabled). Only changes behavior when explicitly set to `false`. | Track 5 Phase A |

---

## Dependencies Between Changes

```
T1.1 (Wire up config) ──────────────────────────────────────── No dependencies
T1.2 (Critical URLs) ───────────────────────────────────────── No dependencies
T2.1 (Token-specs shared layer) ─────────────────────────────── No dependencies
T2.2 (Regional-avail RETRY_CONFIG) ──────────────────────────── No dependencies
T2.6 (Analytics print→logger) ───────────────────────────────── No dependencies
T2.7 (Delete legacy artifacts) ──────────────────────────────── No dependencies
T4.1 (Update CLAUDE.md) ─────────────────────────────────────── No dependencies
T4.2 (Add requirements.txt) ─────────────────────────────────── No dependencies

T1.3 (High-priority values) ─────────── Depends on: T1.2 (config structure established)
T1.4 (Medium-priority values) ──────── Depends on: T1.2 (config structure established)
T2.3 (Provider prefix consolidation) ── Depends on: T1.3 (provider_prefixes in config)
T2.8 (Doc link dedup) ──────────────── Depends on: T1.1 (config loader pattern)
T2.9 (Mantle dedup) ────────────────── Resolved by: T1.2

T3.1 (Config validation) ───────────── Depends on: T1.2 (config structure finalized)
T3.2 (Dynamic ASL regions) ─────────── No dependencies (but test after T1.2)
T3.4 (Parallel S3 reads) ───────────── No dependencies (but easier after T2.4)
T3.5 (LiteLLM caching) ─────────────── Depends on: T2.1 (token-specs uses shared layer)

T2.4 (Final aggregator decomposition) ── Depends on: T2.3 (provider prefixes consolidated)
T2.5 (Pricing aggregator decomposition) ── No dependencies
T3.3 (Declarative pricing) ─────────── Depends on: T2.5 (functions decomposed first)
T3.6 (Separate frontend config) ────── Depends on: T1.2 (config structure established)
T4.4 (Error handling decorator) ────── No dependencies

Track 5 (Self-healing) ─────────────── Depends on: T1.1, T1.2 (config externalization complete)
```

**Critical path**: T1.2 → T1.3 → T2.3 → T2.4 (config externalization → provider consolidation → monolith decomposition)

---

## Success Metrics

| Metric | Current State | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|--------------|---------------|---------------|---------------|
| Hard-coded values requiring code deploy to change | 18 (recommended for externalization) | 8 | 0 | 0 |
| Config sections that exist but are ignored by code | 6 | 0 | 0 | 0 |
| Duplicate RETRY_CONFIG definitions | 3 | 1 | 1 | 1 |
| Duplicate S3 helper implementations | 2 (shared + token-specs) | 1 | 1 | 1 |
| Provider prefix list locations | 3 (out of sync) | 3 | 1 | 1 |
| Largest Lambda (lines) | 1,365 (final-aggregator) | 1,365 | 1,365 | < 1,000 |
| Dead code lines | 987 (legacy artifacts) | 0 | 0 | 0 |
| Lambdas using `print()` for logging | 1 (analytics) | 0 | 0 | 0 |
| CloudWatch pipeline health metrics | 0 | 0 | 6+ | 6+ |
| Config validation on startup | No | No | Yes | Yes |
| New provider requires code change | Yes | Partially | No | No |
| New region requires ASL change | Yes | Yes | No | No |

---

## Appendix: Change Item Index

Quick reference for all 36 change items:

| # | ID | Description | Phase | Track | Effort |
|---|-----|-------------|-------|-------|--------|
| 1 | T1.1 | Wire up existing config sections | 1 | Track 1 | S |
| 2 | T1.2 | Externalize critical API URLs | 1 | Track 1 | S |
| 3 | T1.3 | Externalize high-priority values | 2 | Track 1 | S |
| 4 | T1.4 | Externalize medium-priority values | 2 | Track 1 | S |
| 5 | T1.5 | Add monitoring for keep-as-is values | 2 | Track 1 | M |
| 6 | T2.1 | Token-specs-collector shared layer adoption | 1 | Track 2 | S |
| 7 | T2.2 | Regional-availability RETRY_CONFIG cleanup | 1 | Track 2 | S |
| 8 | T2.3 | Provider prefix list consolidation | 2 | Track 2 | M |
| 9 | T2.4 | Final aggregator decomposition | 3 | Track 2 | XL |
| 10 | T2.5 | Pricing aggregator function decomposition | 3 | Track 2 | L |
| 11 | T2.6 | Analytics Lambda print→logger | 1 | Track 2 | S |
| 12 | T2.7 | Delete legacy pricing-linker artifacts | 1 | Track 2 | S |
| 13 | T2.8 | Duplicate documentation link logic | 2 | Track 2 | S |
| 14 | T2.9 | Mantle endpoint deduplication | 2 | Track 2 | S |
| 15 | T2.10 | (Reserved — config loader init pattern, P3) | 4 | Track 2 | S |
| 16 | T3.1 | Config validation on startup | 2 | Track 3 | M |
| 17 | T3.2 | Dynamic region handling in ASL | 2 | Track 3 | M |
| 18 | T3.3 | Declarative pricing classification | 3 | Track 3 | L |
| 19 | T3.4 | Parallel S3 reads in final aggregator | 2 | Track 3 | M |
| 20 | T3.5 | LiteLLM data caching with ETag | 2 | Track 3 | S |
| 21 | T3.6 | Separate frontend/backend config | 3 | Track 3 | M |
| 22 | T4.1 | Update CLAUDE.md Lambda count | 1 | Track 4 | S |
| 23 | T4.2 | Add missing requirements.txt files | 1 | Track 4 | S |
| 24 | T4.3 | Update README to cover all 17 Lambdas | 2 | Track 4 | M |
| 25 | T4.4 | Consistent error handling decorator | 3 | Track 4 | M |
| 26 | SH-A | Self-healing: Gate & Monitor | 4 | Track 5 | S |
| 27 | SH-B | Self-healing: Validate & Rollback | 4 | Track 5 | L |
| 28 | SH-C | Self-healing: History & Learning | 4 | Track 5 | L |
| 29 | SH-D | Self-healing: Expand Scope | 4 | Track 5 | XL |
| 30 | E-9.1a | Collect deprecation dates | 4 | Track 3 | S |
| 31 | E-9.1b | Collect prompt caching support | 4 | Track 3 | S |
| 32 | E-9.4a | Circuit breaker for external APIs | 4 | Track 3 | M |
| 33 | E-9.4b | Inter-wave data validation | 4 | Track 3 | M |
| 34 | E-9.5a | Unified model ID normalization | 4 | Track 3 | M |
| 35 | E-9.5b | Unified provider canonicalization | 4 | Track 3 | M |
| 36 | E-9.6 | Structured logging + CloudWatch dashboard | 4 | Track 4 | M |
