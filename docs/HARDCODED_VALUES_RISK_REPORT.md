# Hard-Coded Values Risk Report

> **Generated**: 2026-02-19  
> **Scope**: All 17 Lambda handlers, shared layer, ASL workflow, profiler config defaults  
> **Source**: Deep code audit (Task 01) + direct code inspection

---

## Executive Summary

| Metric | Count |
|--------|-------|
| **Total hard-coded values assessed** | 62 |
| **Critical risk** | 4 |
| **High risk** | 12 |
| **Medium risk** | 22 |
| **Low risk** | 24 |
| **Recommended for externalization** | 18 |
| **Recommended for monitoring** | 10 |
| **Keep as-is** | 34 |

The highest-risk hard-coded values are **external API URLs** (LiteLLM GitHub, Mantle endpoint pattern, Bulk Pricing URL) and **fallback region lists** that will silently produce incomplete data when AWS adds new regions. Three Lambdas duplicate the shared layer's retry configuration with different values, creating inconsistent timeout behavior that is invisible to operators.

---

## Top 10 Highest-Risk Values

| # | Value | Location | Risk | Breakage Scenario | Recommendation |
|---|-------|----------|------|-------------------|----------------|
| 1 | `LITELLM_MODEL_DB_URL` — `https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json` | `token-specs-collector/handler.py:29` | **CRITICAL** | BerriAI renames the repo, moves the file, changes the JSON schema, or GitHub rate-limits the Lambda. Token specs collection silently returns empty data — no models get context window or max output token information. The frontend shows "N/A" for all token specs. | **Externalize** to `profiler-config.json` → `external_apis.litellm_url`. Add schema validation on the response. Add CloudWatch alarm on `modelsWithSpecs == 0`. |
| 2 | `MANTLE_ENDPOINT_PATTERN` — `"bedrock-mantle.{region}.api.aws"` | `mantle-collector/handler.py:30`, `final-aggregator/handler.py:228` | **CRITICAL** | AWS changes the Mantle API hostname pattern (e.g., migrates to `bedrock.{region}.api.aws/mantle/v1`). All Mantle data collection fails across every region. Mantle model data disappears from output. Duplicated in two files — one could be updated while the other is missed. | **Externalize** to `profiler-config.json` → `external_apis.mantle_endpoint_pattern`. Remove the duplicate in `final-aggregator` (it only uses it for metadata display). |
| 3 | Fallback region list (16 regions) | `region-discovery/handler.py:38-43, 136-141` | **HIGH** | The `ec2.describe_regions` call fails (IAM permission removed, transient error). The Lambda falls back to a stale 16-region list. AWS has added regions since this list was written (e.g., `ap-southeast-3`, `me-south-1`, `il-central-1`). Pipeline runs with incomplete region coverage — models in new regions are invisible. Returns `status: SUCCESS` even on fallback, so no alarm fires. | **Externalize** to `profiler-config.json` → `region_configuration.fallback_regions`. Add a CloudWatch metric when fallback is used. Consider returning `status: PARTIAL` instead of `SUCCESS`. |
| 4 | `BULK_PRICING_URL` — `"https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/{service_code}/current/{region}/index.json"` | `pricing-collector/handler.py:38` | **HIGH** | AWS changes the Bulk Pricing API URL structure (has happened before — the `/offers/v1.0/` prefix is versioned). Pricing collection loses coverage for models not in the GetProducts API (e.g., Stability AI). Silent failure — the Lambda logs a warning but returns `SUCCESS` with partial data. | **Externalize** to `profiler-config.json` → `external_apis.bulk_pricing_url_template`. Add a CloudWatch alarm when `bulkApiCount == 0` for service codes that previously had bulk data. |
| 5 | Duplicate `RETRY_CONFIG` in `regional-availability` — `connect_timeout=5` (vs shared layer's `10`) | `regional-availability/handler.py:47-51` | **HIGH** | The lower `connect_timeout=5` causes premature timeouts in regions with higher latency (e.g., `ap-southeast-3`, `me-south-1`). Regional availability data is silently incomplete for slow regions. Operators see no error because the Lambda catches exceptions per-region and continues. The inconsistency is invisible — both configs look similar but behave differently. | **Remove** the local `RETRY_CONFIG` and import from shared layer. If the lower timeout is intentional, document why and add a comment. |
| 6 | Duplicate `RETRY_CONFIG` + S3 helpers in `token-specs-collector` | `token-specs-collector/handler.py:22-26, 32-50` | **HIGH** | Bug fixes or improvements to the shared layer's `read_from_s3`/`write_to_s3` (e.g., adding retry logic, better error messages, content-type handling) are not picked up by `token-specs-collector`. The Lambda silently uses its own inferior implementations. If the shared layer adds compression or encryption, this Lambda breaks. | **Refactor** to import from shared layer: `from shared import RETRY_CONFIG, get_s3_client, read_from_s3, write_to_s3, validate_required_params, ValidationError`. |
| 7 | `x-console-consumer: true` header | `model-extractor/handler.py:186` | **HIGH** | AWS changes or removes the console metadata API behavior. This is an undocumented internal API — there is no SLA or deprecation notice. If the header is ignored, context windows, descriptions, languages, and use cases are lost for ~53 models. The Lambda gracefully degrades (returns empty metadata), but the data loss is silent. | **Keep as-is** (cannot externalize an API contract), but add monitoring: CloudWatch alarm when `console_metadata` attachment count drops below a threshold (e.g., < 30 models). Document the risk in a runbook. |
| 8 | `anthropic_version: 'bedrock-2023-05-31'` | `self-healing-agent/handler.py:162` | **HIGH** | Anthropic deprecates this API version. The self-healing agent's `invoke_model` call fails or returns unexpected response format. The agent cannot analyze gaps or apply fixes. The pipeline continues but gap detection becomes a no-op. | **Externalize** to `profiler-config.json` → `agent_configuration.anthropic_version`. |
| 9 | `User-Agent: BedrockProfiler/1.0` | `token-specs-collector/handler.py:57` | **MEDIUM** | GitHub or a CDN blocks this User-Agent string (rate limiting, bot detection). LiteLLM data fetch fails silently. Low likelihood but non-zero — GitHub has increasingly aggressive bot detection. | **Externalize** to `profiler-config.json` → `external_apis.user_agent`. |
| 10 | S3 key patterns: `latest/bedrock_models.json`, `latest/bedrock_pricing.json`, `latest/manifest.json` | `copy-to-latest/handler.py:149-150, 184` | **HIGH** | Frontend code, CloudFront behaviors, or monitoring tools depend on these exact paths. If changed in one place but not others, the frontend breaks or serves stale data. The paths are duplicated across `copy-to-latest`, `gap-detection` (L262: `latest/bedrock_models.json`), and the frontend's `dataSource.js`. | **Externalize** to `profiler-config.json` → `s3_paths.latest_models`, `s3_paths.latest_pricing`, `s3_paths.latest_manifest`. Reference from all consumers. |

---

## Complete Inventory by Category

### 1. External URLs & Endpoints

| # | Value | Location | Likelihood | Impact | Detection | Mitigation | Recommendation |
|---|-------|----------|-----------|--------|-----------|------------|----------------|
| 1 | `https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json` | `token-specs-collector:29` | **High** — third-party repo, no SLA | **High** — all token specs lost | Silent — Lambda returns SUCCESS with 0 specs | None — no fallback, no alarm | **Externalize** to config; add response validation + alarm |
| 2 | `"bedrock-mantle.{region}.api.aws"` | `mantle-collector:30` | **Medium** — AWS internal API | **High** — all Mantle data lost | Visible — per-region FAILED status | Per-region error handling | **Externalize** to config; deduplicate (also in final-aggregator:228) |
| 3 | `"bedrock-mantle.{region}.api.aws"` (duplicate) | `final-aggregator:228` | **Medium** | **Low** — only used in metadata display | Silent | None | **Remove** duplicate; reference from config or pass from mantle-collector |
| 4 | `https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/{service_code}/current/{region}/index.json` | `pricing-collector:38` | **Medium** — AWS public API, versioned URL | **Medium** — loses bulk pricing coverage | Warning logged, returns SUCCESS | Graceful degradation (returns empty list) | **Externalize** to config |
| 5 | `https://bedrock.{region}.amazonaws.com/foundation-models` | `model-extractor:182` | **Low** — standard AWS endpoint pattern | **Medium** — console metadata lost | Warning logged | Graceful degradation | **Keep as-is** — standard AWS endpoint pattern |
| 6 | `https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html` | `final-aggregator:912` | **Low** — AWS docs URL | **Low** — wrong link in output | Silent | None | **Keep as-is** — low change velocity |
| 7 | `https://aws.amazon.com/bedrock/pricing/` | `final-aggregator:915` | **Low** — AWS marketing URL | **Low** — wrong link in output | Silent | None | **Keep as-is** — low change velocity |
| 8 | `User-Agent: BedrockProfiler/1.0` | `token-specs-collector:57` | **Low** | **Medium** — LiteLLM fetch blocked | Silent | None | **Externalize** to config |

### 2. Region Lists

| # | Value | Location | Likelihood | Impact | Detection | Mitigation | Recommendation |
|---|-------|----------|-----------|--------|-----------|------------|----------------|
| 9 | 16 fallback regions (L38-43) | `region-discovery:38-43` | **High** — AWS adds ~2-4 regions/year | **High** — new regions invisible on fallback | Silent — returns SUCCESS | Fallback only on API failure | **Externalize** to config; add metric on fallback use |
| 10 | 16 fallback regions (L136-141) — duplicate of above | `region-discovery:136-141` | **High** | **High** | Silent | Same as above | **Deduplicate** — define once as module constant, then externalize |
| 11 | `model_regions: ["us-east-1", "us-west-2"]` | `config_loader.py:97` (DEFAULT_CONFIG) | **Medium** — only used when S3 config unavailable | **High** — only 2 regions scanned for models | Silent | S3 config has full list | **Keep as-is** — fallback only; S3 config is authoritative |
| 12 | `quota_regions` (16 regions) | `config_loader.py:98-104` (DEFAULT_CONFIG) | **Medium** | **Medium** — quota data incomplete | Silent | S3 config has full list | **Keep as-is** — fallback only |
| 13 | `feature_regions` (16 regions) | `config_loader.py:105-111` (DEFAULT_CONFIG) | **Medium** | **Medium** — feature data incomplete | Silent | S3 config has full list | **Keep as-is** — fallback only |
| 14 | `region_locations` (4 regions only) | `config_loader.py:112-117` (DEFAULT_CONFIG) | **Medium** | **Low** — display names missing for most regions | Silent | S3 config has full list | **Keep as-is** — fallback only |

### 3. Retry & Timeout Configurations

| # | Value | Location | Likelihood | Impact | Detection | Mitigation | Recommendation |
|---|-------|----------|-----------|--------|-----------|------------|----------------|
| 15 | Shared `RETRY_CONFIG`: `max_attempts=3, mode=adaptive, connect_timeout=10, read_timeout=30` | `shared/config.py:11-15` | **Low** | **Low** — canonical config | N/A | N/A | **Keep as-is** — this is the authoritative config |
| 16 | `regional-availability` `RETRY_CONFIG`: `connect_timeout=5, read_timeout=30` | `regional-availability:47-51` | **Medium** — inconsistency causes confusion | **Medium** — premature timeouts in slow regions | Silent — per-region errors caught | Per-region error handling | **Remove** — use shared layer's RETRY_CONFIG |
| 17 | `token-specs-collector` `RETRY_CONFIG`: `connect_timeout=10, read_timeout=30` | `token-specs-collector:22-26` | **Medium** — inconsistency | **Low** — values happen to match shared layer | Silent | None | **Remove** — use shared layer's RETRY_CONFIG |
| 18 | `timeout=60` (Bulk Pricing API) | `pricing-collector:69` | **Low** | **Low** — appropriate for large JSON download | Silent on timeout | Returns empty list | **Keep as-is** |
| 19 | `timeout=30` (Console metadata) | `model-extractor:194` | **Low** | **Low** — appropriate for API call | Silent on timeout | Returns empty dict | **Keep as-is** |
| 20 | `timeout=30` (LiteLLM fetch) | `token-specs-collector:61` | **Low** | **Medium** — LiteLLM JSON is large | Silent on timeout | Returns empty dict | **Keep as-is** |
| 21 | `REQUEST_TIMEOUT_SECONDS = 10` | `mantle-collector:31` | **Low** | **Low** — appropriate for per-region API call | Visible — error returned | Error handling present | **Keep as-is** |
| 22 | `time.sleep(0.5)` (throttle pause) | `pricing-collector:162` | **Low** | **Low** — rate limiting | N/A | N/A | **Keep as-is** |
| 23 | `time.sleep(2)` (throttle backoff) | `pricing-collector:168` | **Low** | **Low** — throttle recovery | N/A | N/A | **Keep as-is** |

### 4. API Parameters & Pagination Limits

| # | Value | Location | Likelihood | Impact | Detection | Mitigation | Recommendation |
|---|-------|----------|-----------|--------|-----------|------------|----------------|
| 24 | `maxResults=1` (Bedrock availability check) | `region-discovery:52` | **Low** — only checking existence | **None** — intentionally minimal | N/A | N/A | **Keep as-is** — intentional |
| 25 | `MaxResults=100` (Pricing API pagination) | `pricing-collector:134` | **Low** — AWS API max page size | **None** — standard pagination | N/A | Pagination loop handles it | **Keep as-is** |
| 26 | `FormatVersion='aws_v1'` | `pricing-collector:135` | **Low** — AWS API parameter | **High** if AWS deprecates v1 | Error from API | None | **Monitor** — add alarm on API errors |
| 27 | `max_batches=100` (safety limit) | `pricing-collector:126` | **Low** | **Medium** — could truncate if >10,000 products | Silent truncation | None | **Externalize** to config or increase to 500 |
| 28 | `MaxResults=100` (Service Quotas pagination) | `quota-collector:52` | **Low** | **None** — standard pagination | N/A | Pagination loop | **Keep as-is** |
| 29 | `SERVICE_CODE = 'bedrock'` | `quota-collector:27` | **Low** — AWS service code is stable | **High** if changed | API error | None | **Keep as-is** — AWS service codes are immutable |
| 30 | `max_workers=20` (region discovery) | `region-discovery:77` | **Low** | **Low** — concurrency limit | N/A | N/A | **Keep as-is** |
| 31 | `max_workers=10` (regional availability) | `regional-availability:94` | **Low** | **Low** — concurrency limit | N/A | N/A | **Keep as-is** |

### 5. S3 Key Patterns

| # | Value | Location | Likelihood | Impact | Detection | Mitigation | Recommendation |
|---|-------|----------|-----------|--------|-----------|------------|----------------|
| 32 | `latest/bedrock_models.json` | `copy-to-latest:149` | **Medium** — cross-system dependency | **High** — frontend breaks | Visible — frontend shows no data | None | **Externalize** to config |
| 33 | `latest/bedrock_pricing.json` | `copy-to-latest:150` | **Medium** | **High** — frontend breaks | Visible | None | **Externalize** to config |
| 34 | `latest/manifest.json` | `copy-to-latest:184` | **Low** | **Low** — manifest is informational | Silent | None | **Keep as-is** |
| 35 | `latest/bedrock_models.json` (previous run reference) | `gap-detection:262` | **Medium** — must match copy-to-latest | **Medium** — new model detection fails | Silent — returns empty new models list | `default_on_missing={}` | **Externalize** to config (same key as #32) |
| 36 | `config/profiler-config.json` | `self-healing-agent:273`, `config_loader.py:157` | **Low** — internal convention | **High** if mismatched | Error on config load | Fallback to DEFAULT_CONFIG | **Keep as-is** — but ensure single source of truth |
| 37 | `config/config-history/profiler-config.{timestamp}.json` | `self-healing-agent:278` | **Low** | **Low** — backup path | N/A | N/A | **Keep as-is** |

### 6. Business Logic Constants

| # | Value | Location | Likelihood | Impact | Detection | Mitigation | Recommendation |
|---|-------|----------|-----------|--------|-----------|------------|----------------|
| 38 | Size category thresholds: `≥128K=Large, ≥32K=Medium, <32K=Small` | `final-aggregator:121-126` | **Medium** — model context windows are growing | **Low** — display categorization only | Silent | None | **Externalize** to `profiler-config.json` → `model_configuration.context_window_thresholds` (section already exists in S3 config) |
| 39 | Size category colors: `#10B981, #3B82F6, #F59E0B, #6B7280` | `final-aggregator:120-126` | **Low** | **Low** — display only | Silent | None | **Keep as-is** — or move to frontend config |
| 40 | Consumption option order: `on_demand, batch, cross_region_inference, mantle, provisioned_throughput` | `final-aggregator:584-590` | **Low** | **Low** — display ordering | Silent | None | **Keep as-is** |
| 41 | `size_variance_threshold = 0.3` (30% size mismatch) | `pricing-linker:225` | **Low** | **Medium** — false positive/negative matches | Silent — wrong pricing linked | Config has this value | **Already externalized** — loaded from config via `get_size_variance_threshold()`. The hard-coded `0.3` in the conflict detection function should reference config instead. |
| 42 | `min_confidence_threshold = 0.7` (DEFAULT_CONFIG) | `config_loader.py:130` | **Low** — fallback only | **Medium** — affects pricing match quality | Silent | S3 config is authoritative | **Keep as-is** — fallback default |
| 43 | `low_confidence_threshold = 0.6` (DEFAULT_CONFIG) | `config_loader.py:137` | **Low** — fallback only | **Medium** — affects agent trigger sensitivity | Silent | S3 config is authoritative | **Keep as-is** — fallback default |
| 44 | `unmatched_models_trigger = 5` (DEFAULT_CONFIG) | `config_loader.py:138` | **Low** — fallback only | **Low** — affects agent trigger | Silent | S3 config is authoritative | **Keep as-is** — fallback default |
| 45 | `temperature: 0.2` (Claude invocation) | `self-healing-agent:170` | **Low** | **Low** — affects response consistency | N/A | N/A | **Externalize** to `profiler-config.json` → `agent_configuration.temperature` |

### 7. String Patterns & Detection Logic

| # | Value | Location | Likelihood | Impact | Detection | Mitigation | Recommendation |
|---|-------|----------|-----------|--------|-----------|------------|----------------|
| 46 | Pricing type detection strings (token, image, video, model_unit, search_unit, etc.) | `pricing-aggregator:69-173` | **Medium** — AWS adds new pricing types | **Medium** — new pricing types miscategorized | Silent — defaults to "token" | Default fallback | **Monitor** — add logging when default is used; consider externalizing type patterns |
| 47 | Pricing group detection strings (Batch, Global, Long Context, Provisioned, Custom) | `pricing-aggregator:176-227` | **Medium** — AWS adds new pricing groups | **Medium** — new groups miscategorized | Silent — defaults to "On-Demand" | Default fallback | **Monitor** — add logging when unexpected patterns found |
| 48 | Suffixes to remove: `['(Amazon Bedrock Edition)', '(Amazon Bedrock)', ...]` | `pricing-aggregator:243-248` | **Low** | **Low** — cosmetic name cleaning | Silent | None | **Keep as-is** |
| 49 | Suffixes to remove: `['-it', '-instruct', '-chat', '-v1', '-v2', '-v3', ':0', ':1', ':2']` | `pricing-linker:248` | **Medium** — new model naming conventions | **Medium** — matching failures | Silent — lower confidence scores | Config has `suffixes_to_remove` | **Already externalized** in config — but this instance is hard-coded. Should reference `config.get_suffixes_to_remove()`. |
| 50 | Type conflict groups (embed vs generation, rerank vs chat, etc.) | `pricing-linker:200-204` | **Low** | **Medium** — false matches if new model types emerge | Silent | None | **Externalize** to `profiler-config.json` → `matching_configuration.type_conflicts` (accessor already exists) |
| 51 | Context window regex: `r':\d+k$'` and `r':(\d+)k$'` | `model-merger:39, 53` | **Low** — AWS naming convention | **Low** — variant dedup fails | Silent | None | **Keep as-is** — tied to AWS model ID format |
| 52 | `_PROVIDER_PREFIXES` list (18 entries) | `final-aggregator:282-303` | **Medium** — new providers added | **Medium** — quota matching fails for new providers | Silent — quotas not linked | None | **Externalize** to `profiler-config.json` → `provider_configuration.provider_prefixes` |
| 53 | Capability inference keywords (code, reasoning, function_calling terms) | `model-enricher:83-92` | **Medium** — new model capabilities emerge | **Medium** — capabilities not detected | Silent | None | **Externalize** to `profiler-config.json` → `model_configuration.capability_keywords` (section already exists in S3 config) |
| 54 | Capability-to-use-case mapping (13 entries) | `model-enricher:105-119` | **Medium** | **Low** — use cases are informational | Silent | None | **Externalize** to config for easier updates |
| 55 | `VALID_EVENT_TYPES` (7 event types) | `analytics:29-33` | **Low** — controlled by frontend | **Low** — new events silently dropped | Silent | None | **Keep as-is** — tightly coupled to frontend |
| 56 | `version: 'v2-port-features'` | `pricing-linker:503` | **Low** | **None** — metadata label | N/A | N/A | **Keep as-is** |

### 8. Analytics & Operational Constants

| # | Value | Location | Likelihood | Impact | Detection | Mitigation | Recommendation |
|---|-------|----------|-----------|--------|-----------|------------|----------------|
| 57 | `SESSION_BUCKET_TTL_DAYS = 7` | `analytics:36` | **Low** | **Low** — DynamoDB TTL | N/A | N/A | **Keep as-is** |
| 58 | 5-minute session bucket window | `analytics:353` | **Low** | **Low** — aggregation granularity | N/A | N/A | **Keep as-is** |
| 59 | `max 50 events per batch` | `analytics:68` | **Low** | **Low** — rate limiting | Visible — 400 error returned | Error response | **Keep as-is** |
| 60 | `max 365 days query` | `analytics:155` | **Low** | **Low** — query limit | Silent — capped | None | **Keep as-is** |
| 61 | `top 20 items` in `_top()` | `analytics:271` | **Low** | **Low** — display limit | N/A | N/A | **Keep as-is** |
| 62 | `ANALYTICS_TABLE` default: `bedrock-profiler-analytics-dev` | `analytics:20` | **Low** — env var overrides | **High** if env var missing in prod | Visible — DynamoDB errors | Env var override | **Monitor** — ensure env var is set in all deployments |

### 9. ASL Workflow Timeouts (Infrastructure-as-Code)

These values are in the Step Functions ASL definition and require redeployment to change. They are a different class of "hard-coded" — expected to be in IaC but worth documenting for operational awareness.

| # | Value | Location | Risk | Notes |
|---|-------|----------|------|-------|
| — | `TimeoutSeconds: 300` (pricing-collector) | `bedrock-profiler.asl.json:114` | **Low** | Appropriate for large API pagination |
| — | `TimeoutSeconds: 120` (model-extractor) | `bedrock-profiler.asl.json:148` | **Low** | Appropriate for per-region extraction |
| — | `TimeoutSeconds: 60` (quota-collector) | `bedrock-profiler.asl.json:193` | **Low** | Appropriate for per-region quota fetch |
| — | `TimeoutSeconds: 300` (pricing-linker) | `bedrock-profiler.asl.json:360` | **Low** | Appropriate for cross-referencing |
| — | `TimeoutSeconds: 120` (regional-availability) | `bedrock-profiler.asl.json:444` | **Low** | Appropriate for multi-region scan |
| — | `TimeoutSeconds: 120` (token-specs) | `bedrock-profiler.asl.json:470` | **Medium** | External HTTP call — could be slow |
| — | `TimeoutSeconds: 180` (final-aggregator) | `bedrock-profiler.asl.json:583` | **Low** | Largest Lambda — appropriate |
| — | `TimeoutSeconds: 120` (gap-detection) | `bedrock-profiler.asl.json:607` | **Low** | Appropriate |
| — | `TimeoutSeconds: 300` (self-healing-agent) | `bedrock-profiler.asl.json:656` | **Medium** | Claude invocation can be slow |
| — | `TimeoutSeconds: 60` (copy-to-latest) | `bedrock-profiler.asl.json:677` | **Low** | Simple S3 copy |
| — | `MaxConcurrency: 3` (pricing Wave1) | `bedrock-profiler.asl.json:75` | **Low** | 3 service codes in parallel |
| — | `MaxConcurrency: 2` (model-extractor) | `bedrock-profiler.asl.json:160` | **Low** | 2 regions in parallel |
| — | `MaxConcurrency: 10` (quota, feature, mantle maps) | `bedrock-profiler.asl.json:239,372,482` | **Low** | Appropriate for per-region parallelism |

> **Note**: ASL timeout values are not recommended for externalization. They are infrastructure configuration and belong in IaC. However, if model count or region count grows significantly, the `TimeoutSeconds` for `final-aggregator` and `self-healing-agent` may need to be increased.

---

## Recommendations Summary

### Externalize to `profiler-config.json` (18 values)

These values should be moved to the S3-hosted `profiler-config.json` for runtime configurability without code changes:

| Priority | Value | Target Config Path | Rationale |
|----------|-------|--------------------|-----------|
| **P0** | LiteLLM URL | `external_apis.litellm_url` | Third-party URL with no SLA; highest breakage risk |
| **P0** | Mantle endpoint pattern | `external_apis.mantle_endpoint_pattern` | AWS internal API; duplicated in 2 files |
| **P0** | Bulk Pricing URL template | `external_apis.bulk_pricing_url_template` | Versioned AWS URL that could change |
| **P0** | Fallback region lists | `region_configuration.fallback_regions` | AWS adds regions frequently; stale fallback is dangerous |
| **P1** | S3 latest paths (`latest/bedrock_models.json`, etc.) | `s3_paths.latest_models`, `s3_paths.latest_pricing` | Cross-system dependency (backend + frontend) |
| **P1** | `anthropic_version` | `agent_configuration.anthropic_version` | API version that will eventually be deprecated |
| **P1** | `_PROVIDER_PREFIXES` list | `provider_configuration.provider_prefixes` | New providers added regularly |
| **P1** | Capability keywords | `model_configuration.capability_keywords` | Already exists in S3 config but not used by code |
| **P2** | Capability-to-use-case mapping | `model_configuration.capability_use_case_mapping` | Informational but changes with new model types |
| **P2** | Type conflict groups | `matching_configuration.type_conflicts` | Accessor already exists in ConfigLoader |
| **P2** | Size category thresholds | `model_configuration.context_window_thresholds` | Already exists in S3 config but not used by code |
| **P2** | `temperature` | `agent_configuration.temperature` | Tuning parameter |
| **P2** | `User-Agent` string | `external_apis.user_agent` | Could be blocked by GitHub |
| **P2** | `max_batches` safety limit | `pricing_configuration.max_batches` | Could truncate large datasets |
| **P2** | Pricing type detection patterns | `pricing_configuration.pricing_type_patterns` | New AWS pricing types emerge |
| **P2** | Pricing group detection patterns | `pricing_configuration.pricing_group_patterns` | New AWS pricing groups emerge |
| **P2** | Suffixes to remove (pricing-linker L248) | Already in config — code should use `config.get_suffixes_to_remove()` | Inconsistency: config has it but code ignores it |
| **P2** | Size variance threshold (pricing-linker L225) | Already in config — code should use `config.get_size_variance_threshold()` | Inconsistency: config has it but code ignores it |

### Add Monitoring (10 values)

These values should stay in code but need operational monitoring to detect when they cause issues:

| Value | Monitoring Action |
|-------|-------------------|
| `x-console-consumer: true` header | CloudWatch alarm when console metadata attachment count < 30 |
| `FormatVersion='aws_v1'` | CloudWatch alarm on Pricing API errors |
| Pricing type detection defaults | Log metric when default "token" type is used for unknown patterns |
| Pricing group detection defaults | Log metric when "On-Demand" default is used for unknown patterns |
| `ANALYTICS_TABLE` default value | Deployment check: ensure env var is set in prod |
| Fallback region list usage | CloudWatch metric when `region-discovery` uses fallback |
| `token-specs-collector` empty results | CloudWatch alarm when `modelsWithSpecs == 0` |
| `mantle-collector` all-region failures | CloudWatch alarm when all Mantle regions fail |
| ASL `TimeoutSeconds` for final-aggregator | Monitor execution duration trends; alert at 80% of timeout |
| ASL `TimeoutSeconds` for self-healing-agent | Monitor execution duration trends; alert at 80% of timeout |

### Keep As-Is (34 values)

These values are appropriately hard-coded because they are:
- **Stable AWS conventions** (service codes, endpoint patterns, API parameters)
- **Intentional design choices** (concurrency limits, pagination sizes, sleep durations)
- **Fallback defaults** in `DEFAULT_CONFIG` that are overridden by S3 config at runtime
- **Infrastructure configuration** (ASL timeouts, MaxConcurrency) that belongs in IaC
- **Tightly coupled constants** (analytics event types, session TTL) that change with code

---

## Cross-Cutting Concerns

### 1. Shared Layer Duplication (3 Lambdas)

Three Lambdas bypass the shared layer, creating maintenance risk:

| Lambda | What's Duplicated | Risk |
|--------|-------------------|------|
| `token-specs-collector` | `RETRY_CONFIG`, `read_from_s3`, `write_to_s3` | Bug fixes to shared layer not applied; inconsistent error handling |
| `regional-availability` | `RETRY_CONFIG` (different `connect_timeout`) | Inconsistent timeout behavior across Lambdas |
| `analytics` | Everything — no shared layer usage at all | Completely standalone; uses `print()` instead of `logger` |

**Recommendation**: Refactor `token-specs-collector` and `regional-availability` to use the shared layer. The `analytics` Lambda is architecturally different (API Gateway, DynamoDB) and may intentionally be standalone, but should at minimum use `get_logger()` for consistent logging.

### 2. Config Values That Exist But Aren't Used

The S3 `profiler-config.json` already contains several sections that the code doesn't reference:

| Config Section | Exists in S3 Config | Used by Code |
|----------------|--------------------:|:-------------|
| `capability_keywords` | ✅ | ❌ — `model-enricher` has its own hard-coded keywords |
| `reasoning_models` | ✅ | ❌ — `model-enricher` uses hard-coded model ID patterns |
| `function_calling_models` | ✅ | ❌ — `model-enricher` uses hard-coded model ID patterns |
| `context_window_thresholds` | ✅ | ❌ — `final-aggregator` uses hard-coded `128K/32K` thresholds |
| `suffixes_to_remove` | ✅ | ⚠️ Partially — `pricing-linker` has its own hard-coded copy |
| `type_conflicts` | ✅ (accessor exists) | ❌ — `pricing-linker` has hard-coded conflict groups |

**Recommendation**: Wire up the existing config sections to the code that should be using them. This is the lowest-effort, highest-impact improvement — the config infrastructure already exists.

### 3. Self-Healing Agent Config Drift

The self-healing agent has modified `profiler-config.json` at least twice (version: `1.0.0-auto-updated-auto-updated`). Each run appends `-auto-updated` to the version string (L270). This means:
- The `DEFAULT_CONFIG` in `config_loader.py` is increasingly divergent from the S3 config
- If S3 config becomes unavailable, the fallback is a minimal subset of the actual config
- There is no mechanism to sync `DEFAULT_CONFIG` with the S3 config

**Recommendation**: Add a periodic check that compares `DEFAULT_CONFIG` keys with S3 config keys and alerts on significant divergence.
