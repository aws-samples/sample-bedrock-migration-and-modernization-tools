# Model Collector Sample 📊

This sample demonstrates how to programmatically collect comprehensive information about Amazon Bedrock foundation models using AWS APIs. It shows implementation patterns for model discovery, data collection, and organization that can be adapted for various model analysis and selection use cases.

## What this sample demonstrates

* 🔍 **Model Discovery Patterns** - How to systematically discover foundation models across 20+ AWS regions using Bedrock APIs
* 📋 **Comprehensive Data Collection** - Methods for gathering 25+ data points per model (specifications, capabilities, lifecycle status)
* 🌍 **Regional Analysis Implementation** - Techniques for mapping regional availability and cross-region inference capabilities
* 📊 **Quota Integration** - Approaches to collect and organize service quotas and rate limits for each model
* ⚡ **Token Specifications Discovery** - LiteLLM-first approach for discovering context windows and max output tokens with community-verified data

## Data Sources (AWS APIs)

### 🏗️ Amazon Bedrock APIs
- **`bedrock:ListFoundationModels`** - Complete catalog of available models
- **`bedrock:GetFoundationModel`** - Detailed specifications (context window, max tokens, etc.)
- **`bedrock:ListInferenceProfiles`** - Cross-region inference capabilities
- **`bedrock:GetInferenceProfile`** - Profile configurations and mappings
- **`bedrock:GetModelInvocationLoggingConfiguration`** - Model configuration details

### 📊 AWS Service Quotas API
- **`servicequotas:ListServiceQuotas`** - Rate limits and quotas (requests/min, tokens/min)

### 🧠 External Data Sources
- **[LiteLLM Database](https://github.com/BerriAI/litellm)** - Primary source for context windows and max output tokens (234+ Bedrock-compatible models)
- **Manual Corrections File** - Fallback corrections for missing or inaccurate specifications

## Running this Sample

### 🌐 **Via Web Interface (Recommended)**
1. Launch the main app: `python3 launch.py`
2. Open `http://localhost:8501` in your browser
3. Go to "⚙️ AWS Setup & Update" in sidebar
4. Select your AWS profile and click "🔄 Update models database"
5. Watch real-time progress as data is collected

### 💻 **Manual Execution**
For development or troubleshooting:
```bash
cd collectors/model-collector/src
python main.py --profile your-aws-profile
```

**Options:**
- `--profile` - AWS profile name
- `--region` - Override default region
- `--threads` - Collection threads (default: 4)
- `--verbose` - Detailed logging

## Data Output

**Main file:** `data/bedrock_models.json`

Sample model structure:
```json
{
  "anthropic.claude-sonnet-4-5-20250929-v1": {
    "model_id": "anthropic.claude-sonnet-4-5-20250929-v1",
    "model_arn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0",
    "model_name": "Claude Sonnet 4.5",
    "model_provider": "Anthropic",
    "model_modalities": {
      "input_modalities": [
        "TEXT",
        "IMAGE"
      ],
      "output_modalities": [
        "TEXT"
      ]
    },
    "streaming_supported": true,
    "customization": {
      "customization_supported": [],
      "customization_options": {}
    },
    "inference_types_supported": [
      "INFERENCE_PROFILE"
    ],
    "model_lifecycle": {
      "status": "ACTIVE",
      "release_date": ""
    },
    "regions_available": [
      "us-east-1",
      "us-east-2",
      "us-west-1",
      "us-west-2"
    ],
    "model_capabilities": [],
    "model_use_cases": [],
    "languages_supported": [],
    "consumption_options": [],
    "cross_region_inference": {
      "supported": true,
      "profiles_count": 35,
      "source_regions": [
        "us-east-1",
      ],
      "profiles": [
        {
          "profile_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
          "profile_name": "US Anthropic Claude Sonnet 4.5",
          "type": "SYSTEM_DEFINED",
          "source_region": "us-east-1",
          "description": "Routes requests to Anthropic Claude Sonnet 4.5 in us-east-1, us-east-2 and us-west-2."
        },
        {
          "profile_id": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
          "profile_name": "Global Claude Sonnet 4.5",
          "type": "SYSTEM_DEFINED",
          "source_region": "us-east-1",
          "description": "Routes requests to Claude Sonnet 4.5 globally across all supported AWS Regions."
        }
      ]
    },
    "documentation_links": {},
    "model_pricing": {
      "is_pricing_available": true,
      "pricing_reference_id": "Anthropic.anthropic.claude-sonnet-4-5",
      "pricing_file_reference": {
        "provider": "Anthropic",
        "model_key": "anthropic.claude-sonnet-4-5",
        "model_name": "Claude Sonnet 4.5"
      },
      "pricing_summary": {
        "total_regions": 23,
        "has_batch_pricing": false,
        "available_regions": [
          "us-east-1",
        ],
        "pricing_types": [
          "on_demand"
        ]
      }
    },
    "model_service_quotas": {
      "us-east-1": [
        {
          "quota_code": "L-8EA73537",
          "quota_name": "Cross-region model inference tokens per minute for Anthropic Claude Sonnet 4.5 V1 1M Context Length",
          "quota_arn": "arn:aws:servicequotas:us-east-1:401243198616:bedrock/L-8EA73537",
          "description": "The maximum number of cross-region tokens that you can submit for model inference in one minute for Anthropic Claude Sonnet 4.5 V1 1M Context Length. The quota considers the combined sum of input and output tokens across all requests to Converse, ConverseStream, InvokeModel and InvokeModelWithResponseStream.",
          "quota_applied_at_level": "ACCOUNT",
          "value": 20000.0,
          "unit": "None",
          "adjustable": true,
          "global_quota": false,
          "usage_metric": {},
          "period": {}
        },
        {
          "quota_code": "L-E107194C",
          "quota_name": "Model invocation max tokens per day for Anthropic Claude Sonnet 4.5 V1 1M Context Length (doubled for cross-region calls)",
          "quota_arn": "arn:aws:servicequotas:us-east-1:401243198616:bedrock/L-E107194C",
          "description": "Daily maximum tokens for model inference for Anthropic Claude Sonnet 4.5 V1 1M Context Length. Combines sum of input and output tokens across all requests to Converse, ConverseStream, InvokeModel and InvokeModelWithResponseStream. Doubled for cross-region calls; not applicable in case of approved TPM increase.",
          "quota_applied_at_level": "ACCOUNT",
          "value": 720000000.0,
          "unit": "None",
          "adjustable": false,
          "global_quota": false,
          "usage_metric": {},
          "period": {}
        }
      ]
    },
    "collection_metadata": {
      "first_discovered_at": "2025-11-03 09:48:51 UTC",
      "first_discovered_in_region": "us-east-1",
      "api_source": "list_foundation_models",
      "dual_region_collection": true,
      "regions_collected_from": [
        "us-east-1",
        "us-west-2"
      ],
      "phase2_regional_discovery": true,
      "regional_data_source": "pricing_data"
    },
    "regional_availability_source": "pricing_data",
    "total_regions_available": 23,
    "batch_inference_supported": {
      "supported": false,
      "supported_regions": [],
      "coverage_percentage": 0.0,
      "detection_method": "pricing_data_analysis"
    },
    "converse_data": {
      "context_window": 200000,
      "max_output_tokens": 4096,
      "size_category": {
        "category": "Large",
        "color": "#10B981",
        "tier": 3
      },
      "verified": true,
      "source": "ctx:litellm|out:litellm_litellm",
      "litellm_verified": true,
      "capabilities_count": 8,
      "use_cases_count": 18,
      "regions_count": 23
    },
    "has_pricing": true,
    "has_quotas": true,
    "total_quotas_assigned": 210
  },
```

## Collection Process

**Phase 1:** Models Extraction from AWS (3-5s) - Discovery using optimal region selection
**Phase 2:** Pricing Integration (1-2s) - Match models to pricing data references
**Phase 3:** Regional Availability Discovery (5-7s) - Map regional availability from pricing
**Phase 4:** Enhanced Features (25-30s) - CRIS profiles, batch inference, customization options
**Phase 4.5:** Token Specifications Discovery (10-15s) - LiteLLM-first context windows and max tokens
**Phase 5:** Service Quotas Collection (15-20s) - Rate limits and quotas by model and region

### Token Specifications Discovery (Phase 4.5)

This phase uses a **LiteLLM-first approach** to discover accurate context windows and max output tokens:

**Primary Source: [LiteLLM Database](https://github.com/BerriAI/litellm)**
- 234+ Bedrock-compatible models with verified specifications
- Community-maintained database with frequent updates
- Enhanced fuzzy matching strategies (direct, path, component matching)
- Cross-reference validation between multiple community sources

**Fallback: Manual Corrections**
- `corrections/model_spec_corrections.json` for missing models
- AWS documentation references for accurate values
- Override mechanism for correcting inaccurate community data

**Benefits of this approach:**
- ✅ 96%+ coverage of Bedrock models with accurate specifications
- ✅ No dependency on AWS Converse API calls
- ✅ Community-verified data with continuous updates
- ✅ Faster collection with minimal API overhead

## Required AWS Permissions

### Bedrock Access
```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:ListFoundationModels",
    "bedrock:GetFoundationModel",
    "bedrock:ListInferenceProfiles",
    "bedrock:GetInferenceProfile"
  ],
  "Resource": "*"
}
```

### Service Quotas Access
```json
{
  "Effect": "Allow",
  "Action": [
    "servicequotas:ListServiceQuotas",
    "servicequotas:GetServiceQuota"
  ],
  "Resource": "*"
}
```

## Configuration

Edit `config.py` for performance tuning:
```python
COLLECTOR_THREADS = 4                    # Concurrent threads
ENABLE_PARALLEL_COLLECTION = True       # Parallel processing
AWS_DEFAULT_REGION = "us-east-1"        # Primary region
```

**Thread Recommendations:**
- **2 threads** - If hitting rate limits
- **4 threads** - Recommended default
- **8+ threads** - High-performance setups

## Troubleshooting

**Access Denied**
→ Check AWS permissions for Bedrock and Service Quotas APIs

**Rate Limit Exceeded**
→ Reduce `COLLECTOR_THREADS` in config.py

**No Models Found**
→ Verify Bedrock access in AWS console, check regional availability

**Region Not Available**
→ Some regions don't have Bedrock yet (collector auto-skips)

**Debug Info:** Check `collectors/model-collector/logs/model_collection.log`

## Integration

- **Works with Pricing Collector** - Provides model data for pricing integration
- **Streamlit Integration** - Runs automatically when updating database
- **Standalone Mode** - Can be used independently for data export
- **LiteLLM Data Source** - Uses community-verified specifications from [LiteLLM database](https://github.com/BerriAI/litellm)
- **External Corrections** - Manual corrections file for edge cases and AWS documentation overrides

---

## Important Notes

**This is a sample implementation for learning and reference purposes.** It demonstrates patterns for collecting Amazon Bedrock model information but should be reviewed and adapted for production use cases.

**For production implementations:**
- Add comprehensive error handling for API failures and rate limits
- Implement proper logging and monitoring
- Consider data validation and consistency checks
- Add retry mechanisms with exponential backoff
- Review security implications of API calls and data storage

**API Dependencies:** This sample queries the latest AWS APIs. Model availability and features vary by region and change over time. Production implementations should include proper error handling for API changes and regional variations.