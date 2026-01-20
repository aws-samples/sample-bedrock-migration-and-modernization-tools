# Amazon Bedrock Model Collector

Comprehensive tool to collect Amazon Bedrock model information across all regions and generate a detailed JSON database with enhanced features, service quotas, and complete model capabilities.

## 🎯 Features

- **Multi-threaded Collection**: Efficient 2-thread concurrent processing across 20+ Bedrock regions
- **Comprehensive Model Data**: 20+ fields per model including capabilities, use cases, languages, and consumption options
- **Service Quotas Integration**: Model-specific quota filtering using AWS Service Quotas API
- **Regional Coverage**: Automatic discovery and collection from all Bedrock-enabled regions
- **Pricing Integration**: Seamless integration with amazon-bedrock-pricing-collector data
- **Cross-Platform Support**: Native scripts for Windows, Linux, and macOS
- **Enhanced Features**: Agreement offers, inference profiles, batch inference detection, and cross-region inference support

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.8+** (tested with 3.9-3.12)
- **AWS CLI** configured with appropriate profile
- **Amazon Bedrock access** in at least one region
- **Service Quotas permissions** for complete quota collection

### 2. Run the Collector

#### **Linux/macOS:**
```bash
# Make script executable
chmod +x run_collector.sh

# Run the collector
./run_collector.sh
```

#### **Windows:**
```cmd
# Run the collector
run_collector.bat
```

Both scripts will:
- ✅ Create and activate a virtual environment
- ✅ Install all dependencies automatically
- ✅ Validate AWS credentials and permissions
- ✅ Run comprehensive model collection across all regions
- ✅ Generate timestamped JSON output in `out/` directory
- ✅ Provide detailed logging and error reporting

### 3. Test API Access (Optional)

```bash
# Linux/macOS
python3 tests/test_api_access.py

# Windows
python tests/test_api_access.py
```

## 📄 Output Structure

The collector generates comprehensive JSON output with 20+ fields per model:

```json
{
  "metadata": {
    "generated_at": "2025-10-06 17:54:53 UTC",
    "version": "1.0.0",
    "description": "Comprehensive Amazon Bedrock Model Database",
    "providers_count": 11,
    "total_models": 69,
    "regions_covered": 20,
    "models_with_pricing": 64,
    "models_with_quotas": 38,
    "regions": ["us-east-1", "us-west-2", ...],
    "data_sources": ["bedrock_api", "service_quotas", "inference_profiles", "agreement_offers", "batch_inference", "pricing_collector", "region_discovery", "model_enhancement"],
    "collection_summary": {
      "raw_models_collected": 100,
      "pricing_integrations": 100,
      "quota_regions": 20,
      "multi_threaded_collection": true
    }
  },
  "providers": {
    "Anthropic": {
      "models": {
        "claude-sonnet-4": {
          "model_id": "anthropic.claude-sonnet-4-20250514-v1:0",
          "model_name": "Claude Sonnet 4",
          "model_provider": "Anthropic",
          "model_family": "Claude",
          "description": { /* Rich description object */ },

          // Model capabilities and technical specs
          "input_modalities": ["TEXT", "IMAGE"],
          "output_modalities": ["TEXT"],
          "capabilities": ["chat", "text_generation", "reasoning", "multimodal"],
          "use_cases": ["chat applications", "content generation", "complex reasoning"],
          "languages_supported": ["English", "Spanish", "French", ...],
          "consumption_options": ["on_demand", "provisioned_throughput", "batch_inference"],

          // Inference and technical details
          "inference_types_supported": ["INFERENCE_PROFILE"],
          "response_streaming_supported": true,
          "customizations_supported": [],
          "guardrails_supported": true,
          "model_lifecycle": {"status": "ACTIVE"},

          // Cross-region and batch inference
          "cross_region_inference": {
            "supported": true,
            "profiles_count": 107,
            "source_regions": ["us-east-1", "us-west-2", ...],
            "destination_regions": ["ap-northeast-1", ...],
            "profiles": [/* Detailed profile information */]
          },
          "batch_inference_supported": {
            "supported": true,
            "supported_regions": ["us-east-1", "us-west-2"],
            "coverage_percentage": 85.0,
            "detection_method": "real_api_testing"
          },

          // Pricing and quotas (model-specific filtering)
          "model_pricing": {
            "is_pricing_available": true,
            "pricing_details": {/* Detailed pricing info */}
          },
          "model_service_quotas": {
            "us-east-1": {
              "L-3D8CC480": {
                "quota_name": "Cross-region model inference requests per minute for Anthropic Claude Sonnet 4",
                "value": 250.0,
                "adjustable": false,
                "unit": "None"
              }
            },
            "quota_metadata": {
              "total_quotas_retrieved": 179,
              "regions_queried": 16,
              "collection_timestamp": "2025-10-06 17:54:51 UTC"
            }
          },

          // Additional features
          "agreement_offers": {"offers_count": 0, "has_offers": false},
          "context_window": "200K",
          "documentation_links": {}
        }
      }
    }
  }
}
```

## ⚙️ Configuration

The collector uses intelligent defaults but can be customized by editing configuration files:

**AWS Configuration:**
- Default region: `us-east-1`
- Auto-discovery: All Bedrock-enabled regions

**Collection Settings:**
- Worker threads: 2 (optimal balance of speed vs. rate limits)
- Timeout: 120s per region
- Retry logic: 3 attempts with exponential backoff

**Output Settings:**
- Format: Pretty-printed JSON
- Timestamped files: `bedrock-models-YYYYMMDD_HHMMSS.json`
- Location: `out/` directory

## 📊 Data Sources & APIs

The collector integrates multiple AWS APIs, data processing, and external sources:

### 🔌 **AWS APIs Used**

| **Data Source** | **API/Service** | **Purpose** |
|-----------------|-----------------|-------------|
| **Model Catalog** | `bedrock.list_foundation_models()` | Base model information across all regions |
| **Region Discovery** | `ec2.describe_regions()` | Auto-discovery of all AWS regions |
| **Service Quotas** | `servicequotas.list_service_quotas(ServiceCode='bedrock')` | Model-specific rate limits and quotas |
| **Inference Profiles** | `bedrock.list_inference_profiles()` | Cross-region inference capabilities |
| **Profile Details** | `bedrock.get_inference_profile()` | Detailed inference profile metadata |
| **Agreement Offers** | `bedrock.list_foundation_model_agreement_offers()` | Commercial terms and pricing agreements |
| **Batch Inference** | `bedrock.list_model_import_jobs()` | Batch processing capability detection |
| **Direct API Calls** | `HTTPS://bedrock.<region>.amazonaws.com/foundation-models` | Fallback model catalog access |

### 📁 **File-Based Data Sources**

| **Data Source** | **Format** | **Purpose** |
|-----------------|------------|-------------|
| **Pricing Data** | `amazon-bedrock-pricing-collector/*.json` | Detailed pricing information integration |
| **Configuration** | `src/config.py` | AWS profiles, regions, and collection settings |

### 🧠 **Data Processing & Enhancement**

The collector performs intelligent data enhancement through multiple processing layers:

| **Enhancement Type** | **Method** | **Generated Fields** |
|---------------------|------------|---------------------|
| **Capabilities Extraction** | `_extract_capabilities()` | `capabilities[]` from input/output modalities |
| **Use Cases Derivation** | `_extract_use_cases()` | `use_cases[]` based on capabilities |
| **Language Support** | `_extract_languages_supported()` | `languages_supported[]` from provider patterns |
| **Consumption Options** | `_extract_consumption_options()` | `consumption_options[]` from inference types |
| **Cross-Region Processing** | `_add_inference_profiles()` | `cross_region_inference{}` with profiles |
| **Batch Inference Testing** | `_add_batch_inference_support_real()` | `batch_inference_supported{}` via real API tests |
| **Quota Filtering** | `_filter_model_quotas()` | `model_service_quotas{}` with precise model matching |
| **Agreement Processing** | `_add_agreement_offers()` | `agreement_offers{}` for commercial terms |

### 🎯 **Intelligent Processing Examples**

**Capabilities Extraction:**
```python
# Input modalities: ["TEXT", "IMAGE"] → Output modalities: ["TEXT"]
# Generated capabilities: ["chat", "text_generation", "multimodal", "reasoning"]
```

**Use Cases Derivation:**
```python
# From capabilities: ["chat", "multimodal"]
# Generated use_cases: ["chat applications", "document analysis", "visual question answering"]
```

**Language Support Inference:**
```python
# Provider: "Anthropic" → Generated: ["English", "Spanish", "French", "German", ...]
# Provider: "DeepSeek" → Generated: ["English", "Chinese"]
```

**Service Quota Filtering:**
```python
# Claude Sonnet 4 gets: "Cross-region requests per minute for Claude Sonnet 4" (179 quotas)
# Nova Lite gets: "Cross-region requests per minute for Nova Lite" (134 quotas)
# Stability AI gets: General quotas + Stability-specific quotas (71 quotas)
```

## 🏗️ Project Structure

```
amazon-bedrock-model-collector/
├── 📄 main.py                    # Entry point
├── 🔧 run_collector.sh           # Linux/macOS launcher
├── 🔧 run_collector.bat          # Windows launcher
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # This documentation
├── 📂 src/                       # Source code
│   ├── 📄 config.py             # Configuration settings
│   ├── 📄 main.py               # Main orchestrator
│   ├── 📂 collectors/           # Data collection modules
│   │   ├── direct_api_collector.py
│   │   └── web_collector.py
│   └── 📂 utils/                # Utility modules
│       ├── data_processor.py    # JSON structure & processing
│       ├── features_enhancer.py # Model enhancement & enrichment
│       ├── model_collector.py   # Main collection orchestrator
│       ├── pricing_integrator.py # Pricing data integration
│       ├── quotas_collector.py  # Service quotas collection
│       └── region_collector.py  # Region discovery
├── 📂 tests/                    # API validation tests
│   └── test_api_access.py       # AWS credentials & access test
├── 📂 out/                      # Generated JSON output files
├── 📂 logs/                     # Collection and error logs
└── 📂 venv/                     # Virtual environment (auto-created)
```

## 📋 Requirements & Dependencies

**System Requirements:**
- Python 3.8+ (tested with 3.9-3.12)
- 2GB+ RAM (for large model datasets)
- Internet connection for API calls

**Python Dependencies:**
```
boto3>=1.34.0          # AWS SDK
requests>=2.31.0       # HTTP requests
beautifulsoup4>=4.12.0 # Web scraping (fallback)
```

**AWS Permissions Required:**

*Core Bedrock Permissions:*
- `bedrock:ListFoundationModels` - Base model catalog access
- `bedrock:GetFoundationModel` - Individual model details

*Cross-Region & Inference:*
- `bedrock:ListInferenceProfiles` - Cross-region inference discovery
- `bedrock:GetInferenceProfile` - Detailed inference profile metadata

*Commercial & Agreements:*
- `bedrock:ListFoundationModelAgreementOffers` - Pricing agreements

*Batch & Import:*
- `bedrock:ListModelImportJobs` - Batch inference capability detection

*Service Management:*
- `servicequotas:ListServiceQuotas` - Quota information (model-specific filtering)

*Regional Discovery:*
- `ec2:DescribeRegions` - Auto-discovery of all AWS regions

*Note: The collector gracefully handles partial permissions - missing permissions result in empty fields rather than failures.*

## 🚨 Troubleshooting

**Common Issues:**

1. **"No quotas found"**: Ensure Service Quotas permissions are granted
2. **"Access denied"**: Check AWS profile configuration and Bedrock access
3. **"Rate limiting"**: The collector includes automatic retry logic and delays
4. **"Empty output"**: Verify Bedrock is available in your configured regions

**Debugging:**
```bash
# Check AWS credentials
aws sts get-caller-identity --profile <your-aws-profile-name>

# Test API access
python tests/test_api_access.py

# Check logs
tail -f logs/model_collection.log
```

## 📄 License

MIT License - see LICENSE file for details