# Pricing Collector Sample 💰

This sample demonstrates how to programmatically collect comprehensive pricing information for Amazon Bedrock foundation models using official AWS APIs. It shows implementation patterns for gathering, processing, and organizing pricing data across multiple consumption options and regions.

## What this sample demonstrates

* 💰 **API Integration Patterns** - How to retrieve current pricing from official AWS Pricing API
* 🌍 **Multi-Region Data Collection** - Methods for collecting pricing across all Amazon Bedrock regions
* 📊 **Complex Data Processing** - Techniques for handling multiple pricing options (on-demand, provisioned throughput, batch, CRIS)
* 🔗 **Data Correlation** - Approaches to link pricing data with specific foundation models
* ⚡ **Data Standardization** - Implementation patterns for normalizing units, regions, and pricing dimensions

## Data Source (AWS APIs)

### 💳 AWS Pricing API
**Service Codes:**
- **`AmazonBedrock`** - Core foundation model pricing (for 1st and 2nd party models)
- **`AmazonBedrockService`** - Service pricing (Guardrails, Knowledge Base)
- **`AmazonBedrockFoundationModels`** - Additional foundation model pricing (for 3rd party models)

**API Calls:**
- **`pricing:GetProducts`** - Primary pricing data retrieval
- **`pricing:DescribeServices`** - Service discovery
- **`pricing:GetAttributeValues`** - Pricing dimension values

## Pricing Categories

### 🚀 On-Demand Pricing
- Pay-per-use token pricing
- Input/output token costs
- Standard and long-context variants

### ⚡ Provisioned Throughput Pricing
- Reserved capacity pricing (model units/hour)
- Commitment options: No commitment, 1 month, 6 months
- Consistent performance guarantees

### 📦 Batch Inference Pricing
- Asynchronous processing pricing
- Bulk inference discounts
- Cost-optimized batch processing

### 🌐 Cross-Region Inference Pricing
- CRIS (Cross-Region Inference Service) pricing
- Global inference endpoint costs
- Multi-region deployment costs

### 🎯 Model Customization Pricing
- Fine-tuning costs
- Continued pre-training pricing
- Custom model import pricing

## Running this Sample

### 🌐 **Via Web Interface (Recommended)**
1. Launch the main app: `python3 launch.py`
2. Open `http://localhost:8501` in your browser
3. Go to "⚙️ AWS Setup & Update" in sidebar
4. Select AWS profile and click "🔄 Update models database"
5. Pricing collection runs automatically with model collection

### 💻 **Manual Execution**
For development or testing:
```bash
cd collectors/pricing-collector/src
python main.py --profile your-aws-profile
```

**Options:**
- `--profile` - AWS profile name
- `--region` - Pricing API region (default: us-east-1)
- `--output` - Output directory
- `--verbose` - Detailed logging
- `--skip-enhancement` - Raw data only

## Data Output

### Individual Pricing Files
**Location:** `data/pricing/[Provider]_[Model]_pricing.json`

Sample structure:
```json
{
  "anthropic.claude-sonnet-4-5": {
    "model_name": "Claude Sonnet 4.5",
    "model_provider": "Anthropic",
    "regions": {
      "us-east-1": {
        "pricing_groups": {
          "On-Demand": [
            {
              "dimension": "USE1-MP:USE1_CacheReadInputTokenCount-Units",
              "price_per_thousand": 0.00033,
              "original_price": 0.33,
              "unit": "1K tokens",
              "description": "AWS Marketplace software usage|us-east-1|thousand Cache Read Input Tokens Regional",
              "source_dataset": "aws_pricing_api",
              "model_id": "anthropic.claude-sonnet-4-5",
              "model_name": "Claude Sonnet 4.5",
              "provider": "Anthropic",
              "model_provider": "Anthropic",
              "location": "US East (N. Virginia)",
              "operation": "Usage",
              "service_code": "AmazonBedrockFoundationModels",
              "pricing_characteristics": {
                "inference_type": "on_demand",
                "context_type": "standard",
                "geographic_scope": "regional"
              },
              "pricing_group": "On-Demand"
            }
          ]
        },
        "total_dimensions": 16,
        "groups_count": 4,
        "group_statistics": {
          "total_entries": 16,
          "total_groups": 4,
          "group_sizes": {
            "On-Demand": 4,
            "On-Demand Long Context": 4,
            "On-Demand Long Context Global": 4,
            "On-Demand Global": 4
          },
          "largest_groups": [
            ["On-Demand",4],
            ["On-Demand Long Context",4],
            ["On-Demand Long Context Global",4],
            ["On-Demand Global",4]
          ],
          "average_entries_per_group": 4.0
        }
      }
    },
    "total_regions": 23,
    "total_pricing_entries": 312
  }
}
```

### Integrated Model Data
Pricing is also embedded in main model database (`data/bedrock_models.json`):
```json
{
  "model_id": "anthropic.claude-sonnet-4-5-20250929-v1",
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
          "us-east-1"
        ],
        "pricing_types": [
          "on_demand"
        ]
      }
  }
}
```

## Collection Process

**Phase 1:** Raw Data Collection (8-15s) - Connect to AWS Pricing API, collect all products
**Phase 2:** Smart Enhancement (5-10s) - Standardize units, map regions, process dimensions
**Phase 3:** Intelligent Organization (5-10s) - Group by provider, categorize pricing types
**Phase 4:** Quality Assurance (2-5s) - Validate data, match to models, format output

## Required AWS Permissions

```json
{
  "Effect": "Allow",
  "Action": [
    "pricing:GetProducts",
    "pricing:DescribeServices",
    "pricing:GetAttributeValues"
  ],
  "Resource": "*"
}
```

**Note:** Pricing API calls are made to us-east-1 (global pricing endpoint)

## Configuration

Edit `config.py` for performance tuning:
```python
PRICING_API_REGION = "us-east-1"         # Pricing API endpoint
PRICING_THREADS = 2                      # Concurrent queries
ENABLE_PRICING_ENHANCEMENT = True        # Smart processing
PRICING_CACHE_TTL = 3600                # Cache for 1 hour
```

## Data Enhancement Features

**Price Standardization:**
- Unit conversion: Million → Thousand tokens
- Currency normalization: All USD per 1K tokens
- Precision handling: Maintains AWS precision

**Region Intelligence:**
- Location mapping: "US East (N. Virginia)" → "us-east-1"
- Geographic grouping by continent
- Regional availability detection

**Smart Processing:**
- Commitment term extraction
- Usage type parsing (input/output)
- Model variant detection
- Duplicate removal

## Troubleshooting

**Access Denied to Pricing API**
→ Verify AWS permissions for Pricing service

**No Pricing Data Found**
→ Check if models exist in pricing API, some new models may have delayed pricing

**Rate Limit Exceeded**
→ Reduce `PRICING_THREADS` in config.py

**Invalid Pricing Format**
→ AWS occasionally changes pricing structure, check for API updates

**Debug Info:** Check `collectors/pricing-collector/logs/pricing_collection.log`

## Integration

- **Works with Model Collector** - Runs in parallel during data updates
- **Streamlit Integration** - Automatic execution when updating database
- **Standalone Mode** - Can be used independently for pricing analysis
- **Data Flow:** Model discovery → Pricing matching → Enhanced database

---

## Important Notes

**This is a sample implementation for learning and reference purposes.** It demonstrates patterns for integrating with AWS Pricing APIs but should be reviewed and adapted for production use cases.

**For production implementations:**
- Add proper error handling and retry logic
- Implement monitoring and alerting for API failures
- Consider data validation and consistency checks
- Review security implications of API calls and data storage

**API Dependencies:** This sample relies on AWS Pricing API availability. AWS occasionally updates pricing formats - the sample includes adaptive processing patterns to handle such changes, but production implementations should include more robust error handling.