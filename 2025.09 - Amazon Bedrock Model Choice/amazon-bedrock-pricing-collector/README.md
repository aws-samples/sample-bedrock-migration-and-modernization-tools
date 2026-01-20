# Amazon Bedrock Pricing Collector

A comprehensive, standalone tool that collects, standardizes, and organizes Amazon Bedrock pricing data with world-class quality enhancements. This tool uses the AWS Pricing API to gather comprehensive pricing information and applies intelligent processing for superior data quality and organization.

## 🚀 How It Gathers Data

### Data Source: AWS Pricing API
The collector uses **only the official AWS Pricing API** to gather all pricing information:

**Service Codes Queried:**
- `AmazonBedrock` - Core Bedrock foundation model pricing
- `AmazonBedrockService` - Bedrock service-specific pricing (Guardrails, Knowledge Base, etc.)
- `AmazonBedrockFoundationModels` - Additional foundation model pricing data

**API Approach:**
- **Direct AWS API Access**: Uses boto3 pricing client for real-time data
- **Comprehensive Coverage**: Queries all 3 service codes for complete pricing information
- **Official Data**: Only uses verified AWS pricing data, no web scraping
- **Real-time Updates**: Always gets the latest pricing information from AWS

### Enhanced Processing Pipeline

1. **Raw Data Collection** (~8 seconds)
   - Connects to AWS Pricing API in us-east-1
   - Queries all 3 Bedrock service codes simultaneously
   - Collects all pricing entries across all regions and models

2. **Smart Data Enhancement**
   - **Price conversions**: Million→thousand standardization when needed
   - **Unit extractions**: Convert vague units ("Units") to specific ("1K tokens")
   - **Region standardizations**: Location names→region codes (100% success)
   - **Dimension enhancements**: Add commitment terms, resolution specs
   - **Commitment lengths**: Extract provisioned throughput terms
   - **Custom model classifications**: Categorize Import vs Training vs Generic

3. **Intelligent Organization**
   - **Providers detected**: Including specialized Custom Model Import provider
   - **Context-aware enhancements**: Speech/text for Nova Sonic, video specs for Luma
   - **Smart deduplication**: Remove duplicates while preserving variants
   - **Pricing group creation**: Organize by On-Demand, Batch, Provisioned Throughput

## 🚀 Key Features

- **🎯 World-Class Data Quality**: Smart conversion, unit extraction, and comprehensive standardization
- **⚡ Lightning Fast**: Quick collection with comprehensive pricing coverage
- **🏢 Multiple Providers Detected**: Including specialized Custom Model Import provider
- **🧠 Context-Aware Enhancement**: Speech/text distinction for Nova Sonic, video resolution specs for Luma
- **💡 Smart Processing**: Million→thousand conversion when needed, vague→specific unit extraction
- **🌍 100% Region Standardization**: Automatic location name→region code conversion
- **📋 Commitment Term Detection**: Provisioned throughput commitment lengths (No Commitment, 1 Month, 6 Months)
- **🔄 Deduplication**: Intelligent duplicate removal while preserving pricing variants
- **📊 Structured Output**: Well-organized JSON with pricing groups and regional breakdown

## Enhanced Data Quality

### Smart Conversions Applied
- **Price conversions**: Intelligent million→thousand standardization when needed
- **Unit extractions**: Convert vague units ("Units", "hour") to specific units ("1K tokens", "second (720p, 24fps)")
- **Region standardizations**: 100% conversion from location names to standardized region codes
- **Dimension enhancements**: Added commitment terms, resolution specs, input/output type specifications
- **Commitment lengths**: Extracted provisioned throughput commitment terms
- **Custom model classifications**: Distinguished Custom Model Import vs Training vs Generic types

### Provider Organization
- **Amazon** (with sub-groups: Nova, Titan, Other Amazon)
- **Anthropic** (Claude models)
- **Meta** (Llama models)
- **Mistral** (unified provider)
- **Cohere**, **AI21 Labs**, **Stability AI**
- **Qwen**, **DeepSeek**, **Writer**
- **TwelveLabs**, **Luma**, **OpenAI**
- **Custom Model Import** (specialized architectures: GPTBig Code, Flan, Mistral, Llama, Qwen2VL, Mixtral)
- **Unknown Models** (unrecognized models)

## Prerequisites

- Python 3.8 or higher
- AWS CLI configured with appropriate permissions
- AWS profile with access to:
  - AWS Pricing API (`pricing:GetProducts`)
- Internet connection for region mapping updates

## Quick Start

### Option 1: Using Shell Scripts (Recommended)

The easiest way to run the collector is using the provided shell scripts:

**Linux/macOS:**
```bash
./run_collector.sh
```

**Windows:**
```cmd
run_collector.bat
```

### Option 2: Manual Execution

1. **Set up virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS credentials**
   ```bash
   aws configure --profile your-profile-name
   ```

4. **Run the collector**
   ```bash
   python3 main.py --profile your-profile-name
   ```

## Enhanced Output Structure

The tool generates comprehensive JSON output with enhanced data quality:

```json
{
  "metadata": {
    "generated_at": "2025-10-21T18:54:08Z",
    "version": "1.0.0",
    "total_pricing_entries": "...",
    "providers_count": "...",
    "currency": "USD",
    "pricing_standardization": "Smart conversion applied: per-million to per-thousand when needed, unit extraction from descriptions",
    "group_types_available": [
      "On-Demand",
      "Batch",
      "Provisioned Throughput",
      "Custom Model"
    ]
  },
  "providers": {
    "Amazon": {
      "amazon.nova-sonic": {
        "model_name": "Nova Sonic",
        "model_provider": "Amazon",
        "model_group": "Amazon Nova",
        "regions": {
          "us-east-1": {
            "pricing_groups": {
              "On-Demand": [
                {
                  "dimension": "On-Demand Inference (Speech Input)",
                  "price_per_thousand": 0.0008,
                  "unit": "1K speech input tokens",
                  "description": "Speech input token processing",
                  "commitment_length": null
                },
                {
                  "dimension": "On-Demand Inference (Text Output)",
                  "price_per_thousand": 0.0016,
                  "unit": "1K text output tokens",
                  "description": "Text output token generation"
                }
              ],
              "Provisioned Throughput": [
                {
                  "dimension": "Provisioned Throughput (6 Months)",
                  "price_per_thousand": 55.0,
                  "unit": "hour",
                  "commitment_length": "6 Months",
                  "description": "6-month commitment provisioned throughput"
                }
              ]
            }
          }
        }
      }
    },
    "Custom Model Import": {
      "openai.gptbig-code": {
        "model_name": "GPTBig Code",
        "model_provider": "OpenAI",
        "custom_model_type": "Custom Model Import",
        "regions": {
          "us-east-1": {
            "pricing_groups": {
              "Custom Model": [
                {
                  "dimension": "Custom Model Import Inference",
                  "price_per_thousand": 0.05718,
                  "unit": "minute",
                  "description": "GPTBigCode architecture model inference",
                  "custom_model_type": "Custom Model Import"
                }
              ]
            }
          }
        }
      }
    }
  }
}
```

## Context-Aware Enhancements

### Nova Sonic Speech/Text Distinction
The collector automatically distinguishes between speech and text processing:
- `1K speech input tokens` vs `1K text input tokens`
- `1K speech output tokens` vs `1K text output tokens`
- Enhanced dimensions: `"On-Demand Inference (Speech Input)"`

### Luma Video Resolution Specifications
Video generation models include precise resolution and frame rate details:
- `second (540p, 24fps)` for standard quality
- `second (720p, 24fps)` for high quality
- Enhanced dimensions: `"On-Demand Inference (720p, 24fps)"`

### Provisioned Throughput Commitment Terms
Automatically extracts and categorizes commitment lengths:
- `"No Commitment"` - On-demand provisioned throughput
- `"1 Month"` - 1-month commitment discount
- `"6 Months"` - 6-month commitment discount
- `"12 Months"` - 1-year commitment discount

### Custom Model Classification
Distinguishes between different custom model types:
- **Custom Model Import**: Pre-trained models from specific architectures (Flan, Llama, Mistral, etc.)
- **Custom Model Training**: Fine-tuning and customization services
- **Custom Model**: Generic custom model services

## Regional Coverage

Automatically processes pricing for all AWS regions where Bedrock is available:
- **US**: us-east-1, us-east-2, us-west-1, us-west-2
- **Europe**: eu-west-1, eu-west-2, eu-west-3, eu-central-1, eu-north-1, eu-south-1
- **Asia Pacific**: ap-northeast-1, ap-northeast-2, ap-southeast-1, ap-southeast-2, ap-south-1
- **Other**: ca-central-1, sa-east-1, me-central-1

All location names like "EU (Ireland)" are automatically standardized to region codes like "eu-west-1".

## 🔍 How Data Flows

1. **API Query**: Collector queries AWS Pricing API for 3 Bedrock service codes
2. **Raw Processing**: Extracts pricing dimensions, units, descriptions, regions
3. **Smart Enhancement**: Applies quality improvements automatically
4. **Organization**: Groups by provider, model, region, and pricing type
5. **Output Generation**: Creates comprehensive JSON with all enhancements
6. **File Export**: Saves to timestamped file in `out/` directory

## Performance & Statistics

- **Collection Time**: Fast collection (typically under 10 seconds)
- **Comprehensive Coverage**: All pricing entries across regions and models
- **Multiple Providers**: Organized provider detection and categorization
- **Regional Processing**: Extensive region-model combinations
- **Quality Enhancements**: Comprehensive data improvements applied

## Configuration

The configuration is fully standalone in `src/config.py`:

```python
# AWS Pricing API Configuration
AWS_REGION = 'us-east-1'
PRICING_API_REGION = 'us-east-1'  # Pricing API region

# Service Codes
AWS_PRICING_SERVICE_CODES = [
    'AmazonBedrock',
    'AmazonBedrockService',
    'AmazonBedrockFoundationModels'
]

# Output Configuration
OUTPUT_DIR = 'out'
LOGS_DIR = 'logs'
```

## Success Indicators

Look for these messages indicating successful execution:

```
🚀 Amazon Bedrock Pricing Collector (AWS API Only)
📋 Collecting from 3 service codes for complete coverage
✅ Successfully updated region mappings: XX locations, XX codes
✅ AWS Pricing API: XXXX entries collected
Processed XXXX entries - Conversions: XXX, Units extracted: XXX, Dimensions enhanced: XXX
Regions standardized: XXX, Commitment lengths: XXX, Custom models classified: XXX
✅ Data saved to: out/bedrock-pricing-YYYYMMDD_HHMMSS.json
Final structure contains:
  - Providers: XX
  - Total pricing entries: XXXX
  - Total regions processed: XXX
  - Total groups created: XXXX
🎉 Collection Complete!
```

## Troubleshooting

### Common Issues

**AWS Credentials**
```bash
aws configure --profile your-profile-name
aws sts get-caller-identity --profile your-profile-name
```

**Required Permissions**
- `pricing:GetProducts` (for AWS Pricing API access)

**Python Version**
```bash
python3 --version  # Should be 3.8+
```

### Error Logs

Detailed logs are saved to `logs/pricing_collection.log` with information about:
- Data collection progress and statistics
- Smart conversion applications
- Region standardization results
- Provider detection and organization
- Error handling and retry attempts

## Data Quality Assurance

### Enhanced Validation Features
- **Smart Conversion Logic**: Only applies million→thousand conversion when appropriate
- **Unit Extraction**: Converts vague units to specific, contextual units
- **Region Standardization**: 100% success rate in standardizing region names
- **Provider Detection**: Flexible keyword matching for robust provider identification
- **Deduplication**: Removes true duplicates while preserving pricing variants
- **Commitment Detection**: Accurate extraction of provisioned throughput terms

### Quality Metrics
- **Conversions Applied**: Smart price standardizations when needed
- **Units Enhanced**: Unit extractions and contextual improvements
- **Regions Standardized**: Location→region code conversions (100% success)
- **Dimensions Enhanced**: Dimensions with added specifications
- **Commitment Terms**: Provisioned throughput commitments extracted
- **Custom Classifications**: Custom models categorized by type

## License

This project is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file for details.

## References

- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [AWS Pricing API Documentation](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/price-changes.html)
- [Amazon Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review logs in `logs/pricing_collection.log`
3. Verify AWS credentials and permissions

---

**Note**: This tool uses official AWS Pricing API data and applies intelligent enhancements for better organization and usability. Always verify pricing information through official AWS channels before making business decisions.