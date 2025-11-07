# Amazon Bedrock Model Profiler - Agent Guide

## Overview
A comprehensive profiling and selection tool for Amazon Bedrock foundation models. Helps users discover, compare, and select the right Bedrock models for their specific use cases through migration planning and capability analysis.

## Core Purpose
- **Model Discovery**: Catalog and explore all available Bedrock foundation models
- **Selection Assistance**: Compare capabilities, pricing, and regional availability
- **Migration Planning**: Evaluate models when migrating from other AI services
- **Cost Optimization**: Analyze pricing across different consumption models

## Architecture Components

### 1. Data Collectors (`/collectors/`)
- **model-collector**: Discovers Bedrock models, capabilities, quotas across regions with LiteLLM-first token specifications
- **pricing-collector**: Gathers comprehensive pricing data from AWS APIs
- **Output**: JSON datasets for model specifications, token specs, and pricing information

### 2. Web Interface (`/ui/`, `app.py`)
- **Streamlit-based**: Interactive model browser and comparison tool
- **Key Views**: Model cards, comparison radar charts, pricing analysis
- **Features**: Filtering, favorites, detailed specifications

### 3. Core Utilities (`/utils/`)
- **Data processing**: Model data transformation and enrichment
- **AWS integration**: Credential management and API interactions
- **Recommendation engine**: Model selection algorithms

## Key Entry Points

### For Users
```bash
python3 launch.py  # Launch full application
```

### For Developers/Agents
```python
# Model data access
from models.new_model_repository import NewModelRepository
repo = NewModelRepository()
models = repo.get_all_models()

# Pricing integration
from collectors.model-collector.src.utils.pricing_integrator import PricingIntegrator
integrator = PricingIntegrator()

# Data collection
from collectors.model-collector.src.main import main as collect_models
from collectors.pricing-collector.src.main import main as collect_pricing
```

## Configuration (`config.py`)
- **AWS settings**: Default region, credential profiles
- **Data paths**: Collectors output directories, model storage locations
- **UI settings**: Streamlit port, logging levels, parallel processing flags
- **Collection settings**: Worker counts, timeout configurations

## Data Flow
1. **Collection**: Collectors gather live data from AWS Bedrock, Pricing APIs, and LiteLLM community database
2. **Processing**: Raw API responses and community data transformed into structured JSON format
3. **Token Enhancement**: LiteLLM-first approach enriches models with context windows and max output tokens
4. **Integration**: Pricing data merged with model specifications, capabilities, and token specifications
5. **Presentation**: Web UI provides interactive exploration and comparison tools

## Key Data Structures

### Model Data Format
```json
{
  "providers": {
    "anthropic": {
      "claude-3-5-sonnet": {
        "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "capabilities": ["text", "multimodal"],
        "regions": ["us-east-1", "us-west-2"],
        "pricing_reference": "pricing_file.json#anthropic.claude-3-5-sonnet",
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
          "litellm_verified": true
        }
      }
    }
  }
}
```

### Pricing Data Format
```json
{
  "providers": {
    "anthropic": {
      "claude-3-5-sonnet": {
        "regions": {
          "us-east-1": {
            "input_tokens": 0.003,
            "output_tokens": 0.015
          }
        }
      }
    }
  }
}
```

## Extension Opportunities for Agents

### Custom Data Sources
- **Token Specifications**: Enhance LiteLLM data with custom model specifications
- **Custom Capabilities**: Add domain-specific capability assessments
- **Usage Analytics**: Incorporate real-world usage patterns
- **Community Corrections**: Contribute to model specification accuracy

### Enhanced Algorithms
- **Smart Recommendations**: ML-based model selection
- **Cost Prediction**: Usage-based cost forecasting
- **Migration Pathways**: Automated migration planning workflows

### Integration APIs
- **REST Endpoints**: Programmatic access to model data
- **CLI Tools**: Command-line model selection utilities
- **SDK Wrappers**: Language-specific integration libraries

### Visualization Extensions
- **Token Specification Charts**: Context window and max output token comparisons
- **Migration Maps**: Visual migration pathway planning
- **Cost Dashboards**: Advanced pricing analysis tools
- **Data Source Tracking**: Visualize LiteLLM vs corrected data sources

## Dependencies
- **Python 3.8+**: Core runtime environment
- **AWS SDK (boto3)**: Bedrock and Pricing API access
- **Streamlit**: Web interface framework
- **Pandas**: Data processing and analysis
- **[LiteLLM Database](https://github.com/BerriAI/litellm)**: External data source for token specifications (234+ Bedrock models)
- **Required AWS Permissions**:
  - `bedrock:ListFoundationModels`
  - `bedrock:GetFoundationModel`
  - `bedrock:ListInferenceProfiles`
  - `pricing:GetProducts`
  - `servicequotas:ListServiceQuotas`

## Key Files for Agent Integration

### Core Application
- `app.py`: Main Streamlit application entry point
- `launch.py`: Application launcher with environment setup
- `config.py`: Central configuration management

### Data Layer
- `models/new_model_repository.py`: Primary data access interface
- `utils/common.py`: Data transformation utilities
- `utils/recommendation.py`: Model selection algorithms

### Collection System
- `collectors/model-collector/src/main.py`: Model data collection
- `collectors/model-collector/src/utils/token_specs_enhancer.py`: LiteLLM-first token specifications discovery
- `collectors/pricing-collector/src/main.py`: Pricing data collection
- `collectors/*/src/config.py`: Collector-specific configurations

### UI Components
- `ui/pages.py`: Streamlit page definitions
- `ui/filters.py`: Model filtering interfaces
- `ui/model_details_modal.py`: Detailed model information displays

## Agent Integration Patterns

### Data Access Pattern
```python
# Initialize repository
repo = NewModelRepository()

# Get all models with pricing
models = repo.get_models_with_pricing()

# Filter by capabilities
text_models = repo.filter_by_capability(models, "text")
```

### Collection Pattern
```python
# Update model database
from utils.data_updater import update_model_database
update_model_database(aws_profile="default")
```

### Recommendation Pattern
```python
# Get model recommendations
from utils.recommendation import get_model_recommendations
recommendations = get_model_recommendations(
    use_case="text_generation",
    budget_limit=100,
    region="us-east-1"
)
```

This tool is designed to be migration-focused, helping organizations make informed decisions when adopting Amazon Bedrock foundation models for their AI workloads.