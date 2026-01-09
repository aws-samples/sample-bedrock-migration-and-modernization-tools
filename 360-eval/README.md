# LLM Benchmarking Framework with LLM-as-a-JURY

A comprehensive framework for benchmarking and evaluating Large Language Models (LLMs) with a specific focus on Amazon Bedrock models.

## Design Introduction

This framework implements a "LLM-as-a-Jury" methodology based on research from [Replacing Judges with Juries:
Evaluating LLM Generations with a Panel of Diverse Models](https://arxiv.org/pdf/2404.18796), each evaluation is composed of an ascending ranking score of 1-5 and, where the total LLMs evaluate responses are tallied and their mean calculated, this becomes the final score (if an average of the evaluation fails to achieve a score of 3+ across any evaluation dimension is deemed to have failed, value can be changed). This technique provides more reliable and balanced evaluations compared to single-judge, PASS | FAIL methods.

Our benchmarking system evaluates model outputs across six core dimensions:
- **Correctness**: Accuracy of information provided
- **Completeness**: Thoroughness of the response
- **Relevance**: How well the response addresses the prompt
- **Format**: Appropriate structure and presentation
- **Coherence**: Logical flow and consistency
- **Following instructions**: Adherence to prompt requirements

Additionally, the framework supports **user-defined** evaluation metrics tailored to specific use cases such as branding style or tone.

The system is designed for scalability, automatically aggregating results from multiple benchmark runs into comprehensive reports, providing an increasingly complete picture of model latency and performance over time.


<img src="./assets/llm-as-judge-background.png" alt="Alt Text" width="600" height="700">


## Overview

This project provides tools for:

- Running performance benchmarks across multiple LLM models (Amazon Bedrock and third-party models)
- Using LLM-as-jury methodology where multiple judge models evaluate other models' responses
- Optimizing prompts for target models using Amazon Bedrock's prompt optimization API
- Managing evaluations through an interactive Streamlit web dashboard
- Measuring key performance metrics (latency, throughput, cost, response quality)
- Visualizing results and generating interactive HTML reports with regional performance analysis

## Features

### Core Capabilities
- **Multi-model benchmarking**: Compare different models side-by-side, including Amazon Bedrock and third-party models (OpenAI, Google Gemini, Azure)
- **LLM-as-jury evaluation**: Leverage multiple LLM judges to evaluate model responses with aggregated scoring
- **Comprehensive metrics**: Track latency (TTFT, TTLB), throughput (tokens/sec), cost, and quality of responses
- **Interactive visualizations**: Generate holistic HTML reports with performance comparisons, heatmaps, radar charts, and error analysis

### Prompt Optimization
- **Automatic prompt optimization**: Use Amazon Bedrock's prompt optimization API to improve prompts for specific target models
- **Compare original vs optimized**: Evaluate both original and optimized prompts side-by-side to measure improvement
- **Optimization tracking**: Detailed logs of optimization attempts with success/failure tracking

### Streamlit Dashboard
- **Interactive web interface**: Launch a full-featured Streamlit dashboard for managing evaluations
- **Real-time monitoring**: Track running evaluations with live progress updates
- **Results visualization**: View and compare evaluation results directly in the browser
- **Model configuration**: Easily configure models, judges, and evaluation parameters
- **Report viewer**: Browse and view generated HTML reports

### Advanced Features
- **Vision model support**: Evaluate vision-capable models with image inputs (JPG, PNG, GIF, WebP, BMP)
- **Temperature variations**: Automatically test models across multiple temperature settings
- **User-defined metrics**: Add custom evaluation criteria beyond the six core dimensions
- **Configuration validation**: Pre-validate configuration files to catch errors before running evaluations
- **Parallel execution**: Run multiple model evaluations concurrently with configurable parallelism
- **Automatic retry logic**: Built-in retry mechanism with tracking for failed API calls
- **Regional performance analysis**: Analyze performance by AWS region with timezone-aware reporting
- **Unprocessed record tracking**: Automatically log failed evaluations for debugging
- **Composite scoring**: Multi-dimensional performance scoring combining accuracy, latency, and cost
- **Rate limiting & reliability testing**: Configure target requests-per-minute (RPM) for models to test reliability at specific load levels

## Installation

```bash
git clone <your-repository-url>
cd 360-eval
```
```bash
pip install -r requirements.txt
```

### Third Party API Setup

To use third-party models in the benchmarking process:

1. Create a `.env` file in the project root directory
2. Add your 3P API keys in the following format:

```
OPENAI_API='your_openai_api_key_here'
GOOGLE_API='your_google_api_key_here'
AZURE_API_KEY='your_azure_api_key_here'
```


## Evaluation Unit Data Format

The benchmarking tool requires input data in JSONL format, with each line containing a scenario to evaluate. Each scenario must follow this schema:

### Field Descriptions

- `text_prompt`: The prompt to send to the model (in the example: "Summarize the principles of edge computing in one sentence.")
  - Should be clear, specific, and aligned with the task type
  - Can include context, instructions, and any formatting requirements

- `expected_output_tokens`: Maximum expected output token count (example: 250)
  - Used for cost calculation and response size estimation

- `task`: Object containing task metadata that guides the evaluation process
  - `task_type`: Category of the task (example: "Summarization")
    - Common types include: Summarization, Question-Answering, Translation, Creative Writing, Code Generation, etc.
    - This categorization helps organize results by task type in the final report

  - `task_criteria`: Specific evaluation criteria for this task (example: "Summarize the text provided ensuring that the main message is not diluted nor changed")
    - Provides detailed instructions on how the response should be evaluated
    - Used by judge models to assess quality along specified dimensions
    - Should align with the core evaluation dimensions (Correctness, Completeness, etc.)

- `golden_answer`: Reference answer for evaluation (example: "Edge computing is a distributed computing paradigm that brings computation and data storage closer to the location where it is needed, improving response times and saving bandwidth by processing data near the source rather than relying on a central data center.")
  - Serves as the ground truth or ideal response
  - Used by judge models to compare model outputs
  - Critical for objective evaluation metrics
  - Should represent a high-quality answer that meets all task criteria

- `url_image` (optional): URL or path to an image for vision-enabled evaluations
  - Required when using `--vision_enabled true`
  - Can be a web URL or local file path
  - Supports JPG, PNG, GIF, WebP, and BMP formats

- `user_defined_metrics` (optional): Per-scenario custom metrics (example: "brand voice, technical accuracy")
  - Comma-separated list of evaluation criteria specific to this scenario
  - Overrides global `--user_defined_metrics` if specified

Example:
```json
{
  "text_prompt": "Summarize the principles of edge computing in one sentence.",
  "expected_output_tokens": 250,
  "task": {
    "task_type": "Summarization",
    "task_criteria": "Summarize the text provided ensuring that the main message is not diluted nor changed"
  },
  "golden_answer": "Edge computing is a distributed computing paradigm that brings computation and data storage closer to the location where it is needed, improving response times and saving bandwidth by processing data near the source rather than relying on a central data center."
}
```

Example with vision:
```json
{
  "text_prompt": "Describe what you see in this image.",
  "expected_output_tokens": 250,
  "task": {
    "task_type": "Image Description",
    "task_criteria": "Provide a detailed description of the image content"
  },
  "golden_answer": "A detailed description of the expected image content.",
  "url_image": "https://example.com/image.jpg"
}
```

## Model Profiles Data Format

The benchmarking tool requires model profiles in JSONL format, with each line containing a model identifier, region (for Bedrock models), and cost data.

### Field Descriptions

- `model_id`: Target model identifier
  - For Bedrock models: `"bedrock/us.amazon.nova-pro-v1:0"`
  - For OpenAI models: `"openai/gpt-4o"`
  - For Google models: `"gemini/gemini-2.0-flash"`

- `region`: AWS region for Bedrock models (example: "us-west-2")
  - **Required for Bedrock models only**
  - Region where the Bedrock model is available
  - Not required for third-party models (OpenAI, Google, Azure)

- `input_token_cost`: Cost per 1,000 input tokens (example: 0.0008)
  - Used for cost calculation and reporting

- `output_token_cost`: Cost per 1,000 output tokens (example: 0.0032)
  - Used for cost calculation and reporting

- `target_rpm` (optional): Target requests per minute for rate limiting (example: 60)
  - Used for reliability testing at specific load levels
  - When set, the framework will throttle requests to maintain this rate
  - Useful for identifying error rates and throttling thresholds
  - Set to `null` or omit the field for no rate limiting

- `service_tier` (optional): Inference service tier for Bedrock models (example: "priority")
  - **Bedrock models only** - controls the processing tier for model requests
  - Valid values: `"default"`, `"priority"`, or `"flex"`
    - `priority`: Higher priority processing with guaranteed capacity
    - `default`: Standard processing tier (used if not specified)
    - `flex`: Cost-optimized processing for batch workloads
  - Multiple tiers can be tested by creating separate entries with the same model_id
  - Each tier will appear as a separate model variant in results (e.g., `model-name_priority`)
  - Unsupported models will fall back to default tier automatically

Examples:

**Bedrock Model with Rate Limiting:**
```json
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032, "target_rpm": 60}
```

**Bedrock Model without Rate Limiting:**
```json
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032}
```

**Third-Party Model (OpenAI):**
```json
{"model_id": "openai/gpt-4o", "input_token_cost": 0.00125, "output_token_cost": 0.01}
```

**Third-Party Model (Google):**
```json
{"model_id": "gemini/gemini-2.0-flash", "input_token_cost": 0.00015, "output_token_cost": 0.0006}
```

**Bedrock Model with Service Tier (Priority):**
```json
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032, "service_tier": "priority"}
```

**Multiple Service Tiers for Same Model:**
```json
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032, "service_tier": "default"}
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032, "service_tier": "priority"}
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032, "service_tier": "flex"}
```

Sample model profiles are provided in:
- `config/models_profiles.jsonl` - Standard model configurations
- `config/models_profiles_service_tier_examples.jsonl` - Service tier examples

## Judge Configuration

Judges are required input data in JSONL format, with each line containing a judge model used to evaluate the models' responses. Currently, only Bedrock models are supported as judges.

### Field Descriptions

- `model_id`: Judge model identifier (example: "bedrock/us.amazon.nova-premier-v1:0")
  - Currently only supporting Bedrock Model Judges

- `region`: AWS region for Bedrock models (example: "us-east-1")
  - Region where the Bedrock model is available

- `input_cost_per_1k`: Input cost per one thousand tokens
  - Used for input pricing calculations

- `output_cost_per_1k`: Output cost per one thousand tokens
  - Used for output pricing calculations

Example:
```json
{"model_id": "bedrock/us.amazon.nova-premier-v1:0", "region": "us-east-2", "input_cost_per_1k": 0.0025, "output_cost_per_1k": 0.0125}
```

Sample judge profiles are provided in `config/judge_profiles.jsonl`.

### 📝 Vision Model Compatibility Important Notes

When using the vision functionality (`--vision_enabled true` flag or enabling "Vision Model" in the dashboard), ensure you're using a model that supports image inputs. Vision mode allows models to process both text prompts and images together.

1. **Error Handling**: If you send image content to a non-vision model, the evaluation will continue but the incorrectly configured evaluations will be stored in the logs as: "Model 'model_name' does not support vision capabilities"

2. **Image Formats**: Supported formats include JPG, PNG, GIF, WebP, and BMP

3. **Image Sources**: You can use either:
   - Local image files (will be automatically base64 encoded)
   - Web URLs pointing to images

4. **Model Selection**: Always verify your chosen model supports vision before enabling vision mode in your evaluation


## Usage

### Configuration Validation

Before running evaluations, validate your configuration files to catch errors early:

```bash
python src/config_validator.py config/
```

This will check:
- Model profiles for correct format and required fields
- Judge profiles for valid configuration
- Duplicate model IDs
- Cost values within reasonable ranges
- Region format for AWS models

### Model Capability Validation

**NEW:** Validate actual Bedrock model availability and service tier support by testing with real API calls. Results are cached to provide accurate service tier options in the dashboard.

```bash
# Validate all models (first-time setup or after adding new models)
python src/validate_model_capabilities.py

# Force re-validation (ignore existing cache)
python src/validate_model_capabilities.py --force

# Validate specific model
python src/validate_model_capabilities.py --model "bedrock/us.amazon.nova-2-lite-v1:0" --region "us-west-2"

# Test specific service tier
python src/validate_model_capabilities.py --model "bedrock/us.amazon.nova-2-lite-v1:0" --region "us-west-2" --tier "priority"
```

**What it does:**
- Tests each model+region combination with a minimal API request (~1 token input, 5 tokens output)
- Tests all service tiers (default, priority, flex) for each model
- Caches results in `.cache/model_capabilities.json`
- Automatically detects when `models_profiles.jsonl` changes and prompts re-validation
- Cost: ~$0.001 per full validation

**Benefits:**
- ✅ Accurate service tier availability (no guesswork)
- ✅ Dashboard shows only available tiers for each model
- ✅ Models unavailable in specific regions are greyed out
- ✅ Cached results avoid repeated API calls

### Streamlit Dashboard

Launch the interactive web dashboard for a user-friendly evaluation experience:

```bash
streamlit run src/streamlit_dashboard.py
```

The dashboard provides:
- **Setup Tab**: Configure evaluations, models, and advanced settings
  - **Model Configuration**: Select models and configure service tiers
    - For supported Bedrock models, check multiple service tier boxes to test different tiers
    - Each selected tier creates a separate model entry in results
    - Service tier selection appears below the model dropdown for compatible models
- **Monitor Tab**: Track running evaluations in real-time
- **Evaluations Tab**: View and analyze evaluation results
- **Reports Tab**: Browse and view generated HTML reports

#### Using Service Tiers in the Dashboard

1. Navigate to **Setup** → **Model Configuration** tab
2. Select a Bedrock model that supports service tiers (e.g., Amazon Nova, Claude 3.5+)
3. Check one or more service tier options (Default, Priority, Flex)
4. Click **Add Model** - this will create separate entries for each selected tier
5. Each tier variant will appear in results as `model-name_priority`, `model-name_flex`, etc.

### File Path Resolution

- **Input evaluation files**: Should be placed in the `runs/` directory. The tool will automatically look for input files in this directory.
- **Model profiles**: If not specified, defaults to `config/models_profiles.jsonl`
- **Judge profiles**: If not specified, defaults to `config/judge_profiles.jsonl`
- **Output files**: Saved to the directory specified by `--output_dir` (default: `outputs/`)
- **Unprocessed records**: Failed evaluations are logged in `outputs/unprocessed/`

### Running Benchmarks (CLI)

```bash
# Basic usage
# Note: Input files should be placed in the runs/ directory
python src/benchmarks_run.py input_file.jsonl
```

```bash
# Advanced usage with all options
python src/benchmarks_run.py input_file.jsonl \
    --output_dir benchmark_results \
    --parallel_calls 4 \
    --invocations_per_scenario 2 \
    --sleep_between_invocations 3 \
    --experiment_counts 2 \
    --experiment_name "My-Benchmark" \
    --experiment_wait_time 0 \
    --temperature_variations 2 \
    --user_defined_metrics "business writing style, brand adherence" \
    --model_file_name "models_profiles.jsonl" \
    --judge_file_name "judge_profiles.jsonl" \
    --evaluation_pass_threshold 3 \
    --report true \
    --vision_enabled false \
    --prompt_optimization_mode none
```

### Prompt Optimization Feature

Amazon Bedrock supports automatic prompt optimization for specific target models. This feature can improve prompt effectiveness by adapting them to each model's strengths.

**Available Modes:**

1. **`none`** (default): No prompt optimization, use original prompts
2. **`optimize_only`**: Replace all prompts with optimized versions
3. **`evaluate_both`**: Run evaluations with both original and optimized prompts for comparison

**Usage:**

```bash
# Optimize prompts for all Bedrock models
python src/benchmarks_run.py input_file.jsonl \
    --prompt_optimization_mode optimize_only

# Compare original vs optimized prompts
python src/benchmarks_run.py input_file.jsonl \
    --prompt_optimization_mode evaluate_both \
    --experiment_name "optimization-comparison"
```

**Output:**
- Optimization logs are saved to `outputs/prompt_optimization_log_<experiment_name>_<timestamp>.json`
- Shows which prompts were successfully optimized, skipped, or failed
- In `evaluate_both` mode, optimized variants are labeled with `_Prompt_Optimized` suffix in reports

**Notes:**
- Only works with Bedrock models (non-Bedrock models will use original prompts)
- Requires access to Amazon Bedrock's prompt optimization API
- Failed optimizations automatically fall back to original prompts

#### Command Line Arguments

- `input_file`: JSONL file with benchmark scenarios (required) - should be placed in `runs/` directory
- `--output_dir`: Directory to save results (default: "outputs")
- `--parallel_calls`: Number of parallel API calls (default: 4)
- `--invocations_per_scenario`: Invocations per scenario (default: 2)
- `--sleep_between_invocations`: Sleep time in seconds between invocations (default: 3)
- `--experiment_counts`: Number of experiment repetitions (default: 2)
- `--experiment_name`: Name for this benchmark run (default: "Benchmark-YYYYMMDD")
- `--experiment_wait_time`: Wait time in seconds between experiments (default: 0, no wait)
- `--temperature_variations`: Number of temperature variations (±25% percentile increments, default: 0)
- `--user_defined_metrics`: Comma-delimited user-defined evaluation metrics tailored to specific use cases
- `--model_file_name`: Name of the JSONL file containing the models to evaluate (defaults to `config/models_profiles.jsonl`)
- `--judge_file_name`: Name of the JSONL file containing the judges used to evaluate (defaults to `config/judge_profiles.jsonl`)
- `--evaluation_pass_threshold`: Threshold score used to determine Pass|Fail (default: 3 out of 5)
- `--report`: Generate HTML report after benchmarking (default: true)
- `--vision_enabled`: Enable vision model capabilities for image inputs (default: false)
- `--prompt_optimization_mode`: Prompt optimization mode - `none`, `optimize_only`, or `evaluate_both` (default: none)

### Visualizing Results

The benchmarking tool automatically generates interactive HTML reports when it completes. These reports can be found in the output directory specified (default: `outputs/`).

**⚠️ Report Generation Requirement:**
HTML report generation requires access to the `us.amazon.nova-premier-v1:0` model in your AWS account. This model is used to analyze evaluation results and create the report content. If this model is not accessible, evaluations will complete successfully but HTML reports will not be generated.

The reports include:
- **Performance comparisons**: Time to first token, tokens per second, cost per response
- **Latency and throughput metrics**: Aggregated statistics with percentile distributions
- **Success rate heatmaps**: Model performance across different task types
- **Bubble charts**: Multi-dimensional performance visualization
- **Radar charts**: Judge score breakdowns by evaluation dimension
- **Error analysis**: Treemap visualization of failure patterns
- **Regional performance**: Latency and cost analysis by AWS region with timezone awareness
- **Temperature analysis**: Performance metrics grouped by temperature settings (if enabled)
- **Statistical distributions**: Histogram overlays with normal distribution curves
- **Composite scoring**: Integrated performance tables with color-coded rankings


## Project Structure
- `assets/html_template.txt`: Web report template
- `assets/scale_icon.png`: Dashboard icon
- `config/`: Default configuration files
  - `models_profiles.jsonl`: Model configuration examples
  - `judge_profiles.jsonl`: Judge configuration examples
- `logs/`: Logs of evaluation sessions are stored here
- `outputs/`: Output directory for results and reports
  - `unprocessed/`: Records that failed to be evaluated
- `runs/`: Input directory for evaluation scenarios
- `src/`: Source code
  - `benchmarks_run.py`: Main benchmarking engine
  - `config_validator.py`: Configuration validation tool
  - `utils.py`: Utility functions for API interactions and data processing
  - `visualize_results.py`: Data visualization and reporting tools
  - `streamlit_dashboard.py`: Streamlit web dashboard
  - `dashboard/`: Dashboard components and utilities

## Requirements

- Python 3.12+
- Boto3
- Plotly
- Pandas
- LiteLLM
- Jinja2
- Dotenv
- Streamlit
- Scipy
- Pytz
- AWS account with Amazon Bedrock access and credentials configured
- Access to `us.amazon.nova-premier-v1:0` model (required for HTML report generation)

## Advanced Features

### Automatic Model Access Verification
Before running evaluations, the tool automatically verifies access to all configured models in parallel, providing immediate feedback on any permission issues.

### Retry Logic with Tracking
Failed API calls are automatically retried with exponential backoff. The `inference_request_count` metric tracks the number of retries per evaluation.

### Per-Scenario Metrics
You can override global user-defined metrics on a per-scenario basis by adding `"user_defined_metrics": "metric1, metric2"` to individual evaluation entries.

### Composite Scoring
The integrated analysis tables use composite scoring that combines:
- Success rate (quality)
- Latency (speed)
- Cost (efficiency)
- Throughput (tokens per second)

### Regional Performance Analysis
Reports include timezone-aware regional analysis showing:
- Performance by AWS region
- Time-of-day correlation with performance
- Average retry counts by region
- Composite scores for optimal region selection

### Rate Limiting & Reliability Testing
The framework supports configurable rate limiting (target RPM) for testing model reliability at specific load levels. This helps identify optimal production settings and throttling thresholds.

**How it works:**
- Configure `target_rpm` in model profiles to set requests per minute limit
- The framework uses a token bucket algorithm to throttle requests
- Rate limiting is applied per model-region combination
- Metrics are tracked for each throttled request

**Metrics tracked:**
- **Target RPM**: Configured requests per minute limit
- **Actual RPM**: Actual average rate achieved during evaluation
- **Throttle Events**: Number of times requests were delayed
- **Wait Time**: Total and average time spent waiting due to rate limiting
- **Success/Error Rates**: Model reliability at the configured RPM

**Use cases:**
- Test model reliability at expected production load
- Identify throttling thresholds before deployment
- Compare error rates across different RPM settings
- Optimize request rate for cost vs. throughput

**Example configuration:**
```json
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032, "target_rpm": 60}
```

**Viewing results:**
In the Streamlit dashboard, completed evaluations will display RPM metrics including target vs actual RPM, throttle events, and success/error rates for models configured with rate limiting.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
