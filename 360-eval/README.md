# LLM Benchmarking Framework with LLM-as-a-JURY

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange.svg)](https://aws.amazon.com/bedrock/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red.svg)](https://streamlit.io/)

## 📑 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
  - [UI Quick Start (Recommended)](#ui-quick-start-recommended)
  - [CLI Quick Start](#cli-quick-start)
- [Design Introduction](#design-introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Complete Dashboard Guide](#complete-dashboard-guide)
  - [Launching the Dashboard](#launching-the-dashboard)
  - [Step 1: Setting Up Your First Evaluation](#step-1-setting-up-your-first-evaluation)
  - [Step 2: Running Evaluations](#step-2-running-evaluations)
  - [Step 3: Analyzing Results](#step-3-analyzing-results)
  - [Step 4: Viewing Reports](#step-4-viewing-reports)
  - [Dashboard Workflow Example](#dashboard-workflow-example)
- [CLI Usage](#cli-usage)
  - [Configuration Validation](#configuration-validation)
  - [Model Capability Validation](#model-capability-validation)
  - [Running Benchmarks (CLI)](#running-benchmarks-cli)
  - [Command Line Arguments](#command-line-arguments)
- [Configuration Reference](#configuration-reference)
  - [Evaluation Unit Data Format](#evaluation-unit-data-format)
  - [Model Profiles Data Format](#model-profiles-data-format)
  - [Jury Configuration](#judge-configuration)
  - [Vision Model Compatibility](#vision-model-compatibility)
- [Advanced Features](#advanced-features)
  - [Prompt Optimization Feature](#prompt-optimization-feature)
  - [Latency-Only Evaluation Mode](#latency-only-evaluation-mode)
  - [Temperature Variations](#temperature-variations)
  - [Rate Limiting & Reliability Testing](#rate-limiting--reliability-testing)
  - [Service Tiers](#service-tiers)
  - [Regional Performance Analysis](#regional-performance-analysis)
- [Reports and Visualizations](#reports-and-visualizations)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Best Practices](#best-practices)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project provides tools for:

- **Running performance benchmarks** across multiple LLM models (Amazon Bedrock and third-party models)
- **Using LLM-as-a-jury methodology** where multiple judge models evaluate other models' responses
- **Optimizing prompts** for target models using Amazon Bedrock's prompt optimization API
- **Managing evaluations** through an interactive Streamlit web dashboard (recommended for most users)
- **Measuring key performance metrics** (latency, throughput, cost, response quality)
- **Visualizing results** and generating interactive HTML reports with regional performance analysis

---

## Features

### 🎯 Core Capabilities
- **Multi-model benchmarking**: Compare different models side-by-side, including Amazon Bedrock and third-party models (OpenAI, Google Gemini, Azure)
- **LLM-as-a-jury evaluation**: Leverage multiple LLM judges to evaluate model responses with aggregated scoring (1-5 scale)
- **Comprehensive metrics**: Track latency (TTFT, TTLB), throughput (tokens/sec), cost, and quality of responses
- **Interactive visualizations**: Generate holistic HTML reports with performance comparisons, heatmaps, radar charts, and error analysis

### 🖥️  Dashboard
- **Interactive web interface**: Launch a full-featured Streamlit dashboard for managing evaluations
- **Real-time monitoring**: Track running evaluations with live progress updates
- **Results visualization**: View and compare evaluation results directly in the browser
- **Model configuration**: Easily configure models, judges, and evaluation parameters
- **Report viewer**: Browse and view generated HTML reports
- **Configuration reuse**: Load previous configurations to run similar evaluations

### 🚀 Advanced Features
- **Vision model support**: Evaluate vision-capable models with image inputs (JPG, PNG, GIF, WebP, BMP)
- **User-defined metrics**: Add custom evaluation criteria beyond the six core dimensions
- **Configuration validation**: Pre-validate configuration files to catch errors before running evaluations
- **Regional performance analysis**: Analyze performance by AWS region with timezone-aware reporting
- **Unprocessed record tracking**: Automatically log failed evaluations for debugging
- **Service tier support**: Test Bedrock models across different service tiers (default, priority, flex)
- **Prompt optimization**: Use Amazon Bedrock's API to optimize compatible models prompts

---

## Quick Start

### UI Quick Start (Recommended)

**Get your first evaluation running in 5 minutes using the dashboard:**

```bash
# 1. Clone and navigate to the project
git clone <https://github.com/aws-samples/sample-bedrock-migration-and-modernization-tools.git>
cd 360-eval

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure AWS credentials (if not already configured)
aws configure

# 4. Launch the dashboard
streamlit run src/streamlit_dashboard.py
```

The dashboard will open at `http://localhost:8501`

**Now in the dashboard:**

1. **Setup Tab** → Upload your CSV with prompts and golden answers
2. **Select columns** for prompts and expected responses
3. **Add models** to evaluate and **judges** to assess responses
4. **Monitor Tab** → Add your evaluation to the execution queue
5. **Evaluations Tab** → View results when complete
6. **Reports Tab** → Explore interactive HTML reports

**Sample CSV format:**
```csv
prompt,golden_answer
"What is the capital of France?","Paris is the capital of France."
"Explain machine learning","Machine learning is a subset of artificial intelligence..."
```

---

### CLI Quick Start

**Run a benchmark from the command line:**

```bash
# 1. Clone and navigate to the project
git clone <https://github.com/aws-samples/sample-bedrock-migration-and-modernization-tools.git>
cd 360-eval

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure AWS credentials
aws configure

# 4. Create your evaluation data (JSONL format)
# Place your input file in the runs/ directory
cat > runs/my_first_benchmark.jsonl << 'EOF'
{"text_prompt": "What is the capital of France?", "expected_output_tokens": 50, "task": {"task_type": "Question-Answering", "task_criteria": "Provide accurate factual information"}, "golden_answer": "Paris is the capital of France."}
{"text_prompt": "Explain machine learning in one sentence.", "expected_output_tokens": 100, "task": {"task_type": "Summarization", "task_criteria": "Concise and accurate explanation"}, "golden_answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."}
EOF

# 5. Validate your configuration files (optional but recommended)
python src/config_validator.py config/

# 6. Run the benchmark
python src/benchmarks_run.py runs/my_first_benchmark.jsonl
```

**View results:**
- CSV results will be in `outputs/`
- HTML report will open automatically in your browser

---

## Design Introduction

A comprehensive framework for benchmarking and evaluating Large Language Models (LLMs) with a specific focus on Amazon Bedrock models. Features an intuitive Streamlit dashboard for easy evaluation management and interactive HTML reports for results analysis.

<img src="./assets/llm-as-judge-background.png" alt="LLM-as-a-Jury methodology diagram showing multiple judge models evaluating model responses across six dimensions" width="600" height="700">

This framework implements a "LLM-as-a-Jury" methodology based on research from [Replacing Jury with Juries: Evaluating LLM Generations with a Panel of Diverse Models](https://arxiv.org/pdf/2404.18796). Each evaluation uses multiple judge models that score responses on a scale of 1-5. The scores are tallied and averaged to calculate the final score. If the average score fails to achieve a threshold of 3+ across any evaluation dimension, the response is deemed to have failed (this threshold is configurable). This technique provides more reliable and balanced evaluations compared to single-judge PASS/FAIL methods.

### Six Core Evaluation Dimensions

Our benchmarking system evaluates model outputs across six core dimensions:
- **Correctness**: Accuracy of information provided
- **Completeness**: Thoroughness of the response
- **Relevance**: How well the response addresses the prompt
- **Format**: Appropriate structure and presentation
- **Coherence**: Logical flow and consistency
- **Following instructions**: Adherence to prompt requirements

Additionally, the framework supports **user-defined** evaluation metrics tailored to specific use cases such as branding style, tone, technical accuracy, or regulatory compliance.

The system is designed for scalability, automatically aggregating results from multiple benchmark runs into comprehensive reports, providing an increasingly complete picture of model latency and performance over time.

---

## Prerequisites

Before installing and using the framework, ensure you have:

### Required
- **Python 3.12 or higher** installed ([Download Python](https://www.python.org/downloads/))
- **AWS Account** with Amazon Bedrock access
- **AWS Credentials** configured on your machine
  ```bash
  aws configure
  # Enter your AWS Access Key ID, Secret Access Key, and default region
  ```
- **Bedrock Model Access**: Enable model access in the AWS Bedrock console
  - Go to AWS Console → Bedrock → Model access
  - Request access to the models you want to evaluate
  - **Required for reports**: `us.amazon.nova-premier-v1:0` (used for HTML report generation)

### Optional (for third-party models)
- **OpenAI API Key** (for GPT models)
- **Google API Key** (for Gemini models)
- **Azure API Key** (for Azure OpenAI models)

Create a `.env` file in the project root:
```env
OPENAI_API='your_openai_api_key_here'
GOOGLE_API='your_google_api_key_here'
AZURE_API_KEY='your_azure_api_key_here'
```

---

## Installation

```bash
# Clone the repository
git clone <https://github.com/aws-samples/sample-bedrock-migration-and-modernization-tools.git>
cd 360-eval

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python src/config_validator.py config/
```

**Key Dependencies:**
- Python 3.12+
- Boto3 (AWS SDK)
- Streamlit (Dashboard UI)
- LiteLLM (Multi-provider LLM client)
- Plotly (Visualizations)
- Pandas (Data processing)
- Jinja2 (Report templates)

---

## Dashboard User Guide

The Streamlit dashboard is the easiest way to use 360-Eval. It provides an interface for configuring, running, analyzing and tracking LLM benchmarks.

### Launching the Dashboard

```bash
# From the project root directory
streamlit run src/streamlit_dashboard.py
```

The dashboard will be available at `http://localhost:8501`

<img src="./assets/main_ui.png" alt="Main dashboard interface showing the 4-tab navigation: Setup, Monitor, Evaluations, and Reports" width="700" height="300">

**To stop the dashboard:**
- Press `Ctrl+C` in the terminal where the dashboard is running

---

### Step 1: Setting Up Your First Evaluation

#### 1.1 Navigate to Setup Tab

The Setup tab contains three sub-tabs:

<img src="./assets/model_config.png" alt="Setup tab showing the three configuration sub-tabs: Evaluation Setup, Model Configuration, and Advanced Configuration" width="700" height="300">

---

#### 🔧 Evaluation Setup Tab

**1. Name Your Evaluation**
- Enter a descriptive name (e.g., "Customer_Support_Bot_V2")
- This name identifies your results and reports

**2. Upload Your Dataset**
- Click "Upload CSV with prompts and golden answers"
- Your CSV should have at least two columns:
  - One for prompts (questions/inputs)
  - One for golden answers (expected responses)

**Example CSV structure:**
```csv
prompt,golden_answer
"What is the capital of France?","Paris"
"Explain machine learning","Machine learning is a subset of AI that enables systems to learn from data..."
"How do I reset my password?","Visit the login page and click 'Forgot Password' to receive a reset link."
```

<img src="./assets/csv_labeling.png" alt="CSV file upload interface with column selection dropdowns for prompt and golden answer columns" width="700" height="300">

**3. Select Data Columns**
- Choose the "Prompt Column" (contains your test questions)
- Choose the "Golden Answer Column" (contains expected responses)
- Preview your data to verify selections

**4. Vision Model Configuration** (Optional)
- Enable the "Vision Model" checkbox if testing vision-capable models
- Select the "Image Column" containing base64-encoded images or image file paths
- This allows evaluation of multimodal models that process both text and images

**5. Configure Multiple Task Evaluations**
- Use "Number of Task Evaluations" to create multiple tests
- For each task evaluation, specify:
  - **Task Type**: e.g., "Question-Answering", "Summarization", "Translation"
  - **Task Criteria**: Detailed evaluation instructions for judges
  - **Temperature**: Controls response creativity (0.01 = deterministic, 1.0 = very creative)
  - **User-Defined Metrics**: Optional custom criteria (e.g., "professional tone, brand consistency")

**Example Multi-Task Setup:**
- Task 1: Question-Answering, Temperature 0.3, Focus on factual accuracy
- Task 2: Creative Writing, Temperature 0.8, Focus on engagement and creativity
- Task 3: Brand Voice, Temperature 0.4, Custom metrics: "professional tone, empathetic language"

<img src="./assets/task_def.png" alt="Multiple task evaluation setup showing task type, criteria, temperature, and custom metrics configuration" width="700" height="300">

---

#### 🤖 Model Configuration Tab

**1. Select Models to Evaluate**
- Choose from available LLM models (Bedrock and third-party)
- Configure regions for Bedrock models
- Set cost parameters for each model (input/output token costs)
- Configure service tiers if applicable (default, priority, flex)

<img src="./assets/model_config.png" alt="Model configuration interface showing region selector, model dropdown, service tier options, and cost settings" width="700" height="300">

**2. Choose Jury Models**
- Select models that will evaluate the responses (judges/jurors)
- Jury assess quality based on your task criteria and the six core dimensions
- Can use different models as judges than those being evaluated
- Configure input/output costs for accurate cost tracking
- Recommended: Use 2-3 judge models for balanced evaluation

**3. Load Previous Configurations** (Optional)
- Use the "Load Config" button from completed evaluations to reuse settings
- This copies all parameters from a previous evaluation
- Useful for testing variations or running similar evaluations

**💡 Tip:** Start with 1-2 models and 2 judges for your first evaluation to understand the workflow before scaling up.

---

#### ⚙️ Advanced Configuration Tab

Fine-tune execution parameters:

<img src="./assets/advance_config.png" alt="Advanced configuration tab showing all parameter controls with sliders, ranges, and help text" width="700" height="300">

**Parameter Descriptions:**

- **Parallel API Calls**: Number of simultaneous API requests (default: 4, range: 1-20)
  - Higher values = faster execution but may hit rate limits
  - Start with 4 for most use cases

- **Invocations per Scenario**: How many times each prompt is tested (default: 2, range: 1-20)
  - Use 3-5 for reliable results with variance analysis
  - Higher values = more reliable metrics but longer runtime

- **Pass|Failure Threshold**: Score cutoff for pass/fail determination (default: 3, range: 2-4)
  - Based on 1-5 scoring scale
  - 3 = responses must score "good" or better to pass

- **Sleep Between Invocations**: Wait time between API calls in seconds (default: 3, range: 0-300)
  - Use 60-120 seconds for production APIs to avoid rate limits
  - Lower values = faster execution but higher risk of throttling

- **Experiment Counts**: Number of complete evaluation runs (default: 2, range: 1-10)
  - Use 1 for testing, 3-5 for production-grade benchmarks
  - Multiple runs help identify variance and reliability

- **Temperature Variations**: Test additional temperature settings automatically (default: 0, range: 0-5)
  - Adds ±25% temperature increments to your configured temperatures
  - Useful for understanding temperature impact on model behavior

- **Experiment Wait Time**: Pause between experiment runs (dropdown: 0 minutes to 3 hours)
  - Use for rate limit management or time-of-day analysis

- **Evaluation Type**:
  - **Full Evaluation (360)**: Complete quality and performance assessment
  - **Latency Only**: Skip judge evaluation, measure performance metrics only

**💾 Save Your Configuration**
Click "Save Configuration" to store your setup for execution.

**⚠️ Model Access Validation**
When you run an evaluation, the system will first check access to all selected models in parallel:
- ✅ Accessible models will be included in the evaluation
- ⚠️ If some models fail access check, evaluation continues with available models
- ❌ If no models are accessible, evaluation fails with clear error messages

---

### Step 2: Running Evaluations

#### 2.1 Navigate to Monitor Tab

The Monitor tab shows "Processing Evaluations" and execution controls.

<img src="./assets/monitor.png" alt="Monitor tab showing evaluation status table, queue status, and execution controls" width="700" height="400">

---

#### 2.2 Queue Your Evaluations

**1. Select Evaluations to Run**
- Use the dropdown to select from available (not yet processed) evaluations
- Only shows evaluations that haven't been completed, failed, or are currently running
- Multiple evaluations can be selected for batch processing

**2. Add to Execution Queue**
- Click "🚀 Add to Execution Queue"
- Evaluations run **sequentially** (one at a time) for stability and API quota management
- Monitor queue status and currently running evaluation in the status panel

---

#### 2.3 Monitor Progress

**Real-time Status Tracking:**
- **Queue Status**: Shows currently running and queued evaluations
- **Manual Refresh**: Click "Refresh Evaluations" to update status (dashboard uses manual refresh, not auto-refresh)
- **Progress Bars**: Real-time completion percentage for running evaluations
- **Log Monitoring**: Check the `logs/` directory for detailed progress and debugging

**💡 Tip:** For long-running evaluations, monitor the log files in the `logs/` directory for detailed progress:
```bash
tail -f logs/evaluation_status_<evaluation_id>.json
```

---

#### 2.4 Delete Evaluations

- **Select Evaluations**: Use multi-select to choose evaluations to remove
- **Delete Process**: Click "🗑️ Delete Selected Evaluations"
- **Confirmation**: Action removes evaluations from all lists and cleans up associated files

---

### Step 3: Analyzing Results

#### 3.1 Navigate to Evaluations Tab

View all completed evaluations with detailed information.

**Filter Options:**
- **All**: All evaluations regardless of status
- **Successful**: Only completed evaluations
- **Failed**: Only failed evaluations

<img src="./assets/evaluations.png" alt="Evaluations tab showing filter buttons and completed evaluations list with detailed metrics" width="700" height="400">

---

#### 3.2 Review Evaluation Data

The main table shows:
- **Name**: Evaluation identifier
- **Task Type**: What was being tested
- **Data File**: Original CSV filename used
- **Temperature**: Temperature setting used
- **Custom Metrics**: Whether custom metrics were applied
- **Models**: Number of models tested
- **Jury**: Number of judge models used
- **Completed**: Completion timestamp

---

#### 3.3 Detailed Analysis

**1. Select an Evaluation**: Choose from the dropdown

**2. Review Configuration**: See all parameters used, including:
- Basic info (task type, criteria, status, duration)
- Models evaluated (displayed as DataFrame with costs)
- Jury models (displayed as DataFrame with costs)
- Results files and configuration details

**3. Model Performance**: Analyze results by model and judge
- Compare scores across the six core dimensions
- Review custom metric assessments
- Analyze latency and cost metrics

**4. Error Analysis**: Check for any issues or failures
- View unprocessed records
- Review retry attempts and API errors

**5. Action Buttons:**
- **📋 Load Config**: Reuse this evaluation's settings for a new evaluation
- **🗑️ Delete**: Remove this evaluation and its associated files
- **📊 View Report**: Open associated HTML report

---

### Step 4: Viewing Reports

#### 4.1 Navigate to Reports Tab

Location for viewing HTML reports that are automatically generated when evaluations complete.

---

#### 4.2 Automatic Report Generation

**Important**: Reports are automatically created during the evaluation process, not manually generated:
- Each completed evaluation has an associated HTML report
- Reports are generated automatically when the benchmark process finishes
- You cannot manually create new reports - they are tied to completed evaluations

**⚠️ Report Generation Requirement:**
HTML report generation requires access to the `us.amazon.nova-premier-v1:0` model in your AWS account. This model is used to analyze evaluation results and create the report content. If this model is not accessible, evaluations will complete successfully but HTML reports will not be generated.

---

#### 4.3 View Reports

**1. Select Evaluation**: Choose from dropdown of completed evaluations that have reports

**2. View Report**: HTML reports display within the dashboard interface showing:
- **Performance charts**: Compare models across latency, throughput, and cost
- **Success rate heatmaps**: Model performance across different task types
- **Cost analysis**: Token usage and pricing breakdown
- **Error analysis**: Treemap visualization of failure patterns
- **Regional performance**: Latency and cost analysis by AWS region
- **Statistical distributions**: Histogram overlays with normal distribution curves

<img src="./assets/report.png" alt="Reports tab showing evaluation selector and embedded HTML report viewer with interactive charts" width="700" height="400">

---

#### 4.4 Report Access

- Reports are linked to their source evaluations
- If an evaluation is deleted, its report may also be removed
- Reports combine data from the evaluation's CSV output files
- Reports are stored in the `outputs/` directory

---

### Dashboard Workflow Example

#### Scenario: Testing a Customer Service Chatbot

**Step 1: Prepare Your Data**

Create a CSV file with customer service scenarios:
```csv
customer_query,expected_response_type
"How do I return a product?","Provide clear return policy steps with timeframe and conditions"
"What are your business hours?","State specific hours, timezone, and any holiday exceptions"
"I'm having trouble with my order","Show empathy, acknowledge the issue, and offer specific troubleshooting steps"
"Can I change my shipping address?","Explain the window for changes and provide clear instructions"
"What payment methods do you accept?","List all accepted payment methods clearly"
```

---

**Step 2: Configure Evaluation**

1. **Upload CSV** and select columns
   - Prompt column: `customer_query`
   - Golden answer column: `expected_response_type`

2. **Create One or Multiple Tasks**:
   - **Task 1**: "Accuracy" - Temperature 0.2 - "Provide factually correct information based on company policies"
   - **Task 2**: "Helpfulness" - Temperature 0.5 - "Be helpful, clear, and customer-friendly"
   - **Task 3**: "Brand Voice" - Temperature 0.4 - Custom metrics: "professional tone, empathetic language, brand consistency"

---

**Step 3: Select Models and Jury**

- **Models to Evaluate**:
  - `bedrock/us.amazon.nova-pro-v1:0` (us-west-2)
  - `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0` (us-east-1)
  - `openai/gpt-4o`

- **Jury Models**:
  - `bedrock/us.amazon.nova-premier-v1:0` (us-east-1)
  - `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0` (us-west-2)

- **Settings**:
  - 3 invocations per scenario
  - 2 experiment counts
  - 60 second sleep between invocations

---

**Step 4: Execute and Monitor**

1. Save configuration and go to Monitor tab
2. Add evaluation to queue
3. Monitor progress with status badges and progress bars
4. Check logs for detailed execution tracking

---

**Step 5: Analyze Results**

1. View completed evaluation in Evaluations tab
2. Compare model performance across the three tasks
3. Review temperature impact on response quality
4. Analyze custom metrics (professional tone, empathy, brand consistency)
5. Check cost and latency metrics for production feasibility

---

**Step 6: Review Reports**

1. Navigate to Reports tab
2. View automatically generated interactive HTML report
3. Analyze visualizations:
   - Which model best balances accuracy and helpfulness?
   - How does temperature affect brand voice consistency?
   - What are the cost/performance trade-offs?
4. Use insights to select the optimal model and configuration for production

---

## CLI Usage

While the dashboard is recommended for most users, the CLI provides powerful options for automation, CI/CD integration, and advanced use cases.
---
`

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

---

### Running Benchmarks (CLI)

**Basic usage:**
```bash
# Note: Input files should be placed in the runs/ directory
python src/benchmarks_run.py runs/input_file.jsonl
```

**Advanced usage with all options:**
```bash
python src/benchmarks_run.py runs/input_file.jsonl \
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
    --prompt_optimization_mode none \
    --latency_only_mode false
```

---

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input_file` | JSONL file with benchmark scenarios (required) - should be placed in `runs/` directory | - |
| `--output_dir` | Directory to save results | `"outputs"` |
| `--parallel_calls` | Number of parallel API calls | `4` (range: 1-20) |
| `--invocations_per_scenario` | Invocations per scenario | `2` (range: 1-20) |
| `--sleep_between_invocations` | Sleep time in seconds between invocations | `3` (range: 0-300) |
| `--experiment_counts` | Number of experiment repetitions | `2` (range: 1-10) |
| `--experiment_name` | Name for this benchmark run | `"Benchmark-YYYYMMDD-HHMMSS"` |
| `--experiment_wait_time` | Wait time in seconds between experiments | `0` |
| `--temperature_variations` | Number of temperature variations (±25% percentile increments) | `0` (range: 0-5) |
| `--user_defined_metrics` | Comma-delimited user-defined evaluation metrics | `""` |
| `--model_file_name` | Name of the JSONL file containing the models to evaluate | `"config/models_profiles.jsonl"` |
| `--judge_file_name` | Name of the JSONL file containing the judges used to evaluate | `"config/judge_profiles.jsonl"` |
| `--evaluation_pass_threshold` | Threshold score used to determine Pass\|Fail | `3` (range: 2-4) |
| `--report` | Generate HTML report after benchmarking | `true` |
| `--vision_enabled` | Enable vision model capabilities for image inputs | `false` |
| `--prompt_optimization_mode` | Prompt optimization mode: `none`, `optimize_only`, or `evaluate_both` | `"none"` |
| `--latency_only_mode` | Enable latency-only evaluation mode (skips judge evaluation) | `false` |

---

## Configuration Reference

### Evaluation Unit Data Format

The benchmarking tool requires input data in **JSONL format**, with each line containing a scenario to evaluate. Each scenario must follow this schema:

#### Field Descriptions

- **`text_prompt`**: The prompt to send to the model
  - Should be clear, specific, and aligned with the task type
  - Can include context, instructions, and any formatting requirements
  - Example: `"Summarize the principles of edge computing in one sentence."`

- **`expected_output_tokens`**: Maximum expected output token count
  - Used for cost calculation and response size estimation
  - Example: `250`

- **`task`**: Object containing task metadata that guides the evaluation process
  - **`task_type`**: Category of the task (e.g., "Summarization", "Question-Answering", "Translation", "Creative Writing", "Code Generation")
    - This categorization helps organize results by task type in the final report

  - **`task_criteria`**: Specific evaluation criteria for this task
    - Provides detailed instructions on how the response should be evaluated
    - Used by judge models to assess quality along specified dimensions
    - Should align with the core evaluation dimensions (Correctness, Completeness, etc.)
    - Example: `"Summarize the text provided ensuring that the main message is not diluted nor changed"`

- **`golden_answer`**: Reference answer for evaluation
  - Serves as the ground truth or ideal response
  - Used by judge models to compare model outputs
  - Critical for objective evaluation metrics
  - Should represent a high-quality answer that meets all task criteria
  - Example: `"Edge computing is a distributed computing paradigm that brings computation and data storage closer to the location where it is needed, improving response times and saving bandwidth by processing data near the source rather than relying on a central data center."`

- **`url_image`** (optional): URL or path to an image for vision-enabled evaluations
  - Required when using `--vision_enabled true`
  - Can be a web URL or local file path
  - Supports JPG, PNG, GIF, WebP, and BMP formats

- **`user_defined_metrics`** (optional): Per-scenario custom metrics
  - Comma-separated list of evaluation criteria specific to this scenario
  - Overrides global `--user_defined_metrics` if specified
  - Example: `"brand voice, technical accuracy, empathetic tone"`

---

#### Example Evaluation Unit (Basic)

<details>
<summary>Click to expand basic example</summary>

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
</details>

---

#### Example Evaluation Unit (Vision)

<details>
<summary>Click to expand vision example</summary>

```json
{
  "text_prompt": "Describe what you see in this image in detail.",
  "expected_output_tokens": 250,
  "task": {
    "task_type": "Image Description",
    "task_criteria": "Provide a detailed and accurate description of the image content, including objects, actions, and context"
  },
  "golden_answer": "A detailed description of the expected image content, including key objects, their arrangement, colors, and any text or notable features visible in the image.",
  "url_image": "https://example.com/image.jpg"
}
```
</details>

---

#### Example Evaluation Unit (Custom Metrics)

<details>
<summary>Click to expand custom metrics example</summary>

```json
{
  "text_prompt": "Write a professional email response to a customer complaint about a delayed shipment.",
  "expected_output_tokens": 300,
  "task": {
    "task_type": "Creative Writing",
    "task_criteria": "Write a professional, empathetic email that acknowledges the issue, apologizes, provides an explanation, and offers a solution"
  },
  "golden_answer": "Dear [Customer Name],\n\nI sincerely apologize for the delay in your shipment. We understand how frustrating this must be, especially when you were expecting your order by [date].\n\nThe delay was caused by [specific reason]. To make this right, we've expedited your shipment at no additional cost, and you should receive it by [new date]. Additionally, we'd like to offer you a [discount/credit] on your next purchase as a gesture of our commitment to your satisfaction.\n\nThank you for your patience and understanding.\n\nBest regards,\n[Company Name] Customer Service",
  "user_defined_metrics": "professional tone, empathetic language, brand voice consistency, solution-oriented"
}
```
</details>

---

### Model Profiles Data Format

The benchmarking tool requires model profiles in **JSONL format**, with each line containing a model identifier, region (for Bedrock models), and cost data.

#### Field Descriptions

- **`model_id`**: Target model identifier
  - For Bedrock models: `"bedrock/us.amazon.nova-pro-v1:0"`
  - For OpenAI models: `"openai/gpt-4o"`
  - For Google models: `"gemini/gemini-2.0-flash-exp"`
  - For Azure models: Follow LiteLLM Azure format

- **`region`**: AWS region for Bedrock models (example: `"us-west-2"`)
  - **Required for Bedrock models only**
  - Region where the Bedrock model is available
  - Not required for third-party models (OpenAI, Google, Azure)

- **`input_token_cost`**: Cost per 1,000 input tokens (example: `0.0008`)
  - Used for cost calculation and reporting

- **`output_token_cost`**: Cost per 1,000 output tokens (example: `0.0032`)
  - Used for cost calculation and reporting

- **`target_rpm`** (optional): Target requests per minute for rate limiting (example: `60`)
  - Used for reliability testing at specific load levels
  - When set, the framework will throttle requests to maintain this rate
  - Useful for identifying error rates and throttling thresholds
  - Set to `null` or omit the field for no rate limiting

- **`service_tier`** (optional): Inference service tier for Bedrock models (example: `"priority"`)
  - **Bedrock models only** - controls the processing tier for model requests
  - Valid values: `"default"`, `"priority"`, or `"flex"`
    - `priority`: Higher priority processing with guaranteed capacity
    - `default`: Standard processing tier (used if not specified)
    - `flex`: Cost-optimized processing for batch workloads
  - Multiple tiers can be tested by creating separate entries with the same model_id
  - Each tier will appear as a separate model variant in results (e.g., `model-name_priority`)
  - Unsupported models will fall back to default tier automatically

---

#### Model Profile Examples

<details>
<summary>Bedrock Model (Basic)</summary>

```json
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032}
```
</details>

<details>
<summary>Bedrock Model with Rate Limiting</summary>

```json
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032, "target_rpm": 60}
```
</details>

<details>
<summary>Bedrock Model with Service Tier</summary>

```json
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032, "service_tier": "priority"}
```
</details>

<details>
<summary>Multiple Service Tiers for Same Model</summary>

```json
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032, "service_tier": "default"}
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032, "service_tier": "priority"}
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032, "service_tier": "flex"}
```
</details>

<details>
<summary>OpenAI Model</summary>

```json
{"model_id": "openai/gpt-4o", "input_token_cost": 0.00250, "output_token_cost": 0.01}
```
</details>

<details>
<summary>Google Gemini Model</summary>

```json
{"model_id": "gemini/gemini-2.0-flash-exp", "input_token_cost": 0.0, "output_token_cost": 0.0}
```
</details>

**Sample model profiles are provided in:**
- `config/models_profiles.jsonl` - Standard model configurations
- `config/models_profiles_service_tier_examples.jsonl` - Service tier examples

---

### Jury Configuration

Jury are required input data in **JSONL format**, with each line containing a judge model used to evaluate the models' responses. Currently, only Bedrock models are supported as judges.

#### Field Descriptions

- **`model_id`**: Jury model identifier (example: `"bedrock/us.amazon.nova-premier-v1:0"`)
  - Currently only supporting Bedrock model judges

- **`region`**: AWS region for Bedrock models (example: `"us-east-1"`)
  - Region where the Bedrock model is available

- **`input_cost_per_1k`**: Input cost per one thousand tokens
  - Used for input pricing calculations

- **`output_cost_per_1k`**: Output cost per one thousand tokens
  - Used for output pricing calculations

#### Jury Profile Example

<details>
<summary>Click to expand example</summary>

```json
{"model_id": "bedrock/us.amazon.nova-premier-v1:0", "region": "us-east-2", "input_cost_per_1k": 0.0025, "output_cost_per_1k": 0.0125}
```
</details>

**Sample judge profiles are provided in:**
- `config/judge_profiles.jsonl`

---

### Vision Model Compatibility

#### 📝 Important Notes

When using the vision functionality (`--vision_enabled true` flag or enabling "Vision Model" in the dashboard), ensure you're using a model that supports image inputs. Vision mode allows models to process both text prompts and images together.

**Key Points:**

1. **Error Handling**: If you send image content to a non-vision model, the evaluation will continue but the incorrectly configured evaluations will be stored in the logs as: `"Model 'model_name' does not support vision capabilities"`

2. **Image Formats**: Supported formats include JPG, PNG, GIF, WebP, and BMP

3. **Image Sources**: You can use either:
   - Local image files (will be automatically base64 encoded)
   - Web URLs pointing to images

4. **Model Selection**: Always verify your chosen model supports vision before enabling vision mode in your evaluation

**Vision-Capable Models (Examples):**
- Amazon Nova Pro/Premier with vision support
- Claude 3.5 Sonnet
- GPT-4 Vision / GPT-4o
- Gemini 2.0 Flash

---

## Advanced Features

### Prompt Optimization Feature

Amazon Bedrock supports automatic prompt optimization for specific target models. This feature can improve prompt effectiveness by adapting them to each model's strengths.

#### Available Modes

1. **`none`** (default): No prompt optimization, use original prompts
2. **`optimize_only`**: Replace all prompts with optimized versions
3. **`evaluate_both`**: Run evaluations with both original and optimized prompts for comparison

#### Usage

**CLI:**
```bash
# Optimize prompts for all Bedrock models
python src/benchmarks_run.py runs/input_file.jsonl \
    --prompt_optimization_mode optimize_only

# Compare original vs optimized prompts
python src/benchmarks_run.py runs/input_file.jsonl \
    --prompt_optimization_mode evaluate_both \
    --experiment_name "optimization-comparison"
```

**Dashboard:**
- Navigate to **Setup** → **Advanced Configuration**
- Select prompt optimization mode from the dropdown
- Options: None, Optimize Only, Evaluate Both

#### Output

- Optimization logs are saved to `outputs/prompt_optimization_log_<experiment_name>_<timestamp>.json`
- Shows which prompts were successfully optimized, skipped, or failed
- In `evaluate_both` mode, optimized variants are labeled with `_Prompt_Optimized` suffix in reports

#### Notes

- Only works with Bedrock models (non-Bedrock models will use original prompts)
- Requires access to Amazon Bedrock's prompt optimization API
- Failed optimizations automatically fall back to original prompts
- Optimization adds slight latency to the evaluation setup phase

---

### Latency-Only Evaluation Mode

The framework supports a **latency-only evaluation mode** that skips the LLM judge evaluation and focuses exclusively on performance metrics. This mode is useful when you only need to measure inference speed, throughput, tokens, and cost without assessing response quality.

#### When to Use Latency-Only Mode

- Performance benchmarking and load testing
- Cost analysis across different models
- Token throughput comparison
- Initial model screening before full evaluation
- Testing infrastructure and API connectivity

#### How to Enable

**CLI:**
```bash
python src/benchmarks_run.py runs/input_file.jsonl --latency_only_mode true
```

**Dashboard:**
1. Navigate to the **Setup** tab
2. Under **Evaluation Type**, check the **"Latency Only Evaluation"** checkbox
3. Configure your evaluation as normal and run

#### What Happens in Latency-Only Mode

- ✅ Model inference runs normally (full response generated)
- ✅ All performance metrics collected (TTFT, TTLB, throughput, tokens, cost)
- ❌ Jury evaluation skipped (no accuracy assessment)
- 📊 CSV output contains `'N/A'` for all judge-related fields
- 📊 `eval_type` column shows `'latency'` (vs `'360'` for full evaluation)
- 📊 HTML reports show placeholder messages for accuracy-related charts
- 📊 Latency/cost/throughput charts display normally

#### CSV Output Structure

```csv
model_id,eval_type,time_to_first_byte,throughput_tps,response_cost,judge_success,judge_explanation,judge_scores
us.amazon.nova-pro-v1:0,latency,0.234,45.2,0.00012,N/A,N/A,{}
```

---

### Temperature Variations

Automatically test models across multiple temperature settings to understand the impact on response quality and creativity.

#### How It Works

- Set `--temperature_variations` to a value between 0-5
- The framework will test additional temperatures in ±25% increments around your base temperature
- Example: Base temperature 0.4 with 2 variations tests: 0.2, 0.4, 0.6

#### Usage

**CLI:**
```bash
python src/benchmarks_run.py runs/input_file.jsonl \
    --temperature_variations 2
```

**Dashboard:**
- Navigate to **Setup** → **Advanced Configuration**
- Set "Temperature Variations" slider (0-5)

#### Use Cases

- **Factual tasks**: Test low temperatures (0.1-0.3) for consistency
- **Creative tasks**: Test higher temperatures (0.7-0.9) for variety
- **Optimization**: Find the optimal temperature for your specific use case
- **Variance analysis**: Understand how temperature affects response stability

---

### Rate Limiting & Reliability Testing

The framework supports configurable rate limiting (target RPM) for testing model reliability at specific load levels. This helps identify optimal production settings and throttling thresholds.

#### How It Works

- Configure `target_rpm` in model profiles to set requests per minute limit
- The framework uses a token bucket algorithm to throttle requests
- Rate limiting is applied per model-region combination
- Metrics are tracked for each throttled request

#### Metrics Tracked

- **Target RPM**: Configured requests per minute limit
- **Actual RPM**: Actual average rate achieved during evaluation
- **Throttle Events**: Number of times requests were delayed
- **Wait Time**: Total and average time spent waiting due to rate limiting
- **Success/Error Rates**: Model reliability at the configured RPM

#### Use Cases

- Test model reliability at expected production load
- Identify throttling thresholds before deployment
- Compare error rates across different RPM settings
- Optimize request rate for cost vs. throughput
- Validate SLA compliance under load

#### Example Configuration

```json
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032, "target_rpm": 60}
```

#### Viewing Results

In the Streamlit dashboard, completed evaluations will display RPM metrics including target vs actual RPM, throttle events, and success/error rates for models configured with rate limiting.

---

### Service Tiers

Amazon Bedrock supports multiple service tiers for model inference, allowing you to balance cost, latency, and capacity guarantees.

#### Available Tiers

- **`default`**: Standard processing tier (used if not specified)
- **`priority`**: Higher priority processing with guaranteed capacity and potentially lower latency
- **`flex`**: Cost-optimized processing for batch workloads with flexible latency

#### Configuration

**Model Profile (JSONL):**
```json
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032, "service_tier": "priority"}
```

**Dashboard:**
1. Navigate to **Setup** → **Model Configuration** tab
2. Select a Bedrock model that supports service tiers (e.g., Amazon Nova, Claude 3.5+)
3. Check one or more service tier options (Default, Priority, Flex)
4. Click **Add Model** - this will create separate entries for each selected tier
5. Each tier variant will appear in results as `model-name_priority`, `model-name_flex`, etc.

#### Testing Multiple Tiers

Create separate entries with the same model_id but different service_tier values:
```json
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032, "service_tier": "default"}
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032, "service_tier": "priority"}
{"model_id": "bedrock/us.amazon.nova-pro-v1:0", "region": "us-west-2", "input_token_cost": 0.0008, "output_token_cost": 0.0032, "service_tier": "flex"}
```

Each tier will be evaluated separately and appear as distinct entries in reports.

#### Validation

Use the model capability validation tool to check service tier support:
```bash
python src/validate_model_capabilities.py --model "bedrock/us.amazon.nova-pro-v1:0" --region "us-west-2" --tier "priority"
```

---

### Regional Performance Analysis

Reports include timezone-aware regional analysis showing:

- **Performance by AWS region**: Latency, throughput, and cost metrics aggregated by region
- **Time-of-day correlation**: Performance variation based on when evaluations run
- **Average retry counts by region**: Reliability metrics per region
- **Composite scores**: Integrated scoring for optimal region selection

This helps identify the best AWS region for your specific use case based on latency, cost, and reliability requirements.

---

## Reports and Visualizations

The benchmarking tool automatically generates interactive HTML reports when evaluations complete. These reports can be found in the output directory specified (default: `outputs/`).

### Report Generation Requirement

**⚠️ Important:** HTML report generation requires access to the `us.amazon.nova-premier-v1:0` model in your AWS account. This model is used to analyze evaluation results and create the report content. If this model is not accessible, evaluations will complete successfully but HTML reports will not be generated.

### Report Contents

The reports include:

- **Performance comparisons**: Time to first token (TTFT), time to last byte (TTLB), tokens per second, cost per response
- **Latency and throughput metrics**: Aggregated statistics with percentile distributions (p50, p95, p99)
- **Success rate heatmaps**: Model performance across different task types
- **Bubble charts**: Multi-dimensional performance visualization (cost vs latency vs quality)
- **Radar charts**: Jury score breakdowns by evaluation dimension (Correctness, Completeness, Relevance, Format, Coherence, Following Instructions)
- **Error analysis**: Treemap visualization of failure patterns and error categorization
- **Regional performance**: Latency and cost analysis by AWS region with timezone awareness
- **Temperature analysis**: Performance metrics grouped by temperature settings (if temperature variations enabled)
- **Statistical distributions**: Histogram overlays with normal distribution curves
- **Composite scoring**: Integrated performance tables with color-coded rankings

### Accessing Reports

**Dashboard:**
- Navigate to **Reports** tab
- Select an evaluation from the dropdown
- View the embedded HTML report

**File System:**
- Reports are stored in `outputs/`
- File naming: `llm_benchmark_report_<timestamp>_<experiment_name>.html`
- Open directly in your browser

### CSV Output Files

In addition to HTML reports, the framework generates detailed CSV files:

- **Model responses**: Raw outputs from each tested model
- **Jury scores**: Numerical ratings (1-5) for each evaluation dimension
- **Latency data**: Response time measurements (TTFT, TTLB, throughput)
- **Cost tracking**: Token usage and pricing information
- **Error logs**: Any failures or issues encountered
- **Configuration data**: Parameters used for each test
- **Retry attempts**: Number of inference retries per evaluation

**CSV Location:** `outputs/invocations_<experiment_name>_<timestamp>.csv`

---

## Troubleshooting

### Common Issues and Solutions

<details>
<summary><strong>Issue: Evaluation gets stuck in "queued" status</strong></summary>

**Solution:**
- Check logs for API errors or rate limiting: `tail -f logs/evaluation_status_*.json`
- Verify AWS credentials are valid: `aws sts get-caller-identity`
- Check for background errors in the terminal running the dashboard

**Prevention:**
- Use appropriate sleep intervals between calls (60-120 seconds for production)
- Reduce parallel calls if hitting rate limits
- Monitor AWS Bedrock quotas in AWS Console
</details>

<details>
<summary><strong>Issue: CSV upload fails in dashboard</strong></summary>

**Solution:**
- Verify CSV format and column headers are correct
- Ensure file encoding is UTF-8 (not UTF-16 or other encodings)
- Check for special characters or malformed rows
- Validate CSV structure with a text editor or CSV validator

**Prevention:**
- Use standard CSV format (comma-separated, quoted strings for text with commas)
- Avoid blank rows or inconsistent column counts
- Test with a small sample CSV first
</details>

<details>
<summary><strong>Issue: Models not available or access denied</strong></summary>

**Solution:**
- Verify AWS credentials and region settings: `aws configure list`
- Ensure Bedrock model access is enabled in your AWS account (AWS Console → Bedrock → Model access)
- Check that you're using the correct region for the model
- Run model capability validation: `python src/validate_model_capabilities.py`

**Note:** The system performs parallel model access checks before evaluation starts and provides immediate feedback on model availability.

**Prevention:**
- Request model access in AWS Bedrock console before configuring evaluations
- Verify model availability in specific regions (some models are region-limited)
- Keep model profiles updated with correct region information
</details>

<details>
<summary><strong>Issue: Reports not generated or not available</strong></summary>

**Solution:**
- Reports are auto-generated, not manually created
- Ensure evaluation completed successfully (check status in Evaluations tab)
- Verify CSV output files are present in `outputs/` directory
- **Check access to `us.amazon.nova-premier-v1:0` model** (required for report generation)

**Note:** Each evaluation automatically creates its own report upon completion.

**Prevention:**
- Enable `us.amazon.nova-premier-v1:0` in AWS Bedrock model access
- Monitor evaluation completion status
- Check terminal/logs for report generation errors
</details>

<details>
<summary><strong>Issue: High costs or unexpected billing</strong></summary>

**Solution:**
- Review token usage in CSV output files
- Check configured models and their pricing in model profiles
- Verify `invocations_per_scenario` and `experiment_counts` settings (multiplied together)
- Monitor judge costs (judges evaluate each response)

**Prevention:**
- Start with small test datasets (5-10 prompts)
- Use fewer invocations per scenario for testing (1-2)
- Disable temperature variations for initial runs
- Use latency-only mode for cost-free quality assessments
</details>

<details>
<summary><strong>Issue: Slow evaluation performance</strong></summary>

**Solution:**
- Increase `--parallel_calls` for faster execution (watch for rate limits)
- Reduce `--sleep_between_invocations` if not hitting rate limits
- Use fewer judge models (2 judges vs 3-4)
- Consider latency-only mode if quality assessment not needed

**Performance Tips:**
- 4-8 parallel calls is optimal for most use cases
- Sleep intervals of 3-5 seconds work well for testing
- Sequential experiment execution prevents conflicts
</details>

<details>
<summary><strong>Issue: Vision model evaluation fails</strong></summary>

**Solution:**
- Verify model supports vision capabilities (e.g., Claude 3.5 Sonnet, GPT-4o, Nova Pro with vision)
- Check image format (JPG, PNG, GIF, WebP, BMP)
- Validate image URLs are accessible or local paths are correct
- Ensure images are properly base64 encoded if using local files

**Prevention:**
- Test with a known vision-capable model first
- Use small, standard-format images for testing
- Verify image column is correctly specified in dashboard
</details>

---

### Debug Tools

- **Sidebar Debug Panel**: Session information and log access in dashboard
- **Log Files**: Detailed execution logs in `logs/` directory
  - `evaluation_status_<evaluation_id>.json` - Real-time status updates
  - Timestamped logs for each evaluation
- **Status Files**: Check evaluation state in status JSON files
- **Manual Refresh**: Use refresh buttons in dashboard (no auto-refresh available)
- **Model Access Check**: Parallel validation provides immediate feedback on model availability
- **Unprocessed Records**: Check `outputs/unprocessed/` for failed evaluation records

---

### Performance Tips

- **Parallel Calls**: Adjust based on API rate limits
  - Start with 4, increase to 8-10 if no throttling occurs
  - Watch for HTTP 429 (Too Many Requests) errors
- **Sleep Intervals**: Increase if experiencing rate limiting
  - 60-120 seconds for production APIs
  - 3-5 seconds for testing with light load
- **Batch Size**: Process smaller evaluation sets for faster feedback
  - Start with 5-10 prompts to validate configuration
  - Scale up to full dataset once validated
- **Resource Monitoring**: Watch CPU and memory usage during execution
  - Dashboard runs in-process, monitor terminal for memory usage
  - Large evaluations may require significant memory for data processing

---

## FAQ

<details>
<summary><strong>Q: How much does it cost to run an evaluation?</strong></summary>

**A:** Cost depends on several factors:
- **Number of prompts** in your dataset
- **Number of models** being evaluated
- **Number of judge models** (they evaluate each response)
- **Invocations per scenario** (how many times each prompt is tested)
- **Experiment counts** (number of complete evaluation runs)
- **Token usage** (input + output tokens per model)
- **Model pricing** (varies by model and region)

**Example calculation:**
- 10 prompts
- 2 models being evaluated
- 2 judge models
- 3 invocations per scenario
- 2 experiment counts
- Average 500 tokens per evaluation (200 input + 300 output)

**Total evaluations:** 10 prompts × 2 models × 3 invocations × 2 experiments = **120 model inferences**
**Total judge evaluations:** 120 inferences × 2 judges = **240 judge evaluations**

**Estimated cost (using Amazon Nova Pro):**
- Model inference: 120 × 500 tokens × $0.0008 (per 1k tokens) ≈ **$0.05**
- Jury evaluation: 240 × 500 tokens × $0.0025 (per 1k tokens) ≈ **$0.30**
- **Total: ~$0.35**

💡 **Tip:** Start with a small test dataset (5-10 prompts) and use latency-only mode to estimate costs before full evaluation.
</details>

<details>
<summary><strong>Q: How long does an evaluation take?</strong></summary>

**A:** Duration depends on:
- **Dataset size** (number of prompts)
- **Number of models** being evaluated
- **Parallel calls** setting
- **Sleep between invocations** setting
- **Experiment counts** and **invocations per scenario**
- **Model latency** (varies by model and region)

**Example timing:**
- 50 prompts
- 3 models
- 4 parallel calls
- 3 second sleep between invocations
- 3 invocations per scenario
- 2 experiment counts

**Calculation:**
- Total inferences: 50 × 3 models × 3 invocations × 2 experiments = 900 inferences
- With 4 parallel: 900 / 4 = 225 sequential batches
- With 3 second sleep: 225 × 3 = 675 seconds ≈ **11 minutes** (plus model inference time)
- Typical model latency: 2-5 seconds per inference
- **Total estimated time: 20-40 minutes**

Jury evaluation adds additional time (runs after model inference completes).

💡 **Tip:** Use higher parallel calls (8-10) and lower sleep intervals (0-3 seconds) for testing to reduce runtime.
</details>

<details>
<summary><strong>Q: Should I use the dashboard or CLI?</strong></summary>

**A:**
- **Use the dashboard** (recommended) if:
  - You want an intuitive visual interface
  - You need to configure evaluations without writing code
  - You prefer real-time monitoring and visualization
  - You're new to the framework
  - You want to easily reuse previous configurations

- **Use the CLI** if:
  - You need to automate evaluations in CI/CD pipelines
  - You're scripting batch evaluations
  - You want programmatic control
  - You're running on a headless server
  - You prefer command-line workflows

Both use the same underlying evaluation engine and produce identical results.
</details>

<details>
<summary><strong>Q: How do I choose which models to use as judges?</strong></summary>

**A:** Best practices for judge selection:
- **Use 2-3 judge models** for balanced evaluation (more judges = more reliable but higher cost)
- **Use capable models** as judges (e.g., Claude 3.5 Sonnet, Nova Premier, GPT-4)
- **Vary judge models** to reduce bias (don't use the same model as judge and evaluated model)
- **Consider cost** (judges evaluate every response, so costs multiply)
- **Match capability to task** (use vision-capable judges for vision evaluations)

**Recommended judge combinations:**
- Budget-conscious: 2 judges (e.g., Nova Pro + Claude 3 Haiku)
- Balanced: 3 judges (e.g., Nova Premier + Claude 3.5 Sonnet + Nova Pro)
- High-accuracy: 3-4 judges with diverse models

Currently, only **Bedrock models** are supported as judges.
</details>

<details>
<summary><strong>Q: What's the difference between full (360) and latency-only evaluation?</strong></summary>

**A:**

**Full (360) Evaluation:**
- Evaluates **quality and performance**
- Uses judge models to assess response quality across 6 dimensions
- Provides detailed scoring and analysis
- Higher cost (includes judge evaluation costs)
- Longer runtime (judge evaluation adds time)
- Best for comprehensive model comparison

**Latency-Only Evaluation:**
- Evaluates **performance metrics only**
- No judge models used (no quality assessment)
- Measures TTFT, TTLB, throughput, tokens, cost
- Lower cost (no judge evaluation)
- Faster runtime
- Best for infrastructure testing, cost analysis, initial screening

**Use latency-only for:**
- Quick performance benchmarks
- Cost estimation
- Infrastructure validation
- Initial model screening

**Use full (360) for:**
- Comprehensive model comparison
- Quality assessment
- Production model selection
- Compliance validation
</details>

<details>
<summary><strong>Q: Can I evaluate non-Bedrock models?</strong></summary>

**A:** **Yes!** The framework supports:
- **Amazon Bedrock models** (native support)
- **OpenAI models** (GPT-3.5, GPT-4, GPT-4o, etc.)
- **Google Gemini models** (Gemini Pro, Gemini 2.0 Flash, etc.)
- **Azure OpenAI models**

**Setup for third-party models:**
1. Create a `.env` file in the project root
2. Add your API keys:
   ```env
   OPENAI_API='your_openai_api_key'
   GOOGLE_API='your_google_api_key'
   AZURE_API_KEY='your_azure_api_key'
   ```
3. Add model profiles in `config/models_profiles.jsonl` using the format:
   - OpenAI: `"openai/gpt-4o"`
   - Google: `"gemini/gemini-2.0-flash-exp"`

**Note:** Jury models currently only support Bedrock models.
</details>

<details>
<summary><strong>Q: How do I interpret the evaluation scores?</strong></summary>

**A:** The framework uses a **1-5 scoring scale** for each of the six evaluation dimensions:

**Score meanings:**
- **5**: Excellent - Exceeds expectations
- **4**: Good - Meets expectations well
- **3**: Acceptable - Meets minimum requirements (default pass threshold)
- **2**: Poor - Below expectations
- **1**: Very Poor - Fails to meet requirements

**Pass/Fail determination:**
- Default threshold: **3** (configurable)
- A response passes if it scores ≥ threshold across **all six dimensions**
- Jury vote independently, scores are averaged
- If average score < threshold on any dimension, the response fails

**Example:**
- Correctness: 4.5
- Completeness: 4.0
- Relevance: 4.5
- Format: 4.0
- Coherence: 4.5
- Following Instructions: 2.5 ← **Below threshold**
- **Result: FAIL** (one dimension below threshold)

**Composite scoring** in reports combines quality (scores), latency, and cost for holistic comparison.
</details>

<details>
<summary><strong>Q: How do I run evaluations in parallel?</strong></summary>

**A:** The framework supports **two types of parallelism**:

**1. Within-evaluation parallelism (recommended):**
- Set `--parallel_calls` to control simultaneous API requests
- Default: 4, Range: 1-20
- Higher values = faster execution but may hit rate limits
- Example: `--parallel_calls 8`

**2. Across-evaluation parallelism (dashboard only):**
- **Not currently supported** - evaluations run sequentially
- This prevents conflicts and ensures reliable execution
- Queue multiple evaluations in Monitor tab - they'll run one after another

**Best practice:**
- Use high `--parallel_calls` (4-10) for speed
- Use appropriate `--sleep_between_invocations` (3-120 seconds) to avoid rate limits
- Balance speed vs rate limits based on your API quotas
</details>

<details>
<summary><strong>Q: What happens if an evaluation fails?</strong></summary>

**A:** The framework has robust error handling:

**Partial failures:**
- Individual model failures don't stop the entire evaluation
- Failed evaluations are logged in `outputs/unprocessed/`
- The system retries failed API calls automatically (with exponential backoff)
- Successful models complete normally
- Reports show which models failed and why

**Complete failures:**
- Evaluation status changes to "Failed"
- Error details logged in `logs/evaluation_status_*.json`
- No HTML report generated if evaluation didn't complete
- CSV output may be partially written with successful results

**Recovery:**
- Check logs for specific error messages
- Fix the issue (e.g., enable model access, update credentials)
- Re-run the evaluation

**Prevention:**
- Run `python src/validate_model_capabilities.py` before evaluations
- Validate configurations with `python src/config_validator.py config/`
- Start with small test datasets to catch issues early
- Monitor AWS Bedrock quotas and rate limits
</details>

---

## Best Practices

### Data Preparation

- **Clear prompts**: Write specific, unambiguous test prompts that represent real-world use cases
- **Quality golden answers**: Provide detailed, accurate expected responses that judges can compare against
- **Balanced dataset**: Include various difficulty levels, task types, and edge cases
- **Consistent format**: Maintain uniform CSV/JSONL structure across all evaluation data
- **Representative samples**: Ensure your test data reflects actual production scenarios

### Evaluation Configuration

- **Start small**: Begin with 1-2 models and simple criteria to understand the workflow
- **Iterative approach**: Add complexity gradually (more models, more judges, more custom metrics)
- **Temperature testing**: Use different temperatures for different task types
  - 0.1-0.3: Factual, deterministic tasks
  - 0.4-0.6: Balanced tasks
  - 0.7-0.9: Creative tasks
- **Multiple judges**: Use 2-3 judge models for reliable, unbiased assessment
- **Validate first**: Run `config_validator.py` and `validate_model_capabilities.py` before evaluations

### Execution Management

- **Sequential processing**: Let evaluations run one at a time (dashboard enforces this)
- **Monitor resources**: Watch for API rate limits, AWS quotas, and costs
- **Log review**: Check logs for detailed progress, errors, and debugging information
- **Patience**: Large evaluations can take significant time - plan accordingly
- **Test configurations**: Use small datasets to validate settings before full runs

### Cost Management

- **Start with latency-only mode**: Validate infrastructure and costs before adding judge evaluation
- **Use fewer invocations**: 1-2 invocations for testing, 3-5 for production
- **Limit experiment counts**: Use 1 for testing, 2-3 for production
- **Monitor token usage**: Review CSV outputs to understand token consumption patterns
- **Choose the jury wisely**: Jurer costs multiply (each judge evaluates every response)

### Report Usage

- **Automatic generation**: Reports are created automatically for each evaluation - no manual action needed
- **Immediate access**: Reports available as soon as evaluation completes
- **Evaluation-specific**: Each report corresponds to exactly one evaluation
- **Comparative analysis**: Use reports to compare models across multiple dimensions
- **Historical tracking**: Keep reports for longitudinal analysis of model performance over time

---

## Project Structure

```
360-eval/
├── assets/
│   ├── html_template.txt           # Web report template
│   ├── llm-as-judge-background.png # Methodology diagram
│   ├── scale_icon.png              # Dashboard icon
│   └── *.png                       # Dashboard screenshots
├── config/
│   ├── models_profiles.jsonl                        # Model configuration examples
│   ├── models_profiles_service_tier_examples.jsonl # Service tier examples
│   └── judge_profiles.jsonl                        # Jurer configuration examples
├── logs/                            # Logs of evaluation sessions
│   └── evaluation_status_*.json    # Real-time status updates
├── outputs/                         # Output directory for results and reports
│   ├── invocations_*.csv           # Detailed evaluation results
│   ├── llm_benchmark_report_*.html # Interactive HTML reports
│   └── unprocessed/                # Records that failed to be evaluated
├── runs/                            # Input directory for evaluation scenarios (JSONL files)
├── src/
│   ├── benchmarks_run.py           # Main benchmarking engine
│   ├── config_validator.py         # Configuration validation tool
│   ├── validate_model_capabilities.py # Model availability and service tier validation
│   ├── utils.py                    # Utility functions for API interactions and data processing
│   ├── visualize_results.py        # Data visualization and reporting tools
│   ├── streamlit_dashboard.py      # Streamlit web dashboard
│   └── dashboard/                  # Dashboard components and utilities
├── .cache/                          # Cache for model capability validation
│   └── model_capabilities.json
├── .env                             # Third-party API keys (create this file)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Requirements

### System Requirements
- **Python 3.12 or higher**
- **Operating System**: macOS, Linux, Windows (with WSL recommended)
- **Memory**: 4GB+ RAM recommended for large evaluations
- **Disk Space**: 1GB+ for outputs and reports

### Python Dependencies
- **Boto3**: AWS SDK for Python (Bedrock API access)
- **Plotly**: Interactive visualizations and charts
- **Pandas**: Data processing and analysis
- **LiteLLM**: Multi-provider LLM client (Bedrock, OpenAI, Google, Azure)
- **Jinja2**: HTML report templating
- **Python-Dotenv**: Environment variable management
- **Streamlit**: Dashboard web interface
- **Scipy**: Statistical analysis
- **Pytz**: Timezone handling for regional analysis

### AWS Requirements
- **AWS Account** with active credentials
- **Amazon Bedrock access** enabled in your account
- **Model access** enabled for:
  - Models you want to evaluate
  - Models you want to use as judges
  - **`us.amazon.nova-premier-v1:0`** (required for HTML report generation)
- **IAM Permissions**: `bedrock:InvokeModel`, `bedrock:InvokeModelWithResponseStream`

### Optional Requirements
- **OpenAI API key** (for GPT models)
- **Google API key** (for Gemini models)
- **Azure API key** (for Azure OpenAI models)

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

**How to contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Areas where contributions are especially welcome:**
- Additional judge model providers (currently Bedrock only)
- New visualization types for reports
- Performance optimizations
- Documentation improvements
- Bug fixes and error handling improvements
- Support for additional LLM providers
- CI/CD integration examples

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Questions or issues?** Please open an issue on GitHub or contact the maintainers.

**Documentation:** This README is comprehensive, but for specific implementation details, refer to the code comments and inline documentation.

**Updates:** This framework is actively maintained. Check the repository for the latest features and improvements.