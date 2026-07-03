import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parents[2]))

# Project root directory - configurable via env var for containerized deployments
PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', str(Path(os.path.abspath(__file__)).parents[3])))

# Default directories - configurable via env vars for S3/container deployments
DEFAULT_OUTPUT_DIR = os.environ.get('OUTPUT_DIR', str(PROJECT_ROOT / "outputs"))
DEFAULT_PROMPT_EVAL_DIR = os.environ.get('PROMPT_EVAL_DIR', str(PROJECT_ROOT / "runs"))
CONFIG_DIR = os.environ.get('CONFIG_DIR', str(PROJECT_ROOT / "config"))
LOGS_DIR = os.environ.get('LOGS_DIR', str(PROJECT_ROOT / "logs"))
STATUS_FILES_DIR = os.environ.get('STATUS_FILES_DIR', str(PROJECT_ROOT / "logs"))

def get_config_path(filename):
    """Get absolute path to a config file"""
    return os.path.join(CONFIG_DIR, filename)

def _read_jsonl_lines(filename):
    """Read JSONL lines from local config dir, falling back to S3 if not found locally."""
    file_path = get_config_path(filename)

    # Try local file first
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.readlines()

    # Fall back to S3
    s3_bucket = os.environ.get('S3_BUCKET')
    if s3_bucket:
        try:
            import boto3
            s3 = boto3.client('s3', region_name=os.environ.get('AWS_REGION', 'us-east-1'))
            resp = s3.get_object(Bucket=s3_bucket, Key=f'config/{filename}')
            content = resp['Body'].read().decode('utf-8')
            # Cache locally for subsequent reads
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"[CONFIG] Downloaded {filename} from S3 and cached locally")
            return content.splitlines()
        except Exception as e:
            print(f"[WARNING] Failed to read {filename} from S3: {e}")

    raise FileNotFoundError(f"'{filename}' not found locally at {file_path} or in S3")


def generate_model_info(filename='models_profiles.jsonl'):
    """
    Load model information from config files.
    Reads from local config dir first, falls back to S3 bucket.
    """
    try:
        lines = _read_jsonl_lines(filename)

        # Initialize empty structures
        bedrock_models = []
        openai_models = []
        cost_map = {}
        model_to_regions = {}
        region_to_models = {}
        model_service_tiers = {}  # (model_id, region) -> list of tiers
        model_tier_pricing = {}   # (model_id, region) -> {tier: {input, output}}
        model_endpoint = {}       # (model_id, region) -> {endpoint, mantle_region}

        # Process JSONL lines
        for line in lines:
                try:
                    data = json.loads(line)
                    model_id = data['model_id']
                    region = None
                    if 'region' in data and 'bedrock/' in model_id:
                        region = data['region']
                    elif 'region' in data and 'bedrock/' not in model_id:
                        region = "N/A"
                    # Categorize models based on prefix
                    if 'bedrock/' not in model_id:
                        openai_models.append([model_id, region])
                    else:
                        bedrock_models.append([model_id, region])
                        
                        # Build region/model mappings for Bedrock models
                        if region and region != "N/A":
                            # Add to model_to_regions mapping
                            if model_id not in model_to_regions:
                                model_to_regions[model_id] = []
                            if region not in model_to_regions[model_id]:
                                model_to_regions[model_id].append(region)
                            
                            # Add to region_to_models mapping
                            if region not in region_to_models:
                                region_to_models[region] = []
                            if model_id not in region_to_models[region]:
                                region_to_models[region].append(model_id)

                    # Build service tiers map
                    if 'service_tiers' in data and region and region != "N/A":
                        model_service_tiers[(model_id, region)] = data['service_tiers']

                    # Build tier pricing map
                    if 'tier_pricing' in data and region and region != "N/A":
                        model_tier_pricing[(model_id, region)] = data['tier_pricing']

                    # Build endpoint map (Mantle / Responses-API models). Carried into
                    # per-eval profiles so the engine routes them correctly.
                    if data.get('endpoint') and region and region != "N/A":
                        model_endpoint[(model_id, region)] = {
                            "endpoint": data['endpoint'],
                            "mantle_region": data.get('mantle_region', region),
                        }

                    # Build cost map entry keyed by (model_id, region) for per-region pricing
                    input_cost_key = 'input_token_cost' if 'input_token_cost' in data else 'input'
                    output_token_key = 'output_token_cost' if 'output_token_cost' in data else 'output'
                    cost_entry = {
                        "input": data[input_cost_key],
                        "output": data[output_token_key]
                    }
                    # Per-region cost map
                    if region and region != "N/A":
                        cost_map[(model_id, region)] = cost_entry
                    else:
                        cost_map[(model_id, "N/A")] = cost_entry
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line}")
                except KeyError as e:
                    print(f"Warning: Missing key in data: {e} for line: {line}")

        # Return the generated structures
        return {
            "DEFAULT_BEDROCK_MODELS": bedrock_models,
            "DEFAULT_OPENAI_MODELS": openai_models,
            "DEFAULT_COST_MAP": cost_map,
            "MODEL_TO_REGIONS": model_to_regions,
            "REGION_TO_MODELS": region_to_models,
            "MODEL_SERVICE_TIERS": model_service_tiers,
            "MODEL_TIER_PRICING": model_tier_pricing,
            "MODEL_ENDPOINT": model_endpoint,
        }

    except FileNotFoundError as fnf:
        print(f"Error: {fnf}")
        return {"DEFAULT_BEDROCK_MODELS": [], "DEFAULT_OPENAI_MODELS": [], "DEFAULT_COST_MAP": {}, "MODEL_TO_REGIONS": {}, "REGION_TO_MODELS": {}, "MODEL_SERVICE_TIERS": {}, "MODEL_TIER_PRICING": {}}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"DEFAULT_BEDROCK_MODELS": [], "DEFAULT_OPENAI_MODELS": [], "DEFAULT_COST_MAP": {}, "MODEL_TO_REGIONS": {}, "REGION_TO_MODELS": {}, "MODEL_SERVICE_TIERS": {}}

"""Constants for the Streamlit dashboard."""

# App title and information
APP_TITLE = "LLM Benchmarking Dashboard"
SIDEBAR_INFO = """
### LLM Benchmarking Dashboard

This dashboard provides an intuitive interface for:
- Setting up evaluations from CSV files
- Configuring model parameters
- Selecting judge models
- Monitoring evaluation progress
- Viewing results and reports

For more details, see the [README.md](https://github.com/aws-samples/sample-bedrock-migration-and-modernization-tools/tree/main/360-eval)
"""

# Evaluation parameters
DEFAULT_PARALLEL_CALLS = 4
DEFAULT_INVOCATIONS_PER_SCENARIO = 3
DEFAULT_SLEEP_BETWEEN_INVOCATIONS = 3
DEFAULT_EXPERIMENT_COUNTS = 1
DEFAULT_TEMPERATURE_VARIATIONS = 0
DEFAULT_FAILURE_THRESHOLD = 3

# Default model regions
AWS_REGIONS = [
        # North America
        'us-east-1',  # N. Virginia
        'us-east-2',  # Ohio
        'us-west-1',  # N. California
        'us-west-2',  # Oregon

        # Africa
        'af-south-1',  # Cape Town

        # Asia Pacific
        'ap-east-1',  # Hong Kong
        'ap-south-2',  # Hyderabad
        'ap-southeast-3',  # Jakarta
        'ap-southeast-5',  # Malaysia
        'ap-southeast-4',  # Melbourne
        'ap-south-1',  # Mumbai
        'ap-northeast-3',  # Osaka
        'ap-northeast-2',  # Seoul
        'ap-southeast-1',  # Singapore
        'ap-southeast-2',  # Sydney
        'ap-southeast-7',  # Thailand
        'ap-northeast-1',  # Tokyo

        # Canada
        'ca-central-1',  # Central
        'ca-west-1',  # Calgary

        # Europe
        'eu-central-1',  # Frankfurt
        'eu-west-1',  # Ireland
        'eu-west-2',  # London
        'eu-south-1',  # Milan
        'eu-west-3',  # Paris
        'eu-south-2',  # Spain
        'eu-north-1',  # Stockholm
        'eu-central-2',  # Zurich

        # Israel
        'il-central-1',  # Tel Aviv

        # Mexico
        'mx-central-1',  # Central

        # Middle East
        'me-south-1',  # Bahrain
        'me-central-1',  # UAE

        # South America
        'sa-east-1',  # São Paulo

        # AWS GovCloud
        'us-gov-east-1',  # US-East
        'us-gov-west-1',  # US-West
    ]


# Load model data
defaults = generate_model_info('models_profiles.jsonl')
DEFAULT_BEDROCK_MODELS = defaults['DEFAULT_BEDROCK_MODELS']
DEFAULT_OPENAI_MODELS = defaults['DEFAULT_OPENAI_MODELS']
DEFAULT_COST_MAP = defaults['DEFAULT_COST_MAP']
MODEL_TO_REGIONS = defaults['MODEL_TO_REGIONS']
REGION_TO_MODELS = defaults['REGION_TO_MODELS']
MODEL_SERVICE_TIERS = defaults['MODEL_SERVICE_TIERS']

# Load judge data
judges = generate_model_info('judge_profiles.jsonl')
DEFAULT_JUDGES = judges['DEFAULT_BEDROCK_MODELS']
DEFAULT_JUDGES_COST = judges['DEFAULT_COST_MAP']
JUDGE_MODEL_TO_REGIONS = judges['MODEL_TO_REGIONS']
JUDGE_REGION_TO_MODELS = judges['REGION_TO_MODELS']
