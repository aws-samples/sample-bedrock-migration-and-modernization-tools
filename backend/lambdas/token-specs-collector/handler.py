"""
Token Specs Collector Lambda

Fetches token specifications (context window, max output) from LiteLLM.
Works with the correct snake_case schema.
"""

import json
import logging
import os
import time
from typing import Any
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import boto3
from botocore.config import Config

logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

RETRY_CONFIG = Config(
    retries={'max_attempts': 3, 'mode': 'adaptive'},
    connect_timeout=10,
    read_timeout=30
)

# LiteLLM model database URL
LITELLM_MODEL_DB_URL = 'https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json'


def get_s3_client():
    return boto3.client('s3', config=RETRY_CONFIG)


def read_from_s3(s3_client: Any, bucket: str, key: str) -> dict:
    """Read JSON data from S3."""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response['Body'].read().decode('utf-8'))


def write_to_s3(s3_client: Any, bucket: str, key: str, data: dict) -> None:
    """Write JSON data to S3."""
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2, default=str),
        ContentType='application/json'
    )
    logger.info(f"Written to s3://{bucket}/{key}")


def fetch_litellm_data() -> dict:
    """Fetch model data from LiteLLM GitHub repository."""
    try:
        request = Request(LITELLM_MODEL_DB_URL, headers={
            'User-Agent': 'BedrockProfiler/1.0',
            'Cache-Control': 'no-cache, no-store',
            'Pragma': 'no-cache'
        })
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            logger.info(f"Fetched {len(data)} models from LiteLLM")
            return data
    except (URLError, HTTPError) as e:
        logger.error(f"Failed to fetch LiteLLM data: {e}")
        return {}


def filter_bedrock_models(litellm_data: dict) -> dict:
    """Filter LiteLLM data to only include Bedrock models."""
    bedrock_models = {}

    for model_key, model_data in litellm_data.items():
        # Check if it's a Bedrock model
        # Include models with 'bedrock' in key or provider containing 'bedrock'
        litellm_provider = model_data.get('litellm_provider', '')
        is_bedrock = (
            'bedrock' in model_key.lower() or
            'bedrock' in litellm_provider.lower() if litellm_provider else False
        )
        if is_bedrock:
            bedrock_models[model_key] = model_data

    logger.info(f"Filtered {len(bedrock_models)} Bedrock models from LiteLLM data")
    return bedrock_models


def match_token_specs(models_data: dict, litellm_bedrock: dict) -> dict:
    """
    Match token specs from LiteLLM to our models.

    Returns dict of model_id -> token_specs (in snake_case)
    """
    token_specs = {}

    # Build lookup maps for flexible matching
    litellm_lookup = {}
    for key, data in litellm_bedrock.items():
        # Extract just the model portion from keys like "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
        normalized = key.lower()
        if '/' in normalized:
            normalized = normalized.split('/')[-1]

        # Use snake_case for output schema
        litellm_lookup[normalized] = {
            'context_window': data.get('max_input_tokens') or data.get('max_tokens'),
            'max_output_tokens': data.get('max_output_tokens'),
            'source': 'litellm',
            'original_key': key,
            'litellm_verified': True
        }

    # Match against our models
    for provider, provider_data in models_data.get('providers', {}).items():
        for model_id, model in provider_data.get('models', {}).items():
            model_id_normalized = model_id.lower()

            # Try exact match first
            if model_id_normalized in litellm_lookup:
                token_specs[model_id] = litellm_lookup[model_id_normalized]
                continue

            # Try partial matching
            for litellm_key, specs in litellm_lookup.items():
                if model_id_normalized in litellm_key or litellm_key in model_id_normalized:
                    token_specs[model_id] = specs
                    break

    return token_specs


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for token specs collection.

    Input:
        {
            "s3Bucket": "bucket-name",
            "executionId": "exec-123",
            "modelsS3Key": "executions/{id}/merged/models.json"
        }

    Output:
        {
            "status": "SUCCESS",
            "s3Key": "executions/{id}/intermediate/token-specs.json",
            "modelsWithSpecs": 104,
            "modelsWithoutSpecs": 4
        }
    """
    start_time = time.time()

    s3_bucket = event['s3Bucket']
    execution_id = event['executionId']
    models_s3_key = event['modelsS3Key']
    dry_run = event.get('dryRun', False)

    if ':' in execution_id:
        execution_id = execution_id.split(':')[-1]

    output_key = f"executions/{execution_id}/intermediate/token-specs.json"

    logger.info("Collecting token specs from LiteLLM")

    try:
        # Fetch LiteLLM data (this is external, so do it even in dry run for testing)
        litellm_data = fetch_litellm_data()
        bedrock_models = filter_bedrock_models(litellm_data)

        s3_client = get_s3_client()

        if not dry_run:
            # Read our models
            models_data = read_from_s3(s3_client, s3_bucket, models_s3_key)

            # Match token specs
            token_specs = match_token_specs(models_data, bedrock_models)

            # Count statistics
            total_models = sum(
                len(p['models'])
                for p in models_data.get('providers', {}).values()
            )
            models_with_specs = len(token_specs)
            models_without_specs = total_models - models_with_specs

            output_data = {
                'metadata': {
                    'models_with_specs': models_with_specs,
                    'models_without_specs': models_without_specs,
                    'litellm_models_available': len(bedrock_models),
                    'source': 'litellm',
                    'collection_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                },
                'token_specs': token_specs
            }

            write_to_s3(s3_client, s3_bucket, output_key, output_data)
        else:
            logger.info("Dry run - skipping S3 operations")
            models_with_specs = len(bedrock_models)
            models_without_specs = 0

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            'status': 'SUCCESS',
            's3Key': output_key,
            'modelsWithSpecs': models_with_specs,
            'modelsWithoutSpecs': models_without_specs,
            'litellmModelsAvailable': len(bedrock_models),
            'durationMs': duration_ms
        }

    except Exception as e:
        logger.error(f"Failed to collect token specs: {e}", exc_info=True)
        return {
            'status': 'FAILED',
            'errorType': type(e).__name__,
            'errorMessage': str(e)
        }
