import os
import json
import logging
import time
from typing import List, Dict, Tuple, Optional
from tenacity import retry, stop_after_delay, wait_exponential, wait_random, retry_if_exception_type
import litellm
from litellm import embedding, RateLimitError, ServiceUnavailableError, APIError, APIConnectionError

litellm.drop_params = True
logger = logging.getLogger(__name__)


# ----------------------------------------
# Embedding Model Profile Loading
# ----------------------------------------

def load_embedding_model_profiles(file_path: str) -> List[Dict]:
    """
    Load embedding model profiles from JSONL file.

    Args:
        file_path: Path to embedding_models_profiles.jsonl

    Returns:
        List of embedding model profile dicts
    """
    profiles = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    profiles.append(json.loads(line))
        logger.info(f"Loaded {len(profiles)} embedding model profiles from {file_path}")
    except FileNotFoundError:
        logger.error(f"Embedding model profiles file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing embedding model profiles: {e}")
        raise
    return profiles


# ----------------------------------------
# Embedding Generation with Retry Logic
# ----------------------------------------

@retry(
    retry=retry_if_exception_type((RateLimitError, ServiceUnavailableError, APIConnectionError)),
    wait=wait_exponential(multiplier=2, min=2, max=60) + wait_random(0, 2),  # Exponential backoff with jitter
    stop=stop_after_delay(300),  # 5 minutes max retry to handle rate limits
    reraise=True
)
def generate_embeddings_litellm(
    texts: List[str],
    model_id: str,
    provider_params: Optional[Dict] = None
) -> Tuple[List[List[float]], Dict]:
    """
    Generate embeddings using LiteLLM with robust retry logic.

    Retry Strategy:
    - Exponential backoff: 2s, 4s, 8s, 16s, 32s, 60s (capped)
    - Jitter: Random 0-2 seconds added to each retry
    - Total timeout: 5 minutes (300 seconds)
    - Retries on: RateLimitError, ServiceUnavailableError, APIConnectionError

    Args:
        texts: List of text strings to embed
        model_id: Embedding model ID (e.g., "openai/text-embedding-3-large")
        provider_params: Additional parameters (api_key, region, etc.)

    Returns:
        Tuple of (embeddings_list, metadata_dict)
        - embeddings_list: List of embedding vectors
        - metadata_dict: Contains tokens used, latency, model info
    """
    if not texts:
        raise ValueError("texts list cannot be empty")

    start_time = time.time()
    params = provider_params or {}

    try:
        logger.debug(f"Generating embeddings for {len(texts)} texts using {model_id}")

        # Call LiteLLM embedding API
        response = embedding(
            model=model_id,
            input=texts,
            **params
        )

        latency = time.time() - start_time

        # Extract embeddings from response
        embeddings_list = [item['embedding'] for item in response['data']]

        # Extract metadata
        metadata = {
            'model_id': model_id,
            'num_texts': len(texts),
            'total_tokens': response.get('usage', {}).get('total_tokens', 0),
            'latency_seconds': latency,
            'dimensions': len(embeddings_list[0]) if embeddings_list else 0
        }

        logger.debug(f"Successfully generated {len(embeddings_list)} embeddings in {latency:.2f}s")

        return embeddings_list, metadata

    except (RateLimitError, ServiceUnavailableError, APIConnectionError) as e:
        logger.warning(f"Retryable error generating embeddings: {type(e).__name__} - {str(e)}")
        raise  # Let @retry handle it
    except Exception as e:
        logger.error(f"Error generating embeddings with {model_id}: {type(e).__name__} - {str(e)}", exc_info=True)
        raise


def generate_single_embedding(
    text: str,
    model_id: str,
    provider_params: Optional[Dict] = None
) -> Tuple[List[float], Dict]:
    """
    Generate embedding for a single text.

    Args:
        text: Text string to embed
        model_id: Embedding model ID
        provider_params: Additional parameters

    Returns:
        Tuple of (embedding_vector, metadata_dict)
    """
    embeddings_list, metadata = generate_embeddings_litellm([text], model_id, provider_params)
    return embeddings_list[0], metadata


# ----------------------------------------
# Batch Embedding Generation
# ----------------------------------------

def generate_embeddings_batch(
    texts: List[str],
    model_id: str,
    provider_params: Optional[Dict] = None,
    batch_size: int = 100,
    sleep_between_batches: float = 2.0
) -> Tuple[List[List[float]], Dict]:
    """
    Generate embeddings in batches to avoid rate limits.

    Args:
        texts: List of text strings to embed
        model_id: Embedding model ID
        provider_params: Additional parameters
        batch_size: Number of texts per batch
        sleep_between_batches: Sleep time between batches (seconds, default 2.0 to avoid rate limits)

    Returns:
        Tuple of (embeddings_list, metadata_dict)
    """
    all_embeddings = []
    total_tokens = 0
    total_latency = 0

    num_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info(f"Generating embeddings for {len(texts)} texts in {num_batches} batches")

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1

        logger.debug(f"Processing batch {batch_num}/{num_batches} ({len(batch)} texts)")

        try:
            embeddings, metadata = generate_embeddings_litellm(batch, model_id, provider_params)
            all_embeddings.extend(embeddings)
            total_tokens += metadata.get('total_tokens', 0)
            total_latency += metadata.get('latency_seconds', 0)

            # Sleep between batches to avoid rate limits
            if i + batch_size < len(texts) and sleep_between_batches > 0:
                time.sleep(sleep_between_batches)

        except Exception as e:
            logger.error(f"Failed to process batch {batch_num}: {str(e)}")
            raise

    aggregate_metadata = {
        'model_id': model_id,
        'num_texts': len(texts),
        'total_tokens': total_tokens,
        'latency_seconds': total_latency,
        'num_batches': num_batches,
        'dimensions': len(all_embeddings[0]) if all_embeddings else 0
    }

    logger.info(f"Successfully generated {len(all_embeddings)} embeddings in {total_latency:.2f}s")

    return all_embeddings, aggregate_metadata


# ----------------------------------------
# Model Access Verification
# ----------------------------------------

def check_embedding_model_access(model_id: str, params: Optional[Dict] = None) -> str:
    """
    Check if embedding model is accessible.

    Args:
        model_id: Embedding model ID to check
        params: Provider-specific parameters (api_key, region, etc.)

    Returns:
        'granted' if accessible, 'failed' otherwise
    """
    try:
        logger.debug(f"Checking access to embedding model: {model_id}")

        # Try to generate a single test embedding
        test_text = "test"
        _, metadata = generate_single_embedding(test_text, model_id, params)

        logger.debug(f"Access granted for {model_id} (dimensions: {metadata.get('dimensions')})")
        return 'granted'

    except Exception as e:
        logger.warning(f"Access denied for {model_id}: {type(e).__name__} - {str(e)}")
        return 'failed'


def embedding_model_sanity_check(embedding_models: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """
    Check access to all embedding models in parallel.

    Args:
        embedding_models: List of embedding model profile dicts

    Returns:
        Tuple of (accessible_models, failed_models)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    logger.info(f"Checking access for {len(embedding_models)} embedding models...")

    accessible = []
    failed = []
    lock = Lock()

    def check_single_model(model):
        params = {}
        model_id = model['model_id']

        # Setup params based on provider
        if "openai/" in model_id:
            params['api_key'] = os.getenv('OPENAI_API')
        elif 'cohere/' in model_id:
            # Cohere via API (not Bedrock)
            params['api_key'] = os.getenv('COHERE_API')
        elif 'bedrock/' in model_id:
            # Bedrock models (bedrock/*)
            params['aws_region_name'] = model.get('region', 'us-east-1')

        try:
            access = check_embedding_model_access(model_id, params)
            return model, access, None
        except Exception as e:
            return model, 'failed', str(e)

    with ThreadPoolExecutor(max_workers=min(10, len(embedding_models))) as executor:
        future_to_model = {
            executor.submit(check_single_model, model): model
            for model in embedding_models
        }

        completed = 0
        total = len(embedding_models)

        for future in as_completed(future_to_model):
            completed += 1
            original_model = future_to_model[future]

            try:
                model, access, error = future.result(timeout=30)

                with lock:
                    if access == 'granted':
                        accessible.append(model)
                        logger.debug(f"✓ Embedding model access granted: {model['model_id']} ({completed}/{total})")
                    else:
                        failed.append(model['model_id'])
                        if error:
                            logger.debug(f"✗ Embedding model access failed: {model['model_id']} - {error} ({completed}/{total})")
                        else:
                            logger.debug(f"✗ Embedding model access denied: {model['model_id']} ({completed}/{total})")

            except Exception as e:
                with lock:
                    failed.append(original_model['model_id'])
                    logger.error(f"✗ Exception checking embedding model {original_model['model_id']}: {str(e)} ({completed}/{total})")

    logger.info(f"Embedding model access check complete: {len(accessible)} accessible, {len(failed)} failed")
    return accessible, failed


# ----------------------------------------
# Cost Calculation
# ----------------------------------------

def calculate_embedding_cost(
    total_tokens: int,
    model_profile: Dict
) -> float:
    """
    Calculate cost for embedding generation.

    Args:
        total_tokens: Total tokens processed
        model_profile: Embedding model profile dict with cost info

    Returns:
        Total cost in USD
    """
    input_cost_per_1k = model_profile.get('input_token_cost', 0)
    cost = (total_tokens / 1000) * input_cost_per_1k
    return cost
