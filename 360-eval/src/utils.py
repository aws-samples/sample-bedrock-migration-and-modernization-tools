import pytz
import datetime
import json
import re
import time
import os
import random
import logging
import base64
import hashlib
from functools import lru_cache
from threading import Lock
import litellm
import requests
import requests.exceptions
from tenacity import retry, stop_after_delay, wait_exponential, retry_if_exception_type
from litellm import completion, RateLimitError, ServiceUnavailableError, APIError, APIConnectionError, BadRequestError, token_counter, Timeout
from botocore.exceptions import ClientError
import litellm

litellm.drop_params = True
# litellm.set_verbose = True

logger = logging.getLogger(__name__)

# Judge prompt template version — increment when rubric/prompt changes to invalidate cache
JUDGE_TEMPLATE_VERSION = "2.0"

# Sharpened metric definitions with boundaries and 1-5 rubrics
METRIC_DEFINITIONS = {
    "Correctness": {
        "definition": "Are the facts, logic, and conclusions accurate?",
        "boundary": "Does NOT assess whether everything was covered (that is Completeness).",
        "rubric": {
            "1": "Factually wrong or fabricated",
            "2": "Major errors that change meaning",
            "3": "Mostly correct with notable gaps in accuracy",
            "4": "Correct with minor inaccuracies",
            "5": "Fully correct and accurate",
        },
    },
    "Completeness": {
        "definition": "Does the response address ALL parts of the prompt?",
        "boundary": "Does NOT assess whether what is included is accurate (that is Correctness).",
        "rubric": {
            "1": "Addresses almost none of the prompt",
            "2": "Addresses some parts, major omissions",
            "3": "Addresses most parts, some gaps",
            "4": "Addresses nearly all parts, minor gaps",
            "5": "Thoroughly addresses every part of the prompt",
        },
    },
    "Relevance": {
        "definition": "Is everything in the response necessary and on-topic? No fluff, no tangents.",
        "boundary": "Does NOT assess structure or format (that is Format).",
        "rubric": {
            "1": "Off-topic or padded with unnecessary content",
            "2": "Partially relevant with significant tangents",
            "3": "Mostly relevant, some unnecessary content",
            "4": "Relevant with minimal tangents",
            "5": "Every sentence directly serves the prompt",
        },
    },
    "Format": {
        "definition": "Does the response match the requested output structure and syntax?",
        "boundary": "Does NOT assess whether the content is good (only the shape).",
        "rubric": {
            "1": "Completely ignores requested format",
            "2": "Partially follows format with major deviations",
            "3": "Follows format with some inconsistencies",
            "4": "Follows format with minor deviations",
            "5": "Perfectly matches the requested output format",
        },
    },
    "Coherence": {
        "definition": "Is the response internally consistent, logical, and well-organized?",
        "boundary": "Does NOT assess factual accuracy (that is Correctness).",
        "rubric": {
            "1": "Disorganized, contradictory, or illogical",
            "2": "Poorly structured with logical gaps",
            "3": "Reasonably structured, some logical issues",
            "4": "Well-structured with minor flow issues",
            "5": "Logically structured, clear, and well-organized",
        },
    },
    "Following-instructions": {
        "definition": "Did the response obey explicit constraints? (length, language, tone, specific requirements)",
        "boundary": "Does NOT assess general quality (only compliance with stated constraints).",
        "rubric": {
            "1": "Ignores explicit constraints entirely",
            "2": "Follows some constraints, misses key ones",
            "3": "Follows most constraints with some deviations",
            "4": "Follows constraints with minor oversights",
            "5": "Precisely obeys every stated constraint",
        },
    },
}

STANDARD_METRICS = list(METRIC_DEFINITIONS.keys())

# ----------------------------------------
# Judge Result Cache
# ----------------------------------------
_judge_cache = None
_judge_cache_lock = Lock()
_judge_cache_path = None
_judge_cache_stats = {"hits": 0, "misses": 0}
_judge_cache_dirty = False


def _get_judge_cache_path():
    """Determine judge cache file path based on output directory."""
    global _judge_cache_path
    if _judge_cache_path:
        return _judge_cache_path
    output_dir = os.environ.get("DEFAULT_OUTPUT_DIR", "outputs")
    _judge_cache_path = os.path.join(output_dir, ".judge_cache.json")
    return _judge_cache_path


def _load_judge_cache():
    """Load judge cache from disk. Returns dict."""
    global _judge_cache
    if _judge_cache is not None:
        return _judge_cache
    path = _get_judge_cache_path()
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                _judge_cache = json.load(f)
            logger.info(f"Judge cache loaded: {len(_judge_cache)} entries from {path}")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Judge cache corrupted, starting fresh: {e}")
            _judge_cache = {}
    else:
        _judge_cache = {}
    return _judge_cache


def _save_judge_cache():
    """Persist judge cache to disk."""
    global _judge_cache
    if _judge_cache is None:
        return
    path = _get_judge_cache_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with open(path, 'w') as f:
            json.dump(_judge_cache, f)
    except IOError as e:
        logger.warning(f"Failed to save judge cache: {e}")


def judge_cache_key(prompt, model_response, metric, judge_model_id):
    """
    Generate a cache key by hashing prompt + model_response + metric + judge_model_id + template_version.
    Template version in the key auto-invalidates on rubric/prompt changes.
    """
    raw = f"{prompt}|{model_response}|{metric}|{judge_model_id}|{JUDGE_TEMPLATE_VERSION}"
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def judge_cache_get(prompt, model_response, metric, judge_model_id):
    """Check cache for a judge result. Returns cached result dict or None."""
    with _judge_cache_lock:
        cache = _load_judge_cache()
        key = judge_cache_key(prompt, model_response, metric, judge_model_id)
        result = cache.get(key)
        if result is not None:
            _judge_cache_stats["hits"] += 1
            logger.info(f"Judge cache HIT for {metric} (judge={judge_model_id}, hash={key[:12]})")
            return result
        _judge_cache_stats["misses"] += 1
        logger.debug(f"Judge cache MISS for {metric} (judge={judge_model_id}, hash={key[:12]})")
        return None


def judge_cache_put(prompt, model_response, metric, judge_model_id, result):
    """Store a judge result in the in-memory cache.

    The previous implementation rewrote the entire cache file to disk on every
    put while holding the lock — O(n) per put (O(n^2) over a run) and serializing
    the whole judge pool on each fsync. We now just mark the cache dirty and rely
    on a single flush_judge_cache() at the end of the run.
    """
    global _judge_cache_dirty
    with _judge_cache_lock:
        cache = _load_judge_cache()
        key = judge_cache_key(prompt, model_response, metric, judge_model_id)
        cache[key] = result
        _judge_cache_dirty = True


def flush_judge_cache():
    """Persist the judge cache to disk once, if it has pending changes."""
    global _judge_cache_dirty
    with _judge_cache_lock:
        if _judge_cache_dirty:
            _save_judge_cache()
            _judge_cache_dirty = False


def get_judge_cache_stats():
    """Return judge cache hit/miss stats."""
    with _judge_cache_lock:
        hits = _judge_cache_stats["hits"]
        misses = _judge_cache_stats["misses"]
        total = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": (hits / total * 100) if total > 0 else 0.0,
            "cache_size": len(_judge_cache) if _judge_cache else 0,
        }


# ----------------------------------------
# Token Counting Cache
# ----------------------------------------

_token_cache_lock = Lock()
_token_cache_stats = {"hits": 0, "misses": 0}


@lru_cache(maxsize=1024)
def _cached_token_count(model: str, content_hash: str, content: str) -> int:
    """
    LRU-cached token counting implementation.

    Args:
        model: Model identifier for token counting
        content_hash: SHA-256 hash of content (used as cache key)
        content: The actual content to count tokens for

    Returns:
        Token count for the content
    """
    # content_hash is used for cache key deduplication, actual counting uses content
    return token_counter(model=model, messages=[{"role": "user", "content": content}])


def cached_token_counter(model: str, content: str) -> int:
    """
    Thread-safe cached token counter.

    Caches token counts based on model and content hash to avoid
    redundant API calls for the same content.

    Args:
        model: Model identifier for token counting
        content: The content to count tokens for

    Returns:
        Token count for the content
    """
    # Generate SHA-256 hash of content for cache key
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

    # lru_cache is internally thread-safe, so no lock is needed here. The previous
    # version took the lock twice per call (and read cache_info twice) purely to
    # maintain a hit/miss counter — that serialized every token-counting thread.
    # Stats are now derived from cache_info() on demand in get_token_cache_stats().
    try:
        return _cached_token_count(model, content_hash, content)
    except Exception as e:
        logger.warning(f"Cached token counter failed, falling back to direct call: {e}")
        # Fall back to direct token_counter call
        return token_counter(model=model, messages=[{"role": "user", "content": content}])


def get_token_cache_stats() -> dict:
    """
    Get token cache statistics for monitoring.

    Returns:
        dict with 'hits', 'misses', 'hit_rate', and 'cache_info'
    """
    info = _cached_token_count.cache_info()
    total = info.hits + info.misses
    hit_rate = (info.hits / total * 100) if total > 0 else 0.0
    return {
        "hits": info.hits,
        "misses": info.misses,
        "total": total,
        "hit_rate": round(hit_rate, 2),
        "cache_info": info._asdict(),
    }


def clear_token_cache():
    """Clear the token counting cache."""
    _cached_token_count.cache_clear()
    logger.info("Token cache cleared")

# ----------------------------------------
# Prompt Optimization Configuration
# ----------------------------------------

# Supported regions for prompt optimization
PROMPT_OPTIMIZATION_SUPPORTED_REGIONS = [
    "us-east-1",      # US East (N. Virginia)
    "us-west-2",      # US West (Oregon)
    "ap-south-1",     # Asia Pacific (Mumbai)
    "ap-southeast-2", # Asia Pacific (Sydney)
    "ca-central-1",   # Canada (Central)
    "eu-central-1",   # Europe (Frankfurt)
    "eu-west-1",      # Europe (Ireland)
    "eu-west-2",      # Europe (London)
    "eu-west-3",      # Europe (Paris)
    "sa-east-1"       # South America (São Paulo)
]

# Map model ID patterns to optimization target models
# This mapping handles various model families and versions
MODEL_FAMILY_OPTIMIZATION_MAP = {
    # Amazon Nova family (v1)
    "amazon.nova-lite-v1": "amazon.nova-lite-v1:0",
    "amazon.nova-micro-v1": "amazon.nova-micro-v1:0",
    "amazon.nova-pro-v1": "amazon.nova-pro-v1:0",
    "amazon.nova-premier-v1": "amazon.nova-premier-v1:0",

    # Amazon Nova 2 family
    "amazon.nova-2-lite": "amazon.nova-2-lite-v1:0",
    "amazon.nova-2-micro": "amazon.nova-2-micro-v1:0",
    "amazon.nova-2-pro": "amazon.nova-2-pro-v1:0",

    # Anthropic Claude 3 family
    "anthropic.claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",

    # Anthropic Claude 3.5 family
    "anthropic.claude-3-5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",

    # Anthropic Claude 3.7/4 family
    "anthropic.claude-3-7-sonnet": "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-sonnet-4": "anthropic.claude-sonnet-4-20250514-v1:0",
    "anthropic.claude-opus-4": "anthropic.claude-opus-4-20250514-v1:0",

    # DeepSeek - Tested and working with real AWS API
    # Note: Config uses us.deepseek.r1-v1:0 (for inference) but optimization API needs deepseek.r1-v1:0
    # The us. prefix is automatically stripped by get_optimization_target_model() before pattern matching
    "deepseek.r1": "deepseek.r1-v1:0",

    # Meta Llama family
    "meta.llama3-70b": "meta.llama3-70b-instruct-v1:0",
    "meta.llama3-1-70b": "meta.llama3-1-70b-instruct-v1:0",
    "meta.llama3-2-11b": "meta.llama3-2-11b-instruct-v1:0",
    "meta.llama3-3-70b": "meta.llama3-3-70b-instruct-v1:0",
    "meta.llama4-maverick-17b": "meta.llama4-maverick-17b-instruct-v1:0",
    "meta.llama4-scout-17b": "meta.llama4-scout-17b-instruct-v1:0",

    # Mistral family
    "mistral.mistral-large-2402": "mistral.mistral-large-2402-v1:0",
    "mistral.mistral-large-2407": "mistral.mistral-large-2407-v1:0",
    "mistral.mixtral": "mistral.mistral-large-2407-v1:0",  # Map Mixtral to Large for optimization
}

# ----------------------------------------
# Service Tier Configuration
# ----------------------------------------

# Valid service tier options
SERVICE_TIER_OPTIONS = ["default", "priority", "flex"]


def is_service_tier_supported(model_id, region=None):
    """
    Check if a model supports service tier selection (priority, default, flex).

    Uses cached validation results from model_capability_validator.

    This function handles various model ID formats including:
    - Regional prefixes (us., eu., ap., ca., sa.)
    - bedrock/ and converse/ prefixes
    - Different version suffixes

    Args:
        model_id: The model ID to check (may include bedrock/ prefix, regional prefix, version)
        region: Optional AWS region for more accurate cache lookup

    Returns:
        bool: True if model supports service tier, False otherwise
        Returns False if cache is unavailable (run validation to populate cache)

    Examples:
        >>> is_service_tier_supported("bedrock/us.amazon.nova-pro-v1:0", "us-west-2")
        True
        >>> is_service_tier_supported("anthropic.claude-3-5-sonnet-20241022-v2:0")
        True
        >>> is_service_tier_supported("openai/gpt-4o")
        False
    """
    # Use cache-based lookup
    try:
        from model_capability_validator import get_available_service_tiers
        tiers = get_available_service_tiers(model_id, region) if region else []

        # If we have cache data and more than just "default" tier, it's supported
        if tiers and len(tiers) > 1:
            return True
        # If cache explicitly shows only default, it's not supported
        elif tiers and len(tiers) == 1:
            return False
    except (ImportError, Exception) as e:
        # Cache not available - recommend running validation
        logger.debug(f"Service tier check failed for {model_id}: {e}. Run validation to populate cache.")
        pass

    # If cache unavailable or no data, return False (conservative default)
    return False


def get_optimization_target_model(model_id):
    """
    Map a model ID to its optimization target model.

    This function handles various model ID formats including:
    - Regional prefixes (us., eu., ap., ca., sa.)
    - bedrock/ and converse/ prefixes
    - Different version suffixes

    Args:
        model_id: The actual model ID (may include bedrock/ prefix, regional prefix, version)

    Returns:
        tuple: (optimization_target_model, is_supported)
               optimization_target_model is None if not supported

    Examples:
        >>> get_optimization_target_model("bedrock/us.amazon.nova-pro-v1:0")
        ("amazon.nova-pro-v1:0", True)
        >>> get_optimization_target_model("anthropic.claude-3-5-sonnet-20241022-v2:0")
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", True)
        >>> get_optimization_target_model("openai/gpt-4o")
        (None, False)
    """
    # Remove bedrock and converse prefixes
    clean_id = model_id.replace("bedrock/", "").replace("converse/", "")

    # Remove regional prefixes (us., eu., ap., ca., sa.)
    if "." in clean_id:
        parts = clean_id.split(".", 1)
        if parts[0] in ["us", "eu", "ap", "ca", "sa"]:
            clean_id = parts[1]

    # Try to match against model family patterns
    for pattern, target in MODEL_FAMILY_OPTIMIZATION_MAP.items():
        if pattern in clean_id:
            return target, True

    return None, False


def optimize_prompt_bedrock(prompt, model_id, region='us-east-1'):
    """
    Optimize a prompt using Bedrock's optimize_prompt API.

    This function:
    1. Validates the region supports prompt optimization
    2. Maps the model ID to a supported optimization target
    3. Calls the Bedrock optimize_prompt API
    4. Parses the streaming response to extract optimized prompt and analysis

    Args:
        prompt: The original prompt text to optimize
        model_id: Model ID to optimize for (cleaned, no bedrock/ prefix)
        region: AWS region where the optimization API will be called

    Returns:
        dict with:
            - 'success': bool - True if optimization succeeded
            - 'optimized_prompt': str - The optimized prompt text (if success=True)
            - 'analysis': str - Analysis from optimization API (if success=True)
            - 'target_model_used': str - The optimization target model used (if success=True)
            - 'error': str - Error message (if success=False)
            - 'skipped': bool - True if region/model not supported (graceful skip)

    Raises:
        No exceptions raised - all errors returned in result dict
    """
    import boto3

    # Check if region supports optimization
    if region not in PROMPT_OPTIMIZATION_SUPPORTED_REGIONS:
        logger.info(f"Region {region} does not support prompt optimization - skipping")
        return {
            "success": False,
            "skipped": True,
            "error": f"Region {region} does not support prompt optimization"
        }

    # Get optimization target model
    target_model, is_supported = get_optimization_target_model(model_id)

    if not is_supported:
        logger.info(f"Model {model_id} not supported for optimization - skipping")
        return {
            "success": False,
            "skipped": True,
            "error": f"Model {model_id} not supported for optimization"
        }

    try:
        logger.info(f"Optimizing prompt for model {model_id} using target {target_model} in region {region}")
        client = boto3.client('bedrock-agent-runtime', region_name=region)

        input_data = {
            "textPrompt": {
                "text": prompt
            }
        }

        response = client.optimize_prompt(
            input=input_data,
            targetModelId=target_model
        )

        # Parse streaming response
        optimized_prompt = None
        analysis = None

        event_stream = response.get('optimizedPrompt', [])

        for event in event_stream:
            if 'optimizedPromptEvent' in event:
                optimized_prompt_event = event['optimizedPromptEvent']
                # Navigate the nested structure to get the text
                text_prompt_data = optimized_prompt_event.get('optimizedPrompt', {}).get('textPrompt', {})
                optimized_prompt = text_prompt_data.get('text', '')

            elif 'analyzePromptEvent' in event:
                analysis_event = event['analyzePromptEvent']
                # Extract analysis text or metadata
                analysis = str(analysis_event)

        if optimized_prompt:
            logger.info(f"Successfully optimized prompt for {model_id} (original: {len(prompt)} chars, optimized: {len(optimized_prompt)} chars)")
            return {
                "success": True,
                "optimized_prompt": optimized_prompt,
                "analysis": analysis or "No analysis provided",
                "target_model_used": target_model,
                "skipped": False
            }
        else:
            logger.error("No optimized prompt returned from API")
            return {
                "success": False,
                "error": "No optimized prompt returned from API",
                "skipped": False
            }

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_msg = e.response.get('Error', {}).get('Message', str(e))
        logger.error(f"AWS ClientError optimizing prompt for {model_id}: {error_code} - {error_msg}")
        return {
            "success": False,
            "error": f"AWS Error {error_code}: {error_msg}",
            "skipped": False
        }
    except Exception as e:
        logger.error(f"Unexpected error optimizing prompt for {model_id}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "skipped": False
        }


# ----------------------------------------
# Request Builders
# ----------------------------------------


def prepare_model_for_litellm(model_id):
    """
    Prepare model ID for litellm completion call.
    For Bedrock models, ensures correct format is bedrock/<model-id>.
    LiteLLM automatically uses the converse route for supported models.

    Handles edge cases:
    - Missing bedrock/ prefix (adds it back)
    - Removes any /converse/ prefix (litellm handles routing automatically)
    - Already correct format (returns as-is)

    Args:
        model_id: Model ID (e.g., "bedrock/us.amazon.nova-pro-v1:0" or "bedrock/converse/us.amazon.nova-pro-v1:0")

    Returns:
        str: Model ID ready for litellm (e.g., "bedrock/us.amazon.nova-pro-v1:0")

    Examples:
        >>> prepare_model_for_litellm("bedrock/us.amazon.nova-pro-v1:0")
        "bedrock/us.amazon.nova-pro-v1:0"
        >>> prepare_model_for_litellm("bedrock/converse/us.amazon.nova-pro-v1:0")
        "bedrock/us.amazon.nova-pro-v1:0"
        >>> prepare_model_for_litellm("converse/us.amazon.nova-pro-v1:0")
        "bedrock/us.amazon.nova-pro-v1:0"
        >>> prepare_model_for_litellm("us.amazon.nova-pro-v1:0")
        "bedrock/us.amazon.nova-pro-v1:0"
        >>> prepare_model_for_litellm("openai/gpt-4o")
        "openai/gpt-4o"
    """

    # Check if this is a Bedrock model (contains regional prefix like us., eu., or anthropic., amazon., etc.)
    is_bedrock_model = model_id.startswith('bedrock/')
    if not is_bedrock_model:
        # Not a Bedrock model, return as-is
        logger.debug(f"Not a Bedrock model, returning as-is: {model_id}")
        return model_id

    # It's a Bedrock model - ensure correct format: bedrock/<model-id>
    # LiteLLM will automatically use converse route for supported models

    # Step 1: Remove any existing bedrock/ and converse/ prefixes (including stacked ones)
    clean_id = model_id
    while clean_id.startswith('bedrock/') or clean_id.startswith('converse/'):
        if clean_id.startswith('bedrock/'):
            clean_id = clean_id.replace('bedrock/', '', 1)
        if clean_id.startswith('converse/'):
            clean_id = clean_id.replace('converse/', '', 1)

    # Step 2: Build the correct format - litellm handles converse routing automatically
    correct_id = f"bedrock/{clean_id}"

    return correct_id


def setup_logging(log_dir='logs', experiment='none'):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/360-benchmark-{ts}-{experiment}.log"

    # Reset root logger and handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure root logger
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode='w'
    )

    # Add console handler for info and above (needed for progress tracking)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(console)

    # Configure logger for this module
    module_logger = logging.getLogger(__name__)
    module_logger.info(f"Logging initialized. Log file: {log_file}")

    return ts, log_file


def get_timestamp():
    return datetime.datetime.fromtimestamp(time.time(), tz=pytz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def calculate_average_scores(dict_list):
    if not dict_list:
        return {}

    # Initialize result dictionary
    result = {}

    # Use the union of keys — judges may report different metric sets (e.g. one judge
    # omits a metric it couldn't score). Averaging only over the dicts that contain a
    # given key avoids a KeyError and prevents a missing metric from being treated as 0.
    keys = []
    for d in dict_list:
        for k in d:
            if k not in keys:
                keys.append(k)

    # Calculate the average for each key over the judges that provided it
    for key in keys:
        values = [d[key] for d in dict_list if key in d]
        if not values:
            continue
        average = sum(values) / len(values)
        result[f'AVG_{key}'] = round(average, 4)
    return result


def extract_json_with_llm(all_metrics, text, judge_model_id, cfg):
    metrics_entries = [f'            "{metric}": <int>' for metric in all_metrics]
    metrics_string = ",\n".join(metrics_entries)

    prompt = f"""## Instruction
Extract and return the JSON object from the given text that matches the specified JSON schema. The schema is:
```json
{{
    "scores": {{
{metrics_string}
            }}
}}
```
## Text
{text}

Provide your response immediately without any preamble or additional information.
            """
    resp = run_inference(model_name=judge_model_id, prompt_text=prompt, provider_params=cfg, stream=False, judge_eval=True)
    text = resp['text']
    payload = extract_json_from_text(text)
    if not payload:
        return None
    return payload


def sanitize_judge_json(raw_json):
    """
    Sanitize raw JSON text from judge responses before parsing.

    Fixes common issues where judges produce structurally valid JSON
    but with characters that break json.loads():
    - Invalid escape sequences (e.g. \\$ \\# \\@ \\& \\% from LaTeX habits)
    - Trailing commas before closing braces/brackets
    - Unescaped double quotes inside string values (e.g. 'free of "weasel" words')

    Args:
        raw_json: Raw JSON string from judge response

    Returns:
        Sanitized JSON string safe for json.loads()
    """
    # Fix invalid backslash escapes: \$ \# \@ \& \% \~ \^ etc.
    # Valid JSON escapes are: \" \\ \/ \b \f \n \r \t \uXXXX
    sanitized = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw_json)

    # Remove trailing commas before } or ] (common LLM output error)
    sanitized = re.sub(r',\s*([}\]])', r'\1', sanitized)

    return sanitized


def _fix_unescaped_quotes(raw_json):
    """
    Fix unescaped double quotes inside JSON string values.

    Walks the JSON character by character, tracking whether we're inside
    a string value. Quotes that appear mid-string (not after a colon/comma
    boundary or at a structural position) are escaped.
    """
    result = []
    in_string = False
    i = 0
    while i < len(raw_json):
        ch = raw_json[i]
        if ch == '\\' and in_string:
            # Escaped character — pass through both chars
            result.append(ch)
            if i + 1 < len(raw_json):
                i += 1
                result.append(raw_json[i])
            i += 1
            continue
        if ch == '"':
            if not in_string:
                in_string = True
                result.append(ch)
            else:
                # Check if this quote ends the string or is embedded
                # Look ahead: if next non-whitespace is : , } ] it's a structural close
                rest = raw_json[i+1:].lstrip()
                if rest and rest[0] in (':', ',', '}', ']'):
                    in_string = False
                    result.append(ch)
                else:
                    # Embedded quote — escape it
                    result.append('\\"')
        else:
            result.append(ch)
        i += 1
    return ''.join(result)


def _try_parse_json(raw):
    """Try json.loads with progressive sanitization fallbacks."""
    # Attempt 1: raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: fix escape sequences + trailing commas
    try:
        return json.loads(sanitize_judge_json(raw))
    except json.JSONDecodeError:
        pass

    # Attempt 3: also fix unescaped quotes
    try:
        return json.loads(_fix_unescaped_quotes(sanitize_judge_json(raw)))
    except json.JSONDecodeError:
        return None


def extract_json_from_text(text):
    """Extract JSON from text. Handles both bundled and single-metric formats."""
    # Try to find any JSON object in the text
    # First try: look for ```json ... ``` code blocks
    code_block = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if code_block:
        parsed = _try_parse_json(code_block.group(1).strip())
        if parsed:
            return parsed

    # Second try: find JSON with "scores" key (bundled mode)
    pattern = re.compile(r'\{[^{}]*"scores"\s*:\s*\{.*?\}\s*\}', re.DOTALL)
    match = pattern.search(text)
    if match:
        parsed = _try_parse_json(match.group(0))
        if parsed:
            return parsed

    # Third try: find JSON with "score" key (single-metric mode)
    pattern = re.compile(r'\{\s*"score"\s*:.*?\}', re.DOTALL)
    match = pattern.search(text)
    if match:
        parsed = _try_parse_json(match.group(0))
        if parsed:
            return parsed

    # Fourth try: find any JSON object
    for match in re.finditer(r'\{[^{}]+\}', text, re.DOTALL):
        parsed = _try_parse_json(match.group(0))
        if parsed and isinstance(parsed, dict) and ('score' in parsed or 'scores' in parsed):
            return parsed

    logger.warning("No matching JSON block found in judge response.")
    return None


def extract_json_response(all_metrics, text, judge_model_id, cfg):
    payload = extract_json_from_text(text)
    # Validate: bundled mode must have "scores" key, single-metric must have "score" key
    if payload:
        if len(all_metrics) == 1:
            # Single-metric: {"score": int, "rationale": "..."} is valid
            if 'score' not in payload and 'scores' not in payload:
                payload = None
        else:
            # Bundled: must have "scores" dict
            if 'scores' not in payload:
                payload = None
    if not payload:
        payload = extract_json_with_llm(all_metrics, text, judge_model_id, cfg)
    return payload


def validate_custom_metrics(custom_metrics):
    """
    Validate custom metric definitions. Each must have metric_name, definition,
    and a rubric with all 5 levels (1-5).

    Args:
        custom_metrics: list of custom metric dicts

    Returns:
        (valid: bool, errors: list of str)
    """
    if not custom_metrics:
        return True, []
    errors = []
    for i, cm in enumerate(custom_metrics):
        if not isinstance(cm, dict):
            errors.append(f"Custom metric #{i+1}: must be a dict, got {type(cm).__name__}")
            continue
        name = cm.get("metric_name")
        if not name:
            errors.append(f"Custom metric #{i+1}: missing 'metric_name'")
        if not cm.get("definition"):
            errors.append(f"Custom metric '{name or f'#{i+1}'}': missing 'definition'")
        rubric = cm.get("rubric")
        if not rubric or not isinstance(rubric, dict):
            errors.append(f"Custom metric '{name or f'#{i+1}'}': missing or invalid 'rubric'")
        else:
            for level in ("1", "2", "3", "4", "5"):
                if level not in rubric:
                    errors.append(f"Custom metric '{name}': missing rubric level {level}")
        if not cm.get("primary"):
            errors.append(f"Custom metric '{name or f'#{i+1}'}': missing 'primary' model assignment")
    return len(errors) == 0, errors


def _build_metric_block(metric_name, structure_validation=None, custom_definition=None):
    """Build the definition + boundary + rubric block for a single metric.

    Args:
        metric_name: Name of the metric
        structure_validation: Structure validation result (for Format metric)
        custom_definition: Optional custom metric dict with definition/boundary/rubric
    """
    defn = custom_definition or METRIC_DEFINITIONS.get(metric_name)
    if not defn:
        # Fallback for custom metric without definition (legacy)
        return f"- {metric_name}"

    rubric_lines = "\n".join(f"  {k} = {v}" for k, v in defn["rubric"].items())
    boundary_line = f"Does NOT assess: {defn['boundary']}\n" if defn.get("boundary") else ""
    block = (
        f"**{metric_name}**\n"
        f"Definition: {defn['definition']}\n"
        f"{boundary_line}"
        f"{rubric_lines}"
    )

    # Add structure validation evidence for Format metric only
    if metric_name == "Format" and structure_validation:
        fmt = structure_validation.get("expected_format", "unknown")
        if structure_validation["valid"]:
            block += (
                f"\n\nData Structure Analysis: The model response was programmatically validated "
                f"against {fmt.upper()} format. Result: PASSED — valid {fmt.upper()}. "
                f"Use this evidence when scoring Format."
            )
        else:
            error = structure_validation.get("error", "validation failed")
            block += (
                f"\n\nData Structure Analysis: The model response was programmatically validated "
                f"against {fmt.upper()} format. Result: FAILED — {error}. "
                f"An invalid structure should score poorly on Format (1)."
            )

    return block


def llm_judge_template(all_metrics,
                       task_types,
                       task_criteria,
                       prompt,
                       model_response,
                       golden_answer,
                       structure_validation=None,
                       custom_metric_definitions=None,
                       success_criteria=None,
                       ):
    """Build the judge evaluation prompt.

    Supports two modes:
    - Bundled (multiple metrics): all metric definitions included, JSON output with all scores + rationales
    - Single metric: only one metric's definition included, simpler JSON output

    Supports two anchor modes:
    - Golden answer: reference response provided (default)
    - Criteria-only: structured success criteria instead of golden answer

    Golden answer appears BEFORE model response to reduce positional bias.
    Rationale is required for every score.

    Args:
        custom_metric_definitions: list of custom metric dicts
        success_criteria: dict with must_include, success_definition, must_not_include,
            expected_format, edge_cases — used when golden_answer is empty/None
    """
    # Build lookup for custom metric definitions
    custom_defs = {}
    if custom_metric_definitions:
        for cm in custom_metric_definitions:
            if isinstance(cm, dict) and cm.get("metric_name"):
                custom_defs[cm["metric_name"]] = cm

    # Build metric blocks with definitions, boundaries, and rubrics
    metric_blocks = []
    for m in all_metrics:
        custom_def = custom_defs.get(m)
        metric_blocks.append(_build_metric_block(m, structure_validation, custom_definition=custom_def))

    metrics_section = "\n\n".join(metric_blocks)

    # Build anchor section: golden answer or success criteria
    has_golden = golden_answer and str(golden_answer).strip() and str(golden_answer).strip().lower() not in ('', 'nan', 'none', 'n/a')
    if has_golden:
        anchor_section = f"""# Golden (Reference) Response:
{golden_answer}"""
    elif success_criteria and isinstance(success_criteria, dict):
        parts = []
        if success_criteria.get('must_include'):
            parts.append(f"- Must include: {success_criteria['must_include']}")
        if success_criteria.get('success_definition'):
            parts.append(f"- Success definition: {success_criteria['success_definition']}")
        if success_criteria.get('must_not_include'):
            parts.append(f"- Must NOT include: {success_criteria['must_not_include']}")
        if success_criteria.get('edge_cases'):
            parts.append(f"- Edge cases: {success_criteria['edge_cases']}")
        criteria_text = "\n".join(parts)
        anchor_section = f"""# Success Criteria:
{criteria_text}"""
    else:
        anchor_section = "# No golden answer or success criteria provided. Evaluate the response based on the prompt and task description alone."

    if len(all_metrics) == 1:
        # Single-metric mode (specialist)
        return f"""## You are an expert evaluator.

# Task: {task_types}
# Task description: {task_criteria}

# Original Prompt:
{prompt}

{anchor_section}

# Model Response (evaluate this):
{model_response}

# Evaluation Metric:
{metrics_section}

# Score this metric from 1 (worst) to 5 (best). Evaluate ONLY this metric — ignore all other quality aspects.

## IMPORTANT: Output JSON only in this format:
```json
{{
  "score": <int>,
  "rationale": "<brief evidence from the response>"
}}
```""".strip()
    else:
        # Bundled mode (multiple metrics)
        metrics_entries = [f'    "{m}": {{"score": <int>, "rationale": "<brief evidence>"}}' for m in all_metrics]
        metrics_json = ",\n".join(metrics_entries)
        return f"""## You are an expert evaluator.

# Task: {task_types}
# Task description: {task_criteria}

# Original Prompt:
{prompt}

{anchor_section}

# Model Response (evaluate this):
{model_response}

# Evaluate the model response on each of the following metrics INDEPENDENTLY.
# Each metric has its own definition, boundary, and rubric. Do not let one metric influence another.

{metrics_section}

# For each metric, assign an integer score from 1 (worst) to 5 (best) and provide a brief rationale citing evidence from the response.

## IMPORTANT: Output JSON only in this format:
```json
{{
  "scores": {{
{metrics_json}
  }}
}}
```""".strip()


# Define which exceptions should trigger a retry
RETRYABLE_EXCEPTIONS = (
    Timeout,
    RateLimitError,
    ServiceUnavailableError,
    APIConnectionError,
    APIError,
    requests.exceptions.RequestException,
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError
)


# Create a class to track retry counts
class RetryTracker:
    def __init__(self):
        self.attempts = 0
        self.had_300_second_wait = False

    def increment(self, retry_state):
        self.attempts = retry_state.attempt_number
        wait_time = retry_state.next_action.sleep if retry_state.next_action else 0

        # Log first retry at INFO, subsequent retries at DEBUG
        if self.attempts == 1:
            logger.info(f"First retry attempt, sleeping for {wait_time} seconds")
        else:
            logger.debug(f"Retry attempt {self.attempts}, sleeping for {wait_time} seconds")

        # If we're about to wait 300 seconds and already had one 300s wait, stop retrying
        if wait_time >= 300:
            if self.had_300_second_wait:
                logger.info("Already waited 300 seconds once, stopping retries")
                raise Exception("Max wait time reached - stopping after one 300-second retry")
            self.had_300_second_wait = True
            logger.warning(f"Long wait time ({wait_time}s) on retry attempt {self.attempts}")


# Retry decorator with exponential backoff
def _call_llm_with_retry(model_name, messages, provider_params, retry_tracker, stream):
    """Wrapper function to call LLM with retry logic"""

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        wait=wait_exponential(multiplier=2, min=1, max=300),  # Start at 1s, exponentially increase, max 60s per attempt
        stop=stop_after_delay(300),  # Total retry time of 5 minutes
        before_sleep=retry_tracker.increment
    )
    def _api_call():
        try:
            time_ = time.time()
            # Prepare model ID for litellm
            litellm_model = prepare_model_for_litellm(model_name)
            completed = completion(
                model=litellm_model,
                messages=messages,
                stream=stream,
                timeout=30,  # 60 second timeout to prevent extreme outliers
                stream_timeout=60,
                # Ask the provider to report token usage in the final stream chunk so we
                # can use exact server counts instead of re-counting client-side.
                **({"stream_options": {"include_usage": True}} if stream else {}),
                **provider_params
                )
            return completed, time_
        except BadRequestError as e:
            error_msg = str(e)
            has_image_content = any(
                isinstance(msg.get('content'), list) and
                any(part.get('type') == 'image_url' for part in msg.get('content', []))
                for msg in messages if isinstance(msg, dict)
            )

            if has_image_content and ("doesn't support the image content block" in error_msg or
                                      "image content block" in error_msg or
                                      "vision" in error_msg.lower() or
                                      "multimodal" in error_msg.lower()):
                logger.error(f"Model {model_name} does not support vision/image inputs: {error_msg}")
                # Create a more informative error message and don't retry
                raise
            else:
                # Other BadRequestErrors should not be retried either
                logger.error(f"BadRequestError (non-retryable): {error_msg}")
                raise
        except Timeout as e:
            error_msg = str(e)
            logger.error(f"Model {model_name} Exceeded set up-time to generate the response: {error_msg}")
            # Create a more informative error message and don't retry
            raise
        except RETRYABLE_EXCEPTIONS as e:
            error_msg = str(e)
            # Don't retry auth/token errors — they won't resolve with retries
            auth_errors = ["expired", "invalid token", "bearer token", "not authorized",
                           "access denied", "invalid credential", "security token"]
            if any(phrase in error_msg.lower() for phrase in auth_errors):
                logger.error(f"Authentication error (non-retryable): {error_msg[:200]}")
                raise Exception(f"API key error: {error_msg[:200]}. Please update your credentials.")
            # Only log first retryable error at WARNING, rest at DEBUG to reduce noise
            if retry_tracker.attempts == 0:
                logger.warning(f"Retryable error occurred: {type(e).__name__}: {str(e)[:100]}")
            else:
                logger.debug(f"Retryable error (attempt {retry_tracker.attempts}): {type(e).__name__}")
            # Add jitter to avoid thundering herd
            jitter = random.uniform(0, 3)
            time.sleep(jitter)
            raise  # Re-raise for the retry decorator to catch
        except Exception as e:
            logger.error(f"Non-retryable error calling LLM: {str(e)}")
            raise

    return _api_call()


def encode_image(image_path):
    """Encode a local image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def validate_image_url(url, timeout=10):
    """
    Validate that a web URL is accessible and points to an image.

    Args:
        url: The URL to validate
        timeout: Request timeout in seconds

    Returns:
        bool: True if URL is valid and accessible

    Raises:
        ValueError: If URL is not accessible or not an image
    """
    try:
        logger.debug(f"Validating image URL: {url}")
        response = requests.head(url, timeout=timeout, allow_redirects=True)

        # Check if request was successful
        if response.status_code != 200:
            raise ValueError(f"URL returned status code {response.status_code}")

        # Check Content-Type header if available
        content_type = response.headers.get('Content-Type', '')
        if content_type and not content_type.startswith(('image/', 'application/octet-stream')):
            logger.warning(f"URL may not be an image. Content-Type: {content_type}")

        logger.debug(f"URL validation successful for: {url}")
        return True

    except requests.exceptions.Timeout:
        raise ValueError(f"URL request timed out after {timeout} seconds")
    except requests.exceptions.ConnectionError:
        raise ValueError(f"Failed to connect to URL: {url}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error accessing URL: {str(e)}")


def validate_local_image(file_path):
    """
    Validate that a local file exists and is a supported image format.

    Args:
        file_path: Path to the local image file

    Returns:
        str: The file extension (without dot)

    Raises:
        ValueError: If file doesn't exist or has unsupported format
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise ValueError(f"Image file not found: {file_path}")

    # Check if it's a file (not directory)
    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")

    # Check file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension.startswith('.'):
        file_extension = file_extension[1:]

    supported_formats = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp']
    if file_extension not in supported_formats:
        raise ValueError(
            f"Unsupported image format: {file_extension}. Supported formats: {', '.join(supported_formats)}")

    # Check if file is readable
    if not os.access(file_path, os.R_OK):
        raise ValueError(f"Image file is not readable: {file_path}")

    # Check file size (warn if too large)
    file_size = os.path.getsize(file_path)
    max_size_mb = 20
    if file_size > max_size_mb * 1024 * 1024:
        logger.warning(f"Image file is large ({file_size / 1024 / 1024:.2f} MB): {file_path}")

    logger.debug(f"Local image validation successful: {file_path}")
    return file_extension


def handle_vision(prompt_text, vision_enabled):
    image_path = vision_enabled.strip()

    if not image_path:
        logger.error("Empty image path provided for vision model")
        raise ValueError("Image path cannot be empty when vision is enabled")

    logger.info(f"Processing image for vision model: {image_path}")

    # Check if the image is a web URL using regex
    url_pattern = r'^https?://'

    if re.match(url_pattern, image_path):
        # It's a web URL, validate it's accessible
        logger.debug("Detected web URL for image")
        try:
            validate_image_url(image_path)
            image_url = image_path
            logger.info(f"Successfully validated web image URL: {image_path}")
        except ValueError as e:
            logger.error(f"Failed to validate image URL {image_path}: {e}")
            raise ValueError(f"Invalid or inaccessible image URL: {e}")
    else:
        # It's a local file, validate and encode it
        logger.debug("Detected local file path for image")
        try:
            # Validate the local image file
            file_extension = validate_local_image(image_path)

            # Map common extensions to MIME types
            mime_type_map = {
                'jpg': 'jpeg',
                'jpeg': 'jpeg',
                'png': 'png',
                'gif': 'gif',
                'webp': 'webp',
                'bmp': 'bmp'
            }
            mime_type = mime_type_map.get(file_extension, 'jpeg')

            # Encode the image
            logger.debug(f"Encoding local image file: {image_path}")
            base64_image = encode_image(image_path)
            image_url = f"data:image/{mime_type};base64,{base64_image}"
            logger.info(f"Successfully encoded local image: {image_path} (size: {len(base64_image)} bytes)")

        except ValueError as e:
            logger.error(f"Image validation failed for {image_path}: {e}")
            raise
        except IOError as e:
            logger.error(f"Failed to read image file {image_path}: {e}")
            raise ValueError(f"Failed to read image file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing image {image_path}: {e}")
            raise ValueError(f"Failed to process image file: {e}")

    # Create message for vision model with image and text
    image_content = {
        "type": "image_url",
        "image_url": {
            "url": image_url
        }
    }
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}, image_content]}]
    logger.debug("Created multimodal message with image and text")
    return messages


def _usage_value(usage, key):
    """Read a token-count field from a litellm usage object (attribute- or dict-style)."""
    if usage is None:
        return None
    val = getattr(usage, key, None)
    if val is None and hasattr(usage, 'get'):
        try:
            val = usage.get(key)
        except Exception:
            val = None
    return val


def _run_inference_responses(model_name, prompt_text, input_cost, output_cost,
                             provider_params, judge_eval, retry_tracker):
    """Inference via the Bedrock Mantle OpenAI-compatible Responses API.

    Mantle models (e.g. openai.gpt-5.5) are NOT Converse models — they're invoked with
    litellm.responses() against https://bedrock-mantle.<region>.api.aws/openai/v1 using
    a Bedrock Mantle API key. provider_params carries api_base + api_key (set by the
    caller). Non-streaming for reliability: clean output_text + usage, with TTFT==TTLB
    ==runtime (same convention as the non-streaming completion path).
    """
    # litellm routes via its OpenAI-compatible handler with the openai/ prefix
    # (the bedrock_mantle/ prefix is buggy in litellm). Strip the catalog bedrock/ prefix:
    # 'bedrock/openai.gpt-5.5' -> 'openai/openai.gpt-5.5'.
    clean = model_name
    for p in ("bedrock/converse/", "bedrock/"):
        if clean.startswith(p):
            clean = clean[len(p):]
            break
    litellm_model = f"openai/{clean}"

    kwargs = {
        "model": litellm_model,
        "input": prompt_text,
        "api_base": provider_params.get("api_base"),
        "api_key": provider_params.get("api_key"),
    }
    # NOTE: GPT-5.x reasoning models on Mantle reject `temperature` ("unsupported_parameter")
    # — do NOT forward it. Only bound the output length.
    if provider_params.get("max_tokens") is not None:
        kwargs["max_output_tokens"] = provider_params["max_tokens"]  # Responses API param name

    start_time = time.time()
    resp = litellm.responses(**kwargs)
    total_runtime = time.time() - start_time

    response_text = getattr(resp, "output_text", "") or ""
    usage = getattr(resp, "usage", None)
    input_tokens = _usage_value(usage, "input_tokens")
    if input_tokens is None:
        input_tokens = _usage_value(usage, "prompt_tokens")
    output_tokens = _usage_value(usage, "output_tokens")
    if output_tokens is None:
        output_tokens = _usage_value(usage, "completion_tokens")
    # Fall back to client-side counting only if the provider returned no usage.
    if input_tokens is None or output_tokens is None:
        try:
            if output_tokens is None:
                output_tokens = cached_token_counter(model=clean, content=response_text)
            if input_tokens is None:
                input_tokens = cached_token_counter(model=clean, content=prompt_text)
        except Exception:
            input_tokens = input_tokens if input_tokens is not None else 0.0000001
            output_tokens = output_tokens if output_tokens is not None else 0.0000001

    if judge_eval:
        return {"text": response_text, "outputTokens": output_tokens, "inputTokens": input_tokens}

    throughput_tps = output_tokens / total_runtime if total_runtime > 0 else 0
    tot_input_cost = input_tokens * (input_cost / 1_000_000)
    tot_output_cost = output_tokens * (output_cost / 1_000_000)
    return {
        "model_response": response_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_runtime": total_runtime,
        "time_to_first_byte": total_runtime,  # non-streaming: TTFT == TTLB == runtime
        "time_to_last_byte": total_runtime,
        "throughput_tps": throughput_tps,
        "total_cost": tot_output_cost + tot_input_cost,
        "retry_count": retry_tracker.attempts,
        "finish_reason": getattr(resp, "status", None),
        "stream_metrics": None,
        "thinking_response": "",
    }


# Run streaming inference and collect metrics
def run_inference(model_name: str,
                  prompt_text: str,
                  input_cost: float = 0.00001,
                  output_cost: float = 0.00001,
                  provider_params: dict = dict,
                  stream: bool = True,
                  vision_enabled: str = None,
                  judge_eval: bool = None,
                  endpoint: str = None):
    # Bedrock Mantle (OpenAI-compatible Responses API) models take a separate path —
    # litellm.responses() with the Mantle api_base/key — not Converse/completion().
    if endpoint == "bedrock_mantle":
        retry_tracker = RetryTracker()
        return _run_inference_responses(model_name, prompt_text, input_cost, output_cost,
                                        provider_params, judge_eval, retry_tracker)
    if vision_enabled:
        messages = handle_vision(prompt_text, vision_enabled)
    else:
        messages = [{"content": prompt_text, "role": "user"}]
    response_chunks = []
    thinking_chunks = []
    first = True
    # Create a retry tracker
    retry_tracker = RetryTracker()

    try:
        if 'gemini' in model_name:
            os.environ['GEMINI_API_KEY'] = provider_params['api_key']
            del provider_params['api_key']
            # Use the retry wrapper for the API call
        payload, start_time = _call_llm_with_retry(
            model_name=model_name,
            messages=messages,
            provider_params=provider_params,
            retry_tracker=retry_tracker,
            stream=stream
        )
        if not stream:
            msg = payload.choices[0].message
            response = msg.content if hasattr(msg, 'content') else str(msg)
            thinking_response = msg.get("reasoning_content", "") if hasattr(msg, 'get') else getattr(msg, 'reasoning_content', "")
            output_tokens = payload.model_extra['usage']['completion_tokens']
            input_tokens = payload.model_extra['usage']['prompt_tokens']

            judge_payload = dict()
            judge_payload["text"] = payload.choices[0].message.content
            judge_payload['outputTokens'] = payload.model_extra['usage']['completion_tokens']
            judge_payload['inputTokens'] = payload.model_extra['usage']['prompt_tokens']

            # If called from judge (judge_eval=True), return judge format
            if judge_eval:
                return judge_payload

            # Otherwise, return full evaluation structure (non-streaming evaluation mode)
            end_time = time.time()
            total_runtime = end_time - start_time

            throughput_tps = output_tokens / total_runtime if total_runtime > 0 else 0
            tot_input_cost = input_tokens * (input_cost / 1_000_000)
            tot_output_cost = output_tokens * (output_cost / 1_000_000)

            # Capture finish_reason
            finish_reason = payload.choices[0].finish_reason if hasattr(payload.choices[0], 'finish_reason') else None

            return {
                "model_response": response,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_runtime": total_runtime,
                "time_to_first_byte": total_runtime,  # Same as total for non-streaming
                "time_to_last_byte": total_runtime,   # Same as total for non-streaming
                "throughput_tps": throughput_tps,
                "total_cost": tot_output_cost + tot_input_cost,
                "retry_count": retry_tracker.attempts,
                "finish_reason": finish_reason,
                "stream_metrics": None,  # N/A for non-streaming
                "thinking_response": thinking_response
            }
        else:
            time_to_first_token = 0
            server_usage = None
            # completion() returns before the stream body is read, so a dead pooled
            # connection (e.g. litellm's background logging loops closing a cached
            # socket's fd — "[Errno 9] Bad file descriptor" on the first read)
            # surfaces here, outside the tenacity retry above. Re-request the stream
            # when it dies before yielding any content; once content has arrived,
            # errors propagate unchanged so partial responses are still returned.
            max_stream_restarts = 2
            for stream_restart in range(max_stream_restarts + 1):
                try:
                    for chunk in payload:
                        if first:
                            time_to_first_token = time.time() - start_time
                            first = False

                        # Capture provider-reported token usage when present. The final usage
                        # chunk (and some providers' interim chunks) carry usage with empty
                        # choices, so record it before skipping the chunk for content.
                        chunk_usage = getattr(chunk, 'usage', None)
                        if chunk_usage is not None:
                            server_usage = chunk_usage

                        # Handle potential None or malformed chunks
                        if not chunk or not hasattr(chunk, 'choices') or len(chunk.choices) == 0:
                            if chunk_usage is None:
                                logger.warning("Received invalid chunk from API")
                            continue
                        delta_obj = chunk.choices[0].delta
                        # Some models return delta as string instead of dict-like object
                        if isinstance(delta_obj, str):
                            if delta_obj:
                                response_chunks.append(delta_obj)
                            continue
                        thinking_delta = delta_obj.get("reasoning_content", "") if hasattr(delta_obj, 'get') else getattr(delta_obj, 'reasoning_content', "")
                        if thinking_delta:
                            thinking_chunks.append(thinking_delta)
                        delta = delta_obj.get("content", "") if hasattr(delta_obj, 'get') else getattr(delta_obj, 'content', "")
                        if delta:
                            response_chunks.append(delta)
                    break
                except RETRYABLE_EXCEPTIONS as e:
                    if response_chunks or stream_restart == max_stream_restarts:
                        raise
                    logger.warning(
                        f"Stream failed before yielding content ({type(e).__name__}); "
                        f"re-requesting stream ({stream_restart + 1}/{max_stream_restarts})")
                    first = True
                    payload, start_time = _call_llm_with_retry(
                        model_name=model_name,
                        messages=messages,
                        provider_params=provider_params,
                        retry_tracker=retry_tracker,
                        stream=stream
                    )

            end = time.time()
            time_to_last_byte = round(end - start_time, 4)
            total_runtime = end - start_time
            actual_response = "".join(response_chunks)
            thinking_response = "".join(thinking_chunks)
            # Prefer the provider's reported token usage — it matches the non-streaming
            # path and correctly accounts for the system/vision content that client-side
            # counting of prompt_text alone misses (notably base64 images). Fall back to
            # the cached client-side counter only when the provider returns no usage.
            input_tokens = _usage_value(server_usage, 'prompt_tokens')
            output_tokens = _usage_value(server_usage, 'completion_tokens')
            if input_tokens is None or output_tokens is None:
                try:
                    counter_id = model_name.replace('converse/', '')  # Converse is needed for inference only
                    output_tokens = cached_token_counter(model=counter_id, content=thinking_response + actual_response)
                    input_tokens = cached_token_counter(model=counter_id, content=prompt_text)
                except Exception as e:
                    logger.error(f"Error counting tokens: {str(e)}")
                    output_tokens = 0.0000001
                    input_tokens = 0.0000001

            tokens_per_sec = output_tokens / total_runtime if total_runtime > 0 else 0
            tot_input_cost = input_tokens * (input_cost / 1_000_000)
            tot_output_cost = output_tokens * (output_cost / 1_000_000)

            return {
                "model_response": actual_response,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_runtime": total_runtime,
                "time_to_first_byte": time_to_first_token,
                "time_to_last_byte": time_to_last_byte,
                "throughput_tps": tokens_per_sec,
                "total_cost": tot_output_cost + tot_input_cost,
                "retry_count": retry_tracker.attempts,
                "thinking_response": thinking_response,
            }

    except Exception as e:
        logger.error(f"Error during inference: {type(e).__name__}: {str(e)}")
        # Return partial results if available, or error information
        if response_chunks:
            partial_response = "".join(response_chunks)
            logger.info(f"Returning partial response of length {len(partial_response)}")
            return {
                "model_response": partial_response,
                "error": str(e),
                "error_type": type(e).__name__,
                "partial_result": True,
                "retry_count": retry_tracker.attempts  # Include the retry count even in error case
            }
        else:
            raise RuntimeError(f"Inference failed after {retry_tracker.attempts} retries: {str(e)}")


def report_summary_template(models, evaluations):
    models_str = '\n'.join(models)
    return f"""
## Task
You are writing an executive summary for an LLM model evaluation report. The report evaluates {len(models)} model(s) across one or more tasks. Your summary should give the reader a complete picture of the evaluation results in 2-4 concise paragraphs.

## Guidelines
1. Start with evaluation context: which tasks were evaluated, how many models, and any reliability concerns (models with low evaluation completion rates should be flagged — their metrics are based on fewer data points and may not be representative).
2. Cover the top performers for accuracy, speed, and cost-efficiency. Also mention the runner-up (second-best) model in each category with the gap between #1 and #2 — this helps readers understand if the winner is a clear standout or barely ahead.
3. Highlight any notable trade-offs (e.g., a model that is the fastest but also the most expensive, or the most accurate but slowest).
4. If any models have low reliability scores (high failure rates), explicitly call this out and note that their accuracy/latency/cost metrics should be interpreted with caution.
5. Use neutral, fact-based language. State numbers and percentages. Avoid subjective judgments like "good" or "poor" — let the data speak.
6. Use plain language. When data uses technical terms like "fat tails" say "highly likely to vary" instead.
7. Use HTML tags "<b><i>TEXT</b></i>" to highlight Model Names and Task Names throughout your response.
8. Do not reference data or analysis sections that are empty or not provided.

## Models Evaluated:
{models_str}

## Evaluation Data
{evaluations}

Write the executive summary now, without any preamble or section headers. Start directly with the findings.
    """.strip()


def convert_scientific_to_decimal(df):
    """
    Converts numeric columns with scientific notation to decimal representation.

    Parameters:
        df (pandas.DataFrame): Input dataframe

    Returns:
        pandas.DataFrame: DataFrame with converted values
    """
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    # Iterate through columns
    for column in result_df.columns:
        try:
            result_df[column] = result_df[column].apply(lambda x: f"{x:.6f}" if x < 0.01 else x)
        except:
            pass

    return result_df


def check_model_access(provider_params, model_id):
    """
    Check if we have access to invoke a specific model
    """
    try:
        messages = [{"content": 'HI', "role": "user"}]
        # Prepare model ID for litellm (adds /converse for Bedrock models)
        litellm_model = prepare_model_for_litellm(model_id)
        # Non-streaming: a stream=True probe left an unconsumed stream holding a
        # pooled connection, whose GC later closed the socket under the first real
        # invocation (httpcore.ReadError: [Errno 9] Bad file descriptor).
        completion(
            model=litellm_model,
            messages=messages,
            stream=False,
            timeout=60,  # 60 second timeout to prevent extreme outliers
            **provider_params
        )

        # If we get a response without error, access is granted
        return 'granted'

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'AccessDeniedException':
            return 'denied'
        elif error_code == 'ValidationException':
            return 'denied'
        elif error_code == 'ThrottlingException':
            return 'granted'
        else:
            return 'denied'
    except Exception:
        return 'denied'


# ----------------------------------------
# System prompt extraction (for APO)
# ----------------------------------------

def find_longest_common_block(strings, min_len=20):
    """Return the longest contiguous substring present in ALL input strings.

    Uses difflib pairwise to harvest candidate blocks, then filters to those
    present in every string. Returns None if no shared block meets `min_len`.
    """
    import difflib
    if not strings or len(strings) < 2:
        return None
    base = strings[0]
    others = strings[1:]
    if not base:
        return None

    candidates = set()
    for other in others:
        if not other:
            continue
        sm = difflib.SequenceMatcher(None, base, other, autojunk=False)
        for block in sm.get_matching_blocks():
            if block.size >= min_len:
                candidates.add(base[block.a:block.a + block.size])

    valid = [c for c in candidates if all(c in s for s in strings)]
    if not valid:
        return None
    return max(valid, key=len)


def _llm_extract_system_prompt(samples, *, model_id, region, api_key=None):
    """Ask a small Bedrock model to extract the verbatim shared system prompt."""
    if not samples:
        return ""
    # Cap each sample to keep the extractor prompt under ~10k tokens
    truncated = [(s[:3000] + ("..." if len(s) > 3000 else "")) for s in samples]
    sample_block = "".join(
        f"\n\n--- SAMPLE {i+1} ---\n{s}" for i, s in enumerate(truncated)
    )
    extractor_prompt = (
        f"You are given {len(samples)} prompts that share a common system-prompt "
        "section (role description, task instructions, output format rules, etc.) "
        "but differ in the variable input each prompt asks about.\n\n"
        "Your job: extract ONLY the shared system-prompt section. Output the "
        "system prompt VERBATIM as it appears in the prompts. Do not paraphrase. "
        "Do not add commentary. Do not include any portion that varies between "
        "prompts (specific questions, transcripts, examples, dates, names, etc.).\n\n"
        f"PROMPTS:{sample_block}\n\n"
        "Reply with ONLY the shared system-prompt text, nothing else."
    )

    params = {"max_tokens": 4000, "temperature": 0.0, "aws_region_name": region}
    if api_key:
        params["api_key"] = api_key
    resp = run_inference(
        model_id, extractor_prompt,
        provider_params=params, stream=False, judge_eval=True,
    )
    return (resp.get("text") or "").strip()


def extract_system_prompt_hybrid(samples, *, min_len=20,
                                 fallback_model_id="us.amazon.nova-lite-v1:0",
                                 region="us-east-1", api_key=None):
    """Detect the shared system-prompt section across N sample prompts.

    Strategy:
      1. Heuristic — longest common contiguous block (>= min_len chars).
      2. If no clean block found, LLM fallback (Bedrock Nova Lite by default)
         is asked to identify the verbatim shared section.
      3. Last-resort: return ("", samples) — caller can treat APO as a no-op.

    Returns `(system_prompt: str, variable_parts: list[str])` where
    `variable_parts[i]` is `samples[i]` minus the detected system prompt.
    """
    if not samples:
        return ("", [])

    block = find_longest_common_block(list(samples), min_len=min_len)
    if block:
        variable_parts = []
        for s in samples:
            # Remove only the FIRST occurrence to avoid stripping coincidental
            # substring matches deeper in the text.
            variable_parts.append(s.replace(block, "", 1).strip())
        logger.info(
            f"[APO extract] heuristic match: {len(block)} chars shared across "
            f"{len(samples)} samples"
        )
        return (block.strip(), variable_parts)

    logger.info(
        f"[APO extract] heuristic found no block >= {min_len} chars; "
        "falling back to LLM extraction"
    )
    try:
        sys_prompt = _llm_extract_system_prompt(
            samples, model_id=fallback_model_id, region=region, api_key=api_key,
        )
    except Exception as e:
        logger.warning(f"[APO extract] LLM fallback failed: {e}")
        sys_prompt = ""

    if not sys_prompt:
        logger.warning(
            "[APO extract] no system prompt detected — caller should treat "
            "APO as a no-op for this evaluation"
        )
        return ("", list(samples))

    variable_parts = []
    for s in samples:
        if sys_prompt in s:
            variable_parts.append(s.replace(sys_prompt, "", 1).strip())
        else:
            # LLM may have summarized slightly; keep original sample as-is.
            variable_parts.append(s)
    return (sys_prompt, variable_parts)

