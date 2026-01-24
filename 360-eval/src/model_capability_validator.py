"""
Model Capability Validator

Tests Bedrock model availability and service tier support by making actual API calls.
Results are cached to avoid repeated validation costs.
"""

import json
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import existing utility functions
from utils import run_inference

logger = logging.getLogger(__name__)

# Cache file location
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_FILE = CACHE_DIR / "model_capabilities.json"
MODELS_PROFILE_PATH = PROJECT_ROOT / "config" / "models_profiles.jsonl"


def get_models_hash() -> str:
    """
    Generate SHA256 hash of models_profiles.jsonl content.
    Used to detect when the file has changed and cache needs refresh.
    """
    try:
        with open(MODELS_PROFILE_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        return hashlib.sha256(content.encode()).hexdigest()
    except FileNotFoundError:
        logger.warning(f"Models profile file not found: {MODELS_PROFILE_PATH}")
        return ""
    except Exception as e:
        logger.error(f"Error reading models profile: {e}")
        return ""


def load_capability_cache() -> Dict:
    """Load cached model capabilities from JSON file."""
    if not CACHE_FILE.exists():
        return {
            "last_updated": None,
            "models_hash": "",
            "capabilities": {}
        }

    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading capability cache: {e}")
        return {
            "last_updated": None,
            "models_hash": "",
            "capabilities": {}
        }


def save_capability_cache(cache: Dict) -> bool:
    """Save model capabilities to JSON cache file."""
    try:
        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Update timestamp
        cache["last_updated"] = datetime.utcnow().isoformat() + "Z"

        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)

        logger.info(f"Capability cache saved to {CACHE_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving capability cache: {e}")
        return False


def is_cache_valid() -> bool:
    """
    Check if the capability cache is still valid.
    Returns False if models_profiles.jsonl has changed since last validation.
    """
    cache = load_capability_cache()
    current_hash = get_models_hash()

    if not cache.get("models_hash"):
        return False

    return cache["models_hash"] == current_hash


def test_service_tier(
    model_id: str,
    region: str,
    service_tier: str,
    timeout: int = 10
) -> Tuple[bool, Optional[str]]:
    """
    Test if a specific service tier is available for a model+region combination.

    Args:
        model_id: Full model ID (e.g., "bedrock/us.amazon.nova-2-lite-v1:0")
        region: AWS region (e.g., "us-west-2")
        service_tier: Service tier to test ("default", "priority", or "flex")
        timeout: Request timeout in seconds

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    try:
        # Prepare minimal test request - only essential parameters
        provider_params = {
            "max_tokens": 5,
            "aws_region_name": region
        }

        # Add service tier if not default
        if service_tier != "default":
            provider_params["serviceTier"] = {"type": service_tier}

        # Make test inference call
        result = run_inference(
            model_name=model_id,
            prompt_text="Hi",
            input_cost=0.0001,
            output_cost=0.0001,
            provider_params=provider_params,
            stream=False,  # Non-streaming for faster response
            vision_enabled=None
        )

        # Check if response is valid
        # For non-streaming, run_inference returns dict with "text" key
        if result:
            response_text = result.get("text", "")
            logger.debug(f"Received response for {model_id}: {type(result)}, keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
            logger.debug(f"Response text: '{response_text[:100] if response_text else 'EMPTY'}'")

            # Check if we got actual text content OR valid token counts
            # Some models (like reasoning models) may return empty text but valid completion
            # If we got inputTokens and outputTokens, the model responded successfully
            if response_text and len(response_text.strip()) > 0:
                return True, None
            elif result.get("outputTokens", 0) > 0 or result.get("inputTokens", 0) > 0:
                # Model responded with tokens, even if text is empty (e.g., reasoning models)
                logger.debug(f"Model responded with tokens (input: {result.get('inputTokens')}, output: {result.get('outputTokens')})")
                return True, None
            else:
                return False, "Empty response from model"
        else:
            return False, "No response object returned"

    except Exception as e:
        error_msg = str(e)

        # Check if error is about serviceTier not being supported
        if "serviceTier" in error_msg and "not permitted" in error_msg:
            logger.debug(f"Service tier '{service_tier}' not supported for {model_id} @ {region} (parameter not recognized by Bedrock)")
            return False, "Service tier parameter not supported by this model"

        logger.debug(f"Service tier test failed for {model_id} @ {region} ({service_tier}): {error_msg}")
        return False, error_msg


def test_model_availability(
    model_id: str,
    region: str
) -> Dict:
    """
    Test model availability and supported service tiers.

    Args:
        model_id: Full model ID
        region: AWS region

    Returns:
        Dict with availability info:
        {
            "available": bool,
            "service_tiers": List[str],
            "last_checked": str (ISO timestamp),
            "error": Optional[str]
        }
    """
    result = {
        "available": False,
        "service_tiers": [],
        "last_checked": datetime.utcnow().isoformat() + "Z",
        "error": None
    }

    # Test default tier first (baseline)
    success, error = test_service_tier(model_id, region, "default")

    if not success:
        result["error"] = "Model not available in region"
        logger.info(f"❌ {model_id} @ {region}: Not available ({error})")
        return result

    # Model is available
    result["available"] = True
    result["service_tiers"].append("default")

    # Test other tiers
    for tier in ["priority", "flex"]:
        logger.debug(f"Testing {tier} tier for {model_id} @ {region}")
        success, error = test_service_tier(model_id, region, tier)

        if success:
            result["service_tiers"].append(tier)
            logger.debug(f"  ✓ {tier} tier supported")
        else:
            logger.debug(f"  ✗ {tier} tier not supported")

        # Small delay between tier tests to avoid rate limiting
        time.sleep(0.5)

    tier_status = ", ".join(result["service_tiers"])
    logger.info(f"✅ {model_id} @ {region}: Available ({tier_status})")

    return result


def load_models_from_profile() -> List[Tuple[str, str]]:
    """
    Load model+region pairs from models_profiles.jsonl.

    Returns:
        List of (model_id, region) tuples
    """
    models = []

    try:
        with open(MODELS_PROFILE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                try:
                    data = json.loads(line)
                    model_id = data.get("model_id")
                    region = data.get("region")

                    if model_id and region and "bedrock/" in model_id:
                        models.append((model_id, region))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line[:50]}...")
                    continue

    except FileNotFoundError:
        logger.error(f"Models profile file not found: {MODELS_PROFILE_PATH}")
    except Exception as e:
        logger.error(f"Error loading models profile: {e}")

    return models


def validate_all_models(force: bool = False) -> Dict:
    """
    Validate all models from models_profiles.jsonl.

    Args:
        force: If True, ignore existing cache and re-validate all models

    Returns:
        Updated capability cache dict
    """
    logger.info("🔍 Starting model capability validation...")

    # Load existing cache
    cache = load_capability_cache() if not force else {
        "last_updated": None,
        "models_hash": "",
        "capabilities": {}
    }

    # Load models from profile
    models = load_models_from_profile()
    logger.info(f"📋 Found {len(models)} model+region combinations")

    if not models:
        logger.warning("No Bedrock models found in profiles")
        return cache

    # Update models hash
    cache["models_hash"] = get_models_hash()

    # Validate each model
    for idx, (model_id, region) in enumerate(models, 1):
        logger.info(f"[{idx}/{len(models)}] Testing {model_id} @ {region}")

        # Initialize model entry if doesn't exist
        if model_id not in cache["capabilities"]:
            cache["capabilities"][model_id] = {}

        # Test this model+region
        result = test_model_availability(model_id, region)
        cache["capabilities"][model_id][region] = result

        # Small delay between models to avoid rate limiting
        time.sleep(1)

    # Save updated cache
    save_capability_cache(cache)

    logger.info("✅ Model validation complete!")
    return cache


def get_model_capabilities(model_id: str, region: Optional[str] = None) -> Optional[Dict]:
    """
    Get cached capabilities for a specific model+region.

    Args:
        model_id: Full model ID
        region: AWS region (optional, returns first available if not specified)

    Returns:
        Capability dict or None if not found
    """
    cache = load_capability_cache()
    capabilities = cache.get("capabilities", {})

    if model_id not in capabilities:
        return None

    model_caps = capabilities[model_id]

    if region:
        return model_caps.get(region)
    else:
        # Return first available region's capabilities
        for reg, caps in model_caps.items():
            if caps.get("available"):
                return caps
        return None


def get_available_service_tiers(model_id: str, region: str) -> List[str]:
    """
    Get list of available service tiers for model+region.

    Args:
        model_id: Full model ID
        region: AWS region

    Returns:
        List of available tier names (e.g., ["default", "priority", "flex"])
    """
    capabilities = get_model_capabilities(model_id, region)

    if not capabilities or not capabilities.get("available"):
        return ["default"]  # Fallback to default only

    return capabilities.get("service_tiers", ["default"])
