"""Bedrock model utilities.

Provides functions to:
- Fetch foundation models and inference profiles from the Bedrock API
- Ensure models_profiles.jsonl exists

Model pricing and service tiers are generated from the bedrock-pricing-scraper CSV
using scripts/generate_profiles_from_csv.py. This file no longer scrapes pricing.
"""

import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import boto3

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_PROFILE_PATH = PROJECT_ROOT / "config" / "models_profiles.jsonl"


def fetch_foundation_models(region: str = "us-east-1") -> tuple[dict, dict]:
    """Fetch all foundation models and build lookup maps.

    Returns:
        (name_to_info, id_to_name) where:
        - name_to_info: modelName -> {modelId, inferenceTypes, provider, ...}
        - id_to_name: modelId -> modelName
    """
    client = boto3.client("bedrock", region_name=region)
    resp = client.list_foundation_models()
    models = resp.get("modelSummaries", [])

    name_to_info = {}
    id_to_name = {}

    for m in models:
        name = m["modelName"]
        mid = m["modelId"]
        inference_types = m.get("inferenceTypesSupported", [])

        # Skip provisioned-only variants (e.g., :256k, :300k suffixes)
        if re.search(r":\d+k$", mid) or re.search(r":mm$", mid):
            continue
        if inference_types == ["PROVISIONED"]:
            continue

        # If duplicate name, prefer ON_DEMAND or INFERENCE_PROFILE over PROVISIONED
        if name in name_to_info:
            if "PROVISIONED" in inference_types:
                continue  # keep existing non-provisioned variant

        name_to_info[name] = {
            "modelId": mid,
            "inferenceTypes": inference_types,
            "provider": m.get("providerName", ""),
            "inputModalities": m.get("inputModalities", []),
            "outputModalities": m.get("outputModalities", []),
        }
        id_to_name[mid] = name

    return name_to_info, id_to_name


def fetch_inference_profiles(region: str = "us-east-1") -> dict:
    """Fetch inference profiles and build cross-region prefix map.

    Returns:
        Dict mapping base_model_id -> list of cross-region profile IDs.
        E.g., anthropic.claude-sonnet-4-5-v1:0 -> [us.anthropic..., global.anthropic...]
    """
    client = boto3.client("bedrock", region_name=region)

    profiles = []
    resp = client.list_inference_profiles(maxResults=100)
    profiles.extend(resp.get("inferenceProfileSummaries", []))
    while resp.get("nextToken"):
        resp = client.list_inference_profiles(maxResults=100, nextToken=resp["nextToken"])
        profiles.extend(resp.get("inferenceProfileSummaries", []))

    cross_region_map = defaultdict(list)
    for p in profiles:
        pid = p["inferenceProfileId"]
        base_models = [m.get("modelArn", "").split("/")[-1] for m in p.get("models", [])]
        for base in set(base_models):
            if base:
                cross_region_map[base].append(pid)

    return cross_region_map


def ensure_models_profiles(models_path: Optional[Path] = None) -> Path:
    """Ensure models_profiles.jsonl exists.

    This function checks if the file exists. If not, it logs a warning.
    Model profiles are now generated from the bedrock-pricing-scraper CSV
    using scripts/generate_profiles_from_csv.py, not by scraping.

    Args:
        models_path: Path to models_profiles.jsonl. Defaults to config/models_profiles.jsonl.

    Returns:
        Path to the models_profiles.jsonl file.
    """
    if models_path is None:
        models_path = MODELS_PROFILE_PATH

    if not models_path.exists():
        logger.warning(
            "models_profiles.jsonl not found at %s. "
            "Generate it using: python scripts/generate_profiles_from_csv.py <pricing_csv>",
            models_path
        )

    return models_path
