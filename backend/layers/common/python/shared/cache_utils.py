"""
Cache utilities for sharing data between Lambda functions.

Provides functions to read and validate cached ListFoundationModels responses
that are stored by model-extractor for reuse by regional-availability.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from .s3_utils import read_from_s3

logger = logging.getLogger(__name__)


def get_cached_models(s3_client: Any, bucket: str, cache_key: str) -> Optional[dict]:
    """
    Get cached ListFoundationModels response from S3.

    Args:
        s3_client: Boto3 S3 client instance
        bucket: S3 bucket name
        cache_key: S3 key for cached data

    Returns:
        Cached data dictionary or None if not found/error
    """
    try:
        return read_from_s3(s3_client, bucket, cache_key, default_on_missing=None)
    except Exception as e:
        logger.warning(
            "Failed to read cache",
            extra={"cache_key": cache_key, "error": str(e)},
        )
        return None


def is_cache_valid(cached_data: Optional[dict], max_age_seconds: int = 3600) -> bool:
    """
    Check if cached data is still valid based on timestamp.

    Args:
        cached_data: Cached data dictionary with 'timestamp' field
        max_age_seconds: Maximum age in seconds (default 1 hour)

    Returns:
        True if cache is valid and not expired, False otherwise
    """
    if not cached_data:
        return False

    timestamp_str = cached_data.get("timestamp")
    if not timestamp_str:
        return False

    try:
        cached_time = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
        age_seconds = (datetime.utcnow() - cached_time).total_seconds()
        return age_seconds < max_age_seconds
    except (ValueError, TypeError) as e:
        logger.warning(
            "Failed to parse cache timestamp",
            extra={"timestamp": timestamp_str, "error": str(e)},
        )
        return False


def build_cache_key(
    execution_id: str, region: str, cache_type: str = "list_foundation_models"
) -> str:
    """
    Build a standardized cache key for a given execution and region.

    Args:
        execution_id: The pipeline execution ID
        region: AWS region
        cache_type: Type of cached data (default: list_foundation_models)

    Returns:
        S3 key for the cache file
    """
    return f"executions/{execution_id}/cache/{cache_type}_{region}.json"
