"""
S3 utility functions with proper exception handling.
"""

import json
import logging
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

from shared.config import RETRY_CONFIG

logger = logging.getLogger(__name__)


class S3ReadError(Exception):
    """Raised when an S3 read operation fails."""

    def __init__(self, message: str, bucket: str, key: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.bucket = bucket
        self.key = key
        self.original_error = original_error


class S3WriteError(Exception):
    """Raised when an S3 write operation fails."""

    def __init__(self, message: str, bucket: str, key: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.bucket = bucket
        self.key = key
        self.original_error = original_error


def get_s3_client() -> Any:
    """
    Get a configured S3 client with retry logic.

    Returns:
        Configured boto3 S3 client.
    """
    return boto3.client('s3', config=RETRY_CONFIG)


def read_from_s3(
    s3_client: Any,
    bucket: str,
    key: str,
    default_on_missing: Optional[dict] = None
) -> dict:
    """
    Read JSON data from S3 with proper exception handling.

    Args:
        s3_client: Boto3 S3 client instance.
        bucket: S3 bucket name.
        key: S3 object key.
        default_on_missing: If provided, return this value when object doesn't exist
                           instead of raising an error. Set to None to raise on missing.

    Returns:
        Parsed JSON data as a dictionary.

    Raises:
        S3ReadError: When the read operation fails (unless default_on_missing is set
                    for NoSuchKey errors).
    """
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')

        if error_code == 'NoSuchKey':
            if default_on_missing is not None:
                logger.warning(f"S3 object not found, using default: s3://{bucket}/{key}")
                return default_on_missing
            logger.error(f"S3 object not found: s3://{bucket}/{key}")
            raise S3ReadError(
                f"S3 object not found: {key}",
                bucket=bucket,
                key=key,
                original_error=e
            )
        elif error_code == 'AccessDenied':
            logger.error(f"Access denied to S3 object: s3://{bucket}/{key}")
            raise S3ReadError(
                f"Access denied to S3 object: {key}",
                bucket=bucket,
                key=key,
                original_error=e
            )
        else:
            logger.error(f"S3 read error ({error_code}): s3://{bucket}/{key} - {e}")
            raise S3ReadError(
                f"Failed to read from S3: {error_code}",
                bucket=bucket,
                key=key,
                original_error=e
            )
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from s3://{bucket}/{key}: {e}")
        raise S3ReadError(
            f"Invalid JSON in S3 object: {key}",
            bucket=bucket,
            key=key,
            original_error=e
        )
    except Exception as e:
        logger.error(f"Unexpected error reading from s3://{bucket}/{key}: {e}")
        raise S3ReadError(
            f"Unexpected error reading from S3: {e}",
            bucket=bucket,
            key=key,
            original_error=e
        )


def write_to_s3(
    s3_client: Any,
    bucket: str,
    key: str,
    data: dict,
    content_type: str = 'application/json'
) -> None:
    """
    Write JSON data to S3 with proper exception handling.

    Args:
        s3_client: Boto3 S3 client instance.
        bucket: S3 bucket name.
        key: S3 object key.
        data: Dictionary to serialize as JSON.
        content_type: Content-Type header for the S3 object.

    Raises:
        S3WriteError: When the write operation fails.
    """
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, indent=2, default=str),
            ContentType=content_type
        )
        logger.info(f"Written to s3://{bucket}/{key}")
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')

        if error_code == 'AccessDenied':
            logger.error(f"Access denied writing to S3: s3://{bucket}/{key}")
            raise S3WriteError(
                f"Access denied writing to S3: {key}",
                bucket=bucket,
                key=key,
                original_error=e
            )
        else:
            logger.error(f"S3 write error ({error_code}): s3://{bucket}/{key} - {e}")
            raise S3WriteError(
                f"Failed to write to S3: {error_code}",
                bucket=bucket,
                key=key,
                original_error=e
            )
    except Exception as e:
        logger.error(f"Unexpected error writing to s3://{bucket}/{key}: {e}")
        raise S3WriteError(
            f"Unexpected error writing to S3: {e}",
            bucket=bucket,
            key=key,
            original_error=e
        )
