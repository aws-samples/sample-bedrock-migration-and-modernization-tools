"""
Region Discovery Lambda

Dynamically discovers all AWS regions where Bedrock inference profiles are available.
This replaces hardcoded region lists with dynamic discovery.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))


def get_all_enabled_regions() -> list[str]:
    """Get all regions enabled in this AWS account."""
    ec2 = boto3.client('ec2', region_name='us-east-1')

    try:
        response = ec2.describe_regions(
            AllRegions=False,  # Only enabled regions
            Filters=[
                {'Name': 'opt-in-status', 'Values': ['opt-in-not-required', 'opted-in']}
            ]
        )
        regions = [r['RegionName'] for r in response.get('Regions', [])]
        logger.info(f"Found {len(regions)} enabled regions")
        return regions
    except ClientError as e:
        logger.error(f"Error getting regions: {e}")
        # Fallback to common regions
        return [
            'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2',
            'eu-west-1', 'eu-west-2', 'eu-west-3', 'eu-central-1', 'eu-north-1',
            'ap-northeast-1', 'ap-northeast-2', 'ap-south-1',
            'ap-southeast-1', 'ap-southeast-2',
            'ca-central-1', 'sa-east-1'
        ]


def check_bedrock_available(region: str) -> tuple[str, bool]:
    """Check if Bedrock inference profiles are available in a region."""
    try:
        bedrock = boto3.client('bedrock', region_name=region)
        # Try to list inference profiles - this will fail if Bedrock isn't available
        bedrock.list_inference_profiles(maxResults=1)
        return (region, True)
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code in ('UnrecognizedClientException', 'InvalidIdentityToken'):
            # Region not enabled or Bedrock not available
            logger.debug(f"Bedrock not available in {region}: {error_code}")
            return (region, False)
        elif error_code == 'AccessDeniedException':
            # Bedrock exists but we don't have access - still count it
            logger.debug(f"Bedrock exists in {region} but access denied")
            return (region, True)
        else:
            logger.warning(f"Error checking Bedrock in {region}: {e}")
            return (region, False)
    except Exception as e:
        logger.warning(f"Unexpected error checking {region}: {e}")
        return (region, False)


def discover_bedrock_regions(all_regions: list[str]) -> list[str]:
    """Discover which regions have Bedrock inference profiles available."""
    bedrock_regions = []

    # Check regions in parallel for speed
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(check_bedrock_available, region): region for region in all_regions}

        for future in as_completed(futures):
            region, available = future.result()
            if available:
                bedrock_regions.append(region)

    # Sort for consistent ordering
    bedrock_regions.sort()
    logger.info(f"Found {len(bedrock_regions)} regions with Bedrock inference profiles")
    return bedrock_regions


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for region discovery.

    Input:
        {} (no input required)

    Output:
        {
            "status": "SUCCESS",
            "featureRegions": ["us-east-1", "us-west-2", ...],
            "totalRegions": 27,
            "discoveryTimestamp": "2024-01-01T00:00:00Z"
        }
    """
    start_time = time.time()

    logger.info("Starting region discovery")

    try:
        # Get all enabled regions in the account
        all_regions = get_all_enabled_regions()
        logger.info(f"Checking {len(all_regions)} enabled regions for Bedrock availability")

        # Filter to regions with Bedrock inference profiles
        bedrock_regions = discover_bedrock_regions(all_regions)

        elapsed = time.time() - start_time
        logger.info(f"Region discovery completed in {elapsed:.2f}s, found {len(bedrock_regions)} Bedrock regions")

        return {
            'status': 'SUCCESS',
            'featureRegions': bedrock_regions,
            'totalRegions': len(bedrock_regions),
            'allEnabledRegions': len(all_regions),
            'discoveryTimestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }

    except Exception as e:
        logger.error(f"Region discovery failed: {e}")
        return {
            'status': 'FAILED',
            'errorType': type(e).__name__,
            'errorMessage': str(e),
            # Fallback to known good regions
            'featureRegions': [
                'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2',
                'eu-west-1', 'eu-west-2', 'eu-west-3', 'eu-central-1', 'eu-north-1',
                'ap-northeast-1', 'ap-northeast-2', 'ap-south-1',
                'ap-southeast-1', 'ap-southeast-2',
                'ca-central-1', 'sa-east-1'
            ],
            'totalRegions': 16
        }
