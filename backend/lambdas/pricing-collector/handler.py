"""
Pricing Collector Lambda

Collects pricing data from AWS Pricing API for a single Bedrock service code.
Also fetches from AWS Bulk Pricing API for additional coverage (e.g., Stability AI).
"""

import json
import logging
import os
import time
from typing import Any
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

import boto3
from botocore.exceptions import ClientError

from shared import (
    RETRY_CONFIG,
    write_to_s3,
    parse_execution_id,
    validate_required_params,
    ValidationError,
    S3WriteError,
)

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

# AWS clients
PRICING_REGION = os.environ.get('PRICING_API_REGION', 'us-east-1')
DATA_BUCKET = os.environ.get('DATA_BUCKET')

# Bulk Pricing API URL template
# Available regions for bulk pricing: us-east-1, ap-south-1
BULK_PRICING_URL = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/{service_code}/current/{region}/index.json"


def get_pricing_client():
    """Create Pricing API client (only available in us-east-1 or ap-south-1)."""
    return boto3.client('pricing', region_name=PRICING_REGION, config=RETRY_CONFIG)


def get_s3_client():
    """Create S3 client."""
    return boto3.client('s3', config=RETRY_CONFIG)


def fetch_bulk_pricing(service_code: str, region: str = 'us-east-1') -> list[dict]:
    """
    Fetch pricing from AWS Bulk Pricing API (public HTTPS endpoint).

    This provides additional coverage for models not in the GetProducts API,
    such as Stability AI models.

    Args:
        service_code: AWS service code (e.g., 'AmazonBedrockFoundationModels')
        region: Region for pricing (default: us-east-1)

    Returns:
        List of product dictionaries in GetProducts-compatible format
    """
    url = BULK_PRICING_URL.format(service_code=service_code, region=region)
    logger.info(f"Fetching bulk pricing from: {url}")

    try:
        with urlopen(url, timeout=60) as response:
            data = json.loads(response.read().decode('utf-8'))
    except HTTPError as e:
        logger.warning(f"Bulk pricing API returned HTTP {e.code} for {service_code}: {e.reason}")
        return []
    except URLError as e:
        logger.warning(f"Failed to fetch bulk pricing for {service_code}: {e.reason}")
        return []
    except Exception as e:
        logger.warning(f"Unexpected error fetching bulk pricing: {e}")
        return []

    products = []

    # Parse bulk pricing format into GetProducts-compatible format
    # Bulk pricing structure: { products: { sku: {...} }, terms: { OnDemand: { sku: {...} } } }
    bulk_products = data.get('products', {})
    bulk_terms = data.get('terms', {})
    on_demand_terms = bulk_terms.get('OnDemand', {})

    for sku, product_info in bulk_products.items():
        attributes = product_info.get('attributes', {})

        # Get the OnDemand terms for this SKU
        sku_terms = on_demand_terms.get(sku, {})

        # Convert to GetProducts format
        product = {
            'product': {
                'sku': sku,
                'attributes': attributes
            },
            'terms': {
                'OnDemand': sku_terms
            },
            'source': 'bulk_pricing_api'
        }
        products.append(product)

    logger.info(f"Fetched {len(products)} products from bulk pricing API for {service_code}")
    return products


def collect_pricing_for_service(pricing_client: Any, service_code: str) -> list[dict]:
    """
    Collect all pricing products for a given service code.

    Args:
        pricing_client: Boto3 Pricing client
        service_code: AWS service code (e.g., 'AmazonBedrock')

    Returns:
        List of pricing product dictionaries
    """
    products = []
    next_token = None
    batch_count = 0
    max_batches = 100  # Safety limit

    logger.info(f"Starting pricing collection for service: {service_code}")

    while batch_count < max_batches:
        try:
            params = {
                'ServiceCode': service_code,
                'MaxResults': 100,
                'FormatVersion': 'aws_v1'
            }

            if next_token:
                params['NextToken'] = next_token

            response = pricing_client.get_products(**params)

            # Parse price list items
            for price_item in response.get('PriceList', []):
                try:
                    product = json.loads(price_item) if isinstance(price_item, str) else price_item
                    products.append(product)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse price item: {e}")
                    continue

            batch_count += 1

            # Check for more results
            next_token = response.get('NextToken')
            if not next_token:
                break

            # Brief pause to avoid throttling
            if batch_count % 10 == 0:
                logger.info(f"Processed {batch_count} batches, {len(products)} products so far...")
                time.sleep(0.5)

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                logger.warning(f"Throttled, waiting before retry...")
                time.sleep(2)
                continue
            else:
                logger.error(f"ClientError collecting pricing: {e}")
                raise

    logger.info(f"Completed: {len(products)} products from {batch_count} batches")
    return products


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for pricing collection.

    Input:
        {
            "serviceCode": "AmazonBedrock",
            "s3Bucket": "bucket-name",
            "s3Key": "executions/{id}/pricing/AmazonBedrock.json",
            "dryRun": false  // Optional: skip S3 write for testing
        }

    Output:
        {
            "status": "SUCCESS",
            "serviceCode": "AmazonBedrock",
            "s3Key": "executions/{id}/pricing/AmazonBedrock.json",
            "recordCount": 1250,
            "durationMs": 45000
        }
    """
    start_time = time.time()

    # Validate required parameters
    try:
        validate_required_params(event, ['serviceCode'], 'PricingCollector')
    except ValidationError as e:
        return {
            'status': 'FAILED',
            'errorType': 'ValidationError',
            'errorMessage': str(e)
        }

    # Extract parameters
    service_code = event['serviceCode']
    s3_bucket = event.get('s3Bucket', DATA_BUCKET)
    s3_key = event.get('s3Key', f'test/{service_code}.json')
    dry_run = event.get('dryRun', False)

    logger.info(f"Starting pricing collection: service={service_code}, bucket={s3_bucket}, dryRun={dry_run}")

    try:
        # Initialize clients
        pricing_client = get_pricing_client()

        # Collect pricing data from GetProducts API
        products = collect_pricing_for_service(pricing_client, service_code)
        api_count = len(products)

        # Also try Bulk Pricing API for additional coverage
        # This catches models like Stability AI that aren't in GetProducts
        bulk_products = fetch_bulk_pricing(service_code)
        bulk_count = len(bulk_products)

        # Merge products, avoiding duplicates by SKU
        existing_skus = {p.get('product', {}).get('sku') for p in products}
        for bp in bulk_products:
            sku = bp.get('product', {}).get('sku')
            if sku and sku not in existing_skus:
                products.append(bp)
                existing_skus.add(sku)

        logger.info(f"Combined {api_count} GetProducts + {bulk_count} Bulk API = {len(products)} total (after dedup)")

        # Structure the output
        output_data = {
            'metadata': {
                'serviceCode': service_code,
                'recordCount': len(products),
                'getProductsCount': api_count,
                'bulkApiCount': bulk_count,
                'collectionTimestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'pricingRegion': PRICING_REGION
            },
            'products': products
        }

        # Write to S3 (skip in dry run mode)
        if not dry_run and s3_bucket:
            s3_client = get_s3_client()
            write_to_s3(s3_client, s3_bucket, s3_key, output_data)
        else:
            logger.info(f"Dry run mode - skipping S3 write. Would write to s3://{s3_bucket}/{s3_key}")

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            'status': 'SUCCESS',
            'serviceCode': service_code,
            's3Key': s3_key,
            'recordCount': len(products),
            'durationMs': duration_ms,
            'dryRun': dry_run
        }

    except Exception as e:
        logger.error(f"Failed to collect pricing: {e}", exc_info=True)

        return {
            'status': 'FAILED',
            'serviceCode': service_code,
            'errorType': type(e).__name__,
            'errorMessage': str(e),
            'retryable': isinstance(e, (ClientError,)) and 'Throttling' in str(e)
        }
