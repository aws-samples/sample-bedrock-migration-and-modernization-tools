#!/usr/bin/env python3
"""
Local test runner for pricing-collector Lambda.

Usage:
    python test_pricing_collector_local.py [--service-code SERVICE_CODE] [--output-dir DIR]

This script tests the pricing collection logic locally without S3.
"""

import argparse
import json
import logging
import os
import sys
import time

# Add lambdas directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lambdas', 'pricing-collector'))

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PRICING_REGION = 'us-east-1'
RETRY_CONFIG = Config(
    retries={'max_attempts': 3, 'mode': 'adaptive'},
    connect_timeout=10,
    read_timeout=30
)


def collect_pricing_for_service(pricing_client, service_code: str, max_batches: int = 100) -> list:
    """Collect all pricing products for a given service code."""
    products = []
    next_token = None
    batch_count = 0

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

            for price_item in response.get('PriceList', []):
                try:
                    product = json.loads(price_item) if isinstance(price_item, str) else price_item
                    products.append(product)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse price item: {e}")
                    continue

            batch_count += 1
            next_token = response.get('NextToken')

            if not next_token:
                break

            if batch_count % 10 == 0:
                logger.info(f"Processed {batch_count} batches, {len(products)} products so far...")
                time.sleep(0.5)

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                logger.warning("Throttled, waiting before retry...")
                time.sleep(2)
                continue
            else:
                logger.error(f"ClientError: {e}")
                raise

    logger.info(f"Completed: {len(products)} products from {batch_count} batches")
    return products


def main():
    parser = argparse.ArgumentParser(description='Test pricing collector locally')
    parser.add_argument('--service-code', default='AmazonBedrock',
                        choices=['AmazonBedrock', 'AmazonBedrockService', 'AmazonBedrockFoundationModels'],
                        help='AWS Pricing service code')
    parser.add_argument('--output-dir', default='./output',
                        help='Output directory for JSON files')
    parser.add_argument('--max-batches', type=int, default=100,
                        help='Maximum number of API batches (for testing, use small number)')
    parser.add_argument('--profile', default=None,
                        help='AWS profile to use')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create pricing client
    session_kwargs = {}
    if args.profile:
        session_kwargs['profile_name'] = args.profile

    session = boto3.Session(**session_kwargs)
    pricing_client = session.client('pricing', region_name=PRICING_REGION, config=RETRY_CONFIG)

    # Verify credentials
    try:
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        logger.info(f"Using AWS account: {identity['Account']}, ARN: {identity['Arn']}")
    except Exception as e:
        logger.error(f"Failed to verify AWS credentials: {e}")
        sys.exit(1)

    # Collect pricing
    start_time = time.time()

    try:
        products = collect_pricing_for_service(
            pricing_client,
            args.service_code,
            max_batches=args.max_batches
        )

        duration_ms = int((time.time() - start_time) * 1000)

        # Structure output
        output_data = {
            'metadata': {
                'serviceCode': args.service_code,
                'recordCount': len(products),
                'collectionTimestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'pricingRegion': PRICING_REGION,
                'durationMs': duration_ms
            },
            'products': products
        }

        # Write to file
        output_file = os.path.join(args.output_dir, f'{args.service_code}.json')
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"Output written to: {output_file}")
        logger.info(f"Total products: {len(products)}")
        logger.info(f"Duration: {duration_ms}ms")

        # Print sample product
        if products:
            logger.info("Sample product structure:")
            sample = products[0]
            print(json.dumps({
                'product': sample.get('product', {}),
                'terms_keys': list(sample.get('terms', {}).keys()) if 'terms' in sample else []
            }, indent=2))

        return {
            'status': 'SUCCESS',
            'serviceCode': args.service_code,
            'recordCount': len(products),
            'durationMs': duration_ms,
            'outputFile': output_file
        }

    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        return {
            'status': 'FAILED',
            'errorType': type(e).__name__,
            'errorMessage': str(e)
        }


if __name__ == '__main__':
    result = main()
    print("\n" + "="*50)
    print("RESULT:")
    print(json.dumps(result, indent=2))
