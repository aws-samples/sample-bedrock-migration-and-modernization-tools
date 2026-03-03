"""
Pricing Collector Lambda

Collects pricing data from AWS Pricing API for a single Bedrock service code.
Also fetches from AWS Bulk Pricing API for additional coverage (e.g., Stability AI).

Configuration (environment variables):
    PRICING_API_REGION: Region for Pricing API (default: us-east-1)
    PRICING_MAX_BATCHES: Safety limit for pagination (default: 100)
    PRICING_API_TIMEOUT: Timeout for bulk pricing API calls in seconds (default: 60)
    PRICING_THROTTLE_DELAY: Delay between batches to avoid throttling in seconds (default: 0.5)
    PRICING_RETRY_DELAY: Delay after throttling error in seconds (default: 2)
"""

import json
import os
import time
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
    get_config_loader,
)
from shared.powertools import logger, tracer, metrics, LambdaContext
from aws_lambda_powertools.metrics import MetricUnit

# Configuration with defaults
PRICING_API_REGION = os.environ.get("PRICING_API_REGION", "us-east-1")
PRICING_MAX_BATCHES = int(os.environ.get("PRICING_MAX_BATCHES", "100"))
PRICING_API_TIMEOUT = int(os.environ.get("PRICING_API_TIMEOUT", "60"))
PRICING_THROTTLE_DELAY = float(os.environ.get("PRICING_THROTTLE_DELAY", "0.5"))
PRICING_RETRY_DELAY = float(os.environ.get("PRICING_RETRY_DELAY", "2"))

DATA_BUCKET = os.environ.get("DATA_BUCKET")


def get_bulk_pricing_url_template() -> str:
    """Get bulk pricing URL template from config."""
    config = get_config_loader()
    return config.get_bulk_pricing_url()


def get_pricing_client():
    """Create Pricing API client (only available in us-east-1 or ap-south-1)."""
    return boto3.client("pricing", region_name=PRICING_API_REGION, config=RETRY_CONFIG)


def get_s3_client():
    """Create S3 client."""
    return boto3.client("s3", config=RETRY_CONFIG)


@tracer.capture_method
def fetch_bulk_pricing(service_code: str, region: str = "us-east-1") -> list[dict]:
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
    url_template = get_bulk_pricing_url_template()
    url = url_template.format(service_code=service_code, region=region)
    logger.info("Fetching bulk pricing", extra={"url": url})

    try:
        with urlopen(url, timeout=PRICING_API_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        logger.warning(
            "Bulk pricing API returned HTTP error",
            extra={
                "http_code": e.code,
                "service_code": service_code,
                "reason": e.reason,
            },
        )
        return []
    except URLError as e:
        logger.warning(
            "Failed to fetch bulk pricing",
            extra={"service_code": service_code, "reason": str(e.reason)},
        )
        return []
    except Exception as e:
        logger.warning(
            "Unexpected error fetching bulk pricing", extra={"error": str(e)}
        )
        return []

    products = []

    # Parse bulk pricing format into GetProducts-compatible format
    # Bulk pricing structure: { products: { sku: {...} }, terms: { OnDemand: { sku: {...} } } }
    bulk_products = data.get("products", {})
    bulk_terms = data.get("terms", {})
    on_demand_terms = bulk_terms.get("OnDemand", {})

    for sku, product_info in bulk_products.items():
        attributes = product_info.get("attributes", {})

        # Get the OnDemand terms for this SKU
        sku_terms = on_demand_terms.get(sku, {})

        # Convert to GetProducts format
        product = {
            "product": {"sku": sku, "attributes": attributes},
            "terms": {"OnDemand": sku_terms},
            "source": "bulk_pricing_api",
        }
        products.append(product)

    logger.info(
        "Fetched products from bulk pricing API",
        extra={"product_count": len(products), "service_code": service_code},
    )
    return products


@tracer.capture_method
def collect_pricing_for_service(pricing_client, service_code: str) -> list[dict]:
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

    logger.info("Starting pricing collection", extra={"service_code": service_code})

    while batch_count < PRICING_MAX_BATCHES:
        try:
            params = {
                "ServiceCode": service_code,
                "MaxResults": 100,
                "FormatVersion": "aws_v1",
            }

            if next_token:
                params["NextToken"] = next_token

            response = pricing_client.get_products(**params)

            # Parse price list items
            for price_item in response.get("PriceList", []):
                try:
                    product = (
                        json.loads(price_item)
                        if isinstance(price_item, str)
                        else price_item
                    )
                    products.append(product)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Failed to parse price item", extra={"error": str(e)}
                    )
                    continue

            batch_count += 1

            # Check for more results
            next_token = response.get("NextToken")
            if not next_token:
                break

            # Brief pause to avoid throttling
            if batch_count % 10 == 0:
                logger.info(
                    "Pricing collection progress",
                    extra={"batch_count": batch_count, "product_count": len(products)},
                )
                time.sleep(PRICING_THROTTLE_DELAY)

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ThrottlingException":
                logger.warning("Throttled, waiting before retry")
                time.sleep(PRICING_RETRY_DELAY)
                continue
            else:
                logger.error("ClientError collecting pricing", extra={"error": str(e)})
                raise

    logger.info(
        "Pricing collection completed",
        extra={"product_count": len(products), "batch_count": batch_count},
    )
    return products


@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
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
        validate_required_params(event, ["serviceCode"], "PricingCollector")
    except ValidationError as e:
        return {
            "status": "FAILED",
            "errorType": "ValidationError",
            "errorMessage": str(e),
        }

    # Extract parameters
    service_code = event["serviceCode"]
    s3_bucket = event.get("s3Bucket", DATA_BUCKET)
    s3_key = event.get("s3Key", f"test/{service_code}.json")
    dry_run = event.get("dryRun", False)

    logger.info(
        "Starting pricing collection",
        extra={"service_code": service_code, "bucket": s3_bucket, "dry_run": dry_run},
    )

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
        existing_skus = {p.get("product", {}).get("sku") for p in products}
        for bp in bulk_products:
            sku = bp.get("product", {}).get("sku")
            if sku and sku not in existing_skus:
                products.append(bp)
                existing_skus.add(sku)

        logger.info(
            "Combined pricing data",
            extra={
                "get_products_count": api_count,
                "bulk_api_count": bulk_count,
                "total_count": len(products),
            },
        )

        # Structure the output
        output_data = {
            "metadata": {
                "serviceCode": service_code,
                "recordCount": len(products),
                "getProductsCount": api_count,
                "bulkApiCount": bulk_count,
                "collectionTimestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "pricingRegion": PRICING_API_REGION,
            },
            "products": products,
        }

        # Write to S3 (skip in dry run mode)
        if not dry_run and s3_bucket:
            s3_client = get_s3_client()
            write_to_s3(s3_client, s3_bucket, s3_key, output_data)
        else:
            logger.info(
                "Dry run mode - skipping S3 write",
                extra={"bucket": s3_bucket, "key": s3_key},
            )

        duration_ms = int((time.time() - start_time) * 1000)

        # Add metrics
        metrics.add_metric(
            name="ProductsCollected", unit=MetricUnit.Count, value=len(products)
        )
        metrics.add_metric(
            name="CollectionDurationMs", unit=MetricUnit.Milliseconds, value=duration_ms
        )

        logger.info(
            "Pricing collection complete",
            extra={"product_count": len(products), "duration_ms": duration_ms},
        )

        return {
            "status": "SUCCESS",
            "serviceCode": service_code,
            "s3Key": s3_key,
            "recordCount": len(products),
            "durationMs": duration_ms,
            "dryRun": dry_run,
        }

    except Exception as e:
        logger.exception(
            "Failed to collect pricing",
            extra={"service_code": service_code, "error": str(e)},
        )

        return {
            "status": "FAILED",
            "serviceCode": service_code,
            "errorType": type(e).__name__,
            "errorMessage": str(e),
            "retryable": isinstance(e, (ClientError,)) and "Throttling" in str(e),
        }
