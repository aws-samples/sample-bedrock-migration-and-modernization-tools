"""
AWS Pricing API Collector (Smart Extraction)
Collects pricing data from AWS Pricing API for Bedrock services using smart adaptive extraction
"""

import json
import logging
from typing import List, Dict
import boto3
from botocore.exceptions import ClientError, BotoCoreError

from config import Config
from .smart_extractor import SmartModelExtractor


logger = logging.getLogger(__name__)


class AWSPricingAPICollector:
    """Collector for AWS Pricing API data with smart model extraction"""

    def __init__(self, profile_name: str = None):
        """
        Initialize the AWS Pricing API collector

        Args:
            profile_name: AWS profile name (defaults to config)
        """
        self.profile_name = profile_name or Config.AWS_PROFILE_NAME
        self.client = None
        self.service_codes = Config.AWS_PRICING_SERVICE_CODES
        self.smart_extractor = SmartModelExtractor()
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Pricing client"""
        try:
            session = boto3.Session(profile_name=self.profile_name)
            # Pricing API is only available in us-east-1 and ap-south-1
            self.client = session.client('pricing', region_name=Config.PRICING_API_REGION)
            logger.info("Initialized AWS Pricing API client")
        except Exception as e:
            logger.error(f"Failed to initialize Pricing client: {str(e)}")
            raise

    def collect_service_pricing(self, service_code: str) -> List[Dict]:
        """
        Collect pricing data for a specific service code

        Args:
            service_code: AWS service code (e.g., 'AmazonBedrock')

        Returns:
            List of pricing entries
        """
        logger.info(f"Collecting pricing data for service: {service_code}")

        all_pricing_data = []
        next_token = None
        batch_count = 0

        try:
            while True:
                batch_count += 1
                params = {
                    'ServiceCode': service_code,
                    'MaxResults': Config.MAX_RESULTS_PER_PAGE
                }

                if next_token:
                    params['NextToken'] = next_token

                response = self.client.get_products(**params)
                products = response.get('PriceList', [])

                if not products:
                    break

                logger.debug(f"Processing batch {batch_count}: {len(products)} products")

                # Process each product using smart extraction
                for product_json in products:
                    try:
                        pricing_entries = self._parse_product(product_json, service_code)
                        all_pricing_data.extend(pricing_entries)
                    except Exception as e:
                        logger.warning(f"Error parsing product: {str(e)}")
                        continue

                next_token = response.get('NextToken')
                if not next_token:
                    break

                # Safety limit
                if batch_count > Config.MAX_BATCHES_LIMIT:
                    logger.warning(f"Reached batch limit for {service_code}")
                    break

        except (ClientError, BotoCoreError) as e:
            logger.error(f"Error collecting pricing for {service_code}: {str(e)}")

        logger.info(f"Collected {len(all_pricing_data)} pricing entries from {service_code}")
        return all_pricing_data

    def _parse_product(self, product_json: str, service_code: str) -> List[Dict]:
        """
        Parse a single product JSON and extract pricing information using smart extraction

        Args:
            product_json: JSON string of the product
            service_code: Service code this product belongs to

        Returns:
            List of pricing entries for this product
        """
        pricing_entries = []

        try:
            product = json.loads(product_json)
            attrs = product.get('product', {}).get('attributes', {})

            # Use smart extractor for model info (replaces all old hardcoded logic)
            model_name, provider, model_id = self.smart_extractor.extract_model_info(attrs, service_code)

            # Extract other essential attributes
            usage_type = attrs.get('usagetype', '')
            location = attrs.get('location', '')
            operation = attrs.get('operation', '')

            # Extract pricing terms
            terms = product.get('terms', {}).get('OnDemand', {})

            for term_key, term_data in terms.items():
                price_dims = term_data.get('priceDimensions', {})

                for dim_key, dim_data in price_dims.items():
                    price_per_unit = dim_data.get('pricePerUnit', {})
                    usd_price = price_per_unit.get('USD', '0')

                    try:
                        price_value = float(usd_price)
                    except (ValueError, TypeError):
                        price_value = 0.0

                    # Only include non-zero prices
                    if price_value > 0:
                        pricing_entries.append({
                            'model_id': model_id,
                            'model_name': model_name,
                            'provider': provider,
                            'model_provider': provider,  # Add model_provider field
                            'dimension': usage_type,
                            'original_price': price_value,
                            'unit': dim_data.get('unit', ''),
                            'description': dim_data.get('description', ''),
                            'location': location,
                            'operation': operation,
                            'service_code': service_code,
                            'source_dataset': 'aws_pricing_api'
                        })

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in product data: {str(e)}")
        except Exception as e:
            logger.warning(f"Error parsing product: {str(e)}")

        return pricing_entries

    def collect_all_pricing_data(self) -> List[Dict]:
        """
        Collect pricing data from all configured service codes

        Returns:
            List of all pricing entries from AWS Pricing API
        """
        logger.info("Starting AWS Pricing API collection")

        all_pricing_data = []

        for service_code in self.service_codes:
            service_pricing = self.collect_service_pricing(service_code)
            all_pricing_data.extend(service_pricing)

        logger.info(f"AWS Pricing API collection complete: {len(all_pricing_data)} total entries")

        return all_pricing_data