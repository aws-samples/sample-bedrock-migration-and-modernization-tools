"""
AWS Pricing API Collector (Smart Extraction)
Collects pricing data from AWS Pricing API for Bedrock services using smart adaptive extraction
"""

import json
import logging
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore.exceptions import ClientError, BotoCoreError

from config import Config
from .smart_extractor import SmartModelExtractor


logger = logging.getLogger(__name__)


class AWSPricingAPICollector:
    """Collector for AWS Pricing API data with smart model extraction"""

    def __init__(self, profile_name: str = None, use_parallel: bool = True, max_workers: int = 3):
        """
        Initialize the AWS Pricing API collector

        Args:
            profile_name: AWS profile name (defaults to config)
            use_parallel: Enable parallel collection of service codes (default: True)
            max_workers: Number of parallel workers for service code collection (default: 3)
        """
        self.profile_name = profile_name or Config.AWS_PROFILE_NAME
        self.use_parallel = use_parallel
        self.max_workers = max_workers
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
            logger.warning(f"Failed to initialize Pricing client: {str(e)}")
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
        Uses parallel processing by default for faster execution

        Returns:
            List of all pricing entries from AWS Pricing API
        """
        if self.use_parallel:
            return self._collect_all_pricing_data_parallel()
        else:
            return self._collect_all_pricing_data_sequential()

    def _collect_all_pricing_data_parallel(self) -> List[Dict]:
        """
        Collect pricing data from all service codes IN PARALLEL
        Provides significant speedup when collecting from 3 service codes
        """
        logger.info(f"🚀 Starting AWS Pricing API collection (parallel, workers={self.max_workers})")
        logger.info(f"Service codes: {self.service_codes}")

        all_pricing_data = []
        start_time = time.time()

        # Use ThreadPoolExecutor for parallel collection
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all service code collection tasks
            future_to_service = {
                executor.submit(self.collect_service_pricing, service_code): service_code
                for service_code in self.service_codes
            }

            # Collect results as they complete
            for future in as_completed(future_to_service):
                service_code = future_to_service[future]

                try:
                    service_pricing = future.result()
                    all_pricing_data.extend(service_pricing)
                    logger.info(f"✅ [{service_code}] Collected {len(service_pricing)} entries")

                except Exception as e:
                    logger.error(f"❌ [{service_code}] Failed: {e}")

        elapsed_time = time.time() - start_time
        logger.info(f"✅ Parallel AWS Pricing API collection complete in {elapsed_time:.2f}s")
        logger.info(f"Total entries before deduplication: {len(all_pricing_data)}")

        # Apply deduplication to prioritize AmazonBedrockFoundationModels over AmazonBedrock
        deduplicated_data = self._deduplicate_pricing_entries(all_pricing_data)
        logger.info(f"Total entries after deduplication: {len(deduplicated_data)}")

        return deduplicated_data

    def _collect_all_pricing_data_sequential(self) -> List[Dict]:
        """
        Collect pricing data from all service codes SEQUENTIALLY (legacy method)
        Note: Parallel collection is recommended for better performance
        """
        logger.info("Starting AWS Pricing API collection (sequential)")
        logger.info(f"Service codes: {self.service_codes}")

        all_pricing_data = []

        for service_code in self.service_codes:
            service_pricing = self.collect_service_pricing(service_code)
            all_pricing_data.extend(service_pricing)

        logger.info(f"Total entries before deduplication: {len(all_pricing_data)}")

        # Apply deduplication to prioritize AmazonBedrockFoundationModels over AmazonBedrock
        deduplicated_data = self._deduplicate_pricing_entries(all_pricing_data)
        logger.info(f"AWS Pricing API collection complete: {len(deduplicated_data)} total entries after deduplication")

        return deduplicated_data

    def _deduplicate_pricing_entries(self, all_pricing_data: List[Dict]) -> List[Dict]:
        """
        Remove duplicate pricing entries using model-level service code prioritization

        Args:
            all_pricing_data: List of all pricing entries from all service codes

        Returns:
            List of deduplicated pricing entries
        """
        logger.info("🔍 Starting model-level pricing deduplication")
        logger.info(f"📊 Input: {len(all_pricing_data)} pricing entries to deduplicate")

        # Track service code statistics
        service_code_stats = {}
        for entry in all_pricing_data:
            service_code = entry.get('service_code', 'unknown')
            service_code_stats[service_code] = service_code_stats.get(service_code, 0) + 1

        # Log service code distribution
        logger.info("📈 Service code distribution:")
        for service_code, count in sorted(service_code_stats.items()):
            logger.info(f"  • {service_code}: {count} entries")

        # Group entries by model ID
        by_model = {}
        for entry in all_pricing_data:
            model_id = self._normalize_model_id_for_grouping(entry.get('model_id', 'unknown'))
            if model_id not in by_model:
                by_model[model_id] = []
            by_model[model_id].append(entry)

        logger.info(f"🎯 Found {len(by_model)} unique models")

        # Process each model and apply model-level deduplication
        deduplicated_entries = []
        total_removed = 0
        models_with_conflicts = 0

        for model_id, model_entries in by_model.items():
            # Group entries by service code for this model
            by_service_code = {}
            for entry in model_entries:
                service_code = entry.get('service_code', 'unknown')
                if service_code not in by_service_code:
                    by_service_code[service_code] = []
                by_service_code[service_code].append(entry)

            if len(by_service_code) == 1:
                # No service code conflicts for this model, keep all entries
                deduplicated_entries.extend(model_entries)
            else:
                # Multiple service codes for this model - apply your suggested logic
                models_with_conflicts += 1

                # Find service code with most entries (your suggestion)
                best_service_code = max(by_service_code.items(), key=lambda x: len(x[1]))[0]
                best_entries = by_service_code[best_service_code]

                # Use priority as tiebreaker if counts are equal
                if len(set(len(entries) for entries in by_service_code.values())) == 1:
                    # All service codes have same count, use priority order
                    service_priority = ["AmazonBedrockFoundationModels", "AmazonBedrock", "AmazonBedrockService"]
                    for priority_service in service_priority:
                        if priority_service in by_service_code:
                            best_service_code = priority_service
                            best_entries = by_service_code[priority_service]
                            break

                # Keep only the best service code entries
                deduplicated_entries.extend(best_entries)

                # Count removed entries
                removed_for_model = len(model_entries) - len(best_entries)
                total_removed += removed_for_model

                # Log the deduplication decision
                removed_services = [sc for sc in by_service_code.keys() if sc != best_service_code]
                logger.debug(f"🔄 Model {model_id}: kept {len(best_entries)} from {best_service_code}, removed {removed_for_model} from {removed_services}")

        # Calculate final service code distribution
        final_service_code_stats = {}
        for entry in deduplicated_entries:
            service_code = entry.get('service_code', 'unknown')
            final_service_code_stats[service_code] = final_service_code_stats.get(service_code, 0) + 1

        logger.info(f"✅ Model-level deduplication complete: removed {total_removed} duplicate entries")
        logger.info(f"📊 Models processed: {len(by_model)}, models with conflicts: {models_with_conflicts}")
        logger.info(f"📊 Breakdown: {len(all_pricing_data)} original → {len(deduplicated_entries)} deduplicated ({((total_removed) / len(all_pricing_data) * 100):.1f}% reduction)")

        logger.info("📈 Final service code distribution:")
        for service_code, count in sorted(final_service_code_stats.items()):
            original_count = service_code_stats.get(service_code, 0)
            reduction = original_count - count
            logger.info(f"  • {service_code}: {count} entries (removed {reduction})")

        # Verify deduplication approach worked
        foundation_models_count = final_service_code_stats.get('AmazonBedrockFoundationModels', 0)
        bedrock_count = final_service_code_stats.get('AmazonBedrock', 0)
        if foundation_models_count > 0 and bedrock_count > 0:
            logger.info("ℹ️  Both AmazonBedrockFoundationModels and AmazonBedrock entries remain (different models)")
        elif foundation_models_count > 0:
            logger.info("✅ Model-level deduplication favoring comprehensive service codes working correctly")

        return deduplicated_entries

    def _normalize_model_id_for_grouping(self, model_id: str) -> str:
        """
        Normalize model ID for grouping purposes in model-level deduplication

        Args:
            model_id: Original model ID

        Returns:
            Normalized model ID for grouping
        """
        if not model_id:
            return 'unknown'

        normalized = model_id.lower().strip()

        # Remove version suffixes that don't affect model identity for pricing purposes
        normalized = normalized.replace(':0', '')
        normalized = normalized.replace('-v1', '')
        normalized = normalized.replace('bedrock-', '')

        # Remove date stamps (like 20240307)
        import re
        normalized = re.sub(r'-\d{8}', '', normalized)

        return normalized

    def _create_semantic_unique_key(self, entry: Dict) -> tuple:
        """
        Create a semantic unique key that identifies the same pricing across different service codes

        Args:
            entry: Pricing entry dictionary

        Returns:
            Tuple representing the semantic unique identifier
        """
        # Extract core pricing components
        model_id = entry.get('model_id', '').lower().strip()

        # Normalize model ID to handle variations
        # Remove common suffixes/prefixes that don't affect pricing semantics
        model_id = model_id.replace('bedrock-', '').replace('-v1', '').replace(':0', '')

        price = round(entry.get('original_price', 0), 8)  # Higher precision for better matching
        unit = entry.get('unit', '').lower().strip()

        # Extract region from location (standardize format)
        region = self._extract_region_semantic(entry)

        # Extract pricing type (input/output tokens, storage, etc.)
        pricing_type = self._extract_pricing_type_semantic(entry)

        # Extract operation type (inference, training, etc.)
        operation_type = self._extract_operation_type_semantic(entry)

        # Normalize description for additional semantic matching
        description = entry.get('description', '').lower().strip()
        # Extract key terms from description for semantic comparison
        semantic_description = self._normalize_description_for_matching(description)

        # Create semantic key focusing on business meaning, not formatting
        semantic_key = (
            model_id,           # Same model (normalized)
            region,             # Same region (normalized)
            pricing_type,       # Same pricing type (input/output/storage/etc.)
            price,              # Same price value (higher precision)
            unit,               # Same unit
            operation_type,     # Same operation (inference/training/etc.)
            semantic_description  # Normalized description for better matching
        )

        return semantic_key

    def _normalize_description_for_matching(self, description: str) -> str:
        """
        Normalize description text for semantic matching

        Args:
            description: Original description text

        Returns:
            Normalized description for matching purposes
        """
        if not description:
            return ''

        # Remove common variations and normalize
        normalized = description.lower().strip()

        # Remove service code prefixes that don't affect semantics
        normalized = normalized.replace('amazombedrock', '').replace('amazonbedrock', '')
        normalized = normalized.replace('amazon bedrock', '').replace('bedrock', '')

        # Extract key semantic elements
        key_terms = []

        # Token type indicators
        if 'input' in normalized:
            key_terms.append('input')
        if 'output' in normalized or 'response' in normalized:
            key_terms.append('output')

        # Model size/type indicators
        if 'haiku' in normalized:
            key_terms.append('haiku')
        elif 'sonnet' in normalized:
            key_terms.append('sonnet')
        elif 'opus' in normalized:
            key_terms.append('opus')

        # Context indicators
        if 'long' in normalized and 'context' in normalized:
            key_terms.append('long_context')

        # Operation type indicators
        if 'inference' in normalized:
            key_terms.append('inference')
        if 'training' in normalized:
            key_terms.append('training')
        if 'batch' in normalized:
            key_terms.append('batch')

        # Sort for consistent matching
        key_terms.sort()

        return '_'.join(key_terms) if key_terms else normalized[:50]  # Truncate if no key terms

    def _extract_region_semantic(self, entry: Dict) -> str:
        """Extract and normalize region from entry"""
        # Try location field first
        location = entry.get('location', '').lower()

        # Common location to region mappings (case insensitive)
        location_mappings = {
            'us east (n. virginia)': 'us-east-1',
            'us east (ohio)': 'us-east-2',
            'us west (oregon)': 'us-west-2',
            'us west (n. california)': 'us-west-1',
            'eu (ireland)': 'eu-west-1',
            'europe (ireland)': 'eu-west-1',
            'eu (london)': 'eu-west-2',
            'europe (london)': 'eu-west-2',
            'eu (paris)': 'eu-west-3',
            'europe (paris)': 'eu-west-3',
            'eu (frankfurt)': 'eu-central-1',
            'europe (frankfurt)': 'eu-central-1',
            'asia pacific (tokyo)': 'ap-northeast-1',
            'asia pacific (seoul)': 'ap-northeast-2',
            'asia pacific (singapore)': 'ap-southeast-1',
            'asia pacific (sydney)': 'ap-southeast-2',
            'asia pacific (mumbai)': 'ap-south-1',
            'canada (central)': 'ca-central-1',
            'south america (são paulo)': 'sa-east-1',
            'south america (sao paulo)': 'sa-east-1'
        }

        if location in location_mappings:
            return location_mappings[location]

        # Try extracting from dimension (e.g., "USE1-MP:USE1_InputTokenCount-Units")
        dimension = entry.get('dimension', '')
        if dimension:
            # Extract region code from dimension
            import re
            region_match = re.search(r'^([A-Z]{2,4}\d?)[-_]', dimension)
            if region_match:
                region_code = region_match.group(1).upper()
                # Map region codes to full region names
                code_mappings = {
                    'USE1': 'us-east-1', 'USE2': 'us-east-2',
                    'USW1': 'us-west-1', 'USW2': 'us-west-2',
                    'EUW1': 'eu-west-1', 'EUW2': 'eu-west-2', 'EUW3': 'eu-west-3',
                    'EUC1': 'eu-central-1', 'EUC2': 'eu-central-2',
                    'EUN1': 'eu-north-1', 'EUS1': 'eu-south-1', 'EUS2': 'eu-south-2',
                    'APN1': 'ap-northeast-1', 'APN2': 'ap-northeast-2', 'APN3': 'ap-northeast-3',
                    'APS1': 'ap-southeast-1', 'APS2': 'ap-southeast-2', 'APS3': 'ap-southeast-3',
                    'APS4': 'ap-southeast-4', 'APE1': 'ap-east-1',
                    'CAN1': 'ca-central-1', 'SAE1': 'sa-east-1'
                }
                if region_code in code_mappings:
                    return code_mappings[region_code]

        return 'unknown'

    def _extract_pricing_type_semantic(self, entry: Dict) -> str:
        """Extract semantic pricing type (input tokens, output tokens, storage, etc.)"""
        description = entry.get('description', '').lower()
        dimension = entry.get('dimension', '').lower()

        # Look for input token patterns
        if any(pattern in description or pattern in dimension for pattern in [
            'input', 'inputtoken', 'input token', 'inputtokencount'
        ]):
            return 'input_tokens'

        # Look for output token patterns
        if any(pattern in description or pattern in dimension for pattern in [
            'output', 'response', 'outputtoken', 'output token', 'responsetoken'
        ]):
            return 'output_tokens'

        # Look for storage patterns
        if any(pattern in description or pattern in dimension for pattern in [
            'storage', 'month', 'customized model storage'
        ]):
            return 'storage'

        # Look for training patterns
        if any(pattern in description or pattern in dimension for pattern in [
            'training', 'fine-tuning', 'customization'
        ]):
            return 'training'

        # Look for provisioned throughput patterns
        if any(pattern in description or pattern in dimension for pattern in [
            'provisioned', 'model units', 'throughput'
        ]):
            return 'provisioned'

        # Look for batch patterns
        if any(pattern in description or pattern in dimension for pattern in [
            'batch', 'bulk'
        ]):
            return 'batch'

        return 'general'

    def _extract_operation_type_semantic(self, entry: Dict) -> str:
        """Extract operation type (inference, training, etc.)"""
        operation = entry.get('operation', '').lower()
        description = entry.get('description', '').lower()

        if 'training' in operation or 'training' in description:
            return 'training'
        elif 'inference' in operation or 'inference' in description:
            return 'inference'
        elif 'storage' in operation or 'storage' in description:
            return 'storage'
        else:
            return 'inference'  # Default assumption