#!/usr/bin/env python3
"""
Amazon Bedrock Pricing Collector

Collects comprehensive pricing data from AWS Pricing API using 3 service codes:
- AmazonBedrock (1st/2nd party models)
- AmazonBedrockService (2nd party infrastructure)
- AmazonBedrockFoundationModels (3rd party models)
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from config import Config
from utils.aws_pricing_api import AWSPricingAPICollector
from utils.data_processor import BedrockPricingDataProcessor


def setup_logging() -> None:
    """Setup logging configuration"""
    # Ensure logs directory exists
    logs_dir = Path(__file__).parent.parent / Config.LOGS_DIR
    logs_dir.mkdir(exist_ok=True)

    log_file_path = logs_dir / Config.LOG_FILENAME

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file_path)
        ]
    )


def ensure_output_directory() -> Path:
    """Ensure output directory exists"""
    output_dir = Path(__file__).parent.parent / Config.OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    return output_dir


def collect_pricing_data() -> Tuple[List[Dict], bool]:
    """
    Collect pricing data from AWS Pricing API (all 3 service codes)

    Returns:
        Tuple containing:
        - List of pricing entries from AWS Pricing API
        - Boolean indicating collection success
    """
    logger = logging.getLogger(__name__)

    logger.info("=== Starting AWS Pricing API Collection (3 service codes) ===")
    logger.info(f"Service codes: {Config.AWS_PRICING_SERVICE_CODES}")

    try:
        pricing_api_collector = AWSPricingAPICollector()
        pricing_data = pricing_api_collector.collect_all_pricing_data()

        logger.info(f"✅ AWS Pricing API: {len(pricing_data)} entries collected")
        logger.info("=== Collection Summary ===")
        logger.info(f"Total entries collected: {len(pricing_data)}")
        logger.info("✅ Collection successful - using simplified AWS Pricing API approach")

        return pricing_data, True

    except Exception as e:
        logger.error(f"❌ AWS Pricing API collection failed: {str(e)}")
        return [], False


def process_and_save_data(pricing_data: List[Dict], success: bool, output_dir: Path) -> str:
    """
    Process collected data and save to JSON file

    Args:
        pricing_data: Raw pricing data from AWS Pricing API
        success: Whether collection was successful
        output_dir: Output directory path

    Returns:
        Path to the saved file
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Processing {len(pricing_data)} total pricing entries")

    # Create simple stats for the processor
    stats = {
        'aws_pricing_api': {'success': success, 'count': len(pricing_data), 'error': None}
    }

    # Process data using the data processor
    processor = BedrockPricingDataProcessor()
    final_structure = processor.create_final_structure(pricing_data, stats)

    # Generate timestamp-based filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{Config.OUTPUT_FILE_PREFIX}-{timestamp}.json"
    filepath = output_dir / filename

    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(final_structure, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Data saved to: {filepath}")
    logger.info(f"Final structure contains:")
    logger.info(f"  - Providers: {final_structure['metadata']['providers_count']}")
    logger.info(f"  - Total pricing entries: {final_structure['metadata']['total_pricing_entries']}")
    logger.info(f"  - Total regions processed: {final_structure['metadata']['total_regions_processed']}")
    logger.info(f"  - Total groups created: {final_structure['metadata']['total_groups_created']}")

    return str(filepath)


def main():
    """
    Main execution function for Amazon Bedrock Pricing Collector

    Orchestrates the complete pricing collection workflow:
    1. Setup logging and output directories
    2. Update region mappings from AWS documentation
    3. Collect pricing from AWS Pricing API (3 service codes)
    4. Process and save structured pricing data
    """
    print("🚀 Amazon Bedrock Pricing Collector (AWS API Only)")
    print("=" * 55)
    print("📋 Collecting from 3 service codes for complete coverage")
    print("⚡ Simplified, fast, and reliable approach")
    print("=" * 55)

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize environment
        output_dir = ensure_output_directory()
        logger.info(f"Output directory: {output_dir}")

        # Update region mappings from AWS documentation
        logger.info("=== Updating Region Mappings ===")
        Config.update_region_mappings()

        # Collect pricing data using AWS Pricing API only
        pricing_data, success = collect_pricing_data()

        if success:
            # Process and save the collected data
            output_file = process_and_save_data(pricing_data, success, output_dir)

            # Display success summary
            print("\n🎉 Collection Complete!")
            print(f"📄 Output file: {Path(output_file).name}")
            print(f"📊 Total entries: {len(pricing_data):,}")
            print("✅ Status: SUCCESS")
            print("🔧 AWS Pricing API (3 service codes)")
        else:
            print("\n❌ Collection Failed!")
            print("⚠️  Check pricing_collection.log for details")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        print(f"\n❌ Collection failed: {str(e)}")
        print("💡 Check your AWS credentials and network connection")
        sys.exit(1)


if __name__ == "__main__":
    main()