#!/usr/bin/env python3
"""
Amazon Bedrock Model Collector
Main script that orchestrates comprehensive model data collection from multiple sources
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from config import Config
from utils.region_collector import BedrockRegionCollector
from utils.model_collector import BedrockModelCollector
from utils.features_enhancer import ModelFeaturesEnhancer
from utils.pricing_integrator import PricingIntegrator
from utils.quotas_collector import ServiceQuotasCollector
from utils.data_processor import ModelDataProcessor


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


def collect_bedrock_data() -> Dict[str, Any]:
    """
    Collect comprehensive Bedrock model data from all sources

    Returns:
        Dictionary with collected data from each phase
    """
    logger = logging.getLogger(__name__)

    collection_data = {
        'regions': [],
        'raw_models': {},
        'enhanced_models': {},
        'pricing_data': {},
        'quotas_data': {},
        'statistics': {}
    }

    # Phase 1: Discover Bedrock regions
    logger.info("=== Phase 1: Discovering Bedrock Regions ===")
    try:
        region_collector = BedrockRegionCollector(
            profile_name=Config.AWS_PROFILE_NAME,
            default_region=Config.AWS_REGION
        )
        bedrock_regions = region_collector.discover_bedrock_regions()
        collection_data['regions'] = bedrock_regions
        logger.info(f"✅ Found Bedrock in {len(bedrock_regions)} regions")
    except Exception as e:
        logger.error(f"❌ Region discovery failed: {str(e)}")
        raise

    # Phase 2: Collect models from all regions (multi-threaded)
    logger.info("=== Phase 2: Collecting Models from All Regions ===")
    try:
        model_collector = BedrockModelCollector(
            profile_name=Config.AWS_PROFILE_NAME,
            regions=bedrock_regions,
            max_workers=Config.MAX_WORKER_THREADS,
            use_direct_api=Config.USE_DIRECT_API
        )
        raw_models = model_collector.collect_models_all_regions()
        collection_data['raw_models'] = raw_models
        logger.info(f"✅ Collected {len(raw_models)} unique models")
    except Exception as e:
        logger.error(f"❌ Model collection failed: {str(e)}")
        raise

    # Phase 3: Enhance models with additional features
    logger.info("=== Phase 3: Enhancing Models with Additional Features ===")
    try:
        features_enhancer = ModelFeaturesEnhancer(
            profile_name=Config.AWS_PROFILE_NAME,
            regions=bedrock_regions
        )
        enhanced_models = features_enhancer.enhance_models(raw_models)
        collection_data['enhanced_models'] = enhanced_models
        logger.info(f"✅ Enhanced {len(enhanced_models)} models with comprehensive features")
    except Exception as e:
        logger.error(f"❌ Model enhancement failed: {str(e)}")
        raise

    # Phase 4: Integrate pricing data
    logger.info("=== Phase 4: Integrating Pricing Data ===")
    try:
        pricing_integrator = PricingIntegrator(
            pricing_collector_path=Config.get_pricing_collector_path()
        )
        pricing_data = pricing_integrator.integrate_pricing_data(enhanced_models)
        collection_data['pricing_data'] = pricing_data
        logger.info(f"✅ Integrated pricing data for models")
    except Exception as e:
        logger.error(f"❌ Pricing integration failed: {str(e)}")
        # Continue without pricing data
        logger.warning("Continuing without pricing data...")

    # Phase 5: Collect service quotas
    logger.info("=== Phase 5: Collecting Service Quotas ===")
    try:
        quotas_collector = ServiceQuotasCollector(
            profile_name=Config.AWS_PROFILE_NAME,
            regions=bedrock_regions
        )
        quotas_data = quotas_collector.collect_service_quotas()
        collection_data['quotas_data'] = quotas_data
        logger.info(f"✅ Collected service quotas from {len(bedrock_regions)} regions")
    except Exception as e:
        logger.error(f"❌ Service quotas collection failed: {str(e)}")
        # Continue without quota data
        logger.warning("Continuing without service quota data...")

    return collection_data


def process_and_save_data(collection_data: Dict[str, Any], output_dir: Path) -> str:
    """
    Process collected data and save to comprehensive JSON file

    Args:
        collection_data: Raw collected data from all phases
        output_dir: Output directory path

    Returns:
        Path to the saved file
    """
    logger = logging.getLogger(__name__)

    logger.info("=== Processing and Structuring Data ===")

    # Process data using the comprehensive data processor
    processor = ModelDataProcessor()
    final_structure = processor.create_comprehensive_structure(
        raw_models=collection_data['raw_models'],
        enhanced_models=collection_data['enhanced_models'],
        pricing_data=collection_data['pricing_data'],
        quotas_data=collection_data['quotas_data'],
        regions=collection_data['regions']
    )

    # Generate timestamp-based filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{Config.OUTPUT_FILE_PREFIX}-{timestamp}.json"
    filepath = output_dir / filename

    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(final_structure, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Data saved to: {filepath}")

    # Log comprehensive statistics
    metadata = final_structure.get('metadata', {})
    logger.info(f"Final structure contains:")
    logger.info(f"  - Providers: {metadata.get('providers_count', 0)}")
    logger.info(f"  - Total models: {metadata.get('total_models', 0)}")
    logger.info(f"  - Regions covered: {metadata.get('regions_covered', 0)}")
    logger.info(f"  - Models with pricing: {metadata.get('models_with_pricing', 0)}")
    logger.info(f"  - Models with quotas: {metadata.get('models_with_quotas', 0)}")

    return str(filepath)


def main():
    """Main execution function"""
    print("🚀 Amazon Bedrock Model Collector")
    print("=" * 50)
    print("Comprehensive model database with enhanced features")
    print(f"AWS Profile: {Config.AWS_PROFILE_NAME}")
    print(f"Threading: {Config.MAX_WORKER_THREADS} workers")
    print("=" * 50)

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Ensure output directory exists
        output_dir = ensure_output_directory()
        logger.info(f"Output directory: {output_dir}")

        # Collect comprehensive Bedrock data
        collection_data = collect_bedrock_data()

        # Process and save data
        output_file = process_and_save_data(collection_data, output_dir)

        print("\n🎉 Collection Complete!")
        print(f"📄 Output file: {output_file}")

        # Display final summary
        regions_count = len(collection_data['regions'])
        models_count = len(collection_data['enhanced_models'])

        print(f"📊 Regions covered: {regions_count}")
        print(f"📊 Total models: {models_count}")
        print(f"📊 Multi-threaded collection with {Config.MAX_WORKER_THREADS} workers")

    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        print(f"❌ Collection failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()