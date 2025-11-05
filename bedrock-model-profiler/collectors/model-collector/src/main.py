#!/usr/bin/env python3
"""
Amazon Bedrock Model Collector
Main script that orchestrates comprehensive model data collection from multiple sources
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

from config import Config
from utils.model_collector import BedrockModelCollector
from utils.regional_availability import RegionalAvailabilityDiscovery
from utils.features_enhancer import ModelFeaturesEnhancer
from utils.token_specs_enhancer import TokenSpecsEnhancer  # Token Specifications Discovery
from utils.pricing_integrator import PricingIntegrator
from utils.quotas_collector import ServiceQuotasCollector
from utils.data_processor import ModelDataProcessor
from utils.region_analysis import get_optimal_regions


class ConsoleFilter(logging.Filter):
    """Filter to show only essential messages on console"""
    def filter(self, record):
        # Show only key phase messages, warnings, errors, and final summaries
        message = record.getMessage()

        # Always show warnings and errors
        if record.levelno >= logging.WARNING:
            return True

        # Show key phase start/completion messages
        essential_patterns = [
            "=== Phase",
            "Collection Complete!",
            "COLLECTION SUMMARY:",
            "✅ Collected",
            "✅ Enhanced",
            "✅ Created pricing",
            "✅ Discovered regional",
            "✅ Added token specifications",
            "✅ Found",
            "✅ Loaded",
            "✅ Clean structure created",
            "✅ Timestamped backup saved",
            "✅ Final data saved",
            "Output file:",
            "Models with",
            "Providers:",
            "Total models:",
            "Regions covered:",
            "🚀 Amazon Bedrock Model Collector",
            "🎯 Using optimal regions",
            "🎯 Discovering token specifications",
            "📊 Total unique models:",
            "📊 Data sources:",
            "📊 TOTAL ASSIGNMENT:",
            "🚀 Using Optimized",
            "🚀 Starting parallel",
            "🚀 Collecting ALL Bedrock"
        ]

        # Check if message contains any essential pattern
        return any(pattern in message for pattern in essential_patterns)

def setup_logging() -> None:
    """Setup logging configuration with separate console and file handlers"""
    # Ensure logs directory exists
    logs_dir = Path(__file__).parent.parent / Config.LOGS_DIR
    logs_dir.mkdir(exist_ok=True)

    log_file_path = logs_dir / Config.LOG_FILENAME

    # Remove any existing handlers
    logging.getLogger().handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')

    # File handler - captures everything at DEBUG level
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Console handler - shows only essential messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(ConsoleFilter())

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Reduce verbosity of specific modules for console
    logging.getLogger('utils.token_specs_enhancer').setLevel(logging.DEBUG)
    logging.getLogger('utils.data_processor').setLevel(logging.DEBUG)
    logging.getLogger('utils.quotas_collector').setLevel(logging.DEBUG)


def ensure_output_directory() -> Path:
    """Ensure output directory exists - use root data/ directory"""
    output_dir = Path(Config.get_root_data_dir())
    output_dir.mkdir(exist_ok=True)
    return output_dir


def collect_bedrock_data(profile_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Collect comprehensive Bedrock model data from all sources

    Args:
        profile_override: AWS profile to use (overrides config.py)

    Returns:
        Dictionary with collected data from each phase
    """
    logger = logging.getLogger(__name__)

    # Use profile override or fall back to config
    aws_profile = profile_override or Config.AWS_PROFILE_NAME

    collection_data = {
        'regions': [],
        'raw_models': {},
        'regional_availability': {},
        'pricing_references': {},
        'enhanced_models': {},
        'quotas_data': {},
        'statistics': {}
    }

    # PHASE 1: Models Extraction from AWS (~3 seconds)
    logger.info("=== Phase 1: Models Extraction from AWS ===")

    # Determine optimal regions for 100% model coverage
    try:
        optimal_regions = get_optimal_regions()
        logger.info(f"🎯 Using optimal regions for 100% coverage: {optimal_regions}")
    except Exception as e:
        logger.warning(f"⚠️  Could not determine optimal regions ({e}), using default")
        optimal_regions = ['us-east-1', 'us-west-2']

    try:
        model_collector = BedrockModelCollector(
            profile_name=aws_profile,
            regions=optimal_regions,  # Dynamic optimal region selection
            max_workers=len(optimal_regions),  # Scale workers with region count
            use_direct_api=False  # Use optimized multi-region collection
        )
        raw_models = model_collector.collect_models_all_regions()
        collection_data['raw_models'] = raw_models
        logger.info(f"✅ Collected {len(raw_models)} unique models from optimal {len(optimal_regions)}-region approach")
    except Exception as e:
        logger.warning(f"❌ Model collection failed: {str(e)}")
        raise

    # PHASE 2: Pricing Integration (Match models to pricing data) (~1 second)
    logger.info("=== Phase 2: Pricing Integration ===")
    try:
        pricing_integrator = PricingIntegrator(
            pricing_collector_path=Config.get_pricing_collector_path()
        )
        pricing_references = pricing_integrator.integrate_pricing_data(raw_models)
        collection_data['pricing_references'] = pricing_references
        logger.info(f"✅ Created pricing references for models")
    except Exception as e:
        logger.error(f"❌ Pricing integration failed: {str(e)}")
        # Continue without pricing references
        logger.warning("Continuing without pricing references...")

    # PHASE 3: Regional Availability Discovery (Use pricing matches to determine regions) (~5 seconds)
    logger.info("=== Phase 3: Regional Availability Discovery ===")
    try:
        regional_discovery = RegionalAvailabilityDiscovery()
        # Pass both raw models AND pricing references to determine regional availability
        regional_availability_data = regional_discovery.discover_regional_availability(
            raw_models, pricing_references=collection_data.get('pricing_references', {})
        )

        collection_data['regional_availability'] = regional_availability_data
        collection_data['regions'] = regional_availability_data['bedrock_regions']

        # Update models with enhanced regional data
        enhanced_regional_models = regional_availability_data['enhanced_models']
        collection_data['raw_models'] = enhanced_regional_models

        bedrock_regions = regional_availability_data['bedrock_regions']
        logger.info(f"✅ Discovered regional availability - Bedrock in {len(bedrock_regions)} regions")
    except Exception as e:
        logger.warning(f"❌ Regional availability discovery failed: {str(e)}")
        raise

    # PHASE 3.5: Merge pricing references into models before Phase 4
    logger.info("🔗 Merging pricing references into models for Phase 4...")
    models_with_pricing = {}
    for model_id, model_data in enhanced_regional_models.items():
        # Copy the model data
        merged_model = model_data.copy()

        # Add pricing reference if available
        if model_id in collection_data.get('pricing_references', {}):
            pricing_ref = collection_data['pricing_references'][model_id]
            merged_model['model_pricing'] = pricing_ref['model_pricing']
        else:
            # Ensure model_pricing exists even if no pricing data
            merged_model['model_pricing'] = {'is_pricing_available': False}

        models_with_pricing[model_id] = merged_model

    logger.info(f"✅ Merged pricing data into {len(models_with_pricing)} models")

    # NEW PHASE 4: Enhanced Features (Optimized) (~30 seconds)
    logger.info("=== Phase 4: Enhanced Features (Optimized) ===")
    try:
        features_enhancer = ModelFeaturesEnhancer(
            profile_name=aws_profile,
            regions=bedrock_regions
        )
        enhanced_models = features_enhancer.enhance_models(models_with_pricing)
        collection_data['enhanced_models'] = enhanced_models
        logger.info(f"✅ Enhanced {len(enhanced_models)} models with optimized features")
    except Exception as e:
        logger.warning(f"❌ Model enhancement failed: {str(e)}")
        # Use models without enhancement
        collection_data['enhanced_models'] = enhanced_regional_models
        logger.warning("Using models without enhancement...")

    # NEW PHASE 4.5: Token Specifications Discovery
    logger.info("=== Phase 4.5: Token Specifications Discovery ===")
    try:
        token_specs_enhancer = TokenSpecsEnhancer()
        models_with_token_specs = token_specs_enhancer.enhance_models(collection_data['enhanced_models'])
        collection_data['enhanced_models'] = models_with_token_specs
        logger.info(f"✅ Added token specifications to {len(models_with_token_specs)} models")
    except Exception as e:
        logger.warning(f"❌ Token specifications enhancement failed: {str(e)}")
        logger.warning("Continuing without token specifications enhancement...")

    # PHASE 5: Service Quotas Collection (UNCHANGED)
    logger.info("=== Phase 5: Collecting Service Quotas ===")
    try:
        quotas_collector = ServiceQuotasCollector(
            profile_name=aws_profile,
            regions=bedrock_regions
        )
        quotas_data = quotas_collector.collect_service_quotas()
        collection_data['quotas_data'] = quotas_data
        collection_data['quotas_collector'] = quotas_collector  # Pass the collector instance
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
    # Reuse the same quotas collector instance to avoid threading issues
    quotas_collector = collection_data.get('quotas_collector')
    processor = ModelDataProcessor(quotas_collector=quotas_collector)
    final_structure = processor.create_comprehensive_structure(
        raw_models=collection_data['raw_models'],
        enhanced_models=collection_data['enhanced_models'],
        pricing_data=collection_data.get('pricing_references', {}),  # Use pricing references
        quotas_data=collection_data['quotas_data'],
        regions=collection_data['regions'],
        regional_availability=collection_data.get('regional_availability', {})  # Add regional availability data
    )

    # Save timestamped backup file
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{Config.OUTPUT_FILE_PREFIX}-{timestamp}.json"
    backup_filepath = output_dir / backup_filename

    with open(backup_filepath, 'w', encoding='utf-8') as f:
        json.dump(final_structure, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Timestamped backup saved to: {backup_filepath}")

    # Save final file that the app expects
    final_filepath = output_dir / Config.FINAL_OUTPUT_FILENAME

    with open(final_filepath, 'w', encoding='utf-8') as f:
        json.dump(final_structure, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Final data saved to: {final_filepath}")


    return str(final_filepath)


def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Amazon Bedrock Model Collector')
    parser.add_argument('--profile', type=str, help='AWS profile to use (overrides config.py)')
    args = parser.parse_args()

    # Determine AWS profile to use
    aws_profile = args.profile or Config.AWS_PROFILE_NAME

    print("🚀 Amazon Bedrock Model Collector")
    print("=" * 50)
    print("Comprehensive model database with enhanced features")
    print(f"AWS Profile: {aws_profile}")
    print(f"Parallel Collection: Enabled (Phase 1: 2 workers, Phase 4: 10 workers, Phase 4.5: 10 workers, Phase 5: 10 workers)")
    print("=" * 50)

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Show log file location
    logs_dir = Path(__file__).parent.parent / Config.LOGS_DIR
    log_file_path = logs_dir / Config.LOG_FILENAME
    print(f"📝 Detailed logs: {log_file_path}")
    print("=" * 50)

    try:
        # Ensure output directory exists
        output_dir = ensure_output_directory()
        logger.info(f"Output directory: {output_dir}")

        # Collect comprehensive Bedrock data
        collection_data = collect_bedrock_data(profile_override=aws_profile)

        # Process and save data
        output_file = process_and_save_data(collection_data, output_dir)

        print("\n🎉 Collection Complete!")

        # Display comprehensive final summary
        try:
            # Read the final structure to get accurate metadata
            import json
            with open(output_file, 'r', encoding='utf-8') as f:
                final_structure = json.load(f)

            metadata = final_structure.get('metadata', {})
            print("\n📊 COLLECTION SUMMARY:")
            print("=" * 50)
            print(f"✅ Providers: {metadata.get('providers_count', 0)}")
            print(f"✅ Total models: {metadata.get('total_models', 0)}")
            print(f"✅ Regions covered: {metadata.get('regions_covered', 0)}")
            print(f"✅ Models with pricing: {metadata.get('models_with_pricing', 0)}")
            print(f"✅ Models with quotas: {metadata.get('models_with_quotas', 0)}")

            # Calculate success rate
            total_models = metadata.get('total_models', 0)
            models_with_quotas = metadata.get('models_with_quotas', 0)
            if total_models > 0:
                success_rate = (models_with_quotas / total_models) * 100
                print(f"✅ Quota assignment success rate: {success_rate:.1f}%")

            print(f"✅ Parallel collection optimized per phase")
            print("=" * 50)

        except Exception as e:
            logger.warning(f"Could not read final summary from output file: {e}")
            # Fallback to basic summary
            regions_count = len(collection_data['regions'])
            models_count = len(collection_data['enhanced_models'])
            print(f"📊 Regions covered: {regions_count}")
            print(f"📊 Total models: {models_count}")
            print(f"📊 Parallel collection optimized per phase")

    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        print(f"❌ Collection failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()