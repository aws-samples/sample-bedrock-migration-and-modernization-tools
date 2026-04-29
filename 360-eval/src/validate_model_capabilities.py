#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Model Capability Validation CLI Tool

Validates Amazon Bedrock model availability and service tier support by making test API calls.
Results are cached for use by the dashboard and benchmark tools.

Usage:
    python src/validate_model_capabilities.py                    # Validate all models
    python src/validate_model_capabilities.py --force            # Force re-validation
    python src/validate_model_capabilities.py --model MODEL --region REGION
    python src/validate_model_capabilities.py --model MODEL --region REGION --tier TIER
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model_capability_validator import (
    validate_all_models,
    test_model_availability,
    test_service_tier,
    load_capability_cache,
    is_cache_valid,
    get_models_hash
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simple format for CLI output
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print CLI banner."""
    print("=" * 60)
    print("🔍 360-Eval Model Capability Validator")
    print("=" * 60)
    print()


def print_summary(cache: dict):
    """Print validation summary."""
    print("\n" + "=" * 60)
    print("📊 Validation Summary")
    print("=" * 60)

    capabilities = cache.get("capabilities", {})
    total_models = len(capabilities)
    total_regions = sum(len(regions) for regions in capabilities.values())

    available_count = 0
    tier_counts = {"default": 0, "priority": 0, "flex": 0}

    for model_id, regions in capabilities.items():
        for region, caps in regions.items():
            if caps.get("available"):
                available_count += 1
                for tier in caps.get("service_tiers", []):
                    tier_counts[tier] += 1

    print(f"📦 Total models: {total_models}")
    print(f"🌍 Total model+region combinations: {total_regions}")
    print(f"✅ Available combinations: {available_count}")
    print(f"❌ Unavailable combinations: {total_regions - available_count}")
    print()
    print("Service Tier Support:")
    print(f"  • Default: {tier_counts['default']} combinations")
    print(f"  • Priority: {tier_counts['priority']} combinations")
    print(f"  • Flex: {tier_counts['flex']} combinations")
    print()
    print(f"💾 Cache updated: {cache.get('last_updated', 'Unknown')}")
    print(f"📄 Models hash: {cache.get('models_hash', 'Unknown')[:16]}...")
    print("=" * 60)


def validate_single_model(model_id: str, region: str, tier: str = None):
    """Validate a single model or model+tier combination."""
    print(f"🔍 Testing {model_id} @ {region}")

    if tier:
        # Test specific tier
        print(f"   Testing {tier} tier...")
        success, error = test_service_tier(model_id, region, tier)

        if success:
            print(f"   ✅ {tier} tier: Supported")
        else:
            print(f"   ❌ {tier} tier: Not supported")
            if error:
                print(f"      Error: {error}")
    else:
        # Test model availability and all tiers
        result = test_model_availability(model_id, region)

        if result["available"]:
            tiers = ", ".join(result["service_tiers"])
            print(f"   ✅ Available (Tiers: {tiers})")
        else:
            print(f"   ❌ Not available")
            if result["error"]:
                print(f"      Error: {result['error']}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Amazon Bedrock model availability and service tier support",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-validation of all models (ignore existing cache)"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Validate specific model (e.g., 'bedrock/us.amazon.nova-2-lite-v1:0')"
    )

    parser.add_argument(
        "--region",
        type=str,
        help="AWS region for model validation (required with --model)"
    )

    parser.add_argument(
        "--tier",
        type=str,
        choices=["default", "priority", "flex"],
        help="Test specific service tier (requires --model and --region)"
    )

    args = parser.parse_args()

    # Validate argument combinations
    if args.tier and (not args.model or not args.region):
        parser.error("--tier requires both --model and --region")

    if args.model and not args.region:
        parser.error("--model requires --region")

    print_banner()

    start_time = time.time()

    try:
        if args.model:
            # Validate single model
            validate_single_model(args.model, args.region, args.tier)
        else:
            # Validate all models
            if not args.force:
                # Check if cache is valid
                if is_cache_valid():
                    cache = load_capability_cache()
                    print("ℹ️  Cache is up-to-date. Use --force to re-validate.")
                    print()
                    print_summary(cache)
                    return

            print("🚀 Starting validation of all models...")
            print("⏱️  This may take 2-3 minutes depending on number of models.")
            print()

            cache = validate_all_models(force=args.force)
            print_summary(cache)

        elapsed = time.time() - start_time
        print(f"\n⏱️  Completed in {elapsed:.1f} seconds")

    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        logger.exception("Validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
