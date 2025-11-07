#!/usr/bin/env python3
"""
Analyze pricing data to determine minimal regions needed to capture all models
"""

import json
import glob
import os
from collections import defaultdict
from typing import Dict, Set, List

def get_latest_pricing_file():
    """Get the most recent pricing JSON file - works from both root and model-collector contexts"""
    # Try different possible locations depending on where the script is run from
    possible_paths = [
        "collectors/pricing-collector/out/bedrock-pricing-*.json",  # From root
        "../pricing-collector/out/bedrock-pricing-*.json",         # From model-collector
        "../../pricing-collector/out/bedrock-pricing-*.json"       # From deeper in model-collector
    ]

    for path_pattern in possible_paths:
        pricing_files = glob.glob(path_pattern)
        if pricing_files:
            # Return the newest file
            return max(pricing_files, key=os.path.getmtime)

    raise FileNotFoundError("No pricing files found. Please run the pricing collector first.")

def analyze_pricing_regions():
    """Analyze pricing data to find minimal regions needed"""

    # Load pricing data
    pricing_file = get_latest_pricing_file()

    with open(pricing_file, 'r', encoding='utf-8') as f:
        pricing_data = json.load(f)

    # Track which models are available in which regions
    models_by_region = defaultdict(set)
    regions_by_model = defaultdict(set)
    all_regions = set()
    all_models = set()

    # Process all providers and models
    for provider_name, provider_data in pricing_data.get('providers', {}).items():
        for model_id, model_data in provider_data.items():
            if isinstance(model_data, dict) and 'regions' in model_data:
                all_models.add(model_id)
                for region in model_data['regions'].keys():
                    all_regions.add(region)
                    models_by_region[region].add(model_id)
                    regions_by_model[model_id].add(region)

    print(f"📊 PRICING DATA ANALYSIS")
    print(f"=" * 50)
    print(f"Total unique models: {len(all_models)}")
    print(f"Total unique regions: {len(all_regions)}")
    print()

    # Find regions with unique models (models only available in that region)
    unique_model_regions = []
    for region in sorted(all_regions):
        region_models = models_by_region[region]
        unique_models = set()

        for model in region_models:
            # Check if model is only available in this region
            if len(regions_by_model[model]) == 1:
                unique_models.add(model)

        if unique_models:
            unique_model_regions.append({
                'region': region,
                'total_models': len(region_models),
                'unique_models': len(unique_models),
                'unique_model_list': sorted(unique_models)
            })

    print(f"🔍 REGIONS WITH UNIQUE MODELS:")
    print(f"=" * 50)
    if unique_model_regions:
        for info in sorted(unique_model_regions, key=lambda x: x['unique_models'], reverse=True):
            print(f"Region: {info['region']}")
            print(f"  Total models: {info['total_models']}")
            print(f"  Unique models: {info['unique_models']}")
            if info['unique_models'] <= 5:  # Show details for regions with few unique models
                for model in info['unique_model_list']:
                    print(f"    - {model}")
            print()
    else:
        print("No regions have unique models")

    print()

    # Find regions with most models (good candidates for collection)
    top_regions = []
    for region in all_regions:
        model_count = len(models_by_region[region])
        top_regions.append((region, model_count))

    # Sort by model count
    top_regions.sort(key=lambda x: x[1], reverse=True)

    print(f"🏆 TOP REGIONS BY MODEL COUNT:")
    print(f"=" * 50)
    for region, count in top_regions[:15]:  # Top 15
        print(f"{region}: {count} models")

    print()

    # Check current regions (us-east-1, us-west-2)
    current_regions = ['us-east-1', 'us-west-2']
    current_models = set()

    print(f"🔍 ANALYSIS OF CURRENT REGIONS:")
    print(f"=" * 50)
    for region in current_regions:
        if region in models_by_region:
            region_models = models_by_region[region]
            current_models.update(region_models)
            print(f"{region}: {len(region_models)} models")
        else:
            print(f"{region}: NOT FOUND in pricing data")

    print(f"Combined unique models from current regions: {len(current_models)}")

    # Find models missing from current regions
    missing_models = all_models - current_models
    if missing_models:
        print(f"Missing models: {len(missing_models)}")
        print("Missing models list:")
        for model in sorted(missing_models):
            available_regions = sorted(regions_by_model[model])
            print(f"  {model} -> available in: {', '.join(available_regions[:3])}{'...' if len(available_regions) > 3 else ''}")
    else:
        print("✅ Current regions capture ALL models!")

    print()

    # Recommend minimal region set
    print(f"💡 MINIMAL REGION RECOMMENDATIONS:")
    print(f"=" * 50)

    if not missing_models:
        print("✅ Current regions (us-east-1, us-west-2) are sufficient!")
    else:
        # Find which regions would capture the missing models
        region_coverage = {}
        for region in all_regions:
            if region not in current_regions:
                coverage = len(missing_models.intersection(models_by_region[region]))
                if coverage > 0:
                    region_coverage[region] = coverage

        if region_coverage:
            print("Regions that would add missing models:")
            for region, coverage in sorted(region_coverage.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {region}: +{coverage} additional models")

            best_additional = max(region_coverage.items(), key=lambda x: x[1])
            print(f"\nRecommendation: Add '{best_additional[0]}' to capture {best_additional[1]} additional models")

def get_optimal_regions() -> List[str]:
    """Get the minimal region set needed to capture all models"""

    try:
        # Load pricing data
        pricing_file = get_latest_pricing_file()
        with open(pricing_file, 'r', encoding='utf-8') as f:
            pricing_data = json.load(f)
    except FileNotFoundError:
        print("⚠️  Pricing data not found, using default regions")
        return ['us-east-1', 'us-west-2']

    # Track which models are available in which regions
    models_by_region = defaultdict(set)
    regions_by_model = defaultdict(set)
    all_models = set()

    # Process all providers and models
    for provider_name, provider_data in pricing_data.get('providers', {}).items():
        for model_id, model_data in provider_data.items():
            if isinstance(model_data, dict) and 'regions' in model_data:
                all_models.add(model_id)
                for region in model_data['regions'].keys():
                    models_by_region[region].add(model_id)
                    regions_by_model[model_id].add(region)

    if not all_models:
        print("⚠️  No models found in pricing data, using default regions")
        return ['us-east-1', 'us-west-2']

    # Start with the region that has the most models
    optimal_regions = []
    covered_models = set()

    # Sort regions by model count (descending)
    region_counts = [(region, len(models)) for region, models in models_by_region.items()]
    region_counts.sort(key=lambda x: x[1], reverse=True)

    # Greedy algorithm to find minimal region set
    for region, count in region_counts:
        region_models = models_by_region[region]
        new_models = region_models - covered_models

        if new_models:  # This region adds new models
            optimal_regions.append(region)
            covered_models.update(region_models)

            # Stop when we have all models
            if len(covered_models) >= len(all_models):
                break

    print(f"🎯 OPTIMAL REGION SELECTION:")
    print(f"   Total models available: {len(all_models)}")
    print(f"   Regions needed: {len(optimal_regions)}")
    print(f"   Selected regions: {optimal_regions}")
    print(f"   Coverage: {len(covered_models)}/{len(all_models)} models ({len(covered_models)/len(all_models)*100:.1f}%)")

    return optimal_regions

if __name__ == "__main__":
    analyze_pricing_regions()