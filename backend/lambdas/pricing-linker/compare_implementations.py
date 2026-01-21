#!/usr/bin/env python3
"""
Comparison Test Script for Pricing Linker V1 vs V2

Compares match rates and differences between the original implementation
(handler_v1.py) and the new implementation with PORT features (handler.py).

This is a standalone script that extracts the core matching logic without
requiring AWS dependencies.

Usage:
    python3 compare_implementations.py [--data-dir PATH]
"""

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


# =============================================================================
# V1 IMPLEMENTATION (Original)
# =============================================================================

def v1_similarity_score(a: str, b: str) -> float:
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def v1_has_on_demand_pricing(pricing_data: dict) -> bool:
    """Check if pricing data has On-Demand pricing in at least one region."""
    regions = pricing_data.get('regions', {})
    for region_data in regions.values():
        pricing_groups = region_data.get('pricing_groups', {})
        on_demand = pricing_groups.get('On-Demand', [])
        if on_demand:
            return True
    return False


def v1_normalize_model_id(model_id: str) -> str:
    """Normalize model ID for matching by removing common suffixes and normalizing format."""
    normalized = model_id.lower()
    suffixes_to_remove = ['-it', '-instruct', '-chat', '-v1', '-v2', '-v3']
    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
    return normalized.replace('-', '').replace('_', '').replace('.', '')


def v1_find_best_pricing_match(model_id: str, model_name: str, pricing_models: dict) -> tuple:
    """V1 matching algorithm (original)."""
    best_on_demand_match = None
    best_on_demand_score = 0.0
    best_other_match = None
    best_other_score = 0.0

    model_id_normalized = v1_normalize_model_id(model_id)
    model_name_normalized = model_name.lower().replace('-', '').replace('_', '').replace('.', '').replace(' ', '')

    for pricing_key, pricing_data in pricing_models.items():
        pricing_model_name = pricing_data.get('model_name', '')
        pricing_key_normalized = v1_normalize_model_id(pricing_key)
        pricing_name_normalized = pricing_model_name.lower().replace('-', '').replace('_', '').replace('.', '').replace(' ', '')

        score = 0.0

        if model_id_normalized == pricing_key_normalized:
            score = 1.0
        elif model_name_normalized == pricing_name_normalized:
            score = 1.0
        elif model_id_normalized.startswith(pricing_key_normalized) or pricing_key_normalized.startswith(model_id_normalized):
            score = 0.95
        elif model_name_normalized.startswith(pricing_name_normalized) or pricing_name_normalized.startswith(model_name_normalized):
            score = 0.95
        else:
            score = max(
                v1_similarity_score(model_id_normalized, pricing_key_normalized),
                v1_similarity_score(model_name_normalized, pricing_name_normalized),
                v1_similarity_score(model_id_normalized, pricing_name_normalized)
            )

        if v1_has_on_demand_pricing(pricing_data):
            if score > best_on_demand_score:
                best_on_demand_score = score
                best_on_demand_match = pricing_key
        else:
            if score > best_other_score:
                best_other_score = score
                best_other_match = pricing_key

    if best_on_demand_match and best_on_demand_score >= 0.7:
        return best_on_demand_match, best_on_demand_score
    if best_other_match and best_other_score > best_on_demand_score:
        return best_other_match, best_other_score
    if best_on_demand_match:
        return best_on_demand_match, best_on_demand_score

    return best_other_match, best_other_score


def v1_link_pricing_to_models(models_data: dict, pricing_data: dict) -> dict:
    """V1 linking algorithm (original)."""
    models_with_pricing = 0
    models_without_pricing = 0

    all_pricing_models = {}
    for provider_name, data in pricing_data.get('providers', {}).items():
        if isinstance(data, dict):
            if 'regions' in data:
                all_pricing_models[provider_name] = {'provider': provider_name, 'data': data}
            elif 'models' in data:
                for model_key, model_pricing in data.get('models', {}).items():
                    all_pricing_models[model_key] = {'provider': provider_name, 'data': model_pricing}
            else:
                for model_key, model_pricing in data.items():
                    if isinstance(model_pricing, dict) and 'regions' in model_pricing:
                        all_pricing_models[model_key] = {'provider': provider_name, 'data': model_pricing}

    pricing_data_only = {k: v['data'] for k, v in all_pricing_models.items()}

    for provider, provider_data in models_data.get('providers', {}).items():
        for model_id, model in provider_data.get('models', {}).items():
            model_name = model.get('model_name', model_id)
            matched_key, confidence = v1_find_best_pricing_match(model_id, model_name, pricing_data_only)

            if matched_key and confidence >= 0.7:
                pricing_entry = all_pricing_models[matched_key]
                model['model_pricing'] = {
                    'is_pricing_available': True,
                    'pricing_reference_id': matched_key,
                    'confidence': round(confidence, 3),
                }
                model['has_pricing'] = True
                models_with_pricing += 1
            else:
                model['model_pricing'] = {
                    'is_pricing_available': False,
                    'pricing_reference_id': None,
                    'confidence': round(confidence, 3) if matched_key else 0,
                }
                model['has_pricing'] = False
                models_without_pricing += 1

    return {
        'models_with_pricing': models_with_pricing,
        'models_without_pricing': models_without_pricing,
        'providers': models_data.get('providers', {})
    }


# =============================================================================
# V2 IMPLEMENTATION (With PORT Features)
# =============================================================================

V2_PROVIDER_ALIASES = {
    'amazon': {'amazon', 'aws'},
    'anthropic': {'anthropic'},
    'meta': {'meta', 'facebook'},
    'mistral ai': {'mistral', 'mistralai', 'mistral ai'},
    'stability ai': {'stability', 'stabilityai', 'stability ai'},
    'cohere': {'cohere'},
    'ai21 labs': {'ai21', 'ai21labs', 'ai21 labs'},
    'luma ai': {'luma', 'lumaai', 'luma ai'},
    'twelvelabs': {'twelvelabs', 'twelve labs', 'twelverlabs'},
    'minimax': {'minimax', 'minimax ai', 'minimax-ai'},
    'moonshot ai': {'moonshot', 'moonshot ai', 'kimi', 'kimi ai'},
    'deepseek': {'deepseek'},
    'qwen': {'qwen', 'qwen2', 'alibaba'},
    'google': {'google'},
    'nvidia': {'nvidia'},
    'openai': {'openai'},
    'writer': {'writer', 'writerai'},
}


def v2_similarity_score(a: str, b: str) -> float:
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def v2_has_on_demand_pricing(pricing_data: dict) -> bool:
    """Check if pricing data has On-Demand pricing in at least one region."""
    regions = pricing_data.get('regions', {})
    for region_data in regions.values():
        pricing_groups = region_data.get('pricing_groups', {})
        on_demand = pricing_groups.get('On-Demand', [])
        if on_demand:
            return True
    return False


def v2_providers_match(model_provider: str, pricing_provider: str) -> bool:
    """Check if model provider matches pricing provider with alias support."""
    if not model_provider or not pricing_provider:
        return False

    model_provider_lower = model_provider.lower().strip()
    pricing_provider_lower = pricing_provider.lower().strip()

    if model_provider_lower == pricing_provider_lower:
        return True

    for canonical, aliases in V2_PROVIDER_ALIASES.items():
        model_in_aliases = (
            model_provider_lower in aliases or
            any(alias in model_provider_lower for alias in aliases)
        )
        pricing_in_aliases = (
            pricing_provider_lower in aliases or
            any(alias in pricing_provider_lower for alias in aliases)
        )
        if model_in_aliases and pricing_in_aliases:
            return True

    return False


def v2_has_semantic_conflict(model_name: str, pricing_name: str, model_id: str = '', pricing_key: str = '') -> bool:
    """Detect semantic conflicts that should block a match."""
    model_lower = (model_name + ' ' + model_id).lower()
    pricing_lower = (pricing_name + ' ' + pricing_key).lower()

    # Claude variant conflicts
    claude_variants = ['haiku', 'sonnet', 'opus']
    model_claude = [v for v in claude_variants if v in model_lower]
    pricing_claude = [v for v in claude_variants if v in pricing_lower]
    if model_claude and pricing_claude and model_claude != pricing_claude:
        return True

    # Nova variant conflicts
    nova_variants = ['micro', 'lite', 'pro', 'premier']
    model_nova = [v for v in nova_variants if f'nova-{v}' in model_lower or f'nova {v}' in model_lower]
    pricing_nova = [v for v in nova_variants if f'nova-{v}' in pricing_lower or f'nova {v}' in pricing_lower]
    if model_nova and pricing_nova and model_nova != pricing_nova:
        return True

    # Llama size conflicts
    llama_sizes = ['8b', '70b', '405b', '3b', '1b', '11b', '90b']
    model_llama_size = [s for s in llama_sizes if s in model_lower]
    pricing_llama_size = [s for s in llama_sizes if s in pricing_lower]
    if model_llama_size and pricing_llama_size and model_llama_size != pricing_llama_size:
        return True

    # Embed vs generator conflict
    model_is_embed = 'embed' in model_lower
    pricing_is_embed = 'embed' in pricing_lower
    if model_is_embed != pricing_is_embed:
        return True

    # General size mismatch
    model_sizes = re.findall(r'(\d+)b', model_lower)
    pricing_sizes = re.findall(r'(\d+)b', pricing_lower)
    if model_sizes and pricing_sizes:
        model_size = int(model_sizes[0])
        pricing_size = int(pricing_sizes[0])
        larger = max(model_size, pricing_size)
        smaller = min(model_size, pricing_size)
        if larger > 0 and (smaller / larger) < 0.7:
            return True

    return False


def v2_normalize_model_id(model_id: str, provider: str = '') -> str:
    """Normalize model ID with provider-specific rules."""
    normalized = model_id.lower()
    provider_lower = provider.lower() if provider else ''

    # Provider-specific normalizations
    if 'qwen' in provider_lower or 'qwen' in normalized:
        normalized = re.sub(r'-instruct.*$', '', normalized)
        normalized = re.sub(r'_instruct.*$', '', normalized)

    if 'deepseek' in provider_lower or 'deepseek' in normalized:
        normalized = re.sub(r'v(\d)', r'\1', normalized)
        normalized = re.sub(r'-v(\d)', r'-\1', normalized)

    if 'cohere' in provider_lower or 'cohere' in normalized:
        normalized = normalized.replace('-model', '').replace('model-', '')

    if 'stability' in provider_lower or 'stable' in normalized:
        normalized = re.sub(r'sd(\d)', r'stable-diffusion-\1', normalized)

    # General suffixes
    suffixes_to_remove = ['-it', '-instruct', '-chat', '-v1', '-v2', '-v3']
    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]

    return normalized.replace('-', '').replace('_', '').replace('.', '')


def v2_find_best_pricing_match(model_id: str, model_name: str, model_provider: str, pricing_models: dict) -> tuple:
    """V2 matching algorithm (with PORT features)."""
    best_on_demand_match = None
    best_on_demand_score = 0.0
    best_other_match = None
    best_other_score = 0.0

    model_id_normalized = v2_normalize_model_id(model_id, model_provider)
    model_name_normalized = model_name.lower().replace('-', '').replace('_', '').replace('.', '').replace(' ', '')

    for pricing_key, pricing_entry in pricing_models.items():
        pricing_data = pricing_entry['data']
        pricing_provider = pricing_entry['provider']
        pricing_model_name = pricing_data.get('model_name', '')

        # PORT Feature 1: Provider-scoped matching
        if not v2_providers_match(model_provider, pricing_provider):
            continue

        # PORT Feature 2: Conflict detection
        if v2_has_semantic_conflict(model_name, pricing_model_name, model_id, pricing_key):
            continue

        pricing_key_normalized = v2_normalize_model_id(pricing_key, pricing_provider)
        pricing_name_normalized = pricing_model_name.lower().replace('-', '').replace('_', '').replace('.', '').replace(' ', '')

        score = 0.0

        if model_id_normalized == pricing_key_normalized:
            score = 1.0
        elif model_name_normalized == pricing_name_normalized:
            score = 1.0
        elif model_id_normalized.startswith(pricing_key_normalized) or pricing_key_normalized.startswith(model_id_normalized):
            score = 0.95
        elif model_name_normalized.startswith(pricing_name_normalized) or pricing_name_normalized.startswith(model_name_normalized):
            score = 0.95
        else:
            score = max(
                v2_similarity_score(model_id_normalized, pricing_key_normalized),
                v2_similarity_score(model_name_normalized, pricing_name_normalized),
                v2_similarity_score(model_id_normalized, pricing_name_normalized)
            )

        if v2_has_on_demand_pricing(pricing_data):
            if score > best_on_demand_score:
                best_on_demand_score = score
                best_on_demand_match = pricing_key
        else:
            if score > best_other_score:
                best_other_score = score
                best_other_match = pricing_key

    if best_on_demand_match and best_on_demand_score >= 0.7:
        return best_on_demand_match, best_on_demand_score
    if best_other_match and best_other_score > best_on_demand_score:
        return best_other_match, best_other_score
    if best_on_demand_match:
        return best_on_demand_match, best_on_demand_score

    return best_other_match, best_other_score


def v2_link_pricing_to_models(models_data: dict, pricing_data: dict) -> dict:
    """V2 linking algorithm (with PORT features)."""
    models_with_pricing = 0
    models_without_pricing = 0

    all_pricing_models = {}
    for provider_name, data in pricing_data.get('providers', {}).items():
        if isinstance(data, dict):
            if 'regions' in data:
                all_pricing_models[provider_name] = {'provider': provider_name, 'data': data}
            elif 'models' in data:
                for model_key, model_pricing in data.get('models', {}).items():
                    all_pricing_models[model_key] = {'provider': provider_name, 'data': model_pricing}
            else:
                for model_key, model_pricing in data.items():
                    if isinstance(model_pricing, dict) and 'regions' in model_pricing:
                        all_pricing_models[model_key] = {'provider': provider_name, 'data': model_pricing}

    for provider, provider_data in models_data.get('providers', {}).items():
        for model_id, model in provider_data.get('models', {}).items():
            model_name = model.get('model_name', model_id)
            matched_key, confidence = v2_find_best_pricing_match(
                model_id, model_name, provider, all_pricing_models
            )

            if matched_key and confidence >= 0.7:
                model['model_pricing'] = {
                    'is_pricing_available': True,
                    'pricing_reference_id': matched_key,
                    'confidence': round(confidence, 3),
                }
                model['has_pricing'] = True
                models_with_pricing += 1
            else:
                model['model_pricing'] = {
                    'is_pricing_available': False,
                    'pricing_reference_id': None,
                    'confidence': round(confidence, 3) if matched_key else 0,
                }
                model['has_pricing'] = False
                models_without_pricing += 1

    return {
        'models_with_pricing': models_with_pricing,
        'models_without_pricing': models_without_pricing,
        'providers': models_data.get('providers', {})
    }


# =============================================================================
# COMPARISON LOGIC
# =============================================================================

def load_json_file(filepath: str) -> dict:
    """Load a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def run_comparison(models_data: dict, pricing_data: dict) -> dict:
    """
    Run both implementations and compare results.

    Returns comparison statistics and detailed differences.
    """
    # Run V1 (original)
    v1_result = v1_link_pricing_to_models(
        json.loads(json.dumps(models_data)),  # Deep copy
        pricing_data
    )

    # Run V2 (with PORT features)
    v2_result = v2_link_pricing_to_models(
        json.loads(json.dumps(models_data)),  # Deep copy
        pricing_data
    )

    # Extract results
    v1_providers = v1_result.get('providers', {})
    v2_providers = v2_result.get('providers', {})

    # Collect detailed comparison
    comparison = {
        'v1_stats': {
            'models_with_pricing': v1_result.get('models_with_pricing', 0),
            'models_without_pricing': v1_result.get('models_without_pricing', 0),
        },
        'v2_stats': {
            'models_with_pricing': v2_result.get('models_with_pricing', 0),
            'models_without_pricing': v2_result.get('models_without_pricing', 0),
        },
        'differences': [],
        'v1_only_matches': [],
        'v2_only_matches': [],
        'different_matches': [],
        'confidence_improvements': [],
        'confidence_regressions': [],
    }

    # Compare model by model
    for provider, provider_data in v1_providers.items():
        v2_provider_data = v2_providers.get(provider, {})

        for model_id, v1_model in provider_data.get('models', {}).items():
            v2_model = v2_provider_data.get('models', {}).get(model_id, {})

            v1_pricing = v1_model.get('model_pricing', {})
            v2_pricing = v2_model.get('model_pricing', {})

            v1_has_pricing = v1_pricing.get('is_pricing_available', False)
            v2_has_pricing = v2_pricing.get('is_pricing_available', False)

            v1_ref = v1_pricing.get('pricing_reference_id')
            v2_ref = v2_pricing.get('pricing_reference_id')

            v1_confidence = v1_pricing.get('confidence', 0)
            v2_confidence = v2_pricing.get('confidence', 0)

            model_name = v1_model.get('model_name', model_id)

            diff_entry = {
                'provider': provider,
                'model_id': model_id,
                'model_name': model_name,
                'v1_ref': v1_ref,
                'v2_ref': v2_ref,
                'v1_confidence': v1_confidence,
                'v2_confidence': v2_confidence,
            }

            # Categorize differences
            if v1_has_pricing and not v2_has_pricing:
                comparison['v1_only_matches'].append(diff_entry)
                comparison['differences'].append({
                    **diff_entry,
                    'type': 'v1_only',
                    'description': f"V1 matched to '{v1_ref}' but V2 found no match"
                })
            elif not v1_has_pricing and v2_has_pricing:
                comparison['v2_only_matches'].append(diff_entry)
                comparison['differences'].append({
                    **diff_entry,
                    'type': 'v2_only',
                    'description': f"V2 matched to '{v2_ref}' but V1 found no match"
                })
            elif v1_ref != v2_ref and v1_has_pricing and v2_has_pricing:
                comparison['different_matches'].append(diff_entry)
                comparison['differences'].append({
                    **diff_entry,
                    'type': 'different_match',
                    'description': f"V1 matched '{v1_ref}' vs V2 matched '{v2_ref}'"
                })
            elif v1_ref == v2_ref and v1_has_pricing:
                # Same match, compare confidence
                conf_diff = v2_confidence - v1_confidence
                if conf_diff > 0.01:
                    comparison['confidence_improvements'].append({
                        **diff_entry,
                        'confidence_change': round(conf_diff, 3)
                    })
                elif conf_diff < -0.01:
                    comparison['confidence_regressions'].append({
                        **diff_entry,
                        'confidence_change': round(conf_diff, 3)
                    })

    return comparison


def print_report(comparison: dict) -> None:
    """Print a formatted comparison report."""
    v1 = comparison['v1_stats']
    v2 = comparison['v2_stats']

    print("=" * 70)
    print("PRICING LINKER IMPLEMENTATION COMPARISON REPORT")
    print("=" * 70)
    print()

    # Summary statistics
    print("MATCH STATISTICS")
    print("-" * 40)
    print(f"{'Metric':<30} {'V1':<10} {'V2':<10} {'Diff':<10}")
    print("-" * 40)

    v1_total = v1['models_with_pricing'] + v1['models_without_pricing']
    v2_total = v2['models_with_pricing'] + v2['models_without_pricing']

    print(f"{'Models with pricing':<30} {v1['models_with_pricing']:<10} {v2['models_with_pricing']:<10} {v2['models_with_pricing'] - v1['models_with_pricing']:+}")
    print(f"{'Models without pricing':<30} {v1['models_without_pricing']:<10} {v2['models_without_pricing']:<10} {v2['models_without_pricing'] - v1['models_without_pricing']:+}")

    v1_rate = (v1['models_with_pricing'] / v1_total * 100) if v1_total > 0 else 0
    v2_rate = (v2['models_with_pricing'] / v2_total * 100) if v2_total > 0 else 0
    print(f"{'Match rate':<30} {v1_rate:.1f}%{'':<5} {v2_rate:.1f}%{'':<5} {v2_rate - v1_rate:+.1f}%")
    print()

    # Differences summary
    print("DIFFERENCES SUMMARY")
    print("-" * 40)
    print(f"V1-only matches (lost in V2):    {len(comparison['v1_only_matches'])}")
    print(f"V2-only matches (gained in V2):  {len(comparison['v2_only_matches'])}")
    print(f"Different match targets:         {len(comparison['different_matches'])}")
    print(f"Confidence improvements:         {len(comparison['confidence_improvements'])}")
    print(f"Confidence regressions:          {len(comparison['confidence_regressions'])}")
    print()

    # Detail: V1-only matches (potential regressions)
    if comparison['v1_only_matches']:
        print("V1-ONLY MATCHES (Potential Regressions)")
        print("-" * 70)
        print("These models matched in V1 but NOT in V2 - review if intentional:")
        print()
        for item in comparison['v1_only_matches'][:10]:  # Limit to 10
            print(f"  [{item['provider']}] {item['model_id']}")
            print(f"    Name: {item['model_name']}")
            print(f"    V1 matched: {item['v1_ref']} (conf: {item['v1_confidence']:.3f})")
            print()
        if len(comparison['v1_only_matches']) > 10:
            print(f"  ... and {len(comparison['v1_only_matches']) - 10} more")
        print()

    # Detail: V2-only matches (improvements)
    if comparison['v2_only_matches']:
        print("V2-ONLY MATCHES (Improvements)")
        print("-" * 70)
        print("These models matched in V2 but NOT in V1 - new matches found:")
        print()
        for item in comparison['v2_only_matches'][:10]:
            print(f"  [{item['provider']}] {item['model_id']}")
            print(f"    Name: {item['model_name']}")
            print(f"    V2 matched: {item['v2_ref']} (conf: {item['v2_confidence']:.3f})")
            print()
        if len(comparison['v2_only_matches']) > 10:
            print(f"  ... and {len(comparison['v2_only_matches']) - 10} more")
        print()

    # Detail: Different matches
    if comparison['different_matches']:
        print("DIFFERENT MATCH TARGETS")
        print("-" * 70)
        print("These models matched to different pricing entries:")
        print()
        for item in comparison['different_matches'][:10]:
            print(f"  [{item['provider']}] {item['model_id']}")
            print(f"    Name: {item['model_name']}")
            print(f"    V1: {item['v1_ref']} (conf: {item['v1_confidence']:.3f})")
            print(f"    V2: {item['v2_ref']} (conf: {item['v2_confidence']:.3f})")
            print()
        if len(comparison['different_matches']) > 10:
            print(f"  ... and {len(comparison['different_matches']) - 10} more")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    net_change = v2['models_with_pricing'] - v1['models_with_pricing']
    if net_change > 0:
        print(f"V2 finds {net_change} MORE matches than V1")
    elif net_change < 0:
        print(f"V2 finds {abs(net_change)} FEWER matches than V1")
    else:
        print("V2 finds the SAME number of matches as V1")

    if comparison['v1_only_matches']:
        print(f"WARNING: {len(comparison['v1_only_matches'])} models lost matches - review for false negatives")

    if comparison['different_matches']:
        print(f"NOTE: {len(comparison['different_matches'])} models have different match targets - review for correctness")

    print()


def main():
    parser = argparse.ArgumentParser(description='Compare pricing linker V1 vs V2')
    parser.add_argument(
        '--data-dir',
        default=None,
        help='Path to directory containing bedrock_models.json and bedrock_pricing.json'
    )
    parser.add_argument(
        '--json-output',
        action='store_true',
        help='Output raw JSON instead of formatted report'
    )
    args = parser.parse_args()

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Default to frontend/public relative to this script
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent.parent.parent / 'frontend' / 'public'

    models_path = data_dir / 'bedrock_models.json'
    pricing_path = data_dir / 'bedrock_pricing.json'

    # Validate files exist
    if not models_path.exists():
        print(f"Error: Models file not found at {models_path}", file=sys.stderr)
        sys.exit(1)
    if not pricing_path.exists():
        print(f"Error: Pricing file not found at {pricing_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading models from: {models_path}")
    print(f"Loading pricing from: {pricing_path}")
    print()

    # Load data
    models_data = load_json_file(str(models_path))
    pricing_data = load_json_file(str(pricing_path))

    # Run comparison
    comparison = run_comparison(models_data, pricing_data)

    # Output results
    if args.json_output:
        print(json.dumps(comparison, indent=2))
    else:
        print_report(comparison)


if __name__ == '__main__':
    main()
