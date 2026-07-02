"""
Generate models_profiles.jsonl from the bedrock-pricing-scraper CSV output.

Usage:
    python scripts/generate_profiles_from_csv.py <csv_path> [--output config/models_profiles.jsonl]

Reads the CSV with columns: provider, model_family, model_name, tier, region_name, region_code, input_1m_tokens, output_1m_tokens
Maps display names to Bedrock model IDs using the ListFoundationModels API.
Outputs JSONL with: model_id, region, input_token_cost, output_token_cost, service_tiers
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from bedrock_pricing import fetch_foundation_models, fetch_inference_profiles


def normalize(name):
    """Normalize a model name for fuzzy matching."""
    s = name.lower().strip()
    s = re.sub(r'\s*\(.*?\)\s*', ' ', s)  # Remove parenthetical notes
    s = re.sub(r'\binstruct\b', '', s)
    s = re.sub(r'\bv\d+\b', '', s)
    s = re.sub(r'\b(w/|with)\b.*', '', s)  # Remove "w/ latency optimized" etc.
    s = re.sub(r'[^a-z0-9. ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def build_name_to_id_map():
    """Build a mapping from display names to Bedrock model IDs."""
    print("Fetching foundation models from Bedrock API...")
    name_to_info, id_to_name = fetch_foundation_models()

    print("Fetching inference profiles for cross-region IDs...")
    cross_region_map = fetch_inference_profiles()

    # Build direct name -> model_id map
    name_to_id = {}
    for name, info in name_to_info.items():
        name_to_id[name.lower().strip()] = info["modelId"]

    # Build normalized name -> model_id map for fuzzy matching
    norm_to_id = {}
    for name, info in name_to_info.items():
        norm_to_id[normalize(name)] = info["modelId"]

    # Build cross-region prefix map: base_id -> preferred cross-region id
    cross_region_ids = {}
    for base_id, profiles in cross_region_map.items():
        # Prefer us. prefix, then global.
        for p in profiles:
            if p.startswith("us."):
                cross_region_ids[base_id] = p
                break
        if base_id not in cross_region_ids and profiles:
            cross_region_ids[base_id] = profiles[0]

    return name_to_id, norm_to_id, cross_region_ids


def resolve_model_id(csv_name, name_to_id, norm_to_id, cross_region_ids):
    """Resolve a CSV model name to a Bedrock model ID with cross-region prefix."""
    # Direct match
    key = csv_name.lower().strip()
    if key in name_to_id:
        base_id = name_to_id[key]
        full_id = cross_region_ids.get(base_id, base_id)
        return f"bedrock/{full_id}"

    # Normalized fuzzy match
    norm_key = normalize(csv_name)
    if norm_key in norm_to_id:
        base_id = norm_to_id[norm_key]
        full_id = cross_region_ids.get(base_id, base_id)
        return f"bedrock/{full_id}"

    # Try partial matching — find the best match by longest common substring
    best_match = None
    best_score = 0
    for api_name, model_id in name_to_id.items():
        # Score by how many words match
        csv_words = set(norm_key.split())
        api_words = set(normalize(api_name).split())
        overlap = len(csv_words & api_words)
        total = max(len(csv_words), len(api_words))
        if total > 0:
            score = overlap / total
            if score > best_score and score >= 0.6:
                best_score = score
                best_match = model_id

    if best_match:
        full_id = cross_region_ids.get(best_match, best_match)
        return f"bedrock/{full_id}"

    return None


def generate_profiles(csv_path, output_path):
    """Generate models_profiles.jsonl from CSV."""
    # Read CSV
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Read {len(rows)} rows from {csv_path}")

    # Build name mapping
    name_to_id, norm_to_id, cross_region_ids = build_name_to_id_map()

    # Group by model_name to collect all tiers and regions
    # Key: (model_name, region_code) -> {tiers, input_cost, output_cost}
    model_data = defaultdict(lambda: {"tiers": set(), "input_cost": None, "output_cost": None})

    unresolved = set()
    resolved_names = {}

    for row in rows:
        model_name = row["model_name"]
        tier = row["tier"]
        region = row["region_code"]
        input_cost = float(row["input_1m_tokens"])  # Cost per 1M tokens
        output_cost = float(row["output_1m_tokens"])

        # Skip batch tier (not an inference service tier)
        if tier.lower() == "batch":
            continue

        # Resolve model ID (only once per model name)
        if model_name not in resolved_names:
            model_id = resolve_model_id(model_name, name_to_id, norm_to_id, cross_region_ids)
            if model_id:
                resolved_names[model_name] = model_id
            else:
                unresolved.add(model_name)
                continue
        elif model_name in unresolved:
            continue

        model_id = resolved_names[model_name]
        key = (model_id, region)

        # Map tier names: Standard -> default, Priority -> priority, Flex -> flex
        tier_mapped = {"standard": "default", "priority": "priority", "flex": "flex"}.get(tier.lower(), tier.lower())
        model_data[key]["tiers"].add(tier_mapped)

        # Use Standard tier pricing as the base cost
        if tier.lower() == "standard":
            model_data[key]["input_cost"] = input_cost
            model_data[key]["output_cost"] = output_cost

    # Generate JSONL entries
    entries = []
    for (model_id, region), data in sorted(model_data.items()):
        if data["input_cost"] is None:
            continue  # No standard pricing found
        entries.append({
            "model_id": model_id,
            "region": region,
            "input_token_cost": data["input_cost"],
            "output_token_cost": data["output_cost"],
            "service_tiers": sorted(data["tiers"]),
        })

    # Write JSONL
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    # Report
    print(f"\nWritten {len(entries)} entries to {output_path}")
    print(f"Resolved: {len(resolved_names)} model names")
    if unresolved:
        print(f"Unresolved ({len(unresolved)}):")
        for name in sorted(unresolved):
            print(f"  - {name}")

    # Tier summary
    from collections import Counter
    tier_counts = Counter()
    multi_tier = 0
    for entry in entries:
        for t in entry["service_tiers"]:
            tier_counts[t] += 1
        if len(entry["service_tiers"]) > 1:
            multi_tier += 1
    print(f"\nTier distribution: {dict(tier_counts)}")
    print(f"Entries with priority/flex: {multi_tier}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate models_profiles.jsonl from pricing CSV")
    parser.add_argument("csv_path", help="Path to bedrock pricing CSV")
    parser.add_argument("--output", default="config/models_profiles.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    generate_profiles(args.csv_path, args.output)