#!/usr/bin/env python3
"""
Pricing Coverage Analyzer

Automatically analyzes pricing coverage after deployments.
Compares new pricing data against a baseline (old version) and reports differences.

Usage:
    python pricing_coverage_analyzer.py [--baseline PATH] [--download] [--region REGION]

Examples:
    # Download latest from S3 and analyze
    python pricing_coverage_analyzer.py --download

    # Analyze local file against baseline
    python pricing_coverage_analyzer.py /path/to/pricing.json

    # Use custom baseline
    python pricing_coverage_analyzer.py --download --baseline /path/to/baseline.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Default paths
DEFAULT_BASELINE = Path(__file__).parent.parent.parent / "shared-model-profiler 2/bedrock-model-profiler_2026/public/bedrock_pricing.json"
DEFAULT_S3_BUCKET = "bedrock-profiler-data-169497827606-dev"
DEFAULT_S3_KEY = "latest/bedrock_pricing.json"
DEFAULT_REGION = "us-west-2"


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.END}")


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BLUE}{'-' * len(text)}{Colors.END}")


def download_from_s3(bucket: str, key: str, region: str) -> str:
    """Download pricing data from S3."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    temp_file.close()

    print(f"{Colors.YELLOW}Downloading from s3://{bucket}/{key}...{Colors.END}")

    try:
        result = subprocess.run(
            ['aws', 's3', 'cp', f's3://{bucket}/{key}', temp_file.name, '--region', region],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"{Colors.GREEN}✓ Downloaded successfully{Colors.END}")
        return temp_file.name
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}✗ Failed to download: {e.stderr}{Colors.END}")
        sys.exit(1)


def load_pricing_data(file_path: str) -> dict:
    """Load pricing data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_pricing(data: dict) -> Tuple[Dict[str, int], int, dict]:
    """Analyze pricing data and return provider counts and total models."""
    providers = data.get('providers', {})
    provider_counts = {}
    total_models = 0
    model_details = {}

    for provider, models in providers.items():
        model_count = len(models)
        provider_counts[provider] = model_count
        total_models += model_count
        model_details[provider] = list(models.keys())

    return provider_counts, total_models, model_details


def compare_coverage(new_counts: Dict[str, int], old_counts: Dict[str, int]) -> dict:
    """Compare pricing coverage between two versions."""
    all_providers = set(new_counts.keys()) | set(old_counts.keys())

    changes = {
        'added': {},      # Providers added (weren't in old)
        'removed': {},    # Providers removed (were in old, not in new)
        'increased': {},  # Providers with more models
        'decreased': {},  # Providers with fewer models
        'unchanged': {},  # Providers with same count
    }

    for provider in all_providers:
        new_count = new_counts.get(provider, 0)
        old_count = old_counts.get(provider, 0)
        diff = new_count - old_count

        if old_count == 0 and new_count > 0:
            changes['added'][provider] = new_count
        elif new_count == 0 and old_count > 0:
            changes['removed'][provider] = old_count
        elif diff > 0:
            changes['increased'][provider] = (old_count, new_count, diff)
        elif diff < 0:
            changes['decreased'][provider] = (old_count, new_count, diff)
        else:
            changes['unchanged'][provider] = new_count

    return changes


def print_provider_table(provider_counts: Dict[str, int], title: str) -> None:
    """Print a formatted provider table."""
    print_section(title)

    # Sort by count descending
    sorted_providers = sorted(provider_counts.items(), key=lambda x: (-x[1], x[0]))

    print(f"{'Provider':<25} {'Models':>8}")
    print(f"{'-' * 25} {'-' * 8}")

    for provider, count in sorted_providers:
        if provider == 'Unknown Models':
            color = Colors.YELLOW
        elif count >= 10:
            color = Colors.GREEN
        else:
            color = Colors.END
        print(f"{color}{provider:<25} {count:>8}{Colors.END}")

    total = sum(provider_counts.values())
    print(f"{'-' * 25} {'-' * 8}")
    print(f"{Colors.BOLD}{'TOTAL':<25} {total:>8}{Colors.END}")


def print_comparison_report(changes: dict, new_total: int, old_total: int) -> None:
    """Print a detailed comparison report."""
    print_header("COVERAGE COMPARISON REPORT")

    # Summary
    total_diff = new_total - old_total
    if total_diff > 0:
        status = f"{Colors.GREEN}✓ IMPROVED{Colors.END}"
        diff_str = f"{Colors.GREEN}+{total_diff}{Colors.END}"
    elif total_diff < 0:
        status = f"{Colors.RED}✗ REGRESSED{Colors.END}"
        diff_str = f"{Colors.RED}{total_diff}{Colors.END}"
    else:
        status = f"{Colors.YELLOW}○ UNCHANGED{Colors.END}"
        diff_str = "0"

    print(f"\n{Colors.BOLD}Summary:{Colors.END}")
    print(f"  Status: {status}")
    print(f"  Total Models: {old_total} → {new_total} ({diff_str})")
    print(f"  Total Providers: {len(changes['unchanged']) + len(changes['increased']) + len(changes['decreased']) + len(changes['added'])} " +
          f"(+{len(changes['added'])} new, -{len(changes['removed'])} removed)")

    # New providers
    if changes['added']:
        print_section(f"✅ New Providers ({len(changes['added'])})")
        for provider, count in sorted(changes['added'].items()):
            print(f"  {Colors.GREEN}+ {provider}: {count} models{Colors.END}")

    # Increased coverage
    if changes['increased']:
        print_section(f"📈 Increased Coverage ({len(changes['increased'])})")
        for provider, (old, new, diff) in sorted(changes['increased'].items()):
            print(f"  {Colors.GREEN}{provider}: {old} → {new} (+{diff}){Colors.END}")

    # Decreased coverage
    if changes['decreased']:
        print_section(f"📉 Decreased Coverage ({len(changes['decreased'])})")
        for provider, (old, new, diff) in sorted(changes['decreased'].items()):
            print(f"  {Colors.RED}{provider}: {old} → {new} ({diff}){Colors.END}")

    # Removed providers
    if changes['removed']:
        print_section(f"❌ Removed Providers ({len(changes['removed'])})")
        for provider, count in sorted(changes['removed'].items()):
            print(f"  {Colors.RED}- {provider}: was {count} models{Colors.END}")

    # Unknown models warning
    unknown_new = changes['unchanged'].get('Unknown Models', 0)
    if 'Unknown Models' in changes['increased']:
        unknown_new = changes['increased']['Unknown Models'][1]
    elif 'Unknown Models' in changes['decreased']:
        unknown_new = changes['decreased']['Unknown Models'][1]

    if unknown_new > 0:
        print_section(f"⚠️  Unknown Models: {unknown_new}")
        print(f"  {Colors.YELLOW}These models couldn't be matched to a provider.{Colors.END}")
        print(f"  {Colors.YELLOW}Consider adding provider patterns for them.{Colors.END}")


def print_unknown_models_details(model_details: dict) -> None:
    """Print details of unknown models."""
    unknown = model_details.get('Unknown Models', [])
    if unknown:
        print_section("Unknown Models Details")
        for model_id in sorted(unknown):
            print(f"  - {model_id}")


def run_analysis(
    pricing_file: str,
    baseline_file: Optional[str] = None,
    show_unknown_details: bool = True
) -> dict:
    """Run the full pricing coverage analysis."""

    print_header("PRICING COVERAGE ANALYSIS")
    print(f"\n{Colors.BOLD}Analysis Time:{Colors.END} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{Colors.BOLD}Pricing File:{Colors.END} {pricing_file}")

    # Load and analyze new pricing data
    new_data = load_pricing_data(pricing_file)
    new_counts, new_total, new_details = analyze_pricing(new_data)

    # Print metadata
    metadata = new_data.get('metadata', {})
    if metadata:
        print(f"{Colors.BOLD}Generated At:{Colors.END} {metadata.get('generated_at', 'Unknown')}")
        print(f"{Colors.BOLD}Total Pricing Entries:{Colors.END} {metadata.get('total_pricing_entries', 'Unknown')}")

    # Print current coverage
    print_provider_table(new_counts, "Current Coverage")

    # Compare with baseline if available
    if baseline_file and os.path.exists(baseline_file):
        print(f"\n{Colors.BOLD}Baseline File:{Colors.END} {baseline_file}")

        old_data = load_pricing_data(baseline_file)
        old_counts, old_total, old_details = analyze_pricing(old_data)

        changes = compare_coverage(new_counts, old_counts)
        print_comparison_report(changes, new_total, old_total)

        # Show unknown models details
        if show_unknown_details and 'Unknown Models' in new_counts and new_counts['Unknown Models'] > 0:
            print_unknown_models_details(new_details)

        return {
            'status': 'improved' if new_total > old_total else ('regressed' if new_total < old_total else 'unchanged'),
            'new_total': new_total,
            'old_total': old_total,
            'diff': new_total - old_total,
            'changes': changes,
            'unknown_count': new_counts.get('Unknown Models', 0)
        }
    else:
        if baseline_file:
            print(f"\n{Colors.YELLOW}⚠️  Baseline file not found: {baseline_file}{Colors.END}")

        # Show unknown models details even without baseline
        if show_unknown_details and 'Unknown Models' in new_counts and new_counts['Unknown Models'] > 0:
            print_unknown_models_details(new_details)

        return {
            'status': 'no_baseline',
            'new_total': new_total,
            'unknown_count': new_counts.get('Unknown Models', 0)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze pricing coverage after deployments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'pricing_file',
        nargs='?',
        help='Path to pricing JSON file (optional if --download is used)'
    )

    parser.add_argument(
        '--download', '-d',
        action='store_true',
        help='Download latest pricing data from S3'
    )

    parser.add_argument(
        '--baseline', '-b',
        type=str,
        default=str(DEFAULT_BASELINE),
        help=f'Path to baseline pricing JSON for comparison (default: {DEFAULT_BASELINE})'
    )

    parser.add_argument(
        '--bucket',
        type=str,
        default=DEFAULT_S3_BUCKET,
        help=f'S3 bucket name (default: {DEFAULT_S3_BUCKET})'
    )

    parser.add_argument(
        '--key',
        type=str,
        default=DEFAULT_S3_KEY,
        help=f'S3 key (default: {DEFAULT_S3_KEY})'
    )

    parser.add_argument(
        '--region', '-r',
        type=str,
        default=DEFAULT_REGION,
        help=f'AWS region (default: {DEFAULT_REGION})'
    )

    parser.add_argument(
        '--no-baseline',
        action='store_true',
        help='Skip baseline comparison'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    args = parser.parse_args()

    # Determine pricing file
    if args.download:
        pricing_file = download_from_s3(args.bucket, args.key, args.region)
    elif args.pricing_file:
        pricing_file = args.pricing_file
    else:
        parser.error('Either provide a pricing file path or use --download')

    # Determine baseline
    baseline_file = None if args.no_baseline else args.baseline

    # Run analysis
    result = run_analysis(pricing_file, baseline_file)

    # Output JSON if requested
    if args.json:
        print(f"\n{Colors.BOLD}JSON Output:{Colors.END}")
        print(json.dumps(result, indent=2, default=str))

    # Clean up temp file if downloaded
    if args.download and os.path.exists(pricing_file):
        os.unlink(pricing_file)

    # Exit with appropriate code
    if result['status'] == 'regressed':
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
