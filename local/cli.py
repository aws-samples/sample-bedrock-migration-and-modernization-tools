"""
CLI entry point for local Bedrock Model Profiler data collection.

Usage:
    python -m local.cli collect
    python -m local.cli collect --profile my-aws-profile
    python -m local.cli collect --output ./my-data
"""

import argparse
import logging
import sys
from pathlib import Path


def setup_logging(verbose: bool = False):
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' if verbose else '%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Reduce noise from boto3/botocore
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def cmd_collect(args):
    """Run the data collection pipeline."""
    from local.collector import LocalCollector

    collector = LocalCollector(
        profile_name=args.profile,
        output_dir=Path(args.output)
    )

    try:
        results = collector.collect_all()
        return 0
    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")
        return 1
    except Exception as e:
        logging.error(f"Collection failed: {e}")
        if args.verbose:
            logging.exception("Full traceback:")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='bedrock-profiler',
        description='Bedrock Model Profiler - Local Data Collection'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Collect command
    collect_parser = subparsers.add_parser(
        'collect',
        help='Collect Bedrock model and pricing data'
    )
    collect_parser.add_argument(
        '--profile', '-p',
        help='AWS profile name (uses default if not specified)'
    )
    collect_parser.add_argument(
        '--output', '-o',
        default='data',
        help='Output directory for JSON files (default: ./data)'
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.command == 'collect':
        sys.exit(cmd_collect(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
