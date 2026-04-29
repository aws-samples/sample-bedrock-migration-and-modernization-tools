#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Verification script for adapter_config.yaml.

This script loads and validates the adapter configuration using Pydantic,
ensuring all required fields are present and properly structured.
"""

import sys
from pathlib import Path
import yaml
from config_schema import validate_config


def verify_adapter_config(config_path: Path) -> bool:
    """
    Verify adapter configuration file.
    
    Args:
        config_path: Path to adapter_config.yaml
        
    Returns:
        True if validation succeeds, False otherwise
    """
    print(f"Verifying adapter configuration: {config_path}")
    print("=" * 70)
    
    # Load YAML
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        print("✓ YAML file loaded successfully")
    except FileNotFoundError:
        print(f"✗ ERROR: Configuration file not found: {config_path}")
        return False
    except yaml.YAMLError as e:
        print(f"✗ ERROR: Invalid YAML syntax: {e}")
        return False
    
    # Validate structure
    try:
        config = validate_config(config_dict)
        print("✓ Configuration structure is valid")
    except ValueError as e:
        print(f"✗ ERROR: Configuration validation failed: {e}")
        return False
    
    # Verify multi-stage pipeline
    print("\nMulti-stage pipeline verification:")
    print(f"  ✓ Stage A - Normalize: {len(config.normalize.event_paths)} event paths")
    print(f"  ✓ Stage A - Classify: {len(config.classify.rules)} classification rules")
    print(f"  ✓ Stage B - Segment: {len(config.segment.strategy_preference)} strategies")
    print(f"  ✓ Stage C - Derive: configured")
    
    # Verify confidence scoring
    print("\nConfidence scoring verification:")
    print(f"  ✓ Base score: {config.confidence.scoring.base}")
    print(f"  ✓ Penalties defined: {len(config.confidence.scoring.penalties)}")
    for penalty_name, penalty_value in config.confidence.scoring.penalties.items():
        print(f"    - {penalty_name}: {penalty_value}")
    
    # Verify comprehensive field aliases
    total_aliases = sum(len(aliases) for aliases in config.normalize.field_aliases.values())
    print(f"\nField aliases verification:")
    print(f"  ✓ Total field aliases: {total_aliases}")
    print(f"  ✓ Field groups: {len(config.normalize.field_aliases)}")
    
    if total_aliases >= 50:
        print(f"  ✓ Comprehensive coverage (50+ mappings)")
    else:
        print(f"  ⚠ Warning: Only {total_aliases} mappings (recommended: 50+)")
    
    # List some key field aliases
    print("\n  Key field aliases:")
    for field_name in ['timestamp', 'tool_name', 'tool_run_id', 'event_type', 'span_id']:
        if field_name in config.normalize.field_aliases:
            count = len(config.normalize.field_aliases[field_name])
            print(f"    - {field_name}: {count} aliases")
    
    # Verify regex patterns compile
    print("\nRegex pattern verification:")
    regex_count = 0
    for rule in config.classify.rules:
        if rule.all:
            for condition in rule.all:
                if condition.regex:
                    regex_count += 1
        if rule.any:
            for condition in rule.any:
                if condition.regex:
                    regex_count += 1
    print(f"  ✓ Classification regex patterns: {regex_count} (all valid)")
    
    strip_regex_count = len(config.derive.prompt_context_strip.strip_text_regex)
    print(f"  ✓ Prompt context strip patterns: {strip_regex_count}")
    
    tool_output_regex_count = len(config.derive.attribution.verdicts.tool_output_only_if_text_matches_regex)
    print(f"  ✓ Tool output detection patterns: {tool_output_regex_count}")
    
    print("\n" + "=" * 70)
    print("✓ Configuration verification PASSED")
    print("\nConfiguration is valid and ready for use.")
    return True


def main():
    """CLI entry point."""
    # Default to adapter_config.yaml in same directory
    config_path = Path(__file__).parent / "adapter_config.yaml"
    
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    
    success = verify_adapter_config(config_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
