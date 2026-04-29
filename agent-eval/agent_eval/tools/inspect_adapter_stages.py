#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Inspection script to dump intermediate normalized state for failing corpus cases.

This script helps localize which pipeline stage contains the bugs by dumping
intermediate state at key points in the adapter pipeline.

Usage:
    python -m agent_eval.tools.inspect_adapter_stages <trace_file>
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_eval.adapters.generic_json.adapter import _TraceNormalizer
from agent_eval.adapters.generic_json.config_loader import AdapterConfig
from agent_eval.adapters.generic_json import DEFAULT_CONFIG_PATH


def inspect_trace(trace_path: str) -> None:
    """
    Inspect a trace file and dump intermediate state at each pipeline stage.
    
    Args:
        trace_path: Path to the trace JSON file
    """
    print(f"\n{'='*80}")
    print(f"INSPECTING TRACE: {trace_path}")
    print(f"{'='*80}\n")
    
    # Load configuration
    config = AdapterConfig(str(DEFAULT_CONFIG_PATH))
    
    # Create normalizer
    normalizer = _TraceNormalizer(config)
    
    # Load trace data
    with open(trace_path, 'r') as f:
        raw_data = json.load(f)
    
    # Extract events
    events = normalizer._extract_events(raw_data)
    print(f"Total events extracted: {len(events)}")
    
    # STAGE A: Normalize events (field extraction)
    print(f"\n{'='*80}")
    print("STAGE A: AFTER FIELD EXTRACTION, BEFORE SEGMENTATION")
    print(f"{'='*80}\n")
    
    normalized_events = normalizer._normalize_events(events)
    
    for idx, event in enumerate(normalized_events):
        print(f"\nEvent {idx}:")
        print(f"  kind: {event.get('kind')}")
        print(f"  turn_id: {event.get('turn_id')}")
        print(f"  tool_run_id: {event.get('tool_run_id')}")
        print(f"  tool_name: {event.get('tool_name')}")
        print(f"  status: {event.get('status')}")
        print(f"  text: {event.get('text', '')[:50]}..." if event.get('text') else "  text: None")
        print(f"  event_type: {event.get('event_type')}")
        print(f"  operation: {event.get('operation')}")
    
    # STAGE B: Segment into turns
    print(f"\n{'='*80}")
    print("STAGE B: SEGMENTATION")
    print(f"{'='*80}\n")
    
    turn_groups, strategy_used, strategy_reason = normalizer._segment_into_turns(normalized_events, raw_data)
    
    print(f"Segmentation strategy: {strategy_used}")
    print(f"Reason: {strategy_reason}")
    print(f"Number of turns created: {len(turn_groups)}")
    
    for turn_idx, turn_events in enumerate(turn_groups):
        print(f"\nTurn {turn_idx}: {len(turn_events)} events")
        for evt in turn_events:
            print(f"  - {evt.get('kind')} (turn_id={evt.get('turn_id')}, tool_run_id={evt.get('tool_run_id')})")
    
    # STAGE C: Process each turn (tool linking, step emission)
    print(f"\n{'='*80}")
    print("STAGE C: AFTER TOOL LINKING, BEFORE STEP EMISSION")
    print(f"{'='*80}\n")
    
    for turn_idx, turn_events in enumerate(turn_groups):
        print(f"\n--- Turn {turn_idx} ---")
        
        # Order events
        ordered_events = normalizer._order_events_within_turn(turn_events)
        
        # Link tool calls and results (this modifies events in-place)
        turn_id = f"turn_{turn_idx}"
        linked_events = normalizer._link_tool_calls_and_results(ordered_events, turn_id)
        
        print(f"\nAfter tool linking:")
        for evt_idx, event in enumerate(linked_events):
            print(f"\nEvent {evt_idx}:")
            print(f"  kind: {event.get('kind')}")
            print(f"  turn_id: {event.get('turn_id')}")
            print(f"  tool_run_id: {event.get('tool_run_id')}")
            print(f"  tool_name: {event.get('tool_name')}")
            print(f"  status: {event.get('status')}")
            print(f"  _linked_by: {event.get('_linked_by')}")
            print(f"  _linked_to_call: {event.get('_linked_to_call')}")
            print(f"  _linked_to_result: {event.get('_linked_to_result')}")
            
            # For TOOL_RESULT events, show the result content
            if event.get('kind') == 'TOOL_RESULT':
                print(f"  tool_result: {str(event.get('tool_result', ''))[:100]}...")
                print(f"  error: {event.get('error')}")
        
        # Convert to steps to see final status values
        print(f"\nAfter step conversion:")
        for evt_idx, event in enumerate(linked_events):
            step = normalizer._event_to_step(event)
            print(f"\nStep {evt_idx}:")
            print(f"  kind: {step.get('kind')}")
            print(f"  tool_name: {step.get('tool_name')}")
            print(f"  status: {step.get('status')}")
            print(f"  text: {step.get('text', '')[:50]}..." if step.get('text') else "  text: None")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m agent_eval.tools.inspect_adapter_stages <trace_file>")
        print("\nExample:")
        print("  python -m agent_eval.tools.inspect_adapter_stages test-fixtures/baseline/good_002_tool_grounded.json")
        sys.exit(1)
    
    trace_path = sys.argv[1]
    
    if not Path(trace_path).exists():
        print(f"Error: Trace file not found: {trace_path}")
        sys.exit(1)
    
    inspect_trace(trace_path)


if __name__ == "__main__":
    main()
