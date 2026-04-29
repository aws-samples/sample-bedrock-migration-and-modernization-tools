#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Quick verification script to test if baseline traces work with the adapter.

This script tests the two most critical traces before running the full test suite:
1. good_001 - Basic trace without tools
2. good_002 - Trace with tool call/result pairing

Usage:
    python verify_adapter_format.py
"""

import json
import sys
from pathlib import Path

# Add agent_eval to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_eval.adapters.generic_json import adapt


def verify_trace(trace_file: str, expected_turns: int, expected_tools: int) -> bool:
    """
    Verify a trace can be parsed and produces expected structure.
    
    Args:
        trace_file: Path to trace JSON file
        expected_turns: Expected number of turns
        expected_tools: Expected number of tool calls
        
    Returns:
        True if verification passes, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Verifying: {trace_file}")
    print(f"{'='*60}")
    
    try:
        # Run adapter
        result = adapt(trace_file)
        
        # Check basic structure
        assert "run_id" in result, "Missing run_id"
        assert "metadata" in result, "Missing metadata"
        assert "turns" in result, "Missing turns"
        assert "adapter_stats" in result, "Missing adapter_stats"
        
        print(f"✓ Basic structure present")
        
        # Check turn count
        actual_turns = len(result["turns"])
        assert actual_turns == expected_turns, f"Expected {expected_turns} turns, got {actual_turns}"
        print(f"✓ Turn count: {actual_turns}")
        
        # Check tool count (check steps with TOOL_CALL kind)
        tool_call_steps = [
            s for s in result["turns"][0].get("steps", [])
            if s.get("kind") == "TOOL_CALL"
        ]
        actual_tools = len(tool_call_steps)
        assert actual_tools == expected_tools, f"Expected {expected_tools} tools, got {actual_tools}"
        print(f"✓ Tool count: {actual_tools}")
        
        # Check session_id extraction
        session_id = result["metadata"].get("session_id")
        print(f"✓ Session ID extracted: {session_id}")
        
        # Check turn_id extraction (first turn)
        if result["turns"]:
            turn_id = result["turns"][0].get("turn_id")
            print(f"✓ Turn ID extracted: {turn_id}")
        
        # Check user input extraction
        if result["turns"]:
            user_query = result["turns"][0].get("user_query")
            assert user_query, "Missing user_query"
            print(f"✓ User query extracted: {user_query[:50]}...")
        
        # Check assistant output extraction
        if result["turns"]:
            final_answer = result["turns"][0].get("final_answer")
            assert final_answer, "Missing final_answer"
            print(f"✓ Final answer extracted: {final_answer[:50]}...")
        
        # For tool traces, check tool linkage
        if expected_tools > 0:
            tool_call_steps = [
                s for s in result["turns"][0].get("steps", [])
                if s.get("kind") == "TOOL_CALL"
            ]
            if tool_call_steps:
                tool_step = tool_call_steps[0]
                # Check that tool_name is preserved (critical for linking)
                tool_name = tool_step.get("tool_name")
                assert tool_name, "Missing tool_name in TOOL_CALL step"
                print(f"✓ Tool name preserved: {tool_name}")
                
                # Check tool_run_id
                tool_run_id = tool_step.get("tool_run_id")
                assert tool_run_id, "Missing tool_run_id"
                print(f"✓ Tool run ID: {tool_run_id}")
                
                # Check if result is linked (look for TOOL_RESULT with same tool_run_id)
                tool_result_steps = [
                    s for s in result["turns"][0].get("steps", [])
                    if s.get("kind") == "TOOL_RESULT" and s.get("tool_run_id") == tool_run_id
                ]
                if tool_result_steps:
                    print(f"✓ Tool result linked by tool_run_id")
        
        # Check confidence score
        if result["turns"]:
            confidence = result["turns"][0].get("confidence")
            assert confidence is not None, "Missing confidence"
            assert 0 <= confidence <= 1, f"Confidence out of range: {confidence}"
            print(f"✓ Confidence score: {confidence:.2f}")
        
        # Check adapter_stats
        stats = result["adapter_stats"]
        assert "total_events_processed" in stats, "Missing total_events_processed"
        assert "turn_count" in stats, "Missing turn_count in stats"
        print(f"✓ Adapter stats: {stats['total_events_processed']} events, {stats['turn_count']} turns")
        
        print(f"\n✅ PASS: {trace_file}")
        return True
        
    except Exception as e:
        print(f"\n❌ FAIL: {trace_file}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run verification on critical traces."""
    baseline_dir = Path(__file__).parent
    
    print("="*60)
    print("BASELINE TRACE FORMAT VERIFICATION")
    print("="*60)
    print("\nThis script verifies that baseline traces work with the adapter.")
    print("Testing the two most critical traces:")
    print("  1. good_001 - Basic trace (no tools)")
    print("  2. good_002 - Tool trace (call + result)")
    
    results = []
    
    # Test 1: Basic trace without tools
    results.append(verify_trace(
        str(baseline_dir / "good_001_direct_answer.json"),
        expected_turns=1,
        expected_tools=0
    ))
    
    # Test 2: Trace with tool call/result
    results.append(verify_trace(
        str(baseline_dir / "good_002_tool_grounded.json"),
        expected_turns=1,
        expected_tools=1
    ))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All critical traces verified successfully!")
        print("The adapter can parse the baseline trace format.")
        print("\nNext step: Run full baseline test suite:")
        print("  pytest agent_eval/tests/baseline/ -v")
        return 0
    else:
        print(f"\n❌ {total - passed} trace(s) failed verification")
        print("Fix the trace format before running full tests.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
