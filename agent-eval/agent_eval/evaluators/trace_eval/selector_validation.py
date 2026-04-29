# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Selector Validation for Agent Traces Evaluator

This module provides validation functions to detect mismatched selectors
in rubric configurations, specifically preventing run-level selectors
from being used in turn-scoped rubrics.
"""

import re
from typing import List


class SelectorValidationError(Exception):
    """Raised when selector validation fails."""
    pass


def validate_turn_scoped_selectors(
    rubric_id: str,
    scope: str,
    turn_selector_mode: str,
    evidence_selectors: List[str]
) -> None:
    """
    Validate that turn-scoped rubrics don't use run-level selectors.
    
    When a rubric has scope='turn' and turn_selector_mode='narrow_to_turn',
    the evidence extraction context is narrowed to a single turn object.
    Using run-level selectors (like $.turns[*]) in this context will fail
    because the selector expects a run object but receives a turn object.
    
    This function detects common run-level selector patterns and raises
    a validation error if found in turn-scoped rubrics.
    
    Args:
        rubric_id: The rubric identifier (for error messages)
        scope: The rubric scope ('turn' or 'run')
        turn_selector_mode: The turn selector mode ('narrow_to_turn' or 'full_context')
        evidence_selectors: List of JSONPath selector strings
        
    Raises:
        SelectorValidationError: If run-level selectors are found in turn-scoped rubrics
    """
    # Only validate turn-scoped rubrics with narrow_to_turn mode
    if scope != "turn" or turn_selector_mode != "narrow_to_turn":
        return
    
    # Patterns that indicate run-level selectors
    # These patterns access the 'turns' array which doesn't exist in turn context
    run_level_patterns = [
        r'\$\.turns\[\*\]',           # $.turns[*]
        r'\$\.turns\[\d+\]',          # $.turns[0], $.turns[1], etc.
        r'\$\.turns\[',               # $.turns[ (any indexed/filtered access)
        r'\$\.\.turns\[',             # $..turns[ (recursive descent)
        r'\$\.turns(?![a-zA-Z_])',    # $.turns (accessing turns array directly, not a field like 'turns_count')
    ]
    
    # Check each selector for run-level patterns
    invalid_selectors = []
    for selector in evidence_selectors:
        for pattern in run_level_patterns:
            if re.search(pattern, selector):
                invalid_selectors.append(selector)
                break  # No need to check other patterns for this selector
    
    # Raise error if any invalid selectors found
    if invalid_selectors:
        error_msg = (
            f"Rubric '{rubric_id}' has scope='turn' with turn_selector_mode='narrow_to_turn' "
            f"but uses run-level selectors that reference $.turns[*] or $.turns[N]. "
            f"When scope='turn', the evidence extraction context is narrowed to a single turn object, "
            f"so selectors must use turn-relative paths (e.g., '$.user_query' instead of '$.turns[*].user_query'). "
            f"\n\nInvalid selectors found:\n"
        )
        for selector in invalid_selectors:
            error_msg += f"  - {selector}\n"
        
        error_msg += (
            f"\nTo fix this:\n"
            f"1. Remove '$.turns[*].' prefix from selectors (e.g., '$.turns[*].user_query' → '$.user_query')\n"
            f"2. Remove indexed access (e.g., '$.turns[0].final_answer' → '$.final_answer')\n"
            f"3. Or change scope to 'run' if you need to access multiple turns"
        )
        
        raise SelectorValidationError(error_msg)
