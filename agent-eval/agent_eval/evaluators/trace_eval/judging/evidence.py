# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Evidence Extraction for Agent Traces Evaluator

This module handles extracting evidence from NormalizedRun data using JSONPath selectors.
It enforces scope constraints, evidence budgets, and PII redaction.
"""

import hashlib
import json
import re
import warnings
from typing import Dict, Any, List, Optional, Tuple
from jsonpath_ng.ext import parse
from jsonpath_ng.exceptions import JsonPathParserError


class EvidenceExtractionError(Exception):
    """Raised when evidence extraction fails."""
    pass


class EvidenceExtractor:
    """Extracts evidence from NormalizedRun using rubric evidence_selectors."""
    
    # PII patterns for basic redaction (conservative to avoid false positives)
    PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        # SSN with various formats
        "ssn": r'\b(?:SSN:?\s*)?(\d{3}[-\s]?\d{2}[-\s]?\d{4})\b',
    }
    
    # Sensitive field names to redact from tool payloads
    SENSITIVE_FIELDS = {
        "password", "api_key", "secret", "token", "auth", "credential",
        "ssn", "social_security", "credit_card", "cvv", "pin",
        "access_token", "id_token", "refresh_token", "authorization",
        "bearer", "x-api-key", "client_secret", "private_key"
    }
    
    # Safe operational fields that should is not expected to be redacted
    SAFE_OPERATIONAL_FIELDS = {
        "tool_name", "tool_id", "status", "latency_ms", "timestamp",
        "turn_id", "step_id", "kind", "type", "name", "id",
        "duration_ms", "start_time", "end_time", "success", "error_code",
        "http_status", "method", "url", "path", "query_params"
    }
    
    # Maximum matches per selector to prevent explosion
    MAX_MATCHES_PER_SELECTOR = 100
    
    def __init__(self, evidence_budget: int = 10000):
        """
        Initialize evidence extractor.
        
        Args:
            evidence_budget: Maximum characters per rubric payload (default 10,000)
        """
        self.evidence_budget = evidence_budget
    
    def extract_evidence(
        self,
        normalized_run: Dict[str, Any],
        evidence_selectors: List[str],
        scope: str,
        turn_id: Optional[str] = None,
        redact_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract evidence from NormalizedRun using JSONPath selectors.
        
        Args:
            normalized_run: The NormalizedRun data dictionary
            evidence_selectors: List of JSONPath expressions
            scope: "turn" or "run" (rubric-level scope)
            turn_id: Turn identifier (required for turn-scoped rubrics)
            redact_fields: Additional field names to redact (beyond defaults)
            
        Returns:
            Dictionary with extracted evidence and metadata including warnings
            
        Raises:
            EvidenceExtractionError: If extraction fails or scope is violated
        """
        # Validate scope value
        if scope not in {"turn", "run"}:
            raise EvidenceExtractionError(
                f"Invalid scope '{scope}'. Must be 'turn' or 'run'"
            )
        
        # Validate scope and turn_id consistency
        if scope == "turn" and not turn_id:
            raise EvidenceExtractionError(
                "turn_id is required for turn-scoped evidence extraction"
            )
        
        # Warn if no selectors provided
        if not evidence_selectors:
            warnings.warn(
                "No evidence_selectors provided; returning empty evidence",
                UserWarning
            )
            return {}
        
        # Prepare data based on scope
        if scope == "turn":
            # Extract specific turn data
            turn_data = self._get_turn_data(normalized_run, turn_id)
            if not turn_data:
                raise EvidenceExtractionError(
                    f"Turn {turn_id} not found in NormalizedRun"
                )
            extraction_context = turn_data
        else:
            # Use full run data
            extraction_context = normalized_run
        
        # Extract evidence using selectors
        extracted = {}
        extraction_warnings = []
        
        for idx, selector in enumerate(evidence_selectors):
            try:
                # Parse and apply JSONPath
                jsonpath_expr = parse(selector)
                matches = jsonpath_expr.find(extraction_context)
                
                # Limit matches to prevent explosion
                if len(matches) > self.MAX_MATCHES_PER_SELECTOR:
                    warning_msg = (
                        f"Selector '{selector}' matched {len(matches)} items, "
                        f"limiting to {self.MAX_MATCHES_PER_SELECTOR}"
                    )
                    warnings.warn(warning_msg, UserWarning)
                    extraction_warnings.append(warning_msg)
                    matches = matches[:self.MAX_MATCHES_PER_SELECTOR]
                
                # Collect matched values - typically return list for consistency
                values = [match.value for match in matches]
                
                # Use stable hash-based key to avoid collisions
                import hashlib
                selector_hash = hashlib.sha256(selector.encode('utf-8')).hexdigest()[:8]
                key = f"s{idx}_{selector_hash}"
                
                extracted[key] = {
                    "selector": selector,
                    "values": values,
                    "count": len(values)
                }
                
            except JsonPathParserError as e:
                raise EvidenceExtractionError(
                    f"Invalid JSONPath selector '{selector}': {str(e)}"
                )
            except Exception as e:
                raise EvidenceExtractionError(
                    f"Failed to extract evidence with selector '{selector}': {str(e)}"
                )
        
        # Apply PII redaction (before budget to avoid leaking PII in truncated data)
        extracted = self._redact_pii(extracted, redact_fields)
        
        # Enforce evidence budget and collect truncation warnings
        extracted, budget_warnings = self._enforce_budget(extracted)
        extraction_warnings.extend(budget_warnings)
        
        # Add warnings to output if any occurred
        if extraction_warnings:
            extracted["_warnings"] = extraction_warnings
        
        return extracted
    
    def _get_turn_data(self, normalized_run: Dict[str, Any], turn_id: str) -> Optional[Dict[str, Any]]:
        """
        Extract specific turn data from NormalizedRun.
        
        Args:
            normalized_run: The NormalizedRun data
            turn_id: Turn identifier
            
        Returns:
            Turn data dictionary or None if not found
        """
        turns = normalized_run.get("turns", [])
        for turn in turns:
            if turn.get("turn_id") == turn_id:
                return turn
        return None
    
    def _redact_pii(
        self,
        data: Dict[str, Any],
        additional_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Redact PII from extracted evidence.
        
        Applies field-based redaction first (safer), then pattern-based on string values only.
        
        Args:
            data: Extracted evidence dictionary
            additional_fields: Additional field names to redact
            
        Returns:
            Redacted evidence dictionary
        """
        # Combine default sensitive fields with additional ones
        sensitive_fields = self.SENSITIVE_FIELDS.copy()
        if additional_fields:
            sensitive_fields.update(f.lower() for f in additional_fields)
        
        # Apply field-based redaction first (recursive)
        redacted = self._redact_sensitive_fields(data, sensitive_fields)
        
        # Apply pattern-based redaction only to string values (avoid type changes)
        redacted = self._redact_pii_patterns(redacted)
        
        return redacted
    
    def _redact_pii_patterns(self, data: Any) -> Any:
        """
        Apply pattern-based PII redaction to string values only.
        
        Args:
            data: Data to redact (dict, list, or primitive)
            
        Returns:
            Redacted data with same types preserved
        """
        if isinstance(data, dict):
            return {
                key: self._redact_pii_patterns(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._redact_pii_patterns(item) for item in data]
        elif isinstance(data, str):
            # Apply pattern redaction only to strings
            redacted_str = data
            for pii_type, pattern in self.PII_PATTERNS.items():
                redacted_str = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted_str, flags=re.IGNORECASE)
            return redacted_str
        else:
            # Preserve non-string types as-is
            return data
    
    def _redact_sensitive_fields(
        self,
        data: Any,
        sensitive_fields: set
    ) -> Any:
        """
        Recursively redact sensitive fields from data structure.
        
        Applies allowlist for safe operational fields to prevent over-redaction.
        
        Args:
            data: Data to redact (dict, list, or primitive)
            sensitive_fields: Set of field names to redact
            
        Returns:
            Redacted data
        """
        if isinstance(data, dict):
            return {
                key: "[REDACTED]" if (
                    key.lower() in sensitive_fields and 
                    key.lower() not in self.SAFE_OPERATIONAL_FIELDS
                )
                else self._redact_sensitive_fields(value, sensitive_fields)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._redact_sensitive_fields(item, sensitive_fields) for item in data]
        else:
            return data
    
    def _enforce_budget(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Enforce evidence budget by truncating if necessary.
        
        Uses intelligent truncation: limits large values before dropping selectors.
        
        Args:
            data: Extracted evidence dictionary
            
        Returns:
            Tuple of (truncated evidence dictionary, list of warning messages)
        """
        budget_warnings = []
        
        # Serialize to measure size accurately
        data_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        data_size = len(data_str)
        
        if data_size <= self.evidence_budget:
            return data, budget_warnings
        
        # Budget exceeded - apply intelligent truncation
        warning_msg = (
            f"Evidence size ({data_size} chars) exceeds budget ({self.evidence_budget} chars). "
            f"Applying intelligent truncation."
        )
        warnings.warn(warning_msg, UserWarning)
        budget_warnings.append(warning_msg)
        
        # Count original selectors (exclude metadata keys)
        original_selector_count = sum(1 for k in data.keys() if not k.startswith('_'))
        
        # Strategy: First truncate large string values, then limit list sizes, then drop selectors
        truncated = {}
        
        # Sort selectors by size (keep smaller ones preferentially)
        sorted_items = sorted(
            data.items(),
            key=lambda item: len(json.dumps(item[1], ensure_ascii=False, separators=(',', ':')))
        )
        
        current_size = 2  # Account for outer dict braces
        
        for key, value in sorted_items:
            # Try to add this selector's data
            truncated_value = self._truncate_value(value, max_chars=5000)
            
            # Calculate size with proper JSON formatting
            item_str = json.dumps({key: truncated_value}, ensure_ascii=False, separators=(',', ':'))
            item_size = len(item_str) - 2  # Subtract outer braces
            
            # Add comma separator if not first item
            if len(truncated) > 0:
                item_size += 1
            
            # Reserve space for metadata (5 keys × ~50 chars each = 250 chars)
            if current_size + item_size <= self.evidence_budget - 250:
                truncated[key] = truncated_value
                current_size += item_size
            else:
                # Can't fit more selectors
                break
        
        # Count kept selectors (exclude metadata keys)
        kept_selector_count = sum(1 for k in truncated.keys() if not k.startswith('_'))
        dropped_selector_count = original_selector_count - kept_selector_count
        
        # Add truncation metadata
        truncated["_truncation_applied"] = True
        truncated["_original_size_chars"] = data_size
        truncated["_budget_chars"] = self.evidence_budget
        truncated["_selectors_kept"] = kept_selector_count
        truncated["_selectors_dropped"] = dropped_selector_count
        
        return truncated, budget_warnings
    
    def _truncate_value(self, value: Any, max_chars: int = 5000) -> Any:
        """
        Truncate large values intelligently.
        
        Args:
            value: Value to potentially truncate
            max_chars: Maximum characters for string/list representation
            
        Returns:
            Truncated value
        """
        if isinstance(value, dict):
            # For selector result dicts, truncate the values list
            if "values" in value and isinstance(value["values"], list):
                truncated_values = []
                for item in value["values"]:
                    if isinstance(item, str) and len(item) > max_chars:
                        truncated_values.append(item[:max_chars] + "...[TRUNCATED]")
                    else:
                        truncated_values.append(item)
                
                # Limit list length
                if len(truncated_values) > 50:
                    truncated_values = truncated_values[:50]
                    value = value.copy()
                    value["values"] = truncated_values
                    value["_list_truncated"] = True
                else:
                    value = value.copy()
                    value["values"] = truncated_values
            
            return value
        elif isinstance(value, str) and len(value) > max_chars:
            return value[:max_chars] + "...[TRUNCATED]"
        elif isinstance(value, list) and len(value) > 50:
            return value[:50] + ["...[TRUNCATED]"]
        else:
            return value
