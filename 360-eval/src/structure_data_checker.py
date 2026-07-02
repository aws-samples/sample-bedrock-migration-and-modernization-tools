"""
Structured Data Checker

Validates LLM responses against expected data structure formats.
Used during evaluation to programmatically verify structured output
before passing results to judge models.
"""

import csv
import io
import json
import logging
import re
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def validate_structure(response_text: str, expected_format: str) -> dict:
    """
    Validate if a response matches the expected data structure format.

    Args:
        response_text: The LLM response text to validate
        expected_format: One of "json", "csv-comma", "csv-pipe",
                        "markdown", "yaml", "html", "xml"

    Returns:
        {"valid": bool, "error": str or None}
    """
    if not response_text or not response_text.strip():
        return {"valid": False, "error": "Empty response"}

    text = response_text.strip()

    # Strip markdown code fences if present (```json ... ``` or ```xml ... ```)
    code_block = re.search(r'```(?:\w+)?\s*\n?(.*?)```', text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    validators = {
        "json": _validate_json,
        "csv-comma": _validate_csv_comma,
        "csv-pipe": _validate_csv_pipe,
        "yaml": _validate_yaml,
        "html": _validate_html,
        "xml": _validate_xml,
        "markdown": _validate_markdown,
    }

    validator = validators.get(expected_format)
    if not validator:
        return {"valid": False, "error": f"Unknown format: {expected_format}"}

    try:
        return validator(text)
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}


def _validate_json(text: str) -> dict:
    """Validate JSON structure."""
    try:
        json.loads(text)
        return {"valid": True, "error": None}
    except json.JSONDecodeError as e:
        return {"valid": False, "error": f"Invalid JSON: {str(e)}"}


def _validate_csv_comma(text: str) -> dict:
    """Validate CSV with comma delimiter."""
    return _validate_csv(text, delimiter=',', name="CSV (comma)")


def _validate_csv_pipe(text: str) -> dict:
    """Validate CSV with pipe delimiter."""
    return _validate_csv(text, delimiter='|', name="CSV (pipe)")


def _validate_csv(text: str, delimiter: str, name: str) -> dict:
    """Validate CSV structure with given delimiter."""
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows = list(reader)

        if len(rows) < 1:
            return {"valid": False, "error": f"Invalid {name}: no rows found"}

        # Check that rows have consistent column count
        col_counts = [len(row) for row in rows if row]
        if not col_counts:
            return {"valid": False, "error": f"Invalid {name}: no data rows"}

        if len(set(col_counts)) > 1:
            return {"valid": False, "error": f"Invalid {name}: inconsistent column count ({min(col_counts)}-{max(col_counts)} columns)"}

        if col_counts[0] < 2:
            return {"valid": False, "error": f"Invalid {name}: only {col_counts[0]} column(s), expected at least 2"}

        return {"valid": True, "error": None}
    except csv.Error as e:
        return {"valid": False, "error": f"Invalid {name}: {str(e)}"}


def _validate_yaml(text: str) -> dict:
    """Validate YAML structure."""
    try:
        import yaml
        result = yaml.safe_load(text)
        if result is None:
            return {"valid": False, "error": "Invalid YAML: empty document"}
        return {"valid": True, "error": None}
    except ImportError:
        return {"valid": False, "error": "YAML validation unavailable (pyyaml not installed)"}
    except Exception as e:
        return {"valid": False, "error": f"Invalid YAML: {str(e)}"}


def _validate_html(text: str) -> dict:
    """Validate HTML structure — checks for basic well-formedness."""
    # Check for at least one HTML tag
    if not re.search(r'<[a-zA-Z][^>]*>', text):
        return {"valid": False, "error": "Invalid HTML: no HTML tags found"}

    # Check for basic tag structure
    open_tags = re.findall(r'<([a-zA-Z][a-zA-Z0-9]*)\b[^>]*/?>|<([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>', text)
    close_tags = re.findall(r'</([a-zA-Z][a-zA-Z0-9]*)>', text)

    if not open_tags:
        return {"valid": False, "error": "Invalid HTML: no opening tags found"}

    # Self-closing tags that don't need closing
    void_elements = {'br', 'hr', 'img', 'input', 'meta', 'link', 'area', 'base', 'col', 'embed', 'source', 'track', 'wbr'}

    # Try parsing with xml.etree (works for well-formed XHTML)
    # Wrap in a root element to handle fragments
    try:
        ET.fromstring(f"<root>{text}</root>")
        return {"valid": True, "error": None}
    except ET.ParseError:
        pass

    # Fallback: if we have tags, consider it valid HTML (browsers are lenient)
    if len(open_tags) >= 1:
        return {"valid": True, "error": None}

    return {"valid": False, "error": "Invalid HTML: malformed structure"}


def _validate_xml(text: str) -> dict:
    """Validate XML structure."""
    try:
        ET.fromstring(text)
        return {"valid": True, "error": None}
    except ET.ParseError as e:
        return {"valid": False, "error": f"Invalid XML: {str(e)}"}


def _validate_markdown(text: str) -> dict:
    """Validate markdown structure — checks for markdown elements."""
    md_patterns = [
        (r'^#{1,6}\s+\S', 'headers'),           # # Header
        (r'^\s*[-*+]\s+\S', 'unordered lists'),  # - list item
        (r'^\s*\d+\.\s+\S', 'ordered lists'),    # 1. list item
        (r'```', 'code blocks'),                  # ```code```
        (r'\|.+\|', 'tables'),                    # | col | col |
        (r'\*\*.+\*\*', 'bold'),                  # **bold**
        (r'_.+_', 'italic'),                      # _italic_
        (r'\[.+\]\(.+\)', 'links'),               # [text](url)
        (r'^>\s+\S', 'blockquotes'),              # > quote
    ]

    found_elements = []
    for pattern, name in md_patterns:
        if re.search(pattern, text, re.MULTILINE):
            found_elements.append(name)

    if not found_elements:
        return {"valid": False, "error": "Invalid Markdown: no markdown elements found (no headers, lists, code blocks, tables, etc.)"}

    return {"valid": True, "error": None}
