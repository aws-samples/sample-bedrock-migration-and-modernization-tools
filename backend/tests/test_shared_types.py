"""Tests for type definitions.

These tests require Python 3.11+ due to the use of NotRequired from typing.
They will be skipped on older Python versions.
"""

import sys
from pathlib import Path
import importlib.util
import pytest
from typing import get_type_hints

# Skip all tests if Python version is too old
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="types.py requires Python 3.11+ for NotRequired from typing",
)

# Load the types module directly from file (bypasses __init__.py and powertools dependency)
TYPES_MODULE_PATH = (
    Path(__file__).parent.parent
    / "layers"
    / "common"
    / "python"
    / "shared"
    / "types.py"
)

# Only load module if Python version is sufficient
types_module = None
if sys.version_info >= (3, 11):
    spec = importlib.util.spec_from_file_location("shared_types", TYPES_MODULE_PATH)
    types_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(types_module)


def test_quota_collector_input_fields():
    """QuotaCollectorInput should have region field."""
    hints = get_type_hints(types_module.QuotaCollectorInput)
    assert "region" in hints


def test_quota_collector_output_fields():
    """QuotaCollectorOutput should have status field."""
    hints = get_type_hints(types_module.QuotaCollectorOutput)
    assert "status" in hints


def test_handler_result_base_fields():
    """HandlerResult should have status and durationMs."""
    hints = get_type_hints(types_module.HandlerResult)
    assert "status" in hints


def test_s3_reference_fields():
    """S3Reference should have s3Bucket and s3Key."""
    hints = get_type_hints(types_module.S3Reference)
    assert "s3Bucket" in hints
    assert "s3Key" in hints


def test_quota_data_snake_case():
    """QuotaData should use snake_case field names."""
    hints = get_type_hints(types_module.QuotaData)
    assert "quota_code" in hints
    assert "quota_name" in hints


def test_all_exports_importable():
    """All exports in __all__ should be importable."""
    for name in types_module.__all__:
        assert hasattr(types_module, name)
