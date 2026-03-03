"""Tests for exception hierarchy."""

import sys
from pathlib import Path
import importlib.util
import pytest

# Load the exceptions module directly from file (bypasses __init__.py and powertools dependency)
EXCEPTIONS_MODULE_PATH = (
    Path(__file__).parent.parent
    / "layers"
    / "common"
    / "python"
    / "shared"
    / "exceptions.py"
)
spec = importlib.util.spec_from_file_location(
    "shared_exceptions", EXCEPTIONS_MODULE_PATH
)
exceptions_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exceptions_module)


def test_profiler_error_base():
    """ProfilerError should have retryable attribute."""
    e = exceptions_module.ProfilerError("test")
    assert hasattr(e, "retryable")
    assert hasattr(e, "error_code")
    assert hasattr(e, "context")


def test_validation_error_not_retryable():
    """ValidationError should not be retryable."""
    e = exceptions_module.ValidationError("invalid input", field="test")
    assert e.retryable is False
    assert e.error_code == "VALIDATION_ERROR"


def test_s3_read_error_retryable():
    """S3ReadError should be retryable."""
    e = exceptions_module.S3ReadError("bucket", "key", "Not found")
    assert e.retryable is True
    assert e.error_code == "S3_READ_ERROR"


def test_s3_write_error_retryable():
    """S3WriteError should be retryable."""
    e = exceptions_module.S3WriteError("bucket", "key", "Access denied")
    assert e.retryable is True


def test_throttling_error_retryable():
    """ThrottlingError should always be retryable."""
    e = exceptions_module.ThrottlingError(
        "bedrock", "ListFoundationModels", "us-east-1"
    )
    assert e.retryable is True
    assert e.error_code == "THROTTLING_ERROR"


def test_configuration_error_not_retryable():
    """ConfigurationError should not be retryable."""
    e = exceptions_module.ConfigurationError("Missing config", config_key="test")
    assert e.retryable is False


def test_exception_to_dict():
    """to_dict() should return structured dictionary."""
    e = exceptions_module.S3ReadError("bucket", "key", "Not found")
    d = e.to_dict()
    assert d["error_code"] == "S3_READ_ERROR"
    assert d["retryable"] is True
    assert "bucket" in d["context"]
    assert "key" in d["context"]


def test_exception_str_with_context():
    """__str__ should include context."""
    e = exceptions_module.S3ReadError("bucket", "key", "Not found")
    s = str(e)
    assert "bucket" in s or "context" in s


def test_exception_inheritance():
    """S3ReadError should inherit from ProfilerError."""
    e = exceptions_module.S3ReadError("bucket", "key")
    assert isinstance(e, exceptions_module.ProfilerError)
    assert isinstance(e, Exception)
