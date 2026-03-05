"""
Tests for Phase 1 Task 03: Dimension Detection Constants

This module tests the dimension detection constants and patterns that were added
to the pricing-aggregator handler for improved pricing dimension classification.

Tests validate:
- INFERENCE_MODES contains all 6 modes
- GEOGRAPHIC_SCOPES contains all 3 scopes
- CACHE_TYPES contains all 4 types including cache_write_1h
- COMMITMENT_TERMS contains all 5 terms
- Detection patterns correctly match expected strings
"""

import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest  # noqa: F401 - used by pytest test discovery


# Mock aws_lambda_powertools before importing handler (must be done before pytest import)
mock_powertools = MagicMock()
mock_powertools.Logger = MagicMock(return_value=MagicMock())
mock_powertools.Tracer = MagicMock(return_value=MagicMock())
mock_powertools.Metrics = MagicMock(return_value=MagicMock())

mock_utilities = MagicMock()
mock_utilities.typing = MagicMock()
mock_utilities.typing.LambdaContext = MagicMock()

mock_metrics = MagicMock()
mock_metrics.MetricUnit = MagicMock()
mock_metrics.MetricUnit.Count = "Count"
mock_metrics.MetricUnit.Milliseconds = "Milliseconds"

sys.modules["aws_lambda_powertools"] = mock_powertools
sys.modules["aws_lambda_powertools.utilities"] = mock_utilities
sys.modules["aws_lambda_powertools.utilities.typing"] = mock_utilities.typing
sys.modules["aws_lambda_powertools.metrics"] = mock_metrics

# Mock shared module
mock_shared = MagicMock()
mock_shared.get_s3_client = MagicMock()
mock_shared.read_from_s3 = MagicMock()
mock_shared.write_to_s3 = MagicMock()
mock_shared.parse_execution_id = MagicMock()
mock_shared.validate_required_params = MagicMock()
mock_shared.ValidationError = Exception
mock_shared.S3ReadError = Exception
mock_shared.get_config_loader = MagicMock()

sys.modules["shared"] = mock_shared

# Mock shared.powertools
mock_shared_powertools = MagicMock()
mock_shared_powertools.logger = MagicMock()
mock_shared_powertools.tracer = MagicMock()
mock_shared_powertools.metrics = MagicMock()
mock_shared_powertools.LambdaContext = MagicMock()

sys.modules["shared.powertools"] = mock_shared_powertools

# Now load the handler module
HANDLER_PATH = (
    Path(__file__).parent.parent / "lambdas" / "pricing-aggregator" / "handler.py"
)

spec = importlib.util.spec_from_file_location("handler", HANDLER_PATH)
handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler)

# Import constants under test
INFERENCE_MODES = handler.INFERENCE_MODES
GEOGRAPHIC_SCOPES = handler.GEOGRAPHIC_SCOPES
CACHE_TYPES = handler.CACHE_TYPES
COMMITMENT_TERMS = handler.COMMITMENT_TERMS
MANTLE_PATTERNS = handler.MANTLE_PATTERNS
CRIS_REGIONAL_PATTERNS = handler.CRIS_REGIONAL_PATTERNS
RESERVED_PATTERNS = handler.RESERVED_PATTERNS
CACHE_PATTERNS = handler.CACHE_PATTERNS


class TestDimensionConstants:
    """Tests for dimension detection constants."""

    def test_inference_modes_complete(self):
        """INFERENCE_MODES should contain all 6 modes."""
        # Arrange
        expected_keys = {
            "ON_DEMAND",
            "BATCH",
            "PROVISIONED",
            "RESERVED",
            "MANTLE",
            "CUSTOM_MODEL",
        }

        # Act
        actual_keys = set(INFERENCE_MODES.keys())

        # Assert
        assert actual_keys == expected_keys, (
            f"Expected keys {expected_keys}, got {actual_keys}"
        )

    def test_inference_modes_values(self):
        """INFERENCE_MODES values should be lowercase snake_case."""
        # Arrange & Act & Assert
        assert INFERENCE_MODES["ON_DEMAND"] == "on_demand"
        assert INFERENCE_MODES["BATCH"] == "batch"
        assert INFERENCE_MODES["PROVISIONED"] == "provisioned"
        assert INFERENCE_MODES["RESERVED"] == "reserved"
        assert INFERENCE_MODES["MANTLE"] == "mantle"
        assert INFERENCE_MODES["CUSTOM_MODEL"] == "custom_model"

    def test_geographic_scopes_complete(self):
        """GEOGRAPHIC_SCOPES should contain all 3 scopes."""
        # Arrange
        expected_keys = {"IN_REGION", "CRIS_GLOBAL", "CRIS_REGIONAL"}

        # Act
        actual_keys = set(GEOGRAPHIC_SCOPES.keys())

        # Assert
        assert actual_keys == expected_keys, (
            f"Expected keys {expected_keys}, got {actual_keys}"
        )

    def test_geographic_scopes_values(self):
        """GEOGRAPHIC_SCOPES values should be lowercase snake_case."""
        # Arrange & Act & Assert
        assert GEOGRAPHIC_SCOPES["IN_REGION"] == "in_region"
        assert GEOGRAPHIC_SCOPES["CRIS_GLOBAL"] == "cris_global"
        assert GEOGRAPHIC_SCOPES["CRIS_REGIONAL"] == "cris_regional"

    def test_cache_types_complete(self):
        """CACHE_TYPES should contain all 4 types including cache_write_1h."""
        # Arrange
        expected_keys = {"NONE", "CACHE_READ", "CACHE_WRITE", "CACHE_WRITE_1H"}

        # Act
        actual_keys = set(CACHE_TYPES.keys())

        # Assert
        assert actual_keys == expected_keys, (
            f"Expected keys {expected_keys}, got {actual_keys}"
        )

    def test_cache_types_values(self):
        """CACHE_TYPES values should be correct."""
        # Arrange & Act & Assert
        assert CACHE_TYPES["NONE"] is None
        assert CACHE_TYPES["CACHE_READ"] == "cache_read"
        assert CACHE_TYPES["CACHE_WRITE"] == "cache_write"
        assert CACHE_TYPES["CACHE_WRITE_1H"] == "cache_write_1h"

    def test_commitment_terms_complete(self):
        """COMMITMENT_TERMS should contain all 5 terms."""
        # Arrange
        expected_keys = {"NONE", "NO_COMMIT", "1_MONTH", "3_MONTH", "6_MONTH"}

        # Act
        actual_keys = set(COMMITMENT_TERMS.keys())

        # Assert
        assert actual_keys == expected_keys, (
            f"Expected keys {expected_keys}, got {actual_keys}"
        )

    def test_commitment_terms_values(self):
        """COMMITMENT_TERMS values should be correct."""
        # Arrange & Act & Assert
        assert COMMITMENT_TERMS["NONE"] is None
        assert COMMITMENT_TERMS["NO_COMMIT"] == "no_commit"
        assert COMMITMENT_TERMS["1_MONTH"] == "1_month"
        assert COMMITMENT_TERMS["3_MONTH"] == "3_month"
        assert COMMITMENT_TERMS["6_MONTH"] == "6_month"


class TestDetectionPatterns:
    """Tests for dimension detection patterns."""

    def test_mantle_pattern_matches_mantle(self):
        """MANTLE_PATTERNS should match 'mantle-pricing'."""
        # Arrange
        test_string = "mantle-pricing"

        # Act
        matched = any(p.search(test_string) for p in MANTLE_PATTERNS)

        # Assert
        assert matched, f"MANTLE_PATTERNS should match '{test_string}'"

    def test_mantle_pattern_matches_mantle_case_insensitive(self):
        """MANTLE_PATTERNS should match 'Mantle' (case insensitive)."""
        # Arrange
        test_string = "Mantle"

        # Act
        matched = any(p.search(test_string) for p in MANTLE_PATTERNS)

        # Assert
        assert matched, f"MANTLE_PATTERNS should match '{test_string}'"

    def test_mantle_pattern_matches_openai_compatible(self):
        """MANTLE_PATTERNS should match 'openai-compatible'."""
        # Arrange
        test_string = "openai-compatible"

        # Act
        matched = any(p.search(test_string) for p in MANTLE_PATTERNS)

        # Assert
        assert matched, f"MANTLE_PATTERNS should match '{test_string}'"

    def test_cris_regional_pattern_matches_regional_cris(self):
        """CRIS_REGIONAL_PATTERNS should match 'Regional CRIS'."""
        # Arrange
        test_string = "Regional CRIS"

        # Act
        matched = any(p.search(test_string) for p in CRIS_REGIONAL_PATTERNS)

        # Assert
        assert matched, f"CRIS_REGIONAL_PATTERNS should match '{test_string}'"

    def test_cris_regional_pattern_matches_geo_suffix(self):
        """CRIS_REGIONAL_PATTERNS should match '_Geo'."""
        # Arrange
        test_string = "USE1_InputTokenCount_Geo"

        # Act
        matched = any(p.search(test_string) for p in CRIS_REGIONAL_PATTERNS)

        # Assert
        assert matched, f"CRIS_REGIONAL_PATTERNS should match '{test_string}'"

    def test_cris_regional_pattern_matches_regional(self):
        """CRIS_REGIONAL_PATTERNS should match 'regional pricing'."""
        # Arrange
        test_string = "regional pricing"

        # Act
        matched = any(p.search(test_string) for p in CRIS_REGIONAL_PATTERNS)

        # Assert
        assert matched, f"CRIS_REGIONAL_PATTERNS should match '{test_string}'"

    def test_reserved_pattern_matches_reserved_1month(self):
        """RESERVED_PATTERNS should match 'Reserved_1Month'."""
        # Arrange
        test_string = "Reserved_1Month"

        # Act
        matched = any(p.search(test_string) for p in RESERVED_PATTERNS)

        # Assert
        assert matched, f"RESERVED_PATTERNS should match '{test_string}'"

    def test_reserved_pattern_matches_reserved(self):
        """RESERVED_PATTERNS should match 'reserved'."""
        # Arrange
        test_string = "reserved"

        # Act
        matched = any(p.search(test_string) for p in RESERVED_PATTERNS)

        # Assert
        assert matched, f"RESERVED_PATTERNS should match '{test_string}'"

    def test_reserved_pattern_matches_tpm(self):
        """RESERVED_PATTERNS should match '_tpm_'."""
        # Arrange
        test_string = "USE1_tpm_input"

        # Act
        matched = any(p.search(test_string) for p in RESERVED_PATTERNS)

        # Assert
        assert matched, f"RESERVED_PATTERNS should match '{test_string}'"

    def test_reserved_pattern_matches_no_commit(self):
        """RESERVED_PATTERNS should match 'no-commit'."""
        # Arrange
        test_string = "no-commit"

        # Act
        matched = any(p.search(test_string) for p in RESERVED_PATTERNS)

        # Assert
        assert matched, f"RESERVED_PATTERNS should match '{test_string}'"

    def test_cache_write_1h_pattern_matches_cache_write_1h(self):
        """CACHE_PATTERNS['cache_write_1h'] should match 'cache-write-1h'."""
        # Arrange
        test_string = "cache-write-1h"
        patterns = CACHE_PATTERNS.get("cache_write_1h", [])

        # Act
        matched = any(p.search(test_string) for p in patterns)

        # Assert
        assert matched, f"CACHE_PATTERNS['cache_write_1h'] should match '{test_string}'"

    def test_cache_write_1h_pattern_matches_underscore_variant(self):
        """CACHE_PATTERNS['cache_write_1h'] should match 'cache_write_1h'."""
        # Arrange
        test_string = "cache_write_1h"
        patterns = CACHE_PATTERNS.get("cache_write_1h", [])

        # Act
        matched = any(p.search(test_string) for p in patterns)

        # Assert
        assert matched, f"CACHE_PATTERNS['cache_write_1h'] should match '{test_string}'"

    def test_cache_write_1h_pattern_matches_1_hour_cache(self):
        """CACHE_PATTERNS['cache_write_1h'] should match '1-hour-cache'."""
        # Arrange
        test_string = "1-hour-cache"
        patterns = CACHE_PATTERNS.get("cache_write_1h", [])

        # Act
        matched = any(p.search(test_string) for p in patterns)

        # Assert
        assert matched, f"CACHE_PATTERNS['cache_write_1h'] should match '{test_string}'"

    def test_cache_read_pattern_exists(self):
        """CACHE_PATTERNS should have 'cache_read' patterns."""
        # Arrange & Act
        patterns = CACHE_PATTERNS.get("cache_read", [])

        # Assert
        assert len(patterns) > 0, "CACHE_PATTERNS should have 'cache_read' patterns"

    def test_cache_write_pattern_exists(self):
        """CACHE_PATTERNS should have 'cache_write' patterns."""
        # Arrange & Act
        patterns = CACHE_PATTERNS.get("cache_write", [])

        # Assert
        assert len(patterns) > 0, "CACHE_PATTERNS should have 'cache_write' patterns"

    def test_cache_write_pattern_does_not_match_1h(self):
        """CACHE_PATTERNS['cache_write'] should NOT match 'cache-write-1h'."""
        # Arrange
        test_string = "cache-write-1h"
        patterns = CACHE_PATTERNS.get("cache_write", [])

        # Act
        matched = any(p.search(test_string) for p in patterns)

        # Assert
        # The cache_write pattern uses negative lookahead to exclude 1h
        assert not matched, (
            f"CACHE_PATTERNS['cache_write'] should NOT match '{test_string}'"
        )
