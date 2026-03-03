"""
Tests for quota-collector naming convention (Task 13).

Validates that quota-collector outputs use snake_case field names
instead of camelCase.
"""

import pytest
from pathlib import Path

HANDLER_PATH = (
    Path(__file__).parent.parent / "lambdas" / "quota-collector" / "handler.py"
)


def get_handler_source() -> str:
    """Read the handler source code."""
    return HANDLER_PATH.read_text()


class TestSnakeCaseFieldNames:
    """Tests that quota output uses snake_case field names."""

    def test_quota_code_field_name(self):
        """Quota output should use 'quota_code' (snake_case)."""
        # Arrange
        source = get_handler_source()

        # Assert - check for both single and double quotes
        assert '"quota_code"' in source or "'quota_code'" in source, (
            "quota-collector should use 'quota_code' field name (snake_case)"
        )

    def test_quota_name_field_name(self):
        """Quota output should use 'quota_name' (snake_case)."""
        # Arrange
        source = get_handler_source()

        # Assert - check for both single and double quotes
        assert '"quota_name"' in source or "'quota_name'" in source, (
            "quota-collector should use 'quota_name' field name (snake_case)"
        )

    def test_quota_arn_field_name(self):
        """Quota output should use 'quota_arn' (snake_case)."""
        # Arrange
        source = get_handler_source()

        # Assert - check for both single and double quotes
        assert '"quota_arn"' in source or "'quota_arn'" in source, (
            "quota-collector should use 'quota_arn' field name (snake_case)"
        )

    def test_global_quota_field_name(self):
        """Quota output should use 'global_quota' (snake_case)."""
        # Arrange
        source = get_handler_source()

        # Assert - check for both single and double quotes
        assert '"global_quota"' in source or "'global_quota'" in source, (
            "quota-collector should use 'global_quota' field name (snake_case)"
        )

    def test_usage_metric_field_name(self):
        """Quota output should use 'usage_metric' (snake_case)."""
        # Arrange
        source = get_handler_source()

        # Assert - check for both single and double quotes
        assert '"usage_metric"' in source or "'usage_metric'" in source, (
            "quota-collector should use 'usage_metric' field name (snake_case)"
        )


class TestNoCamelCaseFieldNames:
    """Tests that quota output does NOT use camelCase field names."""

    def test_no_camel_case_quota_code(self):
        """Quota output should NOT use 'quotaCode' (camelCase)."""
        # Arrange
        source = get_handler_source()

        # Assert
        assert "'quotaCode'" not in source, (
            "quota-collector should NOT use 'quotaCode' (camelCase). "
            "Use 'quota_code' instead."
        )

    def test_no_camel_case_quota_name(self):
        """Quota output should NOT use 'quotaName' (camelCase)."""
        # Arrange
        source = get_handler_source()

        # Assert
        assert "'quotaName'" not in source, (
            "quota-collector should NOT use 'quotaName' (camelCase). "
            "Use 'quota_name' instead."
        )

    def test_no_camel_case_quota_arn(self):
        """Quota output should NOT use 'quotaArn' (camelCase)."""
        # Arrange
        source = get_handler_source()

        # Assert
        assert "'quotaArn'" not in source, (
            "quota-collector should NOT use 'quotaArn' (camelCase). "
            "Use 'quota_arn' instead."
        )

    def test_no_camel_case_global_quota(self):
        """Quota output should NOT use 'globalQuota' (camelCase)."""
        # Arrange
        source = get_handler_source()

        # Assert
        assert "'globalQuota'" not in source, (
            "quota-collector should NOT use 'globalQuota' (camelCase). "
            "Use 'global_quota' instead."
        )

    def test_no_camel_case_usage_metric(self):
        """Quota output should NOT use 'usageMetric' (camelCase)."""
        # Arrange
        source = get_handler_source()

        # Assert
        assert "'usageMetric'" not in source, (
            "quota-collector should NOT use 'usageMetric' (camelCase). "
            "Use 'usage_metric' instead."
        )


class TestQuotaOutputStructure:
    """Tests for the overall quota output structure."""

    def test_quota_output_uses_snake_case_consistently(self):
        """All quota output fields should use snake_case consistently."""
        # Arrange
        source = get_handler_source()

        # Assert - all expected snake_case fields are present (check both quote styles)
        expected_fields = [
            "quota_code",
            "quota_name",
            "quota_arn",
            "global_quota",
            "usage_metric",
        ]

        for field in expected_fields:
            assert f'"{field}"' in source or f"'{field}'" in source, (
                f"Expected {field} in quota output"
            )

    def test_normalized_dict_structure(self):
        """Quota collector should create a normalized dict with snake_case keys."""
        # Arrange
        source = get_handler_source()

        # Assert - the normalized dict assignment pattern should exist
        assert "normalized = {" in source, (
            "quota-collector should create a 'normalized' dict for quota output"
        )
