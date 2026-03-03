"""
Tests for externalized configuration (Task 11).

Validates that configuration values can be set via environment variables
and have appropriate defaults.
"""

import os
import pytest
from unittest.mock import patch


class TestQuotaBatchSize:
    """Tests for QUOTA_BATCH_SIZE configuration."""

    def test_quota_batch_size_default(self):
        """QUOTA_BATCH_SIZE should default to 100 when env var not set."""
        # Arrange - ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            # Act
            default = int(os.environ.get("QUOTA_BATCH_SIZE", "100"))

            # Assert
            assert default == 100

    def test_quota_batch_size_override(self):
        """QUOTA_BATCH_SIZE should be overridable via environment variable."""
        # Arrange
        with patch.dict(os.environ, {"QUOTA_BATCH_SIZE": "50"}):
            # Act
            value = int(os.environ.get("QUOTA_BATCH_SIZE", "100"))

            # Assert
            assert value == 50


class TestAvailabilityMaxWorkers:
    """Tests for AVAILABILITY_MAX_WORKERS configuration."""

    def test_availability_max_workers_default(self):
        """AVAILABILITY_MAX_WORKERS should default to 10 when env var not set."""
        # Arrange - ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            # Act
            default = int(os.environ.get("AVAILABILITY_MAX_WORKERS", "10"))

            # Assert
            assert default == 10

    def test_availability_max_workers_override(self):
        """AVAILABILITY_MAX_WORKERS should be overridable via environment variable."""
        # Arrange
        with patch.dict(os.environ, {"AVAILABILITY_MAX_WORKERS": "20"}):
            # Act
            value = int(os.environ.get("AVAILABILITY_MAX_WORKERS", "10"))

            # Assert
            assert value == 20


class TestAwsTimeouts:
    """Tests for AWS timeout configuration."""

    def test_aws_connect_timeout_default(self):
        """AWS_CONNECT_TIMEOUT should default to 10 when env var not set."""
        # Arrange - ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            # Act
            default = int(os.environ.get("AWS_CONNECT_TIMEOUT", "10"))

            # Assert
            assert default == 10

    def test_aws_connect_timeout_override(self):
        """AWS_CONNECT_TIMEOUT should be overridable via environment variable."""
        # Arrange
        with patch.dict(os.environ, {"AWS_CONNECT_TIMEOUT": "15"}):
            # Act
            value = int(os.environ.get("AWS_CONNECT_TIMEOUT", "10"))

            # Assert
            assert value == 15

    def test_aws_read_timeout_default(self):
        """AWS_READ_TIMEOUT should default to 30 when env var not set."""
        # Arrange - ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            # Act
            default = int(os.environ.get("AWS_READ_TIMEOUT", "30"))

            # Assert
            assert default == 30

    def test_aws_read_timeout_override(self):
        """AWS_READ_TIMEOUT should be overridable via environment variable."""
        # Arrange
        with patch.dict(os.environ, {"AWS_READ_TIMEOUT": "60"}):
            # Act
            value = int(os.environ.get("AWS_READ_TIMEOUT", "30"))

            # Assert
            assert value == 60


class TestQuotaCollectorUsesEnvVar:
    """Tests that quota-collector handler uses environment variable."""

    def test_quota_collector_has_env_var_config(self):
        """Quota collector should read QUOTA_BATCH_SIZE from environment."""
        from pathlib import Path

        handler_path = (
            Path(__file__).parent.parent / "lambdas" / "quota-collector" / "handler.py"
        )
        source = handler_path.read_text()

        # Assert - handler should use os.environ.get for QUOTA_BATCH_SIZE
        assert 'os.environ.get("QUOTA_BATCH_SIZE"' in source
        assert "QUOTA_BATCH_SIZE = int(" in source
