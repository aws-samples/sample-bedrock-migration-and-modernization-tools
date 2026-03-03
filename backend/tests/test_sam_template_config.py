"""Tests for SAM template configuration (Phase 4 - Task 16).

Tests verify that the SAM template has proper configuration parameters
and environment variables for Lambda functions.
"""

import subprocess
import pytest
import yaml
from pathlib import Path

SAM_TEMPLATE_PATH = (
    Path(__file__).parent.parent.parent / "infra" / "backend-template.yaml"
)


class CloudFormationLoader(yaml.SafeLoader):
    """Custom YAML loader that handles CloudFormation intrinsic functions."""

    pass


def _cfn_intrinsic_constructor(loader, tag_suffix, node):
    """Handle CloudFormation intrinsic functions like !Ref, !Sub, !If, etc."""
    if isinstance(node, yaml.ScalarNode):
        return {tag_suffix: loader.construct_scalar(node)}
    elif isinstance(node, yaml.SequenceNode):
        return {tag_suffix: loader.construct_sequence(node)}
    elif isinstance(node, yaml.MappingNode):
        return {tag_suffix: loader.construct_mapping(node)}
    return {tag_suffix: None}


# Add multi-constructor for all CloudFormation tags
CloudFormationLoader.add_multi_constructor("!", _cfn_intrinsic_constructor)


def load_template():
    """Load and parse the SAM template with CloudFormation intrinsic function support."""
    with open(SAM_TEMPLATE_PATH) as f:
        return yaml.load(f, Loader=CloudFormationLoader)


class TestSAMParameters:
    """Tests for SAM template parameters."""

    def test_sam_has_log_level_parameter(self):
        """Template should have LogLevel parameter with INFO default."""
        # Arrange
        template = load_template()

        # Act
        params = template.get("Parameters", {})

        # Assert
        assert "LogLevel" in params, "LogLevel parameter not found"
        assert params["LogLevel"]["Default"] == "INFO", (
            "LogLevel default should be INFO"
        )
        assert "AllowedValues" in params["LogLevel"], (
            "LogLevel should have AllowedValues"
        )

    def test_sam_has_availability_workers_parameter(self):
        """Template should have AvailabilityMaxWorkers parameter."""
        # Arrange
        template = load_template()

        # Act
        params = template.get("Parameters", {})

        # Assert
        assert "AvailabilityMaxWorkers" in params, (
            "AvailabilityMaxWorkers parameter not found"
        )
        assert params["AvailabilityMaxWorkers"]["Type"] == "Number", (
            "AvailabilityMaxWorkers should be Number type"
        )
        assert "Default" in params["AvailabilityMaxWorkers"], (
            "AvailabilityMaxWorkers should have default value"
        )

    def test_sam_has_quota_batch_size_parameter(self):
        """Template should have QuotaBatchSize parameter."""
        # Arrange
        template = load_template()

        # Act
        params = template.get("Parameters", {})

        # Assert
        assert "QuotaBatchSize" in params, "QuotaBatchSize parameter not found"
        assert params["QuotaBatchSize"]["Type"] == "Number", (
            "QuotaBatchSize should be Number type"
        )
        assert "Default" in params["QuotaBatchSize"], (
            "QuotaBatchSize should have default value"
        )


class TestSAMGlobals:
    """Tests for SAM template Globals configuration."""

    def test_sam_globals_has_powertools_env_vars(self):
        """Globals should have Powertools environment variables."""
        # Arrange
        template = load_template()

        # Act
        env_vars = (
            template.get("Globals", {})
            .get("Function", {})
            .get("Environment", {})
            .get("Variables", {})
        )

        # Assert
        assert "POWERTOOLS_SERVICE_NAME" in env_vars, (
            "POWERTOOLS_SERVICE_NAME not in Globals"
        )
        assert "POWERTOOLS_METRICS_NAMESPACE" in env_vars, (
            "POWERTOOLS_METRICS_NAMESPACE not in Globals"
        )
        assert "POWERTOOLS_LOG_LEVEL" in env_vars, "POWERTOOLS_LOG_LEVEL not in Globals"


class TestSAMFunctionConfig:
    """Tests for individual Lambda function configurations."""

    def test_sam_quota_collector_has_batch_size_env(self):
        """QuotaCollectorFunction should have QUOTA_BATCH_SIZE env var."""
        # Arrange
        template = load_template()

        # Act
        resources = template.get("Resources", {})
        quota_func = resources.get("QuotaCollectorFunction", {})
        env_vars = (
            quota_func.get("Properties", {}).get("Environment", {}).get("Variables", {})
        )

        # Assert
        assert "QUOTA_BATCH_SIZE" in env_vars, (
            "QUOTA_BATCH_SIZE not in QuotaCollectorFunction"
        )

    def test_sam_regional_availability_has_workers_env(self):
        """RegionalAvailabilityFunction should have AVAILABILITY_MAX_WORKERS env var."""
        # Arrange
        template = load_template()

        # Act
        resources = template.get("Resources", {})
        avail_func = resources.get("RegionalAvailabilityFunction", {})
        env_vars = (
            avail_func.get("Properties", {}).get("Environment", {}).get("Variables", {})
        )

        # Assert
        assert "AVAILABILITY_MAX_WORKERS" in env_vars, (
            "AVAILABILITY_MAX_WORKERS not in RegionalAvailabilityFunction"
        )


class TestSAMBuild:
    """Tests for SAM template build validation."""

    @pytest.mark.slow
    def test_sam_build_succeeds(self):
        """SAM build should succeed with the template."""
        # Arrange - ensure template exists
        assert SAM_TEMPLATE_PATH.exists(), f"Template not found: {SAM_TEMPLATE_PATH}"

        # Act
        result = subprocess.run(
            ["sam", "build", "-t", str(SAM_TEMPLATE_PATH)],
            cwd=SAM_TEMPLATE_PATH.parent,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        # Assert
        assert result.returncode == 0, f"SAM build failed: {result.stderr}"
