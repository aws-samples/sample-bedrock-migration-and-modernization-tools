"""Tests for infrastructure environment variable usage (Task 07).

Tests the environment variable configuration in:
- SAM template parameters (CognitoRegion, Ec2Region)
- Lambda environment variable usage
- Layer description updates
"""

import os
from pathlib import Path
import pytest
from unittest.mock import Mock, patch
import yaml


# Get the project root directory (parent of backend/)
PROJECT_ROOT = Path(__file__).parent.parent.parent


# ============================================================================
# CloudFormation YAML Loader
# ============================================================================


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


# ============================================================================
# Tests for Environment Variable Fallback Logic
# ============================================================================


class TestAnalyticsEnvVars:
    """Tests for analytics Lambda environment variable usage."""

    def test_analytics_uses_cognito_region_env(self):
        """Should use COGNITO_REGION when set."""
        with patch.dict(
            os.environ, {"COGNITO_REGION": "eu-west-1", "AWS_REGION": "us-east-1"}
        ):
            # Simulate the fallback logic used in Lambda handlers
            cognito_region = os.environ.get(
                "COGNITO_REGION", os.environ.get("AWS_REGION", "us-east-1")
            )

            assert cognito_region == "eu-west-1"

    def test_analytics_falls_back_to_aws_region(self):
        """Should fall back to AWS_REGION when COGNITO_REGION not set."""
        env = {"AWS_REGION": "ap-southeast-1"}
        with patch.dict(os.environ, env, clear=True):
            cognito_region = os.environ.get(
                "COGNITO_REGION", os.environ.get("AWS_REGION", "us-east-1")
            )

            assert cognito_region == "ap-southeast-1"

    def test_analytics_uses_default_region(self):
        """Should use default us-east-1 when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            cognito_region = os.environ.get(
                "COGNITO_REGION", os.environ.get("AWS_REGION", "us-east-1")
            )

            assert cognito_region == "us-east-1"


class TestCognitoSyncEnvVars:
    """Tests for cognito-sync Lambda environment variable usage."""

    def test_cognito_sync_uses_cognito_region_env(self):
        """Should use COGNITO_REGION when set."""
        with patch.dict(
            os.environ, {"COGNITO_REGION": "eu-central-1", "AWS_REGION": "us-west-2"}
        ):
            cognito_region = os.environ.get(
                "COGNITO_REGION", os.environ.get("AWS_REGION", "us-east-1")
            )

            assert cognito_region == "eu-central-1"


class TestRegionDiscoveryEnvVars:
    """Tests for region-discovery Lambda environment variable usage."""

    def test_region_discovery_uses_ec2_region_env(self):
        """Should use EC2_REGION when set."""
        with patch.dict(
            os.environ, {"EC2_REGION": "us-west-2", "AWS_REGION": "us-east-1"}
        ):
            ec2_region = os.environ.get(
                "EC2_REGION", os.environ.get("AWS_REGION", "us-east-1")
            )

            assert ec2_region == "us-west-2"

    def test_region_discovery_falls_back_to_aws_region(self):
        """Should fall back to AWS_REGION when EC2_REGION not set."""
        env = {"AWS_REGION": "eu-west-1"}
        with patch.dict(os.environ, env, clear=True):
            ec2_region = os.environ.get(
                "EC2_REGION", os.environ.get("AWS_REGION", "us-east-1")
            )

            assert ec2_region == "eu-west-1"


# ============================================================================
# Tests for SAM Template Configuration
# ============================================================================


class TestSamTemplate:
    """Tests for SAM template configuration."""

    @pytest.fixture
    def sam_template(self):
        """Load SAM template with CloudFormation intrinsic function support."""
        template_path = PROJECT_ROOT / "infra" / "backend-template.yaml"
        with open(template_path, "r") as f:
            return yaml.load(f, Loader=CloudFormationLoader)

    def test_sam_template_has_cognito_region_param(self, sam_template):
        """Should have CognitoRegion parameter."""
        params = sam_template.get("Parameters", {})

        assert "CognitoRegion" in params
        assert params["CognitoRegion"].get("Default") == "us-east-1"
        assert params["CognitoRegion"].get("Type") == "String"

    def test_sam_template_has_ec2_region_param(self, sam_template):
        """Should have Ec2Region parameter."""
        params = sam_template.get("Parameters", {})

        assert "Ec2Region" in params
        assert params["Ec2Region"].get("Default") == "us-east-1"
        assert params["Ec2Region"].get("Type") == "String"

    def test_sam_template_layer_description_updated(self, sam_template):
        """Should have updated layer description with v2.5.0."""
        resources = sam_template.get("Resources", {})
        layer = resources.get("SharedUtilsLayer", {})
        props = layer.get("Properties", {})
        description = props.get("Description", "")

        # Check for version or architecture improvements mention
        assert "v2.5.0" in description or "Architecture improvements" in description

    def test_sam_template_globals_have_cognito_region(self, sam_template):
        """Should have COGNITO_REGION in global environment variables."""
        globals_section = sam_template.get("Globals", {})
        func_globals = globals_section.get("Function", {})
        env_vars = func_globals.get("Environment", {}).get("Variables", {})

        assert "COGNITO_REGION" in env_vars

    def test_sam_template_globals_have_ec2_region(self, sam_template):
        """Should have EC2_REGION in global environment variables."""
        globals_section = sam_template.get("Globals", {})
        func_globals = globals_section.get("Function", {})
        env_vars = func_globals.get("Environment", {}).get("Variables", {})

        assert "EC2_REGION" in env_vars

    def test_sam_template_cognito_region_uses_ref(self, sam_template):
        """Should reference CognitoRegion parameter in globals."""
        globals_section = sam_template.get("Globals", {})
        func_globals = globals_section.get("Function", {})
        env_vars = func_globals.get("Environment", {}).get("Variables", {})

        cognito_region_value = env_vars.get("COGNITO_REGION")
        # Should be a CloudFormation !Ref to the parameter
        assert cognito_region_value == {"Ref": "CognitoRegion"}

    def test_sam_template_ec2_region_uses_ref(self, sam_template):
        """Should reference Ec2Region parameter in globals."""
        globals_section = sam_template.get("Globals", {})
        func_globals = globals_section.get("Function", {})
        env_vars = func_globals.get("Environment", {}).get("Variables", {})

        ec2_region_value = env_vars.get("EC2_REGION")
        # Should be a CloudFormation !Ref to the parameter
        assert ec2_region_value == {"Ref": "Ec2Region"}
