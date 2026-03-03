"""
Integration tests for production deployment verification.

These tests verify that the deployment to bedrock-profiler-prod was successful.
Run after deployment to confirm all resources are correctly configured.

Usage:
    STACK_NAME=bedrock-profiler-prod pytest tests/test_deployment.py -v
"""

import os
import pytest
import boto3
from botocore.exceptions import ClientError

# Configuration
STACK_NAME = os.environ.get("STACK_NAME", "bedrock-profiler-prod")
LAYER_NAME = f"bedrock-profiler-shared-utils-{STACK_NAME.split('-')[-1]}"


@pytest.fixture(scope="module")
def cloudformation_client():
    """Create CloudFormation client."""
    return boto3.client("cloudformation")


@pytest.fixture(scope="module")
def lambda_client():
    """Create Lambda client."""
    return boto3.client("lambda")


@pytest.fixture(scope="module")
def s3_client():
    """Create S3 client."""
    return boto3.client("s3")


@pytest.fixture(scope="module")
def stack_outputs(cloudformation_client):
    """Get stack outputs."""
    try:
        response = cloudformation_client.describe_stacks(StackName=STACK_NAME)
        outputs = response["Stacks"][0].get("Outputs", [])
        return {o["OutputKey"]: o["OutputValue"] for o in outputs}
    except ClientError as e:
        pytest.skip(f"Could not describe stack {STACK_NAME}: {e}")


class TestStackDeployment:
    """Tests for stack deployment status."""

    def test_stack_exists_and_complete(self, cloudformation_client):
        """Stack should exist and be in COMPLETE status."""
        try:
            response = cloudformation_client.describe_stacks(StackName=STACK_NAME)
        except ClientError as e:
            pytest.fail(f"Stack {STACK_NAME} not found: {e}")

        assert len(response["Stacks"]) == 1
        status = response["Stacks"][0]["StackStatus"]
        assert status in ["CREATE_COMPLETE", "UPDATE_COMPLETE"], (
            f"Stack status: {status}"
        )

    def test_layer_version_updated(self, lambda_client):
        """Layer should have updated description with v2.5.0."""
        try:
            response = lambda_client.list_layer_versions(LayerName=LAYER_NAME)
        except ClientError as e:
            pytest.skip(f"Layer {LAYER_NAME} not found: {e}")

        assert len(response["LayerVersions"]) > 0, "No layer versions found"
        latest_version = response["LayerVersions"][0]
        description = latest_version.get("Description", "")

        # Check for v2.5.0 or architecture improvements indicator
        assert "v2.5.0" in description or "Architecture improvements" in description, (
            f"Layer description: {description}"
        )


class TestLambdaConfiguration:
    """Tests for Lambda configuration."""

    def test_config_sync_lambda_exists(self, lambda_client):
        """config-sync Lambda should exist."""
        env_suffix = STACK_NAME.split("-")[-1]
        function_name = f"bedrock-profiler-config-sync-{env_suffix}"

        try:
            response = lambda_client.get_function(FunctionName=function_name)
        except ClientError as e:
            pytest.fail(f"Function {function_name} not found: {e}")

        assert response["Configuration"]["FunctionName"] == function_name

    def test_lambdas_have_cognito_region_env(self, lambda_client):
        """Lambdas should have COGNITO_REGION environment variable."""
        env_suffix = STACK_NAME.split("-")[-1]
        functions_to_check = [
            f"bedrock-profiler-analytics-{env_suffix}",
            f"bedrock-profiler-cognito-sync-{env_suffix}",
        ]

        checked_count = 0
        for function_name in functions_to_check:
            try:
                response = lambda_client.get_function_configuration(
                    FunctionName=function_name
                )
                env_vars = response.get("Environment", {}).get("Variables", {})

                assert "COGNITO_REGION" in env_vars, (
                    f"{function_name} missing COGNITO_REGION"
                )
                checked_count += 1
            except ClientError:
                # Function may not exist in all deployments
                continue

        if checked_count == 0:
            pytest.skip("No functions with COGNITO_REGION requirement found")


class TestS3Configuration:
    """Tests for S3 configuration."""

    def test_config_uploaded_to_s3(self, s3_client, stack_outputs):
        """profiler-config.json should exist in S3."""
        bucket = stack_outputs.get("DataBucketName")
        if not bucket:
            pytest.skip("DataBucketName not in stack outputs")

        try:
            response = s3_client.head_object(
                Bucket=bucket, Key="config/profiler-config.json"
            )
        except ClientError as e:
            pytest.fail(f"Config file not found in S3: {e}")

        assert response["ContentLength"] > 0, "Config file is empty"
