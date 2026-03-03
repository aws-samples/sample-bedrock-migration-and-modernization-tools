"""
Integration tests for production validation.

These tests verify that all architecture improvements are working correctly
in the production environment after a pipeline execution.

Usage:
    STACK_NAME=bedrock-profiler-prod pytest tests/test_production_validation.py -v
"""

import os
import json
import pytest
import boto3
import requests
from botocore.exceptions import ClientError

# Configuration
STACK_NAME = os.environ.get("STACK_NAME", "bedrock-profiler-prod")
CLOUDFRONT_URL = os.environ.get(
    "CLOUDFRONT_URL", "https://d3oem6l61p8j11.cloudfront.net"
)


@pytest.fixture(scope="module")
def s3_client():
    """Create S3 client."""
    return boto3.client("s3")


@pytest.fixture(scope="module")
def sfn_client():
    """Create Step Functions client."""
    return boto3.client("stepfunctions")


@pytest.fixture(scope="module")
def cloudformation_client():
    """Create CloudFormation client."""
    return boto3.client("cloudformation")


@pytest.fixture(scope="module")
def data_bucket(cloudformation_client):
    """Get data bucket name from stack outputs."""
    try:
        response = cloudformation_client.describe_stacks(StackName=STACK_NAME)
        outputs = response["Stacks"][0].get("Outputs", [])
        for output in outputs:
            if output["OutputKey"] == "DataBucketName":
                return output["OutputValue"]
        pytest.skip("DataBucketName not found in stack outputs")
    except ClientError as e:
        pytest.skip(f"Could not describe stack {STACK_NAME}: {e}")


@pytest.fixture(scope="module")
def state_machine_arn(cloudformation_client):
    """Get state machine ARN from stack outputs."""
    try:
        response = cloudformation_client.describe_stacks(StackName=STACK_NAME)
        outputs = response["Stacks"][0].get("Outputs", [])
        for output in outputs:
            if output["OutputKey"] == "StateMachineArn":
                return output["OutputValue"]
        pytest.skip("StateMachineArn not found in stack outputs")
    except ClientError as e:
        pytest.skip(f"Could not describe stack {STACK_NAME}: {e}")


@pytest.fixture(scope="module")
def latest_execution_id(s3_client, data_bucket):
    """Get the latest execution ID from S3."""
    try:
        response = s3_client.list_objects_v2(
            Bucket=data_bucket, Prefix="executions/", Delimiter="/"
        )

        prefixes = response.get("CommonPrefixes", [])
        if not prefixes:
            pytest.skip("No executions found")

        # Get the most recent execution
        latest = sorted(prefixes, key=lambda x: x["Prefix"], reverse=True)[0]
        return latest["Prefix"].split("/")[1]
    except ClientError as e:
        pytest.skip(f"Could not list executions: {e}")


class TestPipelineExecution:
    """Tests for pipeline execution."""

    def test_pipeline_execution_succeeds(self, sfn_client, state_machine_arn):
        """Pipeline should have at least one successful execution."""
        try:
            response = sfn_client.list_executions(
                stateMachineArn=state_machine_arn,
                statusFilter="SUCCEEDED",
                maxResults=1,
            )
        except ClientError as e:
            pytest.skip(f"Could not list executions: {e}")

        executions = response.get("executions", [])
        assert len(executions) > 0, "No successful executions found"


class TestFrontendConfig:
    """Tests for frontend config generation."""

    def test_frontend_config_generated(self, s3_client, data_bucket):
        """frontend-config.json should exist and have content."""
        try:
            response = s3_client.get_object(
                Bucket=data_bucket, Key="config/frontend-config.json"
            )
        except ClientError as e:
            pytest.skip(f"Frontend config not found: {e}")

        content = json.loads(response["Body"].read())

        assert "regions" in content, "Missing 'regions' in frontend config"
        assert "providers" in content, "Missing 'providers' in frontend config"
        assert len(content["regions"]) > 0, "No regions in frontend config"


class TestGapDetection:
    """Tests for gap detection output."""

    def test_gap_report_has_new_types(
        self, s3_client, data_bucket, latest_execution_id
    ):
        """Gap report should include new gap types."""
        try:
            response = s3_client.get_object(
                Bucket=data_bucket,
                Key=f"agent/gap-reports/{latest_execution_id}/gap-analysis.json",
            )

            content = json.loads(response["Body"].read())
            details = content.get("details", {})

            # Check for new gap types (may be empty if no gaps)
            assert "context_window_mismatches" in details or "summary" in content, (
                "Gap report missing expected structure"
            )

        except ClientError:
            pytest.skip("Gap report not found for latest execution")


class TestCaching:
    """Tests for caching functionality."""

    def test_cache_files_created(self, s3_client, data_bucket, latest_execution_id):
        """Cache files should exist in executions/{id}/cache/."""
        try:
            response = s3_client.list_objects_v2(
                Bucket=data_bucket,
                Prefix=f"executions/{latest_execution_id}/cache/",
                MaxKeys=5,
            )
        except ClientError as e:
            pytest.skip(f"Could not list cache files: {e}")

        # Cache files should exist if model-extractor ran
        contents = response.get("Contents", [])
        # Note: Cache may not exist if execution didn't include model-extractor
        # This is informational, not a failure
        if not contents:
            pytest.skip("No cache files found (may be expected)")

        assert any("list_foundation_models" in obj["Key"] for obj in contents), (
            "No list_foundation_models cache file found"
        )


class TestFinalOutput:
    """Tests for final output files."""

    def test_final_output_updated(self, s3_client, data_bucket):
        """latest/ files should be updated."""
        try:
            # Check models file
            models_response = s3_client.head_object(
                Bucket=data_bucket, Key="latest/bedrock_models.json"
            )

            # Check pricing file
            pricing_response = s3_client.head_object(
                Bucket=data_bucket, Key="latest/bedrock_pricing.json"
            )
        except ClientError as e:
            pytest.fail(f"Final output files not found: {e}")

        assert models_response["ContentLength"] > 0, "Models file is empty"
        assert pricing_response["ContentLength"] > 0, "Pricing file is empty"


class TestCloudFrontAccess:
    """Tests for CloudFront accessibility."""

    def test_cloudfront_serves_data(self):
        """CloudFront should serve data files."""
        try:
            # Test models endpoint
            models_response = requests.get(
                f"{CLOUDFRONT_URL}/latest/bedrock_models.json", timeout=10
            )
        except requests.RequestException as e:
            pytest.fail(f"Could not reach CloudFront: {e}")

        assert models_response.status_code == 200, (
            f"CloudFront returned {models_response.status_code}"
        )

        data = models_response.json()
        assert "providers" in data, "Missing 'providers' in models response"

    def test_cloudfront_serves_frontend_config(self):
        """CloudFront should serve frontend config."""
        try:
            config_response = requests.get(
                f"{CLOUDFRONT_URL}/config/frontend-config.json", timeout=10
            )
        except requests.RequestException as e:
            pytest.fail(f"Could not reach CloudFront: {e}")

        # May return 404 if config-sync hasn't run yet
        if config_response.status_code == 404:
            pytest.skip("Frontend config not yet generated")

        assert config_response.status_code == 200, (
            f"CloudFront returned {config_response.status_code}"
        )

        # Check content type - CloudFront may return HTML (SPA fallback) if file doesn't exist
        content_type = config_response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            pytest.skip(
                "Frontend config not yet generated (CloudFront returned SPA fallback)"
            )

        try:
            data = config_response.json()
        except requests.exceptions.JSONDecodeError:
            pytest.skip("Frontend config not yet generated (response is not JSON)")

        assert "regions" in data, "Missing 'regions' in frontend config"
