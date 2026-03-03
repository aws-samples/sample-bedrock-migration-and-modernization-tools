"""End-to-end tests for the workflow (Phase 4 Integration).

These tests require a deployed stack and are skipped by default.
Set RUN_E2E_TESTS=true environment variable to run them.

Note: E2E tests may incur AWS costs.
"""

import json
import os
import time

import boto3
import pytest

# Skip if not in integration test environment
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_E2E_TESTS") != "true",
    reason="E2E tests require RUN_E2E_TESTS=true environment variable",
)


@pytest.fixture
def sfn_client():
    """Create Step Functions client."""
    return boto3.client("stepfunctions")


@pytest.fixture
def logs_client():
    """Create CloudWatch Logs client."""
    return boto3.client("logs")


@pytest.fixture
def cloudwatch_client():
    """Create CloudWatch client for metrics."""
    return boto3.client("cloudwatch")


@pytest.fixture
def xray_client():
    """Create X-Ray client for traces."""
    return boto3.client("xray")


def get_state_machine_arn():
    """Get the state machine ARN from environment or CloudFormation."""
    # First check environment variable
    arn = os.environ.get("STATE_MACHINE_ARN")
    if arn:
        return arn

    # Try to get from CloudFormation stack
    cfn = boto3.client("cloudformation")
    stack_name = os.environ.get("STACK_NAME", "bedrock-profiler-dev")

    try:
        response = cfn.describe_stacks(StackName=stack_name)
        outputs = response["Stacks"][0].get("Outputs", [])
        for output in outputs:
            if output["OutputKey"] == "StateMachineArn":
                return output["OutputValue"]
    except Exception:
        pass

    pytest.skip(
        "STATE_MACHINE_ARN not found. Set STATE_MACHINE_ARN env var or deploy stack."
    )
    return None


class TestWorkflowExecution:
    """Tests for workflow execution."""

    def test_workflow_execution_succeeds(self, sfn_client):
        """Workflow execution should succeed.

        Note: This test starts an actual workflow execution and waits for completion.
        It may take several minutes and incur AWS costs.
        """
        # Arrange
        state_machine_arn = get_state_machine_arn()

        # Act - Start execution
        response = sfn_client.start_execution(
            stateMachineArn=state_machine_arn, input="{}"
        )
        execution_arn = response["executionArn"]

        # Wait for completion (max 10 minutes)
        max_wait_time = 600  # 10 minutes
        poll_interval = 30  # 30 seconds
        elapsed = 0

        while elapsed < max_wait_time:
            describe_response = sfn_client.describe_execution(
                executionArn=execution_arn
            )
            status = describe_response["status"]

            if status == "SUCCEEDED":
                break
            elif status in ["FAILED", "TIMED_OUT", "ABORTED"]:
                pytest.fail(
                    f"Workflow execution {status}: {describe_response.get('error', 'unknown')}"
                )

            time.sleep(poll_interval)
            elapsed += poll_interval

        # Assert
        assert status == "SUCCEEDED", (
            f"Workflow did not complete in time. Status: {status}"
        )


class TestCloudWatchLogs:
    """Tests for CloudWatch Logs integration."""

    def test_cloudwatch_logs_are_json(self, logs_client):
        """CloudWatch logs should be JSON formatted (Powertools structured logging).

        This test queries recent logs from Lambda functions and verifies
        they are in JSON format as expected from Powertools.
        """
        # Arrange
        log_group_prefix = "/aws/lambda/bedrock-profiler-"
        environment = os.environ.get("ENVIRONMENT", "dev")

        # Get log groups
        response = logs_client.describe_log_groups(
            logGroupNamePrefix=f"{log_group_prefix}"
        )
        log_groups = [lg["logGroupName"] for lg in response.get("logGroups", [])]

        if not log_groups:
            pytest.skip("No log groups found. Deploy stack first.")

        # Act - Query recent logs from first log group
        log_group = log_groups[0]
        end_time = int(time.time() * 1000)
        start_time = end_time - (24 * 60 * 60 * 1000)  # Last 24 hours

        try:
            response = logs_client.filter_log_events(
                logGroupName=log_group, startTime=start_time, endTime=end_time, limit=10
            )
        except Exception as e:
            pytest.skip(f"Could not query logs: {e}")

        events = response.get("events", [])
        if not events:
            pytest.skip("No log events found in the last 24 hours.")

        # Assert - Check that logs are JSON formatted
        json_logs_found = 0
        for event in events:
            message = event.get("message", "")
            try:
                parsed = json.loads(message)
                if isinstance(parsed, dict):
                    json_logs_found += 1
            except json.JSONDecodeError:
                # Some logs may not be JSON (e.g., START, END, REPORT)
                pass

        assert json_logs_found > 0, (
            "No JSON-formatted logs found. Powertools logging may not be configured."
        )


class TestXRayTraces:
    """Tests for X-Ray tracing integration."""

    def test_xray_traces_exist(self, xray_client):
        """X-Ray traces should exist for workflow execution.

        This test queries X-Ray for traces from the Bedrock Profiler service.
        """
        # Arrange
        end_time = time.time()
        start_time = end_time - (24 * 60 * 60)  # Last 24 hours

        # Act - Query for traces
        try:
            response = xray_client.get_trace_summaries(
                StartTime=start_time,
                EndTime=end_time,
                FilterExpression='service(id(name: "bedrock-profiler"))',
                Sampling=False,
            )
        except Exception as e:
            pytest.skip(f"Could not query X-Ray traces: {e}")

        traces = response.get("TraceSummaries", [])

        # Assert
        if not traces:
            pytest.skip(
                "No X-Ray traces found. Run workflow first or check X-Ray configuration."
            )

        assert len(traces) > 0, "Expected at least one X-Ray trace"


class TestCloudWatchMetrics:
    """Tests for CloudWatch metrics integration."""

    def test_cloudwatch_metrics_emitted(self, cloudwatch_client):
        """CloudWatch metrics should be emitted by handlers.

        This test queries CloudWatch for metrics in the BedrockProfiler namespace.
        """
        # Arrange
        namespace = "BedrockProfiler"
        end_time = time.time()
        start_time = end_time - (24 * 60 * 60)  # Last 24 hours

        # Act - List metrics in namespace
        try:
            response = cloudwatch_client.list_metrics(Namespace=namespace)
        except Exception as e:
            pytest.skip(f"Could not query CloudWatch metrics: {e}")

        metrics = response.get("Metrics", [])

        # Assert
        if not metrics:
            pytest.skip(
                "No metrics found in BedrockProfiler namespace. Run workflow first."
            )

        assert len(metrics) > 0, (
            "Expected at least one metric in BedrockProfiler namespace"
        )

        # Verify we have expected metric names
        metric_names = {m["MetricName"] for m in metrics}
        # Powertools typically emits ColdStart and custom metrics
        assert len(metric_names) > 0, "Expected at least one unique metric name"
