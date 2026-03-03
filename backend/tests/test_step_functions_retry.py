"""Tests for Step Functions retry configuration (Phase 4 - Task 17).

Tests verify that the Step Functions workflow has proper retry configuration
with JitterStrategy and MaxDelaySeconds for all retry blocks.
"""

import json
import pytest
from pathlib import Path

WORKFLOW_PATH = (
    Path(__file__).parent.parent / "statemachine" / "bedrock-profiler.asl.json"
)


def load_workflow():
    """Load and parse the workflow definition."""
    with open(WORKFLOW_PATH) as f:
        return json.load(f)


def find_all_retries(obj, retries=None):
    """Recursively find all Retry configurations in the workflow.

    Args:
        obj: The object to search (dict, list, or other)
        retries: Accumulator list for found retry configs

    Returns:
        List of all Retry configuration objects found
    """
    if retries is None:
        retries = []

    if isinstance(obj, dict):
        if "Retry" in obj:
            retries.extend(obj["Retry"])
        for value in obj.values():
            find_all_retries(value, retries)
    elif isinstance(obj, list):
        for item in obj:
            find_all_retries(item, retries)

    return retries


class TestWorkflowValidity:
    """Tests for workflow JSON validity."""

    def test_workflow_json_valid(self):
        """Workflow JSON should be valid and have required structure."""
        # Arrange & Act
        workflow = load_workflow()

        # Assert
        assert workflow is not None, "Workflow should not be None"
        assert "States" in workflow, "Workflow should have States"
        assert "StartAt" in workflow, "Workflow should have StartAt"
        assert isinstance(workflow["States"], dict), "States should be a dict"


class TestRetryJitterStrategy:
    """Tests for JitterStrategy configuration in retries."""

    def test_all_retries_have_jitter_strategy(self):
        """All retry configurations should have JitterStrategy."""
        # Arrange
        workflow = load_workflow()

        # Act
        retries = find_all_retries(workflow)

        # Assert
        assert len(retries) > 0, "No retry configurations found"
        for retry in retries:
            assert "JitterStrategy" in retry, (
                f"Retry missing JitterStrategy: {retry.get('ErrorEquals', 'unknown')}"
            )

    def test_jitter_strategy_is_full(self):
        """JitterStrategy should be FULL for all retries."""
        # Arrange
        workflow = load_workflow()

        # Act
        retries = find_all_retries(workflow)

        # Assert
        for retry in retries:
            if "JitterStrategy" in retry:
                assert retry["JitterStrategy"] == "FULL", (
                    f"JitterStrategy should be FULL, got {retry['JitterStrategy']} "
                    f"for {retry.get('ErrorEquals', 'unknown')}"
                )


class TestRetryMaxDelay:
    """Tests for MaxDelaySeconds configuration in retries."""

    def test_all_retries_have_max_delay_seconds(self):
        """All retry configurations should have MaxDelaySeconds."""
        # Arrange
        workflow = load_workflow()

        # Act
        retries = find_all_retries(workflow)

        # Assert
        assert len(retries) > 0, "No retry configurations found"
        for retry in retries:
            assert "MaxDelaySeconds" in retry, (
                f"Retry missing MaxDelaySeconds: {retry.get('ErrorEquals', 'unknown')}"
            )

    def test_throttling_retry_has_longer_delay(self):
        """Throttling retries should have MaxDelaySeconds >= 120."""
        # Arrange
        workflow = load_workflow()

        # Act
        retries = find_all_retries(workflow)
        throttling_retries = [
            r
            for r in retries
            if "ThrottlingException" in r.get("ErrorEquals", [])
            or "ProvisionedThroughputExceededException" in r.get("ErrorEquals", [])
        ]

        # Assert
        assert len(throttling_retries) > 0, "No throttling retry configurations found"
        for retry in throttling_retries:
            assert retry.get("MaxDelaySeconds", 0) >= 120, (
                f"Throttling retry should have MaxDelaySeconds >= 120, "
                f"got {retry.get('MaxDelaySeconds')} for {retry.get('ErrorEquals')}"
            )

    def test_standard_retry_has_reasonable_delay(self):
        """Standard retries (non-throttling) should have MaxDelaySeconds <= 120.

        Note: Most standard retries have MaxDelaySeconds <= 60, but some tasks
        like the self-healing agent may have longer delays (up to 120) due to
        the nature of their operations (e.g., calling Bedrock Claude).
        """
        # Arrange
        workflow = load_workflow()

        # Act
        retries = find_all_retries(workflow)
        standard_retries = [
            r
            for r in retries
            if "ThrottlingException" not in r.get("ErrorEquals", [])
            and "ProvisionedThroughputExceededException" not in r.get("ErrorEquals", [])
        ]

        # Assert
        assert len(standard_retries) > 0, "No standard retry configurations found"
        for retry in standard_retries:
            # Standard retries should have reasonable delays (up to 120 for special cases)
            assert retry.get("MaxDelaySeconds", 0) <= 120, (
                f"Standard retry should have MaxDelaySeconds <= 120, "
                f"got {retry.get('MaxDelaySeconds')} for {retry.get('ErrorEquals')}"
            )


class TestRetryCount:
    """Tests for retry configuration coverage."""

    def test_workflow_has_multiple_retry_configs(self):
        """Workflow should have multiple retry configurations."""
        # Arrange
        workflow = load_workflow()

        # Act
        retries = find_all_retries(workflow)

        # Assert - workflow has many Lambda invocations, should have many retries
        assert len(retries) >= 10, (
            f"Expected at least 10 retry configurations, found {len(retries)}"
        )

    def test_all_task_states_have_retry_or_catch(self):
        """All Task states invoking Lambda should have retry or catch configuration.

        Note: Some tasks like DiscoverRegions intentionally use only Catch
        (not Retry) because they are designed to fail gracefully and continue
        with default values.
        """
        # Arrange
        workflow = load_workflow()

        # Act - find all Task states
        def find_task_states(obj, tasks=None):
            if tasks is None:
                tasks = []
            if isinstance(obj, dict):
                if obj.get("Type") == "Task" and "Resource" in obj:
                    tasks.append(obj)
                for value in obj.values():
                    find_task_states(value, tasks)
            elif isinstance(obj, list):
                for item in obj:
                    find_task_states(item, tasks)
            return tasks

        task_states = find_task_states(workflow)

        # Assert - all Task states should have Retry OR Catch
        for task in task_states:
            has_retry = "Retry" in task
            has_catch = "Catch" in task
            assert has_retry or has_catch, (
                f"Task state missing both Retry and Catch configuration: "
                f"{task.get('Resource', 'unknown')}"
            )

    def test_most_task_states_have_retry(self):
        """Most Task states should have retry configuration for resilience."""
        # Arrange
        workflow = load_workflow()

        # Act - find all Task states
        def find_task_states(obj, tasks=None):
            if tasks is None:
                tasks = []
            if isinstance(obj, dict):
                if obj.get("Type") == "Task" and "Resource" in obj:
                    tasks.append(obj)
                for value in obj.values():
                    find_task_states(value, tasks)
            elif isinstance(obj, list):
                for item in obj:
                    find_task_states(item, tasks)
            return tasks

        task_states = find_task_states(workflow)
        tasks_with_retry = [t for t in task_states if "Retry" in t]

        # Assert - at least 90% of tasks should have Retry
        retry_percentage = len(tasks_with_retry) / len(task_states) * 100
        assert retry_percentage >= 90, (
            f"Expected at least 90% of tasks to have Retry, got {retry_percentage:.1f}%"
        )
