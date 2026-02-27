"""
Tests for Step Functions workflow integration.

Tests the ASL JSON structure and workflow configuration for lifecycle data collection.
"""

import json
import os
import pytest


# Path to the ASL file
ASL_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "statemachine", "bedrock-profiler.asl.json"
)


@pytest.fixture
def asl_content():
    """Load the ASL JSON content."""
    with open(ASL_FILE_PATH, "r") as f:
        return json.load(f)


class TestASLStructure:
    """Tests for ASL JSON structure and validity."""

    def test_asl_json_valid(self, asl_content):
        """Test that ASL JSON is valid and parseable."""
        # Assert - If we got here, JSON is valid
        assert asl_content is not None
        assert isinstance(asl_content, dict)

        # Verify required top-level keys
        assert "StartAt" in asl_content
        assert "States" in asl_content
        assert isinstance(asl_content["States"], dict)

    def test_asl_has_lifecycle_branch(self, asl_content):
        """Test that CollectLifecycle state exists in Wave 2."""
        states = asl_content.get("States", {})

        # Find Wave2_EnrichmentProcessing
        wave2 = states.get("Wave2_EnrichmentProcessing")
        assert wave2 is not None, "Wave2_EnrichmentProcessing state not found"
        assert wave2.get("Type") == "Parallel"

        # Check that one of the branches contains CollectLifecycle
        branches = wave2.get("Branches", [])
        assert len(branches) > 0, "Wave2 has no branches"

        # Find the lifecycle branch
        lifecycle_branch_found = False
        for branch in branches:
            start_state = branch.get("StartAt")
            if start_state == "CollectLifecycle":
                lifecycle_branch_found = True
                # Verify the state exists in the branch
                branch_states = branch.get("States", {})
                assert "CollectLifecycle" in branch_states

                # Verify CollectLifecycle is a Task state
                lifecycle_state = branch_states["CollectLifecycle"]
                assert lifecycle_state.get("Type") == "Task"
                assert "Resource" in lifecycle_state
                break

        assert lifecycle_branch_found, "CollectLifecycle branch not found in Wave2"

    def test_asl_prepare_aggregation_lifecycle(self, asl_content):
        """Test that PrepareAggregation includes lifecycleData."""
        states = asl_content.get("States", {})

        # Find PrepareAggregation state
        prepare_agg = states.get("PrepareAggregation")
        assert prepare_agg is not None, "PrepareAggregation state not found"
        assert prepare_agg.get("Type") == "Pass"

        # Check Parameters include lifecycleData
        parameters = prepare_agg.get("Parameters", {})
        assert "lifecycleData.$" in parameters, (
            "lifecycleData not in PrepareAggregation Parameters"
        )

        # Verify the JSONPath reference
        lifecycle_path = parameters.get("lifecycleData.$")
        assert lifecycle_path is not None
        # Should reference wave2Results (lifecycle is in Wave 2)
        assert "wave2Results" in lifecycle_path


class TestWorkflowFlow:
    """Tests for workflow execution flow."""

    def test_lifecycle_collector_retry_config(self, asl_content):
        """Test that lifecycle collector has proper retry configuration."""
        states = asl_content.get("States", {})
        wave2 = states.get("Wave2_EnrichmentProcessing", {})
        branches = wave2.get("Branches", [])

        # Find lifecycle branch
        for branch in branches:
            if branch.get("StartAt") == "CollectLifecycle":
                lifecycle_state = branch.get("States", {}).get("CollectLifecycle", {})

                # Check retry configuration
                retry = lifecycle_state.get("Retry", [])
                assert len(retry) > 0, "CollectLifecycle has no retry configuration"

                # Verify retry catches States.ALL
                retry_config = retry[0]
                assert "States.ALL" in retry_config.get("ErrorEquals", [])
                assert retry_config.get("MaxAttempts", 0) >= 2
                break

    def test_lifecycle_collector_error_handling(self, asl_content):
        """Test that lifecycle collector has error handling (Catch)."""
        states = asl_content.get("States", {})
        wave2 = states.get("Wave2_EnrichmentProcessing", {})
        branches = wave2.get("Branches", [])

        # Find lifecycle branch
        for branch in branches:
            if branch.get("StartAt") == "CollectLifecycle":
                lifecycle_state = branch.get("States", {}).get("CollectLifecycle", {})

                # Check catch configuration
                catch = lifecycle_state.get("Catch", [])
                assert len(catch) > 0, "CollectLifecycle has no Catch configuration"

                # Verify catch handles States.ALL
                catch_config = catch[0]
                assert "States.ALL" in catch_config.get("ErrorEquals", [])
                assert "Next" in catch_config  # Should transition to failed state
                break

    def test_final_aggregation_receives_lifecycle(self, asl_content):
        """Test that FinalAggregation receives lifecycleData parameter."""
        states = asl_content.get("States", {})

        # Find FinalAggregation state
        final_agg = states.get("FinalAggregation")
        assert final_agg is not None, "FinalAggregation state not found"

        # Check Parameters include lifecycleData
        parameters = final_agg.get("Parameters", {})
        assert "lifecycleData.$" in parameters, (
            "lifecycleData not passed to FinalAggregation"
        )
