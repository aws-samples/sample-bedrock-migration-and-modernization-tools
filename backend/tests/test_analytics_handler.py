"""
Tests for analytics handler — Phase 1, Phase 2, Phase 3 & Phase 4.

Covers:
  Task 01: Event counting fixes (BUGs 3, 4, 7)
  Task 02: Aggregation fixes (BUGs 1, 2, 8)
  Task 03: SAM template validation
  Task 06: Fix frontend event collection (sendBeacon Blob, page_view)
  Task 07: Add missing event tracking (search_filter, region_availability_view)
  Task 09: Dashboard API — Cognito data integration
  Task 10: Dashboard API — New metrics (comparison wins, engagement, regions)
  Task 11: Frontend utility function validation (Python equivalents)
  Task 13-16: Dashboard response shape tests (all 4 redesigned tabs)
  Task 17: Performance optimization tests (Cognito cache for returning users)

Run:
    cd backend && python3 -m pytest tests/test_analytics_handler.py -v

Requires: pytest, pyyaml
    pip install pytest pyyaml
"""

import importlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Path setup — use importlib to avoid name collisions with other handler.py
# ---------------------------------------------------------------------------
LAMBDA_DIR = os.path.join(os.path.dirname(__file__), "..", "lambdas", "analytics")
_HANDLER_PATH = os.path.join(LAMBDA_DIR, "handler.py")
_MODULE_NAME = "analytics_handler"  # unique name to avoid collision


def _import_analytics_handler():
    """Import the analytics handler module using importlib to avoid name collisions.

    Both analytics and cognito-sync have handler.py — using importlib with a
    unique module name prevents cross-contamination when tests run together.
    """
    # Remove stale entry if present
    if _MODULE_NAME in sys.modules:
        del sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _HANDLER_PATH)
    assert spec is not None, f"Could not find module spec for {_HANDLER_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    assert spec.loader is not None, f"Module spec has no loader for {_HANDLER_PATH}"

    # Mock shared.powertools BEFORE loading the module
    # Create pass-through decorators that preserve function behavior
    mock_logger = MagicMock()
    mock_logger.inject_lambda_context = lambda **kwargs: lambda f: f
    mock_logger.info = MagicMock()
    mock_logger.warning = MagicMock()
    mock_logger.error = MagicMock()
    mock_logger.debug = MagicMock()

    mock_tracer = MagicMock()
    mock_tracer.capture_method = lambda f: f
    mock_tracer.capture_lambda_handler = lambda f: f

    mock_metrics = MagicMock()
    mock_metrics.log_metrics = lambda **kwargs: lambda f: f
    mock_metrics.add_metric = MagicMock()

    mock_powertools = MagicMock()
    mock_powertools.logger = mock_logger
    mock_powertools.tracer = mock_tracer
    mock_powertools.metrics = mock_metrics
    mock_powertools.LambdaContext = MagicMock
    mock_powertools.MetricUnit = MagicMock()

    # Mock aws_lambda_powertools.metrics.MetricUnit
    mock_aws_powertools_metrics = MagicMock()
    mock_aws_powertools_metrics.MetricUnit = MagicMock()
    mock_aws_powertools_metrics.MetricUnit.Count = "Count"

    sys.modules["shared"] = MagicMock()
    sys.modules["shared.powertools"] = mock_powertools
    sys.modules["aws_lambda_powertools"] = MagicMock()
    sys.modules["aws_lambda_powertools.metrics"] = mock_aws_powertools_metrics
    sys.modules["aws_lambda_powertools.utilities.typing"] = MagicMock()

    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures — mock DynamoDB before importing the handler module
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    """Set environment variables before every test."""
    monkeypatch.setenv("ANALYTICS_TABLE", "test-analytics-table")
    monkeypatch.setenv("ADMIN_GROUP", "admins")
    monkeypatch.setenv("ALLOWED_ORIGINS", "*")


@pytest.fixture()
def mock_dynamodb():
    """Patch boto3.resource so the handler gets a mock DynamoDB table."""
    mock_table = MagicMock()
    mock_table.update_item = MagicMock()
    mock_table.put_item = MagicMock()
    mock_table.get_item = MagicMock(return_value={"Item": {"firstSeen": "2025-01-01"}})
    mock_table.query = MagicMock(return_value={"Items": []})

    mock_resource = MagicMock()
    mock_resource.Table.return_value = mock_table
    mock_resource.meta.client.batch_get_item = MagicMock(
        return_value={"Responses": {"test-analytics-table": []}}
    )

    with patch("boto3.resource", return_value=mock_resource):
        handler = _import_analytics_handler()

        yield {
            "handler": handler,
            "table": mock_table,
            "resource": mock_resource,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(events_list, auid="test-user-1", country="US", region="us-east-1"):
    """Build an API Gateway v2 event for POST /events."""
    return {
        "routeKey": "POST /events",
        "body": json.dumps(
            {
                "events": events_list,
                "auid": auid,
                "country": country,
                "region": region,
            }
        ),
    }


def _make_dashboard_event(params=None, groups=None):
    """Build an API Gateway v2 event for GET /dashboard."""
    if groups is None:
        groups = ["admins"]
    return {
        "routeKey": "GET /dashboard",
        "queryStringParameters": params or {},
        "requestContext": {
            "authorizer": {
                "jwt": {
                    "claims": {
                        "cognito:groups": groups,
                    }
                }
            }
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Task 01 — Event Counting Fixes (BUGs 3, 4, 7)
# ═══════════════════════════════════════════════════════════════════════════


class TestSectionChangeIncrementsViews:
    """BUG 3/4 fix: section_change events should increment view_count."""

    def test_section_change_increments_views(self, mock_dynamodb):
        """3 section_change events should produce view_count == 3."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        events = [{"type": "section_change", "section": "overview"}] * 3
        result = handler.handle_post_events(_make_event(events))

        body = json.loads(result["body"])
        assert result["statusCode"] == 200
        assert body["recorded"] == 3

        # Verify _update_daily_aggregate was called with views=3
        # The first positional call to table.update_item is the daily aggregate
        update_calls = table.update_item.call_args_list
        # The daily aggregate update should include views=3 via ADD #v :views
        agg_call = update_calls[0]
        expr_values = agg_call.kwargs.get("ExpressionAttributeValues", {})
        assert expr_values.get(":views") == 3

    def test_page_view_still_increments_views(self, mock_dynamodb):
        """2 page_view events should produce view_count == 2."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        events = [{"type": "page_view", "section": "home"}] * 2
        result = handler.handle_post_events(_make_event(events))

        body = json.loads(result["body"])
        assert body["recorded"] == 2

        update_calls = table.update_item.call_args_list
        agg_call = update_calls[0]
        expr_values = agg_call.kwargs.get("ExpressionAttributeValues", {})
        assert expr_values.get(":views") == 2

    def test_mixed_events_view_count(self, mock_dynamodb):
        """2 page_view + 3 section_change = view_count 5."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        events = [
            {"type": "page_view", "section": "home"},
            {"type": "page_view", "section": "home"},
            {"type": "section_change", "section": "pricing"},
            {"type": "section_change", "section": "regions"},
            {"type": "section_change", "section": "overview"},
        ]
        result = handler.handle_post_events(_make_event(events))

        body = json.loads(result["body"])
        assert body["recorded"] == 5

        update_calls = table.update_item.call_args_list
        agg_call = update_calls[0]
        expr_values = agg_call.kwargs.get("ExpressionAttributeValues", {})
        assert expr_values.get(":views") == 5


class TestComparisonInflation:
    """BUG 7 fix: comparison_remove/clear should NOT inflate comparison counts."""

    def test_comparison_remove_no_inflation(self, mock_dynamodb):
        """comparison_remove should not increment feature_counts['comparisons']."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        events = [{"type": "comparison_remove", "meta": {"modelId": "m1"}}]
        result = handler.handle_post_events(_make_event(events))

        body = json.loads(result["body"])
        assert body["recorded"] == 1

        # The daily aggregate update should NOT have views (views == 0 means no ADD #v)
        # and the features map should not get a comparisons increment.
        # Check that _increment_map_counter for "features" with "comparisons" was NOT called
        # by inspecting all update_item calls for features.comparisons
        update_calls = table.update_item.call_args_list
        for c in update_calls:
            expr_names = c.kwargs.get("ExpressionAttributeNames", {})
            # If this is a map counter update for features.comparisons, fail
            if (
                expr_names.get("#m") == "features"
                and expr_names.get("#k") == "comparisons"
            ):
                pytest.fail(
                    "comparison_remove should not increment features.comparisons"
                )

    def test_comparison_clear_no_inflation(self, mock_dynamodb):
        """comparison_clear should not increment feature_counts['comparisons']."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        events = [{"type": "comparison_clear"}]
        result = handler.handle_post_events(_make_event(events))

        body = json.loads(result["body"])
        assert body["recorded"] == 1

        update_calls = table.update_item.call_args_list
        for c in update_calls:
            expr_names = c.kwargs.get("ExpressionAttributeNames", {})
            if (
                expr_names.get("#m") == "features"
                and expr_names.get("#k") == "comparisons"
            ):
                pytest.fail(
                    "comparison_clear should not increment features.comparisons"
                )

    def test_comparison_add_still_counts(self, mock_dynamodb):
        """comparison_add should still increment feature_counts['comparisons']."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        events = [
            {
                "type": "comparison_add",
                "meta": {"modelId": "m1", "provider": "Anthropic"},
            }
        ]
        result = handler.handle_post_events(_make_event(events))

        body = json.loads(result["body"])
        assert body["recorded"] == 1

        # Verify that _increment_map_counter IS called for features.comparisons
        update_calls = table.update_item.call_args_list
        found_comparison_increment = False
        for c in update_calls:
            expr_names = c.kwargs.get("ExpressionAttributeNames", {})
            if (
                expr_names.get("#m") == "features"
                and expr_names.get("#k") == "comparisons"
            ):
                found_comparison_increment = True
                break
        assert found_comparison_increment, (
            "comparison_add should increment features.comparisons"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Task 02 — Aggregation Fixes (BUGs 1, 2, 8)
# ═══════════════════════════════════════════════════════════════════════════


class TestCountryRegionViewsMaps:
    """BUG 1 & 8 fix: countryViews/regionViews map attributes are created."""

    def test_country_views_map_created(self, mock_dynamodb):
        """Event with country='US' should call _increment_map_counter for countryViews."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        events = [{"type": "page_view", "section": "home"}]
        handler.handle_post_events(_make_event(events, country="US"))

        update_calls = table.update_item.call_args_list
        found_country_views = False
        for c in update_calls:
            expr_names = c.kwargs.get("ExpressionAttributeNames", {})
            if expr_names.get("#m") == "countryViews" and expr_names.get("#k") == "US":
                found_country_views = True
                break
        assert found_country_views, "countryViews map should be updated for country US"

    def test_region_views_map_created(self, mock_dynamodb):
        """Event with region='us-east-1' should call _increment_map_counter for regionViews."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        events = [{"type": "page_view", "section": "home"}]
        handler.handle_post_events(_make_event(events, region="us-east-1"))

        update_calls = table.update_item.call_args_list
        found_region_views = False
        for c in update_calls:
            expr_names = c.kwargs.get("ExpressionAttributeNames", {})
            if (
                expr_names.get("#m") == "regionViews"
                and expr_names.get("#k") == "us-east-1"
            ):
                found_region_views = True
                break
        assert found_region_views, (
            "regionViews map should be updated for region us-east-1"
        )


class TestCountryCountsUsesViews:
    """BUG 1 fix: countryCounts should use actual view counts from countryViews map."""

    def test_country_counts_uses_views_not_presence(self, mock_dynamodb):
        """Two daily items with countryViews={US:5} and {US:3} should sum to 8."""
        handler = mock_dynamodb["handler"]

        items = [
            {
                "PK": "AGG#daily",
                "SK": "2025-01-01",
                "views": 5,
                "countryViews": {"US": 5},
                "regionViews": {},
                "uniqueUsers": set(),
                "newUsers": set(),
                "countries": {"US"},
                "regions": set(),
                "sections": {},
                "features": {},
                "topModels": {},
                "comparedModels": {},
                "favoritedModels": {},
                "providerComparisons": {},
                "providerFavorites": {},
            },
            {
                "PK": "AGG#daily",
                "SK": "2025-01-02",
                "views": 3,
                "countryViews": {"US": 3},
                "regionViews": {},
                "uniqueUsers": set(),
                "newUsers": set(),
                "countries": {"US"},
                "regions": set(),
                "sections": {},
                "features": {},
                "topModels": {},
                "comparedModels": {},
                "favoritedModels": {},
                "providerComparisons": {},
                "providerFavorites": {},
            },
        ]

        now = datetime(2025, 1, 3, 12, 0, 0, tzinfo=timezone.utc)
        summary, _, _, _ = handler._build_summary(items, now, "2025-01-01")

        assert summary["countryCounts"] == [{"id": "US", "count": 8}]


class TestReturningUsersFirstSeen:
    """BUG 2 fix: returning users should be based on firstSeen < start_date."""

    def test_returning_users_uses_first_seen(self, mock_dynamodb):
        """5 unique users, 2 with firstSeen before start_date => returningUsers == 2."""
        handler = mock_dynamodb["handler"]
        resource = mock_dynamodb["resource"]

        # Set up batch_get_item to return 5 users, 2 with firstSeen before start
        resource.meta.client.batch_get_item.return_value = {
            "Responses": {
                "test-analytics-table": [
                    {"firstSeen": "2024-12-01"},  # before start -> returning
                    {"firstSeen": "2024-12-15"},  # before start -> returning
                    {"firstSeen": "2025-01-01"},  # same as start -> NOT returning
                    {"firstSeen": "2025-01-02"},  # after start -> NOT returning
                    {"firstSeen": "2025-01-03"},  # after start -> NOT returning
                ]
            }
        }

        unique_user_ids = {"u1", "u2", "u3", "u4", "u5"}
        result = handler._count_returning_users(unique_user_ids, "2025-01-01")

        assert result == 2

    def test_returning_users_empty_set(self, mock_dynamodb):
        """Empty user set should return 0 returning users."""
        handler = mock_dynamodb["handler"]

        result = handler._count_returning_users(set(), "2025-01-01")
        assert result == 0


class TestBuildSummaryBackwardCompat:
    """BUG 1/8 fix: backward compatibility with old data lacking countryViews/regionViews."""

    def test_build_summary_backward_compat(self, mock_dynamodb):
        """Daily item without countryViews/regionViews should not error."""
        handler = mock_dynamodb["handler"]

        items = [
            {
                "PK": "AGG#daily",
                "SK": "2025-01-01",
                "views": 10,
                # No countryViews or regionViews keys — old data format
                "uniqueUsers": set(),
                "newUsers": set(),
                "countries": {"US"},
                "regions": {"us-east-1"},
                "sections": {},
                "features": {},
                "topModels": {},
                "comparedModels": {},
                "favoritedModels": {},
                "providerComparisons": {},
                "providerFavorites": {},
            },
        ]

        now = datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        summary, _, _, _ = handler._build_summary(items, now, "2025-01-01")

        # Should produce empty counts, not crash
        assert summary["countryCounts"] == []
        assert summary["regionCounts"] == []
        assert summary["totalViews"] == 10


# ═══════════════════════════════════════════════════════════════════════════
# Task 03 — SAM Template Validation
# ═══════════════════════════════════════════════════════════════════════════


class TestSAMTemplate:
    """Validate the analytics SAM template structure."""

    TEMPLATE_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..", "infra", "analytics-template.yaml"
    )

    @staticmethod
    def _load_cfn_yaml(path):
        """Load a CloudFormation/SAM YAML template with intrinsic function support.

        CloudFormation uses custom YAML tags (!Sub, !Ref, !GetAtt, !Split, etc.)
        that yaml.safe_load cannot handle. We register constructors that pass
        through the tag values so the template can be parsed for structural checks.
        """
        import yaml

        class CfnLoader(yaml.SafeLoader):
            pass

        # Register constructors for common CloudFormation intrinsic functions
        cfn_tags = [
            "!Sub",
            "!Ref",
            "!GetAtt",
            "!Split",
            "!Select",
            "!Join",
            "!If",
            "!Not",
            "!Equals",
            "!And",
            "!Or",
            "!FindInMap",
            "!ImportValue",
            "!Condition",
            "!Base64",
            "!Cidr",
            "!GetAZs",
            "!Transform",
        ]

        def _cfn_constructor(loader, node):
            if isinstance(node, yaml.ScalarNode):
                return loader.construct_scalar(node)
            if isinstance(node, yaml.SequenceNode):
                return loader.construct_sequence(node)
            return loader.construct_mapping(node)  # type: ignore[arg-type]

        for tag in cfn_tags:
            CfnLoader.add_constructor(tag, _cfn_constructor)

        with open(path, "r") as f:
            return yaml.load(f, Loader=CfnLoader)

    def test_sam_template_valid_yaml(self):
        """Template should be valid YAML with no parse errors."""
        template = self._load_cfn_yaml(self.TEMPLATE_PATH)

        assert template is not None
        assert "Resources" in template
        assert "AWSTemplateFormatVersion" in template

    def test_sam_template_has_cognito_sync(self):
        """Template should define a CognitoSyncFunction resource."""
        template = self._load_cfn_yaml(self.TEMPLATE_PATH)

        resources = template.get("Resources", {})
        assert "CognitoSyncFunction" in resources, (
            "CognitoSyncFunction resource must be defined in the template"
        )

        cognito_sync = resources["CognitoSyncFunction"]
        assert cognito_sync["Type"] == "AWS::Serverless::Function"
        assert "handler.lambda_handler" in cognito_sync["Properties"]["Handler"]

    def test_sam_template_has_batch_get_item(self):
        """Analytics function policy should include BatchGetItem permission."""
        template = self._load_cfn_yaml(self.TEMPLATE_PATH)

        analytics_fn = template["Resources"]["AnalyticsFunction"]
        policies = analytics_fn["Properties"]["Policies"]

        # Flatten all actions across all policy statements
        all_actions = []
        for policy in policies:
            for stmt in policy.get("Statement", []):
                actions = stmt.get("Action", [])
                if isinstance(actions, str):
                    actions = [actions]
                all_actions.extend(actions)

        assert "dynamodb:BatchGetItem" in all_actions, (
            "AnalyticsFunction must have dynamodb:BatchGetItem permission"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Task 06 — Fix Frontend Event Collection (BUG 5 sendBeacon, BUG 3 page_view)
# ═══════════════════════════════════════════════════════════════════════════


class TestSendBeaconBlobFormat:
    """BUG 5 fix: sendBeacon with Blob(Content-Type: application/json) accepted."""

    def test_events_endpoint_accepts_json_content_type(self, mock_dynamodb):
        """POST /events with JSON body (as from sendBeacon Blob) returns 200 OK."""
        handler = mock_dynamodb["handler"]

        # Arrange: simulate the exact payload shape that sendBeacon sends
        # when using new Blob([JSON.stringify(payload)], {type: 'application/json'})
        events = [
            {"type": "page_view", "section": "home"},
            {"type": "section_change", "section": "pricing"},
        ]
        event = _make_event(
            events, auid="beacon-user-1", country="DE", region="eu-west-1"
        )

        # Act
        result = handler.handle_post_events(event)

        # Assert
        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["status"] == "ok"
        assert body["recorded"] == 2

    def test_section_change_event_has_section_field(self, mock_dynamodb):
        """section_change event with section='favorites' preserves section metadata."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        # Arrange
        events = [{"type": "section_change", "section": "favorites"}]

        # Act
        result = handler.handle_post_events(_make_event(events))

        # Assert: event processed successfully
        body = json.loads(result["body"])
        assert result["statusCode"] == 200
        assert body["recorded"] == 1

        # Assert: section 'favorites' was counted in the sections map
        update_calls = table.update_item.call_args_list
        found_favorites_section = False
        for c in update_calls:
            expr_names = c.kwargs.get("ExpressionAttributeNames", {})
            if (
                expr_names.get("#m") == "sections"
                and expr_names.get("#k") == "favorites"
            ):
                found_favorites_section = True
                break
        assert found_favorites_section, (
            "section_change with section='favorites' should increment sections.favorites"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Task 07 — Add Missing Event Tracking (Favorites, comparison removal, new events)
# ═══════════════════════════════════════════════════════════════════════════


class TestNewEventTypes:
    """Task 07: New event types search_filter and region_availability_view are accepted."""

    def test_search_filter_event_accepted(self, mock_dynamodb):
        """Event with type='search_filter' should be counted (in VALID_EVENT_TYPES)."""
        handler = mock_dynamodb["handler"]

        # Arrange
        events = [{"type": "search_filter", "meta": {"query": "anthropic"}}]

        # Act
        result = handler.handle_post_events(_make_event(events))

        # Assert
        body = json.loads(result["body"])
        assert result["statusCode"] == 200
        assert body["recorded"] == 1, (
            "search_filter should be in VALID_EVENT_TYPES and increment event_count"
        )

    def test_region_availability_view_accepted(self, mock_dynamodb):
        """Event with type='region_availability_view' should be counted."""
        handler = mock_dynamodb["handler"]

        # Arrange
        events = [
            {
                "type": "region_availability_view",
                "meta": {"modelId": "anthropic.claude-3-sonnet"},
            }
        ]

        # Act
        result = handler.handle_post_events(_make_event(events))

        # Assert
        body = json.loads(result["body"])
        assert result["statusCode"] == 200
        assert body["recorded"] == 1, (
            "region_availability_view should be in VALID_EVENT_TYPES and increment event_count"
        )


class TestComparisonRemoveTracking:
    """Task 07: comparison_remove with modelId tracked without inflating comparisons."""

    def test_comparison_remove_with_model_id(self, mock_dynamodb):
        """comparison_remove with modelId should be accepted without comparison inflation."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        # Arrange
        events = [{"type": "comparison_remove", "meta": {"modelId": "meta.llama3-70b"}}]

        # Act
        result = handler.handle_post_events(_make_event(events))

        # Assert: event accepted
        body = json.loads(result["body"])
        assert result["statusCode"] == 200
        assert body["recorded"] == 1

        # Assert: no comparison inflation — features.comparisons NOT incremented
        update_calls = table.update_item.call_args_list
        for c in update_calls:
            expr_names = c.kwargs.get("ExpressionAttributeNames", {})
            if (
                expr_names.get("#m") == "features"
                and expr_names.get("#k") == "comparisons"
            ):
                pytest.fail(
                    "comparison_remove should not inflate features.comparisons count"
                )


class TestFavoriteToggleWithSection:
    """Task 07: favorite_toggle with section='favorites' tracks favorites section."""

    def test_favorite_toggle_with_section_favorites(self, mock_dynamodb):
        """favorite_toggle with section='favorites' should count in sections map."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        # Arrange
        events = [
            {
                "type": "favorite_toggle",
                "section": "favorites",
                "meta": {"modelId": "amazon.titan-text-express", "provider": "Amazon"},
            }
        ]

        # Act
        result = handler.handle_post_events(_make_event(events))

        # Assert: event accepted
        body = json.loads(result["body"])
        assert result["statusCode"] == 200
        assert body["recorded"] == 1

        # Assert: section 'favorites' was counted in the sections map
        update_calls = table.update_item.call_args_list
        found_favorites_section = False
        for c in update_calls:
            expr_names = c.kwargs.get("ExpressionAttributeNames", {})
            if (
                expr_names.get("#m") == "sections"
                and expr_names.get("#k") == "favorites"
            ):
                found_favorites_section = True
                break
        assert found_favorites_section, (
            "favorite_toggle with section='favorites' should increment sections.favorites"
        )

        # Assert: features.favorites was also incremented
        found_favorites_feature = False
        for c in update_calls:
            expr_names = c.kwargs.get("ExpressionAttributeNames", {})
            if (
                expr_names.get("#m") == "features"
                and expr_names.get("#k") == "favorites"
            ):
                found_favorites_feature = True
                break
        assert found_favorites_feature, (
            "favorite_toggle should increment features.favorites"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Task 10 — Dashboard API New Metrics
# ═══════════════════════════════════════════════════════════════════════════


def _make_daily_item(
    date,
    views=0,
    unique_users=None,
    new_users=None,
    sections=None,
    features=None,
    compared_models=None,
    country_views=None,
    region_views=None,
    countries=None,
    regions=None,
):
    """Helper to build a daily aggregate item for _build_summary tests."""
    return {
        "PK": "AGG#daily",
        "SK": date,
        "views": views,
        "uniqueUsers": unique_users or set(),
        "newUsers": new_users or set(),
        "countries": countries or set(),
        "regions": regions or set(),
        "sections": sections or {},
        "features": features or {},
        "topModels": {},
        "comparedModels": compared_models or {},
        "favoritedModels": {},
        "providerComparisons": {},
        "providerFavorites": {},
        "countryViews": country_views or {},
        "regionViews": region_views or {},
    }


class TestComparisonWinner:
    """Task 10: comparisonWinner identifies the most compared model."""

    def test_comparison_winner_calculated(self, mock_dynamodb):
        """Model with highest comparison count should be the winner."""
        handler = mock_dynamodb["handler"]

        items = [
            _make_daily_item(
                "2025-01-01",
                views=10,
                unique_users={"u1", "u2"},
                compared_models={"model-a": 5, "model-b": 3},
            ),
            _make_daily_item(
                "2025-01-02",
                views=8,
                unique_users={"u2", "u3"},
                compared_models={"model-a": 2, "model-b": 7},
            ),
        ]

        now = datetime(2025, 1, 3, 12, 0, 0, tzinfo=timezone.utc)
        summary, _, _, _ = handler._build_summary(items, now, "2025-01-01")

        # model-b: 3+7=10, model-a: 5+2=7 => model-b wins
        assert summary["comparisonWinner"] is not None
        assert summary["comparisonWinner"]["modelId"] == "model-b"
        assert summary["comparisonWinner"]["comparisons"] == 10
        assert summary["comparisonWinner"]["totalComparisons"] == 17

    def test_comparison_winner_none_when_empty(self, mock_dynamodb):
        """comparisonWinner should be None when no comparison data exists."""
        handler = mock_dynamodb["handler"]

        items = [
            _make_daily_item("2025-01-01", views=5, unique_users={"u1"}),
        ]

        now = datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        summary, _, _, _ = handler._build_summary(items, now, "2025-01-01")

        assert summary["comparisonWinner"] is None


class TestAvgDailyUsers:
    """Task 10: avgDailyUsers calculates average unique users per active day."""

    def test_avg_daily_users_calculated(self, mock_dynamodb):
        """Average of 2 users on day 1 and 4 users on day 2 should be 3.0."""
        handler = mock_dynamodb["handler"]

        items = [
            _make_daily_item("2025-01-01", views=5, unique_users={"u1", "u2"}),
            _make_daily_item(
                "2025-01-02", views=8, unique_users={"u1", "u2", "u3", "u4"}
            ),
        ]

        now = datetime(2025, 1, 3, 12, 0, 0, tzinfo=timezone.utc)
        summary, _, _, _ = handler._build_summary(items, now, "2025-01-01")

        # Day 1: 2 users, Day 2: 4 users => avg = 3.0
        assert summary["avgDailyUsers"] == 3.0

    def test_avg_daily_users_zero_when_no_data(self, mock_dynamodb):
        """avgDailyUsers should be 0 when no items have users."""
        handler = mock_dynamodb["handler"]

        items = [
            _make_daily_item("2025-01-01", views=0),
        ]

        now = datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        summary, _, _, _ = handler._build_summary(items, now, "2025-01-01")

        assert summary["avgDailyUsers"] == 0.0


class TestViewsPerUser:
    """Task 10: viewsPerUser engagement metric."""

    def test_views_per_user_calculated(self, mock_dynamodb):
        """20 views across 4 unique users should give 5.0 views per user."""
        handler = mock_dynamodb["handler"]

        items = [
            _make_daily_item("2025-01-01", views=12, unique_users={"u1", "u2"}),
            _make_daily_item("2025-01-02", views=8, unique_users={"u3", "u4"}),
        ]

        now = datetime(2025, 1, 3, 12, 0, 0, tzinfo=timezone.utc)
        summary, _, _, _ = handler._build_summary(items, now, "2025-01-01")

        # 20 views / 4 unique users = 5.0
        assert summary["viewsPerUser"] == 5.0

    def test_views_per_user_zero_users(self, mock_dynamodb):
        """viewsPerUser should handle zero users gracefully."""
        handler = mock_dynamodb["handler"]

        items = [
            _make_daily_item("2025-01-01", views=10),
        ]

        now = datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        summary, _, _, _ = handler._build_summary(items, now, "2025-01-01")

        # 10 views / max(0, 1) = 10.0
        assert summary["viewsPerUser"] == 10.0


class TestMostActiveSection:
    """Task 10: mostActiveSection identifies the section with most activity."""

    def test_most_active_section(self, mock_dynamodb):
        """Section with highest total count should be identified."""
        handler = mock_dynamodb["handler"]

        items = [
            _make_daily_item(
                "2025-01-01",
                views=10,
                sections={"overview": 5, "pricing": 3, "regions": 2},
            ),
            _make_daily_item(
                "2025-01-02",
                views=8,
                sections={"overview": 2, "pricing": 6},
            ),
        ]

        now = datetime(2025, 1, 3, 12, 0, 0, tzinfo=timezone.utc)
        summary, _, _, _ = handler._build_summary(items, now, "2025-01-01")

        # pricing: 3+6=9, overview: 5+2=7, regions: 2 => pricing wins
        assert summary["mostActiveSection"] == "pricing"

    def test_most_active_section_none_when_empty(self, mock_dynamodb):
        """mostActiveSection should be None when no section data exists."""
        handler = mock_dynamodb["handler"]

        items = [
            _make_daily_item("2025-01-01", views=5),
        ]

        now = datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        summary, _, _, _ = handler._build_summary(items, now, "2025-01-01")

        assert summary["mostActiveSection"] is None


class TestTotalRegions:
    """Task 10: totalRegions count in summary."""

    def test_total_regions_counted(self, mock_dynamodb):
        """totalRegions should reflect the union of all regions across days."""
        handler = mock_dynamodb["handler"]

        items = [
            _make_daily_item(
                "2025-01-01",
                views=5,
                regions={"us-east-1", "eu-west-1"},
            ),
            _make_daily_item(
                "2025-01-02",
                views=3,
                regions={"us-east-1", "ap-southeast-1"},
            ),
        ]

        now = datetime(2025, 1, 3, 12, 0, 0, tzinfo=timezone.utc)
        summary, _, _, _ = handler._build_summary(items, now, "2025-01-01")

        # Union: us-east-1, eu-west-1, ap-southeast-1 = 3
        assert summary["totalRegions"] == 3


class TestRegionCountsInResponse:
    """Task 10: regionCounts includes actual view counts per region."""

    def test_region_counts_in_response(self, mock_dynamodb):
        """regionCounts should include region view counts from regionViews map."""
        handler = mock_dynamodb["handler"]

        items = [
            _make_daily_item(
                "2025-01-01",
                views=50,
                region_views={"us-east-1": 30, "eu-west-1": 20},
                regions={"us-east-1", "eu-west-1"},
            ),
            _make_daily_item(
                "2025-01-02",
                views=25,
                region_views={"us-east-1": 20, "ap-southeast-1": 5},
                regions={"us-east-1", "ap-southeast-1"},
            ),
        ]

        now = datetime(2025, 1, 3, 12, 0, 0, tzinfo=timezone.utc)
        summary, _, _, _ = handler._build_summary(items, now, "2025-01-01")

        # us-east-1: 30+20=50, eu-west-1: 20, ap-southeast-1: 5
        region_counts = {r["id"]: r["count"] for r in summary["regionCounts"]}
        assert region_counts["us-east-1"] == 50
        assert region_counts["eu-west-1"] == 20
        assert region_counts["ap-southeast-1"] == 5


# ═══════════════════════════════════════════════════════════════════════════
# Task 09 — Dashboard API: Cognito Data Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestDashboardIncludesCognitoKey:
    """Task 09: Dashboard response includes 'cognito' key with Cognito data."""

    def test_dashboard_includes_cognito_key(self, mock_dynamodb):
        """GET /dashboard response should have a 'cognito' key."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        # Arrange: mock table.query to return empty items for aggregates and cognito
        table.query.return_value = {"Items": []}

        # Mock the Cognito client to return a user count
        mock_cognito = MagicMock()
        mock_cognito.describe_user_pool.return_value = {
            "UserPool": {"EstimatedNumberOfUsers": 42}
        }

        with (
            patch.object(handler, "_cognito_client", mock_cognito),
            patch.object(handler, "USER_POOL_ID", "us-east-1_TestPool"),
        ):
            result = handler.handle_get_dashboard(_make_dashboard_event())

        body = json.loads(result["body"])
        assert result["statusCode"] == 200
        assert "cognito" in body, "Dashboard response must include 'cognito' key"
        assert "loggedInUsers" in body["cognito"]


class TestCognitoLoggedInUsers:
    """Task 09: loggedInUsers reflects Cognito EstimatedNumberOfUsers."""

    def test_cognito_logged_in_users(self, mock_dynamodb):
        """loggedInUsers should equal EstimatedNumberOfUsers from describe_user_pool."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        table.query.return_value = {"Items": []}

        mock_cognito = MagicMock()
        mock_cognito.describe_user_pool.return_value = {
            "UserPool": {"EstimatedNumberOfUsers": 150}
        }

        with (
            patch.object(handler, "_cognito_client", mock_cognito),
            patch.object(handler, "USER_POOL_ID", "us-east-1_TestPool"),
        ):
            result = handler.handle_get_dashboard(_make_dashboard_event())

        body = json.loads(result["body"])
        assert body["cognito"]["loggedInUsers"] == 150


class TestCognitoSummaryAggregation:
    """Task 09: COGNITO#SUMMARY records are aggregated correctly."""

    def test_cognito_summary_aggregation(self, mock_dynamodb):
        """3 COGNITO#SUMMARY records should produce correct totals."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        # Arrange: table.query is called multiple times:
        #   1. current period aggregates (AGG#daily)
        #   2. previous period aggregates (AGG#daily)
        #   3. COGNITO#SUMMARY query
        cognito_items = [
            {
                "PK": "COGNITO#SUMMARY",
                "SK": "2025-01-01",
                "newUsersToday": 5,
                "returningUsersToday": 10,
                "totalUsers": 100,
                "usersByCountry": {"US": 3, "DE": 2},
            },
            {
                "PK": "COGNITO#SUMMARY",
                "SK": "2025-01-02",
                "newUsersToday": 3,
                "returningUsersToday": 12,
                "totalUsers": 103,
                "usersByCountry": {"US": 2, "GB": 1},
            },
            {
                "PK": "COGNITO#SUMMARY",
                "SK": "2025-01-03",
                "newUsersToday": 2,
                "returningUsersToday": 15,
                "totalUsers": 105,
                "usersByCountry": {"US": 1, "DE": 1},
            },
        ]

        # Mock query to return empty for AGG#daily (current + previous), then cognito items
        # handle_get_dashboard calls:
        #   1. _query_aggregates(current) → table.query (AGG#daily)
        #   2. _query_aggregates(previous) → table.query (AGG#daily)
        #   3. _get_hourly_series(today) → table.query (SESSION#)
        #   4. _get_cognito_summary → table.query (COGNITO#SUMMARY)
        table.query.side_effect = [
            {"Items": []},  # current period aggregates
            {"Items": []},  # previous period aggregates
            {"Items": []},  # hourly series for today
            {"Items": cognito_items},  # cognito summary
        ]

        mock_cognito = MagicMock()
        mock_cognito.describe_user_pool.return_value = {
            "UserPool": {"EstimatedNumberOfUsers": 105}
        }

        with (
            patch.object(handler, "_cognito_client", mock_cognito),
            patch.object(handler, "USER_POOL_ID", "us-east-1_TestPool"),
        ):
            result = handler.handle_get_dashboard(_make_dashboard_event())

        body = json.loads(result["body"])
        cognito = body["cognito"]

        # Verify aggregation: newUsersInPeriod = 5+3+2 = 10
        assert cognito["newUsersInPeriod"] == 10
        # returningUsersInPeriod = 10+12+15 = 37
        assert cognito["returningUsersInPeriod"] == 37
        # totalRegistered = max(100, 103, 105) = 105
        assert cognito["totalRegistered"] == 105


class TestCognitoNoPoolIdReturnsZero:
    """Task 09: When USER_POOL_ID is empty, loggedInUsers returns 0."""

    def test_cognito_no_pool_id_returns_zero(self, mock_dynamodb):
        """Empty USER_POOL_ID should result in loggedInUsers == 0."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        table.query.return_value = {"Items": []}

        # Ensure USER_POOL_ID is empty (graceful fallback)
        with patch.object(handler, "USER_POOL_ID", ""):
            result = handler.handle_get_dashboard(_make_dashboard_event())

        body = json.loads(result["body"])
        assert body["cognito"]["loggedInUsers"] == 0


class TestCognitoCountriesAggregated:
    """Task 09: Cognito country data is aggregated across days."""

    def test_cognito_countries_aggregated(self, mock_dynamodb):
        """2 days with overlapping countries should sum user counts correctly."""
        handler = mock_dynamodb["handler"]

        # Test _build_cognito_summary directly for isolation
        cognito_items = [
            {
                "PK": "COGNITO#SUMMARY",
                "SK": "2025-01-01",
                "newUsersToday": 3,
                "returningUsersToday": 5,
                "totalUsers": 50,
                "usersByCountry": {"US": 10, "DE": 5},
            },
            {
                "PK": "COGNITO#SUMMARY",
                "SK": "2025-01-02",
                "newUsersToday": 2,
                "returningUsersToday": 8,
                "totalUsers": 52,
                "usersByCountry": {"US": 8, "GB": 3},
            },
        ]

        result = handler._build_cognito_summary(cognito_items)

        # US: 10+8=18, DE: 5, GB: 3
        countries = {c["id"]: c["count"] for c in result["usersByCountry"]}
        assert countries["US"] == 18
        assert countries["DE"] == 5
        assert countries["GB"] == 3

    def test_cognito_countries_empty_when_no_data(self, mock_dynamodb):
        """Empty cognito items should return empty country list."""
        handler = mock_dynamodb["handler"]

        result = handler._build_cognito_summary([])
        assert result["usersByCountry"] == []
        assert result["totalRegistered"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Task 11 — Frontend Utility Function Validation (Python Equivalents)
# ═══════════════════════════════════════════════════════════════════════════


class TestMergeCountryData:
    """Task 11: Validate mergeCountryData logic (Python equivalent)."""

    @staticmethod
    def _merge_country_data(analytics_counts, cognito_countries):
        """Python equivalent of frontend mergeCountryData function."""
        merged = {}

        for item in analytics_counts:
            cid = item["id"]
            if cid not in merged:
                merged[cid] = {"id": cid, "views": 0, "users": 0}
            merged[cid]["views"] = item["count"]

        for item in cognito_countries:
            cid = item["id"]
            if cid not in merged:
                merged[cid] = {"id": cid, "views": 0, "users": 0}
            merged[cid]["users"] = item["count"]

        return sorted(
            merged.values(),
            key=lambda x: x["views"] + x["users"],
            reverse=True,
        )

    def test_merge_country_data(self):
        """Merge analytics views with cognito users per country."""
        analytics = [{"id": "US", "count": 5}]
        cognito = [{"id": "US", "count": 3}, {"id": "DE", "count": 2}]

        result = self._merge_country_data(analytics, cognito)

        result_dict = {r["id"]: r for r in result}
        assert result_dict["US"]["views"] == 5
        assert result_dict["US"]["users"] == 3
        assert result_dict["DE"]["views"] == 0
        assert result_dict["DE"]["users"] == 2

    def test_merge_country_data_empty_inputs(self):
        """Empty inputs should return empty list."""
        result = self._merge_country_data([], [])
        assert result == []

    def test_merge_country_data_analytics_only(self):
        """Analytics-only data should have zero users."""
        analytics = [{"id": "JP", "count": 10}]
        result = self._merge_country_data(analytics, [])

        assert len(result) == 1
        assert result[0]["id"] == "JP"
        assert result[0]["views"] == 10
        assert result[0]["users"] == 0


class TestGetWinnerDisplay:
    """Task 11: Validate getWinnerDisplay logic (Python equivalent)."""

    @staticmethod
    def _get_winner_display(winner):
        """Python equivalent of frontend getWinnerDisplay function."""
        if not winner:
            return None

        model_id = winner["modelId"]
        # Extract short name: 'anthropic.claude-3-sonnet' → 'Claude 3 Sonnet'
        raw_name = model_id.split(".")[-1] if "." in model_id else model_id
        display_name = " ".join(
            word.capitalize() for word in raw_name.replace("-", " ").split()
        )

        total = winner.get("totalComparisons", 0)
        percentage = round((winner["comparisons"] / total) * 100) if total > 0 else 0

        return {
            "modelId": model_id,
            "displayName": display_name,
            "count": winner["comparisons"],
            "total": total,
            "percentage": percentage,
        }

    def test_get_winner_display(self):
        """Winner formatting should extract display name and calculate percentage."""
        winner = {
            "modelId": "anthropic.claude-3-sonnet",
            "comparisons": 10,
            "totalComparisons": 50,
        }

        result = self._get_winner_display(winner)

        assert result is not None
        assert "Claude" in result["displayName"]
        assert "Sonnet" in result["displayName"]
        assert result["percentage"] == 20
        assert result["count"] == 10
        assert result["total"] == 50

    def test_get_winner_display_none(self):
        """None winner should return None."""
        result = self._get_winner_display(None)
        assert result is None

    def test_get_winner_display_zero_total(self):
        """Zero totalComparisons should produce 0 percentage."""
        winner = {
            "modelId": "amazon.titan-text-express",
            "comparisons": 5,
            "totalComparisons": 0,
        }

        result = self._get_winner_display(winner)
        assert result is not None
        assert result["percentage"] == 0


class TestFmtPctZeroTotal:
    """Task 11: Validate fmtPct zero-division handling (Python equivalent)."""

    @staticmethod
    def _fmt_pct(value, total):
        """Python equivalent of frontend fmtPct function."""
        if not total or total == 0:
            return "0%"
        return f"{round((value / total) * 100)}%"

    def test_fmt_pct_zero_total(self):
        """fmtPct(5, 0) should return '0%' (no division by zero)."""
        assert self._fmt_pct(5, 0) == "0%"

    def test_fmt_pct_none_total(self):
        """fmtPct(5, None) should return '0%'."""
        assert self._fmt_pct(5, None) == "0%"

    def test_fmt_pct_normal(self):
        """fmtPct(25, 100) should return '25%'."""
        assert self._fmt_pct(25, 100) == "25%"

    def test_fmt_pct_rounding(self):
        """fmtPct(1, 3) should round to '33%'."""
        assert self._fmt_pct(1, 3) == "33%"


# ═══════════════════════════════════════════════════════════════════════════
# Tasks 13-16 — Phase 4: Dashboard Response Shape Tests (All 4 Tabs)
# ═══════════════════════════════════════════════════════════════════════════


def _make_full_daily_items():
    """Build a realistic set of daily aggregate items for full dashboard tests."""
    return [
        _make_daily_item(
            "2025-01-01",
            views=50,
            unique_users={"u1", "u2", "u3"},
            new_users={"u1", "u2"},
            sections={"overview": 20, "pricing": 15, "regions": 10, "favorites": 5},
            features={"modelDetails": 8, "comparisons": 5, "favorites": 3},
            compared_models={"anthropic.claude-3-sonnet": 3, "meta.llama3-70b": 2},
            country_views={"US": 30, "DE": 15, "GB": 5},
            region_views={"us-east-1": 25, "eu-west-1": 20, "ap-southeast-1": 5},
            countries={"US", "DE", "GB"},
            regions={"us-east-1", "eu-west-1", "ap-southeast-1"},
        ),
        _make_daily_item(
            "2025-01-02",
            views=40,
            unique_users={"u2", "u3", "u4"},
            new_users={"u4"},
            sections={"overview": 15, "pricing": 10, "regions": 8, "favorites": 7},
            features={"modelDetails": 6, "comparisons": 4, "favorites": 5},
            compared_models={
                "anthropic.claude-3-sonnet": 2,
                "amazon.titan-text-express": 2,
            },
            country_views={"US": 20, "JP": 10, "DE": 10},
            region_views={"us-east-1": 20, "eu-west-1": 10, "ap-northeast-1": 10},
            countries={"US", "JP", "DE"},
            regions={"us-east-1", "eu-west-1", "ap-northeast-1"},
        ),
    ]


class TestDashboardResponseOverviewFields:
    """Task 13: Overview tab requires totalViews, uniqueUsers, newUsers,
    returningUsers, activeToday, viewsPerUser, avgDailyUsers, mostActiveSection."""

    def test_dashboard_response_has_overview_fields(self, mock_dynamodb):
        """Response summary must contain all fields consumed by the Overview tab."""
        handler = mock_dynamodb["handler"]

        items = _make_full_daily_items()
        now = datetime(2025, 1, 2, 14, 0, 0, tzinfo=timezone.utc)
        summary, _, _, _ = handler._build_summary(items, now, "2025-01-01")

        overview_fields = [
            "totalViews",
            "uniqueUsers",
            "newUsers",
            "returningUsers",
            "activeToday",
            "viewsPerUser",
            "avgDailyUsers",
            "mostActiveSection",
        ]
        for field in overview_fields:
            assert field in summary, (
                f"Overview tab field '{field}' missing from summary"
            )

        # Verify values are reasonable
        assert summary["totalViews"] == 90  # 50 + 40
        assert summary["uniqueUsers"] == 4  # u1, u2, u3, u4
        assert summary["newUsers"] == 3  # u1, u2, u4
        assert isinstance(summary["viewsPerUser"], (int, float))
        assert isinstance(summary["avgDailyUsers"], (int, float))
        assert summary["mostActiveSection"] is not None


class TestDashboardResponseAudienceFields:
    """Task 14: Audience tab requires cognito data + countryCounts + avgDailyUsers."""

    def test_dashboard_response_has_audience_fields(self, mock_dynamodb):
        """Full dashboard response must include cognito and country data for Audience tab."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        # Set up cognito summary items
        cognito_items = [
            {
                "PK": "COGNITO#SUMMARY",
                "SK": "2025-01-01",
                "newUsersToday": 5,
                "returningUsersToday": 10,
                "totalUsers": 100,
                "usersByCountry": {"US": 8, "DE": 3},
            },
            {
                "PK": "COGNITO#SUMMARY",
                "SK": "2025-01-02",
                "newUsersToday": 3,
                "returningUsersToday": 12,
                "totalUsers": 103,
                "usersByCountry": {"US": 5, "GB": 2},
            },
        ]

        # Mock query: current AGG, previous AGG, hourly SESSION, COGNITO#SUMMARY
        table.query.side_effect = [
            {"Items": _make_full_daily_items()},  # current period
            {"Items": []},  # previous period
            {"Items": []},  # hourly series
            {"Items": cognito_items},  # cognito summary
        ]

        mock_cognito = MagicMock()
        mock_cognito.describe_user_pool.return_value = {
            "UserPool": {"EstimatedNumberOfUsers": 103}
        }

        with (
            patch.object(handler, "_cognito_client", mock_cognito),
            patch.object(handler, "USER_POOL_ID", "us-east-1_TestPool"),
        ):
            result = handler.handle_get_dashboard(_make_dashboard_event())

        body = json.loads(result["body"])
        assert result["statusCode"] == 200

        # Cognito fields
        cognito = body["cognito"]
        assert "loggedInUsers" in cognito
        assert "totalRegistered" in cognito
        assert "newUsersInPeriod" in cognito
        assert "returningUsersInPeriod" in cognito
        assert "usersByCountry" in cognito
        assert cognito["loggedInUsers"] == 103

        # Country counts from analytics
        summary = body["summary"]
        assert "countryCounts" in summary
        assert isinstance(summary["countryCounts"], list)

        # avgDailyUsers in summary
        assert "avgDailyUsers" in summary


class TestDashboardResponseContentFields:
    """Task 15: Content tab requires comparisonWinner, regionCounts, totalRegions,
    topModels, topComparedModels, topFavoritedModels, providerComparisons, providerFavorites."""

    def test_dashboard_response_has_content_fields(self, mock_dynamodb):
        """Response summary must contain all fields consumed by the Content tab."""
        handler = mock_dynamodb["handler"]

        items = _make_full_daily_items()
        now = datetime(2025, 1, 2, 14, 0, 0, tzinfo=timezone.utc)
        summary, _, _, _ = handler._build_summary(items, now, "2025-01-01")

        content_fields = [
            "comparisonWinner",
            "regionCounts",
            "totalRegions",
            "topModels",
            "topComparedModels",
            "topFavoritedModels",
            "providerComparisons",
            "providerFavorites",
        ]
        for field in content_fields:
            assert field in summary, f"Content tab field '{field}' missing from summary"

        # comparisonWinner should be the model with most comparisons
        assert summary["comparisonWinner"] is not None
        assert "modelId" in summary["comparisonWinner"]
        assert "comparisons" in summary["comparisonWinner"]
        assert "totalComparisons" in summary["comparisonWinner"]

        # regionCounts should be a list of {id, count}
        assert isinstance(summary["regionCounts"], list)
        assert len(summary["regionCounts"]) > 0
        assert "id" in summary["regionCounts"][0]
        assert "count" in summary["regionCounts"][0]

        # totalRegions should reflect union of all regions
        assert (
            summary["totalRegions"] == 4
        )  # us-east-1, eu-west-1, ap-southeast-1, ap-northeast-1


class TestDashboardResponseRealtimeFields:
    """Task 16: Realtime tab requires hourlySeries with views/events/uniqueUsers per hour."""

    def test_dashboard_response_has_realtime_fields(self, mock_dynamodb):
        """Full dashboard response must include hourlySeries for the Realtime tab."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        # Mock session bucket items for today's hourly data
        session_items = [
            {
                "PK": "SESSION#2025-01-02",
                "SK": "10:00#user-a",
                "views": 5,
                "events": 8,
                "auid": "user-a",
            },
            {
                "PK": "SESSION#2025-01-02",
                "SK": "10:05#user-b",
                "views": 3,
                "events": 4,
                "auid": "user-b",
            },
            {
                "PK": "SESSION#2025-01-02",
                "SK": "14:00#user-a",
                "views": 2,
                "events": 3,
                "auid": "user-a",
            },
        ]

        # Mock query: current AGG, previous AGG, hourly SESSION, COGNITO#SUMMARY
        table.query.side_effect = [
            {"Items": _make_full_daily_items()},  # current period
            {"Items": []},  # previous period
            {"Items": session_items},  # hourly series for today
            {"Items": []},  # cognito summary
        ]

        with patch.object(handler, "USER_POOL_ID", ""):
            result = handler.handle_get_dashboard(_make_dashboard_event())

        body = json.loads(result["body"])
        assert result["statusCode"] == 200

        # hourlySeries should be present and have 24 entries
        assert "hourlySeries" in body
        hourly = body["hourlySeries"]
        assert len(hourly) == 24

        # Each entry should have hour, views, events, uniqueUsers
        for entry in hourly:
            assert "hour" in entry
            assert "views" in entry
            assert "events" in entry
            assert "uniqueUsers" in entry

        # Verify hour 10 has aggregated data from both session items
        hour_10 = next(h for h in hourly if h["hour"] == "10:00")
        assert hour_10["views"] == 8  # 5 + 3
        assert hour_10["events"] == 12  # 8 + 4
        assert hour_10["uniqueUsers"] == 2  # user-a, user-b


class TestDashboardFullResponseSerializable:
    """Task 13-16: Full dashboard response must be JSON-serializable (no Decimal, set errors)."""

    def test_dashboard_full_response_serializable(self, mock_dynamodb):
        """Full dashboard response should serialize to JSON without errors."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]

        # Use items with Decimal values (as DynamoDB returns)
        items_with_decimals = [
            {
                "PK": "AGG#daily",
                "SK": "2025-01-01",
                "views": Decimal("50"),
                "uniqueUsers": {"u1", "u2"},
                "newUsers": {"u1"},
                "countries": {"US"},
                "regions": {"us-east-1"},
                "sections": {"overview": Decimal("20")},
                "features": {
                    "modelDetails": Decimal("5"),
                    "comparisons": Decimal("3"),
                    "favorites": Decimal("2"),
                },
                "topModels": {"model-a": Decimal("10")},
                "comparedModels": {"model-a": Decimal("3")},
                "favoritedModels": {"model-a": Decimal("2")},
                "providerComparisons": {"Anthropic": Decimal("3")},
                "providerFavorites": {"Anthropic": Decimal("2")},
                "countryViews": {"US": Decimal("50")},
                "regionViews": {"us-east-1": Decimal("50")},
            },
        ]

        # Mock query: current AGG, previous AGG, hourly SESSION, COGNITO#SUMMARY
        table.query.side_effect = [
            {"Items": items_with_decimals},  # current period
            {"Items": []},  # previous period
            {"Items": []},  # hourly series
            {"Items": []},  # cognito summary
        ]

        with patch.object(handler, "USER_POOL_ID", ""):
            result = handler.handle_get_dashboard(_make_dashboard_event())

        # The response body should already be JSON (handler calls json.dumps)
        assert result["statusCode"] == 200
        body_str = result["body"]

        # Verify it's valid JSON (no Decimal/set serialization errors)
        parsed = json.loads(body_str)
        assert isinstance(parsed, dict)

        # Double-check by re-serializing (standard json, no custom serializer)
        re_serialized = json.dumps(parsed)
        assert isinstance(re_serialized, str)


# ═══════════════════════════════════════════════════════════════════════════
# Task 17 — Performance Optimization Tests (Cognito Cache for Returning Users)
# ═══════════════════════════════════════════════════════════════════════════


class TestReturningUsersCognitoCache:
    """Task 17: _count_returning_users uses Cognito cache for large user sets."""

    def test_returning_users_uses_cognito_cache_for_large_sets(self, mock_dynamodb):
        """600 unique users with COGNITO#SUMMARY cache should use cached value."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]
        resource = mock_dynamodb["resource"]

        # Arrange: 600 unique user IDs
        large_user_set = {f"user-{i}" for i in range(600)}

        # Mock table.get_item to return a cached COGNITO#SUMMARY with returningUsersToday
        table.get_item.return_value = {"Item": {"returningUsersToday": 450}}

        # Act
        result = handler._count_returning_users(large_user_set, "2025-01-01")

        # Assert: should use cached value (450), not batch_get_item
        assert result == 450

        # batch_get_item should NOT have been called (cache hit)
        resource.meta.client.batch_get_item.assert_not_called()

    def test_returning_users_fallback_for_small_sets(self, mock_dynamodb):
        """50 unique users should use batch_get_item directly (no cache check)."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]
        resource = mock_dynamodb["resource"]

        # Arrange: 50 unique user IDs
        small_user_set = {f"user-{i}" for i in range(50)}

        # Mock batch_get_item to return users with firstSeen before start_date
        resource.meta.client.batch_get_item.return_value = {
            "Responses": {
                "test-analytics-table": [{"firstSeen": "2024-12-01"} for _ in range(20)]
                + [{"firstSeen": "2025-01-15"} for _ in range(30)]
            }
        }

        # Act
        result = handler._count_returning_users(small_user_set, "2025-01-01")

        # Assert: should use batch_get_item (20 users with firstSeen before start)
        assert result == 20

        # table.get_item should NOT have been called for cache (set too small)
        # Note: get_item is called in the fixture setup, so we check the call args
        cache_calls = [
            c
            for c in table.get_item.call_args_list
            if c.kwargs.get("Key", {}).get("PK") == "COGNITO#SUMMARY"
        ]
        assert len(cache_calls) == 0

    def test_returning_users_fallback_on_cache_miss(self, mock_dynamodb):
        """600 users with no COGNITO#SUMMARY should fall back to batch_get_item."""
        handler = mock_dynamodb["handler"]
        table = mock_dynamodb["table"]
        resource = mock_dynamodb["resource"]

        # Arrange: 600 unique user IDs
        large_user_set = {f"user-{i}" for i in range(600)}

        # Mock table.get_item to return no cached item (cache miss)
        table.get_item.return_value = {"Item": {}}

        # Mock batch_get_item to return users (in batches of 100)
        resource.meta.client.batch_get_item.return_value = {
            "Responses": {
                "test-analytics-table": [{"firstSeen": "2024-11-01"} for _ in range(30)]
                + [{"firstSeen": "2025-02-01"} for _ in range(70)]
            }
        }

        # Act
        result = handler._count_returning_users(large_user_set, "2025-01-01")

        # Assert: should have fallen back to batch_get_item
        # 600 users / 100 per batch = 6 batch calls, each returning 30 returning
        assert result == 30 * 6  # 30 returning per batch × 6 batches
        assert resource.meta.client.batch_get_item.call_count == 6
