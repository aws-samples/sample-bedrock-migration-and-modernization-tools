"""
Tests for Cognito sync Lambda — Task 04.

Covers:
  - Skipping when USER_POOL_ID is empty
  - Empty user pool handling
  - Confirmed-only user counting
  - New user detection by create date
  - Returning user detection by lastModified
  - Country extraction from locale and custom attribute
  - Summary record written to DynamoDB
  - Individual user records written via batch_writer
  - Disabled user exclusion

Run:
    cd backend && python3 -m pytest tests/test_cognito_sync.py -v

Requires: pytest
    pip install pytest
"""

import importlib
import importlib.util
import os
import sys
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Path setup — use importlib to avoid name collisions with other handler.py
# ---------------------------------------------------------------------------
LAMBDA_DIR = os.path.join(os.path.dirname(__file__), "..", "lambdas", "cognito-sync")
_HANDLER_PATH = os.path.join(LAMBDA_DIR, "handler.py")
_MODULE_NAME = "cognito_sync_handler"  # unique name to avoid collision


def _import_cognito_sync_handler():
    """Import the cognito-sync handler module using importlib to avoid name collisions."""
    if _MODULE_NAME in sys.modules:
        del sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _HANDLER_PATH)
    assert spec is not None, f"Could not find module spec for {_HANDLER_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    assert spec.loader is not None, f"Module spec has no loader for {_HANDLER_PATH}"
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    """Set environment variables before every test."""
    monkeypatch.setenv("ANALYTICS_TABLE", "test-analytics-table")
    monkeypatch.setenv("USER_POOL_ID", "us-east-1_TestPool")
    monkeypatch.setenv("COGNITO_REGION", "us-east-1")


@pytest.fixture()
def mock_services():
    """Patch boto3 so the handler gets mock Cognito + DynamoDB."""
    mock_table = MagicMock()
    mock_table.put_item = MagicMock()

    # batch_writer context manager
    mock_batch = MagicMock()
    mock_batch.__enter__ = MagicMock(return_value=mock_batch)
    mock_batch.__exit__ = MagicMock(return_value=False)
    mock_batch.put_item = MagicMock()
    mock_table.batch_writer.return_value = mock_batch

    mock_resource = MagicMock()
    mock_resource.Table.return_value = mock_table

    mock_cognito = MagicMock()
    mock_cognito.describe_user_pool.return_value = {
        "UserPool": {"EstimatedNumberOfUsers": 0}
    }
    mock_cognito.list_users.return_value = {"Users": []}

    with (
        patch("boto3.resource", return_value=mock_resource),
        patch("boto3.client", return_value=mock_cognito),
    ):
        handler = _import_cognito_sync_handler()

        yield {
            "handler": handler,
            "table": mock_table,
            "batch": mock_batch,
            "resource": mock_resource,
            "cognito": mock_cognito,
        }


def _make_cognito_user(
    sub,
    status="CONFIRMED",
    enabled=True,
    created=None,
    last_modified=None,
    locale="",
    custom_country="",
):
    """Build a Cognito ListUsers response user object."""
    now = datetime.now(timezone.utc)
    if created is None:
        created = now - timedelta(days=30)
    if last_modified is None:
        last_modified = now - timedelta(days=1)

    attrs = [{"Name": "sub", "Value": sub}]
    if locale:
        attrs.append({"Name": "locale", "Value": locale})
    if custom_country:
        attrs.append({"Name": "custom:country", "Value": custom_country})

    return {
        "Username": f"user-{sub}",
        "Attributes": attrs,
        "UserStatus": status,
        "Enabled": enabled,
        "UserCreateDate": created,
        "UserLastModifiedDate": last_modified,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCognitoSyncSkip:
    """Handler should skip when USER_POOL_ID is not configured."""

    def test_sync_skips_without_pool_id(self, mock_services, monkeypatch):
        """Empty USER_POOL_ID should return SKIPPED status."""
        monkeypatch.setenv("USER_POOL_ID", "")
        handler = mock_services["handler"]

        # Re-set the module-level variable to empty
        handler.USER_POOL_ID = ""

        result = handler.lambda_handler({}, None)
        assert result["status"] == "SKIPPED"


class TestCognitoSyncEmptyPool:
    """Handler should handle an empty user pool gracefully."""

    def test_sync_empty_pool(self, mock_services):
        """Empty pool should return totalUsers == 0."""
        handler = mock_services["handler"]
        cognito = mock_services["cognito"]

        cognito.describe_user_pool.return_value = {
            "UserPool": {"EstimatedNumberOfUsers": 0}
        }
        cognito.list_users.return_value = {"Users": []}

        result = handler.lambda_handler({}, None)
        assert result["status"] == "SUCCESS"
        assert result["totalUsers"] == 0


class TestCognitoSyncConfirmedUsers:
    """Only CONFIRMED users should be counted."""

    def test_sync_counts_confirmed_users(self, mock_services):
        """5 users: 3 CONFIRMED, 2 UNCONFIRMED => totalUsers == 3."""
        handler = mock_services["handler"]
        cognito = mock_services["cognito"]

        now = datetime.now(timezone.utc)
        users = [
            _make_cognito_user(
                "u1", status="CONFIRMED", created=now - timedelta(days=10)
            ),
            _make_cognito_user(
                "u2", status="CONFIRMED", created=now - timedelta(days=5)
            ),
            _make_cognito_user(
                "u3", status="CONFIRMED", created=now - timedelta(days=3)
            ),
            _make_cognito_user(
                "u4", status="UNCONFIRMED", created=now - timedelta(days=2)
            ),
            _make_cognito_user(
                "u5", status="FORCE_CHANGE_PASSWORD", created=now - timedelta(days=1)
            ),
        ]

        cognito.describe_user_pool.return_value = {
            "UserPool": {"EstimatedNumberOfUsers": 5}
        }
        cognito.list_users.return_value = {"Users": users}

        result = handler.lambda_handler({}, None)
        # totalUsers in the return value is len(users) from _list_all_users (all 5)
        # but the summary's totalUsers should be 3 (confirmed only)
        # The handler returns len(users) which is all users listed, not just confirmed
        # Let's check the summary written to DynamoDB instead
        table = mock_services["table"]
        put_calls = table.put_item.call_args_list
        assert len(put_calls) >= 1

        # The summary item should have totalUsers == 3
        summary_item = put_calls[0].kwargs.get("Item", {})
        assert summary_item["totalUsers"] == 3

    def test_sync_excludes_disabled_users(self, mock_services):
        """2 enabled + 1 disabled => totalUsers == 2 in summary."""
        handler = mock_services["handler"]
        cognito = mock_services["cognito"]

        now = datetime.now(timezone.utc)
        users = [
            _make_cognito_user("u1", enabled=True, created=now - timedelta(days=10)),
            _make_cognito_user("u2", enabled=True, created=now - timedelta(days=5)),
            _make_cognito_user("u3", enabled=False, created=now - timedelta(days=3)),
        ]

        cognito.describe_user_pool.return_value = {
            "UserPool": {"EstimatedNumberOfUsers": 3}
        }
        cognito.list_users.return_value = {"Users": users}

        result = handler.lambda_handler({}, None)

        table = mock_services["table"]
        put_calls = table.put_item.call_args_list
        summary_item = put_calls[0].kwargs.get("Item", {})
        assert summary_item["totalUsers"] == 2


class TestCognitoSyncNewUsers:
    """New users should be detected by create date == today."""

    def test_sync_detects_new_users(self, mock_services):
        """2 users created today => newUsersToday == 2."""
        handler = mock_services["handler"]
        cognito = mock_services["cognito"]

        now = datetime.now(timezone.utc)
        today = now.replace(hour=6, minute=0, second=0, microsecond=0)

        users = [
            _make_cognito_user("u1", created=today, last_modified=today),
            _make_cognito_user("u2", created=today, last_modified=today),
            _make_cognito_user(
                "u3",
                created=now - timedelta(days=10),
                last_modified=now - timedelta(days=5),
            ),
        ]

        cognito.describe_user_pool.return_value = {
            "UserPool": {"EstimatedNumberOfUsers": 3}
        }
        cognito.list_users.return_value = {"Users": users}

        result = handler.lambda_handler({}, None)

        table = mock_services["table"]
        put_calls = table.put_item.call_args_list
        summary_item = put_calls[0].kwargs.get("Item", {})
        assert summary_item["newUsersToday"] == 2


class TestCognitoSyncReturningUsers:
    """Returning users: created before today, lastModified >= yesterday."""

    def test_sync_detects_returning_users(self, mock_services):
        """3 users created before today, 2 modified yesterday => returningUsersToday == 2."""
        handler = mock_services["handler"]
        cognito = mock_services["cognito"]

        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        users = [
            _make_cognito_user(
                "u1",
                created=now - timedelta(days=30),
                last_modified=yesterday,
            ),
            _make_cognito_user(
                "u2",
                created=now - timedelta(days=20),
                last_modified=yesterday,
            ),
            _make_cognito_user(
                "u3",
                created=now - timedelta(days=10),
                last_modified=now - timedelta(days=5),
            ),
        ]

        cognito.describe_user_pool.return_value = {
            "UserPool": {"EstimatedNumberOfUsers": 3}
        }
        cognito.list_users.return_value = {"Users": users}

        result = handler.lambda_handler({}, None)

        table = mock_services["table"]
        put_calls = table.put_item.call_args_list
        summary_item = put_calls[0].kwargs.get("Item", {})
        assert summary_item["returningUsersToday"] == 2


class TestCognitoSyncCountryExtraction:
    """Country should be extracted from locale or custom:country attribute."""

    def test_sync_extracts_country_from_locale(self, mock_services):
        """User with locale='en-US' => usersByCountry == {'US': 1}."""
        handler = mock_services["handler"]
        cognito = mock_services["cognito"]

        now = datetime.now(timezone.utc)
        users = [
            _make_cognito_user(
                "u1",
                created=now - timedelta(days=10),
                locale="en-US",
            ),
        ]

        cognito.describe_user_pool.return_value = {
            "UserPool": {"EstimatedNumberOfUsers": 1}
        }
        cognito.list_users.return_value = {"Users": users}

        result = handler.lambda_handler({}, None)

        table = mock_services["table"]
        put_calls = table.put_item.call_args_list
        summary_item = put_calls[0].kwargs.get("Item", {})
        assert summary_item["usersByCountry"] == {"US": 1}

    def test_sync_extracts_country_from_custom_attr(self, mock_services):
        """User with custom:country='DE' => usersByCountry == {'DE': 1}."""
        handler = mock_services["handler"]
        cognito = mock_services["cognito"]

        now = datetime.now(timezone.utc)
        users = [
            _make_cognito_user(
                "u1",
                created=now - timedelta(days=10),
                custom_country="DE",
            ),
        ]

        cognito.describe_user_pool.return_value = {
            "UserPool": {"EstimatedNumberOfUsers": 1}
        }
        cognito.list_users.return_value = {"Users": users}

        result = handler.lambda_handler({}, None)

        table = mock_services["table"]
        put_calls = table.put_item.call_args_list
        summary_item = put_calls[0].kwargs.get("Item", {})
        assert summary_item["usersByCountry"] == {"DE": 1}


class TestCognitoSyncDynamoDBWrites:
    """Verify DynamoDB write operations."""

    def test_sync_writes_summary_record(self, mock_services):
        """Summary should be written with PK='COGNITO#SUMMARY'."""
        handler = mock_services["handler"]
        cognito = mock_services["cognito"]

        now = datetime.now(timezone.utc)
        users = [
            _make_cognito_user("u1", created=now - timedelta(days=10)),
            _make_cognito_user("u2", created=now - timedelta(days=5)),
            _make_cognito_user("u3", created=now - timedelta(days=3)),
        ]

        cognito.describe_user_pool.return_value = {
            "UserPool": {"EstimatedNumberOfUsers": 3}
        }
        cognito.list_users.return_value = {"Users": users}

        handler.lambda_handler({}, None)

        table = mock_services["table"]
        put_calls = table.put_item.call_args_list
        assert len(put_calls) >= 1

        summary_item = put_calls[0].kwargs.get("Item", {})
        assert summary_item["PK"] == "COGNITO#SUMMARY"
        assert "totalUsers" in summary_item
        assert "syncTimestamp" in summary_item

    def test_sync_writes_user_records(self, mock_services):
        """3 users should produce 3 batch_writer.put_item calls."""
        handler = mock_services["handler"]
        cognito = mock_services["cognito"]

        now = datetime.now(timezone.utc)
        users = [
            _make_cognito_user("u1", created=now - timedelta(days=10)),
            _make_cognito_user("u2", created=now - timedelta(days=5)),
            _make_cognito_user("u3", created=now - timedelta(days=3)),
        ]

        cognito.describe_user_pool.return_value = {
            "UserPool": {"EstimatedNumberOfUsers": 3}
        }
        cognito.list_users.return_value = {"Users": users}

        handler.lambda_handler({}, None)

        batch = mock_services["batch"]
        assert batch.put_item.call_count == 3

        # Verify each call has PK='COGNITO#USERS'
        for c in batch.put_item.call_args_list:
            item = c.kwargs.get("Item", {})
            assert item["PK"] == "COGNITO#USERS"
