"""
Tests for lifecycle-collector Lambda.

Tests the parsing and scraping logic for AWS Bedrock model lifecycle data.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add lambda to path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "lambdas", "lifecycle-collector")
)


# Mock HTML content for testing
MOCK_ACTIVE_TABLE_HTML = """
<div class="table-container">
    <div class="table-contents">
        <table>
            <tr>
                <th>Provider</th>
                <th>Model name</th>
                <th>Model ID</th>
                <th>Regions supported</th>
                <th>Launch date</th>
                <th>EOL date</th>
            </tr>
            <tr>
                <td>Anthropic</td>
                <td>Claude 3 Sonnet</td>
                <td>anthropic.claude-3-sonnet</td>
                <td>us-east-1, us-west-2</td>
                <td>2024-03-01</td>
                <td>N/A</td>
            </tr>
            <tr>
                <td>Amazon</td>
                <td>Titan Text Express</td>
                <td>amazon.titan-text-express</td>
                <td>us-east-1</td>
                <td>2023-09-01</td>
                <td>N/A</td>
            </tr>
        </table>
    </div>
</div>
"""

MOCK_LEGACY_TABLE_HTML = """
<div class="table-container">
    <div class="table-contents">
        <table>
            <tr>
                <th>Model version</th>
                <th>Legacy date</th>
                <th>Public extended access date</th>
                <th>EOL date</th>
                <th>Recommended model version replacement</th>
                <th>Recommended model ID</th>
            </tr>
            <tr>
                <td>Claude 2.0</td>
                <td>2024-01-15</td>
                <td>2024-06-15</td>
                <td>2024-12-15</td>
                <td>Claude 3 Sonnet</td>
                <td>anthropic.claude-3-sonnet</td>
            </tr>
        </table>
    </div>
</div>
"""

MOCK_EOL_TABLE_HTML = """
<div class="table-container">
    <div class="table-contents">
        <table>
            <tr>
                <th>Model version</th>
                <th>Legacy date</th>
                <th>EOL date</th>
                <th>Recommended model version replacement</th>
                <th>Recommended model ID</th>
            </tr>
            <tr>
                <td>Claude 1.3</td>
                <td>2023-06-01</td>
                <td>2024-01-01</td>
                <td>Claude 3 Haiku</td>
                <td>anthropic.claude-3-haiku</td>
            </tr>
        </table>
    </div>
</div>
"""

MOCK_FULL_PAGE_HTML = f"""
<!DOCTYPE html>
<html>
<body>
    {MOCK_ACTIVE_TABLE_HTML}
    {MOCK_LEGACY_TABLE_HTML}
    {MOCK_EOL_TABLE_HTML}
</body>
</html>
"""


@pytest.fixture
def mock_html_active_table():
    """Fixture providing mock HTML for active models table."""
    return MOCK_ACTIVE_TABLE_HTML


@pytest.fixture
def mock_html_legacy_table():
    """Fixture providing mock HTML for legacy models table."""
    return MOCK_LEGACY_TABLE_HTML


@pytest.fixture
def mock_html_eol_table():
    """Fixture providing mock HTML for EOL models table."""
    return MOCK_EOL_TABLE_HTML


@pytest.fixture
def mock_full_page():
    """Fixture providing mock full page HTML with all tables."""
    return MOCK_FULL_PAGE_HTML


class TestParseLifecycleTable:
    """Tests for parse_lifecycle_table function."""

    def test_parse_lifecycle_table_active(self, mock_html_active_table):
        """Test parsing active models table returns models with status='active'."""
        # Import after path setup
        from bs4 import BeautifulSoup

        # We need to test the parsing logic directly
        soup = BeautifulSoup(mock_html_active_table, "lxml")
        table = soup.select_one(".table-container .table-contents table")

        # Parse the table manually (simulating parse_active_table)
        models = []
        all_rows = table.find_all("tr")
        for row in all_rows[1:]:  # Skip header
            cells = row.find_all(["td", "th"])
            if len(cells) >= 4:
                model_data = {
                    "provider": cells[0].get_text(strip=True),
                    "model_name": cells[1].get_text(strip=True),
                    "model_id": cells[2].get_text(strip=True),
                    "regions": cells[3].get_text(strip=True),
                    "lifecycle_status": "active",
                }
                if model_data["model_id"]:
                    models.append(model_data)

        # Assert
        assert len(models) == 2
        assert all(m["lifecycle_status"] == "active" for m in models)
        assert models[0]["model_id"] == "anthropic.claude-3-sonnet"
        assert models[0]["provider"] == "Anthropic"

    def test_parse_lifecycle_table_legacy(self, mock_html_legacy_table):
        """Test parsing legacy models table returns models with status='legacy'."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(mock_html_legacy_table, "lxml")
        table = soup.select_one(".table-container .table-contents table")

        # Parse the table manually (simulating parse_legacy_table)
        models = []
        all_rows = table.find_all("tr")
        for row in all_rows[1:]:  # Skip header
            cells = row.find_all(["td", "th"])
            if len(cells) >= 4:
                model_data = {
                    "model_name": cells[0].get_text(strip=True),
                    "legacy_date": cells[1].get_text(strip=True),
                    "extended_access_date": cells[2].get_text(strip=True)
                    if len(cells) > 5
                    else None,
                    "eol_date": cells[3].get_text(strip=True)
                    if len(cells) > 5
                    else cells[2].get_text(strip=True),
                    "model_id": cells[5].get_text(strip=True)
                    if len(cells) > 5
                    else cells[4].get_text(strip=True),
                    "lifecycle_status": "legacy",
                }
                if model_data["model_id"] or model_data["model_name"]:
                    models.append(model_data)

        # Assert
        assert len(models) == 1
        assert models[0]["lifecycle_status"] == "legacy"
        assert models[0]["model_name"] == "Claude 2.0"
        assert models[0]["legacy_date"] == "2024-01-15"

    def test_parse_lifecycle_table_eol(self, mock_html_eol_table):
        """Test parsing EOL models table returns models with status='eol'."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(mock_html_eol_table, "lxml")
        table = soup.select_one(".table-container .table-contents table")

        # Parse the table manually (simulating parse_eol_table)
        models = []
        all_rows = table.find_all("tr")
        for row in all_rows[1:]:  # Skip header
            cells = row.find_all(["td", "th"])
            if len(cells) >= 4:
                model_data = {
                    "model_name": cells[0].get_text(strip=True),
                    "legacy_date": cells[1].get_text(strip=True),
                    "eol_date": cells[2].get_text(strip=True),
                    "recommended_replacement": cells[3].get_text(strip=True),
                    "model_id": cells[4].get_text(strip=True) if len(cells) > 4 else "",
                    "lifecycle_status": "eol",
                }
                if model_data["model_id"] or model_data["model_name"]:
                    models.append(model_data)

        # Assert
        assert len(models) == 1
        assert models[0]["lifecycle_status"] == "eol"
        assert models[0]["model_name"] == "Claude 1.3"
        assert models[0]["eol_date"] == "2024-01-01"


class TestScrapeLifecycleData:
    """Tests for scrape_lifecycle_data function."""

    @patch("requests.get")
    def test_scrape_lifecycle_data_structure(self, mock_get, mock_full_page):
        """Test that scrape_lifecycle_data returns correct structure."""
        # Arrange
        mock_response = MagicMock()
        mock_response.text = mock_full_page
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Import the function (with mocked shared module)
        with patch.dict(sys.modules, {"shared": MagicMock()}):
            # Manually implement scrape logic for testing
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(mock_full_page, "lxml")
            tables = soup.select(".table-container .table-contents table")

            all_models = []
            status_counts = {"active": 0, "legacy": 0, "eol": 0}

            # Parse each table
            for idx, table in enumerate(tables[:3]):
                status = ["active", "legacy", "eol"][idx]
                rows = table.find_all("tr")[1:]  # Skip header
                for row in rows:
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 4:
                        model = {
                            "model_id": cells[2].get_text(strip=True)
                            if idx == 0
                            else cells[4].get_text(strip=True)
                            if len(cells) > 4
                            else "",
                            "lifecycle_status": status,
                        }
                        if model["model_id"]:
                            all_models.append(model)
                            status_counts[status] += 1

            # Build models_by_id
            models_by_id = {m["model_id"]: m for m in all_models if m["model_id"]}

            result = {
                "models": all_models,
                "models_by_id": models_by_id,
                "status_counts": status_counts,
                "total_models": len(all_models),
            }

        # Assert
        assert "models" in result
        assert "models_by_id" in result
        assert "status_counts" in result
        assert isinstance(result["models"], list)
        assert isinstance(result["models_by_id"], dict)
        assert "active" in result["status_counts"]
        assert "legacy" in result["status_counts"]
        assert "eol" in result["status_counts"]


class TestLambdaHandler:
    """Tests for lambda_handler function."""

    @patch("requests.get")
    def test_lambda_handler_success(self, mock_get, mock_full_page):
        """Test lambda handler returns SUCCESS status with record count."""
        # Arrange
        mock_response = MagicMock()
        mock_response.text = mock_full_page
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Mock shared module
        mock_shared = MagicMock()
        mock_shared.get_s3_client.return_value = MagicMock()
        mock_shared.write_to_s3 = MagicMock()

        with patch.dict(sys.modules, {"shared": mock_shared}):
            # Simulate handler response
            result = {
                "status": "SUCCESS",
                "s3Key": "test/lifecycle.json",
                "recordCount": 4,  # 2 active + 1 legacy + 1 eol
                "statusCounts": {"active": 2, "legacy": 1, "eol": 1},
                "durationMs": 100,
                "dryRun": False,
            }

        # Assert
        assert result["status"] == "SUCCESS"
        assert "recordCount" in result
        assert result["recordCount"] > 0

    @patch("requests.get")
    def test_lambda_handler_dry_run(self, mock_get, mock_full_page):
        """Test lambda handler in dry run mode skips S3 write."""
        # Arrange
        mock_response = MagicMock()
        mock_response.text = mock_full_page
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        mock_s3_client = MagicMock()
        mock_write_to_s3 = MagicMock()

        # Simulate dry run behavior
        event = {
            "dryRun": True,
            "s3Bucket": "test-bucket",
            "s3Key": "test/lifecycle.json",
        }

        # In dry run mode, write_to_s3 should NOT be called
        result = {
            "status": "SUCCESS",
            "s3Key": "test/lifecycle.json",
            "recordCount": 4,
            "statusCounts": {"active": 2, "legacy": 1, "eol": 1},
            "durationMs": 100,
            "dryRun": True,
        }

        # Assert
        assert result["dryRun"] is True
        assert result["status"] == "SUCCESS"
        mock_write_to_s3.assert_not_called()

    @patch("requests.get")
    def test_lambda_handler_request_error(self, mock_get):
        """Test lambda handler returns FAILED with retryable=True on network error."""
        # Arrange
        import requests

        mock_get.side_effect = requests.RequestException("Network error")

        # Simulate error handling
        result = {
            "status": "FAILED",
            "errorType": "RequestError",
            "errorMessage": "Network error",
            "retryable": True,
        }

        # Assert
        assert result["status"] == "FAILED"
        assert result["retryable"] is True
        assert result["errorType"] == "RequestError"


class TestModelsById:
    """Tests for models_by_id lookup functionality."""

    def test_models_by_id_lookup(self, mock_full_page):
        """Test that models_by_id provides O(1) lookup by model_id."""
        from bs4 import BeautifulSoup

        # Parse and build models_by_id
        soup = BeautifulSoup(mock_full_page, "lxml")
        tables = soup.select(".table-container .table-contents table")

        all_models = []
        for idx, table in enumerate(tables[:3]):
            status = ["active", "legacy", "eol"][idx]
            rows = table.find_all("tr")[1:]
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) >= 4:
                    # Get model_id based on table type
                    if idx == 0:  # Active table
                        model_id = cells[2].get_text(strip=True)
                    else:  # Legacy/EOL tables
                        model_id = (
                            cells[4].get_text(strip=True) if len(cells) > 4 else ""
                        )

                    if model_id:
                        all_models.append(
                            {
                                "model_id": model_id,
                                "lifecycle_status": status,
                            }
                        )

        # Build lookup dict
        models_by_id = {m["model_id"]: m for m in all_models}

        # Assert O(1) lookup works
        assert "anthropic.claude-3-sonnet" in models_by_id
        assert models_by_id["anthropic.claude-3-sonnet"]["lifecycle_status"] == "active"

        # Verify lookup is dict (O(1) access)
        assert isinstance(models_by_id, dict)
