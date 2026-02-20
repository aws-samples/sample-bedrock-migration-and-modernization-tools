"""
Tests for dashboard utility functions — Phase 4 (Python equivalents).

Validates the algorithm logic of frontend utility functions from
frontend/src/components/admin/utils/dashboardUtils.js using Python
equivalents. Since the project has no JS test framework (Vitest/Jest),
these tests ensure the core logic is correct.

Covers:
  - mergeCountryData: Combine analytics views with Cognito user counts
  - getWinnerDisplay: Format comparison winner for display
  - fmtPct: Percentage formatting with zero-division safety
  - formatRegion: Region code to display name conversion
  - engagementScore: Views per user per day calculation
  - fmt: Compact number formatting

Run:
    cd backend && python3 -m pytest tests/test_dashboard_utils.py -v

Requires: pytest
    pip install pytest
"""

import math

import pytest


# ─── Python equivalents of frontend utility functions ──────────────────────


def merge_country_data(analytics_counts=None, cognito_countries=None):
    """Python equivalent of frontend mergeCountryData function.

    Merges analytics view counts with Cognito registered user counts per country.
    """
    if analytics_counts is None:
        analytics_counts = []
    if cognito_countries is None:
        cognito_countries = []

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


def get_winner_display(winner):
    """Python equivalent of frontend getWinnerDisplay function.

    Formats comparison winner data for display, extracting a human-readable
    model name and calculating win percentage.
    """
    if not winner:
        return None

    model_id = winner["modelId"]
    # Extract short name: 'anthropic.claude-3-sonnet' -> 'Claude 3 Sonnet'
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


def fmt_pct(value, total):
    """Python equivalent of frontend fmtPct function.

    Formats a value/total as a percentage string, handling zero-division.
    """
    if not total or total == 0:
        return "0%"
    return f"{round((value / total) * 100)}%"


def format_region(region_code):
    """Python equivalent of frontend formatRegion function.

    Converts region codes like 'us-east-1' to 'US East 1'.
    """
    if not region_code:
        return ""
    parts = region_code.split("-")
    result = []
    for i, part in enumerate(parts):
        if i == 0:
            result.append(part.upper())
        else:
            result.append(part[0].upper() + part[1:] if part else part)
    return " ".join(result)


def engagement_score(views, users, days):
    """Python equivalent of frontend engagementScore function.

    Calculates views per user per day, rounded to 2 decimal places.
    """
    if not users or not days:
        return 0
    return round((views / users / days) * 100) / 100


def fmt(n):
    """Python equivalent of frontend fmt function.

    Formats numbers with K/M suffixes for compact display.
    """
    if n is None:
        return "0"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ═══════════════════════════════════════════════════════════════════════════
# mergeCountryData tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMergeCountryData:
    """Validate mergeCountryData combines analytics views with Cognito users."""

    def test_merge_country_data_both_sources(self):
        """Overlapping countries should merge views and users correctly."""
        analytics = [{"id": "US", "count": 10}]
        cognito = [{"id": "US", "count": 5}, {"id": "DE", "count": 3}]

        result = merge_country_data(analytics, cognito)
        result_dict = {r["id"]: r for r in result}

        assert result_dict["US"]["views"] == 10
        assert result_dict["US"]["users"] == 5
        assert result_dict["DE"]["views"] == 0
        assert result_dict["DE"]["users"] == 3

    def test_merge_country_data_empty_cognito(self):
        """Analytics-only data should have zero users."""
        analytics = [{"id": "US", "count": 10}]

        result = merge_country_data(analytics, [])

        assert len(result) == 1
        assert result[0]["id"] == "US"
        assert result[0]["views"] == 10
        assert result[0]["users"] == 0

    def test_merge_country_data_empty_both(self):
        """Empty inputs should return empty list."""
        result = merge_country_data([], [])
        assert result == []

    def test_merge_country_data_sorted_by_total(self):
        """Results should be sorted by views + users descending."""
        analytics = [{"id": "US", "count": 5}, {"id": "JP", "count": 20}]
        cognito = [{"id": "US", "count": 10}, {"id": "JP", "count": 1}]

        result = merge_country_data(analytics, cognito)

        # JP: 20+1=21, US: 5+10=15 => JP first
        assert result[0]["id"] == "JP"
        assert result[1]["id"] == "US"


# ═══════════════════════════════════════════════════════════════════════════
# getWinnerDisplay tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGetWinnerDisplay:
    """Validate getWinnerDisplay formats comparison winner for display."""

    def test_get_winner_display_formats_name(self):
        """Winner display should extract model name and calculate percentage."""
        winner = {
            "modelId": "anthropic.claude-3-sonnet",
            "comparisons": 10,
            "totalComparisons": 50,
        }

        result = get_winner_display(winner)

        assert result is not None
        assert "Claude" in result["displayName"]
        assert "Sonnet" in result["displayName"]
        assert result["percentage"] == 20
        assert result["count"] == 10
        assert result["total"] == 50

    def test_get_winner_display_null_input(self):
        """None winner should return None."""
        result = get_winner_display(None)
        assert result is None

    def test_get_winner_display_zero_total(self):
        """Zero totalComparisons should produce 0 percentage."""
        winner = {
            "modelId": "amazon.titan-text-express",
            "comparisons": 5,
            "totalComparisons": 0,
        }

        result = get_winner_display(winner)
        assert result is not None
        assert result["percentage"] == 0

    def test_get_winner_display_no_dot_in_model_id(self):
        """Model ID without dot should use full ID as name."""
        winner = {
            "modelId": "custom-model-v2",
            "comparisons": 3,
            "totalComparisons": 10,
        }

        result = get_winner_display(winner)
        assert result is not None
        assert "Custom" in result["displayName"]
        assert result["percentage"] == 30


# ═══════════════════════════════════════════════════════════════════════════
# fmtPct tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFmtPct:
    """Validate fmtPct percentage formatting with edge cases."""

    def test_fmt_pct_edge_cases(self):
        """fmtPct(0,0), fmtPct(5,10), fmtPct(3,0) should handle all edge cases."""
        assert fmt_pct(0, 0) == "0%"
        assert fmt_pct(5, 10) == "50%"
        assert fmt_pct(3, 0) == "0%"

    def test_fmt_pct_none_total(self):
        """fmtPct with None total should return '0%'."""
        assert fmt_pct(5, None) == "0%"

    def test_fmt_pct_normal(self):
        """fmtPct(25, 100) should return '25%'."""
        assert fmt_pct(25, 100) == "25%"

    def test_fmt_pct_rounding(self):
        """fmtPct(1, 3) should round to '33%'."""
        assert fmt_pct(1, 3) == "33%"

    def test_fmt_pct_full(self):
        """fmtPct(100, 100) should return '100%'."""
        assert fmt_pct(100, 100) == "100%"


# ═══════════════════════════════════════════════════════════════════════════
# formatRegion tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFormatRegion:
    """Validate formatRegion converts region codes to display names."""

    def test_format_region_code(self):
        """'us-east-1' should become 'US East 1'."""
        assert format_region("us-east-1") == "US East 1"

    def test_format_region_three_parts(self):
        """'ap-southeast-1' should become 'AP Southeast 1'."""
        assert format_region("ap-southeast-1") == "AP Southeast 1"

    def test_format_region_eu(self):
        """'eu-west-1' should become 'EU West 1'."""
        assert format_region("eu-west-1") == "EU West 1"

    def test_format_region_empty(self):
        """Empty string should return empty string."""
        assert format_region("") == ""

    def test_format_region_none(self):
        """None should return empty string."""
        assert format_region(None) == ""


# ═══════════════════════════════════════════════════════════════════════════
# engagementScore tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEngagementScore:
    """Validate engagementScore calculates views per user per day."""

    def test_engagement_score_normal(self):
        """100 views, 10 users, 5 days = 2.0 views/user/day."""
        result = engagement_score(100, 10, 5)
        assert result == 2.0

    def test_engagement_score_zero_users(self):
        """Zero users should return 0 (no division by zero)."""
        result = engagement_score(100, 0, 5)
        assert result == 0

    def test_engagement_score_zero_days(self):
        """Zero days should return 0 (no division by zero)."""
        result = engagement_score(100, 10, 0)
        assert result == 0

    def test_engagement_score_fractional(self):
        """Fractional result should be rounded to 2 decimal places."""
        result = engagement_score(7, 3, 2)
        # 7 / 3 / 2 = 1.1666... → round(1.1666 * 100) / 100 = 1.17
        assert result == 1.17


# ═══════════════════════════════════════════════════════════════════════════
# fmt (compact number formatting) tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFmt:
    """Validate fmt compact number formatting."""

    def test_fmt_millions(self):
        """1,500,000 should format as '1.5M'."""
        assert fmt(1_500_000) == "1.5M"

    def test_fmt_thousands(self):
        """2,500 should format as '2.5K'."""
        assert fmt(2_500) == "2.5K"

    def test_fmt_small_number(self):
        """42 should format as '42'."""
        assert fmt(42) == "42"

    def test_fmt_none(self):
        """None should format as '0'."""
        assert fmt(None) == "0"

    def test_fmt_zero(self):
        """0 should format as '0'."""
        assert fmt(0) == "0"
