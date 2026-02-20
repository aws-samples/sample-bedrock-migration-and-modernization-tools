"""
Cognito Sync Lambda — syncs Cognito user data to DynamoDB cache.

Scheduled daily. Paginates all Cognito users and writes:
  - COGNITO#SUMMARY / {YYYY-MM-DD} — daily summary (total, new, returning, by country)
  - COGNITO#USERS / {sub} — individual user cache records with dateCreated

DynamoDB Schema:
  PK=COGNITO#SUMMARY, SK=YYYY-MM-DD:
    totalUsers, newUsersToday, returningUsersToday, usersByCountry (Map),
    estimatedTotal, syncTimestamp

  PK=COGNITO#USERS, SK={sub}:
    status, enabled, country, locale,
    dateCreated (ISO string from identities attribute — epoch ms of first login),
    dateCreatedMs (int — raw epoch ms for range queries),
    lastModified (ISO string)
"""

import json
import os
from datetime import datetime, timezone, timedelta

import boto3

TABLE_NAME = os.environ.get("ANALYTICS_TABLE", "bedrock-profiler-analytics-dev")
USER_POOL_ID = os.environ.get("USER_POOL_ID", "")
COGNITO_REGION = os.environ.get("COGNITO_REGION", "us-east-1")

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(TABLE_NAME)
cognito = boto3.client("cognito-idp", region_name=COGNITO_REGION)

# ISO 3166-1 alpha-3 to alpha-2 country code mapping
# Midway uses 3-letter codes, frontend expects 2-letter codes
COUNTRY_CODE_MAP = {
    "USA": "US",
    "GBR": "GB",
    "DEU": "DE",
    "FRA": "FR",
    "ESP": "ES",
    "ITA": "IT",
    "BRA": "BR",
    "IND": "IN",
    "JPN": "JP",
    "CHN": "CN",
    "AUS": "AU",
    "CAN": "CA",
    "MEX": "MX",
    "KOR": "KR",
    "RUS": "RU",
    "ZAF": "ZA",
    "ARG": "AR",
    "COL": "CO",
    "CHL": "CL",
    "PER": "PE",
    "NLD": "NL",
    "BEL": "BE",
    "SWE": "SE",
    "NOR": "NO",
    "FIN": "FI",
    "DNK": "DK",
    "POL": "PL",
    "AUT": "AT",
    "CHE": "CH",
    "PRT": "PT",
    "IRL": "IE",
    "ISR": "IL",
    "ARE": "AE",
    "SAU": "SA",
    "SGP": "SG",
    "MYS": "MY",
    "THA": "TH",
    "IDN": "ID",
    "PHL": "PH",
    "VNM": "VN",
    "TWN": "TW",
    "HKG": "HK",
    "NZL": "NZ",
    "UKR": "UA",
    "EGY": "EG",
    "NGA": "NG",
    "KEN": "KE",
    "GHA": "GH",
    "TUR": "TR",
    "PAK": "PK",
    "BGD": "BD",
    "CZE": "CZ",
    "ROU": "RO",
    "HUN": "HU",
    "GRC": "GR",
    "SVK": "SK",
    "BGR": "BG",
    "HRV": "HR",
    "SRB": "RS",
    "LTU": "LT",
    "LVA": "LV",
    "EST": "EE",
    "SVN": "SI",
    "LUX": "LU",
    "CRI": "CR",
}


def _normalize_country_code(code):
    """Convert 3-letter country code to 2-letter ISO code.

    Midway provides 3-letter codes (e.g., "IRL", "USA").
    Frontend expects 2-letter codes (e.g., "IE", "US").
    If already 2-letter or not in mapping, return as-is (uppercase).
    """
    if not code:
        return ""
    code = code.upper().strip()
    # If it's a 3-letter code, try to convert
    if len(code) == 3:
        return COUNTRY_CODE_MAP.get(code, code)
    # If it's already 2-letter, return as-is
    return code


def lambda_handler(event, context):
    """Main handler — sync Cognito users to DynamoDB cache."""
    if not USER_POOL_ID:
        return {"status": "SKIPPED", "reason": "USER_POOL_ID not configured"}

    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")

    # Step 1: Quick count via DescribeUserPool
    pool_info = cognito.describe_user_pool(UserPoolId=USER_POOL_ID)
    estimated_total = pool_info["UserPool"].get("EstimatedNumberOfUsers", 0)

    # Step 2: Paginate all users
    users = _list_all_users()

    # Step 3: Compute summary
    summary = _compute_summary(users, today)
    summary["estimatedTotal"] = estimated_total
    summary["syncTimestamp"] = int(now.timestamp() * 1000)
    summary["syncDate"] = today

    # Step 4: Write summary to DynamoDB
    _write_summary(today, summary)

    # Step 5: Write/update individual user cache records
    _write_user_records(users)

    return {
        "status": "SUCCESS",
        "totalUsers": len(users),
        "estimatedTotal": estimated_total,
        "newToday": summary.get("newUsersToday", 0),
        "returningToday": summary.get("returningUsersToday", 0),
        "countries": len(summary.get("usersByCountry", {})),
        "syncDate": today,
    }


def _parse_identity_date_created(attrs):
    """Extract dateCreated from the identities attribute.

    The identities attribute is a JSON array like:
    [{"dateCreated":"1770124854931","userId":"...","providerName":"...","providerType":"OIDC",...}]

    dateCreated is a Unix timestamp in milliseconds.

    Returns:
        tuple: (datetime_utc, epoch_ms_int) or (None, None) if unparseable.
    """
    identities_str = attrs.get("identities", "")
    if not identities_str:
        return None, None

    try:
        identities = json.loads(identities_str)
        if isinstance(identities, list) and len(identities) > 0:
            date_created_ms = identities[0].get("dateCreated")
            if date_created_ms is not None:
                epoch_ms = int(date_created_ms)
                dt = datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
                return dt, epoch_ms
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        pass

    return None, None


def _list_all_users():
    """Paginate all Cognito users. Respects 30 RPS / 60 per page."""
    users = []
    params = {
        "UserPoolId": USER_POOL_ID,
        "Limit": 60,  # Max per page
    }

    while True:
        resp = cognito.list_users(**params)
        for user in resp.get("Users", []):
            attrs = {a["Name"]: a["Value"] for a in user.get("Attributes", [])}

            # Parse dateCreated from identities attribute (first login date)
            identity_dt, identity_ms = _parse_identity_date_created(attrs)

            # Fall back to UserCreateDate if identities is missing/unparseable
            if identity_dt is not None:
                created_dt = identity_dt
                created_ms = identity_ms
            else:
                created_dt = user.get("UserCreateDate")
                created_ms = (
                    int(created_dt.timestamp() * 1000)
                    if created_dt and hasattr(created_dt, "timestamp")
                    else None
                )

            users.append(
                {
                    "sub": attrs.get("sub", ""),
                    "email_verified": attrs.get("email_verified", "false") == "true",
                    "status": user.get("UserStatus", "UNKNOWN"),
                    "enabled": user.get("Enabled", False),
                    "created": created_dt,
                    "createdMs": created_ms,
                    "lastModified": user.get("UserLastModifiedDate"),
                    "locale": attrs.get("locale", ""),
                    "country": _normalize_country_code(attrs.get("custom:country", ""))
                    or "Unknown",
                }
            )

        pagination_token = resp.get("PaginationToken")
        if not pagination_token:
            break
        params["PaginationToken"] = pagination_token

    return users


def _compute_summary(users, today):
    """Compute daily summary from user list.

    - totalUsers: all confirmed or federated (enabled) users
    - newUsersToday: users whose dateCreated (from identities) is today
    - returningUsersToday: users whose dateCreated is before today
    - usersByCountry: from custom:country attribute only ("Unknown" excluded from counts)
    """
    total = 0
    new_today = 0
    returning_today = 0
    by_country = {}
    confirmed_users = 0

    for user in users:
        if not user.get("enabled", False):
            continue
        if user.get("status") not in ("CONFIRMED", "EXTERNAL_PROVIDER"):
            continue

        confirmed_users += 1
        total += 1

        # Determine if new (first login today) or returning (first login before today)
        created = user.get("created")
        if created:
            created_date = (
                created.strftime("%Y-%m-%d")
                if hasattr(created, "strftime")
                else str(created)[:10]
            )
            if created_date == today:
                new_today += 1
            else:
                returning_today += 1

        # Country from custom:country attribute (already resolved in _list_all_users)
        country = user.get("country") or ""
        if country and country != "Unknown":
            by_country[country] = by_country.get(country, 0) + 1

    return {
        "totalUsers": total,
        "confirmedUsers": confirmed_users,
        "newUsersToday": new_today,
        "returningUsersToday": returning_today,
        "usersByCountry": by_country,
    }


def _write_summary(today, summary):
    """Write daily summary to DynamoDB."""
    item = {
        "PK": "COGNITO#SUMMARY",
        "SK": today,
        **{k: v for k, v in summary.items() if v is not None},
    }
    # Convert dict values for DynamoDB compatibility
    if "usersByCountry" in item and isinstance(item["usersByCountry"], dict):
        item["usersByCountry"] = {k: int(v) for k, v in item["usersByCountry"].items()}

    table.put_item(Item=item)


def _write_user_records(users):
    """Batch write individual user cache records with dateCreated for period queries."""
    with table.batch_writer() as batch:
        for user in users:
            if not user.get("sub"):
                continue
            item = {
                "PK": "COGNITO#USERS",
                "SK": user["sub"],
                "status": user.get("status", "UNKNOWN"),
                "enabled": user.get("enabled", False),
                "country": user.get("country", ""),
                "locale": user.get("locale", ""),
            }
            # Store dateCreated as ISO string and raw epoch ms
            if user.get("created"):
                item["dateCreated"] = (
                    user["created"].isoformat()
                    if hasattr(user["created"], "isoformat")
                    else str(user["created"])
                )
            if user.get("createdMs"):
                item["dateCreatedMs"] = user["createdMs"]
            if user.get("lastModified"):
                item["lastModified"] = (
                    user["lastModified"].isoformat()
                    if hasattr(user["lastModified"], "isoformat")
                    else str(user["lastModified"])
                )

            batch.put_item(Item=item)
