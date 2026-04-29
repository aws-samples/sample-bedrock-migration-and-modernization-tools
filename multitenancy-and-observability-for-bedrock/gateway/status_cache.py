# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""In-memory profile status cache with TTL.

Persists across warm Lambda invocations since it lives at module level.
"""

import time


class ProfileStatusCache:
    """Simple TTL cache for profile records.

    Avoids hitting DynamoDB on every request when the Lambda container
    is reused (warm start).  A short TTL (default 30 s) ensures that
    status changes (suspend / throttle) propagate quickly.
    """

    def __init__(self, ttl_seconds: int = 30):
        self._ttl_seconds = ttl_seconds
        # { profile_id: (profile_record, expiry_timestamp) }
        self._store: dict[str, tuple[dict, float]] = {}

    def get(self, profile_id: str) -> dict | None:
        """Return the cached profile record, or *None* if expired / missing."""
        entry = self._store.get(profile_id)
        if entry is None:
            return None
        record, expiry = entry
        if time.monotonic() > expiry:
            del self._store[profile_id]
            return None
        return record

    def set(self, profile_id: str, profile_record: dict) -> None:
        """Cache a profile record with the configured TTL."""
        expiry = time.monotonic() + self._ttl_seconds
        self._store[profile_id] = (profile_record, expiry)

    def invalidate(self, profile_id: str) -> None:
        """Remove a profile from the cache."""
        self._store.pop(profile_id, None)
