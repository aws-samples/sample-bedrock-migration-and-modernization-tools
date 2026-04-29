# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Data models for ISV Amazon Bedrock Observability platform.

Uses plain dataclasses for Lambda compatibility without extra dependencies.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional  # noqa: F401 – used by callers


@dataclass
class Profile:
    """Represents a profile with an associated Amazon Bedrock inference profile."""

    tenant_id: str
    tenant_name: str
    status: str = "active"  # active, throttled, suspended
    model_id: str = ""
    region: str = ""
    inference_profile_id: str = ""
    inference_profile_arn: str = ""
    profile_strategy: str = "dedicated"  # dedicated, shared
    tags: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class PricingEntry:
    """Cached pricing data for a model in a specific region."""

    cache_key: str  # region#model_id
    model_id: str
    region: str
    input_cost: float
    output_cost: float
    pricing_source: str  # api, api_partial, fallback
    cached_at: float = 0.0
    ttl: int = 0  # epoch seconds, 24h from cached_at

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PricingEntry":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        # helps numeric types
        for num_field in ("input_cost", "output_cost", "cached_at"):
            if num_field in filtered:
                filtered[num_field] = float(filtered[num_field])
        if "ttl" in filtered:
            filtered["ttl"] = int(filtered["ttl"])
        return cls(**filtered)
