"""Type definitions for Bedrock Model Profiler Lambda handlers.

This module provides TypedDict definitions for Lambda handler inputs and outputs,
enabling static type checking and IDE autocompletion.

Usage:
    from shared.types import QuotaCollectorInput, QuotaCollectorOutput

    def lambda_handler(event: QuotaCollectorInput, context: LambdaContext) -> QuotaCollectorOutput:
        region = event['region']  # IDE knows this is a string
        return {'status': 'SUCCESS', 'region': region, ...}
"""

from typing import TypedDict, NotRequired, Any

# =============================================================================
# Common Types
# =============================================================================


class S3Reference(TypedDict):
    """Reference to an S3 object."""

    s3Bucket: str
    s3Key: str


class HandlerResult(TypedDict):
    """Base result structure for all handlers."""

    status: str  # 'SUCCESS' or 'FAILED'
    durationMs: NotRequired[int]
    errorType: NotRequired[str]
    errorMessage: NotRequired[str]
    retryable: NotRequired[bool]


# =============================================================================
# Quota Collector Types
# =============================================================================


class QuotaCollectorInput(TypedDict, total=False):
    """Input event for quota-collector Lambda.

    Attributes:
        region: AWS region to collect quotas from (required)
        s3Bucket: S3 bucket for output (optional)
        s3Key: S3 key for output (optional)
        dryRun: If True, skip S3 write (optional)
    """

    region: str
    s3Bucket: str
    s3Key: str
    dryRun: bool


class QuotaCollectorOutput(TypedDict):
    """Output response from quota-collector Lambda."""

    status: str
    region: NotRequired[str]
    s3Key: NotRequired[str]
    quotaCount: NotRequired[int]
    durationMs: NotRequired[int]
    errorType: NotRequired[str]
    errorMessage: NotRequired[str]
    retryable: NotRequired[bool]


# =============================================================================
# Pricing Collector Types
# =============================================================================


class PricingCollectorInput(TypedDict, total=False):
    """Input event for pricing-collector Lambda."""

    serviceCode: str
    s3Bucket: str
    s3Key: str
    dryRun: bool


class PricingCollectorOutput(TypedDict):
    """Output response from pricing-collector Lambda."""

    status: str
    serviceCode: NotRequired[str]
    s3Key: NotRequired[str]
    productCount: NotRequired[int]
    durationMs: NotRequired[int]
    errorType: NotRequired[str]
    errorMessage: NotRequired[str]


# =============================================================================
# Model Extractor Types
# =============================================================================


class ModelExtractorInput(TypedDict, total=False):
    """Input event for model-extractor Lambda."""

    region: str
    s3Bucket: str
    s3Key: str
    dryRun: bool


class ModelExtractorOutput(TypedDict):
    """Output response from model-extractor Lambda."""

    status: str
    region: NotRequired[str]
    s3Key: NotRequired[str]
    modelCount: NotRequired[int]
    durationMs: NotRequired[int]
    errorType: NotRequired[str]
    errorMessage: NotRequired[str]


# =============================================================================
# Final Aggregator Types
# =============================================================================


class FinalAggregatorInput(TypedDict, total=False):
    """Input event for final-aggregator Lambda."""

    s3Bucket: str
    executionId: str
    pricingS3Key: str
    modelsS3Key: str
    quotaResults: list[dict]
    pricingLinked: dict
    regionalAvailability: dict
    featureResults: list[dict]
    tokenSpecs: dict
    enrichedModels: dict
    mantleResults: list[dict]
    lifecycleData: dict
    dryRun: bool


class FinalAggregatorOutput(TypedDict):
    """Output response from final-aggregator Lambda."""

    status: str
    modelsS3Key: NotRequired[str]
    pricingS3Key: NotRequired[str]
    totalModels: NotRequired[int]
    totalProviders: NotRequired[int]
    totalRegions: NotRequired[int]
    durationMs: NotRequired[int]
    errorType: NotRequired[str]
    errorMessage: NotRequired[str]


# =============================================================================
# Regional Availability Types
# =============================================================================


class RegionalAvailabilityInput(TypedDict, total=False):
    """Input event for regional-availability Lambda."""

    s3Bucket: str
    executionId: str
    pricingS3Key: str
    regions: list[str]
    dryRun: bool


class RegionalAvailabilityOutput(TypedDict):
    """Output response from regional-availability Lambda."""

    status: str
    s3Key: NotRequired[str]
    regionsChecked: NotRequired[int]
    modelsFound: NotRequired[int]
    durationMs: NotRequired[int]
    errorType: NotRequired[str]
    errorMessage: NotRequired[str]


# =============================================================================
# Feature Collector Types
# =============================================================================


class FeatureCollectorInput(TypedDict, total=False):
    """Input event for feature-collector Lambda."""

    region: str
    s3Bucket: str
    s3Key: str
    dryRun: bool


class FeatureCollectorOutput(TypedDict):
    """Output response from feature-collector Lambda."""

    status: str
    region: NotRequired[str]
    s3Key: NotRequired[str]
    profileCount: NotRequired[int]
    durationMs: NotRequired[int]
    errorType: NotRequired[str]
    errorMessage: NotRequired[str]


# =============================================================================
# Copy to Latest Types
# =============================================================================


class CopyToLatestInput(TypedDict, total=False):
    """Input event for copy-to-latest Lambda."""

    s3Bucket: str
    executionId: str
    finalResult: dict
    dryRun: bool


class CopyToLatestOutput(TypedDict):
    """Output response from copy-to-latest Lambda."""

    status: str
    copiedFiles: NotRequired[list[str]]
    durationMs: NotRequired[int]
    errorType: NotRequired[str]
    errorMessage: NotRequired[str]


# =============================================================================
# Data Structure Types
# =============================================================================


class QuotaData(TypedDict):
    """Individual quota data structure (snake_case)."""

    quota_code: str
    quota_name: str
    quota_arn: str
    value: float
    unit: str
    adjustable: bool
    global_quota: bool
    usage_metric: dict
    period: dict
    region: str


class ModelModalities(TypedDict):
    """Model input/output modalities."""

    input_modalities: list[str]
    output_modalities: list[str]


class ModelLifecycle(TypedDict, total=False):
    """Model lifecycle information."""

    status: str
    release_date: str
    legacy_date: str
    eol_date: str
    recommended_replacement: str


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Common
    "S3Reference",
    "HandlerResult",
    # Quota Collector
    "QuotaCollectorInput",
    "QuotaCollectorOutput",
    "QuotaData",
    # Pricing Collector
    "PricingCollectorInput",
    "PricingCollectorOutput",
    # Model Extractor
    "ModelExtractorInput",
    "ModelExtractorOutput",
    # Final Aggregator
    "FinalAggregatorInput",
    "FinalAggregatorOutput",
    # Regional Availability
    "RegionalAvailabilityInput",
    "RegionalAvailabilityOutput",
    # Feature Collector
    "FeatureCollectorInput",
    "FeatureCollectorOutput",
    # Copy to Latest
    "CopyToLatestInput",
    "CopyToLatestOutput",
    # Data Structures
    "ModelModalities",
    "ModelLifecycle",
]
