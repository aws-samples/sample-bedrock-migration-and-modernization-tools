"""
Shared utilities for Bedrock Model Profiler Lambda functions.

This module provides common functionality used across all Lambda handlers:
- S3 operations with proper exception handling
- AWS client configuration with retry logic
- Execution ID parsing
- Input validation utilities
- Configuration loading from S3 with fallback defaults
- Type definitions for Lambda handler inputs and outputs
- Custom exception hierarchy with retry classification
- Model ID matching and normalization utilities
"""

from shared.config import RETRY_CONFIG, get_logger
from shared.s3_utils import (
    get_s3_client,
    read_from_s3,
    write_to_s3,
)
from shared.s3_utils import (
    S3ReadError as LegacyS3ReadError,
    S3WriteError as LegacyS3WriteError,
)
from shared.execution import parse_execution_id
from shared.validation import validate_required_params
from shared.validation import ValidationError as LegacyValidationError
from shared.config_loader import ConfigLoader, get_config_loader
from shared.powertools import logger, tracer, metrics, LambdaContext, MetricUnit
from shared.exceptions import (
    ProfilerError,
    ValidationError,
    S3Error,
    S3ReadError,
    S3WriteError,
    APIError,
    ThrottlingError,
    BedrockAPIError,
    PricingAPIError,
    QuotaAPIError,
    ConfigurationError,
    DataProcessingError,
    AggregationError,
    TransformationError,
)
from shared.model_matcher import (
    # Type definitions
    ModelVariantInfo,
    # Core functions
    get_canonical_model_id,
    get_model_variant_info,
    calculate_match_score,
    find_best_match,
    find_all_matches,
    has_semantic_conflict,
    # Utility functions
    normalize_provider_prefix,
    is_variant_of,
)
from shared.cache_utils import get_cached_models, is_cache_valid, build_cache_key
from shared.types import (
    # Common types
    S3Reference,
    HandlerResult,
    # Quota Collector
    QuotaCollectorInput,
    QuotaCollectorOutput,
    QuotaData,
    # Pricing Collector
    PricingCollectorInput,
    PricingCollectorOutput,
    # Model Extractor
    ModelExtractorInput,
    ModelExtractorOutput,
    # Final Aggregator
    FinalAggregatorInput,
    FinalAggregatorOutput,
    # Regional Availability
    RegionalAvailabilityInput,
    RegionalAvailabilityOutput,
    # Feature Collector
    FeatureCollectorInput,
    FeatureCollectorOutput,
    # Copy to Latest
    CopyToLatestInput,
    CopyToLatestOutput,
    # Data Structures
    ModelModalities,
    ModelLifecycle,
)

__all__ = [
    # Config
    "RETRY_CONFIG",
    "get_logger",
    # S3 utilities
    "get_s3_client",
    "read_from_s3",
    "write_to_s3",
    # Execution utilities
    "parse_execution_id",
    # Validation utilities
    "validate_required_params",
    # Configuration loader
    "ConfigLoader",
    "get_config_loader",
    # Powertools
    "logger",
    "tracer",
    "metrics",
    "LambdaContext",
    "MetricUnit",
    # Exceptions (new hierarchy)
    "ProfilerError",
    "ValidationError",
    "S3Error",
    "S3ReadError",
    "S3WriteError",
    "APIError",
    "ThrottlingError",
    "BedrockAPIError",
    "PricingAPIError",
    "QuotaAPIError",
    "ConfigurationError",
    "DataProcessingError",
    "AggregationError",
    "TransformationError",
    # Legacy exceptions (for backward compatibility during migration)
    "LegacyS3ReadError",
    "LegacyS3WriteError",
    "LegacyValidationError",
    # Model matcher utilities
    "ModelVariantInfo",
    "get_canonical_model_id",
    "get_model_variant_info",
    "calculate_match_score",
    "find_best_match",
    "find_all_matches",
    "has_semantic_conflict",
    "normalize_provider_prefix",
    "is_variant_of",
    # Cache utilities
    "get_cached_models",
    "is_cache_valid",
    "build_cache_key",
    # Type definitions - Common
    "S3Reference",
    "HandlerResult",
    # Type definitions - Quota Collector
    "QuotaCollectorInput",
    "QuotaCollectorOutput",
    "QuotaData",
    # Type definitions - Pricing Collector
    "PricingCollectorInput",
    "PricingCollectorOutput",
    # Type definitions - Model Extractor
    "ModelExtractorInput",
    "ModelExtractorOutput",
    # Type definitions - Final Aggregator
    "FinalAggregatorInput",
    "FinalAggregatorOutput",
    # Type definitions - Regional Availability
    "RegionalAvailabilityInput",
    "RegionalAvailabilityOutput",
    # Type definitions - Feature Collector
    "FeatureCollectorInput",
    "FeatureCollectorOutput",
    # Type definitions - Copy to Latest
    "CopyToLatestInput",
    "CopyToLatestOutput",
    # Type definitions - Data Structures
    "ModelModalities",
    "ModelLifecycle",
]
