"""
Local handlers for Bedrock Model Profiler.

These are standalone versions of the Lambda handlers that run locally
using the same transformation logic as the AWS Step Functions pipeline.
"""

from local.handlers.pricing_aggregator import aggregate_pricing
from local.handlers.model_merger import merge_models
from local.handlers.model_enricher import enrich_providers
from local.handlers.pricing_linker import link_pricing_to_models
from local.handlers.regional_availability import compute_regional_availability
from local.handlers.final_aggregator import build_final_models

__all__ = [
    'aggregate_pricing',
    'merge_models',
    'enrich_providers',
    'link_pricing_to_models',
    'compute_regional_availability',
    'build_final_models',
]
