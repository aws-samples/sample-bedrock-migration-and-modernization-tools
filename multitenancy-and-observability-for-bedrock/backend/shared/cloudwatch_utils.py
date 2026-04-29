# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""CloudWatch metric publishing for Amazon Bedrock inference tracing.

Namespace: Amazon BedrockInvocationTracing
Dimensions: InferenceProfile, TenantId, ModelId
"""

import logging

logger = logging.getLogger(__name__)

NAMESPACE = "ISVBedrock/Gateway"


def publish_inference_metrics(
    cloudwatch_client,
    profile_name: str,
    profile_id: str,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    input_cost: float,
    output_cost: float,
    latency_ms: float,
    success: bool = True,
) -> None:
    """Publish all inference metrics to CloudWatch.

    Emits the following metrics:
        - InputTokens
        - OutputTokens
        - InputTokensCost (USD)
        - OutputTokensCost (USD)
        - InvocationLatencyMs (milliseconds)
        - InvocationSuccess (1 per successful call)
        - InvocationFailure (1 per failed call)
    """
    dimensions = [
        {"Name": "InferenceProfile", "Value": profile_name},
        {"Name": "TenantId", "Value": profile_id},
        {"Name": "ModelId", "Value": model_id},
    ]

    metric_data = [
        {"MetricName": "InputTokens", "Dimensions": dimensions, "Value": float(input_tokens), "Unit": "Count"},
        {"MetricName": "OutputTokens", "Dimensions": dimensions, "Value": float(output_tokens), "Unit": "Count"},
        {"MetricName": "InputTokensCost", "Dimensions": dimensions, "Value": input_cost, "Unit": "None"},
        {"MetricName": "OutputTokensCost", "Dimensions": dimensions, "Value": output_cost, "Unit": "None"},
        {"MetricName": "InvocationLatencyMs", "Dimensions": dimensions, "Value": latency_ms, "Unit": "Milliseconds"},
        {"MetricName": "InvocationSuccess" if success else "InvocationFailure", "Dimensions": dimensions, "Value": 1, "Unit": "Count"},
    ]

    try:
        # CloudWatch PutMetricData accepts up to 1000 metric data points per call
        cloudwatch_client.put_metric_data(
            Namespace=NAMESPACE,
            MetricData=metric_data,
        )
        logger.info(
            "Published %d metrics for profile=%s profile_name=%s model=%s",
            len(metric_data),
            profile_id,
            profile_name,
            model_id,
        )
    except Exception as exc:
        logger.error(
            "Failed to publish CloudWatch metrics for profile=%s: %s",
            profile_id,
            exc,
        )
        raise
