#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
import aws_cdk as cdk

from stacks.foundation_stack import FoundationStack
from stacks.gateway_stack import GatewayStack
from stacks.backend_stack import BackendStack

app = cdk.App()

# Foundation stack: DynamoDB tables, S3 bucket, IAM roles
foundation = FoundationStack(
    app,
    "IsvObservabilityFoundation",
    description="ISV Amazon Bedrock Observability - Foundation (DynamoDB, S3, IAM)",
)

# Gateway stack: API Gateway + Lambda for model invocation proxy
gateway = GatewayStack(
    app,
    "IsvObservabilityGateway",
    tenants_table=foundation.tenants_table,
    pricing_cache_table=foundation.pricing_cache_table,
    storage_bucket=foundation.storage_bucket,
    gateway_lambda_role=foundation.gateway_lambda_role,
    description="ISV Amazon Bedrock Observability - Gateway (API GW + Lambda)",
)
gateway.add_dependency(foundation)

# Backend stack: API Gateway + Lambda for tenant management and discovery
backend = BackendStack(
    app,
    "IsvObservabilityBackend",
    tenants_table=foundation.tenants_table,
    profile_mappings_table=foundation.profile_mappings_table,
    pricing_cache_table=foundation.pricing_cache_table,
    dashboards_table=foundation.dashboards_table,
    alerts_table=foundation.alerts_table,
    storage_bucket=foundation.storage_bucket,
    backend_lambda_role=foundation.backend_lambda_role,
    description="ISV Amazon Bedrock Observability - Backend (Tenants + Discovery API)",
)
backend.add_dependency(foundation)

app.synth()
