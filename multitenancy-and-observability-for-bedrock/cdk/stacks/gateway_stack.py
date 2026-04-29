# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
from aws_cdk import (
    Stack,
    Duration,
    aws_lambda as _lambda,
    aws_apigateway as apigw,
    aws_dynamodb as dynamodb,
    aws_s3 as s3,
    aws_iam as iam,
)
from constructs import Construct


class GatewayStack(Stack):
    """Gateway stack: API Gateway + Gateway Lambda for /invoke endpoint."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        tenants_table: dynamodb.ITable,
        pricing_cache_table: dynamodb.ITable,
        storage_bucket: s3.IBucket,
        gateway_lambda_role: iam.IRole,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---------------------------------------------------------------------
        # Gateway Lambda Function
        # ---------------------------------------------------------------------

        self.gateway_lambda = _lambda.Function(
            self,
            "GatewayLambda",
            function_name="isv-observability-gateway",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=_lambda.Code.from_asset("../gateway"),
            memory_size=512,
            timeout=Duration.seconds(300),
            role=gateway_lambda_role,
            environment={
                "TENANTS_TABLE": tenants_table.table_name,
                "PRICING_CACHE_TABLE": pricing_cache_table.table_name,
            },
        )

        # ---------------------------------------------------------------------
        # REST API Gateway
        # ---------------------------------------------------------------------

        self.api = apigw.RestApi(
            self,
            "GatewayApi",
            rest_api_name="isv-observability-gateway-api",
            description="ISV Amazon Bedrock Observability Gateway API",
            endpoint_types=[apigw.EndpointType.REGIONAL],
            deploy_options=apigw.StageOptions(stage_name="v1"),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=["POST", "OPTIONS"],
                allow_headers=["Content-Type", "Tenant-Id"],
            ),
        )

        # POST /invoke
        invoke_resource = self.api.root.add_resource("invoke")
        invoke_resource.add_method(
            "POST",
            apigw.LambdaIntegration(
                self.gateway_lambda,
                proxy=True,
            ),
        )
