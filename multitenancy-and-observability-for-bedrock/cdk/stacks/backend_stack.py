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


class BackendStack(Stack):
    """Backend stack: API Gateway + Lambda handlers for tenants and discovery."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        tenants_table: dynamodb.ITable,
        profile_mappings_table: dynamodb.ITable,
        pricing_cache_table: dynamodb.ITable,
        dashboards_table: dynamodb.ITable,
        alerts_table: dynamodb.ITable,
        storage_bucket: s3.IBucket,
        backend_lambda_role: iam.IRole,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---------------------------------------------------------------------
        # Backend Lambda Function
        # ---------------------------------------------------------------------

        self.backend_lambda = _lambda.Function(
            self,
            "BackendLambda",
            function_name="isv-observability-backend",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="handler.lambda_handler",
            code=_lambda.Code.from_asset("../backend"),
            memory_size=256,
            timeout=Duration.seconds(30),
            role=backend_lambda_role,
            environment={
                "TENANTS_TABLE": tenants_table.table_name,
                "PROFILE_MAPPINGS_TABLE": profile_mappings_table.table_name,
                "PRICING_CACHE_TABLE": pricing_cache_table.table_name,
                "STORAGE_BUCKET": storage_bucket.bucket_name,
                "DASHBOARDS_TABLE": dashboards_table.table_name,
                "ALERTS_TABLE": alerts_table.table_name,
            },
        )

        # ---------------------------------------------------------------------
        # REST API Gateway
        # ---------------------------------------------------------------------

        self.api = apigw.RestApi(
            self,
            "BackendApi",
            rest_api_name="isv-observability-backend-api",
            description="ISV Amazon Bedrock Observability Backend API",
            endpoint_types=[apigw.EndpointType.REGIONAL],
            deploy_options=apigw.StageOptions(stage_name="v1"),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
                allow_headers=["Content-Type", "Authorization"],
            ),
        )

        backend_integration = apigw.LambdaIntegration(
            self.backend_lambda,
            proxy=True,
        )

        # Single proxy resource catches all routes — the Lambda router
        # (handler.py) already handles path-based routing internally.
        # This avoids the 20KB Lambda resource policy limit that occurs
        # when each route generates its own permission.
        self.api.root.add_proxy(
            default_integration=backend_integration,
            any_method=True,
        )
