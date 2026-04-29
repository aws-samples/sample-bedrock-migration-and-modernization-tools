# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
from aws_cdk import (
    Stack,
    RemovalPolicy,
    aws_dynamodb as dynamodb,
    aws_s3 as s3,
    aws_iam as iam,
)
from constructs import Construct


class FoundationStack(Stack):
    """Foundation stack: DynamoDB tables, S3 bucket, and IAM roles."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---------------------------------------------------------------------
        # DynamoDB Tables
        # ---------------------------------------------------------------------

        # Tenants table
        self.tenants_table = dynamodb.Table(
            self,
            "TenantsTable",
            table_name="isv-observability-tenants",
            partition_key=dynamodb.Attribute(
                name="tenant_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
            time_to_live_attribute="ttl",
        )

        # ProfileMappings table
        self.profile_mappings_table = dynamodb.Table(
            self,
            "ProfileMappingsTable",
            table_name="isv-observability-profile-mappings",
            partition_key=dynamodb.Attribute(
                name="inference_profile_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # GSI on tenant_id for ProfileMappings
        self.profile_mappings_table.add_global_secondary_index(
            index_name="tenant_id-index",
            partition_key=dynamodb.Attribute(
                name="tenant_id", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # PricingCache table
        self.pricing_cache_table = dynamodb.Table(
            self,
            "PricingCacheTable",
            table_name="isv-observability-pricing-cache",
            partition_key=dynamodb.Attribute(
                name="region#model_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
            time_to_live_attribute="ttl",
        )

        # Dashboards table
        self.dashboards_table = dynamodb.Table(
            self,
            "DashboardsTable",
            table_name="isv-observability-dashboards",
            partition_key=dynamodb.Attribute(
                name="dashboard_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.dashboards_table.add_global_secondary_index(
            index_name="tenant_id_index",
            partition_key=dynamodb.Attribute(
                name="tenant_id", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # Alerts table
        self.alerts_table = dynamodb.Table(
            self,
            "AlertsTable",
            table_name="isv-observability-alerts",
            partition_key=dynamodb.Attribute(
                name="alert_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.alerts_table.add_global_secondary_index(
            index_name="tenant_id_index",
            partition_key=dynamodb.Attribute(
                name="tenant_id", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # ---------------------------------------------------------------------
        # S3 Bucket
        # ---------------------------------------------------------------------

        self.storage_bucket = s3.Bucket(
            self,
            "StorageBucket",
            bucket_name=None,  # auto-generated
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
        )

        # ---------------------------------------------------------------------
        # IAM Roles
        # ---------------------------------------------------------------------

        # Gateway Lambda execution role
        self.gateway_lambda_role = iam.Role(
            self,
            "GatewayLambdaRole",
            role_name="isv-observability-gateway-lambda-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
            ],
        )

        # Gateway Lambda permissions: Amazon Bedrock invoke
        self.gateway_lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=["*"],
            )
        )

        # Gateway Lambda permissions: CloudWatch put metrics
        self.gateway_lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
            )
        )

        # Gateway Lambda permissions: DynamoDB read on Tenants + PricingCache
        self.gateway_lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "dynamodb:GetItem",
                    "dynamodb:Query",
                    "dynamodb:Scan",
                    "dynamodb:BatchGetItem",
                ],
                resources=[
                    self.tenants_table.table_arn,
                    self.pricing_cache_table.table_arn,
                ],
            )
        )

        # Backend Lambda execution role
        self.backend_lambda_role = iam.Role(
            self,
            "BackendLambdaRole",
            role_name="isv-observability-backend-lambda-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
            ],
        )

        # Backend Lambda permissions: DynamoDB full CRUD on all tables
        self.backend_lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "dynamodb:GetItem",
                    "dynamodb:PutItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:DeleteItem",
                    "dynamodb:Query",
                    "dynamodb:Scan",
                    "dynamodb:BatchGetItem",
                    "dynamodb:BatchWriteItem",
                ],
                resources=[
                    self.tenants_table.table_arn,
                    f"{self.tenants_table.table_arn}/index/*",
                    self.profile_mappings_table.table_arn,
                    f"{self.profile_mappings_table.table_arn}/index/*",
                    self.pricing_cache_table.table_arn,
                    f"{self.pricing_cache_table.table_arn}/index/*",
                    self.dashboards_table.table_arn,
                    f"{self.dashboards_table.table_arn}/index/*",
                    self.alerts_table.table_arn,
                    f"{self.alerts_table.table_arn}/index/*",
                ],
            )
        )

        # Backend Lambda permissions: Amazon Bedrock manage inference profiles
        self.backend_lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:ListFoundationModels",
                    "bedrock:GetFoundationModel",
                    "bedrock:ListInferenceProfiles",
                    "bedrock:GetInferenceProfile",
                    "bedrock:CreateInferenceProfile",
                    "bedrock:DeleteInferenceProfile",
                    "bedrock:TagResource",
                    "bedrock:UntagResource",
                    "bedrock:ListTagsForResource",
                ],
                resources=["*"],
            )
        )

        # Backend Lambda permissions: Pricing API read
        self.backend_lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "pricing:GetProducts",
                    "pricing:DescribeServices",
                    "pricing:GetAttributeValues",
                ],
                resources=["*"],
            )
        )

        # Backend Lambda permissions: CloudWatch read
        self.backend_lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "cloudwatch:GetMetricData",
                    "cloudwatch:GetMetricStatistics",
                    "cloudwatch:ListMetrics",
                    "cloudwatch:DescribeAlarms",
                ],
                resources=["*"],
            )
        )

        # Backend Lambda permissions: CloudWatch dashboard/alarm management and SNS
        self.backend_lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "cloudwatch:PutDashboard",
                    "cloudwatch:DeleteDashboards",
                    "cloudwatch:GetDashboard",
                    "cloudwatch:ListDashboards",
                    "cloudwatch:PutMetricAlarm",
                    "cloudwatch:DeleteAlarms",
                    "sns:CreateTopic",
                    "sns:DeleteTopic",
                    "sns:Subscribe",
                    "sns:Unsubscribe",
                    "sns:ListSubscriptionsByTopic",
                    "sns:GetTopicAttributes",
                    "sns:TagResource",
                ],
                resources=["*"],
            )
        )

        # Backend Lambda permissions: Cost Explorer
        self.backend_lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "ce:GetCostAndUsage",
                    "ce:GetCostForecast",
                    "ce:UpdateCostAllocationTagsStatus",
                    "ce:ListCostAllocationTags",
                ],
                resources=["*"],
            )
        )

        # Backend Lambda permissions: S3 access to storage bucket
        self.storage_bucket.grant_read_write(self.backend_lambda_role)
