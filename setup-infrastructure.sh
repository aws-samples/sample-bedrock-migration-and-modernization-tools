#!/bin/bash
set -e

# Bedrock Model Profiler - Full Infrastructure Setup
#
# Deploy order:
#   1. Frontend stack (creates data bucket + CloudFront + frontend S3)
#   2. Seed placeholder data in S3 (prevents broken site while pipeline runs)
#   3. Backend stack (Lambdas + Step Functions, uses data bucket from frontend)
#   4. Build and upload frontend files
#   5. Trigger data pipeline
#
# Usage:
#   ./setup-infrastructure.sh <stack-name>
#   STACK_NAME=bmp ./setup-infrastructure.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REGION="${AWS_REGION:-us-east-1}"
STACK_NAME="${STACK_NAME:-${1:-}}"
DOMAIN_NAME="${DOMAIN_NAME:-}"
HOSTED_ZONE_ID="${HOSTED_ZONE_ID:-}"

if [ -z "$STACK_NAME" ]; then
    echo "Usage: ./setup-infrastructure.sh <stack-name>"
    echo ""
    echo "  Example: ./setup-infrastructure.sh bmp"
    echo "  Example: STACK_NAME=bmp ./setup-infrastructure.sh"
    echo ""
    echo "This will create two CloudFormation stacks:"
    echo "  <stack-name>-frontend  — CloudFront, S3 buckets (frontend + data)"
    echo "  <stack-name>-backend   — Lambdas, Step Functions, EventBridge"
    exit 1
fi

FRONTEND_STACK="${STACK_NAME}-frontend"
BACKEND_STACK="${STACK_NAME}-backend"

echo "=========================================="
echo "Bedrock Model Profiler - Infrastructure Setup"
echo "=========================================="
echo "Frontend Stack: ${FRONTEND_STACK}"
echo "Backend Stack:  ${BACKEND_STACK}"
echo "Region:         ${REGION}"
if [ -n "$DOMAIN_NAME" ]; then
    echo "Custom Domain:  ${DOMAIN_NAME}"
fi
echo ""

# Check for required tools
command -v aws >/dev/null 2>&1 || { echo "Error: AWS CLI is required"; exit 1; }
command -v sam >/dev/null 2>&1 || { echo "Error: SAM CLI is required"; exit 1; }

# ==========================================================
# Step 1: Deploy frontend infrastructure
# Creates: data S3 bucket, frontend S3 bucket, CloudFront
# ==========================================================
echo "Step 1: Deploying frontend infrastructure..."
cd "$SCRIPT_DIR/infra"

sam build -t frontend-template.yaml

FRONTEND_DEPLOY_ARGS="--stack-name $FRONTEND_STACK --region $REGION --capabilities CAPABILITY_IAM --resolve-s3 --no-confirm-changeset --no-fail-on-empty-changeset"
if [ -n "$DOMAIN_NAME" ] && [ -n "$HOSTED_ZONE_ID" ]; then
    FRONTEND_DEPLOY_ARGS="$FRONTEND_DEPLOY_ARGS --parameter-overrides DomainName=${DOMAIN_NAME} HostedZoneId=${HOSTED_ZONE_ID}"
fi

sam deploy $FRONTEND_DEPLOY_ARGS

# Read outputs from frontend stack
DATA_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name "$FRONTEND_STACK" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" \
    --output text)

DISTRIBUTION_ID=$(aws cloudformation describe-stacks \
    --stack-name "$FRONTEND_STACK" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='CloudFrontDistributionId'].OutputValue" \
    --output text)

echo "Data bucket:      ${DATA_BUCKET}"
echo "Distribution ID:  ${DISTRIBUTION_ID}"

# ==========================================================
# Step 2: Seed placeholder data
# Prevents "Failed to load models" while pipeline runs
# ==========================================================
echo ""
echo "Step 2: Seeding placeholder data..."

TMPDIR=$(mktemp -d)
echo '{"metadata":{"pipeline_status":"pending","message":"Data pipeline is running. Please refresh in a few minutes."},"providers":{}}' > "$TMPDIR/models.json"
echo '{"metadata":{"pipeline_status":"pending"},"providers":{}}' > "$TMPDIR/pricing.json"

aws s3 cp "$TMPDIR/models.json" "s3://${DATA_BUCKET}/latest/bedrock_models.json" \
    --content-type "application/json" --region "$REGION" > /dev/null
aws s3 cp "$TMPDIR/pricing.json" "s3://${DATA_BUCKET}/latest/bedrock_pricing.json" \
    --content-type "application/json" --region "$REGION" > /dev/null

rm -rf "$TMPDIR"

echo "Placeholder data uploaded."

# ==========================================================
# Step 3: Deploy backend (single deploy — no circular dependency)
# ==========================================================
echo ""
echo "Step 3: Deploying backend..."

sam build -t backend-template.yaml

sam deploy \
    --stack-name "$BACKEND_STACK" \
    --region "$REGION" \
    --capabilities CAPABILITY_NAMED_IAM \
    --resolve-s3 \
    --parameter-overrides "DataBucketName=${DATA_BUCKET} FrontendStackName=${FRONTEND_STACK}" \
    --no-confirm-changeset \
    --no-fail-on-empty-changeset

echo "Backend deployed."

# ==========================================================
# Step 4: Build and deploy frontend files
# ==========================================================
echo ""
echo "Step 4: Building and deploying frontend..."
cd "$SCRIPT_DIR/frontend"
npm install
npm run build
FRONTEND_STACK="$FRONTEND_STACK" ./scripts/deploy.sh

# ==========================================================
# Step 5: Trigger the data pipeline
# ==========================================================
echo ""
echo "Step 5: Triggering data pipeline..."

STATE_MACHINE_ARN=$(aws cloudformation describe-stacks \
    --stack-name "$BACKEND_STACK" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='StateMachineArn'].OutputValue" \
    --output text)

if [ -n "$STATE_MACHINE_ARN" ] && [ "$STATE_MACHINE_ARN" != "None" ]; then
    EXECUTION_ARN=$(aws stepfunctions start-execution \
        --state-machine-arn "$STATE_MACHINE_ARN" \
        --region "$REGION" \
        --query "executionArn" \
        --output text)
    echo "Data pipeline started: ${EXECUTION_ARN}"
    echo "The pipeline takes ~2-3 minutes. Data will appear on the site once finished."
else
    echo "Warning: Could not find state machine. Trigger the pipeline manually."
fi

# ==========================================================
# Done
# ==========================================================
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="

CLOUDFRONT_URL=$(aws cloudformation describe-stacks \
    --stack-name "$FRONTEND_STACK" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='CloudFrontURL'].OutputValue" \
    --output text)

echo ""
echo "Your application is available at:"
echo "  ${CLOUDFRONT_URL}"

CUSTOM_DOMAIN_URL=$(aws cloudformation describe-stacks \
    --stack-name "$FRONTEND_STACK" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='CustomDomainURL'].OutputValue" \
    --output text 2>/dev/null || echo "None")

if [ -n "$CUSTOM_DOMAIN_URL" ] && [ "$CUSTOM_DOMAIN_URL" != "None" ]; then
    echo "  ${CUSTOM_DOMAIN_URL}"
fi
echo ""
echo "Data freshness:"
echo "  - The data pipeline was triggered and takes ~2-3 minutes to complete."
echo "  - After that, data refreshes automatically twice daily (6 AM and 6 PM UTC)."
echo "  - Model availability and pricing can change at any time — always verify"
echo "    in the AWS Console before making production decisions."
echo "  - To trigger a manual refresh, start the Step Functions workflow:"
echo "    aws stepfunctions start-execution --state-machine-arn \\"
echo "      $([ -n "$STATE_MACHINE_ARN" ] && echo "$STATE_MACHINE_ARN" || echo "<state-machine-arn>")"
echo ""
