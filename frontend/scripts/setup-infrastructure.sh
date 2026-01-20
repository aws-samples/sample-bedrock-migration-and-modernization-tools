#!/bin/bash
set -e

# Bedrock Model Profiler - Full Infrastructure Setup
# This script deploys both backend and frontend infrastructure

ENVIRONMENT="${ENVIRONMENT:-dev}"
REGION="${AWS_REGION:-us-east-1}"
BACKEND_STACK="bedrock-profiler-${ENVIRONMENT}"
FRONTEND_STACK="bedrock-profiler-frontend-${ENVIRONMENT}"

echo "=========================================="
echo "Bedrock Model Profiler - Infrastructure Setup"
echo "=========================================="
echo "Environment: ${ENVIRONMENT}"
echo "Region: ${REGION}"
echo ""

# Check for required tools
command -v aws >/dev/null 2>&1 || { echo "Error: AWS CLI is required"; exit 1; }
command -v sam >/dev/null 2>&1 || { echo "Error: SAM CLI is required"; exit 1; }

# Step 1: Check if backend stack exists
echo "Step 1: Checking backend stack..."
BACKEND_EXISTS=$(aws cloudformation describe-stacks \
    --stack-name "$BACKEND_STACK" \
    --region "$REGION" \
    --query "Stacks[0].StackStatus" \
    --output text 2>/dev/null || echo "NOT_FOUND")

if [ "$BACKEND_EXISTS" == "NOT_FOUND" ]; then
    echo "Backend stack not found. Please deploy the backend first:"
    echo ""
    echo "  cd infra"
    echo "  sam build -t backend-template.yaml"
    echo "  sam deploy --guided"
    echo ""
    exit 1
fi
echo "Backend stack found: $BACKEND_EXISTS"

# Get backend data bucket name
DATA_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name "$BACKEND_STACK" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" \
    --output text)

echo "Data bucket: ${DATA_BUCKET}"

# Step 2: Deploy frontend infrastructure
echo ""
echo "Step 2: Deploying frontend infrastructure..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../../infra"

sam build -t frontend-template.yaml

sam deploy \
    --stack-name "$FRONTEND_STACK" \
    --region "$REGION" \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides "Environment=${ENVIRONMENT}" \
    --no-confirm-changeset \
    --no-fail-on-empty-changeset

# Get CloudFront distribution ARN
DISTRIBUTION_ID=$(aws cloudformation describe-stacks \
    --stack-name "$FRONTEND_STACK" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='CloudFrontDistributionId'].OutputValue" \
    --output text)

# Get AWS Account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
CLOUDFRONT_ARN="arn:aws:cloudfront::${ACCOUNT_ID}:distribution/${DISTRIBUTION_ID}"

echo "CloudFront Distribution ID: ${DISTRIBUTION_ID}"
echo "CloudFront ARN: ${CLOUDFRONT_ARN}"

# Step 3: Update backend stack with CloudFront ARN for bucket policy
echo ""
echo "Step 3: Updating backend stack with CloudFront access..."
cd "$SCRIPT_DIR/../../infra"

sam build -t backend-template.yaml

sam deploy \
    --stack-name "$BACKEND_STACK" \
    --region "$REGION" \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameter-overrides "Environment=${ENVIRONMENT} CloudFrontDistributionArn=${CLOUDFRONT_ARN}" \
    --no-confirm-changeset \
    --no-fail-on-empty-changeset

# Step 4: Build and deploy frontend files
echo ""
echo "Step 4: Building and deploying frontend..."
cd "$SCRIPT_DIR/.."
npm install
npm run build
./scripts/deploy.sh

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
echo "Your application is now available at:"
echo "${CLOUDFRONT_URL}"
echo ""
