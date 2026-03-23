#!/bin/bash
set -e

# Bedrock Model Profiler - Cleanup Script
# Removes all deployed resources and stops incurring charges.
#
# Usage:
#   ./cleanup.sh <stack-name>
#   STACK_NAME=bmp ./cleanup.sh

REGION="${AWS_REGION:-us-east-1}"
STACK_NAME="${STACK_NAME:-${1:-}}"

if [ -z "$STACK_NAME" ]; then
    echo "Usage: ./cleanup.sh <stack-name>"
    echo ""
    echo "  Example: ./cleanup.sh bmp"
    echo "  Example: AWS_REGION=eu-west-1 ./cleanup.sh bmp"
    echo ""
    echo "This will delete both stacks:"
    echo "  <stack-name>-frontend  — CloudFront, S3 buckets"
    echo "  <stack-name>-backend   — Lambdas, Step Functions, EventBridge"
    exit 1
fi

FRONTEND_STACK="${STACK_NAME}-frontend"
BACKEND_STACK="${STACK_NAME}-backend"

echo "==========================================="
echo "Bedrock Model Profiler - Cleanup"
echo "==========================================="
echo "Frontend Stack: ${FRONTEND_STACK}"
echo "Backend Stack:  ${BACKEND_STACK}"
echo "Region:         ${REGION}"
echo ""

read -p "Are you sure you want to delete all resources? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# 1. Empty the S3 buckets (required before stack deletion)
echo ""
echo "Step 1: Emptying S3 buckets..."

DATA_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name "$FRONTEND_STACK" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" \
    --output text 2>/dev/null || echo "None")

FRONTEND_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name "$FRONTEND_STACK" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='FrontendBucketName'].OutputValue" \
    --output text 2>/dev/null || echo "None")

if [ -n "$DATA_BUCKET" ] && [ "$DATA_BUCKET" != "None" ]; then
    echo "Emptying data bucket: ${DATA_BUCKET}"
    aws s3 rm "s3://${DATA_BUCKET}" --recursive --region "$REGION"
fi

if [ -n "$FRONTEND_BUCKET" ] && [ "$FRONTEND_BUCKET" != "None" ]; then
    echo "Emptying frontend bucket: ${FRONTEND_BUCKET}"
    aws s3 rm "s3://${FRONTEND_BUCKET}" --recursive --region "$REGION"
fi

# 2. Delete backend stack first (it imports from frontend stack)
echo ""
echo "Step 2: Deleting backend stack..."
aws cloudformation delete-stack \
    --stack-name "$BACKEND_STACK" \
    --region "$REGION"

echo "Waiting for backend stack deletion..."
aws cloudformation wait stack-delete-complete \
    --stack-name "$BACKEND_STACK" \
    --region "$REGION"

echo "Backend stack deleted."

# 3. Delete frontend stack
echo ""
echo "Step 3: Deleting frontend stack..."
aws cloudformation delete-stack \
    --stack-name "$FRONTEND_STACK" \
    --region "$REGION"

echo "Waiting for frontend stack deletion..."
aws cloudformation wait stack-delete-complete \
    --stack-name "$FRONTEND_STACK" \
    --region "$REGION"

echo ""
echo "==========================================="
echo "All resources deleted."
echo "==========================================="
