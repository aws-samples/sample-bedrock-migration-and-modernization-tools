#!/bin/bash
set -e

# Bedrock Model Profiler - Frontend Deployment Script
# Uploads built files to S3 and invalidates CloudFront cache

# Configuration
STACK_NAME="${FRONTEND_STACK:-${1:-}}"
REGION="${AWS_REGION:-us-east-1}"

if [ -z "$STACK_NAME" ]; then
    echo "Usage: ./scripts/deploy.sh <frontend-stack-name>"
    echo "  Example: ./scripts/deploy.sh bmp-frontend"
    echo "  Example: FRONTEND_STACK=bmp-frontend ./scripts/deploy.sh"
    exit 1
fi
DIST_DIR="dist"

echo "=========================================="
echo "Bedrock Model Profiler - Frontend Deploy"
echo "=========================================="
echo "Stack Name: ${STACK_NAME}"
echo "Region: ${REGION}"
echo ""

# Check if dist directory exists
if [ ! -d "$DIST_DIR" ]; then
    echo "Error: $DIST_DIR directory not found. Run 'npm run build' first."
    exit 1
fi

# Get S3 bucket name from CloudFormation stack
echo "Fetching S3 bucket name from CloudFormation..."
BUCKET_NAME=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='FrontendBucketName'].OutputValue" \
    --output text)

if [ -z "$BUCKET_NAME" ] || [ "$BUCKET_NAME" == "None" ]; then
    echo "Error: Could not find S3 bucket. Make sure the infrastructure is deployed."
    echo "Run: ./setup-infrastructure.sh <stack-name>"
    exit 1
fi

echo "S3 Bucket: ${BUCKET_NAME}"

# Get CloudFront distribution ID
echo "Fetching CloudFront distribution ID..."
DISTRIBUTION_ID=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='CloudFrontDistributionId'].OutputValue" \
    --output text)

echo "Distribution ID: ${DISTRIBUTION_ID}"

# Sync files to S3
echo ""
echo "Uploading files to S3..."
aws s3 sync "$DIST_DIR" "s3://${BUCKET_NAME}" \
    --delete \
    --cache-control "public, max-age=31536000, immutable" \
    --exclude "index.html" \
    --region "$REGION"

# Upload index.html with no-cache
aws s3 cp "$DIST_DIR/index.html" "s3://${BUCKET_NAME}/index.html" \
    --cache-control "no-cache, no-store, must-revalidate" \
    --content-type "text/html" \
    --region "$REGION"

echo "Upload complete!"

# Invalidate CloudFront cache
if [ -n "$DISTRIBUTION_ID" ] && [ "$DISTRIBUTION_ID" != "None" ]; then
    echo ""
    echo "Invalidating CloudFront cache..."
    INVALIDATION_ID=$(aws cloudfront create-invalidation \
        --distribution-id "$DISTRIBUTION_ID" \
        --paths "/*" \
        --query "Invalidation.Id" \
        --output text)
    echo "Invalidation created: ${INVALIDATION_ID}"
fi

# Get the CloudFront URL
CLOUDFRONT_URL=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='CloudFrontURL'].OutputValue" \
    --output text)

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo "URL: ${CLOUDFRONT_URL}"

# Show custom domain URL if configured
CUSTOM_DOMAIN_URL=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='CustomDomainURL'].OutputValue" \
    --output text 2>/dev/null || echo "None")

if [ -n "$CUSTOM_DOMAIN_URL" ] && [ "$CUSTOM_DOMAIN_URL" != "None" ]; then
    echo "Custom domain: ${CUSTOM_DOMAIN_URL}"
fi
echo ""
echo "Note: CloudFront cache invalidation may take a few minutes to propagate."
