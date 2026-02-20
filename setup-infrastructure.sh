#!/bin/bash
set -e

# Bedrock Model Profiler - Full Infrastructure Setup
# This script deploys both backend and frontend infrastructure

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENVIRONMENT="${ENVIRONMENT:-dev}"
REGION="${AWS_REGION:-us-east-1}"
BACKEND_STACK="bedrock-profiler-${ENVIRONMENT}"
FRONTEND_STACK="bedrock-profiler-frontend-${ENVIRONMENT}"
ANALYTICS_STACK="bedrock-profiler-analytics-${ENVIRONMENT}"
DOMAIN_NAME="${DOMAIN_NAME:-}"
HOSTED_ZONE_ID="${HOSTED_ZONE_ID:-}"

echo "=========================================="
echo "Bedrock Model Profiler - Infrastructure Setup"
echo "=========================================="
echo "Environment: ${ENVIRONMENT}"
echo "Region: ${REGION}"
if [ -n "$DOMAIN_NAME" ]; then
    echo "Custom Domain: ${DOMAIN_NAME}"
    echo "Hosted Zone ID: ${HOSTED_ZONE_ID}"
fi
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
cd "$SCRIPT_DIR/infra"

sam build -t frontend-template.yaml

# Build parameter overrides
FRONTEND_PARAMS="Environment=${ENVIRONMENT}"
if [ -n "$DOMAIN_NAME" ] && [ -n "$HOSTED_ZONE_ID" ]; then
    FRONTEND_PARAMS="${FRONTEND_PARAMS} DomainName=${DOMAIN_NAME} HostedZoneId=${HOSTED_ZONE_ID}"
fi

sam deploy \
    --stack-name "$FRONTEND_STACK" \
    --region "$REGION" \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides "$FRONTEND_PARAMS" \
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
sam build -t backend-template.yaml

sam deploy \
    --stack-name "$BACKEND_STACK" \
    --region "$REGION" \
    --capabilities CAPABILITY_NAMED_IAM \
    --resolve-s3 \
    --parameter-overrides "Environment=${ENVIRONMENT} CloudFrontDistributionArn=${CLOUDFRONT_ARN}" \
    --no-confirm-changeset \
    --no-fail-on-empty-changeset

# Step 4: Deploy analytics stack
echo ""
echo "Step 4: Deploying analytics stack..."

# Get CloudFront URL for AllowedOrigins
CLOUDFRONT_URL=$(aws cloudformation describe-stacks \
    --stack-name "$FRONTEND_STACK" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='CloudFrontURL'].OutputValue" \
    --output text)

# Extract Cognito values from frontend/.env
ENV_FILE="$SCRIPT_DIR/frontend/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Warning: frontend/.env not found. Skipping analytics stack deployment."
    echo "  Create frontend/.env with VITE_COGNITO_AUTHORITY_URL and VITE_COGNITO_CLIENT_ID to enable analytics."
else
    COGNITO_AUTHORITY_URL=$(grep '^VITE_COGNITO_AUTHORITY_URL=' "$ENV_FILE" | cut -d'=' -f2-)
    COGNITO_CLIENT_ID=$(grep '^VITE_COGNITO_CLIENT_ID=' "$ENV_FILE" | cut -d'=' -f2-)

    if [ -z "$COGNITO_AUTHORITY_URL" ] || [ -z "$COGNITO_CLIENT_ID" ]; then
        echo "Warning: Cognito values not found in frontend/.env. Skipping analytics stack deployment."
        echo "  Set VITE_COGNITO_AUTHORITY_URL and VITE_COGNITO_CLIENT_ID to enable analytics."
    else
        # Extract User Pool ID (last path segment of the authority URL)
        COGNITO_USER_POOL_ID=$(echo "$COGNITO_AUTHORITY_URL" | awk -F'/' '{print $NF}')
        # Extract Cognito region from the authority URL
        COGNITO_REGION=$(echo "$COGNITO_AUTHORITY_URL" | sed -n 's|.*cognito-idp\.\([^.]*\)\.amazonaws\.com.*|\1|p')
        COGNITO_REGION="${COGNITO_REGION:-us-east-1}"

        echo "Cognito User Pool ID: ${COGNITO_USER_POOL_ID}"
        echo "Cognito Client ID: ${COGNITO_CLIENT_ID}"
        echo "Cognito Region: ${COGNITO_REGION}"

        # Build AllowedOrigins: CloudFront URL + localhost for dev
        ALLOWED_ORIGINS="${CLOUDFRONT_URL},http://localhost:5173"

        # Add custom domain to AllowedOrigins if configured
        if [ -n "$DOMAIN_NAME" ]; then
            ALLOWED_ORIGINS="${ALLOWED_ORIGINS},https://${DOMAIN_NAME}"
        fi

        echo "Allowed Origins: ${ALLOWED_ORIGINS}"

        cd "$SCRIPT_DIR/infra"
        sam build -t analytics-template.yaml

        ANALYTICS_PARAMS="Environment=${ENVIRONMENT}"
        ANALYTICS_PARAMS="${ANALYTICS_PARAMS} CognitoUserPoolId=${COGNITO_USER_POOL_ID}"
        ANALYTICS_PARAMS="${ANALYTICS_PARAMS} CognitoClientId=${COGNITO_CLIENT_ID}"
        ANALYTICS_PARAMS="${ANALYTICS_PARAMS} CognitoRegion=${COGNITO_REGION}"
        ANALYTICS_PARAMS="${ANALYTICS_PARAMS} AllowedOrigins=${ALLOWED_ORIGINS}"

        sam deploy \
            --stack-name "$ANALYTICS_STACK" \
            --region "$REGION" \
            --capabilities CAPABILITY_IAM \
            --parameter-overrides "$ANALYTICS_PARAMS" \
            --no-confirm-changeset \
            --no-fail-on-empty-changeset

        ANALYTICS_API_URL=$(aws cloudformation describe-stacks \
            --stack-name "$ANALYTICS_STACK" \
            --region "$REGION" \
            --query "Stacks[0].Outputs[?OutputKey=='AnalyticsApiUrl'].OutputValue" \
            --output text)

        echo "Analytics API URL: ${ANALYTICS_API_URL}"
    fi
fi

# Step 5: Build and deploy frontend files
echo ""
echo "Step 5: Building and deploying frontend..."
cd "$SCRIPT_DIR/frontend"
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

# Show analytics API URL if analytics stack was deployed
ANALYTICS_API_URL=$(aws cloudformation describe-stacks \
    --stack-name "$ANALYTICS_STACK" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='AnalyticsApiUrl'].OutputValue" \
    --output text 2>/dev/null || echo "")

if [ -n "$ANALYTICS_API_URL" ] && [ "$ANALYTICS_API_URL" != "None" ]; then
    echo "Analytics API: ${ANALYTICS_API_URL}"
fi

# Show custom domain URL if configured
CUSTOM_DOMAIN_URL=$(aws cloudformation describe-stacks \
    --stack-name "$FRONTEND_STACK" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='CustomDomainURL'].OutputValue" \
    --output text 2>/dev/null || echo "None")

if [ -n "$CUSTOM_DOMAIN_URL" ] && [ "$CUSTOM_DOMAIN_URL" != "None" ]; then
    echo "Custom domain: ${CUSTOM_DOMAIN_URL}"
fi
echo ""
