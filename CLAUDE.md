# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Amazon Bedrock Model Profiler - A full-stack tool for exploring, analyzing, and comparing Amazon Bedrock foundation models. Live at https://d3oem6l61p8j11.cloudfront.net

## Commands

### Frontend Development
```bash
cd frontend
npm install          # Install dependencies
npm run dev          # Start dev server at localhost:5173
npm run build        # Production build
npm run preview      # Preview production build
```

### Backend Deployment
```bash
cd infra
sam build -t backend-template.yaml
sam deploy --stack-name bedrock-profiler-dev --capabilities CAPABILITY_NAMED_IAM --resolve-s3
```

### Full Deployment
```bash
./setup-infrastructure.sh    # Deploy backend + frontend infrastructure
cd frontend && ./scripts/deploy.sh  # Deploy frontend files only
```

### Backend Testing
```bash
cd backend/tests
python test_pricing_collector_local.py
python test_workflow_local.py
```

## Architecture

### Data Flow
```
Step Functions Workflow (daily at 6 AM UTC)
    ├→ pricing-collector (3x parallel) → pricing-aggregator
    ├→ model-extractor (2 regions) → model-merger
    └→ quota-collector (20 regions parallel)
         ↓
    pricing-linker + regional-availability + feature-collector + token-specs-collector
         ↓
    final-aggregator → copy-to-latest
         ↓
    S3: latest/bedrock_models.json, latest/bedrock_pricing.json
         ↓
    CloudFront → Frontend
```

### Project Structure
- **frontend/**: React + Vite + Tailwind CSS + Radix UI
- **backend/lambdas/**: 11 Python Lambda functions (see backend/lambdas/README.md for contracts)
- **backend/layers/common/**: Shared utilities (S3, config, validation)
- **backend/statemachine/**: Step Functions workflow definition (ASL)
- **infra/**: SAM templates (CloudFormation)

### Frontend Architecture
- **State**: Zustand stores for model comparison (`stores/comparisonStore.js`) and auth (`stores/authStore.js`)
- **Auth**: AWS Cognito OIDC via `react-oidc-context` (see `auth/` directory)
- **Data**: Custom hook `useModels.js` fetches from S3 via CloudFront (prod) or Vite proxy (dev)
- **Components**: Radix UI primitives in `components/ui/`, features in `components/models/` and `components/comparison/`
- **Data source**: `config/dataSource.js` handles environment-aware URLs

### Backend Architecture
- All Lambdas import from shared layer: `from shared import s3_utils, config, validation, execution`
- Consistent response format: `{status: "SUCCESS"|"FAILED", ...metadata}`
- Retry config: 3 retries with exponential backoff for throttling
- S3 data stored in `executions/{execution-id}/` with final output copied to `latest/`

## Key Patterns

### Lambda Handler Pattern
```python
import boto3
from shared import s3_utils, config, validation

def lambda_handler(event, context):
    # Validate input
    validation.require_params(event, ['s3Bucket', 's3Key'])
    # Process
    # Write results to S3
    s3_utils.write_json(bucket, key, data)
    # Return structured response
    return {"status": "SUCCESS", ...}
```

### Frontend Data Flow
Development uses Vite S3 proxy plugin (vite.config.js) with local AWS credentials. Production fetches directly from CloudFront `/latest/*` paths.

## Authentication Setup

The frontend uses AWS Cognito for authentication. To configure:

1. Copy `frontend/template.env` to `frontend/.env`
2. Set the Cognito environment variables:
   - `VITE_COGNITO_AUTHORITY_URL`: Cognito User Pool authority (e.g., `https://cognito-idp.us-east-1.amazonaws.com/us-east-1_xxxxx`)
   - `VITE_COGNITO_CLIENT_ID`: Cognito App Client ID

If environment variables are not set, the app runs without authentication.

## AWS Services Used
- **Cognito**: User authentication (OIDC)
- **Bedrock**: ListFoundationModels, ListInferenceProfiles
- **Pricing API**: GetProducts (us-east-1 only)
- **Service Quotas**: ListServiceQuotas
- **Step Functions**: Workflow orchestration
- **S3**: Data storage, frontend hosting
- **CloudFront**: CDN distribution
