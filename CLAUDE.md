# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Amazon Bedrock Model Profiler - A full-stack serverless tool for exploring, analyzing, and comparing Amazon Bedrock foundation models. Features a self-healing data pipeline with 18 Lambda functions, inter-Lambda caching, and Claude-powered gap detection.

## Commands

### Frontend Development
```bash
cd frontend
npm install          # Install dependencies
npm run dev          # Start dev server at localhost:5173
npm run build        # Production build
npm run preview      # Preview production build
```

### Local Data Collection
```bash
pip install -r local/requirements.txt
python -m local collect
python -m local collect --profile my-aws-profile --output ./data
```

### Backend-Only Deployment
```bash
cd infra
sam build -t backend-template.yaml
sam deploy --stack-name bmp-backend --capabilities CAPABILITY_NAMED_IAM --resolve-s3 \
  --parameter-overrides "DataBucketName=<data-bucket> FrontendStackName=bmp-frontend"
```

### Full AWS Deployment
```bash
./setup-infrastructure.sh bmp                                                      # Deploy everything
AWS_REGION=eu-west-1 ./setup-infrastructure.sh bmp                                 # Deploy to a specific region
cd frontend && AWS_REGION=eu-west-1 FRONTEND_STACK=bmp-frontend ./scripts/deploy.sh  # Redeploy frontend only
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
Step Functions Workflow (daily)
  Phase 0: region-discovery → config-sync
  Phase 1 (parallel):
    ├→ pricing-collector (3x service codes) → pricing-aggregator
    ├→ model-extractor (27+ regions, caches to S3) → model-merger
    └→ quota-collector (N regions)
  Phase 2 (parallel):
    ├→ pricing-linker + regional-availability (from cache) + feature-collector (from cache)
    └→ token-specs-collector + mantle-collector + lifecycle-collector
  Phase 3:
    final-aggregator → gap-detection → [self-healing-agent] → copy-to-latest
      ↓
    S3: latest/bedrock_models.json, latest/bedrock_pricing.json
      ↓
    CloudFront → Frontend
```

### Project Structure
- **frontend/**: React 18 + Vite + Tailwind CSS v4 + Radix UI
- **backend/lambdas/**: 17 Python Lambda functions (see backend/lambdas/README.md for contracts)
- **backend/layers/common/**: Shared utilities (model_matcher, cache_utils, config_loader, s3_utils, validation)
- **backend/statemachine/**: Step Functions workflow definition (ASL)
- **backend/config/**: Externalized configuration (profiler-config.json)
- **backend/tests/**: ~150 tests (pytest)
- **local/**: CLI tool for local data collection
- **data/**: Sample data files
- **infra/**: SAM templates (CloudFormation)
- **docs/**: Architecture docs, data sources, analysis

### Frontend Architecture
- **State**: Zustand stores for comparison (`stores/comparisonStore.js`) and favorites (`stores/favoritesStore.js`)
- **Auth**: Optional AWS Cognito OIDC via `react-oidc-context` — app runs without auth if env vars not set
- **Data**: Custom hook `useModels.js` fetches models + pricing JSON from S3 via CloudFront (prod) or Vite proxy (dev)
- **Components**: Radix UI primitives in `components/ui/`, features in `components/models/` and `components/comparison/`
- **Config**: `config/constants.js` + `config/generated-constants.js` for app constants, `config/dataSource.js` for environment-aware URLs

### Backend Architecture
- 18 Lambda functions (Python 3.11) orchestrated by Step Functions
- All Lambdas import from shared layer: `from shared import s3_utils, config, validation, model_matcher, cache_utils`
- Consistent response format: `{status: "SUCCESS"|"FAILED", ...metadata}`
- Retry config: 3 retries with exponential backoff for throttling
- Inter-Lambda S3 caching: ~97% cache hit rate, ~29 API calls per execution (down from ~480)
- Self-healing: gap-detection → Claude auto-applies safe config fixes
- Externalized config in `backend/config/profiler-config.json`
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

## Authentication (Optional)

The frontend supports optional AWS Cognito authentication. To configure:

1. Copy `frontend/template.env` to `frontend/.env`
2. Set the Cognito environment variables:
   - `VITE_COGNITO_AUTHORITY_URL`: Cognito User Pool authority
   - `VITE_COGNITO_CLIENT_ID`: Cognito App Client ID

If environment variables are not set, the app runs without authentication.

## AWS Services Used
- **Bedrock**: ListFoundationModels, ListInferenceProfiles, InvokeModel (self-healing agent)
- **Pricing API**: GetProducts (us-east-1 only, 3 service codes)
- **Service Quotas**: ListServiceQuotas (27+ regions)
- **Step Functions**: Workflow orchestration (4 phases)
- **S3**: Data storage, frontend hosting, inter-Lambda caching, config storage
- **CloudFront**: CDN distribution with OAC
- **EventBridge**: Scheduled execution
- **Lambda Powertools**: Structured logging, tracing, metrics
