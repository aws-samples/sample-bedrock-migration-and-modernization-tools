# Bedrock Model Profiler

A web application for exploring and comparing Amazon Bedrock foundation models with pricing, regional availability, and technical specifications.

## Live URL

**Production**: https://d13th0vs8a20t3.cloudfront.net

## Project Structure

```
bedrock-model-profiler/
├── frontend/          # React + Vite web application
├── backend/           # AWS Lambda functions + Step Functions state machine
└── infra/             # SAM templates (CloudFormation)
```

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.11
- AWS CLI configured
- AWS SAM CLI

### Development

```bash
cd frontend
npm install
npm run dev
```

### Deployment

**Full deployment** (infrastructure + code):
```bash
cd frontend
./scripts/setup-infrastructure.sh
```

**Frontend only** (after infrastructure exists):
```bash
cd frontend
npm run build
./scripts/deploy.sh
```

**Backend only**:
```bash
cd infra
sam build -t backend-template.yaml
sam deploy --stack-name bedrock-profiler-dev --capabilities CAPABILITY_NAMED_IAM
```

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      CloudFront Distribution                    │
├────────────────────────────────────────────────────────────────┤
│   ┌─────────────────┐          ┌─────────────────────────┐    │
│   │ Frontend Origin │          │     Data Origin         │    │
│   │   (default)     │          │    (/latest/*)          │    │
│   └────────┬────────┘          └───────────┬─────────────┘    │
│            │                               │                   │
│            ▼                               ▼                   │
│   ┌─────────────────┐          ┌─────────────────────────┐    │
│   │ Frontend S3     │          │ Data S3 Bucket          │    │
│   │ (static files)  │          │ (models + pricing JSON) │    │
│   └─────────────────┘          └─────────────────────────┘    │
│                                           ▲                    │
└───────────────────────────────────────────┼────────────────────┘
                                            │
                              Step Functions Workflow
                              (daily @ 6 AM UTC)
```

### Backend Workflow

Parallel Step Functions workflow with 12 Lambda functions:

```
Wave 1 (parallel):
├── Pricing (3 service codes) → Aggregate
├── Models (2 regions) → Merge
└── Quotas (20 regions)

Wave 2 (parallel):
├── Link Pricing
├── Regional Availability
├── Features (20 regions)
└── Token Specs (LiteLLM)

Final: Aggregate all → Copy to S3 latest/
```

## Features

- **Model Explorer**: Browse 108+ Bedrock models from 17 providers
- **Model Comparison**: Compare up to 5 models side-by-side
- **Regional Pricing**: View pricing across all Bedrock regions
- **Responsive Design**: Mobile, tablet, and desktop support
- **Dark/Light Theme**: Toggle between themes

## AWS Stacks

| Stack | Description |
|-------|-------------|
| `bedrock-profiler-dev` | Backend (Lambda, Step Functions, S3 data bucket) |
| `bedrock-profiler-frontend-dev` | Frontend (CloudFront, S3 static files) |

## Tech Stack

**Frontend**: React 18, Vite, Tailwind CSS v4, Radix UI, Zustand

**Backend**: Python 3.11, AWS Lambda, Step Functions, EventBridge

**Infrastructure**: AWS SAM, CloudFront, S3, IAM
