# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Live URL

**Production**: https://d13th0vs8a20t3.cloudfront.net

## Frontend Commands

```bash
cd bedrock-model-profiler_2026

# Install dependencies
npm install

# Development server (proxies data from S3)
npm run dev

# Production build (outputs to dist/)
npm run build

# Preview production build locally
npm run preview

# Deploy frontend files to S3 + invalidate CloudFront
./scripts/deploy.sh

# Full infrastructure setup (first time or updates)
./scripts/setup-infrastructure.sh
```

No test or lint scripts are currently configured.

## Architecture

### Deployment Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      CloudFront Distribution                    │
│                   (bedrock-profiler-frontend-dev)              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   ┌─────────────────┐          ┌─────────────────────────┐    │
│   │ Frontend Origin │          │     Data Origin         │    │
│   │   (default)     │          │    (/latest/*)          │    │
│   └────────┬────────┘          └───────────┬─────────────┘    │
│            │                               │                   │
│            ▼                               ▼                   │
│   ┌─────────────────┐          ┌─────────────────────────┐    │
│   │ Frontend S3     │          │ Data S3 Bucket          │    │
│   │ (static files)  │          │ (bedrock-profiler-data) │    │
│   └─────────────────┘          └─────────────────────────┘    │
│                                           ▲                    │
└───────────────────────────────────────────┼────────────────────┘
                                            │
                              Step Functions Workflow
                              (daily @ 6 AM UTC)
```

### State Management
- **Zustand store** (`src/stores/comparisonStore.js`): Manages model comparison selections (max 5 models), persisted to localStorage under `bedrock-comparison-storage`
- **useModels hook** (`src/hooks/useModels.js`): Core data hook that loads model and pricing data, flattens provider/model hierarchy, extracts filter options

### Component Organization
- `src/components/ui/` - Radix UI-based primitives (button, card, dialog, select, tabs, tooltip)
- `src/components/layout/` - App shell: Layout, Sidebar, ThemeProvider, MainContent (responsive with mobile hamburger menu)
- `src/components/models/` - Model Explorer feature: ModelExplorer, ModelCard, ModelGrid, ModelFilters, Pagination
- `src/components/comparison/` - Comparison feature with tabs/ subdirectory for OverviewTab, PricingTab, AvailabilityTab, TechSpecsTab

### Data Flow
1. **Production**: CloudFront serves static files from Frontend S3, data files from Data S3 (`/latest/*` path)
2. **Development**: Vite proxies `/s3-data/*` requests to Data S3 bucket using local AWS credentials
3. `useModels()` fetches JSON data (auto-selects URL based on environment via `src/config/dataSource.js`)
4. Components receive models through hook and apply filters via `src/utils/filters.js`
5. Comparison selections flow through `useComparisonStore()` Zustand store
6. Pricing lookups use `getPricingForModel()` helper from useModels

### Styling
- Tailwind CSS v4 with dark/light theme support via CSS variables and `dark:` class
- Fully responsive design (mobile, tablet, desktop breakpoints)
- Theme managed by ThemeProvider context
- Utility merging via `cn()` function (clsx + tailwind-merge)
- Provider-specific color palette: AWS (orange), Anthropic (tan), Meta (blue), Mistral (red), Cohere (green), AI21 (purple), Stability (purple)

### Data Files (served from S3)
- `bedrock_models.json` (~5MB): 108 models from 17 providers with capabilities, modalities, regions, quotas
- `bedrock_pricing.json` (~8MB): Regional pricing for 86 models

### Frontend Infrastructure
- **Stack**: `bedrock-profiler-frontend-dev`
- **SAM Template**: `bedrock-model-profiler_2026/infrastructure/template.yaml`
- **Resources**: CloudFront distribution, Frontend S3 bucket, OAC, Cache policies, Security headers

## Backend (Step Functions)

The data collection pipeline is in `bedrock-profiler-stepfunctions/`.

**Schedule**: Runs daily at 6 AM UTC via EventBridge Scheduler

### Commands

```bash
cd bedrock-profiler-stepfunctions

# Build
sam build -t infrastructure/template.yaml

# Deploy (first time)
sam deploy --guided --stack-name bedrock-profiler-dev --capabilities CAPABILITY_NAMED_IAM

# Deploy with CloudFront access (after frontend is deployed)
sam deploy \
  --stack-name bedrock-profiler-dev \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides "CloudFrontDistributionArn=arn:aws:cloudfront::ACCOUNT:distribution/DIST_ID"

# Manual execution
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:ACCOUNT:stateMachine:bedrock-profiler-workflow-dev
```

### Architecture

Parallel Step Functions workflow with 12 Lambda functions:

```
Wave 1 (parallel):
├── Pricing (3 service codes) → Aggregate
├── Models (2 regions) → Merge
└── Quotas (20 regions)

Wave 2 (parallel, after pricing+models ready):
├── Link Pricing
├── Regional Availability
├── Features (20 regions)
└── Token Specs (LiteLLM)

Final: Aggregate all → Copy to S3 latest/
```

### Key Files
- `statemachine/bedrock-profiler.asl.json` - State machine definition
- `infrastructure/template.yaml` - SAM template (Lambda, S3, IAM, EventBridge)
- `lambdas/README.md` - Lambda function input/output contracts
- `lambdas/*/handler.py` - Lambda implementations

### Backend Infrastructure
- **Stack**: `bedrock-profiler-dev`
- **Data Bucket**: `bedrock-profiler-data-{account}-dev`
- **State Machine**: `bedrock-profiler-workflow-dev`
- **CloudFront Access**: Data bucket has policy allowing CloudFront OAC to read `/latest/*`
