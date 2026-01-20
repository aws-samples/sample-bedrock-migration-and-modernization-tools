# Bedrock Model Profiler - Frontend

A React-based web application for exploring and comparing Amazon Bedrock foundation models with pricing, regional availability, and technical specifications.

## Live URL

**Production**: https://d13th0vs8a20t3.cloudfront.net

## Features

- **Model Explorer**: Browse 108+ Bedrock models from 17 providers with filtering by provider, capabilities, modalities, and regional availability
- **Model Comparison**: Compare up to 5 models side-by-side across pricing, availability, and technical specs
- **Regional Pricing**: View pricing data across all Bedrock regions
- **Responsive Design**: Fully responsive UI for mobile, tablet, and desktop
- **Dark/Light Theme**: Toggle between dark and light modes

## Prerequisites

- Node.js 18+
- npm
- AWS CLI (for deployment)
- AWS SAM CLI (for infrastructure deployment)

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The dev server runs at `http://localhost:5173` and proxies data requests to S3 using your local AWS credentials.

## Production Build

```bash
# Build for production
npm run build

# Preview production build locally
npm run preview
```

## Deployment

### First Time Setup

Run the full infrastructure setup script:

```bash
./scripts/setup-infrastructure.sh
```

This will:
1. Check that the backend stack exists
2. Deploy CloudFront + S3 infrastructure
3. Update backend with CloudFront access
4. Build and deploy frontend files

### Deploy Frontend Only

If infrastructure is already deployed, deploy just the frontend files:

```bash
npm run build
./scripts/deploy.sh
```

## Architecture

### Frontend Stack

```
CloudFront Distribution
├── Default Behavior → Frontend S3 (static files)
└── /latest/* Behavior → Data S3 (models/pricing JSON)
```

**AWS Resources:**
- CloudFront distribution with OAC
- S3 bucket for static files
- Cache policies (1 day default, 1 hour for data)
- Security headers policy (HSTS, X-Frame-Options, etc.)

### Data Flow

```
Production:
  Browser → CloudFront → S3 buckets

Development:
  Browser → Vite (localhost:5173)
            └→ Proxy → S3 data bucket
```

Data is fetched from:
- Production: `/latest/bedrock_models.json`, `/latest/bedrock_pricing.json`
- Development: `/s3-data/latest/bedrock_models.json`, `/s3-data/latest/bedrock_pricing.json` (proxied to S3)

## Project Structure

```
bedrock-model-profiler_2026/
├── src/
│   ├── components/
│   │   ├── ui/          # Radix UI primitives (button, card, dialog, etc.)
│   │   ├── layout/      # App shell (Layout, Sidebar, MainContent)
│   │   ├── models/      # Model Explorer (filters, grid, cards, pagination)
│   │   └── comparison/  # Comparison feature (tabs for overview, pricing, etc.)
│   ├── config/
│   │   └── dataSource.js    # Environment-aware data URL configuration
│   ├── hooks/
│   │   └── useModels.js     # Core data fetching hook
│   ├── stores/
│   │   └── comparisonStore.js  # Zustand store for comparison selections
│   └── utils/
│       └── filters.js       # Filter logic for model list
├── infrastructure/
│   └── template.yaml        # SAM template for CloudFront + S3
├── scripts/
│   ├── deploy.sh            # Deploy frontend to S3
│   └── setup-infrastructure.sh  # Full infrastructure setup
├── public/                  # Static assets
└── dist/                    # Production build output
```

## Configuration

### Environment Variables

Configuration is handled via `src/config/dataSource.js`:
- `S3_BUCKET`: Data bucket name
- `S3_REGION`: Data bucket region
- `S3_PREFIX`: Data path prefix (default: `latest`)

### Vite Proxy (Development)

The development server proxies `/s3-data/*` requests to S3. Configure in `vite.config.js`.

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool
- **Tailwind CSS v4** - Styling
- **Radix UI** - Accessible component primitives
- **Zustand** - State management
- **Lucide React** - Icons
- **AWS SAM** - Infrastructure as code

## Related

- **Backend**: See `bedrock-profiler-stepfunctions/` for the Step Functions data collection pipeline
- **Data**: Updated daily at 6 AM UTC via EventBridge scheduler
