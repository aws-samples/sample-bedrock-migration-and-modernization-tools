# Amazon Bedrock Model Profiler

A comprehensive profiling tool for exploring, analyzing, and comparing Amazon Bedrock foundation models. Whether you're building new applications, optimizing existing workloads, or migrating from other AI services, this tool provides deep insights into model capabilities, pricing, and regional availability across 100+ models from 17+ providers.

## Features

### Model Discovery and Exploration
Browse the complete Bedrock model catalog with powerful filtering:
- 13 filter types: provider, region, status, CRIS scope, streaming, context window, modality, capabilities, use cases, customization, languages, consumption options, and geographic region
- Sorting by name, provider, context window size, or pricing
- Adjustable grid density and pagination
- Light and dark theme support

### Model Comparison Framework
Compare up to 5 models side-by-side with four specialized views:
- **Overview** — Capabilities, specifications, and metadata at a glance
- **Pricing** — Per-1M-token costs with region-level granularity (token, image, video, and search unit pricing)
- **Availability** — Interactive region map showing on-demand, CRIS, Mantle, batch, and provisioned availability
- **Tech Specs** — Context windows, token limits, streaming support, and customization options

### Pricing Analysis
Detailed pricing information from AWS Pricing APIs:
- Input/output token pricing per 1M tokens
- Region-by-region price comparison
- Multiple pricing models: token, image generation, video generation, video per-second, search units
- On-demand, batch, and provisioned throughput pricing groups

### Additional Features
- **Favorites** — Save a personal shortlist of models for quick access (persisted in browser)

## Getting Started

### Option 1: Local development with sample data

```bash
# Clone the repo
git clone https://github.com/your-org/bedrock-model-profiler.git
cd bedrock-model-profiler

# Install frontend dependencies
cd frontend
npm install

# Copy sample data for local development
cp ../data/bedrock_models.json public/latest/bedrock_models.json
cp ../data/bedrock_pricing.json public/latest/bedrock_pricing.json

# Start dev server
npm run dev
```

Open `http://localhost:5173` and start exploring.

### Option 2: Local data collection with your AWS account

Use the included CLI tool to collect fresh data from your AWS account:

```bash
# Install Python dependencies
pip install -r local/requirements.txt

# Collect data (requires AWS credentials)
python -m local collect

# Or specify a profile and output directory
python -m local collect --profile my-aws-profile --output ./data
```

### Option 3: Full AWS deployment

Deploy the complete serverless pipeline with automated daily data collection:

```bash
# Deploy backend infrastructure
cd infra
sam build -t backend-template.yaml
sam deploy --stack-name bedrock-profiler --capabilities CAPABILITY_NAMED_IAM --resolve-s3

# Deploy frontend
cd ../frontend
npm run build
./scripts/deploy.sh
```

### Optional: Authentication

The app supports optional AWS Cognito authentication. To enable it:

```bash
cp frontend/template.env frontend/.env
```

Edit `.env` with your Cognito configuration. If not set, the app runs without authentication.

## How It Works

The tool runs a fully automated serverless pipeline that collects data from multiple AWS APIs:

```
Daily (Step Functions):

  Phase 1 — Collect raw data from AWS APIs
    Pricing API (3 service codes) + Bedrock API (27+ regions) + Service Quotas

  Phase 2 — Enrich and cross-reference
    Pricing matching + Regional availability + Inference profiles
    + Token specs + Mantle data + Lifecycle status

  Phase 3 — Aggregate and validate
    Final merge → Gap detection → Self-healing → Publish to CloudFront
```

17 Lambda functions process the data, with inter-Lambda S3 caching reducing API calls from ~480 to ~29 per execution. A self-healing system powered by Claude detects data gaps and automatically applies safe configuration fixes.

## Project Structure

```
bedrock-model-profiler/
├── frontend/                  # React + Vite web application
│   ├── src/components/        # UI components (models, comparison, layout)
│   ├── src/hooks/             # Data fetching (useModels.js)
│   ├── src/stores/            # State management (comparison, favorites)
│   └── src/config/            # Constants and data source config
├── backend/
│   ├── lambdas/               # 17 Python Lambda functions
│   ├── layers/common/         # Shared utilities (model matcher, caching, config)
│   ├── statemachine/          # Step Functions workflow definition
│   ├── config/                # Externalized pipeline configuration
│   └── tests/                 # ~150 tests
├── local/                     # CLI tool for local data collection
├── data/                      # Sample data files
├── infra/                     # SAM templates (CloudFormation)
├── docs/                      # Architecture and design documentation
└── setup-infrastructure.sh    # Full deployment script
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, data flow, Lambda functions |
| [docs/DATA-SOURCES.md](docs/DATA-SOURCES.md) | Data sources, reliability notes, fallback mechanisms |
| [docs/DATA-SCHEMA.md](docs/DATA-SCHEMA.md) | Model JSON structure, field explanations |
| [docs/PRICING-SCHEMA.md](docs/PRICING-SCHEMA.md) | Pricing JSON structure, groups, dimensions |
| [backend/lambdas/README.md](backend/lambdas/README.md) | Lambda function contracts |

## Requirements

**For Local Development:**
- Node.js 18+
- Web browser

**For Data Collection:**
- Python 3.11+
- AWS credentials with Bedrock and Pricing API access

**For Full Deployment:**
- AWS CLI configured
- AWS SAM CLI
- Python 3.11

## Troubleshooting

**No models showing?**
Make sure sample data files exist in `frontend/public/latest/`, or run the backend pipeline.

**Build fails?**
Make sure you have Node.js 18+ and run `npm install`.

---

## Important Notes

**This is a profiling tool for analysis and reference purposes.** It provides comprehensive model information to support decision-making, but should be verified against official sources for production decisions.

- Always verify pricing and availability in the AWS Console and official AWS documentation
- Review and adapt the code according to your specific security, compliance, and requirements
- Test thoroughly with your specific use cases and data

## License

MIT License — see [LICENSE](LICENSE) for details.
