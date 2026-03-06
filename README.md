# Amazon Bedrock Model Profiler

A comprehensive profiling tool for exploring, analyzing, and comparing Amazon Bedrock foundation models to make informed model selection decisions. Whether you're building new applications, optimizing existing workloads, or migrating from other AI services, this tool provides deep insights into model capabilities, pricing, and regional availability across 100+ models from 17+ providers.

**Live:** https://d3oem6l61p8j11.cloudfront.net

## Use Cases This Tool Supports

* **Model Selection Decisions** - Compare capabilities, specifications, and pricing across Amazon Bedrock foundation models to find the best fit for your workload
* **Migration Planning** - Evaluate Bedrock models side-by-side when migrating from OpenAI, Anthropic direct, or internal models
* **Regional Availability Analysis** - Review model availability across 27+ AWS regions with detailed consumption type breakdowns (on-demand, CRIS, Mantle, batch, provisioned)
* **Cost Optimization** - Compare per-1M-token pricing across models, regions, and consumption options for accurate budget planning
* **Capability Matching** - Find models that match specific requirements for context windows, multimodal capabilities, streaming, or specialized tasks
* **Deployment Planning** - Analyze cross-region inference options, geographic scopes, and service quotas to plan multi-region architectures

## Getting Started

### 1. Clone the repository
```bash
git clone git@ssh.gitlab.aws.dev:molivac/bedrock-model-profiler.git
cd bedrock-model-profiler
```

### 2. Install dependencies
```bash
cd frontend
npm install
```

### 3. Configure authentication (required for internal use)
```bash
cp template.env .env
```

Edit `.env` and set your Cognito credentials:
```
VITE_COGNITO_AUTHORITY_URL=https://cognito-idp.us-east-1.amazonaws.com/us-east-1_xxxxx
VITE_COGNITO_CLIENT_ID=your-client-id
```

### 4. Launch it
```bash
npm run dev
```

### 5. Start exploring!
1. Open `http://localhost:5173` in your browser
2. Browse models using filters (provider, region, capabilities, modalities, and more)
3. Click any model card for detailed specs, pricing, and availability
4. Select models to compare them side-by-side across 4 dimensions

## Key Tool Capabilities

### Model Discovery and Exploration
Browse the complete Bedrock model catalog with powerful filtering:
- 13 filter types: provider, region, status, CRIS scope, streaming, context window, modality, capabilities, use cases, customization, languages, consumption options, and geographic region
- Sorting by name, provider, context window size, or pricing
- Adjustable grid density and pagination
- Light and dark theme support

### Model Comparison Framework
Compare up to 5 models side-by-side with four specialized views:
- **Overview** - Capabilities, specifications, and metadata at a glance
- **Pricing** - Per-1M-token costs with region-level granularity (token, image, video, and search unit pricing)
- **Availability** - Interactive region map showing on-demand, CRIS, Mantle, batch, and provisioned availability
- **Tech Specs** - Context windows, token limits, streaming support, and customization options

### Regional Availability Matrix
Comprehensive model-by-region availability analysis:
- All 27+ AWS regions with geographic grouping (NAMER, EMEA, APAC, LATAM)
- Five consumption types tracked: on-demand, CRIS, Mantle, batch, provisioned throughput
- Filter by availability type, provider, or model status
- Expandable detail sections showing inference profiles, source regions, and geographic scopes

### Pricing Analysis
Detailed pricing information updated daily from AWS Pricing APIs:
- Input/output token pricing per 1M tokens
- Region-by-region price comparison
- Multiple pricing models: token, image generation, video generation, video per-second, search units
- On-demand, batch, and provisioned throughput pricing groups

### Additional Features
- **Favorites** - Save a personal shortlist of models for quick access (persisted across sessions)
- **Service Quotas** - View internal service quotas by region and category (requires authorization)
- **Region Roadmap** - Internal planning tool for tracking model launch schedules (operators)
- **Usage Analytics** - Dashboards for tracking tool adoption and usage patterns (admins)

## How It Works

The tool runs a fully automated serverless pipeline that collects data from 7 AWS and external sources daily:

```
Daily at 6 AM UTC (Step Functions):

  Phase 1 - Collect raw data from AWS APIs
    Pricing API (3 service codes) + Bedrock API (27 regions) + Service Quotas

  Phase 2 - Enrich and cross-reference
    Pricing matching + Regional availability + Inference profiles
    + Token specs (LiteLLM) + Mantle data + Lifecycle status

  Phase 3 - Aggregate and validate
    Final merge → Gap detection → Self-healing → Publish to CloudFront
```

17 Lambda functions process the data, with inter-Lambda S3 caching reducing API calls from ~480 to ~29 per execution. A self-healing system powered by Claude detects data gaps and automatically applies safe configuration fixes.

## Project Structure

```
bedrock-model-profiler/
├── frontend/                  # React + Vite web application
│   ├── src/components/        # UI components (models, comparison, layout)
│   ├── src/hooks/             # Data fetching (useModels.js)
│   ├── src/stores/            # State management (comparison, favorites, auth)
│   └── src/config/            # Constants, permissions, data source config
├── backend/
│   ├── lambdas/               # 17 Python Lambda functions
│   ├── layers/common/         # Shared utilities (model matcher, caching, config)
│   ├── statemachine/          # Step Functions workflow definition
│   ├── config/                # Externalized pipeline configuration
│   └── tests/                 # ~150 tests
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
| [docs/model-matching-issues.md](docs/model-matching-issues.md) | Known model matching issues |

## What You Need

**For Local Development:**
- Node.js 18+
- Web browser
- AWS credentials configured (for S3 data proxy in development)

**For Deployment:**
- AWS CLI configured
- AWS SAM CLI
- Python 3.11

## Deployment

**Full deployment** (infrastructure + frontend):
```bash
./setup-infrastructure.sh
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
sam deploy --stack-name bedrock-profiler-dev --capabilities CAPABILITY_NAMED_IAM --resolve-s3
```

## Troubleshooting

**No models showing?**
Check that the backend Step Functions workflow has run at least once.

**Access denied errors in dev?**
Check your AWS credentials are configured for S3 access.

**Build fails?**
Make sure you have Node.js 18+ and run `npm install`.

**Missing features (Regional Availability, Quotas)?**
These require authentication. Check your Cognito user group membership (beta-access-users, admins).

---

## Important Notes

**This is a profiling tool for analysis and reference purposes.** It provides comprehensive model information to support decision-making, but should be verified against official sources for production decisions.

**For production decisions:**
- Always verify pricing and availability in the AWS Console and official AWS documentation
- Contact your AWS account team for guidance on model selection for production workloads
- Review and adapt the code according to your specific security, compliance, and requirements
- Test thoroughly with your specific use cases and data

**Data freshness:** The tool refreshes data daily at 6 AM UTC. For the most current information on new model launches or pricing changes, consult official AWS channels.

## License

This project is for internal use.
