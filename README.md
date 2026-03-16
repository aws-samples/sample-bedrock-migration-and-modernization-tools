# Amazon Bedrock Model Profiler

A comprehensive tool for exploring, analyzing, and comparing Amazon Bedrock foundation models. Whether you're building new AI applications, optimizing existing workloads, or migrating from other AI services, this profiler provides deep insights into model capabilities, pricing, and regional availability across 100+ models from 17+ providers.

![Model Explorer](images/model_explorer.png)

## Use Cases

- **Model Selection** — Compare capabilities, context windows, and specifications to find the right model for your use case
- **Migration Planning** — Analyze Bedrock models when migrating workloads from other AI providers
- **Cost Optimization** — Compare pricing across models, regions, and consumption options (on-demand, batch, provisioned, CRIS, Mantle)
- **Regional Planning** — Identify model availability across 27+ AWS regions for multi-region deployments
- **Capability Matching** — Find models with specific features: vision, code generation, embeddings, function calling
- **Quota Analysis** — Review service quotas and throughput limits for capacity planning

## Key Features

### Model Explorer
Browse 100+ foundation models with powerful filtering by provider, capabilities, modalities, regions, context windows, and more. Adjustable grid density, sorting, search, and light/dark themes.

![Model Explorer with Filters](images/model_explorer_filtering.png)

### Model Comparison
Compare up to 5 models side-by-side across four specialized views:
- **Overview** — Capabilities, specifications, and metadata at a glance
- **Pricing** — Per-1M-token costs with region-level granularity (token, image, video, search unit pricing)
- **Availability** — Interactive region map showing on-demand, CRIS, Mantle, batch, and provisioned availability
- **Tech Specs** — Context windows, token limits, streaming support, and customization options

![Model Comparison](images/model_comparison.png)

### Regional Availability Matrix
Comprehensive model-by-region availability analysis:
- 27+ AWS regions with geographic grouping (NAMER, EMEA, APAC, LATAM)
- Five consumption types: on-demand, CRIS, Mantle, batch, provisioned throughput
- Filter by availability type, provider, or model status
- Expandable detail sections showing inference profiles, source regions, and geographic scopes

### Detailed Model Cards
Access comprehensive information for each model including:
- Input/output modalities and supported features
- Regional availability with cross-region inference support
- Pricing breakdown by region and consumption type
- Service quotas and throughput limits
- Documentation links and provider information

![Model Card - Pricing](images/model_card_pricing.png)

### Additional Features
- **Favorites** — Save a personal shortlist of models for quick access (persisted in browser across sessions)
- **13 Filter Types** — Provider, region, status, CRIS scope, streaming, context window, modality, capabilities, use cases, customization, languages, consumption, and geographic region

## Getting Started

### Option 1: Local Development (No AWS Infrastructure)

Run the profiler locally using your AWS credentials. No cloud deployment required.

**Prerequisites:**
- Python 3.11+
- Node.js 18+
- AWS credentials with read access to Bedrock and Pricing APIs

```bash
# Clone the repository
git clone https://github.com/aws-samples/bedrock-model-profiler.git
cd bedrock-model-profiler

# Set up Python environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r local/requirements.txt

# Collect model data (takes ~90 seconds)
python -m local collect --profile your-aws-profile

# Start the frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

> **Tip:** Sample data is included in the `data/` directory. To use it without collecting fresh data:
> ```bash
> mkdir -p frontend/public/latest
> cp data/bedrock_models.json frontend/public/latest/
> cp data/bedrock_pricing.json frontend/public/latest/
> ```

### Option 2: Full AWS Deployment

Deploy the complete solution with automated daily data refresh.

**Prerequisites:**
- All local prerequisites, plus:
- AWS SAM CLI
- AWS credentials with deployment permissions (CloudFormation, S3, Lambda, Step Functions)

```bash
# 1. Deploy backend (data pipeline)
cd infra
sam build -t backend-template.yaml
sam deploy --guided

# 2. Deploy everything (frontend + backend linking)
cd ..
./setup-infrastructure.sh
```

This deploys:
- **S3 buckets** for frontend hosting and data storage
- **CloudFront distribution** for global access
- **Step Functions workflow** for automated data collection (daily at 6 AM UTC)
- **16 Lambda functions** for data processing and enrichment
- **EventBridge rule** for scheduling

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CloudFront                              │
│                    (Frontend + Data CDN)                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────┐           ┌─────────────────────┐
│  Frontend S3    │           │    Data S3           │
│  (React App)    │           │ (models + pricing)   │
└─────────────────┘           └──────────▲───────────┘
                                         │
┌────────────────────────────────────────┴──────────────────────────┐
│                     Step Functions Workflow                       │
│                     (Daily at 6 AM UTC)                           │
├─────────────┬─────────────────┬──────────────┬───────────────────┤
│  Phase 1    │  Phase 1        │  Phase 1     │  Phase 2+3        │
│  Pricing    │  Model          │  Quota       │  Enrichment,      │
│  Collector  │  Extractor      │  Collector   │  Aggregation,     │
│  (3 svc)    │  (27+ regions)  │  (27+ reg)   │  Gap Detection    │
└─────────────┴─────────────────┴──────────────┴───────────────────┘
```

The pipeline collects data from multiple AWS APIs, enriches it with cross-references (pricing matching, inference profiles, token specs, lifecycle status), and publishes the final dataset to S3. Inter-Lambda S3 caching reduces API calls from ~480 to ~29 per execution. A self-healing system detects data gaps and automatically applies safe configuration fixes.

## Project Structure

```
bedrock-model-profiler/
├── frontend/                 # React + Vite + Tailwind CSS application
│   ├── src/
│   │   ├── components/       # UI components (models/, comparison/, layout/, ui/)
│   │   ├── hooks/            # Data fetching hooks
│   │   ├── stores/           # Zustand state management (comparison, favorites)
│   │   ├── utils/            # Filters, region utilities
│   │   └── config/           # Constants and data source config
│   └── scripts/              # Deployment scripts
├── backend/
│   ├── lambdas/              # 16 Python Lambda functions
│   ├── layers/common/        # Shared Lambda layer (model matching, caching, config)
│   ├── statemachine/         # Step Functions ASL workflow definition
│   ├── config/               # Externalized pipeline configuration
│   └── tests/                # ~150 tests
├── local/                    # CLI tool for local data collection
├── data/                     # Sample data files
├── infra/                    # SAM/CloudFormation templates
├── docs/                     # Architecture and schema documentation
└── setup-infrastructure.sh   # One-command deployment
```

## AWS Permissions Required

**For local data collection (Option 1):**
```
bedrock:ListFoundationModels
bedrock:ListInferenceProfiles
pricing:GetProducts
pricing:DescribeServices
servicequotas:ListServiceQuotas
```

**For full AWS deployment (Option 2) — additional:**
```
cloudformation:*
s3:*
lambda:*
states:*
iam:CreateRole, iam:AttachRolePolicy, iam:PassRole
cloudfront:*
events:*
logs:*
```

## Cost Estimate

The solution is designed to minimize costs using serverless, pay-per-use services.

### Running Costs (Monthly)

| Service | Usage | Estimated Cost |
|---------|-------|---------------|
| **Lambda** | 16 functions × 1 execution/day × ~30s avg | ~$0.50/mo |
| **Step Functions** | 1 workflow/day × ~50 state transitions | ~$0.01/mo |
| **S3** | ~50 MB data + frontend assets, daily writes | ~$0.10/mo |
| **CloudFront** | Depends on traffic; 10K requests/mo | ~$0.10/mo |
| **EventBridge** | 1 scheduled rule | Free tier |
| **CloudWatch Logs** | ~100 MB/mo log ingestion | ~$0.05/mo |
| **Bedrock API calls** | ListFoundationModels, ListInferenceProfiles (~30 calls/day) | Free (list operations) |
| **Pricing API** | GetProducts (3 calls/day) | Free |
| **Service Quotas API** | ListServiceQuotas (~27 calls/day) | Free |

**Estimated total: ~$1–2/month** for typical usage with low traffic.

> **Note:** The self-healing agent (optional) uses Bedrock InvokeModel with Claude, which incurs token-based charges. This only runs when data gaps are detected (~1-2 times/month) and costs approximately $0.01–0.05 per invocation.

### One-Time Deployment

CloudFormation stack creation and SAM deployment are free. You only pay for the resources created.

## Cleanup

To remove all deployed resources and stop incurring charges:

```bash
# Set your environment (default: dev)
ENVIRONMENT="${ENVIRONMENT:-dev}"
REGION="${AWS_REGION:-us-east-1}"

# 1. Empty the S3 buckets (required before stack deletion)
DATA_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name "bedrock-profiler-${ENVIRONMENT}" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" \
    --output text)

FRONTEND_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name "bedrock-profiler-frontend-${ENVIRONMENT}" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='FrontendBucketName'].OutputValue" \
    --output text)

aws s3 rm "s3://${DATA_BUCKET}" --recursive
aws s3 rm "s3://${FRONTEND_BUCKET}" --recursive

# 2. Delete the stacks (frontend first, then backend)
aws cloudformation delete-stack \
    --stack-name "bedrock-profiler-frontend-${ENVIRONMENT}" \
    --region "$REGION"

aws cloudformation wait stack-delete-complete \
    --stack-name "bedrock-profiler-frontend-${ENVIRONMENT}" \
    --region "$REGION"

aws cloudformation delete-stack \
    --stack-name "bedrock-profiler-${ENVIRONMENT}" \
    --region "$REGION"

echo "All resources deleted."
```

## Documentation

| Document | Description |
|---------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, data flow, Lambda functions |
| [docs/DATA-SOURCES.md](docs/DATA-SOURCES.md) | Data sources, reliability notes, fallback mechanisms |
| [docs/DATA-SCHEMA.md](docs/DATA-SCHEMA.md) | Model JSON structure, field explanations |
| [docs/PRICING-SCHEMA.md](docs/PRICING-SCHEMA.md) | Pricing JSON structure, groups, dimensions |
| [backend/lambdas/README.md](backend/lambdas/README.md) | Lambda function contracts and I/O specifications |

## Troubleshooting

### Local Development

**"File not found" error in browser**
- Run `python -m local collect` to generate data files, or copy sample data from `data/`
- Verify `frontend/public/latest/bedrock_models.json` exists

**"Access Denied" during data collection**
- Verify AWS credentials: `aws sts get-caller-identity`
- Check your profile has Bedrock and Pricing API access
- Some regions may fail due to Bedrock not being available — this is normal

**Frontend won't start**
- Delete `node_modules` and `package-lock.json`, then run `npm install`
- Ensure Node.js 18+ is installed: `node --version`

### AWS Deployment

**No models showing after deployment**
- The Step Functions workflow runs daily at 6 AM UTC
- Manually trigger: AWS Console → Step Functions → Start execution
- Check CloudWatch Logs for Lambda errors

**CloudFormation deployment fails**
- Verify SAM CLI is installed: `sam --version`
- Check you have sufficient IAM permissions
- Review the error in CloudFormation console for specific resource failures

**Stale data**
- Data refreshes daily at 6 AM UTC
- Manually trigger the Step Functions workflow for immediate refresh

## Additional Screenshots

<details>
<summary>Click to expand</summary>

### Model Card - Specifications
![Model Card Specs](images/model_card_specs.png)

### Model Card - Quotas
![Model Card Quotas](images/model_card_quotas.png)

### Comparison - Regional Availability
![Comparison Availability](images/model_comparison_availability.png)

</details>

## Important Notes

This tool is designed for exploration, analysis, and planning. While it uses official AWS APIs, please note:

- **Verify before production decisions** — Always confirm pricing and availability in the [AWS Console](https://console.aws.amazon.com/bedrock/)
- **Data freshness** — Model availability and pricing can change; data refreshes daily
- **Regional variations** — Some features may not be available in all regions
- **Consult AWS** — Contact your AWS account team for production workload guidance

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
