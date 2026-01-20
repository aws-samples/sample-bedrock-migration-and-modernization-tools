# Amazon Bedrock Model Profiler

A comprehensive web application for exploring, analyzing, and comparing Amazon Bedrock foundation models to make informed model selection decisions. Whether you're building new applications, optimizing existing workloads, or migrating from other AI services, this tool provides deep insights into model capabilities, pricing, and regional availability.

**Live Demo:** https://d13th0vs8a20t3.cloudfront.net

## Use Cases This Tool Supports

* **Model Selection Decisions** - Compare capabilities, specifications, and performance characteristics across Amazon Bedrock foundation models
* **Migration Planning** - Evaluate and select the right Amazon Bedrock models when migrating workloads or modernizing AI applications
* **Regional Availability Analysis** - Review model availability across AWS regions to plan deployments and ensure service coverage
* **Cost Optimization** - Compare pricing across models and consumption options (on-demand, provisioned throughput, batch inference) for budget planning
* **Capability Matching** - Find models that match specific requirements for context windows, multimodal capabilities, or specialized tasks
* **Comprehensive Model Discovery** - Explore the complete Amazon Bedrock model catalog with detailed specifications and real-time availability

## Key Features

### Model Discovery and Exploration
- Dynamic filtering by provider, capabilities, modalities, and regions
- Search functionality across model attributes
- Flexible grid layouts with adjustable column density
- Detailed model cards with specifications and pricing

### Model Comparison Framework
- Compare up to 5 models side-by-side
- Regional availability visualizations
- Pricing analysis across regions
- Technical specification comparisons

### Pricing Integration
- Real-time data from AWS Pricing APIs
- Regional price variation analysis
- Input/output token pricing breakdown
- On-demand and provisioned throughput options

### Regional Analysis
- Model availability mapping across 20+ AWS regions
- Cross-region inference profile support
- Service quota information by region

## Project Structure

```
bedrock-model-profiler/
├── frontend/                  # React + Vite web application
├── backend/                   # AWS Lambda functions + Step Functions
├── infra/                     # SAM templates (CloudFormation)
└── setup-infrastructure.sh    # Full deployment script
```

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.11
- AWS CLI configured
- AWS SAM CLI

### Local Development

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

### Deployment

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

## Tech Stack

| Component | Technologies |
|-----------|-------------|
| **Frontend** | React 18, Vite, Tailwind CSS v4, Radix UI, Zustand |
| **Backend** | Python 3.11, AWS Lambda, Step Functions, EventBridge |
| **Infrastructure** | AWS SAM, CloudFront, S3, IAM |

---

## Important Notes

**Data accuracy:** While this tool uses official AWS APIs, model availability and pricing can change. Always verify current information through official AWS channels before making final decisions.

**For production decisions:**
- Verify pricing and availability in the AWS console and official AWS documentation
- Contact your AWS account team for guidance on model selection for production workloads
- Test thoroughly with your specific use cases and data

## License

This project is for internal use.
