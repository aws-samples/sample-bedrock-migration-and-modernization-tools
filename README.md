# Amazon Bedrock Model Profiler 🤖

A comprehensive profiling tool for exploring, analyzing, and comparing Amazon Bedrock foundation models to make informed model selection decisions. Whether you're building new applications, optimizing existing workloads, or migrating from other AI services or internal models, this tool provides deep insights into model capabilities, pricing, and regional availability to guide your choice of Amazon Bedrock foundation models.

**🌐 Live Demo:** https://d13th0vs8a20t3.cloudfront.net

## Use Cases This Tool Supports

* 🎯 **Model Selection Decisions** - Compare capabilities, specifications, and performance characteristics across Amazon Bedrock foundation models
* 🔄 **Migration Planning** - Evaluate and select the right Amazon Bedrock models when migrating workloads or modernizing AI applications
* 🌍 **Regional Availability Analysis** - Review model availability across AWS regions to plan deployments and ensure service coverage
* 💰 **Cost Optimization** - Compare pricing across models and consumption options (on-demand, provisioned throughput, batch inference) for budget planning
* ⚖️ **Capability Matching** - Find models that match specific requirements for context windows, multimodal capabilities, or specialized tasks
* 📊 **Performance Planning** - Analyze throughput options, latency characteristics, and scaling capabilities for workload requirements
* 🔍 **Comprehensive Model Discovery** - Explore the complete Amazon Bedrock model catalog with detailed specifications and real-time availability

## Getting Started with this Tool

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
2. Browse models using filters (provider, capabilities, regions)
3. Click models for detailed specs and pricing
4. Select models to compare them side-by-side

## Key Tool Capabilities

### 🔍 Model Discovery and Exploration
This tool provides comprehensive model browsing capabilities:
- Dynamic filtering by provider, capabilities, and regions
- Search functionality across model attributes
- Flexible grid layouts with adjustable density
- Drill-down interfaces for detailed specifications

### ⚖️ Model Comparison Framework
The tool implements advanced model comparison features:
- Compare up to 5 models side-by-side
- Regional availability visualizations
- Pricing analysis dashboards
- Technical specification comparisons

### 💰 Pricing Integration Features
The tool provides comprehensive pricing analysis capabilities:
- Real-time data integration from AWS Pricing APIs
- Regional price variation analysis and visualization
- Input/output token pricing breakdown
- On-demand and provisioned throughput options

### 🌍 Regional Analysis Features
The tool offers multi-region model analysis capabilities:
- Model availability mapping across 20+ AWS regions
- Regional availability comparisons
- Cross-region inference capability integration
- Service quota analysis and display by region

## What You Need

**For Local Development:**
- Node.js 18+
- Web browser
- AWS credentials configured (for S3 data proxy)

**For Deployment:**
- AWS CLI configured
- AWS SAM CLI
- Python 3.11

**AWS Permissions (for data collection backend):**
- **Bedrock:** `bedrock:ListFoundationModels`, `bedrock:ListInferenceProfiles`
- **Pricing API:** `pricing:GetProducts`, `pricing:DescribeServices`
- **Service Quotas:** `servicequotas:ListServiceQuotas`, `servicequotas:GetServiceQuota`

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

## Project Structure

```
bedrock-model-profiler/
├── frontend/                  # React + Vite web application
├── backend/                   # AWS Lambda functions + Step Functions
├── infra/                     # SAM templates (CloudFormation)
└── setup-infrastructure.sh    # Full deployment script
```

## Troubleshooting

**No models showing?**
→ Check that the backend Step Functions workflow has run at least once

**Access denied errors in dev?**
→ Check your AWS credentials are configured for S3 access

**Build fails?**
→ Make sure you have Node.js 18+ and run `npm install`

---

## Important Notes

**This is a profiling tool for learning and reference purposes.** It provides comprehensive model profiling and selection capabilities to help you understand implementation approaches, but should be reviewed and adapted for your specific production requirements.

**For production decisions:**
- Always verify pricing and availability in the AWS console and official AWS documentation
- Contact your AWS account team for guidance on model selection for production workloads
- Review and adapt the code according to your specific security, compliance, and requirements
- Test thoroughly with your specific use cases and data

**Data accuracy:** While this tool uses official AWS APIs, model availability and pricing can change. Always verify current information through official AWS channels before making final decisions.

## License

This project is for internal use.
