# Amazon Bedrock Migration & Modernization Toolkit

One-stop shop for migrating and modernizing your LLM workloads to Amazon Bedrock.

## Your Migration Journey

### 1. Profile - Explore & Compare Models
**[bedrock-model-profiler](bedrock-model-profiler/)**

Explore, analyze, and compare 100+ Amazon Bedrock foundation models.
- Compare capabilities, pricing, and specifications side-by-side
- Regional availability across all AWS regions with CRIS, Mantle, batch, and provisioned options
- Filter by provider, modality, context window, and 13+ other dimensions

### 2. Evaluate - Compare Model Quality
**[360-eval](360-eval/)**

Comprehensive LLM evaluation framework with LLM-as-a-Jury methodology.
- Multi-model comparison (Amazon Bedrock, OpenAI, Azure, Gemini)
- Quality scoring across 6 dimensions
- Interactive HTML reports

### 3. Implement - Use Case Examples
**[usecase-examples](usecase-examples/)**

Production-ready patterns and examples.
- Common migration patterns
- Best practices
- Reference architectures

### 4. Agentic Evaluation — Evaluate agent behavior from traces (offline)
**[agent-eval](agent-eval/)**

Agentic Evaluation evaluates agent behavior and outcome quality from recorded traces — without invoking the runtime.

This module analyzes:
- Orchestrator responses
- Tool/sub-agent invocation paths
- Latency and failures
- Correctness vs golden answers (via judge model)

It is designed to be:
- Runtime-agnostic
- Cloud-agnostic
- Offline-first
- CI-friendly

## Prerequisites

- AWS Account with Amazon Bedrock access
- Python 3.10+
- AWS CLI configured

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md)

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting and [ARCHITECTURE.md](ARCHITECTURE.md) for security design documentation.

### Shared Responsibility

This sample code operates under the [AWS Shared Responsibility Model](https://aws.amazon.com/compliance/shared-responsibility-model/). AWS manages the security of Amazon Bedrock, Amazon S3, AWS Lambda, and other services. You are responsible for configuring IAM policies, encryption, network security, and input validation in your deployment.

### Before Deploying to Production

1. **Review IAM policies** — scope down permissions to least privilege. Replace `Resource: "*"` with specific ARNs.
2. **Enable encryption** — configure SSE-KMS for S3 buckets and DynamoDB tables. Enforce TLS 1.2+.
3. **Validate inputs** — sanitize all user inputs before passing to Amazon Bedrock or other AWS APIs.
4. **Manage secrets** — use AWS Secrets Manager for API keys. Do not store credentials in code or environment variables.
5. **Enable logging** — turn on CloudTrail, CloudWatch Logs, and S3 access logging.
6. **Configure AI safety** — implement Amazon Bedrock Guardrails for content filtering. Add human review for healthcare and financial use cases.

## License

MIT-0 License. See [LICENSE](LICENSE)