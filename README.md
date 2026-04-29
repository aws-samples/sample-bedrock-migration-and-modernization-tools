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

## License

MIT-0 License. See [LICENSE](LICENSE)