# Security

## Reporting Security Issues

If you discover a potential security issue in this project, we ask that you notify AWS Security via our
[vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/). Please do **not**
create a public GitHub issue.

## Security Considerations

This repository contains sample code for Amazon Bedrock migration and modernization tools. The code is
provided as-is for educational and evaluation purposes. Before deploying to production:

1. **IAM Permissions**: Review and scope down all IAM policies to follow least-privilege principles.
   Replace any `Resource: "*"` with specific resource ARNs.
2. **Encryption**: Enable encryption at rest (SSE-KMS) for all S3 buckets and DynamoDB tables.
   Enable encryption in transit by enforcing TLS 1.2+ on all endpoints.
3. **Input Validation**: Validate and sanitize all user inputs before passing to AWS APIs or LLM prompts.
4. **Secrets Management**: Use AWS Secrets Manager or SSM Parameter Store for API keys and credentials.
   Never commit secrets to source control.
5. **Network Security**: Deploy Lambda functions in VPCs with appropriate security groups when accessing
   sensitive resources.
6. **Logging and Monitoring**: Enable CloudTrail, CloudWatch Logs, and S3 access logging for audit trails.
7. **AI/ML Safety**: Implement Amazon Bedrock Guardrails for content filtering. Add human review for
   high-risk use cases (healthcare, financial).

## Shared Responsibility Model

This sample code operates under the [AWS Shared Responsibility Model](https://aws.amazon.com/compliance/shared-responsibility-model/):

- **AWS is responsible for**: Security of the cloud infrastructure, including Amazon Bedrock service
  availability, model hosting, and underlying compute/storage/networking.
- **You are responsible for**: Security in the cloud, including IAM policies, data encryption,
  network configuration, input validation, output filtering, and compliance with applicable regulations.

## Dependencies

This project uses third-party dependencies. Review the `requirements.txt` and `package.json` files
for each component. Keep dependencies updated and monitor for known vulnerabilities.
