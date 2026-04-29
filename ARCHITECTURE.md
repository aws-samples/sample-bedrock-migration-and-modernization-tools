# Architecture and Security Design

## System Overview

This toolkit contains four independent components for Amazon Bedrock migration and modernization:

```
┌─────────────────────────────────────────────────────────────┐
│                    User / Developer                          │
├──────────┬──────────┬───────────────┬───────────────────────┤
│ 360-Eval │agent-eval│ bedrock-model │ multitenancy-and-     │
│Dashboard │   CLI    │  -profiler    │ observability         │
├──────────┴──────────┴───────────────┴───────────────────────┤
│              Amazon Bedrock APIs                             │
│  InvokeModel │ Converse │ CreateInferenceProfile             │
├──────────────┴──────────┴───────────────────────────────────┤
│  S3  │ DynamoDB │ CloudWatch │ Step Functions │ Lambda       │
└──────────────────────────────────────────────────────────────┘
```

## Security Design Considerations

### 1. Authentication and Authorization
- All AWS API calls use IAM roles with temporary credentials via boto3 default credential chain
- API Gateway endpoints use Cognito User Pools for authentication
- No hardcoded credentials in source code

### 2. Data Protection
- S3 buckets configured with server-side encryption (AES-256), Block Public Access, and TLS enforcement
- DynamoDB tables use AWS-managed encryption at rest
- All data in transit encrypted via HTTPS/TLS 1.2+

### 3. Multi-Tenant Isolation
- Tenant data isolated via DynamoDB partition keys and IAM resource-level policies
- Amazon Bedrock inference profiles scoped per tenant
- Cost allocation tags enable per-tenant billing visibility

### 4. Input Validation
- CLI arguments validated via argparse with type checking
- API request bodies validated against JSON schemas
- File paths validated to prevent directory traversal

### 5. Logging and Monitoring
- CloudWatch Logs for Lambda function execution
- Step Functions execution history with X-Ray tracing
- S3 access logging for audit trails
- CloudFront access logs for frontend traffic

## Threat Model

### Trust Boundaries
1. **User → Frontend**: CloudFront with HTTPS, security headers (HSTS, X-Frame-Options, CSP)
2. **Frontend → API Gateway**: Cognito JWT authentication, CORS restrictions
3. **API Gateway → Lambda**: IAM execution roles, VPC security groups (when configured)
4. **Lambda → AWS Services**: IAM policies with resource-level scoping
5. **Lambda → Amazon Bedrock**: Model access controlled via IAM, inference profiles per tenant

### Identified Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Prompt injection via user input | High | Validate inputs, use Amazon Bedrock Guardrails, implement output filtering |
| Unauthorized model access | High | IAM policies scoped to specific model ARNs, tenant-level inference profiles |
| Data exfiltration via S3 | Medium | Block Public Access, bucket policies with TLS enforcement, access logging |
| Cost abuse via excessive API calls | Medium | API Gateway throttling, Lambda concurrency limits, CloudWatch alarms |
| Cross-tenant data leakage | High | DynamoDB partition isolation, IAM condition keys, separate inference profiles |
| Credential exposure | High | No hardcoded credentials, temporary IAM credentials only, Secrets Manager for API keys |

## AWS Service Security Guidelines

### Amazon Bedrock
- Use IAM policies to restrict `bedrock:InvokeModel` to specific model ARNs
- Enable Amazon Bedrock Guardrails for content filtering
- Monitor usage with CloudWatch metrics and set billing alarms
- Use inference profiles for multi-tenant model access isolation

### Amazon S3
- Enable Block Public Access on all buckets
- Enforce TLS via bucket policy (`aws:SecureTransport` condition)
- Enable server-side encryption (SSE-S3 or SSE-KMS)
- Enable versioning and access logging
- Configure lifecycle policies for data retention

### AWS Lambda
- Follow least-privilege for execution role permissions
- Set appropriate memory and timeout limits
- Enable X-Ray tracing for debugging
- Use environment variables (not hardcoded values) for configuration

### Amazon DynamoDB
- Enable encryption at rest (AWS-managed or customer-managed KMS key)
- Use IAM condition keys for tenant isolation
- Enable point-in-time recovery for data protection
- Monitor with CloudWatch and set capacity alarms

### Amazon CloudFront
- Enforce HTTPS with TLS 1.2+ minimum
- Configure security response headers (HSTS, X-Frame-Options, X-Content-Type-Options)
- Use Origin Access Control for S3 origins
- Enable access logging

### AWS Step Functions
- Enable CloudWatch Logs for execution history
- Use IAM roles with least-privilege for state machine execution
- Enable X-Ray tracing
