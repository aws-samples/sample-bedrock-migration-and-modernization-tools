# Midway Authentication Integration Plan

This document outlines the plan for integrating AWS Cognito authentication (Midway) into the Bedrock Model Profiler.

## Current Status

**Completed:**
- Frontend authentication code integrated (`react-oidc-context`)
- Auth components: `AuthProvider`, `AuthGate`, `UserProfile`
- Zustand auth store for user state
- Environment variable configuration (`template.env`)
- Graceful fallback when auth not configured

**Pending:**
- Cognito infrastructure deployment (CDK → SAM conversion)
- Secrets management setup
- Build/deploy script integration

## Source Projects

- **Frontend auth code**: Already integrated from `/Users/molivac/aws/projects/react-midway`
- **Cognito infrastructure**: `/Users/molivac/aws/projects/cognito-midway-authentication` (CDK TypeScript)

## Architecture Decision

**Separate Auth Stack** - Keep Cognito resources in a dedicated SAM template:

```
infra/
├── auth-template.yaml      # NEW - Cognito User Pool, Identity Provider, Client
├── auth-samconfig.toml     # NEW - SAM deployment configuration
├── auth-parameters-template.json  # NEW - Parameter template (gitignored)
├── backend-template.yaml   # Existing - Step Functions + Lambdas
└── frontend-template.yaml  # Existing - CloudFront + S3
```

**Rationale:**
- Isolation of concerns
- Independent lifecycle management
- Easier rollback
- Can be shared across multiple applications

## Cognito Resources to Create

From the CDK stack, convert to CloudFormation/SAM:

1. **AWS::Cognito::UserPool** - Email-based sign-in
2. **AWS::Cognito::UserPoolDomain** - Hosted UI for OAuth
3. **AWS::Cognito::UserPoolIdentityProvider** - OIDC connection to Midway/Federate
4. **AWS::Cognito::UserPoolClient** - OAuth 2.0 Authorization Code Grant
5. **AWS::Cognito::IdentityPool** - For AWS credential vending (optional)
6. **AWS::IAM::Role** (x2) - Authenticated and unauthenticated roles

## Stack Outputs Required

| Output | Used For |
|--------|----------|
| `UserPoolId` | Part of authority URL |
| `UserPoolClientId` | `VITE_COGNITO_CLIENT_ID` |
| `CognitoAuthorityUrl` | `VITE_COGNITO_AUTHORITY_URL` |
| `IdentityPoolId` | AWS credential vending (if needed) |

## Implementation Phases

### Phase 1: CDK to SAM Translation
- Create `infra/auth-template.yaml`
- Translate CDK L2 constructs to CloudFormation resources
- Add parameters for environment-specific configuration
- Use Secrets Manager dynamic reference for OIDC client secret

### Phase 2: Secrets Management Setup
- Create Secrets Manager secret for OIDC client secret
- Create `auth-parameters-template.json` for configuration
- Update `.gitignore` to exclude `auth-parameters.json`

### Phase 3: SAM Configuration
- Create `infra/auth-samconfig.toml`
- Configure dev/staging/prod environments

### Phase 4: Infrastructure Script Integration
- Update `setup-infrastructure.sh` to deploy auth stack
- Extract stack outputs and export as environment variables
- Pass outputs to frontend build process

### Phase 5: Frontend Build Integration
- Inject Cognito outputs into frontend `.env` during build
- Verify built assets contain correct configuration

### Phase 6: Documentation and Testing
- Update README with auth setup instructions
- Create testing checklist
- Document troubleshooting steps

## Open Questions

### Prerequisites
1. **Do you have an existing Amazon Federate Service Profile configured?**
   - Required for OIDC integration
   - Provides `oidc_client_id` and `oidc_client_secret`
   - Guide: https://quip-amazon.com/tPJFAkXmRLu4/Midway-Authentication-Setup

### Configuration
2. **Which AWS account/region should host Cognito?**
   - Same as backend? Different account for security?

3. **What domain prefix to use for Cognito?**
   - Must be globally unique across AWS
   - Suggestion: `bedrock-profiler-{account-id}-{env}`

4. **Callback URLs needed?**
   - `http://localhost:5173` (local dev)
   - `https://d13th0vs8a20t3.cloudfront.net` (current CloudFront)
   - Any additional domains?

### Architecture
5. **Should Identity Pool be included?**
   - Only needed if frontend makes direct AWS API calls
   - Current frontend only uses S3 via CloudFront (no direct AWS calls)
   - Recommendation: Include initially, can remove if unused

6. **Removal policy for production?**
   - Dev/Staging: `Delete` (clean up on stack deletion)
   - Prod: `Retain` (prevent accidental data loss)

### Security
7. **Backend Lambda authorization needed?**
   - Current: Lambdas invoked by Step Functions (internal)
   - No API Gateway endpoints exposed
   - Recommendation: Frontend-only auth initially

## Environment Variables

Frontend requires these in `.env`:

```bash
# From Cognito stack outputs
VITE_COGNITO_AUTHORITY_URL=https://cognito-idp.{region}.amazonaws.com/{userPoolId}
VITE_COGNITO_CLIENT_ID={userPoolClientId}
```

## Parameter Flow

```
Secrets Manager (oidc_client_secret)
       ↓
CloudFormation Parameters (auth-parameters.json)
       ↓
SAM Deploy (auth-template.yaml)
       ↓
Stack Outputs (UserPoolId, ClientId)
       ↓
Shell Script (setup-infrastructure.sh)
       ↓
Environment Variables (VITE_*)
       ↓
Frontend Build (npm run build)
       ↓
Compiled Assets (dist/)
```

## Testing Checklist

- [ ] Deploy auth stack independently
- [ ] Verify User Pool in Cognito console
- [ ] Check OIDC Identity Provider configuration
- [ ] Test OAuth flow with Midway authentication
- [ ] Verify frontend redirects to Cognito hosted UI
- [ ] Confirm successful callback with JWT tokens
- [ ] Test user profile display in sidebar
- [ ] Test logout functionality
- [ ] Verify unauthenticated fallback (remove env vars)
- [ ] Validate CloudFront + Cognito CORS behavior

## Rollback Strategy

1. Delete auth stack: `aws cloudformation delete-stack --stack-name bedrock-profiler-auth-dev`
2. Remove `.env` from frontend
3. Rebuild frontend: `npm run build`
4. Redeploy frontend: `./scripts/deploy.sh`

App will gracefully fall back to unauthenticated mode.
