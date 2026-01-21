#!/bin/bash
#
# Deploy and Analyze Script
#
# Deploys the backend and optionally runs the Step Functions workflow,
# then analyzes pricing coverage.
#
# Usage:
#   ./deploy_and_analyze.sh [--run-workflow] [--skip-deploy]
#
# Options:
#   --run-workflow    Run the Step Functions workflow after deployment
#   --skip-deploy     Skip deployment, only run workflow and analysis
#   --help            Show this help message
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INFRA_DIR="$PROJECT_ROOT/infra"
STACK_NAME="bedrock-profiler-dev"
REGION="us-west-2"
STATE_MACHINE_ARN="arn:aws:states:${REGION}:169497827606:stateMachine:bedrock-profiler-workflow-dev"

# Parse arguments
RUN_WORKFLOW=false
SKIP_DEPLOY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --run-workflow)
            RUN_WORKFLOW=true
            shift
            ;;
        --skip-deploy)
            SKIP_DEPLOY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--run-workflow] [--skip-deploy]"
            echo ""
            echo "Options:"
            echo "  --run-workflow    Run the Step Functions workflow after deployment"
            echo "  --skip-deploy     Skip deployment, only run workflow and analysis"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

print_header() {
    echo ""
    echo -e "${BOLD}${CYAN}============================================================${NC}"
    echo -e "${BOLD}${CYAN}$1${NC}"
    echo -e "${BOLD}${CYAN}============================================================${NC}"
}

print_step() {
    echo ""
    echo -e "${BOLD}${BLUE}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Step 1: Build and Deploy
if [ "$SKIP_DEPLOY" = false ]; then
    print_header "STEP 1: BUILD AND DEPLOY"

    cd "$INFRA_DIR"

    print_step "Building SAM application..."
    sam build -t backend-template.yaml

    print_step "Deploying to AWS..."
    sam deploy \
        --stack-name "$STACK_NAME" \
        --capabilities CAPABILITY_NAMED_IAM \
        --no-fail-on-empty-changeset \
        --no-confirm-changeset \
        --resolve-s3 \
        --region "$REGION"

    print_success "Deployment completed"
else
    print_warning "Skipping deployment (--skip-deploy)"
fi

# Step 2: Run Workflow (optional)
if [ "$RUN_WORKFLOW" = true ]; then
    print_header "STEP 2: RUN STEP FUNCTIONS WORKFLOW"

    print_step "Starting workflow execution..."
    EXECUTION_ARN=$(aws stepfunctions start-execution \
        --state-machine-arn "$STATE_MACHINE_ARN" \
        --region "$REGION" \
        --query 'executionArn' \
        --output text)

    echo "Execution ARN: $EXECUTION_ARN"

    print_step "Waiting for workflow to complete..."
    while true; do
        STATUS=$(aws stepfunctions describe-execution \
            --execution-arn "$EXECUTION_ARN" \
            --region "$REGION" \
            --query 'status' \
            --output text)

        echo -ne "\rStatus: $STATUS"

        if [ "$STATUS" = "SUCCEEDED" ]; then
            echo ""
            print_success "Workflow completed successfully"
            break
        elif [ "$STATUS" = "FAILED" ] || [ "$STATUS" = "TIMED_OUT" ] || [ "$STATUS" = "ABORTED" ]; then
            echo ""
            print_error "Workflow failed with status: $STATUS"
            exit 1
        fi

        sleep 10
    done
fi

# Step 3: Analyze Pricing Coverage
print_header "STEP 3: PRICING COVERAGE ANALYSIS"

cd "$PROJECT_ROOT"
python3 "$SCRIPT_DIR/pricing_coverage_analyzer.py" --download --region "$REGION"

ANALYSIS_EXIT_CODE=$?

if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    print_success "Analysis completed - coverage is stable or improved"
else
    print_warning "Analysis completed - coverage may have regressed"
fi

print_header "DEPLOYMENT AND ANALYSIS COMPLETE"
echo ""
echo -e "To run workflow and analyze again:"
echo -e "  ${CYAN}$0 --skip-deploy --run-workflow${NC}"
echo ""
echo -e "To just analyze current data:"
echo -e "  ${CYAN}python3 $SCRIPT_DIR/pricing_coverage_analyzer.py --download${NC}"
echo ""

exit $ANALYSIS_EXIT_CODE
