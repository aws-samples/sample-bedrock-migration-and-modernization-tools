#!/bin/bash
# backend/tests/run_production_tests.sh

# Run production validation tests
# Usage: ./run_production_tests.sh [stack-name]

set -e

STACK_NAME=${1:-bedrock-profiler-prod}

echo "Running production tests for stack: ${STACK_NAME}"
echo "=============================================="

# Set environment variables
export STACK_NAME=${STACK_NAME}
export CLOUDFRONT_URL="https://d3oem6l61p8j11.cloudfront.net"

# Change to tests directory
cd "$(dirname "$0")"

# Run deployment tests
echo ""
echo "=== Deployment Tests ==="
python -m pytest test_deployment.py -v

# Run validation tests
echo ""
echo "=== Validation Tests ==="
python -m pytest test_production_validation.py -v

echo ""
echo "=== Tests Complete ==="
