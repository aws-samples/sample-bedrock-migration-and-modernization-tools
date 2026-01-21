#!/bin/bash
#
# Quick script to run the Step Functions workflow and analyze pricing coverage
# Use this after manual deployments or when you want to refresh the data
#
# Usage: ./run_workflow_and_analyze.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/deploy_and_analyze.sh" --skip-deploy --run-workflow
