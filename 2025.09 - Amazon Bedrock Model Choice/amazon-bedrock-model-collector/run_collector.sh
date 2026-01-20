#!/bin/bash

# Amazon Bedrock Model Collector Runner Script
# Linux/macOS version

set -e  # Exit on error

echo "🚀 Amazon Bedrock Model Collector - Linux/macOS Runner"
echo "========================================================"
echo "Comprehensive model database with enhanced features"

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
MAIN_SCRIPT="$SCRIPT_DIR/main.py"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
print_status "Using Python $PYTHON_VERSION"

# Create/activate virtual environment
if [ ! -d "$VENV_DIR" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created"
fi

print_status "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install requirements
print_status "Installing requirements..."
pip install --upgrade pip
pip install -r "$REQUIREMENTS_FILE"

# Create directories
mkdir -p "$SCRIPT_DIR/out" "$SCRIPT_DIR/logs"

# Check AWS credentials
print_status "Checking AWS credentials for profile: avelizf+main-Admin"
if ! python3 -c "import boto3; boto3.Session(profile_name='avelizf+main-Admin')" 2>/dev/null; then
    echo "⚠️  AWS credentials may not be configured for profile 'avelizf+main-Admin'"
fi

# Run collector
print_status "Starting Amazon Bedrock Model Collector..."
echo ""
python3 main.py

if [ $? -eq 0 ]; then
    print_success "Model collection completed successfully!"
    print_status "Output files: $SCRIPT_DIR/out/"
    print_status "Log files: $SCRIPT_DIR/logs/"
else
    echo "❌ Collection failed. Check logs for details."
    exit 1
fi

deactivate
print_success "Script completed!"