#!/bin/bash

# Amazon Bedrock Pricing Collector Runner Script
# Linux/macOS version

set -e  # Exit on error

echo "🚀 Amazon Bedrock Pricing Collector - Linux/macOS Runner"
echo "========================================================"

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
MAIN_SCRIPT="$SCRIPT_DIR/main.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
print_status "Using Python $PYTHON_VERSION"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created at $VENV_DIR"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install/upgrade requirements
if [ -f "$REQUIREMENTS_FILE" ]; then
    print_status "Installing requirements from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
    print_success "Requirements installed successfully"
else
    print_warning "requirements.txt not found. Installing basic dependencies..."
    pip install boto3 requests
fi

# Ensure output and logs directories exist
mkdir -p "$SCRIPT_DIR/out"
mkdir -p "$SCRIPT_DIR/logs"

# Check for AWS credentials
print_status "Checking AWS credentials..."
if ! python3 -c "import boto3; boto3.Session(profile_name='avelizf+main-Admin')" 2>/dev/null; then
    print_warning "AWS credentials may not be configured properly"
    print_warning "Make sure your AWS profile 'avelizf+main-Admin' is set up"
fi

# Run the pricing collector
print_status "Starting Amazon Bedrock Pricing Collector..."
echo ""

cd "$SCRIPT_DIR"
python3 main.py

# Check exit code
if [ $? -eq 0 ]; then
    print_success "Pricing collection completed successfully!"
    echo ""
    print_status "Output files are available in: $SCRIPT_DIR/out/"
    print_status "Log files are available in: $SCRIPT_DIR/logs/"
else
    print_error "Pricing collection failed. Check the logs for details."
    exit 1
fi

# Deactivate virtual environment
deactivate

print_success "Script completed successfully!"