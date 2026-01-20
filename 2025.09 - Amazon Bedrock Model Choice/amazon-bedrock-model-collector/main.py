#!/usr/bin/env python3
"""
Amazon Bedrock Model Collector - Entry Point
Entry point script that runs the main collector from the src directory
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

# Change working directory to src for relative imports
os.chdir(src_dir)

# Import and run the main function
import main

if __name__ == "__main__":
    main.main()