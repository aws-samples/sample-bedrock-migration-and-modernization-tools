# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Module entry point for running as: python -m agent_eval.tools.agentcore_pipeline.export_agentcore_pipeline
"""
from .export_agentcore_pipeline import main
import sys

if __name__ == "__main__":
    sys.exit(main())
