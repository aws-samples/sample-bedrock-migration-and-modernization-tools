"""
Data Updater for Amazon Bedrock Model Profiler

This module provides functionality to update the model database using the new
amazon-bedrock-model-collector and amazon-bedrock-pricing-collector systems.

Author: AWS
License: MIT
"""

import subprocess  # nosec B404 - Required for AWS collector execution with proper input validation
import sys
import os
import glob
import json
import shutil
import re
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _validate_aws_profile_name(profile_name):
    """
    Validate AWS profile name to prevent command injection.
    AWS profile names should only contain alphanumeric characters, hyphens, underscores, and plus signs.
    """
    if not profile_name:
        return False

    # AWS profile names: alphanumeric, hyphens, underscores, plus signs, dots
    # Maximum length is typically 64 characters
    pattern = re.compile(r'^[a-zA-Z0-9._+-]+$')
    return len(profile_name) <= 64 and pattern.match(profile_name)


class ModelDataUpdater:
    """
    Handles updating the model database using the new collection system.
    """

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.model_collector_path = self.project_root / "collectors" / "model-collector"
        self.pricing_collector_path = self.project_root / "collectors" / "pricing-collector"
        self.data_path = self.project_root / "data"

    def run_pricing_collector(self):
        """
        Run the amazon-bedrock-pricing-collector to generate pricing data.
        """
        try:
            logger.info("Running amazon-bedrock-pricing-collector...")

            # Change to pricing collector directory
            original_dir = os.getcwd()
            os.chdir(self.pricing_collector_path)

            # Run the collector
            # nosemgrep: dangerous-subprocess-use-audit
            result = subprocess.run(
                [sys.executable, "main.py"],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout (pricing collection can take longer)
                shell=False  # nosec B603
            )

            os.chdir(original_dir)

            if result.returncode != 0:
                logger.error(f"Pricing collector failed: {result.stderr}")
                logger.warning("Continuing without pricing data...")
                return False

            logger.info("Pricing collector completed successfully")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Pricing collector timed out")
            logger.warning("Continuing without pricing data...")
            return False
        except Exception as e:
            logger.error(f"Error running pricing collector: {str(e)}")
            logger.warning("Continuing without pricing data...")
            return False

    def run_model_collector(self, profile_name=None):
        """
        Run the amazon-bedrock-model-collector to generate new model data.

        Args:
            profile_name: AWS profile to use (overrides config.py)

        Security: Uses list-based subprocess calls with shell=False.
        AWS profile names are validated using regex patterns to ensure
        they contain only safe characters. List format prevents command injection.
        """
        try:
            logger.info("Running amazon-bedrock-model-collector...")

            # Validate profile name to prevent command injection
            if profile_name and not _validate_aws_profile_name(profile_name):
                logger.error(f"Invalid AWS profile name: {profile_name}")
                logger.error("Profile names must contain only alphanumeric characters, dots, hyphens, underscores, and plus signs (max 64 chars)")
                return False

            # Build command with optional profile parameter (with additional security hardening)
            cmd = [sys.executable, "main.py"]
            if profile_name:
                # Profile name is already validated, no escaping needed for list-based subprocess
                cmd.extend(["--profile", profile_name])

            # Run the collector with real-time output
            logger.info("Starting model collector (this may take 10-20 minutes)...")

            # nosemgrep: dangerous-subprocess-use-audit
            process = subprocess.Popen(
                cmd,
                cwd=str(self.model_collector_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr with stdout
                text=True,
                bufsize=1,  # Line buffering
                universal_newlines=True,
                shell=False  # nosec B603
            )

            # Stream output in real-time with timeout
            import time
            output_lines = []
            start_time = time.time()
            timeout_seconds = 1200  # 20 minutes

            while True:
                # Check for timeout
                if time.time() - start_time > timeout_seconds:
                    logger.error(f"Model collector timed out after {timeout_seconds} seconds")
                    process.terminate()
                    return False

                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Log each line as it comes
                    line = output.strip()
                    logger.info(f"Collector: {line}")
                    output_lines.append(line)

            # Wait for process to complete
            return_code = process.poll()

            if return_code != 0:
                logger.error(f"Model collector failed with return code {return_code}")
                # Show last few lines for context
                if len(output_lines) > 5:
                    logger.error("Last few lines of output:")
                    for line in output_lines[-5:]:
                        logger.error(f"  {line}")
                return False

            logger.info("Model collector completed successfully")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Model collector timed out")
            return False
        except Exception as e:
            logger.error(f"Error running model collector: {str(e)}")
            return False

    def verify_model_data_exists(self):
        """
        Verify that the model data was created successfully (model collector now saves directly to data/ folder)
        """
        try:
            # Check if the target file exists (model collector saves directly here now)
            target_file = self.data_path / "bedrock_models.json"

            if not target_file.exists():
                logger.error("Model data file not found - model collector may have failed")
                return False

            # Check file is not empty and has recent timestamp
            file_stats = target_file.stat()
            if file_stats.st_size == 0:
                logger.error("Model data file is empty")
                return False

            # Check if file was modified in the last hour (fresh collection)
            current_time = datetime.now().timestamp()
            if current_time - file_stats.st_mtime > 3600:  # 1 hour
                logger.warning("Model data file is older than 1 hour - may not be fresh")

            logger.info(f"✅ Model data verified: {target_file}")
            logger.info(f"   File size: {file_stats.st_size:,} bytes")
            logger.info(f"   Last modified: {datetime.fromtimestamp(file_stats.st_mtime)}")
            return True

        except Exception as e:
            logger.error(f"Error verifying model data: {str(e)}")
            return False

    def validate_new_data(self):
        """
        Validate the new model data structure.
        """
        try:
            data_file = self.data_path / "bedrock_models.json"
            if not data_file.exists():
                logger.error("Model data file not found")
                return False

            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate structure
            required_keys = ['metadata', 'providers']
            for key in required_keys:
                if key not in data:
                    logger.error(f"Missing required key: {key}")
                    return False

            # Check metadata
            metadata = data['metadata']
            if 'total_models' not in metadata or metadata['total_models'] == 0:
                logger.error("No models found in data")
                return False

            logger.info(f"Data validation successful: {metadata['total_models']} models from {metadata['providers_count']} providers")
            return True

        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return False

    def update_model_data(self, profile_name=None):
        """
        Complete workflow to update model data.

        Args:
            profile_name: AWS profile to use (overrides config.py)
        """
        logger.info("Starting model data update...")

        # First run the pricing collector to get pricing data
        logger.info("Step 1: Running pricing collector...")
        pricing_success = self.run_pricing_collector()
        if pricing_success:
            logger.info("✅ Pricing data collection completed successfully")
        else:
            logger.warning("⚠️ Pricing data collection failed, continuing without pricing data")

        # Then run the model collector (which now saves directly to root data/ folder)
        logger.info("Step 2: Running model collector with pricing integration...")
        if not self.run_model_collector(profile_name=profile_name):
            return False

        # Verify the model data was created successfully
        if not self.verify_model_data_exists():
            return False

        # Validate the new data
        if not self.validate_new_data():
            return False

        logger.info("🎉 Model data update completed successfully!")
        logger.info("✅ Pricing collector: " + ("Completed" if pricing_success else "Failed (continuing without pricing)"))
        logger.info("✅ Model collector: Completed with direct output to data/ folder")
        logger.info("✅ Data verification: Passed")
        logger.info("✅ Data validation: Passed")
        return True


def update_data(profile_name=None):
    """
    Main function to update all data.

    Args:
        profile_name: AWS profile to use (overrides config.py)
    """
    updater = ModelDataUpdater()
    return updater.update_model_data(profile_name=profile_name)


if __name__ == "__main__":
    success = update_data()
    sys.exit(0 if success else 1)