"""
Local file output manager.

Provides file I/O operations that replace S3 for local execution.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OutputManager:
    """Manages local file output for collected data."""

    def __init__(self, output_dir: Path):
        """
        Initialize the output manager.

        Args:
            output_dir: Base directory for output files (e.g., ./data)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, filename: str, data: Any) -> Path:
        """
        Write data to a JSON file.

        Args:
            filename: Name of the file (e.g., 'bedrock_models.json')
            data: Data to write

        Returns:
            Path to the written file
        """
        filepath = self.output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f"Wrote {filepath} ({filepath.stat().st_size:,} bytes)")
        return filepath

    def read_json(self, filename: str) -> Any:
        """
        Read data from a JSON file.

        Args:
            filename: Name of the file

        Returns:
            Parsed JSON data
        """
        filepath = self.output_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
