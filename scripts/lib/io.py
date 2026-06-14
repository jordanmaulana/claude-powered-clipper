"""Shared file I/O helpers."""

import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    """Read a JSON file, exiting with a clear error if it is missing."""
    if not path.exists():
        sys.exit(f"error: {path} not found")
    return json.loads(path.read_text())
