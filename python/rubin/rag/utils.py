"""Utility functions for the Rubin RAG application."""

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parents[3] / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    """Load and return the YAML config file as a dictionary."""
    with path.open() as f:
        return yaml.safe_load(f)
