"""Sanborn building footprints."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=ROOT / "config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
