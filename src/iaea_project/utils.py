from __future__ import annotations

from pathlib import Path

# Repo root = .../my-project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

CACHE_DIR = PROCESSED_DIR / "cache"


def ensure_dirs() -> None:
    """Create expected repo directories if missing."""
    for p in [RAW_DIR, PROCESSED_DIR, CACHE_DIR, FIGURES_DIR, REPORTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)
