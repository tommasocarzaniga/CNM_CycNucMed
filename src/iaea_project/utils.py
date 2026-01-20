from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../my-project
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

def ensure_dirs() -> None:
    for p in [RAW_DIR, PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)
