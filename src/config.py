"""Central paths and ranking defaults."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"

TOP_K = 10
RETRIEVAL_WEIGHT = 0.50
SKILL_WEIGHT = 0.35
EXPERIENCE_WEIGHT = 0.15

for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
