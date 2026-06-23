"""Central paths and ranking defaults."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"
CANDIDATE_EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "candidate_embeddings.npy"

TOP_K = 10
TOP_RETRIEVAL_CANDIDATES = 50
USE_CROSS_ENCODER = False

for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
