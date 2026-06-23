"""Persist ranking outputs for downstream review."""
import json
from pathlib import Path

import pandas as pd


def write_outputs(results: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    """Write CSV and JSON ranking results, returning both paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "ranked_candidates.csv"
    json_path = directory / "ranked_candidates.json"
    results.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(results.to_dict(orient="records"), indent=2), encoding="utf-8")
    return csv_path, json_path
