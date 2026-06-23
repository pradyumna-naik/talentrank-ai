"""Persist ranking outputs for downstream review."""
import json
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = ("job_id", "candidate_id", "rank", "final_score", "explanation")


def write_outputs(results: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    """Write the required ranked output CSV and matching JSON representation."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output = results.loc[:, OUTPUT_COLUMNS]
    csv_path = directory / "ranked_output.csv"
    json_path = directory / "ranked_output.json"
    output.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(output.to_dict(orient="records"), indent=2), encoding="utf-8")
    return csv_path, json_path
