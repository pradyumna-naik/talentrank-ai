"""Persist ranking outputs and responsible-AI feature reports."""
import json
from pathlib import Path
from typing import Mapping

import pandas as pd


OUTPUT_COLUMNS = ("job_id", "candidate_id", "rank", "final_score", "explanation")


def write_outputs(results: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path, Path]:
    """Write the required ranked output CSV, JSON, and XLSX representations."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output = results.loc[:, OUTPUT_COLUMNS]
    csv_path = directory / "ranked_output.csv"
    json_path = directory / "ranked_output.json"
    xlsx_path = directory / "ranked_output.xlsx"
    output.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(output.to_dict(orient="records"), indent=2), encoding="utf-8")
    output.to_excel(xlsx_path, index=False, sheet_name="ranked_candidates")
    return csv_path, json_path, xlsx_path


def write_used_features_report(report: Mapping[str, object], output_dir: str | Path) -> Path:
    """Write the responsible-AI report describing fields used for ranking."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "used_features_report.json"
    report_path.write_text(json.dumps(dict(report), indent=2), encoding="utf-8")
    return report_path
