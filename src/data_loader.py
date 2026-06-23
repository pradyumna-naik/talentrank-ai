"""CSV loading and lightweight validation utilities."""
from pathlib import Path

import pandas as pd


REQUIRED_CANDIDATE_COLUMNS = {"candidate_id", "name", "skills", "experience_years", "profile"}
REQUIRED_JOB_COLUMNS = {"job_id", "title", "description"}


def load_csv(path: str | Path, required_columns: set[str]) -> pd.DataFrame:
    """Load a CSV file and raise a clear error if required columns are missing."""
    dataframe = pd.read_csv(path)
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if dataframe.empty:
        raise ValueError(f"{path} contains no records")
    return dataframe


def load_candidates(path: str | Path) -> pd.DataFrame:
    """Load candidate data."""
    return load_csv(path, REQUIRED_CANDIDATE_COLUMNS)


def load_jobs(path: str | Path) -> pd.DataFrame:
    """Load job-description data."""
    return load_csv(path, REQUIRED_JOB_COLUMNS)
