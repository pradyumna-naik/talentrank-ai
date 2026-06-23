"""CSV loading and schema validation utilities."""
from pathlib import Path

import pandas as pd


REQUIRED_CANDIDATE_COLUMNS = {
    "candidate_id", "name", "skills", "experience_years", "education",
    "projects", "summary", "activity_score",
}
REQUIRED_JOB_COLUMNS = {
    "job_id", "title", "description", "required_skills", "preferred_skills",
    "min_experience", "max_experience",
}


def load_csv(path: str | Path, required_columns: set[str]) -> pd.DataFrame:
    """Load a non-empty CSV and validate its required columns."""
    dataframe = pd.read_csv(path)
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if dataframe.empty:
        raise ValueError(f"{path} contains no records")
    return dataframe


def load_candidates(path: str | Path) -> pd.DataFrame:
    """Load candidates.csv using the baseline candidate schema."""
    return load_csv(path, REQUIRED_CANDIDATE_COLUMNS)


def load_jobs(path: str | Path) -> pd.DataFrame:
    """Load jobs.csv using the baseline job schema."""
    return load_csv(path, REQUIRED_JOB_COLUMNS)
