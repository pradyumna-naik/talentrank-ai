"""Job parsing and TF-IDF profile construction."""
from dataclasses import dataclass

import pandas as pd

from src.preprocessing import join_text, split_skills


@dataclass(frozen=True)
class JobProfile:
    """Structured job data needed by baseline ranking."""

    job_id: str
    required_skills: list[str]
    profile_text: str


def parse_job(row: pd.Series) -> JobProfile:
    """Parse one jobs.csv row into required skills and profile text."""
    return JobProfile(
        job_id=str(row["job_id"]),
        required_skills=split_skills(row["required_skills"]),
        profile_text=join_text([
            row["title"], row["description"], row["required_skills"], row["preferred_skills"],
        ]),
    )
