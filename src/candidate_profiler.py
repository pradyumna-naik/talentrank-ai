"""Turn raw candidate rows into consistent candidate profiles."""
from dataclasses import dataclass

import pandas as pd

from src.preprocessing import join_text, split_skills


@dataclass(frozen=True)
class CandidateProfile:
    """A normalized candidate profile used by the ranking pipeline."""

    candidate_id: str
    name: str
    skills: list[str]
    experience_years: float
    text: str


def build_candidate_profiles(candidates: pd.DataFrame) -> list[CandidateProfile]:
    """Build profiles from the candidate input dataframe."""
    profiles: list[CandidateProfile] = []
    for row in candidates.to_dict(orient="records"):
        skills = split_skills(row["skills"])
        profiles.append(CandidateProfile(
            candidate_id=str(row["candidate_id"]), name=str(row["name"]), skills=skills,
            experience_years=float(row["experience_years"]),
            text=join_text([row["profile"], row["skills"]]),
        ))
    return profiles
