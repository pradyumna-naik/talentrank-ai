"""Turn raw candidate rows into consistent candidate profiles."""
from dataclasses import dataclass

import pandas as pd

from src.preprocessing import create_candidate_profile_text, split_skills


@dataclass(frozen=True)
class CandidateProfile:
    """A normalized candidate profile used by ranking helpers."""

    candidate_id: str
    name: str
    skills: list[str]
    experience_years: float
    profile_text: str


def build_candidate_profiles(candidates: pd.DataFrame) -> list[CandidateProfile]:
    """Build normalized candidate profiles from the baseline input schema."""
    profiles: list[CandidateProfile] = []
    for _, row in candidates.iterrows():
        profiles.append(CandidateProfile(
            candidate_id=str(row["candidate_id"]),
            name=str(row["name"]),
            skills=split_skills(row["skills"]),
            experience_years=float(row["experience_years"]),
            profile_text=create_candidate_profile_text(row),
        ))
    return profiles
