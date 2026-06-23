"""Text cleaning and profile construction for the baseline pipeline."""
import re
from collections.abc import Iterable

import pandas as pd


def normalize_text(value: object) -> str:
    """Lowercase text and collapse whitespace, treating missing values as empty."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def split_skills(value: object) -> list[str]:
    """Split a comma-, semicolon-, or pipe-delimited skills field."""
    parts = re.split(r"[,;|]", normalize_text(value))
    return list(dict.fromkeys(part.strip() for part in parts if part.strip()))


def join_text(parts: Iterable[object]) -> str:
    """Join non-empty values into normalized model input text."""
    return normalize_text(" ".join(str(part) for part in parts if normalize_text(part)))


def clean_text_fields(dataframe: pd.DataFrame, fields: Iterable[str]) -> pd.DataFrame:
    """Return a copy with all specified text fields normalized."""
    cleaned = dataframe.copy()
    for field in fields:
        cleaned[field] = cleaned[field].map(normalize_text)
    return cleaned


def create_candidate_profile_text(candidate: pd.Series) -> str:
    """Combine the requested candidate fields into one retrieval profile."""
    return join_text([
        candidate["skills"], candidate["projects"], candidate["summary"],
        candidate["education"], f"{candidate['experience_years']} years experience",
    ])
