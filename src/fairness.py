"""Responsible-AI safeguards for candidate ranking inputs."""
from __future__ import annotations

import warnings
from collections.abc import Iterable


SENSITIVE_COLUMNS = frozenset({
    "name", "gender", "religion", "caste", "age", "date_of_birth", "photo",
    "marital_status", "address",
})
CANDIDATE_RANKING_FEATURES = ("skills", "education", "projects", "summary", "experience_years", "activity_score")
JOB_RANKING_FEATURES = ("title", "description", "required_skills", "preferred_skills", "min_experience", "max_experience")
DISCLAIMER = "This system is a recruiter decision-support tool. Final hiring decisions should be human-reviewed."


def find_sensitive_columns(columns: Iterable[object]) -> list[str]:
    """Return sensitive input columns found in a dataset, case-insensitively."""
    return sorted(str(column) for column in columns if str(column).lower() in SENSITIVE_COLUMNS)


def build_responsible_ai_report(candidate_columns: Iterable[object], job_columns: Iterable[object]) -> dict[str, object]:
    """Build a serializable report of used and deliberately excluded fields."""
    found = find_sensitive_columns(candidate_columns)
    if found:
        warnings.warn(
            f"Sensitive candidate columns detected and excluded from ranking: {', '.join(found)}.",
            UserWarning,
            stacklevel=2,
        )
    return {
        "used_candidate_features": list(CANDIDATE_RANKING_FEATURES),
        "used_job_features": list(JOB_RANKING_FEATURES),
        "identifier_only_fields": ["candidate_id", "job_id"],
        "sensitive_columns_detected": found,
        "excluded_sensitive_columns": sorted(SENSITIVE_COLUMNS),
        "sensitive_columns_used_for_ranking": [],
        "disclaimer": DISCLAIMER,
    }
