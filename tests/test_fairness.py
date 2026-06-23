import pandas as pd

from src.fairness import CANDIDATE_RANKING_FEATURES, build_responsible_ai_report
from src.preprocessing import create_candidate_profile_text


def test_sensitive_columns_are_reported_and_excluded():
    report = build_responsible_ai_report(
        ["candidate_id", "name", "gender", "skills", "age", "summary"],
        ["job_id", "title", "description"],
    )
    assert report["sensitive_columns_detected"] == ["age", "gender", "name"]
    assert report["sensitive_columns_used_for_ranking"] == []
    assert "name" not in CANDIDATE_RANKING_FEATURES


def test_candidate_profile_text_excludes_sensitive_values():
    candidate = pd.Series({
        "name": "Private Name", "gender": "private gender", "address": "private address",
        "skills": "Python", "education": "B.Tech", "projects": "Built API", "summary": "Engineer",
        "experience_years": 3,
    })
    profile = create_candidate_profile_text(candidate)
    assert "private name" not in profile
    assert "private gender" not in profile
    assert "private address" not in profile
    assert "python" in profile
