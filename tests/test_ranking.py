import pandas as pd

from run_pipeline import rank_candidates


def test_stronger_skill_match_ranks_higher():
    job = pd.Series({"title": "Data Scientist", "description": "Need 3 years of Python, SQL and machine learning."})
    candidates = pd.DataFrame([
        {"candidate_id": "1", "name": "Strong", "skills": "Python, SQL, machine learning", "experience_years": 3, "profile": "Data scientist"},
        {"candidate_id": "2", "name": "Weak", "skills": "React, JavaScript", "experience_years": 3, "profile": "Frontend developer"},
    ])
    results = rank_candidates(job, candidates)
    assert results.iloc[0]["name"] == "Strong"
    assert "python" in results.iloc[0]["matched_skills"]
