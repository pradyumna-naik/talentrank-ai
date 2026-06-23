import pandas as pd

from src.explainability import build_explanation_card
from src.jd_parser import parse_jd


def test_explanation_card_contains_required_evidence():
    job = parse_jd(pd.Series({
        "title": "NLP Engineer", "description": "Build NLP APIs with FastAPI.",
        "required_skills": "Python, NLP, FastAPI, Docker", "preferred_skills": "AWS",
        "min_experience": 2, "max_experience": 4,
    }))
    candidate = pd.Series({
        "candidate_id": "C102", "skills": "Python, NLP, FastAPI, AWS", "experience_years": 3,
        "projects": "Built NLP classification APIs using FastAPI.", "summary": "NLP engineer", "activity_score": 85,
    })
    card = build_explanation_card(candidate, job, final_score=0.82, coverage_score=0.75)
    assert card.confidence_level == "High"
    assert card.matched_must_have_skills == ["Python", "NLP", "FastAPI"]
    assert card.matched_nice_to_have_skills == ["AWS"]
    assert card.missing_required_skills == ["Docker"]
    assert "3 years of experience for the 2-4 year role range" in card.experience_reason
    assert "NLP, FastAPI" in card.project_reason
    assert "Gap: Docker not found" in card.risk_or_gap
    assert "match 3/4 required skills" in card.fit_summary
