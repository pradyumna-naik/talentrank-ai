import numpy as np
import pandas as pd

from run_pipeline import rank_all_jobs
from src.retrieval import calculate_hybrid_score, load_or_create_candidate_embeddings


class FakeSemanticModel:
    """Deterministic test encoder that avoids downloading a model."""

    def encode(self, texts, **kwargs):
        return np.array([
            [1.0, 0.0] if "python" in text or "data scientist" in text else [0.0, 1.0]
            for text in texts
        ])


def test_hybrid_score_uses_required_weights():
    assert calculate_hybrid_score(1.0, 0.5, 0.75) == 0.775


def test_candidate_embeddings_are_cached(tmp_path):
    model = FakeSemanticModel()
    cache_path = tmp_path / "candidate_embeddings.npy"
    first = load_or_create_candidate_embeddings(["python candidate"], cache_path, model)
    second = load_or_create_candidate_embeddings(["python candidate"], cache_path, model)
    assert cache_path.exists()
    assert np.array_equal(first, second)


def test_hybrid_ranking_exposes_component_scores(tmp_path):
    jobs = pd.DataFrame([{
        "job_id": "J1", "title": "Data Scientist", "description": "Build machine learning models.",
        "required_skills": "Python, SQL, machine learning", "preferred_skills": "pandas",
        "min_experience": 3, "max_experience": 6,
    }])
    candidates = pd.DataFrame([
        {"candidate_id": "1", "name": "Strong", "skills": "Python, SQL, machine learning", "experience_years": 3, "education": "B.Tech", "projects": "ML model", "summary": "Data scientist", "activity_score": 80},
        {"candidate_id": "2", "name": "Weak", "skills": "React, JavaScript", "experience_years": 3, "education": "B.Tech", "projects": "Web app", "summary": "Frontend developer", "activity_score": 80},
    ])
    results = rank_all_jobs(jobs, candidates, FakeSemanticModel(), tmp_path / "candidate_embeddings.npy")
    assert results.iloc[0]["candidate_id"] == "1"
    assert {"semantic_score", "tfidf_score", "skill_match_score"}.issubset(results.columns)
    assert "Semantic fit:" in results.iloc[0]["explanation"]
