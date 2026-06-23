import pandas as pd
import pytest

from src.evaluation import build_ablation_table, evaluate_with_labels, unsupervised_diagnostics


def _results() -> pd.DataFrame:
    return pd.DataFrame({
        "job_id": ["J1", "J1", "J1"],
        "candidate_id": ["C1", "C2", "C3"],
        "rank": [1, 2, 3],
        "tfidf_score": [90.0, 60.0, 20.0],
        "semantic_score": [80.0, 50.0, 40.0],
        "skill_match_score": [100.0, 50.0, 0.0],
        "experience_match_score": [100.0, 100.0, 50.0],
        "cross_encoder_score": [95.0, 40.0, 10.0],
        "final_score": [90.0, 50.0, 20.0],
    })


def test_supervised_ranking_metrics():
    labels = pd.DataFrame({
        "job_id": ["J1", "J1", "J1"],
        "candidate_id": ["C1", "C2", "C3"],
        "relevance": [1, 0, 1],
    })
    metrics = evaluate_with_labels(_results(), labels)
    assert metrics["Precision@5"] == pytest.approx(2 / 3)
    assert metrics["Precision@10"] == pytest.approx(2 / 3)
    assert metrics["Recall@10"] == 1.0
    assert metrics["MRR"] == 1.0
    assert metrics["NDCG@10"] > 0.9


def test_unsupervised_diagnostics_and_ablation_table():
    results = _results()
    diagnostics = unsupervised_diagnostics(results, ranking_latency_seconds=0.25)
    assert diagnostics["average_skill_coverage_top_10"] == pytest.approx(50.0)
    assert diagnostics["ranking_latency_seconds"] == 0.25
    table = build_ablation_table(results, ranking_latency_seconds=0.25)
    assert set(table["variant"]) == {"TF-IDF only", "Semantic only", "Hybrid", "Hybrid + CrossEncoder"}
    assert set(table["status"]) == {"available"}
