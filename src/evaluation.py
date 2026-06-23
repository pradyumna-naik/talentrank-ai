"""Ranking evaluation metrics, diagnostics, and ablation analysis."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd


LABEL_COLUMNS = {"job_id", "candidate_id", "relevance"}


def has_relevance_labels(labels: pd.DataFrame | None) -> bool:
    """Return whether a label dataframe has the required usable relevance fields."""
    return labels is not None and LABEL_COLUMNS.issubset(labels.columns) and not labels.empty


def _rank_by_score(results: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """Re-rank candidate rows independently for each job by a score column."""
    ranked = results.sort_values(["job_id", score_column], ascending=[True, False]).copy()
    ranked["rank"] = ranked.groupby("job_id").cumcount() + 1
    return ranked


def _dcg(relevances: list[float]) -> float:
    """Compute discounted cumulative gain for a relevance sequence."""
    return sum((2**relevance - 1) / math.log2(position + 2) for position, relevance in enumerate(relevances))


def evaluate_with_labels(results: pd.DataFrame, labels: pd.DataFrame) -> dict[str, float]:
    """Calculate Precision@5, Precision@10, Recall@10, NDCG@10, and MRR."""
    if not has_relevance_labels(labels):
        raise ValueError("Labels must include non-empty job_id, candidate_id, and relevance columns")
    ranked = results.merge(labels.loc[:, list(LABEL_COLUMNS)], on=["job_id", "candidate_id"], how="left")
    ranked["relevance"] = ranked["relevance"].fillna(0.0).astype(float)
    grouped_labels = labels.groupby("job_id")["relevance"].apply(list)
    precision_5: list[float] = []
    precision_10: list[float] = []
    recall_10: list[float] = []
    ndcg_10: list[float] = []
    reciprocal_ranks: list[float] = []

    for job_id, group in ranked.sort_values("rank").groupby("job_id"):
        relevance = group["relevance"].tolist()
        relevant_total = sum(value > 0 for value in grouped_labels.get(job_id, []))
        top_5 = relevance[:5]
        top_10 = relevance[:10]
        precision_5.append(sum(value > 0 for value in top_5) / max(1, len(top_5)))
        precision_10.append(sum(value > 0 for value in top_10) / max(1, len(top_10)))
        recall_10.append(sum(value > 0 for value in top_10) / relevant_total if relevant_total else 0.0)
        ideal = sorted(grouped_labels.get(job_id, []), reverse=True)[:10]
        ideal_dcg = _dcg(ideal)
        ndcg_10.append(_dcg(top_10) / ideal_dcg if ideal_dcg else 0.0)
        first_relevant = next((index + 1 for index, value in enumerate(relevance) if value > 0), None)
        reciprocal_ranks.append(1 / first_relevant if first_relevant else 0.0)

    return {
        "Precision@5": float(np.mean(precision_5)),
        "Precision@10": float(np.mean(precision_10)),
        "Recall@10": float(np.mean(recall_10)),
        "NDCG@10": float(np.mean(ndcg_10)),
        "MRR": float(np.mean(reciprocal_ranks)),
    }


def unsupervised_diagnostics(results: pd.DataFrame, ranking_latency_seconds: float | None = None) -> dict[str, float]:
    """Summarize top-10 evidence when relevance labels are unavailable."""
    top_10 = results.sort_values("rank").groupby("job_id").head(10)
    diagnostics = {
        "average_skill_coverage_top_10": float(top_10["skill_match_score"].mean()) if not top_10.empty else 0.0,
        "average_semantic_score_top_10": float(top_10["semantic_score"].mean()) if not top_10.empty else 0.0,
        "average_experience_match_top_10": float(top_10["experience_match_score"].mean()) if not top_10.empty else 0.0,
        "ranking_latency_seconds": float(ranking_latency_seconds or 0.0),
    }
    return diagnostics


def build_ablation_table(
    results: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    ranking_latency_seconds: float | None = None,
) -> pd.DataFrame:
    """Compare score-only variants with supervised metrics or diagnostics."""
    hybrid_score = (
        0.45 * results["semantic_score"] + 0.35 * results["tfidf_score"]
        + 0.20 * results["skill_match_score"]
    )
    variants: dict[str, pd.Series | None] = {
        "TF-IDF only": results["tfidf_score"],
        "Semantic only": results["semantic_score"],
        "Hybrid": hybrid_score,
        "Hybrid + CrossEncoder": results["final_score"] if results["cross_encoder_score"].notna().any() else None,
    }
    rows: list[dict[str, object]] = []
    for name, scores in variants.items():
        if scores is None:
            rows.append({"variant": name, "status": "CrossEncoder not enabled"})
            continue
        variant_results = results.copy()
        variant_results["variant_score"] = scores
        variant_results = _rank_by_score(variant_results, "variant_score")
        row: dict[str, object] = {"variant": name, "status": "available"}
        if has_relevance_labels(labels):
            row.update(evaluate_with_labels(variant_results, labels))
        else:
            row.update(unsupervised_diagnostics(variant_results, ranking_latency_seconds))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_ranking(
    results: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    ranking_latency_seconds: float | None = None,
) -> tuple[str, Mapping[str, float]]:
    """Choose supervised evaluation when labels exist, otherwise diagnostics."""
    if has_relevance_labels(labels):
        return "Supervised metrics", evaluate_with_labels(results, labels)
    return "Unsupervised diagnostics", unsupervised_diagnostics(results, ranking_latency_seconds)
