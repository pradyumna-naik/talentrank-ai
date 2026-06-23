"""Run hybrid retrieval with optional CrossEncoder candidate reranking."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    CANDIDATE_EMBEDDINGS_PATH,
    OUTPUT_DIR,
    RAW_DATA_DIR,
    TOP_RETRIEVAL_CANDIDATES,
    USE_CROSS_ENCODER,
)
from src.data_loader import load_candidates, load_jobs
from src.explainability import build_explanation_card, skill_match_score
from src.fairness import build_responsible_ai_report
from src.jd_parser import parse_job
from src.output_writer import write_outputs, write_used_features_report
from src.preprocessing import clean_text_fields, create_candidate_profile_text
from src.retrieval import (
    calculate_cross_encoder_scores,
    calculate_hybrid_score,
    calculate_semantic_scores,
    calculate_tfidf_scores,
    load_cross_encoder,
    load_or_create_candidate_embeddings,
    load_sentence_model,
)


logger = logging.getLogger(__name__)
CANDIDATE_TEXT_FIELDS = ("skills", "education", "projects", "summary")
JOB_TEXT_FIELDS = ("title", "description", "required_skills", "preferred_skills")


def experience_match_score(experience_years: object, min_experience: float | None, max_experience: float | None) -> float:
    """Score experience against a job range, retaining partial credit near its bounds."""
    experience = float(experience_years)
    if min_experience is not None and experience < min_experience:
        return max(0.0, experience / min_experience)
    if max_experience is not None and experience > max_experience:
        return max(0.0, max_experience / experience)
    return 1.0


def normalized_activity_score(value: object) -> float:
    """Clamp an activity score supplied on a 0–100 scale to a 0–1 score."""
    numeric = pd.to_numeric(value, errors="coerce")
    return 0.0 if pd.isna(numeric) else float(np.clip(float(numeric) / 100, 0.0, 1.0))


def calculate_final_score(
    cross_encoder_score: float | None,
    skill_score: float,
    semantic_score: float,
    tfidf_score: float,
    experience_score: float,
    activity_score: float,
) -> float:
    """Calculate final ranking score, redistributing CrossEncoder weight when disabled."""
    if cross_encoder_score is None:
        return (
            0.35 * skill_score + 0.35 * semantic_score + 0.15 * tfidf_score
            + 0.10 * experience_score + 0.05 * activity_score
        )
    return (
        0.30 * cross_encoder_score + 0.20 * skill_score + 0.20 * semantic_score
        + 0.15 * tfidf_score + 0.10 * experience_score + 0.05 * activity_score
    )


def _rank_prepared(
    job_row: pd.Series,
    candidates: pd.DataFrame,
    candidate_texts: list[str],
    candidate_embeddings: Any,
    semantic_model: Any,
    use_cross_encoder: bool,
    cross_encoder_model: Any | None,
) -> pd.DataFrame:
    """Retrieve the top 50 candidates, then optionally rerank them with CrossEncoder."""
    job = parse_job(job_row)
    candidate_rows = candidates.reset_index(drop=True)
    tfidf_scores = calculate_tfidf_scores(job.profile_text, candidate_texts)
    semantic_scores = calculate_semantic_scores(job.profile_text, candidate_embeddings, semantic_model)
    skill_scores = np.array([skill_match_score(row["skills"], job.required_skills) for _, row in candidate_rows.iterrows()])
    retrieval_scores = np.array([
        calculate_hybrid_score(float(semantic_scores[index]), float(tfidf_scores[index]), float(skill_scores[index]))
        for index in range(len(candidate_rows))
    ])
    top_indices = np.argsort(-retrieval_scores)[:TOP_RETRIEVAL_CANDIDATES]

    cross_scores: dict[int, float] = {}
    if use_cross_encoder:
        if cross_encoder_model is None:
            raise ValueError("CrossEncoder model is required when reranking is enabled")
        rerank_texts = [candidate_texts[int(index)] for index in top_indices]
        values = calculate_cross_encoder_scores(job.profile_text, rerank_texts, cross_encoder_model)
        cross_scores = {int(index): float(value) for index, value in zip(top_indices, values)}

    records: list[dict[str, object]] = []
    for index in top_indices:
        candidate = candidate_rows.iloc[int(index)]
        experience_score = experience_match_score(
            candidate["experience_years"], job.intelligence.min_experience, job.intelligence.max_experience,
        )
        activity_score = normalized_activity_score(candidate["activity_score"])
        cross_score = cross_scores.get(int(index)) if use_cross_encoder else None
        final_score = calculate_final_score(
            cross_score, float(skill_scores[index]), float(semantic_scores[index]), float(tfidf_scores[index]),
            experience_score, activity_score,
        )
        explanation_card = build_explanation_card(candidate, job.intelligence, final_score, float(skill_scores[index]))
        records.append({
            "job_id": job.job_id,
            "candidate_id": str(candidate["candidate_id"]),
            "cross_encoder_score": round(cross_score * 100, 2) if cross_score is not None else None,
            "semantic_score": round(float(semantic_scores[index]) * 100, 2),
            "tfidf_score": round(float(tfidf_scores[index]) * 100, 2),
            "skill_match_score": round(float(skill_scores[index]) * 100, 2),
            "experience_match_score": round(experience_score * 100, 2),
            "activity_score": round(activity_score * 100, 2),
            "final_score": round(final_score * 100, 2),
            "explanation": explanation_card.fit_summary,
            "explanation_card": explanation_card.to_dict(),
        })
    results = pd.DataFrame(records).sort_values("final_score", ascending=False, ignore_index=True)
    results.insert(2, "rank", range(1, len(results) + 1))
    return results


def rank_candidates(
    job_row: pd.Series,
    candidates: pd.DataFrame,
    semantic_model: Any | None = None,
    embedding_cache_path: str | Path = CANDIDATE_EMBEDDINGS_PATH,
    use_cross_encoder: bool | None = None,
    cross_encoder_model: Any | None = None,
) -> pd.DataFrame:
    """Rank candidates for one job and log the elapsed ranking time."""
    started_at = time.perf_counter()
    use_cross_encoder = USE_CROSS_ENCODER if use_cross_encoder is None else use_cross_encoder
    cleaned_candidates = clean_text_fields(candidates, CANDIDATE_TEXT_FIELDS)
    cleaned_job = clean_text_fields(pd.DataFrame([job_row]), JOB_TEXT_FIELDS).iloc[0]
    model = semantic_model or load_sentence_model()
    reranker = cross_encoder_model or (load_cross_encoder() if use_cross_encoder else None)
    candidate_texts = cleaned_candidates.apply(create_candidate_profile_text, axis=1).tolist()
    candidate_embeddings = load_or_create_candidate_embeddings(candidate_texts, embedding_cache_path, model)
    results = _rank_prepared(cleaned_job, cleaned_candidates, candidate_texts, candidate_embeddings, model, use_cross_encoder, reranker)
    elapsed = time.perf_counter() - started_at
    results.attrs["ranking_latency_seconds"] = elapsed
    logger.info("Ranked %d candidates in %.2f seconds (cross_encoder=%s)", len(candidates), elapsed, use_cross_encoder)
    return results


def rank_all_jobs(
    jobs: pd.DataFrame,
    candidates: pd.DataFrame,
    semantic_model: Any | None = None,
    embedding_cache_path: str | Path = CANDIDATE_EMBEDDINGS_PATH,
    use_cross_encoder: bool | None = None,
    cross_encoder_model: Any | None = None,
) -> pd.DataFrame:
    """Rank all jobs and log total elapsed ranking time."""
    started_at = time.perf_counter()
    use_cross_encoder = USE_CROSS_ENCODER if use_cross_encoder is None else use_cross_encoder
    cleaned_candidates = clean_text_fields(candidates, CANDIDATE_TEXT_FIELDS)
    cleaned_jobs = clean_text_fields(jobs, JOB_TEXT_FIELDS)
    model = semantic_model or load_sentence_model()
    reranker = cross_encoder_model or (load_cross_encoder() if use_cross_encoder else None)
    candidate_texts = cleaned_candidates.apply(create_candidate_profile_text, axis=1).tolist()
    candidate_embeddings = load_or_create_candidate_embeddings(candidate_texts, embedding_cache_path, model)
    rankings = [
        _rank_prepared(job, cleaned_candidates, candidate_texts, candidate_embeddings, model, use_cross_encoder, reranker)
        for _, job in cleaned_jobs.iterrows()
    ]
    results = pd.concat(rankings, ignore_index=True)
    elapsed = time.perf_counter() - started_at
    results.attrs["ranking_latency_seconds"] = elapsed
    logger.info("Ranked %d jobs against %d candidates in %.2f seconds (cross_encoder=%s)", len(jobs), len(candidates), elapsed, use_cross_encoder)
    return results


def main() -> None:
    """Run ranking from CSV inputs and write ranked_output.csv."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="Rank candidates with optional CrossEncoder reranking.")
    parser.add_argument("--jobs", default=RAW_DATA_DIR / "jobs.csv", help="Path to jobs CSV")
    parser.add_argument("--candidates", default=RAW_DATA_DIR / "candidates.csv", help="Path to candidates CSV")
    parser.add_argument("--cross-encoder", action="store_true", help="Enable slower CrossEncoder reranking")
    args = parser.parse_args()
    use_cross_encoder = True if args.cross_encoder else USE_CROSS_ENCODER
    jobs = load_jobs(args.jobs)
    candidates = load_candidates(args.candidates)
    responsible_ai_report = build_responsible_ai_report(candidates.columns, jobs.columns)
    report_path = write_used_features_report(responsible_ai_report, OUTPUT_DIR)
    results = rank_all_jobs(jobs, candidates, use_cross_encoder=use_cross_encoder)
    csv_path, _ = write_outputs(results, OUTPUT_DIR)
    print(results[["job_id", "candidate_id", "rank", "final_score", "explanation"]].to_string(index=False))
    print(f"\nSaved: {csv_path}\nSaved: {report_path}")


if __name__ == "__main__":
    main()





