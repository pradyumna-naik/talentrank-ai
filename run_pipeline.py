"""Run the hybrid TF-IDF and semantic candidate-ranking pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CANDIDATE_EMBEDDINGS_PATH, OUTPUT_DIR, RAW_DATA_DIR
from src.data_loader import load_candidates, load_jobs
from src.explainability import build_explanation, skill_match_score
from src.jd_parser import parse_job
from src.output_writer import write_outputs
from src.preprocessing import clean_text_fields, create_candidate_profile_text
from src.retrieval import (
    calculate_hybrid_score,
    calculate_semantic_scores,
    calculate_tfidf_scores,
    load_or_create_candidate_embeddings,
    load_sentence_model,
)


CANDIDATE_TEXT_FIELDS = ("skills", "education", "projects", "summary")
JOB_TEXT_FIELDS = ("title", "description", "required_skills", "preferred_skills")


def _rank_prepared(
    job_row: pd.Series,
    candidates: pd.DataFrame,
    candidate_texts: list[str],
    candidate_embeddings: Any,
    semantic_model: Any,
) -> pd.DataFrame:
    """Rank pre-cleaned candidates for one pre-cleaned job."""
    job = parse_job(job_row)
    tfidf_scores = calculate_tfidf_scores(job.profile_text, candidate_texts)
    semantic_scores = calculate_semantic_scores(job.profile_text, candidate_embeddings, semantic_model)
    records: list[dict[str, object]] = []
    for index, candidate in candidates.reset_index(drop=True).iterrows():
        coverage = skill_match_score(candidate["skills"], job.required_skills)
        hybrid = calculate_hybrid_score(float(semantic_scores[index]), float(tfidf_scores[index]), coverage)
        records.append({
            "job_id": job.job_id,
            "candidate_id": str(candidate["candidate_id"]),
            "semantic_score": round(float(semantic_scores[index]) * 100, 2),
            "tfidf_score": round(float(tfidf_scores[index]) * 100, 2),
            "skill_match_score": round(coverage * 100, 2),
            "final_score": round(hybrid * 100, 2),
            "explanation": build_explanation(
                candidate["skills"], job.required_skills,
                float(semantic_scores[index]), float(tfidf_scores[index]), coverage,
            ),
        })
    results = pd.DataFrame(records).sort_values("final_score", ascending=False, ignore_index=True)
    results.insert(2, "rank", range(1, len(results) + 1))
    return results


def rank_candidates(
    job_row: pd.Series,
    candidates: pd.DataFrame,
    semantic_model: Any | None = None,
    embedding_cache_path: str | Path = CANDIDATE_EMBEDDINGS_PATH,
) -> pd.DataFrame:
    """Rank candidates for one job using the hybrid semantic score."""
    cleaned_candidates = clean_text_fields(candidates, CANDIDATE_TEXT_FIELDS)
    cleaned_job = clean_text_fields(pd.DataFrame([job_row]), JOB_TEXT_FIELDS).iloc[0]
    model = semantic_model or load_sentence_model()
    candidate_texts = cleaned_candidates.apply(create_candidate_profile_text, axis=1).tolist()
    candidate_embeddings = load_or_create_candidate_embeddings(candidate_texts, embedding_cache_path, model)
    return _rank_prepared(cleaned_job, cleaned_candidates, candidate_texts, candidate_embeddings, model)


def rank_all_jobs(
    jobs: pd.DataFrame,
    candidates: pd.DataFrame,
    semantic_model: Any | None = None,
    embedding_cache_path: str | Path = CANDIDATE_EMBEDDINGS_PATH,
) -> pd.DataFrame:
    """Clean inputs and produce hybrid rankings for every job-candidate pairing."""
    cleaned_candidates = clean_text_fields(candidates, CANDIDATE_TEXT_FIELDS)
    cleaned_jobs = clean_text_fields(jobs, JOB_TEXT_FIELDS)
    model = semantic_model or load_sentence_model()
    candidate_texts = cleaned_candidates.apply(create_candidate_profile_text, axis=1).tolist()
    candidate_embeddings = load_or_create_candidate_embeddings(candidate_texts, embedding_cache_path, model)
    rankings = [
        _rank_prepared(job, cleaned_candidates, candidate_texts, candidate_embeddings, model)
        for _, job in cleaned_jobs.iterrows()
    ]
    return pd.concat(rankings, ignore_index=True)


def main() -> None:
    """Run hybrid ranking from CSV inputs and write ranked_output.csv."""
    parser = argparse.ArgumentParser(description="Rank candidates with hybrid semantic similarity.")
    parser.add_argument("--jobs", default=RAW_DATA_DIR / "jobs.csv", help="Path to jobs CSV")
    parser.add_argument("--candidates", default=RAW_DATA_DIR / "candidates.csv", help="Path to candidates CSV")
    args = parser.parse_args()
    results = rank_all_jobs(load_jobs(args.jobs), load_candidates(args.candidates))
    csv_path, _ = write_outputs(results, OUTPUT_DIR)
    print(results.to_string(index=False))
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
