"""Run TalentRank AI against a selected job and a candidate CSV."""
from __future__ import annotations

import argparse

import pandas as pd

from src.candidate_profiler import build_candidate_profiles
from src.config import OUTPUT_DIR, RAW_DATA_DIR, TOP_K
from src.data_loader import load_candidates, load_jobs
from src.explainability import build_explanation
from src.feature_engineering import build_match_features
from src.jd_parser import parse_job_description
from src.output_writer import write_outputs
from src.reranker import rerank
from src.retrieval import retrieve_candidates


def rank_candidates(job_row: pd.Series, candidates: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    """Rank candidates for a single job row and return explainable results."""
    job = parse_job_description(str(job_row["title"]), str(job_row["description"]))
    profiles = build_candidate_profiles(candidates)
    retrieved = retrieve_candidates(job.text, [profile.text for profile in profiles], top_k)
    records: list[dict[str, object]] = []
    for index, relevance in retrieved:
        candidate = profiles[index]
        features = build_match_features(job, candidate)
        score = rerank(relevance, features)
        records.append({
            "candidate_id": candidate.candidate_id,
            "name": candidate.name,
            "experience_years": candidate.experience_years,
            "matched_skills": ", ".join(features.matched_skills),
            "missing_skills": ", ".join(features.missing_skills),
            "relevance_score": round(score.relevance_score * 100, 2),
            "skill_score": round(score.skill_score * 100, 2),
            "experience_score": round(score.experience_score * 100, 2),
            "final_score": round(score.final_score, 2),
            "explanation": build_explanation(candidate, features),
        })
    return pd.DataFrame(records).sort_values("final_score", ascending=False, ignore_index=True)


def main() -> None:
    """Execute the default demo pipeline."""
    parser = argparse.ArgumentParser(description="Rank candidates for a job description.")
    parser.add_argument("--jobs", default=RAW_DATA_DIR / "jobs.csv", help="Path to jobs CSV")
    parser.add_argument("--candidates", default=RAW_DATA_DIR / "candidates.csv", help="Path to candidates CSV")
    parser.add_argument("--job-id", default=None, help="Optional job_id to rank")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    jobs = load_jobs(args.jobs)
    candidates = load_candidates(args.candidates)
    selected = jobs if args.job_id is None else jobs[jobs["job_id"].astype(str) == args.job_id]
    if selected.empty:
        raise ValueError(f"No job found for job_id={args.job_id}")
    results = rank_candidates(selected.iloc[0], candidates, args.top_k)
    csv_path, json_path = write_outputs(results, OUTPUT_DIR)
    print(results.to_string(index=False))
    print(f"\nSaved: {csv_path}\nSaved: {json_path}")


if __name__ == "__main__":
    main()
