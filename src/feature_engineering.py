"""Explainable matching features used by optional ranking helpers."""
from dataclasses import dataclass

from src.candidate_profiler import CandidateProfile
from src.jd_parser import JobProfile


@dataclass(frozen=True)
class MatchFeatures:
    """Feature values for one job-candidate pair."""

    matched_skills: list[str]
    missing_skills: list[str]
    skill_coverage: float
    experience_fit: float


def build_match_features(job: JobProfile, candidate: CandidateProfile) -> MatchFeatures:
    """Calculate required-skill coverage for a candidate and job pair."""
    candidate_skills = set(candidate.skills)
    matched = sorted(set(job.required_skills) & candidate_skills)
    missing = sorted(set(job.required_skills) - candidate_skills)
    coverage = len(matched) / len(job.required_skills) if job.required_skills else 1.0
    return MatchFeatures(matched, missing, coverage, 1.0)
