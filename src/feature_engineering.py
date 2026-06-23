"""Explainable matching features used by the reranker."""
from dataclasses import dataclass

from src.candidate_profiler import CandidateProfile
from src.jd_parser import JobRequirements


@dataclass(frozen=True)
class MatchFeatures:
    """Feature values for one job-candidate pair."""

    matched_skills: list[str]
    missing_skills: list[str]
    skill_coverage: float
    experience_fit: float


def build_match_features(job: JobRequirements, candidate: CandidateProfile) -> MatchFeatures:
    """Calculate skill coverage and a capped experience-fit score."""
    candidate_skills = set(candidate.skills)
    matched = sorted(set(job.skills) & candidate_skills)
    missing = sorted(set(job.skills) - candidate_skills)
    coverage = len(matched) / len(job.skills) if job.skills else 1.0
    if job.min_experience_years <= 0:
        experience_fit = 1.0
    else:
        experience_fit = min(candidate.experience_years / job.min_experience_years, 1.0)
    return MatchFeatures(matched, missing, coverage, experience_fit)
