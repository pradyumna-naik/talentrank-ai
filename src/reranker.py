"""Combine retrieval and structured features into a simple ranking score."""
from dataclasses import dataclass

from src.config import EXPERIENCE_WEIGHT, RETRIEVAL_WEIGHT, SKILL_WEIGHT
from src.feature_engineering import MatchFeatures


@dataclass(frozen=True)
class RankingScore:
    """Score components retained for transparent output."""

    relevance_score: float
    skill_score: float
    experience_score: float
    final_score: float


def rerank(relevance_score: float, features: MatchFeatures) -> RankingScore:
    """Calculate a weighted score on a 0-100 scale."""
    final = 100 * (
        RETRIEVAL_WEIGHT * relevance_score
        + SKILL_WEIGHT * features.skill_coverage
        + EXPERIENCE_WEIGHT * features.experience_fit
    )
    return RankingScore(relevance_score, features.skill_coverage, features.experience_fit, final)
