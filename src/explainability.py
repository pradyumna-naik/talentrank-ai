"""Human-readable explanations for each ranking decision."""
from src.candidate_profiler import CandidateProfile
from src.feature_engineering import MatchFeatures


def build_explanation(candidate: CandidateProfile, features: MatchFeatures) -> str:
    """Summarize the strongest evidence and the most relevant gaps."""
    matched = ", ".join(features.matched_skills) or "no explicitly detected job skills"
    explanation = f"Matches: {matched}. Experience: {candidate.experience_years:g} years."
    if features.missing_skills:
        explanation += f" Gaps: {', '.join(features.missing_skills)}."
    return explanation
