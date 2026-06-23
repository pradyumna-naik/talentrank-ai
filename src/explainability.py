"""Human-readable explanations for hybrid ranking decisions."""
from src.preprocessing import split_skills


def skill_match_score(candidate_skills: object, required_skills: list[str]) -> float:
    """Return the fraction of required skills listed by the candidate."""
    if not required_skills:
        return 1.0
    return len(set(split_skills(candidate_skills)) & set(required_skills)) / len(required_skills)


def build_explanation(
    candidate_skills: object,
    required_skills: list[str],
    semantic_score: float,
    tfidf_score: float,
    coverage_score: float,
) -> str:
    """Explain component scores plus matched and missing required skills."""
    skills = set(split_skills(candidate_skills))
    matched = sorted(skills & set(required_skills))
    missing = sorted(set(required_skills) - skills)
    matched_text = ", ".join(matched) if matched else "none"
    missing_text = ", ".join(missing) if missing else "none"
    return (
        f"Semantic fit: {semantic_score * 100:.2f}%. "
        f"Keyword fit: {tfidf_score * 100:.2f}%. "
        f"Skill coverage: {coverage_score * 100:.2f}%. "
        f"Matched skills: {matched_text}. Missing required skills: {missing_text}."
    )
