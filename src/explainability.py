"""Structured explanation cards for candidate-ranking decisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from src.jd_parser import JDIntelligence
from src.preprocessing import normalize_text, split_skills


@dataclass(frozen=True)
class ExplanationCard:
    """Evidence and gaps supporting a candidate's ranking outcome."""

    fit_summary: str
    matched_must_have_skills: list[str]
    matched_nice_to_have_skills: list[str]
    missing_required_skills: list[str]
    experience_reason: str
    project_reason: str
    activity_reason: str
    risk_or_gap: str
    confidence_level: str

    def to_dict(self) -> dict[str, object]:
        """Return a UI-friendly serializable representation of the card."""
        return asdict(self)


def skill_match_score(candidate_skills: object, required_skills: list[str]) -> float:
    """Return the fraction of required skills listed by the candidate."""
    if not required_skills:
        return 1.0
    return len(set(split_skills(candidate_skills)) & set(required_skills)) / len(required_skills)


def _matching_skills(candidate_skills: object, job_skills: list[str]) -> tuple[list[str], list[str]]:
    """Return display-form matched and missing job skills."""
    candidate_set = set(split_skills(candidate_skills))
    matched = [skill for skill in job_skills if normalize_text(skill) in candidate_set]
    missing = [skill for skill in job_skills if normalize_text(skill) not in candidate_set]
    return matched, missing


def _experience_reason(experience_years: object, intelligence: JDIntelligence) -> str:
    """Describe candidate experience relative to the parsed job range."""
    years = float(experience_years)
    minimum, maximum = intelligence.min_experience, intelligence.max_experience
    if minimum is None and maximum is None:
        return f"Has {years:g} years of experience; the role does not specify an experience range."
    if minimum is not None and years < minimum:
        return f"Has {years:g} years of experience, below the {minimum:g}-year minimum."
    if maximum is not None and years > maximum:
        return f"Has {years:g} years of experience, above the {maximum:g}-year maximum."
    if minimum is not None and maximum is not None:
        return f"Has {years:g} years of experience for the {minimum:g}-{maximum:g} year role range."
    return f"Has {years:g} years of experience, meeting the {minimum:g}+ year minimum."


def _project_reason(candidate: pd.Series, related_skills: list[str]) -> str:
    """Find job-related skills mentioned in the candidate's projects or summary."""
    project_text = normalize_text(f"{candidate.get('projects', '')} {candidate.get('summary', '')}")
    related = [skill for skill in related_skills if normalize_text(skill) in project_text]
    if related:
        return f"Project experience mentions {', '.join(related)}."
    projects = str(candidate.get("projects", "")).strip()
    if projects:
        return "Projects are listed, but no direct job-skill evidence was detected."
    return "No project details were provided."


def _activity_reason(activity_score: object) -> str:
    """Convert a 0–100 activity score into plain-language evidence."""
    score = pd.to_numeric(activity_score, errors="coerce")
    if pd.isna(score):
        return "No activity score was provided."
    numeric = float(score)
    if numeric >= 80:
        return f"Activity score of {numeric:g}/100 indicates strong recent engagement."
    if numeric >= 60:
        return f"Activity score of {numeric:g}/100 indicates consistent engagement."
    return f"Activity score of {numeric:g}/100 is a weaker supporting signal."


def _confidence_level(final_score: float, coverage_score: float) -> str:
    """Assign a confidence tier from aggregate score and required-skill coverage."""
    if final_score >= 0.75 and coverage_score >= 0.70:
        return "High"
    if final_score >= 0.45 or coverage_score >= 0.50:
        return "Medium"
    return "Low"


def build_explanation_card(
    candidate: pd.Series,
    intelligence: JDIntelligence,
    final_score: float,
    coverage_score: float,
) -> ExplanationCard:
    """Build an explanation card for one candidate and parsed job description."""
    matched_must, missing_required = _matching_skills(candidate["skills"], intelligence.must_have_skills)
    matched_nice, _ = _matching_skills(candidate["skills"], intelligence.nice_to_have_skills)
    experience_reason = _experience_reason(candidate["experience_years"], intelligence)
    project_reason = _project_reason(candidate, intelligence.must_have_skills + intelligence.nice_to_have_skills)
    activity_reason = _activity_reason(candidate.get("activity_score"))
    confidence = _confidence_level(final_score, coverage_score)
    fit_label = {"High": "strong", "Medium": "moderate", "Low": "limited"}[confidence]
    required_count = len(intelligence.must_have_skills)
    experience_summary = experience_reason.replace("Has ", "have ", 1)
    summary = (
        f"Candidate {candidate['candidate_id']} is a {fit_label} fit because they match "
        f"{len(matched_must)}/{required_count} required skills and {experience_summary} "
        f"{project_reason}"
    )
    risk_parts: list[str] = []
    if missing_required:
        risk_parts.append(f"Gap: {', '.join(missing_required)} not found.")
    if "below" in experience_reason:
        risk_parts.append("Experience is below the requested minimum.")
    if not risk_parts:
        risk_parts.append("No material required-skill or experience gap detected.")
    return ExplanationCard(
        fit_summary=summary,
        matched_must_have_skills=matched_must,
        matched_nice_to_have_skills=matched_nice,
        missing_required_skills=missing_required,
        experience_reason=experience_reason,
        project_reason=project_reason,
        activity_reason=activity_reason,
        risk_or_gap=" ".join(risk_parts),
        confidence_level=confidence,
    )


def build_explanation(
    candidate: pd.Series,
    intelligence: JDIntelligence,
    final_score: float,
    coverage_score: float,
) -> str:
    """Return the concise explanation text used in the ranked-output CSV."""
    return build_explanation_card(candidate, intelligence, final_score, coverage_score).fit_summary


