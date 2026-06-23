"""Extract practical, explainable job requirements from job-description text."""
import re
from dataclasses import dataclass

from src.preprocessing import normalize_text


KNOWN_SKILLS = {
    "python", "sql", "pandas", "numpy", "scikit-learn", "machine learning",
    "data analysis", "data visualization", "tableau", "power bi", "aws", "azure",
    "docker", "kubernetes", "java", "javascript", "react", "node.js", "git",
    "nlp", "streamlit", "pytorch", "tensorflow", "excel",
}


@dataclass(frozen=True)
class JobRequirements:
    """Requirements derived from a job description."""

    title: str
    skills: list[str]
    min_experience_years: float
    text: str


def parse_job_description(title: str, description: str) -> JobRequirements:
    """Identify mentioned known skills and a minimum experience value if present."""
    text = normalize_text(description)
    skills = sorted(skill for skill in KNOWN_SKILLS if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", text))
    experience_matches = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", text)
    min_experience = max((float(match) for match in experience_matches), default=0.0)
    return JobRequirements(title=title, skills=skills, min_experience_years=min_experience, text=text)
