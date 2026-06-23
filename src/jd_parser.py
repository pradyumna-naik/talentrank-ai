"""Rule-based job-description intelligence extraction."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from src.preprocessing import join_text, normalize_text, split_skills


class SkillTaxonomy:
    """Common technical skills and regex aliases used for JD extraction."""

    SKILLS: ClassVar[dict[str, tuple[str, ...]]] = {
        "Python": (r"python",),
        "Java": (r"java",),
        "SQL": (r"sql",),
        "Machine Learning": (r"machine[ -]?learning", r"\bml\b"),
        "Deep Learning": (r"deep[ -]?learning", r"\bdl\b"),
        "NLP": (r"natural language processing", r"\bnlp\b"),
        "React": (r"react(?:\.js)?",),
        "Node.js": (r"node(?:\.js|\s?js)?",),
        "FastAPI": (r"fastapi",),
        "Flask": (r"flask",),
        "Django": (r"django",),
        "Docker": (r"docker",),
        "Kubernetes": (r"kubernetes", r"\bk8s\b"),
        "AWS": (r"aws", r"amazon web services"),
        "Azure": (r"azure",),
        "GCP": (r"gcp", r"google cloud platform"),
        "Git": (r"git",),
        "Excel": (r"excel",),
        "Power BI": (r"power\s?bi",),
        "Tableau": (r"tableau",),
    }

    @classmethod
    def extract(cls, text: object) -> list[str]:
        """Return canonical skills found in arbitrary job-description text."""
        normalized = normalize_text(text)
        found: list[str] = []
        for skill, aliases in cls.SKILLS.items():
            if any(re.search(rf"(?<!\w){alias}(?!\w)", normalized) for alias in aliases):
                found.append(skill)
        return found

    @classmethod
    def canonicalize(cls, skill: object) -> str:
        """Return a taxonomy display name when a supplied skill matches one."""
        normalized = normalize_text(skill)
        for canonical, aliases in cls.SKILLS.items():
            if any(re.fullmatch(alias, normalized) for alias in aliases):
                return canonical
        return str(skill).strip()


@dataclass(frozen=True)
class JDIntelligence:
    """Rule-based structured intelligence extracted from a job description."""

    role_title: str
    seniority_level: str
    must_have_skills: list[str]
    nice_to_have_skills: list[str]
    responsibilities: list[str]
    domain: str
    min_experience: float | None
    max_experience: float | None


@dataclass(frozen=True)
class JobProfile:
    """Ranking-oriented job profile with attached JD intelligence."""

    job_id: str
    required_skills: list[str]
    profile_text: str
    intelligence: JDIntelligence


MUST_HAVE_CUES = ("must have", "required", "requirement", "mandatory", "essential", "need to have")
NICE_TO_HAVE_CUES = ("nice to have", "preferred", "bonus", "plus", "good to have", "desired")
RESPONSIBILITY_VERBS = (
    "build", "develop", "design", "implement", "create", "maintain", "lead",
    "collaborate", "analyze", "deploy", "optimize", "manage", "work with",
)


def _unique(values: list[str]) -> list[str]:
    """Deduplicate values without changing their first-seen display form."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = normalize_text(value)
        if key and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def _structured_skills(value: object) -> list[str]:
    """Canonicalize known skills while retaining additional structured skills."""
    return _unique([SkillTaxonomy.canonicalize(skill) for skill in split_skills(value)])


def _sentences(description: object) -> list[str]:
    """Split a description into simple sentence or bullet-sized units."""
    text = str(description or "").strip()
    return [part.strip(" -•\t") for part in re.split(r"(?:\r?\n|(?<=[.!?;])\s+)", text) if part.strip(" -•\t")]


def _experience_from_text(text: str) -> tuple[float | None, float | None]:
    """Extract a numeric experience range or minimum from free text."""
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", text)
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))
    minimum_match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", text)
    if minimum_match:
        return float(minimum_match.group(1)), None
    return None, None


def _provided_experience(row: pd.Series, column: str) -> float | None:
    """Read an optional numeric experience field without treating NaN as a value."""
    value = row.get(column)
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return float(value)


def _seniority_level(title: str, text: str, min_experience: float | None) -> str:
    """Classify one of the required seniority levels from explicit cues first."""
    source = normalize_text(f"{title} {text}")
    if re.search(r"\bintern(ship)?\b", source):
        return "intern"
    if re.search(r"\bfresher\b|\bgraduate\b|\bentry[ -]?level\b", source):
        return "fresher"
    if re.search(r"\bjunior\b|\bassociate\b", source):
        return "junior"
    if re.search(r"\bsenior\b|\blead\b|\bprincipal\b|\bstaff\b", source):
        return "senior"
    if re.search(r"\bmid(?:[- ]level)?\b|\bintermediate\b", source):
        return "mid"
    if min_experience is not None:
        if min_experience <= 0:
            return "fresher"
        if min_experience <= 2:
            return "junior"
        if min_experience >= 5:
            return "senior"
    return "mid"


def _domain(text: str) -> str:
    """Infer a single broad domain from skill and responsibility evidence."""
    normalized = normalize_text(text)
    domains = (
        ("Machine Learning / AI", ("machine learning", "deep learning", "nlp", "model")),
        ("Data Analytics", ("analytics", "tableau", "power bi", "excel", "dashboard")),
        ("Cloud / DevOps", ("docker", "kubernetes", "aws", "azure", "gcp", "devops")),
        ("Backend Engineering", ("fastapi", "flask", "django", "api", "backend")),
        ("Frontend Engineering", ("react", "frontend", "javascript", "node.js")),
    )
    for domain, keywords in domains:
        if any(keyword in normalized for keyword in keywords):
            return domain
    return "Software Engineering"


def parse_jd(row: pd.Series) -> JDIntelligence:
    """Extract explainable JD intelligence using rules and the skill taxonomy."""
    title = str(row.get("title", "")).strip()
    description = str(row.get("description", "") or "")
    text = normalize_text(f"{title} {description}")
    sentences = _sentences(description)

    must_have = _structured_skills(row.get("required_skills", ""))
    nice_to_have = _structured_skills(row.get("preferred_skills", ""))
    for sentence in sentences:
        skills = SkillTaxonomy.extract(sentence)
        normalized_sentence = normalize_text(sentence)
        if any(cue in normalized_sentence for cue in NICE_TO_HAVE_CUES):
            nice_to_have.extend(skills)
        elif any(cue in normalized_sentence for cue in MUST_HAVE_CUES):
            must_have.extend(skills)
        else:
            must_have.extend(skill for skill in skills if skill not in nice_to_have)

    must_have = _unique(must_have)
    nice_to_have = [skill for skill in _unique(nice_to_have) if normalize_text(skill) not in {normalize_text(item) for item in must_have}]
    responsibilities = [
        sentence for sentence in sentences
        if any(re.search(rf"\b{re.escape(verb)}\b", normalize_text(sentence)) for verb in RESPONSIBILITY_VERBS)
    ]
    if not responsibilities and sentences:
        responsibilities = [sentences[0]]

    parsed_minimum, parsed_maximum = _experience_from_text(text)
    min_experience = _provided_experience(row, "min_experience")
    max_experience = _provided_experience(row, "max_experience")
    min_experience = min_experience if min_experience is not None else parsed_minimum
    max_experience = max_experience if max_experience is not None else parsed_maximum
    return JDIntelligence(
        role_title=title,
        seniority_level=_seniority_level(title, description, min_experience),
        must_have_skills=must_have,
        nice_to_have_skills=nice_to_have,
        responsibilities=responsibilities,
        domain=_domain(text),
        min_experience=min_experience,
        max_experience=max_experience,
    )


def parse_job(row: pd.Series) -> JobProfile:
    """Create the ranking profile while retaining its parsed JD intelligence."""
    intelligence = parse_jd(row)
    return JobProfile(
        job_id=str(row["job_id"]),
        required_skills=[normalize_text(skill) for skill in intelligence.must_have_skills],
        profile_text=join_text([
            row["title"], row["description"], row.get("required_skills", ""), row.get("preferred_skills", ""),
        ]),
        intelligence=intelligence,
    )
