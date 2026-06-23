"""Small, deterministic text-cleaning helpers."""
import re
from collections.abc import Iterable


def normalize_text(value: object) -> str:
    """Lowercase text and collapse whitespace without removing useful symbols."""
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.strip().lower())


def split_skills(value: object) -> list[str]:
    """Convert a comma- or semicolon-delimited skills field into unique skills."""
    parts = re.split(r"[,;|]", normalize_text(value))
    return list(dict.fromkeys(part.strip() for part in parts if part.strip()))


def join_text(parts: Iterable[object]) -> str:
    """Join non-empty values into normalized model input text."""
    return normalize_text(" ".join(str(part) for part in parts if str(part).strip()))
