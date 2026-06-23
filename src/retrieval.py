"""Candidate retrieval using semantic embeddings with an offline TF-IDF fallback."""
from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def retrieve_candidates(job_text: str, candidate_texts: Sequence[str], top_k: int) -> list[tuple[int, float]]:
    """Return candidate indices and relevance scores, highest relevance first.

    TF-IDF keeps the first version runnable offline. The sentence-transformers
    dependency is retained for a future drop-in semantic model upgrade.
    """
    if not candidate_texts:
        return []
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform([job_text, *candidate_texts])
    scores = cosine_similarity(matrix[0], matrix[1:]).ravel()
    ordered = np.argsort(-scores)[:top_k]
    return [(int(index), float(scores[index])) for index in ordered]
