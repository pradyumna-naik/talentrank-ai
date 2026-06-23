"""Retrieval helpers for TF-IDF, semantic embeddings, and CrossEncoder reranking."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def calculate_tfidf_scores(job_text: str, candidate_texts: Sequence[str]) -> np.ndarray:
    """Calculate lexical cosine similarity between a job and candidate profiles."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform([job_text, *candidate_texts])
    return cosine_similarity(matrix[0], matrix[1:]).ravel()


def load_sentence_model() -> Any:
    """Load the configured Sentence Transformers embedding model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, local_files_only=True)


def load_cross_encoder() -> Any:
    """Load the optional Sentence Transformers CrossEncoder reranker."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(CROSS_ENCODER_MODEL_NAME, local_files_only=True)


def _text_fingerprint(texts: Sequence[str]) -> str:
    """Build a stable cache key from ordered normalized profile text."""
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def load_or_create_candidate_embeddings(candidate_texts: Sequence[str], cache_path: str | Path, model: Any) -> np.ndarray:
    """Load cached candidate embeddings or generate and save normalized vectors."""
    cache = Path(cache_path)
    metadata_path = cache.with_suffix(".meta.json")
    fingerprint = _text_fingerprint(candidate_texts)
    if cache.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            embeddings = np.load(cache)
            if embeddings.shape[0] == len(candidate_texts):
                return embeddings
    embeddings = np.asarray(model.encode(list(candidate_texts), normalize_embeddings=True, show_progress_bar=False))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, embeddings)
    metadata_path.write_text(json.dumps({"fingerprint": fingerprint, "model": MODEL_NAME}), encoding="utf-8")
    return embeddings


def calculate_semantic_scores(job_text: str, candidate_embeddings: np.ndarray, model: Any) -> np.ndarray:
    """Calculate semantic cosine similarity from a job embedding and cached candidates."""
    job_embedding = np.asarray(model.encode([job_text], normalize_embeddings=True, show_progress_bar=False))
    return cosine_similarity(job_embedding, candidate_embeddings).ravel()


def calculate_cross_encoder_scores(job_text: str, candidate_texts: Sequence[str], model: Any) -> np.ndarray:
    """Score job-candidate pairs and map CrossEncoder logits to a 0–1 range."""
    pairs = [(job_text, candidate_text) for candidate_text in candidate_texts]
    raw_scores = np.asarray(model.predict(pairs, show_progress_bar=False), dtype=float)
    clipped_scores = np.clip(raw_scores, -30, 30)
    return 1.0 / (1.0 + np.exp(-clipped_scores))


def calculate_hybrid_score(semantic_score: float, tfidf_score: float, skill_match_score: float) -> float:
    """Combine semantic, lexical, and skill-match evidence for top-50 retrieval."""
    return 0.45 * semantic_score + 0.35 * tfidf_score + 0.20 * skill_match_score

