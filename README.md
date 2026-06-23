# TalentRank AI

TalentRank AI produces an explainable shortlist for every job-candidate pairing. It combines semantic profile similarity, TF-IDF keyword relevance, and required-skill coverage.

## Requirements

Python 3.10+.

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py
streamlit run app/streamlit_app.py
```

The first semantic run downloads the `all-MiniLM-L6-v2` Sentence Transformers model. Candidate embeddings are then cached at `data/processed/candidate_embeddings.npy` and regenerated only if the normalized candidate profile text changes.

## Input schema

`data/raw/candidates.csv` must contain:

`candidate_id`, `name`, `skills`, `experience_years`, `education`, `projects`, `summary`, `activity_score`

`data/raw/jobs.csv` must contain:

`job_id`, `title`, `description`, `required_skills`, `preferred_skills`, `min_experience`, `max_experience`

## Hybrid ranking

The pipeline cleans text, builds profiles, then calculates:

```text
hybrid_score = 0.45 * semantic_score + 0.35 * tfidf_score + 0.20 * skill_match_score
```

- Semantic score: cosine similarity between MiniLM embeddings.
- TF-IDF score: lexical cosine similarity between profile texts.
- Skill-match score: proportion of required skills matched.

`data/outputs/ranked_output.csv` keeps the delivery schema: `job_id`, `candidate_id`, `rank`, `final_score`, and `explanation`. Explanations show semantic fit, keyword fit, skill coverage, matched skills, and missing required skills. The dashboard exposes the three component scores separately.

## Tests

```powershell
python -m pytest -q tests
```
