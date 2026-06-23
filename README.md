# TalentRank AI

An explainable candidate-ranking baseline for hackathon demos. It retrieves profiles relevant to a job description, scores skill coverage and experience fit, then produces a ranked shortlist with visible strengths and gaps.

## Requirements

Python 3.10+.

```bash
python -m pip install -r requirements.txt
python run_pipeline.py
streamlit run app/streamlit_app.py
```

The pipeline reads `data/raw/jobs.csv` and `data/raw/candidates.csv`, then writes CSV and JSON results to `data/outputs/`.

## Input schema

`jobs.csv`: `job_id`, `title`, `description`

`candidates.csv`: `candidate_id`, `name`, `skills`, `experience_years`, `profile`

## Ranking approach

The baseline uses TF-IDF retrieval so the demo works offline, followed by a weighted score:

- 50% profile relevance
- 35% required-skill coverage
- 15% experience fit

Each result includes matched skills, missing skills, component scores, and an explanation. `sentence-transformers` is included as a dependency for an easy later semantic-retrieval upgrade, without making the initial demo require a model download.

## Tests

```bash
pytest
```
