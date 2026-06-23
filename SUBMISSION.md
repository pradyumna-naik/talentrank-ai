# TalentRank AI — Hackathon Submission

## One-line pitch

TalentRank AI is an explainable AI candidate discovery engine that turns a job description and candidate pool into a transparent, evidence-backed shortlist.

## Problem

Recruiters need to review large candidate pools quickly, but keyword filters are brittle and opaque. They can miss semantically relevant talent and give little justification for why a person is ranked.

## What we built

TalentRank AI combines:

- Rule-based JD intelligence
- TF-IDF and MiniLM semantic retrieval
- Skill, experience, and activity fit
- Optional CrossEncoder reranking
- Candidate explanation cards
- Responsible-AI exclusions and feature-use reporting
- Evaluation metrics and ablation analysis

## Demo flow

1. Choose a job on the **Home** page.
2. Review parsed role requirements in **JD Intelligence**.
3. Run ranking and inspect **Ranked Candidates**.
4. Open **Candidate Explanation** for supporting evidence and gaps.
5. Open **Evaluation** for diagnostics or upload relevance labels.
6. Review **Responsible AI** safeguards.
7. Download the five-column shortlist from **Export Output**.

## Technical highlights

- Hybrid retrieval retains the top 50 candidates before optional pairwise CrossEncoder reranking.
- Candidate embeddings are cached in `data/processed/candidate_embeddings.npy`.
- Sensitive fields are excluded from profile text and scoring.
- `used_features_report.json` provides an audit-friendly feature-use record.

## Running the project

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py
streamlit run app/streamlit_app.py
```

Optional high-fidelity reranking:

```powershell
python run_pipeline.py --cross-encoder
```

## Deliverables

- Runtime output: `data/outputs/ranked_output.csv`
- Feature-use audit: `data/outputs/used_features_report.json`
- Sample output: `data/outputs/sample_ranked_output.csv`
- Test suite: `python -m pytest -q tests`

## Responsible AI statement

This system is a recruiter decision-support tool. Final hiring decisions should be human-reviewed.
