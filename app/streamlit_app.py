"""Streamlit dashboard for TalentRank AI."""
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from run_pipeline import rank_candidates  # noqa: E402
from src.config import RAW_DATA_DIR  # noqa: E402
from src.data_loader import load_candidates, load_jobs  # noqa: E402


st.set_page_config(page_title="TalentRank AI", page_icon="🏅", layout="wide")
st.title("TalentRank AI")
st.caption("Explainable candidate shortlisting beyond keyword counts.")

@st.cache_data
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load demo input data once per dashboard session."""
    return load_jobs(RAW_DATA_DIR / "jobs.csv"), load_candidates(RAW_DATA_DIR / "candidates.csv")


jobs, candidates = get_data()
labels = {str(row.job_id): f"{row.job_id} — {row.title}" for row in jobs.itertuples()}
selected_id = st.selectbox("Job description", options=list(labels), format_func=labels.get)
top_k = st.slider("Shortlist size", min_value=1, max_value=len(candidates), value=min(5, len(candidates)))
job = jobs[jobs["job_id"].astype(str) == selected_id].iloc[0]

with st.expander("Job description", expanded=True):
    st.write(job["description"])

if st.button("Rank candidates", type="primary"):
    results = rank_candidates(job, candidates, top_k)
    st.subheader("Ranked shortlist")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", results.to_csv(index=False), "ranked_candidates.csv", "text/csv")
