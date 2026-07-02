"""Hackathon demo dashboard for TalentRank AI."""
from io import BytesIO
from pathlib import Path
import sys
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from run_pipeline import rank_candidates  # noqa: E402
from src.config import RAW_DATA_DIR, USE_CROSS_ENCODER  # noqa: E402
from src.data_loader import load_candidates, load_jobs  # noqa: E402
from src.evaluation import build_ablation_table, evaluate_ranking  # noqa: E402
from src.fairness import DISCLAIMER, build_responsible_ai_report  # noqa: E402
from src.jd_parser import parse_jd  # noqa: E402
from src.output_writer import OUTPUT_COLUMNS  # noqa: E402


st.set_page_config(page_title="TalentRank AI", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
    .stApp { background: radial-gradient(circle at top left, #1b1030 0%, #0E1117 48%); }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1b2030, #12151f);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 14px;
        padding: 14px 16px 8px 16px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.35);
    }
    div[data-testid="stMetric"] label { color: #A5B4CF !important; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #171225 0%, #0E1117 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.2);
    }
    .brand-title {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #8B5CF6, #22D3EE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: -6px;
    }
    .hero-banner {
        background: linear-gradient(120deg, rgba(139,92,246,0.18), rgba(34,211,238,0.10));
        border: 1px solid rgba(139,92,246,0.35);
        border-radius: 18px;
        padding: 26px 30px;
        margin-bottom: 14px;
    }
    .fit-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .fit-strong { background: rgba(34,197,94,0.15); color: #4ADE80; border: 1px solid rgba(74,222,128,0.45); }
    .fit-moderate { background: rgba(250,204,21,0.15); color: #FACC15; border: 1px solid rgba(250,204,21,0.45); }
    .fit-limited { background: rgba(248,113,113,0.15); color: #F87171; border: 1px solid rgba(248,113,113,0.45); }
    .fit-neutral { background: rgba(148,163,184,0.15); color: #CBD5E1; border: 1px solid rgba(148,163,184,0.45); }
    .podium-card {
        background: linear-gradient(160deg, #1b2030, #12151f);
        border: 1px solid rgba(139,92,246,0.25);
        border-radius: 14px;
        padding: 16px 18px;
        height: 100%;
    }
    .podium-rank {
        font-size: 1.6rem;
        font-weight: 800;
        color: #8B5CF6;
    }
    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PAGE_ICONS = {
    "Home": "🏠",
    "JD Intelligence": "🧭",
    "Ranked Candidates": "🏆",
    "Candidate Explanation": "🔍",
    "Evaluation": "📊",
    "Responsible AI": "🛡️",
    "Export Output": "📤",
}


def fit_badge_html(text: str) -> str:
    """Map a fit-summary sentence to a color-coded badge."""
    lowered = text.lower()
    if "strong fit" in lowered:
        css_class, label = "fit-strong", "Strong fit"
    elif "moderate fit" in lowered:
        css_class, label = "fit-moderate", "Moderate fit"
    elif "limited fit" in lowered:
        css_class, label = "fit-limited", "Limited fit"
    else:
        css_class, label = "fit-neutral", "Reviewed"
    return f'<span class="fit-badge {css_class}">{label}</span>'


@st.cache_data
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load demo input data once per dashboard session."""
    return load_jobs(RAW_DATA_DIR / "jobs.csv"), load_candidates(RAW_DATA_DIR / "candidates.csv")


def title_block(title: str, subtitle: str) -> None:
    """Render a consistent title and subtitle."""
    st.title(title)
    st.caption(subtitle)


def active_results() -> pd.DataFrame | None:
    """Return session results only for the currently selected job."""
    results = st.session_state.get("ranking_results")
    if results is None or st.session_state.get("ranked_job_id") != st.session_state.get("active_job_id"):
        return None
    return results


def run_ranking(job: pd.Series, candidates: pd.DataFrame, use_cross_encoder: bool, top_k: int) -> None:
    """Rank candidates and retain results across dashboard pages."""
    started = time.perf_counter()
    results = rank_candidates(job, candidates, use_cross_encoder=use_cross_encoder).head(top_k)
    st.session_state["ranking_results"] = results
    st.session_state["ranking_latency_seconds"] = results.attrs.get("ranking_latency_seconds", time.perf_counter() - started)
    st.session_state["ranked_job_id"] = str(job["job_id"])


def require_results(results: pd.DataFrame | None) -> bool:
    """Show a helpful empty state when ranking has not run."""
    if results is None:
        st.info("Run ranking from the sidebar or Home page to populate this view.")
        return False
    return True


def score_breakdown(candidate: pd.Series) -> None:
    """Show metric cards and a score chart for one candidate."""
    fields = [
        ("Final", "final_score"), ("Semantic", "semantic_score"),
        ("Keyword", "tfidf_score"), ("Skills", "skill_match_score"),
        ("Experience", "experience_match_score"), ("Activity", "activity_score"),
        ("CrossEncoder", "cross_encoder_score"),
    ]
    cards = st.columns(4)
    for index, (label, field) in enumerate(fields):
        value = candidate[field]
        cards[index % 4].metric(label, "Not used" if pd.isna(value) else f"{value:.1f}")
    labels = [label for label, _ in fields]
    values = [candidate[field] if pd.notna(candidate[field]) else 0 for _, field in fields]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=values, colorscale=[[0, "#F87171"], [0.5, "#FACC15"], [1, "#4ADE80"]], cmin=0, cmax=100),
            text=[f"{value:.1f}" for value in values],
            textposition="outside",
        )
    )
    figure.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EDF3"),
        xaxis=dict(range=[0, 112], gridcolor="rgba(148,163,184,0.15)"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(figure, use_container_width=True)


def explanation_card(candidate: pd.Series) -> None:
    """Render the structured evidence card for a candidate."""
    card = candidate["explanation_card"]
    st.info(card["fit_summary"])
    columns = st.columns(2)
    columns[0].markdown("**Matched must-have skills**\n\n" + (", ".join(card["matched_must_have_skills"]) or "None"))
    columns[1].markdown("**Matched nice-to-have skills**\n\n" + (", ".join(card["matched_nice_to_have_skills"]) or "None"))
    columns[0].markdown("**Missing required skills**\n\n" + (", ".join(card["missing_required_skills"]) or "None"))
    columns[1].markdown(f"**Confidence level**\n\n{card['confidence_level']}")
    with st.expander("Evidence and risk details", expanded=True):
        st.markdown(f"**Experience:** {card['experience_reason']}")
        st.markdown(f"**Projects:** {card['project_reason']}")
        st.markdown(f"**Activity:** {card['activity_reason']}")
        st.warning(card["risk_or_gap"])


jobs, candidates = get_data()
responsible_report = build_responsible_ai_report(candidates.columns, jobs.columns)

with st.sidebar:
    st.markdown('<div class="brand-title">🧭 TalentRank AI</div>', unsafe_allow_html=True)
    st.caption("Explainable AI Candidate Discovery Engine")
    st.divider()
    page = st.radio("Demo pages", list(PAGE_ICONS), format_func=lambda label: f"{PAGE_ICONS[label]}  {label}")
    st.divider()
    labels = {str(row.job_id): f"{row.job_id} — {row.title}" for row in jobs.itertuples()}
    selected_job_id = st.selectbox("Active job", list(labels), format_func=labels.get)
    st.session_state["active_job_id"] = selected_job_id
    top_k = st.slider("Shortlist size", 1, len(candidates), min(5, len(candidates)))
    use_cross_encoder = st.checkbox("Use Cross-Encoder Reranking", value=USE_CROSS_ENCODER, help="More accurate pairwise scoring; slower on CPU.")
    run_requested = st.button("Run ranking", type="primary", use_container_width=True)

job = jobs[jobs["job_id"].astype(str) == selected_job_id].iloc[0]
intelligence = parse_jd(job)
if run_requested:
    with st.spinner("Ranking candidates..."):
        run_ranking(job, candidates, use_cross_encoder, top_k)
results = active_results()

if page == "Home":
    title_block("🏠 TalentRank AI", "Explainable AI Candidate Discovery Engine")
    st.markdown('<div class="hero-banner">', unsafe_allow_html=True)
    hero, status = st.columns([2, 1])
    with hero:
        st.markdown("### ✨ Explainable candidate discovery for fast, defensible shortlists")
        st.write("Combine job intelligence, semantic relevance, skills, experience, activity, and optional CrossEncoder evidence in one recruiter-ready workflow.")
        st.caption(DISCLAIMER)
    with status:
        st.metric("Active role", intelligence.role_title)
        st.metric("Available candidates", len(candidates))
    st.markdown("</div>", unsafe_allow_html=True)
    cards = st.columns(4)
    cards[0].metric("Jobs", len(jobs))
    cards[1].metric("Must-have skills", len(intelligence.must_have_skills))
    cards[2].metric("Seniority", intelligence.seniority_level.title())
    cards[3].metric("Mode", "CrossEncoder" if use_cross_encoder else "Hybrid")
    with st.expander("Demo flow", expanded=True):
        st.markdown("1. Explore **JD Intelligence**\n2. Run ranking\n3. Inspect **Ranked Candidates**\n4. Open **Candidate Explanation**\n5. Review **Evaluation**, **Responsible AI**, and **Export Output**")
    if results is not None:
        st.success(f"Shortlist ready: {len(results)} candidates ranked in {st.session_state.get('ranking_latency_seconds', 0.0):.2f}s.")
        st.dataframe(results[["rank", "candidate_id", "final_score", "semantic_score", "skill_match_score"]], use_container_width=True, hide_index=True)
    else:
        require_results(results)

elif page == "JD Intelligence":
    title_block("🧭 JD Intelligence", "Rule-based role, skill, responsibility, and experience extraction")
    with st.expander("Original job description", expanded=True):
        st.write(job["description"])
    cards = st.columns(5)
    cards[0].metric("Role", intelligence.role_title)
    cards[1].metric("Seniority", intelligence.seniority_level.title())
    cards[2].metric("Domain", intelligence.domain)
    cards[3].metric("Min experience", f"{intelligence.min_experience:g} years" if intelligence.min_experience is not None else "Not specified")
    cards[4].metric("Max experience", f"{intelligence.max_experience:g} years" if intelligence.max_experience is not None else "Not specified")
    columns = st.columns(3)
    columns[0].markdown("**Must-have skills**\n\n" + (", ".join(intelligence.must_have_skills) or "Not specified"))
    columns[1].markdown("**Nice-to-have skills**\n\n" + (", ".join(intelligence.nice_to_have_skills) or "Not specified"))
    columns[2].markdown("**Responsibilities**\n\n" + ("\n\n".join(f"- {item}" for item in intelligence.responsibilities) or "Not specified"))

elif page == "Ranked Candidates":
    title_block("🏆 Ranked Candidates", "Shortlist ordered by explainable evidence")
    if require_results(results):
        cards = st.columns(4)
        cards[0].metric("Shortlist", len(results))
        cards[1].metric("Top score", f"{results.iloc[0]['final_score']:.1f}")
        cards[2].metric("Average score", f"{results['final_score'].mean():.1f}")
        cards[3].metric("Latency", f"{st.session_state.get('ranking_latency_seconds', 0.0):.2f}s")

        st.subheader("Podium")
        podium = st.columns(min(3, len(results)))
        for column, (_, row) in zip(podium, results.head(3).iterrows()):
            badge = fit_badge_html(row["explanation_card"]["fit_summary"])
            column.markdown(
                f"""<div class="podium-card">
                    <div class="podium-rank">#{row['rank']}</div>
                    <div style="font-weight:600; margin-bottom:6px;">{row['candidate_id']}</div>
                    {badge}
                    <div style="margin-top:10px; font-size:1.4rem; font-weight:700;">{row['final_score']:.1f}</div>
                    <div style="color:#94A3B8; font-size:0.8rem;">final score</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.subheader("Full shortlist")
        columns = ["rank", "candidate_id", "final_score", "cross_encoder_score", "semantic_score", "tfidf_score", "skill_match_score", "experience_match_score", "activity_score"]
        st.dataframe(results[columns], use_container_width=True, hide_index=True)
        with st.expander("Score guide"):
            st.write("Final score combines semantic relevance, TF-IDF, skill coverage, experience, activity, and optional CrossEncoder reranking.")

elif page == "Candidate Explanation":
    title_block("🔍 Candidate Explanation", "Explanation card and score breakdown")
    if require_results(results):
        candidate_id = st.selectbox("Select candidate", results["candidate_id"].tolist(), format_func=lambda value: f"#{results.loc[results['candidate_id'] == value, 'rank'].iloc[0]} — {value}")
        candidate = results[results["candidate_id"] == candidate_id].iloc[0]
        cards = st.columns(3)
        cards[0].metric("Rank", f"#{candidate['rank']}")
        cards[1].metric("Final score", f"{candidate['final_score']:.1f}")
        cards[2].metric("Confidence", candidate["explanation_card"]["confidence_level"])
        st.markdown(fit_badge_html(candidate["explanation_card"]["fit_summary"]), unsafe_allow_html=True)
        explanation_card(candidate)
        st.subheader("Score breakdown")
        score_breakdown(candidate)

elif page == "Evaluation":
    title_block("📊 Evaluation", "Validate ranking quality with labels or top-10 diagnostics")
    if require_results(results):
        uploaded_labels = st.file_uploader("Optional relevance labels CSV", type="csv", help="Required columns: job_id, candidate_id, relevance")
        labels_frame = pd.read_csv(uploaded_labels) if uploaded_labels is not None else None
        latency = st.session_state.get("ranking_latency_seconds", 0.0)
        evaluation_type, metrics_data = evaluate_ranking(results, labels_frame, latency)
        st.subheader(evaluation_type)
        cards = st.columns(len(metrics_data))
        for column, (name, value) in zip(cards, metrics_data.items()):
            display = f"{value:.3f}" if "latency" in name else f"{value:.2%}" if evaluation_type == "Supervised metrics" else f"{value:.2f}"
            column.metric(name.replace("_", " "), display)
        st.subheader("Ablation table")
        st.dataframe(build_ablation_table(results, labels_frame, latency), use_container_width=True, hide_index=True)

elif page == "Responsible AI":
    title_block("🛡️ Responsible AI", "Safeguards, exclusions, and human oversight")
    st.info(DISCLAIMER)
    cards = st.columns(3)
    cards[0].metric("Sensitive fields detected", len(responsible_report["sensitive_columns_detected"]))
    cards[1].metric("Ranking features used", len(responsible_report["used_candidate_features"]))
    cards[2].metric("Sensitive ranking features", len(responsible_report["sensitive_columns_used_for_ranking"]))
    if responsible_report["sensitive_columns_detected"]:
        st.warning("Excluded from ranking: " + ", ".join(responsible_report["sensitive_columns_detected"]))
    with st.expander("Feature-use report", expanded=True):
        st.json(responsible_report)
    st.caption("Sensitive fields are never added to profile text, ranking scores, or shortlist outputs.")

else:
    title_block("📤 Export Output", "Download the shortlist in the required delivery format")
    if require_results(results):
        export_data = results.loc[:, OUTPUT_COLUMNS]
        cards = st.columns(3)
        cards[0].metric("Rows ready", len(export_data))
        cards[1].metric("Columns", len(export_data.columns))
        cards[2].metric("File", "ranked_output.csv")
        st.dataframe(export_data, use_container_width=True, hide_index=True)
        download_cols = st.columns(2)
        excel_buffer = BytesIO()
        export_data.to_excel(excel_buffer, index=False, sheet_name="ranked_candidates")
        download_cols[0].download_button("⬇ Download ranked_output.csv", export_data.to_csv(index=False), "ranked_output.csv", "text/csv", type="primary", use_container_width=True)
        download_cols[1].download_button("⬇ Download ranked_output.xlsx", excel_buffer.getvalue(), "ranked_output.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with st.expander("Export schema"):
            st.code(", ".join(OUTPUT_COLUMNS))
