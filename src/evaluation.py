"""Basic ranking-output checks for demos and regression tests."""
import pandas as pd


def ranking_summary(results: pd.DataFrame) -> dict[str, float]:
    """Return compact, non-label-based summary metrics."""
    return {
        "candidate_count": float(len(results)),
        "mean_score": float(results["final_score"].mean()) if not results.empty else 0.0,
        "top_score": float(results["final_score"].max()) if not results.empty else 0.0,
    }
