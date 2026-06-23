import pandas as pd
import pytest

from src.data_loader import load_candidates


def test_load_candidates_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "bad_candidates.csv"
    pd.DataFrame({"candidate_id": ["C1"], "name": ["Ada"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_candidates(path)
