import pandas as pd
from src.data import basic_clean


def test_basic_clean_runs():
    df = pd.DataFrame({
        "title": ["A", None],
        "description": ["desc", None],
        "release_year": ["2020", "not_a_year"],
        "date_added": ["September 9, 2020", None],
    })
    out = basic_clean(df)
    assert "title" in out.columns
    assert out["title"].isna().sum() == 0
    assert out["release_year"].isna().sum() == 1
