from __future__ import annotations

import re
import pandas as pd


def _clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_text_feature(df: pd.DataFrame) -> pd.Series:
    """Combine multiple columns into one text field for vectorization."""
    cols = [c for c in ["title", "director", "cast", "country", "rating", "listed_in", "description"] if c in df.columns]
    if not cols:
        raise ValueError("No expected text columns found to build features.")
    text = df[cols].astype(str).agg(" ".join, axis=1)
    return text.map(_clean_text)
