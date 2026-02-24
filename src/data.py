from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


DEFAULT_FILENAME = "NETFLIX_MOVIES_AND_TV_SHOWS_CLUSTERING.csv"


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    raw: Path
    processed: Path
    models: Path
    reports: Path

    @staticmethod
    def from_repo_root(repo_root: Path) -> "Paths":
        return Paths(
            repo_root=repo_root,
            raw=repo_root / "data" / "raw",
            processed=repo_root / "data" / "processed",
            models=repo_root / "models",
            reports=repo_root / "reports",
        )


def load_raw_csv(repo_root: Path, filename: str = DEFAULT_FILENAME) -> pd.DataFrame:
    """Load the raw Netflix titles dataset from data/raw."""
    paths = Paths.from_repo_root(repo_root)
    path = paths.raw / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put the CSV in data/raw/ or run scripts/make_dataset.py"
        )
    return pd.read_csv(path)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning that is safe for EDA + text clustering."""
    out = df.copy()

    # Normalize column names
    out.columns = [c.strip() for c in out.columns]

    # Common columns in this dataset
    for col in ["title", "type", "rating", "country", "listed_in", "description", "director", "cast"]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)

    # Dates / numerics (if present)
    if "release_year" in out.columns:
        out["release_year"] = pd.to_numeric(out["release_year"], errors="coerce")

    if "date_added" in out.columns:
        out["date_added"] = pd.to_datetime(out["date_added"], errors="coerce")

    return out
