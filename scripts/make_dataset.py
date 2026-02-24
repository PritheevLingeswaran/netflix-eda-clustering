from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy the dataset CSV into data/raw/")
    parser.add_argument("--input", required=True, help="Path to your CSV file")
    parser.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[1]), help="Repo root")
    parser.add_argument("--name", default="NETFLIX_MOVIES_AND_TV_SHOWS_CLUSTERING.csv", help="Target filename")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    raw_dir = repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    dst = raw_dir / args.name
    shutil.copy(src, dst)
    print(f"Copied dataset to: {dst}")


if __name__ == "__main__":
    main()
