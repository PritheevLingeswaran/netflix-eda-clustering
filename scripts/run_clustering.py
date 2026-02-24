from __future__ import annotations

import argparse
from pathlib import Path
import joblib

from src.data import load_raw_csv, basic_clean
from src.features import build_text_feature
from src.clustering import fit_kmeans_text, top_terms_per_cluster, make_cluster_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Netflix text clustering end-to-end.")
    parser.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[1]), help="Repo root")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max_features", type=int, default=20000)
    parser.add_argument("--svd_components", type=int, default=50)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_df", type=float, default=0.95)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--silhouette_sample_size", type=int, default=8000)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    df = basic_clean(load_raw_csv(repo_root))
    texts = build_text_feature(df).tolist()

    art = fit_kmeans_text(
        texts=texts,
        k=args.k,
        max_features=args.max_features,
        svd_components=args.svd_components,
        min_df=args.min_df,
        max_df=args.max_df,
        random_state=args.random_state,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        silhouette_sample_size=args.silhouette_sample_size,
    )

    df["cluster"] = art.labels
    keywords = top_terms_per_cluster(art, top_n=12)
    summary = make_cluster_summary(df, art.labels, keywords, n_samples=10)

    # Save artifacts
    models_dir = repo_root / "models"
    reports_dir = repo_root / "reports"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    model_path = models_dir / f"kmeans_k{args.k}.joblib"
    joblib.dump(
        {
            "vectorizer": art.vectorizer,
            "svd": art.svd,
            "normalizer": art.normalizer,
            "kmeans": art.kmeans,
            "metrics": art.metrics,
            "top_terms": keywords,
        },
        model_path,
    )

    out_csv = reports_dir / f"cluster_summary_k{args.k}.csv"
    summary.to_csv(out_csv, index=False)

    print("Done.")
    print(f"Model saved: {model_path}")
    print(f"Summary saved: {out_csv}")
    print(f"Metrics: {art.metrics}")


if __name__ == "__main__":
    main()
