from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline


@dataclass
class ClusterArtifacts:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    normalizer: Normalizer
    kmeans: MiniBatchKMeans
    X_tfidf: "scipy.sparse.spmatrix"  # type: ignore[name-defined]
    X_reduced: np.ndarray
    labels: np.ndarray
    metrics: Dict[str, float]


def fit_kmeans_text(
    texts: List[str],
    k: int = 8,
    max_features: int = 20000,
    svd_components: int = 50,
    min_df: int = 2,
    max_df: float = 0.95,
    random_state: int = 42,
    batch_size: int = 2048,
    max_iter: int = 200,
    silhouette_sample_size: int = 8000,
) -> ClusterArtifacts:
    """Vectorize text -> SVD -> KMeans, with quick evaluation metrics."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),
    )
    X_tfidf = vectorizer.fit_transform(texts)

    svd = TruncatedSVD(n_components=svd_components, random_state=random_state)
    normalizer = Normalizer(copy=False)
    reducer = make_pipeline(svd, normalizer)
    X_reduced = reducer.fit_transform(X_tfidf)

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init="auto",
    )
    labels = kmeans.fit_predict(X_reduced)

    metrics: Dict[str, float] = {"inertia": float(kmeans.inertia_)}

    # Silhouette can be expensive; sample for speed
    n = X_reduced.shape[0]
    if n >= 3:
        sample_n = min(silhouette_sample_size, n)
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_n, replace=False)
        try:
            metrics["silhouette_sampled"] = float(silhouette_score(X_reduced[idx], labels[idx]))
        except Exception:
            metrics["silhouette_sampled"] = float("nan")

    return ClusterArtifacts(
        vectorizer=vectorizer,
        svd=svd,
        normalizer=normalizer,
        kmeans=kmeans,
        X_tfidf=X_tfidf,
        X_reduced=X_reduced,
        labels=labels,
        metrics=metrics,
    )


def top_terms_per_cluster(art: ClusterArtifacts, top_n: int = 12) -> Dict[int, List[str]]:
    """Approximate top terms per cluster by centroid weights in TF-IDF space."""
    centroids_reduced = art.kmeans.cluster_centers_
    centroids_tfidf_approx = np.dot(centroids_reduced, art.svd.components_)
    terms = np.array(art.vectorizer.get_feature_names_out())

    out: Dict[int, List[str]] = {}
    for i, row in enumerate(centroids_tfidf_approx):
        top_idx = np.argsort(row)[::-1][:top_n]
        out[i] = terms[top_idx].tolist()
    return out


def make_cluster_summary(df: pd.DataFrame, labels: np.ndarray, keywords: Dict[int, List[str]], n_samples: int = 8) -> pd.DataFrame:
    """Create a human-readable cluster summary table."""
    out_rows = []
    tmp = df.copy()
    tmp["cluster"] = labels

    for c in sorted(tmp["cluster"].unique()):
        block = tmp[tmp["cluster"] == c]
        titles = (
            block["title"].dropna().astype(str).head(n_samples).tolist()
            if "title" in block.columns
            else block.index.astype(str).head(n_samples).tolist()
        )
        out_rows.append(
            {
                "cluster": int(c),
                "count": int(len(block)),
                "top_keywords": ", ".join(keywords.get(int(c), [])),
                "sample_titles": " | ".join(titles),
            }
        )
    return pd.DataFrame(out_rows).sort_values("count", ascending=False).reset_index(drop=True)
