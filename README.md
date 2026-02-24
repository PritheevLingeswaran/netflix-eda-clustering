# Netflix EDA + Clustering (GitHub Project)

This repo contains a complete end-to-end mini project:
- Exploratory Data Analysis (EDA) of the Netflix titles dataset
- Text-based clustering of titles using TF-IDF + SVD + MiniBatchKMeans
- Cluster interpretation (top keywords + sample titles)
- Optional similarity-based recommendations (Nearest Neighbors)

## Project structure
```
.
├─ notebooks/
│  └─ Netflix_EDA_Clustering.ipynb
├─ src/
│  ├─ data.py
│  ├─ features.py
│  ├─ clustering.py
│  └─ viz.py
├─ scripts/
│  ├─ make_dataset.py
│  └─ run_clustering.py
├─ configs/
│  └─ params.yaml
├─ data/
│  ├─ raw/           # put the CSV here (not tracked in git)
│  └─ processed/
├─ reports/
│  └─ figures/
├─ models/
├─ tests/
│  └─ test_smoke.py
├─ requirements.txt
└─ README.md
```

## Dataset
Place your dataset CSV inside:
`data/raw/NETFLIX_MOVIES_AND_TV_SHOWS_CLUSTERING.csv`

> Tip: Do **not** commit large datasets into GitHub unless your course requires it. Keep it local.

You can also run:
```bash
python scripts/make_dataset.py --input /path/to/your.csv
```
to copy it into the correct location.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run clustering from CLI
```bash
python scripts/run_clustering.py --k 8 --max_features 20000 --svd_components 50
```

Outputs:
- `models/kmeans_k{K}.joblib`
- `reports/cluster_summary_k{K}.csv`

## Notebook
Open and run:
`notebooks/Netflix_EDA_Clustering.ipynb`

## Notes
If your instructor asks “why K=8”, justify it using:
- elbow curve (inertia)
- silhouette (sampled)
- interpretability of clusters

## License
MIT
