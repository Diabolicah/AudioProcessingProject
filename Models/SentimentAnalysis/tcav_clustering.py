"""Clustering / PCA analysis of the TCAV concept vectors (book, section 5.1.5).

This is the logic the three analysis notebooks had copy-pasted between them. It
regenerates, from a per-sample TCAV CSV:

    * Tables 4, 12, 15  - clustering metrics, all concepts
    * Tables 5, 13, 16  - clustering metrics, "good" CAVs only (>= 85% accuracy)
    * Tables 6, 14, 17  - average magnitude per concept per label
    * Tables 7, 18, 19  - true-label distribution over centroid-per-label clusters
    * Figures 15-18, 25-29, 32 - PCA scatter plots and the silhouette curve

    python tcav_clustering.py --tcav ravdess_tcav.csv --out-dir results/ravdess

The metrics are the ones the book defines: ARI, silhouette, mean intra-cluster
distance, mean inter-cluster distance and their ratio (separation), each in both
Euclidean and cosine space.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score
from sklearn.preprocessing import normalize

# KMeans on Windows + MKL leaks memory unless the thread count is pinned; the
# notebooks set this by hand in a cell.
os.environ.setdefault("OMP_NUM_THREADS", "6")

DEFAULT_SEED = 42
GOOD_CAV_ACCURACY = 0.85
CLUSTER_RANGE = range(2, 11)


def load_tcav_csv(path: Path) -> pd.DataFrame:
    """Read a per-sample TCAV table, accepting both column spellings.

    Older CSVs were written with spaces ("true label", "concept name",
    "cav acc"); everything is written with underscores now.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.replace(" ", "_") for c in df.columns if " " in c})

    required = {"path", "true_label", "predicted_label", "concept_name", "magnitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def concept_vectors(df: pd.DataFrame, correct_only: bool = True) -> pd.DataFrame:
    """One row per recording, one column per concept, values = TCAV magnitude.

    :param correct_only: keep only recordings the classifier got right, as the
        notebooks do - a TCAV score explains the *predicted* class, so mixing in
        misclassified samples would attribute concepts to the wrong emotion.
    """
    if correct_only:
        df = df[df["true_label"] == df["predicted_label"]]
    return df.pivot_table(
        index=["path", "true_label"],
        columns="concept_name",
        values="magnitude",
    ).reset_index(drop=False)


def good_cav_names(df: pd.DataFrame, min_accuracy: float = GOOD_CAV_ACCURACY) -> list[str]:
    """Concepts whose linear CAV classifier reached `min_accuracy` (book: 85%)."""
    if "cav_acc" not in df.columns:
        raise ValueError("TCAV table has no 'cav_acc' column; regenerate it with tcav_demo.py")
    mean_acc = df.groupby("concept_name")["cav_acc"].mean()
    return sorted(mean_acc[mean_acc >= min_accuracy].index)


def split_xy(vector_df: pd.DataFrame, concepts: Optional[Sequence[str]] = None):
    """Feature matrix and label vector from a concept-vector table."""
    feature_df = vector_df.drop(columns=["path", "true_label"])
    if concepts is not None:
        feature_df = feature_df[list(concepts)]
    return feature_df.to_numpy(), vector_df["true_label"].to_numpy()


def intra_inter_scores(X, y, metric: str = "euclidean") -> dict:
    """Mean within-class distance, mean between-class distance, and their ratio."""
    dist_matrix = pairwise_distances(X, metric=metric)
    labels = np.unique(y)

    intra_dists = []
    inter_dists = []

    for label in labels:
        idx = np.where(y == label)[0]
        if len(idx) > 1:
            d = dist_matrix[np.ix_(idx, idx)]
            intra_dists.append(d[np.triu_indices_from(d, k=1)].mean())

    for i, lbl1 in enumerate(labels):
        for lbl2 in labels[i + 1:]:
            idx1 = np.where(y == lbl1)[0]
            idx2 = np.where(y == lbl2)[0]
            inter_dists.append(dist_matrix[np.ix_(idx1, idx2)].mean())

    return {
        "intra_mean": float(np.mean(intra_dists)),
        "inter_mean": float(np.mean(inter_dists)),
        "separation": float(np.mean(inter_dists) / np.mean(intra_dists)),
    }


def evaluate_clustering(X, y, n_clusters: int, random_state: int = DEFAULT_SEED) -> dict:
    """KMeans in Euclidean and in cosine space, scored against the true labels.

    Cosine clustering is KMeans on L2-normalised rows, which is the standard
    "spherical k-means" approximation: on the unit sphere, Euclidean distance is
    a monotone function of cosine distance.
    """
    clusters_euc = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(X)

    X_cosine = normalize(X, norm='l2', axis=1)   # put points on the unit sphere
    clusters_cos = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(X_cosine)

    return {
        "Euclidean clustering ARI": adjusted_rand_score(y, clusters_euc),
        "Cosine clustering ARI": adjusted_rand_score(y, clusters_cos),
        "Euclidean Silhouette": silhouette_score(X, clusters_euc),
        "Cosine Silhouette": silhouette_score(X_cosine, clusters_cos),
        "Euclidean": intra_inter_scores(X, clusters_euc, metric="euclidean"),
        "Cosine": intra_inter_scores(X, clusters_cos, metric="cosine"),
    }


def clustering_metrics_table(X, y, cluster_range: Sequence[int] = CLUSTER_RANGE,
                             random_state: int = DEFAULT_SEED) -> pd.DataFrame:
    """Tables 4/5, 12/13, 15/16: one row per cluster count."""
    rows = {}
    # KMeans needs n_clusters <= n_samples, and silhouette needs at least one
    # sample more than clusters. A run with few correctly-classified samples
    # would otherwise abort partway through the table.
    usable = [n for n in cluster_range if n < len(X)]
    skipped = [n for n in cluster_range if n not in usable]
    if skipped:
        print(f"[WARN] only {len(X)} samples; skipping cluster counts {skipped}")
    if not usable:
        raise ValueError(f"Need at least 3 samples to cluster, got {len(X)}")

    for n_clusters in usable:
        flat = {}
        for key, value in evaluate_clustering(X, y, n_clusters, random_state).items():
            if isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    flat[f"{key} {inner_key}"] = round(inner_value, 3)
            else:
                flat[key] = round(value, 3)
        rows[n_clusters] = flat

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "num clusters"
    return table.reset_index()


def average_magnitude_per_label(vector_df: pd.DataFrame) -> pd.DataFrame:
    """Tables 6, 14, 17: mean TCAV magnitude per concept, per true label."""
    return (vector_df.drop(columns=["path"])
            .groupby("true_label").mean()
            .round(3).reset_index())


def centroid_per_label_distribution(X, y) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tables 7, 18, 19: nearest-centroid assignment using one centroid per label.

    Rows are the true labels; columns are the label whose centroid each sample
    was closest to. A perfectly separable concept space would be the identity.
    """
    X = normalize(X, norm='l2', axis=1)
    labels = np.unique(y)

    centroids = np.vstack([X[y == label].mean(axis=0) for label in labels])
    centroids_df = pd.DataFrame(centroids, index=labels)

    eps = 1e-12
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
    Cn = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + eps)
    assigned = labels[np.argmax(Xn @ Cn.T, axis=1)]

    distribution = (pd.DataFrame({"true_label": y, "cluster": assigned})
                    .groupby("true_label")["cluster"]
                    .value_counts(normalize=True)
                    .unstack(fill_value=0)
                    .round(3))
    distribution.columns = [f"{col} clus" for col in distribution.columns]
    return distribution, centroids_df


def pca_scatter(X, y, out_path: Optional[Path] = None, title_suffix: str = ""):
    """Figures 15-17, 25-27, 29-31: 2D PCA of the concept vectors."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_

    labels = np.unique(y)
    colors = mpl.colormaps.get_cmap("tab20")(np.linspace(0, 1, len(labels)))

    fig = plt.figure(figsize=(8, 6))
    for i, lbl in enumerate(labels):
        mask = (y == lbl)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=20, alpha=0.8,
                    color=colors[i], label=str(lbl), edgecolors="none")

    plt.title(f"PCA 2D: PC1 {evr[0]*100:.1f}% | PC2 {evr[1]*100:.1f}% "
              f"(Total {evr.sum()*100:.1f}%){title_suffix}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="True label", markerscale=1.5, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    return X_pca, evr


def silhouette_curve(metrics_table: pd.DataFrame, out_path: Optional[Path] = None):
    """Figures 18, 28, 32: cosine silhouette against the number of clusters."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 4))
    plt.plot(metrics_table["num clusters"], metrics_table["Cosine Silhouette"], marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Cosine silhouette")
    plt.title("Cosine Silhouette score per number of clusters")
    plt.grid(True)
    plt.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def run_full_analysis(tcav_csv: Path, out_dir: Path,
                      min_cav_accuracy: float = GOOD_CAV_ACCURACY,
                      random_state: int = DEFAULT_SEED) -> dict[str, pd.DataFrame]:
    """Regenerate every TCAV table and figure for one dataset."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_tcav_csv(tcav_csv)
    vectors = concept_vectors(raw, correct_only=True)
    X_all, y = split_xy(vectors)

    outputs: dict[str, pd.DataFrame] = {}

    outputs["clustering_all_concepts"] = clustering_metrics_table(X_all, y, random_state=random_state)
    outputs["average_magnitude_per_label"] = average_magnitude_per_label(vectors)

    distribution, centroids = centroid_per_label_distribution(X_all, y)
    outputs["centroid_label_distribution"] = distribution.reset_index()
    outputs["label_centroids"] = centroids.reset_index(names="true_label")

    if "cav_acc" in raw.columns:
        good = good_cav_names(raw, min_cav_accuracy)
        print(f"{len(good)}/{vectors.shape[1] - 2} concepts pass the {min_cav_accuracy:.0%} CAV-accuracy bar")
        if len(good) >= 2:
            X_good, _ = split_xy(vectors, concepts=good)
            outputs["clustering_good_cavs"] = clustering_metrics_table(X_good, y, random_state=random_state)
            pca_scatter(X_good, y, out_dir / "pca_good_cavs.png", " - good CAVs")
        else:
            # With captum's default classifier the reported accuracy is the test
            # split's class balance, so this legitimately comes out empty. Skip
            # rather than abort the rest of the analysis.
            print("[WARN] fewer than 2 concepts pass the bar; skipping the good-CAV tables")

    pca_scatter(X_all, y, out_dir / "pca_all_concepts.png", " - all concepts")

    without_hn = ~np.isin(y, ["happy", "neutral"])
    if without_hn.any():
        pca_scatter(X_all[without_hn], y[without_hn],
                    out_dir / "pca_no_happy_neutral.png", " - happy/neutral removed")

    silhouette_curve(outputs["clustering_all_concepts"], out_dir / "cosine_silhouette.png")

    for name, table in outputs.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"wrote {out_dir / f'{name}.csv'}")

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tcav", type=Path, required=True, help="per-sample TCAV CSV")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-cav-accuracy", type=float, default=GOOD_CAV_ACCURACY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    run_full_analysis(args.tcav, args.out_dir, args.min_cav_accuracy, args.seed)
