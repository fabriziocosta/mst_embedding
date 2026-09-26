"""Compare exact and hierarchical IMSTE graph construction on sklearn digits.

Run with: python notebooks/compare_exact_hierarchical.py
"""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

from mst_embedding import IteratedMinimumSpanningTreeEmbedder
from mst_embedding._estimator import _hierarchical_imst_edges, _iterated_mst_edges

MINIBATCH_SIZE = 256


def main():
    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data[:600])
    common = dict(n_msts=3, n_epochs=150, batch_size=1024, random_state=42)

    start = perf_counter()
    exact_distances = cdist(X, X)
    exact_edges, _ = _iterated_mst_edges(exact_distances, common["n_msts"])
    exact_graph_seconds = perf_counter() - start
    exact_start = perf_counter()
    exact_model = IteratedMinimumSpanningTreeEmbedder(**common).fit(X)
    exact_total_seconds = perf_counter() - exact_start

    rows = []
    first_hierarchical = None
    for n_clusters in (10, 20, 40, 80):
        start = perf_counter()
        clustering = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=MINIBATCH_SIZE,
            random_state=common["random_state"],
        ).fit(X)
        _edges, _weights = _hierarchical_imst_edges(
            X,
            clustering.labels_,
            clustering.cluster_centers_,
            n_clusters,
            n_coarse_msts=2,
            n_local_msts=3,
            rank_weight_exponent=1.0,
            n_jobs=-1,
        )
        graph_seconds = perf_counter() - start
        model_start = perf_counter()
        model = IteratedMinimumSpanningTreeEmbedder(
            **common,
            graph_mode="hierarchical",
            n_clusters=n_clusters,
            n_coarse_msts=2,
            n_local_msts=3,
            n_jobs=-1,
            minibatch_size=MINIBATCH_SIZE,
        ).fit(X)
        total_seconds = perf_counter() - model_start
        exact_set = {tuple(sorted(map(int, edge))) for edge in exact_edges}
        hierarchical_set = {tuple(sorted(map(int, edge))) for edge in model.graph_edges_}
        recovered = len(hierarchical_set & exact_set) / max(len(exact_set), 1)
        rows.append((n_clusters, graph_seconds, total_seconds, len(model.graph_edges_), recovered))
        if first_hierarchical is None:
            first_hierarchical = model

    print(f"exact graph={exact_graph_seconds:.3f}s total={exact_total_seconds:.3f}s edges={len(exact_edges)}")
    print("clusters graph_s total_s unique_edges exact_edge_recall")
    for row in rows:
        print(f"{row[0]:8d} {row[1]:7.3f} {row[2]:7.3f} {row[3]:12d} {row[4]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, model, title in (
        (axes[0], exact_model, "Exact IMSTE"),
        (axes[1], first_hierarchical, "Hierarchical IMSTE (10 clusters)"),
    ):
        points = ax.scatter(
            model.embedding_[:, 0], model.embedding_[:, 1],
            c=digits.target[: len(X)], s=8, cmap="tab10", alpha=0.8,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(points, ax=axes, label="Digit")
    plt.show()


if __name__ == "__main__":
    main()
