"""MNIST quality and runtime benchmark helpers for the companion notebook."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.manifold import trustworthiness
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from mst_embedding import IteratedMinimumSpanningTreeEmbedder


def run_trials(
    configurations: Iterable[Mapping[str, Any]],
    run_one: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every configuration, collecting failures without aborting the sweep."""
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for configuration in configurations:
        try:
            results.append(run_one(configuration))
        except Exception as exc:
            failures.append(
                {
                    **configuration,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return results, failures


def _load_mnist(max_samples: int | None, random_state: int):
    dataset = fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="auto", cache=True
    )
    X = np.asarray(dataset.data, dtype=np.float64)
    y = np.unique(dataset.target, return_inverse=True)[1]
    if max_samples is not None and max_samples < len(X):
        indices, _ = train_test_split(
            np.arange(len(X)),
            train_size=max_samples,
            random_state=random_state,
            stratify=y,
        )
        X, y = X[indices], y[indices]
    return StandardScaler().fit_transform(X), y


def _validate_fitted_model(model, n_samples: int):
    embedding = np.asarray(model.embedding_)
    edges = np.asarray(model.graph_edges_)
    weights = np.asarray(model.graph_weights_)
    if embedding.ndim != 2 or embedding.shape[0] != n_samples:
        raise ValueError(f"Unexpected embedding shape: {embedding.shape}")
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains non-finite coordinates.")
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) != len(weights):
        raise ValueError("Graph edge and weight arrays have inconsistent shapes.")
    if not np.isfinite(weights).all():
        raise ValueError("Graph weights contain non-finite values.")
    if edges.size:
        if np.any(edges < 0) or np.any(edges >= n_samples):
            raise ValueError("Graph contains an out-of-range sample index.")
        if np.any(edges[:, 0] == edges[:, 1]):
            raise ValueError("Graph contains a self-edge.")
    canonical_edges = [tuple(sorted(map(int, edge))) for edge in edges]
    if len(canonical_edges) != len(set(canonical_edges)):
        raise ValueError("Graph contains duplicate undirected edges.")

    timing_names = (
        "graph_construction_time_",
        "embedding_optimization_time_",
        "fit_time_",
    )
    timings = {name: float(getattr(model, name)) for name in timing_names}
    if not np.isfinite(list(timings.values())).all() or any(
        value < 0 for value in timings.values()
    ):
        raise ValueError("Fit timing diagnostics must be finite and non-negative.")
    if any(timings[name] > timings["fit_time_"] for name in timing_names[:2]):
        raise ValueError("A stage timing exceeds total fit time.")
    if sum(timings[name] for name in timing_names[:2]) > timings["fit_time_"] + 1e-9:
        raise ValueError("Stage timings exceed total fit time.")
    return set(canonical_edges), timings


def _summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "unique_edges",
        "exact_edge_precision",
        "exact_edge_recall",
        "trustworthiness",
        "5NN_accuracy",
        "delta_trustworthiness",
        "delta_5NN_accuracy",
        "delta_fit_time_seconds",
        "graph_construction_seconds",
        "embedding_optimization_seconds",
        "fit_time_seconds",
        "wall_time_seconds",
    ]
    rows = []
    if not runs.empty:
        for (mode, n_clusters), group in runs.groupby(
            ["mode", "n_clusters"], dropna=False, sort=False
        ):
            row = {
                "mode": mode,
                "n_clusters": n_clusters,
                "successful_runs": len(group),
            }
            for metric in metric_columns:
                values = group[metric].dropna()
                if not values.empty:
                    row[f"{metric}_median"] = values.median()
                    row[f"{metric}_q25"] = values.quantile(0.25)
                    row[f"{metric}_q75"] = values.quantile(0.75)
            rows.append(row)
    return pd.DataFrame(rows)


def run_mnist_parameter_grid(
    *,
    coarse_mst_values: Iterable[int] = (1, 2, 4),
    local_mst_values: Iterable[int] = (1, 2, 4),
    max_samples: int = 8_000,
    data_seed: int = 42,
    model_seed: int = 42,
    n_msts: int = 8,
    exact_mst_values: Iterable[int] = (2, 3, 4, 8),
    n_epochs: int = 200,
    n_clusters: int = 100,
    minibatch_size: int = 256,
    n_jobs: int = -1,
    device: str = "cpu",
    minkowski_p: float = 2.0,
) -> tuple[
    pd.DataFrame,
    dict[str, float | int],
    dict[tuple[int, int], np.ndarray],
    np.ndarray,
    dict[int, np.ndarray],
]:
    """Compare coarse/local MST settings on a shared stratified MNIST sample."""
    coarse_values = tuple(int(value) for value in coarse_mst_values)
    local_values = tuple(int(value) for value in local_mst_values)
    exact_values = tuple(int(value) for value in exact_mst_values)
    if not coarse_values or not local_values or not exact_values:
        raise ValueError("Provide at least one exact, coarse, and local MST value.")
    if min(coarse_values) < 1 or min(local_values) < 1 or min(exact_values) < 1:
        raise ValueError("Exact, coarse, and local MST values must be positive integers.")
    if max_samples < 10:
        raise ValueError("max_samples must be at least 10.")

    X, y = _load_mnist(max_samples, data_seed)
    print(
        f"Fixed MNIST grid data: {len(X):,} samples × {X.shape[1]} features; "
        f"coarse values={coarse_values}; local values={local_values}",
        flush=True,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=data_seed)

    def score_embedding(embedding):
        return {
            "trustworthiness": float(
                trustworthiness(X, embedding, n_neighbors=10)
            ),
            "5NN_accuracy": float(
                cross_val_score(
                    KNeighborsClassifier(n_neighbors=5),
                    embedding,
                    y,
                    cv=cv,
                    n_jobs=1,
                ).mean()
            ),
        }

    # Exclude one-time torch and process-pool startup from measured fits.
    warmup_common = {
        "n_epochs": 2,
        "batch_size": 4096,
        "random_state": model_seed,
        "device": device,
        "minkowski_p": minkowski_p,
    }
    IteratedMinimumSpanningTreeEmbedder(
        graph_mode="exact", n_msts=n_msts, **warmup_common
    ).fit(X)
    IteratedMinimumSpanningTreeEmbedder(
        graph_mode="hierarchical",
        n_clusters=n_clusters,
        n_coarse_msts=coarse_values[0],
        n_local_msts=local_values[0],
        n_jobs=n_jobs,
        minibatch_size=minibatch_size,
        **warmup_common,
    ).fit(X)

    exact = IteratedMinimumSpanningTreeEmbedder(
        graph_mode="exact",
        n_msts=n_msts,
        n_epochs=n_epochs,
        batch_size=4096,
        random_state=model_seed,
        device=device,
        minkowski_p=minkowski_p,
    )
    exact_wall_started = time.perf_counter()
    exact.fit(X)
    exact_wall_time = time.perf_counter() - exact_wall_started
    exact_edges, exact_timings = _validate_fitted_model(exact, len(X))
    exact_scores = score_embedding(exact.embedding_)
    exact_embedding = exact.embedding_.copy()
    exact_embeddings: dict[int, np.ndarray] = {}
    for exact_n_msts in exact_values:
        if exact_n_msts == n_msts:
            exact_embeddings[exact_n_msts] = exact_embedding
            continue
        exact_model = IteratedMinimumSpanningTreeEmbedder(
            graph_mode="exact",
            n_msts=exact_n_msts,
            n_epochs=n_epochs,
            batch_size=4096,
            random_state=model_seed,
            device=device,
            minkowski_p=minkowski_p,
        )
        exact_model.fit(X)
        exact_embeddings[exact_n_msts] = exact_model.embedding_.copy()
    embeddings: dict[tuple[int, int], np.ndarray] = {}
    exact_summary: dict[str, float | int] = {
        "minkowski_p": minkowski_p,
        "unique_edges": len(exact_edges),
        **exact_scores,
        "graph_construction_seconds": exact_timings["graph_construction_time_"],
        "embedding_optimization_seconds": exact_timings[
            "embedding_optimization_time_"
        ],
        "fit_time_seconds": exact_timings["fit_time_"],
        "wall_time_seconds": exact_wall_time,
    }

    rows = []
    for coarse_msts in coarse_values:
        for local_msts in local_values:
            estimator = IteratedMinimumSpanningTreeEmbedder(
                graph_mode="hierarchical",
                n_clusters=n_clusters,
                n_coarse_msts=coarse_msts,
                n_local_msts=local_msts,
                n_jobs=n_jobs,
                minibatch_size=minibatch_size,
                n_epochs=n_epochs,
                batch_size=4096,
                random_state=model_seed,
                device=device,
                minkowski_p=minkowski_p,
            )
            wall_started = time.perf_counter()
            estimator.fit(X)
            wall_time = time.perf_counter() - wall_started
            edges, timings = _validate_fitted_model(estimator, len(X))
            scores = score_embedding(estimator.embedding_)
            embeddings[(coarse_msts, local_msts)] = estimator.embedding_.copy()
            intersection = len(edges & exact_edges)
            rows.append(
                {
                    "minkowski_p": minkowski_p,
                    "n_coarse_msts": coarse_msts,
                    "n_local_msts": local_msts,
                    "effective_coarse_msts": estimator.n_coarse_msts_,
                    "unique_edges": len(edges),
                    "exact_edge_precision": intersection / max(len(edges), 1),
                    "exact_edge_recall": intersection / max(len(exact_edges), 1),
                    **scores,
                    "delta_trustworthiness": (
                        scores["trustworthiness"] - exact_scores["trustworthiness"]
                    ),
                    "delta_5NN_accuracy": (
                        scores["5NN_accuracy"] - exact_scores["5NN_accuracy"]
                    ),
                    "delta_fit_time_seconds": (
                        timings["fit_time_"] - exact_timings["fit_time_"]
                    ),
                    "graph_construction_seconds": timings[
                        "graph_construction_time_"
                    ],
                    "embedding_optimization_seconds": timings[
                        "embedding_optimization_time_"
                    ],
                    "fit_time_seconds": timings["fit_time_"],
                    "wall_time_seconds": wall_time,
                }
            )
    return pd.DataFrame(rows), exact_summary, embeddings, y, exact_embeddings


def plot_mnist_embedding_grid(
    embeddings: Mapping[tuple[int, int], np.ndarray],
    labels: np.ndarray,
    exact_embeddings: Mapping[int, np.ndarray],
    *,
    coarse_mst_values: Iterable[int],
    local_mst_values: Iterable[int],
):
    """Plot exact MST iteration embeddings above the coarse/local grid."""
    import matplotlib.pyplot as plt

    exact_values = tuple(sorted(int(value) for value in exact_embeddings))
    coarse_values = tuple(sorted(int(value) for value in coarse_mst_values))
    local_values = tuple(sorted(int(value) for value in local_mst_values))
    n_columns = max(len(exact_values), len(local_values))
    n_rows = len(coarse_values) + 1
    fig = plt.figure(
        figsize=(3.2 * n_columns, 3.0 * n_rows),
        constrained_layout=True,
    )
    layout = fig.add_gridspec(n_rows, n_columns)
    fig.suptitle("MNIST 2D embeddings — exact MST iteration grid and hierarchical grid")

    axes = []
    points = None
    for column, n_msts in enumerate(exact_values):
        ax = fig.add_subplot(layout[0, column])
        axes.append(ax)
        embedding = exact_embeddings[n_msts]
        points = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10",
            vmin=-0.5, vmax=9.5, s=2, alpha=0.65, linewidths=0,
            rasterized=True,
        )
        ax.set_title(f"Exact, N_MSTS={n_msts}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")

    for row, coarse_msts in enumerate(coarse_values, start=1):
        for column, local_msts in enumerate(local_values):
            ax = fig.add_subplot(layout[row, column])
            axes.append(ax)
            embedding = embeddings[(coarse_msts, local_msts)]
            points = ax.scatter(
                embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10",
                vmin=-0.5, vmax=9.5, s=2, alpha=0.65, linewidths=0,
                rasterized=True,
            )
            ax.set_title(f"Coarse={coarse_msts}, local={local_msts}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")

    if points is not None:
        colorbar = fig.colorbar(points, ax=axes, ticks=range(10), shrink=0.85)
        colorbar.set_label("MNIST digit")
    plt.close(fig)
    return fig


def run_mnist_quality_benchmark(
    *,
    seeds: Iterable[int],
    max_samples: int | None = 10_000,
    data_seed: int = 42,
    n_msts: int = 8,
    n_epochs: int = 500,
    n_clusters_to_try: Iterable[int] = (50, 100),
    n_coarse_msts: int = 1,
    n_local_msts: int = 1,
    minibatch_size: int = 256,
    n_jobs: int = -1,
    device: str = "cpu",
    minkowski_p: float = 2.0,
) -> dict[str, pd.DataFrame]:
    """Load MNIST, run paired exact/hierarchical fits, and summarize results."""
    seeds = tuple(int(seed) for seed in seeds)
    cluster_counts = tuple(int(k) for k in n_clusters_to_try)
    if not seeds:
        raise ValueError("Provide at least one model seed.")
    if not cluster_counts:
        raise ValueError("Provide at least one cluster count.")
    if max_samples is not None and max_samples < 10:
        raise ValueError("max_samples must be at least 10 or None.")

    X, y = _load_mnist(max_samples, data_seed)
    print(
        f"Fixed MNIST benchmark data: {len(X):,} samples × {X.shape[1]} features; "
        f"seeds={seeds}",
        flush=True,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=data_seed)

    # Warm up torch and joblib so one-time startup costs do not skew timings.
    warmup_common = {
        "n_epochs": 2,
        "batch_size": 4096,
        "random_state": data_seed,
        "device": device,
        "minkowski_p": minkowski_p,
    }
    IteratedMinimumSpanningTreeEmbedder(
        graph_mode="exact", n_msts=n_msts, **warmup_common
    ).fit(X)
    IteratedMinimumSpanningTreeEmbedder(
        graph_mode="hierarchical",
        n_clusters=cluster_counts[0],
        n_coarse_msts=n_coarse_msts,
        n_local_msts=n_local_msts,
        n_jobs=n_jobs,
        minibatch_size=minibatch_size,
        **warmup_common,
    ).fit(X)

    exact_graphs_by_seed: dict[int, set[tuple[int, int]]] = {}
    exact_scores_by_seed: dict[int, dict[str, float]] = {}
    exact_fit_times_by_seed: dict[int, float] = {}
    configurations = []
    for seed in seeds:
        configurations.append({"mode": "exact", "seed": seed, "n_clusters": None})
        configurations.extend(
            {"mode": "hierarchical", "seed": seed, "n_clusters": k}
            for k in cluster_counts
        )

    def run_one(config: Mapping[str, Any]) -> dict[str, Any]:
        mode, seed, n_clusters = config["mode"], config["seed"], config["n_clusters"]
        if mode == "exact":
            estimator = IteratedMinimumSpanningTreeEmbedder(
                graph_mode="exact",
                n_msts=n_msts,
                n_epochs=n_epochs,
                batch_size=4096,
                random_state=seed,
                device=device,
                minkowski_p=minkowski_p,
            )
        else:
            if seed not in exact_graphs_by_seed:
                raise RuntimeError("Exact reference failed for this seed.")
            estimator = IteratedMinimumSpanningTreeEmbedder(
                graph_mode="hierarchical",
                n_clusters=n_clusters,
                n_coarse_msts=n_coarse_msts,
                n_local_msts=n_local_msts,
                n_jobs=n_jobs,
                minibatch_size=minibatch_size,
                n_epochs=n_epochs,
                batch_size=4096,
                random_state=seed,
                device=device,
                minkowski_p=minkowski_p,
            )

        wall_started = time.perf_counter()
        estimator.fit(X)
        wall_time = time.perf_counter() - wall_started
        graph_edges, timings = _validate_fitted_model(estimator, len(X))
        scores = {
            "trustworthiness": float(
                trustworthiness(X, estimator.embedding_, n_neighbors=10)
            ),
            # Labels are used only after fitting for this transductive diagnostic.
            "5NN_accuracy": float(
                cross_val_score(
                    KNeighborsClassifier(n_neighbors=5),
                    estimator.embedding_,
                    y,
                    cv=cv,
                    n_jobs=1,
                ).mean()
            ),
        }

        if mode == "exact":
            exact_graphs_by_seed[seed] = graph_edges
            exact_scores_by_seed[seed] = scores
            exact_fit_times_by_seed[seed] = timings["fit_time_"]
            precision = recall = np.nan
            delta_trust = delta_knn = delta_fit = np.nan
        else:
            reference_edges = exact_graphs_by_seed[seed]
            intersection = len(graph_edges & reference_edges)
            precision = intersection / max(len(graph_edges), 1)
            recall = intersection / max(len(reference_edges), 1)
            reference_scores = exact_scores_by_seed[seed]
            delta_trust = scores["trustworthiness"] - reference_scores["trustworthiness"]
            delta_knn = scores["5NN_accuracy"] - reference_scores["5NN_accuracy"]
            delta_fit = timings["fit_time_"] - exact_fit_times_by_seed[seed]

        return {
            "mode": mode,
            "minkowski_p": minkowski_p,
            "seed": seed,
            "n_clusters": n_clusters,
            "unique_edges": len(graph_edges),
            "exact_edge_precision": precision,
            "exact_edge_recall": recall,
            **scores,
            "delta_trustworthiness": delta_trust,
            "delta_5NN_accuracy": delta_knn,
            "delta_fit_time_seconds": delta_fit,
            "graph_construction_seconds": timings["graph_construction_time_"],
            "embedding_optimization_seconds": timings["embedding_optimization_time_"],
            "fit_time_seconds": timings["fit_time_"],
            "wall_time_seconds": wall_time,
        }

    runs, failures = run_trials(configurations, run_one)
    for failure in failures:
        print(
            f"FAILED mode={failure['mode']} seed={failure['seed']} "
            f"clusters={failure['n_clusters']}: "
            f"{failure['error_type']}: {failure['error']}",
            flush=True,
        )
    print(
        f"Completed {len(runs)} of {len(configurations)} trials; "
        f"{len(failures)} failed.",
        flush=True,
    )
    runs_df = pd.DataFrame(runs)
    failures_df = pd.DataFrame(failures)
    return {
        "runs": runs_df,
        "summary": _summarize_runs(runs_df),
        "failures": failures_df,
    }


def plot_quality_results(summary: pd.DataFrame):
    """Plot graph quality, embedding quality, runtime, and paired deltas."""
    import matplotlib.pyplot as plt

    def plot_median_iqr(ax, frame, x, metric, label, marker="o"):
        frame = frame.sort_values(x)
        median = frame[f"{metric}_median"].to_numpy(dtype=float)
        low = median - frame[f"{metric}_q25"].to_numpy(dtype=float)
        high = frame[f"{metric}_q75"].to_numpy(dtype=float) - median
        ax.errorbar(
            frame[x], median, yerr=np.vstack([low, high]),
            marker=marker, capsize=3, label=label,
        )

    hierarchical = summary[summary["mode"] == "hierarchical"].copy()
    exact = summary[summary["mode"] == "exact"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    if not hierarchical.empty:
        plot_median_iqr(axes[0, 0], hierarchical, "n_clusters", "exact_edge_recall", "Recall")
        plot_median_iqr(
            axes[0, 0], hierarchical, "n_clusters", "exact_edge_precision",
            "Precision", marker="s",
        )
        axes[0, 0].set(
            title="Hierarchical graph overlap with exact",
            xlabel="Cluster count",
            ylabel="Score",
        )
        axes[0, 0].legend()

        for metric, label, marker in [
            ("trustworthiness", "Trustworthiness", "o"),
            ("5NN_accuracy", "5-NN accuracy", "s"),
        ]:
            plot_median_iqr(axes[0, 1], hierarchical, "n_clusters", metric, label, marker)
            if not exact.empty:
                axes[0, 1].axhline(
                    exact.iloc[0][f"{metric}_median"], linestyle="--", alpha=0.7
                )
        axes[0, 1].set(
            title="Embedding quality (median and IQR)",
            xlabel="Cluster count",
            ylabel="Score",
        )
        axes[0, 1].legend()

    rows = []
    for _, row in summary.iterrows():
        label = "Exact" if row["mode"] == "exact" else f"k={int(row['n_clusters'])}"
        rows.append((label, row))
    labels = [label for label, _ in rows]
    positions = np.arange(len(rows))
    for index, (label, row) in enumerate(rows):
        median, q25, q75 = (
            row[f"fit_time_seconds_{suffix}"] for suffix in ("median", "q25", "q75")
        )
        axes[1, 0].errorbar(
            index,
            median,
            yerr=[[median - q25], [q75 - median]],
            fmt="o",
            capsize=3,
        )
    axes[1, 0].set_xticks(positions, labels, rotation=30)
    axes[1, 0].set_title("Total estimator fit time")
    axes[1, 0].set_ylabel("Seconds (median and IQR)")
    axes[1, 0].grid(axis="y", alpha=0.25)

    for index, (_, row) in enumerate(rows):
        for metric, marker, label in [
            ("graph_construction_seconds", "o", "Graph construction"),
            ("embedding_optimization_seconds", "s", "Embedding prep + optimization"),
        ]:
            median, q25, q75 = (
                row[f"{metric}_{suffix}"] for suffix in ("median", "q25", "q75")
            )
            axes[1, 1].errorbar(
                index,
                median,
                yerr=[[median - q25], [q75 - median]],
                fmt=marker,
                capsize=3,
                label=label if index == 0 else None,
            )
    axes[1, 1].set_xticks(positions, labels, rotation=30)
    axes[1, 1].set_title("Fit-stage timings")
    axes[1, 1].set_ylabel("Seconds (median and IQR)")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend()

    paired_columns = [
        "n_clusters",
        "successful_runs",
        "delta_trustworthiness_median",
        "delta_trustworthiness_q25",
        "delta_trustworthiness_q75",
        "delta_5NN_accuracy_median",
        "delta_5NN_accuracy_q25",
        "delta_5NN_accuracy_q75",
        "delta_fit_time_seconds_median",
        "delta_fit_time_seconds_q25",
        "delta_fit_time_seconds_q75",
    ]
    paired_deltas = summary.loc[
        summary["mode"] == "hierarchical", paired_columns
    ].copy()
    return fig, paired_deltas
