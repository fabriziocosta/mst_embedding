"""Reusable helpers for the digits parameter-sweep notebook."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.datasets import load_digits
from sklearn.manifold import trustworthiness
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Support running from a source checkout without requiring an editable install.
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from mst_embedding import IteratedMSTEmbedding


def load_digits_data():
    """Load digits and return standardized features and labels."""
    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)
    return X, digits.target


def format_duration(seconds: float) -> str:
    """Format elapsed time using the largest useful units."""
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours} hr {minutes} min" if minutes else f"{hours} hr"
    if minutes:
        return f"{minutes} min {seconds} sec"
    return f"{seconds} sec"


def _fit_and_score(X, labels, parameter, value, base_params, n_neighbors):
    params = dict(base_params)
    params[parameter] = value
    estimator = IteratedMSTEmbedding(**params)
    started = time.perf_counter()
    print(f"Fitting {parameter}={value} ...", flush=True)
    embedding = estimator.fit_transform(X)
    elapsed = time.perf_counter() - started
    print(
        f"Finished {parameter}={value} in {format_duration(elapsed)} "
        f"on {estimator.device_}",
        flush=True,
    )

    score = trustworthiness(
        X, embedding, n_neighbors=min(n_neighbors, len(X) - 1)
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn_accuracy = cross_val_score(
        KNeighborsClassifier(n_neighbors=5), embedding, labels, cv=cv
    ).mean()
    return {
        parameter: value,
        "trustworthiness": score,
        "5-NN 5-fold CV accuracy": knn_accuracy,
        "seconds": elapsed,
        "elapsed": format_duration(elapsed),
        "device": estimator.device_,
        "embedding": embedding,
    }


def run_sweep(X, labels, parameter, values, base_params, n_neighbors=10, n_jobs=1):
    """Fit and plot one digits embedding per value of an estimator parameter.

    Independent parameter values can run in parallel with ``n_jobs``. Keep the
    worker count modest because each fit builds a dense pairwise distance matrix.
    Returns result dictionaries (including embeddings) and a summary DataFrame.
    """
    if parameter not in IteratedMSTEmbedding().get_params():
        raise ValueError(f"Unknown estimator parameter: {parameter!r}")
    if not values:
        raise ValueError("values must contain at least one candidate")

    arguments = (
        (X, labels, parameter, value, base_params, n_neighbors)
        for value in values
    )
    if n_jobs == 1:
        results = [_fit_and_score(*args) for args in arguments]
    else:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(_fit_and_score)(*args) for args in arguments
            )

    ncols = min(3, len(results))
    nrows = math.ceil(len(results) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False,
        constrained_layout=True,
    )

    for index, result in enumerate(results):
        embedding = result["embedding"]
        value = result[parameter]
        score = result["trustworthiness"]
        knn_accuracy = result["5-NN 5-fold CV accuracy"]
        ax = axes.flat[index]
        points = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10",
            vmin=-0.5, vmax=9.5, s=7, alpha=0.8, linewidths=0,
        )
        ax.set_title(
            f"{parameter}={value}\n"
            f"trustworthiness={score:.3f}\n"
            f"5-NN 5-fold CV accuracy={knn_accuracy:.3f}"
        )
        ax.set_xlabel("Embedding dimension 1")
        ax.set_ylabel("Embedding dimension 2")

    for ax in axes.flat[len(values):]:
        ax.set_visible(False)
    colorbar = fig.colorbar(points, ax=list(axes.flat[:len(values)]), ticks=range(10))
    colorbar.set_label("Digit label")
    plt.show()

    summary = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "embedding"}
         for row in results]
    )
    return results, summary
