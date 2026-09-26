"""Interactive 3D MST embedding visualization for the MNIST notebook."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import trustworthiness
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

try:  # Notebook execution puts this directory directly on sys.path.
    from digits_sweep import format_duration
except ModuleNotFoundError:  # Also support importing as notebooks.digits_3d.
    from .digits_sweep import format_duration
from mst_embedding import IteratedMSTEmbedding


def fit_and_plot_mst_3d(
    X,
    labels,
    n_msts=8,
    n_epochs=500,
    batch_size=4096,
    learning_rate=0.05,
    negative_ratio=4,
    lambda_rep=1.0,
    epsilon=1e-4,
    random_state=42,
    device="auto",
    trustworthiness_neighbors=10,
):
    """Fit one 3D MST embedding and return its scores and Plotly figure."""
    X = np.asarray(X)
    labels = np.asarray(labels)
    if X.ndim != 2 or len(X) != len(labels):
        raise ValueError("X must be 2D and have one label per row")
    if len(X) < 25:
        raise ValueError("At least 25 samples are required for 5-fold 5-NN scoring")

    estimator = IteratedMSTEmbedding(
        n_msts=n_msts,
        n_components=3,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        negative_ratio=negative_ratio,
        lambda_rep=lambda_rep,
        epsilon=epsilon,
        random_state=random_state,
        device=device,
    )
    print(f"Fitting 3D MST embedding with n_msts={n_msts} ...", flush=True)
    started = time.perf_counter()
    embedding = estimator.fit_transform(X)
    elapsed = time.perf_counter() - started

    n_neighbors = min(trustworthiness_neighbors, (len(X) - 1) // 2)
    score = trustworthiness(X, embedding, n_neighbors=n_neighbors)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    knn_accuracy = cross_val_score(
        KNeighborsClassifier(n_neighbors=5), embedding, labels, cv=cv
    ).mean()
    elapsed_text = format_duration(elapsed)
    print(
        f"Finished 3D MST embedding in {elapsed_text} on {estimator.device_}",
        flush=True,
    )

    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=embedding[:, 0],
                y=embedding[:, 1],
                z=embedding[:, 2],
                mode="markers",
                marker={
                    "size": 3,
                    "opacity": 0.8,
                    "color": labels.astype(int),
                    "colorscale": "Turbo",
                    "cmin": 0,
                    "cmax": 9,
                    "colorbar": {
                        "title": "Digit",
                        "tickmode": "array",
                        "tickvals": list(range(10)),
                    },
                },
                customdata=labels,
                hovertemplate="Digit %{customdata}<extra></extra>",
                showlegend=False,
            )
        ]
    )
    figure.update_layout(
        title=(
            f"3D Iterated MST embedding (n_msts={n_msts})<br>"
            f"trustworthiness={score:.3f}; 5-NN CV={knn_accuracy:.3f}; "
            f"runtime={elapsed_text} ({estimator.device_})"
        ),
        scene={
            "xaxis_title": "Dimension 1",
            "yaxis_title": "Dimension 2",
            "zaxis_title": "Dimension 3",
            "aspectmode": "cube",
        },
        height=750,
        width=1000,
        margin={"l": 0, "r": 0, "t": 110, "b": 0},
        legend={
            "title": {"text": "Digit"},
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.08,
        },
    )
    summary = pd.DataFrame([{
        "n_msts": n_msts,
        "n_components": 3,
        "trustworthiness": score,
        "5-NN 5-fold CV accuracy": knn_accuracy,
        "seconds": elapsed,
        "elapsed": elapsed_text,
        "device": estimator.device_,
    }])
    return embedding, summary, figure
