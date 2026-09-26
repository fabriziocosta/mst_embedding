"""Interactive 3D MNIST embedding comparisons for the companion notebook."""

from __future__ import annotations

import inspect
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE

from digits_sweep import format_duration


def run_tsne_3d_comparisons(
    X,
    labels,
    configurations,
    n_iterations=1000,
    random_state=42,
):
    """Fit 3D t-SNE embeddings and show them in one interactive Plotly row.

    Each subplot has an independent 3D camera, so drag to rotate, scroll to
    zoom, and hover over points to see their digit labels.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    if X.ndim != 2 or len(X) != len(labels):
        raise ValueError("X must be 2D and have one label per row")
    if len(X) < 6:
        raise ValueError("At least 6 samples are required for 5-fold 3D t-SNE")
    if not configurations:
        raise ValueError("configurations must contain at least one setting")

    iteration_parameter = (
        "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    )
    results = []
    for configuration in configurations:
        name = configuration.get("name", "t-SNE")
        perplexity = configuration["perplexity"]
        if not 0 < perplexity < len(X):
            raise ValueError(
                f"perplexity must be positive and smaller than the sample count; got {perplexity}"
            )
        reducer = TSNE(
            n_components=3,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
            n_jobs=-1,
            **{iteration_parameter: n_iterations},
        )
        print(f"Fitting 3D t-SNE: {name} ...", flush=True)
        started = time.perf_counter()
        embedding = reducer.fit_transform(X)
        elapsed = time.perf_counter() - started
        results.append({
            "configuration": name,
            "perplexity": perplexity,
            "embedding": embedding,
            "seconds": elapsed,
            "elapsed": format_duration(elapsed),
        })
        print(f"Finished 3D t-SNE: {name} in {format_duration(elapsed)}", flush=True)

    figure = make_subplots(
        rows=1,
        cols=len(results),
        specs=[[{"type": "scene"} for _ in results]],
        subplot_titles=[
            f"{item['configuration']}<br>perplexity={item['perplexity']}, {item['elapsed']}"
            for item in results
        ],
        horizontal_spacing=0.025,
    )
    digit_labels = labels.astype(int)
    for column, result in enumerate(results, start=1):
        embedding = result["embedding"]
        figure.add_trace(
            go.Scatter3d(
                x=embedding[:, 0],
                y=embedding[:, 1],
                z=embedding[:, 2],
                mode="markers",
                name=result["configuration"],
                marker={
                    "size": 2.5,
                    "opacity": 0.8,
                    "color": digit_labels,
                    "colorscale": "Turbo",
                    "cmin": 0,
                    "cmax": 9,
                    "showscale": column == 1,
                    "colorbar": {
                        "title": "Digit",
                        "tickmode": "array",
                        "tickvals": list(range(10)),
                        "x": 0.285,
                        "len": 0.75,
                    },
                },
                customdata=digit_labels,
                hovertemplate="Digit %{customdata}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.update_scenes(
            dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3",
                aspectmode="cube",
            ),
            row=1,
            col=column,
        )

    figure.update_layout(
        title="MNIST t-SNE embeddings in 3D — drag each panel to rotate",
        height=650,
        width=max(1100, 480 * len(results)),
        margin={"l": 0, "r": 0, "t": 100, "b": 10},
    )
    return results, figure
