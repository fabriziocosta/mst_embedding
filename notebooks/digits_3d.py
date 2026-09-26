"""Interactive 3D MST embedding visualization for the MNIST notebook."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
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

    digit_palette = qualitative.Plotly
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
                x=embedding[:, 0],
                y=embedding[:, 1],
                z=embedding[:, 2],
                mode="markers",
                marker={"size": 3, "color": "#636EFA"},
                customdata=labels,
                hovertemplate="Digit %{customdata}<extra></extra>",
                showlegend=False,
            )
    )
    for digit, color in enumerate(digit_palette):
        figure.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                name=str(digit),
                marker={"size": 6, "color": color},
                hoverinfo="skip",
                showlegend=True,
            )
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
        uirevision="mnist-depth-fog",
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


def render_with_camera_fog(figure, embedding, labels, min_opacity=0.16, max_opacity=0.95):
    """Return a FigureWidget with point fading linked to its live camera."""
    if not 0 <= min_opacity <= max_opacity <= 1:
        raise ValueError("opacity bounds must satisfy 0 <= min_opacity <= max_opacity <= 1")
    coordinates = np.asarray(embedding, dtype=float)
    labels = np.asarray(labels, dtype=int)
    mins = coordinates.min(axis=0)
    spans = coordinates.max(axis=0) - mins
    spans[spans == 0] = 1.0
    normalized = 2.0 * (coordinates - mins) / spans - 1.0
    palette_rgb = [
        tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
        for color in qualitative.Plotly
    ]
    widget = go.FigureWidget(figure)

    def update_depth_fog(scene, camera):
        del scene
        eye = getattr(camera, "eye", None)
        center = getattr(camera, "center", None)
        eye = np.array([
            getattr(eye, axis, None) or default
            for axis, default in zip(("x", "y", "z"), (1.25, 1.25, 1.25))
        ], dtype=float)
        center = np.array([
            getattr(center, axis, None) or 0.0
            for axis in ("x", "y", "z")
        ], dtype=float)
        direction = eye / (np.linalg.norm(eye) or 1.0)
        depths = (normalized - center) @ direction
        depth_span = np.ptp(depths) or 1.0
        proximity = (depths - depths.min()) / depth_span
        opacity = min_opacity + (max_opacity - min_opacity) * np.power(proximity, 0.8)
        colors = [
            f"rgba({r},{g},{b},{alpha:.3f})"
            for (r, g, b), alpha in zip(
                (palette_rgb[label % len(palette_rgb)] for label in labels), opacity
            )
        ]
        widget.data[0].marker.color = colors

    widget.layout.scene.on_change(update_depth_fog, "camera")
    update_depth_fog(widget.layout.scene, widget.layout.scene.camera)
    return widget
