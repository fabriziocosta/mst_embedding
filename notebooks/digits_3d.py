"""Interactive 3D MST embedding visualization for the MNIST notebook."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
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
    point_colors = _depth_fog_colors(embedding, labels)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
                x=embedding[:, 0],
                y=embedding[:, 1],
                z=embedding[:, 2],
                mode="markers",
                marker={"size": 3, "color": point_colors},
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


def _depth_fog_colors(embedding, labels, eye=(1.25, 1.25, 1.25), min_opacity=0.16, max_opacity=0.95):
    """Color points with RGBA alpha based on distance along the camera view."""
    if not 0 <= min_opacity <= max_opacity <= 1:
        raise ValueError("opacity bounds must satisfy 0 <= min_opacity <= max_opacity <= 1")
    coordinates = np.asarray(embedding, dtype=float)
    labels = np.asarray(labels, dtype=int)
    mins = coordinates.min(axis=0)
    spans = coordinates.max(axis=0) - mins
    spans[spans == 0] = 1.0
    normalized = 2.0 * (coordinates - mins) / spans - 1.0
    direction = np.asarray(eye, dtype=float)
    direction /= np.linalg.norm(direction) or 1.0
    depths = normalized @ direction
    depth_span = np.ptp(depths) or 1.0
    proximity = (depths - depths.min()) / depth_span
    opacity = min_opacity + (max_opacity - min_opacity) * np.power(proximity, 0.8)
    palette_rgb = [
        tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
        for color in qualitative.Plotly
    ]
    return [
        f"rgba({r},{g},{b},{alpha:.3f})"
        for (r, g, b), alpha in zip(
            (palette_rgb[label % len(palette_rgb)] for label in labels), opacity
        )
    ]


def write_camera_fog_html(
    figure,
    embedding,
    labels,
    output_path=None,
    min_opacity=0.16,
    max_opacity=0.95,
):
    """Save a self-contained Plotly page whose fog follows camera rotation."""
    if not 0 <= min_opacity <= max_opacity <= 1:
        raise ValueError("opacity bounds must satisfy 0 <= min_opacity <= max_opacity <= 1")
    coordinates = np.asarray(embedding, dtype=float)
    labels = np.asarray(labels, dtype=int)
    post_script = f"""
const gd = document.getElementById('{{plot_id}}');
const points = {json.dumps(coordinates.tolist(), separators=(',', ':'))};
const labels = {json.dumps(labels.tolist(), separators=(',', ':'))};
const palette = {json.dumps([
    [int(color[index:index + 2], 16) for index in (1, 3, 5)]
    for color in qualitative.Plotly
], separators=(',', ':'))};
const minAlpha = {float(min_opacity)};
const maxAlpha = {float(max_opacity)};
let scheduled = false;

function updateDepthFog() {{
  scheduled = false;
  const scene = gd._fullLayout && gd._fullLayout.scene;
  const camera = scene && scene.camera;
  if (!camera) return;
  const eye = camera.eye || {{x: 1.25, y: 1.25, z: 1.25}};
  const center = camera.center || {{x: 0, y: 0, z: 0}};
  const length = Math.hypot(eye.x, eye.y, eye.z) || 1;
  const view = [eye.x / length, eye.y / length, eye.z / length];
const ranges = [0, 1, 2].map(dimension => {{
    let low = Infinity;
    let high = -Infinity;
    for (let i = 0; i < points.length; i++) {{
      low = Math.min(low, points[i][dimension]);
      high = Math.max(high, points[i][dimension]);
    }}
    return [low, high];
  }});
  const normalized = points.map(point => point.map((value, dimension) =>
    2 * (value - ranges[dimension][0]) /
      (ranges[dimension][1] - ranges[dimension][0] || 1) - 1
  ));
  const depths = normalized.map(point =>
    (point[0] - center.x) * view[0] +
    (point[1] - center.y) * view[1] +
    (point[2] - center.z) * view[2]
  );
  let far = Infinity;
  let near = -Infinity;
  for (let i = 0; i < depths.length; i++) {{
    far = Math.min(far, depths[i]);
    near = Math.max(near, depths[i]);
  }}
  const span = near - far || 1;
  const colors = labels.map((label, index) => {{
    const alpha = minAlpha + (maxAlpha - minAlpha) *
      Math.pow((depths[index] - far) / span, 0.8);
    const [r, g, b] = palette[label % palette.length];
    return `rgba(${{r}},${{g}},${{b}},${{alpha.toFixed(3)}})`;
  }});
  Plotly.restyle(gd, {{'marker.color': [colors]}}, [0]);
}}

function scheduleFogUpdate() {{
  if (!scheduled) {{
    scheduled = true;
    setTimeout(updateDepthFog, points.length > 10000 ? 100 : 25);
  }}
}}
gd.on('plotly_relayouting', scheduleFogUpdate);
gd.on('plotly_relayout', scheduleFogUpdate);
updateDepthFog();
"""
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "digits_3d_interactive.html"
    output_path = Path(output_path).expanduser().resolve()
    pio.write_html(
        figure,
        file=output_path,
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "scrollZoom": True},
        post_script=post_script,
    )
    return output_path
