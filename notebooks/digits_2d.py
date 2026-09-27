"""Widget helpers for the interactive 2D MNIST embedding notebook."""

from __future__ import annotations

import time
import sys
import traceback

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML, clear_output, display
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

try:  # Notebook execution puts this directory directly on sys.path.
    from digits_sweep import load_mnist_data
except ModuleNotFoundError:  # Also support importing as notebooks.digits_2d.
    from .digits_sweep import load_mnist_data
from mst_embedding import IteratedMinimumSpanningTreeEmbedder


def _load_balanced_pool(max_instances: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    X_pool, labels_pool = load_mnist_data(
        sample_size=max_instances,
        random_state=random_state,
    )

    rng = np.random.RandomState(random_state)
    class_indices = [
        rng.permutation(np.flatnonzero(labels_pool == label))
        for label in np.unique(labels_pool)
    ]
    per_class = min(len(indices) for indices in class_indices)
    balanced_order = np.stack(
        [indices[:per_class] for indices in class_indices], axis=1
    ).reshape(-1)
    return X_pool[balanced_order], labels_pool[balanced_order]


def display_interactive_embedding(
    max_instances: int = 4000,
    initial_sample_count: int = 1000,
    random_state: int = 42,
) -> dict[str, object]:
    """Load a fixed data pool and display controls for fitting a 2D embedding.

    Sliders configure sample count, MST count, rank weighting, training epochs,
    learning rate, repulsion, and negative sampling. The embedding is fit only
    when the user clicks the fit button. Returns widget references for notebook
    customization.
    """
    if max_instances < 500:
        raise ValueError("max_instances must be at least 500.")
    if initial_sample_count < 500:
        raise ValueError("initial_sample_count must be at least 500.")
    initial_sample_count = min(initial_sample_count, max_instances)

    status = widgets.HTML(value="Loading stratified MNIST sample pool...")
    display(status)
    X_pool, labels_pool = _load_balanced_pool(max_instances, random_state)
    max_instances = len(X_pool)
    initial_sample_count = min(initial_sample_count, max_instances)
    minimum_sample_count = min(500, max_instances)
    status.value = (
        f"Balanced pool available: {max_instances:,} samples. "
        f"Default selection: {initial_sample_count:,} samples, 8 MSTs, "
        "1,000 epochs. Click Fit embedding to run."
    )

    sample_count = widgets.IntSlider(
        value=initial_sample_count,
        min=minimum_sample_count,
        max=max_instances,
        step=250,
        description="Instances",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    n_msts = widgets.IntSlider(
        value=8,
        min=1,
        max=64,
        step=1,
        description="MSTs",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    rank_exponent = widgets.FloatSlider(
        value=1.0,
        min=0.0,
        max=3.0,
        step=0.01,
        description="Rank exponent",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    n_epochs = widgets.IntSlider(
        value=1000,
        min=10,
        max=2000,
        step=10,
        description="Epochs",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    learning_rate = widgets.FloatSlider(
        value=0.05,
        min=0.01,
        max=0.10,
        step=0.01,
        readout_format=".2f",
        description="Learning rate",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    lambda_rep = widgets.FloatSlider(
        value=1.0,
        min=0.0,
        max=2.0,
        step=0.1,
        readout_format=".1f",
        description="Repulsion",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    negative_ratio = widgets.IntSlider(
        value=4,
        min=0,
        max=30,
        step=1,
        description="Negative ratio",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    compute_knn = widgets.Checkbox(
        value=False,
        description="Estimate 5-fold 5-NN accuracy",
        indent=False,
    )
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(
        sys.platform == "darwin"
        and mps_backend is not None
        and mps_backend.is_available()
    )
    device_default = "mps" if mps_available else "auto"
    device = widgets.Dropdown(
        options=["auto", "cpu"] + (["mps"] if mps_available else []),
        value=device_default,
        description="Device",
        style={"description_width": "initial"},
    )

    fit_button = widgets.Button(
        description="Fit embedding",
        button_style="primary",
        icon="play",
    )
    reset_button = widgets.Button(
        description="Reset sliders",
        tooltip="Restore default values",
        icon="undo",
    )
    output = widgets.Output()

    def reset_sliders(_=None) -> None:
        sample_count.value = min(1000, max_instances)
        n_msts.value = 8
        rank_exponent.value = 1.0
        n_epochs.value = 1000
        learning_rate.value = 0.05
        lambda_rep.value = 1.0
        negative_ratio.value = 4
        compute_knn.value = False
        device.value = device_default

    def fit_and_display(_=None) -> None:
        fit_button.disabled = True
        status.value = "Fitting the 2D embedding with the selected settings..."
        with output:
            clear_output(wait=True)
            try:
                n_samples = sample_count.value
                X = X_pool[:n_samples]
                labels = labels_pool[:n_samples]
                estimator = IteratedMinimumSpanningTreeEmbedder(
                    n_msts=n_msts.value,
                    rank_weight_exponent=rank_exponent.value,
                    n_components=2,
                    n_epochs=n_epochs.value,
                    batch_size=8192,
                    learning_rate=learning_rate.value,
                    negative_ratio=negative_ratio.value,
                    lambda_rep=lambda_rep.value,
                    random_state=random_state,
                    device=device.value,
                )
                started = time.perf_counter()
                embedding = estimator.fit_transform(X)
                elapsed = time.perf_counter() - started

                knn_title = ""
                if compute_knn.value:
                    cv = StratifiedKFold(
                        n_splits=5, shuffle=True, random_state=random_state
                    )
                    knn_accuracy = cross_val_score(
                        KNeighborsClassifier(n_neighbors=5),
                        embedding,
                        labels,
                        cv=cv,
                    ).mean()
                    knn_title = f" · 5-NN 5-fold CV accuracy={knn_accuracy:.3f}"

                fig, ax = plt.subplots(figsize=(8, 6))
                points = ax.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c=labels,
                    cmap="tab10",
                    vmin=-0.5,
                    vmax=9.5,
                    s=10,
                    alpha=0.8,
                    linewidths=0,
                )
                ax.set(
                    title=(
                        f"2D IMSTE · {n_samples:,} samples · "
                        f"{n_msts.value} MSTs · α={rank_exponent.value:g} · "
                        f"{n_epochs.value} epochs{knn_title}"
                    ),
                    xlabel="Embedding dimension 1",
                    ylabel="Embedding dimension 2",
                )
                fig.colorbar(points, ax=ax, ticks=range(10), label="Digit")
                fig.tight_layout()
                display(fig)
                plt.close(fig)
                display(
                    HTML(
                        f"<p>Fit in {elapsed:.1f} seconds on "
                        f"<b>{estimator.device_}</b>.</p>"
                    )
                )
                status.value = (
                    "Embedding ready. Adjust the sliders and click Fit embedding "
                    "to update it."
                    if compute_knn.value
                    else "Embedding ready. The optional 5-NN estimate was skipped."
                )
            except Exception:
                status.value = "The fit failed; see the error details below."
                traceback.print_exc()
            finally:
                fit_button.disabled = False

    fit_button.on_click(fit_and_display)
    reset_button.on_click(reset_sliders)

    controls = widgets.VBox(
        [
            widgets.HBox([sample_count, n_msts, rank_exponent]),
            widgets.HBox([n_epochs, learning_rate, lambda_rep, negative_ratio]),
            widgets.HBox([compute_knn, device]),
        ]
    )
    display(controls, widgets.HBox([fit_button, reset_button]), output)

    return {
        "controls": controls,
        "fit_button": fit_button,
        "reset_button": reset_button,
        "output": output,
        "status": status,
        "sliders": {
            "sample_count": sample_count,
            "n_msts": n_msts,
            "rank_weight_exponent": rank_exponent,
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "lambda_rep": lambda_rep,
            "negative_ratio": negative_ratio,
            "compute_knn": compute_knn,
            "device": device,
        },
    }
