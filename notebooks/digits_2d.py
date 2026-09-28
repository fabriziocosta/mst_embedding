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
from sklearn.preprocessing import StandardScaler

try:  # Notebook execution puts this directory directly on sys.path.
    from digits_sweep import load_mnist_data
    from high_dim_mst_gallery import _load_kmnist, _load_openml, _stratified_sample
except ModuleNotFoundError:  # Also support importing as notebooks.digits_2d.
    from .digits_sweep import load_mnist_data
    from .high_dim_mst_gallery import _load_kmnist, _load_openml, _stratified_sample
from mst_embedding import IteratedMinimumSpanningTreeEmbedder


DATASET_LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "kmnist": "Kuzushiji-MNIST",
}


def _load_balanced_pool(
    dataset: str,
    max_instances: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "mnist":
        X_pool, labels_pool = load_mnist_data(
            sample_size=max_instances,
            random_state=random_state,
        )
    elif dataset == "fashion_mnist":
        X, labels = _load_openml("Fashion-MNIST")
        X_pool, labels_pool = _stratified_sample(
            X, labels, max_samples=max_instances, random_state=random_state
        )
        X_pool = StandardScaler().fit_transform(
            np.asarray(X_pool, dtype=np.float32)
        )
    elif dataset == "kmnist":
        X, labels = _load_kmnist()
        X_pool, labels_pool = _stratified_sample(
            X, labels, max_samples=max_instances, random_state=random_state
        )
        X_pool = StandardScaler().fit_transform(
            np.asarray(X_pool, dtype=np.float32)
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}.")

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
    """Load MNIST and display controls for fitting a 2D embedding.

    Dataset choices load lazily and cache balanced sample pools. Controls
    configure sample count, MST count, distance and loss shapes, and negative
    sampling. Attraction uses fixed weight-sum normalization. Fits
    use 1,000 epochs and a 0.05 learning rate. The embedding is fit only when
    the user clicks the fit button. Returns widget references for notebook
    customization.
    """
    if max_instances < 500:
        raise ValueError("max_instances must be at least 500.")
    if initial_sample_count < 500:
        raise ValueError("initial_sample_count must be at least 500.")
    initial_sample_count = min(initial_sample_count, max_instances)

    dataset = widgets.Dropdown(
        options=[
            ("MNIST", "mnist"),
            ("Fashion-MNIST", "fashion_mnist"),
            ("Kuzushiji-MNIST", "kmnist"),
        ],
        value="mnist",
        description="Dataset",
        style={"description_width": "initial"},
    )
    status = widgets.HTML(value="Loading stratified MNIST sample pool...")
    display(status)
    max_pool_instances = max_instances
    X_pool, labels_pool = _load_balanced_pool("mnist", max_instances, random_state)
    pool_cache = {"mnist": (X_pool, labels_pool)}
    active_dataset = {"name": "mnist"}
    max_instances = len(X_pool)
    initial_sample_count = min(initial_sample_count, max_instances)
    minimum_sample_count = min(500, max_instances)
    status.value = (
        f"{DATASET_LABELS['mnist']} pool available: {max_instances:,} samples. "
        f"Default selection: {initial_sample_count:,} samples, 10 MSTs, "
        "squared distance, log attraction, Bernoulli repulsion, "
        "weight_sum normalization, 1,000 epochs. "
        "Click Fit embedding to run."
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
        value=10,
        min=1,
        max=64,
        step=1,
        description="MSTs",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    lambda_rep = widgets.FloatSlider(
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.05,
        readout_format=".2f",
        description="Repulsion share (λ)",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    repulsion_type = widgets.Dropdown(
        options=[
            ("Log / Bernoulli", "bernoulli"),
            ("Inverse distance", "inverse_distance"),
            ("Logistic margin", "logistic"),
        ],
        value="bernoulli",
        description="Repulsion dampening",
        style={"description_width": "initial"},
    )
    distance_type = widgets.Dropdown(
        options=[("Squared Euclidean", "squared"), ("Euclidean", "euclidean")],
        value="squared",
        description="Distance",
        style={"description_width": "initial"},
    )
    attraction_dampening = widgets.Dropdown(
        options=[
            ("Direct", "direct"),
            ("Log", "log"),
            ("Logistic margin", "logistic"),
            ("Huber", "huber"),
        ],
        value="log",
        description="Attraction dampening",
        style={"description_width": "initial"},
    )
    logistic_margin = widgets.FloatSlider(
        value=1.0, min=0.0, max=5.0, step=0.1,
        description="Logistic margin",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    logistic_temperature = widgets.FloatSlider(
        value=0.5, min=0.05, max=2.0, step=0.05,
        description="Logistic temperature",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    negative_ratio = widgets.IntSlider(
        value=5,
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

    def select_dataset(change) -> None:
        selected = change["new"]
        previous = active_dataset["name"]
        fit_button.disabled = True
        status.value = f"Loading {DATASET_LABELS[selected]} sample pool..."
        try:
            if selected not in pool_cache:
                pool_cache[selected] = _load_balanced_pool(
                    selected, max_pool_instances, random_state
                )
            active_dataset["name"] = selected
            X_selected, _ = pool_cache[selected]
            sample_count.max = len(X_selected)
            sample_count.value = min(sample_count.value, len(X_selected))
            status.value = (
                f"{DATASET_LABELS[selected]} pool available: "
                f"{len(X_selected):,} samples. Adjust controls and click Fit embedding."
            )
        except Exception:
            status.value = f"Could not load {DATASET_LABELS[selected]}; see error details."
            traceback.print_exc()
            dataset.unobserve(select_dataset, names="value")
            dataset.value = previous
            dataset.observe(select_dataset, names="value")
        finally:
            fit_button.disabled = False

    def reset_sliders(_=None) -> None:
        dataset.value = "mnist"
        sample_count.value = min(1000, sample_count.max)
        n_msts.value = 10
        lambda_rep.value = 0.5
        repulsion_type.value = "bernoulli"
        distance_type.value = "squared"
        attraction_dampening.value = "log"
        logistic_margin.value = 1.0
        logistic_temperature.value = 0.5
        negative_ratio.value = 5
        compute_knn.value = False
        device.value = device_default

    def fit_and_display(_=None) -> None:
        fit_button.disabled = True
        status.value = "Fitting the 2D embedding with the selected settings..."
        with output:
            clear_output(wait=True)
            try:
                n_samples = sample_count.value
                X_pool, labels_pool = pool_cache[active_dataset["name"]]
                X = X_pool[:n_samples]
                labels = labels_pool[:n_samples]
                estimator = IteratedMinimumSpanningTreeEmbedder(
                    n_msts=n_msts.value,
                    n_components=2,
                    n_epochs=1000,
                    batch_size=8192,
                    learning_rate=0.05,
                    negative_ratio=negative_ratio.value,
                    lambda_rep=lambda_rep.value,
                    repulsion_type=repulsion_type.value,
                    distance_type=distance_type.value,
                    attraction_dampening=attraction_dampening.value,
                    logistic_margin=logistic_margin.value,
                    logistic_temperature=logistic_temperature.value,
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
                logistic_title = ""
                if (
                    attraction_dampening.value == "logistic"
                    or repulsion_type.value == "logistic"
                ):
                    logistic_title = (
                        f" · logistic m={logistic_margin.value:.1f}, "
                        f"τ={logistic_temperature.value:.2f}"
                    )

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
                        f"2D IMSTE · {DATASET_LABELS[active_dataset['name']]} · "
                        f"{n_samples:,} samples · "
                        f"{n_msts.value} MSTs · inverse-rank weights · "
                        f"{distance_type.value} distance · "
                        f"{attraction_dampening.value} attraction · "
                        f"{repulsion_type.value} repulsion · weight_sum · "
                        f"1,000 epochs{logistic_title}{knn_title}"
                    ),
                    xlabel="Embedding dimension 1",
                    ylabel="Embedding dimension 2",
                )
                fig.colorbar(points, ax=ax, ticks=range(10), label="Class label")
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

    dataset.observe(select_dataset, names="value")
    fit_button.on_click(fit_and_display)
    reset_button.on_click(reset_sliders)

    controls = widgets.VBox(
        [
            widgets.HBox([dataset]),
            widgets.HBox([sample_count, n_msts]),
            widgets.HBox([lambda_rep, negative_ratio]),
            widgets.HBox([distance_type]),
            widgets.HBox([attraction_dampening, repulsion_type]),
            widgets.HBox([logistic_margin, logistic_temperature]),
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
            "dataset": dataset,
            "sample_count": sample_count,
            "n_msts": n_msts,
            "lambda_rep": lambda_rep,
            "distance_type": distance_type,
            "attraction_dampening": attraction_dampening,
            "repulsion_type": repulsion_type,
            "logistic_margin": logistic_margin,
            "logistic_temperature": logistic_temperature,
            "negative_ratio": negative_ratio,
            "compute_knn": compute_knn,
            "device": device,
        },
    }
