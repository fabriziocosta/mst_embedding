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
    """Load image pools and display controls for fitting a 2D embedding.

    Dataset choices load lazily and cache balanced sample pools with separate,
    disjoint training and novel rows. Controls configure sample count, MST
    count, embedding batch size, distance and loss shapes, and negative
    sampling. Attraction uses fixed weight-sum normalization. Fits use 1,000
    epochs and a 0.05 learning rate. The embedding is fit only when the user
    clicks the fit button.
    Returns widget references for notebook customization.
    """
    if max_instances < 500:
        raise ValueError("max_instances must be at least 500.")
    if max_instances > 35_000:
        raise ValueError(
            "max_instances must be at most 35,000 to reserve an equally sized "
            "novel batch from the 70,000-row dataset pools."
        )
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
    max_training_instances = max_instances
    max_pool_instances = 2 * max_training_instances
    X_pool, labels_pool = _load_balanced_pool(
        "mnist", max_pool_instances, random_state
    )
    pool_cache = {"mnist": (X_pool, labels_pool)}
    active_dataset = {"name": "mnist"}
    max_training_instances = min(max_training_instances, len(X_pool) // 2)
    initial_sample_count = min(initial_sample_count, max_training_instances)
    minimum_sample_count = min(500, max_training_instances)
    status.value = (
        f"{DATASET_LABELS['mnist']} pool available: {len(X_pool):,} samples "
        f"({max_training_instances:,} training plus an equally sized novel pool). "
        f"Default selection: {initial_sample_count:,} samples, 10 MSTs, "
        "squared distance, log attraction, logistic repulsion, "
        "weight_sum normalization, 1,000 epochs, and an 8,192 embedding batch. "
        "Click Fit embedding to run."
    )

    sample_count = widgets.IntSlider(
        value=initial_sample_count,
        min=minimum_sample_count,
        max=max_training_instances,
        step=250,
        description="Instances",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    batch_size = widgets.IntSlider(
        value=8192,
        min=256,
        max=16384,
        step=256,
        description="Embedding batch size",
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
            ("Logistic margin", "logistic"),
            ("Log / Bernoulli", "bernoulli"),
            ("Inverse distance", "inverse_distance"),
        ],
        value="logistic",
        description="Repulsion dampening",
        style={"description_width": "initial"},
    )
    logistic_margin = widgets.FloatSlider(
        value=1.0, min=0.0, max=5.0, step=0.1,
        description="Squared-distance margin",
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
    transform_method = widgets.Dropdown(
        options=[("Direct (training rows only)", "direct"), ("MLP projection", "mlp")],
        value="direct",
        description="Transform",
        style={"description_width": "initial"},
    )
    mlp_n_layers = widgets.IntSlider(
        value=6,
        min=1,
        max=10,
        step=1,
        description="MLP layers",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    mlp_layer_size = widgets.IntSlider(
        value=128,
        min=32,
        max=512,
        step=32,
        description="Units per layer",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    mlp_dropout = widgets.FloatSlider(
        value=0.1,
        min=0.0,
        max=0.5,
        step=0.05,
        description="MLP dropout",
        continuous_update=False,
        style={"description_width": "initial"},
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
    mlp_button = widgets.Button(
        description="Show MLP predictions",
        tooltip=(
            "Compare the training embedding with predictions for a disjoint, "
            "same-size batch from the same dataset."
        ),
        icon="eye",
        disabled=True,
    )
    output = widgets.Output()
    projection_output = widgets.Output()
    current_fit: dict[str, object] = {}

    def show_mlp_predictions(_=None) -> None:
        fitted = current_fit
        if not fitted:
            return
        with projection_output:
            clear_output(wait=True)
            estimator = fitted["estimator"]
            labels = fitted["labels"]
            novel_X = fitted["novel_X"]
            novel_labels = fitted["novel_labels"]
            embedding = fitted["embedding"]
            predicted = estimator.transform_with_mlp(novel_X)

            fig, axes = plt.subplots(
                1, 2, figsize=(13, 5.5), sharex=True, sharey=True,
                constrained_layout=True,
            )
            for ax, coordinates, title in zip(
                axes,
                (embedding, predicted),
                ("Optimized training embedding", "MLP predictions (novel rows)"),
            ):
                if ax is axes[1]:
                    ax.scatter(
                        embedding[:, 0],
                        embedding[:, 1],
                        c="lightgray",
                        s=8,
                        alpha=0.35,
                        linewidths=0,
                        zorder=1,
                    )
                points = ax.scatter(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    c=labels if ax is axes[0] else novel_labels,
                    cmap="tab10",
                    vmin=-0.5,
                    vmax=9.5,
                    s=10,
                    alpha=0.8,
                    linewidths=0,
                    zorder=2,
                )
                ax.set(title=title, xlabel="Embedding dimension 1")
                ax.set_aspect("equal", adjustable="box")
            axes[0].set_ylabel("Embedding dimension 2")
            combined = np.vstack((embedding, predicted))
            # Use central 99.5% intervals per coordinate. By the union bound,
            # this keeps at least 99% of points inside the shared 2D viewport.
            x_min, x_max = np.quantile(combined[:, 0], [0.0025, 0.9975])
            y_min, y_max = np.quantile(combined[:, 1], [0.0025, 0.9975])
            x_margin = max((x_max - x_min) * 0.03, 0.1)
            y_margin = max((y_max - y_min) * 0.03, 0.1)
            for ax in axes:
                ax.set_xlim(
                    x_min - x_margin,
                    x_max + x_margin,
                )
                ax.set_ylim(
                    y_min - y_margin,
                    y_max + y_margin,
                )
            fig.colorbar(
                points,
                ax=axes,
                ticks=range(10),
                label="Class label",
                location="right",
                pad=0.03,
            )
            fig.suptitle(
                f"{DATASET_LABELS[fitted['dataset']]} · MLP prediction on "
                f"{len(novel_labels):,} novel rows"
            )
            display(fig)
            plt.close(fig)

    def update_mlp_button(*_) -> None:
        mlp_button.disabled = not (
            current_fit
            and current_fit.get("transform_method") == "mlp"
            and transform_method.value == "mlp"
        )

    def select_dataset(change) -> None:
        selected = change["new"]
        previous = active_dataset["name"]
        fit_button.disabled = True
        mlp_button.disabled = True
        current_fit.clear()
        projection_output.clear_output(wait=True)
        status.value = f"Loading {DATASET_LABELS[selected]} sample pool..."
        try:
            if selected not in pool_cache:
                pool_cache[selected] = _load_balanced_pool(
                    selected, max_pool_instances, random_state
                )
            active_dataset["name"] = selected
            X_selected, _ = pool_cache[selected]
            sample_count.max = min(max_instances, len(X_selected) // 2)
            sample_count.value = min(sample_count.value, sample_count.max)
            status.value = (
                f"{DATASET_LABELS[selected]} pool available: "
                f"{len(X_selected):,} samples ({sample_count.max:,} training plus "
                "an equally sized novel pool). Adjust controls and click Fit embedding."
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
        current_fit.clear()
        mlp_button.disabled = True
        projection_output.clear_output(wait=True)
        dataset.value = "mnist"
        sample_count.value = min(1000, sample_count.max)
        n_msts.value = 10
        lambda_rep.value = 0.5
        repulsion_type.value = "logistic"
        logistic_margin.value = 1.0
        logistic_temperature.value = 0.5
        negative_ratio.value = 5
        compute_knn.value = False
        batch_size.value = 8192
        transform_method.value = "direct"
        mlp_n_layers.value = 6
        mlp_layer_size.value = 128
        mlp_dropout.value = 0.1
        device.value = device_default

    def fit_and_display(_=None) -> None:
        fit_button.disabled = True
        mlp_button.disabled = True
        current_fit.clear()
        projection_output.clear_output(wait=True)
        status.value = "Fitting the 2D embedding with the selected settings..."
        with output:
            clear_output(wait=True)
            try:
                n_samples = sample_count.value
                X_pool, labels_pool = pool_cache[active_dataset["name"]]
                X = X_pool[:n_samples]
                labels = labels_pool[:n_samples]
                novel_X = X_pool[n_samples : 2 * n_samples]
                novel_labels = labels_pool[n_samples : 2 * n_samples]
                if len(novel_X) != n_samples:
                    raise ValueError(
                        "The selected dataset pool does not contain enough novel "
                        "rows for a same-size prediction batch."
                    )
                estimator = IteratedMinimumSpanningTreeEmbedder(
                    n_msts=n_msts.value,
                    n_components=2,
                    n_epochs=1000,
                    batch_size=batch_size.value,
                    learning_rate=0.05,
                    negative_ratio=negative_ratio.value,
                    lambda_rep=lambda_rep.value,
                    repulsion_type=repulsion_type.value,
                    logistic_margin=logistic_margin.value,
                    logistic_temperature=logistic_temperature.value,
                    transform_method=transform_method.value,
                    mlp_n_layers=mlp_n_layers.value,
                    mlp_layer_size=mlp_layer_size.value,
                    mlp_dropout=mlp_dropout.value,
                    mlp_min_epochs=100,
                    mlp_patience=20,
                    random_state=random_state,
                    device=device.value,
                )
                started = time.perf_counter()
                embedding = estimator.fit_transform(X)
                elapsed = time.perf_counter() - started
                current_fit.update(
                    {
                        "estimator": estimator,
                        "labels": labels.copy(),
                        "novel_X": novel_X,
                        "novel_labels": novel_labels.copy(),
                        "embedding": embedding.copy(),
                        "dataset": active_dataset["name"],
                        "transform_method": transform_method.value,
                    }
                )
                update_mlp_button()

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
                if repulsion_type.value == "logistic":
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
                    alpha=1.0,
                    linewidths=0,
                    zorder=2,
                )
                ax.set(
                    title=(
                        f"2D IMSTE · {DATASET_LABELS[active_dataset['name']]} · "
                        f"{n_samples:,} samples · "
                        f"{n_msts.value} MSTs · inverse-rank weights · "
                        "squared distance · log attraction · "
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
                status.value = "Embedding ready. "
                if not compute_knn.value:
                    status.value += "The optional cross-validation estimate was skipped. "
                if transform_method.value == "mlp":
                    status.value += (
                        "Click Show MLP predictions to project an equally sized "
                        "novel batch. "
                    )
                status.value += "Adjust the controls and click Fit embedding to update it."
            except Exception:
                status.value = "The fit failed; see the error details below."
                traceback.print_exc()
            finally:
                fit_button.disabled = False

    dataset.observe(select_dataset, names="value")
    transform_method.observe(update_mlp_button, names="value")
    fit_button.on_click(fit_and_display)
    reset_button.on_click(reset_sliders)
    mlp_button.on_click(show_mlp_predictions)

    controls = widgets.VBox(
        [
            widgets.HBox([dataset]),
            widgets.HBox([sample_count, n_msts]),
            widgets.HBox([batch_size]),
            widgets.HBox([lambda_rep, negative_ratio]),
            widgets.HBox([repulsion_type]),
            widgets.HBox([logistic_margin, logistic_temperature]),
            widgets.HBox([compute_knn]),
            widgets.HBox([transform_method]),
            widgets.HBox([mlp_n_layers, mlp_layer_size, mlp_dropout]),
            widgets.HBox([device]),
        ]
    )
    display(
        controls,
        widgets.HBox([fit_button, reset_button, mlp_button]),
        output,
        projection_output,
    )

    return {
        "controls": controls,
        "fit_button": fit_button,
        "mlp_button": mlp_button,
        "reset_button": reset_button,
        "output": output,
        "projection_output": projection_output,
        "status": status,
        "sliders": {
            "dataset": dataset,
            "sample_count": sample_count,
            "batch_size": batch_size,
            "n_msts": n_msts,
            "lambda_rep": lambda_rep,
            "repulsion_type": repulsion_type,
            "logistic_margin": logistic_margin,
            "logistic_temperature": logistic_temperature,
            "negative_ratio": negative_ratio,
            "compute_knn": compute_knn,
            "transform_method": transform_method,
            "mlp_n_layers": mlp_n_layers,
            "mlp_layer_size": mlp_layer_size,
            "mlp_dropout": mlp_dropout,
            "device": device,
        },
    }
