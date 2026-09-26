"""Load, embed, and visualize several high-dimensional image datasets."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from sklearn.datasets import fetch_openml, fetch_olivetti_faces, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:  # Notebook execution puts the repository root on sys.path.
    from mst_embedding import IteratedMSTEmbedding
except ModuleNotFoundError:  # Also support importing as notebooks.high_dim_mst_gallery.
    from ..mst_embedding import IteratedMSTEmbedding


def _stratified_standardized_sample(
    X: object,
    labels: object,
    *,
    max_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    if len(X) > max_samples:
        indices, _ = train_test_split(
            np.arange(len(X)),
            train_size=max_samples,
            random_state=random_state,
            stratify=labels,
        )
        X = X[indices]
        labels = labels[indices]
    return StandardScaler().fit_transform(X), labels


def _load_openml(name: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = fetch_openml(
        name,
        version=1,
        as_frame=False,
        parser="auto",
        cache=True,
    )
    return np.asarray(dataset.data), np.asarray(dataset.target, dtype=np.int64)


def _load_datasets(max_samples: int, random_state: int):
    digits = load_digits()
    loaders = {
        "Digits (8×8)": lambda: (digits.data, digits.target),
        "MNIST (28×28)": lambda: _load_openml("mnist_784"),
        "Fashion-MNIST (28×28)": lambda: _load_openml("Fashion-MNIST"),
        "Olivetti faces (64×64)": lambda: _load_olivetti(random_state),
    }
    datasets = {}
    errors = {}
    for name, loader in loaders.items():
        print(f"Loading {name} ...", flush=True)
        try:
            raw_X, raw_labels = loader()
            X, labels = _stratified_standardized_sample(
                raw_X,
                raw_labels,
                max_samples=max_samples,
                random_state=random_state,
            )
            datasets[name] = (X, labels)
            print(
                f"  {len(X):,} samples × {X.shape[1]:,} features; "
                f"{len(np.unique(labels))} classes",
                flush=True,
            )
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
            print(f"  skipped: {errors[name]}", flush=True)
    if not datasets:
        raise RuntimeError("No datasets loaded successfully.")
    return datasets, errors


def _load_olivetti(random_state: int) -> tuple[np.ndarray, np.ndarray]:
    bunch = fetch_olivetti_faces(shuffle=True, random_state=random_state)
    return bunch.data, bunch.target


def _default_device() -> str:
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else "auto"


def run_high_dim_mst_gallery(
    *,
    max_samples: int = 1000,
    n_msts: int = 8,
    n_epochs: int = 1000,
    batch_size: int = 4096,
    random_state: int = 42,
    device: str | None = None,
) -> dict[str, object]:
    """Fit and display 2D MST embeddings for several image datasets.

    Dataset loading, standardization, fitting, and plotting happen here so the
    notebook can remain a thin launcher. Failed remote dataset loads are
    reported and skipped; successful fits are plotted as they finish.
    """
    if max_samples < 10:
        raise ValueError("max_samples must be at least 10.")
    resolved_device = _default_device() if device is None else device
    datasets, load_errors = _load_datasets(max_samples, random_state)

    embeddings = {}
    summaries = []
    for name, (X, labels) in datasets.items():
        print(f"Fitting {name} on {resolved_device} ...", flush=True)
        estimator = IteratedMSTEmbedding(
            n_msts=n_msts,
            n_components=2,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=0.05,
            negative_ratio=4,
            lambda_rep=1.0,
            rank_weight_exponent=1.0,
            random_state=random_state,
            device=resolved_device,
        )
        started = time.perf_counter()
        embedding = estimator.fit_transform(X)
        elapsed = time.perf_counter() - started
        embeddings[name] = embedding
        summaries.append(
            {
                "dataset": name,
                "samples": len(X),
                "features": X.shape[1],
                "classes": len(np.unique(labels)),
                "n_msts": n_msts,
                "n_epochs": n_epochs,
                "seconds": elapsed,
                "device": estimator.device_,
            }
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        points = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap="tab20",
            s=12,
            alpha=0.8,
            linewidths=0,
        )
        ax.set_title(
            f"{name} — {len(X):,} samples × {X.shape[1]:,} features; "
            f"{n_msts} MSTs; {n_epochs} epochs"
        )
        ax.set_xlabel("Embedding dimension 1")
        ax.set_ylabel("Embedding dimension 2")
        fig.colorbar(points, ax=ax, label="Class label")
        fig.tight_layout()
        display(fig)
        plt.close(fig)

    return {
        "embeddings": embeddings,
        "summary": pd.DataFrame(summaries),
        "load_errors": load_errors,
    }
