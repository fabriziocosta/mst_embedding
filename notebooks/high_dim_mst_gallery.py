"""Load datasets and visualize their high-dimensional IMSTE embeddings."""

from __future__ import annotations

import hashlib
import pickle
import tarfile
import time
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from sklearn.datasets import fetch_openml, get_data_home
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # Notebook execution puts the repository root on sys.path.
    from mst_embedding import (
        ImagePatchRandomProjection,
        IteratedMinimumSpanningTreeEmbedder,
    )
except ModuleNotFoundError:  # Also support importing as notebooks.high_dim_mst_gallery.
    from ..mst_embedding import (
        ImagePatchRandomProjection,
        IteratedMinimumSpanningTreeEmbedder,
    )


def _stratified_sample(
    X: object,
    labels: object,
    *,
    max_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X)
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
    return X, labels


def _load_openml(name: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = fetch_openml(
        name,
        version=1,
        as_frame=False,
        parser="auto",
        cache=True,
    )
    labels = np.unique(dataset.target, return_inverse=True)[1]
    return np.asarray(dataset.data), labels


def _load_kmnist() -> tuple[np.ndarray, np.ndarray]:
    """Load Kuzushiji-MNIST from its official NumPy archives and cache locally."""
    base_url = "https://codh.rois.ac.jp/kmnist/dataset/kmnist"
    cache_dir = Path(get_data_home()) / "kmnist"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def load_archive(filename: str) -> np.ndarray:
        path = cache_dir / filename
        if not path.exists():
            temporary_path = path.with_suffix(path.suffix + ".tmp")
            try:
                urlretrieve(f"{base_url}/{filename}", temporary_path)
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)
        with np.load(path) as archive:
            return np.asarray(archive["arr_0"])

    images = np.concatenate(
        [load_archive("kmnist-train-imgs.npz"), load_archive("kmnist-test-imgs.npz")]
    )
    labels = np.concatenate(
        [load_archive("kmnist-train-labels.npz"), load_archive("kmnist-test-labels.npz")]
    )
    return images.reshape(len(images), -1), labels


def _load_cifar10() -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10's official Python archive and cache the verified download."""
    archive_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    expected_md5 = "c58f30108f718f92721af3b95e74349a"
    archive_path = Path(get_data_home()) / "cifar-10-python.tar.gz"

    def has_expected_checksum(path: Path) -> bool:
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as archive_file:
            for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == expected_md5

    if not archive_path.exists() or not has_expected_checksum(archive_path):
        temporary_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
        try:
            urlretrieve(archive_url, temporary_path)
            if not has_expected_checksum(temporary_path):
                raise ValueError("Downloaded CIFAR-10 archive failed its MD5 checksum.")
            temporary_path.replace(archive_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    images = []
    labels = []
    batch_names = [
        *(f"cifar-10-batches-py/data_batch_{index}" for index in range(1, 6)),
        "cifar-10-batches-py/test_batch",
    ]
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for batch_name in batch_names:
            batch_file = archive.extractfile(batch_name)
            if batch_file is None:
                raise ValueError(f"CIFAR-10 archive is missing {batch_name}.")
            with batch_file:
                batch = pickle.load(batch_file, encoding="bytes")
            images.append(batch[b"data"])
            labels.extend(batch[b"labels"])

    images = np.concatenate(images)
    rgb_images = images.reshape(len(images), 3, 32, 32).transpose(0, 2, 3, 1)
    return rgb_images.reshape(len(images), -1), np.asarray(labels, dtype=np.int64)


def _load_datasets(max_samples: int, random_state: int):
    loaders = {
        "MNIST (28×28)": lambda: _load_openml("mnist_784"),
        "Fashion-MNIST (28×28)": lambda: _load_openml("Fashion-MNIST"),
        "Kuzushiji-MNIST (28×28)": _load_kmnist,
        "CIFAR-10 RGB (32×32×3)": _load_cifar10,
    }
    datasets = {}
    errors = {}
    for name, loader in loaders.items():
        print(f"Loading {name} ...", flush=True)
        try:
            raw_X, raw_labels = loader()
            X, labels = _stratified_sample(
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


def _default_device() -> str:
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else "auto"


def run_high_dim_mst_gallery(
    *,
    max_samples: int = 1000,
    n_patches: int | tuple[int, int] = 5,
    patch_n_components: int = 10,
    position_encoding_size: int | None = None,
    n_msts: int = 8,
    n_epochs: int = 1000,
    batch_size: int = 4096,
    graph_mode: str = "exact",
    n_clusters: int = 100,
    n_coarse_msts: int = 4,
    n_local_msts: int = 8,
    n_jobs: int = -1,
    minibatch_size: int = 1024,
    random_state: int = 42,
    device: str | None = None,
) -> dict[str, object]:
    """Fit and display 2D IMSTE embeddings for several image datasets.

    Dataset loading, normalization, patch projection, fitting, and plotting
    happen here so the notebook can remain a thin launcher. Failed remote
    dataset loads are reported and skipped; successful fits are plotted as
    they finish.
    """
    if max_samples < 10:
        raise ValueError("max_samples must be at least 10.")
    resolved_device = _default_device() if device is None else device
    datasets, load_errors = _load_datasets(max_samples, random_state)

    embeddings = {}
    summaries = []
    for name, (X, labels) in datasets.items():
        print(f"Fitting {name} on {resolved_device} ...", flush=True)
        pipeline = Pipeline(
            [
                ("normalize", StandardScaler()),
                (
                    "patch_projection",
                    ImagePatchRandomProjection(
                        n_patches=n_patches,
                        n_components=patch_n_components,
                        position_encoding_size=position_encoding_size,
                        image_shape=(32, 32, 3)
                        if name.startswith("CIFAR-10")
                        else None,
                        random_state=random_state,
                    ),
                ),
                (
                    "imste",
                    IteratedMinimumSpanningTreeEmbedder(
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
                        graph_mode=graph_mode,
                        n_clusters=n_clusters,
                        n_coarse_msts=n_coarse_msts,
                        n_local_msts=n_local_msts,
                        n_jobs=n_jobs,
                        minibatch_size=minibatch_size,
                    ),
                ),
            ]
        )
        started = time.perf_counter()
        embedding = pipeline.fit_transform(X)
        elapsed = time.perf_counter() - started
        print(f"  pipeline completed in {elapsed:.2f} seconds", flush=True)
        projection = pipeline.named_steps["patch_projection"]
        estimator = pipeline.named_steps["imste"]
        mst_summary = (
            f"{n_local_msts} local / {n_coarse_msts} coarse MSTs"
            if graph_mode == "hierarchical"
            else f"{n_msts} MSTs"
        )
        patch_height, patch_width = projection.patch_shape_
        patch_channels = projection.image_shape_[2]
        patch_grid = (projection.n_patch_rows_, projection.n_patch_columns_)
        print(
            f"  preprocessing output: {embedding.shape[0]:,} samples × "
            f"{projection.n_features_out_:,} features "
            f"({projection.n_patches_} patches × "
            f"{projection.n_features_per_patch_} values per patch)\n"
            f"  patch features: {projection.n_components} projected + "
            f"{projection.position_encoding_size_} positional = "
            f"{projection.n_features_per_patch_} values\n"
            f"  patch layout: {patch_grid[0]}×{patch_grid[1]} grid; each padded patch "
            f"is {patch_height}×{patch_width}×{patch_channels} "
            "(height × width × channels)",
            flush=True,
        )
        embeddings[name] = embedding
        summaries.append(
            {
                "dataset": name,
                "samples": len(X),
                "input_features": X.shape[1],
                "preprocessed_features": projection.n_features_out_,
                "patch_grid": f"{patch_grid[0]}×{patch_grid[1]}",
                "padded_patch_shape": (
                    f"{patch_height}×{patch_width}×{patch_channels}"
                ),
                "patch_features": projection.n_components,
                "position_encoding_size": projection.position_encoding_size_,
                "features_per_patch": projection.n_features_per_patch_,
                "classes": len(np.unique(labels)),
                "n_msts": n_msts if graph_mode == "exact" else None,
                "graph_mode": graph_mode,
                "n_clusters": n_clusters if graph_mode == "hierarchical" else None,
                "n_coarse_msts": n_coarse_msts if graph_mode == "hierarchical" else None,
                "n_local_msts": n_local_msts if graph_mode == "hierarchical" else None,
                "minibatch_size": minibatch_size if graph_mode == "hierarchical" else None,
                "n_epochs": n_epochs,
                "unique_graph_edges": len(estimator.graph_edges_),
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
            f"{name} — {len(X):,} samples × "
            f"{projection.n_features_out_:,} projected features; "
            f"{graph_mode} graph; {mst_summary}; "
            f"{n_epochs} epochs; {elapsed:.2f} s"
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
