"""Scikit-learn estimator for IMSTE embeddings."""

from __future__ import annotations

import numbers
import sys
import time

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .losses import (
    log_attraction_loss,
    logistic_repulsion_loss,
)


def _positive_integer(name: str, value: object, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    return result


def _finite_nonnegative(name: str, value: object, *, strictly_positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number.") from exc
    if not np.isfinite(result) or result < 0 or (strictly_positive and result == 0):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must be a finite {qualifier} number.")
    return result


def _resolve_device(requested: str, n_samples: int) -> torch.device:
    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("device must be one of 'auto', 'cpu', or 'mps'.")
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(
        sys.platform == "darwin"
        and mps_backend is not None
        and mps_backend.is_available()
    )
    if requested == "mps":
        if not mps_available:
            raise ValueError("device='mps' requires macOS and a PyTorch build with MPS support.")
        return torch.device("mps")
    # MPS launch overhead dominated for the small digits dataset in local
    # benchmarks; larger embeddings benefit from its parallel tensor kernels.
    if requested == "auto" and mps_available and n_samples >= 2048:
        return torch.device("mps")
    return torch.device("cpu")


def _sample_negative_targets(
    rng: np.random.RandomState,
    packed_adjacency: np.ndarray,
    negative_sources: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Sample non-neighbors with rejection and an exact dense-graph fallback."""
    targets = rng.randint(n_samples, size=negative_sources.size)
    invalid = np.ones(targets.size, dtype=bool)
    for _ in range(8):
        invalid_indices = np.flatnonzero(invalid)
        if invalid_indices.size == 0:
            break
        candidate_targets = targets[invalid_indices]
        blocked = packed_adjacency[
            negative_sources[invalid_indices], candidate_targets >> 3
        ]
        bit_mask = 1 << (candidate_targets & 7)
        accepted = (blocked & bit_mask) == 0
        invalid[invalid_indices[accepted]] = False
        invalid_indices = invalid_indices[~accepted]
        targets[invalid_indices] = rng.randint(n_samples, size=invalid_indices.size)

    for source in np.unique(negative_sources[invalid]):
        source_indices = np.flatnonzero(invalid & (negative_sources == source))
        blocked = np.unpackbits(
            packed_adjacency[source], bitorder="little", count=n_samples
        )
        candidates = np.flatnonzero(blocked == 0)
        if candidates.size == 0:
            raise ValueError("Cannot sample a negative pair for a fully connected node.")
        targets[source_indices] = rng.choice(candidates, size=source_indices.size)
    return targets


def _squared_euclidean_distance(delta: torch.Tensor) -> torch.Tensor:
    """Return squared Euclidean distances for row-wise coordinate differences."""
    return torch.sum(delta * delta, dim=1)


def _iterated_mst_edges(
    distances: np.ndarray,
    n_msts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return edge-disjoint MSTs, preserving zero-distance edges.

    Dense Prim avoids treating zero-weight edges as missing, which is important
    when the input contains duplicate samples.
    """
    n_samples = distances.shape[0]
    available = np.full(
        (n_samples, (n_samples + 7) // 8), 0xFF, dtype=np.uint8
    )
    sample_ids = np.arange(n_samples)
    available[sample_ids, sample_ids >> 3] &= np.bitwise_not(
        (1 << (sample_ids & 7)).astype(np.uint8)
    )
    edges: list[tuple[int, int]] = []
    weights: list[float] = []

    for rank in range(1, n_msts + 1):
        in_tree = np.zeros(n_samples, dtype=bool)
        best = np.full(n_samples, np.inf, dtype=np.float64)
        parent = np.full(n_samples, -1, dtype=np.intp)
        best[0] = 0.0

        for _ in range(n_samples):
            candidates = np.where(in_tree, np.inf, best)
            node = int(np.argmin(candidates))
            if not np.isfinite(candidates[node]):
                raise ValueError(
                    f"Could not construct MST {rank}: removing earlier trees left "
                    "the available graph disconnected. Reduce n_msts."
                )

            in_tree[node] = True
            if parent[node] != -1:
                other = int(parent[node])
                edges.append((other, node))
                weights.append(1.0 / rank)
                available[other, node >> 3] &= np.uint8(
                    0xFF ^ (1 << (node & 7))
                )
                available[node, other >> 3] &= np.uint8(
                    0xFF ^ (1 << (other & 7))
                )

            remaining = ~in_tree
            available_row = np.unpackbits(
                available[node], bitorder="little", count=n_samples
            ).astype(bool)
            connectable = remaining & available_row
            improve = connectable & (distances[node] < best)
            best[improve] = distances[node, improve]
            parent[improve] = node

    edge_array = np.asarray(edges, dtype=np.intp).reshape(-1, 2)
    weight_array = np.asarray(weights, dtype=np.float64)
    return edge_array, weight_array


class IteratedMinimumSpanningTreeEmbedder(TransformerMixin, BaseEstimator):
    """Learn training-row coordinates from edge-disjoint minimum spanning trees.

    This is the transductive estimator. ``transform`` returns the learned
    coordinates only for the exact training matrix in its original row order.
    Use :class:`imste.inductive.InductiveIMSTE` to project new rows.

    Parameters
    ----------
    n_msts : int, default=10
        Number of edge-disjoint minimum spanning trees to construct.
    n_components : int, default=2
        Number of embedding coordinates per sample.
    n_epochs : int, default=1000
        Number of optimization epochs.
    batch_size : int, default=4096
        Number of positive graph edges per optimization step.
    learning_rate : float, default=0.05
        Adam learning rate for coordinate optimization.
    negative_ratio : int, default=5
        Number of sampled non-neighbors per endpoint of each positive edge.
    lambda_rep : float, default=0.5
        Repulsion share in the objective, from zero to one.
    logistic_margin : float, default=1.0
        Squared-distance margin used by logistic repulsion.
    logistic_temperature : float, default=0.5
        Positive temperature controlling logistic repulsion softness.
    random_state : int or None, default=42
        Seed controlling initialization, edge ordering, and negative sampling.
    epsilon : float, default=1e-4
        Smoothing added to squared distances in the logarithmic attraction.
    device : {'auto', 'cpu', 'mps'}, default='auto'
        Compute device. ``'auto'`` selects MPS on macOS for datasets with at
        least 2048 rows and CPU otherwise.
    """

    def __init__(
        self,
        n_msts: int = 10,
        n_components: int = 2,
        n_epochs: int = 1000,
        batch_size: int = 4096,
        learning_rate: float = 0.05,
        negative_ratio: int = 5,
        lambda_rep: float = 0.5,
        random_state: int | None = 42,
        epsilon: float = 1e-4,
        device: str = "auto",
        logistic_margin: float = 1.0,
        logistic_temperature: float = 0.5,
    ) -> None:
        self.n_msts = n_msts
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.negative_ratio = negative_ratio
        self.lambda_rep = lambda_rep
        self.random_state = random_state
        self.epsilon = epsilon
        self.device = device
        self.logistic_margin = logistic_margin
        self.logistic_temperature = logistic_temperature

    def fit(self, X: object, y: object = None) -> "IteratedMinimumSpanningTreeEmbedder":
        """Fit the embedding and store coordinates for the input rows.

        Successful fits expose `graph_construction_time_`,
        `embedding_optimization_time_`, and total `fit_time_` in seconds.
        """
        fit_started = time.perf_counter()
        del y  # Unsupervised: labels are intentionally never used.
        n_msts = _positive_integer("n_msts", self.n_msts)
        n_components = _positive_integer("n_components", self.n_components)
        n_epochs = _positive_integer("n_epochs", self.n_epochs, allow_zero=True)
        batch_size = _positive_integer("batch_size", self.batch_size)
        negative_ratio = _positive_integer(
            "negative_ratio", self.negative_ratio, allow_zero=True
        )
        learning_rate = _finite_nonnegative(
            "learning_rate", self.learning_rate, strictly_positive=True
        )
        lambda_rep = _finite_nonnegative("lambda_rep", self.lambda_rep)
        if lambda_rep > 1.0:
            raise ValueError("lambda_rep must be between 0 and 1 inclusive.")
        epsilon = _finite_nonnegative("epsilon", self.epsilon, strictly_positive=True)
        logistic_margin = _finite_nonnegative("logistic_margin", self.logistic_margin)
        logistic_temperature = _finite_nonnegative(
            "logistic_temperature", self.logistic_temperature, strictly_positive=True
        )
        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)
        n_samples = X_checked.shape[0]
        if n_samples < 2:
            raise ValueError("X must contain at least two samples to construct an MST.")
        compute_device = _resolve_device(self.device, n_samples)
        compute_dtype = torch.float32 if compute_device.type == "mps" else torch.float64

        graph_started = time.perf_counter()
        distances = cdist(X_checked, X_checked, metric="euclidean")
        if not np.isfinite(distances).all():
            raise ValueError("Pairwise distances overflowed; rescale X before fitting.")
        np.fill_diagonal(distances, 0.0)
        edges, edge_weights = _iterated_mst_edges(
            distances, n_msts
        )
        del distances
        self.graph_construction_time_ = time.perf_counter() - graph_started

        embedding_started = time.perf_counter()
        rng = np.random.RandomState(self.random_state)
        torch_generator = torch.Generator(device="cpu")
        seed = int(rng.randint(0, np.iinfo(np.int32).max))
        torch_generator.manual_seed(seed)
        initial = torch.randn(
            (n_samples, n_components), generator=torch_generator, dtype=compute_dtype
        ).to(compute_device)
        coordinates = torch.nn.Parameter(initial * 1e-3)
        optimizer = torch.optim.Adam([coordinates], lr=learning_rate)

        edge_array = edges
        edge_weights_t = torch.as_tensor(
            edge_weights, dtype=compute_dtype, device=compute_device
        )
        attraction_scale = 1.0
        if edge_weights.size:
            # Minibatches are sampled uniformly by edge. Multiplying their
            # mean by |E| / sum(w) gives an unbiased gradient estimate of the
            # graph-wide weight-normalized attraction objective.
            attraction_scale = len(edge_weights) / float(np.sum(edge_weights))
        edge_sources = edge_array[:, 0]
        edge_targets = edge_array[:, 1]
        packed_adjacency = np.zeros(
            (n_samples, (n_samples + 7) // 8), dtype=np.uint8
        )
        sample_ids = np.arange(n_samples)
        np.bitwise_or.at(
            packed_adjacency,
            (sample_ids, sample_ids >> 3),
            (1 << (sample_ids & 7)).astype(np.uint8),
        )
        np.bitwise_or.at(
            packed_adjacency,
            (edge_sources, edge_targets >> 3),
            (1 << (edge_targets & 7)).astype(np.uint8),
        )
        np.bitwise_or.at(
            packed_adjacency,
            (edge_targets, edge_sources >> 3),
            (1 << (edge_sources & 7)).astype(np.uint8),
        )
        node_degrees = np.bincount(
            np.concatenate((edge_sources, edge_targets)), minlength=n_samples
        )
        negative_counts = n_samples - 1 - node_degrees

        for _ in range(n_epochs):
            order = rng.permutation(len(edge_array))
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                src = edge_sources[batch]
                dst = edge_targets[batch]
                src_t = torch.as_tensor(src, dtype=torch.long, device=compute_device)
                dst_t = torch.as_tensor(dst, dtype=torch.long, device=compute_device)
                batch_t = torch.as_tensor(batch, dtype=torch.long, device=compute_device)
                weights_t = edge_weights_t[batch_t] * attraction_scale

                positive_delta = coordinates[src_t] - coordinates[dst_t]
                positive_d2 = _squared_euclidean_distance(positive_delta)
                attraction = log_attraction_loss(
                    positive_d2, weights_t, epsilon
                )

                # Positive edges are undirected, so both endpoints contribute
                # negative anchors regardless of Prim's stored orientation.
                edge_endpoints = np.concatenate((src, dst))
                eligible_sources = edge_endpoints[
                    negative_counts[edge_endpoints] > 0
                ]
                if negative_ratio and eligible_sources.size:
                    negative_sources = np.repeat(eligible_sources, negative_ratio)
                    negative_targets = _sample_negative_targets(
                        rng, packed_adjacency, negative_sources, n_samples
                    )
                    ni = torch.as_tensor(
                        negative_sources, dtype=torch.long, device=compute_device
                    )
                    nj = torch.as_tensor(
                        negative_targets, dtype=torch.long, device=compute_device
                    )
                    negative_delta = coordinates[ni] - coordinates[nj]
                    negative_d2 = _squared_euclidean_distance(negative_delta)
                    repulsion = logistic_repulsion_loss(
                        negative_d2,
                        logistic_margin,
                        logistic_temperature,
                    )
                else:
                    # Some small or dense graphs have no eligible negatives.
                    repulsion = coordinates.sum() * 0.0

                loss = (1.0 - lambda_rep) * attraction + lambda_rep * repulsion
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    coordinates -= coordinates.mean(dim=0, keepdim=True)

        if compute_device.type == "mps":
            torch.mps.synchronize()
        self.embedding_optimization_time_ = time.perf_counter() - embedding_started
        self.n_features_in_ = X_checked.shape[1]
        self.device_ = str(compute_device)
        if hasattr(X, "columns") and all(isinstance(c, str) for c in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self.X_fit_ = X_checked.copy()
        self.embedding_ = coordinates.detach().cpu().numpy().copy()
        self.graph_edges_ = edge_array.copy()
        self.graph_weights_ = edge_weights.copy()
        self.fit_time_ = time.perf_counter() - fit_started
        return self

    def transform(self, X: object) -> np.ndarray:
        """Return fitted coordinates for the exact training rows only."""
        check_is_fitted(self, attributes=["embedding_", "X_fit_"])
        self._check_feature_names(X)
        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but this estimator was fitted "
                f"with {self.n_features_in_} features."
            )
        if X_checked.shape == self.X_fit_.shape and np.array_equal(
            X_checked, self.X_fit_
        ):
            return self.embedding_.copy()
        raise ValueError(
            "transform is available only for the original training rows in the "
            "same order. Use imste.inductive.InductiveIMSTE to project new rows."
        )

    def _check_feature_names(self, X: object) -> None:
        fitted_names = getattr(self, "feature_names_in_", None)
        if fitted_names is None:
            return
        input_columns = getattr(X, "columns", None)
        if input_columns is None:
            raise ValueError(
                "X must be a DataFrame with the same named columns used during fit."
            )
        input_names = np.asarray(input_columns, dtype=object)
        if not np.array_equal(input_names, fitted_names):
            raise ValueError(
                "X must have the same feature names in the same order as during fit."
            )
