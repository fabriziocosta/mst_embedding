"""Scikit-learn estimator for an iterated-MST 2D embedding."""

from __future__ import annotations

import numbers

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


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


def _iterated_mst_edges(distances: np.ndarray, n_msts: int) -> tuple[np.ndarray, np.ndarray]:
    """Return edge-disjoint MSTs, preserving zero-distance edges.

    Dense Prim avoids treating zero-weight edges as missing, which is important
    when the input contains duplicate samples.
    """
    n_samples = distances.shape[0]
    available = ~np.eye(n_samples, dtype=bool)
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
                available[other, node] = False
                available[node, other] = False

            remaining = ~in_tree
            connectable = remaining & available[node]
            improve = connectable & (distances[node] < best)
            best[improve] = distances[node, improve]
            parent[improve] = node

    edge_array = np.asarray(edges, dtype=np.intp).reshape(-1, 2)
    weight_array = np.asarray(weights, dtype=np.float64)
    return edge_array, weight_array


class IteratedMSTEmbedding(TransformerMixin, BaseEstimator):
    """Embed a dataset in two dimensions using edge-disjoint minimum trees.

    The learned coordinates are attached to the training rows. Since the
    objective jointly optimizes all rows, this estimator does not define an
    out-of-sample projection: :meth:`transform` accepts only the original
    training matrix, in its original row order.

    Parameters
    ----------
    n_msts : int, default=4
        Number of edge-disjoint minimum spanning trees to construct.
    n_epochs : int, default=500
        Number of optimization epochs.
    batch_size : int, default=4096
        Number of positive edges per optimization step.
    learning_rate : float, default=0.05
        Adam learning rate.
    negative_ratio : int, default=4
        Number of sampled non-neighbors per positive edge.
    lambda_rep : float, default=1.0
        Coefficient multiplying the repulsive loss.
    random_state : int or None, default=42
        Seed controlling initialization, edge shuffling, and negative sampling.
    epsilon : float, default=1e-4
        Stabilizing constant in the repulsive loss.
    """

    def __init__(
        self,
        n_msts: int = 4,
        n_epochs: int = 500,
        batch_size: int = 4096,
        learning_rate: float = 0.05,
        negative_ratio: int = 4,
        lambda_rep: float = 1.0,
        random_state: int | None = 42,
        epsilon: float = 1e-4,
    ) -> None:
        self.n_msts = n_msts
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.negative_ratio = negative_ratio
        self.lambda_rep = lambda_rep
        self.random_state = random_state
        self.epsilon = epsilon

    def fit(self, X: object, y: object = None) -> "IteratedMSTEmbedding":
        """Fit the embedding and store coordinates for the input rows."""
        del y  # Unsupervised: labels are intentionally never used.
        n_msts = _positive_integer("n_msts", self.n_msts)
        n_epochs = _positive_integer("n_epochs", self.n_epochs, allow_zero=True)
        batch_size = _positive_integer("batch_size", self.batch_size)
        negative_ratio = _positive_integer(
            "negative_ratio", self.negative_ratio, allow_zero=True
        )
        learning_rate = _finite_nonnegative(
            "learning_rate", self.learning_rate, strictly_positive=True
        )
        lambda_rep = _finite_nonnegative("lambda_rep", self.lambda_rep)
        epsilon = _finite_nonnegative("epsilon", self.epsilon, strictly_positive=True)

        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)
        n_samples = X_checked.shape[0]
        if n_samples < 2:
            raise ValueError("X must contain at least two samples to construct an MST.")

        distances = cdist(X_checked, X_checked, metric="euclidean")
        if not np.isfinite(distances).all():
            raise ValueError("Pairwise distances overflowed; rescale X before fitting.")
        np.fill_diagonal(distances, 0.0)
        edges, edge_weights = _iterated_mst_edges(distances, n_msts)

        rng = np.random.RandomState(self.random_state)
        torch_generator = torch.Generator(device="cpu")
        seed = int(rng.randint(0, np.iinfo(np.int32).max))
        torch_generator.manual_seed(seed)
        initial = torch.randn((n_samples, 2), generator=torch_generator, dtype=torch.float64)
        coordinates = torch.nn.Parameter(initial * 1e-3)
        optimizer = torch.optim.Adam([coordinates], lr=learning_rate)

        edge_array = edges
        edge_weights_t = torch.as_tensor(edge_weights, dtype=torch.float64)
        edge_sources = edge_array[:, 0]
        edge_targets = edge_array[:, 1]
        adjacency = np.zeros((n_samples, n_samples), dtype=bool)
        adjacency[edge_sources, edge_targets] = True
        adjacency[edge_targets, edge_sources] = True
        valid_negative_nodes = [np.flatnonzero(~adjacency[i] & (np.arange(n_samples) != i)) for i in range(n_samples)]

        for _ in range(n_epochs):
            order = rng.permutation(len(edge_array))
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                src = edge_sources[batch]
                dst = edge_targets[batch]
                src_t = torch.as_tensor(src, dtype=torch.long)
                dst_t = torch.as_tensor(dst, dtype=torch.long)
                weights_t = edge_weights_t[batch]

                positive_delta = coordinates[src_t] - coordinates[dst_t]
                positive_d2 = torch.sum(positive_delta * positive_delta, dim=1)
                attraction = torch.mean(weights_t * torch.log1p(positive_d2))

                negative_sources: list[int] = []
                negative_targets: list[int] = []
                if negative_ratio:
                    for source in src:
                        choices = valid_negative_nodes[int(source)]
                        if choices.size:
                            sampled = rng.choice(choices, size=negative_ratio, replace=True)
                            negative_sources.extend([int(source)] * negative_ratio)
                            negative_targets.extend(sampled.tolist())

                if negative_sources:
                    ni = torch.as_tensor(negative_sources, dtype=torch.long)
                    nj = torch.as_tensor(negative_targets, dtype=torch.long)
                    negative_delta = coordinates[ni] - coordinates[nj]
                    negative_d2 = torch.sum(negative_delta * negative_delta, dim=1)
                    repulsion = torch.mean(1.0 / (1.0 + negative_d2 + epsilon))
                else:
                    # Some small or dense graphs have no eligible negatives.
                    repulsion = coordinates.sum() * 0.0

                loss = attraction + lambda_rep * repulsion
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    coordinates -= coordinates.mean(dim=0, keepdim=True)

        self.n_features_in_ = X_checked.shape[1]
        if hasattr(X, "columns") and all(isinstance(c, str) for c in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self.X_fit_ = X_checked.copy()
        self.embedding_ = coordinates.detach().cpu().numpy().copy()
        self.graph_edges_ = edge_array.copy()
        self.graph_weights_ = edge_weights.copy()
        return self

    def transform(self, X: object) -> np.ndarray:
        """Return stored coordinates for the exact training matrix only."""
        check_is_fitted(self, attributes=["embedding_", "X_fit_"])
        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)
        if X_checked.shape != self.X_fit_.shape or not np.array_equal(X_checked, self.X_fit_):
            raise ValueError(
                "transform is available only for the original training rows in "
                "the same order; this estimator has no out-of-sample projection."
            )
        return self.embedding_.copy()
