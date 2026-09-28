"""Scikit-learn estimator for IMSTE embeddings."""

from __future__ import annotations

import numbers
import sys
import time
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .losses import (
    bernoulli_repulsion_loss,
    direct_attraction_loss,
    huber_attraction_loss,
    inverse_distance_repulsion_loss,
    logistic_attraction_loss,
    logistic_repulsion_loss,
    log_attraction_loss,
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


def _validate_loss_output(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 0 or not value.requires_grad:
        raise ValueError(
            f"{name} must return a scalar torch.Tensor that is differentiable "
            "with respect to the supplied distances."
        )
    return value


class _ResidualBlock(nn.Module):
    """A fully connected residual block with dropout on its update."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.linear1(inputs)
        update = self.dropout(self.activation(update))
        update = self.linear2(update)
        return self.activation(inputs + self.dropout(update))


class _ResidualProjector(nn.Module):
    """Map standardized input features to standardized embedding coordinates."""

    def __init__(
        self, n_features: int, n_components: int, width: int, n_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.input_layer = nn.Linear(n_features, width)
        self.activation = nn.ReLU()
        self.blocks = nn.Sequential(
            *(_ResidualBlock(width, dropout) for _ in range(n_layers))
        )
        self.output_layer = nn.Linear(width, n_components)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.input_layer(inputs))
        return self.output_layer(self.blocks(hidden))


def _squared_euclidean_distance(delta: torch.Tensor) -> torch.Tensor:
    """Return squared Euclidean distances for row-wise coordinate differences."""
    return torch.sum(delta * delta, dim=1)


def _distance_value(
    squared_distance: torch.Tensor,
    distance_type: str,
    epsilon: float,
) -> torch.Tensor:
    """Convert squared Euclidean distance to the configured scalar distance."""
    if distance_type == "euclidean":
        return torch.sqrt(squared_distance + epsilon) - epsilon**0.5
    return squared_distance


def _iterated_mst_edges(
    distances: np.ndarray,
    n_msts: int,
) -> tuple[np.ndarray, np.ndarray]:
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


class IteratedMinimumSpanningTreeEmbedder(TransformerMixin, BaseEstimator):
    """Embed a dataset with IMSTE using edge-disjoint minimum spanning trees.

    The learned coordinates are attached to the training rows. By default,
    :meth:`transform` accepts only the original training matrix in its original
    row order. Set ``transform_method="resnet"`` to learn a residual-network
    projection for new rows.

    Parameters
    ----------
    n_msts : int, default=10
        Number of edge-disjoint minimum spanning trees to construct.
    attraction_normalization : {'weight_sum'}, default='weight_sum'
        Attraction is normalized by the sum of graph edge weights. This option
        is fixed to ``'weight_sum'``.
    distance_type : {'euclidean', 'squared'}, default='squared'
        Distance value passed to both attraction and repulsion losses.
    attraction_dampening : {'direct', 'log', 'logistic', 'huber'}, default='log'
        Positive-edge loss shape. ``'direct'`` uses the selected distance,
        ``'log'`` applies ``log(1 + distance + epsilon)``, ``'logistic'`` uses
        the logistic margin loss, and ``'huber'`` uses a Huber penalty with
        ``delta=1``. An ``attraction_loss_fn`` overrides this selection.
    logistic_margin : float, default=1.0
        Distance margin used by logistic attraction and repulsion losses.
    logistic_temperature : float, default=0.5
        Positive temperature controlling the softness of the logistic losses.
    n_components : int, default=2
        Number of embedding coordinates per sample.
    n_epochs : int, default=1000
        Number of optimization epochs.
    batch_size : int, default=4096
        Number of positive edges per optimization step.
    learning_rate : float, default=0.05
        Adam learning rate.
    negative_ratio : int, default=5
        Number of sampled non-neighbors per endpoint of each positive edge.
    lambda_rep : float, default=0.5
        Repulsion share in the convex combination of attraction and repulsion.
        Must be between zero and one; attraction receives weight ``1-lambda_rep``.
    random_state : int or None, default=42
        Seed controlling initialization, edge shuffling, and negative sampling.
    epsilon : float, default=1e-4
        Smoothing constant in the pairwise edge probability
        and distance-based losses. It keeps probabilities strictly between
        zero and one for both positive and negative pairs.
    device : {'auto', 'cpu', 'mps'}, default='auto'
        Compute device for embedding optimization. ``auto`` selects Apple's
        Metal Performance Shaders (MPS) backend on macOS for datasets with at
        least 2048 samples; smaller datasets use the CPU to avoid GPU launch
        overhead. Explicitly select ``mps`` or ``cpu`` to override this choice.
    transform_method : {'direct', 'resnet'}, default='direct'
        ``'direct'`` returns fitted coordinates for the original training rows
        and rejects new rows. ``'resnet'`` additionally fits a residual network
        to the learned coordinates and uses it to transform new rows.
    resnet_n_layers : int, default=3
        Number of fully connected residual blocks in the projection network.
    resnet_layer_size : int, default=256
        Hidden width used by the projection network and its residual blocks.
    resnet_dropout : float, default=0.1
        Dropout probability in each residual block; must be in ``[0, 1)``.
    resnet_epochs : int, default=200
        Number of epochs used to fit the projection network.
    resnet_batch_size : int, default=256
        Number of training rows per projection-network update.
    resnet_learning_rate : float, default=0.001
        AdamW learning rate for the projection network.
    resnet_weight_decay : float, default=0.0001
        AdamW L2 weight decay for the projection network.
    attraction_loss_fn : callable or None, default=None
        Optional callable with signature ``fn(positive_squared_distances,
        edge_weights)`` that returns a scalar differentiable PyTorch tensor.
        When supplied, this overrides the built-in loss selected by
        ``attraction_dampening``.
    repulsion_loss_fn : callable or None, default=None
        Optional callable with signature ``fn(negative_squared_distances,
        epsilon)`` that returns a scalar differentiable PyTorch tensor. The
        default is :func:`mst_embedding.bernoulli_repulsion_loss`, which
        evaluates the negative-pair loss ``-log(1 - q)``.
    repulsion_type : {'bernoulli', 'inverse_distance', 'logistic'}, default='bernoulli'
        Built-in negative-pair loss. ``'inverse_distance'`` selects the
        ``1 / (1 + distance + epsilon)`` penalty; ``'logistic'`` selects the
        negative logistic margin loss. This is ignored when
        ``repulsion_loss_fn`` is supplied.
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
        attraction_normalization: str = "weight_sum",
        distance_type: str = "squared",
        attraction_dampening: str = "log",
        logistic_margin: float = 1.0,
        logistic_temperature: float = 0.5,
        attraction_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
        repulsion_loss_fn: Callable[[torch.Tensor, float], torch.Tensor] | None = None,
        repulsion_type: str = "bernoulli",
        transform_method: str = "direct",
        resnet_n_layers: int = 3,
        resnet_layer_size: int = 256,
        resnet_dropout: float = 0.1,
        resnet_epochs: int = 200,
        resnet_batch_size: int = 256,
        resnet_learning_rate: float = 0.001,
        resnet_weight_decay: float = 0.0001,
    ) -> None:
        self.n_msts = n_msts
        self.attraction_normalization = attraction_normalization
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.negative_ratio = negative_ratio
        self.lambda_rep = lambda_rep
        self.random_state = random_state
        self.epsilon = epsilon
        self.device = device
        self.distance_type = distance_type
        self.attraction_dampening = attraction_dampening
        self.logistic_margin = logistic_margin
        self.logistic_temperature = logistic_temperature
        self.attraction_loss_fn = attraction_loss_fn
        self.repulsion_loss_fn = repulsion_loss_fn
        self.repulsion_type = repulsion_type
        self.transform_method = transform_method
        self.resnet_n_layers = resnet_n_layers
        self.resnet_layer_size = resnet_layer_size
        self.resnet_dropout = resnet_dropout
        self.resnet_epochs = resnet_epochs
        self.resnet_batch_size = resnet_batch_size
        self.resnet_learning_rate = resnet_learning_rate
        self.resnet_weight_decay = resnet_weight_decay

    def fit(self, X: object, y: object = None) -> "IteratedMinimumSpanningTreeEmbedder":
        """Fit the embedding and store coordinates for the input rows.

        Successful fits expose `graph_construction_time_`,
        `embedding_optimization_time_`, and total `fit_time_` in seconds.
        """
        fit_started = time.perf_counter()
        del y  # Unsupervised: labels are intentionally never used.
        n_msts = _positive_integer("n_msts", self.n_msts)
        if self.transform_method not in {"direct", "resnet"}:
            raise ValueError("transform_method must be 'direct' or 'resnet'.")
        resnet_n_layers = _positive_integer("resnet_n_layers", self.resnet_n_layers)
        resnet_layer_size = _positive_integer(
            "resnet_layer_size", self.resnet_layer_size
        )
        resnet_epochs = _positive_integer("resnet_epochs", self.resnet_epochs)
        resnet_batch_size = _positive_integer(
            "resnet_batch_size", self.resnet_batch_size
        )
        resnet_learning_rate = _finite_nonnegative(
            "resnet_learning_rate", self.resnet_learning_rate, strictly_positive=True
        )
        resnet_weight_decay = _finite_nonnegative(
            "resnet_weight_decay", self.resnet_weight_decay
        )
        resnet_dropout = _finite_nonnegative(
            "resnet_dropout", self.resnet_dropout
        )
        if resnet_dropout >= 1:
            raise ValueError("resnet_dropout must be a finite number in [0, 1).")
        if self.attraction_normalization != "weight_sum":
            raise ValueError("attraction_normalization must be 'weight_sum'.")
        if self.distance_type not in {"euclidean", "squared"}:
            raise ValueError(
                "distance_type must be 'euclidean' or 'squared'."
            )
        if self.attraction_dampening not in {"direct", "log", "logistic", "huber"}:
            raise ValueError(
                "attraction_dampening must be 'direct', 'log', 'logistic', or 'huber'."
            )
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
        if self.attraction_loss_fn is not None and not callable(self.attraction_loss_fn):
            raise ValueError("attraction_loss_fn must be callable or None.")
        if self.repulsion_loss_fn is not None and not callable(self.repulsion_loss_fn):
            raise ValueError("repulsion_loss_fn must be callable or None.")
        if self.repulsion_type not in {"bernoulli", "inverse_distance", "logistic"}:
            raise ValueError(
                "repulsion_type must be 'bernoulli', 'inverse_distance', or 'logistic'."
            )
        use_builtin_attraction_loss = self.attraction_loss_fn is None
        attraction_loss_fns = {
            "direct": direct_attraction_loss,
            "log": log_attraction_loss,
            "logistic": logistic_attraction_loss,
            "huber": huber_attraction_loss,
        }
        attraction_loss_fn = attraction_loss_fns[self.attraction_dampening]
        if not use_builtin_attraction_loss:
            attraction_loss_fn = self.attraction_loss_fn
        if self.repulsion_loss_fn is not None:
            repulsion_loss_fn = self.repulsion_loss_fn
        elif self.repulsion_type == "inverse_distance":
            repulsion_loss_fn = inverse_distance_repulsion_loss
        elif self.repulsion_type == "logistic":
            repulsion_loss_fn = logistic_repulsion_loss
        else:
            repulsion_loss_fn = bernoulli_repulsion_loss
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
        adjacency = np.zeros((n_samples, n_samples), dtype=bool)
        adjacency[edge_sources, edge_targets] = True
        adjacency[edge_targets, edge_sources] = True
        sample_ids = np.arange(n_samples)
        negative_counts = np.count_nonzero(
            ~adjacency & (sample_ids[None, :] != sample_ids[:, None]), axis=1
        )
        max_negative_count = int(negative_counts.max(initial=0))
        index_dtype = np.int32 if n_samples <= np.iinfo(np.int32).max else np.intp
        negative_candidates = np.empty(
            (n_samples, max_negative_count), dtype=index_dtype
        )
        for sample_id in range(n_samples):
            candidates = np.flatnonzero(
                ~adjacency[sample_id] & (sample_ids != sample_id)
            )
            negative_candidates[sample_id, : candidates.size] = candidates
        del adjacency, sample_ids

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
                positive_distance = _distance_value(
                    positive_d2, self.distance_type, epsilon
                )
                if use_builtin_attraction_loss:
                    if self.attraction_dampening == "logistic":
                        attraction_value = attraction_loss_fn(
                            positive_distance,
                            weights_t,
                            epsilon,
                            logistic_margin,
                            logistic_temperature,
                        )
                    else:
                        attraction_value = attraction_loss_fn(
                            positive_distance, weights_t, epsilon
                        )
                else:
                    attraction_value = attraction_loss_fn(positive_d2, weights_t)
                attraction = _validate_loss_output(
                    "attraction_loss_fn", attraction_value
                )

                # Positive edges are undirected, so both endpoints contribute
                # negative anchors regardless of Prim's stored orientation.
                edge_endpoints = np.concatenate((src, dst))
                eligible_sources = edge_endpoints[
                    negative_counts[edge_endpoints] > 0
                ]
                if negative_ratio and eligible_sources.size:
                    eligible_counts = negative_counts[eligible_sources]
                    sampled_offsets = (
                        rng.random_sample((eligible_sources.size, negative_ratio))
                        * eligible_counts[:, None]
                    ).astype(np.intp)
                    negative_targets = negative_candidates[
                        eligible_sources[:, None], sampled_offsets
                    ].reshape(-1)
                    negative_sources = np.repeat(eligible_sources, negative_ratio)
                    ni = torch.as_tensor(
                        negative_sources, dtype=torch.long, device=compute_device
                    )
                    nj = torch.as_tensor(
                        negative_targets, dtype=torch.long, device=compute_device
                    )
                    negative_delta = coordinates[ni] - coordinates[nj]
                    negative_d2 = _squared_euclidean_distance(negative_delta)
                    negative_distance = _distance_value(
                        negative_d2, self.distance_type, epsilon
                    )
                    if self.repulsion_loss_fn is not None:
                        negative_loss = repulsion_loss_fn(negative_d2, epsilon)
                    elif self.repulsion_type == "logistic":
                        negative_loss = repulsion_loss_fn(
                            negative_distance,
                            epsilon,
                            logistic_margin,
                            logistic_temperature,
                        )
                    else:
                        negative_loss = repulsion_loss_fn(negative_distance, epsilon)
                    repulsion = _validate_loss_output(
                        "repulsion_loss_fn",
                        negative_loss,
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
        for attribute in (
            "resnet_",
            "resnet_X_mean_",
            "resnet_X_scale_",
            "resnet_embedding_mean_",
            "resnet_embedding_scale_",
            "resnet_fit_time_",
            "resnet_training_loss_",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)
        if self.transform_method == "resnet":
            resnet_started = time.perf_counter()
            self.resnet_X_mean_ = X_checked.mean(axis=0)
            self.resnet_X_scale_ = X_checked.std(axis=0)
            self.resnet_X_scale_[self.resnet_X_scale_ == 0.0] = 1.0
            self.resnet_embedding_mean_ = self.embedding_.mean(axis=0)
            self.resnet_embedding_scale_ = self.embedding_.std(axis=0)
            self.resnet_embedding_scale_[self.resnet_embedding_scale_ == 0.0] = 1.0

            X_resnet = np.asarray(
                (X_checked - self.resnet_X_mean_) / self.resnet_X_scale_,
                dtype=np.float32,
            )
            y_resnet = np.asarray(
                (self.embedding_ - self.resnet_embedding_mean_)
                / self.resnet_embedding_scale_,
                dtype=np.float32,
            )
            X_resnet_t = torch.as_tensor(X_resnet, device=compute_device)
            y_resnet_t = torch.as_tensor(y_resnet, device=compute_device)
            resnet_seed = int(rng.randint(0, np.iinfo(np.int32).max))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(resnet_seed)
                self.resnet_ = _ResidualProjector(
                    X_checked.shape[1],
                    n_components,
                    resnet_layer_size,
                    resnet_n_layers,
                    resnet_dropout,
                ).to(compute_device)
                resnet_optimizer = torch.optim.AdamW(
                    self.resnet_.parameters(),
                    lr=resnet_learning_rate,
                    weight_decay=resnet_weight_decay,
                )
                self.resnet_.train()
                for _ in range(resnet_epochs):
                    order = rng.permutation(n_samples)
                    for start in range(0, n_samples, resnet_batch_size):
                        batch = order[start : start + resnet_batch_size]
                        batch_t = torch.as_tensor(
                            batch, dtype=torch.long, device=compute_device
                        )
                        prediction = self.resnet_(X_resnet_t[batch_t])
                        loss = torch.mean(
                            (prediction - y_resnet_t[batch_t]) ** 2
                        )
                        resnet_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        resnet_optimizer.step()
                self.resnet_.eval()
            if compute_device.type == "mps":
                torch.mps.synchronize()
            self.resnet_fit_time_ = time.perf_counter() - resnet_started
            self.resnet_training_loss_ = float(loss.detach().cpu())
        self.fit_time_ = time.perf_counter() - fit_started
        return self

    def transform(self, X: object) -> np.ndarray:
        """Return fitted coordinates or project new rows with the selected model."""
        check_is_fitted(self, attributes=["embedding_", "X_fit_"])
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
        if self.transform_method != "resnet":
            raise ValueError(
                "transform is available only for the original training rows in "
                "the same order when transform_method='direct'. Set "
                "transform_method='resnet' to project new rows."
            )
        return self.transform_with_resnet(X_checked)

    def transform_with_resnet(self, X: object) -> np.ndarray:
        """Apply the fitted ResNet projection, including to training rows."""
        check_is_fitted(self, attributes=["resnet_"])
        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but this estimator was fitted "
                f"with {self.n_features_in_} features."
            )
        X_resnet = np.asarray(
            (X_checked - self.resnet_X_mean_) / self.resnet_X_scale_,
            dtype=np.float32,
        )
        predictions: list[np.ndarray] = []
        device = torch.device(self.device_)
        self.resnet_.eval()
        with torch.no_grad():
            for start in range(0, len(X_resnet), self.resnet_batch_size):
                batch = torch.as_tensor(
                    X_resnet[start : start + self.resnet_batch_size],
                    device=device,
                )
                predictions.append(self.resnet_(batch).cpu().numpy())
        transformed = np.concatenate(predictions, axis=0).astype(np.float64)
        return (
            transformed * self.resnet_embedding_scale_
            + self.resnet_embedding_mean_
        )
