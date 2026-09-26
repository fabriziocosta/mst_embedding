"""Scikit-learn estimator for IMSTE embeddings."""

from __future__ import annotations

import numbers
import sys
from collections.abc import Callable

import numpy as np
import torch
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.validation import check_array, check_is_fitted

from .losses import inverse_distance_repulsion_loss, log_attraction_loss


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


def _iterated_mst_edges(
    distances: np.ndarray,
    n_msts: int,
    rank_weight_exponent: float = 1.0,
    *,
    allow_partial: bool = False,
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
        rank_edge_start = len(edges)
        in_tree = np.zeros(n_samples, dtype=bool)
        best = np.full(n_samples, np.inf, dtype=np.float64)
        parent = np.full(n_samples, -1, dtype=np.intp)
        best[0] = 0.0
        completed = True

        for _ in range(n_samples):
            candidates = np.where(in_tree, np.inf, best)
            node = int(np.argmin(candidates))
            if not np.isfinite(candidates[node]):
                if allow_partial:
                    completed = False
                    break
                raise ValueError(
                    f"Could not construct MST {rank}: removing earlier trees left "
                    "the available graph disconnected. Reduce n_msts."
                )

            in_tree[node] = True
            if parent[node] != -1:
                other = int(parent[node])
                edges.append((other, node))
                weights.append(rank ** (-rank_weight_exponent))
                available[other, node] = False
                available[node, other] = False

            remaining = ~in_tree
            connectable = remaining & available[node]
            improve = connectable & (distances[node] < best)
            best[improve] = distances[node, improve]
            parent[improve] = node
        if not completed:
            del edges[rank_edge_start:]
            del weights[rank_edge_start:]
            break

    edge_array = np.asarray(edges, dtype=np.intp).reshape(-1, 2)
    weight_array = np.asarray(weights, dtype=np.float64)
    return edge_array, weight_array


def _local_cluster_pair_imst(
    X: np.ndarray,
    sample_indices: np.ndarray,
    n_local_msts: int,
    rank_weight_exponent: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the local IMST on the union of two clusters, in global indices."""
    if sample_indices.size < 2:
        return np.empty((0, 2), dtype=np.intp), np.empty(0, dtype=np.float64)
    local_X = X[sample_indices]
    distances = cdist(local_X, local_X, metric="euclidean")
    if not np.isfinite(distances).all():
        raise ValueError("Pairwise distances overflowed; rescale X before fitting.")
    np.fill_diagonal(distances, 0.0)
    local_edges, weights = _iterated_mst_edges(
        distances, n_local_msts, rank_weight_exponent, allow_partial=True
    )
    return sample_indices[local_edges], weights


def _hierarchical_imst_edges(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
    n_clusters: int,
    n_coarse_msts: int,
    n_local_msts: int,
    rank_weight_exponent: float,
    n_jobs: int,
    coarse_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build centroid IMSTs then union parallel IMSTs over adjacent clusters."""
    members = [np.flatnonzero(cluster_labels == cluster) for cluster in range(n_clusters)]
    if coarse_edges is None:
        coarse_distances = cdist(cluster_centers, cluster_centers, metric="euclidean")
        np.fill_diagonal(coarse_distances, 0.0)
        coarse_edges, _ = _iterated_mst_edges(
            coarse_distances, n_coarse_msts, rank_weight_exponent
        )
    jobs = [
        delayed(_local_cluster_pair_imst)(
            X,
            np.concatenate((members[a], members[b])),
            n_local_msts,
            rank_weight_exponent,
        )
        for a, b in coarse_edges
    ]
    local_results = Parallel(n_jobs=n_jobs)(jobs)

    strongest_weight: dict[tuple[int, int], float] = {}
    for edges, weights in local_results:
        for (u, v), weight in zip(edges, weights):
            key = (int(min(u, v)), int(max(u, v)))
            strongest_weight[key] = max(strongest_weight.get(key, -np.inf), float(weight))
    ordered_edges = sorted(strongest_weight)
    edges = np.asarray(ordered_edges, dtype=np.intp).reshape(-1, 2)
    weights = np.asarray([strongest_weight[e] for e in ordered_edges], dtype=np.float64)
    return edges, weights


class IteratedMinimumSpanningTreeEmbedder(TransformerMixin, BaseEstimator):
    """Embed a dataset with IMSTE using edge-disjoint minimum spanning trees.

    The learned coordinates are attached to the training rows. Since the
    objective jointly optimizes all rows, this estimator does not define an
    out-of-sample projection: :meth:`transform` accepts only the original
    training matrix, in its original row order.

    Parameters
    ----------
    n_msts : int, default=8
        Number of edge-disjoint minimum spanning trees to construct.
    rank_weight_exponent : float, default=1.0
        Exponent controlling how edge weights decay with MST rank. An edge
        first appearing in rank ``r`` receives weight ``r**(-rank_weight_exponent)``.
        Set to 0 for equal weights across ranks; the default of 1 gives the
        original inverse-rank weighting.
    graph_mode : {'exact', 'hierarchical'}, default='exact'
        Select the exact graph or a MiniBatchKMeans-based hierarchical approximation.
    n_clusters : int, default=100
        Number of MiniBatchKMeans clusters for hierarchical graph construction.
    n_coarse_msts : int, default=4
        Number of edge-disjoint MSTs over cluster centroids.
    n_local_msts : int, default=8
        Number of edge-disjoint MSTs requested for each adjacent cluster pair.
    n_jobs : int, default=-1
        Parallel workers used for independent local graph construction.
    minibatch_size : int, default=1024
        MiniBatchKMeans batch size.
    n_components : int, default=2
        Number of embedding coordinates per sample.
    n_epochs : int, default=1000
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
    device : {'auto', 'cpu', 'mps'}, default='auto'
        Compute device for embedding optimization. ``auto`` selects Apple's
        Metal Performance Shaders (MPS) backend on macOS for datasets with at
        least 2048 samples; smaller datasets use the CPU to avoid GPU launch
        overhead. Explicitly select ``mps`` or ``cpu`` to override this choice.
    attraction_loss_fn : callable or None, default=None
        Optional callable with signature ``fn(positive_squared_distances,
        edge_weights)`` that returns a scalar differentiable PyTorch tensor.
        The default is :func:`mst_embedding.log_attraction_loss`.
    repulsion_loss_fn : callable or None, default=None
        Optional callable with signature ``fn(negative_squared_distances,
        epsilon)`` that returns a scalar differentiable PyTorch tensor. The
        default is :func:`mst_embedding.inverse_distance_repulsion_loss`.
    """

    def __init__(
        self,
        n_msts: int = 8,
        n_components: int = 2,
        n_epochs: int = 1000,
        batch_size: int = 4096,
        learning_rate: float = 0.05,
        negative_ratio: int = 4,
        lambda_rep: float = 1.0,
        random_state: int | None = 42,
        epsilon: float = 1e-4,
        device: str = "auto",
        rank_weight_exponent: float = 1.0,
        attraction_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
        repulsion_loss_fn: Callable[[torch.Tensor, float], torch.Tensor] | None = None,
        graph_mode: str = "exact",
        n_clusters: int = 100,
        n_coarse_msts: int = 4,
        n_local_msts: int = 8,
        n_jobs: int = -1,
        minibatch_size: int = 1024,
    ) -> None:
        self.n_msts = n_msts
        self.rank_weight_exponent = rank_weight_exponent
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.negative_ratio = negative_ratio
        self.lambda_rep = lambda_rep
        self.random_state = random_state
        self.epsilon = epsilon
        self.device = device
        self.attraction_loss_fn = attraction_loss_fn
        self.repulsion_loss_fn = repulsion_loss_fn
        self.graph_mode = graph_mode
        self.n_clusters = n_clusters
        self.n_coarse_msts = n_coarse_msts
        self.n_local_msts = n_local_msts
        self.n_jobs = n_jobs
        self.minibatch_size = minibatch_size

    def fit(self, X: object, y: object = None) -> "IteratedMinimumSpanningTreeEmbedder":
        """Fit the embedding and store coordinates for the input rows."""
        del y  # Unsupervised: labels are intentionally never used.
        n_msts = _positive_integer("n_msts", self.n_msts)
        rank_weight_exponent = _finite_nonnegative(
            "rank_weight_exponent", self.rank_weight_exponent
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
        epsilon = _finite_nonnegative("epsilon", self.epsilon, strictly_positive=True)
        if self.graph_mode not in {"exact", "hierarchical"}:
            raise ValueError("graph_mode must be 'exact' or 'hierarchical'.")
        if self.attraction_loss_fn is not None and not callable(self.attraction_loss_fn):
            raise ValueError("attraction_loss_fn must be callable or None.")
        if self.repulsion_loss_fn is not None and not callable(self.repulsion_loss_fn):
            raise ValueError("repulsion_loss_fn must be callable or None.")
        attraction_loss_fn = (
            log_attraction_loss
            if self.attraction_loss_fn is None
            else self.attraction_loss_fn
        )
        repulsion_loss_fn = (
            inverse_distance_repulsion_loss
            if self.repulsion_loss_fn is None
            else self.repulsion_loss_fn
        )
        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)
        n_samples = X_checked.shape[0]
        if n_samples < 2:
            raise ValueError("X must contain at least two samples to construct an MST.")
        compute_device = _resolve_device(self.device, n_samples)
        compute_dtype = torch.float32 if compute_device.type == "mps" else torch.float64

        if self.graph_mode == "exact":
            distances = cdist(X_checked, X_checked, metric="euclidean")
            if not np.isfinite(distances).all():
                raise ValueError("Pairwise distances overflowed; rescale X before fitting.")
            np.fill_diagonal(distances, 0.0)
            edges, edge_weights = _iterated_mst_edges(
                distances, n_msts, rank_weight_exponent
            )
            self.cluster_labels_ = None
            self.cluster_centers_ = None
            self.cluster_sample_indices_ = None
            self.coarse_graph_edges_ = None
            self.n_unique_local_edges_ = None
            self.n_duplicate_local_edges_ = None
        else:
            n_clusters = _positive_integer("n_clusters", self.n_clusters)
            n_coarse_msts = _positive_integer("n_coarse_msts", self.n_coarse_msts)
            n_local_msts = _positive_integer("n_local_msts", self.n_local_msts)
            minibatch_size = _positive_integer("minibatch_size", self.minibatch_size)
            if n_clusters < 2:
                raise ValueError("n_clusters must be at least 2 for hierarchical mode.")
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                batch_size=minibatch_size,
            )
            labels = kmeans.fit_predict(X_checked)
            coarse_distances = cdist(kmeans.cluster_centers_, kmeans.cluster_centers_)
            if not np.isfinite(coarse_distances).all():
                raise ValueError("Centroid distances overflowed; rescale X before fitting.")
            np.fill_diagonal(coarse_distances, 0.0)
            coarse_edges = _iterated_mst_edges(
                coarse_distances, n_coarse_msts, rank_weight_exponent
            )[0]
            edges, edge_weights = _hierarchical_imst_edges(
                X_checked,
                labels,
                kmeans.cluster_centers_,
                n_clusters,
                n_coarse_msts,
                n_local_msts,
                rank_weight_exponent,
                self.n_jobs,
                coarse_edges=coarse_edges,
            )
            self.cluster_labels_ = labels.copy()
            self.cluster_centers_ = kmeans.cluster_centers_.copy()
            self.cluster_sample_indices_ = [
                np.flatnonzero(labels == cluster) for cluster in range(n_clusters)
            ]
            self.coarse_graph_edges_ = coarse_edges.copy()
            self.n_unique_local_edges_ = int(edges.shape[0])
            self.n_duplicate_local_edges_ = None

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
                weights_t = edge_weights_t[batch_t]

                positive_delta = coordinates[src_t] - coordinates[dst_t]
                positive_d2 = torch.sum(positive_delta * positive_delta, dim=1)
                attraction = _validate_loss_output(
                    "attraction_loss_fn",
                    attraction_loss_fn(positive_d2, weights_t),
                )

                eligible_sources = src[negative_counts[src] > 0]
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
                    negative_d2 = torch.sum(negative_delta * negative_delta, dim=1)
                    repulsion = _validate_loss_output(
                        "repulsion_loss_fn",
                        repulsion_loss_fn(negative_d2, epsilon),
                    )
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
        self.device_ = str(compute_device)
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
