"""Numba kernels for deterministic rank-scheduled IMSTE SGD."""

from __future__ import annotations

import numpy as np
from numba import njit


def rank_phases(tree_ranks: np.ndarray, random_state: int | None) -> np.ndarray:
    """Choose one reproducible epoch phase per tree rank."""
    n_ranks = int(np.max(tree_ranks)) if tree_ranks.size else 0
    phases = np.zeros(n_ranks + 1, dtype=np.int64)
    rng = np.random.RandomState(random_state)
    for rank in range(2, n_ranks + 1):
        phases[rank] = rng.randint(rank)
    return phases


def active_edges_for_epoch(
    tree_ranks: np.ndarray, epoch: int, phases: np.ndarray
) -> np.ndarray:
    """Return scheduled edge indices; exposed for exact scheduler tests."""
    if not tree_ranks.size:
        return np.empty(0, dtype=np.intp)
    return np.flatnonzero(epoch % tree_ranks == phases[tree_ranks])


def expected_active_edge_count(tree_ranks: np.ndarray) -> float:
    """Expected active edge count, used for the normalized rank-weighted mean."""
    return float(np.sum(1.0 / tree_ranks)) if tree_ranks.size else 1.0


@njit(cache=True)
def _attraction_factor(d2: float, epsilon: float) -> float:
    return 2.0 / (1.0 + d2 + epsilon)


@njit(cache=True)
def _repulsion_factor(d2: float, margin: float, temperature: float) -> float:
    z = (margin - d2) / temperature
    if z >= 0.0:
        sigmoid = 1.0 / (1.0 + np.exp(-z))
    else:
        exp_z = np.exp(z)
        sigmoid = exp_z / (1.0 + exp_z)
    return 2.0 * sigmoid / temperature


@njit(cache=True)
def _find_neighbor(indices: np.ndarray, start: int, stop: int, target: int) -> bool:
    left = start
    right = stop
    while left < right:
        middle = (left + right) // 2
        value = indices[middle]
        if value < target:
            left = middle + 1
        else:
            right = middle
    return left < stop and indices[left] == target


@njit(cache=True)
def _sample_target(
    source: int, n_samples: int, indptr: np.ndarray, indices: np.ndarray
) -> int:
    # Rejection is fast for sparse graphs. Reservoir fallback remains exact for
    # unusually dense neighborhoods without allocating a complement array.
    for _ in range(32):
        target = np.random.randint(0, n_samples)
        if target != source and not _find_neighbor(
            indices, indptr[source], indptr[source + 1], target
        ):
            return target
    selected = -1
    available = 0
    for target in range(n_samples):
        if target == source or _find_neighbor(
            indices, indptr[source], indptr[source + 1], target
        ):
            continue
        available += 1
        if np.random.randint(0, available) == 0:
            selected = target
    return selected


@njit(cache=True)
def _recenter(embedding: np.ndarray) -> None:
    n_samples, n_components = embedding.shape
    for component in range(n_components):
        mean = 0.0
        for sample in range(n_samples):
            mean += embedding[sample, component]
        mean /= n_samples
        for sample in range(n_samples):
            embedding[sample, component] -= mean


@njit(cache=True)
def _optimize_epoch(
    embedding: np.ndarray,
    edges: np.ndarray,
    tree_ranks: np.ndarray,
    phases: np.ndarray,
    indptr: np.ndarray,
    adjacency: np.ndarray,
    epoch: int,
    n_epochs: int,
    expected_active_edges: float,
    learning_rate: float,
    negative_ratio: int,
    lambda_rep: float,
    epsilon: float,
    margin: float,
    temperature: float,
    seed: int,
) -> None:
    np.random.seed(seed)
    n_edges = edges.shape[0]
    n_samples, n_components = embedding.shape
    active = np.empty(n_edges, dtype=np.int64)
    degree = np.diff(indptr)

    active_count = 0
    for edge_index in range(n_edges):
        rank = tree_ranks[edge_index]
        if epoch % rank == phases[rank]:
            active[active_count] = edge_index
            active_count += 1
    # Fisher-Yates ordering keeps sequential SGD reproducible.
    for idx in range(active_count - 1, 0, -1):
        swap = np.random.randint(0, idx + 1)
        value = active[idx]
        active[idx] = active[swap]
        active[swap] = value

    eligible_endpoints = 0
    if negative_ratio > 0:
        for item in range(active_count):
            edge_index = active[item]
            left = edges[edge_index, 0]
            right = edges[edge_index, 1]
            if degree[left] < n_samples - 1:
                eligible_endpoints += 1
            if degree[right] < n_samples - 1:
                eligible_endpoints += 1
    negative_count = eligible_endpoints * negative_ratio
    # The mean-loss normalization divides per-coordinate updates by the graph
    # population. Undo that dilution so the configured learning rate remains a
    # usable per-sample step size as datasets grow. The factor of five was
    # calibrated against the former Adam default on representative embeddings.
    alpha = learning_rate * (5.0 * n_samples) * (1.0 - epoch / max(1, n_epochs))
    positive_scale = (1.0 - lambda_rep) / max(1.0, expected_active_edges)
    negative_scale = lambda_rep / max(1, negative_count)

    for item in range(active_count):
        edge_index = active[item]
        left = edges[edge_index, 0]
        right = edges[edge_index, 1]
        d2 = 0.0
        for component in range(n_components):
            delta = embedding[left, component] - embedding[right, component]
            d2 += delta * delta
        attraction_factor = alpha * positive_scale * _attraction_factor(d2, epsilon)
        for component in range(n_components):
            delta = embedding[left, component] - embedding[right, component]
            update = attraction_factor * delta
            embedding[left, component] -= update
            embedding[right, component] += update

        if negative_ratio == 0:
            continue
        for endpoint_index in range(2):
            source = left if endpoint_index == 0 else right
            if degree[source] >= n_samples - 1:
                continue
            for _ in range(negative_ratio):
                target = _sample_target(source, n_samples, indptr, adjacency)
                if target < 0:
                    continue
                d2 = 0.0
                for component in range(n_components):
                    delta = embedding[source, component] - embedding[target, component]
                    d2 += delta * delta
                repulsion_factor = (
                    alpha * negative_scale * _repulsion_factor(d2, margin, temperature)
                )
                for component in range(n_components):
                    delta = embedding[source, component] - embedding[target, component]
                    update = repulsion_factor * delta
                    embedding[source, component] += update
                    embedding[target, component] -= update

    if n_samples:
        _recenter(embedding)


def optimize_embedding(
    embedding: np.ndarray,
    edges: np.ndarray,
    tree_ranks: np.ndarray,
    *,
    n_epochs: int,
    learning_rate: float,
    negative_ratio: int,
    lambda_rep: float,
    epsilon: float,
    margin: float,
    temperature: float,
    random_state: int | None,
    progress_callback=None,
) -> None:
    """Optimize coordinates in place using rank-scheduled, normalized SGD."""
    n_samples = embedding.shape[0]
    edge_pairs = np.asarray(edges, dtype=np.int64)
    ranks = np.asarray(tree_ranks, dtype=np.int64)
    phases = rank_phases(ranks, random_state)
    expected_active_edges = expected_active_edge_count(ranks)
    source = np.concatenate((edge_pairs[:, 0], edge_pairs[:, 1]))
    target = np.concatenate((edge_pairs[:, 1], edge_pairs[:, 0]))
    order = np.lexsort((target, source))
    source = source[order]
    target = target[order]
    counts = np.bincount(source, minlength=n_samples)
    indptr = np.empty(n_samples + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    adjacency = target.astype(np.int64, copy=False)
    seed_rng = np.random.RandomState(random_state)
    for epoch in range(n_epochs):
        epoch_seed = int(seed_rng.randint(0, np.iinfo(np.int32).max))
        _optimize_epoch(
            embedding,
            edge_pairs,
            ranks,
            phases,
            indptr,
            adjacency,
            epoch,
            n_epochs,
            expected_active_edges,
            learning_rate,
            negative_ratio,
            lambda_rep,
            epsilon,
            margin,
            temperature,
            epoch_seed,
        )
        if progress_callback is not None:
            progress_callback(epoch + 1, n_epochs)
