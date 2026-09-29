"""Exact dense Prim construction for edge-disjoint Euclidean MSTs."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def iterated_prim_edges(
    X: np.ndarray,
    n_msts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact edge-disjoint MSTs using dense Prim.

    Pairwise distances are computed once and retained while each tree is
    constructed. This preserves exact behavior but requires quadratic memory.
    """
    distances = cdist(X, X, metric="euclidean")
    if not np.isfinite(distances).all():
        raise ValueError("Pairwise distances overflowed; rescale X before fitting.")
    np.fill_diagonal(distances, 0.0)
    return iterated_prim_distance_edges(distances, n_msts)


def iterated_prim_distance_edges(
    distances: np.ndarray,
    n_msts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct edge-disjoint MSTs from a precomputed distance matrix."""
    n_samples = distances.shape[0]

    # A packed bit matrix stores which edges remain available after earlier
    # trees are removed. This also preserves legitimate zero-distance edges.
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

    return (
        np.asarray(edges, dtype=np.intp).reshape(-1, 2),
        np.asarray(weights, dtype=np.float64),
    )
