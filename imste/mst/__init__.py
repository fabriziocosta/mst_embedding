"""Pluggable minimum-spanning-tree construction backends."""

from __future__ import annotations

import numpy as np

from .prim import iterated_prim_edges


def build_mst_edges(
    X: np.ndarray,
    *,
    n_msts: int,
    random_state: int | None,
    method: str,
    neighbors: int,
    inter_component_edges: int,
    max_neighbors: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch MST construction to the selected backend."""
    if method == "prim":
        return iterated_prim_edges(X, n_msts)
    if method == "famst":
        from .famst import iterated_famst_edges

        return iterated_famst_edges(
            X,
            n_msts,
            random_state,
            neighbors,
            inter_component_edges,
            max_neighbors,
        )
    raise ValueError("mst_method must be either 'prim' or 'famst'.")


__all__ = ["build_mst_edges", "iterated_prim_edges"]
