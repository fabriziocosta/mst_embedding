"""Sparse FAMST-style approximate Euclidean MST construction.

This follows the FAMST paper's ANN graph, component-connection, local
inter-component refinement, then Kruskal pipeline. Unlike the paper's single
tree output, this implementation extracts edge-disjoint trees from the sparse
candidate graph and increases the ANN neighborhood size when necessary.
"""

from __future__ import annotations

import numpy as np


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.intp)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            next_item = int(self.parent[item])
            self.parent[item] = root
            item = next_item
        return root

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def _components(n_samples: int, edges: dict[tuple[int, int], float]):
    sets = _DisjointSet(n_samples)
    for left, right in edges:
        sets.union(left, right)
    roots = np.fromiter((sets.find(i) for i in range(n_samples)), dtype=np.intp)
    _, labels = np.unique(roots, return_inverse=True)
    group_lists: list[list[int]] = [[] for _ in range(int(labels.max()) + 1)]
    for point, label in enumerate(labels):
        group_lists[int(label)].append(point)
    groups = [np.asarray(group, dtype=np.intp) for group in group_lists]
    return labels, groups


def _distance(X: np.ndarray, left: int, right: int) -> float:
    delta = X[left] - X[right]
    value = float(np.sqrt(np.dot(delta, delta)))
    if not np.isfinite(value):
        raise ValueError("Pairwise distances overflowed; rescale X before fitting.")
    return value


def _ann_edges(
    X: np.ndarray,
    neighbors: int,
    random_state: int | None,
) -> dict[tuple[int, int], float]:
    try:
        from pynndescent import NNDescent
    except ImportError as exc:
        raise ImportError(
            "mst_method='famst' requires the optional dependency. Install it with "
            "`python -m pip install 'imste[famst]'`."
        ) from exc

    index = NNDescent(
        X,
        n_neighbors=min(neighbors + 1, len(X)),
        metric="euclidean",
        random_state=random_state,
        n_jobs=-1,
    )
    indices, _ = index.neighbor_graph
    edges: dict[tuple[int, int], float] = {}
    for left, row in enumerate(indices):
        for right_value in row:
            right = int(right_value)
            if right < 0 or right == left:
                continue
            edge = (left, right) if left < right else (right, left)
            if edge not in edges:
                edges[edge] = _distance(X, *edge)
    return edges


def _connect_components(
    X: np.ndarray,
    edges: dict[tuple[int, int], float],
    n_candidates: int,
    rng: np.random.RandomState,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """Add randomized inter-component candidates and refine locally.

    Each candidate stores (left, right, left_component, right_component).
    """
    labels, groups = _components(len(X), edges)
    if len(groups) <= 1:
        return [], labels

    bridges: list[tuple[int, int, int, int]] = []
    for left_component in range(len(groups)):
        left_points = groups[left_component]
        for right_component in range(left_component + 1, len(groups)):
            right_points = groups[right_component]
            candidates: dict[tuple[int, int], float] = {}
            for _ in range(n_candidates * n_candidates):
                left = int(rng.choice(left_points))
                right = int(rng.choice(right_points))
                candidates[(left, right)] = _distance(X, left, right)
            best = sorted(candidates, key=candidates.__getitem__)[:n_candidates]
            bridges.extend(
                (left, right, left_component, right_component)
                for left, right in best
            )

    neighbors_by_node: list[list[int]] = [[] for _ in range(len(X))]
    for left, right in edges:
        neighbors_by_node[left].append(right)
        neighbors_by_node[right].append(left)

    # Repeatedly move bridge endpoints to closer vertices in the same ANN
    # component. Each update strictly decreases a finite set of edge distances.
    changed = True
    while changed:
        changed = False
        refined: list[tuple[int, int, int, int]] = []
        for left, right, left_component, right_component in bridges:
            best_left, best_right = left, right
            best_distance = _distance(X, left, right)
            for candidate_left in neighbors_by_node[left]:
                if labels[candidate_left] != left_component:
                    continue
                candidate_distance = _distance(X, candidate_left, best_right)
                if candidate_distance < best_distance:
                    best_left, best_distance = candidate_left, candidate_distance
            for candidate_right in neighbors_by_node[right]:
                if labels[candidate_right] != right_component:
                    continue
                candidate_distance = _distance(X, best_left, candidate_right)
                if candidate_distance < best_distance:
                    best_right, best_distance = candidate_right, candidate_distance
            changed |= (best_left, best_right) != (left, right)
            refined.append(
                (best_left, best_right, left_component, right_component)
            )
        bridges = refined

    for left, right, _, _ in bridges:
        edge = (left, right) if left < right else (right, left)
        edges.setdefault(edge, _distance(X, *edge))
    return bridges, labels


def _kruskal_edge_disjoint_trees(
    n_samples: int,
    candidate_edges: dict[tuple[int, int], float],
    n_msts: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    available = dict(candidate_edges)
    result_edges: list[tuple[int, int]] = []
    result_weights: list[float] = []
    ordered = sorted(available, key=available.__getitem__)

    for rank in range(1, n_msts + 1):
        sets = _DisjointSet(n_samples)
        tree: list[tuple[int, int]] = []
        for edge in ordered:
            if edge not in available:
                continue
            if sets.union(*edge):
                tree.append(edge)
                if len(tree) == n_samples - 1:
                    break
        if len(tree) != n_samples - 1:
            return None
        result_edges.extend(tree)
        result_weights.extend([1.0 / rank] * len(tree))
        for edge in tree:
            del available[edge]
    return (
        np.asarray(result_edges, dtype=np.intp).reshape(-1, 2),
        np.asarray(result_weights, dtype=np.float64),
    )


def iterated_famst_edges(
    X: np.ndarray,
    n_msts: int,
    random_state: int | None,
    neighbors: int,
    inter_component_edges: int,
    max_neighbors: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build approximate edge-disjoint trees from an adaptive sparse graph."""
    n_samples = len(X)
    if n_samples <= 3:
        from .prim import iterated_prim_edges

        return iterated_prim_edges(X, n_msts)

    rng = np.random.RandomState(random_state)
    start_neighbors = min(neighbors, n_samples - 1)
    cap = min(
        n_samples - 1,
        max_neighbors
        if max_neighbors is not None
        else max(start_neighbors, 4 * start_neighbors, 4 * n_msts),
    )
    if cap < start_neighbors:
        raise ValueError("mst_max_neighbors must be at least mst_neighbors.")

    current_neighbors = start_neighbors
    while True:
        if current_neighbors >= n_samples - 1:
            from .prim import iterated_prim_edges

            return iterated_prim_edges(X, n_msts)
        seed = int(rng.randint(0, np.iinfo(np.int32).max))
        candidate_edges = _ann_edges(X, current_neighbors, seed)
        _connect_components(X, candidate_edges, inter_component_edges, rng)
        trees = _kruskal_edge_disjoint_trees(n_samples, candidate_edges, n_msts)
        if trees is not None:
            return trees
        if current_neighbors >= cap:
            raise ValueError(
                f"FAMST candidate graph could not supply {n_msts} edge-disjoint "
                f"spanning trees with up to {current_neighbors} neighbors. "
                "Increase mst_max_neighbors or reduce n_msts."
            )
        current_neighbors = min(cap, max(current_neighbors + 1, current_neighbors * 2))
