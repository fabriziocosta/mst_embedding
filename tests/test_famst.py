import numpy as np
import sys
import types

from imste.mst.famst import (
    _connect_components,
    _kruskal_edge_disjoint_trees,
    _ann_edges,
    _representative_component_pairs,
    _sample_component_representatives,
    iterated_famst_edges,
)


def _separated_components():
    X = np.array(
        [[0.0, 0.0], [10.0, 0.0], [1.0, 0.0], [11.0, 0.0], [20.0, 0.0], [21.0, 0.0]]
    )
    edges = {(0, 1): 10.0, (2, 3): 10.0, (4, 5): 1.0}
    groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    return X, edges, groups


def test_component_representatives_are_bounded_and_seeded():
    groups = [np.array([0, 1, 2]), np.arange(3, 23), np.array([23, 24])]
    points, components = _sample_component_representatives(
        groups, 5, np.random.RandomState(41)
    )
    repeated_points, repeated_components = _sample_component_representatives(
        groups, 5, np.random.RandomState(41)
    )

    np.testing.assert_array_equal(points, repeated_points)
    np.testing.assert_array_equal(components, repeated_components)
    np.testing.assert_array_equal(np.bincount(components), [3, 5, 2])
    assert set(points[components == 0]) == {0, 1, 2}
    assert set(points[components == 2]) == {23, 24}


def test_representative_mst_returns_unique_cross_component_pairs():
    X, _, groups = _separated_components()
    pairs = _representative_component_pairs(
        X, groups, 10, np.random.RandomState(7)
    )

    # The representative MST crosses components 0-1 twice, then 1-2.
    assert pairs == [(0, 1), (1, 2)]


def test_bridge_search_runs_once_for_each_representative_mst_pair():
    X, edges, _ = _separated_components()

    class RecordingRng:
        def __init__(self):
            self.rng = np.random.RandomState(5)
            self.choice_calls = 0

        def choice(self, values, size=None):
            self.choice_calls += 1 if size is None else size
            return self.rng.choice(values, size=size)

    rng = RecordingRng()
    bridges, labels = _connect_components(
        X, edges, n_candidates=1, rng=rng, representatives_per_component=2
    )

    bridge_pairs = {
        tuple(sorted((int(labels[left]), int(labels[right]))))
        for left, right, _, _ in bridges
    }
    assert bridge_pairs == {(0, 1), (1, 2)}
    assert len(bridges) == 2
    # One left and one right endpoint draw for each unique component pair.
    assert rng.choice_calls == 4


def test_famst_still_constructs_requested_tree(monkeypatch):
    X, edges, _ = _separated_components()
    monkeypatch.setattr(
        "imste.mst.famst._ann_edges",
        lambda X, neighbors, random_state: dict(edges),
    )

    result_edges, weights = iterated_famst_edges(
        X,
        n_msts=1,
        random_state=9,
        neighbors=1,
        inter_component_edges=1,
        max_neighbors=4,
        representatives_per_component=2,
    )

    assert result_edges.shape == (len(X) - 1, 2)
    np.testing.assert_array_equal(weights, np.ones(len(X) - 1))


def test_compiled_kruskal_returns_edge_disjoint_trees_and_integer_ranks():
    candidates = {
        (0, 1): 1.0,
        (1, 2): 1.0,
        (2, 3): 1.0,
        (0, 3): 1.0,
        (0, 2): 2.0,
        (1, 3): 2.0,
    }
    result = _kruskal_edge_disjoint_trees(4, candidates, 2)
    assert result is not None
    edges, ranks = result
    assert edges.shape == (2 * (4 - 1), 2)
    np.testing.assert_array_equal(ranks, [1, 1, 1, 2, 2, 2])
    first = {tuple(sorted(map(int, edge))) for edge in edges[ranks == 1]}
    second = {tuple(sorted(map(int, edge))) for edge in edges[ranks == 2]}
    assert len(first) == len(second) == 3
    assert first.isdisjoint(second)


def test_famst_can_return_ranks_without_changing_weight_default(monkeypatch):
    X, edges, _ = _separated_components()
    monkeypatch.setattr(
        "imste.mst.famst._ann_edges",
        lambda X, neighbors, random_state: dict(edges),
    )
    kwargs = dict(
        X=X,
        n_msts=1,
        random_state=9,
        neighbors=1,
        inter_component_edges=1,
        max_neighbors=4,
        representatives_per_component=2,
    )
    edge_weights, weights = iterated_famst_edges(**kwargs)
    edge_ranks, ranks = iterated_famst_edges(**kwargs, return_ranks=True)
    np.testing.assert_array_equal(edge_weights, edge_ranks)
    np.testing.assert_array_equal(ranks, np.ones(len(ranks), dtype=np.int32))
    np.testing.assert_array_equal(weights, 1.0 / ranks)


def test_ann_edges_reuse_minimum_finite_returned_distance(monkeypatch):
    class FakeNNDescent:
        def __init__(self, *args, **kwargs):
            self.neighbor_graph = (
                np.array([[1, 2, -1], [0, 2, 2], [1, 0, 1]]),
                np.array([[3.0, 4.0, np.inf], [2.0, 1.5, np.nan], [2.5, 4.0, 3.5]]),
            )

    monkeypatch.setitem(
        sys.modules, "pynndescent", types.SimpleNamespace(NNDescent=FakeNNDescent)
    )
    edges = _ann_edges(np.zeros((3, 2)), neighbors=2, random_state=1)
    assert edges == {(0, 1): 2.0, (0, 2): 4.0, (1, 2): 1.5}
