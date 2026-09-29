import numpy as np

from imste.mst.famst import (
    _connect_components,
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

        def choice(self, values):
            self.choice_calls += 1
            return self.rng.choice(values)

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
