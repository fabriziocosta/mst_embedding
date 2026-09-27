import numpy as np
import pytest
import torch
import sys
from pathlib import Path
from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

from mst_embedding import IteratedMinimumSpanningTreeEmbedder
from mst_embedding._estimator import _hierarchical_imst_edges, _iterated_mst_edges

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
from hierarchical_quality_utils import run_trials


def small_data():
    return np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]],
        dtype=float,
    )


def test_sklearn_api_and_training_transform():
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1, n_epochs=3, batch_size=4, random_state=7
    )
    assert clone(estimator).get_params() == estimator.get_params()

    X = small_data()
    embedding = estimator.fit_transform(X)
    assert embedding.shape == (len(X), 2)
    assert np.isfinite(embedding).all()
    np.testing.assert_array_equal(estimator.transform(X), embedding)
    assert estimator.n_features_in_ == X.shape[1]
    assert estimator.device_ == "cpu"


@pytest.mark.parametrize("graph_mode", ["exact", "hierarchical"])
def test_fit_timing_diagnostics_are_finite_and_bounded(graph_mode):
    params = dict(n_msts=1, n_epochs=0, random_state=5, graph_mode=graph_mode)
    if graph_mode == "hierarchical":
        params.update(n_clusters=3, n_coarse_msts=1, n_local_msts=1, n_jobs=1)
    estimator = IteratedMinimumSpanningTreeEmbedder(**params).fit(small_data())

    timings = np.array([
        estimator.graph_construction_time_,
        estimator.embedding_optimization_time_,
        estimator.fit_time_,
    ])
    assert np.isfinite(timings).all()
    assert np.all(timings >= 0)
    assert estimator.graph_construction_time_ <= estimator.fit_time_
    assert estimator.embedding_optimization_time_ <= estimator.fit_time_
    assert (
        estimator.graph_construction_time_ + estimator.embedding_optimization_time_
        <= estimator.fit_time_
    )


def test_quality_sweep_records_failure_and_continues():
    configurations = [
        {"n_clusters": 10},
        {"n_clusters": 25},
        {"n_clusters": 50},
    ]
    attempted = []

    def run_one(configuration):
        n_clusters = configuration["n_clusters"]
        attempted.append(n_clusters)
        if n_clusters == 25:
            raise RuntimeError("simulated trial failure")
        return {"n_clusters": n_clusters, "score": 0.9}

    results, failures = run_trials(configurations, run_one)

    assert attempted == [10, 25, 50]
    assert [row["n_clusters"] for row in results] == [10, 50]
    assert failures == [{
        "n_clusters": 25,
        "error_type": "RuntimeError",
        "error": "simulated trial failure",
    }]


def test_seed_reproduces_embedding():
    X = small_data()
    params = dict(n_msts=1, n_epochs=4, batch_size=3, random_state=13)
    first = IteratedMinimumSpanningTreeEmbedder(**params).fit_transform(X)
    second = IteratedMinimumSpanningTreeEmbedder(**params).fit_transform(X)
    np.testing.assert_array_equal(first, second)


def test_transform_rejects_new_or_reordered_rows():
    X = small_data()
    estimator = IteratedMinimumSpanningTreeEmbedder(n_msts=1, n_epochs=0).fit(X)
    with pytest.raises(ValueError, match="original training rows"):
        estimator.transform(X[::-1])
    with pytest.raises(ValueError, match="original training rows"):
        estimator.transform(np.vstack([X, [3.0, 3.0]]))


def test_duplicate_samples_keep_zero_distance_mst_edges():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    estimator = IteratedMinimumSpanningTreeEmbedder(n_msts=1, n_epochs=2, random_state=2).fit(X)
    assert estimator.graph_edges_.shape == (len(X) - 1, 2)
    assert np.isfinite(estimator.embedding_).all()


def test_insufficient_edge_disjoint_msts_raise_clear_error():
    X = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="Reduce n_msts"):
        IteratedMinimumSpanningTreeEmbedder(n_msts=2, n_epochs=0).fit(X)


def test_no_eligible_negative_pairs_is_supported():
    X = np.array([[0.0], [1.0]])
    estimator = IteratedMinimumSpanningTreeEmbedder(n_msts=1, n_epochs=2, random_state=3).fit(X)
    assert estimator.embedding_.shape == (2, 2)
    assert np.isfinite(estimator.embedding_).all()


@pytest.mark.skipif(
    not (torch.backends.mps.is_built() and torch.backends.mps.is_available()),
    reason="PyTorch MPS is unavailable",
)
def test_mps_device_runs_when_available():
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1, n_epochs=2, random_state=5, device="mps"
    ).fit(small_data())
    assert estimator.device_ == "mps"
    assert np.isfinite(estimator.embedding_).all()


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("n_msts", 0),
        ("n_epochs", -1),
        ("batch_size", 0),
        ("negative_ratio", -1),
        ("learning_rate", 0),
        ("lambda_rep", -1),
        ("epsilon", 0),
        ("attraction_normalization", "invalid"),
    ],
)
def test_invalid_parameters_raise_value_error(parameter, value):
    estimator = IteratedMinimumSpanningTreeEmbedder(n_msts=1, n_epochs=0)
    setattr(estimator, parameter, value)
    with pytest.raises(ValueError):
        estimator.fit(small_data())


def test_invalid_data_raise_clear_error():
    with pytest.raises(ValueError, match="at least two samples"):
        IteratedMinimumSpanningTreeEmbedder(n_msts=1, n_epochs=0).fit([[1.0, 2.0]])
    with pytest.raises(ValueError):
        IteratedMinimumSpanningTreeEmbedder(n_msts=1, n_epochs=0).fit([[0.0], [np.nan]])


def hierarchical_data():
    rng = np.random.RandomState(12)
    return np.vstack([rng.normal(loc=i * 4.0, scale=0.3, size=(8, 3)) for i in range(4)])


def test_hierarchical_embedding_and_indices_are_valid():
    X = hierarchical_data()
    estimator = IteratedMinimumSpanningTreeEmbedder(
        graph_mode="hierarchical", n_clusters=4, n_coarse_msts=1,
        n_local_msts=2, n_epochs=2, random_state=4,
    ).fit(X)
    assert estimator.embedding_.shape == (len(X), 2)
    assert np.isfinite(estimator.embedding_).all()
    assert estimator.graph_edges_.ndim == 2 and estimator.graph_edges_.shape[1] == 2
    assert np.all((estimator.graph_edges_ >= 0) & (estimator.graph_edges_ < len(X)))
    canonical = [tuple(sorted(edge)) for edge in estimator.graph_edges_]
    assert len(canonical) == len(set(canonical))
    assert estimator.cluster_labels_.shape == (len(X),)
    assert estimator.cluster_centers_.shape == (4, X.shape[1])


def test_hierarchical_parallel_is_reproducible_and_exact_default_unchanged():
    X = hierarchical_data()
    params = dict(
        graph_mode="hierarchical", n_clusters=4, n_coarse_msts=1,
        n_local_msts=2, n_epochs=0, random_state=19,
    )
    serial = IteratedMinimumSpanningTreeEmbedder(n_jobs=1, **params).fit(X)
    parallel = IteratedMinimumSpanningTreeEmbedder(n_jobs=2, **params).fit(X)
    np.testing.assert_array_equal(serial.graph_edges_, parallel.graph_edges_)
    np.testing.assert_array_equal(serial.graph_weights_, parallel.graph_weights_)
    np.testing.assert_array_equal(serial.cluster_labels_, parallel.cluster_labels_)
    exact = IteratedMinimumSpanningTreeEmbedder(n_msts=1, n_epochs=0).fit(X)
    expected, weights = _iterated_mst_edges(cdist(X, X), 1, 1.0)
    np.testing.assert_array_equal(exact.graph_edges_, expected)
    np.testing.assert_array_equal(exact.graph_weights_, weights)


def test_hierarchical_cluster_and_pair_reproducibility_and_union_edges():
    X = hierarchical_data()
    first = MiniBatchKMeans(n_clusters=4, random_state=23, batch_size=8).fit(X)
    second = MiniBatchKMeans(n_clusters=4, random_state=23, batch_size=8).fit(X)
    np.testing.assert_array_equal(first.labels_, second.labels_)
    np.testing.assert_array_equal(first.cluster_centers_, second.cluster_centers_)
    edges, weights = _hierarchical_imst_edges(
        X, first.labels_, first.cluster_centers_, 4, 1, 2, 1.0, 1
    )
    coarse, _ = _iterated_mst_edges(cdist(first.cluster_centers_, first.cluster_centers_), 1)
    assert len(coarse) > 0
    # Each pair union's tree naturally includes edges internal to a cluster
    # alongside edges crossing the two endpoint clusters.
    a, b = coarse[0]
    pair = np.flatnonzero((first.labels_ == a) | (first.labels_ == b))
    assert any(first.labels_[u] == first.labels_[v] for u, v in edges if u in pair and v in pair)
    assert any(first.labels_[u] != first.labels_[v] for u, v in edges if u in pair and v in pair)
    assert np.isfinite(weights).all()


def test_duplicate_discoveries_use_set_union_and_strongest_weight(monkeypatch):
    import mst_embedding._estimator as estimator_module

    class InlineParallel:
        def __init__(self, n_jobs):
            assert n_jobs == 1

        def __call__(self, jobs):
            return [func(*args, **kwargs) for func, args, kwargs in jobs]

    results = iter([
        (np.array([[1, 0], [2, 1]]), np.array([0.5, 0.25])),
        (np.array([[0, 1], [3, 4]]), np.array([1.0, 0.25])),
    ])
    monkeypatch.setattr(estimator_module, "Parallel", InlineParallel)
    monkeypatch.setattr(estimator_module, "_local_cluster_pair_imst", lambda *args: next(results))
    X = np.arange(10, dtype=float).reshape(5, 2)
    labels = np.array([0, 0, 1, 2, 2])
    centers = np.array([[0., 0.], [1., 1.], [2., 2.]])
    # Coarse chain yields two local jobs sharing cluster 0 only after replacing
    # its edge list with the two desired adjacent pairs.
    monkeypatch.setattr(
        estimator_module, "_iterated_mst_edges",
        lambda *args, **kwargs: (np.array([[0, 1], [0, 2]]), np.ones(2)),
    )
    edges, weights = _hierarchical_imst_edges(
        X, labels, centers, 3, 1, 2, 1.0, 1
    )
    np.testing.assert_array_equal(edges, [[0, 1], [1, 2], [3, 4]])
    np.testing.assert_array_equal(weights, [1.0, 0.25, 0.25])
    assert len(edges) == 3


def test_hierarchical_zero_distance_and_singletons_are_supported():
    X = np.array([[0., 0.], [0., 0.], [10., 0.], [20., 0.]])
    estimator = IteratedMinimumSpanningTreeEmbedder(
        graph_mode="hierarchical", n_clusters=3, n_coarse_msts=1,
        n_local_msts=8, minibatch_size=2, n_epochs=0, random_state=2, n_jobs=1,
    ).fit(X)
    assert np.isfinite(estimator.graph_weights_).all()
    assert np.isfinite(estimator.embedding_).all()
    assert estimator.graph_edges_.size == 0 or np.all(estimator.graph_edges_[:, 0] != estimator.graph_edges_[:, 1])
    assert len({tuple(edge) for edge in estimator.graph_edges_}) == len(estimator.graph_edges_)
    assert (0, 1) in {tuple(sorted(map(int, edge))) for edge in estimator.graph_edges_}
    assert len(estimator.cluster_sample_indices_) == 3


def test_weight_sum_attraction_normalization_uses_global_weight_sum_scale():
    X = small_data()
    observed_weights = []

    def capture_attraction_weights(positive_squared_distances, edge_weights):
        observed_weights.extend(edge_weights.detach().cpu().numpy())
        return torch.mean(edge_weights * torch.log1p(positive_squared_distances))

    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=2,
        n_epochs=1,
        batch_size=100,
        negative_ratio=0,
        random_state=3,
        attraction_normalization="weight_sum",
        attraction_loss_fn=capture_attraction_weights,
    ).fit(X)

    expected = estimator.graph_weights_ * (
        len(estimator.graph_weights_) / estimator.graph_weights_.sum()
    )
    np.testing.assert_allclose(np.sort(observed_weights), np.sort(expected))
    assert np.isclose(np.sum(observed_weights), len(estimator.graph_weights_))
