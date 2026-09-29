import numpy as np
import pytest
import torch
from sklearn.base import clone
from scipy.spatial.distance import cdist

from imste import IteratedMinimumSpanningTreeEmbedder
from imste._estimator import (
    _iterated_mst_edges,
    _sample_negative_targets,
    log_attraction_loss,
)


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


def test_fit_timing_diagnostics_are_finite_and_bounded():
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1, n_epochs=0, random_state=5
    ).fit(small_data())

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


def test_mlp_projection_fits_and_transforms_unseen_rows():
    X = small_data()
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1,
        n_epochs=0,
        transform_method="mlp",
        mlp_n_layers=1,
        mlp_layer_size=8,
        mlp_dropout=0.0,
        mlp_epochs=4,
        mlp_min_epochs=2,
        mlp_patience=2,
        mlp_batch_size=2,
        random_state=11,
    ).fit(X)

    prediction = estimator.transform([[3.0, 3.0]])
    assert prediction.shape == (1, 2)
    assert np.isfinite(prediction).all()
    assert 2 <= estimator.mlp_epochs_trained_ <= 4
    assert np.isfinite(estimator.mlp_best_validation_loss_)
    assert np.isfinite(estimator.mlp_training_loss_)


def test_dataframe_transform_rejects_reordered_feature_names():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(small_data(), columns=["left", "right"])
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1, n_epochs=0, random_state=3
    ).fit(X)

    with pytest.raises(ValueError, match="same feature names in the same order"):
        estimator.transform(X[["right", "left"]])


def test_dense_negative_sampling_falls_back_to_exact_complement():
    n_samples = 64
    packed_adjacency = np.full((n_samples, n_samples // 8), 0xFF, dtype=np.uint8)
    # Node 0 has exactly one non-neighbor. Rejection sampling alone would need
    # many draws; the bounded sampler should return that complement directly.
    packed_adjacency[0, 63 // 8] &= np.uint8(0xFF ^ (1 << (63 & 7)))
    sampled = _sample_negative_targets(
        np.random.RandomState(17),
        packed_adjacency,
        np.zeros(32, dtype=np.intp),
        n_samples,
    )
    np.testing.assert_array_equal(sampled, np.full(32, 63))


def test_mlp_stops_after_patience_once_minimum_epochs_are_met(monkeypatch):
    class ConstantMLP(torch.nn.Module):
        def __init__(self, n_features, n_components, n_layers, width, dropout):
            super().__init__()
            self.output = torch.nn.Parameter(torch.ones(n_components))

        def forward(self, inputs):
            return (self.output * 0).expand(inputs.shape[0], -1)

    monkeypatch.setattr("imste._estimator._MLPProjector", ConstantMLP)
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1,
        n_epochs=0,
        transform_method="mlp",
        mlp_n_layers=1,
        mlp_layer_size=4,
        mlp_dropout=0.0,
        mlp_epochs=8,
        mlp_min_epochs=2,
        mlp_patience=2,
        random_state=19,
    ).fit(small_data())

    assert estimator.mlp_epochs_trained_ == 3


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


def test_exact_graph_uses_euclidean_distances():
    X = small_data()
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=2, n_epochs=0, random_state=19
    ).fit(X)
    expected, weights = _iterated_mst_edges(cdist(X, X, metric="euclidean"), 2)
    np.testing.assert_array_equal(estimator.graph_edges_, expected)
    np.testing.assert_array_equal(estimator.graph_weights_, weights)
    assert set(estimator.graph_weights_) == {1.0, 0.5}


def test_attraction_uses_global_weight_sum_scale(monkeypatch):
    X = small_data()
    observed_weights = []
    original_log_attraction_loss = log_attraction_loss

    def capture_attraction_weights(positive_squared_distances, edge_weights, epsilon):
        observed_weights.extend(edge_weights.detach().cpu().numpy())
        return original_log_attraction_loss(
            positive_squared_distances, edge_weights, epsilon
        )

    monkeypatch.setattr(
        "imste._estimator.log_attraction_loss", capture_attraction_weights
    )

    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=2,
        n_epochs=1,
        batch_size=100,
        negative_ratio=0,
        random_state=3,
    ).fit(X)

    expected = estimator.graph_weights_ * (
        len(estimator.graph_weights_) / estimator.graph_weights_.sum()
    )
    np.testing.assert_allclose(np.sort(observed_weights), np.sort(expected))
    assert np.isclose(np.sum(observed_weights), len(estimator.graph_weights_))
