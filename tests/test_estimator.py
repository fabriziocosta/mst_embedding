import numpy as np
import pytest
from sklearn.base import clone

from mst_embedding import IteratedMSTEmbedding


def small_data():
    return np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]],
        dtype=float,
    )


def test_sklearn_api_and_training_transform():
    estimator = IteratedMSTEmbedding(
        n_msts=1, n_epochs=3, batch_size=4, random_state=7
    )
    assert clone(estimator).get_params() == estimator.get_params()

    X = small_data()
    embedding = estimator.fit_transform(X)
    assert embedding.shape == (len(X), 2)
    assert np.isfinite(embedding).all()
    np.testing.assert_array_equal(estimator.transform(X), embedding)
    assert estimator.n_features_in_ == X.shape[1]


def test_seed_reproduces_embedding():
    X = small_data()
    params = dict(n_msts=1, n_epochs=4, batch_size=3, random_state=13)
    first = IteratedMSTEmbedding(**params).fit_transform(X)
    second = IteratedMSTEmbedding(**params).fit_transform(X)
    np.testing.assert_array_equal(first, second)


def test_transform_rejects_new_or_reordered_rows():
    X = small_data()
    estimator = IteratedMSTEmbedding(n_msts=1, n_epochs=0).fit(X)
    with pytest.raises(ValueError, match="original training rows"):
        estimator.transform(X[::-1])
    with pytest.raises(ValueError, match="original training rows"):
        estimator.transform(np.vstack([X, [3.0, 3.0]]))


def test_duplicate_samples_keep_zero_distance_mst_edges():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    estimator = IteratedMSTEmbedding(n_msts=1, n_epochs=2, random_state=2).fit(X)
    assert estimator.graph_edges_.shape == (len(X) - 1, 2)
    assert np.isfinite(estimator.embedding_).all()


def test_insufficient_edge_disjoint_msts_raise_clear_error():
    X = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="Reduce n_msts"):
        IteratedMSTEmbedding(n_msts=2, n_epochs=0).fit(X)


def test_no_eligible_negative_pairs_is_supported():
    X = np.array([[0.0], [1.0]])
    estimator = IteratedMSTEmbedding(n_msts=1, n_epochs=2, random_state=3).fit(X)
    assert estimator.embedding_.shape == (2, 2)
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
    estimator = IteratedMSTEmbedding(n_msts=1, n_epochs=0)
    setattr(estimator, parameter, value)
    with pytest.raises(ValueError):
        estimator.fit(small_data())


def test_invalid_data_raise_clear_error():
    with pytest.raises(ValueError, match="at least two samples"):
        IteratedMSTEmbedding(n_msts=1, n_epochs=0).fit([[1.0, 2.0]])
    with pytest.raises(ValueError):
        IteratedMSTEmbedding(n_msts=1, n_epochs=0).fit([[0.0], [np.nan]])
