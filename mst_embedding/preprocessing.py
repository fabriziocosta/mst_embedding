"""Image preprocessing utilities for IMSTE pipelines."""

from __future__ import annotations

import math
import numbers

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


def _pair_of_positive_integers(name: str, value: object) -> tuple[int, int]:
    if isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)):
        pair = (int(value), int(value))
    else:
        try:
            pair = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"{name} must be a positive integer or a pair of them.") from exc
        if len(pair) != 2:
            raise ValueError(f"{name} must contain exactly two dimensions.")

    if any(
        isinstance(dimension, (bool, np.bool_))
        or not isinstance(dimension, numbers.Integral)
        or dimension < 1
        for dimension in pair
    ):
        raise ValueError(f"{name} dimensions must be positive integers.")
    return int(pair[0]), int(pair[1])


class ImagePatchRandomProjection(TransformerMixin, BaseEstimator):
    """Project a grid of non-overlapping image patches with one shared matrix.

    Input rows are flattened single-channel images. If ``image_shape`` is not
    supplied, the feature count must be a perfect square. Image borders are
    padded with zeros after normalization to create a regular patch grid. The
    same Gaussian projection matrix is applied to every patch.

    Parameters
    ----------
    n_patches : int or pair of int, default=5
        Number of patch rows and columns. A scalar creates a square grid.
    n_components : int, default=10
        Number of projected values produced by each patch.
    image_shape : pair of int or None, default=None
        Height and width of each input image. Inferred as square when omitted.
    random_state : int or None, default=42
        Seed used to generate the shared projection matrix.
    """

    def __init__(
        self,
        n_patches: int | tuple[int, int] = 5,
        n_components: int = 10,
        image_shape: tuple[int, int] | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.n_patches = n_patches
        self.n_components = n_components
        self.image_shape = image_shape
        self.random_state = random_state

    def fit(self, X: object, y: object = None) -> "ImagePatchRandomProjection":
        """Infer the image layout and create the shared projection matrix."""
        del y
        if (
            isinstance(self.n_components, (bool, np.bool_))
            or not isinstance(self.n_components, numbers.Integral)
            or self.n_components < 1
        ):
            raise ValueError("n_components must be a positive integer.")
        n_patch_rows, n_patch_columns = _pair_of_positive_integers(
            "n_patches", self.n_patches
        )
        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)

        if self.image_shape is None:
            side = math.isqrt(X_checked.shape[1])
            if side * side != X_checked.shape[1]:
                raise ValueError(
                    "Cannot infer image_shape: the number of features is not a "
                    "perfect square. Set image_shape=(height, width)."
                )
            image_height, image_width = side, side
        else:
            image_height, image_width = _pair_of_positive_integers(
                "image_shape", self.image_shape
            )
            if image_height * image_width != X_checked.shape[1]:
                raise ValueError(
                    "image_shape dimensions must multiply to the number of input features."
                )

        patch_height = math.ceil(image_height / n_patch_rows)
        patch_width = math.ceil(image_width / n_patch_columns)

        rng = np.random.RandomState(self.random_state)
        self.components_ = rng.normal(
            loc=0.0,
            scale=1.0 / math.sqrt(int(self.n_components)),
            size=(patch_height * patch_width, int(self.n_components)),
        )
        self.image_shape_ = (image_height, image_width)
        self.patch_shape_ = (patch_height, patch_width)
        self.n_patch_rows_ = n_patch_rows
        self.n_patch_columns_ = n_patch_columns
        self.n_features_in_ = X_checked.shape[1]
        self.n_patches_ = n_patch_rows * n_patch_columns
        self.n_features_out_ = self.n_patches_ * int(self.n_components)
        return self

    def transform(self, X: object) -> np.ndarray:
        """Return concatenated random projections of the image patches."""
        check_is_fitted(self, attributes=["components_", "image_shape_"])
        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, expected {self.n_features_in_}."
            )

        n_samples = X_checked.shape[0]
        image_height, image_width = self.image_shape_
        patch_height, patch_width = self.patch_shape_
        images = X_checked.reshape(n_samples, image_height, image_width)
        pad_height = (-image_height) % patch_height
        pad_width = (-image_width) % patch_width
        if pad_height or pad_width:
            images = np.pad(
                images,
                ((0, 0), (0, pad_height), (0, pad_width)),
                mode="constant",
                constant_values=0.0,
            )

        patches = images.reshape(
            n_samples,
            self.n_patch_rows_,
            patch_height,
            self.n_patch_columns_,
            patch_width,
        )
        patches = patches.transpose(0, 1, 3, 2, 4).reshape(
            n_samples, self.n_patches_, patch_height * patch_width
        )
        projected = patches @ self.components_
        return projected.reshape(n_samples, self.n_features_out_)
