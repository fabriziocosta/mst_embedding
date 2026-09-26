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


def _sinusoidal_2d_encoding(
    n_rows: int, n_columns: int, vector_size: int
) -> np.ndarray:
    """Create fixed sinusoidal encodings for row and column patch positions."""
    row_positions, column_positions = np.meshgrid(
        np.arange(n_rows, dtype=np.float64),
        np.arange(n_columns, dtype=np.float64),
        indexing="ij",
    )
    coordinates = (row_positions.ravel(), column_positions.ravel())
    axis_sizes = ((vector_size + 1) // 2, vector_size // 2)
    encoding = np.empty((n_rows * n_columns, vector_size), dtype=np.float64)

    offset = 0
    for positions, axis_size in zip(coordinates, axis_sizes):
        frequency_count = (axis_size + 1) // 2
        frequencies = np.power(
            10_000.0,
            -2.0 * np.arange(frequency_count) / max(axis_size, 1),
        )
        angles = positions[:, None] * frequencies[None, :]
        axis_encoding = np.stack((np.sin(angles), np.cos(angles)), axis=-1).reshape(
            len(positions), -1
        )
        encoding[:, offset : offset + axis_size] = axis_encoding[:, :axis_size]
        offset += axis_size
    return encoding


class ImagePatchRandomProjection(TransformerMixin, BaseEstimator):
    """Project a grid of non-overlapping image patches with one shared matrix.

    Input rows are flattened channel-last images. ``image_shape`` may be
    ``(height, width)`` for grayscale or ``(height, width, channels)`` for
    multichannel images. If omitted, the feature count must be a perfect
    square and is interpreted as a grayscale image. Image borders are padded
    with zeros after normalization to create a regular spatial patch grid. The
    same Gaussian projection matrix is applied to every patch.

    Parameters
    ----------
    n_patches : int or pair of int, default=5
        Number of patch rows and columns. A scalar creates a square grid.
    n_components : int, default=10
        Number of projected values produced by each patch.
    position_encoding_size : int or None, default=None
        Number of 2D sinusoidal position values appended to each patch. When
        omitted, defaults to ``n_components`` so concatenation doubles the
        per-patch feature count.
    image_shape : pair or triple of int, or None, default=None
        Height, width, and optionally channel count of each image. Inferred as
        square grayscale when omitted. Multichannel rows must be flattened in
        channel-last (HWC) order.
    random_state : int or None, default=42
        Seed used to generate the shared projection matrix.
    """

    def __init__(
        self,
        n_patches: int | tuple[int, int] = 5,
        n_components: int = 10,
        position_encoding_size: int | None = None,
        image_shape: tuple[int, ...] | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.n_patches = n_patches
        self.n_components = n_components
        self.position_encoding_size = position_encoding_size
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
        position_encoding_size = self.position_encoding_size
        if position_encoding_size is None:
            position_encoding_size = int(self.n_components)
        elif (
            isinstance(position_encoding_size, (bool, np.bool_))
            or not isinstance(position_encoding_size, numbers.Integral)
            or position_encoding_size < 1
        ):
            raise ValueError("position_encoding_size must be a positive integer or None.")
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
            image_height, image_width, image_channels = side, side, 1
        else:
            shape = tuple(self.image_shape)
            if len(shape) == 2:
                image_height, image_width = _pair_of_positive_integers(
                    "image_shape", shape
                )
                image_channels = 1
            elif len(shape) == 3:
                image_height, image_width = _pair_of_positive_integers(
                    "image_shape", shape[:2]
                )
                image_channels = shape[2]
                if (
                    isinstance(image_channels, (bool, np.bool_))
                    or not isinstance(image_channels, numbers.Integral)
                    or image_channels < 1
                ):
                    raise ValueError("image_shape channel count must be a positive integer.")
                image_channels = int(image_channels)
            else:
                raise ValueError(
                    "image_shape must contain height, width, and optional channels."
                )
            if image_height * image_width * image_channels != X_checked.shape[1]:
                raise ValueError(
                    "image_shape dimensions must multiply to the number of input features."
                )

        patch_height = math.ceil(image_height / n_patch_rows)
        patch_width = math.ceil(image_width / n_patch_columns)

        rng = np.random.RandomState(self.random_state)
        self.components_ = rng.normal(
            loc=0.0,
            scale=1.0 / math.sqrt(int(self.n_components)),
            size=(patch_height * patch_width * image_channels, int(self.n_components)),
        )
        self.image_shape_ = (image_height, image_width, image_channels)
        self.patch_shape_ = (patch_height, patch_width)
        self.n_patch_rows_ = n_patch_rows
        self.n_patch_columns_ = n_patch_columns
        self.position_encoding_size_ = int(position_encoding_size)
        self.position_encoding_ = _sinusoidal_2d_encoding(
            n_patch_rows, n_patch_columns, self.position_encoding_size_
        )
        self.n_features_in_ = X_checked.shape[1]
        self.n_patches_ = n_patch_rows * n_patch_columns
        self.n_features_per_patch_ = int(self.n_components) + self.position_encoding_size_
        self.n_features_out_ = self.n_patches_ * self.n_features_per_patch_
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
        image_height, image_width, image_channels = self.image_shape_
        patch_height, patch_width = self.patch_shape_
        images = X_checked.reshape(
            n_samples, image_height, image_width, image_channels
        )
        pad_height = (-image_height) % patch_height
        pad_width = (-image_width) % patch_width
        if pad_height or pad_width:
            images = np.pad(
                images,
                ((0, 0), (0, pad_height), (0, pad_width), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )

        patches = images.reshape(
            n_samples,
            self.n_patch_rows_,
            patch_height,
            self.n_patch_columns_,
            patch_width,
            image_channels,
        )
        patches = patches.transpose(0, 1, 3, 2, 4, 5).reshape(
            n_samples, self.n_patches_, patch_height * patch_width * image_channels
        )
        projected = patches @ self.components_
        position_encoding = np.broadcast_to(
            self.position_encoding_[None, :, :],
            (n_samples, self.n_patches_, self.position_encoding_size_),
        )
        encoded_patches = np.concatenate((projected, position_encoding), axis=2)
        return encoded_patches.reshape(n_samples, self.n_features_out_)
