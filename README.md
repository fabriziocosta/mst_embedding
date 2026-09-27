# IMSTE: Iterated Minimum Spanning Tree Embedding

`mst-embedding` provides IMSTE, a scikit-learn-compatible transformer that
learns an embedding from a union of edge-disjoint minimum spanning trees. It
defaults to two dimensions and supports other output dimensions through `n_components`.
The method has two stages: it builds a weighted graph from repeated MSTs, then
optimizes one coordinate vector per input sample. For a detailed explanation,
see the [whitepaper](WHITEPAPER.md).

## Install

```bash
python -m pip install .
```

To run the notebooks, install their plotting dependencies:

```bash
python -m pip install ".[notebook]"
```

## Example

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mst_embedding import IteratedMinimumSpanningTreeEmbedder

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
indices, _ = train_test_split(
    np.arange(len(mnist.data)), train_size=2000,
    random_state=42, stratify=mnist.target,
)
X = StandardScaler().fit_transform(
    np.asarray(mnist.data, dtype=np.float32)[indices]
)

embedding = IteratedMinimumSpanningTreeEmbedder(random_state=42).fit_transform(X)
assert embedding.shape == (len(X), 2)

# Request 3D coordinates for interactive visualization.
embedding_3d = IteratedMinimumSpanningTreeEmbedder(
    n_components=3, random_state=42
).fit_transform(X)
assert embedding_3d.shape == (len(X), 3)
```

## Algorithm

### Build the graph

The method computes all pairwise Euclidean distances and repeatedly builds an
MST, removing each selected edge before constructing the next tree. The union
contains `n_msts` edge-disjoint trees. An edge first selected at rank `r` gets
weight `1 / r`, so later trees contribute less.

### Optimize coordinates

Each observation gets a trainable coordinate vector `y_i` in
`n_components` dimensions. For a pair, let `t = sum_k (y_ik - y_jk) ** 2` be
its squared Euclidean embedding distance. Define the pairwise edge probability
`q(t) = 1 / (1 + t + epsilon)`. A positive graph edge uses the Bernoulli
negative log-likelihood `-log(q) = log(1 + t + epsilon)`, multiplied by its
MST-rank weight. For each positive-edge endpoint, the optimizer samples
`negative_ratio` graph non-neighbors and applies the selected repulsion loss.
The default `repulsion_type="bernoulli"` uses `-log(1 - q)`, equivalently
`log(1 + 1 / (t + epsilon))`. Set `repulsion_type="inverse_distance"` to
restore the earlier `1 / (1 + t + epsilon)` penalty. Negative losses are
averaged without MST-rank weights. Adam minimizes the convex combination
`(1 - lambda_rep) * attraction + lambda_rep * repulsion`. Here `lambda_rep` is
the repulsion share in `[0, 1]`;
the attraction share is `1 - lambda_rep`. The default `0.5` gives equal weight
to the two mean losses.

Training shuffles the graph edges each epoch and processes them in batches.
The random seed controls initialization, edge ordering, and negative sampling.
Labels are not used during fitting; they can be used afterward to color a
visualization. `transform` returns the fitted coordinates only for the original
training matrix because the model does not define an out-of-sample projection.

For flattened images, `ImagePatchRandomProjection` can reduce features after
normalization while retaining local pixel layout. It splits each image into a
2D grid of non-overlapping spatial patches, flattens all channels within each
patch, applies one shared random projection to every patch, and flattens the
projected patches for IMSTE. Set `image_shape=(height, width, channels)` for
channel-last color images; grayscale images can use `(height, width)` or infer
a square shape. Each projected patch also receives a fixed 2D sinusoidal
position vector; `position_encoding_size` controls its length and defaults to
the projected patch size, doubling the combined per-patch feature count. The
default grid is 5×5 patches and each patch maps to 10 image features plus 10
position features; incomplete border patches are zero-padded.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mst_embedding import (
    ImagePatchRandomProjection,
    IteratedMinimumSpanningTreeEmbedder,
)

image_pipeline = make_pipeline(
    StandardScaler(),
    ImagePatchRandomProjection(n_patches=5, n_components=10, random_state=42),
    IteratedMinimumSpanningTreeEmbedder(random_state=42),
)
embedding = image_pipeline.fit_transform(X)
```

The [parameter-sweep notebook](notebooks/digits_parameter_sweep.ipynb) loads and
caches real MNIST and exposes a configurable, stratified sample size (default
2,000). The [interactive 3D notebook](notebooks/digits_3d_interactive.ipynb)
uses sliders for sample size and estimator settings, plus a
dropdown for Bernoulli or inverse-distance repulsion, with `n_components=3`
and Plotly controls to rotate the learned embedding.
The [interactive 2D notebook](notebooks/digits_2d_interactive.ipynb) adds
sliders for sample count and embedding parameters, plus a
dropdown for Bernoulli or the earlier inverse-distance repulsion; it requires
the notebook extras, including `ipywidgets`.
The [high-dimensional dataset gallery](notebooks/high_dim_mst_gallery.ipynb)
runs and plots 2D IMSTE embeddings across several image datasets.

The estimator exposes `fit`, `fit_transform`, and `transform`. Since the
coordinates are optimized jointly for all training samples, `transform` returns
the stored coordinates only for the exact training matrix in its original row
order. It does not project unseen samples.

The main parameters are `n_msts=10`, `attraction_normalization="mean"`,
`n_components=2`, `n_epochs=1000`, `batch_size=4096`, `learning_rate=0.05`,
`negative_ratio=5`, `lambda_rep=0.5`, `epsilon=1e-4`, `random_state=42`, and
`device="auto"`. Edge weights decay by MST rank as
`1 / rank`, so later trees receive smaller weights. `negative_ratio` samples
that many non-neighbors from each endpoint of each positive edge. On macOS,
`auto` uses PyTorch's MPS
backend for datasets with at least 2,048 samples; smaller workloads use the CPU
because GPU launch overhead was higher for a smaller handwritten-digits dataset.
Set `device="mps"` to force Metal acceleration or `device="cpu"` to force CPU execution. The
selected backend is available as `estimator.device_` after fitting.

`attraction_normalization="mean"` preserves the original attraction loss
normalization by edge count. Set it to `"weight_sum"` to normalize by the sum
of edge weights, which reduces the attraction-scale change as more low-weight
MST ranks are added. This option is a comparison variant; it does not change
the default objective.

The attraction and repulsion losses can be replaced independently with
`attraction_loss_fn` and `repulsion_loss_fn`. Each callable must return a scalar
PyTorch tensor that remains differentiable with respect to its distance input.
For example, to use a quadratic attraction while keeping the default
repulsion:

```python
import torch
from mst_embedding import IteratedMinimumSpanningTreeEmbedder

def quadratic_attraction(positive_squared_distances, edge_weights):
    return torch.mean(edge_weights * positive_squared_distances)

embedding = IteratedMinimumSpanningTreeEmbedder(
    attraction_loss_fn=quadratic_attraction,
).fit_transform(X)
```

The built-in `log_attraction_loss`, `bernoulli_repulsion_loss`, and
`inverse_distance_repulsion_loss` functions are exported from `mst_embedding`
for reuse or comparison.
