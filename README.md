# IMSTE: Iterated Minimum Spanning Tree Embedding

`mst-embedding` provides IMSTE, a scikit-learn-compatible transformer that
learns an embedding from a union of edge-disjoint minimum spanning trees. It defaults to
two dimensions and supports other output dimensions through `n_components`.
It follows the algorithm described in [IDEA.md](IDEA.md).
For a detailed explanation of the approach, see the [whitepaper](WHITEPAPER.md).

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
uses sliders for sample size and estimator settings, with `n_components=3`
and Plotly controls to rotate the learned embedding.
The [interactive 2D notebook](notebooks/digits_2d_interactive.ipynb) adds
sliders for sample count and embedding parameters; it requires the notebook
extras, including `ipywidgets`.
The [high-dimensional dataset gallery](notebooks/high_dim_mst_gallery.ipynb)
runs and plots 2D IMSTE embeddings across several image datasets.

The estimator exposes `fit`, `fit_transform`, and `transform`. Since the
coordinates are optimized jointly for all training samples, `transform` returns
the stored coordinates only for the exact training matrix in its original row
order. It does not project unseen samples.

The main parameters are `n_msts=8`, `rank_weight_exponent=1.0`,
`n_components=2`, `n_epochs=1000`, `batch_size=4096`, `learning_rate=0.05`,
`negative_ratio=4`, `lambda_rep=1.0`, `epsilon=1e-4`, `random_state=42`, and
`device="auto"`. Edge weights decay by MST rank as
`rank ** (-rank_weight_exponent)`: the default of 1.0 gives inverse-rank
weighting, while 0 gives equal weights to all ranks. `negative_ratio` samples
that many non-neighbors from each endpoint of each positive edge. On macOS,
`auto` uses PyTorch's MPS
backend for datasets with at least 2,048 samples; smaller workloads use the CPU
because GPU launch overhead was higher for a smaller handwritten-digits dataset.
Set `device="mps"` to force Metal acceleration or `device="cpu"` to force CPU execution. The
selected backend is available as `estimator.device_` after fitting.

## Hierarchical graph construction

The exact IMSTE graph remains the default. For larger datasets, select the
optional hierarchical approximation:

```python
embedding = IteratedMinimumSpanningTreeEmbedder(
    graph_mode="hierarchical",
    n_clusters=100,
    n_coarse_msts=4,
    n_local_msts=8,
    n_jobs=-1,
    minibatch_size=1024,
    random_state=42,
).fit_transform(X)
```

Its graph construction follows:

```text
MiniBatchKMeans
    ↓
IMST over cluster centroids
    ↓
for every centroid edge (a,b):
    IMST over C_a ∪ C_b
    ↓
parallel execution
    ↓
set union of global sample edges
    ↓
existing IMSTE embedding objective
```

MiniBatchKMeans only limits which dataset regions are compared. The final graph
contains edges between original observations, including intra-cluster edges
that arise within each cluster-pair union. Local graphs can rediscover the
same undirected sample edge; the estimator keeps one copy with its strongest
(earliest-rank) weight. Fitted diagnostics include `cluster_labels_`,
`cluster_centers_`, `coarse_graph_edges_`, `graph_edges_`, and `graph_weights_`.
The [comparison script](notebooks/compare_exact_hierarchical.py) measures graph
construction and total runtime over several cluster counts.

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

The built-in `log_attraction_loss` and `inverse_distance_repulsion_loss`
functions are exported from `mst_embedding` for reuse or comparison.
