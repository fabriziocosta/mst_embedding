# IMSTE: Iterated Minimum Spanning Tree Embedding

`imste` provides IMSTE, a scikit-learn-compatible transformer that
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

from imste import IteratedMinimumSpanningTreeEmbedder

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
`n_components` dimensions. For a pair, let `t = sum_k (y_ik - y_jk) ** 2`.
The estimator always uses squared Euclidean distance. Positive graph edges use
logarithmic attraction, `log1p(t + epsilon)`, and sampled non-edges use a
logistic margin loss. Negative losses are averaged without MST-rank weights.
Adam minimizes the convex combination
`(1 - lambda_rep) * attraction + lambda_rep * repulsion`. Here `lambda_rep` is
the repulsion share in `[0, 1]`;
the attraction share is `1 - lambda_rep`. The default `0.5` gives equal weight
to the two mean losses.

Training shuffles the graph edges each epoch and processes them in batches.
The random seed controls initialization, edge ordering, and negative sampling.
Labels are not used during fitting; they can be used afterward to color a
visualization.

### Transform new samples

By default, `transform` returns the fitted coordinates only for the original
training matrix. To project unseen rows, set `transform_method="mlp"`. After
optimizing the embedding, the estimator trains a standard MLP to predict those
coordinates. Its defaults are six hidden layers of 128 units with 0.1 dropout.
It standardizes its inputs and targets and uses a shuffled 10% validation split
for early stopping. The best validation checkpoint is restored, then
predictions are returned in the embedding's original coordinate scale.

```python
from imste import IteratedMinimumSpanningTreeEmbedder

mapper = IteratedMinimumSpanningTreeEmbedder(
    transform_method="mlp",
    mlp_n_layers=6,
    mlp_layer_size=128,
    mlp_dropout=0.1,
    mlp_min_epochs=100,
    mlp_patience=20,
    random_state=42,
).fit(X_train)

Z_train = mapper.transform(X_train)  # returns the optimized IMSTE coordinates
Z_new = mapper.transform(X_new)      # uses the learned MLP
Z_train_predicted = mapper.transform_with_mlp(X_train)  # apply net to any rows
```

The MLP approximates the fitted coordinates; it does not re-optimize the IMSTE
graph for new points. Its training controls are `mlp_epochs` (maximum epochs,
default 200), `mlp_min_epochs` (default 100), `mlp_patience` (default 20),
`mlp_batch_size`, and `mlp_learning_rate`. `transform_with_mlp` explicitly
applies the network to any rows, including training rows; with the default
`transform`, exact training rows continue to return the optimized IMSTE
coordinates.

The [parameter-sweep notebook](notebooks/digits_parameter_sweep.ipynb) loads and
caches real MNIST and exposes a configurable, stratified sample size (default
2,000). The [interactive 3D notebook](notebooks/digits_3d_interactive.ipynb)
uses sliders for sample size and estimator settings, with `n_components=3`
and Plotly controls to rotate the learned embedding. The [interactive 2D
notebook](notebooks/digits_2d_interactive.ipynb) adds controls for sample
count and embedding parameters; it requires the notebook extras, including
`ipywidgets`.
The [high-dimensional dataset gallery](notebooks/high_dim_mst_gallery.ipynb)
runs and plots 2D IMSTE embeddings across several image datasets.

The estimator exposes `fit`, `fit_transform`, and `transform`. The default
`transform_method="direct"` returns stored coordinates for the exact training
matrix only. `transform_method="mlp"` additionally fits an MLP so `transform`
can project unseen rows.

The main parameters are `n_msts=10`, `logistic_margin=1.0`,
`logistic_temperature=0.5`, `n_components=2`, `n_epochs=1000`,
`batch_size=4096`, `learning_rate=0.05`,
`negative_ratio=5`, `lambda_rep=0.5`, `epsilon=1e-4`, `random_state=42`, and
`device="auto"`. Projection parameters are `transform_method="direct"`,
`mlp_n_layers=6`, `mlp_layer_size=128`, `mlp_dropout=0.1`, `mlp_epochs=200`,
`mlp_min_epochs=100`, `mlp_patience=20`,
`mlp_batch_size=256`, and `mlp_learning_rate=0.001`. Edge weights decay by MST
rank as `1 / rank`, so later trees receive smaller weights. `negative_ratio` samples
that many non-neighbors from each endpoint of each positive edge. On macOS,
`auto` uses PyTorch's MPS
backend for datasets with at least 2,048 samples; smaller workloads use the CPU
because GPU launch overhead was higher for a smaller handwritten-digits dataset.
Set `device="mps"` to force Metal acceleration or `device="cpu"` to force CPU execution. The
selected backend is available as `estimator.device_` after fitting.

Attraction is normalized by the sum of graph edge weights. This keeps its
scale more consistent as additional low-weight MST ranks are added.
Logistic margin and temperature control the negative-pair loss. The built-in
`log_attraction_loss` and `logistic_repulsion_loss` functions are exported from
`imste` for reuse.
