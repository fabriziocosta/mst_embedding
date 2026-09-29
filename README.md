# IMSTE

**Iterated Minimum Spanning Tree Embedding (IMSTE)** is a scikit-learn
transformer for learning low-dimensional coordinates from numeric data. It
builds a weighted graph from edge-disjoint minimum spanning trees, then
optimizes one coordinate vector per sample.

The package has two explicit estimators: `IteratedMinimumSpanningTreeEmbedder`
learns coordinates for the training rows, while `InductiveIMSTE` also learns
an MLP that maps new rows into that space.

## Install

From a source checkout:

```bash
python -m pip install .
```

For notebooks, install the optional plotting and widget dependencies:

```bash
python -m pip install ".[notebook]"
```

## Quick start

IMSTE uses Euclidean distances to build its graph. Scale features first when
they have different ranges. This small example uses scikit-learn's built-in
digits dataset, so it does not need a download:

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from imste import IteratedMinimumSpanningTreeEmbedder

digits = load_digits()
rng = np.random.default_rng(42)
rows = rng.choice(len(digits.data), size=500, replace=False)
X = StandardScaler().fit_transform(digits.data[rows])

mapper = IteratedMinimumSpanningTreeEmbedder(
    n_msts=5,
    n_epochs=300,
    random_state=42,
)
embedding = mapper.fit_transform(X)

print(embedding.shape)  # (500, 2)
```

The labels are not passed to the estimator. Use them afterward to color or
score a visualization if needed.

## How it works

### Build a graph

The estimator computes pairwise Euclidean distances, constructs `n_msts`
edge-disjoint minimum spanning trees, and joins their edges into one graph.
An edge selected in tree rank `r` receives weight `1 / r`, so edges from later
trees contribute less to attraction.

### Optimize coordinates

The embedding objective uses squared Euclidean distances. For an embedded pair
`i, j`, let `d² = sum((y_i - y_j) ** 2)`. Graph edges use logarithmic
attraction, `log1p(d² + epsilon)`. Sampled non-edges use logistic repulsion,
`softplus((logistic_margin - d²) / logistic_temperature)`.

`lambda_rep` sets the repulsion share of the objective; attraction receives
`1 - lambda_rep`. Positive edge losses are normalized by the total graph-edge
weight. Negative losses use an unweighted mean. Each positive edge contributes
`negative_ratio` sampled non-neighbors from each endpoint.

The graph is exact and dense pairwise distances are computed, so graph
construction uses quadratic time and memory in the number of samples. For
larger datasets, start with a subset and increase its size as needed.

## Choose how to transform rows

### Transductive: training rows only

Use the core estimator when you want optimized coordinates for the fitted
rows. Its `transform` accepts only the exact training matrix in its original
row order; it raises an error for new or reordered rows.

```python
from imste import IteratedMinimumSpanningTreeEmbedder

mapper = IteratedMinimumSpanningTreeEmbedder(n_msts=10).fit(X_train)
Z_train = mapper.transform(X_train)
```

### Inductive: map new rows with an MLP

Import `InductiveIMSTE` from the explicit `imste.inductive` module. It fits a
transductive IMSTE embedding, then trains an MLP to predict those coordinates:

```python
from imste import IteratedMinimumSpanningTreeEmbedder
from imste.inductive import InductiveIMSTE

mapper = InductiveIMSTE(
    embedder=IteratedMinimumSpanningTreeEmbedder(
        n_msts=10,
        random_state=42,
    ),
    n_layers=6,
    layer_size=128,
    dropout=0.1,
    min_epochs=100,
    patience=20,
).fit(X_train)

Z_train = mapper.transform(X_train)  # MLP predictions for training rows
Z_new = mapper.transform(X_new)      # MLP predictions for new rows
Z_reference = mapper.reference_embedding_  # exact optimized IMSTE coordinates
```

`InductiveIMSTE.transform` always uses the MLP, including for training rows.
Its `reference_embedding_` contains the exact transductive coordinates that
serve as MLP targets. The MLP standardizes its inputs and targets, holds out
10% of the input rows for validation, and restores the checkpoint with the
best validation loss. The IMSTE coordinates use all input rows; the validation
split only controls MLP early stopping. Predictions approximate the reference
embedding and do not extend or re-optimize its graph.

## Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `n_msts` | `10` | Number of edge-disjoint spanning trees |
| `n_components` | `2` | Embedding dimensions |
| `n_epochs` | `1000` | Coordinate-optimization epochs |
| `batch_size` | `4096` | Positive graph edges per optimization step |
| `learning_rate` | `0.05` | Coordinate optimizer learning rate |
| `negative_ratio` | `5` | Non-neighbors sampled per endpoint of each positive edge |
| `lambda_rep` | `0.5` | Repulsion share, from 0 to 1 |
| `logistic_margin` | `1.0` | Squared-distance margin for repulsion |
| `logistic_temperature` | `0.5` | Logistic repulsion softness; must be positive |
| `epsilon` | `1e-4` | Smoothing for the logarithmic attraction |
| `random_state` | `42` | Seed for initialization, shuffling, and sampling |
| `device` | `"auto"` | `"cpu"`, `"mps"`, or automatic selection |

For `device="auto"`, macOS uses PyTorch MPS for datasets with at least 2,048
rows; smaller datasets use the CPU. Other systems use the CPU. The selected
device is available as `mapper.device_` after fitting.

### MLP parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `n_layers` | `6` | Hidden layers in `InductiveIMSTE` |
| `layer_size` | `128` | Units in each hidden layer |
| `dropout` | `0.1` | Dropout probability |
| `epochs` | `200` | Maximum MLP training epochs |
| `min_epochs` | `100` | Minimum epochs before early stopping |
| `patience` | `20` | Epochs without validation improvement before stopping |
| `batch_size` | `256` | Rows per MLP update and prediction batch |
| `learning_rate` | `0.001` | MLP optimizer learning rate |

These are `InductiveIMSTE` parameters. Configure the graph and coordinate
optimization by passing an `IteratedMinimumSpanningTreeEmbedder` as `embedder`.

## Notebooks

- [Interactive 2D embedding](notebooks/digits_2d_interactive.ipynb): choose a
  dataset, fit settings, and optionally compare MLP predictions with the
  optimized coordinates.
- [Interactive 3D embedding](notebooks/digits_3d_interactive.ipynb): explore
  three-dimensional embeddings with Plotly.
- [Parameter sweep](notebooks/digits_parameter_sweep.ipynb): compare IMSTE
  settings and reference embeddings on MNIST.
- [High-dimensional dataset gallery](notebooks/high_dim_mst_gallery.ipynb):
  view embeddings across image datasets.

For the full method description and equations, see the
[IMSTE whitepaper](WHITEPAPER.md).
