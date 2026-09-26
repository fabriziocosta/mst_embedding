# Iterated MST Embedding

`mst-embedding` provides a scikit-learn-compatible transformer that learns a
two-dimensional embedding from a union of edge-disjoint minimum spanning trees.
It follows the algorithm described in [IDEA.md](IDEA.md).

## Install

```bash
python -m pip install .
```

To run the parameter-sweep notebook, install its plotting and UMAP dependencies:

```bash
python -m pip install ".[notebook]"
```

## Example

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mst_embedding import IteratedMSTEmbedding

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
indices, _ = train_test_split(
    np.arange(len(mnist.data)), train_size=2000,
    random_state=42, stratify=mnist.target,
)
X = StandardScaler().fit_transform(
    np.asarray(mnist.data, dtype=np.float32)[indices] / 255.0
)

embedding = IteratedMSTEmbedding(random_state=42).fit_transform(X)
assert embedding.shape == (len(X), 2)
```

The [parameter-sweep notebook](notebooks/digits_parameter_sweep.ipynb) loads and
caches real MNIST and exposes a configurable, stratified sample size (default
2,000).

The estimator exposes `fit`, `fit_transform`, and `transform`. Since the
coordinates are optimized jointly for all training samples, `transform` returns
the stored coordinates only for the exact training matrix in its original row
order. It does not project unseen samples.

The main parameters are `n_msts=4`, `n_epochs=500`, `batch_size=4096`,
`learning_rate=0.05`, `negative_ratio=4`, `lambda_rep=1.0`, `epsilon=1e-4`,
`random_state=42`, and `device="auto"`. On macOS, `auto` uses PyTorch's MPS
backend for datasets with at least 2,048 samples; smaller workloads use the CPU
because GPU launch overhead was higher for a smaller handwritten-digits dataset.
Set `device="mps"` to force Metal acceleration or `device="cpu"` to force CPU execution. The
selected backend is available as `estimator.device_` after fitting.
