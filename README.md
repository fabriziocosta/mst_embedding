# Iterated MST Embedding

`mst-embedding` provides a scikit-learn-compatible transformer that learns a
two-dimensional embedding from a union of edge-disjoint minimum spanning trees.
It follows the algorithm described in [IDEA.md](IDEA.md).

## Install

```bash
python -m pip install .
```

## Example

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from mst_embedding import IteratedMSTEmbedding

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)

embedding = IteratedMSTEmbedding(random_state=42).fit_transform(X)
assert embedding.shape == (len(X), 2)
```

The estimator exposes `fit`, `fit_transform`, and `transform`. Since the
coordinates are optimized jointly for all training samples, `transform` returns
the stored coordinates only for the exact training matrix in its original row
order. It does not project unseen samples.

The main parameters are `n_msts=4`, `n_epochs=500`, `batch_size=4096`,
`learning_rate=0.05`, `negative_ratio=4`, `lambda_rep=1.0`, `epsilon=1e-4`, and
`random_state=42`.
