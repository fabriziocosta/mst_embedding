IMSTE (Iterated Minimum Spanning Tree Embedding) is an unsupervised method based on repeated minimum spanning trees.
The original/default output is 2D; `n_components` generalizes the optimized
coordinate matrix to other dimensions, including 3D for interactive plots.

Input:
- A data matrix `X` of shape `(n_samples, n_features)`.
- Number of MST iterations `R`.
- Rank-weight exponent `alpha` (default: 1.0).
- Number of embedding dimensions `n_components` (default: 2).
- Number of embedding epochs.
- Negative sampling ratio.
- Repulsion coefficient `lambda_rep`.
- Random seed.

The algorithm has two stages: graph construction and coordinate optimization.

### Optional hierarchical approximation

`graph_mode="hierarchical"` replaces only graph construction. MiniBatchKMeans
partitions the observations; its centroids are connected with the ordinary
edge-disjoint IMST routine. For every coarse centroid edge `(a, b)`, the
ordinary local IMST is then run on the complete union `C_a ∪ C_b`, using the
original observations. Independent local problems can run in parallel.

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

Clustering restricts which regions are compared; every final edge still joins
two original observations. Local problems may rediscover an undirected edge.
The result is a set union: duplicate edges appear once and keep the strongest
local rank weight. The exact graph remains the default mode.

## 1. Construct the IMSTE graph

Compute the full pairwise Euclidean distance matrix:

$$
D_{ij}=\|x_i-x_j\|_2.
$$

Set the diagonal to zero.

Create a working copy:

```python
D_work = D.copy()
```

Then repeat for ranks

$$
r=1,\ldots,R.
$$

At each iteration:

1. Compute a minimum spanning tree over the current distance graph `D_work`.
2. Treat the resulting MST as undirected.
3. Add all MST edges to the embedding graph.
4. Assign every edge discovered at this iteration the weight

$$
w_{ij}=r^{-\alpha},
$$

where `alpha` controls how quickly later MSTs lose influence. The default
`alpha=1` gives the original inverse-rank weighting; `alpha=0` gives every
MST rank equal weight, and larger values reduce the weight of later ranks more
quickly.

5. Remove the selected MST edges from `D_work` in both directions so that later MSTs cannot reuse them.

Conceptually:

```python
for rank in range(1, R + 1):
    T = minimum_spanning_tree(D_work)

    for (i, j) in T.edges:
        edges.append((i, j))
        weights.append(rank ** (-alpha))

        D_work[i, j] = 0
        D_work[j, i] = 0
```

After `R` iterations, the graph is the union of `R` edge-disjoint minimum spanning trees.

Do not add k-nearest-neighbor edges, closure edges, or any other graph construction.

## 2. Initialize the embedding

Create one trainable coordinate vector per input sample:

$$
y_i\in\mathbb R^{n_{components}}.
$$

Initialize all coordinates with small random Gaussian noise, for example:

```python
Y ~ Normal(0, 1e-3)
```

Use a fixed random seed for reproducibility.

Use Adam to optimize the coordinates.

A learning rate around `0.05` worked well in the initial experiments.

## 3. Attractive loss

For every positive graph edge `(i,j)` with weight `w_ij`, compute

$$
d_{ij}^2=\|y_i-y_j\|^2.
$$

The attractive loss is

$$
L_{\text{attr}}
=
\frac{1}{|E|}
\sum_{(i,j)\in E}
w_{ij}
\log\left(1+d_{ij}^2\right).
$$

Equivalently, in PyTorch:

```python
d2_pos = ((Y[i] - Y[j]) ** 2).sum(dim=1)
attraction = (w * torch.log1p(d2_pos)).mean()
```

Do not use a UMAP cross-entropy objective. This simple attraction function is the baseline algorithm.

## 4. Random non-neighbor repulsion

For each positive edge in a minibatch, randomly sample several negative pairs.

A useful initial value is:

```python
negative_ratio = 4
```

For each positive source node `i`, sample random nodes `j_neg` uniformly from the dataset.

Reject a sampled pair if:
- `i == j_neg`, or
- `(i, j_neg)` is already an edge in the IMSTE graph.

For every accepted negative pair compute

$$
d_{\text{neg}}^2
=
\|y_i-y_j\|^2.
$$

Use the repulsive loss

$$
L_{\text{rep}}
=
\frac{1}{N_{\text{neg}}}
\sum
\frac{1}
{1+d_{\text{neg}}^2+\epsilon},
$$

where a small value such as

$$
\epsilon=10^{-4}
$$

can be used for numerical safety.

In PyTorch:

```python
d2_neg = ((Y[ni] - Y[nj]) ** 2).sum(dim=1)
repulsion = (1.0 / (1.0 + d2_neg + 1e-4)).mean()
```

Because the optimizer minimizes the loss, this term pushes randomly sampled non-neighbors farther apart.

## 5. Total objective

Optimize

$$
L
=
L_{\text{attr}}
+
\lambda_{\text{rep}}L_{\text{rep}}.
$$

Use

```python
lambda_rep = 1.0
```

as the default baseline.

The initial experiments also tested other values, but `1.0` should be the default implementation.

## 6. Optimization

For each epoch:

1. Randomly shuffle all positive graph edges.
2. Process them in minibatches.
3. Compute positive attraction.
4. Generate random negative pairs for that minibatch.
5. Compute repulsion.
6. Backpropagate the combined loss.
7. Update the coordinates with Adam.

After each optimization step, optionally remove global translation drift:

```python
with torch.no_grad():
    Y -= Y.mean(dim=0, keepdim=True)
```

This does not affect relative geometry.

A reasonable initial configuration is:

```python
epochs = 500
batch_size = 4096
learning_rate = 0.05
negative_ratio = 4
lambda_rep = 1.0
```

## 7. Output

Return the final matrix

```python
Y.shape == (n_samples, n_components)
```

Optionally normalize each output dimension by its standard deviation for visualization:

```python
Y /= Y.std(axis=0, keepdims=True)
```

## 8. Reference experiment

Use the scikit-learn digits dataset as the first test.

Steps:

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
labels = digits.target
```

Run the embedding for several values of the number of MST iterations:

```python
R = [1, 2, 4, 8, 16, 32]
```

For each value of `R`, create a 2D scatter plot:
- x coordinate = embedding dimension 1,
- y coordinate = embedding dimension 2,
- color points according to `digits.target`.

Use the same random seed and optimization parameters for every run.

The purpose of this experiment is to observe how the geometry changes as more edge-disjoint MSTs are added.

## 9. Important constraints

Keep the first implementation deliberately simple.

Do not add:
- kNN graphs,
- UMAP graph construction,
- UMAP stochastic edge sampling,
- probabilistic cross-entropy,
- local distance scaling,
- PCA initialization,
- spectral initialization,
- graph-distance-2 closure,
- distance-dependent edge weighting,
- class labels during training.

The graph-edge weight depends only on the MST iteration rank:

$$
w_{ij}=r^{-\alpha},
$$

where `r` is the MST iteration in which the edge first appears and `alpha`
is the configurable `rank_weight_exponent` parameter (default `1.0`).

The labels are used only for coloring the final visualization.
