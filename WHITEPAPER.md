# IMSTE: Iterated Minimum Spanning Tree Embedding

## Abstract

Iterated Minimum Spanning Tree Embedding (IMSTE) is an unsupervised method for
placing a finite set of high-dimensional observations in a lower-dimensional
coordinate space. It builds a weighted graph by repeatedly finding a minimum
spanning tree (MST) and removing its edges before finding the next tree. The
union of these edge-disjoint trees supplies an adaptive set of relationships:
early trees preserve the shortest connections, while later trees add longer
connections that help describe broader structure. A rank-based weight controls
the influence of each tree. The graph is then embedded by optimizing an
attractive loss on graph edges together with a sampled repulsive loss on
non-edges.

The method is deliberately direct. It uses Euclidean distances, a dense MST
construction, and a small differentiable objective. It does not use class
labels, a nearest-neighbor graph, or an out-of-sample projection. This paper
describes the implementation in this repository and its practical limits; it
makes no claim that the method is a drop-in replacement for established
embedding methods.

## 1. Motivation and scope

An embedding aims to represent relationships among observations with fewer
coordinates than the original feature space. IMSTE represents those
relationships through a sequence of spanning trees. The first tree connects
each sample to the dataset through the shortest possible total edge length.
After its edges are removed, the next tree must use different connections.
Repeating this process exposes additional relationships without choosing a
fixed number of nearest neighbors for every point.

The estimator is intended for exploratory visualization and geometric
inspection of a fixed dataset. Its output is a learned coordinate for each
training row. The current implementation does not define coordinates for new,
unseen rows.

## 2. Notation

Let the input be a matrix

$$
X = [x_1,\ldots,x_n]^\top \in \mathbb{R}^{n\times p},
$$

In the matrix dimensions, n is the number of samples and p is the number of
features.

Each sample receives a coordinate in the embedding space, where the output
dimension is q:

$$
y_i \in \mathbb{R}^q.
$$

The number of trees is R.

## 3. Graph construction

The method computes all pairwise Euclidean distances:

$$
D_{ij}=\sqrt{\sum_{k=1}^{p}(x_{ik}-x_{jk})^2}.
$$

For each requested rank, the method finds an MST over the remaining available
edges, adds the tree's edges to the embedding graph, and removes those edges
in both directions. Thus no undirected edge appears in more than one tree.
An edge first selected at rank $r$ receives the fixed inverse-rank weight:

$$
w_{ij}=\frac{1}{r}.
$$

The edge weight depends on its tree rank, not on a separately applied distance
decay.

When all requested trees can be constructed, the resulting graph contains:

$$
|E|=R(n-1).
$$

If removing earlier trees disconnects the available graph, fitting stops with
an error asking for fewer trees. The implementation uses dense Prim
construction with an explicit edge-availability mask. This allows distinct
samples at zero distance to remain valid MST edges.

## 4. Coordinate objective

Each sample receives a trainable coordinate, initialized with small Gaussian
noise. The coordinates are jointly optimized with Adam.

### 4.1 Attraction on graph edges

For each edge in the graph, the squared Euclidean distance between its
embedded coordinates is:

$$
d_{ij}^2=\sum_{k=1}^{q}(y_{ik}-y_{jk})^2.
$$

The estimator uses squared Euclidean distance, so $z_{ij}=d_{ij}^2$. The
logarithmic objective uses the pairwise edge probability:

$$
q_{ij}=\frac{1}{1+z_{ij}+\epsilon}.
$$

The attractive term is

$$
L_{\mathrm{attr}}=
\frac{1}{\sum_{(i,j)\in E}w_{ij}}\sum_{(i,j)\in E}
w_{ij}\bigl[-\log(q_{ij})\bigr]
=\frac{1}{\sum_{(i,j)\in E}w_{ij}}\sum_{(i,j)\in E}
w_{ij}\log(1+z_{ij}+\epsilon).
$$

Minimizing it brings graph-connected samples together. The MST-rank weight
applies to positive edges only.

Weight-sum normalization keeps the attraction scale from falling simply
because more, lower-weight MST ranks were added. With minibatch optimization,
the implementation scales each minibatch's weighted mean by the graph-wide
ratio `|E| / sum(w)` to estimate this normalized objective.

This logarithmic objective is the only built-in attraction. It always acts on
squared Euclidean distances and uses the same edge weights and weight-sum
normalization described above.

### 4.2 Repulsion on sampled non-edges

For each undirected positive edge in a minibatch, the implementation samples
negative targets for both endpoints. This makes sampling independent of the
arbitrary orientation used to store an edge. Targets are sampled uniformly
from nodes that are neither the source nor one of its graph neighbors. The
squared embedding distance for a sampled pair is:

$$
d_{uv}^2=\sum_{k=1}^{q}(y_{uk}-y_{vk})^2.
$$

For each sampled negative pair, the logistic margin loss is:

Here $z_{uv}=d_{uv}^2$ is the squared Euclidean embedding distance.

$$
L_{uv}^{-}=\operatorname{softplus}\left(\frac{m-z_{uv}}{\tau}\right),
$$

with `logistic_margin` $m$ and `logistic_temperature` $\tau$.
The negative loss is averaged over sampled pairs without MST-rank weights.
Nearby negative pairs incur a larger cost and are pushed apart. If a graph
has no eligible negative pairs, the repulsive term is zero for that step.

The combined minibatch objective is

$$
L=(1-\lambda)L_{\mathrm{attr}}+\lambda L_{\mathrm{rep}},
$$

The parameter `lambda_rep` is the repulsion share $\lambda\in[0,1]$; the
attraction share is $1-\lambda$. The default $\lambda=0.5$ gives equal
weight to the two terms. The implementation normalizes the weighted
positive-edge losses by their graph-wide total weight; the negative term is
unchanged.

The estimator uses the logarithmic attraction and logistic repulsion described
above. The estimator optimizes sample coordinates directly.

## 5. Optimization procedure

For each epoch, the implementation shuffles the positive edges and processes
them in minibatches. For each minibatch, it computes edge attraction, samples
negative pairs for eligible sources, computes repulsion, and updates all
coordinates with Adam. After each update it subtracts the coordinate mean,
removing global translation drift without changing pairwise differences.

The estimator uses Euclidean distance for graph construction and squared
Euclidean distance in the embedding objective, with a fixed inverse-rank
weight for each edge.

| Parameter | Default | Role |
| --- | ---: | --- |
| `n_msts` | 10 | Number of edge-disjoint MSTs |
| `logistic_margin` | 1.0 | Squared-distance margin for logistic repulsion |
| `logistic_temperature` | 0.5 | Softness of the logistic repulsion; must be positive |
| `n_components` | 2 | Number of output dimensions |
| `n_epochs` | 1000 | Number of passes over the positive edges |
| `batch_size` | 4096 | Positive edges per optimization step |
| `learning_rate` | 0.05 | Adam learning rate |
| `negative_ratio` | 5 | Negative samples per endpoint of each positive edge |
| `lambda_rep` | 0.5 | Repulsion share; attraction uses `1 - lambda_rep` |
| `epsilon` | `1e-4` | Smoothing for the pairwise edge probability |
| `random_state` | 42 | Seed for initialization, shuffling, and sampling |
| `device` | `auto` | CPU or Apple MPS optimization backend |

On macOS, `device="auto"` selects MPS for at least 2,048 samples and CPU for
smaller datasets. MPS uses single-precision coordinates; CPU uses
double-precision coordinates. The graph and pairwise distances are constructed
on the CPU in either case.

## 6. Practical use

Euclidean distance makes feature scale part of the model. Features with larger
numeric ranges can dominate graph construction, so inputs should be scaled
appropriately for the dataset. The gallery standardizes a stratified sample
before fitting. Labels in that gallery are used only to color plots; they are
not passed into the estimator's optimization.

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from mst_embedding import IteratedMinimumSpanningTreeEmbedder

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)

embedding = IteratedMinimumSpanningTreeEmbedder(
    n_msts=10,
    n_components=2,
    random_state=42,
).fit_transform(X)
```

The returned array has shape `(n_samples, n_components)`. Coordinates are
centered after every optimization step, but their orientation, reflection, and
scale are not otherwise fixed. Interpret the embedding through relative
relationships and visual patterns, rather than as a coordinate system with
absolute units.

## 7. Computational cost

The pairwise distance matrix requires quadratic memory in the number of
samples:

$$
O(n^2)
$$

Computing all distances scales with the number of samples squared times the
number of features:

$$
O(n^2p)
$$

The dense repeated MST construction takes:

$$
O(Rn^2)
$$

When all requested trees can be constructed, the graph contains:

$$
|E|=R(n-1)
$$

Optimization cost grows with the number of epochs, graph edges, negative ratio,
and output dimension. Let m denote the negative ratio and q the output
dimension. A rough per-epoch bound is:

$$
O(|E|(1+m)q).
$$

Large datasets can therefore require substantial memory and runtime even
when the final embedding has only two dimensions. Sampling a subset before
fitting is often practical for visualization.

## 8. Reproducibility and estimator behavior

`random_state` controls the initial coordinates, edge order within epochs, and
negative sampling. It makes CPU runs repeatable under the same software and
hardware configuration. Different compute backends or software versions may
produce small numerical differences.

The estimator follows the scikit-learn `fit`, `fit_transform`, and `transform`
interface. Since all coordinates are optimized jointly, `transform` only
returns stored coordinates when given the exact training matrix in its
original row order. In the direct mode it raises an error for new or reordered
rows. The estimator also offers an optional post-hoc MLP projection to
approximate the optimized coordinates. The default has six hidden layers of
128 units and 0.1 dropout. It uses a 10% validation split with at least 100
epochs of training and a patience of 20. This learned regressor is an extension
around the core IMSTE objective, not part of the embedding algorithm described
here.

## 9. Limitations and interpretation

- Poor feature scaling can cause some dimensions to dominate Euclidean distances
  and produce a poor graph.
- Dense pairwise distances limit practical sample counts and require quadratic
  memory.
- Repeated MSTs may become impossible before the requested rank because
  previously selected edges can disconnect the remaining graph.
- The embedding is stochastic and non-convex. A fixed seed supports
  reproducibility but does not establish that a result is unique or globally
  optimal.
- The repulsive term samples graph non-neighbors rather than modeling all
  non-edge pairs exactly; the sampled negative losses are unweighted.
- Output distances and axis values are not calibrated quantities. Rotation,
  reflection, and overall scale do not carry intrinsic meaning.
- The core IMSTE objective does not itself define an out-of-sample projection.
  The optional MLP projection is an approximation learned from fitted
  coordinates. Labels are not used during either fit.

## 10. Summary

IMSTE constructs a rank-weighted union of edge-disjoint minimum
spanning trees and learns low-dimensional coordinates by balancing attraction
along graph edges against sampled repulsion between graph non-neighbors. The
approach is compact and easy to inspect, with its main tradeoff being the
quadratic cost of dense graph construction. It is best treated as an
exploratory embedding for appropriately scaled, manageable datasets.
