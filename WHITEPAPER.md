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
relationships through a hierarchy of spanning trees. The first tree connects
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

The number of trees is R. The non-negative exponent that controls how quickly
later trees lose influence satisfies:

$$
\alpha \geq 0.
$$

## 3. Graph construction

The method first computes all pairwise Euclidean distances:

$$
D_{ij}=\lVert x_i-x_j\rVert_2.
$$

For each requested rank, the method finds an MST over the remaining available
edges, adds the tree's edges to the embedding graph, and removes those edges
in both directions. Thus no undirected edge appears in more than one tree.
Each edge receives a weight based on the rank of the tree that first selected
it:

$$
w_{ij}=r^{-\alpha}.
$$

The default exponent gives inverse-rank weights. Setting the exponent to zero
gives every rank equal weight; a larger exponent reduces later trees'
influence more quickly. The edge weight depends on its tree rank, not on a
separately applied distance decay. The two common settings are:

$$
\alpha=1, w_r=r^{-1}.
$$

$$
\alpha=0, w_r=1.
$$

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

For each edge in the graph, the squared distance between its embedded
coordinates is:

$$
d_{ij}^2=\lVert y_i-y_j\rVert_2^2.
$$

The attractive term is

$$
L_{\mathrm{attr}}=
\frac{1}{|E|}\sum_{(i,j)\in E}
w_{ij}\log(1+d_{ij}^2).
$$

Minimizing this term brings graph-connected samples together. The logarithm
keeps the attraction increasing while reducing its growth for already distant
pairs.

### 4.2 Repulsion on sampled non-edges

For each positive-edge source in a minibatch, the implementation samples
negative targets uniformly from nodes that are neither the source nor one of
its graph neighbors. The squared embedding distance for a sampled pair is:

$$
d_{uv}^2=\lVert y_u-y_v\rVert_2^2.
$$

The repulsive contribution is:

$$
L_{\mathrm{rep}}=
\frac{1}{|S|}\sum_{(u,v)\in S}
\frac{1}{1+d_{uv}^2+\epsilon},
$$

Here, S is the sampled set of negative pairs and epsilon is a small
positive constant that prevents division by zero. Because optimization
minimizes this term, nearby negative pairs incur a larger cost and are pushed
apart. If a graph has no eligible negative pairs, the repulsive term is set to
zero for that step.

The combined minibatch objective is

$$
L=L_{\mathrm{attr}}+\lambda_{\mathrm{rep}}L_{\mathrm{rep}},
$$

The parameter lambda_rep sets the relative strength of repulsion. The
implementation uses the mean of weighted positive-edge losses and the mean of
sampled negative losses; it does not normalize the positive term by the sum of
edge weights.

Both loss functions are modular. The estimator accepts an optional
`attraction_loss_fn` callable that receives positive squared distances and edge
weights, and an optional `repulsion_loss_fn` callable that receives negative
squared distances and epsilon. Each callable must return a scalar
differentiable PyTorch tensor. If either hook is omitted, its built-in default
is used. The estimator optimizes the sample coordinates; custom loss functions
are not themselves trained.

## 5. Optimization procedure

For each epoch, the implementation shuffles the positive edges and processes
them in minibatches. For each minibatch, it computes edge attraction, samples
negative pairs for eligible sources, computes repulsion, and updates all
coordinates with Adam. After each update it subtracts the coordinate mean,
removing global translation drift without changing pairwise differences.

The default estimator settings are:

| Parameter | Default | Role |
| --- | ---: | --- |
| `n_msts` | 8 | Number of edge-disjoint MSTs |
| `rank_weight_exponent` | 1.0 | Decay of edge weights by tree rank |
| `n_components` | 2 | Number of output dimensions |
| `n_epochs` | 1000 | Number of passes over the positive edges |
| `batch_size` | 4096 | Positive edges per optimization step |
| `learning_rate` | 0.05 | Adam learning rate |
| `negative_ratio` | 4 | Negative samples per positive edge source |
| `lambda_rep` | 1.0 | Repulsion coefficient |
| `epsilon` | `1e-4` | Repulsion stabilizer |
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
    n_msts=8,
    n_components=2,
    random_state=42,
).fit_transform(X)
```

The returned array has shape `(n_samples, n_components)`. Coordinates are
centered after every optimization step, but their orientation, reflection, and
scale are not otherwise fixed. Interpret the embedding through relative
relationships and visual patterns, rather than as a coordinate system with
absolute units.

For flattened images, the package also provides `ImagePatchRandomProjection`.
In an sklearn pipeline it can follow `StandardScaler`: it reshapes each row
into a grayscale image or a channel-last color tensor, tiles non-overlapping
patches over the two spatial dimensions, projects each flattened patch with
one shared seeded Gaussian matrix, and flattens the projected patches. Color
channels stay together within each spatial patch. The default grid contains 5
rows and 5 columns of patches. Each projected patch is concatenated with a
fixed 2D sinusoidal position vector whose frequencies vary geometrically.
`position_encoding_size` controls its length and defaults to the number of
projected patch features, doubling the concatenated per-patch feature count.
Image borders are zero-padded after normalization as needed to form a regular
grid.

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
original row order. It raises an error for new or reordered rows. A future
out-of-sample transform would require an additional method for locating and
optimizing new points; it is not part of this algorithm's current interface.

## 9. Limitations and interpretation

- The method captures Euclidean structure in the supplied feature space; poor
  scaling or an unsuitable distance representation can produce a poor graph.
- Dense pairwise distances limit practical sample counts and require quadratic
  memory.
- Repeated MSTs may become impossible before the requested rank because
  previously selected edges can disconnect the remaining graph.
- The embedding is stochastic and non-convex. A fixed seed supports
  reproducibility but does not establish that a result is unique or globally
  optimal.
- The repulsive term samples graph non-neighbors rather than modeling all
  non-edge pairs exactly.
- Output distances and axis values are not calibrated quantities. Rotation,
  reflection, and overall scale do not carry intrinsic meaning.
- The estimator has no out-of-sample projection and does not use labels during
  fitting.

## 10. Summary

IMSTE constructs a rank-weighted union of edge-disjoint minimum
spanning trees and learns low-dimensional coordinates by balancing attraction
along graph edges against sampled repulsion between graph non-neighbors. The
approach is compact and easy to inspect, with its main tradeoff being the
quadratic cost of dense graph construction. It is best treated as an
exploratory embedding for appropriately scaled, manageable datasets.
