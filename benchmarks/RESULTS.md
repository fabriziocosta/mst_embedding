# Performance refactor measurements

These measurements are a local CPU reference, not a hardware-independent
performance claim. Inputs are 32-feature, 20-center `make_blobs` datasets with
`cluster_std=3`, seed 42. Fits used 15 MSTs and 50 epochs. The 2,000-row
results use seeds 0 and 1; the 10,000-row results use seed 0. Numba and
PyNNDescent were warmed on a separate 256-row input. UMAP used 15
neighbors, 50 epochs, and matching seeds. Trustworthiness was measured on at
most 2,000 rows; kNN preservation is mean overlap of 10-neighbor sets.

| Samples | Version / method | Graph (s) | Optimizer (s) | Fit (s) | Trustworthiness | kNN preservation |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2,000 | Baseline Torch / FAMST | 2.477 | 0.825 | 3.302 | 0.9816 | 0.1323 |
| 2,000 | Numba SGD / FAMST | 0.992 | 0.157 | 1.150 | 0.9808 | 0.1336 |
| 2,000 | UMAP reference | — | — | 3.989 | 0.9822 | 0.1495 |
| 10,000 | Numba SGD / FAMST | 19.295 | 0.971 | 20.266 | 0.9807 | 0.0321 |
| 10,000 | UMAP reference | — | — | 3.811 | 0.9805 | 0.0342 |

The 2,000-row comparison used two warmed runs for baseline and refactor. The
10,000-row FAMST run needed a 120-neighbor starting graph and cap of 512 to
construct all requested trees; its figures are from one run. A 50,000-
row run was not practical in this pass because graph construction already
dominated the 10,000-row fit. The 10,000-row result shows that optimizer work is
small after the refactor, while adaptive ANN construction remains the main
performance target.

Reproduce with:

```sh
python benchmarks/performance_refactor.py --sizes 2000 10000 50000 --repeats 3 --epochs 50
```

The script prints graph-construction, optimizer, and total-fit timings along
with the two neighborhood metrics. It warms compilation outside measured
runs and includes UMAP when the notebook extras are installed.
