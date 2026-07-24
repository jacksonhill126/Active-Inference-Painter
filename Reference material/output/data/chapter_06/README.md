# Chapter 6 Raw Data

Generated `--save` runs write paired NPZ arrays plus JSON metadata here for
Chapter 6 generalized-filtering figures. Files are reproducible and ignored by
git; this README keeps the chapter data directory visible.

## Expected sidecars

| Stem | Reconstructs |
|---|---|
| `example_6_1_generalized_filter` | Univariate dynamic process observations, beliefs, prediction errors, and free energy. |
| `example_6_2_multivariate_filter` | Hooke oscillator state, observations, vector beliefs, errors, and free energy. |
| `example_6_6_generalized_coordinates` | Generalized-coordinate measurements, belief orders, recovered velocity, errors, and free energy. |
| `visualize_6_6_correlated_embedding_orders` | `S(γ)`-derived generalized precision matrices for each plotted smoothness value. |
| `example_6_7_multivariate_generalized_coordinates` | Vector generalized-coordinate trajectories, ordinary-filter baseline, full precisions, prediction errors, and summary tracking metrics. |

Each stem has both `<stem>.npz` and `<stem>.json`. The NPZ arrays are numeric
only; the JSON manifest records provenance, shapes, dtypes, figure paths, and
summary statistics.
