# Chapter 7 Raw Data

Generated `--save` runs write paired NPZ arrays plus JSON metadata here for
Chapter 7 active-inference figures and animations. Files are reproducible and
ignored by git; this README keeps the chapter data directory visible.

## Expected sidecars

| Stem | Reconstructs |
|---|---|
| `example_7_2_active_inference` | Univariate state, belief, action, observations, prediction errors, and free energy. |
| `example_7_5_multivariate_active_inference` | 2-D active and no-action trajectories, generalized measurements, action vectors, prediction errors, free energy, and preference-error metrics. |
| `animation_7_5_multivariate_active_inference` | The same §7.5 time-series plus frame stride/settings used to render the GIF. |

Each stem has both `<stem>.npz` and `<stem>.json`. The NPZ arrays are numeric
only; the JSON manifest records provenance, shapes, dtypes, figure paths, and
summary statistics.
