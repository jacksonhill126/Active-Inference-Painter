# `output/figures/chapter_05/` — Chapter 5 figures

PNG and GIF outputs from the Chapter 5 orchestrators
(`chapters/chapter_05/`). **Ephemeral and gitignored** — regenerate with:

```bash
python scripts/run_all_figures.py --chapters 5
```

Each saved artifact has a matching raw-data sidecar under `output/data/chapter_05/`: compressed `NPZ` arrays plus a `JSON` manifest with script provenance, figure path, array shape/dtype contracts, and summary statistics.

## Expected files

| File | Producing script |
|---|---|
| `example_5_1_mle_free_energy.png` / `example_5_1_prediction_errors.png` | `example_5_1_prediction_errors.py` |
| `example_5_3_multivariate.png` | `example_5_3_multivariate.py` |
| `example_5_4_recognition_nonlinear.png` (default) / `example_5_4_recognition_linear.png` (`--linear`) | `example_5_4_recognition_dynamics.py` |
| `example_5_7_hierarchical.png` | `example_5_7_hierarchical.py` |
| `animation_recognition_linear.gif` (default) / `animation_recognition_nonlinear.gif` (`--nonlinear`) | `animation_recognition_descent.py` |
| `animation_hierarchical.gif` | `animation_hierarchical.py` |

> `run_all_figures.py` runs `example_5_4` without `--linear`, producing the
> *nonlinear* PNG; pass `--linear` manually for the cross-chapter oracle figure.
