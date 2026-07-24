# `output/figures/chapter_07/` — Chapter 7 figures

Two PNG outputs and one GIF from the Chapter 7 orchestrators (`chapters/chapter_07/`).
**Ephemeral and gitignored** — regenerate with:

```bash
python scripts/run_all_figures.py --chapters 7
```

Each saved artifact has a matching raw-data sidecar under `output/data/chapter_07/`: compressed `NPZ` arrays plus a `JSON` manifest with script provenance, figure path, array shape/dtype contracts, and summary statistics.

## Expected files

| File | Producing script |
|---|---|
| `example_7_2_active_inference.png` | `example_7_2_active_inference.py` |
| `example_7_5_multivariate_active_inference.png` | `example_7_5_multivariate_active_inference.py` |
| `animation_7_5_multivariate_active_inference.gif` | `animation_7_5_multivariate_active_inference.py` |
