# `output/figures/chapter_06/` — Chapter 6 figures

Five PNG outputs from the Chapter 6 orchestrators (`chapters/chapter_06/`).
**Ephemeral and gitignored** — regenerate with:

```bash
python scripts/run_all_figures.py --chapters 6
```

Each saved artifact has a matching raw-data sidecar under `output/data/chapter_06/`: compressed `NPZ` arrays plus a `JSON` manifest with script provenance, figure path, array shape/dtype contracts, and summary statistics.

## Expected files

| File | Producing script |
|---|---|
| `example_6_1_generalized_filter.png` | `example_6_1_generalized_filter.py` |
| `example_6_2_multivariate_filter.png` | `example_6_2_multivariate_filter.py` |
| `example_6_6_generalized_coordinates.png` | `example_6_6_generalized_coordinates.py` |
| `visualize_6_6_correlated_embedding_orders.png` | `visualize_6_6_correlated_embedding_orders.py` |
| `example_6_7_multivariate_generalized_coordinates.png` | `example_6_7_multivariate_generalized_coordinates.py` |
