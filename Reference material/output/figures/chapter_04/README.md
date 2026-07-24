# `output/figures/chapter_04/` — Chapter 4 figures

PNG and GIF outputs from the Chapter 4 orchestrators
(`chapters/chapter_04/`). **Ephemeral and gitignored** — regenerate with:

```bash
python scripts/run_all_figures.py --chapters 4
```

Each saved artifact has a matching raw-data sidecar under `output/data/chapter_04/`: compressed `NPZ` arrays plus a `JSON` manifest with script provenance, figure path, array shape/dtype contracts, and summary statistics.

## Expected files

| File | Producing script |
|---|---|
| `example_4_1_vfe_contour.png` / `example_4_1_density_evolution.png` | `example_4_1_coordinate_search.py` |
| `example_4_2_surprisal.png` | `example_4_2_surprisal.py` |
| `example_4_3_vfe_forms.png` | `example_4_3_vfe_forms.py` |
| `example_4_6_cavi.png` | `example_4_6_free_form_cavi.py` |
| `example_4_7_vfe_contour.png` / `example_4_7_density_evolution.png` / `example_4_7_decomposition.png` | `example_4_7_fixed_form.py` |
| `visualize_kl_loss.png` | `visualize_kl_loss.py` |
| `visualize_vfe_intuition.png` | `visualize_vfe_intuition.py` |
| `visualize_model_comparison.png` | `visualize_model_comparison.py` |
| `animation_vfe_descent.gif` | `animation_vfe_descent.py` |
