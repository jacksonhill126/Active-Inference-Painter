# `output/figures/chapter_08/` — Chapter 8 figures

PNG and GIF outputs from the Chapter 8 orchestrators (`chapters/chapter_08/`).

```bash
python scripts/run_all_figures.py --chapters 8
```

Files here are derived artifacts and can be recreated from the chapter scripts.

Each saved artifact has a matching raw-data sidecar under `output/data/chapter_08/`: compressed `NPZ` arrays plus a `JSON` manifest with script provenance, figure path, array shape/dtype contracts, and summary statistics.

## Expected files

| File | Producing script |
|---|---|
| `example_8_1_learning_attention.png` | `example_8_1_learning_attention.py` |
| `example_8_2_hierarchical_continuous.png` | `example_8_2_hierarchical_continuous.py` |
| `visualize_message_passing.png` | `visualize_message_passing.py` |
| `animation_learning_attention.gif` | `animation_learning_attention.py` |
