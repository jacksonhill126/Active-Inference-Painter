# `output/figures/chapter_09/` — Chapter 9 figures

PNG and GIF outputs from the Chapter 9 orchestrators (`chapters/chapter_09/`).
**Ephemeral and gitignored** — regenerate with:

```bash
python scripts/run_all_figures.py --chapters 9
```

Each saved artifact has a matching raw-data sidecar under `output/data/chapter_09/`: compressed `NPZ` arrays plus a `JSON` manifest with script provenance, figure path, array shape/dtype contracts, and summary statistics.

## Expected files

| File | Producing script |
|---|---|
| `example_9_1_state_inference.png` | `example_9_1_state_inference.py` |
| `example_9_2_dynamic_filtering.png` | `example_9_2_dynamic_filtering.py` |
| `example_9_3_discrete_vfe.png` | `example_9_3_discrete_vfe.py` |
| `example_9_4_gridworld.png` | `example_9_4_gridworld.py` |
| `example_9_6_exploration_exploitation.png` | `example_9_6_exploration_exploitation.py` |
| `animation_belief_filtering.gif` | `animation_belief_filtering.py` |
| `animation_efe_tradeoff.gif` | `animation_efe_tradeoff.py` |
