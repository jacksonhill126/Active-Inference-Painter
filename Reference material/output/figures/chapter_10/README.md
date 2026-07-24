# `output/figures/chapter_10/` — Chapter 10 figures

PNG outputs from the Chapter 10 orchestrators (`chapters/chapter_10/`).
**Ephemeral and gitignored** — regenerate with:

```bash
python scripts/run_all_figures.py --chapters 10
```

Each saved artifact has a matching raw-data sidecar under `output/data/chapter_10/`: compressed `NPZ` arrays plus a `JSON` manifest with script provenance, figure path, array shape/dtype contracts, and summary statistics.

## Expected files

| File | Producing script |
|---|---|
| `example_10_1_learn_D.png` | `example_10_1_learn_D.py` |
| `example_10_2_learn_A.png` | `example_10_2_learn_A.py` |
| `example_10_3_learn_B.png` | `example_10_3_learn_B.py` |
| `example_10_4_novelty.png` | `example_10_4_novelty.py` |
| `example_10_5_precision.png` | `example_10_5_precision.py` |
| `example_10_6_precision_learning.png` | `example_10_6_precision_learning.py` |
| `example_10_7_bandit.png` / `example_10_7_bandit_explore.png` | `example_10_7_two_armed_bandit.py` `[--explore]` |
| `example_10_8_hierarchical.png` | `example_10_8_hierarchical.py` |
| `factorial_likelihood_structure.png` | `visualize_factorial_structure.py` |
| `animation_learning_A.gif` / `animation_learning_B.gif` | `animation_learning.py` `[--transition]` |
| `animation_precision.gif` / `animation_precision_strong.gif` | `animation_precision.py` `[--strong]` |
| `animation_bandit.gif` / `animation_bandit_explore.gif` | `animation_bandit.py` `[--explore]` |
