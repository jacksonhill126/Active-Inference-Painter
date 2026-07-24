# chapters/chapter_01/ — The Hypothesis-Testing Brain

Chapter 1 scripts reproduce the core thought experiments from the book's
opening chapter. Each one is a thin orchestrator that imports from
`active_inference` and standard library.

## Scripts

| Script | Lines | What it shows |
|---|---|---|
| [`01_box_scenario.py`](01_box_scenario.py) | ~110 | The "agent in a box" thought experiment as a stream of noisy sensor readings |
| [`02_three_perspectives.py`](02_three_perspectives.py) | ~110 | Side-by-side simulation of the *scientific*, *hypothesis-testing*, and *statistical* views |
| [`03_bayes_intuition.py`](03_bayes_intuition.py) | ~110 | Bayes' theorem step-by-step on a single-state, single-observation toy |
| [`04_inverse_problem.py`](04_inverse_problem.py) | ~110 | Non-injective generator → bi-modal posterior |
| [`05_belief_from_stream.py`](05_belief_from_stream.py) | ~110 | **Animation** (GIF): sensor stream arriving beside the belief sharpening toward `x*`, with the `σ·√N` concentration statistic cross-checked against the MLE |
| [`interactive_inverse_problem.py`](interactive_inverse_problem.py) | ~30 | **Interactive** (GUI / web-launchable): `y` / `σ_y²` sliders drive the bi-modal posterior of `04_inverse_problem.py`'s non-injective generator |

## Running

```bash
# Run a single script (headless — saves to output/figures/chapter_01/)
python chapters/chapter_01/03_bayes_intuition.py --save

# Run all Chapter 1 scripts at once
python scripts/run_all_figures.py --chapters 1
```

Each script accepts `--save` for headless rendering; stochastic scripts also
accept `--seed` for reproducibility. With no flags, an interactive matplotlib
window opens (`interactive_inverse_problem.py` always opens a GUI window —
it has no `--save` path).

## Library Usage

The numbered scripts import from the top-level `active_inference` package:

```python
from active_inference import (
    GridBayesianInference,
    gaussian_pdf,
    gaussian_log_pdf,
    make_grid,
    get_logger,
)
```

`05_belief_from_stream.py` additionally pulls `LinearGaussianModel`,
`LinearGaussianProcess`, `mle_analytic_linear`, `oracle_agreement`, and the
`animate_stream_belief` / `save_animation` visualization helpers.
`interactive_inverse_problem.py` is a thin wrapper around
`active_inference.visualizations.interactive_inverse_problem`.

## Smoke Tests

`tests/chapters/test_smoke.py` runs each script with `--save` via
subprocess and asserts exit code 0 (the single parametrized test
`test_chapter_script_runs_and_exports_raw_data`, run over every discovered
chapter script). Discovery is glob-driven, so any new
`0N_*.py` file in this folder is picked up automatically — there is no
hand-maintained list of scripts. `interactive_inverse_problem.py` has no
`--save` path and is exercised separately by
`tests/visualizations/test_interactive.py`.

## Key Concepts

- **Scientific view:** The "agent in a box" sees measurements and reasons
  about hidden generative causes.
- **Hypothesis-testing view:** Each observation is a datum; the agent updates
  belief distributions over candidate hypotheses.
- **Statistical view:** Bayes' theorem in its elementary form — prior ×
  likelihood ∝ posterior.
- **Inverse problem:** When the generative function is non-injective, the
  posterior can be genuinely multi-modal even from a unimodal prior.