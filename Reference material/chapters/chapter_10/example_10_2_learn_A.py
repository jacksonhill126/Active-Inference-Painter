"""Example 10.2 — learning the likelihood matrix ``A`` (Chapter 10, §10.1).

Run::

    python chapters/chapter_10/example_10_2_learn_A.py [--save]

Reproduces Figure 10.1.3. An agent with **perfect state knowledge** runs 5 trials of
``T = 1000`` steps, each step counting the (observation, state) pair it sees into the
Dirichlet pseudocounts ``a`` (Eq. 4a, ``a ← a + Σ o ∘ s``). The expected likelihood
``A = E[Dir(a)]`` (Eq. 5) converges on the true ``A``, while the concentration parameters
grow linearly — confidence that makes the estimate progressively harder to move.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import get_logger, simulate_array_learning
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show
from active_inference.visualizations.unified import plot_parameter_learning

LOG = get_logger("ch10.ex2")
# Book Example 10.2 generative process (Eq. 9).
A_TRUE = np.array([[0.7, 0.6], [0.3, 0.4]])
B_TRUE = np.array([[0.0, 1.0], [1.0, 0.0]])


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--steps", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    res = simulate_array_learning(A_true=A_TRUE, B_true=B_TRUE, learn="A",
                                  n_trials=args.trials, steps_per_trial=args.steps)
    LOG.info("learned A=%s  true=%s  max|A−A*|=%.4f",
             np.round(res.A_history[-1], 3).tolist(), A_TRUE.tolist(), res.final_A_error())

    fig = plot_parameter_learning(
        res.A_history, res.a_confidence, truth=A_TRUE, symbol="A",
        title="Fig. 10.1.3 · learning the likelihood A (Example 10.2)",
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_10")
        save_or_show(fig, out / "example_10_2_learn_A.png")
        LOG.info("Saved to %s", out / "example_10_2_learn_A.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
