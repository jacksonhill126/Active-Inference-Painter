"""Example 10.3 — learning the transition matrix ``B`` (Chapter 10, §10.1).

Run::

    python chapters/chapter_10/example_10_3_learn_B.py [--save]

Reproduces Figure 10.1.4. The procedure mirrors learning ``A`` (Example 10.2) but now the
agent counts (next-state, current-state) **transition** pairs into the Dirichlet pseudocounts
``b`` (Eq. 4b, ``b ← b + Σ s^{(τ)} ∘ s^{(τ-1)}``). The expected transition
``B = E[Dir(b)]`` (Eq. 5) converges on the true ``B`` over trials.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import get_logger, simulate_array_learning
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show
from active_inference.visualizations.unified import plot_parameter_learning

LOG = get_logger("ch10.ex3")
# Book Example 10.3 generative process (Eq. 10).
A_TRUE = np.array([[0.7, 0.6], [0.3, 0.4]])
B_TRUE = np.array([[0.2, 0.6], [0.8, 0.4]])


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--steps", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    res = simulate_array_learning(A_true=A_TRUE, B_true=B_TRUE, learn="B",
                                  n_trials=args.trials, steps_per_trial=args.steps)
    err = float(np.max(np.abs(res.B_history[-1] - B_TRUE)))
    LOG.info("learned B=%s  true=%s  max|B−B*|=%.4f",
             np.round(res.B_history[-1], 3).tolist(), B_TRUE.tolist(), err)

    fig = plot_parameter_learning(
        res.B_history, res.b_confidence, truth=B_TRUE, symbol="B",
        title="Fig. 10.1.4 · learning the transition B (Example 10.3)",
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_10")
        save_or_show(fig, out / "example_10_3_learn_B.png")
        LOG.info("Saved to %s", out / "example_10_3_learn_B.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
