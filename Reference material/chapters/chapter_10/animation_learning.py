"""Animation — Dirichlet parameter learning over trials (Chapter 10 §10.1).

Run::

    python chapters/chapter_10/animation_learning.py [--save] [--transition]

Animates Example 10.2 (learning the likelihood ``A``) or, with ``--transition``, Example 10.3
(learning the transition ``B``). Two panels grow trial-by-trial: the learned matrix entries
converging on their true values (dots), and the Dirichlet concentration parameters
(confidence) accumulating. Built on the composable
:func:`~active_inference.visualizations.animate_parameter_learning`.
"""

from __future__ import annotations

import argparse

from active_inference import get_logger, simulate_array_learning
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import animate_parameter_learning, save_animation
import numpy as np

LOG = get_logger("ch10.anim_learn")
A_TRUE = np.array([[0.7, 0.6], [0.3, 0.4]])
B_TRUE_A = np.array([[0.0, 1.0], [1.0, 0.0]])    # for A-learning generative process
B_TRUE_B = np.array([[0.2, 0.6], [0.8, 0.4]])    # the B we learn in --transition mode


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--transition", action="store_true", help="learn B instead of A")
    p.add_argument("--trials", type=int, default=8)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    if args.transition:
        res = simulate_array_learning(A_true=A_TRUE, B_true=B_TRUE_B, learn="B",
                                      n_trials=args.trials, steps_per_trial=1500)
        anim = animate_parameter_learning(
            res.B_history, res.b_confidence, truth=B_TRUE_B, symbol="B",
            title="Ch.10 · learning the transition B over trials (Example 10.3)")
        name = "animation_learning_B.gif"
    else:
        res = simulate_array_learning(A_true=A_TRUE, B_true=B_TRUE_A, learn="A",
                                      n_trials=args.trials, steps_per_trial=800)
        anim = animate_parameter_learning(
            res.A_history, res.a_confidence, truth=A_TRUE, symbol="A",
            title="Ch.10 · learning the likelihood A over trials (Example 10.2)")
        name = "animation_learning_A.gif"

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_10")
        save_animation(anim, out / name)
        LOG.info("Saved to %s", out / name)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
