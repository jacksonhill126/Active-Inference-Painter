"""Example 8.1 — learning first- and second-order parameters.

Run::

    python chapters/chapter_08/example_8_1_learning_attention.py [--save]

The agent observes ``y = x* - theta* + noise`` while learning a hidden-state belief,
a first-order observation parameter ``theta``, and a second-order log precision
``zeta``. This is the Chapter 8 triple-estimation problem in a compact linear model.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    LearningAttentionModel,
    LearningAttentionState,
    get_logger,
    simulate_learning_attention,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_learning_attention, save_or_show

LOG = get_logger("ch8.ex1")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=420)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    t = np.linspace(0.0, 5.0, args.n_steps)
    truth_x = 5.0 + 0.12 * np.sin(t)
    theta_true = 3.0
    ys = truth_x - theta_true + rng.normal(0.0, 0.08, size=args.n_steps)

    model = LearningAttentionModel(
        state_attractor=5.0,
        theta_prior_mean=0.0,
        zeta_prior_mean=0.0,
        sigma2_y=0.02,
        sigma2_theta=50.0,
        sigma2_zeta=50.0,
    )
    res = simulate_learning_attention(
        model,
        ys,
        initial=LearningAttentionState(mu_x=1.5, mu_theta=0.0, mu_zeta=0.0),
        dt=0.03,
        kappa_x=0.7,
        kappa_theta=2.0,
        kappa_zeta=0.25,
        damping=1.2,
    )
    LOG.info("final mu_x=%.3f theta=%.3f variance=%.4f tracking=%.3f",
             res.final_state.mu_x, res.final_state.mu_theta, res.final_variance_x,
             res.tracking_error(truth_x, burn_in=args.n_steps // 2))

    fig = plot_learning_attention(res, truth_x=truth_x, truth_theta=theta_true)
    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_08")
        save_or_show(fig, out / "example_8_1_learning_attention.png")
        LOG.info("Saved to %s", out / "example_8_1_learning_attention.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
