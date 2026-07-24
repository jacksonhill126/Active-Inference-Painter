"""Animate Chapter 8 learning and attention convergence."""

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
from active_inference.visualizations import animate_learning_attention, save_animation

LOG = get_logger("ch8.animation")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--frames", type=int, default=160)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    truth_x = np.full(args.frames, 5.0)
    theta_true = 3.0
    ys = truth_x - theta_true + rng.normal(0.0, 0.08, size=args.frames)
    model = LearningAttentionModel(5.0, sigma2_y=0.02, sigma2_theta=50.0, sigma2_zeta=50.0)
    res = simulate_learning_attention(
        model,
        ys,
        initial=LearningAttentionState(1.5, 0.0, 0.0),
        dt=0.03,
        kappa_x=0.7,
        kappa_theta=2.0,
        kappa_zeta=0.25,
        damping=1.2,
    )
    anim = animate_learning_attention(
        res.mus,
        res.mu_thetas,
        res.mu_zetas,
        res.free_energies,
        truth_x=truth_x,
        truth_theta=theta_true,
        title="Fig. 8.1 animation · learning and attention",
    )
    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_08")
        save_animation(anim, out / "animation_learning_attention.gif", fps=12, dpi=110)
        LOG.info("Saved to %s", out / "animation_learning_attention.gif")
    else:
        plt.show()


if __name__ == "__main__":
    main()
