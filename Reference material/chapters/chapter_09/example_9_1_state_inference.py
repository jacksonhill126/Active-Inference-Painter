"""Example 9.1 — discrete hidden-state inference in a POMDP (Chapter 9 §9.1).

Run::

    python chapters/chapter_09/example_9_1_state_inference.py [--save]

The discrete (categorical) counterpart of Chapters 1–3 Bayesian inference. A weather
agent has a likelihood matrix ``A`` mapping four hidden states (rainy, cloudy, sunny,
snowy) to three observations (water, hot, bright) and a uniform prior ``D``. Given a
one-hot observation it inverts the generative model exactly,
``s = σ(log Aᵀô + log D)`` (Eq. 13). The two panels show the posterior over states after
observing **water** and after observing **hot** — reproducing Figure 9.1.3.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import POMDPModel, get_logger, infer_states
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import finalize, panel_grid, save_or_show
from active_inference.visualizations.style import COLORS, annotate_stat_box

LOG = get_logger("ch9.ex1")

STATES = ["rainy", "cloudy", "sunny", "snowy"]
OBS = ["water", "hot", "bright"]
A = np.array([[0.80, 0.33, 0.05, 0.40],
              [0.15, 0.33, 0.30, 0.05],
              [0.05, 0.34, 0.65, 0.55]])
D = np.array([0.25, 0.25, 0.25, 0.25])


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    model = POMDPModel(A=A, D=D)

    post_water = infer_states(model, OBS.index("water"))
    post_hot = infer_states(model, OBS.index("hot"))
    LOG.info("observe water -> %s", dict(zip(STATES, np.round(post_water, 3))))
    LOG.info("observe hot   -> %s (book Eq.15: [0.18,0.40,0.36,0.06])",
             dict(zip(STATES, np.round(post_hot, 3))))

    fig, axes = panel_grid(2, title="Fig. 9.1.3 · discrete hidden-state inference (POMDP)",
                           figsize=(11, 4.4))
    for ax, post, obs, color in ((axes[0], post_water, "water", COLORS["prior"]),
                                 (axes[1], post_hot, "hot", COLORS["likelihood"])):
        ax.bar(STATES, post, color=color, alpha=0.85)
        ax.set_ylim(0, 0.7)
        annotate_stat_box(ax, "  ".join(f"{s[:2]}={p:.2f}" for s, p in zip(STATES, post)),
                          loc="upper right", fontsize=10)
        finalize(ax, xlabel="state", ylabel="P(state | obs)",
                 title=f"observed: {obs}", legend=False)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_09")
        save_or_show(fig, out / "example_9_1_state_inference.png")
        LOG.info("Saved to %s", out / "example_9_1_state_inference.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
