"""Example 9.3 — discrete variational free energy on a simplex (Chapter 9 §9.3).

Run::

    python chapters/chapter_09/example_9_3_discrete_vfe.py [--save] [--seed 0]

For a two-state POMDP the variational belief is a single number, ``q = Q(s=0)``.
Sweeping ``q`` across the simplex shows the categorical variational free energy

``F(q) = q · (log q - log prior - log Aᵀo)``

is minimized exactly at the Bayes posterior from §9.1. This is the discrete analogue of the
Chapter 4 free-energy bound: the minimum touches surprisal, ``-log P(o)``.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import POMDPModel, discrete_vfe, get_logger, infer_states
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import finalize, panel_grid, save_or_show
from active_inference.visualizations.style import COLORS, annotate_point, annotate_stat_box

LOG = get_logger("ch9.ex3")

A = np.array([[0.85, 0.20],
              [0.15, 0.80]])
D = np.array([0.55, 0.45])
OBS_INDEX = 0
STATE_LABELS = ["state 0", "state 1"]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="Accepted for CLI consistency.")
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    _ = args.seed
    model = POMDPModel(A=A, D=D)
    posterior = infer_states(model, OBS_INDEX)
    likelihood = A.T @ np.array([1.0, 0.0])
    evidence = float(likelihood @ D)
    surprisal = -np.log(evidence)

    q0 = np.linspace(1e-4, 1.0 - 1e-4, 800)
    beliefs = np.column_stack([q0, 1.0 - q0])
    F = np.array([discrete_vfe(model, q, OBS_INDEX, D) for q in beliefs])
    q_min = beliefs[int(np.argmin(F))]

    LOG.info("posterior=%s | grid minimum=%s | surprisal=%.4f",
             np.round(posterior, 4).tolist(), np.round(q_min, 4).tolist(), surprisal)

    fig, axes = panel_grid(2, title="Fig. 9.3 · discrete VFE is minimized by the posterior",
                           figsize=(12, 4.8))

    ax = axes[0]
    ax.plot(q0, F, color=COLORS["data"], lw=2.6, label=r"$\mathcal{F}(q)$")
    ax.axhline(surprisal, color=COLORS["likelihood"], ls="--", lw=1.8,
               label=r"$-\log P(o)$")
    ax.axvline(posterior[0], color=COLORS["posterior"], ls=":", lw=2.0,
               label="exact posterior")
    annotate_point(ax, q_min[0], F.min(), f"min q={q_min[0]:.3f}",
                   color=COLORS["posterior"], dx=0.08, dy=0.20)
    annotate_stat_box(ax, f"posterior q0 = {posterior[0]:.4f}\n"
                          f"min F       = {F.min():.4f}\n"
                          f"surprisal   = {surprisal:.4f}",
                      loc="upper center")
    finalize(ax, xlabel=r"$q = Q(s=0)$", ylabel=r"$\mathcal{F}$",
             title="VFE over the two-state simplex")

    ax = axes[1]
    x = np.arange(2)
    width = 0.26
    ax.bar(x - width, D, width, color=COLORS["prior"], label="prior D")
    ax.bar(x, likelihood / likelihood.sum(), width, color=COLORS["likelihood"],
           label="likelihood row")
    ax.bar(x + width, posterior, width, color=COLORS["posterior"], label="posterior")
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_LABELS)
    ax.set_ylim(0, 1.0)
    finalize(ax, xlabel="state", ylabel="probability",
             title="prior × likelihood → posterior")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_09")
        save_or_show(fig, out / "example_9_3_discrete_vfe.png")
        LOG.info("Saved to %s", out / "example_9_3_discrete_vfe.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
