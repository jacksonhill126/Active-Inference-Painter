"""Example 10.1 — learning the state-prior vector ``D`` (Chapter 10, §10.1).

Run::

    python chapters/chapter_10/example_10_1_learn_D.py [--save]

The simplest case of POMDP **learning**: the agent starts with uniform Dirichlet
concentration parameters ``d = [1, 1]`` (it believes both initial states equally likely)
and, on each trial, observes an initial-state belief ``s^{(0)} = [0.9, 0.1]``. The update
rule ``d ← d + s^{(0)}`` (Eq. 4c) accumulates pseudocounts, and the expected value
``D = d / Σd`` (Eq. 5) converges on the true proportion ``[0.9, 0.1]``.

Reproduces the book's worked numbers: after 49 trials ``d = [45.1, 5.9]`` and
``D ≈ [0.884, 0.116]`` (Eq. 7/8), shown as the converging curve.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import get_logger, learn_D_vector
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import finalize, panel_grid, save_or_show
from active_inference.visualizations.style import COLORS, annotate_stat_box

LOG = get_logger("ch10.ex1")
TRUE_S0 = np.array([0.9, 0.1])
N_TRIALS = 49


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    res = learn_D_vector([1.0, 1.0], np.tile(TRUE_S0, (N_TRIALS, 1)))
    LOG.info("d_final=%s  D_final=%s (book d=[45.1,5.9], D=[0.884,0.116])",
             np.round(res.d_final, 2).tolist(), np.round(res.D_history[-1], 3).tolist())

    trials = np.arange(res.D_history.shape[0])
    fig, axes = panel_grid(2, title="Fig. 10.1 · learning the state prior D (Example 10.1)",
                           figsize=(12.5, 4.8))

    ax = axes[0]
    ax.plot(trials, res.D_history[:, 0], color=COLORS["posterior"], lw=2.6, label=r"$D_0$")
    ax.plot(trials, res.D_history[:, 1], color=COLORS["likelihood"], lw=2.6, label=r"$D_1$")
    ax.axhline(TRUE_S0[0], color=COLORS["truth"], ls="--", lw=1.8, label=r"true $s^{(0)}_0=0.9$")
    ax.axhline(TRUE_S0[1], color=COLORS["neutral"], ls="--", lw=1.4, label=r"true $s^{(0)}_1=0.1$")
    annotate_stat_box(ax, f"D* = [{res.D_history[-1][0]:.3f}, {res.D_history[-1][1]:.3f}]\n"
                          "book [0.884, 0.116]", loc="center right")
    finalize(ax, xlabel="trial", ylabel="probability",
             title="D converges on the true initial-state proportion")

    ax = axes[1]
    d_hist = np.array([[1.0, 1.0]] + [[1.0, 1.0] + (k + 1) * TRUE_S0 for k in range(N_TRIALS)])
    ax.plot(trials, d_hist[:, 0], color=COLORS["posterior"], lw=2.6, label=r"$d_0$")
    ax.plot(trials, d_hist[:, 1], color=COLORS["likelihood"], lw=2.6, label=r"$d_1$")
    annotate_stat_box(ax, f"d* = [{res.d_final[0]:.1f}, {res.d_final[1]:.1f}]\n"
                          "Σd = 51 (book)", loc="upper left")
    finalize(ax, xlabel="trial", ylabel="concentration (pseudocount)",
             title="Dirichlet pseudocounts accumulate (confidence)")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_10")
        save_or_show(fig, out / "example_10_1_learn_D.png")
        LOG.info("Saved to %s", out / "example_10_1_learn_D.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
