"""Example 10.5 — precision- and habit-based policy optimization (Chapter 10, §10.2).

Run::

    python chapters/chapter_10/example_10_5_precision.py [--save]

Reproduces Figures 10.2.2/10.2.3. The variational policy posterior gains two knobs beyond
the Chapter 9 `σ(−γG)`: a **baseline prior / habit** `E` (a learned bias toward policies)
and a **policy precision** `γ` (confidence in the expected free energy `G`). Eq. 22:

    Q(π) = σ(log E − F − γ G)

Sweeping `γ` from 0 (uniform / pure-habit) upward concentrates probability on the
lowest-EFE policy (here policy 2, `G = 1.5`). With a non-uniform habit `E` the prior shifts
mass toward the preferred policies, and the posterior blends habit with EFE.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import get_logger, precision_policy_sweep
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import finalize, panel_grid, save_or_show
from active_inference.visualizations.style import annotate_stat_box
from active_inference.visualizations.unified import layer_colors

LOG = get_logger("ch10.ex5")
# Book Example 10.5 (Eq. 27/28).
G = np.array([3.0, 2.0, 1.5, 2.2, 3.2])
E_UNIFORM = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
E_STRONG = np.array([0.2, 0.4, 0.1, 0.4, 0.2])


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--gamma-max", type=float, default=3.0)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    gammas = np.linspace(0.0, args.gamma_max, 80)
    sweep_uniform = precision_policy_sweep(G, gammas, E=E_UNIFORM)
    sweep_strong = precision_policy_sweep(G, gammas, E=E_STRONG)
    LOG.info("uniform habits @γ=1.5: %s (argmax policy %d)",
             np.round(precision_policy_sweep(G, [1.5], E=E_UNIFORM)[0], 3).tolist(),
             int(np.argmax(precision_policy_sweep(G, [1.5], E=E_UNIFORM)[0])))

    fig, axes = panel_grid(2, title="Fig. 10.2.2 · policy posterior vs precision γ (Example 10.5)",
                           figsize=(12.5, 4.8))
    colors = layer_colors(len(G))

    for ax, sweep, name in ((axes[0], sweep_uniform, "uniform habits"),
                            (axes[1], sweep_strong, "strong habits")):
        for k in range(len(G)):
            ax.plot(gammas, sweep[:, k], color=colors[k], lw=2.4, label=rf"$\pi^{{({k})}}$")
        ax.axvline(1.0, color="0.4", ls="--", lw=1.2)
        annotate_stat_box(ax, f"lowest EFE: policy 2\n(G = {G.min()})", loc="upper left", fontsize=10)
        finalize(ax, xlabel=r"$\gamma$ (confidence)", ylabel="policy probability",
                 title=f"{name}: γ concentrates mass on min-EFE policy")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_10")
        save_or_show(fig, out / "example_10_5_precision.png")
        LOG.info("Saved to %s", out / "example_10_5_precision.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
