"""Example 8.2 — hierarchical continuous model with top-down context.

Run::

    python chapters/chapter_08/example_8_2_hierarchical_continuous.py [--save]

A lower hidden state explains sensory data while an upper context predicts that
lower state. Gradient descent on one VFE objective aligns bottom-up evidence with
top-down contextual prediction.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    ContinuousHierarchyLayer,
    HierarchicalContinuousModel,
    get_logger,
    hierarchical_continuous_free_energy,
    hierarchical_continuous_grad,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import COLORS, finalize, panel_grid, save_or_show

LOG = get_logger("ch8.ex2")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--n-steps", type=int, default=220)
    p.add_argument("--dt", type=float, default=0.02)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    model = HierarchicalContinuousModel(
        lower=ContinuousHierarchyLayer(obs_offset=3.0, sensory_precision=20.0),
        link_precision=2.0,
        context_prior_mean=5.0,
        context_precision=0.5,
    )
    y = 2.0
    belief = np.array([1.0, 0.0])
    path = np.empty((args.n_steps, 2))
    fes = np.empty(args.n_steps)
    for t in range(args.n_steps):
        path[t] = belief
        fes[t] = hierarchical_continuous_free_energy(model, y, belief)
        belief = belief - args.dt * 0.7 * hierarchical_continuous_grad(model, y, belief)

    LOG.info("final lower=%.3f upper=%.3f F=%.3f", path[-1, 0], path[-1, 1], fes[-1])
    fig, axes = panel_grid(2, title="Fig. 8.2 · continuous hierarchy with top-down context",
                           figsize=(12.4, 4.8))
    it = np.arange(args.n_steps) * args.dt

    ax = axes[0]
    ax.plot(it, path[:, 0], color=COLORS["posterior"], label=r"lower $\mu_x$")
    ax.plot(it, path[:, 1], color=COLORS["prior"], label=r"upper $\mu_v$")
    ax.axhline(5.0, color=COLORS["truth"], ls="--", label="context prior")
    finalize(ax, xlabel="time", ylabel="belief mean", title="top-down context pulls lower state")

    ax = axes[1]
    ax.plot(it, fes, color=COLORS["data"], label=r"$\mathcal{F}$")
    finalize(ax, xlabel="time", ylabel="nats", title="hierarchical free energy")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_08")
        save_or_show(fig, out / "example_8_2_hierarchical_continuous.png")
        LOG.info("Saved to %s", out / "example_8_2_hierarchical_continuous.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
