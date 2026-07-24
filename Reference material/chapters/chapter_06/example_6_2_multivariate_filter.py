"""Example 6.2 — the multivariate generalized filter (Chapter 6 §6.2).

Run::

    python chapters/chapter_06/example_6_2_multivariate_filter.py [--save]

A 2-D hidden state obeys **Hooke's law** (a mass on a spring): `ẋ₁ = x₂`,
`ẋ₂ = (k/m)(v₀ − x₁)`, an oscillator orbiting the stable fixed point `(v₀, 0) = (5, 0)`.
The agent observes `y = [x₁ − θ_y, x₂ − θ_y]` with noise and filters it online with the
multivariate generalized filter (Eq. 14/15). Both elements of the belief `μ_x` track the
oscillating true state — with the slight *perception lag* the book notes at the turning
points (which generalized coordinates, §6.3, are introduced to reduce). Reproduces
Figures 6.2.2 / 6.2.3.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    LinearVectorFunction,
    MultivariateDynamicModel,
    MultivariateDynamicProcess,
    get_logger,
    multivariate_generalized_filter,
    simulate_multivariate_process,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import finalize, panel_grid, save_or_show
from active_inference.visualizations.style import COLORS, annotate_stat_box

LOG = get_logger("ch6.ex2")

K, M, V0, THETA_Y = 4.0, 3.0, 5.0, 3.0


def hooke_functions():
    """Compute a chapter-local helper quantity for the orchestrated example."""
    A_f = np.array([[0.0, 1.0], [-K / M, 0.0]])
    b_f = np.array([0.0, (K / M) * V0])           # ẋ₂ = (k/m)(v₀ − x₁)
    A_g = np.eye(2)
    b_g = np.array([-THETA_Y, -THETA_Y])
    return LinearVectorFunction(A_f, b_f), LinearVectorFunction(A_g, b_g)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.01)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    f, g = hooke_functions()

    process = MultivariateDynamicProcess(f=f, g=g, omega_x=0.1, omega_y=0.1)
    tr = simulate_multivariate_process(process, x0=np.array([0.0, 5.0]),
                                       n_steps=args.n_steps, dt=args.dt, rng=rng)

    model = MultivariateDynamicModel(f=f, g=g, precision_x=0.5, precision_y=10.0,
                                     dim_x=2, dim_y=2)   # Σ_x=2I, Σ_y=0.1I
    res = multivariate_generalized_filter(model, tr.ys, dt=args.dt, kappa=1.0,
                                          mu0=np.array([8.0, 8.0]))

    burn = args.n_steps // 3
    te = res.tracking_error(tr.xs, burn_in=burn)
    unfiltered = float(np.mean(np.linalg.norm(np.array([8.0, 8.0]) - tr.xs[burn:], axis=1)))
    LOG.info("Hooke oscillator tracking ||mu - x*|| = %.4f (unfiltered prior = %.4f)",
             te, unfiltered)

    t = np.arange(res.mus.shape[0]) * args.dt
    fig, axes = panel_grid(3, title="Fig. 6.2.3 · multivariate generalized filter (Hooke's law)",
                           figsize=(15.5, 4.8))
    elem_colors = (COLORS["prior"], COLORS["state"])
    ax = axes[0]
    for c in range(2):
        ax.plot(t, tr.xs[:, c], color=COLORS["truth"], lw=1.2, alpha=0.5)
        ax.plot(t, res.mus[:, c], color=elem_colors[c], lw=2.2, label=rf"$\mu_x^{{[{c}]}}$")
    annotate_stat_box(ax, f"track ‖μ−x*‖ = {te:.3f}\nμ₀ = [8, 8]", loc="lower right")
    finalize(ax, xlabel="time", ylabel=r"$x^*$ / $\mu_x$", title="state tracking (grey = true)")

    ax = axes[1]
    for c in range(2):
        ax.plot(t, tr.ys[:, c], color=COLORS["neutral"], lw=1.0, alpha=0.5)
        ax.plot(t, res.mu_ys[:, c], color=elem_colors[c], lw=2.2, label=rf"$\mu_y^{{[{c}]}}$")
    finalize(ax, xlabel="time", ylabel=r"$y$ / $\mu_y$", title="predicted vs received obs")

    ax = axes[2]
    ax.plot(t, res.free_energies, color=COLORS["data"], lw=2.2, label=r"$\mathcal{F}$")
    finalize(ax, xlabel="time", ylabel=r"$\mathcal{F}$", title="free energy")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_06")
        save_or_show(fig, out / "example_6_2_multivariate_filter.png")
        LOG.info("Saved to %s", out / "example_6_2_multivariate_filter.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
