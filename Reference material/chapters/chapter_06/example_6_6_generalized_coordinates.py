"""Example 6.6 — univariate generalized filtering in generalized coordinates (§6.5).

Run::

    python chapters/chapter_06/example_6_6_generalized_coordinates.py [--save]

This repeats Example 6.1 (an object drifting to a point attractor at ``θ_x=10``) but
in **generalized coordinates of motion**: the belief is now a trajectory
``μ̃_x = [μ_x, μ_x', μ_x'', …]`` carrying the inferred position *and its velocity and
acceleration*. The recognition flow ``μ̃̇_x = D μ̃_x − κ ∂F/∂μ̃_x`` (Eq. 51) locks the
belief into a moving reference frame. The payoff over §6.1: the agent recovers not just
*where* the state is but *how it is moving* — the velocity belief ``μ_x'`` tracks the
true velocity ``ẋ* = θ_x − x*``.

The point attractor has analytic derivatives (``x*(t)=θ_x−(θ_x−x_0)e^{-t}``,
``ẋ*=θ_x−x*``, ``ẍ*=x*−θ_x``), which we embed as the generalized measurement
``ỹ=[g(x*), g'·ẋ*, g'·ẍ*]`` (+ small smooth noise). Proper colored-noise smoothing is
covered in §6.6. Reproduces the spirit of Figures 6.5.3 / 6.5.5.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    GeneralizedModel,
    LinearFunction,
    generalized_filter_gc,
    get_logger,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import finalize, panel_grid, save_or_show
from active_inference.visualizations.style import COLORS, annotate_stat_box

LOG = get_logger("ch6.ex6")

THETA_X, THETA_Y, X0 = 10.0, 3.0, 5.0


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--embedding-dim", type=int, default=3)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    t = np.arange(args.n_steps + 1) * args.dt

    # Analytic point-attractor trajectory and its temporal derivatives.
    x_star = THETA_X - (THETA_X - X0) * np.exp(-t)        # position
    xdot = THETA_X - x_star                               # velocity  (= (θ_x−x0)e^{-t})
    xddot = x_star - THETA_X                              # acceleration

    # Generalized measurements ỹ = [g(x*), g'·ẋ*, g'·ẍ*]  (g(x)=x−θ_y, g'=1) + smooth noise.
    g_vals = x_star - THETA_Y
    ys = np.stack([g_vals, xdot, xddot], axis=1)[:, : args.embedding_dim]
    ys = ys + 0.05 * rng.standard_normal(ys.shape)

    model = GeneralizedModel(
        f=LinearFunction(-1.0, THETA_X), g=LinearFunction(1.0, -THETA_Y),
        precision_x=np.full(args.embedding_dim, 0.2),
        precision_y=np.full(args.embedding_dim, 50.0),
        embedding_dim=args.embedding_dim)
    mu0 = np.zeros(args.embedding_dim)
    mu0[0] = 15.0
    res = generalized_filter_gc(model, ys, dt=args.dt, kappa=1.0, mu0_tilde=mu0)

    burn = args.n_steps // 3
    pos_err = float(np.mean(np.abs(res.positions[burn:] - x_star[burn:])))
    vel_err = float(np.mean(np.abs(res.velocities[burn:] - xdot[burn:])))
    LOG.info("position tracking err = %.4f | velocity recovery err = %.4f", pos_err, vel_err)

    fig, axes = panel_grid(3, title="Fig. 6.5 · generalized filtering in generalized coordinates",
                           figsize=(15.5, 4.8))
    ax = axes[0]
    ax.plot(t, x_star, color=COLORS["truth"], lw=1.6, alpha=0.6, label=r"true $x^*$")
    ax.plot(t, res.positions, color=COLORS["posterior"], lw=2.4, label=r"belief $\mu_x^{[0]}$")
    annotate_stat_box(ax, f"pos err = {pos_err:.3f}\nμ₀ = 15", loc="lower right")
    finalize(ax, xlabel="time", ylabel="position", title="position (order 0)")

    ax = axes[1]
    ax.plot(t, xdot, color=COLORS["truth"], lw=1.6, alpha=0.6, label=r"true $\dot x^*$")
    ax.plot(t, res.velocities, color=COLORS["state"], lw=2.4, label=r"belief $\mu_x^{[1]}$")
    annotate_stat_box(ax, f"vel err = {vel_err:.3f}", loc="upper right")
    finalize(ax, xlabel="time", ylabel="velocity", title="velocity recovered (order 1)")

    ax = axes[2]
    ax.plot(t, res.free_energies, color=COLORS["data"], lw=2.2, label=r"$\mathcal{F}$")
    finalize(ax, xlabel="time", ylabel=r"$\mathcal{F}$", title="free energy")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_06")
        save_or_show(fig, out / "example_6_6_generalized_coordinates.png")
        LOG.info("Saved to %s", out / "example_6_6_generalized_coordinates.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
