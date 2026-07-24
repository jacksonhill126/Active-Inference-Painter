"""Example 7.2 — active inference: countering an exogenous force (Chapter 7 §7.4).

Run::

    python chapters/chapter_07/example_7_2_active_inference.py [--save]

An agent in a 1-D environment is pushed by an exogenous current toward `x* = v* = 10`.
The agent *prefers* to be at `v = 0` (its set-point, encoded as the point attractor of
its state-transition model). Before action turns on it can only perceive — its belief
`μ_x` tracks the true state, which sits at the exogenous attractor. Once action turns on
(dashed line), the agent applies a control force `a` that drives the true state down to
its preferred set-point and holds it there against the current (action settles at
`a ≈ −v*`). This is the **action-perception cycle** (Algorithm 7.2.1): *the agent changes
the world to match its model*. Reproduces Figures 7.4.4 / 7.4.5.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    ActiveEnvironment,
    ActiveInferenceAgent,
    DynamicStateSpaceModel,
    LinearFunction,
    get_logger,
    simulate_active_inference,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import finalize, panel_grid, save_or_show
from active_inference.visualizations.style import COLORS, annotate_stat_box

LOG = get_logger("ch7.ex2")

V_STAR, V_PREF, THETA_Y = 10.0, 0.0, 3.0


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=6000)
    p.add_argument("--dt", type=float, default=0.01)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    action_start = args.n_steps // 3

    # Environment E: drift = v* − x* (exogenous attractor at v*), obs g_E = x − θ_y.
    env = ActiveEnvironment(drift=LinearFunction(-1.0, V_STAR), g=LinearFunction(1.0, -THETA_Y),
                            omega_x=0.05, omega_y=0.05)
    # Agent M: preference v=0 (attractor f_M = v − μ), matched obs g_M = μ − θ_y.
    model = DynamicStateSpaceModel(f=LinearFunction(-1.0, V_PREF), g=LinearFunction(1.0, -THETA_Y),
                                   s2_x=1.0, sigma2_y=0.05)
    agent = ActiveInferenceAgent(perception_model=model, forward_model=1.0,
                                 kappa_x=0.2, kappa_a=0.4)
    res = simulate_active_inference(agent, env, x0=5.0, mu0=5.0, n_steps=args.n_steps,
                                    dt=args.dt, action_start=action_start,
                                    rng=np.random.default_rng(args.seed))

    LOG.info("settled state x* = %.3f (preference v=%.1f) | settled action a = %.3f (≈ −v*=%.1f)",
             res.settled_state(), V_PREF, res.settled_action(), -V_STAR)

    t = np.arange(res.xs.shape[0]) * args.dt
    t_on = action_start * args.dt
    fig, axes = panel_grid(3, title="Fig. 7.4.5 · active inference — countering an exogenous force",
                           figsize=(15.5, 4.8))

    ax = axes[0]
    ax.plot(t, res.xs, color=COLORS["truth"], lw=1.4, alpha=0.6, label=r"true $x^*$")
    ax.plot(t, res.mus, color=COLORS["posterior"], lw=2.2, label=r"belief $\mu_x$")
    ax.plot(t, res.actions, color=COLORS["state"], lw=2.0, label=r"action $a$")
    ax.axhline(V_PREF, color=COLORS["likelihood"], ls=":", lw=1.6, label=r"preference $v$")
    ax.axhline(V_STAR, color=COLORS["neutral"], ls="--", lw=1.2, label=r"exogenous $v^*$")
    ax.axvline(t_on, color="black", ls="--", lw=1.4)
    annotate_stat_box(ax, f"x*→{res.settled_state():.2f}\na→{res.settled_action():.2f}",
                      loc="center right")
    finalize(ax, xlabel="time", ylabel="states and action", title="action drives x* → preference")

    ax = axes[1]
    ax.plot(t, res.ys, color=COLORS["sensory"], lw=1.2, alpha=0.7, label=r"obs $y$")
    ax.plot(t, res.eps_y, color=COLORS["state"], lw=1.6, label=r"$\varepsilon_y$")
    ax.axhline(0.0, color=COLORS["neutral"], ls="--", lw=1.0)
    ax.axvline(t_on, color="black", ls="--", lw=1.4)
    finalize(ax, xlabel="time", ylabel="obs / error", title="sensory error → 0 under action")

    ax = axes[2]
    ax.plot(t, res.free_energies, color=COLORS["data"], lw=2.0, label=r"$\mathcal{F}$")
    ax.axvline(t_on, color="black", ls="--", lw=1.4, label="action on")
    finalize(ax, xlabel="time", ylabel=r"$\mathcal{F}$", title="free energy")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_07")
        save_or_show(fig, out / "example_7_2_active_inference.png")
        LOG.info("Saved to %s", out / "example_7_2_active_inference.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
