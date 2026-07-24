"""Example 6.1 — univariate generalized filtering for perception (Chapter 6).

Run::

    python chapters/chapter_06/example_6_1_generalized_filter.py [--save]

An agent tracks the moving position of an object (hidden state ``x*``, a point
attractor at ``θ_x*=10``) using only a noisy olfactory observation ``y``. The
environment is simulated by Euler-integrating the stochastic generative process;
the agent then filters the observation stream online, taking one Euler step down
the variational free energy per time point (Algorithm 6.1.1). With a high sensory
precision (``λ_y=50``) relative to the prior precision (``λ_x=0.2``), the belief
``μ_x`` locks onto and tracks the true state — passive perception in a dynamic
world. Reproduces Figures 6.1.2 / 6.1.3.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    DynamicProcess,
    DynamicStateSpaceModel,
    LinearFunction,
    generalized_filter,
    get_logger,
    gf_fixed_point_linear,
    simulate_dynamic_process,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_generalized_filter, save_or_show

LOG = get_logger("ch6.ex1")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--kappa", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Generative process E (Eq. 9): f_E(x)=θ_x*−x (attractor 10), g_E(x)=x−θ_y* (θ_y*=3).
    process = DynamicProcess(f=LinearFunction(-1.0, 10.0), g=LinearFunction(1.0, -3.0),
                             omega_x=0.1, omega_y=0.1)
    tr = simulate_dynamic_process(process, x0=5.0, n_steps=args.n_steps, dt=args.dt, rng=rng)

    # Generative model M (Eq. 10): same functions; book precisions λ_x=0.2, λ_y=50.
    model = DynamicStateSpaceModel(f=LinearFunction(-1.0, 10.0), g=LinearFunction(1.0, -3.0),
                                   s2_x=5.0, sigma2_y=0.02)
    res = generalized_filter(model, tr.ys, dt=args.dt, kappa=args.kappa, mu0=15.0)

    track_err = res.tracking_error(tr.xs, burn_in=args.n_steps // 3)
    unfiltered = float(np.mean(np.abs(15.0 - tr.xs[args.n_steps // 3:])))
    y_ss = float(tr.ys[-50:].mean())
    LOG.info("tracking error |mu_x - x*| = %.4f (unfiltered prior baseline = %.4f)",
             track_err, unfiltered)
    LOG.info("steady-state y=%.3f -> closed-form fixed point mu*=%.3f (true x*~%.2f)",
             y_ss, gf_fixed_point_linear(model, y_ss), float(tr.xs[-50:].mean()))

    fig = plot_generalized_filter(res, truth=tr.xs, dt=args.dt,
                                  title="Fig. 6.1.3 · generalized filtering for perception")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_06")
        save_or_show(fig, out / "example_6_1_generalized_filter.png")
        LOG.info("Saved to %s", out / "example_6_1_generalized_filter.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
