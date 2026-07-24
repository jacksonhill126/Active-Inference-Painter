"""§5.4 — Hierarchical predictive coding (Example 5.7, Fig. 5.4.4).

Run::

    python chapters/chapter_05/example_5_7_hierarchical.py [--save]

A two-hidden-layer hierarchy with generating function ``g(x)=x²+1`` at each
transition and unit variances. The true bottom-level cause is ``x*=1`` so the
(noise-free) sensory input is ``y = 1²+1 = 2``. With an unconstrained top layer
(``g^[L+1]=0``) and hidden states initialised at 3, the recognition dynamics
drive ``μ^[1] → 1`` and ``μ^[2] → 0`` (since ``g(0)=1=μ^[1]``); every layer-wise
prediction error and the summed free energy go to zero — "bottom-up prediction
errors meet top-down predictions."
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from active_inference import (
    HierarchicalPCModel,
    QuadraticFunction,
    get_logger,
    hierarchical_predictive_coding,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_hierarchical_pc, save_or_show

LOG = get_logger("ch5.ex7")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--kappa", type=float, default=0.03)
    p.add_argument("--n-iter", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    x_star = 1.0
    y_obs = x_star**2 + 1.0  # = 2.0, the noise-free sensory input

    model = HierarchicalPCModel(
        generators=[QuadraticFunction(1.0, 1.0), QuadraticFunction(1.0, 1.0)],
        variances=[1.0, 1.0, 1.0],
        m_x=0.0,  # unconstrained top (Example 5.7)
    )
    res = hierarchical_predictive_coding(
        model, y_obs, mu0=[3.0, 3.0], kappa=args.kappa, n_iter=args.n_iter, tol=1e-14)
    LOG.info("hierarchical mu*=%s  summed F=%.4g  converged=%s (n=%d)",
             [round(v, 4) for v in res.mu_star], res.final_free_energy,
             res.converged, res.n_iter_run)

    fig = plot_hierarchical_pc(res, truth=[y_obs, 1.0, 0.0])

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_05")
        save_or_show(fig, out / "example_5_7_hierarchical.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
