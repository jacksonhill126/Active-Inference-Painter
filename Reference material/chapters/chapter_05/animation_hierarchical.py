"""Animation — hierarchical predictive coding converging (Chapter 5, bonus).

Run::

    python chapters/chapter_05/animation_hierarchical.py [--save]

Renders a GIF of §5.4 in motion (Example 5.7): the layer beliefs ``μ^{[l]}``
settling to ``[2, 1, 0]``, every layer prediction error ``ε^{[l]}`` decaying to
zero, and the per-layer and summed free energy collapsing to ``Σ F = 0``. The
dynamic version of Figure 5.4.4.

Built on the composable :func:`animate_hierarchical_pc`, which animates any
``HierarchicalPCResult``.
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
from active_inference.visualizations import animate_hierarchical_pc, save_animation

LOG = get_logger("ch5.anim_hierarchical")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--kappa", type=float, default=0.03)
    p.add_argument("--n-iter", type=int, default=600)
    p.add_argument("--stride", type=int, default=6,
                   help="animate every Nth iteration (keeps the GIF small)")
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    y_obs = 2.0
    model = HierarchicalPCModel(
        generators=[QuadraticFunction(1.0, 1.0), QuadraticFunction(1.0, 1.0)],
        variances=[1.0, 1.0, 1.0],
        m_x=0.0,  # unconstrained top (Example 5.7)
    )
    res = hierarchical_predictive_coding(
        model, y_obs, mu0=[3.0, 3.0], kappa=args.kappa, n_iter=args.n_iter, tol=1e-14)
    LOG.info("hierarchical mu*=%s  summed F=%.3e  converged=%s (n=%d)",
             [round(v, 3) for v in res.mu_star], float(res.free_energies[-1]),
             res.converged, res.n_iter_run)

    anim = animate_hierarchical_pc(
        res, truth=[y_obs, 1.0, 0.0],
        title="Example 5.7 · hierarchical predictive coding converging",
        stride=args.stride)
    LOG.info("built hierarchical animation: %d iterations, stride=%d",
             res.mus.shape[0], args.stride)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_05")
        path = save_animation(anim, out / "animation_hierarchical.gif", fps=14)
        LOG.info("Saved to %s", path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
