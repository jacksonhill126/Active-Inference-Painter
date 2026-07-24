"""Example 4.1 — Finding the minima of VFE with coordinate search (§4.2).

Run::

    python chapters/chapter_04/example_4_1_coordinate_search.py [--save]

Reproduces Figure 4.2.2. The agent infers food size ``x`` from light intensity
``y`` by minimizing variational free energy with **coordinate search**
(Algorithm 4.2.1): a zero-order optimizer that, from the current ``(μ_x, σ²_x)``,
evaluates VFE at the eight neighbours a step ``κ`` away and jumps to the lowest.

Generative process (Eq. 11): ``y = β₀ + β₁·x* + ω``,  β₀=3, β₁=2, σ²=1.
Generative model    (Eq. 12): likelihood ``N(y; β₀+β₁x, σ²_y)``, prior
``N(x; m_x, s²_x)`` with ``σ²_y=0.25, s²_x=0.25, m_x=4``. With the observation
clamped at ``ŷ=7`` the exact posterior is ``N(2.4, 0.05)``; coordinate search
with the book's ``κ=0.01`` over 20 iterations intentionally stops short of it
(book §4.4), which is exactly why VFE never quite reaches the surprisal bound.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    GaussianBelief,
    GridBayesianInference,
    LinearGaussianModel,
    coordinate_search_vfe,
    get_logger,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import (
    plot_density_evolution,
    plot_vfe_contour,
    save_or_show,
)

LOG = get_logger("ch4.ex1")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--y", type=float, default=7.0, help="clamped observation ŷ")
    p.add_argument("--kappa", type=float, default=0.01, help="search step κ")
    p.add_argument("--iters", type=int, default=20, help="iteration budget")
    p.add_argument("--extended", action="store_true",
                   help="run long enough (κ=0.05, 200 it) to reach the minimum")
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    x_grid = np.linspace(-6.0, 12.0, 2001)
    model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                                m_x=4.0, s2_x=0.25)
    exact = GridBayesianInference(model=model, x_grid=x_grid).infer(args.y)

    kappa, iters = (0.05, 200) if args.extended else (args.kappa, args.iters)
    cs = coordinate_search_vfe(model, args.y, x_grid, kappa=kappa, n_iter=iters)
    LOG.info("coordinate search: μ=%.3f σ²=%.3f F=%.3f (bound −logZ=%.3f) after %d steps",
             cs.belief.mu, cs.belief.var, cs.final_free_energy,
             -exact.log_evidence, cs.n_iter_run)
    LOG.info("exact posterior: mean=%.3f var=%.3f", exact.posterior_mean,
             exact.posterior_variance)

    # Figure 4.2.2 left — VFE contour with the search path.
    fig_contour = plot_vfe_contour(
        model, args.y, x_grid, mu_lo=0.0, mu_hi=5.0, var_lo=0.02, var_hi=2.0,
        n_mu=60, n_var=60, path_mus=cs.mus, path_vars=cs.vars_,
        truth=(exact.posterior_mean, exact.posterior_variance),
        title="Example 4.1 · VFE contour and coordinate-search path",
    )

    # Figure 4.2.2 right — variational density over iterations.
    beliefs = [GaussianBelief(mu, var) for mu, var in zip(cs.mus, cs.vars_)]
    fig_density = plot_density_evolution(
        x_grid, beliefs, posterior=exact.posterior,
        title="Example 4.1 · q(x) converging toward the posterior",
        xlabel="food size x",
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_04")
        save_or_show(fig_contour, out / "example_4_1_vfe_contour.png")
        save_or_show(fig_density, out / "example_4_1_density_evolution.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
