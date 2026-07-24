"""Example 4.2 — Surprisal and the marginal likelihood (§4.3).

Run::

    python chapters/chapter_04/example_4_2_surprisal.py [--save]

Reproduces Figure 4.3.1. Surprisal is ``−log p(y)``: a high-probability (expected)
observation has *low* surprisal, a low-probability (unexpected) observation has
*high* surprisal. Variational free energy is an upper bound on surprisal
(Eq. 26), so minimizing VFE pushes the agent toward a model under which its
sensory data are unsurprising — equivalently, toward higher model evidence.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    GaussianBelief,
    GridBayesianInference,
    LinearGaussianModel,
    free_energy_bound_gap,
    get_logger,
    surprisal,
    variational_free_energy,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_surprisal_relationship, save_or_show

LOG = get_logger("ch4.ex2")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    fig = plot_surprisal_relationship(mu=0.0, sigma2=0.64,
                                      title="Example 4.2 · evidence p(y) vs surprisal")

    # Numerical sanity: a typical observation is less surprising than an outlier.
    x_grid = np.linspace(-6.0, 12.0, 2001)
    model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                                m_x=4.0, s2_x=0.25)
    inf = GridBayesianInference(model=model, x_grid=x_grid)
    for y in (7.0, 0.0):
        s = surprisal(model, y, x_grid)
        LOG.info("ŷ=%.1f → log p(y)=%.3f  surprisal=%.3f",
                 y, inf.infer(y).log_evidence, s)

    # Book Eq. 26: F(q) ≥ −log p(y) for *any* q, tight (gap → 0) only at the exact
    # posterior. Demonstrate the inequality holds and is minimized at the posterior.
    y_demo = 7.0
    post = inf.infer(y_demo)
    posterior_belief = GaussianBelief(mu=post.posterior_mean, var=post.posterior_variance)
    beliefs = {
        "prior-centred q": GaussianBelief(mu=4.0, var=0.25),
        "misfit q": GaussianBelief(mu=0.0, var=1.0),
        "≈posterior q": posterior_belief,
    }
    surp = surprisal(model, y_demo, x_grid)
    min_gap = float("inf")
    for name, q in beliefs.items():
        F = variational_free_energy(q, model, y_demo, x_grid).free_energy
        gap = free_energy_bound_gap(q, model, y_demo, x_grid)
        min_gap = min(min_gap, gap)
        LOG.info("  %-16s F=%.4f  ≥ surprisal=%.4f ?  gap=%.3e (≥0=%s)",
                 name, F, surp, gap, bool(gap >= -1e-8))
    LOG.info("upper bound Eq.26 verified: min gap over beliefs = %.3e "
             "(tight at the posterior)", min_gap)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_04")
        save_or_show(fig, out / "example_4_2_surprisal.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
