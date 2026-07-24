"""§5.1 — Prediction errors and the free-energy landscape (Figs. 5.1.1 / 5.1.2).

Run::

    python chapters/chapter_05/example_5_1_prediction_errors.py [--save]

Predictive coding re-reads variational free energy as precision-weighted
prediction-error minimization. With the linear generating function ``g(x)=2x+3``
(the Chapter 3–4 model) and ``y=7`` the free energy ``F(μ_x)`` bottoms out where
the prediction errors balance:

* **Fig. 5.1.1** — a (near-)uniform prior leaves only the sensory term, so
  ``F = ε_y²/(2σ_y²)`` is minimized at the data-consistent state ``μ_x = 2 = x*``
  (the MLE), reaching the surprisal floor.
* **Fig. 5.1.2** — adding a Gaussian prior ``N(m_x, s_x²)`` pulls the minimum
  toward ``m_x``; the two squared precision-weighted prediction errors cross,
  vanishing at the data-consistent state and at ``m_x`` respectively.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    LinearFunction,
    PredictiveCodingModel,
    get_logger,
    predictive_coding_free_energy,
    surprisal,
)
from active_inference.core.generative_model import LinearGaussianModel
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import (
    finalize,
    panel_grid,
    plot_prediction_errors,
    save_or_show,
)
from active_inference.visualizations.style import COLORS

LOG = get_logger("ch5.ex1")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--y", type=float, default=7.0)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    mu_grid = np.linspace(0.0, 5.0, 400)

    # --- Fig 5.1.1: flat prior (MLE) — F = ε_y²/(2σ_y²), minimum at x* = 2 ---
    flat = PredictiveCodingModel(g=LinearFunction(2.0, 3.0), sigma2_y=1.0,
                                 m_x=4.0, s2_x=1e6)
    F_flat = np.array([predictive_coding_free_energy(flat, args.y, float(m)).free_energy
                       for m in mu_grid])
    # Surprisal floor from the equivalent Chapter-4 grid model (uniform-ish prior).
    lgm = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=1.0,
                              prior_kind="uniform", uniform_low=-50.0, uniform_high=50.0)
    floor = surprisal(lgm, args.y, np.linspace(-60, 60, 6001))
    mu_min_flat = float(mu_grid[int(np.argmin(F_flat))])

    fig1, axes = panel_grid(1, figsize=(7, 4.6),
                            title="Fig. 5.1.1 · free energy under a flat prior (MLE)")
    ax = axes[0]
    ax.plot(mu_grid, F_flat, color=COLORS["prior"], lw=2.2, label=r"$\mathcal{F}$")
    ax.plot(mu_min_flat, F_flat.min(), "o", color=COLORS["likelihood"], ms=9,
            label=rf"min at $\mu_x={mu_min_flat:.2f}$")
    ax.axhline(floor, color=COLORS["data"], ls="--", lw=1.4, label=r"$-\log p(y)$")
    finalize(ax, xlabel=r"$\mu_x$", ylabel=r"$\mathcal{F}$")
    LOG.info("Fig 5.1.1: flat-prior min at mu=%.3f (x*=2), surprisal floor=%.3f",
             mu_min_flat, floor)

    # --- Fig 5.1.2: Gaussian prior — both prediction errors contribute ---
    gauss = PredictiveCodingModel(g=LinearFunction(2.0, 3.0), sigma2_y=1.0,
                                  m_x=4.0, s2_x=1.0)
    fig2 = plot_prediction_errors(gauss, args.y, mu_grid, truth=2.0)
    mu_min = float(mu_grid[int(np.argmin(
        [predictive_coding_free_energy(gauss, args.y, float(m)).free_energy
         for m in mu_grid]))])
    LOG.info("Fig 5.1.2: Gaussian-prior min at mu=%.3f (between x*=2 and m_x=4)", mu_min)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_05")
        save_or_show(fig1, out / "example_5_1_mle_free_energy.png")
        save_or_show(fig2, out / "example_5_1_prediction_errors.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
