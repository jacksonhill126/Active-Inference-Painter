"""Examples 4.3–4.5 — The G, C, and E forms of variational free energy (§4.4).

Run::

    python chapters/chapter_04/example_4_3_vfe_forms.py [--save]

Reproduces Figure 4.4.1. Variational free energy is one number with several
algebraically-equivalent decompositions, each giving different intuition:

* **G / D form** (Eq. 27): ``F = divergence − surprisal``. Surprisal ``−log p(y)``
  is the constant lower boundary the iterations descend toward.
* **C form** (Eq. 28): ``F = complexity − accuracy``. Accuracy rises (better
  prediction) while complexity rises (the prior is updated) — a trade-off.
* **E form** (Eq. 29): ``F = −entropy − energy``. Average energy rises and entropy
  rises together as VFE falls (maximum-entropy behaviour).

We trace all three over the iterations of coordinate search on the Example-4.1
model and verify every form reconstructs the same ``F`` at each step.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    GaussianBelief,
    LinearGaussianModel,
    coordinate_search_vfe,
    get_logger,
    variational_free_energy,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_vfe_decomposition, save_or_show

LOG = get_logger("ch4.ex3")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--y", type=float, default=7.0)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    x_grid = np.linspace(-6.0, 12.0, 2001)
    model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                                m_x=4.0, s2_x=0.25)

    cs = coordinate_search_vfe(model, args.y, x_grid, kappa=0.05, n_iter=20)
    components = [
        variational_free_energy(GaussianBelief(mu, var), model, args.y, x_grid)
        for mu, var in zip(cs.mus, cs.vars_)
    ]

    # Verify all four forms agree at every iteration (the chapter's core claim).
    max_disagreement = 0.0
    for c in components:
        c.check(atol=1e-6)
        max_disagreement = max(
            max_disagreement,
            abs(c.g_form - c.free_energy), abs(c.d_form - c.free_energy),
            abs(c.c_form - c.free_energy), abs(c.e_form - c.free_energy),
        )
    LOG.info("all four VFE forms agree across %d iterations (max Δ = %.2e)",
             len(components), max_disagreement)
    LOG.info("final F=%.3f  surprisal=%.3f  divergence=%.3f",
             components[-1].free_energy, components[-1].surprisal,
             components[-1].divergence)

    fig = plot_vfe_decomposition(
        components, title="Examples 4.3–4.5 · G / C / E decompositions of VFE")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_04")
        save_or_show(fig, out / "example_4_3_vfe_forms.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
