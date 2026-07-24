"""§4.1 — KL divergence as a loss function for the posterior.

Run::

    python chapters/chapter_04/visualize_kl_loss.py [--save]

Section 4.1 proposes ``D_KL(q(x) ‖ p(x|y))`` as the loss to minimize: it is zero
exactly when ``q`` equals the true posterior (Gibbs inequality). The catch is
that this loss *contains the posterior we are trying to find*. This figure plots
that divergence — the D-form ``bound_gap`` — over the variational parameters
``(μ_x, σ²_x)``, showing the single minimum at the true posterior, and overlays
the VFE surface that §4.2 uses to dodge the circularity (the two surfaces differ
only by the constant surprisal).
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    GaussianBelief,
    GridBayesianInference,
    LinearGaussianModel,
    get_logger,
    variational_free_energy,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show
from active_inference.visualizations.style import COLORS

LOG = get_logger("ch4.kl_loss")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--y", type=float, default=7.0)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    x_grid = np.linspace(-6.0, 12.0, 1501)
    model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                                m_x=4.0, s2_x=0.25)
    exact = GridBayesianInference(model=model, x_grid=x_grid).infer(args.y)
    le, post = float(exact.log_evidence), np.asarray(exact.posterior)

    mus = np.linspace(0.0, 5.0, 60)
    vars_ = np.linspace(0.02, 2.0, 60)
    MU, VAR = np.meshgrid(mus, vars_)
    KL = np.empty_like(MU)
    F = np.empty_like(MU)
    for i in range(MU.shape[0]):
        for j in range(MU.shape[1]):
            c = variational_free_energy(GaussianBelief(MU[i, j], VAR[i, j]),
                                        model, args.y, x_grid,
                                        log_evidence=le, posterior=post)
            KL[i, j] = c.divergence
            F[i, j] = c.free_energy

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    c0 = a0.contourf(MU, VAR, KL, levels=20, cmap="magma")
    fig.colorbar(c0, ax=a0, label=r"$D_{KL}(q\,\|\,p(x|y))$")
    a0.plot(exact.posterior_mean, exact.posterior_variance, "+",
            color=COLORS["posterior"], ms=16, mew=3, label="true posterior")
    a0.set_title("§4.1 · KL loss (min = posterior, but needs p(x|y))")
    a0.set_xlabel(r"$\mu_x$")
    a0.set_ylabel(r"$\sigma_x^2$")
    a0.legend(fontsize=9)

    c1 = a1.contourf(MU, VAR, F, levels=20, cmap="viridis")
    fig.colorbar(c1, ax=a1, label=r"$\mathcal{F}$")
    a1.plot(exact.posterior_mean, exact.posterior_variance, "+",
            color=COLORS["posterior"], ms=16, mew=3)
    a1.set_title("§4.2 · VFE (same minimum, no posterior needed)")
    a1.set_xlabel(r"$\mu_x$")
    a1.set_ylabel(r"$\sigma_x^2$")

    LOG.info("KL surface min = %.3e at posterior; VFE − KL = surprisal = %.3f",
             KL.min(), -exact.log_evidence)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_04")
        save_or_show(fig, out / "visualize_kl_loss.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
