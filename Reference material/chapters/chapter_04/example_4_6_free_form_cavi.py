"""Example 4.6 — Free-form mean-field coordinate-ascent VI (§4.5).

Run::

    python chapters/chapter_04/example_4_6_free_form_cavi.py [--save]

Walks through Algorithm 4.5.1 on the three-unknown model of Eq. 32–34, where the
linear coefficients ``β₀`` and ``β₁`` are random variables alongside the hidden
state ``x`` (Fig. 4.5.1's Bayesian network). Under the **mean-field
approximation** ``q(ϑ) = q(x)q(β₀)q(β₁)``, each partition is updated in turn via
the fundamental theorem of mean-field VI (Eq. 43)::

    q(ϑ_s) ∝ exp( E_{q(ϑ_∖s)}[ log p(y, ϑ) ] )

For this conjugate linear-Gaussian model each update is closed-form Gaussian, so
the sweep is exact and VFE decreases monotonically until convergence. With a
single observation ``ŷ`` the problem is under-determined, so the partitions settle
on the prior-regularized fixed point whose reconstruction ``β₀+β₁·x ≈ ŷ``.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import MeanFieldConfig, free_form_cavi, get_logger
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show
from active_inference.visualizations.style import COLORS

LOG = get_logger("ch4.ex6")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--y", type=float, default=7.0)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    cfg = MeanFieldConfig()  # book priors (Eq. 34)
    cavi = free_form_cavi(args.y, cfg)
    recon = cavi.q_b0.mu + cavi.q_b1.mu * cavi.q_x.mu
    LOG.info("CAVI converged=%s in %d sweeps; q(x)=N(%.3f,%.3f) "
             "q(β0)=N(%.3f,%.3f) q(β1)=N(%.3f,%.3f)",
             cavi.converged, cavi.n_sweeps_run,
             cavi.q_x.mu, cavi.q_x.var, cavi.q_b0.mu, cavi.q_b0.var,
             cavi.q_b1.mu, cavi.q_b1.var)
    LOG.info("reconstruction β0+β1·x = %.3f (target ŷ = %.3f); final F = %.4f",
             recon, args.y, cavi.final_free_energy)

    fig, (ax_f, ax_mu) = plt.subplots(1, 2, figsize=(13, 4.5),
                                      constrained_layout=True)
    it = np.arange(len(cavi.free_energies))
    ax_f.plot(it, cavi.free_energies, color=COLORS["prior"], marker="o", ms=4)
    ax_f.set_xlabel("sweep")
    ax_f.set_ylabel(r"$\mathcal{F}$ (mean-field)")
    ax_f.set_title("VFE decreases monotonically (CAVI)")

    ax_mu.plot(it, cavi.mu_x, color=COLORS["posterior"], marker="o", ms=4,
               label=r"$\mu_x$")
    ax_mu.plot(it, cavi.mu_b0, color=COLORS["prior"], marker="s", ms=4,
               label=r"$\mu_{\beta_0}$")
    ax_mu.plot(it, cavi.mu_b1, color=COLORS["likelihood"], marker="^", ms=4,
               label=r"$\mu_{\beta_1}$")
    ax_mu.axhline(3.0, color=COLORS["prior"], ls=":", alpha=0.5)
    ax_mu.axhline(2.0, color=COLORS["likelihood"], ls=":", alpha=0.5)
    ax_mu.set_xlabel("sweep")
    ax_mu.set_ylabel("partition mean")
    ax_mu.set_title("Partition means over sweeps")
    ax_mu.legend(fontsize=9)
    fig.suptitle("Example 4.6 · free-form mean-field CAVI on (x, β₀, β₁)")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_04")
        save_or_show(fig, out / "example_4_6_cavi.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
