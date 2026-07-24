"""§5.2 — Precision balances prediction errors (Fig. 5.1.4).

Run::

    python chapters/chapter_05/example_5_2_precision.py [--save]

The predictive-coding free energy is a sum of two precision-weighted squared
prediction errors, ``F = ½(λ_y ε_y² + λ_x ε_x²) + const`` (book Eq. 7b), so the
belief that minimizes it is a *precision-weighted compromise* between the
data-consistent state (where the sensory error ``ε_y = y − g(μ)`` vanishes) and
the prior mean ``m_x`` (where the state error ``ε_x = μ − m_x`` vanishes).

Using the Chapter-4/5 running model ``g(x) = 2x + 3`` with ``ŷ = 7`` (so the
data-consistent state is ``x* = 2``) and prior mean ``m_x = 4``, this example
sweeps the book's three variance settings ``(s_x², σ_y²)``:

* ``(0.5, 2.0)`` — loose likelihood, tighter prior → belief pulled toward ``m_x``;
* ``(0.1, 1.0)`` — balanced;
* ``(1.0, 0.1)`` — sharp likelihood → belief pulled toward the data ``x* = 2``.

The single governing quantity is the **precision ratio** ``λ_x / λ_y = σ_y² / s_x²``:
the free-energy minimum lands at the ratio-weighted average
``μ* = (λ_x m_x + λ_y·g⁻¹-pull) / …`` — computed in closed form by
:func:`~active_inference.pc_linear_fixed_point` and cross-checked against the grid
minimum for every setting (analytic-vs-numerical oracle).
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    LinearFunction,
    PredictiveCodingModel,
    get_logger,
    oracle_agreement,
    pc_linear_fixed_point,
    predictive_coding_free_energy,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import finalize, panel_grid, save_or_show
from active_inference.visualizations.style import COLORS

LOG = get_logger("ch5.ex2")

# Book §5.2 variance settings (s_x², σ_y²) and the running model constants.
SETTINGS = [(0.5, 2.0), (0.1, 1.0), (1.0, 0.1)]
M_X = 4.0
Y_OBS = 7.0
X_STAR = 2.0  # data-consistent state: g(2) = 2·2 + 3 = 7 = ŷ


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--y", type=float, default=Y_OBS)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    mu_grid = np.linspace(0.0, 6.0, 1201)
    palette = [COLORS["prior"], COLORS["neutral"], COLORS["likelihood"]]

    fig, axes = panel_grid(2, figsize=(12, 4.6),
                           title="§5.2 · precision balances the two prediction errors")
    ax_fe, ax_pull = axes

    ratios, minima = [], []
    for (s2_x, sigma2_y), color in zip(SETTINGS, palette):
        model = PredictiveCodingModel(g=LinearFunction(2.0, 3.0), sigma2_y=sigma2_y,
                                      m_x=M_X, s2_x=s2_x)
        fe = np.array([predictive_coding_free_energy(model, args.y, float(m)).free_energy
                       for m in mu_grid])
        grid_min = float(mu_grid[int(np.argmin(fe))])
        oracle = pc_linear_fixed_point(model, args.y)  # closed-form FE minimizer
        agree = oracle_agreement(grid_min, oracle, tol=1e-2)
        ratio = sigma2_y / s2_x  # λ_x / λ_y
        ratios.append(ratio)
        minima.append(oracle)
        LOG.info("(s²_x=%.2f, σ²_y=%.2f) | λ_x/λ_y=%.2f | μ*=%.4f (grid %.4f, agree=%s) ",
                 s2_x, sigma2_y, ratio, oracle, grid_min, agree.passed)

        label = rf"$s_x^2$={s2_x}, $\sigma_y^2$={sigma2_y} ($\lambda_x/\lambda_y$={ratio:.2g})"
        ax_fe.plot(mu_grid, fe, color=color, lw=2.2, label=label)
        ax_fe.plot(oracle, float(np.min(fe)), "o", color=color, ms=8)

    ax_fe.axvline(X_STAR, color=COLORS["truth"], ls=":", lw=1.6,
                  label=rf"data $x^*$={X_STAR}")
    ax_fe.axvline(M_X, color=COLORS["data"], ls="--", lw=1.4, label=rf"prior $m_x$={M_X}")
    finalize(ax_fe, xlabel=r"$\mu_x$", ylabel=r"$\mathcal{F}$", title="free-energy landscape")

    # Right: the FE minimum slides from the prior mean toward the data as the
    # precision ratio λ_x/λ_y falls (sharper likelihood ⇒ trust the data).
    order = np.argsort(ratios)
    r_sorted = np.asarray(ratios)[order]
    m_sorted = np.asarray(minima)[order]
    ax_pull.plot(r_sorted, m_sorted, "o-", color=COLORS["posterior"], lw=2.2, ms=9)
    ax_pull.axhline(X_STAR, color=COLORS["truth"], ls=":", lw=1.6, label=rf"data $x^*$={X_STAR}")
    ax_pull.axhline(M_X, color=COLORS["data"], ls="--", lw=1.4, label=rf"prior $m_x$={M_X}")
    ax_pull.set_xscale("log")
    finalize(ax_pull, xlabel=r"precision ratio $\lambda_x/\lambda_y=\sigma_y^2/s_x^2$",
             ylabel=r"belief $\mu^*$", title="prior ↔ data pull")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_05")
        save_or_show(fig, out / "example_5_2_precision.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
