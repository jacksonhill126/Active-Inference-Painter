"""Example 4.7 — Fixed-form variational inference (§4.6).

Run::

    python chapters/chapter_04/example_4_7_fixed_form.py [--save]

Reproduces Figures 4.6.1 and 4.6.2. Same model as Example 4.1, but instead of
zero-order coordinate search we assume ``q(x)=N(μ_x, σ²_x)`` and follow the
**gradient** of VFE with respect to its parameters (Algorithm 4.6.1, Eq. 47)::

    μ_x  ← μ_x  − κ · ∂F/∂μ_x
    σ²_x ← σ²_x − κ · ∂F/∂σ²_x

The book used PyTorch autodiff; this companion keeps its dependency set to
numpy/scipy/matplotlib and estimates the gradient by central finite differences
— the same method, no autodiff dependency. The descent is smooth and converges
onto the exact posterior ``N(2.4, 0.05)``, where ``F = −log p(y)`` (the bound).
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    GaussianBelief,
    GridBayesianInference,
    LinearGaussianModel,
    fixed_form_vi,
    get_logger,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import (
    plot_density_evolution,
    plot_vfe_contour,
    plot_vfe_decomposition,
    save_or_show,
)

LOG = get_logger("ch4.ex7")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--y", type=float, default=7.0)
    p.add_argument("--lr", type=float, default=5e-3, help="learning rate κ")
    p.add_argument("--iters", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    x_grid = np.linspace(-6.0, 12.0, 2001)
    model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                                m_x=4.0, s2_x=0.25)
    exact = GridBayesianInference(model=model, x_grid=x_grid).infer(args.y)

    ff = fixed_form_vi(model, args.y, x_grid, lr=args.lr, n_iter=args.iters)
    LOG.info("fixed-form VI: μ=%.4f σ²=%.4f F=%.4f (bound −logZ=%.4f) converged=%s",
             ff.belief.mu, ff.belief.var, ff.final_free_energy,
             -exact.log_evidence, ff.converged)
    LOG.info("exact posterior: mean=%.4f var=%.4f", exact.posterior_mean,
             exact.posterior_variance)

    # Thin the iterate trace for readable plots.
    step = max(1, len(ff.mus) // 25)
    idx = list(range(0, len(ff.mus), step)) + [len(ff.mus) - 1]
    beliefs = [GaussianBelief(ff.mus[i], ff.vars_[i]) for i in idx]

    # Figure 4.6.1 — contour with smooth descent path + density evolution.
    fig_contour = plot_vfe_contour(
        model, args.y, x_grid, mu_lo=0.0, mu_hi=5.0, var_lo=0.02, var_hi=2.0,
        n_mu=60, n_var=60, path_mus=ff.mus[::step], path_vars=ff.vars_[::step],
        truth=(exact.posterior_mean, exact.posterior_variance),
        title="Example 4.7 · fixed-form VI gradient descent on VFE",
    )
    fig_density = plot_density_evolution(
        x_grid, beliefs, posterior=exact.posterior,
        title="Example 4.7 · q(x) converging (smooth gradient descent)",
        xlabel="food size x",
    )

    # Figure 4.6.2 — G/C/E decompositions over iterations.
    comps = [ff.components[i] for i in idx]
    fig_decomp = plot_vfe_decomposition(
        comps, title="Example 4.7 · VFE decompositions over iterations")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_04")
        save_or_show(fig_contour, out / "example_4_7_vfe_contour.png")
        save_or_show(fig_density, out / "example_4_7_density_evolution.png")
        save_or_show(fig_decomp, out / "example_4_7_decomposition.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
