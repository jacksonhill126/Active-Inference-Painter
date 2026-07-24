"""§4.3 — Model evidence and Bayesian model comparison (Figs. 4.3.2 / 4.3.3).

Run::

    python chapters/chapter_04/visualize_model_comparison.py [--save]

A good generative model assigns high evidence ``p(y)`` (low surprisal) to the
data the environment actually produces; a bad one does not. Here we compare the
evidence density ``p(y) = ∫ p(y|x)p(x) dx`` of a well-tuned "good" model and a
mis-specified "bad" model against the true data-generating distribution, and draw
samples from each (reproducing the spirit of Figs. 4.3.2 and 4.3.3). The good
model's evidence overlaps the true input distribution; the bad model's does not.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import LinearGaussianModel, get_logger
from active_inference.core.distributions import gaussian_pdf
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show
from active_inference.visualizations.style import COLORS

LOG = get_logger("ch4.model_cmp")


def evidence_density(model: LinearGaussianModel, y_grid: np.ndarray,
                     x_grid: np.ndarray) -> np.ndarray:
    """``p(y) = ∫ p(y|x) p(x) dx`` evaluated across ``y_grid`` (grid integral)."""
    prior = np.exp(model.log_prior(x_grid))
    out = np.empty_like(y_grid)
    for i, y in enumerate(y_grid):
        lik = np.exp(model.log_likelihood(float(y), x_grid))
        out[i] = np.trapezoid(lik * prior, x_grid)
    return out


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    x_grid = np.linspace(-8.0, 8.0, 1201)
    y_grid = np.linspace(-5.0, 6.0, 400)

    # True data-generating distribution of y (standardized for the comparison).
    true_mu, true_s2 = 0.0, 0.6
    true_py = gaussian_pdf(y_grid, true_mu, true_s2)
    data = rng.normal(true_mu, np.sqrt(true_s2), size=40)

    # Good model: prior + likelihood tuned so its evidence matches the data.
    good = LinearGaussianModel(beta0=0.0, beta1=1.0, sigma2_y=0.4, m_x=0.0, s2_x=0.2)
    # Bad model: mis-centered prior so its evidence sits to the left.
    bad = LinearGaussianModel(beta0=0.0, beta1=1.0, sigma2_y=0.9, m_x=-2.0, s2_x=0.6)

    good_py = evidence_density(good, y_grid, x_grid)
    bad_py = evidence_density(bad, y_grid, x_grid)

    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    ax.plot(y_grid, true_py, color=COLORS["prior"], lw=2.5, label="true model")
    ax.plot(y_grid, good_py, color=COLORS["posterior"], lw=2.5, label="good model")
    ax.plot(y_grid, bad_py, color=COLORS["likelihood"], lw=2.5, label="bad model")
    ax.scatter(data, np.full_like(data, 0.005), marker="|", color="black",
               alpha=0.7, label="data")
    ax.set_xlabel("y")
    ax.set_ylabel("p(y)")
    ax.set_title("Figs. 4.3.2/4.3.3 · model evidence vs the true input distribution")
    ax.legend(loc="upper right", fontsize=10)

    # Report mean log-evidence (negative mean surprisal) of the observed data.
    for name, model in (("good", good), ("bad", bad)):
        py_data = np.interp(data, y_grid, evidence_density(model, y_grid, x_grid))
        mean_surprisal = float(-np.mean(np.log(np.clip(py_data, 1e-12, None))))
        LOG.info("%-4s model: mean surprisal over data = %.3f", name, mean_surprisal)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_04")
        save_or_show(fig, out / "visualize_model_comparison.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()