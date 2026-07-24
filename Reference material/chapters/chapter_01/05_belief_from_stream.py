"""Chapter 1 · Animated belief formation from the box-scenario sensor stream.

Run::

    python chapters/chapter_01/05_belief_from_stream.py [--save] [--n 60]

The static ``01_box_scenario.py`` shows the stream the agent receives; this
animation shows the agent *making sense of it* (§1.1–1.3). Observations arrive one
at a time and the posterior over the hidden state ``x`` sharpens toward the true
``x*`` as evidence accumulates — the chapter's central picture of perception as
inverting a generative process.

Two validated statistics are surfaced as the belief tightens:

* **Consistency** — the final posterior mode matches the closed-form uniform-prior
  inverse ``(ȳ − β₀)/β₁`` (the MLE), cross-checked with :func:`oracle_agreement`.
* **Concentration** — the posterior standard deviation ``σ_n`` shrinks like
  ``1/√N``; the product ``σ_n·√N`` is approximately constant, logged as a range so
  the ``1/√N`` law is visible rather than asserted.

Configurable via ``--n``, ``--x-true``, ``--sigma2-y``, ``--prior-mean``,
``--prior-var``, ``--seed``, and ``--fps``. Saves a GIF (plus NPZ+JSON provenance)
when ``--save`` is passed.
"""

from __future__ import annotations

import argparse

import numpy as np

from active_inference import (
    LinearGaussianModel,
    LinearGaussianProcess,
    get_logger,
    make_grid,
    mle_analytic_linear,
    oracle_agreement,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import animate_stream_belief, save_animation

LOG = get_logger("ch1.stream_belief")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n", type=int, default=60, help="number of streamed observations")
    p.add_argument("--x-true", type=float, default=2.5, help="hidden environmental state x*")
    p.add_argument("--sigma2-y", type=float, default=0.5, help="observation noise variance")
    p.add_argument("--prior-mean", type=float, default=4.0)
    p.add_argument("--prior-var", type=float, default=1.0)
    p.add_argument("--fps", type=int, default=12)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=args.sigma2_y, rng=rng)
    observations = process.sample(args.x_true, n=args.n).flatten()

    x_grid = make_grid(0.0, 5.0, 500)
    model = LinearGaussianModel(
        beta0=3.0, beta1=2.0, sigma2_y=args.sigma2_y,
        m_x=args.prior_mean, s2_x=args.prior_var, prior_kind="gaussian",
    )

    # Sequential assimilation in log-space (numerically stable running posterior).
    log_state = model.log_prior(x_grid).copy()
    prior_density = np.exp(log_state - np.max(log_state))
    prior_density /= np.trapezoid(prior_density, x_grid)

    posteriors: list[np.ndarray] = []
    posterior_stds: list[float] = []
    for y in observations:
        log_state = log_state + model.log_likelihood(float(y), x_grid)
        density = np.exp(log_state - np.max(log_state))
        density /= np.trapezoid(density, x_grid)
        mean = float(np.trapezoid(x_grid * density, x_grid))
        var = float(np.trapezoid((x_grid - mean) ** 2 * density, x_grid))
        posteriors.append(density)
        posterior_stds.append(np.sqrt(max(var, 0.0)))

    final_mode = float(x_grid[int(np.argmax(posteriors[-1]))])
    mle = mle_analytic_linear(observations, beta0=3.0, beta1=2.0)
    agree = oracle_agreement(final_mode, float(mle), tol=5e-2)
    prod = np.asarray(posterior_stds) * np.sqrt(np.arange(1, args.n + 1))
    LOG.info("final mode=%.4f | MLE=%.4f | x*=%.3f | oracle agree=%s (|Δ|=%.2e)",
             final_mode, mle, args.x_true, agree.passed, agree.abs_error)
    LOG.info("1/√N concentration: σ·√N ranges %.3f–%.3f over N=1..%d (≈constant)",
             float(prod.min()), float(prod.max()), args.n)

    anim = animate_stream_belief(
        x_grid, observations, posteriors,
        truth=args.x_true, true_mean=float(process.mean(args.x_true)),
        prior=prior_density, posterior_stds=posterior_stds,
        title="Chapter 1 · reconstructing the hidden state from the stream",
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_01")
        path = save_animation(anim, out / "05_belief_from_stream.gif", fps=args.fps)
        LOG.info("Saved animation to %s", path)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
