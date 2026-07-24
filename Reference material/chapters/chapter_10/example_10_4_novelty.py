"""Example 10.4 — parameter information gain (novelty) and novelty-driven learning (§10.1).

Run::

    python chapters/chapter_10/example_10_4_novelty.py [--save]

The novelty-augmented expected free energy (Eq. 15) adds a **parameter information-gain**
term, ``novelty = o · (W s)`` with ``W ≈ ½(1/a − 1/a₀)`` (Eq. 12). It is large for
state–observation pairs the agent has few counts for, driving *exploration that learns*.

Left: the book's Example 10.4 oracle — for ``a = [[2.5, 0.15], [0.8, 3]]`` the novelty
matrix is ``W = [[0.048, 3.175], [0.473, 0.008]]`` (Eq. 18) and the expected gain is
``o·(Ws) = 0.483`` (Eq. 19); novelty decays as pseudocounts grow.
Right: Algorithm 10.1.1 — a novelty-seeking agent that starts knowing nothing about ``A``
chooses which mappings to probe and learns the true likelihood over trials.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    get_logger,
    novelty_matrix,
    parameter_novelty,
    simulate_learning_agent,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import finalize, panel_grid, save_or_show
from active_inference.visualizations.style import COLORS, annotate_stat_box

LOG = get_logger("ch10.ex4")
A_ORACLE = np.array([[2.5, 0.15], [0.8, 3.0]])
S_ORACLE = np.array([0.15, 0.85])
A_TRUE = np.array([[0.8, 0.2], [0.2, 0.8]])


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()

    W = novelty_matrix(A_ORACLE)
    nov = parameter_novelty(A_ORACLE, S_ORACLE)
    LOG.info("Example 10.4 oracle: W=%s  novelty o·(Ws)=%.3f (book 0.483)",
             np.round(W, 3).tolist(), nov)

    # Novelty as a function of confidence: scale the oracle pseudocounts up.
    scales = np.linspace(1.0, 40.0, 60)
    novelties = [parameter_novelty(A_ORACLE * k, S_ORACLE) for k in scales]

    agent = simulate_learning_agent(A_true=A_TRUE, n_states=2, n_trials=40, steps_per_trial=20)
    LOG.info("learning agent: novelty(trial0)=%.3f  final max|A−A*|=%.4f",
             agent.novelty_first_trial, agent.final_A_error())

    fig, axes = panel_grid(2, title="Fig. 10.4 · parameter information gain (novelty) drives learning",
                           figsize=(12.5, 4.8))

    ax = axes[0]
    total_counts = scales * A_ORACLE.sum()
    ax.plot(total_counts, novelties, color=COLORS["posterior"], lw=2.6, label="expected gain")
    ax.plot(A_ORACLE.sum(), nov, "o", color=COLORS["likelihood"], ms=11,
            markeredgecolor="black", zorder=5, label="Example 10.4")
    annotate_stat_box(ax, f"oracle o·(Ws) = {nov:.3f}\nbook = 0.483\n"
                          "gain ↓ as counts ↑", loc="upper right")
    finalize(ax, xlabel="total pseudocounts Σa", ylabel="parameter novelty",
             title="novelty shrinks as confidence grows")

    ax = axes[1]
    trials = np.arange(agent.A_history.shape[0])
    errs = np.max(np.abs(agent.A_history - A_TRUE), axis=(1, 2))
    ax.plot(trials, errs, color=COLORS["prior"], lw=2.6, label=r"max $|A-A^*|$")
    ax.axhline(0.0, color=COLORS["neutral"], ls="--", lw=1.2)
    annotate_stat_box(ax, f"novelty(trial 0) = {agent.novelty_first_trial:.3f}\n"
                          f"final err = {agent.final_A_error():.3f}", loc="upper right")
    finalize(ax, xlabel="trial", ylabel="likelihood error",
             title="novelty-seeking agent learns A (Alg. 10.1.1)")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_10")
        save_or_show(fig, out / "example_10_4_novelty.png")
        LOG.info("Saved to %s", out / "example_10_4_novelty.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
