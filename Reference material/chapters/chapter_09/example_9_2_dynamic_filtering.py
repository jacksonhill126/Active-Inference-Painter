"""Example 9.2 — dynamic POMDP forward filtering (Chapter 9 §9.2).

Run::

    python chapters/chapter_09/example_9_2_dynamic_filtering.py [--save] [--seed 0]

Chapter 9 starts with static inference from a single categorical observation. Section 9.2
adds dynamics: yesterday's posterior is pushed through the transition matrix ``B`` to become
today's prior, then the new observation is assimilated with the same ``A``-matrix inversion.
The figure shows the filtered belief sequence and the per-step discrete VFE evaluated at the
posterior from §9.3.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import POMDPModel, discrete_vfe, forward_filter, get_logger
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_discrete_belief_sequence, save_or_show

LOG = get_logger("ch9.ex2")

STATES = ["rainy", "cloudy", "sunny", "snowy"]
OBS = ["water", "hot", "bright"]
A = np.array([[0.80, 0.33, 0.05, 0.40],
              [0.15, 0.33, 0.30, 0.05],
              [0.05, 0.34, 0.65, 0.55]])
D = np.array([0.25, 0.25, 0.25, 0.25])
# Columns are current states; rows are next states. The diagonal dominance makes weather
# persistent, while off-diagonal mass lets observations revise a stale prior.
B = np.array([[0.82, 0.10, 0.03, 0.05],
              [0.12, 0.78, 0.12, 0.15],
              [0.03, 0.09, 0.80, 0.05],
              [0.03, 0.03, 0.05, 0.75]])


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=10)
    return p.parse_args()


def simulate_observations(seed: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a small reproducible hidden weather path and observation sequence."""
    rng = np.random.default_rng(seed)
    states = np.zeros(steps, dtype=int)
    obs = np.zeros(steps, dtype=int)
    states[0] = int(rng.choice(len(STATES), p=D))
    for t in range(steps):
        obs[t] = int(rng.choice(len(OBS), p=A[:, states[t]]))
        if t + 1 < steps:
            states[t + 1] = int(rng.choice(len(STATES), p=B[:, states[t]]))
    return states, obs


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    true_states, observations = simulate_observations(args.seed, args.steps)
    model = POMDPModel(A=A, D=D, B=B)
    beliefs = forward_filter(model, observations)
    free_energies = []
    for t, obs in enumerate(observations):
        prior = D if t == 0 else B @ beliefs[t - 1]
        free_energies.append(discrete_vfe(model, beliefs[t], int(obs), prior))
    free_energies = np.asarray(free_energies)

    LOG.info("observations=%s", [OBS[i] for i in observations])
    LOG.info("true states=%s", [STATES[i] for i in true_states])
    LOG.info("final belief=%s", dict(zip(STATES, np.round(beliefs[-1], 3))))

    fig = plot_discrete_belief_sequence(
        beliefs,
        observations=[OBS[i] for i in observations],
        state_labels=STATES,
        free_energies=free_energies,
        title="Fig. 9.2/9.3 · dynamic POMDP filtering and discrete VFE",
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_09")
        save_or_show(fig, out / "example_9_2_dynamic_filtering.png")
        LOG.info("Saved to %s", out / "example_9_2_dynamic_filtering.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
