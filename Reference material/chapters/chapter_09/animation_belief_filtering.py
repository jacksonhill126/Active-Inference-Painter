"""Animation — dynamic POMDP belief filtering (Chapter 9 §9.2).

Run::

    python chapters/chapter_09/animation_belief_filtering.py [--save] [--seed 0]

Animates the same weather POMDP as ``example_9_2_dynamic_filtering.py``. Each frame shows
the current categorical posterior and the accumulated belief trajectories, making the
``predict with B, update with A`` rhythm of the forward pass visible.
"""

from __future__ import annotations

import argparse

import numpy as np

from active_inference import POMDPModel, forward_filter, get_logger
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import animate_discrete_beliefs, save_animation

LOG = get_logger("ch9.anim_filter")

STATES = ["rainy", "cloudy", "sunny", "snowy"]
OBS = ["water", "hot", "bright"]
A = np.array([[0.80, 0.33, 0.05, 0.40],
              [0.15, 0.33, 0.30, 0.05],
              [0.05, 0.34, 0.65, 0.55]])
D = np.array([0.25, 0.25, 0.25, 0.25])
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


def simulate_observations(seed: int, steps: int) -> np.ndarray:
    """Compute a chapter-local helper quantity for the orchestrated example."""
    rng = np.random.default_rng(seed)
    state = int(rng.choice(len(STATES), p=D))
    obs = np.zeros(steps, dtype=int)
    for t in range(steps):
        obs[t] = int(rng.choice(len(OBS), p=A[:, state]))
        state = int(rng.choice(len(STATES), p=B[:, state]))
    return obs


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    observations = simulate_observations(args.seed, args.steps)
    model = POMDPModel(A=A, D=D, B=B)
    beliefs = forward_filter(model, observations)
    anim = animate_discrete_beliefs(
        beliefs,
        observations=[OBS[i] for i in observations],
        state_labels=STATES,
        title="Ch.9 · dynamic POMDP filtering",
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_09")
        save_animation(anim, out / "animation_belief_filtering.gif", fps=3)
        LOG.info("Saved to %s", out / "animation_belief_filtering.gif")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
