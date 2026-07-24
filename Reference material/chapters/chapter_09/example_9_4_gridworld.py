"""Example 9.4/9.5 — Grid World planning by expected free energy (Chapter 9, Alg. 9.5.1).

Run::

    python chapters/chapter_09/example_9_4_gridworld.py [--save]

A discrete active-inference agent in a 3×3 Grid World. Its preference ``C`` is a one-hot on
the goal cell. At each step it enumerates length-``H`` policies (action sequences), scores
each by **expected free energy** ``G = risk + ambiguity`` (Eq. 52), forms the policy
posterior ``σ(−γG)`` (Eq. 55), marginalizes it to an action (Eq. 69), and acts — navigating
to the goal in the minimum number of steps. **Planning as inference.**

Left: the grid with the agent's path. Right: the expected free energy of taking each first
action from the start (lower = preferred), showing how EFE drives the choice.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    enumerate_policies,
    evaluate_policy,
    get_logger,
    make_gridworld,
    one_hot,
    simulate_pomdp_agent,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import finalize, panel_grid, save_or_show
from active_inference.visualizations.style import COLORS, annotate_stat_box

LOG = get_logger("ch9.ex4")
ROWS, COLS, GOAL, HORIZON = 3, 3, 8, 4
ACTION_NAMES = ["up", "down", "left", "right"]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--gamma", type=float, default=8.0)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    model = make_gridworld(ROWS, COLS, goal=GOAL)
    res = simulate_pomdp_agent(model, start=0, horizon=HORIZON, gamma=args.gamma, max_steps=12)
    LOG.info("reached goal=%s in %s steps | path=%s | actions=%s",
             res.reached, res.n_steps_to_goal, res.states.tolist(),
             [ACTION_NAMES[a] for a in res.actions])

    # EFE of each first action from the start (best length-H policy starting with that action).
    policies = enumerate_policies(4, HORIZON)
    belief0 = one_hot(0, ROWS * COLS)
    best_efe = []
    for a in range(4):
        gs = [evaluate_policy(model, belief0, pol, np.asarray(model.C))
              for pol in policies if pol[0] == a]
        best_efe.append(min(gs))

    fig, axes = panel_grid(2, title="Fig. 9.4 · Grid World planning by expected free energy",
                           figsize=(11, 4.6))
    # Left: grid + path.
    ax = axes[0]
    grid = np.zeros((ROWS, COLS))
    path = res.states
    for k, cell in enumerate(path):
        grid[divmod(cell, COLS)] = k + 1
    ax.imshow(grid, cmap="Blues", origin="upper")
    for cell in path:
        r, c = divmod(cell, COLS)
        ax.plot(c, r, "o", color=COLORS["posterior"], ms=10)
    gr, gc = divmod(GOAL, COLS)
    ax.plot(gc, gr, "*", color=COLORS["likelihood"], ms=24, label="goal")
    ax.plot(0, 0, "s", color=COLORS["state"], ms=12, label="start")
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    annotate_stat_box(ax, f"reached in {res.n_steps_to_goal} steps\n(Manhattan dist = 4)",
                      loc="lower left", fontsize=10)
    finalize(ax, xlabel="column", ylabel="row", title="agent path (blue = later)", legend=True)

    # Right: EFE per first action.
    ax = axes[1]
    colors = [COLORS["posterior"] if e == min(best_efe) else COLORS["neutral"] for e in best_efe]
    ax.bar(ACTION_NAMES, best_efe, color=colors, alpha=0.85)
    annotate_stat_box(ax, f"chosen: {ACTION_NAMES[int(np.argmin(best_efe))]}",
                      loc="upper right", fontsize=10)
    finalize(ax, xlabel="first action", ylabel="min G(π) over policies",
             title="expected free energy drives the choice", legend=False)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_09")
        save_or_show(fig, out / "example_9_4_gridworld.png")
        LOG.info("Saved to %s", out / "example_9_4_gridworld.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
