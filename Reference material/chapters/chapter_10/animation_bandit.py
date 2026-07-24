"""Animation — the two-armed bandit agent over time (Chapter 10 §10.3, Figs 10.3.6/7).

Run::

    python chapters/chapter_10/animation_bandit.py [--save] [--explore]

Animates a two-armed bandit run: the context belief converging on the truly-better machine
and the policy posterior over the four choice actions evolving step-by-step — the
explore-then-exploit trajectory. Built on the composable
:func:`~active_inference.visualizations.animate_two_armed_bandit`.
"""

from __future__ import annotations

import argparse

from active_inference import get_logger, make_two_armed_bandit, simulate_two_armed_bandit
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import animate_two_armed_bandit, save_animation

LOG = get_logger("ch10.anim_bandit")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--explore", action="store_true", help="less risk-averse agent")
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    reward_prefs = (0.0, 0.0, 4.0) if args.explore else (0.0, -3.0, 4.0)
    model = make_two_armed_bandit(reward_prefs=reward_prefs)
    res = simulate_two_armed_bandit(model, true_context=1, n_steps=15, gamma=4.0)
    anim = animate_two_armed_bandit(res)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_10")
        name = "animation_bandit_explore.gif" if args.explore else "animation_bandit.gif"
        save_animation(anim, out / name)
        LOG.info("Saved to %s", out / name)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
