"""Animation — policy posterior as precision γ sweeps (Chapter 10 §10.2, Example 10.5).

Run::

    python chapters/chapter_10/animation_precision.py [--save] [--strong]

Animates Figure 10.2.2/10.2.3: as the policy precision ``γ`` ramps from 0 upward, the policy
posterior ``Q(π) = σ(log E − γ G)`` redraws from uniform/habit-shaped toward concentration on
the lowest-EFE policy. With ``--strong`` the agent has a non-uniform habit prior ``E``. Built
on the composable :func:`~active_inference.visualizations.animate_policy_precision`.
"""

from __future__ import annotations

import argparse

import numpy as np

from active_inference import get_logger
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import animate_policy_precision, save_animation

LOG = get_logger("ch10.anim_prec")
G = np.array([3.0, 2.0, 1.5, 2.2, 3.2])
E_STRONG = np.array([0.2, 0.4, 0.1, 0.4, 0.2])


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--strong", action="store_true", help="use a strong (non-uniform) habit E")
    p.add_argument("--gamma-max", type=float, default=3.0)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    gammas = np.linspace(0.0, args.gamma_max, 60)
    E = E_STRONG if args.strong else None
    anim = animate_policy_precision(
        G, gammas, E=E,
        title=f"Ch.10 · policy precision sweep ({'strong' if args.strong else 'uniform'} habits)")
    name = "animation_precision_strong.gif" if args.strong else "animation_precision.gif"

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_10")
        save_animation(anim, out / name)
        LOG.info("Saved to %s", out / name)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
