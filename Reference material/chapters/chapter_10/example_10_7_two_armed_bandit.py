"""Example 10.7 — the two-armed bandit task (Chapter 10, §10.3 factorial depth).

Run::

    python chapters/chapter_10/example_10_7_two_armed_bandit.py [--save] [--explore]

A factorial active-inference agent with two hidden state factors (a fixed **context** — which
machine pays better — and a controllable **choice**) generating three observation modalities
(hint, reward, choice echo). The agent infers the hidden context and exploits the better
machine, trading exploration against exploitation by minimizing factorial expected free
energy (Eq. 38). Reproduces Figs 10.3.6 (risk-averse) / 10.3.7 (less risk-averse).

``--explore`` runs the less risk-averse agent (loss aversion −3 → 0).
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from active_inference import get_logger, make_two_armed_bandit, simulate_two_armed_bandit
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show
from active_inference.visualizations.unified import plot_two_armed_bandit

LOG = get_logger("ch10.ex7")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--explore", action="store_true", help="less risk-averse agent (loss pref 0)")
    p.add_argument("--context", type=int, default=1, choices=[0, 1], help="true better machine")
    p.add_argument("--steps", type=int, default=15)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    reward_prefs = (0.0, 0.0, 4.0) if args.explore else (0.0, -3.0, 4.0)
    model = make_two_armed_bandit(reward_prefs=reward_prefs)
    res = simulate_two_armed_bandit(model, true_context=args.context, n_steps=args.steps, gamma=4.0)
    LOG.info("true context=%s | final belief=%s | wins=%d/%d | hints=%d | choices=%s",
             "right" if args.context else "left",
             [round(x, 3) for x in res.context_belief[-1]], res.n_wins, args.steps,
             res.n_hints, res.choices.tolist())

    kind = "less risk-averse" if args.explore else "risk-averse"
    fig = plot_two_armed_bandit(res, title=f"Fig. 10.3.6 · two-armed bandit ({kind} agent)")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_10")
        name = "example_10_7_bandit_explore.png" if args.explore else "example_10_7_bandit.png"
        save_or_show(fig, out / name)
        LOG.info("Saved to %s", out / name)
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
