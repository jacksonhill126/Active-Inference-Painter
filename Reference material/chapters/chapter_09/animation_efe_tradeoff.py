"""Animation — policy risk/ambiguity trade-off (Chapter 9 §9.6).

Run::

    python chapters/chapter_09/animation_efe_tradeoff.py [--save]

Sweeps the strength of the preference for observation 0. With weak preferences, the
ambiguity term can make the clearer ``explore`` policy competitive; as the preference
sharpens, the reward-seeking risk term dominates and the posterior concentrates on
``exploit``. This is the book's exploration/exploitation mechanism in motion.
"""

from __future__ import annotations

import argparse

import numpy as np

from active_inference import (
    POMDPModel,
    evaluate_policy_components,
    get_logger,
    policy_posterior,
    softmax,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import animate_policy_efe_tradeoff, save_animation

LOG = get_logger("ch9.anim_efe")

POLICY_LABELS = ["exploit", "explore"]
A = np.array([[0.60, 0.15],
              [0.40, 0.85]])
D = np.array([0.5, 0.5])
B = np.array([
    [[0.9, 0.9], [0.1, 0.1]],
    [[0.1, 0.1], [0.9, 0.9]],
])


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--frames", type=int, default=18)
    p.add_argument("--gamma", type=float, default=4.0)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    model = POMDPModel(A=A, D=D, B=B)
    strengths = np.linspace(0.0, 3.0, args.frames)
    risks, ambiguities, posteriors = [], [], []
    for strength in strengths:
        C = softmax(np.array([strength, 0.0]))
        traces = [evaluate_policy_components(model, D, [u], C) for u in range(2)]
        risks.append([tr.risk_total for tr in traces])
        ambiguities.append([tr.ambiguity_total for tr in traces])
        posteriors.append(policy_posterior(np.array([tr.total for tr in traces]),
                                           gamma=args.gamma))
    risks = np.asarray(risks)
    ambiguities = np.asarray(ambiguities)
    posteriors = np.asarray(posteriors)
    frame_labels = [f"pref logit={v:.2f}" for v in strengths]

    anim = animate_policy_efe_tradeoff(
        risks,
        ambiguities,
        posteriors=posteriors,
        frame_labels=frame_labels,
        policy_labels=POLICY_LABELS,
        title="Ch.9 · EFE balances exploration and exploitation",
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_09")
        save_animation(anim, out / "animation_efe_tradeoff.gif", fps=5)
        LOG.info("Saved to %s", out / "animation_efe_tradeoff.gif")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
