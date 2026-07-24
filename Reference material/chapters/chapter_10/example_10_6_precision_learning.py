"""Example 10.6 — learning the policy precision γ from data (Chapter 10, §10.2).

Run::

    python chapters/chapter_10/example_10_6_precision_learning.py [--save]

Reproduces the idea of Figure 10.2.4. The policy precision `γ = 1/β` is itself **learned**
from a Gamma prior by descending VFE in `γ` (Eq. 23–25): `β ← β − κ_γ ∂F/∂γ`, `γ = 1/β`.

When the policy-dependent variational free energy `F` (the evidence each policy actually
accrued) is **close** to the expected free energy `G`, the agent's expectations were
confirmed, so confidence in `G` is high (large γ). When `F` is **far** from `G`, confidence
drops (small γ) and the posterior leans on the VFE evidence / habits instead.

Left: γ descent for the two cases. Right: the resulting policy posteriors `σ(log E − F − γG)`.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import get_logger, learn_precision, policy_posterior_full
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import finalize, panel_grid, save_or_show
from active_inference.visualizations.style import COLORS, annotate_stat_box

LOG = get_logger("ch10.ex6")
# Book Example 10.6 (Eq. 29–30).
G = np.array([3.0, 2.0, 1.5, 2.2, 3.2])
E_WEAK = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
F_CLOSE = np.array([3.5, 2.3, 2.0, 2.5, 4.0])
F_FAR = np.array([0.05, 3.2, 5.0, 6.0, 0.003])


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    res_close = learn_precision(G, F_CLOSE, E=E_WEAK)
    res_far = learn_precision(G, F_FAR, E=E_WEAK)
    LOG.info("γ(F close to G)=%.3f  γ(F far from G)=%.3f  (close>far=%s)",
             res_close.gamma, res_far.gamma, res_close.gamma > res_far.gamma)

    q_close = policy_posterior_full(G, F=F_CLOSE, E=E_WEAK, gamma=res_close.gamma)
    q_far = policy_posterior_full(G, F=F_FAR, E=E_WEAK, gamma=res_far.gamma)
    labels = [rf"$\pi^{{({k})}}$" for k in range(len(G))]

    fig, axes = panel_grid(2, title="Fig. 10.2.4 · learning policy precision γ (Example 10.6)",
                           figsize=(12.5, 4.8))

    ax = axes[0]
    ax.plot(res_close.gamma_trace, color=COLORS["posterior"], lw=2.6, label="F close to G")
    ax.plot(res_far.gamma_trace, color=COLORS["likelihood"], lw=2.6, label="F far from G")
    annotate_stat_box(ax, f"γ* close = {res_close.gamma:.3f}\nγ* far   = {res_far.gamma:.3f}\n"
                          "close ⇒ higher confidence", loc="center right", fontsize=10)
    finalize(ax, xlabel="iteration", ylabel=r"precision $\gamma = 1/\beta$",
             title="γ descent (self-consistent fixed point)")

    ax = axes[1]
    x = np.arange(len(G))
    ax.bar(x - 0.2, q_close, width=0.4, color=COLORS["posterior"], label=f"close (γ={res_close.gamma:.2f})")
    ax.bar(x + 0.2, q_far, width=0.4, color=COLORS["likelihood"], label=f"far (γ={res_far.gamma:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    annotate_stat_box(ax, "high γ ⇒ posterior follows G\nlow γ ⇒ leans on F / habits",
                      loc="upper right", fontsize=10)
    finalize(ax, xlabel="policy", ylabel=r"$Q(\pi)$", title="resulting policy posteriors")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_10")
        save_or_show(fig, out / "example_10_6_precision_learning.png")
        LOG.info("Saved to %s", out / "example_10_6_precision_learning.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
