"""Visualize the factorial likelihood structure of the two-armed bandit (§10.3, Fig 10.3.4).

Run::

    python chapters/chapter_10/visualize_factorial_structure.py [--save]

Renders one heatmap per observation modality of the two-armed bandit's ``A`` set. Each
``A^(m)`` is conditioned on *all* state factors (context × choice), flattened along the
x-axis, showing how every joint state maps to an observation distribution — the multimodal,
multi-factor structure that distinguishes §10.3 from the flat Chapter 9 model.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from active_inference import get_logger, make_two_armed_bandit
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show
from active_inference.visualizations.unified import plot_factorial_likelihood

LOG = get_logger("ch10.viz_factorial")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    model = make_two_armed_bandit()
    LOG.info("A shapes: %s", [a.shape for a in model.A])
    fig = plot_factorial_likelihood(model)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_10")
        save_or_show(fig, out / "factorial_likelihood_structure.png")
        LOG.info("Saved to %s", out / "factorial_likelihood_structure.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
