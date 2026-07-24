"""Visualize Chapter 8 hierarchical forward/backward message passing."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import ContinuousHierarchyLayer, HierarchicalContinuousModel, get_logger
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_hierarchical_message_passing, save_or_show

LOG = get_logger("ch8.messages")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    model = HierarchicalContinuousModel(
        lower=ContinuousHierarchyLayer(obs_offset=3.0, sensory_precision=20.0),
        link_precision=2.0,
        context_prior_mean=5.0,
        context_precision=0.5,
    )
    fig = plot_hierarchical_message_passing(model, y=2.0, belief=np.array([4.0, 2.0]))
    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_08")
        save_or_show(fig, out / "visualize_message_passing.png")
        LOG.info("Saved to %s", out / "visualize_message_passing.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
