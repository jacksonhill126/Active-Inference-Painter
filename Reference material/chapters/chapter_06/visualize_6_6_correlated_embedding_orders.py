"""Visualize Chapter 6 §6.6 correlated embedding-order precision matrices.

Run::

    python chapters/chapter_06/visualize_6_6_correlated_embedding_orders.py [--save]

The generalized precision ``Π̃(γ)`` couples embedding orders through the inverse of
the Gaussian temporal covariance ``S(γ)``. Larger ``γ`` makes higher-order motions
more precise; smaller ``γ`` pushes the model toward white-noise assumptions where
higher derivatives are less informative.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import correlated_embedding_precision, get_logger, save_chapter_data
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_correlated_embedding_precision, save_or_show

LOG = get_logger("ch6.correlation")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the correlated-precision visualization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    """Render the generalized precision matrices and optional raw data sidecar."""
    args = parse_args()
    gammas = np.array([1.0, 1.8, 2.6, 3.4], dtype=float)
    matrices = np.stack([
        correlated_embedding_precision(1.0, args.embedding_dim, gamma=float(gamma))
        for gamma in gammas
    ])
    fig = plot_correlated_embedding_precision(
        matrices,
        gammas,
        title="Fig. 6.6.2 · generalized precision from correlated embedding orders",
    )
    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_06")
        figure = out / "visualize_6_6_correlated_embedding_orders.png"
        save_or_show(fig, figure)
        save_chapter_data(
            6,
            figure.stem,
            {"gammas": gammas, "precision_matrices": matrices},
            {"seed": args.seed, "embedding_dim": args.embedding_dim},
            figures=[figure],
        )
        LOG.info("Saved correlated precision visualization to %s", figure)
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
