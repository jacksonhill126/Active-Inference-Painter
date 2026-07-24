"""Chapter 4 (bonus) · Interactive variational free energy explorer (GUI / web-launchable).

Run::

    python chapters/chapter_04/interactive_vfe_explorer.py     # opens a GUI window

Also launchable from the local web interface (``./run.sh --web`` → Chapter 4 →
"launch interactive"). Opens a matplotlib window with two sliders that move the
variational density ``q(x) = N(μ_x, σ²_x)``:

* **μ_x** — the mean of the variational belief;
* **σ²_x** — the variance of the variational belief.

The top panel overlays ``q(x)`` on the exact posterior; the bottom panel shows
the live decomposition of variational free energy (free energy, divergence
from the posterior, complexity, accuracy) so you can *feel* how the terms
trade off as you drag the belief around. Free energy bottoms out exactly when
``q`` sits on the posterior — at which point divergence is zero and
``F = −log p(y)``.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from active_inference.visualizations import interactive_variational_free_energy


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    """Run the chapter orchestrator and display the interactive figure."""
    parse_args()
    interactive_variational_free_energy()
    plt.show()


if __name__ == "__main__":
    main()
