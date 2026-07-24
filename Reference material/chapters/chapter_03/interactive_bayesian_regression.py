"""Chapter 3 · Interactive Bayesian linear regression (GUI / web-launchable).

Run::

    python chapters/chapter_03/interactive_bayesian_regression.py   # opens a GUI window

Also launchable from the local web interface (``./run.sh --web`` → Chapter 3 →
"launch interactive"). Opens a matplotlib window with two sliders:

* **sample size ``N``** — the ±2σ posterior-predictive band tightens as more of the
  fixed observation pool is assimilated (a true sequential-evidence story);
* **prior precision** — a stronger prior regularizes the fit toward the prior mean.

The live readout reports the recovered slope and intercept with their posterior
standard deviations against the truth. Companion to
``example_3_5_bayesian_linear_regression.py``.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from active_inference.visualizations import interactive_bayesian_regression


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    """Run the chapter orchestrator and display the interactive figure."""
    parse_args()
    interactive_bayesian_regression()
    plt.show()


if __name__ == "__main__":
    main()
