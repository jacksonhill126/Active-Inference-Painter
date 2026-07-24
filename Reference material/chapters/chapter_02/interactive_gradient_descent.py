"""Chapter 2 · Interactive gradient-descent trajectory scrubber (GUI / web-launchable).

Run::

    python chapters/chapter_02/interactive_gradient_descent.py     # opens a GUI window

Also launchable from the local web interface (``./run.sh --web`` → Chapter 2 →
"launch interactive"). Opens a matplotlib window with two sliders over the MLE
loss surface for a fixed pool of observations:

* **log10(learning rate)** — recomputes the whole trajectory from the same
  fixed start point ``x0``. Drag it past the stability threshold
  ``2 / (n · β₁² / σ_y²)`` and the iterate overshoots and diverges instead of
  descending;
* **iteration** — scrubs through the *current* trajectory, redrawing the
  iterate on the loss surface and the loss-vs-iteration trace up to that step.

The live readout reports the current iterate, its loss, the last step size,
and a converging/diverging heuristic. Companion to
``example_2_10_gradient_descent.py`` and ``animation_gradient_descent.py``.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from active_inference.visualizations import interactive_gradient_descent


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    """Run the chapter orchestrator and display the interactive figure."""
    parse_args()
    interactive_gradient_descent()
    plt.show()


if __name__ == "__main__":
    main()
