"""Chapter 3 · Interactive 2-D LGS food-localization explorer (GUI / web-launchable).

Run::

    python chapters/chapter_03/interactive_lgs_localization.py    # opens a GUI window

Also launchable from the local web interface (``./run.sh --web`` → Chapter 3 →
"launch interactive"). Opens a matplotlib window with two sliders — the
coordinates ``y1`` and ``y2`` of a single noisy observation ``y = (y1, y2)``:

* **y1 / y2** — dragging the observation slides the posterior mean ellipse along
  the precision-weighted line between the fixed prior mean and ``y``, the 2-D
  analogue of the scalar precision-weighting story from Chapters 2 and 5.

The prior ellipse stays fixed as a visual anchor; the posterior ellipse moves
and the live readout reports the posterior mean/std and its distance from both
the prior mean and the dragged observation. Companion to
``example_3_6_lgs_food_localization.py``.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from active_inference.visualizations import interactive_lgs_localization


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    """Run the chapter orchestrator and display the interactive figure."""
    parse_args()
    interactive_lgs_localization()
    plt.show()


if __name__ == "__main__":
    main()
