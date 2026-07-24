"""Chapter 1 · Interactive inverse-problem explorer (GUI / web-launchable).

Run::

    python chapters/chapter_01/interactive_inverse_problem.py     # opens a GUI window

Also launchable from the local web interface (``./run.sh --web`` → Chapter 1 →
"launch interactive"). Opens a matplotlib window with two sliders:

* **observation ``y``** — as it grows, the two posterior modes of the
  non-injective generator ``y = 3 + 2 x²`` separate;
* **likelihood variance ``σ_y²``** — shrink it to sharpen the modes, grow it to
  merge them.

The live readout reports both modes, their separation, and the posterior entropy.
This is the hands-on companion to ``04_inverse_problem.py``.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from active_inference.visualizations import interactive_inverse_problem


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    """Run the chapter orchestrator and display the interactive figure."""
    parse_args()
    interactive_inverse_problem()
    plt.show()


if __name__ == "__main__":
    main()
