"""Chapter 5 · Interactive predictive-coding explorer (GUI / web-launchable).

Run::

    python chapters/chapter_05/interactive_predictive_coding.py    # opens a GUI window

Also launchable from the local web interface (``./run.sh --web`` → Chapter 5 →
"launch interactive"). Opens a matplotlib window with four sliders — observation
``y``, prior mean ``m_x``, prior variance ``s_x²``, and likelihood variance
``σ_y²`` — over the linear model ``g(x)=2x+3``.

The left panel is the free-energy landscape ``F(μ)`` with its closed-form minimum
``μ*`` marked; the right panel shows the two precision-weighted prediction errors
``λ_y ε_y²`` and ``λ_x ε_x²`` trading off. Dragging the sliders slides ``μ*``
between the data-consistent state and the prior mean — the live, hands-on form of
Example 5.2.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from active_inference.visualizations import interactive_predictive_coding


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    """Run the chapter orchestrator and display the interactive figure."""
    parse_args()
    interactive_predictive_coding()
    plt.show()


if __name__ == "__main__":
    main()
