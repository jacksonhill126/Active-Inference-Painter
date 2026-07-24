"""Shared visual style for figures and animations.

This module collects the small style decisions that were previously
duplicated across ``plotting.py``, ``diagnostics.py``, and
``animations.py``: stat-box bbox kwargs, the brand color palette, the
default ellipse drawing settings, and the global font / line / grid
settings used across every chapter orchestrator. Centralising them
prevents drift and makes it easy to re-skin the whole package.

The default font sizes are intentionally larger than matplotlib's
defaults — chapter figures are intended to be projected, embedded in
slides, and skim-readable in print, not just rendered at native
resolution on a laptop. Override individual entries via
``set_default_style(...)`` if you need tighter typography.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Mapping, Optional

import matplotlib as mpl

# Palette — the Okabe-Ito colourblind-safe palette (Okabe & Ito 2008), chosen so
# every pair of roles stays distinguishable under deuteranopia, protanopia and
# tritanopia (the red+green ``tab10`` defaults are the classic accessibility
# failure). Hues are also separated in luminance so the figures survive greyscale
# printing. Keys are stable; only the hex values changed.
COLORS: Mapping[str, str] = {
    "prior":      "#0072B2",  # blue
    "likelihood": "#D55E00",  # vermillion (distinct from green for red-green CVD)
    "posterior":  "#009E73",  # bluish green
    "data":       "#000000",  # black
    "truth":      "#CC79A7",  # reddish purple
    "neutral":    "#6E6E6E",  # mid grey (darker than before for contrast)
    "state":      "#E69F00",  # orange  — state prediction error ε_x (was inline)
    "sensory":    "#56B4E9",  # sky blue — sensory prediction error ε_y (was inline)
}


def stat_box_bbox(
    *,
    facecolor: str = "white",
    edgecolor: str = "black",
    pad: float = 0.25,
    alpha: float = 0.85,
) -> dict:
    """The bbox style used for all in-figure stat-readout text boxes.

    Returns a dict suitable for `matplotlib.text.Text(bbox=...)`. Keep
    every stat-box readout in the package on this single style so they
    look like siblings.
    """
    return dict(
        boxstyle=f"round,pad={pad}",
        fc=facecolor,
        ec=edgecolor,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Global defaults — applied at import time so every figure picks them up.
# ---------------------------------------------------------------------------


DEFAULT_RC: Mapping[str, object] = {
    "font.size":         15,
    "axes.titlesize":    18,
    "axes.titleweight":  "bold",
    "axes.labelsize":    16,
    "axes.labelweight":  "bold",
    "xtick.labelsize":   14,
    "ytick.labelsize":   14,
    "legend.fontsize":   13,
    "legend.framealpha": 0.95,
    "legend.edgecolor":  "0.55",
    "legend.borderpad":  0.6,
    "figure.titlesize":  20,
    "figure.titleweight": "bold",
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "axes.linewidth":    1.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   2.6,
    "lines.markersize":  8.0,
    "savefig.dpi":       170,
    "savefig.bbox":      "tight",
    "figure.constrained_layout.use": True,
}


def set_default_style(overrides: Optional[Mapping[str, object]] = None) -> None:
    """Apply the package's default rcParams (large fonts, grid on, etc.).

    Idempotent. Pass ``overrides`` to tweak specific entries — useful when
    a script wants oversized fonts for a slide deck or smaller fonts for
    print embedding.
    """
    rc = dict(DEFAULT_RC)
    if overrides:
        rc.update(overrides)
    mpl.rcParams.update(rc)


@contextmanager
def figure_style(overrides: Optional[Mapping[str, object]] = None):
    """Temporarily apply package defaults inside a ``with`` block.

    Restores the previous rcParams on exit so chapter scripts can opt
    into different styles per figure without leaking state.
    """
    saved = mpl.rcParams.copy()
    try:
        set_default_style(overrides)
        yield
    finally:
        mpl.rcParams.update(saved)


# Apply once at import. Chapter scripts can re-call set_default_style
# with overrides; this gives every figure a sensible baseline.
set_default_style()


def annotate_stat_box(
    ax,
    text: str,
    *,
    loc: str = "upper left",
    fontsize: int = 12,
    monospace: bool = True,
    **bbox_overrides,
):
    """Place a stat-box at a named corner of the axis.

    ``loc`` accepts the same names as `Axes.legend(loc=...)` (only the
    four corners and "center" — anything else falls back to top-left).
    ``monospace`` renders the readout in a fixed-width font so columns of
    numbers line up — the default for statistics panels.
    """
    corners = {
        "upper left":  (0.025, 0.97, "left",  "top"),
        "upper right": (0.975, 0.97, "right", "top"),
        "lower left":  (0.025, 0.03, "left",  "bottom"),
        "lower right": (0.975, 0.03, "right", "bottom"),
        "center":      (0.50, 0.50, "center", "center"),
    }
    x, y, ha, va = corners.get(loc, corners["upper left"])
    bbox = stat_box_bbox(**bbox_overrides)
    kw = dict(family="monospace") if monospace else {}
    return ax.text(
        x, y, text,
        transform=ax.transAxes,
        fontsize=fontsize, ha=ha, va=va,
        bbox=bbox, **kw,
    )


def annotate_point(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    color: str = "#000000",
    dx: float = 0.0,
    dy: float = 0.0,
    marker: str = "o",
    ms: float = 9.0,
    fontsize: int = 12,
    arrow: bool = True,
):
    """Mark a single ``(x, y)`` point and label it with an arrow callout.

    Used to pin **analytical** landmarks on a figure — a fixed point, an
    oracle value, a closed-form minimum — so the reader sees the exact
    coordinate, not just the curve. ``dx``/``dy`` offset the text label (in
    data coordinates) from the marker; set ``arrow=False`` for a bare label.
    """
    ax.plot([x], [y], marker=marker, color=color, ms=ms, zorder=6,
            markeredgecolor="white", markeredgewidth=1.2)
    ann_kw = dict(fontsize=fontsize, color=color, fontweight="bold",
                  ha="left", va="center", zorder=7)
    if arrow:
        ax.annotate(
            text, xy=(x, y), xytext=(x + dx, y + dy),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.4),
            **ann_kw,
        )
    else:
        ax.text(x + dx, y + dy, text, **ann_kw)
