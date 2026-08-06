"""Declared support intervals for painting-policy proposals and latents.

Three tables with three DISTINCT roles. They are not interchangeable and are
deliberately not collapsed into one:

``MARK_ACTION_RANGES``
    A representational invariant of :class:`env.StrokeAction`. Whatever the
    proposal draws, a decoded mark must carry coordinates, width, and amount
    inside these bounds; the decoders in :mod:`policies` enforce it.

``PASSAGE_LATENT_RANGES`` (with ``POLYLINE_LATENT_RANGES`` overrides)
    A representational invariant of :class:`policies.PassageLatent`. The
    passage posterior (`passage_inference.PassageBelief`) clips its mean into
    this box after every Kalman update, so a passage latent means the same thing
    wherever it came from.

``PROPOSAL_SUPPORT``
    The SUPPORT OF A PROPOSAL DISTRIBUTION -- exactly the intervals the
    hand-written :class:`policies.PolicySampler` draws from, and exactly the
    intervals the learned :mod:`proposal` density is normalized over. This is
    the only one of the three that is a probabilistic object; the other two are
    invariants of a representation.

Keeping them separate is a modelling statement, not tidiness: a proposal may be
narrower than the representation it writes into (and it is), while the
representation may be narrower than what a decoder will accept. Collapsing them
would silently change which candidates exist. :func:`assert_nested_ranges`
states the containment relations that must hold, and a test calls it, so a fifth
divergent copy of these numbers becomes a test failure rather than silent drift.

This module imports only :mod:`config`, so :mod:`policies`,
:mod:`passage_inference`, :mod:`proposal`, and the tests can all read the same
tables without an import cycle.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from .config import PainterConfig


class Interval(NamedTuple):
    """Closed support interval ``[low, high]`` of one scalar parameter."""

    low: float
    high: float

    @property
    def width(self) -> float:
        return float(self.high - self.low)

    @property
    def log_width(self) -> float:
        """Width in log space, for the log-uniform / log-logit-normal factors."""

        return float(math.log(self.high) - math.log(self.low))

    def contains(self, value: float, *, tolerance: float = 0.0) -> bool:
        return bool(self.low - tolerance <= float(value) <= self.high + tolerance)


# --- Representational invariant: what a decoded StrokeAction may carry -------
# Source of truth for `policies.PolicySampler._stroke_from_center`,
# `_stroke_from_start`, and the polyline branch of `_passage_actions`.
MARK_ACTION_RANGES: dict[str, Interval] = {
    "x": Interval(0.03, 0.97),
    "y": Interval(0.03, 0.97),
    "width": Interval(0.02, 0.34),
    "amount": Interval(0.05, 0.95),
}

# Inward margin used when a mark's full sampled length is shifted to fit on
# canvas, and by `policies.fit_polyline_latent`.
CANVAS_MARGIN: float = 0.03


# --- Representational invariant: what a PassageLatent may carry --------------
# Source of truth for `passage_inference._clip_latent_values`. `direction` is
# taken modulo 2*pi rather than clipped, so it carries the circle's range.
PASSAGE_LATENT_RANGES: dict[str, Interval] = {
    "center_x": Interval(0.03, 0.97),
    "center_y": Interval(0.03, 0.97),
    "direction": Interval(0.0, 2.0 * math.pi),
    "length": Interval(0.04, 1.0),
    "spacing": Interval(0.01, 1.0),
    "width": Interval(0.01, 1.0),
    "amount": Interval(0.01, 1.0),
}

# A polyline latent reinterprets `length` as total path length and `spacing` as
# the SIGNED turn in radians, so those two coordinates carry different bounds.
POLYLINE_LATENT_RANGES: dict[str, Interval] = {
    "length": Interval(0.04, 0.86),
    "spacing": Interval(-1.2, 1.2),
}


# --- Proposal support: exactly what the hand-written sampler draws -----------
# Every entry is the interval a `PolicySampler` draw is uniform (or log-uniform)
# over, and the interval the learned logit-normal factors are normalized over.
PROPOSAL_SUPPORT: dict[str, Interval] = {
    # `_start_point`: the uniform component, and the per-cell clip of the
    # low-coverage component.
    "start_x": Interval(0.05, 0.95),
    "start_y": Interval(0.05, 0.95),
    # `_stroke`: longer sweeps than a dab, log-uniform width, bounded amount.
    "mark_length": Interval(0.20, 0.60),
    "mark_width": Interval(0.03, 0.30),
    "mark_amount": Interval(0.12, 0.75),
    # `_passage_geometry`, band/chain branch.
    "passage_band_length": Interval(0.16, 0.54),
    "passage_band_spacing": Interval(0.045, 0.15),
    "passage_chain_length": Interval(0.16, 0.54),
    "passage_chain_spacing": Interval(0.045, 0.15),
    # `_passage_geometry`, polyline branch. `spacing` is drawn as a MAGNITUDE
    # and then given a fair random sign, so the proposal's support is the union
    # of two intervals and the sign is a declared Bernoulli(0.5) factor.
    "passage_polyline_length": Interval(0.30, 0.72),
    "passage_polyline_spacing": Interval(0.18, 0.85),
    # `_passage_policy`.
    "passage_width": Interval(0.035, 0.24),
    "passage_amount": Interval(0.16, 0.70),
    # `_passage_plan_policy`. The plan family is out of the learned proposal's
    # scope; the entries exist so the plan draws read from the same table.
    "plan_width": Interval(0.035, 0.22),
    "plan_amount": Interval(0.16, 0.68),
}

# Direction is drawn uniformly on the whole circle in every family.
PROPOSAL_DIRECTION: Interval = Interval(0.0, 2.0 * math.pi)

PROPOSAL_FAMILIES: tuple[str, ...] = ("mark", "passage")
PROPOSAL_KINDS: tuple[str, ...] = ("band", "chain", "polyline")


def latent_ranges_for_kind(kind: str) -> dict[str, Interval]:
    """Representational bounds of a `PassageLatent` of this kind."""

    ranges = dict(PASSAGE_LATENT_RANGES)
    if kind == "polyline":
        ranges.update(POLYLINE_LATENT_RANGES)
    return ranges


def proposal_support_for(family: str, kind: str) -> dict[str, Interval]:
    """Support of the proposal over one family/kind's continuous parameters.

    Keys are the parameter names the proposal factorizes over. `spacing` is
    absent for the mark family (a single mark has no passage spacing), and for
    the polyline kind it is the support of the turn MAGNITUDE, with the sign
    carried by a separate declared Bernoulli factor.
    """

    if family == "mark":
        return {
            "center_x": PROPOSAL_SUPPORT["start_x"],
            "center_y": PROPOSAL_SUPPORT["start_y"],
            "direction": PROPOSAL_DIRECTION,
            "length": PROPOSAL_SUPPORT["mark_length"],
            "width": PROPOSAL_SUPPORT["mark_width"],
            "amount": PROPOSAL_SUPPORT["mark_amount"],
        }
    if family != "passage":
        raise ValueError(f"Unknown proposal family {family!r}; expected one of {PROPOSAL_FAMILIES}.")
    if kind not in PROPOSAL_KINDS:
        raise ValueError(f"Unknown passage kind {kind!r}; expected one of {PROPOSAL_KINDS}.")
    return {
        "center_x": PROPOSAL_SUPPORT["start_x"],
        "center_y": PROPOSAL_SUPPORT["start_y"],
        "direction": PROPOSAL_DIRECTION,
        "length": PROPOSAL_SUPPORT[f"passage_{kind}_length"],
        "spacing": PROPOSAL_SUPPORT[f"passage_{kind}_spacing"],
        "width": PROPOSAL_SUPPORT["passage_width"],
        "amount": PROPOSAL_SUPPORT["passage_amount"],
    }


def passage_stroke_count_range(config: PainterConfig) -> Interval:
    """Inclusive integer range `_passage_policy` draws `stroke_count` from.

    Duplicated nowhere: `policies.PolicySampler._passage_policy` reads this, and
    so does the learned proposal's masked categorical, which is what keeps a
    learned sample from raising inside `Policy.__post_init__`.
    """

    max_strokes = min(max(1, config.planning_horizon), max(1, config.passage_max_strokes))
    min_strokes = min(max_strokes, max(2, config.passage_min_strokes))
    return Interval(float(min_strokes), float(max_strokes))


def clip_to(value: float, interval: Interval) -> float:
    return float(min(max(float(value), interval.low), interval.high))


def assert_nested_ranges() -> None:
    """State the containment relations the three tables must satisfy.

    Called by `tests/test_proposal.py`, not at import: a failure here is a
    modelling inconsistency to be read by a human, not something to hide behind
    a module-load traceback.
    """

    # A proposed mark start point and a proposed passage centre must be a legal
    # coordinate for the representation they are written into.
    for axis in ("x", "y"):
        support = PROPOSAL_SUPPORT[f"start_{axis}"]
        action = MARK_ACTION_RANGES[axis]
        latent = PASSAGE_LATENT_RANGES[f"center_{axis}"]
        if not (action.low <= support.low and support.high <= action.high):
            raise AssertionError(f"start_{axis} proposal support escapes MARK_ACTION_RANGES")
        if not (latent.low <= support.low and support.high <= latent.high):
            raise AssertionError(f"start_{axis} proposal support escapes PASSAGE_LATENT_RANGES")

    mark = proposal_support_for("mark", "mark")
    if not (
        MARK_ACTION_RANGES["width"].low <= mark["width"].low
        and mark["width"].high <= MARK_ACTION_RANGES["width"].high
    ):
        raise AssertionError("mark width proposal support escapes MARK_ACTION_RANGES")
    if not (
        MARK_ACTION_RANGES["amount"].low <= mark["amount"].low
        and mark["amount"].high <= MARK_ACTION_RANGES["amount"].high
    ):
        raise AssertionError("mark amount proposal support escapes MARK_ACTION_RANGES")

    for kind in PROPOSAL_KINDS:
        support = proposal_support_for("passage", kind)
        latent = latent_ranges_for_kind(kind)
        for name in ("length", "width", "amount"):
            bound = latent[name]
            if not (bound.low <= support[name].low and support[name].high <= bound.high):
                raise AssertionError(f"passage {kind} {name} proposal support escapes the latent range")
        spacing = latent["spacing"]
        proposed = support["spacing"]
        if kind == "polyline":
            # Signed: the proposal's support is [-high, -low] U [low, high].
            if not (spacing.low <= -proposed.high and proposed.high <= spacing.high):
                raise AssertionError("polyline turn proposal support escapes the latent range")
        elif not (spacing.low <= proposed.low and proposed.high <= spacing.high):
            raise AssertionError(f"passage {kind} spacing proposal support escapes the latent range")

    # A passage's decoded marks are clipped into MARK_ACTION_RANGES, so the
    # passage width/amount proposal must also be legal there.
    passage = proposal_support_for("passage", "band")
    if not (
        MARK_ACTION_RANGES["width"].low <= passage["width"].low
        and passage["width"].high <= MARK_ACTION_RANGES["width"].high
    ):
        raise AssertionError("passage width proposal support escapes MARK_ACTION_RANGES")
    if not (
        MARK_ACTION_RANGES["amount"].low <= passage["amount"].low
        and passage["amount"].high <= MARK_ACTION_RANGES["amount"].high
    ):
        raise AssertionError("passage amount proposal support escapes MARK_ACTION_RANGES")
