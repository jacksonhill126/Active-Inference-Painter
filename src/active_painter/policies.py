from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from .config import PainterConfig
from .env import StrokeAction
from .policy_ranges import (
    CANVAS_MARGIN,
    MARK_ACTION_RANGES,
    PROPOSAL_SUPPORT,
    latent_ranges_for_kind,
    passage_stroke_count_range,
)
from .precision_beliefs import GapIncrementBelief
from .proposal import ProposalLatent, ProposalRecord

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .proposal import PolicyProposalNetwork


# Conditional band/chain split inside the hand-written passage proposal, once the
# polyline branch has been declined. Named because `proposal.hand_written_log_density`
# must read the same numbers the sampler draws with, or the declared divergence
# between the two proposals would be measured against an idealization of the code
# rather than the code.
PASSAGE_BAND_PROBABILITY: float = 0.65
PASSAGE_PLAN_BAND_PROBABILITY: float = 0.55


@dataclass(frozen=True, slots=True)
class BrushPreparationPolicy:
    """Instantaneous tool-preparation policy below a selected mark policy."""

    kind: str
    selected_tone: float

    def __post_init__(self) -> None:
        if self.kind not in {"preserve", "reload"}:
            raise ValueError("Brush preparation kind must be preserve or reload.")
        if self.selected_tone not in {0.0, 1.0}:
            raise ValueError("selected_tone must be binary white/black.")


@dataclass(frozen=True, slots=True)
class MotorPrimitiveLatent:
    """Embodied realization latent for the first non-stop mark in a policy.

    This is a policy prior/transition-likelihood latent, not a reward. It says
    how the body proposes to realize the selected mark so the generative model
    can predict both canvas and proprioceptive outcomes before posterior policy
    selection.
    """

    kind: str
    scope: str = "first_stroke"
    pivot_joint: str = ""
    description: str = ""
    roll_start_deg: float = 0.0
    roll_end_deg: float = 0.0


@dataclass(frozen=True, slots=True)
class _MarkDraw:
    """One mark's proposal parameters before the deterministic decoder runs.

    These are exactly the coordinates the mark proposal is a distribution OVER,
    which is why the anchor is the START point rather than the mark centre: the
    hand-written proposal draws a start point, so a learned proposal must be
    scored on the same coordinate for the two densities to be comparable and for
    the shared decoder to cancel out of their ratio.
    """

    x0: float
    y0: float
    angle: float
    length: float
    width: float
    amount: float
    tone: float


@dataclass(frozen=True, slots=True)
class PassageLatent:
    """Higher-level latent policy prior over a related sequence of marks.

    This is not an outcome preference or reward. It is a transition prior over
    mark trajectories: a latent passage generates several strokes that share a
    coarse region, direction, scale, tone, and amount. For `polyline`, length is
    total path length and spacing is the signed turn in radians between
    connected straight segments. EFE still evaluates every predicted segment
    consequence and the completed policy, including terminal stop.
    """

    kind: str
    center_x: float
    center_y: float
    direction: float
    length: float
    spacing: float
    stroke_count: int
    width: float
    amount: float
    tone: float

    @property
    def turn(self) -> float:
        return float(self.spacing) if self.kind == "polyline" else 0.0


def _polyline_relative_vertices(latent: PassageLatent) -> np.ndarray:
    bounds = latent_ranges_for_kind("polyline")
    segment_count = max(2, int(latent.stroke_count))
    total_length = float(np.clip(latent.length, *bounds["length"]))
    turn = float(np.clip(latent.spacing, *bounds["spacing"]))
    center_index = 0.5 * (segment_count - 1)
    angles = latent.direction + (np.arange(segment_count, dtype=np.float64) - center_index) * turn
    segment_length = total_length / segment_count
    vertices = np.zeros((segment_count + 1, 2), dtype=np.float64)
    for index, angle in enumerate(angles):
        vertices[index + 1] = vertices[index] + segment_length * np.asarray(
            [np.cos(angle), np.sin(angle)], dtype=np.float64
        )
    segment_centers = 0.5 * (vertices[:-1] + vertices[1:])
    return vertices - segment_centers.mean(axis=0, keepdims=True)


def fit_polyline_latent(latent: PassageLatent, margin: float = CANVAS_MARGIN) -> PassageLatent:
    """Shift a polyline center inward while preserving all segment geometry."""

    if latent.kind != "polyline":
        return latent
    relative = _polyline_relative_vertices(latent)
    lower = float(margin) - relative.min(axis=0)
    upper = 1.0 - float(margin) - relative.max(axis=0)
    requested = np.asarray([latent.center_x, latent.center_y], dtype=np.float64)
    center = np.clip(requested, lower, upper)
    return replace(latent, center_x=float(center[0]), center_y=float(center[1]))


def polyline_vertices(latent: PassageLatent, margin: float = CANVAS_MARGIN) -> np.ndarray:
    """Decode one low-dimensional passage latent into connected vertices."""

    if latent.kind != "polyline":
        raise ValueError("Polyline vertices require a polyline passage latent.")
    fitted = fit_polyline_latent(latent, margin)
    center = np.asarray([fitted.center_x, fitted.center_y], dtype=np.float64)
    return _polyline_relative_vertices(fitted) + center


@dataclass(frozen=True, slots=True)
class PassagePlanLatent:
    """Slower latent policy prior over multiple related passages.

    The plan is not a preference. It samples a sequence of passage latents
    whose centers and directions evolve slowly, then EFE scores the resulting
    multi-mark terminal consequences.
    """

    kind: str
    center_x: float
    center_y: float
    direction: float
    passage_count: int
    total_stroke_count: int
    passage_spacing: float
    turn: float
    width: float
    amount: float
    tone: float
    passages: tuple[PassageLatent, ...]


@dataclass(frozen=True, slots=True)
class Policy:
    actions: tuple[StrokeAction, ...]
    passage: PassageLatent | None = None
    passage_plan: PassagePlanLatent | None = None
    motor_primitive: MotorPrimitiveLatent | None = None
    brush_preparation: BrushPreparationPolicy | None = None
    # Local receding-horizon inference can begin partway through a persistent
    # passage. Zero denotes a globally proposed complete passage.
    passage_start_index: int = 0

    def __post_init__(self) -> None:
        if not self.actions or not self.actions[-1].stop:
            raise ValueError("Every painting policy must terminate in stop.")
        if any(action.stop for action in self.actions[:-1]):
            raise ValueError("Stop may appear only as the final painting policy action.")
        if self.passage is not None and self.passage_plan is not None:
            raise ValueError("A policy may carry either passage or passage-plan metadata, not both.")
        if self.passage_start_index < 0:
            raise ValueError("Passage start index must be non-negative.")
        if self.passage is None and self.passage_start_index != 0:
            raise ValueError("Passage start index requires passage metadata.")
        if self.passage_plan is not None and self.passage_start_index != 0:
            raise ValueError("Passage-plan policies must begin at their first passage.")
        if self.passage is not None and len(self.actions) < 3 and self.passage_start_index == 0:
            raise ValueError("A passage policy must contain multiple marks before stop.")
        if self.passage is not None:
            mark_count = len(self.actions) - 1
            remaining = self.passage.stroke_count - self.passage_start_index
            if remaining < mark_count:
                raise ValueError("Passage metadata does not contain all policy marks.")
        if self.motor_primitive is not None and self.actions[0].stop:
            raise ValueError("A motor realization latent requires a non-stop first action.")
        if self.brush_preparation is not None:
            if self.actions[0].stop:
                raise ValueError("A brush-preparation policy requires a non-stop first action.")
            selected_tone = float(self.actions[0].tone >= 0.5)
            if self.brush_preparation.selected_tone != selected_tone:
                raise ValueError("Brush preparation must select the first action's dedicated color brush.")
        if self.passage_plan is not None:
            if self.passage_plan.passage_count < 2:
                raise ValueError("A passage-plan policy must contain multiple passages.")
            if self.passage_plan.total_stroke_count != len(self.actions) - 1:
                raise ValueError("A passage-plan latent must match the policy mark count.")

    @property
    def passage_boundaries(self) -> tuple[tuple[int, int], ...]:
        """Half-open mark-index ranges generated by each passage latent."""

        if self.passage is not None:
            return ((0, len(self.actions) - 1),)
        if self.passage_plan is None:
            return ()
        boundaries: list[tuple[int, int]] = []
        start = 0
        for passage in self.passage_plan.passages:
            end = start + passage.stroke_count
            boundaries.append((start, end))
            start = end
        return tuple(boundaries)


def policy_posterior_from_efe(
    g: torch.Tensor,
    log_prior: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Single source of truth for the repo's policy posterior Q(pi).

    ``Q(pi) = softmax(log p(pi) - gamma * (G - min G))``. The shift by
    ``min G`` is exactly cancelled by the softmax normalizer, so it is a
    numerical-stability rewrite of the canonical ``softmax(log E - gamma G)``
    and not a reparameterization. ``gamma`` may be a declared constant or the
    posterior mean of a Gamma precision belief; this function must not be able
    to tell the difference.
    """

    return torch.softmax(-float(gamma) * (g - g.min()) + log_prior, dim=0)


def policy_stop_log_prior(
    policy: Policy,
    believed_coverage: float,
    config: PainterConfig,
    gap_progress: GapIncrementBelief | None = None,
) -> float:
    """Declared policy prior term log p(pi) for premature termination.

    The prior over immediate termination is now a PRODUCT of two declared prior
    factors, hence a sum of two log-sigmoids. Both are bounded above by zero, so
    neither can manufacture positive value for any candidate:

    1. A sigmoid in believed material coverage centered at
       `minimum_stop_coverage`. Unchanged.
    2. A gap-progress factor
       ``logsigmoid(-sharpness * E[dGap] / sd[dGap])`` over the believed
       compression-gap increment per completed mark. As the believed increment
       approaches zero (further marks are no longer buying structure) the factor
       rises toward zero and stopping becomes a priori less unlikely. It is
       EXACTLY 0.0 when no belief is supplied, when the declared flag is off, or
       before the belief has any observations, so the coverage factor's
       behaviour -- including ``log(0.5)`` at the midpoint -- is preserved
       verbatim.

    Continuation policies carry a flat prior of zero log-weight; normalization
    over the sampled candidate set is absorbed by the policy softmax. Stopping
    stays admissible at every coverage, merely a priori unlikely before the
    midpoint. The believed gap increment is a transition-prior belief and NEVER
    enters expected free energy: this is its only consumer.
    """

    if not policy.actions[0].stop:
        return 0.0
    logit = config.stop_prior_sharpness * (float(believed_coverage) - config.minimum_stop_coverage)
    if logit >= 0.0:
        coverage_term = -float(np.log1p(np.exp(-logit)))
    else:
        coverage_term = float(logit - np.log1p(np.exp(logit)))
    if gap_progress is None:
        return coverage_term
    return coverage_term + gap_progress.stop_log_prior_term(config)


class PolicySampler:
    """Candidate-policy proposal distribution.

    A PROPOSAL, NOT A PRIOR. Every mixture in here -- the low-coverage
    start-point oversampling, the passage and plan fractions, the polyline
    fraction, the planning-depth categorical -- decides which candidates EXIST to
    be scored. None of them is a normalized `p(pi)` factor and none appears in the
    policy posterior; selection runs the identical
    `softmax(log p_stop + painting_log_evidence)` over whatever set is produced.
    Earlier comments here called the low-coverage mixture an "empirical policy
    prior"; it never was one, and spec §10.2 records the correction.

    A learned proposal (`proposal.PolicyProposalNetwork`) may generate a declared
    fraction `config.learned_proposal_mix` of the candidates, but only when the
    caller supplies belief features -- i.e. only from the spatial planner, which
    is the only path that has a canvas/relational posterior to condition on. At
    `mix = 0.0`, or with no features, the learned branches take ZERO draws from
    `self.rng` and this class reproduces its previous candidate stream
    byte-identically. The hand-written branch always generates the remaining
    `(1 - mix)` in the SAME planning round under the SAME belief, so it is a
    paired control rather than a historical one.

    `_stroke` DELIBERATELY has no learned branch. `bootstrap_dynamics` and
    `demo.pretrain` call it directly for motor babbling, whose purpose is to cover
    the transition model's input space rather than to paint well; sampling that
    from a partially-trained proposal would narrow the very coverage it exists to
    provide.
    """

    def __init__(
        self,
        config: PainterConfig,
        seed: int = 0,
        learned_proposal: "PolicyProposalNetwork | None" = None,
    ) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        # Assigned after construction by `SpatialActiveInferencePainter`, so the
        # construction order of every other module (and therefore the global torch
        # RNG stream) stays exactly as it was.
        self.learned_proposal = learned_proposal
        # Aligned index-for-index with the list `sample` last returned. `None`
        # marks a candidate no modelled proposal is responsible for: the immediate
        # stop policy, and passage-plan compounds, which stay hand-written.
        self.last_proposal_records: tuple[ProposalRecord | None, ...] = ()
        self.last_learned_candidate_count = 0

    def _start_point(self, coverage_field: np.ndarray | None) -> tuple[float, float]:
        if (
            coverage_field is not None
            and coverage_field.ndim == 2
            and coverage_field.size > 0
            and self.rng.uniform() < self.cfg.proposal_low_coverage_mix
        ):
            weights = np.clip(1.0 - coverage_field.astype(np.float64), 0.0, 1.0) + 1e-3
            probabilities = (weights / weights.sum()).ravel()
            index = int(self.rng.choice(coverage_field.size, p=probabilities))
            rows, cols = coverage_field.shape
            row, col = divmod(index, cols)
            x0 = float(np.clip((col + self.rng.uniform()) / cols, *PROPOSAL_SUPPORT["start_x"]))
            y0 = float(np.clip((row + self.rng.uniform()) / rows, *PROPOSAL_SUPPORT["start_y"]))
            return x0, y0
        x0, y0 = self.rng.uniform(PROPOSAL_SUPPORT["start_x"].low, PROPOSAL_SUPPORT["start_x"].high, size=2)
        return float(x0), float(y0)

    def _tone(self) -> float:
        return float(self.rng.integers(0, 2)) if self.cfg.stroke_tone_prior is None else float(self.cfg.stroke_tone_prior)

    def _tone_support(self) -> tuple[float, ...]:
        if self.cfg.stroke_tone_prior is None:
            return (0.0, 1.0)
        return (float(self.cfg.stroke_tone_prior),)

    @staticmethod
    def _action_with_tone(action: StrokeAction, tone: float) -> StrokeAction:
        if action.stop:
            return action
        return replace(action, tone=float(tone))

    def _passage_kind(self, *, band_probability: float) -> str:
        if self.rng.uniform() < np.clip(self.cfg.passage_polyline_mix, 0.0, 1.0):
            return "polyline"
        return "band" if self.rng.uniform() < band_probability else "chain"

    def _passage_geometry(self, kind: str) -> tuple[float, float]:
        if kind == "polyline":
            total_length = float(self.rng.uniform(*PROPOSAL_SUPPORT["passage_polyline_length"]))
            turn_magnitude = float(self.rng.uniform(*PROPOSAL_SUPPORT["passage_polyline_spacing"]))
            turn = turn_magnitude if self.rng.uniform() < 0.5 else -turn_magnitude
            return total_length, turn
        return (
            float(self.rng.uniform(*PROPOSAL_SUPPORT[f"passage_{kind}_length"])),
            float(self.rng.uniform(*PROPOSAL_SUPPORT[f"passage_{kind}_spacing"])),
        )

    def _policy_tone_alternatives(self, policy: Policy) -> list[Policy]:
        alternatives: list[Policy] = []
        for tone in self._tone_support():
            passage = replace(policy.passage, tone=tone) if policy.passage is not None else None
            passage_plan = self._passage_plan_with_tone(policy.passage_plan, tone)
            actions = tuple(self._action_with_tone(action, tone) for action in policy.actions)
            brush_preparation = (
                replace(policy.brush_preparation, selected_tone=float(tone))
                if policy.brush_preparation is not None
                else None
            )
            alternatives.append(
                Policy(
                    actions,
                    passage=passage,
                    passage_plan=passage_plan,
                    motor_primitive=policy.motor_primitive,
                    brush_preparation=brush_preparation,
                    passage_start_index=policy.passage_start_index,
                )
            )
        return alternatives

    @staticmethod
    def _passage_plan_with_tone(
        passage_plan: PassagePlanLatent | None,
        tone: float,
    ) -> PassagePlanLatent | None:
        if passage_plan is None:
            return None
        passages = tuple(replace(passage, tone=float(tone)) for passage in passage_plan.passages)
        return replace(passage_plan, tone=float(tone), passages=passages)

    def _stroke_from_center(
        self,
        x: float,
        y: float,
        angle: float,
        length: float,
        width: float,
        amount: float,
        tone: float,
    ) -> StrokeAction:
        dx = 0.5 * length * np.cos(angle)
        dy = 0.5 * length * np.sin(angle)
        # Shift the center inward so the full sampled length fits: clipping the
        # endpoints instead collapses edge strokes into dwell-dabs, which paint
        # as solid discs no real brush could make.
        x_bounds = MARK_ACTION_RANGES["x"]
        y_bounds = MARK_ACTION_RANGES["y"]
        x = np.clip(x, x_bounds.low + abs(dx), x_bounds.high - abs(dx))
        y = np.clip(y, y_bounds.low + abs(dy), y_bounds.high - abs(dy))
        x0 = np.clip(x - dx, *x_bounds)
        y0 = np.clip(y - dy, *y_bounds)
        x1 = np.clip(x + dx, *x_bounds)
        y1 = np.clip(y + dy, *y_bounds)
        return StrokeAction(
            float(x0),
            float(y0),
            float(x1),
            float(y1),
            float(np.clip(width, *MARK_ACTION_RANGES["width"])),
            float(np.clip(amount, *MARK_ACTION_RANGES["amount"])),
            tone,
        )

    def _stroke_draw(self, coverage_field: np.ndarray | None = None) -> _MarkDraw:
        """Draw one mark's proposal parameters, before any decoding.

        Split out of `_stroke` so the learned proposal in `proposal.py` can be
        scored against exactly the parameters the hand-written proposal draws:
        the START point, direction, length, width, and amount. Everything after
        this point (the inward shift, the endpoint clips) is a deterministic
        shared decoder that cancels in the ratio of the two densities. The RNG
        draw ORDER is byte-identical to the previous inline body.
        """

        x0, y0 = self._start_point(coverage_field)
        angle = self.rng.uniform(0, 2 * np.pi)
        # Bias the proposal prior toward longer sweeps: short marks read as dabs
        # (a round brush over a short span is a blob), so the candidate set now
        # favours strokes long enough to show brush character.
        length = self.rng.uniform(*PROPOSAL_SUPPORT["mark_length"])
        # Log-uniform width: mostly fine marks with a heavy tail of broad ones,
        # so candidate policies span a real range of mark scales.
        width_support = PROPOSAL_SUPPORT["mark_width"]
        width = float(np.exp(self.rng.uniform(np.log(width_support.low), np.log(width_support.high))))
        amount = self.rng.uniform(*PROPOSAL_SUPPORT["mark_amount"])
        tone = self._tone()
        return _MarkDraw(
            float(x0),
            float(y0),
            float(angle),
            float(length),
            float(width),
            float(amount),
            float(tone),
        )

    @staticmethod
    def _stroke_from_draw(draw: _MarkDraw) -> StrokeAction:
        """Decode one mark draw. Deterministic: consumes no randomness."""

        x_bounds = MARK_ACTION_RANGES["x"]
        y_bounds = MARK_ACTION_RANGES["y"]
        dxv = draw.length * np.cos(draw.angle)
        dyv = draw.length * np.sin(draw.angle)
        # Shift the start inward so the full length fits on canvas; clipping the
        # endpoint instead collapses edge strokes into dwell-dabs (solid discs).
        x0 = float(np.clip(draw.x0, x_bounds.low + max(0.0, -dxv), x_bounds.high - max(0.0, dxv)))
        y0 = float(np.clip(draw.y0, y_bounds.low + max(0.0, -dyv), y_bounds.high - max(0.0, dyv)))
        x1 = np.clip(x0 + dxv, *x_bounds)
        y1 = np.clip(y0 + dyv, *y_bounds)
        return StrokeAction(
            float(x0),
            float(y0),
            float(x1),
            float(y1),
            float(draw.width),
            float(draw.amount),
            draw.tone,
        )

    def _stroke(self, coverage_field: np.ndarray | None = None) -> StrokeAction:
        return self._stroke_from_draw(self._stroke_draw(coverage_field))

    def _passage_draw(self, coverage_field: np.ndarray | None = None) -> PassageLatent:
        """Draw one passage latent, BEFORE `fit_polyline_latent` runs.

        Split out of `_passage_policy` for the same reason `_stroke_draw` was: the
        pre-fit latent is the random variable the proposal is a distribution over,
        and the inward polyline fit plus `_passage_actions` are a deterministic
        shared decoder that cancels in the ratio of the hand-written and learned
        densities. The RNG draw order is byte-identical to the previous body.
        """

        stroke_range = passage_stroke_count_range(self.cfg)
        stroke_count = int(self.rng.integers(int(stroke_range.low), int(stroke_range.high) + 1))
        center_x, center_y = self._start_point(coverage_field)
        direction = float(self.rng.uniform(0, 2 * np.pi))
        kind = self._passage_kind(band_probability=PASSAGE_BAND_PROBABILITY)
        length, spacing = self._passage_geometry(kind)
        width_support = PROPOSAL_SUPPORT["passage_width"]
        width = float(np.exp(self.rng.uniform(np.log(width_support.low), np.log(width_support.high))))
        amount = float(self.rng.uniform(*PROPOSAL_SUPPORT["passage_amount"]))
        tone = self._tone()
        return PassageLatent(
            kind=kind,
            center_x=float(center_x),
            center_y=float(center_y),
            direction=direction,
            length=length,
            spacing=spacing,
            stroke_count=stroke_count,
            width=width,
            amount=amount,
            tone=tone,
        )

    def _passage_policy_from_latent(self, latent: PassageLatent) -> Policy:
        """Decode one passage latent. Consumes RNG only through per-mark jitter."""

        latent = fit_polyline_latent(latent)
        actions = self._passage_actions(latent)
        return Policy(tuple(actions) + (StrokeAction.stop_action(),), passage=latent)

    def _passage_policy(self, coverage_field: np.ndarray | None = None) -> Policy:
        return self._passage_policy_from_latent(self._passage_draw(coverage_field))

    def passage_actions(self, latent: PassageLatent, start_index: int = 0) -> list[StrokeAction]:
        return self._passage_actions(latent, start_index=start_index)

    def _passage_actions(self, latent: PassageLatent, start_index: int = 0) -> list[StrokeAction]:
        if latent.kind == "polyline":
            fitted = fit_polyline_latent(latent)
            vertices = polyline_vertices(fitted)
            return [
                StrokeAction(
                    float(vertices[index, 0]),
                    float(vertices[index, 1]),
                    float(vertices[index + 1, 0]),
                    float(vertices[index + 1, 1]),
                    float(np.clip(fitted.width, *MARK_ACTION_RANGES["width"])),
                    float(np.clip(fitted.amount, *MARK_ACTION_RANGES["amount"])),
                    float(fitted.tone),
                )
                for index in range(max(0, int(start_index)), fitted.stroke_count)
            ]

        direction = np.asarray([np.cos(latent.direction), np.sin(latent.direction)], dtype=np.float64)
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        midpoint = 0.5 * (latent.stroke_count - 1)
        actions: list[StrokeAction] = []
        for index in range(max(0, int(start_index)), latent.stroke_count):
            offset = index - midpoint
            if latent.kind == "chain":
                passage_offset = direction * offset * latent.spacing
                stroke_angle = latent.direction + self.rng.normal(0.0, 0.16)
                stroke_length = latent.length * float(np.exp(self.rng.normal(-0.15, 0.10)))
            else:
                passage_offset = normal * offset * latent.spacing
                stroke_angle = latent.direction + self.rng.normal(0.0, 0.10)
                stroke_length = latent.length * float(np.exp(self.rng.normal(0.0, 0.10)))
            jitter = (
                direction * self.rng.normal(0.0, self.cfg.passage_longitudinal_jitter)
                + normal * self.rng.normal(0.0, self.cfg.passage_lateral_jitter)
            )
            center = np.asarray([latent.center_x, latent.center_y], dtype=np.float64) + passage_offset + jitter
            local_width = latent.width * float(np.exp(self.rng.normal(0.0, 0.16)))
            local_amount = latent.amount * float(np.exp(self.rng.normal(0.0, 0.12)))
            actions.append(
                self._stroke_from_center(
                    float(np.clip(center[0], *PROPOSAL_SUPPORT["start_x"])),
                    float(np.clip(center[1], *PROPOSAL_SUPPORT["start_y"])),
                    stroke_angle,
                    stroke_length,
                    local_width,
                    local_amount,
                    latent.tone,
                )
            )
        return actions

    def _passage_plan_policy(self, coverage_field: np.ndarray | None = None) -> Policy:
        min_passages = max(2, self.cfg.passage_plan_min_passages)
        max_passages = max(min_passages, self.cfg.passage_plan_max_passages)
        min_strokes = max(2, self.cfg.passage_min_strokes)
        if self.cfg.planning_horizon < min_passages * min_strokes:
            return self._passage_policy(coverage_field)
        max_passages = min(max_passages, max(2, self.cfg.planning_horizon // min_strokes))
        passage_count = int(self.rng.integers(min_passages, max_passages + 1))
        stroke_counts = [min_strokes for _ in range(passage_count)]
        remaining = self.cfg.planning_horizon - sum(stroke_counts)
        max_extra = max(0, self.cfg.passage_max_strokes - min_strokes)
        while remaining > 0 and max_extra > 0:
            eligible = [index for index, count in enumerate(stroke_counts) if count < self.cfg.passage_max_strokes]
            if not eligible:
                break
            index = int(self.rng.choice(eligible))
            stroke_counts[index] += 1
            remaining -= 1

        center_x, center_y = self._start_point(coverage_field)
        direction = float(self.rng.uniform(0, 2 * np.pi))
        turn = float(self.rng.normal(0.0, self.cfg.passage_plan_turn_jitter))
        plan_width_support = PROPOSAL_SUPPORT["plan_width"]
        width = float(
            np.exp(self.rng.uniform(np.log(plan_width_support.low), np.log(plan_width_support.high)))
        )
        amount = float(self.rng.uniform(*PROPOSAL_SUPPORT["plan_amount"]))
        tone = self._tone()
        kind = "progression" if abs(turn) < 0.25 else "arc"
        direction_vec = np.asarray([np.cos(direction), np.sin(direction)], dtype=np.float64)
        normal = np.asarray([-direction_vec[1], direction_vec[0]], dtype=np.float64)
        midpoint = 0.5 * (passage_count - 1)
        passages: list[PassageLatent] = []
        actions: list[StrokeAction] = []

        for index, stroke_count in enumerate(stroke_counts):
            offset = index - midpoint
            passage_direction = direction + turn * offset
            passage_kind = self._passage_kind(band_probability=PASSAGE_PLAN_BAND_PROBABILITY)
            jitter = (
                direction_vec * self.rng.normal(0.0, self.cfg.passage_plan_center_jitter)
                + normal * self.rng.normal(0.0, self.cfg.passage_plan_center_jitter)
            )
            center = (
                np.asarray([center_x, center_y], dtype=np.float64)
                + direction_vec * offset * self.cfg.passage_plan_spacing
                + jitter
            )
            passage_length, passage_spacing = self._passage_geometry(passage_kind)
            latent = PassageLatent(
                kind=passage_kind,
                center_x=float(np.clip(center[0], *PROPOSAL_SUPPORT["start_x"])),
                center_y=float(np.clip(center[1], *PROPOSAL_SUPPORT["start_y"])),
                direction=float(passage_direction),
                length=passage_length,
                spacing=passage_spacing,
                stroke_count=int(stroke_count),
                width=float(width * np.exp(self.rng.normal(0.0, 0.16))),
                amount=float(amount * np.exp(self.rng.normal(0.0, 0.12))),
                tone=tone,
            )
            latent = fit_polyline_latent(latent)
            passages.append(latent)
            actions.extend(self._passage_actions(latent))

        plan = PassagePlanLatent(
            kind=kind,
            center_x=float(center_x),
            center_y=float(center_y),
            direction=direction,
            passage_count=passage_count,
            total_stroke_count=len(actions),
            passage_spacing=float(self.cfg.passage_plan_spacing),
            turn=turn,
            width=width,
            amount=amount,
            tone=tone,
            passages=tuple(passages),
        )
        return Policy(tuple(actions) + (StrokeAction.stop_action(),), passage_plan=plan)

    # -- proposal records ---------------------------------------------------
    # These describe WHICH proposal generated a candidate and with what latents,
    # so the amortization target can be attributed and the learned and
    # hand-written branches can be compared as a paired same-round control. They
    # are derived from parameters already drawn and consume no randomness.

    @staticmethod
    def _mark_record(draws: Sequence[_MarkDraw], source: str) -> ProposalRecord:
        return ProposalRecord(
            family="mark",
            source=source,
            latents=tuple(
                ProposalLatent(
                    family="mark",
                    kind="mark",
                    center_x=draw.x0,
                    center_y=draw.y0,
                    direction=draw.angle,
                    length=draw.length,
                    spacing=0.0,
                    width=draw.width,
                    amount=draw.amount,
                    stroke_count=1,
                    tone=draw.tone,
                )
                for draw in draws
            ),
        )

    @staticmethod
    def _passage_record(latent: PassageLatent, source: str) -> ProposalRecord:
        return ProposalRecord(
            family="passage",
            source=source,
            latents=(
                ProposalLatent(
                    family="passage",
                    kind=latent.kind,
                    center_x=float(latent.center_x),
                    center_y=float(latent.center_y),
                    direction=float(latent.direction),
                    length=float(latent.length),
                    spacing=float(latent.spacing),
                    width=float(latent.width),
                    amount=float(latent.amount),
                    stroke_count=int(latent.stroke_count),
                    tone=float(latent.tone),
                ),
            ),
        )

    @staticmethod
    def _record_with_tone(record: ProposalRecord, tone: float) -> ProposalRecord:
        """Re-tone a record to match the tone alternative actually emitted.

        The DECLARED `stroke_tone_prior` policy prior keeps exclusive control of
        emission through `_tone` / `_tone_support` / `_policy_tone_alternatives`.
        The proposal's tone factor is therefore scored against the tone that was
        emitted rather than the one it drew -- so "has the agent developed a tone
        preference" stays measurable without the proposal becoming an
        unaccountable sampling confound.
        """

        return replace(
            record,
            latents=tuple(replace(latent, tone=float(tone)) for latent in record.latents),
        )

    # -- learned branches ---------------------------------------------------

    def _learned_mark_policy(
        self,
        features: torch.Tensor,
        depth: int,
    ) -> tuple[Policy, ProposalRecord]:
        assert self.learned_proposal is not None
        record = self.learned_proposal.sample(
            features, family="mark", config=self.cfg, count=max(1, int(depth))
        )
        draws = [
            _MarkDraw(
                latent.center_x,
                latent.center_y,
                latent.direction,
                latent.length,
                latent.width,
                latent.amount,
                # D4: the learned tone draw is scored but never emitted.
                self._tone(),
            )
            for latent in record.latents
        ]
        actions = tuple(self._stroke_from_draw(draw) for draw in draws) + (StrokeAction.stop_action(),)
        return Policy(actions), record

    def _learned_passage_policy(self, features: torch.Tensor) -> tuple[Policy, ProposalRecord]:
        assert self.learned_proposal is not None
        record = self.learned_proposal.sample(features, family="passage", config=self.cfg, count=1)
        sampled = record.latents[0]
        stroke_range = passage_stroke_count_range(self.cfg)
        latent = PassageLatent(
            kind=sampled.kind,
            center_x=sampled.center_x,
            center_y=sampled.center_y,
            direction=sampled.direction,
            length=sampled.length,
            spacing=sampled.spacing,
            # The declared config range dominates the learned categorical, which
            # is what keeps a learned sample from raising inside
            # `Policy.__post_init__` on the planner thread, where the exception
            # would surface only as a diagnostics string.
            stroke_count=int(
                min(max(int(sampled.stroke_count), int(stroke_range.low)), int(stroke_range.high))
            ),
            width=sampled.width,
            amount=sampled.amount,
            tone=self._tone(),
        )
        return self._passage_policy_from_latent(latent), self._passage_record(latent, "learned")

    def _learned_mixture_weight(self, features: torch.Tensor | None) -> float:
        """Fraction of candidates the learned proposal generates this round.

        Zero unless BOTH a trained-or-untrained proposal and belief features are
        available, so every caller that does not condition on a canvas/relational
        posterior (the summary planner, the tests, `bootstrap_dynamics`) keeps the
        hand-written proposal exactly.
        """

        if self.learned_proposal is None or features is None:
            return 0.0
        return float(np.clip(self.cfg.learned_proposal_mix, 0.0, 1.0))

    def sample(
        self,
        coverage_field: np.ndarray | None = None,
        belief_features: torch.Tensor | None = None,
    ) -> list[Policy]:
        policies = [Policy((StrokeAction.stop_action(),))]
        continuation_count = max(0, self.cfg.candidate_policies - 1)
        passage_capacity = max(0, self.cfg.planning_horizon - 1)
        plan_capacity = max(0, self.cfg.planning_horizon - max(2, self.cfg.passage_plan_min_passages) * max(2, self.cfg.passage_min_strokes) + 1)
        plan_count = 0
        if plan_capacity > 0 and self.cfg.passage_plan_proposal_mix > 0.0:
            plan_count = int(round(continuation_count * np.clip(self.cfg.passage_plan_proposal_mix, 0.0, 1.0)))
            plan_count = min(continuation_count, plan_count)
        passage_count = 0
        if passage_capacity > 0 and self.cfg.passage_proposal_mix > 0.0:
            passage_count = int(round(continuation_count * np.clip(self.cfg.passage_proposal_mix, 0.0, 1.0)))
            passage_count = min(continuation_count - plan_count, passage_count)
        mark_count = continuation_count - passage_count - plan_count
        continuations: list[Policy] = []
        # Keyed by identity, so the post-shuffle re-association consumes no
        # randomness and makes no assumption about `Generator.shuffle`'s
        # internals. Shuffling a list of (policy, record) pairs instead would
        # change the RNG stream and break five seeded tests.
        record_by_id: dict[int, ProposalRecord] = {}

        def add_alternatives(
            base_policy: Policy,
            limit: int,
            record: ProposalRecord | None = None,
        ) -> None:
            for alternative in self._policy_tone_alternatives(base_policy):
                if len(continuations) >= limit:
                    break
                continuations.append(alternative)
                if record is not None:
                    record_by_id[id(alternative)] = self._record_with_tone(
                        record, float(alternative.actions[0].tone)
                    )

        mixture_weight = self._learned_mixture_weight(belief_features)
        learned_count = 0

        # DETERMINISTIC COUNTS, not a per-candidate Bernoulli: at mix 0.0 both
        # learned limits are at or below the current length, so neither loop body
        # executes and `self.rng` is consumed exactly as before.
        mark_limit = mark_count
        learned_mark_limit = min(mark_limit, int(round(mark_count * mixture_weight)))
        while len(continuations) < learned_mark_limit:
            depth = int(self.rng.integers(1, self.cfg.planning_horizon + 1))
            assert belief_features is not None
            learned_policy, learned_record = self._learned_mark_policy(belief_features, depth)
            before = len(continuations)
            add_alternatives(learned_policy, learned_mark_limit, learned_record)
            learned_count += len(continuations) - before
        while len(continuations) < mark_limit:
            depth = int(self.rng.integers(1, self.cfg.planning_horizon + 1))
            draws = [self._stroke_draw(coverage_field) for _ in range(depth)]
            actions = tuple(self._stroke_from_draw(draw) for draw in draws) + (StrokeAction.stop_action(),)
            add_alternatives(Policy(actions), mark_limit, self._mark_record(draws, "hand"))

        passage_limit = continuation_count
        passage_target = mark_count + passage_count
        learned_passage_limit = min(
            passage_target, len(continuations) + int(round(passage_count * mixture_weight))
        )
        while len(continuations) < learned_passage_limit:
            assert belief_features is not None
            learned_policy, learned_record = self._learned_passage_policy(belief_features)
            before = len(continuations)
            add_alternatives(learned_policy, learned_passage_limit, learned_record)
            learned_count += len(continuations) - before
        while len(continuations) < passage_target:
            latent = self._passage_draw(coverage_field)
            add_alternatives(
                self._passage_policy_from_latent(latent),
                passage_limit,
                self._passage_record(latent, "hand"),
            )
        while len(continuations) < passage_limit:
            # Passage-plan compounds carry no record: the plan family is outside
            # the learned proposal's declared scope, so it has no density on
            # either side and must not contribute to the amortization target.
            add_alternatives(self._passage_plan_policy(coverage_field), passage_limit)
        self.rng.shuffle(continuations)
        policies.extend(continuations)
        self.last_learned_candidate_count = learned_count
        self.last_proposal_records = (None,) + tuple(
            record_by_id.get(id(policy)) for policy in continuations
        )
        return policies
