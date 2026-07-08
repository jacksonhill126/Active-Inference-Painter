from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import PainterConfig
from .env import StrokeAction


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


@dataclass(frozen=True, slots=True)
class PassageLatent:
    """Higher-level latent policy prior over a related sequence of marks.

    This is not an outcome preference or reward. It is a transition prior over
    mark trajectories: a latent passage generates several strokes that share a
    coarse region, direction, scale, tone, and amount. EFE still evaluates the
    predicted consequences of the completed policy, including terminal stop.
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

    def __post_init__(self) -> None:
        if not self.actions or not self.actions[-1].stop:
            raise ValueError("Every painting policy must terminate in stop.")
        if any(action.stop for action in self.actions[:-1]):
            raise ValueError("Stop may appear only as the final painting policy action.")
        if self.passage is not None and self.passage_plan is not None:
            raise ValueError("A policy may carry either passage or passage-plan metadata, not both.")
        if self.passage is not None and len(self.actions) < 3:
            raise ValueError("A passage policy must contain multiple marks before stop.")
        if self.motor_primitive is not None and self.actions[0].stop:
            raise ValueError("A motor realization latent requires a non-stop first action.")
        if self.passage_plan is not None:
            if self.passage_plan.passage_count < 2:
                raise ValueError("A passage-plan policy must contain multiple passages.")
            if self.passage_plan.total_stroke_count != len(self.actions) - 1:
                raise ValueError("A passage-plan latent must match the policy mark count.")


def policy_stop_log_prior(policy: Policy, believed_coverage: float, config: PainterConfig) -> float:
    """Declared policy prior term log p(pi) for premature termination.

    The prior probability that a policy terminates immediately follows a
    sigmoid in believed material coverage centered at
    `minimum_stop_coverage`. Continuation policies carry a flat prior of zero
    log-weight; normalization over the sampled candidate set is absorbed by
    the policy softmax. This replaces the previous procedural stop veto with
    an explicit prior inside policy inference: stopping stays admissible at
    every coverage, merely a priori unlikely before the midpoint.
    """

    if not policy.actions[0].stop:
        return 0.0
    logit = config.stop_prior_sharpness * (float(believed_coverage) - config.minimum_stop_coverage)
    if logit >= 0.0:
        return -float(np.log1p(np.exp(-logit)))
    return float(logit - np.log1p(np.exp(logit)))


class PolicySampler:
    """Candidate-policy proposal distribution.

    When a spatial coverage belief field is provided, a declared fraction of
    stroke proposals start in low-coverage regions. This is an empirical
    policy prior over candidates, not a reward term: candidates are still
    scored purely by expected free energy.
    """

    def __init__(self, config: PainterConfig, seed: int = 0) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(seed)

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
            x0 = float(np.clip((col + self.rng.uniform()) / cols, 0.05, 0.95))
            y0 = float(np.clip((row + self.rng.uniform()) / rows, 0.05, 0.95))
            return x0, y0
        x0, y0 = self.rng.uniform(0.05, 0.95, size=2)
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
        return StrokeAction(
            action.x0,
            action.y0,
            action.x1,
            action.y1,
            action.width,
            action.amount,
            float(tone),
        )

    def _policy_tone_alternatives(self, policy: Policy) -> list[Policy]:
        alternatives: list[Policy] = []
        for tone in self._tone_support():
            passage = replace(policy.passage, tone=tone) if policy.passage is not None else None
            passage_plan = self._passage_plan_with_tone(policy.passage_plan, tone)
            actions = tuple(self._action_with_tone(action, tone) for action in policy.actions)
            alternatives.append(
                Policy(
                    actions,
                    passage=passage,
                    passage_plan=passage_plan,
                    motor_primitive=policy.motor_primitive,
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
        x0 = np.clip(x - dx, 0.03, 0.97)
        y0 = np.clip(y - dy, 0.03, 0.97)
        x1 = np.clip(x + dx, 0.03, 0.97)
        y1 = np.clip(y + dy, 0.03, 0.97)
        return StrokeAction(
            float(x0),
            float(y0),
            float(x1),
            float(y1),
            float(np.clip(width, 0.02, 0.34)),
            float(np.clip(amount, 0.05, 0.95)),
            tone,
        )

    def _stroke(self, coverage_field: np.ndarray | None = None) -> StrokeAction:
        x0, y0 = self._start_point(coverage_field)
        angle = self.rng.uniform(0, 2 * np.pi)
        length = self.rng.uniform(0.08, 0.48)
        x1 = np.clip(x0 + length * np.cos(angle), 0.03, 0.97)
        y1 = np.clip(y0 + length * np.sin(angle), 0.03, 0.97)
        # Log-uniform width: mostly fine marks with a heavy tail of broad ones,
        # so candidate policies span a real range of mark scales.
        width = float(np.exp(self.rng.uniform(np.log(0.03), np.log(0.30))))
        amount = self.rng.uniform(0.12, 0.75)
        tone = self._tone()
        return StrokeAction(float(x0), float(y0), float(x1), float(y1), float(width), float(amount), tone)

    def _passage_policy(self, coverage_field: np.ndarray | None = None) -> Policy:
        max_strokes = min(max(1, self.cfg.planning_horizon), max(1, self.cfg.passage_max_strokes))
        min_strokes = min(max_strokes, max(2, self.cfg.passage_min_strokes))
        stroke_count = int(self.rng.integers(min_strokes, max_strokes + 1))
        center_x, center_y = self._start_point(coverage_field)
        direction = float(self.rng.uniform(0, 2 * np.pi))
        length = float(self.rng.uniform(0.16, 0.54))
        spacing = float(self.rng.uniform(0.045, 0.15))
        width = float(np.exp(self.rng.uniform(np.log(0.035), np.log(0.24))))
        amount = float(self.rng.uniform(0.16, 0.7))
        tone = self._tone()
        kind = "band" if self.rng.uniform() < 0.65 else "chain"
        latent = PassageLatent(
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
        actions = self._passage_actions(latent)
        return Policy(tuple(actions) + (StrokeAction.stop_action(),), passage=latent)

    def _passage_actions(self, latent: PassageLatent) -> list[StrokeAction]:
        direction = np.asarray([np.cos(latent.direction), np.sin(latent.direction)], dtype=np.float64)
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        midpoint = 0.5 * (latent.stroke_count - 1)
        actions: list[StrokeAction] = []
        for index in range(latent.stroke_count):
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
                    float(np.clip(center[0], 0.05, 0.95)),
                    float(np.clip(center[1], 0.05, 0.95)),
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
        width = float(np.exp(self.rng.uniform(np.log(0.035), np.log(0.22))))
        amount = float(self.rng.uniform(0.16, 0.68))
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
            passage_kind = "band" if self.rng.uniform() < 0.55 else "chain"
            jitter = (
                direction_vec * self.rng.normal(0.0, self.cfg.passage_plan_center_jitter)
                + normal * self.rng.normal(0.0, self.cfg.passage_plan_center_jitter)
            )
            center = (
                np.asarray([center_x, center_y], dtype=np.float64)
                + direction_vec * offset * self.cfg.passage_plan_spacing
                + jitter
            )
            latent = PassageLatent(
                kind=passage_kind,
                center_x=float(np.clip(center[0], 0.05, 0.95)),
                center_y=float(np.clip(center[1], 0.05, 0.95)),
                direction=float(passage_direction),
                length=float(self.rng.uniform(0.14, 0.48)),
                spacing=float(self.rng.uniform(0.045, 0.13)),
                stroke_count=int(stroke_count),
                width=float(width * np.exp(self.rng.normal(0.0, 0.16))),
                amount=float(amount * np.exp(self.rng.normal(0.0, 0.12))),
                tone=tone,
            )
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

    def sample(self, coverage_field: np.ndarray | None = None) -> list[Policy]:
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

        def add_alternatives(base_policy: Policy, limit: int) -> None:
            for alternative in self._policy_tone_alternatives(base_policy):
                if len(continuations) >= limit:
                    break
                continuations.append(alternative)

        mark_limit = mark_count
        while len(continuations) < mark_limit:
            depth = int(self.rng.integers(1, self.cfg.planning_horizon + 1))
            actions = tuple(self._stroke(coverage_field) for _ in range(depth)) + (StrokeAction.stop_action(),)
            add_alternatives(Policy(actions), mark_limit)

        passage_limit = continuation_count
        while len(continuations) < mark_count + passage_count:
            add_alternatives(self._passage_policy(coverage_field), passage_limit)
        while len(continuations) < passage_limit:
            add_alternatives(self._passage_plan_policy(coverage_field), passage_limit)
        self.rng.shuffle(continuations)
        policies.extend(continuations)
        return policies
