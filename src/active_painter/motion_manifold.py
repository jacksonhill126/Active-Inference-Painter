"""Joint-space sweeps of the arm's own reachable-motion manifold.

WHAT THIS IS, IN GENERATIVE-MODEL TERMS
=======================================
This module is GENERATIVE-PROCESS exploration of the agent's own body. It is
NOT a painting policy, not a preference, and not a score.

Every quantity it produces is one of exactly two things:

1. A sample of an existing, already-declared TRANSITION PRIOR. A sweep is
   re-expressed as a `policies.PassageLatent`, whose own docstring states the
   identity verbatim: "This is not an outcome preference or reward. It is a
   transition prior over mark trajectories." This module changes only the
   proposal distribution those bootstrap samples are drawn from - iid marks
   become body-feasible arcs - and the latent is always re-decoded through
   `PolicySampler.passage_actions`, so the latent still generates its own
   actions rather than having geometry glued onto it.
2. TRAINING DATA FOR LIKELIHOODS: the pixel/material transition likelihood
   (GM-PIX-TRANS) via the driver's replay, and the canvas/relational
   likelihoods (Q-CANVAS / Q-REL) via whole-canvas composition replay entries.
   `HierarchicalCanvasModel.training_loss` is a canvas VFE plus a relational
   ELBO, i.e. likelihood/recognition parameters, never a preference.

`ManifoldSweep.declared_as` carries that statement on the dataclass, because
"a number in a dataclass that looks like a score" is the pattern the charter
forbids.

WHY THIS IS INSIDE THE CHARTER
==============================
The mechanics are conventional forward kinematics plus a numerical Jacobian
null-space projection, evaluated BELOW the selected painting policy - the same
category as `arm_control.ik_pose_for_canvas_point`, which the charter's
"Allowed conventional engineering boundary" clause blesses by name. No painting
policy is chosen here: the live policy posterior still comes solely from
`SpatialActiveInferencePainter.infer_policy`. A sweep enters no expected free
energy term, no variational free energy term, and no policy prior.

It supplies NO aesthetic content and NO demonstrated painting policy. What it
supplies is a sensorimotor history: the first regularity the composition
hierarchy learns is a property of the agent's own embodiment, because a
revolute arm sweeps arcs and iid mark sampling cannot produce correlated
curvature, direction, and length. It is attributable because it is behind the
declared `config.bootstrap_generator` flag with the previous iid source
retained as `'random_stroke'`.

WHAT IT DOES NOT FIX
====================
It RELOCATES, and does not remove, the circularity named in the approximation
register: the structural prior is still evaluated by a code fitted on the
agent's own history. The origin of that history moves from an arbitrary noise
source to the body. Claiming more would be the "rename an ordinary controller
as active inference" failure the charter forbids.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields

import numpy as np

from .arm_control import ik_pose_for_canvas_point
from .arm_sim import NEAR_SURFACE_TOLERANCE, ArmKinematics, ArmPose, VerticalCanvas
from .config import PainterConfig
from .env import StrokeAction
from .policies import PassageLatent, PolicySampler, fit_polyline_latent
from .stroke_execution import CANVAS_REACH_FRACTION


# Mirrors ArmPose.clipped(): the sweep integrator explores the body's REAL
# reachable set and stops at its edge rather than silently clipping into a pose
# the plant would refuse. Hard limits are external safety, not a model term.
JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-90.0, 90.0),
    (-90.0, 90.0),
    (-180.0, 180.0),
    (0.0, 150.0),
)

MANIFOLD_FAMILIES: tuple[str, ...] = (
    "yaw_dominant",
    "pitch_dominant",
    "roll_dominant",
    "elbow_dominant",
    "coordinated",
)

_DOMINANT_JOINT_INDEX: dict[str, int] = {
    "yaw_dominant": 0,
    "pitch_dominant": 1,
    "roll_dominant": 2,
    "elbow_dominant": 3,
}

# Fraction of the bushing travel the sweep may sink past the canvas plane. Kept
# strictly below 1.0 so `VerticalCanvas.too_deep` can never fire on a sweep
# pose: overtravel is an external hard limit, so the exploration must stay
# inside it by construction rather than rely on a rejection test.
DEPTH_BAND_TRAVEL_FRACTION = 0.8

# Numerical step for the tip-depth gradient, in degrees.
DEPTH_GRADIENT_EPS_DEGREES = 1e-3

_MIN_SWEEP_POINTS = 3

# Mirrors the total-length clip inside `policies._polyline_relative_vertices`.
# The segmentation rule has to know it: a sweep longer than this cannot lend its
# full arclength to the latent, so mark count must be derived from the clipped
# length or the decoded segments would fall below the minimum mark length.
POLYLINE_MAX_TOTAL_LENGTH = 0.86

# Longest mark `PolicySampler._stroke_from_center` can place without its
# inward-shift clip shortening the requested span: the margins are 0.03 on both
# sides, so a mark centred anywhere still fits up to 0.94.
MAX_BARE_MARK_LENGTH = 0.94

_CANVAS_FIELD_DEFAULTS = {
    field.name: field.default for field in dataclass_fields(VerticalCanvas)
}


@dataclass(frozen=True, slots=True)
class ManifoldSweep:
    """One integrated joint-space sweep of the reachable-motion manifold.

    NOT A DECISION QUANTITY. Every float here is a property of a realized
    forward-kinematic path through the body's own configuration space. None of
    them is summed into an expected or variational free energy, ranked against
    anything, or read by any preference. `declared_as` says so on the object.
    """

    family: str
    joint_path: np.ndarray
    tip_path: np.ndarray
    canvas_path: np.ndarray
    start_pose: ArmPose
    end_pose: ArmPose
    normalized_path_length: float
    chord_length: float
    total_turn_radians: float
    amplitude_degrees: float
    step_degrees: float
    declared_as: str = (
        "generative-process sample of the body's reachable motion; "
        "not a decision quantity, not scored, carries no preference"
    )


class MotionManifoldSampler:
    """Proposal distribution over body-feasible bootstrap marks.

    Owns its own `numpy` Generator. It must never draw from
    `agent.policy_sampler.rng`: sharing that stream would make switching
    `bootstrap_generator` also change the live planner's candidate proposals,
    which would confound the attribution comparison the charter requires.
    """

    def __init__(
        self,
        config: PainterConfig,
        *,
        seed: int | None = None,
        kinematics: ArmKinematics | None = None,
        canvas: VerticalCanvas | None = None,
    ) -> None:
        self.cfg = config
        # Validate eagerly: a typo'd family name would otherwise degrade silently
        # into an undeclared coordinated sweep and quietly break attribution.
        self.families = validated_families(config)
        self.rng = np.random.default_rng(
            int(config.bootstrap_manifold_seed if seed is None else seed)
        )
        self.kinematics = kinematics if kinematics is not None else ArmKinematics()
        # Geometry is read from the live canvas when available and otherwise from
        # VerticalCanvas' own field defaults, so no canvas dimension is
        # re-hardcoded here.
        self.canvas_width = float(
            canvas.width if canvas is not None else _CANVAS_FIELD_DEFAULTS["width"]
        )
        self.canvas_height = float(
            canvas.height if canvas is not None else _CANVAS_FIELD_DEFAULTS["height"]
        )
        self.canvas_distance = float(
            canvas.distance if canvas is not None else _CANVAS_FIELD_DEFAULTS["distance"]
        )
        bushing_travel = float(
            canvas.bushing_travel
            if canvas is not None
            else _CANVAS_FIELD_DEFAULTS["bushing_travel"]
        )
        self.depth_lower = self.canvas_distance - NEAR_SURFACE_TOLERANCE
        self.depth_upper = self.canvas_distance + DEPTH_BAND_TRAVEL_FRACTION * bushing_travel
        self.reach_half_width = 0.5 * self.canvas_width * CANVAS_REACH_FRACTION
        self.reach_half_height = 0.5 * self.canvas_height * CANVAS_REACH_FRACTION
        # A private PolicySampler, seeded from the manifold seed rather than the
        # agent's, used ONLY to decode a fitted latent back into actions. Its
        # `passage_actions` is deterministic for polyline latents, so this
        # consumes no randomness for the emitted geometry.
        self._policy_sampler = PolicySampler(config, seed=int(self.rng.integers(0, 2**31 - 1)))
        self._chain_pose: ArmPose | None = None
        self.sweep_count: int = 0
        self.rejected_sweeps: int = 0
        self._turn_sum: float = 0.0
        self._path_chord_ratio_sum: float = 0.0
        self._normalized_path_sum: float = 0.0

    # ------------------------------------------------------------------ chain

    def reset_chain(self) -> None:
        """Forget the previous sweep's end pose (called at episode boundaries)."""

        self._chain_pose = None

    # --------------------------------------------------------------- sampling

    def sample_sweep(self, *, family: str | None = None) -> ManifoldSweep | None:
        """Integrate one sweep, or return None when the manifold edge is hit."""

        families = self.families
        chosen = str(family) if family is not None else str(
            families[int(self.rng.integers(0, len(families)))]
        )
        start = self._start_joints()
        if start is None:
            self.rejected_sweeps += 1
            return None
        direction = self.rng.normal(size=4) * self._family_weights(chosen)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            self.rejected_sweeps += 1
            return None
        amplitude = float(
            self.rng.uniform(*_ordered_range(self.cfg.bootstrap_manifold_amplitude_degrees))
        )
        step = float(
            self.rng.uniform(*_ordered_range(self.cfg.bootstrap_manifold_step_degrees))
        )
        joint_path, tip_path = self._integrate(start, direction / norm, amplitude, step)
        if len(joint_path) < _MIN_SWEEP_POINTS:
            self.rejected_sweeps += 1
            return None
        canvas_path = self._to_normalized(tip_path)
        path_length, chord, turn = _path_statistics(canvas_path)
        sweep = ManifoldSweep(
            family=chosen,
            joint_path=joint_path,
            tip_path=tip_path,
            canvas_path=canvas_path,
            start_pose=_pose_from_joints(joint_path[0]),
            end_pose=_pose_from_joints(joint_path[-1]),
            normalized_path_length=path_length,
            chord_length=chord,
            total_turn_radians=turn,
            amplitude_degrees=amplitude,
            step_degrees=step,
        )
        self.sweep_count += 1
        self._turn_sum += abs(turn)
        self._normalized_path_sum += path_length
        self._path_chord_ratio_sum += path_length / max(1e-9, chord)
        if self.cfg.bootstrap_manifold_chain_sweeps:
            self._chain_pose = sweep.end_pose
        return sweep

    def sample_marks(
        self,
        *,
        attempts: int = 8,
    ) -> tuple[PassageLatent | None, tuple[StrokeAction, ...]] | None:
        """One sweep re-expressed as body-feasible marks, or None on failure.

        Segmentation is by PATH BUDGET, not by a fixed mark count: a sweep that
        can only carry one mark is emitted WITHOUT a passage latent, because a
        polyline `PassageLatent` needs at least two marks
        (`Policy.__post_init__`). Sweeps shorter than the minimum mark length
        are rejected outright so the emitted marks never collapse into dwell
        dabs.

        MEASURED LIMITATION, not fixed by this rule: the contact depth band caps
        a sweep's usable arclength at a measured mean of ~0.49 normalized, which
        segments into about two marks of ~0.24 each, whereas
        `PolicySampler._stroke` samples lengths uniformly on [0.20, 0.60] (mean
        0.40). At equal mark counts the manifold generator therefore lays
        SUBSTANTIALLY LESS painted path than the iid baseline (measured 22.75 vs
        38.62 over 96 marks; coverage 0.169 vs 0.334). Any A/B between the two
        generators must report `paintedPathLength` and `episodeCoverageMean` per
        arm, because the comparison is otherwise confounded by paint budget.
        """

        min_length = max(1e-3, float(self.cfg.bootstrap_manifold_min_mark_length))
        max_marks = max(
            1,
            min(
                max(1, int(self.cfg.planning_horizon)),
                max(1, int(self.cfg.passage_max_strokes)),
            ),
        )
        for _ in range(max(1, int(attempts))):
            sweep = self.sample_sweep()
            if sweep is None:
                continue
            if sweep.normalized_path_length < min_length:
                self.rejected_sweeps += 1
                continue
            # Derive the mark count from the arclength the latent can actually
            # carry, so every decoded segment is at least `min_length`.
            usable = min(sweep.normalized_path_length, POLYLINE_MAX_TOTAL_LENGTH)
            mark_count = int(np.clip(np.floor(usable / min_length), 1, max_marks))
            width, amount, tone = self._mark_material()
            if mark_count == 1:
                # A one-mark sweep is emitted as its CHORD, so the chord (not the
                # arclength) is what has to clear the minimum. A tightly curved
                # short sweep is rejected rather than flattened into a dab.
                if sweep.chord_length < min_length:
                    self.rejected_sweeps += 1
                    continue
                return None, (self._bare_mark(sweep, width, amount, tone),)
            latent = self._fit_polyline_latent(sweep, mark_count, width, amount, tone)
            actions = tuple(self._policy_sampler.passage_actions(latent))
            if len(actions) < 2:
                self.rejected_sweeps += 1
                continue
            return latent, actions
        return None

    # -------------------------------------------------------------- statistics

    def statistics(self) -> dict[str, float | int]:
        """Plain-float sweep geometry summary for the evidence block."""

        count = max(1, self.sweep_count)
        return {
            "sweepCount": int(self.sweep_count),
            "rejectedSweeps": int(self.rejected_sweeps),
            "meanSweepTurnRadians": float(self._turn_sum / count),
            "meanSweepPathChordRatio": float(self._path_chord_ratio_sum / count),
            "meanSweepNormalizedPathLength": float(self._normalized_path_sum / count),
        }

    # ---------------------------------------------------------------- privates

    def _tip(self, joints: np.ndarray) -> np.ndarray:
        return self.kinematics.tip(_pose_from_joints(joints))

    def _depth_gradient(
        self,
        joints: np.ndarray,
        eps: float = DEPTH_GRADIENT_EPS_DEGREES,
    ) -> np.ndarray:
        """Central-difference d(tip depth)/d(joint angle), four extra FK evals.

        A numerical Jacobian row, not a learned model. It exists so the sweep can
        stay on the canvas plane; the projection onto its null space is a
        first-order constraint re-evaluated every step, named as such in the
        approximation register.
        """

        gradient = np.zeros(4, dtype=np.float64)
        for index in range(4):
            forward = joints.copy()
            forward[index] += eps
            backward = joints.copy()
            backward[index] -= eps
            gradient[index] = float(
                self._tip(forward)[1] - self._tip(backward)[1]
            ) / (2.0 * eps)
        return gradient

    def _start_joints(self) -> np.ndarray | None:
        if self._chain_pose is not None:
            jitter = float(self.cfg.bootstrap_manifold_chain_jitter_degrees)
            joints = _joints_from_pose(self._chain_pose)
            if jitter > 0.0:
                joints = joints + self.rng.normal(0.0, jitter, size=4)
            joints = np.clip(joints, _LIMIT_LOWER, _LIMIT_UPPER)
            if self._start_is_admissible(joints):
                return joints
            # The chained pose drifted off the contact band; fall through to a
            # fresh IK start rather than integrating from an illegal pose.
            self._chain_pose = None
        for _ in range(8):
            x = float(self.rng.uniform(-0.75, 0.75)) * self.reach_half_width
            z = float(self.rng.uniform(-0.75, 0.75)) * self.reach_half_height
            try:
                pose = ik_pose_for_canvas_point(x, z, self.canvas_distance)
            except ValueError:
                continue
            joints = _joints_from_pose(pose)
            if self._start_is_admissible(joints):
                return joints
        return None

    def _start_is_admissible(self, joints: np.ndarray) -> bool:
        if np.any(joints < _LIMIT_LOWER - 1e-9) or np.any(joints > _LIMIT_UPPER + 1e-9):
            return False
        return self._tip_admissible(self._tip(joints))

    def _tip_admissible(self, tip: np.ndarray) -> bool:
        depth = float(tip[1])
        if depth < self.depth_lower or depth > self.depth_upper:
            return False
        return (
            abs(float(tip[0])) <= self.reach_half_width
            and abs(float(tip[2])) <= self.reach_half_height
        )

    def _family_weights(self, family: str) -> np.ndarray:
        """Per-joint velocity weights for one manifold family.

        A family only UP-WEIGHTS its joint inside an otherwise coordinated
        sweep. Measured: holding three joints fixed and sweeping the fourth from
        the canvas-centre IK pose leaves the tip inside the near-contact band for
        only a few degrees, so a strictly single-joint sweep paints a dab, not a
        mark. `test_single_joint_sweeps_cannot_paint` pins that.
        """

        ratio = float(np.clip(self.cfg.bootstrap_manifold_dominance_ratio, 0.0, 1.0))
        index = _DOMINANT_JOINT_INDEX.get(family)
        if index is None:
            return np.ones(4, dtype=np.float64)
        weights = np.full(4, ratio, dtype=np.float64)
        weights[index] = 1.0
        return weights

    def _integrate(
        self,
        start: np.ndarray,
        unit_direction: np.ndarray,
        amplitude: float,
        step: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate a fixed joint-velocity direction along the contact surface.

        Each step projects the requested joint velocity onto the null space of
        the tip-depth gradient, so the tip slides across the canvas plane rather
        than into or away from it. Integration stops at the manifold's real
        edge: a joint limit, the contact depth band, the canvas border, or the
        requested joint-space arclength.
        """

        joints = np.asarray(start, dtype=np.float64).copy()
        joint_path: list[np.ndarray] = [joints.copy()]
        tip_path: list[np.ndarray] = [self._tip(joints)]
        travelled = 0.0
        step = max(1e-3, float(step))
        while travelled < float(amplitude):
            gradient = self._depth_gradient(joints)
            squared = float(gradient @ gradient)
            if squared > 1e-12:
                tangent = unit_direction - gradient * (
                    float(gradient @ unit_direction) / squared
                )
            else:
                tangent = unit_direction
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm < 1e-9:
                break
            candidate = joints + step * tangent / tangent_norm
            if np.any(candidate < _LIMIT_LOWER) or np.any(candidate > _LIMIT_UPPER):
                break
            tip = self._tip(candidate)
            if not self._tip_admissible(tip):
                break
            joints = candidate
            joint_path.append(joints.copy())
            tip_path.append(tip)
            travelled += step
        return (
            np.asarray(joint_path, dtype=np.float64),
            np.asarray(tip_path, dtype=np.float64),
        )

    def _to_normalized(self, tip_path: np.ndarray) -> np.ndarray:
        """Invert `stroke_execution.stroke_world_endpoints`.

        Deliberately NOT `VerticalCanvas.world_to_pixel`: normalized stroke
        space is the coordinate system `StrokeAction` lives in, and it is the
        reach-fraction map, not the pixel map. Using the pixel map would emit
        marks whose realized geometry differs from the requested geometry by a
        factor of `CANVAS_REACH_FRACTION`.
        """

        x = tip_path[:, 0] / (self.canvas_width * CANVAS_REACH_FRACTION) + 0.5
        y = 0.5 - tip_path[:, 2] / (self.canvas_height * CANVAS_REACH_FRACTION)
        return np.stack([x, y], axis=1)

    def _mark_material(self) -> tuple[float, float, float]:
        """Width/amount/tone for the emitted marks.

        Drawn from the same declared candidate ranges `PolicySampler`'s passage
        proposal uses, so the manifold generator differs from the iid baseline in
        GEOMETRY only. `amount` is a deposition quantity; nothing kinematic is
        folded into it.
        """

        width = float(np.exp(self.rng.uniform(np.log(0.035), np.log(0.24))))
        amount = float(self.rng.uniform(0.16, 0.7))
        if self.cfg.stroke_tone_prior is None:
            tone = float(self.rng.integers(0, 2))
        else:
            tone = float(self.cfg.stroke_tone_prior)
        return width, amount, tone

    def _bare_mark(
        self,
        sweep: ManifoldSweep,
        width: float,
        amount: float,
        tone: float,
    ) -> StrokeAction:
        """A single mark on the sweep's chord, with no passage latent.

        Geometry goes through `PolicySampler._stroke_from_center`, the repo's one
        implementation of the inward-shift rule that keeps an edge mark from
        collapsing into a dwell dab (clipping the endpoints instead paints a
        solid disc no real brush makes).
        """

        start = sweep.canvas_path[0]
        end = sweep.canvas_path[-1]
        center = 0.5 * (start + end)
        delta = end - start
        angle = float(np.arctan2(delta[1], delta[0]))
        return self._policy_sampler._stroke_from_center(
            float(center[0]),
            float(center[1]),
            angle,
            float(min(sweep.chord_length, MAX_BARE_MARK_LENGTH)),
            width,
            amount,
            tone,
        )

    def _fit_polyline_latent(
        self,
        sweep: ManifoldSweep,
        mark_count: int,
        width: float,
        amount: float,
        tone: float,
    ) -> PassageLatent:
        """Best-fit constant-turn polyline latent for the sweep's arc.

        LOSSY BY CONSTRUCTION. `_polyline_relative_vertices` is an equal-segment
        constant-turn model with clipped total length and turn, so the emitted
        marks are the LATENT's arc, not the body's exact arc. That is the correct
        trade: it keeps the latent self-consistent for `infer_passage_observation`
        and `PassageBelief`, which a best-fit latent glued onto raw FK segments
        would not. Named in the approximation register.
        """

        vertices = _resample_equal_arclength(sweep.canvas_path, mark_count + 1)
        segments = np.diff(vertices, axis=0)
        angles = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
        turn = float(np.mean(np.diff(angles))) if len(angles) > 1 else 0.0
        centers = 0.5 * (vertices[:-1] + vertices[1:])
        center = centers.mean(axis=0)
        latent = PassageLatent(
            kind="polyline",
            center_x=float(center[0]),
            center_y=float(center[1]),
            direction=float(np.mean(angles)),
            length=float(sweep.normalized_path_length),
            spacing=turn,
            stroke_count=int(mark_count),
            width=float(width),
            amount=float(amount),
            tone=float(tone),
        )
        return fit_polyline_latent(latent)


_LIMIT_LOWER = np.asarray([limit[0] for limit in JOINT_LIMITS], dtype=np.float64)
_LIMIT_UPPER = np.asarray([limit[1] for limit in JOINT_LIMITS], dtype=np.float64)


def _ordered_range(bounds: tuple[float, float]) -> tuple[float, float]:
    low, high = float(bounds[0]), float(bounds[1])
    return (low, high) if low <= high else (high, low)


def _pose_from_joints(joints: np.ndarray) -> ArmPose:
    return ArmPose(
        yaw=float(joints[0]),
        pitch=float(joints[1]),
        roll=float(joints[2]),
        elbow=float(joints[3]),
    )


def _joints_from_pose(pose: ArmPose) -> np.ndarray:
    return np.asarray([pose.yaw, pose.pitch, pose.roll, pose.elbow], dtype=np.float64)


def _path_statistics(canvas_path: np.ndarray) -> tuple[float, float, float]:
    """Normalized path length, chord length, and total signed turn in radians."""

    segments = np.diff(canvas_path, axis=0)
    lengths = np.hypot(segments[:, 0], segments[:, 1])
    path_length = float(lengths.sum())
    chord = float(np.hypot(*(canvas_path[-1] - canvas_path[0])))
    moving = segments[lengths > 1e-12]
    if len(moving) > 1:
        angles = np.unwrap(np.arctan2(moving[:, 1], moving[:, 0]))
        turn = float(angles[-1] - angles[0])
    else:
        turn = 0.0
    return path_length, chord, turn


def _resample_equal_arclength(path: np.ndarray, count: int) -> np.ndarray:
    """Resample a polyline into `count` equally spaced-by-arclength vertices."""

    target = max(2, int(count))
    segments = np.diff(path, axis=0)
    lengths = np.hypot(segments[:, 0], segments[:, 1])
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.repeat(path[:1], target, axis=0)
    positions = np.linspace(0.0, total, target)
    x = np.interp(positions, cumulative, path[:, 0])
    y = np.interp(positions, cumulative, path[:, 1])
    return np.stack([x, y], axis=1)


def validated_families(config: PainterConfig) -> tuple[str, ...]:
    """Declared families, checked against the module's supported set."""

    declared = tuple(str(family) for family in config.bootstrap_manifold_families)
    if not declared:
        return MANIFOLD_FAMILIES
    unknown = [name for name in declared if name not in MANIFOLD_FAMILIES]
    if unknown:
        raise ValueError(
            f"unknown motion-manifold families {unknown}; supported: {MANIFOLD_FAMILIES}"
        )
    return declared


# --------------------------------------------------------------------- probes
# The three context-free reference canvases the bootstrap evidence block is
# measured against. They live here, beside the generator, so tests and the
# driver share ONE implementation and cannot silently diverge. None of them is a
# preference or a target: they are inputs to a MEASUREMENT of the
# canvas/relational likelihood after bootstrap.


def blank_probe_fields(config: PainterConfig, count: int = 4) -> np.ndarray:
    """A blank canvas's material fields.

    Exactly zeros for all six channels: an unpainted VerticalCanvas has zero
    thickness, wetness, and pigment mass, zero surface tone, zero coverage, and
    zero ground contrast (observed tone equals the ground tone).
    """

    return np.zeros(
        (
            max(1, int(count)),
            int(config.spatial_material_channels),
            int(config.spatial_grid_size),
            int(config.spatial_grid_size),
        ),
        dtype=np.float32,
    )


def shuffled_probe_fields(fields: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-sample cell permutation, the SAME permutation across channels.

    One of the acceptance criterion's two null models. It preserves each image's
    per-channel marginals exactly, so `flat_log_likelihood` is identical between
    a canvas and its shuffle and the comparison isolates the hierarchical code
    rather than the marginals. Mirrors `shuffle_cells` in
    tests/test_composition_hierarchy.py.

    MEASURED: it is the tighter null only while the code is undertrained. After
    2700 gradient steps the shuffle scores -74 nats, i.e. strongly
    out-of-distribution, while blank scores -0.05, so the criterion's
    `max(blank, shuffled)` selects BLANK. The max is therefore load-bearing: a
    shuffle-only criterion would report a margin earned by overconfidence.
    """

    array = np.asarray(fields, dtype=np.float32)
    count, channels, grid, _ = array.shape
    flat = array.reshape(count, channels, grid * grid).copy()
    for index in range(count):
        permutation = rng.permutation(grid * grid)
        flat[index] = flat[index][:, permutation]
    return flat.reshape(count, channels, grid, grid)


def iid_scatter_probe_fields(
    config: PainterConfig,
    rng: np.random.Generator,
    count: int = 8,
) -> np.ndarray:
    """Per-cell independent uniform material: no spatial organisation at all.

    Deliberately NOT matched to the bootstrapped canvases' marginals - that
    role belongs to `shuffled_probe_fields`. This probe asks a different
    question (is the code specific to organized material at all?) and both are
    reported, because a probe range computed against an out-of-distribution
    reference grows with model CONFIDENCE and not with discriminative validity.
    """

    return rng.uniform(
        0.0,
        1.0,
        size=(
            max(1, int(count)),
            int(config.spatial_material_channels),
            int(config.spatial_grid_size),
            int(config.spatial_grid_size),
        ),
    ).astype(np.float32)
