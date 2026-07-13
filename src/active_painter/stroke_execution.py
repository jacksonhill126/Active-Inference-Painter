from __future__ import annotations

import copy
from dataclasses import dataclass, fields, replace
from typing import Callable

import numpy as np

from .arm_control import ik_pose_for_canvas_point
from .arm_sim import ArmPainterSim, ArmPose, JOINT_NAMES, clip_scalar
from .env import StrokeAction
from .policies import MotorPrimitiveLatent


@dataclass(frozen=True, slots=True)
class StrokeTiming:
    approach: float = 0.82
    press: float = 0.48
    paint: float = 1.45
    lift: float = 0.58

    @property
    def total(self) -> float:
        return self.approach + self.press + self.paint + self.lift


@dataclass(frozen=True, slots=True)
class StrokeReference:
    phase: str
    t: float
    x: float
    z: float
    depth: float
    pressure: float
    brush_down: bool
    intended_start: tuple[float, float]
    intended_end: tuple[float, float]
    feasible: bool


@dataclass(frozen=True, slots=True)
class StrokeCommand:
    pose: ArmPose
    brush_down: bool
    intended_pressure: float
    reference: StrokeReference


@dataclass(frozen=True, slots=True)
class ExecutionForecast:
    next_state_mean: np.ndarray
    next_state_variance: np.ndarray
    canvas_delta_mean: np.ndarray
    intended_start: tuple[float, float]
    intended_end: tuple[float, float]
    realized_start: tuple[float, float]
    realized_end: tuple[float, float]
    intended_path_length: float
    realized_path_span: float
    paint_motion_fraction: float
    path_covariance: tuple[float, float]
    pressure_mean: float
    pressure_variance: float
    target_pressure_mean: float
    contact_loss_probability: float
    overshoot: float
    execution_uncertainty: float
    feasible: bool
    motor_primitive_kind: str = "cartesian_ik"
    joint_current_rms: float = 0.0
    joint_torque_rms: float = 0.0
    joint_velocity_rms: float = 0.0
    joint_acceleration_rms: float = 0.0
    joint_path_length_deg: float = 0.0
    joint_limit_proximity: float = 0.0
    joint_target_error_rms: float = 0.0
    proprioceptive_observation_dim: int = 0
    proprioceptive_labels: tuple[str, ...] = ()
    proprioceptive_mean: tuple[float, ...] = ()
    proprioceptive_predictive_variance: tuple[float, ...] = ()
    proprioceptive_likelihood_variance: tuple[float, ...] = ()
    motor_rollout_samples: int = 1
    feasibility_probability: float = 1.0

    def diagnostics(self, *, include_state_fields: bool = True) -> dict[str, object]:
        omitted = {"next_state_mean", "next_state_variance", "canvas_delta_mean"}
        payload = {
            field.name: _json_ready(getattr(self, field.name))
            for field in fields(self)
            if include_state_fields or field.name not in omitted
        }
        if not include_state_fields:
            payload.update(
                {
                    "state_vector_dim": int(self.next_state_mean.size),
                    "canvas_delta_abs_mean": float(np.mean(np.abs(self.canvas_delta_mean))),
                    "canvas_delta_abs_max": float(np.max(np.abs(self.canvas_delta_mean), initial=0.0)),
                    "state_fields_omitted": (
                        "dense material forecast arrays are retained for inference but omitted from runtime diagnostics"
                    ),
                }
            )
        return payload


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


CANVAS_REACH_FRACTION = 0.98
PAINT_SPEED_UNITS_PER_SECOND = 5.5


def stroke_world_endpoints(action: StrokeAction, canvas) -> tuple[float, float, float, float]:
    # The normalized stroke space maps to nearly the full canvas; genuinely
    # unreachable strokes are rejected per candidate by the motor feasibility
    # forecast rather than excluded by a blanket margin, which previously
    # capped achievable coverage (~65% of area) below the terminal preference.
    x0 = (action.x0 - 0.5) * canvas.width * CANVAS_REACH_FRACTION
    z0 = (0.5 - action.y0) * canvas.height * CANVAS_REACH_FRACTION
    x1 = (action.x1 - 0.5) * canvas.width * CANVAS_REACH_FRACTION
    z1 = (0.5 - action.y1) * canvas.height * CANVAS_REACH_FRACTION
    return x0, z0, x1, z1


def adaptive_stroke_timing(sim: ArmPainterSim, action: StrokeAction) -> StrokeTiming:
    """Phase durations scaled to the stroke's actual geometry.

    Fixed timing made distant stroke starts unreachable within the approach
    phase and swept long strokes faster than the servo could track, so paint
    gating dropped the middle of the mark. Approach time scales with the
    distance from the current tip to the stroke start; paint time bounds the
    sweep speed.
    """

    x0, z0, x1, z1 = stroke_world_endpoints(action, sim.canvas)
    tip = sim.kinematics.tip(sim.actual_pose)
    approach_distance = float(np.hypot(float(tip[0]) - x0, float(tip[2]) - z0))
    stroke_length = float(np.hypot(x1 - x0, z1 - z0))
    approach = clip_scalar(0.5 + approach_distance / 9.0, 0.6, 2.4)
    paint = clip_scalar(stroke_length / PAINT_SPEED_UNITS_PER_SECOND, 0.9, 3.6)
    return StrokeTiming(approach=approach, paint=paint)


def stroke_reference(action: StrokeAction, sim: ArmPainterSim, t: float, timing: StrokeTiming) -> StrokeReference:
    c = sim.canvas
    x0, z0, x1, z1 = stroke_world_endpoints(action, c)
    feasible = c.contains(x0, z0) and c.contains(x1, z1)
    pressure_base = 0.08 + 0.62 * clip_scalar(action.amount, 0.0, 1.0)
    curvature = float(np.hypot(x1 - x0, z1 - z0))
    speed_factor = clip_scalar(curvature / max(0.2, timing.paint), 0.0, 12.0) / 12.0
    t = clip_scalar(t, 0.0, timing.total)

    if t < timing.approach:
        u = t / timing.approach
        smooth = smootherstep(u)
        return StrokeReference(
            phase="approach",
            t=t,
            x=x0,
            z=z0,
            depth=c.distance - 1.08 + 0.96 * smooth,
            pressure=0.0,
            brush_down=False,
            intended_start=(x0, z0),
            intended_end=(x1, z1),
            feasible=feasible,
        )
    if t < timing.approach + timing.press:
        u = (t - timing.approach) / timing.press
        smooth = smootherstep(u)
        return StrokeReference(
            phase="press",
            t=t,
            x=x0,
            z=z0,
            # Press slightly past the canvas plane: real bushing deflection
            # keeps contact robust to servo depth undershoot, which otherwise
            # breaks the near-surface pressure gate at extended reach.
            depth=c.distance - 0.12 + 0.32 * smooth,
            pressure=float(smooth * pressure_base),
            brush_down=False,
            intended_start=(x0, z0),
            intended_end=(x1, z1),
            feasible=feasible,
        )
    if t < timing.approach + timing.press + timing.paint:
        u = (t - timing.approach - timing.press) / timing.paint
        smooth = smootherstep(u)
        phase_pressure = pressure_base * (0.72 + 0.28 * np.sin(np.pi * u) ** 2)
        width_pressure = 0.42 * clip_scalar(action.width / 0.30, 0.0, 1.0)
        consequence_pressure = clip_scalar(phase_pressure + width_pressure - 0.12 * speed_factor, 0.04, 0.92)
        return StrokeReference(
            phase="paint",
            t=t,
            x=float((1.0 - smooth) * x0 + smooth * x1),
            z=float((1.0 - smooth) * z0 + smooth * z1),
            depth=c.distance + 0.2,
            pressure=float(consequence_pressure),
            brush_down=True,
            intended_start=(x0, z0),
            intended_end=(x1, z1),
            feasible=feasible,
        )

    u = (t - timing.approach - timing.press - timing.paint) / timing.lift
    smooth = smootherstep(u)
    return StrokeReference(
        phase="lift",
        t=t,
        x=x1,
        z=z1,
        depth=c.distance + 0.2 - 1.06 * smooth,
        pressure=float(max(0.0, pressure_base * (1.0 - smooth))),
        brush_down=False,
        intended_start=(x0, z0),
        intended_end=(x1, z1),
        feasible=feasible,
    )


def pose_for_reference(reference: StrokeReference) -> ArmPose:
    return ik_pose_for_canvas_point(reference.x, reference.z, reference.depth)


class DirectStrokeController:
    def reset(self, sim: ArmPainterSim, action: StrokeAction, timing: StrokeTiming) -> None:
        _ = sim, action, timing

    def command(self, sim: ArmPainterSim, action: StrokeAction, t: float, dt: float, timing: StrokeTiming) -> StrokeCommand:
        _ = dt
        reference = stroke_reference(action, sim, t, timing)
        return StrokeCommand(
            pose=pose_for_reference(reference),
            brush_down=reference.brush_down,
            intended_pressure=reference.pressure,
            reference=reference,
        )


class ContactAwareStrokeController:
    """Conventional lower-level controller for realizing a selected stroke.

    The controller previews the stroke reference, rate-limits joint targets, and
    ramps contact before paint is enabled. It does not choose painting policy.
    """

    def __init__(
        self,
        preview_time: float = 0.22,
        filter_time: float = 0.18,
        max_joint_speed_deg: float = 72.0,
        paint_tracking_tolerance: float = 0.65,
        paint_engage_fraction: float = 0.045,
    ) -> None:
        self.preview_time = preview_time
        self.filter_time = filter_time
        self.max_joint_speed_deg = max_joint_speed_deg
        self.paint_tracking_tolerance = paint_tracking_tolerance
        self.paint_engage_fraction = paint_engage_fraction
        self._filtered_pose: ArmPose | None = None

    def reset(self, sim: ArmPainterSim, action: StrokeAction, timing: StrokeTiming) -> None:
        _ = action, timing
        self._filtered_pose = sim.actual_pose

    def command(self, sim: ArmPainterSim, action: StrokeAction, t: float, dt: float, timing: StrokeTiming) -> StrokeCommand:
        current_reference = stroke_reference(action, sim, t, timing)
        preview_reference = stroke_reference(action, sim, min(t + self.preview_time, timing.total), timing)
        # While the tip is far from the reference, travel with the brush
        # pulled back off the canvas and command a bounded Cartesian step (a
        # carrot) toward the target instead of the target itself: joint-space
        # interpolation toward distant targets can swing the tip through the
        # canvas plane, where the overtravel safety rollback wedges the arm in
        # place. Tracking IK solutions of nearby on-path points keeps
        # intermediate joint configurations close to the safe manifold. Normal
        # tracking lag during the paint phase stays within the paint tolerance
        # and never triggers pullback, so contact is not disturbed mid-stroke.
        tip = sim.kinematics.tip(sim.actual_pose)
        lateral_error = float(np.hypot(float(tip[0]) - current_reference.x, float(tip[2]) - current_reference.z))
        pullback_threshold = (
            self.paint_tracking_tolerance if current_reference.phase == "paint" else 0.35
        )
        travel_pullback = clip_scalar(0.9 * (lateral_error - pullback_threshold), 0.0, 1.8)
        target_x, target_z = preview_reference.x, preview_reference.z
        to_target_x = target_x - float(tip[0])
        to_target_z = target_z - float(tip[2])
        to_target = float(np.hypot(to_target_x, to_target_z))
        if travel_pullback > 0.0 and to_target > 2.0:
            scale = 2.0 / to_target
            target_x = float(tip[0]) + scale * to_target_x
            target_z = float(tip[2]) + scale * to_target_z
        desired_pose = ik_pose_for_canvas_point(
            target_x,
            target_z,
            preview_reference.depth - travel_pullback,
        )
        if self._filtered_pose is None:
            self._filtered_pose = sim.actual_pose
        alpha = 1.0 - float(np.exp(-dt / max(1e-5, self.filter_time)))
        pose = rate_limit_pose(
            interpolate_pose(self._filtered_pose, desired_pose, alpha),
            self._filtered_pose,
            max_delta=self.max_joint_speed_deg * dt,
        )
        self._filtered_pose = pose
        brush_down = current_reference.brush_down
        intended_pressure = current_reference.pressure
        if current_reference.phase == "paint":
            brush_down = self._paint_contact_is_ready(sim, current_reference)
            if not brush_down:
                intended_pressure = 0.0
        return StrokeCommand(
            pose=pose,
            brush_down=brush_down,
            intended_pressure=float(intended_pressure),
            reference=current_reference,
        )

    def _paint_contact_is_ready(self, sim: ArmPainterSim, reference: StrokeReference) -> bool:
        start = np.asarray(reference.intended_start, dtype=np.float64)
        end = np.asarray(reference.intended_end, dtype=np.float64)
        current = np.asarray([reference.x, reference.z], dtype=np.float64)
        tip = sim.kinematics.tip(sim.actual_pose)
        actual = np.asarray([float(tip[0]), float(tip[2])], dtype=np.float64)
        intended_length = float(np.linalg.norm(end - start))
        if intended_length < 0.18:
            return False
        min_progress = max(0.035, self.paint_engage_fraction * intended_length)
        reference_progress = float(np.linalg.norm(current - start))
        actual_progress = float(np.linalg.norm(actual - start))
        lateral_error = float(np.linalg.norm(actual - current))
        return (
            reference.feasible
            and reference_progress >= min_progress
            and actual_progress >= 0.45 * min_progress
            and lateral_error <= self.paint_tracking_tolerance
        )


class JointSpaceStrokeController:
    """Conventional motor primitive controller for joint-space mark realization.

    Unlike the Cartesian IK controller, the paint phase is generated as a
    joint-space trajectory between contact poses. The painting policy is still
    selected by active inference; this lower-level controller only exposes a
    bodily realization whose canvas and proprioceptive outcomes can be
    forecast as part of policy inference.
    """

    def __init__(
        self,
        kind: str = "joint_spline",
        filter_time: float = 0.16,
        max_joint_speed_deg: float = 72.0,
        paint_tracking_tolerance: float = 0.85,
        paint_engage_fraction: float = 0.045,
    ) -> None:
        self.kind = kind
        self.filter_time = filter_time
        self.max_joint_speed_deg = max_joint_speed_deg
        self.paint_tracking_tolerance = paint_tracking_tolerance
        self.paint_engage_fraction = paint_engage_fraction
        self._filtered_pose: ArmPose | None = None
        self._start_pose: ArmPose | None = None
        self._end_pose: ArmPose | None = None
        self._intended_start: tuple[float, float] = (0.0, 0.0)
        self._intended_end: tuple[float, float] = (0.0, 0.0)

    def reset(self, sim: ArmPainterSim, action: StrokeAction, timing: StrokeTiming) -> None:
        self._filtered_pose = sim.actual_pose
        start_ref = stroke_reference(action, sim, timing.approach + timing.press, timing)
        end_ref = stroke_reference(action, sim, timing.approach + timing.press + timing.paint, timing)
        self._start_pose = pose_for_reference(start_ref)
        self._end_pose = pose_for_reference(end_ref)
        start_tip = sim.kinematics.tip(self._start_pose)
        end_tip = sim.kinematics.tip(self._end_pose)
        self._intended_start = (float(start_tip[0]), float(start_tip[2]))
        self._intended_end = (float(end_tip[0]), float(end_tip[2]))

    def command(self, sim: ArmPainterSim, action: StrokeAction, t: float, dt: float, timing: StrokeTiming) -> StrokeCommand:
        reference = stroke_reference(action, sim, t, timing)
        desired_pose = pose_for_reference(reference)
        intended_start = reference.intended_start
        intended_end = reference.intended_end
        if reference.phase == "paint":
            u = (t - timing.approach - timing.press) / max(1e-6, timing.paint)
            desired_pose = self._paint_pose(clip_scalar(u, 0.0, 1.0))
            tip = sim.kinematics.tip(desired_pose)
            intended_start = self._intended_start
            intended_end = self._intended_end
            reference = StrokeReference(
                phase="paint",
                t=reference.t,
                x=float(tip[0]),
                z=float(tip[2]),
                depth=float(tip[1]),
                pressure=reference.pressure,
                brush_down=True,
                intended_start=intended_start,
                intended_end=intended_end,
                feasible=reference.feasible and sim.canvas.contains(float(tip[0]), float(tip[2])),
            )

        if self._filtered_pose is None:
            self._filtered_pose = sim.actual_pose
        alpha = 1.0 - float(np.exp(-dt / max(1e-5, self.filter_time)))
        pose = rate_limit_pose(
            interpolate_pose(self._filtered_pose, desired_pose, alpha),
            self._filtered_pose,
            max_delta=self.max_joint_speed_deg * dt,
        )
        self._filtered_pose = pose
        brush_down = reference.brush_down
        intended_pressure = reference.pressure
        if reference.phase == "paint":
            brush_down = self._paint_contact_is_ready(sim, reference)
            if not brush_down:
                intended_pressure = 0.0
        return StrokeCommand(
            pose=pose,
            brush_down=brush_down,
            intended_pressure=float(intended_pressure),
            reference=reference,
        )

    def _paint_pose(self, u: float) -> ArmPose:
        if self._start_pose is None or self._end_pose is None:
            raise RuntimeError("Joint-space controller must be reset before command().")
        smooth = smootherstep(u)
        if self.kind == "elbow_pivot":
            joint_alpha = {
                "yaw": smootherstep(smooth**1.65),
                "pitch": smootherstep(smooth**1.55),
                "roll": smootherstep(smooth**1.30),
                "elbow": smootherstep(smooth**0.72),
            }
        elif self.kind == "shoulder_yaw_arc":
            joint_alpha = {
                "yaw": smootherstep(smooth**0.72),
                "pitch": smootherstep(smooth**1.45),
                "roll": smootherstep(smooth**1.35),
                "elbow": smootherstep(smooth**1.45),
            }
        else:
            joint_alpha = dict.fromkeys(JOINT_NAMES, smooth)
        return ArmPose(
            **{
                name: float(
                    (1.0 - float(joint_alpha[name])) * getattr(self._start_pose, name)
                    + float(joint_alpha[name]) * getattr(self._end_pose, name)
                )
                for name in JOINT_NAMES
            }
        ).clipped()

    def _paint_contact_is_ready(self, sim: ArmPainterSim, reference: StrokeReference) -> bool:
        start = np.asarray(reference.intended_start, dtype=np.float64)
        end = np.asarray(reference.intended_end, dtype=np.float64)
        current = np.asarray([reference.x, reference.z], dtype=np.float64)
        tip = sim.kinematics.tip(sim.actual_pose)
        actual = np.asarray([float(tip[0]), float(tip[2])], dtype=np.float64)
        intended_length = float(np.linalg.norm(end - start))
        if intended_length < 0.18:
            return False
        min_progress = max(0.035, self.paint_engage_fraction * intended_length)
        reference_progress = float(np.linalg.norm(current - start))
        actual_progress = float(np.linalg.norm(actual - start))
        lateral_error = float(np.linalg.norm(actual - current))
        near_surface = bool(sim.contact.on_canvas and sim.contact.pressure > 0.005)
        return (
            reference.feasible
            and near_surface
            and reference_progress >= min_progress
            and actual_progress >= 0.35 * min_progress
            and lateral_error <= self.paint_tracking_tolerance
        )


def controller_for_motor_primitive(
    motor_primitive: MotorPrimitiveLatent | None,
) -> ContactAwareStrokeController | DirectStrokeController | JointSpaceStrokeController:
    kind = "cartesian_ik" if motor_primitive is None else motor_primitive.kind
    if kind in ("cartesian", "cartesian_ik", ""):
        return ContactAwareStrokeController()
    return JointSpaceStrokeController(kind=kind)


def _forecast_stroke_execution_once(
    sim: ArmPainterSim,
    action: StrokeAction,
    summary_fn: Callable[[ArmPainterSim], np.ndarray],
    timing: StrokeTiming | None = None,
    controller: ContactAwareStrokeController | DirectStrokeController | None = None,
    motor_primitive: MotorPrimitiveLatent | None = None,
    dt: float = 1.0 / 90.0,
    noise_sample_index: int = 0,
) -> ExecutionForecast:
    timing = timing or adaptive_stroke_timing(sim, action)
    controller = controller or controller_for_motor_primitive(motor_primitive)
    motor_primitive_kind = "cartesian_ik" if motor_primitive is None else motor_primitive.kind
    working = copy.deepcopy(sim)
    working.plant.select_forecast_noise_sample(noise_sample_index)
    before_state = summary_fn(working)
    controller.reset(working, action, timing)

    intended_points: list[tuple[float, float]] = []
    realized_points: list[tuple[float, float]] = []
    pressures: list[float] = []
    target_pressures: list[float] = []
    pose_samples: list[np.ndarray] = []
    velocity_samples: list[np.ndarray] = []
    current_samples: list[np.ndarray] = []
    torque_samples: list[np.ndarray] = []
    acceleration_samples: list[np.ndarray] = []
    target_error_samples: list[np.ndarray] = []
    limit_proximity_samples: list[np.ndarray] = []
    encoder_std_samples: list[np.ndarray] = []
    process_torque_samples: list[np.ndarray] = []
    contact_losses = 0
    paint_samples = 0
    feasible = True
    previous_velocity: np.ndarray | None = None
    t = 0.0
    while t < timing.total:
        t = min(timing.total, t + dt)
        command = controller.command(working, action, t, dt, timing)
        feasible = feasible and command.reference.feasible
        working.set_target(command.pose)
        working.paint_enabled = command.brush_down
        working.intended_contact_pressure = command.intended_pressure
        working.brush_tone = float(action.tone >= 0.5)
        working.step(dt)
        pose_vec = _pose_vector(working.actual_pose)
        target_vec = _pose_vector(working.target_pose)
        velocity_vec = np.asarray([working.plant.velocity[name] for name in JOINT_NAMES], dtype=np.float64)
        current_vec = np.asarray([working.plant.telemetry.current[name] for name in JOINT_NAMES], dtype=np.float64)
        torque_vec = np.asarray([working.plant.telemetry.torque[name] for name in JOINT_NAMES], dtype=np.float64)
        pose_samples.append(pose_vec)
        velocity_samples.append(velocity_vec)
        current_samples.append(current_vec)
        torque_samples.append(torque_vec)
        target_error_samples.append(target_vec - pose_vec)
        limit_proximity_samples.append(
            _joint_limit_proximity_vector(working.actual_pose, working.config.motor_limit_margin_degrees)
        )
        encoder_std_samples.append(
            np.asarray(
                [working.plant.telemetry.encoder_std_deg[name] for name in JOINT_NAMES],
                dtype=np.float64,
            )
        )
        process_torque_samples.append(
            np.asarray(
                [working.plant.telemetry.process_torque[name] for name in JOINT_NAMES],
                dtype=np.float64,
            )
        )
        if previous_velocity is not None:
            acceleration_samples.append((velocity_vec - previous_velocity) / max(1e-6, dt))
        previous_velocity = velocity_vec
        if command.reference.phase == "paint":
            paint_samples += 1
            intended_points.append((command.reference.x, command.reference.z))
            realized_points.append((float(working.contact.brush_world[0]), float(working.contact.brush_world[2])))
            pressures.append(float(working.contact.pressure))
            target_pressures.append(command.reference.pressure)
            if not command.brush_down or working.contact.pressure <= 0.01:
                contact_losses += 1

    after_state = summary_fn(working)
    intended_start = stroke_reference(action, working, timing.approach + timing.press, timing).intended_start
    intended_end = stroke_reference(action, working, timing.approach + timing.press + timing.paint, timing).intended_end
    intended_path_length = float(np.linalg.norm(np.asarray(intended_end) - np.asarray(intended_start)))

    if realized_points:
        realized = np.asarray(realized_points, dtype=np.float64)
        intended = np.asarray(intended_points, dtype=np.float64)
        errors = realized - intended
        path_cov = np.var(errors, axis=0)
        path_rmse = float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))
        realized_start = tuple(float(x) for x in realized[0])
        realized_end = tuple(float(x) for x in realized[-1])
        realized_path_span = float(np.linalg.norm(realized[-1] - realized[0]))
        overshoot = float(max(np.linalg.norm(realized[0] - np.asarray(intended_start)), path_rmse))
    else:
        path_cov = np.asarray([1.0, 1.0], dtype=np.float64)
        path_rmse = 1.0
        realized_start = intended_start
        realized_end = intended_end
        realized_path_span = 0.0
        overshoot = 1.0
        feasible = False

    pressure_arr = np.asarray(pressures or [0.0], dtype=np.float64)
    target_pressure_arr = np.asarray(target_pressures or [0.0], dtype=np.float64)
    pressure_error = float(np.sqrt(np.mean((pressure_arr - target_pressure_arr) ** 2)))
    contact_loss_probability = float(contact_losses / max(1, paint_samples))
    execution_uncertainty = float(path_rmse + 0.35 * pressure_error + 0.55 * contact_loss_probability)
    minimum_realized_span = min(0.35, max(0.08, 0.2 * intended_path_length))
    paint_motion_fraction = realized_path_span / max(1e-6, intended_path_length)
    feasible = (
        feasible
        and contact_loss_probability < 0.85
        and intended_path_length >= 0.18
        and realized_path_span >= minimum_realized_span
    )

    variance = np.full_like(after_state, 1e-6, dtype=np.float32)
    variance[0] += np.float32((0.015 * execution_uncertainty + 0.025 * contact_loss_probability) ** 2)
    variance[1:] += np.float32((0.01 * execution_uncertainty) ** 2)

    current_by_joint = _sample_rms_by_joint(current_samples) / max(1e-6, working.plant.current_limit)
    torque_by_joint = _sample_rms_by_joint(torque_samples) / max(
        1e-6, working.plant.kt * working.plant.current_limit
    )
    velocity_by_joint = _sample_rms_by_joint(velocity_samples) / max(1e-6, working.plant.max_link_velocity)
    joint_inertias = np.asarray(
        [
            working.plant._joint_param(working.plant.link_inertia, name, working.plant.inertia)
            + working.plant._joint_param(working.plant.motor_inertia, name, 0.0)
            for name in JOINT_NAMES
        ],
        dtype=np.float64,
    )
    acceleration_scales = working.plant.kt * working.plant.current_limit / np.maximum(joint_inertias, 1e-6)
    acceleration_by_joint = _sample_rms_by_joint(acceleration_samples) / acceleration_scales
    target_error_by_joint = _sample_rms_by_joint(target_error_samples) / 45.0
    limit_by_joint = np.mean(np.stack(limit_proximity_samples), axis=0) if limit_proximity_samples else np.zeros(4)
    joint_current_rms = _sample_rms(current_samples) / max(1e-6, working.plant.current_limit)
    joint_torque_rms = _sample_rms(torque_samples) / max(1e-6, working.plant.kt * working.plant.current_limit)
    joint_velocity_rms = _sample_rms(velocity_samples)
    joint_acceleration_rms = _sample_rms(acceleration_samples)
    joint_target_error_rms = _sample_rms(target_error_samples)
    joint_path_length_deg = _joint_path_length(pose_samples)
    joint_limit_proximity = float(limit_by_joint.mean())
    encoder_std_by_joint = (
        np.mean(np.stack(encoder_std_samples), axis=0)
        if encoder_std_samples
        else np.full(len(JOINT_NAMES), working.plant.encoder_base_noise_deg)
    )
    process_torque_std = np.asarray(
        [working.plant._joint_param(working.plant.process_torque_noise_std, name, 0.0) for name in JOINT_NAMES],
        dtype=np.float64,
    )
    current_sensor_variance = np.full(len(JOINT_NAMES), 0.02**2, dtype=np.float64)
    torque_sensor_variance = np.full(len(JOINT_NAMES), 0.02**2, dtype=np.float64)
    velocity_sensor_variance = np.full(
        len(JOINT_NAMES),
        (np.deg2rad(working.plant.encoder_velocity_noise_deg) / max(1e-6, working.plant.max_link_velocity)) ** 2,
        dtype=np.float64,
    )
    acceleration_sensor_variance = (
        process_torque_std / np.maximum(joint_inertias * acceleration_scales, 1e-6)
    ) ** 2
    target_sensor_variance = (encoder_std_by_joint / 45.0) ** 2
    limit_sensor_variance = (
        encoder_std_by_joint / max(1e-6, working.config.motor_limit_margin_degrees)
    ) ** 2
    contact_likelihood_variance = 0.02**2 + contact_loss_probability * (1.0 - contact_loss_probability) / max(
        1, paint_samples
    )
    path_error_norm = path_rmse / max(1e-6, working.canvas.width)
    pressure_error_norm = pressure_error
    cartesian_encoder_std = (
        np.deg2rad(float(encoder_std_by_joint.mean()))
        * (working.kinematics.upper_arm + working.kinematics.lower_arm)
        / max(1e-6, working.canvas.width)
    )
    labels = tuple(
        [f"current_{name}" for name in JOINT_NAMES]
        + [f"torque_{name}" for name in JOINT_NAMES]
        + [f"velocity_{name}" for name in JOINT_NAMES]
        + [f"acceleration_{name}" for name in JOINT_NAMES]
        + [f"target_error_{name}" for name in JOINT_NAMES]
        + [f"limit_proximity_{name}" for name in JOINT_NAMES]
        + ["contact_loss", "pressure_error", "path_error"]
    )
    proprioceptive_mean = np.concatenate(
        [
            current_by_joint,
            torque_by_joint,
            velocity_by_joint,
            acceleration_by_joint,
            target_error_by_joint,
            limit_by_joint,
            np.asarray([contact_loss_probability, pressure_error_norm, path_error_norm]),
        ]
    )
    likelihood_variance = np.concatenate(
        [
            current_sensor_variance,
            torque_sensor_variance,
            velocity_sensor_variance,
            acceleration_sensor_variance,
            target_sensor_variance,
            limit_sensor_variance,
            np.asarray([contact_likelihood_variance, 0.02**2, max(cartesian_encoder_std**2, 1e-8)]),
        ]
    )

    return ExecutionForecast(
        next_state_mean=after_state.astype(np.float32),
        next_state_variance=variance,
        canvas_delta_mean=(after_state - before_state).astype(np.float32),
        intended_start=tuple(float(x) for x in intended_start),
        intended_end=tuple(float(x) for x in intended_end),
        realized_start=realized_start,
        realized_end=realized_end,
        intended_path_length=intended_path_length,
        realized_path_span=realized_path_span,
        paint_motion_fraction=float(paint_motion_fraction),
        path_covariance=(float(path_cov[0]), float(path_cov[1])),
        pressure_mean=float(pressure_arr.mean()),
        pressure_variance=float(pressure_arr.var()),
        target_pressure_mean=float(target_pressure_arr.mean()),
        contact_loss_probability=contact_loss_probability,
        overshoot=overshoot,
        execution_uncertainty=execution_uncertainty,
        feasible=feasible,
        motor_primitive_kind=motor_primitive_kind,
        joint_current_rms=float(joint_current_rms),
        joint_torque_rms=float(joint_torque_rms),
        joint_velocity_rms=float(joint_velocity_rms),
        joint_acceleration_rms=float(joint_acceleration_rms),
        joint_path_length_deg=float(joint_path_length_deg),
        joint_limit_proximity=joint_limit_proximity,
        joint_target_error_rms=float(joint_target_error_rms),
        proprioceptive_observation_dim=len(labels),
        proprioceptive_labels=labels,
        proprioceptive_mean=tuple(float(value) for value in proprioceptive_mean),
        proprioceptive_predictive_variance=tuple(0.0 for _ in labels),
        proprioceptive_likelihood_variance=tuple(float(max(value, 1e-8)) for value in likelihood_variance),
        motor_rollout_samples=1,
        feasibility_probability=float(feasible),
    )


def forecast_stroke_execution(
    sim: ArmPainterSim,
    action: StrokeAction,
    summary_fn: Callable[[ArmPainterSim], np.ndarray],
    timing: StrokeTiming | None = None,
    controller: ContactAwareStrokeController | DirectStrokeController | JointSpaceStrokeController | None = None,
    motor_primitive: MotorPrimitiveLatent | None = None,
    dt: float = 1.0 / 90.0,
    rollout_samples: int | None = None,
) -> ExecutionForecast:
    """Monte Carlo predictive density over canvas and proprioceptive outcomes."""

    sample_count = max(
        1,
        int(sim.config.motor_forecast_samples if rollout_samples is None else rollout_samples),
    )
    forecasts = [
        _forecast_stroke_execution_once(
            sim,
            action,
            summary_fn,
            timing=timing,
            controller=controller,
            motor_primitive=motor_primitive,
            dt=dt,
            noise_sample_index=index,
        )
        for index in range(sample_count)
    ]
    if sample_count == 1:
        return forecasts[0]

    base = forecasts[0]
    state_samples = np.stack([forecast.next_state_mean for forecast in forecasts]).astype(np.float64)
    state_mean = state_samples.mean(axis=0)
    state_within = np.stack([forecast.next_state_variance for forecast in forecasts]).astype(np.float64)
    state_variance = np.mean(state_within + (state_samples - state_mean) ** 2, axis=0)
    proprio_samples = np.asarray([forecast.proprioceptive_mean for forecast in forecasts], dtype=np.float64)
    proprio_mean = proprio_samples.mean(axis=0)
    proprio_variance = proprio_samples.var(axis=0)
    likelihood_variance = np.asarray(
        [forecast.proprioceptive_likelihood_variance for forecast in forecasts],
        dtype=np.float64,
    ).mean(axis=0)
    realized_starts = np.asarray([forecast.realized_start for forecast in forecasts], dtype=np.float64)
    realized_ends = np.asarray([forecast.realized_end for forecast in forecasts], dtype=np.float64)
    within_path_covariance = np.asarray([forecast.path_covariance for forecast in forecasts], dtype=np.float64)
    between_path_covariance = 0.5 * (realized_starts.var(axis=0) + realized_ends.var(axis=0))
    path_covariance = within_path_covariance.mean(axis=0) + between_path_covariance
    pressure_means = np.asarray([forecast.pressure_mean for forecast in forecasts], dtype=np.float64)
    pressure_variance = float(
        np.mean([forecast.pressure_variance for forecast in forecasts]) + pressure_means.var()
    )
    feasibility_probability = float(np.mean([forecast.feasible for forecast in forecasts]))

    def mean_field(name: str) -> float:
        return float(np.mean([float(getattr(forecast, name)) for forecast in forecasts]))

    return replace(
        base,
        next_state_mean=state_mean.astype(np.float32),
        next_state_variance=np.clip(state_variance, 1e-8, None).astype(np.float32),
        canvas_delta_mean=np.mean(
            np.stack([forecast.canvas_delta_mean for forecast in forecasts]),
            axis=0,
        ).astype(np.float32),
        realized_start=tuple(float(value) for value in realized_starts.mean(axis=0)),
        realized_end=tuple(float(value) for value in realized_ends.mean(axis=0)),
        realized_path_span=mean_field("realized_path_span"),
        paint_motion_fraction=mean_field("paint_motion_fraction"),
        path_covariance=(float(path_covariance[0]), float(path_covariance[1])),
        pressure_mean=float(pressure_means.mean()),
        pressure_variance=pressure_variance,
        target_pressure_mean=mean_field("target_pressure_mean"),
        contact_loss_probability=mean_field("contact_loss_probability"),
        overshoot=mean_field("overshoot"),
        execution_uncertainty=mean_field("execution_uncertainty"),
        feasible=feasibility_probability >= 0.5,
        joint_current_rms=mean_field("joint_current_rms"),
        joint_torque_rms=mean_field("joint_torque_rms"),
        joint_velocity_rms=mean_field("joint_velocity_rms"),
        joint_acceleration_rms=mean_field("joint_acceleration_rms"),
        joint_path_length_deg=mean_field("joint_path_length_deg"),
        joint_limit_proximity=mean_field("joint_limit_proximity"),
        joint_target_error_rms=mean_field("joint_target_error_rms"),
        proprioceptive_mean=tuple(float(value) for value in proprio_mean),
        proprioceptive_predictive_variance=tuple(float(max(value, 0.0)) for value in proprio_variance),
        proprioceptive_likelihood_variance=tuple(float(max(value, 1e-8)) for value in likelihood_variance),
        motor_rollout_samples=sample_count,
        feasibility_probability=feasibility_probability,
    )


def smootherstep(u: float) -> float:
    u = clip_scalar(u, 0.0, 1.0)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def interpolate_pose(a: ArmPose, b: ArmPose, alpha: float) -> ArmPose:
    alpha = clip_scalar(alpha, 0.0, 1.0)
    return ArmPose(
        **{
            name: float((1.0 - alpha) * getattr(a, name) + alpha * getattr(b, name))
            for name in JOINT_NAMES
        }
    ).clipped()


def rate_limit_pose(target: ArmPose, previous: ArmPose, max_delta: float) -> ArmPose:
    return ArmPose(
        **{
            name: float(
                getattr(previous, name)
                + clip_scalar(getattr(target, name) - getattr(previous, name), -max_delta, max_delta)
            )
            for name in JOINT_NAMES
        }
    ).clipped()


def _pose_vector(pose: ArmPose) -> np.ndarray:
    return np.asarray([getattr(pose, name) for name in JOINT_NAMES], dtype=np.float64)


def _sample_rms(samples: list[np.ndarray]) -> float:
    if not samples:
        return 0.0
    values = np.stack(samples).astype(np.float64)
    return float(np.sqrt(np.mean(values * values)))


def _sample_rms_by_joint(samples: list[np.ndarray]) -> np.ndarray:
    if not samples:
        return np.zeros(len(JOINT_NAMES), dtype=np.float64)
    values = np.stack(samples).astype(np.float64)
    return np.sqrt(np.mean(values * values, axis=0))


def _joint_path_length(samples: list[np.ndarray]) -> float:
    if len(samples) < 2:
        return 0.0
    values = np.stack(samples).astype(np.float64)
    return float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())


def _joint_limit_proximity(pose: ArmPose, margin_degrees: float) -> float:
    return float(_joint_limit_proximity_vector(pose, margin_degrees).mean())


def _joint_limit_proximity_vector(pose: ArmPose, margin_degrees: float) -> np.ndarray:
    margin = max(1e-6, float(margin_degrees))
    limits = {
        "yaw": (-90.0, 90.0),
        "pitch": (-90.0, 90.0),
        "roll": (-180.0, 180.0),
        "elbow": (0.0, 150.0),
    }
    proximity = []
    for name, (lo, hi) in limits.items():
        value = float(getattr(pose.clipped(), name))
        distance = min(value - lo, hi - value)
        proximity.append(clip_scalar((margin - distance) / margin, 0.0, 1.0))
    return np.asarray(proximity, dtype=np.float64)
