from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import threading
import time
from typing import Any, Callable

import numpy as np
import torch

from .action_encoding import encoded_action_vector
from .agent import ActiveInferencePainter
from .arm_control import ik_pose_for_canvas_point
from .arm_sim import ArmPainterSim, ArmPose, JOINT_NAMES
from .config import PainterConfig
from .efe import EFEComponents
from .env import StrokeAction
from .models import GaussianBelief
from .motor_planning import motor_efe_terms, motor_policy_log_prior, motor_realization_policy_alternatives
from .policies import MotorPrimitiveLatent, PassageLatent, PassagePlanLatent, Policy, policy_stop_log_prior
from .spatial_agent import SpatialActiveInferencePainter
from .spatial_efe import SpatialEFEComponents
from .spatial_hierarchy import infer_mark_event_belief
from .local_spatial import pixel_material_from_state
from .spatial_state import MATERIAL_CHANNELS, SpatialCanvasState, spatial_canvas_state, spatial_state_diagnostics
from .stroke_execution import (
    ContactAwareStrokeController,
    ExecutionForecast,
    StrokeTiming,
    adaptive_stroke_timing,
    controller_for_motor_primitive,
    forecast_stroke_execution,
    pose_for_reference,
    rate_limit_pose,
    stroke_world_endpoints,
    stroke_reference,
)


@dataclass(slots=True)
class StrokeExecution:
    action: StrokeAction
    efe: EFEComponents | SpatialEFEComponents
    posterior: float
    initial_state: np.ndarray | SpatialCanvasState | None = None
    forecast: ExecutionForecast | None = None
    motor_primitive: MotorPrimitiveLatent | None = None
    timing: StrokeTiming = field(default_factory=StrokeTiming)
    controller: ContactAwareStrokeController = field(default_factory=ContactAwareStrokeController)
    initialized: bool = False
    t: float = 0.0

    @property
    def approach(self) -> float:
        return self.timing.approach

    @property
    def press(self) -> float:
        return self.timing.press

    @property
    def paint(self) -> float:
        return self.timing.paint

    @property
    def lift(self) -> float:
        return self.timing.lift

    @property
    def total(self) -> float:
        return self.timing.total


@dataclass(slots=True)
class ArmActiveInferenceDriver:
    config: PainterConfig = field(
        default_factory=lambda: PainterConfig(
            canvas_size=64,
            candidate_policies=80,
            planning_horizon=3,
            policy_precision=0.35,
            batch_size=32,
        )
    )
    bootstrap_transitions: int = 96
    bootstrap_train_steps: int = 180
    checkpoint_path: Path | str | None = None
    checkpoint_save_every_transitions: int = 1
    enabled: bool = True
    on_stop: Callable[[], None] | None = None
    agent: ActiveInferencePainter | SpatialActiveInferencePainter = field(init=False)
    belief: GaussianBelief | SpatialCanvasState = field(init=False)
    current: StrokeExecution | None = field(default=None, init=False)
    stopped: bool = field(default=False, init=False)
    last_ranked: list[tuple[Policy, EFEComponents | SpatialEFEComponents, float]] = field(default_factory=list, init=False)
    last_components: EFEComponents | SpatialEFEComponents | None = field(default=None, init=False)
    stroke_count: int = field(default=0, init=False)
    trained_transitions: int = field(default=0, init=False)
    last_training_loss: float | None = field(default=None, init=False)
    last_training_seconds: float = field(default=0.0, init=False)
    checkpoint_status: str = field(default="disabled", init=False)
    checkpoint_loaded: bool = field(default=False, init=False)
    checkpoint_last_saved: str | None = field(default=None, init=False)
    checkpoint_last_error: str | None = field(default=None, init=False)
    checkpoint_architecture: dict[str, object] = field(default_factory=dict, init=False)
    last_stop_blocked: bool = field(default=False, init=False)
    last_execution_forecast: ExecutionForecast | None = field(default=None, init=False)
    last_motor_rejections: int = field(default=0, init=False)
    last_motor_primitive_candidates: int = field(default=0, init=False)
    planning: bool = field(default=False, init=False)
    last_planning_seconds: float = field(default=0.0, init=False)
    last_planning_profile: dict[str, object] = field(default_factory=dict, init=False)
    _planning_profile_current: dict[str, object] | None = field(default=None, init=False)
    _planning_forecast_cache: dict[tuple[object, ...], ExecutionForecast] = field(default_factory=dict, init=False)
    _planning_started_at: float | None = field(default=None, init=False)
    _planner_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _planner_thread: threading.Thread | None = field(default=None, init=False)
    _pending_current: StrokeExecution | None = field(default=None, init=False)
    _pending_stopped: bool = field(default=False, init=False)
    _pending_ranked: list[tuple[Policy, EFEComponents | SpatialEFEComponents, float]] | None = field(default=None, init=False)
    _pending_components: EFEComponents | SpatialEFEComponents | None = field(default=None, init=False)
    _pending_passage_queue: tuple[StrokeAction, ...] = field(default_factory=tuple, init=False)
    _pending_passage: PassageLatent | None = field(default=None, init=False)
    _pending_passage_plan: PassagePlanLatent | None = field(default=None, init=False)
    _pending_error: str | None = field(default=None, init=False)
    _transition_to_learn: tuple[
        np.ndarray | SpatialCanvasState,
        StrokeAction,
        MotorPrimitiveLatent | None,
        np.ndarray | SpatialCanvasState,
    ] | None = field(default=None, init=False)
    _post_stroke_retract_remaining: float = field(default=0.0, init=False)
    _hold_pose: ArmPose | None = field(default=None, init=False)
    _hold_command_pose: ArmPose | None = field(default=None, init=False)
    _hold_command_velocity: dict[str, float] = field(default_factory=dict, init=False)
    _hold_scope: str = field(default="global", init=False)
    _passage_queue: list[StrokeAction] = field(default_factory=list, init=False)
    _active_passage: PassageLatent | None = field(default=None, init=False)
    _active_passage_plan: PassagePlanLatent | None = field(default=None, init=False)
    _active_passage_total_strokes: int = field(default=0, init=False)
    _active_passage_completed_strokes: int = field(default=0, init=False)
    _contact_release_count: int = field(default=0, init=False)
    _cached_belief_gap: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self._uses_spatial_planner():
            self.agent = SpatialActiveInferencePainter(self.config, seed=17, device="cpu")
        else:
            self.agent = ActiveInferencePainter(self.config, seed=17, device="cpu")
        self.belief = self.agent.belief
        self.checkpoint_architecture = self._checkpoint_architecture_metadata()
        loaded = self._load_checkpoint_if_available()
        if self.bootstrap_transitions > 0 and not loaded:
            self.bootstrap_dynamics()
            self._save_checkpoint_if_due(force=True)

    def bootstrap_dynamics(self) -> None:
        sim = ArmPainterSim(replace(self.config))
        for i in range(self.bootstrap_transitions):
            current_state = self._planner_state(sim)
            if self._state_coverage(current_state) > 0.94 or i % 24 == 23:
                sim.reset_pose()
                sim.canvas.clear()
            action = self.agent.policy_sampler._stroke()
            state = self._planner_state(sim)
            execute_stroke_action(sim, action, dt=1.0 / 90.0)
            next_state = self._planner_state(sim)
            self._add_transition_to_agent(state, action, next_state)
            self.trained_transitions += 1
            if len(self.agent.replay) >= self.config.batch_size and i % 4 == 0:
                self.last_training_loss = self.agent.train_dynamics(gradient_steps=2)
        if len(self.agent.replay) >= self.config.batch_size:
            self.last_training_loss = self.agent.train_dynamics(gradient_steps=self.bootstrap_train_steps)

    def _checkpoint_architecture_metadata(self) -> dict[str, object]:
        cfg = self.config
        return {
            "schema_version": 1,
            "agent_kind": "spatial_material" if self._uses_spatial_planner() else "summary",
            "state_dim": cfg.state_dim,
            "action_dim": cfg.action_dim,
            "planner_state_kind": cfg.planner_state_kind,
            "spatial_grid_size": cfg.spatial_grid_size,
            "material_pyramid_levels": tuple(cfg.material_pyramid_levels),
            "spatial_material_channels": cfg.spatial_material_channels,
            "spatial_action_channels": cfg.spatial_action_channels,
            "spatial_transition_mode": cfg.spatial_transition_mode,
            "spatial_hidden_channels": cfg.spatial_hidden_channels,
            "spatial_residual_blocks": cfg.spatial_residual_blocks,
            "spatial_ensemble_size": cfg.spatial_ensemble_size,
            "ensemble_size": cfg.ensemble_size,
            "hidden_dim": cfg.hidden_dim,
            "composition_enabled": bool(
                self._uses_spatial_planner() and cfg.composition_gap_precision > 0.0
            ),
            "composition_latent_dim": cfg.composition_latent_dim,
            "composition_hidden_channels": cfg.composition_hidden_channels,
            "motor_realization_kinds": tuple(cfg.motor_realization_kinds),
            "thickness_scale": cfg.thickness_scale,
            "canvas_ground_tone": cfg.canvas_ground_tone,
        }

    def _checkpoint_file(self) -> Path | None:
        if self.checkpoint_path is None:
            self.checkpoint_status = "disabled"
            return None
        return Path(self.checkpoint_path)

    def _load_checkpoint_if_available(self) -> bool:
        path = self._checkpoint_file()
        if path is None:
            return False
        if not path.is_file():
            self.checkpoint_status = "not_found"
            return False
        try:
            try:
                payload = torch.load(path, map_location=self.agent.device, weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location=self.agent.device)
            if not isinstance(payload, dict):
                raise ValueError("checkpoint payload is not a dictionary")
            expected = self._checkpoint_architecture_metadata()
            found = payload.get("architecture")
            if found != expected:
                self.checkpoint_status = "incompatible"
                self.checkpoint_last_error = self._checkpoint_mismatch_summary(found, expected)
                return False
            self.agent.dynamics.load_state_dict(payload["dynamics_state"])
            if "optimizer_state" in payload:
                self.agent.optimizer.load_state_dict(payload["optimizer_state"])
            self._restore_replay(self.agent.replay, payload.get("replay"))
            if (
                isinstance(self.agent, SpatialActiveInferencePainter)
                and self.agent.composition is not None
                and payload.get("composition_state") is not None
            ):
                self.agent.composition.load_state_dict(payload["composition_state"])
                if (
                    self.agent.composition_optimizer is not None
                    and payload.get("composition_optimizer_state") is not None
                ):
                    self.agent.composition_optimizer.load_state_dict(payload["composition_optimizer_state"])
                self.agent.last_composition_loss = payload.get("last_composition_loss")
            if isinstance(self.agent, SpatialActiveInferencePainter):
                self._restore_replay(self.agent.composition_replay, payload.get("composition_replay"))
            self.trained_transitions = int(payload.get("trained_transitions", 0))
            self.last_training_loss = payload.get("last_training_loss")
            self.checkpoint_loaded = True
            self.checkpoint_status = "loaded"
            self.checkpoint_last_error = None
            return True
        except Exception as exc:  # pragma: no cover - surfaced in diagnostics.
            self.checkpoint_status = "load_failed"
            self.checkpoint_last_error = repr(exc)
            return False

    def _save_checkpoint_if_due(self, *, force: bool = False) -> None:
        path = self._checkpoint_file()
        if path is None:
            return
        if self.checkpoint_status in {"incompatible", "load_failed"} and not self.checkpoint_loaded:
            return
        interval = max(1, int(self.checkpoint_save_every_transitions))
        if not force and self.trained_transitions % interval != 0:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, object] = {
                "schema_version": 1,
                "architecture": self._checkpoint_architecture_metadata(),
                "config": asdict(self.config),
                "trained_transitions": self.trained_transitions,
                "last_training_loss": self.last_training_loss,
                "dynamics_state": self.agent.dynamics.state_dict(),
                "optimizer_state": self.agent.optimizer.state_dict(),
                "replay": self._replay_snapshot(self.agent.replay),
            }
            if isinstance(self.agent, SpatialActiveInferencePainter):
                payload["composition_replay"] = self._replay_snapshot(self.agent.composition_replay)
                payload["last_composition_loss"] = self.agent.last_composition_loss
                if self.agent.composition is not None:
                    payload["composition_state"] = self.agent.composition.state_dict()
                if self.agent.composition_optimizer is not None:
                    payload["composition_optimizer_state"] = self.agent.composition_optimizer.state_dict()
            temp_path = path.with_name(f"{path.name}.tmp")
            torch.save(payload, temp_path)
            temp_path.replace(path)
            self.checkpoint_status = "saved"
            self.checkpoint_last_saved = str(path)
            self.checkpoint_last_error = None
        except Exception as exc:  # pragma: no cover - surfaced in diagnostics.
            self.checkpoint_status = "save_failed"
            self.checkpoint_last_error = repr(exc)

    @staticmethod
    def _replay_snapshot(replay: object) -> dict[str, object] | None:
        data = getattr(replay, "data", None)
        if data is None:
            return None
        return {
            "data": list(data),
            "maxlen": getattr(data, "maxlen", None),
        }

    @staticmethod
    def _restore_replay(replay: object, snapshot: object) -> None:
        if not isinstance(snapshot, dict) or not hasattr(replay, "data"):
            return
        data = getattr(replay, "data")
        maxlen = getattr(data, "maxlen", None)
        restored = list(snapshot.get("data", []))
        data.clear()
        if maxlen is not None:
            restored = restored[-int(maxlen):]
        data.extend(restored)

    @staticmethod
    def _checkpoint_mismatch_summary(found: object, expected: dict[str, object]) -> str:
        if not isinstance(found, dict):
            return "checkpoint has no architecture metadata"
        changed = [
            f"{key}: checkpoint={found.get(key)!r}, current={value!r}"
            for key, value in expected.items()
            if found.get(key) != value
        ]
        extra = [f"{key}: checkpoint={value!r}, current=<missing>" for key, value in found.items() if key not in expected]
        details = changed + extra
        if not details:
            return "architecture metadata differs"
        return "; ".join(details[:6])

    def reset(self, sim: ArmPainterSim) -> None:
        with self._planner_lock:
            self.current = None
            self.stopped = False
            self.stroke_count = 0
            self.last_ranked = []
            self.last_components = None
            self.last_stop_blocked = False
            self.last_execution_forecast = None
            self.last_motor_rejections = 0
            self.last_motor_primitive_candidates = 0
            self.planning = False
            self.last_planning_profile = {}
            self._planning_profile_current = None
            self._planning_forecast_cache = {}
            self._planning_started_at = None
            self._planner_thread = None
            self._pending_current = None
            self._pending_stopped = False
            self._pending_ranked = None
            self._pending_components = None
            self._pending_passage_queue = ()
            self._pending_passage = None
            self._pending_passage_plan = None
            self._pending_error = None
            self._transition_to_learn = None
            self._post_stroke_retract_remaining = 0.0
            self._hold_pose = None
            self._hold_command_pose = None
            self._hold_command_velocity = {}
            self._hold_scope = "global"
            self._passage_queue = []
            self._active_passage = None
            self._active_passage_plan = None
            self._active_passage_total_strokes = 0
            self._active_passage_completed_strokes = 0
            self._contact_release_count = 0
        self._observe(sim)

    def _observe(self, sim: ArmPainterSim) -> GaussianBelief | SpatialCanvasState:
        state = self._planner_state(sim)
        self._reset_agent_belief(state)
        self.belief = self.agent.belief
        return self.belief

    def _uses_spatial_planner(self) -> bool:
        return self.config.planner_state_kind == "spatial_material"

    def _planner_state(self, sim: ArmPainterSim) -> np.ndarray | SpatialCanvasState:
        if self._uses_spatial_planner():
            return spatial_canvas_state(sim, self.config)
        return canvas_summary_state(sim)

    def _state_coverage(self, state: np.ndarray | SpatialCanvasState) -> float:
        if isinstance(state, SpatialCanvasState):
            return state.material_coverage_mean(self.config.thickness_scale)
        return float(state[0])

    def _reset_agent_belief(self, state: np.ndarray | SpatialCanvasState) -> None:
        if self._uses_spatial_planner():
            assert isinstance(self.agent, SpatialActiveInferencePainter)
            assert isinstance(state, SpatialCanvasState)
            self.agent.reset_belief(state)
        else:
            assert isinstance(self.agent, ActiveInferencePainter)
            assert isinstance(state, np.ndarray)
            self.agent.reset_belief(state)

    def _update_agent_belief(
        self,
        action: StrokeAction,
        state: np.ndarray | SpatialCanvasState,
        motor_primitive: MotorPrimitiveLatent | None = None,
    ) -> None:
        if self._uses_spatial_planner():
            assert isinstance(self.agent, SpatialActiveInferencePainter)
            assert isinstance(state, SpatialCanvasState)
            self.agent.update_belief(action, state)
        else:
            assert isinstance(self.agent, ActiveInferencePainter)
            assert isinstance(state, np.ndarray)
            self.agent.update_belief(action, state, motor_primitive)

    def _add_transition_to_agent(
        self,
        state: np.ndarray | SpatialCanvasState,
        action: StrokeAction,
        next_state: np.ndarray | SpatialCanvasState,
        motor_primitive: MotorPrimitiveLatent | None = None,
    ) -> None:
        if self._uses_spatial_planner():
            assert isinstance(self.agent, SpatialActiveInferencePainter)
            assert isinstance(state, SpatialCanvasState)
            assert isinstance(next_state, SpatialCanvasState)
            self.agent.add_transition(state, action, next_state, motor_primitive)
        else:
            assert isinstance(self.agent, ActiveInferencePainter)
            assert isinstance(state, np.ndarray)
            assert isinstance(next_state, np.ndarray)
            self.agent.replay.add(state, encoded_action_vector(action, self.config, motor_primitive), next_state)

    def step(self, sim: ArmPainterSim, dt: float) -> None:
        if not self.enabled or self.stopped:
            self._hold_retracted(sim, dt, scope="global")
            return
        if self._post_stroke_retract_remaining > 0.0:
            hold_scope = "passage" if self._passage_queue else "global"
            self._hold_retracted(sim, dt, scope=hold_scope)
            self._post_stroke_retract_remaining = max(0.0, self._post_stroke_retract_remaining - dt)
            if self._post_stroke_retract_remaining <= 0.0 and self._passage_queue:
                self._start_next_passage_stroke(sim)
            if not self._passage_queue and self.current is None:
                self._start_background_plan(sim)
            return
        if self._consume_background_plan():
            self._hold_retracted(sim, dt, scope="global")
            return
        if self.current is None:
            if self._passage_queue:
                self._start_next_passage_stroke(sim)
                return
            self._hold_retracted(sim, dt, scope="global")
            self._start_background_plan(sim)
            return
        self._hold_pose = None
        self._execute_current(sim, dt)

    def _hold_retracted(self, sim: ArmPainterSim, dt: float, *, scope: str) -> None:
        sim.paint_enabled = False
        sim.intended_contact_pressure = 0.0
        sim.control_damping_multiplier = max(1.0, float(self.config.hold_damping_multiplier))
        contact_escape = self._contact_escape_pose(sim, scope)
        if contact_escape is not None:
            self._hold_scope = scope
            desired = contact_escape
        else:
            if self._hold_pose is None or self._hold_scope != scope:
                self._hold_scope = scope
                self._hold_pose = self._passage_hold_pose(sim) if scope == "passage" else self._global_hold_pose(sim)
                self._hold_command_pose = sim.target_pose
                self._hold_command_velocity = dict.fromkeys(JOINT_NAMES, 0.0)
            desired = self._hold_pose
        self._apply_contact_release(sim, desired, dt)
        sim.set_target(self._shaped_hold_target(sim, desired, dt))

    def _shaped_hold_target(self, sim: ArmPainterSim, desired: ArmPose, dt: float) -> ArmPose:
        if self._hold_command_pose is None:
            self._hold_command_pose = sim.target_pose
            self._hold_command_velocity = dict.fromkeys(JOINT_NAMES, 0.0)
        dt_eff = max(float(dt), 1.0 / 240.0)
        max_speed = max(1.0, float(self.config.hold_target_joint_speed_deg_s))
        max_accel = max(1.0, float(self.config.hold_target_joint_accel_deg_s2))
        values: dict[str, float] = {}
        for name in JOINT_NAMES:
            current = float(getattr(self._hold_command_pose, name))
            target = float(getattr(desired, name))
            error = target - current
            old_velocity = float(self._hold_command_velocity.get(name, 0.0))
            if abs(error) < 1e-6:
                velocity = 0.0
                value = target
            else:
                braking_speed = float(np.sqrt(max(0.0, 2.0 * max_accel * abs(error))))
                desired_velocity = float(np.sign(error) * min(max_speed, braking_speed))
                velocity_delta = float(np.clip(desired_velocity - old_velocity, -max_accel * dt_eff, max_accel * dt_eff))
                velocity = float(np.clip(old_velocity + velocity_delta, -max_speed, max_speed))
                step = velocity * dt_eff
                if abs(step) >= abs(error):
                    value = target
                    velocity = 0.0
                else:
                    value = current + step
            values[name] = value
            self._hold_command_velocity[name] = velocity
        self._hold_command_pose = ArmPose(**values).clipped()
        return self._hold_command_pose

    def _apply_contact_release(self, sim: ArmPainterSim, desired: ArmPose, dt: float) -> None:
        threshold = max(0.0, float(self.config.contact_release_pressure_threshold))
        if sim.contact.pressure <= threshold:
            return
        tip = sim.kinematics.tip(sim.actual_pose)
        current_overtravel = sim.canvas.overtravel_depth(tip)
        max_delta = max(82.0, float(self.config.contact_release_joint_speed_deg_s)) * max(float(dt), 1.0 / 240.0)
        release_pose = rate_limit_pose(desired, sim.actual_pose, max_delta=max_delta).clipped()
        release_tip = sim.kinematics.tip(release_pose)
        release_overtravel = sim.canvas.overtravel_depth(release_tip)
        if release_overtravel > current_overtravel and float(release_tip[1]) >= float(tip[1]):
            return
        sim.actual_pose = release_pose
        sim.target_pose = release_pose
        self._hold_command_pose = release_pose
        self._hold_command_velocity = dict.fromkeys(JOINT_NAMES, 0.0)
        sim.plant.reset_state(sim.actual_pose)
        sim.intended_contact_pressure = 0.0
        sim.paint_enabled = False
        sim.contact = sim.canvas.contact_from_tip(release_tip, sim.intended_contact_pressure)
        self._contact_release_count += 1

    def _contact_escape_pose(self, sim: ArmPainterSim, scope: str) -> ArmPose | None:
        tip = sim.kinematics.tip(sim.actual_pose)
        near_or_on_canvas = sim.contact.on_canvas and (
            sim.contact.pressure > 0.01 or float(tip[1]) > sim.canvas.distance - 0.08
        )
        if not near_or_on_canvas:
            return None
        lateral_limit = 0.46 * min(sim.canvas.width, sim.canvas.height)
        x = float(np.clip(tip[0], -lateral_limit, lateral_limit))
        z = float(np.clip(tip[2], -lateral_limit, lateral_limit))
        if not np.isfinite(x) or not np.isfinite(z):
            x, z = self._active_passage_world_center(sim) if scope == "passage" else (0.0, 0.0)
        depth = self.config.local_passage_retract_depth if scope == "passage" else self.config.global_planning_retract_depth
        pose = ik_pose_for_canvas_point(x, z, sim.canvas.distance - depth)
        tip = sim.kinematics.tip(pose)
        min_clearance = max(0.35, 0.5 * depth)
        if not np.all(np.isfinite(tip)) or float(tip[1]) > sim.canvas.distance - min_clearance:
            return self._passage_hold_pose(sim) if scope == "passage" else self._global_hold_pose(sim)
        return pose

    def _global_hold_pose(self, sim: ArmPainterSim) -> ArmPose:
        x = float(self.config.global_planning_park_x_fraction) * sim.canvas.width
        z = 0.0
        return ik_pose_for_canvas_point(x, z, sim.canvas.distance - self.config.global_planning_retract_depth)

    def _passage_hold_pose(self, sim: ArmPainterSim) -> ArmPose:
        x, z = self._active_passage_world_center(sim)
        return ik_pose_for_canvas_point(x, z, sim.canvas.distance - self.config.local_passage_retract_depth)

    def _active_passage_world_center(self, sim: ArmPainterSim) -> tuple[float, float]:
        if self._active_passage is not None:
            x = (self._active_passage.center_x - 0.5) * sim.canvas.width * 0.98
            z = (0.5 - self._active_passage.center_y) * sim.canvas.height * 0.98
            return float(x), float(z)
        if self._active_passage_plan is not None:
            x = (self._active_passage_plan.center_x - 0.5) * sim.canvas.width * 0.98
            z = (0.5 - self._active_passage_plan.center_y) * sim.canvas.height * 0.98
            return float(x), float(z)
        actions = self._passage_queue
        if actions:
            centers: list[tuple[float, float]] = []
            for action in actions[: max(1, min(3, len(actions)))]:
                x0, z0, x1, z1 = stroke_world_endpoints(action, sim.canvas)
                centers.append((0.5 * (x0 + x1), 0.5 * (z0 + z1)))
            return tuple(float(v) for v in np.mean(np.asarray(centers, dtype=np.float64), axis=0))  # type: ignore[return-value]
        tip = sim.kinematics.tip(sim.actual_pose)
        lateral_limit = 0.46 * min(sim.canvas.width, sim.canvas.height)
        x = float(np.clip(tip[0], -lateral_limit, lateral_limit))
        z = float(np.clip(tip[2], -lateral_limit, lateral_limit))
        if not np.isfinite(x) or not np.isfinite(z):
            return 0.0, 0.0
        return x, z

    def _start_background_plan(self, sim: ArmPainterSim) -> None:
        if self.current is not None or self._passage_queue:
            return
        with self._planner_lock:
            if (
                self.planning
                or self._pending_ranked is not None
                or self._pending_current is not None
                or self._pending_stopped
            ):
                return
            # The previous planner thread may still be training after its plan
            # was published; model updates must not race the next evaluation.
            if self._planner_thread is not None and self._planner_thread.is_alive():
                return
            transition = self._transition_to_learn
            self._transition_to_learn = None
            state = self._planner_state(sim)
            body_snapshot = copy.deepcopy(sim)
            self.planning = True
            self._planning_started_at = time.perf_counter()
            self._pending_error = None
        thread = threading.Thread(
            target=self._background_plan,
            args=(state, transition, body_snapshot),
            name="active-painter-policy-plan",
            daemon=True,
        )
        with self._planner_lock:
            self._planner_thread = thread
        thread.start()

    def _background_plan(
        self,
        state: np.ndarray | SpatialCanvasState,
        transition: tuple[
            np.ndarray | SpatialCanvasState,
            StrokeAction,
            MotorPrimitiveLatent | None,
            np.ndarray | SpatialCanvasState,
        ] | None,
        body_snapshot: ArmPainterSim | None = None,
    ) -> None:
        started = time.perf_counter()
        pending_current: StrokeExecution | None = None
        pending_stopped = False
        pending_ranked: list[tuple[Policy, EFEComponents | SpatialEFEComponents, float]] | None = None
        pending_components: EFEComponents | SpatialEFEComponents | None = None
        pending_passage_queue: tuple[StrokeAction, ...] = ()
        pending_passage: PassageLatent | None = None
        pending_passage_plan: PassagePlanLatent | None = None
        error: str | None = None
        profile: dict[str, object] = {
            "kind": "planning_profile",
            "totalSeconds": 0.0,
            "beliefUpdateSeconds": 0.0,
            "policyInferenceSeconds": 0.0,
            "policySampleSeconds": 0.0,
            "baseEFESeconds": 0.0,
            "motorForecastSeconds": 0.0,
            "motorEFERescoreSeconds": 0.0,
            "posteriorSeconds": 0.0,
            "compositionDiagnosticSeconds": 0.0,
            "selectedForecastSeconds": 0.0,
            "selectedForecastCacheHits": 0,
            "trailingTrainingSeconds": 0.0,
            "publishSeconds": 0.0,
            "policyCount": 0,
            "motorForecastCount": 0,
            "motorForecastCacheHits": 0,
            "candidateMotorRealizations": 0,
            "trainingAfterPublish": False,
            "beliefUpdateRequiredBeforeInference": transition is not None,
        }
        self._planning_profile_current = profile
        self._planning_forecast_cache = {}
        try:
            phase_started = time.perf_counter()
            if transition is not None:
                before, action, motor_primitive, after = transition
                self._add_transition_to_agent(before, action, after, motor_primitive)
                self.trained_transitions += 1
                self._update_agent_belief(action, after, motor_primitive)
            else:
                self._reset_agent_belief(state)
            self.belief = self.agent.belief
            self._profile_add_seconds("beliefUpdateSeconds", time.perf_counter() - phase_started)
            phase_started = time.perf_counter()
            if body_snapshot is None:
                _, _, ranked = self.agent.infer_policy()
                self.last_motor_rejections = 0
                self.last_motor_primitive_candidates = 0
                self._profile_set("policyCount", len(ranked))
            elif self._uses_spatial_planner():
                ranked = self._infer_spatial_policy_with_execution_forecasts(body_snapshot)
            else:
                ranked = self._infer_policy_with_execution_forecasts(body_snapshot)
            self._profile_add_seconds("policyInferenceSeconds", time.perf_counter() - phase_started)
            pending_ranked = ranked
            policy, efe, prob = ranked[0]
            action = policy.actions[0]
            pending_components = efe
            # Premature termination is handled by the declared stop prior
            # inside policy inference, not by a procedural veto here.
            if action.stop:
                pending_stopped = True
            else:
                motor_primitive = policy.motor_primitive
                if policy.passage is not None or policy.passage_plan is not None:
                    pending_passage_queue = tuple(action for action in policy.actions[1:] if not action.stop)
                    pending_passage = policy.passage
                    pending_passage_plan = policy.passage_plan
                pending_current = StrokeExecution(
                    action=action,
                    efe=efe,
                    posterior=float(prob),
                    initial_state=state,
                    forecast=self._profiled_forecast_action(body_snapshot, action, motor_primitive),
                    motor_primitive=motor_primitive,
                    controller=controller_for_motor_primitive(motor_primitive),
                )
        except Exception as exc:  # pragma: no cover - surfaced in diagnostics.
            error = repr(exc)
        if error is None:
            phase_started = time.perf_counter()
            self._refresh_composition_diagnostics()
            self._profile_add_seconds("compositionDiagnosticSeconds", time.perf_counter() - phase_started)
        phase_started = time.perf_counter()
        profile["totalSeconds"] = time.perf_counter() - started
        with self._planner_lock:
            self._pending_current = pending_current
            self._pending_stopped = pending_stopped
            self._pending_ranked = pending_ranked
            self._pending_components = pending_components
            self._pending_passage_queue = pending_passage_queue
            self._pending_passage = pending_passage
            self._pending_passage_plan = pending_passage_plan
            self._pending_error = error
            self.last_planning_seconds = time.perf_counter() - started
            profile["totalSeconds"] = self.last_planning_seconds
            self.last_planning_profile = dict(profile)
            self.planning = False
            self._planning_started_at = None
        self._profile_add_seconds("publishSeconds", time.perf_counter() - phase_started)
        with self._planner_lock:
            self.last_planning_profile = dict(profile)
        # Model learning runs after the plan is published, so it overlaps the
        # selected stroke's execution instead of extending the planning gap.
        # _start_background_plan will not launch the next planner thread until
        # this one exits, so training never races policy evaluation.
        if error is None and transition is not None:
            try:
                train_started = time.perf_counter()
                self.last_training_loss = self.agent.train_dynamics(gradient_steps=2)
                self.last_training_seconds = time.perf_counter() - train_started
                self._save_checkpoint_if_due()
                with self._planner_lock:
                    self.last_planning_profile = {
                        **self.last_planning_profile,
                        "trailingTrainingSeconds": self.last_training_seconds,
                        "trainingAfterPublish": True,
                    }
            except Exception as exc:  # pragma: no cover - surfaced in diagnostics.
                with self._planner_lock:
                    self._pending_error = repr(exc)
        self._planning_profile_current = None
        self._planning_forecast_cache = {}

    def _profile_add_seconds(self, key: str, seconds: float) -> None:
        profile = self._planning_profile_current
        if profile is None:
            return
        profile[key] = float(profile.get(key, 0.0)) + max(0.0, float(seconds))

    def _profile_increment(self, key: str, amount: int = 1) -> None:
        profile = self._planning_profile_current
        if profile is None:
            return
        profile[key] = int(profile.get(key, 0)) + int(amount)

    def _profile_set(self, key: str, value: object) -> None:
        profile = self._planning_profile_current
        if profile is not None:
            profile[key] = value

    def _profiled_forecast_action(
        self,
        sim: ArmPainterSim | None,
        action: StrokeAction,
        motor_primitive: MotorPrimitiveLatent | None = None,
    ) -> ExecutionForecast | None:
        if sim is None or action.stop:
            return None
        key = self._forecast_cache_key(action, motor_primitive)
        cached = self._planning_forecast_cache.get(key)
        if cached is not None:
            self._profile_increment("selectedForecastCacheHits")
            return cached
        started = time.perf_counter()
        forecast = self._forecast_action(sim, action, motor_primitive)
        self._profile_add_seconds("selectedForecastSeconds", time.perf_counter() - started)
        return forecast

    def _forecast_cache_key(
        self,
        action: StrokeAction,
        motor_primitive: MotorPrimitiveLatent | None,
    ) -> tuple[object, ...]:
        primitive_key = "" if motor_primitive is None else motor_primitive.kind
        return tuple(float(x) for x in action.vector()) + (primitive_key,)

    def _refresh_composition_diagnostics(self) -> None:
        # Cached so UI polling never runs a model forward concurrently with
        # background training.
        if isinstance(self.agent, SpatialActiveInferencePainter) and self.agent.composition is not None:
            self._cached_belief_gap = self.agent.belief_composition_gap()

    def _current_planning_seconds(self) -> float:
        if not self.planning or self._planning_started_at is None:
            return 0.0
        return max(0.0, time.perf_counter() - self._planning_started_at)

    def _consume_background_plan(self) -> bool:
        with self._planner_lock:
            if self.planning:
                return False
            if self._pending_ranked is None and self._pending_current is None and not self._pending_stopped:
                return False
            pending_current = self._pending_current
            pending_stopped = self._pending_stopped
            pending_ranked = self._pending_ranked
            pending_components = self._pending_components
            pending_passage_queue = self._pending_passage_queue
            pending_passage = self._pending_passage
            pending_passage_plan = self._pending_passage_plan
            # Diagnostic: stop had the lowest expected free energy, but the
            # declared stop prior demoted it below a continuation policy.
            stop_blocked = False
            if pending_ranked:
                lowest_efe = min(pending_ranked, key=lambda item: item[1].total)
                stop_blocked = bool(
                    lowest_efe[0].actions[0].stop and not pending_ranked[0][0].actions[0].stop
                )
            self._pending_current = None
            self._pending_stopped = False
            self._pending_ranked = None
            self._pending_components = None
            self._pending_passage_queue = ()
            self._pending_passage = None
            self._pending_passage_plan = None
        if pending_ranked is not None:
            self.last_ranked = pending_ranked
        if pending_components is not None:
            self.last_components = pending_components
        self.last_stop_blocked = stop_blocked
        if pending_stopped:
            self.stopped = True
            if self.on_stop is not None:
                self.on_stop()
            return True
        self.current = pending_current
        if pending_current is not None:
            self._passage_queue = list(pending_passage_queue)
            self._active_passage = pending_passage
            self._active_passage_plan = pending_passage_plan
            self._active_passage_total_strokes = 1 + len(self._passage_queue) if self._passage_queue else 0
            self._active_passage_completed_strokes = 0
            self._hold_pose = None
            self._hold_command_pose = None
            self._hold_command_velocity = {}
        return False

    def _start_next_passage_stroke(self, sim: ArmPainterSim) -> None:
        if not self._passage_queue:
            return
        action = self._passage_queue.pop(0)
        self.current = StrokeExecution(
            action=action,
            efe=self.last_components if self.last_components is not None else EFEComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            posterior=1.0,
            initial_state=self._planner_state(sim),
            controller=ContactAwareStrokeController(),
        )
        self._hold_pose = None
        self._hold_command_pose = None
        self._hold_command_velocity = {}

    def _execute_current(self, sim: ArmPainterSim, dt: float) -> None:
        assert self.current is not None
        ex = self.current
        self._hold_command_pose = None
        self._hold_command_velocity = {}
        if not ex.initialized:
            ex.timing = adaptive_stroke_timing(sim, ex.action)
            ex.controller.reset(sim, ex.action, ex.timing)
            ex.initialized = True
        ex.t += dt
        command = ex.controller.command(sim, ex.action, ex.t, dt, ex.timing)
        sim.control_damping_multiplier = 1.0
        sim.set_target(command.pose)
        sim.paint_enabled = command.brush_down
        sim.intended_contact_pressure = command.intended_pressure
        sim.brush_tone = float(ex.action.tone >= 0.5)
        if ex.t >= ex.total:
            self.stroke_count += 1
            self.last_execution_forecast = ex.forecast
            self.current = None
            after = self._planner_state(sim)
            passage_continues = bool(self._passage_queue)
            if self._active_passage_total_strokes > 0:
                self._active_passage_completed_strokes += 1
            if ex.initial_state is not None:
                if passage_continues:
                    self._add_transition_to_agent(ex.initial_state, ex.action, after, ex.motor_primitive)
                    self.trained_transitions += 1
                    self._update_agent_belief(ex.action, after, ex.motor_primitive)
                    self.belief = self.agent.belief
                else:
                    with self._planner_lock:
                        self._transition_to_learn = (ex.initial_state, ex.action, ex.motor_primitive, after)
            if passage_continues:
                self._post_stroke_retract_remaining = max(0.0, self.config.passage_local_retract_seconds)
                self._hold_retracted(sim, dt, scope="passage")
            else:
                had_active_passage = self._active_passage_total_strokes > 0
                self._active_passage = None
                self._active_passage_plan = None
                self._active_passage_total_strokes = 0
                self._active_passage_completed_strokes = 0
                retract_seconds = (
                    self.config.passage_center_retract_seconds
                    if had_active_passage
                    else self.config.post_stroke_retract_seconds
                )
                self._post_stroke_retract_remaining = max(0.0, retract_seconds)
                self._hold_retracted(sim, dt, scope="global")

    def _yield_to_runtime(self) -> None:
        delay = max(0.0, float(self.config.background_planner_yield_seconds))
        if delay > 0.0:
            time.sleep(delay)
        else:
            time.sleep(0)

    def _infer_policy_with_execution_forecasts(
        self,
        body_snapshot: ArmPainterSim,
    ) -> list[tuple[Policy, EFEComponents, float]]:
        assert isinstance(self.agent, ActiveInferencePainter)
        assert isinstance(self.belief, GaussianBelief)
        agent = self.agent
        belief = self.belief
        phase_started = time.perf_counter()
        policies = agent.policy_sampler.sample()
        self._profile_add_seconds("policySampleSeconds", time.perf_counter() - phase_started)
        self._profile_set("policyCount", len(policies))
        phase_started = time.perf_counter()
        base_components = agent.efe.evaluate_batch(belief, policies)
        self._profile_add_seconds("baseEFESeconds", time.perf_counter() - phase_started)
        believed_coverage = float(belief.mean[0].item())
        stop_indices = [i for i, policy in enumerate(policies) if policy.actions[0].stop]
        non_stop_indices = [i for i, policy in enumerate(policies) if not policy.actions[0].stop]
        non_stop_indices = sorted(non_stop_indices, key=lambda i: base_components[i].total)
        forecast_budget = max(1, self.config.motor_forecast_candidates)
        components: list[EFEComponents] = list(base_components)
        rejections = 0
        motor_primitive_candidates = 0
        forecasted_indices: set[int] = set()
        active_indices: list[int] = list(stop_indices)
        forecast_cache: dict[tuple[object, ...], ExecutionForecast] = {}

        def forecast_index(index: int) -> bool:
            nonlocal rejections, motor_primitive_candidates
            policy = policies[index]
            first_action = policy.actions[0]
            if first_action.stop:
                return True
            best_policy = policy
            best_component: EFEComponents | None = None
            best_feasible = False
            for motor_policy in motor_realization_policy_alternatives(policy, self.config):
                motor_primitive_candidates += 1
                self._profile_increment("candidateMotorRealizations")
                primitive = motor_policy.motor_primitive
                key = self._forecast_cache_key(first_action, primitive)
                forecast = forecast_cache.get(key)
                if forecast is None:
                    forecast_started = time.perf_counter()
                    forecast = forecast_stroke_execution(
                        body_snapshot,
                        first_action,
                        canvas_summary_state,
                        motor_primitive=primitive,
                        dt=1.0 / 45.0,
                    )
                    self._profile_add_seconds("motorForecastSeconds", time.perf_counter() - forecast_started)
                    self._profile_increment("motorForecastCount")
                    forecast_cache[key] = forecast
                    self._planning_forecast_cache[key] = forecast
                    self._yield_to_runtime()
                else:
                    self._profile_increment("motorForecastCacheHits")
                rescore_started = time.perf_counter()
                motor_terms = motor_efe_terms(forecast, self.config)
                mean = torch.tensor(forecast.next_state_mean, device=agent.device)
                variance = torch.tensor(forecast.next_state_variance, device=agent.device)
                comp = agent.efe.evaluate_with_first_transition(
                    belief,
                    motor_policy,
                    mean,
                    variance,
                    execution_uncertainty=forecast.execution_uncertainty,
                    contact_loss_probability=forecast.contact_loss_probability,
                    motor_overshoot=forecast.overshoot,
                    motor_feasible=forecast.feasible,
                    motor_risk=motor_terms.risk,
                    motor_ambiguity=motor_terms.ambiguity,
                    motor_epistemic_value=motor_terms.epistemic_value,
                    motor_efe_approximation=motor_terms.approximation,
                )
                self._profile_add_seconds("motorEFERescoreSeconds", time.perf_counter() - rescore_started)
                if not forecast.feasible:
                    rejections += 1
                    comp = replace(comp, motor_feasible=False)
                if (
                    best_component is None
                    or (forecast.feasible and not best_feasible)
                    or (forecast.feasible == best_feasible and comp.total < best_component.total)
                ):
                    best_policy = motor_policy
                    best_component = comp
                    best_feasible = bool(forecast.feasible)
            assert best_component is not None
            policies[index] = best_policy
            components[index] = best_component
            forecasted_indices.add(index)
            if best_feasible:
                active_indices.append(index)
            return best_feasible

        has_feasible_non_stop = False
        for index in non_stop_indices[:forecast_budget]:
            has_feasible_non_stop = forecast_index(index) or has_feasible_non_stop
        for index in non_stop_indices[forecast_budget:]:
            if has_feasible_non_stop:
                break
            has_feasible_non_stop = forecast_index(index) or has_feasible_non_stop

        for index in non_stop_indices:
            if index not in forecasted_indices:
                components[index] = replace(base_components[index], motor_feasible=False)

        active_indices = list(dict.fromkeys(active_indices))
        if not active_indices:
            active_indices = [0]

        self.last_motor_rejections = rejections
        self.last_motor_primitive_candidates = motor_primitive_candidates
        phase_started = time.perf_counter()
        active_g = torch.tensor([components[i].total for i in active_indices], device=agent.device)
        active_log_prior = torch.tensor(
            [
                policy_stop_log_prior(policies[i], believed_coverage, self.config)
                + motor_policy_log_prior(policies[i], self.config)
                for i in active_indices
            ],
            device=agent.device,
        )
        active_posterior = torch.softmax(
            -self.config.policy_precision * (active_g - active_g.min()) + active_log_prior,
            dim=0,
        )
        posterior_values = [0.0 for _ in policies]
        for index, prob in zip(active_indices, active_posterior.detach().cpu().tolist()):
            posterior_values[index] = prob
        ranked = sorted(
            zip(policies, components, posterior_values),
            key=lambda item: item[2],
            reverse=True,
        )
        self._profile_add_seconds("posteriorSeconds", time.perf_counter() - phase_started)
        return ranked

    def _infer_spatial_policy_with_execution_forecasts(
        self,
        body_snapshot: ArmPainterSim,
    ) -> list[tuple[Policy, SpatialEFEComponents, float]]:
        assert isinstance(self.agent, SpatialActiveInferencePainter)
        assert isinstance(self.belief, SpatialCanvasState)
        agent = self.agent
        belief = self.belief
        phase_started = time.perf_counter()
        policies = agent.policy_sampler.sample(belief.coverage(self.config.thickness_scale))
        self._profile_add_seconds("policySampleSeconds", time.perf_counter() - phase_started)
        self._profile_set("policyCount", len(policies))
        phase_started = time.perf_counter()
        base_components = agent.efe.evaluate_batch(belief, policies)
        self._profile_add_seconds("baseEFESeconds", time.perf_counter() - phase_started)
        believed_coverage = belief.material_coverage_mean(self.config.thickness_scale)
        stop_indices = [i for i, policy in enumerate(policies) if policy.actions[0].stop]
        non_stop_indices = [i for i, policy in enumerate(policies) if not policy.actions[0].stop]
        non_stop_indices = sorted(non_stop_indices, key=lambda i: base_components[i].total)
        forecast_budget = max(1, self.config.motor_forecast_candidates)
        components: list[SpatialEFEComponents] = list(base_components)
        rejections = 0
        motor_primitive_candidates = 0
        forecasted_indices: set[int] = set()
        active_indices: list[int] = list(stop_indices)
        forecast_cache: dict[tuple[object, ...], ExecutionForecast] = {}
        rollout_grid_size = (
            pixel_material_from_state(belief).shape[-1]
            if self.config.spatial_transition_mode == "local_patch"
            else self.config.spatial_grid_size
        )
        material_shape = (
            self.config.spatial_material_channels,
            rollout_grid_size,
            rollout_grid_size,
        )

        def spatial_flat_state(working: ArmPainterSim) -> np.ndarray:
            state = spatial_canvas_state(working, self.config)
            if self.config.spatial_transition_mode == "local_patch":
                return pixel_material_from_state(state).reshape(-1)
            return state.flatten_mean()

        def forecast_index(index: int) -> bool:
            nonlocal rejections, motor_primitive_candidates
            policy = policies[index]
            first_action = policy.actions[0]
            if first_action.stop:
                return True
            best_policy = policy
            best_component: SpatialEFEComponents | None = None
            best_feasible = False
            for motor_policy in motor_realization_policy_alternatives(policy, self.config):
                motor_primitive_candidates += 1
                self._profile_increment("candidateMotorRealizations")
                primitive = motor_policy.motor_primitive
                key = self._forecast_cache_key(first_action, primitive)
                forecast = forecast_cache.get(key)
                if forecast is None:
                    forecast_started = time.perf_counter()
                    forecast = forecast_stroke_execution(
                        body_snapshot,
                        first_action,
                        spatial_flat_state,
                        motor_primitive=primitive,
                        dt=1.0 / 45.0,
                    )
                    self._profile_add_seconds("motorForecastSeconds", time.perf_counter() - forecast_started)
                    self._profile_increment("motorForecastCount")
                    forecast_cache[key] = forecast
                    self._planning_forecast_cache[key] = forecast
                    self._yield_to_runtime()
                else:
                    self._profile_increment("motorForecastCacheHits")
                rescore_started = time.perf_counter()
                next_material = forecast.next_state_mean.reshape(material_shape)
                mean = torch.tensor(next_material, device=agent.device, dtype=torch.float32)
                variance = torch.tensor(
                    self._spatial_material_variance_from_forecast(belief, next_material, forecast, body_snapshot),
                    device=agent.device,
                    dtype=torch.float32,
                )
                motor_terms = motor_efe_terms(forecast, self.config)
                comp = agent.efe.evaluate_with_first_transition(
                    belief,
                    motor_policy,
                    mean,
                    variance,
                    execution_uncertainty=forecast.execution_uncertainty,
                    contact_loss_probability=forecast.contact_loss_probability,
                    motor_overshoot=forecast.overshoot,
                    motor_feasible=forecast.feasible,
                    motor_risk=motor_terms.risk,
                    motor_ambiguity=motor_terms.ambiguity,
                    motor_epistemic_value=motor_terms.epistemic_value,
                    motor_efe_approximation=motor_terms.approximation,
                )
                self._profile_add_seconds("motorEFERescoreSeconds", time.perf_counter() - rescore_started)
                if not forecast.feasible:
                    rejections += 1
                    comp = replace(comp, motor_feasible=False)
                if (
                    best_component is None
                    or (forecast.feasible and not best_feasible)
                    or (forecast.feasible == best_feasible and comp.total < best_component.total)
                ):
                    best_policy = motor_policy
                    best_component = comp
                    best_feasible = bool(forecast.feasible)
            assert best_component is not None
            policies[index] = best_policy
            components[index] = best_component
            forecasted_indices.add(index)
            if best_feasible:
                active_indices.append(index)
            return best_feasible

        has_feasible_non_stop = False
        for index in non_stop_indices[:forecast_budget]:
            has_feasible_non_stop = forecast_index(index) or has_feasible_non_stop
        for index in non_stop_indices[forecast_budget:]:
            if has_feasible_non_stop:
                break
            has_feasible_non_stop = forecast_index(index) or has_feasible_non_stop

        for index in non_stop_indices:
            if index not in forecasted_indices:
                components[index] = replace(base_components[index], motor_feasible=False)

        active_indices = list(dict.fromkeys(active_indices))
        if not active_indices:
            active_indices = [0]

        self.last_motor_rejections = rejections
        self.last_motor_primitive_candidates = motor_primitive_candidates
        phase_started = time.perf_counter()
        active_g = torch.tensor([components[i].total for i in active_indices], device=agent.device)
        active_log_prior = torch.tensor(
            [
                policy_stop_log_prior(policies[i], believed_coverage, self.config)
                + motor_policy_log_prior(policies[i], self.config)
                for i in active_indices
            ],
            device=agent.device,
        )
        active_posterior = torch.softmax(
            -self.config.policy_precision * (active_g - active_g.min()) + active_log_prior,
            dim=0,
        )
        posterior_values = [0.0 for _ in policies]
        for index, prob in zip(active_indices, active_posterior.detach().cpu().tolist()):
            posterior_values[index] = prob
        ranked = sorted(
            zip(policies, components, posterior_values),
            key=lambda item: item[2],
            reverse=True,
        )
        self._profile_add_seconds("posteriorSeconds", time.perf_counter() - phase_started)
        return ranked

    def _spatial_material_variance_from_forecast(
        self,
        belief: SpatialCanvasState,
        next_material: np.ndarray,
        forecast: ExecutionForecast,
        body_snapshot: ArmPainterSim,
    ) -> np.ndarray:
        # Approximation: first-order propagation of execution dispersion into
        # material-field uncertainty. Spatial path covariance moves deposited
        # material across grid cells; pressure and contact-loss uncertainty
        # scale the deposited delta. This remains a predictive covariance, not a
        # scalar motor-ease objective.
        current_material = belief.material
        if current_material.shape != next_material.shape:
            current_material = pixel_material_from_state(belief)
        delta = next_material - current_material
        variance = np.full_like(next_material, 1e-6, dtype=np.float32)
        grid_size = max(1, int(next_material.shape[-1]))
        cell_width = max(1e-6, body_snapshot.canvas.width / grid_size)
        cell_height = max(1e-6, body_snapshot.canvas.height / grid_size)
        path_var_x = max(0.0, float(forecast.path_covariance[0])) / (cell_width * cell_width)
        path_var_z = max(0.0, float(forecast.path_covariance[1])) / (cell_height * cell_height)
        for channel in range(next_material.shape[0]):
            grad_z, grad_x = np.gradient(next_material[channel].astype(np.float64))
            variance[channel] += (path_var_x * grad_x * grad_x + path_var_z * grad_z * grad_z).astype(np.float32)
        contact_var = forecast.contact_loss_probability * (1.0 - forecast.contact_loss_probability)
        pressure_denominator = max(1e-6, forecast.target_pressure_mean * forecast.target_pressure_mean)
        pressure_var = max(0.0, forecast.pressure_variance) / pressure_denominator
        variance += np.asarray((contact_var + pressure_var) * delta * delta, dtype=np.float32)
        return np.clip(variance, 1e-8, 1.0).astype(np.float32)

    def _forecast_action(
        self,
        sim: ArmPainterSim | None,
        action: StrokeAction,
        motor_primitive: MotorPrimitiveLatent | None = None,
    ) -> ExecutionForecast | None:
        if sim is None or action.stop:
            return None
        return forecast_stroke_execution(
            sim,
            action,
            canvas_summary_state,
            motor_primitive=motor_primitive,
            dt=1.0 / 45.0,
        )

    def diagnostics(self) -> dict[str, Any]:
        action = asdict(self.current.action) if self.current is not None else None
        efe = asdict(self.last_components) if self.last_components is not None else None
        posterior_values = [prob for _, _, prob in self.last_ranked]
        posterior_entropy = float(
            -sum(prob * np.log(max(prob, 1e-12)) for prob in posterior_values)
        ) if posterior_values else 0.0
        passage_values = [prob for policy, _, prob in self.last_ranked if policy.passage is not None]
        passage_posterior_mass = float(sum(passage_values)) if passage_values else 0.0
        passage_plan_values = [prob for policy, _, prob in self.last_ranked if policy.passage_plan is not None]
        passage_plan_posterior_mass = float(sum(passage_plan_values)) if passage_plan_values else 0.0
        motor_values = [prob for policy, _, prob in self.last_ranked if policy.motor_primitive is not None]
        motor_posterior_mass = float(sum(motor_values)) if motor_values else 0.0
        spatial_belief = (
            spatial_state_diagnostics(self.belief, self.config)
            if isinstance(self.belief, SpatialCanvasState)
            else None
        )
        mark_events = (
            infer_mark_event_belief(self.belief, self.config).diagnostics()
            if isinstance(self.belief, SpatialCanvasState)
            else None
        )
        composition = None
        if isinstance(self.agent, SpatialActiveInferencePainter) and self.agent.composition is not None:
            composition = {
                "currentBeliefGap": self._cached_belief_gap,
                "gapPrecision": self.config.composition_gap_precision,
                "lastTrainingLoss": self.agent.last_composition_loss,
                "declaredAs": (
                    "structural prior p*(s_T) ~ exp(precision * compression_gap); "
                    "gap = hierarchical ELBO minus context-free flat code, nats/cell-channel"
                ),
            }
        return {
            "enabled": self.enabled,
            "stopped": self.stopped,
            "planning": self.planning,
            "plannerError": self._pending_error,
            "lastPlanningSeconds": self.last_planning_seconds,
            "currentPlanningSeconds": self._current_planning_seconds(),
            "planningProfile": dict(self.last_planning_profile),
            "lastTrainingSeconds": self.last_training_seconds,
            "checkpoint": {
                "path": str(self.checkpoint_path) if self.checkpoint_path is not None else None,
                "status": self.checkpoint_status,
                "loaded": self.checkpoint_loaded,
                "lastSaved": self.checkpoint_last_saved,
                "lastError": self.checkpoint_last_error,
                "saveEveryTransitions": self.checkpoint_save_every_transitions,
                "architecture": dict(self.checkpoint_architecture),
            },
            "postStrokeRetractRemaining": self._post_stroke_retract_remaining,
            "planningScope": self._planning_scope(),
            "holdScope": self._hold_scope,
            "contactReleaseCount": self._contact_release_count,
            "passageQueueLength": len(self._passage_queue),
            "activePassage": asdict(self._active_passage) if self._active_passage is not None else None,
            "activePassagePlan": asdict(self._active_passage_plan) if self._active_passage_plan is not None else None,
            "activePassageTotalStrokes": self._active_passage_total_strokes,
            "activePassageCompletedStrokes": self._active_passage_completed_strokes,
            "minimumStopCoverage": self.config.minimum_stop_coverage,
            "lastStopBlocked": self.last_stop_blocked,
            "motorRejections": self.last_motor_rejections,
            "motorPrimitiveCandidateCount": self.last_motor_primitive_candidates,
            "motorPrimitivePosteriorMass": motor_posterior_mass,
            "executionForecast": self._execution_forecast_diagnostics(),
            "stateRepresentation": self._state_representation_diagnostics(),
            "transitionModel": self._transition_model_diagnostics(),
            "spatialTransitionMode": (
                self.config.spatial_transition_mode
                if isinstance(self.belief, SpatialCanvasState)
                else None
            ),
            "policyPrecision": self.config.policy_precision,
            "posteriorEntropy": posterior_entropy,
            "passageCandidateCount": len(passage_values),
            "passagePosteriorMass": passage_posterior_mass,
            "passagePlanCandidateCount": len(passage_plan_values),
            "passagePlanPosteriorMass": passage_plan_posterior_mass,
            "trainedTransitions": self.trained_transitions,
            "lastTrainingLoss": self.last_training_loss,
            "belief": self._belief_diagnostics(),
            "spatialBelief": spatial_belief,
            "markEvents": mark_events,
            "composition": composition,
            "strokeCount": self.stroke_count,
            "executing": action,
            "executingMotorPrimitive": (
                asdict(self.current.motor_primitive)
                if self.current is not None and self.current.motor_primitive is not None
                else None
            ),
            "efe": efe,
            "phase": self.phase_label(),
            "posterior": self.current.posterior if self.current is not None else None,
            "topPolicies": [
                {
                    "length": len(policy.actions),
                    "firstStop": policy.actions[0].stop,
                    "passage": asdict(policy.passage) if policy.passage is not None else None,
                    "passagePlan": asdict(policy.passage_plan) if policy.passage_plan is not None else None,
                    "motorPrimitive": asdict(policy.motor_primitive) if policy.motor_primitive is not None else None,
                    "posterior": prob,
                    "total": comp.total,
                    "terminalRisk": comp.terminal_risk,
                    "ambiguity": comp.ambiguity,
                    "epistemicValue": comp.epistemic_value,
                    "terminalEntropy": comp.terminal_entropy,
                    "pragmaticValue": comp.pragmatic_value,
                    "transitionRisk": comp.transition_risk,
                    "transitionAmbiguity": comp.transition_ambiguity,
                    "motorRisk": comp.motor_risk,
                    "motorAmbiguity": comp.motor_ambiguity,
                    "motorEpistemicValue": comp.motor_epistemic_value,
                    "motorEFEApproximation": comp.motor_efe_approximation,
                    "compositionGap": getattr(comp, "composition_gap", 0.0),
                    "compositionRisk": getattr(comp, "composition_risk", 0.0),
                    "terminalCoverageMean": comp.terminal_coverage_mean,
                    "rolloutMode": getattr(comp, "rollout_mode", "dense_grid"),
                    "rolloutGridSize": getattr(comp, "rollout_grid_size", 0),
                    "activePatchAreaFraction": getattr(comp, "active_patch_area_fraction", 0.0),
                    "localTransitionSteps": getattr(comp, "local_transition_steps", 0),
                    "sequentialPatchSteps": getattr(comp, "sequential_patch_steps", 0),
                    "identityTransitionApproximation": getattr(comp, "identity_transition_approximation", ""),
                    "executionUncertainty": comp.execution_uncertainty,
                    "contactLossProbability": comp.contact_loss_probability,
                    "motorOvershoot": comp.motor_overshoot,
                    "motorFeasible": comp.motor_feasible,
                }
                for policy, comp, prob in self.last_ranked[:5]
            ],
        }

    def _belief_diagnostics(self) -> dict[str, object]:
        if isinstance(self.belief, SpatialCanvasState):
            std = np.sqrt(np.exp(np.clip(self.belief.logvar, -30.0, 20.0)))
            return {
                "names": list(MATERIAL_CHANNELS[: self.belief.material.shape[0]]),
                "mean": self.belief.material.mean(axis=(1, 2)).astype(float).tolist(),
                "std": std.mean(axis=(1, 2)).astype(float).tolist(),
            }
        assert isinstance(self.belief, GaussianBelief)
        belief_std = torch.sqrt(self.belief.logvar.exp())
        return {
            "names": [
                "coverage",
                "mean_thickness",
                "max_thickness",
                "mean_wetness",
                "overlap_fraction",
                "mean_ground_contrast",
            ],
            "mean": self.belief.mean.detach().cpu().tolist(),
            "std": belief_std.detach().cpu().tolist(),
        }

    def _state_representation_diagnostics(self) -> str:
        if isinstance(self.belief, SpatialCanvasState):
            if self.config.spatial_transition_mode == "local_patch":
                pixel_grid = pixel_material_from_state(self.belief).shape[-1]
                return (
                    f"Spatial Gaussian q(s) with pixel-local rollouts over {pixel_grid}x{pixel_grid} "
                    f"material fields and coarse {self.belief.grid_size}x{self.belief.grid_size} "
                    "composition/planner observations; six canvas summaries are diagnostics only"
                )
            return (
                f"Spatial Gaussian q(s_grid) over {self.belief.grid_size}x{self.belief.grid_size} "
                "material fields: thickness, wetness, black_mass, observed_tone, ground_contrast, material_coverage; "
                "six canvas summaries are diagnostics only"
            )
        return "Gaussian q(s) over six canvas summary states; spatial hierarchy is not active in this runtime mode"

    def _transition_model_diagnostics(self) -> str:
        if isinstance(self.agent, SpatialActiveInferencePainter):
            if self.config.spatial_transition_mode == "local_patch":
                return (
                    "learned LocalSpatialDynamicsEnsemble p_theta(s_patch_next | s_patch, rasterized stroke patch) "
                    "with identity transition prior outside local support"
                )
            return (
                "learned SpatialDynamicsEnsemble p_theta(s_grid_next | s_grid, rasterized stroke) "
                "trained from this arm/canvas simulator"
            )
        return "learned DynamicsEnsemble p_theta(s_next | s, realized execution forecast) trained from this arm/canvas simulator"

    def _execution_forecast_diagnostics(self) -> dict[str, object] | None:
        forecast = self.current.forecast if self.current is not None else self.last_execution_forecast
        return forecast.diagnostics() if forecast is not None else None

    def phase_label(self) -> str:
        if self.current is not None:
            return execution_phase(self.current)
        if self.stopped:
            return "stop"
        if self._post_stroke_retract_remaining > 0.0:
            return "local_passage_hold" if self._passage_queue else "return_center"
        if self._passage_queue:
            return "local_passage_hold"
        return "global_planning"

    def _planning_scope(self) -> str:
        if self.current is not None:
            return "stroke_execution"
        if self._passage_queue or self._active_passage_total_strokes > 0:
            return "passage_local"
        return "global"


def canvas_summary_state(sim: ArmPainterSim) -> np.ndarray:
    canvas = sim.canvas
    coverage = canvas.coverage_field()
    painted = canvas.thickness > 0.02
    overlap = canvas.thickness > sim.config.thickness_scale
    return np.asarray(
        [
            float(coverage.mean()),
            float(canvas.thickness.mean()),
            float(canvas.thickness.max(initial=0.0)),
            float(canvas.wetness.mean()),
            float(overlap.mean()),
            float((canvas.ground_contrast_field() * painted).mean()),
        ],
        dtype=np.float32,
    )


def execute_stroke_action(sim: ArmPainterSim, action: StrokeAction, dt: float = 1.0 / 120.0) -> None:
    ex = StrokeExecution(
        action=action,
        efe=EFEComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        posterior=1.0,
        initial_state=canvas_summary_state(sim),
    )
    ex.timing = adaptive_stroke_timing(sim, action)
    ex.controller.reset(sim, ex.action, ex.timing)
    ex.initialized = True
    while ex.t < ex.total:
        ex.t += dt
        command = ex.controller.command(sim, ex.action, ex.t, dt, ex.timing)
        sim.control_damping_multiplier = 1.0
        sim.set_target(command.pose)
        sim.paint_enabled = command.brush_down
        sim.intended_contact_pressure = command.intended_pressure
        sim.brush_tone = float(action.tone >= 0.5)
        sim.step(dt)


def execution_phase(ex: StrokeExecution | None) -> str:
    if ex is None:
        return "planning"
    if ex.t < ex.approach:
        return "approach"
    if ex.t < ex.approach + ex.press:
        return "press"
    if ex.t < ex.approach + ex.press + ex.paint:
        return "paint"
    return "lift"


def pose_for_execution(sim: ArmPainterSim, ex: StrokeExecution) -> tuple[ArmPose, bool, float]:
    reference = stroke_reference(ex.action, sim, ex.t, ex.timing)
    return pose_for_reference(reference), reference.brush_down, reference.pressure
