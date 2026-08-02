from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .arm_agent_driver import (
    OBSERVATION_ACCESS_MODE,
    ArmActiveInferenceDriver,
)
from .arm_control import scripted_contact_pressure, scripted_pose
from .arm_sim import ArmPainterSim, JOINT_NAMES
from .config import PainterConfig, SPATIAL_MATERIAL_PLANNER_STATE_KIND
from .telemetry_log import ArmTelemetryLog
from .version import CodeBuildInfo, code_build_info
from .web_robot_model import load_robot_visual_model, retarget_legacy_robot_state


FOVEA_TRACE_INTERFACE_VERSION = "fovea-trace-v0"
DEFAULT_FOVEA_TRACE_RETENTION_S = 10.0


@dataclass(slots=True)
class WebSimRuntime:
    canvas_size: int = 256
    speed: float = 1.0
    planner_state_kind: str = SPATIAL_MATERIAL_PLANNER_STATE_KIND
    spatial_grid_size: int = 16
    stroke_tone_prior: float | None = None
    save_every_paintings: int = 5
    archive_dir: Path | str = Path("runs/web")
    telemetry_max_samples: int = 54_000
    telemetry_sample_period: float = 1.0 / 15.0
    driver_bootstrap_transitions: int = 96
    driver_bootstrap_train_steps: int = 24
    checkpoint_path: Path | str | None = None
    checkpoint_save_every_transitions: int = 10
    device: str | None = None
    plant_backend: str = "native"
    observation_access_mode: str = OBSERVATION_ACCESS_MODE
    fovea_trace_retention_s: float = DEFAULT_FOVEA_TRACE_RETENTION_S
    sim: ArmPainterSim = field(init=False)
    agent_driver: ArmActiveInferenceDriver = field(init=False)
    telemetry_log: ArmTelemetryLog = field(init=False)
    code_build: CodeBuildInfo = field(init=False)
    robot_model: dict[str, Any] = field(init=False)
    sim_time: float = field(default=0.0, init=False)
    painting_count: int = field(default=0, init=False)
    last_saved_canvas: str | None = field(default=None, init=False)
    paused: bool = field(default=False, init=False)
    max_speed: bool = field(default=False, init=False)
    agent_enabled: bool = field(default=True, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _next_telemetry_time: float = field(default=0.0, init=False)
    _last_robot_joint_position_deg: dict[str, float] | None = field(default=None, init=False)
    _camera_process: Any | None = field(default=None, init=False)
    _fovea_trace: list[dict[str, Any]] = field(default_factory=list, init=False)
    _pending_fovea_delivery_deadlines: dict[str, float] = field(
        default_factory=dict,
        init=False,
    )
    _next_camera_delivery_poll_time_s: float = field(default=0.0, init=False)
    _operator_fovea_sequence: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.fovea_trace_retention_s) or self.fovea_trace_retention_s <= 0.0:
            raise ValueError("fovea_trace_retention_s must be finite and positive")
        self.code_build = code_build_info()
        self.robot_model = load_robot_visual_model()
        sim_config = PainterConfig(
            canvas_size=self.canvas_size,
            planner_state_kind=self.planner_state_kind,
            spatial_grid_size=self.spatial_grid_size,
            stroke_tone_prior=self.stroke_tone_prior,
        )
        # Replay capacity must be sized to the state representation. In spatial
        # mode each transition holds full-resolution material patches (~200 KB
        # for the local-patch buffer, plus composition/passage states), so the
        # 50k default would grow the three buffers to ~15-20 GB over an
        # overnight run -- a rolling window of a few thousand recent strokes is
        # ample for online continual learning. Summary mode's 6-float states
        # are negligible, so it keeps the large default.
        replay_capacity = 5_000 if self.planner_state_kind == "spatial_material" else 50_000
        driver_config = PainterConfig(
            canvas_size=64,
            candidate_policies=32,
            planning_horizon=4,
            passage_proposal_mix=0.45,
            passage_plan_proposal_mix=0.15,
            policy_precision=0.35,
            batch_size=32,
            motor_forecast_candidates=2,
            planner_state_kind=self.planner_state_kind,
            spatial_grid_size=self.spatial_grid_size,
            stroke_tone_prior=self.stroke_tone_prior,
            replay_capacity=replay_capacity,
        )
        self.sim = ArmPainterSim(sim_config)
        if self.plant_backend == "mujoco":
            from .mujoco_backend import MujocoJointPlant

            self.sim.plant = MujocoJointPlant()
            self.sim.reset_pose()
        elif self.plant_backend != "native":
            raise ValueError(f"unsupported plant backend: {self.plant_backend}")
        self.agent_driver = ArmActiveInferenceDriver(
            config=driver_config,
            bootstrap_transitions=self.driver_bootstrap_transitions,
            bootstrap_train_steps=self.driver_bootstrap_train_steps,
            checkpoint_path=self.checkpoint_path,
            checkpoint_save_every_transitions=self.checkpoint_save_every_transitions,
            on_stop=self._complete_stopped_painting,
            device=self.device,
            observation_access_mode=self.observation_access_mode,
        )
        if self.agent_driver.observation_boundary_blocked:
            self.agent_enabled = False
        self.telemetry_log = ArmTelemetryLog(max_samples=self.telemetry_max_samples)
        self.agent_driver.reset(self.sim)
        self._record_telemetry(force=True)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="active-painter-web-sim", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._camera_process is not None:
            self._camera_process.close()
            self._camera_process = None
        close = getattr(self.sim.plant, "close", None)
        if callable(close):
            close()

    def _run(self) -> None:
        last = time.perf_counter()
        fixed_dt = 1.0 / 240.0
        while not self._stop.is_set():
            now = time.perf_counter()
            wall_dt = min(0.05, now - last)
            last = now
            steps = self._simulation_steps_for_wall_time(wall_dt, fixed_dt)
            for index in range(steps):
                if self._stop.is_set():
                    break
                with self._lock:
                    if self.paused:
                        break
                    self._advance_one_step(fixed_dt)
                if self.max_speed and index % 8 == 7:
                    time.sleep(0)
            time.sleep(0.001 if self.max_speed else 0.01)

    def _simulation_steps_for_wall_time(self, wall_dt: float, fixed_dt: float) -> int:
        with self._lock:
            if self.paused:
                return 0
            if self.max_speed:
                return 120
            return max(1, int(np.ceil(wall_dt * self.speed / fixed_dt)))

    def _advance_one_step(self, fixed_dt: float) -> None:
        self.sim_time += fixed_dt * self.speed
        if self.agent_enabled:
            painting_count_before = self.painting_count
            self.agent_driver.step(self.sim, fixed_dt)
            if self.painting_count != painting_count_before or self._restart_after_stop_if_needed():
                self._record_telemetry(force=True)
                return
        else:
            self.sim.control_damping_multiplier = 1.0
            self.sim.set_target(scripted_pose(self.sim_time))
            self.sim.intended_contact_pressure = scripted_contact_pressure(self.sim_time)
        self.sim.step(fixed_dt)
        self._advance_pending_camera_delivery()
        self._record_telemetry()

    def _record_telemetry(self, *, force: bool = False) -> None:
        if self.telemetry_sample_period <= 0.0:
            return
        if not force and self.sim_time + 1e-12 < self._next_telemetry_time:
            return
        self.telemetry_log.append_from_sim(
            self.sim_time,
            self.sim,
            phase=self.agent_driver.phase_label() if self.agent_enabled else "scripted",
            painting_count=self.painting_count,
            agent_enabled=self.agent_enabled,
        )
        self._next_telemetry_time = self.sim_time + self.telemetry_sample_period

    def command(self, data: dict[str, Any]) -> dict[str, Any]:
        action = str(data.get("type", ""))
        with self._lock:
            if action == "toggle_max_speed":
                self.max_speed = not self.max_speed
            elif action == "set_max_speed":
                self.max_speed = bool(data.get("value", False))
            elif action == "toggle_pause":
                self.paused = not self.paused
            elif action == "set_pause":
                self.paused = bool(data.get("value", False))
            elif action == "reset":
                self.sim.reset_pose()
                self.sim.canvas.clear()
                if self._camera_process is not None:
                    self._camera_process.reset()
                self._clear_fovea_trace()
                self.sim_time = 0.0
                self.painting_count = 0
                self.last_saved_canvas = None
                self.telemetry_log.clear()
                self._next_telemetry_time = 0.0
                self.agent_driver.reset(self.sim)
                self._record_telemetry(force=True)
            elif action == "clear":
                self.sim.canvas.clear()
                self._clear_fovea_trace()
                self.agent_driver.reset(self.sim)
            elif action == "clear_telemetry":
                self.telemetry_log.clear()
                self._next_telemetry_time = self.sim_time
                self._record_telemetry(force=True)
            elif action in {"toggle_brush_load", "toggle_paint"}:
                if self.sim.brush.loaded:
                    self.sim.unload_brush()
                else:
                    self.sim.load_brush(1.0, self.sim.brush_tone)
            elif action == "toggle_agent":
                if self.agent_driver.observation_boundary_blocked:
                    return {
                        "ok": False,
                        "error": (
                            "active inference is fail-closed: camera-conditioned "
                            "painting inference is connected, but sensor-conditioned "
                            "body initialization and the action-conditioned live "
                            "observation loop are not"
                        ),
                    }
                self.agent_enabled = not self.agent_enabled
                self.agent_driver.enabled = self.agent_enabled
                if not self.agent_enabled and not self.sim.brush.loaded:
                    self.sim.load_brush(1.0, self.sim.brush_tone)
            elif action == "tone":
                tone = 1.0 if str(data.get("value", "black")) == "black" else 0.0
                self.sim.brush_tone = tone
                if self.sim.brush.loaded:
                    self.sim.load_brush(self.sim.brush.load_amount, tone)
            elif action == "request_fovea":
                try:
                    self._request_operator_fovea(data)
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    return {"ok": False, "error": str(exc)}
            else:
                return {"ok": False, "error": f"unknown command: {action}"}
        return {"ok": True, "state": self.state()}

    def state(self) -> dict[str, Any]:
        with self._lock:
            self._restart_after_stop_if_needed()
            points = self.sim.kinematics.joint_points(self.sim.actual_pose)
            render_points = self.sim.render_points()
            pose = self.sim.actual_pose
            target = self.sim.target_pose
            contact = self.sim.contact
            telemetry = self.sim.plant.telemetry
            direct_robot_state = getattr(self.sim.plant, "web_robot_state", None)
            if callable(direct_robot_state):
                robot_state = direct_robot_state(pose, target)
            else:
                robot_state = retarget_legacy_robot_state(
                    self.robot_model,
                    {name: float(getattr(pose, name)) for name in JOINT_NAMES},
                    render_points[-1],
                    self._last_robot_joint_position_deg,
                )
                self._last_robot_joint_position_deg = robot_state["jointPositionDeg"]
                robot_state["brushCompressionM"] = min(
                    self.robot_model["brush"]["compressionTravel"],
                    max(0.0, float(contact.deflection) * 0.0254),
                )
                robot_state["brushBendRad"] = {"x": 0.0, "z": 0.0}
            return {
                "simTime": self.sim_time,
                "codeVersion": self.code_build.version,
                "paused": self.paused,
                "maxSpeed": self.max_speed,
                "agentEnabled": self.agent_enabled,
                "paintingCount": self.painting_count,
                "saveEveryPaintings": self.save_every_paintings,
                "lastSavedCanvas": self.last_saved_canvas,
                "telemetryLog": self.telemetry_log.summary(self.sim_time),
                "agent": self.agent_driver.diagnostics(),
                "brushLoaded": self.sim.brush.loaded,
                "depositingPaint": self.sim.depositing_paint,
                "brushLoadAmount": self.sim.brush.load_amount,
                "plantBackend": self.plant_backend,
                "cameraObservation": {
                    "available": self.plant_backend == "mujoco",
                    "interfaceVersion": "camera-observation-interface-v1",
                    "productContract": (
                        self.robot_model["cameraRig"]["productContract"]
                    ),
                    "nativeFrameRetained": True,
                    "derivedProducts": [
                        "global_canvas",
                        "fovea_canvas",
                        "edge_profile",
                    ],
                    "foveaAddressing": (
                        self.robot_model["cameraRig"]["foveaAddressing"]
                    ),
                    "foveaDefault": None,
                    "modelInputEndpoint": (
                        "/api/camera/{camera_name}.png"
                        if self.plant_backend == "mujoco"
                        else None
                    ),
                    "consumedByInference": True,
                    "likelihoodModel": self.robot_model["cameraRig"][
                        "likelihoodModel"
                    ],
                    "consumptionBoundary": (
                        "registered global/foveal products only; native and "
                        "edge products ignored by painting-state likelihood"
                    ),
                    "foveation": self._foveation_state(),
                },
                "counterfactualPlantBackend": getattr(
                    self.sim.plant,
                    "counterfactual_backend_id",
                    self.plant_backend,
                ),
                "brushTone": "black" if self.sim.brush_tone >= 0.5 else "white",
                "canvas": {
                    "width": self.sim.canvas.width,
                    "height": self.sim.canvas.height,
                    "distance": self.sim.canvas.distance,
                    "coverage": self.sim.canvas.material_coverage(),
                    "size": self.sim.canvas.config.canvas_size,
                },
                "pose": {name: float(getattr(pose, name)) for name in JOINT_NAMES},
                "targetPose": {name: float(getattr(target, name)) for name in JOINT_NAMES},
                "points": points.astype(float).tolist(),
                "renderPoints": render_points.astype(float).tolist(),
                "tip": points[-1].astype(float).tolist(),
                "renderTip": render_points[-1].astype(float).tolist(),
                "robot": robot_state,
                "upperArmAxis": self.sim.kinematics.upper_arm_axis(pose).astype(float).tolist(),
                "elbowHingeAxis": self.sim.kinematics.elbow_hinge_axis(pose).astype(float).tolist(),
                "contact": {
                    "onCanvas": contact.on_canvas,
                    "projectedOnCanvas": contact.on_canvas,
                    "touching": bool(contact.pressure > 0.001 or contact.force > 0.001),
                    "deflection": contact.deflection,
                    "force": contact.force,
                    "pressure": contact.pressure,
                    "brushWidthPx": contact.brush_width_px,
                    "brushWorld": contact.brush_world.astype(float).tolist(),
                },
                "motor": {
                    name: {
                        "voltage": telemetry.voltage[name],
                        "current": telemetry.current[name],
                        "torque": telemetry.torque[name],
                        "velocityRadS": self.sim.plant.velocity[name],
                        "velocityDegS": float(np.rad2deg(self.sim.plant.velocity[name])),
                        "actuatorAngleDeg": telemetry.actuator_angle_deg[name],
                        "actuatorVelocityRadS": telemetry.actuator_velocity_rad_s[name],
                        "encoderAngleDeg": telemetry.encoder_angle_deg[name],
                        "encoderVelocityRadS": telemetry.encoder_velocity_rad_s[name],
                        "positionErrorDeg": telemetry.position_error_deg[name],
                        "elasticDeflectionDeg": telemetry.elastic_deflection_deg[name],
                        "backlashDeflectionDeg": telemetry.backlash_deflection_deg[name],
                        "frictionTorque": telemetry.friction_torque[name],
                        "loadTorque": telemetry.load_torque[name],
                        "encoderStdDeg": telemetry.encoder_std_deg[name],
                        "thermalFraction": telemetry.thermal_fraction[name],
                        "torqueLimitFraction": telemetry.torque_limit_fraction[name],
                    }
                    for name in JOINT_NAMES
                },
            }

    def telemetry_csv(self) -> bytes:
        with self._lock:
            return self.telemetry_log.to_csv().encode("utf-8")

    def canvas_png(self) -> bytes:
        with self._lock:
            gray = self._render_canvas_gray()
        image = Image.fromarray(gray, mode="L")
        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()

    def camera_png(self, camera_name: str) -> bytes:
        """Render the global/edge derived product for diagnostics."""

        with self._lock:
            process, backend = self._camera_backend()
            camera = process.rig.camera(camera_name)
            inspection_available = self._inspection_camera_available(
                process,
                backend,
            )
            if camera.availability == "park_only" and not inspection_available:
                raise RuntimeError(
                    f"camera {camera_name!r} is available only at camera_clear_park"
                )
            frame = process.render_immediate(
                camera_name,
                monotonic_time_s=float(backend.data.time),
                qpos=backend.data.qpos.copy(),
                canvas_grayscale=1.0 - self.sim.canvas.observed_tone(),
            )
        image = Image.fromarray(
            np.rint(frame.grayscale * 255.0).astype(np.uint8),
            mode="L",
        )
        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()

    def camera_observations(self, *, fovea_requests: tuple[Any, ...] = ()) -> Any:
        """Deliver due camera products to inference and the fovea trace."""

        with self._lock:
            return self._camera_observations_locked(fovea_requests=fovea_requests)

    def _camera_observations_locked(
        self,
        *,
        fovea_requests: tuple[Any, ...] = (),
    ) -> Any:
        process, backend = self._camera_backend()
        now = float(backend.data.time)
        for request in fovea_requests:
            self._pending_fovea_delivery_deadlines[request.request_id] = max(
                float(request.expires_time_s),
                now,
            ) + float(process.rig.camera(request.camera_name).latency_s) + 0.5
        observation = process.observe(
            now,
            qpos=backend.data.qpos.copy(),
            canvas_grayscale=1.0 - self.sim.canvas.observed_tone(),
            inspection_available=self._inspection_camera_available(
                process,
                backend,
            ),
            fovea_requests=fovea_requests,
        )
        if self.observation_access_mode == OBSERVATION_ACCESS_MODE:
            self.agent_driver.ingest_camera_observation(observation)
        self._record_fovea_observations(observation)
        return observation

    def _request_operator_fovea(self, data: dict[str, Any]) -> None:
        if self.plant_backend != "mujoco":
            raise RuntimeError("fovea requests require the MuJoCo camera backend")
        from .camera_observation import FoveaRequest

        process, backend = self._camera_backend()
        camera_name = str(data.get("cameraName", "canvas_right_oblique"))
        camera = process.rig.camera(camera_name)
        if camera.registration != "canvas_plane_homography" or camera.foveal_resolution_px is None:
            raise ValueError(f"camera {camera_name!r} has no canvas foveal product")
        center = self._normalized_pair(data.get("centerCanvasUv"), "centerCanvasUv")
        span = self._normalized_pair(
            data.get("spanCanvasUv", (0.22, 0.22)),
            "spanCanvasUv",
            strictly_positive=True,
        )
        now = float(backend.data.time)
        request_id = f"operator-{self._operator_fovea_sequence}"
        self._operator_fovea_sequence += 1
        request = FoveaRequest(
            request_id=request_id,
            camera_name=camera_name,
            requested_time_s=now,
            expires_time_s=now + 1.0,
            center_canvas_uv=center,
            span_canvas_uv=span,
            selection_basis="operator_diagnostic",
            selection_revision="web-viewer-pointer-v0",
        )
        self._camera_observations_locked(fovea_requests=(request,))

    @staticmethod
    def _normalized_pair(
        value: Any,
        name: str,
        *,
        strictly_positive: bool = False,
    ) -> tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must contain two normalized values")
        pair = (float(value[0]), float(value[1]))
        lower_valid = all(component > 0.0 for component in pair) if strictly_positive else all(
            component >= 0.0 for component in pair
        )
        if not lower_valid or any(component > 1.0 for component in pair) or not all(
            np.isfinite(component) for component in pair
        ):
            interval = "(0, 1]" if strictly_positive else "[0, 1]"
            raise ValueError(f"{name} must lie in {interval}")
        return pair

    def _advance_pending_camera_delivery(self) -> None:
        if self.plant_backend != "mujoco" or not self._pending_fovea_delivery_deadlines:
            return
        _, backend = self._camera_backend()
        now = float(backend.data.time)
        self._pending_fovea_delivery_deadlines = {
            request_id: deadline
            for request_id, deadline in self._pending_fovea_delivery_deadlines.items()
            if deadline >= now - 1e-12
        }
        if (
            not self._pending_fovea_delivery_deadlines
            or now + 1e-12 < self._next_camera_delivery_poll_time_s
        ):
            return
        self._next_camera_delivery_poll_time_s = now + 1.0 / 120.0
        self._camera_observations_locked()

    def _record_fovea_observations(self, observation: Any) -> None:
        from .camera_observation import FOVEA_CANVAS_PRODUCT

        for frame in observation.frames:
            if frame.product_kind != FOVEA_CANVAS_PRODUCT:
                continue
            event = {
                "eventId": f"{frame.camera_name}:{frame.sequence}:{frame.fovea_request_id}",
                "requestId": frame.fovea_request_id,
                "cameraName": frame.camera_name,
                "sequence": frame.sequence,
                "captureTimeS": frame.capture_time_s,
                "availableTimeS": frame.available_time_s,
                "centerCanvasUv": list(frame.center_canvas_uv),
                "spanCanvasUv": list(frame.span_canvas_uv),
                "selectionBasis": frame.selection_basis,
                "selectionRevision": frame.selection_revision,
            }
            self._fovea_trace.append(event)
            self._pending_fovea_delivery_deadlines.pop(
                str(frame.fovea_request_id),
                None,
            )
        if len(self._fovea_trace) > 512:
            self._fovea_trace = self._fovea_trace[-512:]
        self._prune_fovea_trace(float(observation.monotonic_time_s))

    def _foveation_state(self) -> dict[str, Any]:
        now = self._camera_time_s()
        retention, retention_source, memory_horizon = self._fovea_retention_contract()
        self._prune_fovea_trace(now, retention_s=retention)
        events = [
            {
                **event,
                "ageSeconds": max(0.0, now - float(event["availableTimeS"])),
            }
            for event in self._fovea_trace
        ]
        return {
            "interfaceVersion": FOVEA_TRACE_INTERFACE_VERSION,
            "traceRetentionSeconds": retention,
            "retentionSource": retention_source,
            "memoryHorizonSeconds": memory_horizon,
            "active": events[-1] if events else None,
            "trace": events,
            "pendingRequestCount": len(self._pending_fovea_delivery_deadlines),
            "operatorRequestCommand": "request_fovea",
        }

    def _fovea_retention_contract(self) -> tuple[float, str, float | None]:
        horizon = getattr(self.agent_driver.agent, "foveation_memory_horizon_s", None)
        if horizon is not None and np.isfinite(horizon) and float(horizon) > 0.0:
            value = float(horizon)
            return value, "agent_foveation_memory_horizon", value
        return (
            float(self.fovea_trace_retention_s),
            "visualization_default_no_foveation_memory_model_declared",
            None,
        )

    def _camera_time_s(self) -> float:
        backend = getattr(self.sim.plant, "backend", None)
        if backend is not None:
            return float(backend.data.time)
        return float(self.sim_time)

    def _prune_fovea_trace(
        self,
        now_s: float,
        *,
        retention_s: float | None = None,
    ) -> None:
        retention = (
            self._fovea_retention_contract()[0]
            if retention_s is None
            else float(retention_s)
        )
        cutoff = now_s - retention
        self._fovea_trace = [
            event
            for event in self._fovea_trace
            if float(event["availableTimeS"]) >= cutoff - 1e-12
        ]

    def _clear_fovea_trace(self) -> None:
        self._fovea_trace.clear()
        self._pending_fovea_delivery_deadlines.clear()
        self._next_camera_delivery_poll_time_s = 0.0

    def _camera_backend(self) -> tuple[Any, Any]:
        backend = getattr(self.sim.plant, "backend", None)
        if self.plant_backend != "mujoco" or backend is None:
            raise RuntimeError(
                "model-facing camera rendering requires --plant-backend mujoco"
            )
        if self._camera_process is None:
            from .camera_observation import CameraObservationProcess

            self._camera_process = CameraObservationProcess()
        return self._camera_process, backend

    @staticmethod
    def _inspection_camera_available(process: Any, backend: Any) -> bool:
        park_qpos = backend.keyframe_qpos("camera_clear_park")
        return bool(
            np.max(np.abs(backend.data.qpos[:4] - park_qpos[:4]))
            <= np.deg2rad(2.0)
        )

    def _restart_after_stop_if_needed(self) -> bool:
        if not self.agent_driver.stopped:
            return False
        self._complete_stopped_painting()
        return True

    def _complete_stopped_painting(self) -> None:
        self.painting_count += 1
        if self.save_every_paintings > 0 and self.painting_count % self.save_every_paintings == 0:
            self.last_saved_canvas = str(self._save_canvas_snapshot(self.painting_count))
        self.sim.reset_pose()
        self.sim.canvas.clear()
        self.sim.intended_contact_pressure = 0.0
        self.sim.refresh_contact()
        self.agent_driver.reset(self.sim)

    def _save_canvas_snapshot(self, painting_index: int) -> Path:
        archive = Path(self.archive_dir)
        archive.mkdir(parents=True, exist_ok=True)
        path = archive / f"painting_{painting_index:04d}.png"
        gray = self._render_canvas_gray()
        Image.fromarray(gray, mode="L").save(path, format="PNG")
        return path

    def _render_canvas_gray(self) -> np.ndarray:
        tone = self.sim.canvas.observed_tone()
        return np.clip((1.0 - tone) * 255.0, 0, 255).astype(np.uint8)
