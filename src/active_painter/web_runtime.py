from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .arm_agent_driver import ArmActiveInferenceDriver
from .arm_control import scripted_contact_pressure, scripted_pose
from .arm_sim import ArmPainterSim, JOINT_NAMES
from .config import PainterConfig
from .telemetry_log import ArmTelemetryLog
from .version import CodeBuildInfo, code_build_info


@dataclass(slots=True)
class WebSimRuntime:
    canvas_size: int = 256
    speed: float = 1.0
    planner_state_kind: str = "summary"
    spatial_grid_size: int = 16
    stroke_tone_prior: float | None = None
    save_every_paintings: int = 5
    archive_dir: Path | str = Path("runs/web")
    telemetry_max_samples: int = 18_000
    telemetry_sample_period: float = 1.0 / 60.0
    driver_bootstrap_transitions: int = 96
    driver_bootstrap_train_steps: int = 180
    sim: ArmPainterSim = field(init=False)
    agent_driver: ArmActiveInferenceDriver = field(init=False)
    telemetry_log: ArmTelemetryLog = field(init=False)
    code_build: CodeBuildInfo = field(init=False)
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

    def __post_init__(self) -> None:
        self.code_build = code_build_info()
        sim_config = PainterConfig(
            canvas_size=self.canvas_size,
            planner_state_kind=self.planner_state_kind,
            spatial_grid_size=self.spatial_grid_size,
            stroke_tone_prior=self.stroke_tone_prior,
        )
        driver_config = PainterConfig(
            canvas_size=64,
            candidate_policies=48,
            planning_horizon=5,
            passage_proposal_mix=0.45,
            passage_plan_proposal_mix=0.20,
            policy_precision=0.35,
            batch_size=32,
            motor_forecast_candidates=3,
            planner_state_kind=self.planner_state_kind,
            spatial_grid_size=self.spatial_grid_size,
            stroke_tone_prior=self.stroke_tone_prior,
        )
        self.sim = ArmPainterSim(sim_config)
        self.agent_driver = ArmActiveInferenceDriver(
            config=driver_config,
            bootstrap_transitions=self.driver_bootstrap_transitions,
            bootstrap_train_steps=self.driver_bootstrap_train_steps,
            on_stop=self._complete_stopped_painting,
        )
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
            self.sim.paint_enabled = True
            self.sim.set_target(scripted_pose(self.sim_time))
            self.sim.intended_contact_pressure = scripted_contact_pressure(self.sim_time)
        self.sim.step(fixed_dt)
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
                self.sim_time = 0.0
                self.painting_count = 0
                self.last_saved_canvas = None
                self.telemetry_log.clear()
                self._next_telemetry_time = 0.0
                self.agent_driver.reset(self.sim)
                self._record_telemetry(force=True)
            elif action == "clear":
                self.sim.canvas.clear()
                self.agent_driver.reset(self.sim)
            elif action == "clear_telemetry":
                self.telemetry_log.clear()
                self._next_telemetry_time = self.sim_time
                self._record_telemetry(force=True)
            elif action == "toggle_paint":
                self.sim.paint_enabled = not self.sim.paint_enabled
            elif action == "toggle_agent":
                self.agent_enabled = not self.agent_enabled
                self.agent_driver.enabled = self.agent_enabled
            elif action == "tone":
                self.sim.brush_tone = 1.0 if str(data.get("value", "black")) == "black" else 0.0
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
                "paintEnabled": self.sim.paint_enabled,
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
                "contact": {
                    "onCanvas": contact.on_canvas,
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
        self.sim.paint_enabled = False
        self.sim.intended_contact_pressure = 0.0
        self.sim.contact = self.sim.canvas.contact_from_tip(
            self.sim.kinematics.tip(self.sim.actual_pose),
            self.sim.intended_contact_pressure,
        )
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
