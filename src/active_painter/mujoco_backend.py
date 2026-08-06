from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from .arm_control import ik_pose_for_canvas_point
from .arm_sim import (
    JOINT_NAMES,
    ArmKinematics,
    ArmPose,
    ContactState,
    MotorTelemetry,
    VerticalCanvas,
    clip_scalar,
)
from .plant_interface import (
    PhysicalSensorPacket,
    PlantCapabilities,
    PlantCommand,
    SimulatorEvaluationTruth,
)
from .web_robot_model import (
    INCH_TO_M,
    ROBOT_MODEL_PATH,
    load_robot_visual_model,
    retarget_legacy_robot_state,
)

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
    raise ImportError(
        "The MuJoCo backend requires the optional dependency: "
        'python -m pip install -e ".[mujoco]"'
    ) from exc


def _named_id(model: mujoco.MjModel, object_type: Any, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise ValueError(f"MuJoCo model has no {name!r} for {object_type}")
    return object_id


class MujocoPlantBackend:
    """SI-unit MuJoCo implementation of the backend-neutral plant contract."""

    def __init__(
        self,
        model_path: Path | str = ROBOT_MODEL_PATH,
        *,
        model: mujoco.MjModel | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.model = model or mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._joint_ids = {
            name: _named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        }
        self._actuator_ids = {
            name: _named_id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"{name}_position",
            )
            for name in JOINT_NAMES
        }
        self._sensor_ids = {
            name: _named_id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            for name in (
                *(f"{joint}_position_sensor" for joint in JOINT_NAMES),
                *(f"{joint}_velocity_sensor" for joint in JOINT_NAMES),
                "brush_bend_x_sensor",
                "brush_bend_z_sensor",
                "brush_compression_sensor",
                "brush_touch_sensor",
                "brush_force_sensor",
                "tip_position_sensor",
                *(f"{joint}_torque_sensor" for joint in JOINT_NAMES),
            )
        }
        self._brush_joint_id = _named_id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "brush_compression",
        )
        self._brush_bend_joint_ids = {
            axis: _named_id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"brush_bend_{axis}",
            )
            for axis in ("x", "z")
        }
        self._canvas_geom_id = _named_id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "canvas_surface",
        )
        self._bristle_geom_id = _named_id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "bristle_contact",
        )
        self._effective_motor_constants = self._custom_numeric(
            "robstride_effective_motor_constant_nm_per_a"
        )
        self._rated_current = self._custom_numeric("robstride_rated_current_arms")
        self._model_peak_current = self._custom_numeric(
            "robstride_model_peak_current_a"
        )
        self._bus_voltage = self._custom_numeric("robstride_rated_voltage_v")
        self._controller_kp = self._custom_numeric(
            "robstride_controller_kp_v_per_rad"
        )
        self._controller_kd = self._custom_numeric(
            "robstride_controller_kd_vs_per_rad"
        )
        self._pending_time_s = 0.0
        self._sequence = 0
        self._last_command: PlantCommand | None = None
        limits = tuple(
            tuple(float(value) for value in self.model.jnt_range[joint_id])
            for joint_id in self._joint_ids.values()
        )
        self._capabilities = PlantCapabilities(
            backend_id="mujoco-robstride-electromechanical-v4",
            joint_names=JOINT_NAMES,
            position_limits_rad=limits,
            command_modes=("position",),
            physical_sensor_fields=(
                "encoder_position_rad",
                "encoder_velocity_rad_s",
                "motor_current_a",
                "bus_voltage_v",
                "contact_force_n",
                "tool_deflection_m",
                "contact_switch",
            ),
            nominal_control_period_s=float(self.model.opt.timestep),
            supports_deterministic_step=True,
            supports_evaluation_truth=True,
        )
        mujoco.mj_forward(self.model, self.data)

    @property
    def capabilities(self) -> PlantCapabilities:
        return self._capabilities

    def _custom_numeric(self, name: str) -> np.ndarray:
        numeric_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
        address = int(self.model.numeric_adr[numeric_id])
        size = int(self.model.numeric_size[numeric_id])
        return np.asarray(self.model.numeric_data[address : address + size], dtype=np.float64)

    def _sensor(self, name: str) -> np.ndarray:
        sensor_id = self._sensor_ids[name]
        address = int(self.model.sensor_adr[sensor_id])
        size = int(self.model.sensor_dim[sensor_id])
        return np.asarray(self.data.sensordata[address : address + size], dtype=np.float64)

    def joint_position_rad(self) -> np.ndarray:
        return np.asarray(
            [
                self.data.qpos[int(self.model.jnt_qposadr[self._joint_ids[name]])]
                for name in JOINT_NAMES
            ],
            dtype=np.float64,
        )

    def joint_velocity_rad_s(self) -> np.ndarray:
        return np.asarray(
            [
                self.data.qvel[int(self.model.jnt_dofadr[self._joint_ids[name]])]
                for name in JOINT_NAMES
            ],
            dtype=np.float64,
        )

    def encoder_position_rad(self) -> np.ndarray:
        return np.asarray(
            [
                float(self._sensor(f"{name}_position_sensor")[0])
                for name in JOINT_NAMES
            ],
            dtype=np.float64,
        )

    def encoder_velocity_rad_s(self) -> np.ndarray:
        return np.asarray(
            [
                float(self._sensor(f"{name}_velocity_sensor")[0])
                for name in JOINT_NAMES
            ],
            dtype=np.float64,
        )

    def actuator_force_nm(self) -> np.ndarray:
        return np.asarray(
            [
                float(self._sensor(f"{name}_torque_sensor")[0])
                for name in JOINT_NAMES
            ],
            dtype=np.float64,
        )

    def actuator_current_a(self) -> np.ndarray:
        """Return the simulated output-equivalent winding-current state.

        With MuJoCo's dcmotor electrical dynamics enabled, the final activation
        slot is armature current. This reports that state without clipping:
        force saturation does not by itself clamp regenerative current, so an
        over-envelope value must remain observable rather than be hidden.
        """
        currents = np.empty(len(JOINT_NAMES), dtype=np.float64)
        torque = self.actuator_force_nm()
        for index, name in enumerate(JOINT_NAMES):
            actuator_id = self._actuator_ids[name]
            activation_address = int(self.model.actuator_actadr[actuator_id])
            activation_count = int(self.model.actuator_actnum[actuator_id])
            if activation_address >= 0 and activation_count > 0:
                current = float(
                    self.data.act[activation_address + activation_count - 1]
                )
            else:
                current = float(
                    torque[index]
                    / max(abs(self._effective_motor_constants[index]), 1e-9)
                )
            currents[index] = current
        return currents

    def rated_current_fraction(self) -> np.ndarray:
        return np.abs(self.actuator_current_a()) / np.maximum(
            self._rated_current,
            1e-9,
        )

    def applied_voltage_v(self) -> np.ndarray:
        """Return the dcmotor position controller's terminal-voltage request.

        The MJCF controller has no integral term, slew limit, or control delay,
        so its voltage request is exactly the clamped proportional-derivative
        expression below. This is distinct from the fixed 48 V DC bus exposed
        in :class:`PhysicalSensorPacket`.
        """
        voltage = np.empty(len(JOINT_NAMES), dtype=np.float64)
        for index, name in enumerate(JOINT_NAMES):
            actuator_id = self._actuator_ids[name]
            position_error = (
                float(self.data.ctrl[actuator_id])
                - float(self.data.actuator_length[actuator_id])
            )
            velocity = float(self.data.actuator_velocity[actuator_id])
            request = (
                self._controller_kp[index] * position_error
                - self._controller_kd[index] * velocity
            )
            voltage[index] = float(
                np.clip(request, -self._bus_voltage[index], self._bus_voltage[index])
            )
        return voltage

    def tip_position_m(self) -> np.ndarray:
        return self._sensor("tip_position_sensor").copy()

    def brush_compression_m(self) -> float:
        value = float(self._sensor("brush_compression_sensor")[0])
        return max(0.0, -value)

    def brush_bend_rad(self) -> np.ndarray:
        """Return the two lumped ferrule-flexure angles in local x/z order."""
        return np.asarray(
            [
                float(self._sensor("brush_bend_x_sensor")[0]),
                float(self._sensor("brush_bend_z_sensor")[0]),
            ],
            dtype=np.float64,
        )

    def contact_force_n(self) -> float:
        return max(0.0, float(self._sensor("brush_touch_sensor")[0]))

    def exact_brush_canvas_contact(self) -> bool:
        expected = frozenset((self._canvas_geom_id, self._bristle_geom_id))
        return any(
            frozenset(
                (
                    int(self.data.contact[index].geom1),
                    int(self.data.contact[index].geom2),
                )
            )
            == expected
            for index in range(self.data.ncon)
        )

    def keyframe_qpos(self, name: str) -> np.ndarray:
        key_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        return np.asarray(self.model.key_qpos[key_id], dtype=np.float64).copy()

    def set_state(
        self,
        joint_position_rad: np.ndarray,
        *,
        joint_velocity_rad_s: np.ndarray | None = None,
        control_rad: np.ndarray | None = None,
    ) -> None:
        joint_position_rad = np.asarray(joint_position_rad, dtype=np.float64)
        if joint_position_rad.shape != (len(JOINT_NAMES),):
            raise ValueError("joint_position_rad must contain yaw, pitch, roll, elbow")
        mujoco.mj_resetData(self.model, self.data)
        for index, name in enumerate(JOINT_NAMES):
            self.data.qpos[int(self.model.jnt_qposadr[self._joint_ids[name]])] = (
                joint_position_rad[index]
            )
        if joint_velocity_rad_s is not None:
            joint_velocity_rad_s = np.asarray(joint_velocity_rad_s, dtype=np.float64)
            for index, name in enumerate(JOINT_NAMES):
                self.data.qvel[int(self.model.jnt_dofadr[self._joint_ids[name]])] = (
                    joint_velocity_rad_s[index]
                )
        self.data.ctrl[:] = (
            joint_position_rad
            if control_rad is None
            else np.asarray(control_rad, dtype=np.float64)
        )
        self._pending_time_s = 0.0
        mujoco.mj_forward(self.model, self.data)

    def send_command(self, command: PlantCommand) -> None:
        command.validate_for(self.capabilities)
        self._last_command = command
        self._sequence = command.sequence
        self.data.ctrl[:] = np.asarray(command.position_target_rad, dtype=np.float64)

    def step(self, duration_s: float) -> None:
        if not np.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("duration_s must be finite and non-negative")
        self._pending_time_s += float(duration_s)
        timestep = float(self.model.opt.timestep)
        while self._pending_time_s + 1e-15 >= timestep:
            mujoco.mj_step(self.model, self.data)
            self._pending_time_s -= timestep

    def read_sensors(self) -> PhysicalSensorPacket:
        current = self.actuator_current_a()
        flags: list[str] = []
        if np.any(np.abs(current) > self._model_peak_current + 1e-9):
            flags.append("model_peak_current_exceeded")
        if not (
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
            and np.isfinite(self.data.sensordata).all()
        ):
            flags.append("nonfinite_state")
        return PhysicalSensorPacket(
            sequence=self._sequence,
            monotonic_time_s=float(self.data.time),
            joint_names=JOINT_NAMES,
            encoder_position_rad=tuple(float(value) for value in self.encoder_position_rad()),
            encoder_velocity_rad_s=tuple(
                float(value) for value in self.encoder_velocity_rad_s()
            ),
            motor_current_a=tuple(float(value) for value in current),
            bus_voltage_v=tuple(float(value) for value in self._bus_voltage),
            contact_force_n=self.contact_force_n(),
            tool_deflection_m=self.brush_compression_m(),
            contact_switch=self.exact_brush_canvas_contact(),
            fault_flags=tuple(flags),
        )

    def read_evaluation_truth(self) -> SimulatorEvaluationTruth:
        qfrc = np.asarray(
            [
                self.data.qfrc_actuator[
                    int(self.model.jnt_dofadr[self._joint_ids[name]])
                ]
                for name in JOINT_NAMES
            ],
            dtype=np.float64,
        )
        return SimulatorEvaluationTruth(
            monotonic_time_s=float(self.data.time),
            joint_names=JOINT_NAMES,
            actual_position_rad=tuple(float(value) for value in self.joint_position_rad()),
            actual_velocity_rad_s=tuple(
                float(value) for value in self.joint_velocity_rad_s()
            ),
            applied_torque_nm=tuple(float(value) for value in qfrc),
            exact_contact=self.exact_brush_canvas_contact(),
            exact_contact_force_n=self.contact_force_n(),
            hidden_state_revision=self._sequence,
        )

    def state_snapshot(self) -> dict[str, Any]:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "act": self.data.act.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": float(self.data.time),
            "pending_time_s": self._pending_time_s,
            "sequence": self._sequence,
            "last_command": self._last_command,
        }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = snapshot["qpos"]
        self.data.qvel[:] = snapshot["qvel"]
        if self.data.act.size:
            self.data.act[:] = snapshot["act"]
        self.data.ctrl[:] = snapshot["ctrl"]
        self.data.time = float(snapshot["time"])
        self._pending_time_s = float(snapshot["pending_time_s"])
        self._sequence = int(snapshot["sequence"])
        self._last_command = snapshot["last_command"]
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        return


class MujocoJointPlant:
    """Compatibility facade from the current canvas controller to MuJoCo.

    Actual execution and deep-copied oracle counterfactuals both use independent
    MuJoCo data under the same immutable MJCF model. The legacy logical
    controller retarget remains an explicit approximation below policy
    selection.
    """

    handles_contact = True
    backend_id = "mujoco-robstride-electromechanical-v4"
    counterfactual_backend_id = "mujoco-robstride-electromechanical-v4"
    counterfactual_initialization = "baseline-oracle-v0 exact MuJoCo process snapshot"
    counterfactual_approximation = (
        "exact MJCF dynamics/contact under legacy_canvas_cartesian_retarget; "
        "MuJoCo body-parameter uncertainty is not yet sampled"
    )

    # Compatibility parameters consumed by existing telemetry/EFE code.
    current_limit = 25.5
    kt = 1.17
    max_link_velocity = 5.0
    inertia = 0.08
    link_inertia: dict[str, float] | float = 0.08
    motor_inertia: dict[str, float] | float = 0.0
    encoder_base_noise_deg = 0.0
    encoder_velocity_noise_deg = 0.0
    process_torque_noise_std: dict[str, float] | float = 0.0

    def __init__(
        self,
        model_path: Path | str = ROBOT_MODEL_PATH,
        *,
        model: mujoco.MjModel | None = None,
    ) -> None:
        self.backend = MujocoPlantBackend(model_path, model=model)
        self.robot_model = load_robot_visual_model(model_path)
        self._native_kinematics = ArmKinematics()
        self.velocity = dict.fromkeys(JOINT_NAMES, 0.0)
        self.motor_angle = dict.fromkeys(JOINT_NAMES, 0.0)
        self.motor_velocity = dict.fromkeys(JOINT_NAMES, 0.0)
        self.temperature = dict.fromkeys(JOINT_NAMES, 0.0)
        self.telemetry = MotorTelemetry()
        self._logical_pose = ArmPose()
        self._physical_target_deg = dict.fromkeys(JOINT_NAMES, 0.0)
        self._last_roll_preserved = True
        self._sequence = 0
        self._last_contact = ContactState(
            False,
            0.0,
            0.0,
            0.0,
            0.0,
            np.zeros(3, dtype=np.float64),
        )
        self.forecast_current_scale = self.backend._model_peak_current.copy()
        self.forecast_torque_scale = self.backend._custom_numeric(
            "robstride_peak_torque_nm"
        )
        self.forecast_velocity_scale = self.backend._custom_numeric(
            "robstride_no_load_speed_rad_s"
        )
        nominal_inertia = np.asarray(
            [
                self.backend.model.dof_M0[
                    int(self.backend.model.jnt_dofadr[self.backend._joint_ids[name]])
                ]
                for name in JOINT_NAMES
            ],
            dtype=np.float64,
        )
        self.forecast_acceleration_scale = self.forecast_torque_scale / np.maximum(
            nominal_inertia,
            1e-6,
        )

    @staticmethod
    def _joint_param(
        values: dict[str, float] | float,
        name: str,
        fallback: float,
    ) -> float:
        if isinstance(values, dict):
            return float(values.get(name, fallback))
        return float(values)

    def __deepcopy__(self, memo: dict[int, Any]) -> MujocoJointPlant:
        clone = type(self)(
            self.backend.model_path,
            model=self.backend.model,
        )
        memo[id(self)] = clone
        clone.restore_state(copy.deepcopy(self.state_snapshot(), memo))
        return clone

    def select_forecast_noise_sample(self, sample_index: int) -> None:
        _ = sample_index

    def _legacy_point_from_physical(self, point_m: np.ndarray) -> np.ndarray:
        canvas = self.robot_model["canvas"]
        source_y = self.robot_model["compatibility"]["sourceCanvasContactY"]
        return np.asarray(
            [
                (float(point_m[0]) - canvas["center"][0]) / INCH_TO_M,
                source_y + (float(point_m[1]) - canvas["contactY"]) / INCH_TO_M,
                (float(point_m[2]) - canvas["center"][2]) / INCH_TO_M,
            ],
            dtype=np.float64,
        )

    def _logical_pose_from_backend(self) -> ArmPose:
        logical_tip = self._legacy_point_from_physical(self.backend.tip_position_m())
        roll_deg = float(np.rad2deg(self.backend.encoder_position_rad()[2]))
        try:
            return ik_pose_for_canvas_point(
                float(logical_tip[0]),
                float(logical_tip[2]),
                float(logical_tip[1]),
                upper_arm_roll_deg=roll_deg,
            )
        except ValueError:
            try:
                return ik_pose_for_canvas_point(
                    float(logical_tip[0]),
                    float(logical_tip[2]),
                    float(logical_tip[1]),
                    upper_arm_roll_deg=0.0,
                )
            except ValueError:
                return self._logical_pose

    def _physical_target_for(self, pose: ArmPose) -> dict[str, float]:
        legacy_tip = self._native_kinematics.tip(pose)
        current = {
            name: float(np.rad2deg(value))
            for name, value in zip(
                JOINT_NAMES,
                self.backend.encoder_position_rad(),
                strict=True,
            )
        }
        target_state = retarget_legacy_robot_state(
            self.robot_model,
            {name: float(getattr(pose, name)) for name in JOINT_NAMES},
            legacy_tip,
            current,
        )
        self._last_roll_preserved = bool(target_state["rollPreserved"])
        return target_state["jointPositionDeg"]

    def telemetry_joint_position_deg(self) -> dict[str, float]:
        """Return physical encoder coordinates for the shared telemetry log."""
        return {
            name: float(np.rad2deg(value))
            for name, value in zip(
                JOINT_NAMES,
                self.backend.encoder_position_rad(),
                strict=True,
            )
        }

    def telemetry_joint_target_deg(self) -> dict[str, float]:
        """Return the physical actuator target corresponding to the controller target."""
        return dict(self._physical_target_deg)

    def forecast_joint_limit_proximity_vector(
        self,
        margin_degrees: float,
    ) -> np.ndarray:
        """Return physical MJCF joint-limit proximity in declared joint order."""

        margin = max(np.deg2rad(float(margin_degrees)), 1e-9)
        position = self.backend.encoder_position_rad()
        proximity = []
        for index, limits in enumerate(self.backend.capabilities.position_limits_rad):
            low, high = limits
            distance = min(position[index] - low, high - position[index])
            proximity.append(clip_scalar((margin - distance) / margin, 0.0, 1.0))
        return np.asarray(proximity, dtype=np.float64)

    def reset_state(self, pose: ArmPose) -> None:
        self._logical_pose = pose.clipped()
        target = self._physical_target_for(self._logical_pose)
        qpos = np.asarray([np.deg2rad(target[name]) for name in JOINT_NAMES])
        self.backend.set_state(qpos, control_rad=qpos)
        self._physical_target_deg = target
        self._sequence = 0
        self._sync_telemetry()

    def step(
        self,
        actual: ArmPose,
        target: ArmPose,
        dt: float,
        contact_force: float = 0.0,
        damping_multiplier: float = 1.0,
    ) -> ArmPose:
        _ = actual, contact_force, damping_multiplier
        self._physical_target_deg = self._physical_target_for(target.clipped())
        self._sequence += 1
        q_target = tuple(
            float(np.deg2rad(self._physical_target_deg[name])) for name in JOINT_NAMES
        )
        self.backend.send_command(
            PlantCommand(
                sequence=self._sequence,
                monotonic_time_s=float(self.backend.data.time),
                position_target_rad=q_target,
            )
        )
        self.backend.step(dt)
        self._logical_pose = self._logical_pose_from_backend()
        self._sync_telemetry()
        return self._logical_pose

    def _sync_telemetry(self) -> None:
        packet = self.backend.read_sensors()
        q = np.asarray(packet.encoder_position_rad)
        velocity = np.asarray(packet.encoder_velocity_rad_s)
        torque = self.backend.actuator_force_nm()
        current = np.asarray(packet.motor_current_a)
        applied_voltage = self.backend.applied_voltage_v()
        target = np.asarray(
            [np.deg2rad(self._physical_target_deg[name]) for name in JOINT_NAMES]
        )
        constraint = np.asarray(
            [
                self.backend.data.qfrc_constraint[
                    int(
                        self.backend.model.jnt_dofadr[
                            self.backend._joint_ids[name]
                        ]
                    )
                ]
                for name in JOINT_NAMES
            ]
        )
        bias = np.asarray(
            [
                self.backend.data.qfrc_bias[
                    int(
                        self.backend.model.jnt_dofadr[
                            self.backend._joint_ids[name]
                        ]
                    )
                ]
                for name in JOINT_NAMES
            ]
        )
        for index, name in enumerate(JOINT_NAMES):
            self.velocity[name] = float(velocity[index])
            self.motor_velocity[name] = float(velocity[index])
            self.motor_angle[name] = float(q[index])
            self.temperature[name] = 0.0
            self.telemetry.voltage[name] = float(applied_voltage[index])
            self.telemetry.current[name] = float(current[index])
            self.telemetry.torque[name] = float(torque[index])
            self.telemetry.actuator_angle_deg[name] = float(np.rad2deg(q[index]))
            self.telemetry.actuator_velocity_rad_s[name] = float(velocity[index])
            self.telemetry.encoder_angle_deg[name] = float(np.rad2deg(q[index]))
            self.telemetry.encoder_velocity_rad_s[name] = float(velocity[index])
            self.telemetry.position_error_deg[name] = float(
                np.rad2deg(target[index] - q[index])
            )
            self.telemetry.elastic_deflection_deg[name] = 0.0
            self.telemetry.backlash_deflection_deg[name] = 0.0
            self.telemetry.friction_torque[name] = 0.0
            self.telemetry.load_torque[name] = float(constraint[index])
            self.telemetry.gravity_torque[name] = float(bias[index])
            self.telemetry.coupling_torque[name] = 0.0
            self.telemetry.process_torque[name] = 0.0
            self.telemetry.encoder_std_deg[name] = 0.0
            self.telemetry.thermal_fraction[name] = 0.0
            self.telemetry.torque_limit_fraction[name] = 1.0

    def contact_state(
        self,
        canvas: VerticalCanvas,
        intended_pressure: float = 0.0,
    ) -> ContactState:
        _ = intended_pressure
        logical_tip = self._legacy_point_from_physical(self.backend.tip_position_m())
        touching = self.backend.exact_brush_canvas_contact()
        on_canvas = canvas.contains(float(logical_tip[0]), float(logical_tip[2]))
        compression_m = min(
            self.robot_model["brush"]["compressionTravel"],
            self.backend.brush_compression_m(),
        )
        force = self.backend.contact_force_n() if touching else 0.0
        spring = float(
            self.backend.model.jnt_stiffness[self.backend._brush_joint_id]
        )
        force_scale = max(
            1e-6,
            spring * self.robot_model["brush"]["compressionTravel"],
        )
        compression_fraction = compression_m / max(
            1e-9,
            self.robot_model["brush"]["compressionTravel"],
        )
        pressure = (
            clip_scalar(
                max(compression_fraction, force / force_scale),
                0.0,
                1.0,
            )
            if touching and on_canvas
            else 0.0
        )
        brush_world = logical_tip.copy()
        if touching and on_canvas:
            brush_world[1] = canvas.distance
        contact = ContactState(
            on_canvas=bool(on_canvas),
            deflection=float(compression_m / INCH_TO_M),
            force=float(force),
            pressure=float(pressure),
            brush_width_px=float(
                2.0 * canvas.brush_radius_world(pressure) * canvas._pixels_per_unit()
            ),
            brush_world=brush_world,
        )
        self._last_contact = contact
        return contact

    def state_snapshot(self) -> dict[str, Any]:
        return {
            "backend": self.backend.state_snapshot(),
            "logical_pose": copy.deepcopy(self._logical_pose),
            "physical_target_deg": dict(self._physical_target_deg),
            "last_roll_preserved": self._last_roll_preserved,
            "sequence": self._sequence,
            "telemetry": copy.deepcopy(self.telemetry),
            "last_contact": copy.deepcopy(self._last_contact),
        }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        self.backend.restore_state(snapshot["backend"])
        self._logical_pose = copy.deepcopy(snapshot["logical_pose"])
        self._physical_target_deg = dict(snapshot["physical_target_deg"])
        self._last_roll_preserved = bool(snapshot["last_roll_preserved"])
        self._sequence = int(snapshot["sequence"])
        self.telemetry = copy.deepcopy(snapshot["telemetry"])
        self._last_contact = copy.deepcopy(snapshot.get("last_contact", self._last_contact))
        self._sync_telemetry()

    def close(self) -> None:
        self.backend.close()

    def web_robot_state(
        self,
        controller_pose: ArmPose,
        controller_target_pose: ArmPose,
    ) -> dict[str, Any]:
        q = self.backend.encoder_position_rad()
        target_m = np.asarray(
            retarget_legacy_robot_state(
                self.robot_model,
                {
                    name: float(getattr(controller_target_pose, name))
                    for name in JOINT_NAMES
                },
                self._native_kinematics.tip(controller_target_pose),
                self._physical_target_deg,
            )["mappedCartesianTargetM"]
        )
        tip_m = self.backend.tip_position_m()
        brush_bend = self.backend.brush_bend_rad()
        return {
            "mode": "mujoco_direct",
            "backendId": self.backend_id,
            "counterfactualBackend": self.counterfactual_backend_id,
            "jointPositionDeg": {
                name: float(np.rad2deg(value))
                for name, value in zip(JOINT_NAMES, q, strict=True)
            },
            "controllerJointPositionDeg": {
                name: float(getattr(controller_pose, name)) for name in JOINT_NAMES
            },
            "tipM": tip_m.astype(float).tolist(),
            "mappedCartesianTargetM": target_m.astype(float).tolist(),
            "alignmentErrorM": float(np.linalg.norm(tip_m - target_m)),
            "rollPreserved": self._last_roll_preserved,
            "brushCompressionM": self.backend.brush_compression_m(),
            "brushBendRad": {
                "x": float(brush_bend[0]),
                "z": float(brush_bend[1]),
            },
        }
