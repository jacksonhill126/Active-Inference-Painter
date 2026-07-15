from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .config import PainterConfig


JOINT_NAMES = ("yaw", "pitch", "roll", "elbow")


def clip_scalar(value: float, lower: float, upper: float) -> float:
    value = float(value)
    if value < lower:
        return float(lower)
    if value > upper:
        return float(upper)
    return value


def safe_home_pose() -> "ArmPose":
    return ArmPose(yaw=0.0, pitch=-50.0, roll=0.0, elbow=100.0)


def _make_canvas_grain(n: int, period_px: float, seed: int) -> np.ndarray:
    """A fixed canvas tooth/height field in [0, 1]: smoothed value noise with a
    faint woven bias, so a lightly loaded brush catches only the raised tooth."""
    rng = np.random.default_rng(int(seed))
    period = max(1.5, float(period_px))
    coarse = max(2, int(np.ceil(n / period)) + 1)
    lattice = rng.random((coarse, coarse)).astype(np.float32)
    # Bilinear upsample the coarse lattice to n x n (separable interpolation).
    idx = np.linspace(0.0, coarse - 1.0, n).astype(np.float32)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, coarse - 1)
    frac = (idx - lo).astype(np.float32)
    rows = lattice[lo] * (1.0 - frac)[:, None] + lattice[hi] * frac[:, None]
    grain = rows[:, lo] * (1.0 - frac)[None, :] + rows[:, hi] * frac[None, :]
    # Faint canvas weave so texture reads as fabric rather than pure blobs.
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    weave = 0.5 + 0.5 * np.sin(2.0 * np.pi * xx / period) * np.sin(2.0 * np.pi * yy / period)
    grain = 0.7 * grain + 0.3 * weave
    # Histogram-equalise to a uniform tooth-height distribution: summing smooth
    # fields bunches values near the middle, which would leave almost no cells
    # below a light-pressure waterline (no dry-brush sparkle) and almost none
    # above a heavy one. Ranks make the reach -> fill mapping predictable.
    order = np.argsort(grain, axis=None)
    ranks = np.empty(order.size, dtype=np.float32)
    ranks[order] = np.arange(order.size, dtype=np.float32)
    grain = (ranks / max(1.0, float(order.size - 1))).reshape(grain.shape)
    return grain.astype(np.float32)


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass(slots=True)
class ArmPose:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    elbow: float = 0.0

    def radians(self) -> dict[str, float]:
        return {name: np.deg2rad(getattr(self, name)) for name in JOINT_NAMES}

    def clipped(self) -> "ArmPose":
        return ArmPose(
            yaw=clip_scalar(self.yaw, -90.0, 90.0),
            pitch=clip_scalar(self.pitch, -90.0, 90.0),
            roll=clip_scalar(self.roll, -180.0, 180.0),
            elbow=clip_scalar(self.elbow, 0.0, 150.0),
        )


@dataclass(slots=True)
class ArmKinematics:
    upper_arm: float = 13.0
    lower_arm: float = 13.0

    def joint_points(self, pose: ArmPose) -> np.ndarray:
        q = pose.clipped().radians()
        base = np.zeros(3, dtype=np.float64)
        r_shoulder = _rot_z(q["yaw"]) @ _rot_x(q["pitch"])
        elbow = r_shoulder @ np.asarray([0.0, self.upper_arm, 0.0])
        r_forearm = r_shoulder @ _rot_y(q["roll"]) @ _rot_x(q["elbow"])
        tip = elbow + r_forearm @ np.asarray([0.0, self.lower_arm, 0.0])
        return np.stack([base, elbow, tip])

    def tip(self, pose: ArmPose) -> np.ndarray:
        return self.joint_points(pose)[-1]

    def upper_arm_axis(self, pose: ArmPose) -> np.ndarray:
        """World-space axis about which the upper-arm roll joint rotates."""

        q = pose.clipped().radians()
        return _rot_z(q["yaw"]) @ _rot_x(q["pitch"]) @ np.asarray([0.0, 1.0, 0.0])

    def elbow_hinge_axis(self, pose: ArmPose) -> np.ndarray:
        """World-space elbow hinge axis after upper-arm roll."""

        q = pose.clipped().radians()
        r_shoulder = _rot_z(q["yaw"]) @ _rot_x(q["pitch"])
        return r_shoulder @ _rot_y(q["roll"]) @ np.asarray([1.0, 0.0, 0.0])


@dataclass(slots=True)
class MotorTelemetry:
    voltage: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    current: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    torque: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    actuator_angle_deg: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    actuator_velocity_rad_s: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    encoder_angle_deg: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    encoder_velocity_rad_s: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    position_error_deg: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    elastic_deflection_deg: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    backlash_deflection_deg: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    friction_torque: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    load_torque: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    gravity_torque: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    coupling_torque: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    process_torque: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    encoder_std_deg: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    thermal_fraction: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    torque_limit_fraction: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))


@dataclass(slots=True)
class JointPlant:
    """Probabilistic coupled actuator/link process beneath painting inference.

    The values are representative small-arm parameters, not vendor-specific
    motor measurements. The plant exposes prediction-error-relevant mechanics
    to policy forecasts while safety limits stay external to painting choice.
    """

    supply_voltage: float = 24.0
    current_limit: float = 7.0
    servo_stiffness: float = 1.0
    damping: float = 0.80
    inertia: float = 0.065
    kt: float = 0.42
    resistance: float = 2.1
    motor_inertia: dict[str, float] | float = field(
        default_factory=lambda: {"yaw": 0.012, "pitch": 0.014, "roll": 0.006, "elbow": 0.010}
    )
    link_inertia: dict[str, float] | float = field(
        default_factory=lambda: {"yaw": 0.060, "pitch": 0.074, "roll": 0.036, "elbow": 0.060}
    )
    transmission_stiffness: dict[str, float] | float = field(
        default_factory=lambda: {"yaw": 28.0, "pitch": 32.0, "roll": 18.0, "elbow": 24.0}
    )
    transmission_damping: dict[str, float] | float = field(
        default_factory=lambda: {"yaw": 0.72, "pitch": 0.82, "roll": 0.46, "elbow": 0.62}
    )
    motor_viscous_friction: dict[str, float] | float = field(
        default_factory=lambda: {"yaw": 0.018, "pitch": 0.022, "roll": 0.012, "elbow": 0.016}
    )
    link_viscous_friction: dict[str, float] | float = field(
        default_factory=lambda: {"yaw": 0.010, "pitch": 0.014, "roll": 0.007, "elbow": 0.010}
    )
    coulomb_friction: dict[str, float] | float = field(
        default_factory=lambda: {"yaw": 0.018, "pitch": 0.025, "roll": 0.012, "elbow": 0.018}
    )
    static_friction: dict[str, float] | float = field(
        default_factory=lambda: {"yaw": 0.030, "pitch": 0.040, "roll": 0.020, "elbow": 0.030}
    )
    backlash_deadband_deg: dict[str, float] | float = field(
        default_factory=lambda: {"yaw": 0.035, "pitch": 0.045, "roll": 0.060, "elbow": 0.040}
    )
    contact_load_gain: dict[str, float] | float = field(
        default_factory=lambda: {"yaw": 0.0015, "pitch": 0.0050, "roll": 0.0010, "elbow": 0.0040}
    )
    pitch_elbow_coupling_inertia: float = 0.018
    yaw_roll_coupling_inertia: float = 0.003
    upper_arm_mass_kg: float = 1.35
    lower_arm_mass_kg: float = 0.85
    brush_payload_mass_kg: float = 0.18
    link_length_m: float = 0.3302
    gravity_m_s2: float = 9.81
    gravity_compensation_fraction: float = 0.985
    process_torque_noise_std: dict[str, float] | float = field(
        default_factory=lambda: {"yaw": 0.006, "pitch": 0.010, "roll": 0.005, "elbow": 0.008}
    )
    max_motor_velocity: float = 7.0
    max_link_velocity: float = 5.0
    thermal_time_constant: float = 18.0
    cooling_time_constant: float = 65.0
    thermal_current_derate: float = 0.35
    encoder_base_noise_deg: float = 0.035
    encoder_velocity_noise_deg: float = 0.020
    encoder_current_noise_deg: float = 0.025
    encoder_contact_noise_deg: float = 0.003
    encoder_position_bias_deg: dict[str, float] | float = 0.0
    encoder_velocity_bias_rad_s: dict[str, float] | float = 0.0
    process_noise_enabled: bool = True
    encoder_noise_enabled: bool = True
    rng_seed: int | None = 0
    velocity: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    motor_angle: dict[str, float] = field(default_factory=dict)
    motor_velocity: dict[str, float] = field(default_factory=dict)
    temperature: dict[str, float] = field(default_factory=dict)
    telemetry: MotorTelemetry = field(default_factory=MotorTelemetry)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.rng_seed)

    def reset_state(self, pose: ArmPose) -> None:
        q = pose.clipped().radians()
        self.velocity = dict.fromkeys(JOINT_NAMES, 0.0)
        self.motor_angle = {name: float(q[name]) for name in JOINT_NAMES}
        self.motor_velocity = dict.fromkeys(JOINT_NAMES, 0.0)
        self.temperature = dict.fromkeys(JOINT_NAMES, 0.0)
        self.telemetry = MotorTelemetry()
        for name in JOINT_NAMES:
            self.telemetry.actuator_angle_deg[name] = float(getattr(pose.clipped(), name))
            self.telemetry.encoder_angle_deg[name] = float(getattr(pose.clipped(), name))

    def select_forecast_noise_sample(self, sample_index: int) -> None:
        """Choose a reproducible independent continuation for a rollout particle."""

        if sample_index > 0:
            self._rng.random(97 * int(sample_index))

    def state_snapshot(self) -> dict[str, object]:
        return {
            "velocity": dict(self.velocity),
            "motor_angle": dict(self.motor_angle),
            "motor_velocity": dict(self.motor_velocity),
            "temperature": dict(self.temperature),
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "telemetry": {
                field_name: dict(getattr(self.telemetry, field_name))
                for field_name in MotorTelemetry.__dataclass_fields__
            },
        }

    def restore_state(self, snapshot: dict[str, object]) -> None:
        self.velocity = dict(snapshot["velocity"])  # type: ignore[arg-type]
        self.motor_angle = dict(snapshot["motor_angle"])  # type: ignore[arg-type]
        self.motor_velocity = dict(snapshot["motor_velocity"])  # type: ignore[arg-type]
        self.temperature = dict(snapshot["temperature"])  # type: ignore[arg-type]
        rng_state = snapshot.get("rng_state")
        if isinstance(rng_state, dict):
            self._rng.bit_generator.state = copy.deepcopy(rng_state)
        telemetry_values = snapshot["telemetry"]
        self.telemetry = MotorTelemetry()
        if isinstance(telemetry_values, dict):
            for field_name in MotorTelemetry.__dataclass_fields__:
                values = telemetry_values.get(field_name, {})
                if isinstance(values, dict):
                    setattr(self.telemetry, field_name, dict(values))

    def _ensure_state(self, pose: ArmPose) -> None:
        q = pose.clipped().radians()
        for name in JOINT_NAMES:
            self.velocity.setdefault(name, 0.0)
            self.motor_angle.setdefault(name, float(q[name]))
            self.motor_velocity.setdefault(name, float(self.velocity[name]))
            self.temperature.setdefault(name, 0.0)

    @staticmethod
    def _joint_param(values: dict[str, float] | float, name: str, fallback: float) -> float:
        if isinstance(values, dict):
            return float(values.get(name, fallback))
        return float(values)

    @staticmethod
    def _compliance_deflection(raw_deflection: float, deadband: float) -> tuple[float, float]:
        if abs(raw_deflection) <= deadband:
            return 0.0, raw_deflection
        sign = float(np.sign(raw_deflection))
        return raw_deflection - sign * deadband, sign * deadband

    def _friction_torque(self, name: str, drive: float, link_velocity: float) -> float:
        static = self._joint_param(self.static_friction, name, 0.15)
        coulomb = self._joint_param(self.coulomb_friction, name, 0.10)
        viscous = self._joint_param(self.link_viscous_friction, name, 0.06)
        if abs(link_velocity) < 0.015 and abs(drive) < static:
            return float(drive)
        direction = float(np.sign(link_velocity if abs(link_velocity) >= 0.015 else drive))
        return float(coulomb * direction + viscous * link_velocity)

    def _encoder_std_deg(self, name: str, velocity: float, current: float, contact_force: float) -> float:
        _ = name
        return float(
            self.encoder_base_noise_deg
            + self.encoder_velocity_noise_deg * abs(velocity)
            + self.encoder_current_noise_deg * abs(current) / max(1e-6, self.current_limit)
            + self.encoder_contact_noise_deg * contact_force
        )

    def _mass_matrix(self, q: np.ndarray) -> np.ndarray:
        diagonal = np.asarray(
            [
                self._joint_param(self.link_inertia, name, self.inertia)
                + self._joint_param(self.motor_inertia, name, 0.0)
                for name in JOINT_NAMES
            ],
            dtype=np.float64,
        )
        matrix = np.diag(np.maximum(diagonal, 1e-5))
        yaw, pitch, roll, elbow = q
        _ = yaw
        pitch_elbow = self.pitch_elbow_coupling_inertia * np.cos(elbow)
        yaw_roll = self.yaw_roll_coupling_inertia * np.cos(pitch) * np.cos(roll)
        matrix[1, 3] = matrix[3, 1] = pitch_elbow
        matrix[0, 2] = matrix[2, 0] = yaw_roll
        return matrix

    def _coupled_loads(self, q: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, pitch, roll, elbow = q
        _, pitch_w, _, elbow_w = velocity
        h = self.pitch_elbow_coupling_inertia * np.sin(elbow)
        coriolis = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        coriolis[1] = -h * (2.0 * pitch_w * elbow_w + elbow_w * elbow_w)
        coriolis[3] = h * pitch_w * pitch_w

        residual = max(0.0, 1.0 - float(self.gravity_compensation_fraction))
        length = self.link_length_m
        lower_moment = (0.5 * self.lower_arm_mass_kg + self.brush_payload_mass_kg) * length
        shoulder_moment = (
            0.5 * self.upper_arm_mass_kg * length
            + (self.lower_arm_mass_kg + self.brush_payload_mass_kg) * length
        )
        gravity = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        roll_projection = np.cos(roll)
        gravity[1] = residual * self.gravity_m_s2 * roll_projection * (
            shoulder_moment * np.cos(pitch)
            + lower_moment * np.cos(pitch + elbow)
        )
        gravity[3] = (
            residual
            * self.gravity_m_s2
            * roll_projection
            * lower_moment
            * np.cos(pitch + elbow)
        )
        return coriolis, gravity

    def step(
        self,
        actual: ArmPose,
        target: ArmPose,
        dt: float,
        contact_force: float = 0.0,
        damping_multiplier: float = 1.0,
    ) -> ArmPose:
        actual = actual.clipped()
        target = target.clipped()
        self._ensure_state(actual)
        contact_force = max(0.0, float(contact_force))
        effective_damping = self.damping * max(0.0, float(damping_multiplier))
        q = np.asarray([np.deg2rad(getattr(actual, name)) for name in JOINT_NAMES], dtype=np.float64)
        q_target = np.asarray([np.deg2rad(getattr(target, name)) for name in JOINT_NAMES], dtype=np.float64)
        link_velocity = np.asarray([self.velocity[name] for name in JOINT_NAMES], dtype=np.float64)
        previous_motor_q = np.asarray([self.motor_angle[name] for name in JOINT_NAMES], dtype=np.float64)
        temperatures = np.asarray([clip_scalar(self.temperature[name], 0.0, 1.0) for name in JOINT_NAMES])
        currents = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        voltages = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        motor_torques = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        load_torques = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        friction_torques = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        command_errors = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        backlash_deflections = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        current_limits = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        encoder_stds = np.zeros(len(JOINT_NAMES), dtype=np.float64)

        for index, name in enumerate(JOINT_NAMES):
            temperature = float(temperatures[index])
            current_limit = self.current_limit * max(0.25, 1.0 - self.thermal_current_derate * temperature)
            current_limits[index] = current_limit
            deadband = np.deg2rad(self._joint_param(self.backlash_deadband_deg, name, 0.2))
            encoder_stds[index] = self._encoder_std_deg(name, link_velocity[index], 0.0, contact_force)
            encoder_noise = self._rng.normal(0.0, encoder_stds[index]) if self.encoder_noise_enabled else 0.0
            measured_q = q[index] + np.deg2rad(
                self._joint_param(self.encoder_position_bias_deg, name, 0.0) + encoder_noise
            )
            velocity_noise = (
                self._rng.normal(0.0, np.deg2rad(self.encoder_velocity_noise_deg))
                if self.encoder_noise_enabled
                else 0.0
            )
            measured_w = (
                link_velocity[index]
                + self._joint_param(self.encoder_velocity_bias_rad_s, name, 0.0)
                + velocity_noise
            )
            command_error = q_target[index] - measured_q
            command_errors[index] = command_error
            _, backlash_deflection = self._compliance_deflection(command_error, deadband)
            backlash_deflections[index] = backlash_deflection
            voltage = clip_scalar(
                self.supply_voltage * self.servo_stiffness * command_error - effective_damping * measured_w,
                -self.supply_voltage,
                self.supply_voltage,
            )
            current = clip_scalar(voltage / self.resistance, -current_limit, current_limit)
            motor_torque = self.kt * current
            voltages[index] = voltage
            currents[index] = current
            motor_torques[index] = motor_torque
            load_direction = float(
                np.sign(link_velocity[index] if abs(link_velocity[index]) >= 0.015 else motor_torque)
            )
            load_torque = contact_force * self._joint_param(self.contact_load_gain, name, 0.0) * load_direction
            load_torques[index] = load_torque
            motor_drag = self._joint_param(self.motor_viscous_friction, name, 0.0) * self.motor_velocity[name]
            drive = motor_torque - motor_drag - load_torque
            friction_torques[index] = self._friction_torque(name, drive, link_velocity[index])

        coriolis_torque, gravity_torque = self._coupled_loads(q, link_velocity)
        process_torque = np.asarray(
            [
                self._rng.normal(0.0, self._joint_param(self.process_torque_noise_std, name, 0.0))
                if self.process_noise_enabled
                else 0.0
                for name in JOINT_NAMES
            ],
            dtype=np.float64,
        )
        generalized_drive = motor_torques - load_torques - friction_torques - coriolis_torque - gravity_torque + process_torque
        link_acceleration = np.linalg.solve(self._mass_matrix(q), generalized_drive)
        link_velocity = np.clip(
            link_velocity + link_acceleration * dt,
            -self.max_link_velocity,
            self.max_link_velocity,
        )
        q = q + link_velocity * dt

        values: dict[str, float] = {}
        for index, name in enumerate(JOINT_NAMES):
            spring_drive = motor_torques[index] - load_torques[index] - friction_torques[index]
            spring_deflection = spring_drive / max(
                1e-5, self._joint_param(self.transmission_stiffness, name, 8.0)
            )
            spring_deflection += self._joint_param(self.transmission_damping, name, 0.35) * link_velocity[index] / max(
                1e-5, self._joint_param(self.transmission_stiffness, name, 8.0)
            )
            motor_q = float(q[index] + backlash_deflections[index] + spring_deflection)
            motor_w = clip_scalar(
                (motor_q - previous_motor_q[index]) / max(1e-6, dt),
                -self.max_motor_velocity,
                self.max_motor_velocity,
            )
            heat = (abs(currents[index]) / max(1e-6, self.current_limit)) ** 2 * dt / max(
                1e-6, self.thermal_time_constant
            )
            cool = temperatures[index] * dt / max(1e-6, self.cooling_time_constant)
            temperature = clip_scalar(temperatures[index] + heat - cool, 0.0, 1.0)
            encoder_stds[index] = self._encoder_std_deg(
                name,
                link_velocity[index],
                currents[index],
                contact_force,
            )
            encoder_noise = self._rng.normal(0.0, encoder_stds[index]) if self.encoder_noise_enabled else 0.0

            self.velocity[name] = float(link_velocity[index])
            self.motor_angle[name] = float(motor_q)
            self.motor_velocity[name] = float(motor_w)
            self.temperature[name] = temperature
            self.telemetry.voltage[name] = float(voltages[index])
            self.telemetry.current[name] = float(currents[index])
            self.telemetry.torque[name] = float(motor_torques[index])
            self.telemetry.actuator_angle_deg[name] = float(np.rad2deg(motor_q))
            self.telemetry.actuator_velocity_rad_s[name] = float(motor_w)
            encoder_q_after = q[index] + np.deg2rad(
                self._joint_param(self.encoder_position_bias_deg, name, 0.0) + encoder_noise
            )
            encoder_w_after = link_velocity[index] + self._joint_param(
                self.encoder_velocity_bias_rad_s, name, 0.0
            )
            self.telemetry.encoder_angle_deg[name] = float(np.rad2deg(encoder_q_after))
            self.telemetry.encoder_velocity_rad_s[name] = float(encoder_w_after)
            self.telemetry.position_error_deg[name] = float(np.rad2deg(command_errors[index]))
            self.telemetry.elastic_deflection_deg[name] = float(np.rad2deg(spring_deflection))
            self.telemetry.backlash_deflection_deg[name] = float(np.rad2deg(backlash_deflections[index]))
            self.telemetry.friction_torque[name] = float(friction_torques[index])
            self.telemetry.load_torque[name] = float(load_torques[index])
            self.telemetry.gravity_torque[name] = float(gravity_torque[index])
            self.telemetry.coupling_torque[name] = float(coriolis_torque[index])
            self.telemetry.process_torque[name] = float(process_torque[index])
            self.telemetry.encoder_std_deg[name] = float(encoder_stds[index])
            self.telemetry.thermal_fraction[name] = temperature
            self.telemetry.torque_limit_fraction[name] = float(
                current_limits[index] / max(1e-6, self.current_limit)
            )
            values[name] = float(np.rad2deg(q[index]))
        return ArmPose(**values).clipped()


@dataclass(slots=True)
class ContactState:
    on_canvas: bool
    deflection: float
    force: float
    pressure: float
    brush_width_px: float
    brush_world: np.ndarray


@dataclass(slots=True)
class Brush:
    """Per-stroke brush state: how heavily it is loaded (a constant fresh-paint
    deposition scale, since oil does not run out or dry mid-stroke), a small
    held reservoir of paint skimmed off the canvas (the "dirty brush" that
    drives wet blending: held volume plus its pigment mass, exactly conserved
    against the canvas ledger), and a fixed bristle furrow pattern. Reset at
    each pen-down from the stroke's ``amount``/``tone`` so that a whole stroke
    stays a deterministic function of the canvas and the action (the learned
    transition model never sees cross-stroke brush memory).
    """

    config: PainterConfig
    rng: np.random.Generator
    load: float = field(default=1.0)
    fresh_tone: float = field(default=0.0)
    held_volume: float = field(default=0.0)
    held_black: float = field(default=0.0)
    carried_tone: float = field(default=0.0)
    path_distance: float = field(default=0.0)
    bristle_offsets: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    bristle_gains: np.ndarray = field(default_factory=lambda: np.ones(0, dtype=np.float32))
    streak_phases: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    wobble_phases: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    wobble_amps: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def reload(self, amount: float, tone: float) -> None:
        cfg = self.config
        frac = clip_scalar(amount, 0.0, 1.0)
        self.load = cfg.brush_load_min + (cfg.brush_load_max - cfg.brush_load_min) * frac
        self.fresh_tone = float(tone >= 0.5)
        self.held_volume = 0.0
        self.held_black = 0.0
        self.carried_tone = self.fresh_tone
        self.path_distance = 0.0
        count = max(1, int(cfg.brush_bristle_count))
        offsets = np.linspace(-1.0, 1.0, count, dtype=np.float32) if count > 1 else np.zeros(1, np.float32)
        jitter = self.rng.uniform(-0.12, 0.12, size=count).astype(np.float32) if count > 1 else np.zeros(1, np.float32)
        depth = clip_scalar(cfg.brush_bristle_depth, 0.0, 1.0)
        gains = 1.0 - depth * self.rng.uniform(0.0, 1.0, size=count).astype(np.float32)
        offsets = np.clip(offsets + jitter, -1.0, 1.0)
        order = np.argsort(offsets)  # np.interp needs ascending sample points
        self.bristle_offsets = offsets[order]
        self.bristle_gains = gains[order]
        # Per-hair phase for intermittent dry gaps that open and close along the
        # path (hairs recharge from surrounding paint), and per-stroke low-order
        # harmonics that make the contact-patch boundary non-circular.
        self.streak_phases = self.rng.uniform(0.0, 1.0, size=count).astype(np.float32)
        self.wobble_phases = self.rng.uniform(0.0, 2.0 * np.pi, size=3).astype(np.float32)
        self.wobble_amps = self.rng.uniform(0.4, 1.0, size=3).astype(np.float32)

    def bristle_gains_at(self, path_distance: float) -> np.ndarray:
        """Per-hair gains with intermittent dry gaps at this point on the path."""
        cfg = self.config
        gap_fraction = clip_scalar(cfg.brush_bristle_gap_fraction, 0.0, 1.0)
        if gap_fraction <= 0.0 or self.streak_phases.size == 0:
            return self.bristle_gains
        wavelength = max(1e-3, float(cfg.brush_streak_length))
        cycle = (path_distance / wavelength + self.streak_phases) % 1.0
        dry = cycle < gap_fraction
        gap_gain = clip_scalar(cfg.brush_bristle_gap_gain, 0.0, 1.0)
        return np.where(dry, gap_gain * self.bristle_gains, self.bristle_gains).astype(np.float32)


@dataclass(slots=True)
class VerticalCanvas:
    config: PainterConfig
    width: float = 20.0
    height: float = 20.0
    distance: float = 17.0
    bushing_travel: float = 0.5
    contact_stiffness: float = 55.0
    thickness: np.ndarray = field(init=False)
    wetness: np.ndarray = field(init=False)
    black_mass: np.ndarray = field(init=False)
    surface_tone: np.ndarray = field(init=False)
    grain: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = self.config.canvas_size
        self.thickness = np.zeros((n, n), dtype=np.float32)
        self.wetness = np.zeros((n, n), dtype=np.float32)
        self.black_mass = np.zeros((n, n), dtype=np.float32)
        self.surface_tone = np.zeros((n, n), dtype=np.float32)
        if self.config.canvas_grain_strength > 0.0:
            self.grain = _make_canvas_grain(n, self.config.canvas_grain_period_px, self.config.canvas_grain_seed)
        else:
            self.grain = np.ones((n, n), dtype=np.float32)

    def clear(self) -> None:
        self.thickness.fill(0.0)
        self.wetness.fill(0.0)
        self.black_mass.fill(0.0)
        self.surface_tone.fill(0.0)

    def coverage_field(self) -> np.ndarray:
        """Binary material occupancy; additional layers do not add coverage."""

        return (self.thickness >= self.config.paint_presence_threshold).astype(np.float32)

    def surface_opacity_field(self) -> np.ndarray:
        """Optical opacity remains thickness-dependent and separate from coverage."""

        return 1.0 - np.exp(-self.thickness / self.config.thickness_scale)

    def visible_tone(self) -> np.ndarray:
        return np.clip(self.surface_tone, 0.0, 1.0)

    def observed_tone(self) -> np.ndarray:
        opacity = self.surface_opacity_field()
        return np.clip(
            (1.0 - opacity) * self.config.canvas_ground_tone + opacity * self.visible_tone(),
            0.0,
            1.0,
        )

    def ground_contrast_field(self) -> np.ndarray:
        return np.abs(self.observed_tone() - self.config.canvas_ground_tone).astype(np.float32)

    def material_coverage(self) -> float:
        return float(self.coverage_field().mean())

    def world_to_pixel(self, x: float, z: float) -> tuple[float, float]:
        u = (x / self.width + 0.5) * (self.config.canvas_size - 1)
        v = (0.5 - z / self.height) * (self.config.canvas_size - 1)
        return u, v

    def contains(self, x: float, z: float) -> bool:
        return abs(x) <= self.width / 2.0 and abs(z) <= self.height / 2.0

    def brush_radius_world(self, pressure: float) -> float:
        # Compact bristle contact patch in world units so physical mark width
        # is canvas-resolution independent; pressure splays the bristles. The
        # patch has hard support: no paint deposits beyond it no matter how
        # long the brush dwells.
        return 0.10 + 0.42 * clip_scalar(pressure, 0.0, 1.0)

    def _pixels_per_unit(self) -> float:
        return (self.config.canvas_size - 1) / max(1e-6, self.width)

    def contact_from_tip(self, tip: np.ndarray, intended_pressure: float = 0.0) -> ContactState:
        on_canvas = self.contains(float(tip[0]), float(tip[2]))
        raw = max(0.0, float(tip[1] - self.distance)) if on_canvas else 0.0
        deflection = min(raw, self.bushing_travel)
        force = self.contact_stiffness * deflection
        geometric_pressure = deflection / max(1e-5, self.bushing_travel)
        near_surface = on_canvas and float(tip[1]) >= self.distance - 0.08
        pressure = max(geometric_pressure, clip_scalar(intended_pressure, 0.0, 1.0) if near_surface else 0.0)
        force = max(force, pressure * self.contact_stiffness * self.bushing_travel)
        brush_width_px = 2.0 * self.brush_radius_world(pressure) * self._pixels_per_unit()
        brush_world = tip.copy()
        if on_canvas and (raw > 0.0 or pressure > 0.0):
            brush_world[1] = self.distance
        return ContactState(on_canvas, deflection, force, pressure, brush_width_px, brush_world)

    def too_deep(self, tip: np.ndarray) -> bool:
        return self.contains(float(tip[0]), float(tip[2])) and float(tip[1] - self.distance) > self.bushing_travel

    def overtravel_depth(self, tip: np.ndarray) -> float:
        if not self.contains(float(tip[0]), float(tip[2])):
            return 0.0
        return max(0.0, float(tip[1] - self.distance) - self.bushing_travel)

    def paint_at(
        self,
        brush_world: np.ndarray,
        pressure: float,
        tone: float,
        dt: float,
        *,
        motion: np.ndarray | None = None,
        brush: Brush | None = None,
        flow: float = 1.0,
    ) -> float:
        """Deposit a stamp and return the peak thickness laid.

        With ``motion`` and ``brush`` omitted this is the legacy isotropic disc
        with a unit deposition scale and binary tone (relied on by direct-call
        tests). ``motion`` (world-space brush displacement since the previous
        deposition) sweeps the disc into a capsule so travel elongates the mark;
        ``brush`` adds loading (a constant deposition scale, never depleting),
        bristle furrows, canvas-grain texture, and wet-drag smear. ``flow`` in
        [0, 1] tapers the brush width (stroke-end envelope). Paint never dries
        or runs out.
        """

        if pressure <= 0.001 or not self.contains(float(brush_world[0]), float(brush_world[2])):
            return 0.0
        n = self.config.canvas_size
        u, v = self.world_to_pixel(float(brush_world[0]), float(brush_world[2]))
        ppu = self._pixels_per_unit()
        radius = max(0.9, self.brush_radius_world(pressure) * ppu)
        # Taper the width toward the stroke ends (flow < 1) so marks come to a
        # point instead of a round cap. Only narrows; never widens.
        if brush is not None and flow < 1.0:
            taper_min = clip_scalar(self.config.brush_taper_min_width, 0.0, 1.0)
            radius = max(0.9, radius * (taper_min + (1.0 - taper_min) * clip_scalar(flow, 0.0, 1.0)))
        edge = max(0.7, 0.18 * radius)
        # Trailing endpoint of the swept segment (previous contact point). With
        # no motion this collapses onto (u, v) and the capsule is a disc, so the
        # legacy distance field is reproduced bit-for-bit.
        if motion is not None and self.config.brush_directional_enabled:
            mu = float(motion[0]) * ppu
            mv = -float(motion[2]) * ppu
        else:
            mu = mv = 0.0
        u0, v0 = u - mu, v - mv
        seg_len = float(np.hypot(mu, mv))
        # The brush has hard support, so deposit only inside the swept bounding box.
        extent = int(np.ceil(radius + edge)) + 1
        col0 = max(0, int(np.floor(min(u, u0))) - extent)
        col1 = min(n, int(np.ceil(max(u, u0))) + extent + 1)
        row0 = max(0, int(np.floor(min(v, v0))) - extent)
        row1 = min(n, int(np.ceil(max(v, v0))) + extent + 1)
        if col0 >= col1 or row0 >= row1:
            return 0.0
        yy, xx = np.mgrid[row0:row1, col0:col1]
        if seg_len > 1e-6:
            # Distance from each pixel to the swept segment [p0, p1].
            tx, ty = mu / seg_len, mv / seg_len
            proj = np.clip(((xx - u0) * tx + (yy - v0) * ty), 0.0, seg_len)
            cx, cy = u0 + proj * tx, v0 + proj * ty
            distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        else:
            distance = np.sqrt((xx - u) ** 2 + (yy - v) ** 2)
        # A real contact patch is never a perfect circle: perturb the boundary
        # radius with fixed per-stroke low-order angular harmonics, so even a
        # stationary dab comes out slightly lumpy.
        if brush is not None and self.config.brush_edge_wobble > 0.0:
            theta = np.arctan2(yy - v, xx - u)
            # Harmonic amps are normalised to the largest, so the leading
            # harmonic swings the boundary by the full configured wobble.
            amp_scale = clip_scalar(self.config.brush_edge_wobble, 0.0, 0.5) / max(
                1e-6, float(brush.wobble_amps.max())
            )
            wobble = 1.0 + amp_scale * (
                brush.wobble_amps[0] * np.sin(2.0 * theta + brush.wobble_phases[0])
                + brush.wobble_amps[1] * np.sin(3.0 * theta + brush.wobble_phases[1])
                + brush.wobble_amps[2] * np.sin(5.0 * theta + brush.wobble_phases[2])
            ) / 3.0
            distance = distance / np.maximum(wobble, 0.5)
        ramp = np.clip((radius + edge - distance) / edge, 0.0, 1.0)
        footprint = (ramp * ramp * (3.0 - 2.0 * ramp)).astype(np.float32)

        # Track along-path distance (world units) for intermittent dry streaks.
        if brush is not None:
            brush.path_distance += seg_len / max(1e-6, ppu)

        # Bristle furrows: gentle striation across the cross-stroke axis, with
        # dry gaps that open and close along the path (hairs recharge), so a
        # furrow textures the mark without splitting it end to end.
        if brush is not None and seg_len > 1e-6 and brush.bristle_offsets.size > 1:
            perp = ((xx - u) * (-ty) + (yy - v) * tx) / max(1e-6, radius)
            gains = np.interp(
                np.clip(perp, -1.0, 1.0).ravel(),
                brush.bristle_offsets,
                brush.bristle_gains_at(brush.path_distance),
            ).reshape(footprint.shape)
            footprint = footprint * gains.astype(np.float32)

        # Canvas tooth/grain: a light brush lays paint only on the raised tooth
        # and leaves unreached valleys genuinely bare (dry-brushing); pressing
        # harder lowers the waterline until reach >= 1 fills everything. Static
        # per canvas, so bare valleys stay bare through repeated light passes ->
        # texture survives opacity build-up.
        if brush is not None and self.config.canvas_grain_strength > 0.0:
            tooth = self.grain[row0:row1, col0:col1]
            reach = clip_scalar(
                self.config.canvas_grain_reach_base + self.config.canvas_grain_reach_pressure * pressure,
                0.0,
                1.0,
            )
            strength = clip_scalar(self.config.canvas_grain_strength, 0.0, 1.0)
            # Fill: 0 for tooth one shoulder below the (1-reach) waterline, 1 at
            # the waterline and above; reach = 1 puts every cell at full fill.
            shoulder = 0.18
            fill = np.clip((tooth - (1.0 - reach)) / shoulder + 1.0, 0.0, 1.0)
            fill = fill * fill * (3.0 - 2.0 * fill)
            footprint = footprint * (1.0 - strength * (1.0 - fill)).astype(np.float32)

        deposition_rate = (
            float(self.config.paint_deposition_base_rate)
            + float(self.config.paint_deposition_pressure_rate) * pressure
        )
        # Brush loading scales deposited thickness uniformly; it never depletes.
        load_scale = brush.load if brush is not None else 1.0
        peak = float(dt) * max(0.0, deposition_rate) * load_scale
        region = np.s_[row0:row1, col0:col1]
        previous_thickness = self.thickness[region]
        previous_tone = self.surface_tone[region]
        previous_wetness = self.wetness[region]

        # --- Pickup: skim wet surface paint into the brush head (dirty brush).
        # Mass and pigment move from the canvas ledger to the held reservoir,
        # exactly conserved; a floor keeps covered pixels above the presence
        # threshold so material coverage never regresses.
        if brush is not None and self.config.brush_pickup_fraction > 0.0:
            skimmable = np.minimum(
                np.maximum(previous_thickness - 2.0 * float(self.config.paint_presence_threshold), 0.0),
                float(self.config.brush_pickup_depth),
            )
            picked = float(self.config.brush_pickup_fraction) * clip_scalar(pressure, 0.0, 1.0) * footprint * skimmable
            capacity = float(self.config.brush_capacity_thickness) * max(1e-8, float(footprint.sum()))
            picked_total = float(picked.sum())
            allowed = max(0.0, capacity - brush.held_volume)
            if picked_total > allowed:
                picked *= allowed / max(1e-12, picked_total)
                picked_total = allowed
            if picked_total > 1e-12:
                bulk_black_fraction = self.black_mass[region] / np.maximum(previous_thickness, 1e-12)
                picked_black = picked * np.clip(bulk_black_fraction, 0.0, 1.0)
                remaining = previous_thickness - picked
                scale = remaining / np.maximum(previous_thickness, 1e-12)
                self.thickness[region] = remaining
                self.black_mass[region] = np.maximum(self.black_mass[region] - picked_black, 0.0)
                self.wetness[region] = previous_wetness * np.clip(scale, 0.0, 1.0)
                brush.held_volume += picked_total
                brush.held_black += float(picked_black.sum())
                previous_thickness = self.thickness[region]
                previous_wetness = self.wetness[region]

        # --- Deposit: fresh paint from the loading plus a released share of the
        # held (dirty) paint. The dirty share is biased toward the leading edge
        # of the swept footprint, so picked-up paint is pushed ahead of the
        # stroke -- the cheap core of wet-into-wet blending (ArtRage/Krita).
        fresh = peak * footprint
        dirty = np.zeros_like(fresh)
        held_tone = brush.fresh_tone if brush is not None else 0.0
        if brush is not None and brush.held_volume > 1e-12 and self.config.brush_release_fraction > 0.0:
            release_total = float(self.config.brush_release_fraction) * brush.held_volume
            if seg_len > 1e-6:
                lead = np.clip(proj / seg_len, 0.0, 1.0)
                push = clip_scalar(self.config.brush_push_forward, 0.0, 1.0)
                release_weight = footprint * (1.0 - push + push * lead)
            else:
                release_weight = footprint
            weight_total = float(release_weight.sum())
            if weight_total > 1e-12:
                held_tone = brush.held_black / max(1e-12, brush.held_volume)
                dirty = release_total * release_weight / weight_total
                brush.held_volume = max(0.0, brush.held_volume - release_total)
                brush.held_black = max(0.0, brush.held_black - release_total * held_tone)

        deposited = fresh + dirty
        if brush is not None:
            incoming_tone = np.where(
                deposited > 1e-12,
                (fresh * brush.fresh_tone + dirty * held_tone) / np.maximum(deposited, 1e-12),
                brush.fresh_tone,
            )
            deposited_total = float(deposited.sum())
            if deposited_total > 1e-12:
                # Diagnostic: the mass-weighted tone the brush is currently laying.
                brush.carried_tone = clip_scalar(
                    float((deposited * incoming_tone).sum() / deposited_total), 0.0, 1.0
                )
        else:
            incoming_tone = float(tone >= 0.5)
        surface_alpha = 1.0 - np.exp(
            -deposited / max(1e-8, float(self.config.oil_surface_opacity_thickness))
        )
        wet_pickup = np.clip(
            float(self.config.oil_wet_pickup_fraction)
            * previous_wetness
            / np.maximum(previous_wetness + deposited, 1e-6),
            0.0,
            0.75,
        )
        loaded_tone = (1.0 - wet_pickup) * incoming_tone + wet_pickup * previous_tone
        new_tone = (1.0 - surface_alpha) * previous_tone + surface_alpha * loaded_tone
        self.thickness[region] = previous_thickness + deposited
        self.wetness[region] = previous_wetness + 0.8 * deposited
        self.black_mass[region] += deposited * incoming_tone
        self.surface_tone[region] = np.clip(new_tone, 0.0, 1.0)
        return peak


@dataclass(slots=True)
class ArmPainterSim:
    config: PainterConfig = field(default_factory=PainterConfig)
    kinematics: ArmKinematics = field(default_factory=ArmKinematics)
    plant: JointPlant = field(default_factory=JointPlant)
    canvas: VerticalCanvas = field(init=False)
    actual_pose: ArmPose = field(init=False)
    target_pose: ArmPose = field(init=False)
    paint_enabled: bool = field(init=False)
    brush_tone: float = field(init=False)
    intended_contact_pressure: float = field(init=False)
    intended_paint_load: float = field(init=False)
    brush_flow: float = field(init=False)
    control_damping_multiplier: float = field(init=False)
    contact: ContactState = field(init=False)
    brush: Brush = field(init=False)
    _painting: bool = field(default=False, init=False)
    _previous_brush_world: np.ndarray | None = field(default=None, init=False)
    _tip_lag_world: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.canvas = VerticalCanvas(self.config)
        self.actual_pose = safe_home_pose()
        self.target_pose = safe_home_pose()
        self.plant.reset_state(self.actual_pose)
        self.paint_enabled = False
        self.brush_tone = 1.0
        self.intended_contact_pressure = 0.0
        self.intended_paint_load = 1.0
        self.brush_flow = 1.0
        self.control_damping_multiplier = 1.0
        self.brush = Brush(self.config, np.random.default_rng(int(self.config.brush_seed)))
        self._painting = False
        self._previous_brush_world = None
        self.contact = self.canvas.contact_from_tip(self.kinematics.tip(self.actual_pose), self.intended_contact_pressure)

    def reset_pose(self) -> None:
        self.actual_pose = safe_home_pose()
        self.target_pose = safe_home_pose()
        self.plant.reset_state(self.actual_pose)
        self.paint_enabled = False
        self.intended_contact_pressure = 0.0
        self.control_damping_multiplier = 1.0
        self._painting = False
        self._previous_brush_world = None
        self._tip_lag_world = None

    def set_target(self, pose: ArmPose) -> None:
        self.target_pose = pose.clipped()

    def step(self, dt: float) -> None:
        previous_pose = self.actual_pose
        previous_plant_state = self.plant.state_snapshot()
        target_pose = self.target_pose
        self.actual_pose = self.plant.step(
            self.actual_pose,
            self.target_pose,
            dt,
            contact_force=self.contact.force,
            damping_multiplier=self.control_damping_multiplier,
        )
        tip = self.kinematics.tip(self.actual_pose)
        previous_tip = self.kinematics.tip(previous_pose)
        if self.canvas.too_deep(tip) and self.canvas.overtravel_depth(tip) >= self.canvas.overtravel_depth(previous_tip):
            self.actual_pose = previous_pose
            self.plant.restore_state(previous_plant_state)
            if self.canvas.too_deep(self.kinematics.tip(target_pose)):
                self.target_pose = self.actual_pose
            tip = self.kinematics.tip(self.actual_pose)
        self.contact = self.canvas.contact_from_tip(tip, self.intended_contact_pressure)
        painting = self.paint_enabled and self.contact.pressure > 0.001 and self.contact.on_canvas
        if painting:
            if not self._painting:
                # Pen-down: reload the brush from the stroke's declared load and
                # tone. Per-stroke reset keeps each stroke a function of the
                # canvas and action, with no cross-stroke brush memory.
                self.brush.reload(self.intended_paint_load, self.brush_tone)
                self._previous_brush_world = None
                self._tip_lag_world = None
            contact_world = self.contact.brush_world
            # Bristle-tip trailer dynamics: the painting point is a damped
            # follower of the contact point, so it lags starts and cuts corners
            # like a pulled brush tip. Reset each pen-down.
            tau = float(self.config.brush_tip_lag_seconds)
            if tau > 1e-6:
                if self._tip_lag_world is None:
                    self._tip_lag_world = contact_world.copy()
                follow = 1.0 - float(np.exp(-dt / tau))
                self._tip_lag_world = self._tip_lag_world + follow * (contact_world - self._tip_lag_world)
                brush_world = self._tip_lag_world
            else:
                brush_world = contact_world
            if self._previous_brush_world is not None:
                motion = brush_world - self._previous_brush_world
            else:
                motion = None
            self.canvas.paint_at(
                brush_world,
                self.contact.pressure,
                self.brush_tone,
                dt,
                motion=motion,
                brush=self.brush,
                flow=self.brush_flow,
            )
            self._previous_brush_world = brush_world.copy()
        self._painting = painting

    def render_points(self) -> np.ndarray:
        points = self.kinematics.joint_points(self.actual_pose).copy()
        if self.contact.on_canvas and self.contact.pressure > 0.0:
            points[-1] = self.contact.brush_world
        return points
