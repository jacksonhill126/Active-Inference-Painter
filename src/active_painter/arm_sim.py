from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import PainterConfig


JOINT_NAMES = ("yaw", "pitch", "roll", "elbow")


def safe_home_pose() -> "ArmPose":
    return ArmPose(yaw=0.0, pitch=-50.0, roll=0.0, elbow=100.0)


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
            yaw=float(np.clip(self.yaw, -90.0, 90.0)),
            pitch=float(np.clip(self.pitch, -90.0, 90.0)),
            roll=float(np.clip(self.roll, -180.0, 180.0)),
            elbow=float(np.clip(self.elbow, 0.0, 150.0)),
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
    encoder_std_deg: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    thermal_fraction: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    torque_limit_fraction: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))


@dataclass(slots=True)
class JointPlant:
    """Deterministic actuator/link approximation beneath painting inference.

    The values are representative small-arm parameters, not vendor-specific
    motor measurements. The plant exposes prediction-error-relevant mechanics
    to policy forecasts while safety limits stay external to painting choice.
    """

    supply_voltage: float = 24.0
    current_limit: float = 7.0
    servo_stiffness: float = 1.0
    damping: float = 0.48
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
    velocity: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    motor_angle: dict[str, float] = field(default_factory=dict)
    motor_velocity: dict[str, float] = field(default_factory=dict)
    temperature: dict[str, float] = field(default_factory=dict)
    telemetry: MotorTelemetry = field(default_factory=MotorTelemetry)

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

    def state_snapshot(self) -> dict[str, object]:
        return {
            "velocity": dict(self.velocity),
            "motor_angle": dict(self.motor_angle),
            "motor_velocity": dict(self.motor_velocity),
            "temperature": dict(self.temperature),
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

    def step(self, actual: ArmPose, target: ArmPose, dt: float, contact_force: float = 0.0) -> ArmPose:
        values: dict[str, float] = {}
        actual = actual.clipped()
        target = target.clipped()
        self._ensure_state(actual)
        contact_force = max(0.0, float(contact_force))
        for name in JOINT_NAMES:
            q = np.deg2rad(getattr(actual, name))
            q_target = np.deg2rad(getattr(target, name))
            link_w = float(self.velocity[name])
            previous_motor_q = float(self.motor_angle[name])
            temperature = float(np.clip(self.temperature[name], 0.0, 1.0))
            current_limit = self.current_limit * max(0.25, 1.0 - self.thermal_current_derate * temperature)
            deadband = np.deg2rad(self._joint_param(self.backlash_deadband_deg, name, 0.2))
            measured_q = q + np.deg2rad(self._joint_param(self.encoder_position_bias_deg, name, 0.0))
            measured_w = link_w + self._joint_param(self.encoder_velocity_bias_rad_s, name, 0.0)
            command_error = q_target - measured_q
            _, backlash_deflection = self._compliance_deflection(command_error, deadband)
            voltage = np.clip(
                self.supply_voltage * self.servo_stiffness * command_error - self.damping * measured_w,
                -self.supply_voltage,
                self.supply_voltage,
            )
            current = np.clip(voltage / self.resistance, -current_limit, current_limit)
            motor_torque = self.kt * current

            load_direction = float(np.sign(link_w if abs(link_w) >= 0.015 else motor_torque))
            load_torque = contact_force * self._joint_param(self.contact_load_gain, name, 0.0) * load_direction
            drive = motor_torque - load_torque
            friction_torque = self._friction_torque(name, drive, link_w)
            link_accel = (drive - friction_torque) / max(1e-5, self._joint_param(self.link_inertia, name, self.inertia))
            link_w = float(np.clip(link_w + link_accel * dt, -self.max_link_velocity, self.max_link_velocity))
            q = q + link_w * dt
            spring_deflection = (drive - friction_torque) / max(
                1e-5, self._joint_param(self.transmission_stiffness, name, 8.0)
            )
            spring_deflection += self._joint_param(self.transmission_damping, name, 0.35) * link_w / max(
                1e-5, self._joint_param(self.transmission_stiffness, name, 8.0)
            )
            motor_q = float(q + backlash_deflection + spring_deflection)
            motor_w = float(
                np.clip(
                    (motor_q - previous_motor_q) / max(1e-6, dt),
                    -self.max_motor_velocity,
                    self.max_motor_velocity,
                )
            )

            heat = (abs(current) / max(1e-6, self.current_limit)) ** 2 * dt / max(1e-6, self.thermal_time_constant)
            cool = temperature * dt / max(1e-6, self.cooling_time_constant)
            temperature = float(np.clip(temperature + heat - cool, 0.0, 1.0))

            self.velocity[name] = float(link_w)
            self.motor_angle[name] = float(motor_q)
            self.motor_velocity[name] = float(motor_w)
            self.temperature[name] = temperature
            self.telemetry.voltage[name] = float(voltage)
            self.telemetry.current[name] = float(current)
            self.telemetry.torque[name] = float(motor_torque)
            self.telemetry.actuator_angle_deg[name] = float(np.rad2deg(motor_q))
            self.telemetry.actuator_velocity_rad_s[name] = float(motor_w)
            encoder_q_after = q + np.deg2rad(self._joint_param(self.encoder_position_bias_deg, name, 0.0))
            encoder_w_after = link_w + self._joint_param(self.encoder_velocity_bias_rad_s, name, 0.0)
            self.telemetry.encoder_angle_deg[name] = float(np.rad2deg(encoder_q_after))
            self.telemetry.encoder_velocity_rad_s[name] = float(encoder_w_after)
            self.telemetry.position_error_deg[name] = float(np.rad2deg(command_error))
            self.telemetry.elastic_deflection_deg[name] = float(np.rad2deg(spring_deflection))
            self.telemetry.backlash_deflection_deg[name] = float(np.rad2deg(backlash_deflection))
            self.telemetry.friction_torque[name] = float(friction_torque)
            self.telemetry.load_torque[name] = float(load_torque)
            self.telemetry.encoder_std_deg[name] = float(
                self.encoder_base_noise_deg
                + self.encoder_velocity_noise_deg * abs(link_w)
                + self.encoder_current_noise_deg * abs(current) / max(1e-6, self.current_limit)
                + self.encoder_contact_noise_deg * contact_force
            )
            self.telemetry.thermal_fraction[name] = temperature
            self.telemetry.torque_limit_fraction[name] = float(current_limit / max(1e-6, self.current_limit))
            values[name] = float(np.rad2deg(q))
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

    def __post_init__(self) -> None:
        n = self.config.canvas_size
        self.thickness = np.zeros((n, n), dtype=np.float32)
        self.wetness = np.zeros((n, n), dtype=np.float32)
        self.black_mass = np.zeros((n, n), dtype=np.float32)

    def clear(self) -> None:
        self.thickness.fill(0.0)
        self.wetness.fill(0.0)
        self.black_mass.fill(0.0)

    def coverage_field(self) -> np.ndarray:
        return 1.0 - np.exp(-self.thickness / self.config.thickness_scale)

    def visible_tone(self) -> np.ndarray:
        denom = np.maximum(self.thickness, 1e-6)
        return np.clip(self.black_mass / denom, 0.0, 1.0)

    def observed_tone(self) -> np.ndarray:
        coverage = self.coverage_field()
        return np.clip(
            (1.0 - coverage) * self.config.canvas_ground_tone + coverage * self.visible_tone(),
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
        return 0.10 + 0.42 * float(np.clip(pressure, 0.0, 1.0))

    def _pixels_per_unit(self) -> float:
        return (self.config.canvas_size - 1) / max(1e-6, self.width)

    def contact_from_tip(self, tip: np.ndarray, intended_pressure: float = 0.0) -> ContactState:
        on_canvas = self.contains(float(tip[0]), float(tip[2]))
        raw = max(0.0, float(tip[1] - self.distance)) if on_canvas else 0.0
        deflection = min(raw, self.bushing_travel)
        force = self.contact_stiffness * deflection
        geometric_pressure = deflection / max(1e-5, self.bushing_travel)
        near_surface = on_canvas and float(tip[1]) >= self.distance - 0.08
        pressure = max(geometric_pressure, float(np.clip(intended_pressure, 0.0, 1.0)) if near_surface else 0.0)
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

    def paint_at(self, brush_world: np.ndarray, pressure: float, tone: float, dt: float) -> None:
        if pressure <= 0.001 or not self.contains(float(brush_world[0]), float(brush_world[2])):
            return
        n = self.config.canvas_size
        u, v = self.world_to_pixel(float(brush_world[0]), float(brush_world[2]))
        radius = max(0.9, self.brush_radius_world(pressure) * self._pixels_per_unit())
        edge = max(0.7, 0.18 * radius)
        # The brush has hard support, so deposit only inside its bounding box.
        extent = int(np.ceil(radius + edge)) + 1
        col0 = max(0, int(np.floor(u)) - extent)
        col1 = min(n, int(np.ceil(u)) + extent + 1)
        row0 = max(0, int(np.floor(v)) - extent)
        row1 = min(n, int(np.ceil(v)) + extent + 1)
        if col0 >= col1 or row0 >= row1:
            return
        yy, xx = np.mgrid[row0:row1, col0:col1]
        distance = np.sqrt((xx - u) ** 2 + (yy - v) ** 2)
        ramp = np.clip((radius + edge - distance) / edge, 0.0, 1.0)
        footprint = (ramp * ramp * (3.0 - 2.0 * ramp)).astype(np.float32)
        deposited = float(dt) * (0.055 + 0.22 * pressure) * footprint
        self.wetness *= self.config.wetness_decay
        self.thickness[row0:row1, col0:col1] += deposited
        self.wetness[row0:row1, col0:col1] += 0.8 * deposited
        self.black_mass[row0:row1, col0:col1] += deposited * float(tone >= 0.5)


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
    contact: ContactState = field(init=False)

    def __post_init__(self) -> None:
        self.canvas = VerticalCanvas(self.config)
        self.actual_pose = safe_home_pose()
        self.target_pose = safe_home_pose()
        self.plant.reset_state(self.actual_pose)
        self.paint_enabled = False
        self.brush_tone = 1.0
        self.intended_contact_pressure = 0.0
        self.contact = self.canvas.contact_from_tip(self.kinematics.tip(self.actual_pose), self.intended_contact_pressure)

    def reset_pose(self) -> None:
        self.actual_pose = safe_home_pose()
        self.target_pose = safe_home_pose()
        self.plant.reset_state(self.actual_pose)
        self.paint_enabled = False
        self.intended_contact_pressure = 0.0

    def set_target(self, pose: ArmPose) -> None:
        self.target_pose = pose.clipped()

    def step(self, dt: float) -> None:
        previous_pose = self.actual_pose
        previous_plant_state = self.plant.state_snapshot()
        target_pose = self.target_pose
        self.actual_pose = self.plant.step(self.actual_pose, self.target_pose, dt, contact_force=self.contact.force)
        tip = self.kinematics.tip(self.actual_pose)
        previous_tip = self.kinematics.tip(previous_pose)
        if self.canvas.too_deep(tip) and self.canvas.overtravel_depth(tip) >= self.canvas.overtravel_depth(previous_tip):
            self.actual_pose = previous_pose
            self.plant.restore_state(previous_plant_state)
            if self.canvas.too_deep(self.kinematics.tip(target_pose)):
                self.target_pose = self.actual_pose
            tip = self.kinematics.tip(self.actual_pose)
        self.contact = self.canvas.contact_from_tip(tip, self.intended_contact_pressure)
        if self.paint_enabled:
            self.canvas.paint_at(self.contact.brush_world, self.contact.pressure, self.brush_tone, dt)

    def render_points(self) -> np.ndarray:
        points = self.kinematics.joint_points(self.actual_pose).copy()
        if self.contact.on_canvas and self.contact.pressure > 0.0:
            points[-1] = self.contact.brush_world
        return points
