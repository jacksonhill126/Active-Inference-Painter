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


@dataclass(slots=True)
class JointPlant:
    supply_voltage: float = 24.0
    current_limit: float = 7.0
    servo_stiffness: float = 0.65
    damping: float = 0.9
    inertia: float = 0.065
    kt: float = 0.42
    resistance: float = 2.1
    velocity: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINT_NAMES, 0.0))
    telemetry: MotorTelemetry = field(default_factory=MotorTelemetry)

    def step(self, actual: ArmPose, target: ArmPose, dt: float) -> ArmPose:
        values: dict[str, float] = {}
        actual = actual.clipped()
        target = target.clipped()
        for name in JOINT_NAMES:
            q = np.deg2rad(getattr(actual, name))
            q_target = np.deg2rad(getattr(target, name))
            err = q_target - q
            w = self.velocity[name]
            voltage = np.clip(
                self.supply_voltage * self.servo_stiffness * err - self.damping * w,
                -self.supply_voltage,
                self.supply_voltage,
            )
            current = np.clip(voltage / self.resistance, -self.current_limit, self.current_limit)
            torque = self.kt * current
            accel = torque / max(1e-5, self.inertia)
            w = np.clip(w + accel * dt, -5.0, 5.0)
            q = q + w * dt
            self.velocity[name] = float(w)
            self.telemetry.voltage[name] = float(voltage)
            self.telemetry.current[name] = float(current)
            self.telemetry.torque[name] = float(torque)
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
        self.paint_enabled = False
        self.brush_tone = 1.0
        self.intended_contact_pressure = 0.0
        self.contact = self.canvas.contact_from_tip(self.kinematics.tip(self.actual_pose), self.intended_contact_pressure)

    def reset_pose(self) -> None:
        self.actual_pose = safe_home_pose()
        self.target_pose = safe_home_pose()
        self.plant.velocity = dict.fromkeys(JOINT_NAMES, 0.0)
        self.paint_enabled = False
        self.intended_contact_pressure = 0.0

    def set_target(self, pose: ArmPose) -> None:
        self.target_pose = pose.clipped()

    def step(self, dt: float) -> None:
        previous_pose = self.actual_pose
        previous_velocity = dict(self.plant.velocity)
        self.actual_pose = self.plant.step(self.actual_pose, self.target_pose, dt)
        tip = self.kinematics.tip(self.actual_pose)
        if self.canvas.too_deep(tip):
            self.actual_pose = previous_pose
            self.target_pose = self.actual_pose
            self.plant.velocity = {name: 0.0 for name in previous_velocity}
            tip = self.kinematics.tip(self.actual_pose)
        self.contact = self.canvas.contact_from_tip(tip, self.intended_contact_pressure)
        if self.paint_enabled:
            self.canvas.paint_at(self.contact.brush_world, self.contact.pressure, self.brush_tone, dt)

    def render_points(self) -> np.ndarray:
        points = self.kinematics.joint_points(self.actual_pose).copy()
        if self.contact.on_canvas and self.contact.pressure > 0.0:
            points[-1] = self.contact.brush_world
        return points
