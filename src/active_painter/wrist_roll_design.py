from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from .camera_geometry import ROBOT_MODEL_PATH


WRIST_ROLL_EXPLORATION_VERSION = "mujoco-robstride-angled-wrist-roll-exploration-v0"
WRIST_ROLL_EXPLORATION_STATUS = "noncanonical_design_exploration_not_hardware_selected"


@dataclass(frozen=True, slots=True)
class AngledWristRollSpec:
    """Declared geometry for the non-canonical four-axis design branch."""

    brush_angle_deg: float = 15.0
    wrist_axis_from_elbow_m: float = 0.2472
    roll_motor_mass_kg: float = 0.405
    roll_rotor_mass_fraction: float = 0.20

    def __post_init__(self) -> None:
        if not 0.0 < self.brush_angle_deg < 45.0:
            raise ValueError("brush_angle_deg must be between 0 and 45 degrees")
        if not 0.20 < self.wrist_axis_from_elbow_m < 0.30:
            raise ValueError("wrist axis must remain on the distal forearm")
        if self.roll_motor_mass_kg <= 0.0:
            raise ValueError("roll motor mass must be positive")
        if not 0.0 < self.roll_rotor_mass_fraction < 1.0:
            raise ValueError("roll_rotor_mass_fraction must lie inside (0, 1)")


@dataclass(frozen=True, slots=True)
class WristRollComparison:
    version: str
    status: str
    spec: dict[str, float]
    roll_sweep_deg: tuple[float, float]
    canonical_tip_path_length_mm: float
    wrist_tip_path_length_mm: float
    wrist_tip_chord_mm: float
    wrist_tip_orbit_radius_mm: float
    wrist_brush_direction_change_deg: float
    canonical_canvas_reachability: float
    wrist_canvas_reachability: float
    canonical_median_tip_residual_mm: float
    wrist_median_tip_residual_mm: float
    canonical_elbow_gravity_bias_nm: float
    wrist_elbow_gravity_bias_nm: float
    elbow_gravity_bias_increase_nm: float
    approximation: str


def build_wrist_roll_exploration_xml(
    spec: AngledWristRollSpec = AngledWristRollSpec(),
    *,
    canonical_path: Path | str = ROBOT_MODEL_PATH,
) -> str:
    """Branch the canonical MJCF in memory without changing its hardware record.

    The shoulder roll joint is removed.  The same named ``roll`` actuator is
    relocated downstream of the elbow, and the existing brush assembly is
    mounted at a fixed angle to its output.  Names used by sensors and
    actuators stay stable, but this model is deliberately not accepted as a
    runtime plant.

    Aggregate link inertias in the source MJCF do not identify stator/rotor and
    brush-handle contributions separately.  The transformation therefore
    relocates the vendor-listed RS02 mass using an explicit 80/20 stator/rotor
    split and records that approximation in custom metadata.
    """

    root = ElementTree.parse(Path(canonical_path)).getroot()
    root.set("model", "active_inference_painter_angled_wrist_roll_exploration")

    custom = _required(root.find("custom"), "custom")
    _set_custom_text(custom, "model_version", WRIST_ROLL_EXPLORATION_VERSION)
    _set_custom_text(
        custom,
        "kinematic_convention",
        "Rz_yaw_Rx_pitch_Rx_elbow_Ry_wrist_roll_Rx_fixed_brush_angle",
    )
    _set_custom_text(custom, "design_status", WRIST_ROLL_EXPLORATION_STATUS)
    _set_custom_text(custom, "roll_joint_location", "distal_wrist_after_elbow")
    _set_custom_text(
        custom,
        "wrist_mass_model",
        "vendor_total_RS02_mass_relocated_with_provisional_stator_rotor_split",
    )
    _set_custom_numeric(
        custom,
        "wrist_roll_brush_angle_rad",
        (math.radians(spec.brush_angle_deg),),
    )
    _set_custom_numeric(
        custom,
        "wrist_axis_from_elbow_m",
        (spec.wrist_axis_from_elbow_m,),
    )
    _set_custom_numeric(
        custom,
        "wrist_roll_motor_mass_split_kg",
        (
            spec.roll_motor_mass_kg * (1.0 - spec.roll_rotor_mass_fraction),
            spec.roll_motor_mass_kg * spec.roll_rotor_mass_fraction,
        ),
    )

    pitch_output = _named(root, "body", "pitch_output")
    upper_arm = _named(root, "body", "roll_output")
    elbow_output = _named(root, "body", "elbow_output")

    shoulder_roll_joint = _direct_named(upper_arm, "joint", "roll")
    upper_arm.remove(shoulder_roll_joint)
    upper_arm.set("name", "upper_arm_structure")
    old_roll_flange = _direct_named(upper_arm, "geom", "roll_output_flange")
    upper_arm.remove(old_roll_flange)

    old_roll_stator = _direct_named(pitch_output, "geom", "roll_rs02_stator")
    pitch_output.remove(old_roll_stator)
    old_roll_stator.set("name", "wrist_roll_rs02_stator")
    half_length = 0.0455 / 2.0
    old_roll_stator.set(
        "fromto",
        _format_numbers(
            (
                0.0,
                spec.wrist_axis_from_elbow_m - half_length,
                0.0,
                0.0,
                spec.wrist_axis_from_elbow_m + half_length,
                0.0,
            )
        ),
    )
    elbow_output.append(old_roll_stator)

    _relocate_roll_motor_mass(pitch_output, elbow_output, spec)

    wrist = ElementTree.Element(
        "body",
        {
            "name": "wrist_roll_output",
            "pos": _format_numbers((0.0, spec.wrist_axis_from_elbow_m, 0.0)),
        },
    )
    rotor_mass = spec.roll_motor_mass_kg * spec.roll_rotor_mass_fraction
    ElementTree.SubElement(
        wrist,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": f"{rotor_mass:.12g}",
            "diaginertia": "0.00008 0.00004 0.00008",
        },
    )
    wrist.append(shoulder_roll_joint)
    ElementTree.SubElement(
        wrist,
        "geom",
        {
            "name": "wrist_roll_output_flange",
            "type": "cylinder",
            "fromto": "0 -0.010 0 0 0.010 0",
            "size": "0.030",
            "material": "output_mat",
        },
    )
    ElementTree.SubElement(
        wrist,
        "site",
        {
            "name": "wrist_roll_axis",
            "type": "sphere",
            "pos": "0 0 0",
            "size": "0.0025",
            "rgba": "0.2 0.8 1 0.55",
        },
    )

    half_angle = 0.5 * math.radians(spec.brush_angle_deg)
    angle_mount = ElementTree.SubElement(
        wrist,
        "body",
        {
            "name": "angled_brush_mount",
            "quat": _format_numbers((math.cos(half_angle), math.sin(half_angle), 0.0, 0.0)),
        },
    )
    for tag, name in (
        ("geom", "brush_handle"),
        ("geom", "brush_ferrule"),
        ("site", "brush_mount"),
        ("body", "brush_compliance"),
    ):
        element = _direct_named(elbow_output, tag, name)
        elbow_output.remove(element)
        _translate_element_y(element, -spec.wrist_axis_from_elbow_m)
        angle_mount.append(element)
    elbow_output.append(wrist)

    # Moving the roll joint after elbow changes qpos tree order from
    # yaw,pitch,roll,elbow,... to yaw,pitch,elbow,roll,... . Actuator ctrl order
    # remains yaw,pitch,roll,elbow because actuator declarations are unchanged.
    keyframe = _required(root.find("keyframe"), "keyframe")
    for key in keyframe.findall("key"):
        qpos = list(_numbers(key.get("qpos")))
        if len(qpos) >= 4:
            qpos[:4] = (qpos[0], qpos[1], qpos[3], qpos[2])
            key.set("qpos", _format_numbers(qpos))

    return ElementTree.tostring(root, encoding="unicode")


def build_wrist_roll_exploration_model(
    spec: AngledWristRollSpec = AngledWristRollSpec(),
):
    import mujoco

    return mujoco.MjModel.from_xml_string(build_wrist_roll_exploration_xml(spec))


def compare_wrist_roll_design(
    spec: AngledWristRollSpec = AngledWristRollSpec(),
    *,
    sweep_degrees: float = 32.0,
    sample_count: int = 65,
) -> WristRollComparison:
    """Compare uncompensated roll consequences at one declared arm pose."""

    import mujoco

    canonical = mujoco.MjModel.from_xml_path(str(ROBOT_MODEL_PATH))
    wrist = build_wrist_roll_exploration_model(spec)
    angles = np.linspace(-sweep_degrees, sweep_degrees, max(3, sample_count))
    pose = {"yaw": 0.0, "pitch": math.radians(-50.0), "elbow": math.radians(100.0)}

    canonical_points, _ = _roll_trace(canonical, angles, pose)
    wrist_points, wrist_directions = _roll_trace(wrist, angles, pose)
    wrist_axis = _site_position(wrist, "wrist_roll_axis", pose, roll_rad=0.0)
    center_tip = _site_position(wrist, "tip", pose, roll_rad=0.0)
    forearm_axis = _site_axis_y(wrist, "wrist_roll_axis", pose, roll_rad=0.0)
    offset = center_tip - wrist_axis
    axial = float(np.dot(offset, forearm_axis)) * forearm_axis
    orbit_radius = float(np.linalg.norm(offset - axial))

    canonical_bias = _joint_gravity_bias(
        canonical,
        "elbow",
        {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "elbow": 0.0},
    )
    wrist_bias = _joint_gravity_bias(
        wrist,
        "elbow",
        {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "elbow": 0.0},
    )
    direction_change = math.degrees(
        math.acos(
            float(
                np.clip(
                    np.dot(wrist_directions[0], wrist_directions[-1]),
                    -1.0,
                    1.0,
                )
            )
        )
    )
    canonical_reach, canonical_residual = _canvas_reachability(canonical)
    wrist_reach, wrist_residual = _canvas_reachability(wrist)
    return WristRollComparison(
        version=WRIST_ROLL_EXPLORATION_VERSION,
        status=WRIST_ROLL_EXPLORATION_STATUS,
        spec={key: float(value) for key, value in asdict(spec).items()},
        roll_sweep_deg=(-float(sweep_degrees), float(sweep_degrees)),
        canonical_tip_path_length_mm=1000.0 * _path_length(canonical_points),
        wrist_tip_path_length_mm=1000.0 * _path_length(wrist_points),
        wrist_tip_chord_mm=1000.0 * float(np.linalg.norm(wrist_points[-1] - wrist_points[0])),
        wrist_tip_orbit_radius_mm=1000.0 * orbit_radius,
        wrist_brush_direction_change_deg=direction_change,
        canonical_canvas_reachability=canonical_reach,
        wrist_canvas_reachability=wrist_reach,
        canonical_median_tip_residual_mm=1000.0 * canonical_residual,
        wrist_median_tip_residual_mm=1000.0 * wrist_residual,
        canonical_elbow_gravity_bias_nm=canonical_bias,
        wrist_elbow_gravity_bias_nm=wrist_bias,
        elbow_gravity_bias_increase_nm=abs(wrist_bias) - abs(canonical_bias),
        approximation=(
            "Uncompensated fixed-yaw/pitch/elbow roll sweep at one representative pose; "
            "canonical aggregate inertias do not identify individual housing/rotor/tool masses, "
            "so the RS02 relocation uses the declared provisional stator/rotor split."
        ),
    )


def _canvas_reachability(model) -> tuple[float, float]:
    """Numerical 3-position-DOF check over a bounded canvas grid and roll set."""

    center_x = 0.075
    center_z = 0.350
    contact_y = 0.4826
    coordinates = np.linspace(-0.19, 0.19, 5)
    rolls = np.radians((-32.0, 0.0, 32.0))
    residuals: list[float] = []
    for roll in rolls:
        for dz in coordinates:
            for dx in coordinates:
                target = np.asarray(
                    (center_x + dx, contact_y, center_z + dz),
                    dtype=np.float64,
                )
                residuals.append(_solve_position_ik(model, target, float(roll)))
    values = np.asarray(residuals, dtype=np.float64)
    return float(np.mean(values <= 0.002)), float(np.median(values))


def _solve_position_ik(model, target: np.ndarray, roll_rad: float) -> float:
    """Damped least-squares IK for yaw/pitch/elbow with declared roll fixed."""

    import mujoco

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    variable_names = ("yaw", "pitch", "elbow")
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in variable_names
    ]
    qpos_addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    dof_addresses = [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
    limits = np.asarray([model.jnt_range[joint_id] for joint_id in joint_ids])
    seeds = (
        (0.0, math.radians(-50.0), math.radians(100.0)),
        (math.radians(-35.0), math.radians(-25.0), math.radians(75.0)),
        (math.radians(35.0), math.radians(-25.0), math.radians(75.0)),
        (0.0, math.radians(5.0), math.radians(55.0)),
    )
    best = float("inf")
    for seed in seeds:
        data = mujoco.MjData(model)
        _set_named_pose(model, data, {"roll": roll_rad})
        for address, value in zip(qpos_addresses, seed, strict=True):
            data.qpos[address] = value
        for _ in range(90):
            mujoco.mj_forward(model, data)
            error = target - np.asarray(data.site_xpos[site_id], dtype=np.float64)
            residual = float(np.linalg.norm(error))
            best = min(best, residual)
            if residual <= 0.0005:
                break
            jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
            jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jacSite(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                site_id,
            )
            jacobian = jacobian_position[:, dof_addresses]
            damping = 2e-4
            step = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(3),
                error,
            )
            norm = float(np.linalg.norm(step))
            if norm > 0.20:
                step *= 0.20 / norm
            for index, address in enumerate(qpos_addresses):
                data.qpos[address] = float(
                    np.clip(
                        data.qpos[address] + step[index],
                        limits[index, 0],
                        limits[index, 1],
                    )
                )
    return best


def _roll_trace(model, angles_deg: np.ndarray, pose: dict[str, float]):
    import mujoco

    data = mujoco.MjData(model)
    points: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    for angle in angles_deg:
        _set_named_pose(model, data, {**pose, "roll": math.radians(float(angle))})
        mujoco.mj_forward(model, data)
        tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        points.append(np.asarray(data.site_xpos[tip_id], dtype=np.float64).copy())
        directions.append(
            np.asarray(data.site_xmat[tip_id], dtype=np.float64).reshape(3, 3)[:, 1].copy()
        )
    return np.stack(points), np.stack(directions)


def _site_position(model, name: str, pose: dict[str, float], *, roll_rad: float) -> np.ndarray:
    import mujoco

    data = mujoco.MjData(model)
    _set_named_pose(model, data, {**pose, "roll": roll_rad})
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()


def _site_axis_y(model, name: str, pose: dict[str, float], *, roll_rad: float) -> np.ndarray:
    import mujoco

    data = mujoco.MjData(model)
    _set_named_pose(model, data, {**pose, "roll": roll_rad})
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)[:, 1].copy()


def _joint_gravity_bias(model, name: str, pose: dict[str, float]) -> float:
    import mujoco

    data = mujoco.MjData(model)
    _set_named_pose(model, data, pose)
    mujoco.mj_forward(model, data)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    dof = int(model.jnt_dofadr[joint_id])
    return float(data.qfrc_bias[dof])


def _set_named_pose(model, data, pose: dict[str, float]) -> None:
    import mujoco

    for name, value in pose.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)


def _path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _relocate_roll_motor_mass(
    pitch_output: ElementTree.Element,
    elbow_output: ElementTree.Element,
    spec: AngledWristRollSpec,
) -> None:
    """Apply the declared point-mass relocation to aggregate link inertias."""

    pitch_inertial = _required(pitch_output.find("inertial"), "pitch inertial")
    elbow_inertial = _required(elbow_output.find("inertial"), "elbow inertial")
    motor_mass = spec.roll_motor_mass_kg
    stator_mass = motor_mass * (1.0 - spec.roll_rotor_mass_fraction)

    pitch_mass = float(pitch_inertial.get("mass", "0"))
    pitch_center = _numbers(pitch_inertial.get("pos"))[1]
    remaining_pitch_mass = pitch_mass - motor_mass
    if remaining_pitch_mass <= 0.0:
        raise ValueError("declared roll motor mass exceeds pitch-output aggregate mass")
    remaining_pitch_center = (
        pitch_mass * pitch_center - motor_mass * 0.083
    ) / remaining_pitch_mass
    pitch_inertial.set("mass", f"{remaining_pitch_mass:.12g}")
    pitch_inertial.set("pos", _format_numbers((0.0, remaining_pitch_center, 0.0)))
    pitch_inertial.set("diaginertia", "0.00190 0.00149 0.00190")

    elbow_mass = float(elbow_inertial.get("mass", "0"))
    elbow_center = _numbers(elbow_inertial.get("pos"))[1]
    branched_elbow_mass = elbow_mass + stator_mass
    branched_elbow_center = (
        elbow_mass * elbow_center
        + stator_mass * spec.wrist_axis_from_elbow_m
    ) / branched_elbow_mass
    elbow_inertial.set("mass", f"{branched_elbow_mass:.12g}")
    elbow_inertial.set("pos", _format_numbers((0.0, branched_elbow_center, 0.0)))
    elbow_inertial.set("diaginertia", "0.0108 0.00155 0.0108")


def _translate_element_y(element: ElementTree.Element, delta_y: float) -> None:
    if element.get("pos"):
        position = list(_numbers(element.get("pos")))
        position[1] += delta_y
        element.set("pos", _format_numbers(position))
    if element.get("fromto"):
        endpoints = list(_numbers(element.get("fromto")))
        endpoints[1] += delta_y
        endpoints[4] += delta_y
        element.set("fromto", _format_numbers(endpoints))


def _named(root: ElementTree.Element, tag: str, name: str) -> ElementTree.Element:
    element = root.find(f".//{tag}[@name='{name}']")
    return _required(element, f"{tag} {name}")


def _direct_named(parent: ElementTree.Element, tag: str, name: str) -> ElementTree.Element:
    element = parent.find(f"./{tag}[@name='{name}']")
    return _required(element, f"direct {tag} {name}")


def _required(element: ElementTree.Element | None, label: str) -> ElementTree.Element:
    if element is None:
        raise ValueError(f"canonical model is missing {label}")
    return element


def _numbers(value: str | None) -> tuple[float, ...]:
    return tuple(float(token) for token in (value or "").split())


def _format_numbers(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _set_custom_text(custom: ElementTree.Element, name: str, value: str) -> None:
    element = custom.find(f"./text[@name='{name}']")
    if element is None:
        element = ElementTree.SubElement(custom, "text", {"name": name})
    element.set("data", value)


def _set_custom_numeric(custom: ElementTree.Element, name: str, values) -> None:
    element = custom.find(f"./numeric[@name='{name}']")
    if element is None:
        element = ElementTree.SubElement(custom, "numeric", {"name": name})
    element.set("data", _format_numbers(values))


def main() -> None:
    print(json.dumps(asdict(compare_wrist_roll_design()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
