from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np


INCH_TO_M = 0.0254
ROBOT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "active_inference_painter.xml"
CONTROLLED_JOINTS = ("yaw", "pitch", "roll", "elbow")


def _numbers(value: str | None, *, default: tuple[float, ...] = ()) -> list[float]:
    if not value:
        return list(default)
    return [float(item) for item in value.split()]


def _vec3(value: str | None) -> list[float]:
    values = _numbers(value, default=(0.0, 0.0, 0.0))
    if len(values) != 3:
        raise ValueError(f"expected three values, got {values}")
    return values


def _geom_payload(element: ElementTree.Element) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": element.get("name", ""),
        "type": element.get("type", "sphere"),
        "position": _vec3(element.get("pos")),
        "size": _numbers(element.get("size")),
        "material": element.get("material"),
    }
    if element.get("fromto"):
        payload["fromTo"] = _numbers(element.get("fromto"))
    if element.get("rgba"):
        payload["rgba"] = _numbers(element.get("rgba"))
    return payload


def _joint_payload(element: ElementTree.Element) -> dict[str, Any]:
    return {
        "name": element.get("name", ""),
        "type": element.get("type", "hinge"),
        "axis": _vec3(element.get("axis")),
        "range": _numbers(element.get("range")),
    }


def _body_payload(element: ElementTree.Element) -> dict[str, Any]:
    return {
        "name": element.get("name", ""),
        "position": _vec3(element.get("pos")),
        "joints": [_joint_payload(joint) for joint in element.findall("joint")],
        "geoms": [_geom_payload(geom) for geom in element.findall("geom")],
        "bodies": [_body_payload(body) for body in element.findall("body")],
    }


def _material_payload(element: ElementTree.Element) -> dict[str, Any]:
    return {
        "name": element.get("name", ""),
        "rgba": _numbers(element.get("rgba"), default=(0.45, 0.48, 0.52, 1.0)),
        "specular": float(element.get("specular", "0.25")),
        "shininess": float(element.get("shininess", "0.25")),
        "reflectance": float(element.get("reflectance", "0")),
        "texture": element.get("texture"),
    }


def _named(root: ElementTree.Element, tag: str, name: str) -> ElementTree.Element:
    element = root.find(f".//{tag}[@name='{name}']")
    if element is None:
        raise ValueError(f"MuJoCo model has no {tag} named {name!r}")
    return element


def _from_to_length(element: ElementTree.Element) -> float:
    values = _numbers(element.get("fromto"))
    if len(values) != 6:
        raise ValueError(f"{element.get('name', element.tag)!r} has no valid fromto")
    return float(np.linalg.norm(np.asarray(values[3:]) - np.asarray(values[:3])))


def _actuator_assignments(value: str) -> dict[str, str]:
    tokens = value.split("_")
    return {
        tokens[index]: tokens[index + 1]
        for index in range(0, len(tokens) - 1, 2)
    }


@lru_cache(maxsize=4)
def _load_robot_visual_model(model_path: str) -> dict[str, Any]:
    root = ElementTree.parse(model_path).getroot()
    custom = root.find("custom")
    if custom is None:
        raise ValueError("MuJoCo model has no custom metadata")

    text = {item.get("name", ""): item.get("data", "") for item in custom.findall("text")}
    numeric = {
        item.get("name", ""): _numbers(item.get("data"))
        for item in custom.findall("numeric")
    }
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MuJoCo model has no worldbody")

    materials: dict[str, dict[str, Any]] = {}
    asset = root.find("asset")
    if asset is not None:
        materials = {
            material.get("name", ""): _material_payload(material)
            for material in asset.findall("material")
        }

    yaw_output = _named(root, "body", "yaw_output")
    pitch_output = _named(root, "body", "pitch_output")
    roll_output = _named(root, "body", "roll_output")
    elbow_output = _named(root, "body", "elbow_output")
    brush_compliance = _named(root, "body", "brush_compliance")
    canvas_body = _named(root, "body", "canvas")
    canvas_surface = _named(root, "geom", "canvas_surface")
    bristle_contact = _named(root, "geom", "bristle_contact")
    brush_compression = _named(root, "joint", "brush_compression")
    tip_site = _named(root, "site", "tip")

    yaw_origin = _vec3(yaw_output.get("pos"))
    yaw_height = yaw_origin[2]
    yaw_to_pitch = _vec3(pitch_output.get("pos"))
    pitch_to_roll_vector = np.asarray(_vec3(roll_output.get("pos")))
    roll_to_elbow_vector = np.asarray(_vec3(elbow_output.get("pos")))
    elbow_to_brush_vector = np.asarray(_vec3(brush_compliance.get("pos")))
    brush_to_tip_vector = np.asarray(_vec3(tip_site.get("pos")))
    pitch_anchor_at_zero = [
        yaw_origin[0] + yaw_to_pitch[0],
        yaw_origin[1] + yaw_to_pitch[1],
        yaw_origin[2] + yaw_to_pitch[2],
    ]
    canvas_body_position = np.asarray(_vec3(canvas_body.get("pos")))
    canvas_geom_position = np.asarray(_vec3(canvas_surface.get("pos")))
    canvas_half_size = _numbers(canvas_surface.get("size"))
    if len(canvas_half_size) != 3:
        raise ValueError("canvas_surface must have three box half-sizes")
    canvas_center = [
        float(canvas_body_position[0] + canvas_geom_position[0]),
        float(canvas_body_position[1] + canvas_geom_position[1] - canvas_half_size[1]),
        float(canvas_body_position[2] + canvas_geom_position[2]),
    ]
    joint_range_deg = {
        name: [
            float(np.rad2deg(value))
            for value in _numbers(_named(root, "joint", name).get("range"))
        ]
        for name in CONTROLLED_JOINTS
    }
    compression_range = _numbers(brush_compression.get("range"))
    if len(compression_range) != 2:
        raise ValueError("brush_compression must have a two-value range")
    bend_range = numeric["brush_tangential_bend_range_rad"]
    if len(bend_range) != 2:
        raise ValueError("brush_tangential_bend_range_rad must have two values")
    actuator_assignments = _actuator_assignments(text["actuator_assignment"])

    return {
        "name": root.get("model", "active_inference_painter"),
        "version": text["model_version"],
        "units": "m",
        "source": "models/active_inference_painter.xml",
        "kinematicConvention": text["kinematic_convention"],
        "jointOrder": list(CONTROLLED_JOINTS),
        "jointRangeDeg": joint_range_deg,
        "fidelity": {
            "actuatorModel": text["actuator_model"],
            "torqueSaturation": text["torque_saturation_semantics"],
            "powerElectronics": text["power_electronics_model"],
            "encoder": text["encoder_model"],
            "thermal": text["thermal_model"],
            "transmission": text["transmission_model"],
        },
        "materials": materials,
        "world": {
            "geoms": [_geom_payload(geom) for geom in worldbody.findall("geom")],
            "bodies": [_body_payload(body) for body in worldbody.findall("body")],
        },
        "kinematics": {
            "yawOrigin": yaw_origin,
            "yawToPitch": yaw_to_pitch,
            "pitchAnchorAtZero": pitch_anchor_at_zero,
            "pitchToRoll": float(np.linalg.norm(pitch_to_roll_vector)),
            "upperArmLength": float(
                np.linalg.norm(pitch_to_roll_vector + roll_to_elbow_vector)
            ),
            "lowerArmLength": float(
                np.linalg.norm(elbow_to_brush_vector + brush_to_tip_vector)
            ),
        },
        "canvas": {
            "width": 2.0 * canvas_half_size[0],
            "height": 2.0 * canvas_half_size[2],
            "center": canvas_center,
            "contactY": canvas_center[1],
            "nativeContactY": numeric["native_canvas_contact_y_m"][0],
        },
        "brush": {
            "diameter": 2.0 * _numbers(bristle_contact.get("size"))[0],
            "bristleLength": _from_to_length(bristle_contact),
            "compressionTravel": compression_range[1] - compression_range[0],
            "bendRangeRad": bend_range,
            "tangentialStiffnessNmPerRad": numeric[
                "brush_tangential_stiffness_nm_per_rad"
            ],
            "tangentialDampingNmsPerRad": numeric[
                "brush_tangential_damping_nms_per_rad"
            ],
        },
        "motors": {
            name: {
                "model": actuator_assignments[name].replace("RS", "RobStride "),
                "actuatorModel": text["actuator_model"],
                "ratedTorqueNm": numeric["robstride_rated_torque_nm"][index],
                "peakTorqueNm": numeric["robstride_peak_torque_nm"][index],
                "ratedVoltageV": numeric["robstride_rated_voltage_v"][index],
                "voltageRangeV": [
                    numeric["robstride_voltage_min_v"][index],
                    numeric["robstride_voltage_max_v"][index],
                ],
                "ratedPowerW": numeric["robstride_rated_power_w"][index],
                "ratedSpeedRadS": numeric["robstride_rated_speed_rad_s"][index],
                "noLoadSpeedRadS": numeric["robstride_no_load_speed_rad_s"][index],
                "noLoadCurrentArms": numeric[
                    "robstride_no_load_current_arms"
                ][index],
                "ratedCurrentArms": numeric["robstride_rated_current_arms"][index],
                "modelPeakCurrentA": numeric["robstride_model_peak_current_a"][index],
                "ratedPhaseCurrentApk": numeric[
                    "robstride_rated_phase_current_apk"
                ][index],
                "peakPhaseCurrentApk": numeric[
                    "robstride_peak_phase_current_apk"
                ][index],
                "torqueConstantNmPerArms": numeric[
                    "robstride_torque_constant_nm_per_arms"
                ][index],
                "backEmfConstantVsPerRad": numeric[
                    "robstride_back_emf_constant_vs_per_rad"
                ][index],
                "effectiveMotorConstantNmPerA": numeric[
                    "robstride_effective_motor_constant_nm_per_a"
                ][index],
                "equivalentTerminalResistanceOhm": numeric[
                    "robstride_equivalent_terminal_resistance_ohm"
                ][index],
                "equivalentViscousLossNmsPerRad": numeric[
                    "robstride_equivalent_viscous_loss_nms_per_rad"
                ][index],
                "lineResistanceOhm": numeric[
                    "robstride_line_resistance_ohm"
                ][index],
                "lineInductanceH": numeric["robstride_line_inductance_h"][index],
                "electricalTimeConstantS": numeric[
                    "robstride_electrical_time_constant_s"
                ][index],
                "controllerKpVPerRad": numeric[
                    "robstride_controller_kp_v_per_rad"
                ][index],
                "controllerKdVsPerRad": numeric[
                    "robstride_controller_kd_vs_per_rad"
                ][index],
                "poleCount": int(numeric["robstride_pole_count"][index]),
                "encoderCount": int(numeric["robstride_encoder_count"][index]),
                "massKg": numeric["robstride_mass_kg"][index],
                "reductionRatio": numeric["robstride_reduction_ratio"][index],
            }
            for index, name in enumerate(CONTROLLED_JOINTS)
        },
        "paintProcess": text["paint_process"],
        "compatibility": {
            "mode": "legacy_canvas_cartesian_retarget",
            "sourceUnits": "in",
            "sourceCanvasContactY": numeric["native_canvas_contact_y_m"][0] / INCH_TO_M,
            "description": (
                "Visualization-only conventional IK maps the legacy Cartesian canvas point "
                "onto the physical MJCF canvas. It does not select painting policies."
            ),
        },
    }


def load_robot_visual_model(model_path: Path | str = ROBOT_MODEL_PATH) -> dict[str, Any]:
    return _load_robot_visual_model(str(Path(model_path).resolve()))


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def physical_tip(
    model: dict[str, Any],
    joint_position_deg: dict[str, float],
) -> np.ndarray:
    kinematics = model["kinematics"]
    yaw = np.deg2rad(joint_position_deg["yaw"])
    pitch = np.deg2rad(joint_position_deg["pitch"])
    roll = np.deg2rad(joint_position_deg["roll"])
    elbow = np.deg2rad(joint_position_deg["elbow"])

    yaw_rotation = _rot_z(yaw)
    yaw_origin = np.asarray(kinematics["yawOrigin"], dtype=np.float64)
    pitch_origin = yaw_origin + yaw_rotation @ np.asarray(
        kinematics["yawToPitch"], dtype=np.float64
    )
    upper_rotation = yaw_rotation @ _rot_x(pitch)
    elbow_origin = pitch_origin + upper_rotation @ np.asarray(
        [0.0, kinematics["upperArmLength"], 0.0], dtype=np.float64
    )
    forearm_rotation = upper_rotation @ _rot_y(roll) @ _rot_x(elbow)
    return elbow_origin + forearm_rotation @ np.asarray(
        [0.0, kinematics["lowerArmLength"], 0.0], dtype=np.float64
    )


def legacy_tip_to_physical_target(
    model: dict[str, Any],
    legacy_tip_in: np.ndarray,
) -> np.ndarray:
    canvas = model["canvas"]
    source_canvas_y = model["compatibility"]["sourceCanvasContactY"]
    return np.asarray(
        [
            canvas["center"][0] + float(legacy_tip_in[0]) * INCH_TO_M,
            canvas["contactY"] + (float(legacy_tip_in[1]) - source_canvas_y) * INCH_TO_M,
            canvas["center"][2] + float(legacy_tip_in[2]) * INCH_TO_M,
        ],
        dtype=np.float64,
    )


def _retarget_fixed_roll(
    model: dict[str, Any],
    target: np.ndarray,
    source_pose_deg: dict[str, float],
    roll_deg: float,
    initial_pose_deg: dict[str, float] | None = None,
) -> tuple[dict[str, float], float]:
    ranges = model["jointRangeDeg"]
    lower = np.deg2rad(
        [ranges["yaw"][0], ranges["pitch"][0], ranges["elbow"][0]]
    )
    upper = np.deg2rad(
        [ranges["yaw"][1], ranges["pitch"][1], ranges["elbow"][1]]
    )
    source = np.deg2rad(
        [source_pose_deg["yaw"], source_pose_deg["pitch"], source_pose_deg["elbow"]]
    )
    initial = (
        source
        if initial_pose_deg is None
        else np.deg2rad(
            [
                initial_pose_deg["yaw"],
                initial_pose_deg["pitch"],
                initial_pose_deg["elbow"],
            ]
        )
    )
    seeds = (
        initial,
        source,
        np.deg2rad([0.0, -48.0, 86.0]),
        np.deg2rad([30.0, -45.0, 75.0]),
        np.deg2rad([-30.0, -45.0, 75.0]),
        np.deg2rad([0.0, -15.0, -55.0]),
    )
    best_pose: dict[str, float] | None = None
    best_error = float("inf")
    best_score = float("inf")

    for seed_index, seed in enumerate(seeds):
        q = np.clip(np.asarray(seed, dtype=np.float64), lower, upper)
        for _ in range(36):
            pose = {
                "yaw": float(np.rad2deg(q[0])),
                "pitch": float(np.rad2deg(q[1])),
                "roll": float(roll_deg),
                "elbow": float(np.rad2deg(q[2])),
            }
            tip = physical_tip(model, pose)
            residual = target - tip
            if float(np.linalg.norm(residual)) < 2e-7:
                break
            jacobian = np.empty((3, 3), dtype=np.float64)
            epsilon = 1e-5
            for column in range(3):
                q_plus = q.copy()
                q_minus = q.copy()
                q_plus[column] += epsilon
                q_minus[column] -= epsilon
                plus_pose = {
                    "yaw": float(np.rad2deg(q_plus[0])),
                    "pitch": float(np.rad2deg(q_plus[1])),
                    "roll": float(roll_deg),
                    "elbow": float(np.rad2deg(q_plus[2])),
                }
                minus_pose = {
                    "yaw": float(np.rad2deg(q_minus[0])),
                    "pitch": float(np.rad2deg(q_minus[1])),
                    "roll": float(roll_deg),
                    "elbow": float(np.rad2deg(q_minus[2])),
                }
                jacobian[:, column] = (
                    physical_tip(model, plus_pose) - physical_tip(model, minus_pose)
                ) / (2.0 * epsilon)
            damping = 2e-5
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(3),
                residual,
            )
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > 0.22:
                delta *= 0.22 / delta_norm
            q = np.clip(q + delta, lower, upper)

        candidate = {
            "yaw": float(np.rad2deg(q[0])),
            "pitch": float(np.rad2deg(q[1])),
            "roll": float(roll_deg),
            "elbow": float(np.rad2deg(q[2])),
        }
        error = float(np.linalg.norm(physical_tip(model, candidate) - target))
        source_distance = float(np.linalg.norm(q - source))
        score = error + 1e-7 * source_distance
        if score < best_score:
            best_pose = candidate
            best_error = error
            best_score = score
        if seed_index == 0 and error < 1e-5:
            return candidate, error

    if best_pose is None:
        raise RuntimeError("physical visualization IK produced no candidate")
    return best_pose, best_error


def retarget_legacy_robot_state(
    model: dict[str, Any],
    controller_pose_deg: dict[str, float],
    legacy_render_tip_in: np.ndarray,
    initial_pose_deg: dict[str, float] | None = None,
) -> dict[str, Any]:
    target = legacy_tip_to_physical_target(model, legacy_render_tip_in)
    preserved_pose, preserved_error = _retarget_fixed_roll(
        model,
        target,
        controller_pose_deg,
        controller_pose_deg["roll"],
        initial_pose_deg,
    )
    best_pose = preserved_pose
    best_error = preserved_error
    roll_preserved = True

    if preserved_error > 5e-4 and abs(controller_pose_deg["roll"]) > 1e-7:
        neutral_pose, neutral_error = _retarget_fixed_roll(
            model,
            target,
            controller_pose_deg,
            0.0,
            initial_pose_deg,
        )
        if neutral_error < best_error:
            best_pose = neutral_pose
            best_error = neutral_error
            roll_preserved = False

    return {
        "mode": model["compatibility"]["mode"],
        "jointPositionDeg": best_pose,
        "controllerJointPositionDeg": {
            name: float(controller_pose_deg[name]) for name in CONTROLLED_JOINTS
        },
        "tipM": physical_tip(model, best_pose).astype(float).tolist(),
        "mappedCartesianTargetM": target.astype(float).tolist(),
        "alignmentErrorM": best_error,
        "rollPreserved": roll_preserved,
        "brushCompressionM": 0.0,
        "brushBendRad": {"x": 0.0, "z": 0.0},
    }
