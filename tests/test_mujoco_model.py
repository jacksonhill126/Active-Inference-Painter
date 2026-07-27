from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from active_painter.arm_sim import JOINT_NAMES, ArmPose, safe_home_pose


MODEL_PATH = Path(__file__).parents[1] / "models" / "active_inference_painter.xml"
LINK_LENGTH_M = 0.3302
PITCH_AXIS_POSITION = np.asarray((0.075, 0.0, 0.391))
PITCH_AXIS_LATERAL_OFFSET_M = 0.075
CANVAS_HALF_SIZE_M = 0.254
CANVAS_CONTACT_Y_M = 0.4826
CANVAS_CENTER_X_M = 0.075
CANVAS_CENTER_Z_M = 0.350
JOINT_RANGES = {
    "yaw": (-math.pi / 2.0, math.pi / 2.0),
    "pitch": (-math.pi / 2.0, math.pi / 2.0),
    "roll": (-math.pi, math.pi),
    "elbow": (math.radians(-85.0), math.radians(150.0)),
}
NATIVE_CONTROLLER_RANGES = {
    "yaw": (-math.pi / 2.0, math.pi / 2.0),
    "pitch": (-math.pi / 2.0, math.pi / 2.0),
    "roll": (-math.pi, math.pi),
    "elbow": (0.0, math.radians(150.0)),
}
JOINT_AXES = {
    "yaw": (0.0, 0.0, 1.0),
    "pitch": (1.0, 0.0, 0.0),
    "roll": (0.0, 1.0, 0.0),
    "elbow": (1.0, 0.0, 0.0),
}
ACTUATOR_PEAK_TORQUES = {
    "yaw_position": 60.0,
    "pitch_position": 60.0,
    "roll_position": 17.0,
    "elbow_position": 17.0,
}


def _root() -> ET.Element:
    return ET.parse(MODEL_PATH).getroot()


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split())


def _named_elements(root: ET.Element, path: str) -> dict[str, ET.Element]:
    return {
        element.attrib["name"]: element
        for element in root.findall(path)
        if "name" in element.attrib
    }


def test_mujoco_model_declares_external_paint_boundary_and_si_radians() -> None:
    root = _root()
    compiler = root.find("compiler")
    texts = _named_elements(root, "./custom/text")

    assert root.attrib["model"] == "active_inference_painter"
    assert compiler is not None
    assert compiler.attrib["angle"] == "radian"
    assert texts["model_version"].attrib["data"] == "mujoco-robstride-direct-v3"
    assert texts["paint_process"].attrib["data"] == "external_VerticalCanvas"

    xml_text = MODEL_PATH.read_text(encoding="utf-8").lower()
    assert "paint deposition" in xml_text
    assert "<plugin" not in xml_text
    assert "<flex" not in xml_text


def test_joint_order_axes_ranges_and_home_preserve_native_command_subset() -> None:
    root = _root()
    joints = [
        element
        for element in root.findall(".//joint")
        if element.attrib.get("name") in JOINT_NAMES
    ]
    assert [joint.attrib["name"] for joint in joints] == list(JOINT_NAMES)

    for joint in joints:
        name = joint.attrib["name"]
        assert _floats(joint.attrib["axis"]) == pytest.approx(JOINT_AXES[name])
        assert _floats(joint.attrib["range"]) == pytest.approx(JOINT_RANGES[name])
        native_lower, native_upper = NATIVE_CONTROLLER_RANGES[name]
        model_lower, model_upper = JOINT_RANGES[name]
        assert model_lower <= native_lower
        assert model_upper >= native_upper

    key = _named_elements(root, "./keyframe/key")["safe_home"]
    expected_home = safe_home_pose()
    expected_qpos = tuple(
        math.radians(float(getattr(expected_home, name))) for name in JOINT_NAMES
    ) + (0.0,)
    assert _floats(key.attrib["qpos"]) == pytest.approx(expected_qpos)
    assert _floats(key.attrib["ctrl"]) == pytest.approx(expected_qpos[:4])
    assert ArmPose(-999, -999, -999, -999).clipped() == ArmPose(-90, -90, -180, 0)
    assert ArmPose(999, 999, 999, 999).clipped() == ArmPose(90, 90, 180, 150)


def test_shoulder_and_roll_axes_are_physically_separated() -> None:
    root = _root()
    bodies = _named_elements(root, ".//body")
    numerics = _named_elements(root, "./custom/numeric")
    canvas_position = _floats(bodies["canvas"].attrib["pos"])
    yaw_body_offset = _floats(bodies["yaw_output"].attrib["pos"])
    pitch_body_offset = _floats(bodies["pitch_output"].attrib["pos"])
    roll_body_offset = _floats(bodies["roll_output"].attrib["pos"])

    assert canvas_position == pytest.approx(
        (CANVAS_CENTER_X_M, 0.48895, CANVAS_CENTER_Z_M)
    )
    assert yaw_body_offset == pytest.approx((0.0, 0.0, 0.285))
    assert pitch_body_offset == pytest.approx((0.075, 0.0, 0.106))
    assert np.linalg.norm(pitch_body_offset) == pytest.approx(0.129849913362)
    assert np.linalg.norm(roll_body_offset) == pytest.approx(0.083)
    assert _floats(numerics["post_yaw_height_m"].attrib["data"]) == pytest.approx(
        (0.285,)
    )
    assert _floats(numerics["yaw_to_pitch_offset_m"].attrib["data"]) == pytest.approx(
        pitch_body_offset
    )
    assert _floats(numerics["yaw_to_pitch_distance_m"].attrib["data"]) == pytest.approx(
        (np.linalg.norm(pitch_body_offset),)
    )
    assert _floats(numerics["link_length_m"].attrib["data"]) == pytest.approx(
        (LINK_LENGTH_M, LINK_LENGTH_M)
    )
    assert _floats(numerics["native_canvas_contact_y_m"].attrib["data"]) == pytest.approx(
        (0.4318,)
    )
    assert _floats(numerics["canvas_contact_y_m"].attrib["data"]) == pytest.approx(
        (CANVAS_CONTACT_Y_M,)
    )
    assert _floats(numerics["canvas_center_x_m"].attrib["data"]) == pytest.approx(
        (CANVAS_CENTER_X_M,)
    )
    assert _floats(numerics["canvas_center_z_m"].attrib["data"]) == pytest.approx(
        (CANVAS_CENTER_Z_M,)
    )
    assert yaw_body_offset[2] > 0.25
    assert pitch_body_offset[0] > 0.05
    assert np.linalg.norm(roll_body_offset) > 0.05


def test_robstride_direct_mount_specs_and_actuator_limits_are_encoded() -> None:
    root = _root()
    numerics = _named_elements(root, "./custom/numeric")
    actuators = _named_elements(root, "./actuator/position")
    joints = _named_elements(root, ".//joint")

    assert _floats(numerics["robstride_rated_torque_nm"].attrib["data"]) == (20, 20, 6, 6)
    assert _floats(numerics["robstride_peak_torque_nm"].attrib["data"]) == (60, 60, 17, 17)
    assert _floats(numerics["robstride_mass_kg"].attrib["data"]) == pytest.approx(
        (0.880, 0.880, 0.405, 0.405)
    )
    assert _floats(numerics["robstride_reduction_ratio"].attrib["data"]) == (
        9,
        9,
        7.75,
        7.75,
    )
    assert _floats(numerics["robstride_rated_voltage_v"].attrib["data"]) == (
        48,
        48,
        48,
        48,
    )
    assert _floats(numerics["robstride_voltage_min_v"].attrib["data"]) == (
        15,
        15,
        24,
        24,
    )
    assert _floats(numerics["robstride_rated_power_w"].attrib["data"]) == (
        380,
        380,
        170,
        170,
    )
    assert _floats(numerics["robstride_rated_speed_rad_s"].attrib["data"]) == (
        pytest.approx(100.0 * 2.0 * math.pi / 60.0),
    ) * 4
    assert _floats(numerics["robstride_no_load_speed_rad_s"].attrib["data"]) == (
        pytest.approx(195.0 * 2.0 * math.pi / 60.0),
        pytest.approx(195.0 * 2.0 * math.pi / 60.0),
        pytest.approx(410.0 * 2.0 * math.pi / 60.0),
        pytest.approx(410.0 * 2.0 * math.pi / 60.0),
    )
    assert _floats(numerics["robstride_rated_phase_current_apk"].attrib["data"]) == (
        12,
        12,
        7,
        7,
    )
    assert _floats(numerics["robstride_peak_phase_current_apk"].attrib["data"]) == (
        43,
        43,
        23,
        23,
    )

    for joint_name, actuator_name in zip(JOINT_NAMES, ACTUATOR_PEAK_TORQUES, strict=True):
        actuator = actuators[actuator_name]
        peak = ACTUATOR_PEAK_TORQUES[actuator_name]
        assert actuator.attrib["joint"] == joint_name
        assert float(actuator.attrib["gear"]) == 1.0
        assert _floats(actuator.attrib["ctrlrange"]) == pytest.approx(JOINT_RANGES[joint_name])
        assert _floats(actuator.attrib["forcerange"]) == pytest.approx((-peak, peak))
        assert _floats(joints[joint_name].attrib["actuatorfrcrange"]) == pytest.approx(
            (-peak, peak)
        )


def test_half_inch_brush_has_axial_compliance_and_bounded_canvas_contact() -> None:
    root = _root()
    numerics = _named_elements(root, "./custom/numeric")
    geoms = _named_elements(root, ".//geom")
    joints = _named_elements(root, ".//joint")
    pairs = _named_elements(root, "./contact/pair")
    sensors = _named_elements(root, "./sensor/*")
    sites = _named_elements(root, ".//site")

    brush_diameter = _floats(numerics["brush_diameter_m"].attrib["data"])[0]
    bristle_radius = _floats(geoms["bristle_contact"].attrib["size"])[0]
    assert brush_diameter == pytest.approx(0.5 * 0.0254)
    assert 2.0 * bristle_radius == pytest.approx(brush_diameter)
    assert geoms["bristle_contact"].attrib["type"] == "cylinder"
    assert _floats(geoms["bristle_contact"].attrib["fromto"])[4] == pytest.approx(0.035)

    compression = joints["brush_compression"]
    assert compression.attrib["type"] == "slide"
    assert _floats(compression.attrib["axis"]) == (0.0, 1.0, 0.0)
    assert _floats(compression.attrib["range"]) == pytest.approx((-0.012, 0.0))
    assert float(compression.attrib["stiffness"]) > 0.0
    assert float(compression.attrib["damping"]) > 0.0

    pair = pairs["brush_canvas_contact"]
    assert pair.attrib["geom1"] == "bristle_contact"
    assert pair.attrib["geom2"] == "canvas_surface"
    assert sensors["brush_touch_sensor"].attrib["site"] == "brush_contact_zone"
    assert "tip" in sites

    canvas = geoms["canvas_surface"]
    assert canvas.attrib["type"] == "box"
    assert _floats(canvas.attrib["size"]) == pytest.approx((0.254, 0.00635, 0.254))


def test_mujoco_compiles_with_stable_adapter_names_and_separated_anchors() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    assert model.nq == 5
    assert model.nv == 5
    assert model.nu == 4
    assert model.nkey == 4
    assert [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    ] == [*JOINT_NAMES, "brush_compression"]

    for name in (
        "tip",
        "brush_touch_sensor",
        "brush_force_sensor",
        "tip_position_sensor",
        "yaw_torque_sensor",
        "pitch_torque_sensor",
        "roll_torque_sensor",
        "elbow_torque_sensor",
    ):
        object_type = (
            mujoco.mjtObj.mjOBJ_SITE
            if name == "tip"
            else mujoco.mjtObj.mjOBJ_SENSOR
        )
        assert mujoco.mj_name2id(model, object_type, name) >= 0

    mujoco.mj_forward(model, data)
    yaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")
    pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pitch")
    roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "roll")
    assert data.xanchor[yaw_id] == pytest.approx((0.0, 0.0, 0.285))
    assert data.xanchor[pitch_id] - data.xanchor[yaw_id] == pytest.approx(
        (0.075, 0.0, 0.106)
    )
    assert np.linalg.norm(data.xanchor[roll_id] - data.xanchor[pitch_id]) == pytest.approx(
        0.083
    )


def test_mujoco_joint_signs_match_the_declared_offset_kinematics() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")

    yaw, pitch, roll, elbow = np.deg2rad((14.0, -38.0, 32.0, 92.0))
    data.qpos[:] = (yaw, pitch, roll, elbow, 0.0)
    mujoco.mj_forward(model, data)

    def rot_x(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.asarray(((1, 0, 0), (0, c, -s), (0, s, c)))

    def rot_y(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.asarray(((c, 0, s), (0, 1, 0), (-s, 0, c)))

    def rot_z(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1)))

    yaw_anchor = np.asarray((0.0, 0.0, 0.285))
    shoulder = yaw_anchor + rot_z(yaw) @ np.asarray((0.075, 0.0, 0.106))
    r_shoulder = rot_z(yaw) @ rot_x(pitch)
    elbow_position = shoulder + r_shoulder @ np.asarray((0.0, LINK_LENGTH_M, 0.0))
    expected_tip = elbow_position + (
        r_shoulder
        @ rot_y(roll)
        @ rot_x(elbow)
        @ np.asarray((0.0, LINK_LENGTH_M, 0.0))
    )
    assert data.site_xpos[tip_id] == pytest.approx(expected_tip, abs=1e-10)


def test_canvas_center_edges_and_corners_are_reachable_in_hardware_ranges() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")

    def pose_for_canvas_point(x: float, z: float) -> tuple[float, float, float]:
        radial_xy = math.hypot(x, CANVAS_CONTACT_Y_M)
        yaw = math.atan2(CANVAS_CONTACT_Y_M, x) - math.acos(
            PITCH_AXIS_LATERAL_OFFSET_M / radial_xy
        )
        planar_y = math.sqrt(
            radial_xy**2 - PITCH_AXIS_LATERAL_OFFSET_M**2
        )
        planar_z = z - PITCH_AXIS_POSITION[2]
        reach = math.hypot(planar_y, planar_z)
        elbow_magnitude = math.acos(
            (reach**2 - 2.0 * LINK_LENGTH_M**2)
            / (2.0 * LINK_LENGTH_M**2)
        )
        for elbow in (-elbow_magnitude, elbow_magnitude):
            pitch = math.atan2(planar_z, planar_y) - math.atan2(
                LINK_LENGTH_M * math.sin(elbow),
                LINK_LENGTH_M * (1.0 + math.cos(elbow)),
            )
            if (
                JOINT_RANGES["yaw"][0] <= yaw <= JOINT_RANGES["yaw"][1]
                and JOINT_RANGES["pitch"][0] <= pitch <= JOINT_RANGES["pitch"][1]
                and JOINT_RANGES["elbow"][0] <= elbow <= JOINT_RANGES["elbow"][1]
            ):
                return yaw, pitch, elbow
        raise AssertionError(f"No in-range branch for canvas point {(x, z)}")

    yaw_by_x: dict[float, float] = {}
    for x in (
        CANVAS_CENTER_X_M - CANVAS_HALF_SIZE_M,
        CANVAS_CENTER_X_M,
        CANVAS_CENTER_X_M + CANVAS_HALF_SIZE_M,
    ):
        for z in (
            CANVAS_CENTER_Z_M - CANVAS_HALF_SIZE_M,
            CANVAS_CENTER_Z_M,
            CANVAS_CENTER_Z_M + CANVAS_HALF_SIZE_M,
        ):
            yaw, pitch, elbow = pose_for_canvas_point(x, z)
            yaw_by_x[x] = yaw
            data.qpos[:] = (yaw, pitch, 0.0, elbow, 0.0)
            mujoco.mj_forward(model, data)
            assert data.site_xpos[tip_id] == pytest.approx(
                (x, CANVAS_CONTACT_Y_M, z), abs=1e-9
            )

    assert math.degrees(yaw_by_x[CANVAS_CENTER_X_M]) == pytest.approx(0.0, abs=1e-10)
    left_yaw_deg = math.degrees(
        yaw_by_x[CANVAS_CENTER_X_M - CANVAS_HALF_SIZE_M]
    )
    right_yaw_deg = math.degrees(
        yaw_by_x[CANVAS_CENTER_X_M + CANVAS_HALF_SIZE_M]
    )
    assert left_yaw_deg == pytest.approx(28.73, abs=0.01)
    assert right_yaw_deg == pytest.approx(-26.91, abs=0.01)


def test_provisional_static_gravity_loads_stay_below_rated_torque() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)

    gravity_bias = dict(zip((*JOINT_NAMES, "brush_compression"), data.qfrc_bias, strict=True))
    assert gravity_bias["yaw"] == pytest.approx(0.0, abs=1e-12)
    assert gravity_bias["roll"] == pytest.approx(0.0, abs=1e-12)
    assert abs(gravity_bias["pitch"]) == pytest.approx(6.6537, abs=5e-4)
    assert abs(gravity_bias["elbow"]) == pytest.approx(1.3263, abs=5e-4)
    assert abs(gravity_bias["pitch"]) < 20.0
    assert abs(gravity_bias["elbow"]) < 6.0


def test_lower_arm_down_key_uses_near_vertical_negative_elbow_travel() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "elbow")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "lower_arm_down")

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    lower_arm = data.site_xpos[tip_id] - data.xanchor[elbow_id]
    lower_arm_pitch_deg = math.degrees(math.atan2(lower_arm[2], lower_arm[1]))
    assert math.degrees(data.qpos[3]) == pytest.approx(-80.0)
    assert lower_arm_pitch_deg == pytest.approx(-80.0)
    assert data.site_xpos[tip_id, 2] > 0.04
    assert data.site_xpos[tip_id, 1] < CANVAS_CONTACT_Y_M
    assert data.ncon == 0

    for _ in range(2000):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert math.degrees(data.qpos[3]) == pytest.approx(-80.0, abs=0.2)


def test_safe_home_is_clear_and_contact_probe_contacts_canvas() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    canvas_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "canvas_surface"
    )
    bristle_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "bristle_contact"
    )

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "safe_home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    assert data.site_xpos[tip_id, 1] < CANVAS_CONTACT_Y_M
    assert data.ncon == 0

    probe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "contact_probe")
    mujoco.mj_resetDataKeyframe(model, data, probe_id)
    mujoco.mj_forward(model, data)
    contact_pairs = {
        frozenset((int(data.contact[index].geom1), int(data.contact[index].geom2)))
        for index in range(data.ncon)
    }
    assert frozenset((canvas_geom_id, bristle_geom_id)) in contact_pairs

    bottom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "bottom_edge_probe"
    )
    mujoco.mj_resetDataKeyframe(model, data, bottom_id)
    mujoco.mj_forward(model, data)
    assert data.site_xpos[tip_id] == pytest.approx((0.075, 0.486, 0.106), abs=1e-9)
    bottom_contact_pairs = {
        frozenset((int(data.contact[index].geom1), int(data.contact[index].geom2)))
        for index in range(data.ncon)
    }
    assert frozenset((canvas_geom_id, bristle_geom_id)) in bottom_contact_pairs
