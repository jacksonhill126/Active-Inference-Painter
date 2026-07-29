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
ACTUATOR_RATED_TORQUES = (20.0, 20.0, 6.0, 6.0)
EFFECTIVE_MOTOR_CONSTANTS = (
    2.327719630643,
    2.327719630643,
    1.147984041196,
    1.147984041196,
)
EQUIVALENT_RESISTANCES = (
    1.862175704515,
    1.862175704515,
    3.241366704555,
    3.241366704555,
)
EQUIVALENT_VISCOUS_LOSS = (
    0.068394108064,
    0.068394108064,
    0.013368829372,
    0.013368829372,
)
ELECTRICAL_TIME_CONSTANTS = (
    0.000705128205,
    0.000705128205,
    0.000453448276,
    0.000453448276,
)


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
    assert (
        texts["model_version"].attrib["data"]
        == "mujoco-robstride-electromechanical-v4"
    )
    assert (
        texts["actuator_model"].attrib["data"]
        == "output_equivalent_dcmotor_position_v1"
    )
    assert (
        texts["torque_saturation_semantics"].attrib["data"]
        == "hard_peak_envelope_without_thermal_derating"
    )
    assert (
        texts["encoder_model"].attrib["data"]
        == "ideal_joint_state_without_noise_or_delay"
    )
    assert texts["thermal_model"].attrib["data"] == "disabled_missing_vendor_constants"
    assert (
        texts["transmission_model"].attrib["data"]
        == "rigid_integrated_reducer_without_backlash"
    )
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
    ) + (0.0, 0.0, 0.0)
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
    actuators = _named_elements(root, "./actuator/dcmotor")
    joints = _named_elements(root, ".//joint")

    assert _floats(
        numerics["robstride_rated_torque_nm"].attrib["data"]
    ) == ACTUATOR_RATED_TORQUES
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
    assert _floats(
        numerics["robstride_effective_motor_constant_nm_per_a"].attrib["data"]
    ) == pytest.approx(EFFECTIVE_MOTOR_CONSTANTS)
    assert _floats(
        numerics["robstride_equivalent_terminal_resistance_ohm"].attrib["data"]
    ) == pytest.approx(EQUIVALENT_RESISTANCES)
    assert _floats(
        numerics["robstride_equivalent_viscous_loss_nms_per_rad"].attrib["data"]
    ) == pytest.approx(EQUIVALENT_VISCOUS_LOSS)
    assert _floats(
        numerics["robstride_electrical_time_constant_s"].attrib["data"]
    ) == pytest.approx(ELECTRICAL_TIME_CONSTANTS)
    assert _floats(
        numerics["robstride_rated_current_arms"].attrib["data"]
    ) == pytest.approx(
        (
            12.0 / math.sqrt(2.0),
            12.0 / math.sqrt(2.0),
            7.0 / math.sqrt(2.0),
            7.0 / math.sqrt(2.0),
        )
    )
    assert _floats(numerics["robstride_pole_count"].attrib["data"]) == (
        42,
        42,
        28,
        28,
    )

    torque_constant = np.asarray(
        _floats(numerics["robstride_torque_constant_nm_per_arms"].attrib["data"])
    )
    back_emf_constant = np.asarray(
        _floats(
            numerics["robstride_back_emf_constant_vs_per_rad"].attrib["data"]
        )
    )
    effective_constant = np.asarray(EFFECTIVE_MOTOR_CONSTANTS)
    equivalent_resistance = np.asarray(EQUIVALENT_RESISTANCES)
    equivalent_viscous_loss = np.asarray(EQUIVALENT_VISCOUS_LOSS)
    no_load_current = np.asarray(
        _floats(numerics["robstride_no_load_current_arms"].attrib["data"])
    )
    no_load_speed = np.asarray(
        _floats(numerics["robstride_no_load_speed_rad_s"].attrib["data"])
    )
    peak_torque = np.asarray(
        _floats(numerics["robstride_peak_torque_nm"].attrib["data"])
    )
    assert effective_constant == pytest.approx(
        np.sqrt(torque_constant * back_emf_constant)
    )
    assert equivalent_resistance == pytest.approx(
        48.0 * effective_constant / peak_torque
    )
    assert 48.0 == pytest.approx(
        equivalent_resistance * no_load_current
        + back_emf_constant * no_load_speed
    )
    assert equivalent_viscous_loss * no_load_speed == pytest.approx(
        effective_constant * no_load_current
    )

    for index, (joint_name, actuator_name) in enumerate(
        zip(JOINT_NAMES, ACTUATOR_PEAK_TORQUES, strict=True)
    ):
        actuator = actuators[actuator_name]
        peak = ACTUATOR_PEAK_TORQUES[actuator_name]
        assert actuator.attrib["joint"] == joint_name
        assert float(actuator.attrib["gear"]) == 1.0
        assert actuator.attrib["input"] == "position"
        assert _floats(actuator.attrib["ctrlrange"]) == pytest.approx(JOINT_RANGES[joint_name])
        assert _floats(actuator.attrib["saturation"]) == pytest.approx((peak, 0.0, 0.0))
        assert _floats(actuator.attrib["inductance"]) == pytest.approx(
            (0.0, ELECTRICAL_TIME_CONSTANTS[index])
        )
        assert float(actuator.attrib["resistance"]) == pytest.approx(
            EQUIVALENT_RESISTANCES[index]
        )
        assert float(joints[joint_name].attrib["damping"]) == pytest.approx(
            EQUIVALENT_VISCOUS_LOSS[index]
        )
        assert float(joints[joint_name].attrib["frictionloss"]) == 0.0
        controller = _floats(actuator.attrib["controller"])
        assert controller[1] == 0.0
        assert controller[3:5] == (0.0, 0.0)
        assert controller[5] == 48.0
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
    bend_range = _floats(
        numerics["brush_tangential_bend_range_rad"].attrib["data"]
    )
    for name, axis in (
        ("brush_bend_x", (1.0, 0.0, 0.0)),
        ("brush_bend_z", (0.0, 0.0, 1.0)),
    ):
        bend = joints[name]
        assert bend.attrib["type"] == "hinge"
        assert _floats(bend.attrib["axis"]) == axis
        assert _floats(bend.attrib["range"]) == pytest.approx(bend_range)
        assert float(bend.attrib["stiffness"]) == pytest.approx(1.2)
        assert float(bend.attrib["damping"]) == pytest.approx(0.01)

    pair = pairs["brush_canvas_contact"]
    assert pair.attrib["geom1"] == "bristle_contact"
    assert pair.attrib["geom2"] == "canvas_surface"
    assert int(pair.attrib["condim"]) == 4
    assert _floats(pair.attrib["friction"]) == pytest.approx(
        (0.85, 0.85, 0.03, 0.001, 0.001)
    )
    assert sensors["brush_touch_sensor"].attrib["site"] == "brush_contact_zone"
    assert "tip" in sites

    canvas = geoms["canvas_surface"]
    assert canvas.attrib["type"] == "box"
    assert _floats(canvas.attrib["size"]) == pytest.approx((0.254, 0.00635, 0.254))


def test_camera_housing_envelopes_share_optical_frames_and_stay_behind_lenses() -> None:
    root = _root()
    worldbody = root.find("./worldbody")
    assert worldbody is not None
    defaults = {
        element.attrib["class"]: element
        for element in root.findall("./default/default")
    }
    camera_visual = defaults["camera_visual"].find("./geom")
    assert camera_visual is not None
    assert camera_visual.attrib["contype"] == "0"
    assert camera_visual.attrib["conaffinity"] == "0"
    assert camera_visual.attrib["group"] == "2"

    cameras = _named_elements(root, "./worldbody/camera")
    bodies = _named_elements(root, "./worldbody/body")
    for camera_name in (
        "canvas_right_oblique",
        "canvas_left_oblique",
        "canvas_inspection_deployed",
        "brush_standoff_overhead",
    ):
        housing_name = {
            "canvas_right_oblique": "canvas_right_camera_housing",
            "canvas_left_oblique": "canvas_left_camera_housing",
            "canvas_inspection_deployed": "canvas_inspection_camera_housing",
            "brush_standoff_overhead": "brush_standoff_camera_housing",
        }[camera_name]
        camera = cameras[camera_name]
        housing = bodies[housing_name]
        assert _floats(housing.attrib["pos"]) == pytest.approx(
            _floats(camera.attrib["pos"])
        )
        assert _floats(housing.attrib["xyaxes"]) == pytest.approx(
            _floats(camera.attrib["xyaxes"])
        )
        geoms = list(housing.findall("./geom"))
        assert len(geoms) == 3
        assert all(geom.attrib["class"] == "camera_visual" for geom in geoms)

        body_geom = next(geom for geom in geoms if geom.attrib["type"] == "box")
        body_position = _floats(body_geom.attrib["pos"])
        body_half_size = _floats(body_geom.attrib["size"])
        assert body_position[2] - body_half_size[2] > 0.0

        axial_geoms = [
            geom for geom in geoms if geom.attrib["type"] == "cylinder"
        ]
        assert all(
            min(_floats(geom.attrib["fromto"])[2::3]) > 0.0
            for geom in axial_geoms
        )


def test_mujoco_compiles_with_stable_adapter_names_and_separated_anchors() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    assert model.nq == 7
    assert model.nv == 7
    assert model.nu == 4
    assert model.pair_dim.tolist() == [4]
    assert model.pair_friction[0] == pytest.approx(
        (0.85, 0.85, 0.03, 0.001, 0.001)
    )
    assert model.nkey == 5
    assert [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
        for index in range(model.ncam)
    ] == [
        "overview",
        "canvas_view",
        "top_view",
            "canvas_right_oblique",
            "canvas_left_oblique",
            "canvas_inspection_deployed",
            "brush_standoff_overhead",
        ]
    assert [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    ] == [
        *JOINT_NAMES,
        "brush_bend_x",
        "brush_bend_z",
        "brush_compression",
    ]

    for name in (
        "tip",
        "brush_bend_x_sensor",
        "brush_bend_z_sensor",
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
    data.qpos[:] = (yaw, pitch, roll, elbow, 0.0, 0.0, 0.0)
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
            data.qpos[:] = (yaw, pitch, 0.0, elbow, 0.0, 0.0, 0.0)
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

    gravity_bias = dict(
        zip(
            (*JOINT_NAMES, "brush_bend_x", "brush_bend_z", "brush_compression"),
            data.qfrc_bias,
            strict=True,
        )
    )
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
