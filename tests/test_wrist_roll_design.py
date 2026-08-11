from __future__ import annotations

import math
from xml.etree import ElementTree

import numpy as np
import pytest

from active_painter.wrist_roll_design import (
    WRIST_ROLL_EXPLORATION_STATUS,
    WRIST_ROLL_EXPLORATION_VERSION,
    AngledWristRollSpec,
    build_wrist_roll_exploration_model,
    build_wrist_roll_exploration_xml,
    compare_wrist_roll_design,
)
from active_painter.wrist_roll_viewer import demo_targets


def test_wrist_roll_branch_is_explicitly_noncanonical_and_relocates_roll() -> None:
    root = ElementTree.fromstring(build_wrist_roll_exploration_xml())
    text = {
        item.get("name"): item.get("data")
        for item in root.findall("./custom/text")
    }
    roll = root.find(".//joint[@name='roll']")
    wrist = root.find(".//body[@name='wrist_roll_output']")
    upper_arm = root.find(".//body[@name='upper_arm_structure']")

    assert root.get("model") == "active_inference_painter_angled_wrist_roll_exploration"
    assert text["model_version"] == WRIST_ROLL_EXPLORATION_VERSION
    assert text["design_status"] == WRIST_ROLL_EXPLORATION_STATUS
    assert wrist is not None
    assert roll is wrist.find("./joint[@name='roll']")
    assert upper_arm is not None
    assert upper_arm.find("./joint[@name='roll']") is None
    assert wrist.find("./body[@name='angled_brush_mount']") is not None


def test_wrist_roll_branch_compiles_with_stable_joint_actuator_and_sensor_names() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = build_wrist_roll_exploration_model()

    for name in ("yaw", "pitch", "roll", "elbow"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_position") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{name}_position_sensor") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "wrist_roll_axis") >= 0


def test_angled_wrist_roll_has_local_tip_orbit_and_orientation_consequence() -> None:
    pytest.importorskip("mujoco")
    result = compare_wrist_roll_design(
        AngledWristRollSpec(brush_angle_deg=15.0),
        sweep_degrees=32.0,
    )

    expected_radius_mm = 1000.0 * (0.3302 - 0.2472) * math.sin(math.radians(15.0))
    assert result.wrist_tip_orbit_radius_mm == pytest.approx(expected_radius_mm, rel=0.03)
    assert result.wrist_tip_path_length_mm > 10.0
    assert result.wrist_tip_chord_mm > 10.0
    assert result.wrist_brush_direction_change_deg > 10.0
    assert result.wrist_canvas_reachability >= 0.80
    assert result.wrist_median_tip_residual_mm <= 2.0
    assert np.isfinite(result.elbow_gravity_bias_increase_nm)


def test_brush_angle_controls_wrist_orbit_radius() -> None:
    pytest.importorskip("mujoco")
    shallow = compare_wrist_roll_design(AngledWristRollSpec(brush_angle_deg=8.0))
    steep = compare_wrist_roll_design(AngledWristRollSpec(brush_angle_deg=22.0))

    assert steep.wrist_tip_orbit_radius_mm > 2.5 * shallow.wrist_tip_orbit_radius_mm
    assert steep.wrist_brush_direction_change_deg > shallow.wrist_brush_direction_change_deg


def test_wrist_roll_viewer_demo_sweeps_symmetrically() -> None:
    positive = demo_targets(1.25)
    negative = demo_targets(3.75)

    assert math.degrees(positive["roll"]) == pytest.approx(32.0)
    assert math.degrees(negative["roll"]) == pytest.approx(-32.0)
