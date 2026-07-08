import pytest
import numpy as np

from active_painter.arm_sim import ArmPainterSim, ArmPose, JointPlant, VerticalCanvas, safe_home_pose
from active_painter.config import PainterConfig


def test_arm_kinematics_home_reaches_forward() -> None:
    sim = ArmPainterSim(PainterConfig())
    tip = sim.kinematics.tip(ArmPose())
    assert np.allclose(tip, np.asarray([0.0, 26.0, 0.0]))


def test_sim_starts_bent_near_seventeen_inch_canvas_without_penetration() -> None:
    sim = ArmPainterSim(PainterConfig())
    tip = sim.kinematics.tip(sim.actual_pose)
    assert sim.canvas.distance == 17.0
    assert 16.0 < tip[1] < sim.canvas.distance
    assert abs(tip[2]) < 0.5


def test_vertical_canvas_contact_pressure_increases_with_penetration() -> None:
    canvas = VerticalCanvas(PainterConfig(canvas_size=24))
    light = canvas.contact_from_tip(np.asarray([0.0, canvas.distance + 0.03, 0.0]))
    deeper = canvas.contact_from_tip(np.asarray([0.0, canvas.distance + 0.12, 0.0]))
    assert light.on_canvas
    assert deeper.pressure > light.pressure
    assert deeper.force > light.force
    assert deeper.brush_width_px > light.brush_width_px


def test_vertical_canvas_white_paint_increases_material_coverage() -> None:
    canvas = VerticalCanvas(PainterConfig(canvas_size=32))
    before = canvas.material_coverage()
    canvas.paint_at(np.asarray([0.0, canvas.distance, 0.0]), pressure=0.8, tone=0.0, dt=0.1)
    after = canvas.material_coverage()
    assert after > before
    assert canvas.visible_tone().mean() < 0.01


def test_vertical_canvas_observed_tone_composites_paint_against_gray_ground() -> None:
    canvas = VerticalCanvas(PainterConfig(canvas_size=32, canvas_ground_tone=0.34))
    blank_tone = canvas.observed_tone()
    assert np.allclose(blank_tone, 0.34)

    canvas.paint_at(np.asarray([-3.0, canvas.distance, 0.0]), pressure=0.8, tone=0.0, dt=0.2)
    after_white = canvas.observed_tone()
    assert after_white.min() < 0.34
    assert canvas.ground_contrast_field().max() > 0.0

    canvas.paint_at(np.asarray([3.0, canvas.distance, 0.0]), pressure=0.8, tone=1.0, dt=0.2)
    after_black = canvas.observed_tone()
    assert after_black.max() > 0.34


def test_joint_plant_moves_actual_pose_toward_target_without_selecting_policy() -> None:
    sim = ArmPainterSim(PainterConfig())
    sim.set_target(ArmPose(yaw=35.0, pitch=-20.0, roll=10.0, elbow=70.0))
    for _ in range(80):
        sim.step(1.0 / 120.0)
    assert sim.actual_pose.yaw > 0.0
    assert sim.actual_pose.elbow > 0.0


def test_joint_plant_backlash_deadband_delays_small_link_motion() -> None:
    plant = JointPlant(backlash_deadband_deg=dict.fromkeys(("yaw", "pitch", "roll", "elbow"), 2.0))
    actual = safe_home_pose()
    target = ArmPose(yaw=actual.yaw + 0.5, pitch=actual.pitch, roll=actual.roll, elbow=actual.elbow)

    after = plant.step(actual, target, 1.0 / 120.0)

    assert abs(after.yaw - actual.yaw) < 0.05
    assert plant.telemetry.position_error_deg["yaw"] == pytest.approx(0.5)
    assert abs(plant.telemetry.current["yaw"]) > 0.0
    assert abs(plant.telemetry.backlash_deflection_deg["yaw"]) > 0.0


def test_encoded_servo_hold_brakes_residual_velocity_without_drifting() -> None:
    plant = JointPlant()
    actual = safe_home_pose()
    plant.reset_state(actual)
    plant.velocity["yaw"] = 1.2
    target = actual

    pose = actual
    for _ in range(480):
        pose = plant.step(pose, target, 1.0 / 240.0)

    assert abs(pose.yaw - actual.yaw) < 0.5
    assert abs(plant.velocity["yaw"]) < 0.05
    assert plant.telemetry.encoder_angle_deg["yaw"] == pytest.approx(pose.yaw)


def test_joint_plant_contact_load_increases_encoder_uncertainty_and_load_torque() -> None:
    unloaded = JointPlant()
    loaded = JointPlant()
    actual = safe_home_pose()
    target = ArmPose(yaw=actual.yaw + 25.0, pitch=actual.pitch, roll=actual.roll, elbow=actual.elbow)

    unloaded.step(actual, target, 1.0 / 60.0, contact_force=0.0)
    loaded.step(actual, target, 1.0 / 60.0, contact_force=18.0)

    assert abs(loaded.telemetry.load_torque["yaw"]) > abs(unloaded.telemetry.load_torque["yaw"])
    assert loaded.telemetry.encoder_std_deg["yaw"] > unloaded.telemetry.encoder_std_deg["yaw"]


def test_reset_pose_realigns_motor_and_link_state() -> None:
    sim = ArmPainterSim(PainterConfig())
    sim.set_target(ArmPose(yaw=20.0, pitch=-30.0, roll=15.0, elbow=80.0))
    for _ in range(10):
        sim.step(1.0 / 120.0)

    sim.reset_pose()

    home = safe_home_pose()
    assert sim.actual_pose == home
    for name, radians in home.radians().items():
        assert sim.plant.motor_angle[name] == pytest.approx(radians)
        assert sim.plant.motor_velocity[name] == pytest.approx(0.0)
        assert sim.plant.velocity[name] == pytest.approx(0.0)


def test_arm_safety_rolls_back_canvas_overtravel() -> None:
    sim = ArmPainterSim(PainterConfig())
    sim.set_target(ArmPose())
    for _ in range(240):
        sim.step(1.0 / 120.0)
        tip = sim.kinematics.tip(sim.actual_pose)
        if sim.canvas.contains(float(tip[0]), float(tip[2])):
            assert tip[1] <= sim.canvas.distance + sim.canvas.bushing_travel + 1e-6


def test_render_points_clamp_contact_tip_to_canvas_face() -> None:
    sim = ArmPainterSim(PainterConfig())
    tip = sim.kinematics.tip(sim.actual_pose).copy()
    tip[1] = sim.canvas.distance + 0.2
    sim.contact = sim.canvas.contact_from_tip(tip)
    render_points = sim.render_points()
    assert render_points[-1, 1] == sim.canvas.distance
