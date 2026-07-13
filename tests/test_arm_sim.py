import pytest
import numpy as np

from active_painter.arm_control import ik_pose_for_canvas_point
from active_painter.arm_sim import ArmPainterSim, ArmPose, JointPlant, VerticalCanvas, clip_scalar, safe_home_pose
from active_painter.config import PainterConfig


def test_arm_kinematics_home_reaches_forward() -> None:
    sim = ArmPainterSim(PainterConfig())
    tip = sim.kinematics.tip(ArmPose())
    assert np.allclose(tip, np.asarray([0.0, 26.0, 0.0]))


def test_scalar_clip_matches_bounded_semantics_and_preserves_nan() -> None:
    assert clip_scalar(-2.0, -1.0, 1.0) == -1.0
    assert clip_scalar(0.25, -1.0, 1.0) == 0.25
    assert clip_scalar(2.0, -1.0, 1.0) == 1.0
    assert np.isnan(clip_scalar(float("nan"), -1.0, 1.0))


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


def test_vertical_canvas_default_paint_deposition_is_more_opaque_than_old_rate() -> None:
    old_cfg = PainterConfig(
        canvas_size=32,
        paint_deposition_base_rate=0.055,
        paint_deposition_pressure_rate=0.22,
    )
    opaque_cfg = PainterConfig(canvas_size=32)
    old_canvas = VerticalCanvas(old_cfg)
    opaque_canvas = VerticalCanvas(opaque_cfg)
    point = np.asarray([0.0, opaque_canvas.distance, 0.0])

    old_canvas.paint_at(point, pressure=0.55, tone=1.0, dt=1.0 / 240.0)
    opaque_canvas.paint_at(point, pressure=0.55, tone=1.0, dt=1.0 / 240.0)

    assert opaque_canvas.material_coverage() > old_canvas.material_coverage() * 2.0
    assert opaque_canvas.thickness.max() > old_canvas.thickness.max() * 2.0


def test_vertical_canvas_paint_deposition_rates_are_configurable() -> None:
    low = VerticalCanvas(PainterConfig(canvas_size=32, paint_deposition_base_rate=0.02, paint_deposition_pressure_rate=0.04))
    high = VerticalCanvas(PainterConfig(canvas_size=32, paint_deposition_base_rate=0.20, paint_deposition_pressure_rate=0.80))
    point = np.asarray([0.0, high.distance, 0.0])

    low.paint_at(point, pressure=0.5, tone=1.0, dt=1.0 / 240.0)
    high.paint_at(point, pressure=0.5, tone=1.0, dt=1.0 / 240.0)

    assert high.thickness.max() > low.thickness.max() * 8.0


def test_oil_white_over_wet_black_is_surface_opaque_with_some_pickup() -> None:
    canvas = VerticalCanvas(PainterConfig(canvas_size=48))
    point = np.asarray([0.0, canvas.distance, 0.0])

    canvas.paint_at(point, pressure=0.8, tone=1.0, dt=1.0 / 120.0)
    after_black_tone = canvas.visible_tone().copy()
    black_mass_before_white = canvas.black_mass.copy()
    canvas.paint_at(point, pressure=0.8, tone=0.0, dt=1.0 / 120.0)
    after_white_tone = canvas.visible_tone()
    u, v = canvas.world_to_pixel(0.0, 0.0)
    row = int(round(v))
    col = int(round(u))

    assert after_black_tone[row, col] > 0.7
    assert 0.02 < after_white_tone[row, col] < 0.35
    assert np.array_equal(canvas.black_mass, black_mass_before_white)


def test_oil_surface_opacity_controls_white_over_black_dominance() -> None:
    point = np.asarray([0.0, 17.0, 0.0])
    transparent = VerticalCanvas(
        PainterConfig(
            canvas_size=48,
            oil_surface_opacity_thickness=0.02,
            oil_wet_pickup_fraction=0.18,
        )
    )
    opaque = VerticalCanvas(
        PainterConfig(
            canvas_size=48,
            oil_surface_opacity_thickness=0.001,
            oil_wet_pickup_fraction=0.18,
        )
    )

    for canvas in (transparent, opaque):
        canvas.paint_at(point, pressure=0.8, tone=1.0, dt=1.0 / 120.0)
        canvas.paint_at(point, pressure=0.8, tone=0.0, dt=1.0 / 120.0)
    u, v = opaque.world_to_pixel(0.0, 0.0)
    row = int(round(v))
    col = int(round(u))

    assert opaque.visible_tone()[row, col] < transparent.visible_tone()[row, col]


def test_oil_paint_wetness_persists_while_brush_is_lifted() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    point = np.asarray([0.0, sim.canvas.distance, 0.0])
    sim.canvas.paint_at(point, pressure=0.8, tone=1.0, dt=1.0 / 120.0)
    wetness_after_paint = sim.canvas.wetness.copy()

    sim.paint_enabled = False
    for _ in range(480):
        sim.step(1.0 / 240.0)

    assert np.array_equal(sim.canvas.wetness, wetness_after_paint)


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
    assert abs(plant.telemetry.position_error_deg["yaw"] - 0.5) <= 3.0 * plant.telemetry.encoder_std_deg["yaw"]
    assert abs(plant.telemetry.current["yaw"]) > 0.0
    assert abs(plant.telemetry.backlash_deflection_deg["yaw"]) > 0.0


def test_joint_plant_process_noise_is_seeded_and_changes_realized_trajectory() -> None:
    plants = [JointPlant(rng_seed=seed) for seed in (11, 11, 12)]
    poses = [safe_home_pose() for _ in plants]
    target = ArmPose(yaw=18.0, pitch=-42.0, roll=7.0, elbow=92.0)

    for _ in range(80):
        poses = [plant.step(pose, target, 1.0 / 120.0) for plant, pose in zip(plants, poses)]

    assert poses[0] == poses[1]
    assert poses[0] != poses[2]


def test_joint_plant_mass_matrix_couples_pitch_drive_into_elbow_motion() -> None:
    common = {
        "process_noise_enabled": False,
        "encoder_noise_enabled": False,
        "gravity_compensation_fraction": 1.0,
    }
    coupled = JointPlant(**common)
    uncoupled = JointPlant(**common, pitch_elbow_coupling_inertia=0.0)
    actual = safe_home_pose()
    target = ArmPose(yaw=actual.yaw, pitch=actual.pitch + 15.0, roll=actual.roll, elbow=actual.elbow)

    coupled_after = coupled.step(actual, target, 1.0 / 120.0)
    uncoupled_after = uncoupled.step(actual, target, 1.0 / 120.0)

    assert abs(coupled_after.elbow - actual.elbow) > abs(uncoupled_after.elbow - actual.elbow)
    assert coupled.telemetry.gravity_torque["pitch"] == pytest.approx(0.0)


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
    assert abs(plant.telemetry.encoder_angle_deg["yaw"] - pose.yaw) <= 3.0 * plant.telemetry.encoder_std_deg["yaw"]


def test_joint_plant_damping_multiplier_reduces_hold_ringing() -> None:
    actual = safe_home_pose()
    target = ArmPose(
        yaw=actual.yaw + 20.0,
        pitch=actual.pitch - 10.0,
        roll=actual.roll,
        elbow=actual.elbow - 15.0,
    )
    low = JointPlant()
    high = JointPlant()
    low.reset_state(actual)
    high.reset_state(actual)
    for plant in (low, high):
        plant.velocity["yaw"] = 1.5
        plant.velocity["pitch"] = -1.0
        plant.velocity["elbow"] = 1.2
    low_pose = actual
    high_pose = actual

    for _ in range(40):
        low_pose = low.step(low_pose, target, 1.0 / 240.0, damping_multiplier=1.0)
        high_pose = high.step(high_pose, target, 1.0 / 240.0, damping_multiplier=2.5)

    assert abs(high_pose.yaw - target.yaw) < abs(low_pose.yaw - target.yaw)
    assert abs(high.velocity["yaw"]) < abs(low.velocity["yaw"])


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


def test_arm_safety_preserves_safe_retract_target_after_overtravel_rollback() -> None:
    sim = ArmPainterSim(PainterConfig())
    unsafe_target = ArmPose()
    sim.set_target(unsafe_target)
    sim.step(1.0 / 120.0)

    safe_target = ik_pose_for_canvas_point(0.0, 0.0, sim.canvas.distance - 4.0)
    sim.set_target(safe_target)
    sim.step(1.0 / 120.0)

    assert sim.target_pose == safe_target


def test_arm_safety_allows_already_overtravel_pose_to_escape_outward() -> None:
    sim = ArmPainterSim(PainterConfig())
    sim.actual_pose = ArmPose()
    sim.plant.reset_state(sim.actual_pose)
    sim.contact = sim.canvas.contact_from_tip(sim.kinematics.tip(sim.actual_pose), 0.0)
    safe_target = ik_pose_for_canvas_point(0.0, 0.0, sim.canvas.distance - 4.0)
    sim.set_target(safe_target)
    initial_y = float(sim.kinematics.tip(sim.actual_pose)[1])

    for _ in range(80):
        sim.step(1.0 / 240.0)

    assert sim.target_pose == safe_target
    assert float(sim.kinematics.tip(sim.actual_pose)[1]) < initial_y


def test_render_points_clamp_contact_tip_to_canvas_face() -> None:
    sim = ArmPainterSim(PainterConfig())
    tip = sim.kinematics.tip(sim.actual_pose).copy()
    tip[1] = sim.canvas.distance + 0.2
    sim.contact = sim.canvas.contact_from_tip(tip)
    render_points = sim.render_points()
    assert render_points[-1, 1] == sim.canvas.distance
