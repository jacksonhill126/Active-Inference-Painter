import pytest
import numpy as np

from active_painter.arm_control import ik_pose_for_canvas_point
from active_painter.arm_sim import Brush, ArmPainterSim, ArmPose, JointPlant, VerticalCanvas, clip_scalar, safe_home_pose
from active_painter.config import PainterConfig


def test_arm_kinematics_home_reaches_forward() -> None:
    sim = ArmPainterSim(PainterConfig())
    tip = sim.kinematics.tip(ArmPose())
    assert np.allclose(tip, np.asarray([0.0, 26.0, 0.0]))


def test_upper_arm_roll_rotates_elbow_hinge_without_moving_upper_arm() -> None:
    sim = ArmPainterSim(PainterConfig())
    negative = ArmPose(yaw=14.0, pitch=-38.0, roll=-32.0, elbow=92.0)
    positive = ArmPose(yaw=14.0, pitch=-38.0, roll=32.0, elbow=92.0)

    negative_points = sim.kinematics.joint_points(negative)
    positive_points = sim.kinematics.joint_points(positive)
    upper_axis = sim.kinematics.upper_arm_axis(positive)
    negative_hinge = sim.kinematics.elbow_hinge_axis(negative)
    positive_hinge = sim.kinematics.elbow_hinge_axis(positive)

    assert np.allclose(negative_points[1], positive_points[1])
    assert not np.allclose(negative_points[2], positive_points[2])
    assert np.linalg.norm(upper_axis) == pytest.approx(1.0)
    assert np.dot(upper_axis, negative_hinge) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(upper_axis, positive_hinge) == pytest.approx(0.0, abs=1e-12)
    assert not np.allclose(negative_hinge, positive_hinge)


@pytest.mark.parametrize("roll_deg", [-32.0, 0.0, 32.0])
def test_fixed_roll_ik_reaches_the_same_canvas_target(roll_deg: float) -> None:
    sim = ArmPainterSim(PainterConfig())
    target = np.asarray([4.0, 17.0, -3.0])

    pose = ik_pose_for_canvas_point(
        float(target[0]),
        float(target[2]),
        float(target[1]),
        upper_arm_roll_deg=roll_deg,
    )

    assert pose.roll == pytest.approx(roll_deg)
    assert sim.kinematics.tip(pose) == pytest.approx(target, abs=1e-10)


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


def test_vertical_canvas_layering_does_not_increase_covered_area() -> None:
    cfg = PainterConfig(canvas_size=16)
    canvas = VerticalCanvas(cfg)
    canvas.thickness[4:12, 3:9] = 2.0 * cfg.paint_presence_threshold

    first_coat_coverage = canvas.material_coverage()
    canvas.thickness[4:12, 3:9] *= 20.0

    assert canvas.material_coverage() == first_coat_coverage
    assert first_coat_coverage == pytest.approx(48.0 / 256.0)


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

    assert opaque_canvas.surface_opacity_field().mean() > old_canvas.surface_opacity_field().mean() * 2.0
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


def _drag(canvas: VerticalCanvas, brush: Brush | None, x0: float, x1: float, z: float,
          pressure: float, tone: float, steps: int, dt: float = 1.0 / 90.0) -> None:
    """Deposit a straight horizontal drag as swept stamps, mirroring how
    ArmPainterSim.step feeds motion + brush into paint_at."""
    prev = None
    for x in np.linspace(x0, x1, steps):
        point = np.asarray([x, canvas.distance, z])
        motion = None if (brush is None or prev is None) else (point - prev)
        canvas.paint_at(point, pressure, tone, dt, motion=motion, brush=brush)
        prev = point


def test_paint_at_legacy_call_is_isotropic_with_unit_deposition() -> None:
    # The bare 4-arg call (used by other tests and any legacy caller) must stay
    # an isotropic disc that never runs out.
    cfg = PainterConfig(canvas_size=64)
    canvas = VerticalCanvas(cfg)
    point = np.asarray([0.0, canvas.distance, 0.0])
    for _ in range(200):
        canvas.paint_at(point, pressure=0.7, tone=1.0, dt=1.0 / 90.0)
    mask = canvas.thickness > 0
    x_span = np.ptp(np.nonzero(mask.any(axis=0))[0])
    y_span = np.ptp(np.nonzero(mask.any(axis=1))[0])
    assert abs(x_span - y_span) <= 1  # isotropic
    # Never runs out: the 200th stamp still deposits as much as the first.
    single = VerticalCanvas(cfg)
    single.paint_at(point, pressure=0.7, tone=1.0, dt=1.0 / 90.0)
    assert canvas.thickness.max() > 100.0 * single.thickness.max()


def test_brush_oil_does_not_dry_thickness_holds_along_a_long_stroke() -> None:
    # Oil: a loaded brush lays paint at a consistent rate; the mark must not
    # fade/thin from start to end of even a long stroke.
    cfg = PainterConfig(canvas_size=96)
    canvas = VerticalCanvas(cfg)
    brush = Brush(cfg, np.random.default_rng(0))
    brush.reload(amount=0.5, tone=1.0)
    _drag(canvas, brush, x0=-8.0, x1=8.0, z=0.0, pressure=0.6, tone=1.0, steps=160)
    row = int(round(canvas.world_to_pixel(0.0, 0.0)[1]))
    profile = canvas.thickness[row]
    nz = np.nonzero(profile)[0]
    start = profile[nz[0] + len(nz) // 6:nz[0] + len(nz) // 3].mean()
    end = profile[nz[-1] - len(nz) // 3:nz[-1] - len(nz) // 6].mean()
    assert end > 0.7 * start  # no drying out toward the end


def test_brush_loading_from_amount_scales_deposited_thickness() -> None:
    # `amount` sets brush loading -> more paint means thicker, more opaque
    # deposition, uniformly (not faster depletion).
    cfg = PainterConfig(canvas_size=64)
    thin_canvas = VerticalCanvas(cfg)
    thick_canvas = VerticalCanvas(cfg)
    point = np.asarray([0.0, thin_canvas.distance, 0.0])
    light = Brush(cfg, np.random.default_rng(0)); light.reload(amount=0.1, tone=1.0)
    heavy = Brush(cfg, np.random.default_rng(0)); heavy.reload(amount=1.0, tone=1.0)
    thin_canvas.paint_at(point, 0.6, 1.0, 1.0 / 90.0, brush=light)
    thick_canvas.paint_at(point, 0.6, 1.0, 1.0 / 90.0, brush=heavy)
    assert thick_canvas.thickness.max() > 1.5 * thin_canvas.thickness.max()


def test_brush_travel_direction_elongates_the_footprint() -> None:
    cfg = PainterConfig(canvas_size=96)
    canvas = VerticalCanvas(cfg)
    brush = Brush(cfg, np.random.default_rng(0))
    brush.reload(amount=1.0, tone=1.0)
    point = np.asarray([0.0, canvas.distance, 0.0])
    canvas.paint_at(point, 0.7, 1.0, 1.0 / 90.0, motion=np.asarray([4.0, 0.0, 0.0]), brush=brush)
    mask = canvas.thickness > 0
    x_span = np.ptp(np.nonzero(mask.any(axis=0))[0])
    y_span = np.ptp(np.nonzero(mask.any(axis=1))[0])
    assert x_span > y_span + 3  # swept along the travel (x) axis


def test_brush_bristles_streak_across_the_stroke_width() -> None:
    cfg = PainterConfig(canvas_size=96)
    canvas = VerticalCanvas(cfg)
    brush = Brush(cfg, np.random.default_rng(3))
    brush.reload(amount=1.0, tone=1.0)
    _drag(canvas, brush, x0=-6.0, x1=6.0, z=0.0, pressure=0.6, tone=1.0, steps=60)
    mid_col = int(round(canvas.world_to_pixel(0.0, 0.0)[0]))
    cross = canvas.thickness[:, mid_col]
    cross = cross[cross > 0]
    # A flat (bristle-free) disc drag would be near-uniform across its width.
    assert cross.std() / cross.mean() > 0.02


def test_brush_flow_taper_narrows_the_mark_toward_stroke_ends() -> None:
    # A low flow (stroke end) must lay a narrower mark than full flow (middle).
    cfg = PainterConfig(canvas_size=96)
    point = np.asarray([0.0, 17.0, 0.0])
    full = VerticalCanvas(cfg)
    tip = VerticalCanvas(cfg)
    bf = Brush(cfg, np.random.default_rng(0)); bf.reload(1.0, 1.0)
    bt = Brush(cfg, np.random.default_rng(0)); bt.reload(1.0, 1.0)
    full.paint_at(point, 0.7, 1.0, 1.0 / 90.0, brush=bf, flow=1.0)
    tip.paint_at(point, 0.7, 1.0, 1.0 / 90.0, brush=bt, flow=0.1)
    full_w = np.ptp(np.nonzero((full.thickness > 0).any(axis=1))[0]) if full.thickness.any() else 0
    tip_w = np.ptp(np.nonzero((tip.thickness > 0).any(axis=1))[0]) if tip.thickness.any() else 0
    assert tip_w < full_w


def test_canvas_grain_lets_light_pressure_leave_bare_tooth() -> None:
    # Under grain at full strength (the default), a light-pressure stroke must
    # NOT fully cover its footprint: unreached valleys stay genuinely bare.
    cfg = PainterConfig(canvas_size=96)
    assert cfg.canvas_grain_strength == 1.0
    grained = VerticalCanvas(cfg)
    b = Brush(cfg, np.random.default_rng(0)); b.reload(1.0, 1.0)
    prev = None
    for x in np.linspace(-3.0, 3.0, 60):
        p = np.asarray([x, 17.0, 0.0])
        m = None if prev is None else p - prev
        grained.paint_at(p, 0.2, 1.0, 1.0 / 90.0, motion=m, brush=b)  # light pressure
        prev = p
    # Within the mark's bounding box there are both painted and bare cells.
    rows = np.nonzero((grained.thickness > 0).any(axis=1))[0]
    cols = np.nonzero((grained.thickness > 0).any(axis=0))[0]
    box = grained.thickness[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    painted_fraction = float((box > 0).mean())
    assert 0.05 < painted_fraction < 0.95  # textured, not solid


def test_bristle_furrows_do_not_split_a_stroke_end_to_end() -> None:
    # Dry-hair gaps must open and close along the path: no lane of the mark may
    # stay unpainted for the whole stroke length (that reads as a split stroke).
    cfg = PainterConfig(canvas_size=96)
    canvas = VerticalCanvas(cfg)
    brush = Brush(cfg, np.random.default_rng(5))
    brush.reload(amount=1.0, tone=1.0)
    _drag(canvas, brush, x0=-8.0, x1=8.0, z=0.0, pressure=0.6, tone=1.0, steps=200)
    mask = canvas.thickness > 0
    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    # middle 60% of the stroke, interior lanes only
    c_lo = cols.min() + int(0.2 * len(cols))
    c_hi = cols.max() - int(0.2 * len(cols))
    interior = mask[rows.min() + 1:rows.max(), c_lo:c_hi]
    lane_has_paint = interior.any(axis=1)
    assert lane_has_paint.all()


def test_brush_dab_is_not_a_perfect_circle() -> None:
    # The per-stroke edge wobble must make even a stationary dab irregular
    # (tested at the web canvas resolution, where the brush spans real pixels).
    cfg = PainterConfig(canvas_size=256)
    from dataclasses import replace as dc_replace
    round_cfg = dc_replace(cfg, brush_edge_wobble=0.0)
    point = np.asarray([0.0, 17.0, 0.0])
    wobbled = VerticalCanvas(cfg)
    perfect = VerticalCanvas(round_cfg)
    bw = Brush(cfg, np.random.default_rng(3)); bw.reload(1.0, 1.0)
    bp = Brush(round_cfg, np.random.default_rng(3)); bp.reload(1.0, 1.0)
    for _ in range(30):
        wobbled.paint_at(point, 0.8, 1.0, 1.0 / 90.0, brush=bw)
        perfect.paint_at(point, 0.8, 1.0, 1.0 / 90.0, brush=bp)
    assert not np.array_equal(wobbled.thickness > 0, perfect.thickness > 0)


def test_stroke_sampler_never_collapses_edge_strokes_into_dabs() -> None:
    # Clipping endpoints used to fold edge strokes onto their start point; the
    # sampler must now preserve the sampled length by shifting inward.
    from active_painter.policies import PolicySampler
    cfg = PainterConfig()
    sampler = PolicySampler(cfg, seed=0)
    for _ in range(500):
        action = sampler._stroke()
        length = float(np.hypot(action.x1 - action.x0, action.y1 - action.y0))
        assert length > 0.19


def test_brush_pickup_conserves_pigment_between_canvas_and_brush() -> None:
    # The dirty-brush transfer moves pigment between the canvas ledger and the
    # held reservoir; a white brush adds volume but zero black, so total black
    # (canvas + brush head) must be exactly conserved through the drag.
    cfg = PainterConfig(canvas_size=96)
    canvas = VerticalCanvas(cfg)
    black = Brush(cfg, np.random.default_rng(1))
    black.reload(amount=1.0, tone=1.0)
    _drag(canvas, black, x0=-6.0, x1=-1.0, z=0.0, pressure=0.8, tone=1.0, steps=60)
    # Paint still held by the black brush leaves the system at pen-up (the
    # per-stroke reset is a brush clean), so conservation is over the white
    # drag alone: canvas black before = canvas black after + white brush head.
    canvas_black_before = float(canvas.black_mass.sum())
    white = Brush(cfg, np.random.default_rng(2))
    white.reload(amount=1.0, tone=0.0)
    _drag(canvas, white, x0=-5.0, x1=6.0, z=0.0, pressure=0.7, tone=0.0, steps=80)
    canvas_black_after = float(canvas.black_mass.sum())
    assert canvas_black_after + white.held_black == pytest.approx(canvas_black_before, rel=1e-5)


def test_brush_drags_picked_up_paint_beyond_the_wet_patch() -> None:
    # Blending is transport, not tinting: black picked up inside the patch must
    # be redeposited past the patch's original right edge.
    cfg = PainterConfig(canvas_size=96)
    canvas = VerticalCanvas(cfg)
    black = Brush(cfg, np.random.default_rng(1))
    black.reload(amount=1.0, tone=1.0)
    _drag(canvas, black, x0=-6.0, x1=-2.0, z=0.0, pressure=0.8, tone=1.0, steps=60)
    edge_col = int(round(canvas.world_to_pixel(-2.0, 0.0)[0]))
    beyond = np.s_[:, edge_col + 8:]
    black_beyond_before = float(canvas.black_mass[beyond].sum())
    white = Brush(cfg, np.random.default_rng(2))
    white.reload(amount=1.0, tone=0.0)
    _drag(canvas, white, x0=-5.0, x1=6.0, z=0.0, pressure=0.7, tone=0.0, steps=80)
    black_beyond_after = float(canvas.black_mass[beyond].sum())
    assert black_beyond_after > black_beyond_before + 1e-6


def test_brush_wet_smear_bleeds_carried_tone_along_the_stroke() -> None:
    cfg = PainterConfig(canvas_size=96)
    canvas = VerticalCanvas(cfg)
    black = Brush(cfg, np.random.default_rng(1))
    black.reload(amount=1.0, tone=1.0)
    _drag(canvas, black, x0=-7.0, x1=-3.0, z=0.0, pressure=0.8, tone=1.0, steps=40)
    white = Brush(cfg, np.random.default_rng(2))
    white.reload(amount=1.0, tone=0.0)
    assert white.carried_tone == 0.0
    _drag(canvas, white, x0=-6.0, x1=6.0, z=0.0, pressure=0.7, tone=0.0, steps=60)
    assert white.carried_tone > 0.0  # picked up wet black it dragged through


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
