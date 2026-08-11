from __future__ import annotations

import math

import numpy as np
import pytest

from active_painter.arm_agent_driver import canvas_summary_state
from active_painter.arm_sim import ArmPainterSim
from active_painter.config import PainterConfig
from active_painter.env import (
    StrokeAction,
    bounded_mark_curvature,
    mark_path_length,
    mark_path_points,
)
from active_painter.motor_planning import (
    motor_realization_policy_alternatives,
)
from active_painter.policies import Policy, PolicySampler
from active_painter.proposal import hand_written_log_density
from active_painter.spatial_state import rasterize_stroke_action
from active_painter.stroke_execution import (
    adaptive_stroke_timing,
    controller_for_motor_primitive,
    forecast_stroke_execution,
    stroke_reference,
)


def _curved_vertical(curvature: float = 0.20) -> StrokeAction:
    return StrokeAction(
        0.50,
        0.80,
        0.50,
        0.20,
        0.06,
        0.55,
        1.0,
        curvature=curvature,
    )


def test_quadratic_mark_curvature_is_signed_midpoint_deflection() -> None:
    action = _curved_vertical(0.20)
    start, midpoint, end = mark_path_points(action, np.asarray((0.0, 0.5, 1.0)))

    assert start == pytest.approx((action.x0, action.y0))
    assert end == pytest.approx((action.x1, action.y1))
    # A downward canvas-UV chord has a positive normal toward +x.
    assert midpoint == pytest.approx((0.62, 0.50))
    assert mark_path_length(action) > math.hypot(
        action.x1 - action.x0, action.y1 - action.y0
    )


def test_curve_decoder_projects_only_curvature_into_canvas_support() -> None:
    straight = StrokeAction(0.05, 0.08, 0.05, 0.92, 0.05, 0.5, 1.0)
    requested = -0.40
    projected = bounded_mark_curvature(straight, requested, margin=0.035)
    curved = StrokeAction(
        straight.x0,
        straight.y0,
        straight.x1,
        straight.y1,
        straight.width,
        straight.amount,
        straight.tone,
        curvature=projected,
    )
    points = mark_path_points(curved, np.linspace(0.0, 1.0, 129))

    assert projected < 0.0
    assert abs(projected) < abs(requested)
    assert float(points.min()) >= 0.035 - 1e-8
    assert float(points.max()) <= 0.965 + 1e-8


def test_spatial_action_raster_follows_curve_not_endpoint_chord() -> None:
    config = PainterConfig(spatial_grid_size=64)
    curved = _curved_vertical(0.20)
    straight = StrokeAction(
        curved.x0,
        curved.y0,
        curved.x1,
        curved.y1,
        curved.width,
        curved.amount,
        curved.tone,
    )
    curved_raster = rasterize_stroke_action(curved, 64, config=config)
    straight_raster = rasterize_stroke_action(straight, 64, config=config)

    midpoint_row = 31
    curved_peak = int(np.argmax(curved_raster[0, midpoint_row]))
    straight_peak = int(np.argmax(straight_raster[0, midpoint_row]))
    assert curved_peak >= straight_peak + 6
    assert np.allclose(curved_raster[6], curved.curvature)


def test_curve_aware_reference_tracks_declared_midpoint() -> None:
    sim = ArmPainterSim(PainterConfig())
    action = _curved_vertical(0.20)
    timing = adaptive_stroke_timing(sim, action)
    paint_midpoint = timing.approach + timing.press + 0.5 * timing.paint
    reference = stroke_reference(action, sim, paint_midpoint, timing)

    expected_uv = mark_path_points(action, 0.5)
    assert reference.phase == "paint"
    assert reference.x == pytest.approx(
        (float(expected_uv[0]) - 0.5) * sim.canvas.width * 0.98
    )
    assert reference.z == pytest.approx(
        (0.5 - float(expected_uv[1])) * sim.canvas.height * 0.98
    )


def test_motor_alternatives_separate_fixed_posture_and_roll_sweep() -> None:
    config = PainterConfig(
        motor_realization_kinds=(
            "cartesian_ik",
            "upper_arm_fixed_roll_positive",
            "upper_arm_fixed_roll_negative",
            "upper_arm_roll_positive",
            "upper_arm_roll_negative",
        ),
        motor_realization_candidate_limit=5,
        motor_fixed_roll_degrees=24.0,
        motor_roll_sweep_degrees=32.0,
    )
    policy = Policy((_curved_vertical(), StrokeAction.stop_action()))
    alternatives = motor_realization_policy_alternatives(policy, config)
    primitives = {item.motor_primitive.kind: item.motor_primitive for item in alternatives}

    assert primitives["cartesian_ik"].roll_start_deg == 0.0
    assert primitives["upper_arm_fixed_roll_positive"].roll_start_deg == 24.0
    assert primitives["upper_arm_fixed_roll_positive"].roll_end_deg == 24.0
    assert primitives["upper_arm_fixed_roll_negative"].roll_start_deg == -24.0
    assert primitives["upper_arm_fixed_roll_negative"].roll_end_deg == -24.0
    assert primitives["upper_arm_roll_positive"].roll_start_deg == -32.0
    assert primitives["upper_arm_roll_positive"].roll_end_deg == 32.0
    assert primitives["upper_arm_roll_negative"].roll_start_deg == 32.0
    assert primitives["upper_arm_roll_negative"].roll_end_deg == -32.0

    controller = controller_for_motor_primitive(
        primitives["upper_arm_fixed_roll_positive"]
    )
    assert controller.roll_start_deg == pytest.approx(24.0)
    assert controller.roll_end_deg == pytest.approx(24.0)


def test_opposite_fixed_roll_postures_predict_distinct_bodily_outcomes() -> None:
    config = PainterConfig(canvas_size=48, motor_forecast_samples=1)
    action = _curved_vertical(0.18)
    policy = Policy((action, StrokeAction.stop_action()))
    alternatives = motor_realization_policy_alternatives(
        policy,
        PainterConfig(
            motor_realization_kinds=(
                "upper_arm_fixed_roll_positive",
                "upper_arm_fixed_roll_negative",
            ),
            motor_realization_candidate_limit=2,
        ),
    )
    forecasts = [
        forecast_stroke_execution(
            ArmPainterSim(config),
            action,
            canvas_summary_state,
            motor_primitive=item.motor_primitive,
            dt=1.0 / 45.0,
        )
        for item in alternatives
    ]

    assert all(forecast.feasible for forecast in forecasts)
    assert forecasts[0].motor_primitive_kind != forecasts[1].motor_primitive_kind
    assert not np.allclose(
        forecasts[0].proprioceptive_mean,
        forecasts[1].proprioceptive_mean,
    )


def test_curved_hand_proposal_has_continuous_symmetric_magnitudes_and_density() -> None:
    config = PainterConfig(
        candidate_policies=3001,
        planning_horizon=1,
        passage_proposal_mix=0.0,
        passage_plan_proposal_mix=0.0,
        curved_mark_proposal_mix=2.0 / 3.0,
        mark_curvature_magnitude=0.24,
        mark_curvature_min_fraction=0.125,
    )
    sampler = PolicySampler(config, seed=91)
    policies = sampler.sample()
    marks = [policy.actions[0] for policy in policies if not policy.actions[0].stop]
    records = [record for record in sampler.last_proposal_records if record is not None]
    requested = np.asarray([record.latents[0].curvature for record in records])

    curved = requested[np.abs(requested) > 1e-12]
    assert np.unique(np.round(curved, 6)).size > 100
    assert np.abs(curved).min() >= 0.03
    assert np.abs(curved).max() <= 0.24
    assert np.std(np.abs(curved)) > 0.04
    assert abs(float((requested > 0).mean()) - 1.0 / 3.0) < 0.08
    assert abs(float((requested < 0).mean()) - 1.0 / 3.0) < 0.08
    assert abs(float((requested == 0).mean()) - 1.0 / 3.0) < 0.08
    assert all(
        np.all((mark_path_points(mark, np.linspace(0.0, 1.0, 65)) >= 0.03))
        and np.all((mark_path_points(mark, np.linspace(0.0, 1.0, 65)) <= 0.97))
        for mark in marks
    )
    straight_record = next(
        record for record in records if abs(record.latents[0].curvature) <= 1e-12
    )
    curved_record = next(
        record for record in records if abs(record.latents[0].curvature) > 1e-12
    )
    assert hand_written_log_density(straight_record, None, config).curvature == pytest.approx(
        math.log(1.0 / 3.0)
    )
    assert hand_written_log_density(curved_record, None, config).curvature == pytest.approx(
        math.log(1.0 / 3.0) - math.log(0.21)
    )


def test_curve_proposal_length_means_brush_travel_not_endpoint_chord() -> None:
    config = PainterConfig(
        candidate_policies=801,
        planning_horizon=1,
        passage_proposal_mix=0.0,
        passage_plan_proposal_mix=0.0,
        curved_mark_proposal_mix=1.0,
        mark_curvature_magnitude=0.24,
        mark_curvature_min_fraction=0.10,
    )
    sampler = PolicySampler(config, seed=97)
    policies = sampler.sample()
    checked = 0
    for policy, record in zip(policies, sampler.last_proposal_records):
        if record is None:
            continue
        action = policy.actions[0]
        latent = record.latents[0]
        # Canvas-bound projection may legitimately reduce curvature. This test
        # isolates decoded curves whose requested geometry fitted unchanged.
        if action.curvature != pytest.approx(latent.curvature, abs=1e-8):
            continue
        assert mark_path_length(action) == pytest.approx(latent.length, rel=2e-5)
        checked += 1
    assert checked > 100


def test_degenerate_curvature_range_falls_back_to_straight_mixed_measure() -> None:
    config = PainterConfig(
        candidate_policies=101,
        planning_horizon=1,
        passage_proposal_mix=0.0,
        passage_plan_proposal_mix=0.0,
        curved_mark_proposal_mix=1.0,
        mark_curvature_magnitude=0.24,
        mark_curvature_min_fraction=1.0,
    )
    sampler = PolicySampler(config, seed=101)
    sampler.sample()
    records = [record for record in sampler.last_proposal_records if record is not None]

    assert all(record.latents[0].curvature == 0.0 for record in records)
    assert all(
        hand_written_log_density(record, None, config).curvature == pytest.approx(0.0)
        for record in records
    )
