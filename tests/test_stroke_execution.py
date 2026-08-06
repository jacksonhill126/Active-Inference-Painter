import copy
import json
import threading
import time

import numpy as np
import pytest
import torch

import active_painter.stroke_execution as stroke_execution_module

from active_painter.arm_agent_driver import canvas_summary_state
from active_painter.arm_sim import ArmPainterSim, JOINT_NAMES, JointPlant
from active_painter.brush_loading import (
    BRUSH_MICROSTRUCTURE_PRIOR_VERSION,
    BrushLoadBelief,
)
from active_painter.config import PainterConfig
from active_painter.env import StrokeAction
from active_painter.motor_planning import motor_efe_terms, motor_realization_log_evidence
from active_painter.plant_interface import BodyBeliefSnapshot
from active_painter.precision_beliefs import constant_modality_weights
from active_painter.policies import MotorPrimitiveLatent, Policy
from active_painter.spatial_state import SpatialCanvasState, material_grid_from_canvas
from active_painter.stroke_execution import (
    ContactAwareStrokeController,
    DirectStrokeController,
    StrokeTiming,
    controller_for_motor_primitive,
    forecast_stroke_execution,
    forecast_stroke_executions_batch,
    pose_for_reference,
    stroke_reference,
)


def test_contact_aware_controller_reduces_overshoot_against_direct_waypoint_baseline() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    action = StrokeAction(0.2, 0.3, 0.8, 0.7, 0.08, 0.7, 1.0)

    direct = forecast_stroke_execution(sim, action, canvas_summary_state, controller=DirectStrokeController())
    aware = forecast_stroke_execution(sim, action, canvas_summary_state, controller=ContactAwareStrokeController())

    assert aware.feasible
    assert aware.execution_uncertainty < direct.execution_uncertainty
    assert aware.joint_target_error_rms < direct.joint_target_error_rms
    assert aware.joint_current_rms < direct.joint_current_rms
    assert sum(aware.path_covariance) < sum(direct.path_covariance)
    # Contact loss now means physical pressure loss only; no controller
    # paint-permission flag contributes to this probability.
    assert aware.contact_loss_probability < 0.35


def test_execution_forecast_diagnostics_are_json_serializable() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    action = StrokeAction(0.2, 0.3, 0.8, 0.7, 0.08, 0.7, 1.0)

    forecast = forecast_stroke_execution(sim, action, canvas_summary_state)
    diagnostics = forecast.diagnostics()

    json.dumps({"executionForecast": diagnostics})
    assert isinstance(diagnostics["next_state_mean"], list)
    assert isinstance(diagnostics["next_state_variance"], list)
    assert isinstance(diagnostics["canvas_delta_mean"], list)


def test_body_posterior_initializes_forecast_particles_with_independent_noise(
    monkeypatch,
) -> None:
    sim = ArmPainterSim(
        PainterConfig(
            canvas_size=24,
            motor_forecast_samples=3,
            body_param_jitter_fraction=0.0,
        )
    )
    belief = BodyBeliefSnapshot(
        monotonic_time_s=2.0,
        joint_names=JOINT_NAMES,
        joint_position_mean_rad=(0.10, -0.70, 0.02, 1.60),
        joint_position_variance_rad2=(4e-4, 9e-4, 4e-4, 9e-4),
        joint_velocity_mean_rad_s=(0.20, -0.10, 0.05, 0.15),
        joint_velocity_variance_rad2_s2=(0.01, 0.02, 0.01, 0.02),
        contact_probability=0.2,
        contact_force_mean_n=0.4,
        contact_force_variance_n2=0.04,
        posterior_revision=11,
        inference_model_id="body-inference-v0:test-sensors-v0",
        calibration_status="synthetic_test_only",
    )
    action = StrokeAction(0.42, 0.45, 0.58, 0.55, 0.05, 0.6, 1.0)
    timing = StrokeTiming(approach=0.02, press=0.02, paint=0.03, lift=0.02)
    initialized: list[tuple[np.ndarray, np.ndarray]] = []
    original_initializer = JointPlant.initialize_forecast_state

    def capture_initializer(self, joint_position_rad, joint_velocity_rad_s):
        initialized.append(
            (
                np.asarray(joint_position_rad, dtype=np.float64).copy(),
                np.asarray(joint_velocity_rad_s, dtype=np.float64).copy(),
            )
        )
        return original_initializer(self, joint_position_rad, joint_velocity_rad_s)

    monkeypatch.setattr(JointPlant, "initialize_forecast_state", capture_initializer)
    live_pose = copy.deepcopy(sim.actual_pose)
    live_plant_state = copy.deepcopy(sim.plant.state_snapshot())
    first = forecast_stroke_execution(
        sim,
        action,
        canvas_summary_state,
        timing=timing,
        dt=0.01,
        initial_body_belief=belief,
        independent_noise_seed=912,
    )
    first_initializations = copy.deepcopy(initialized)

    assert len(first_initializations) == 3
    np.testing.assert_allclose(
        first_initializations[0][0], belief.joint_position_mean_rad
    )
    np.testing.assert_allclose(
        first_initializations[0][1], belief.joint_velocity_mean_rad_s
    )
    assert not np.allclose(
        first_initializations[1][0], first_initializations[0][0]
    )
    assert not np.allclose(
        first_initializations[2][1], first_initializations[0][1]
    )
    assert sim.actual_pose == live_pose
    assert sim.plant.state_snapshot()["rng_state"] == live_plant_state["rng_state"]
    assert first.body_posterior_revision == 11
    assert first.body_inference_model_id == belief.inference_model_id
    assert first.body_calibration_status == belief.calibration_status
    assert "BodyBeliefSnapshot revision 11" in first.forecast_initialization
    assert "contact probability/force" in first.forecast_approximation

    # Advancing the live process RNG cannot alter belief-conditioned particles:
    # future rollout noise is initialized from the request seed instead.
    sim.plant._rng.random(37)
    initialized.clear()
    second = forecast_stroke_execution(
        sim,
        action,
        canvas_summary_state,
        timing=timing,
        dt=0.01,
        initial_body_belief=belief,
        independent_noise_seed=912,
    )
    for expected, actual in zip(first_initializations, initialized, strict=True):
        np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(second.next_state_mean, first.next_state_mean)
    np.testing.assert_allclose(second.next_state_variance, first.next_state_variance)

    with pytest.raises(ValueError, match="independent_noise_seed"):
        forecast_stroke_execution(
            sim,
            action,
            canvas_summary_state,
            timing=timing,
            dt=0.01,
            initial_body_belief=belief,
            independent_noise_seed=-1,
        )


def test_material_posterior_replaces_hidden_canvas_and_samples_declared_variance() -> None:
    cfg = PainterConfig(
        canvas_size=24,
        spatial_grid_size=2,
        motor_forecast_samples=3,
        body_param_jitter_fraction=0.0,
    )
    sim = ArmPainterSim(cfg)
    sim.canvas.thickness.fill(0.8)
    sim.canvas.wetness.fill(0.7)
    sim.canvas.black_mass.fill(0.6)
    sim.canvas.surface_tone.fill(0.9)
    live_material = (
        sim.canvas.thickness.copy(),
        sim.canvas.wetness.copy(),
        sim.canvas.black_mass.copy(),
        sim.canvas.surface_tone.copy(),
    )

    material = np.zeros((6, 2, 2), dtype=np.float32)
    material[0] = np.asarray([[0.010, 0.012], [0.014, 0.016]])
    material[1] = 0.004
    material[2] = 0.020  # projected to black_mass <= thickness
    material[3] = np.asarray([[0.15, 0.25], [0.35, 0.45]])
    belief = SpatialCanvasState(
        material=material,
        logvar=np.full_like(material, -12.0),
        posterior_revision=7,
        inference_model_id="camera-spatial-likelihood-v0:test-camera-v0",
        calibration_status="synthetic_test_only",
    )
    captured: list[np.ndarray] = []
    captured_native_thickness: list[np.ndarray] = []

    def summary(working: ArmPainterSim) -> np.ndarray:
        state = material_grid_from_canvas(working.canvas, 2, channel_count=4)
        captured.append(state.copy())
        captured_native_thickness.append(working.canvas.thickness.copy())
        return state.reshape(-1)

    action = StrokeAction(0.42, 0.45, 0.58, 0.55, 0.05, 0.6, 1.0)
    timing = StrokeTiming(approach=0.02, press=0.02, paint=0.03, lift=0.02)
    first = forecast_stroke_execution(
        sim,
        action,
        summary,
        timing=timing,
        dt=0.01,
        initial_material_belief=belief,
        independent_noise_seed=313,
    )
    first_initializations = [captured[index].copy() for index in (0, 2, 4)]

    np.testing.assert_allclose(first_initializations[0][0], material[0], atol=1e-7)
    np.testing.assert_allclose(first_initializations[0][1], material[1], atol=1e-7)
    np.testing.assert_allclose(first_initializations[0][2], material[0], atol=1e-7)
    np.testing.assert_allclose(first_initializations[0][3], material[3], atol=1e-7)
    assert not np.allclose(first_initializations[1], first_initializations[0])
    assert not np.allclose(first_initializations[2], first_initializations[0])
    assert np.all(first_initializations[1][2] <= first_initializations[1][0])
    assert np.all(first_initializations[2][2] <= first_initializations[2][0])
    for native in (
        captured_native_thickness[0],
        captured_native_thickness[2],
        captured_native_thickness[4],
    ):
        for row_slice in (slice(0, 12), slice(12, 24)):
            for col_slice in (slice(0, 12), slice(12, 24)):
                assert np.unique(native[row_slice, col_slice]).size == 1
    for actual, expected in zip(
        (sim.canvas.thickness, sim.canvas.wetness, sim.canvas.black_mass, sim.canvas.surface_tone),
        live_material,
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected)
    assert first.material_posterior_revision == 7
    assert first.material_inference_model_id == belief.inference_model_id
    assert first.material_calibration_status == belief.calibration_status
    assert "SpatialCanvasState revision 7" in first.forecast_initialization
    assert "substrate grain" in first.forecast_approximation

    captured.clear()
    captured_native_thickness.clear()
    mean_particle = forecast_stroke_execution(
        sim,
        action,
        summary,
        timing=timing,
        dt=0.01,
        rollout_samples=1,
        initial_material_belief=belief,
        independent_noise_seed=313,
    )
    assert np.any(first.next_state_variance > mean_particle.next_state_variance)

    # Changing only the hidden live material cannot change the posterior-seeded
    # initial particles. Future forecast noise uses the request seed.
    sim.canvas.thickness.fill(0.2)
    sim.canvas.wetness.fill(0.1)
    sim.canvas.black_mass.fill(0.05)
    sim.canvas.surface_tone.fill(0.75)
    captured.clear()
    captured_native_thickness.clear()
    second = forecast_stroke_execution(
        sim,
        action,
        summary,
        timing=timing,
        dt=0.01,
        initial_material_belief=belief,
        independent_noise_seed=313,
    )
    second_initializations = [captured[index] for index in (0, 2, 4)]
    for expected, actual in zip(first_initializations, second_initializations, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(second.next_state_mean, first.next_state_mean)
    np.testing.assert_allclose(second.next_state_variance, first.next_state_variance)


def test_brush_posterior_replaces_live_rng_and_samples_mark_variation(
    monkeypatch,
) -> None:
    cfg = PainterConfig(
        canvas_size=24,
        motor_forecast_samples=3,
        body_param_jitter_fraction=0.0,
        canvas_grain_strength=0.0,
    )
    sim = ArmPainterSim(cfg)
    sim.plant.process_noise_enabled = False
    sim.load_brush(0.22, 1.0)
    live_brush_state = {
        "load": sim.brush.load,
        "fresh_tone": sim.brush.fresh_tone,
        "offsets": sim.brush.bristle_offsets.copy(),
        "gains": sim.brush.bristle_gains.copy(),
        "streaks": sim.brush.streak_phases.copy(),
        "wobble_phases": sim.brush.wobble_phases.copy(),
        "wobble_amps": sim.brush.wobble_amps.copy(),
        "rng": copy.deepcopy(sim.brush.rng.bit_generator.state),
    }
    belief = BrushLoadBelief(
        load_mean=0.63,
        load_variance=0.01,
        black_fraction_mean=0.74,
        black_fraction_variance=0.02,
        revision=9,
        inference_model_id="brush-loading-belief-v0:test-camera-v0",
        calibration_status="synthetic_test_only",
    )
    action = StrokeAction(0.42, 0.45, 0.58, 0.55, 0.05, 0.6, 1.0)
    captured: list[dict[str, object]] = []
    original_reset = ContactAwareStrokeController.reset

    def capture_reset(self, working, reset_action, timing):
        brush = working.brush
        captured.append(
            {
                "load": brush.load,
                "fresh_tone": brush.fresh_tone,
                "offsets": brush.bristle_offsets.copy(),
                "gains": brush.bristle_gains.copy(),
                "streaks": brush.streak_phases.copy(),
                "wobble_phases": brush.wobble_phases.copy(),
                "wobble_amps": brush.wobble_amps.copy(),
                "rng": copy.deepcopy(brush.rng.bit_generator.state),
            }
        )
        return original_reset(self, working, reset_action, timing)

    monkeypatch.setattr(ContactAwareStrokeController, "reset", capture_reset)
    first = forecast_stroke_execution(
        sim,
        action,
        canvas_summary_state,
        dt=0.02,
        brush_reload=False,
        brush_belief=belief,
        independent_noise_seed=719,
    )
    first_initializations = copy.deepcopy(captured)

    assert len(first_initializations) == 3
    assert first_initializations[0]["load"] == pytest.approx(belief.load_mean)
    assert first_initializations[0]["fresh_tone"] == pytest.approx(
        belief.black_fraction_mean
    )
    np.testing.assert_allclose(
        first_initializations[0]["offsets"],
        np.linspace(-1.0, 1.0, cfg.brush_bristle_count, dtype=np.float32),
    )
    np.testing.assert_allclose(
        first_initializations[0]["gains"],
        1.0 - 0.5 * cfg.brush_bristle_depth,
    )
    np.testing.assert_allclose(first_initializations[0]["wobble_amps"], 0.0)
    assert first_initializations[1]["load"] != pytest.approx(
        first_initializations[0]["load"]
    )
    assert not np.allclose(
        first_initializations[1]["offsets"],
        first_initializations[0]["offsets"],
    )
    assert not np.allclose(
        first_initializations[2]["streaks"],
        first_initializations[0]["streaks"],
    )
    assert first.brush_posterior_revision == belief.revision
    assert first.brush_inference_model_id == belief.inference_model_id
    assert first.brush_calibration_status == belief.calibration_status
    assert first.brush_microstructure_prior_id == BRUSH_MICROSTRUCTURE_PRIOR_VERSION
    assert first.brush_preparation_kind == "preserve"
    assert "BrushLoadBelief revision 9" in first.forecast_initialization
    assert BRUSH_MICROSTRUCTURE_PRIOR_VERSION in first.forecast_approximation
    assert sim.brush.load == pytest.approx(live_brush_state["load"])
    assert sim.brush.fresh_tone == pytest.approx(live_brush_state["fresh_tone"])
    np.testing.assert_allclose(sim.brush.bristle_offsets, live_brush_state["offsets"])
    np.testing.assert_allclose(sim.brush.bristle_gains, live_brush_state["gains"])
    assert sim.brush.rng.bit_generator.state == live_brush_state["rng"]

    captured.clear()
    mean_particle = forecast_stroke_execution(
        sim,
        action,
        canvas_summary_state,
        dt=0.02,
        rollout_samples=1,
        brush_reload=False,
        brush_belief=belief,
        independent_noise_seed=719,
    )
    assert np.any(first.next_state_variance > mean_particle.next_state_variance)

    # Neither exact brush fields nor continuation of its RNG can affect a
    # belief-conditioned forecast under the same independent request seed.
    sim.brush.load = 0.04
    sim.brush.fresh_tone = 0.05
    sim.brush.bristle_offsets = sim.brush.bristle_offsets[::-1].copy()
    sim.brush.bristle_gains.fill(0.13)
    sim.brush.streak_phases.fill(0.91)
    sim.brush.wobble_phases.fill(2.4)
    sim.brush.wobble_amps.fill(0.99)
    sim.brush.rng.random(127)
    captured.clear()
    second = forecast_stroke_execution(
        sim,
        action,
        canvas_summary_state,
        dt=0.02,
        brush_reload=False,
        brush_belief=belief,
        independent_noise_seed=719,
    )
    for expected, actual in zip(first_initializations, captured, strict=True):
        for key in ("load", "fresh_tone"):
            assert actual[key] == pytest.approx(expected[key])
        for key in (
            "offsets",
            "gains",
            "streaks",
            "wobble_phases",
            "wobble_amps",
        ):
            np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=0.0)
        assert actual["rng"] == expected["rng"]
    np.testing.assert_allclose(second.next_state_mean, first.next_state_mean)
    np.testing.assert_allclose(second.next_state_variance, first.next_state_variance)


def test_brush_reload_forecast_uses_reload_transition_and_preserve_requires_belief(
    monkeypatch,
) -> None:
    cfg = PainterConfig(canvas_size=24, motor_forecast_samples=1)
    sim = ArmPainterSim(cfg)
    belief = BrushLoadBelief(0.2, 0.04, 0.3, 0.05, revision=4)
    action = StrokeAction(0.42, 0.45, 0.58, 0.55, 0.05, 0.6, 1.0)
    captured: list[tuple[float, float]] = []
    original_reset = ContactAwareStrokeController.reset

    def capture_reset(self, working, reset_action, timing):
        captured.append((working.brush.load, working.brush.fresh_tone))
        return original_reset(self, working, reset_action, timing)

    monkeypatch.setattr(ContactAwareStrokeController, "reset", capture_reset)
    forecast = forecast_stroke_execution(
        sim,
        action,
        canvas_summary_state,
        rollout_samples=1,
        brush_reload=True,
        brush_belief=belief,
        independent_noise_seed=91,
    )

    assert captured == [(1.0, 1.0)]
    assert forecast.brush_posterior_revision == belief.revision + 1
    assert forecast.brush_preparation_kind == "reload"

    with pytest.raises(ValueError, match="Preserve forecasts require"):
        forecast_stroke_execution(
            sim,
            action,
            canvas_summary_state,
            rollout_samples=1,
            brush_reload=False,
            brush_belief=None,
            independent_noise_seed=91,
        )


def test_contact_pressure_ramps_before_the_stroke_sweep() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    action = StrokeAction(0.2, 0.3, 0.8, 0.7, 0.08, 0.7, 1.0)
    timing = StrokeTiming()

    press_reference = stroke_reference(action, sim, timing.approach + 0.5 * timing.press, timing)
    paint_reference = stroke_reference(action, sim, timing.approach + timing.press + 0.1, timing)

    assert press_reference.pressure > 0.0
    assert press_reference.phase == "press"
    assert paint_reference.phase == "paint"
    assert paint_reference.pressure > 0.0


def test_contact_controller_does_not_retract_or_disable_pressure_when_tracking_lags() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    action = StrokeAction(0.2, 0.3, 0.8, 0.7, 0.08, 0.7, 1.0)
    timing = StrokeTiming()
    controller = ContactAwareStrokeController(
        preview_time=0.0,
        filter_time=1e-6,
        max_joint_speed_deg=1e6,
    )
    controller.reset(sim, action, timing)

    mid_paint_t = timing.approach + timing.press + 0.5 * timing.paint
    command = controller.command(sim, action, mid_paint_t, 1.0 / 240.0, timing)
    commanded_tip = sim.kinematics.tip(command.pose)

    assert command.reference.phase == "paint"
    assert command.intended_pressure > 0.0
    assert commanded_tip[1] == pytest.approx(sim.canvas.distance + 0.2, abs=0.03)


def test_execution_forecast_rejects_degenerate_stationary_paint_realization() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    action = StrokeAction(0.5, 0.5, 0.502, 0.5, 0.08, 0.7, 1.0)

    forecast = forecast_stroke_execution(sim, action, canvas_summary_state)

    assert not forecast.feasible
    assert forecast.intended_path_length < 0.18
    assert forecast.realized_path_span < 0.2


def test_forecast_proprioceptive_risk_depends_on_body_state_controllability() -> None:
    cfg = PainterConfig(canvas_size=48)
    near = ArmPainterSim(cfg)
    far = ArmPainterSim(cfg)
    action = StrokeAction(0.50, 0.50, 0.62, 0.50, 0.08, 0.7, 1.0)
    far.actual_pose.yaw = 75.0
    far.target_pose = far.actual_pose

    near_forecast = forecast_stroke_execution(near, action, canvas_summary_state)
    far_forecast = forecast_stroke_execution(far, action, canvas_summary_state)

    assert far_forecast.joint_target_error_rms > near_forecast.joint_target_error_rms
    # `risk` is the pragmatic term E_q[-log p*(o)] by declared contract, not a
    # KL, so this ordering is a statement about expected preference violation
    # alone. If risk were ever redefined as a full KL (i.e. minus the forecast
    # outcome entropy) the far forecast's larger entropy could invert it, so
    # this monotonicity is what that migration would have to re-establish.
    assert motor_efe_terms(far_forecast, cfg).risk > motor_efe_terms(near_forecast, cfg).risk


def test_joint_space_motor_primitive_forecast_reports_proprioceptive_outcomes() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=48))
    action = StrokeAction(0.2, 0.35, 0.8, 0.55, 0.08, 0.7, 1.0)
    primitive = MotorPrimitiveLatent("elbow_pivot", pivot_joint="elbow")

    forecast = forecast_stroke_execution(
        sim,
        action,
        canvas_summary_state,
        motor_primitive=primitive,
        dt=1.0 / 45.0,
    )

    assert forecast.motor_primitive_kind == "elbow_pivot"
    assert forecast.proprioceptive_observation_dim == 27
    assert len(forecast.proprioceptive_labels) == forecast.proprioceptive_observation_dim
    assert forecast.motor_rollout_samples == sim.config.motor_forecast_samples
    assert sum(forecast.proprioceptive_predictive_variance) > 0.0
    assert forecast.joint_current_rms >= 0.0
    assert forecast.joint_torque_rms >= 0.0
    assert forecast.joint_path_length_deg > 0.0


def test_upper_arm_roll_motor_primitive_uses_contact_aware_roll_sweep() -> None:
    primitive = MotorPrimitiveLatent(
        "upper_arm_roll_positive",
        pivot_joint="roll",
        roll_start_deg=-32.0,
        roll_end_deg=32.0,
    )

    controller = controller_for_motor_primitive(primitive)

    assert isinstance(controller, ContactAwareStrokeController)
    assert controller.roll_start_deg == pytest.approx(-32.0)
    assert controller.roll_end_deg == pytest.approx(32.0)
    timing = StrokeTiming()
    paint_midpoint = stroke_reference(
        StrokeAction(0.2, 0.35, 0.8, 0.55, 0.08, 0.7, 1.0),
        ArmPainterSim(PainterConfig(canvas_size=48)),
        timing.approach + timing.press + 0.5 * timing.paint,
        timing,
    )
    assert controller._roll_for_reference(paint_midpoint, timing) == pytest.approx(0.0)


def test_opposite_upper_arm_roll_policies_have_distinct_feasible_likelihoods() -> None:
    cfg = PainterConfig(canvas_size=48, motor_forecast_samples=1)
    action = StrokeAction(0.2, 0.35, 0.8, 0.55, 0.08, 0.7, 1.0)
    forecasts = []
    for kind, start, end in (
        ("upper_arm_roll_positive", -32.0, 32.0),
        ("upper_arm_roll_negative", 32.0, -32.0),
    ):
        forecasts.append(
            forecast_stroke_execution(
                ArmPainterSim(cfg),
                action,
                canvas_summary_state,
                motor_primitive=MotorPrimitiveLatent(
                    kind,
                    pivot_joint="roll",
                    roll_start_deg=start,
                    roll_end_deg=end,
                ),
                dt=1.0 / 45.0,
            )
        )

    assert all(forecast.feasible for forecast in forecasts)
    assert forecasts[0].motor_primitive_kind != forecasts[1].motor_primitive_kind
    assert not np.allclose(forecasts[0].next_state_mean, forecasts[1].next_state_mean)
    assert forecasts[0].joint_path_length_deg != pytest.approx(forecasts[1].joint_path_length_deg)


def test_motor_efe_terms_are_separate_precision_weighted_proprioceptive_terms() -> None:
    cfg = PainterConfig(
        motor_proprioceptive_risk_precision=0.5,
        motor_proprioceptive_ambiguity_precision=0.25,
        motor_reliability_novelty_precision=0.25,
    )
    sim = ArmPainterSim(cfg)
    action = StrokeAction(0.2, 0.35, 0.8, 0.55, 0.08, 0.7, 1.0)
    forecast = forecast_stroke_execution(
        sim,
        action,
        canvas_summary_state,
        motor_primitive=MotorPrimitiveLatent("joint_spline"),
        dt=1.0 / 45.0,
    )

    terms = motor_efe_terms(forecast, cfg)

    assert terms.risk >= 0.0
    assert terms.ambiguity >= 0.0
    assert terms.epistemic_value > 0.0
    # The subtracted information gain is separately attributable: state/observation
    # mutual information under one declared precision, learned-reliability
    # parameter novelty under another. This call supplies no reliability
    # evidence, so only the former is non-zero.
    assert terms.mutual_information > 0.0
    assert terms.reliability_novelty == pytest.approx(0.0)
    assert terms.epistemic_value == pytest.approx(terms.mutual_information + terms.reliability_novelty)
    # H[q(o)] is logged so the canonical KL form of risk stays derivable.
    assert terms.forecast_outcome_entropy != 0.0
    assert "analytic in nats" in terms.approximation
    assert "hard safety limits remain external" in terms.approximation
    # With no modality weights supplied the historical arithmetic is reproduced
    # exactly, so no normalization clause is claimed.
    assert terms.normalizer == 1.0
    assert "nats per proprioceptive channel" not in terms.approximation

    normalized = motor_efe_terms(forecast, cfg, weights=constant_modality_weights(cfg))
    channel_count = len(forecast.proprioceptive_labels)
    assert channel_count == 27
    assert normalized.normalizer == pytest.approx(1.0 / channel_count)
    assert normalized.normalizer_name == "nats_per_proprioceptive_channel"
    assert normalized.risk == pytest.approx(terms.risk / channel_count)
    # The two pinned substrings survive the appended clause.
    assert "analytic in nats" in normalized.approximation
    assert "hard safety limits remain external" in normalized.approximation
    assert "nats per proprioceptive channel" in normalized.approximation


@pytest.mark.parametrize("precision", [0.035, 0.35, 2.5, 3.5])
def test_motor_realization_evidence_marginalizes_declared_prior_without_candidate_count_bonus(
    precision: float,
) -> None:
    # Swept over a drifting policy precision, including the declared clamp
    # boundaries 0.1x and 10x the 0.35 web prior mean. The no-candidate-count
    # bonus property is a logsumexp NORMALIZATION property and must therefore be
    # gamma-independent: the driver now feeds a belief posterior mean here.
    single_evidence, single_posterior = motor_realization_log_evidence([1.2], [0.0], precision)
    repeated_evidence, repeated_posterior = motor_realization_log_evidence(
        [1.2, 1.2, 1.2],
        [-float(torch.log(torch.tensor(3.0)))] * 3,
        precision,
    )

    assert repeated_evidence == pytest.approx(single_evidence)
    assert single_posterior.tolist() == pytest.approx([1.0])
    assert repeated_posterior.tolist() == pytest.approx([1.0 / 3.0] * 3)


def test_execution_forecast_changes_efe_through_realized_canvas_distribution() -> None:
    from active_painter.efe import ExpectedFreeEnergy
    from active_painter.models import DynamicsEnsemble, GaussianBelief, ObservationModel
    from active_painter.preferences import TerminalCoveragePreference

    cfg = PainterConfig()
    efe = ExpectedFreeEnergy(cfg, DynamicsEnsemble(cfg), ObservationModel(cfg), TerminalCoveragePreference(cfg))
    belief = GaussianBelief(
        torch.tensor([0.68, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.full((6,), -14.0),
    )
    policy = Policy((StrokeAction(0.2, 0.2, 0.8, 0.8, 0.08, 0.7, 1.0), StrokeAction.stop_action()))
    precise_realization = efe.evaluate_with_first_transition(
        belief,
        policy,
        torch.tensor([0.87, 0.1, 0.1, 0.0, 0.0, 0.0]),
        torch.full((6,), 1e-5),
        execution_uncertainty=0.01,
        contact_loss_probability=0.0,
        motor_overshoot=0.01,
        motor_feasible=True,
    )
    uncertain_realization = efe.evaluate_with_first_transition(
        belief,
        policy,
        torch.tensor([0.78, 0.1, 0.1, 0.0, 0.0, 0.0]),
        torch.tensor([0.04, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]),
        execution_uncertainty=0.8,
        contact_loss_probability=0.5,
        motor_overshoot=0.8,
        motor_feasible=True,
    )

    assert precise_realization.total < uncertain_realization.total
    assert precise_realization.execution_forecast_used
    assert uncertain_realization.execution_forecast_used
    assert precise_realization.execution_uncertainty < uncertain_realization.execution_uncertainty


def test_motor_efe_terms_contribute_to_total_without_mixing_with_coverage_terms() -> None:
    from active_painter.efe import ExpectedFreeEnergy
    from active_painter.models import DynamicsEnsemble, GaussianBelief, ObservationModel
    from active_painter.preferences import TerminalCoveragePreference

    cfg = PainterConfig()
    efe = ExpectedFreeEnergy(cfg, DynamicsEnsemble(cfg), ObservationModel(cfg), TerminalCoveragePreference(cfg))
    belief = GaussianBelief(
        torch.tensor([0.68, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.full((6,), -14.0),
    )
    policy = Policy((StrokeAction(0.2, 0.2, 0.8, 0.8, 0.08, 0.7, 1.0), StrokeAction.stop_action()))

    base = efe.evaluate_with_first_transition(
        belief,
        policy,
        torch.tensor([0.87, 0.1, 0.1, 0.0, 0.0, 0.0]),
        torch.full((6,), 1e-5),
        execution_uncertainty=0.01,
        contact_loss_probability=0.0,
        motor_overshoot=0.01,
        motor_feasible=True,
    )
    motor_loaded = efe.evaluate_with_first_transition(
        belief,
        policy,
        torch.tensor([0.87, 0.1, 0.1, 0.0, 0.0, 0.0]),
        torch.full((6,), 1e-5),
        execution_uncertainty=0.01,
        contact_loss_probability=0.0,
        motor_overshoot=0.01,
        motor_feasible=True,
        motor_risk=0.4,
        motor_ambiguity=0.2,
        motor_epistemic_value=0.1,
        motor_efe_approximation="test proprioceptive modality",
    )

    assert motor_loaded.terminal_risk == base.terminal_risk
    assert motor_loaded.transition_risk == base.transition_risk
    assert motor_loaded.motor_risk == pytest.approx(0.4)
    assert motor_loaded.motor_ambiguity == pytest.approx(0.2)
    assert motor_loaded.motor_epistemic_value == pytest.approx(0.1)
    # The motor modality enters G as pragmatic - information gain: 0.4 - 0.1.
    # motor_ambiguity is likelihood entropy in excess of the preference scale,
    # logged above but never a summand -- adding it would double count ambiguity,
    # because -I(s;o) already carries the canonical ambiguity contribution.
    assert motor_loaded.total == pytest.approx(base.total + 0.3)
    assert motor_loaded.total == pytest.approx(
        base.total + motor_loaded.motor_risk - motor_loaded.motor_epistemic_value
    )
    # Explicit regression pin against the old `+ motor_ambiguity` arithmetic.
    assert motor_loaded.total != pytest.approx(base.total + 0.5)


def test_motor_forecast_batch_overlaps_independent_requests_and_preserves_order(monkeypatch) -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=24, motor_forecast_samples=1))
    action = StrokeAction(0.35, 0.45, 0.55, 0.45, 0.06, 0.5, 1.0)
    primitives = [
        MotorPrimitiveLatent("cartesian_ik"),
        MotorPrimitiveLatent("joint_spline"),
        MotorPrimitiveLatent("elbow_pivot", pivot_joint="elbow"),
    ]
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_forecast(*args, motor_primitive=None, dt=0.0, rollout_samples=None, **kwargs):
        nonlocal active, maximum_active
        assert args[0] is sim
        assert dt == pytest.approx(1.0 / 45.0)
        assert rollout_samples == 3
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return motor_primitive.kind

    monkeypatch.setattr(stroke_execution_module, "forecast_stroke_execution", fake_forecast)

    results = forecast_stroke_executions_batch(
        sim,
        [(action, primitive) for primitive in primitives],
        canvas_summary_state,
        dt=1.0 / 45.0,
        rollout_samples=3,
        max_workers=3,
    )

    assert results == [primitive.kind for primitive in primitives]
    assert maximum_active > 1


def test_batched_and_sequential_motor_likelihoods_are_numerically_identical() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=24, motor_forecast_samples=2))
    action = StrokeAction(0.38, 0.48, 0.58, 0.48, 0.06, 0.55, 1.0)
    requests = [
        (action, MotorPrimitiveLatent("cartesian_ik")),
        (action, MotorPrimitiveLatent("joint_spline")),
    ]

    sequential = forecast_stroke_executions_batch(
        sim,
        requests,
        canvas_summary_state,
        dt=1.0 / 45.0,
        max_workers=1,
    )
    batched = forecast_stroke_executions_batch(
        sim,
        requests,
        canvas_summary_state,
        dt=1.0 / 45.0,
        max_workers=2,
    )

    for expected, actual in zip(sequential, batched):
        np.testing.assert_allclose(actual.next_state_mean, expected.next_state_mean, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(actual.next_state_variance, expected.next_state_variance, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(actual.proprioceptive_mean, expected.proprioceptive_mean, rtol=0.0, atol=0.0)
        assert actual.feasibility_probability == pytest.approx(expected.feasibility_probability)
