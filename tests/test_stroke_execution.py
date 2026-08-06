import json
import threading
import time

import numpy as np
import pytest
import torch

import active_painter.stroke_execution as stroke_execution_module

from active_painter.arm_agent_driver import canvas_summary_state
from active_painter.arm_sim import ArmPainterSim
from active_painter.config import PainterConfig
from active_painter.env import StrokeAction
from active_painter.motor_planning import motor_efe_terms, motor_realization_log_evidence
from active_painter.precision_beliefs import constant_modality_weights
from active_painter.policies import MotorPrimitiveLatent, Policy
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
