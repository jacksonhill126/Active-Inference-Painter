import numpy as np
import pytest

from active_painter.config import PainterConfig
from active_painter.motor_planning import motor_efe_terms
from active_painter.motor_reliability import (
    MotionReliabilityBelief,
    MotionReliabilityLedger,
    execution_error_ratio_sq,
)
from active_painter.stroke_execution import forecast_stroke_execution
from active_painter.arm_sim import ArmPainterSim
from active_painter.env import StrokeAction


def test_reliability_belief_converges_toward_observed_ratio() -> None:
    reliable = MotionReliabilityBelief.from_prior(mean=1.6, strength=4.0)
    jittery = MotionReliabilityBelief.from_prior(mean=1.6, strength=4.0)
    for _ in range(200):
        reliable.update(1.0)
        jittery.update(9.0)  # realized error 3x predicted
    assert reliable.expected_inflation() == pytest.approx(1.0, abs=0.1)
    assert jittery.expected_inflation() == pytest.approx(9.0, abs=0.5)
    # Evidence resolves uncertainty: epistemic value shrinks with observations.
    fresh = MotionReliabilityBelief.from_prior(mean=1.6, strength=4.0)
    assert reliable.epistemic_nats() < fresh.epistemic_nats()


def test_execution_error_ratio_is_floored_and_capped() -> None:
    assert execution_error_ratio_sq(0.0, 0.0) <= 1.0
    assert execution_error_ratio_sq(10.0, 0.0001) == 25.0  # capped
    assert execution_error_ratio_sq(0.02, 0.02) == pytest.approx(1.0)


def test_reliability_inflation_raises_motor_risk_for_the_same_forecast() -> None:
    cfg = PainterConfig(canvas_size=48)
    sim = ArmPainterSim(cfg)
    action = StrokeAction(0.3, 0.4, 0.7, 0.6, 0.08, 0.6, 1.0)

    def summary(s: ArmPainterSim) -> np.ndarray:
        return np.zeros(cfg.state_dim, dtype=np.float32)

    forecast = forecast_stroke_execution(sim, action, summary, rollout_samples=1)
    trusted = motor_efe_terms(forecast, cfg, reliability_inflation=1.0)
    jittery = motor_efe_terms(forecast, cfg, reliability_inflation=4.0)
    assert jittery.risk > trusted.risk
    assert jittery.ambiguity > trusted.ambiguity
    # Unresolved reliability uncertainty is credited as information gain.
    curious = motor_efe_terms(forecast, cfg, reliability_inflation=1.0, reliability_epistemic_nats=0.3)
    assert curious.epistemic_value > trusted.epistemic_value


def test_reliability_ledger_snapshot_round_trip() -> None:
    cfg = PainterConfig()
    ledger = MotionReliabilityLedger(cfg)
    for _ in range(20):
        ledger.observe("upper_arm_roll_positive", 6.0)
        ledger.observe("cartesian_ik", 1.0)
    restored = MotionReliabilityLedger(cfg)
    restored.restore(ledger.snapshot())
    assert restored.expected_inflation("upper_arm_roll_positive") == pytest.approx(
        ledger.expected_inflation("upper_arm_roll_positive")
    )
    assert restored.expected_inflation("cartesian_ik") == pytest.approx(
        ledger.expected_inflation("cartesian_ik")
    )
    assert restored.expected_inflation("upper_arm_roll_positive") > restored.expected_inflation("cartesian_ik")


def test_body_param_jitter_keeps_forecasts_deterministic_and_spreads_particles() -> None:
    cfg = PainterConfig(canvas_size=48)
    assert cfg.body_param_jitter_fraction > 0.0
    sim = ArmPainterSim(cfg)
    action = StrokeAction(0.3, 0.4, 0.7, 0.6, 0.08, 0.6, 1.0)

    def summary(s: ArmPainterSim) -> np.ndarray:
        return np.asarray([s.canvas.material_coverage()], dtype=np.float32)

    first = forecast_stroke_execution(sim, action, summary, rollout_samples=2)
    second = forecast_stroke_execution(sim, action, summary, rollout_samples=2)
    # Deterministic per (sim, action, sample count): resume/cache safe.
    assert np.allclose(first.proprioceptive_mean, second.proprioceptive_mean)
    # Particle 0 keeps the mean body model: a 1-sample forecast matches a
    # no-jitter 1-sample forecast exactly.
    from dataclasses import replace as dc_replace
    no_jitter_cfg = dc_replace(cfg, body_param_jitter_fraction=0.0)
    no_jitter_sim = ArmPainterSim(no_jitter_cfg)
    base = forecast_stroke_execution(sim, action, summary, rollout_samples=1)
    unjittered = forecast_stroke_execution(no_jitter_sim, action, summary, rollout_samples=1)
    assert np.allclose(base.proprioceptive_mean, unjittered.proprioceptive_mean)
