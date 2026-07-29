import pytest

from active_painter.arm_agent_driver import (
    ORACLE_OBSERVATION_ACCESS_MODE,
    ArmActiveInferenceDriver,
    StrokeExecution,
    canvas_summary_state,
)
from active_painter.arm_sim import ArmPainterSim
from active_painter.brush_loading import BrushLoadBelief, BrushLoadingModel
from active_painter.config import PainterConfig
from active_painter.efe import EFEComponents
from active_painter.env import StrokeAction
from active_painter.policies import BrushPreparationPolicy


def _black_mark(amount: float = 0.6) -> StrokeAction:
    return StrokeAction(0.2, 0.3, 0.8, 0.7, 0.08, amount, 1.0)


def test_reload_transition_is_full_and_uniformly_selected_color() -> None:
    model = BrushLoadingModel(PainterConfig())
    contaminated = BrushLoadBelief(
        load_mean=0.22,
        load_variance=0.04,
        black_fraction_mean=0.43,
        black_fraction_variance=0.08,
    )

    white = model.reload_transition(contaminated, 0.0)
    black = model.reload_transition(contaminated, 1.0)

    assert white.load_mean == 1.0
    assert white.black_fraction_mean == 0.0
    assert black.load_mean == 1.0
    assert black.black_fraction_mean == 1.0
    assert white.revision == contaminated.revision + 1
    assert white.load_variance == pytest.approx(
        model.config.brush_reload_load_std**2
    )


def test_preparation_posterior_preserves_full_matching_brush_and_reloads_low_brush() -> None:
    model = BrushLoadingModel(PainterConfig())
    action = _black_mark()
    full = BrushLoadBelief(1.0, 0.001, 1.0, 0.001)
    low = BrushLoadBelief(0.2, 0.01, 1.0, 0.01)

    full_inference = model.infer_preparation(full, action)
    low_inference = model.infer_preparation(low, action)

    assert full_inference.selected.kind == "preserve"
    assert low_inference.selected.kind == "reload"
    assert sum(probability for _, probability in full_inference.posterior) == pytest.approx(1.0)
    assert sum(probability for _, probability in low_inference.posterior) == pytest.approx(1.0)
    assert all(component.total >= 0.0 for component in low_inference.components)
    assert all("hard threshold" in component.approximation for component in low_inference.components)


def test_mixture_mismatch_is_a_conditional_pigment_risk_not_a_cleaning_rule() -> None:
    model = BrushLoadingModel(PainterConfig())
    action = _black_mark()
    wrong_mixture = BrushLoadBelief(0.85, 0.01, 0.15, 0.01)

    inference = model.infer_preparation(wrong_mixture, action)
    preserve, reload = inference.components

    assert inference.selected.kind == "reload"
    assert preserve.pigment_risk > reload.pigment_risk
    assert preserve.predicted_black_fraction_mean == pytest.approx(0.15)
    assert reload.predicted_black_fraction_mean == 1.0


def test_camera_derived_mark_likelihood_updates_load_and_reports_vfe() -> None:
    model = BrushLoadingModel(PainterConfig())
    prior = BrushLoadBelief(0.8, 0.09, 1.0, 0.01)
    action = _black_mark(amount=0.5)

    posterior = model.infer_load_from_mark(
        prior,
        action,
        observed_deposition=0.12,
        observation_variance=0.01,
    )

    assert posterior.load_mean < prior.load_mean
    assert posterior.load_variance < prior.load_variance
    assert model.last_vfe is not None
    assert model.last_vfe.total == pytest.approx(
        model.last_vfe.complexity
        + model.last_vfe.negative_log_likelihood
    )
    assert model.last_vfe.expected_log_likelihood == pytest.approx(
        -model.last_vfe.negative_log_likelihood
    )


def test_driver_preserve_policy_does_not_implicitly_reload_process_brush() -> None:
    config = PainterConfig(candidate_policies=2, planning_horizon=1)
    sim = ArmPainterSim(config)
    driver = ArmActiveInferenceDriver(
        config=config,
        bootstrap_transitions=0,
        bootstrap_train_steps=0,
        observation_access_mode=ORACLE_OBSERVATION_ACCESS_MODE,
    )
    action = _black_mark()
    driver.brush_load_beliefs["black"] = BrushLoadBelief(
        0.9,
        0.001,
        1.0,
        0.001,
    )
    sim.load_brush(0.47, 1.0)
    driver.current = StrokeExecution(
        action=action,
        efe=EFEComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        posterior=1.0,
        initial_state=canvas_summary_state(sim),
        brush_preparation=BrushPreparationPolicy("preserve", 1.0),
    )

    driver._execute_current(sim, 1.0 / 240.0)

    assert sim.brush.load == pytest.approx(0.47)
    assert sim.deposition_amount == pytest.approx(action.amount)
    assert driver.current is not None
    assert driver.current.brush_preparation.kind == "preserve"
