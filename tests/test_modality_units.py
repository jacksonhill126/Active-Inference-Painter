"""Per-observation-channel unit declarations and the silent-domination tripwire.

The EFE modalities were historically summed in incommensurable units (a scalar
Beta KL, per-cell-channel densities, per-latent-dim KLs, and a raw sum over 27
proprioceptive channels). Precision beliefs are meaningless across such a mix,
so this suite pins:

* that no single modality's contribution silently dominates the sum, and that
  the assertion is a genuine tripwire rather than a tautology (the same
  assertion FAILS with the declared forecast-family restriction removed);
* that every modality records the name of the normalizer it was reduced by;
* that the believed compression-gap increment provably cannot leak into G.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from types import SimpleNamespace

import numpy as np
import pytest

from active_painter.config import PainterConfig
from active_painter.motor_planning import motor_efe_terms
from active_painter.policies import Policy, policy_stop_log_prior
from active_painter.precision_beliefs import (
    DEFAULT_PROPRIOCEPTIVE_CHANNELS,
    MODALITY_NAMES,
    NORMALIZER_NAMES,
    GapIncrementBelief,
    PrecisionLedger,
    constant_modality_weights,
)
from active_painter.spatial_agent import SpatialActiveInferencePainter
from active_painter.spatial_state import SpatialCanvasState
from active_painter.env import StrokeAction


# The cold-start regime: a genuinely blank canvas with a diffuse belief. This is
# where the moment-matched terminal Beta degenerates (digamma(alpha -> 0)) and
# therefore where a unit tripwire actually has to bite.
_BLANK_LOGVAR = -6.0


def _modality_contributions(component) -> dict[str, float]:
    values = {
        "terminal_coverage": component.terminal_risk,
        "observation_ambiguity": component.ambiguity,
        "transition": component.transition_risk + component.transition_ambiguity,
        "composition_gap": component.composition_risk,
        "canvas_latent_transition": (
            component.canvas_transition_risk + component.passage_canvas_trajectory_risk
        ),
        "relational_transition": (
            component.relational_transition_risk
            + component.passage_relational_trajectory_risk
        ),
        "motor_proprioceptive": component.motor_risk - component.motor_epistemic_value,
    }
    return {name: float(value) for name, value in values.items()}


def _blank_candidate_set(**overrides):
    cfg = PainterConfig(
        candidate_policies=24,
        planning_horizon=3,
        spatial_grid_size=16,
        spatial_transition_mode="local_patch",
        composition_enabled=True,
        composition_gap_precision=1.0,
        **overrides,
    )
    agent = SpatialActiveInferencePainter(cfg, seed=5, device="cpu")
    material = np.zeros(
        (cfg.spatial_material_channels, cfg.spatial_grid_size, cfg.spatial_grid_size),
        dtype=np.float32,
    )
    agent.belief = SpatialCanvasState(
        material=material,
        logvar=np.full_like(material, _BLANK_LOGVAR),
    )
    policies = agent.policy_sampler.sample(agent.belief.coverage(cfg.paint_presence_threshold))
    return cfg, agent, policies, agent.efe.evaluate_batch(agent.belief, policies)


def _domination_ratios(components) -> tuple[float, float, dict[str, float]]:
    """Mean-|contribution| ratio to the median, and the worst single candidate.

    Structurally-zero modalities are excluded: canvas/relational/passage/motor
    are exactly 0.0 without an execution forecast, so an unfiltered median is 0
    and the ratio is undefined rather than large.
    """

    per_modality = {
        name: np.abs([_modality_contributions(component)[name] for component in components])
        for name in MODALITY_NAMES
    }
    active = {name: values for name, values in per_modality.items() if values.max() > 0.0}
    assert len(active) >= 3, sorted(active)
    means = {name: float(values.mean()) for name, values in active.items()}
    median = float(np.median(list(means.values())))
    assert median > 0.0
    worst_mean = max(means.values()) / median
    worst_single = max(float(values.max()) for values in active.values()) / median
    return worst_mean, worst_single, means


def test_no_modality_dominates_the_declared_multiple_of_the_median() -> None:
    """Silent-domination tripwire, stated on mean |contribution| per modality.

    Measured on a 24-candidate blank-canvas set: terminal coverage's mean
    |contribution| is 20.8x the median of the other active modalities, inside the
    declared 50x bound.
    """

    _, _, _, components = _blank_candidate_set()
    worst_mean, worst_single, means = _domination_ratios(components)
    assert worst_mean <= 50.0, means
    # The tail is RECORDED, not hidden: one candidate's terminal risk still
    # reaches 329x the median modality contribution, because the floored forecast
    # is only bounded, not made commensurable.
    assert worst_single <= 400.0, worst_single


def test_the_tripwire_is_not_a_tautology() -> None:
    """The SAME assertion fails without the declared forecast-family restriction.

    Measured: removing the interior-unimodal restriction (concentration floor
    0.0) takes terminal coverage's mean |contribution| from 20.8x the median to
    1592x, and its worst single candidate from 329x to 3.8e4x. The floor is a
    declared forecast-family restriction, not a clamp on risk -- it is inert on
    every well-conditioned forecast.
    """

    _, _, _, components = _blank_candidate_set(terminal_forecast_concentration_floor=0.0)
    worst_mean, worst_single, _ = _domination_ratios(components)
    assert worst_mean > 50.0
    assert worst_single > 400.0


def test_every_modality_records_its_declared_normalizer_name() -> None:
    _, _, _, components = _blank_candidate_set()
    component = components[0]
    assert component.modality_units == "nats_per_observation_channel"
    assert component.approximation
    for name in MODALITY_NAMES:
        recorded = getattr(component, f"{name}_normalizer_name")
        assert recorded == NORMALIZER_NAMES[name], name
        assert recorded
    # Six modalities are already per-observation-channel densities; only the
    # motor modality needs a genuinely new divisor.
    for name in MODALITY_NAMES:
        if name == "motor_proprioceptive":
            continue
        assert getattr(component, f"{name}_normalizer") == 1.0, name
    assert component.motor_proprioceptive_normalizer == pytest.approx(
        1.0 / DEFAULT_PROPRIOCEPTIVE_CHANNELS
    )
    assert DEFAULT_PROPRIOCEPTIVE_CHANNELS == 27


def test_normalization_flag_off_restores_the_historical_mixed_units() -> None:
    weights = constant_modality_weights(PainterConfig(modality_normalization_enabled=False))
    for name in MODALITY_NAMES:
        assert weights.normalizer[name] == 1.0, name
    assert weights.concentration_floor == 0.0
    assert weights.motor_weight(27) == weights.gamma["motor_proprioceptive"]


def test_recorded_precision_and_normalizer_recover_the_raw_composition_gap() -> None:
    """Attribution assertion: raw * precision * normalizer == stored."""

    cfg, agent, _, components = _blank_candidate_set()
    ledger = agent.precision_ledger
    component = components[0]
    assert component.composition_gap_precision == pytest.approx(
        ledger.mean("composition_gap")
    )
    assert component.composition_risk == pytest.approx(
        -component.composition_gap
        * component.composition_gap_precision
        * component.composition_gap_normalizer,
        abs=1e-6,
    )


# --------------------------------------------------------------------------
# Motor normalizer
# --------------------------------------------------------------------------


def _stub_forecast(channel_count: int):
    labels = tuple(f"target_error_{index}" for index in range(channel_count))
    return SimpleNamespace(
        proprioceptive_labels=labels,
        proprioceptive_mean=np.full(channel_count, 0.05),
        proprioceptive_predictive_variance=np.full(channel_count, 1e-3),
        proprioceptive_likelihood_variance=np.full(channel_count, 2e-3),
    )


def test_motor_terms_divide_by_the_actual_proprioceptive_channel_count() -> None:
    cfg = PainterConfig()
    weights = constant_modality_weights(cfg)
    for channel_count in (9, 27):
        forecast = _stub_forecast(channel_count)
        unnormalized = motor_efe_terms(forecast, cfg)
        normalized = motor_efe_terms(forecast, cfg, weights=weights)
        assert normalized.normalizer == pytest.approx(1.0 / channel_count)
        assert normalized.normalizer_name == NORMALIZER_NAMES["motor_proprioceptive"]
        assert normalized.risk == pytest.approx(unnormalized.risk / channel_count)
        assert normalized.epistemic_value == pytest.approx(
            unnormalized.epistemic_value / channel_count
        )
        # The two pinned approximation substrings survive, and the new clause
        # names the divisor.
        assert "analytic in nats" in normalized.approximation
        assert "hard safety limits remain external" in normalized.approximation
        assert "nats per proprioceptive channel" in normalized.approximation


def test_motor_terms_are_invariant_to_duplicating_proprioceptive_channels() -> None:
    """A per-channel density must not grow when the same channel is listed twice.

    This is the cleanest available proof that the motor normalizer is a genuine
    per-channel density rather than a cosmetic constant.
    """

    cfg = PainterConfig()
    weights = constant_modality_weights(cfg)
    single = motor_efe_terms(_stub_forecast(9), cfg, weights=weights)
    doubled = motor_efe_terms(_stub_forecast(18), cfg, weights=weights)
    assert doubled.risk == pytest.approx(single.risk, rel=1e-12)
    assert doubled.epistemic_value == pytest.approx(single.epistemic_value, rel=1e-12)
    assert doubled.ambiguity == pytest.approx(single.ambiguity, rel=1e-12)


# --------------------------------------------------------------------------
# Delta-gap containment
# --------------------------------------------------------------------------


def test_gap_increment_belief_cannot_leak_into_expected_free_energy() -> None:
    """Delta-gap is a transition prior over a rate, never a reward.

    Feeding the belief a large increment must leave EVERY field of EVERY EFE
    component bit-identical. If Delta-gap ever appears inside G, this fails --
    which is exactly the "weighted score soup with extra steps" failure mode the
    feature is most at risk of.
    """

    cfg, agent, policies, before = _blank_candidate_set(
        gap_progress_stop_enabled=True
    )
    agent.gap_increment.observe(0.0, 0)
    agent.gap_increment.observe(25.0, 1)
    assert agent.gap_increment.has_observations()
    assert agent.gap_increment.posterior_mean() > 1.0

    after = agent.efe.evaluate_batch(agent.belief, policies)
    assert len(before) == len(after)
    names = [entry.name for entry in dataclass_fields(before[0])]
    for left, right in zip(before, after):
        for name in names:
            assert getattr(left, name) == getattr(right, name), name

    # ... while the STOP POLICY PRIOR does move, which is where it belongs.
    stop = Policy((StrokeAction.stop_action(),))
    continuation = Policy(
        (StrokeAction(0.1, 0.1, 0.9, 0.9, 0.08, 0.5, 1.0), StrokeAction.stop_action())
    )
    coverage_only = policy_stop_log_prior(stop, 0.5, cfg)
    with_progress = policy_stop_log_prior(stop, 0.5, cfg, agent.gap_increment)
    assert with_progress < coverage_only
    assert with_progress <= 0.0
    # Continuations keep a flat prior of exactly zero even with a fully observed
    # belief, so the progress term is a factor on stopping, not a global bias.
    assert policy_stop_log_prior(continuation, 0.5, cfg, agent.gap_increment) == 0.0


def test_ledger_weights_carry_the_declared_normalizer_names() -> None:
    ledger = PrecisionLedger(PainterConfig())
    weights = ledger.weights()
    for name in MODALITY_NAMES:
        assert weights.normalizer_name[name] == NORMALIZER_NAMES[name]
        assert weights.weight(name) == pytest.approx(
            ledger.mean(name) * weights.normalizer[name]
        )


def test_gap_belief_is_unobserved_by_default_so_the_stop_prior_is_unchanged() -> None:
    """Regression guard for the log(0.5) midpoint identity."""

    cfg = PainterConfig()
    stop = Policy((StrokeAction.stop_action(),))
    fresh = GapIncrementBelief.from_config(cfg)
    assert policy_stop_log_prior(
        stop, cfg.minimum_stop_coverage, cfg, fresh
    ) == policy_stop_log_prior(stop, cfg.minimum_stop_coverage, cfg)


def test_motor_normalization_flattens_the_realization_posterior_without_flipping_it() -> None:
    """MEASURED COST of the per-proprioceptive-channel unit, pinned not hidden.

    Reducing the motor modality to nats per channel divides its cross-kind spread
    by 27. At the web runtime's 0.35 policy precision the modal realization is
    unchanged, but the conditional motor posterior becomes nearly uniform -- so
    motor realization selection becomes close to indiscriminate. That is a real
    cost of the declared unit convention, and it must not be compensated by
    hand-tuning `motor_modality_precision`, which is exactly the practice this
    feature exists to remove.
    """

    from active_painter.motor_planning import motor_realization_log_evidence

    raw = [-4.408514, 1.466367, 1.057071, -4.408514, -4.408514]
    normalized = [value / 27.0 for value in raw]
    priors = [-float(np.log(5.0))] * 5

    _, raw_posterior = motor_realization_log_evidence(raw, priors, 0.35)
    _, normalized_posterior = motor_realization_log_evidence(normalized, priors, 0.35)

    def entropy(posterior: np.ndarray) -> float:
        return float(-(posterior * np.log(posterior + 1e-300)).sum())

    assert int(np.argmax(raw_posterior)) == int(np.argmax(normalized_posterior))
    assert entropy(normalized_posterior) > entropy(raw_posterior)
    assert entropy(raw_posterior) == pytest.approx(1.353, abs=5e-3)
    assert entropy(normalized_posterior) == pytest.approx(1.609, abs=5e-3)
    # Nearly the uniform maximum log(5): the spread has effectively been erased.
    assert entropy(normalized_posterior) > 0.999 * float(np.log(5.0))
