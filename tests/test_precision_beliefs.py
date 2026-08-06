"""Reference parity and degeneracy pins for the Gamma precision beliefs.

Every test here names the identity or the negative result it pins. The point of
the suite is not that precision learning "works" -- measured, it barely moves --
but that the mechanism is the reference's rule, that its degeneracies are
reported rather than dressed up as convergence, and that it cannot become a
backdoor around "preferences are never learned from outcomes".
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from _reference_loader import load_reference_module

from active_painter.config import PainterConfig
from active_painter.precision_beliefs import (
    LEDGER_KEYS,
    MODALITY_NAMES,
    NORMALIZER_NAMES,
    POLICY_PRECISION_KEY,
    GammaPrecisionBelief,
    GapIncrementBelief,
    PrecisionLedger,
    learn_precision,
    precision_gradient,
)


@pytest.fixture(scope="module")
def pomdp():
    return load_reference_module("core.pomdp")


_PAIRS = (
    (np.array([1.0, 2.0, 3.0, 0.5]), np.array([0.5, 1.0, 1.5, 0.25])),
    (np.array([0.0, 1.0, -1.0]), np.array([2.0, -0.5, 0.75])),
    (np.array([4.0, 4.5, 3.0, 9.0, 1.0]), np.array([-1.0, 0.0, 1.0, 2.0, 3.0])),
)


@pytest.mark.parametrize("G,F", _PAIRS)
def test_precision_gradient_matches_reference_at_unit_shape(pomdp, G, F) -> None:
    """dF/dgamma = (beta - beta0) + (pi - pi0).(-G) equals the reference at alpha=1."""

    for gamma in (0.35, 1.0, 3.0):
        mine = precision_gradient(G, F, gamma, alpha=1.0, beta0=1.0)
        theirs = pomdp.precision_gradient(G, F, gamma, beta0=1.0)
        assert mine == pytest.approx(theirs, abs=1e-12)


@pytest.mark.parametrize("G,F", _PAIRS)
def test_learn_precision_matches_reference_at_unit_shape(pomdp, G, F) -> None:
    """The whole beta <- beta - kappa*grad descent equals the reference at alpha=1."""

    for beta0 in (1.0, 1.0 / 3.0, 1.0 / 0.35):
        mine = learn_precision(G, F, alpha=1.0, beta0=beta0)
        theirs = pomdp.learn_precision(G, F, beta0=beta0)
        assert mine.gamma == pytest.approx(theirs.gamma, abs=1e-12)
        assert mine.beta == pytest.approx(theirs.beta, abs=1e-12)
        assert mine.converged == theirs.converged
        assert len(mine.gamma_trace) == len(theirs.gamma_trace)


def test_flat_free_energy_makes_the_rule_provably_inert(pomdp) -> None:
    """NEGATIVE RESULT, pinned: a flat F gives gradient exactly 0.0.

    This is the measured degeneracy behind the whole feature's honest framing.
    The reference reports converged=True with gradient 0.0, which LOOKS like a
    learned fixed point but is the trivial gamma = alpha / beta0. Anything that
    "learns" a precision from a flat F has learned nothing.
    """

    G = np.array([1.0, 2.0, 3.0, 0.5])
    zeros = np.zeros_like(G)

    assert precision_gradient(G, zeros, 1.0, beta0=1.0) == 0.0
    assert pomdp.precision_gradient(G, zeros, 1.0, beta0=1.0) == 0.0
    for beta0 in (1.0, 1.0 / 3.0, 1.0 / 0.35):
        result = learn_precision(G, zeros, beta0=beta0)
        assert result.converged
        assert result.gamma == pytest.approx(1.0 / beta0, abs=1e-12)
    # A flat but non-zero F is equally inert: only differences of F matter.
    flat = np.full_like(G, 3.7)
    assert learn_precision(G, flat, beta0=1.0).gamma == pytest.approx(1.0, abs=1e-12)


def test_ledger_reports_flat_free_energy_as_degenerate_not_updated() -> None:
    """The ledger must refuse to call a provable no-op an update."""

    ledger = PrecisionLedger(PainterConfig())
    before = ledger.mean("transition")
    update = ledger.observe("transition", [1.0, 2.0, 3.0], [0.5, 0.5, 0.5])
    assert update.status == "degenerate_flat_F"
    assert ledger.mean("transition") == before


def test_unobserved_belief_mean_is_bit_identical_to_the_declared_constant() -> None:
    """Attribution must be exact, not approximate.

    Turning the beliefs off and leaving them unobserved must be the SAME
    arithmetic, so the all-off configuration reproduces the pre-feature repo bit
    for bit.
    """

    cfg = PainterConfig(policy_precision=3.0, terminal_risk_precision=1.0)
    ledger = PrecisionLedger(cfg)
    assert ledger.mean(POLICY_PRECISION_KEY) == 3.0
    assert ledger.mean("terminal_coverage") == 1.0
    assert ledger.mean("canvas_latent_transition") == cfg.canvas_latent_transition_precision

    web_like = PainterConfig(policy_precision=0.35)
    assert PrecisionLedger(web_like).mean(POLICY_PRECISION_KEY) == 0.35

    disabled = PainterConfig(policy_precision=0.35, precision_beliefs_enabled=False)
    assert PrecisionLedger(disabled).mean(POLICY_PRECISION_KEY) == 0.35


def test_zero_declared_constant_creates_no_belief_and_returns_exactly_zero() -> None:
    """A declared constant of exactly 0.0 is a STRUCTURAL off switch.

    Every precision=0.0 isolation config in the suite depends on this: a learned
    precision must never resurrect a modality the config switched off.
    """

    cfg = PainterConfig(transition_precision=0.0, composition_gap_precision=0.0)
    ledger = PrecisionLedger(cfg)
    assert "transition" not in ledger.beliefs
    assert "composition_gap" not in ledger.beliefs
    assert ledger.mean("transition") == 0.0
    assert ledger.mean("composition_gap") == 0.0
    update = ledger.observe("transition", [1.0, 2.0], [0.0, 1.0])
    assert update.status == "structurally_off"
    assert ledger.mean("transition") == 0.0
    weights = ledger.weights()
    assert weights.transition == 0.0
    assert weights.composition == 0.0


def test_modality_flag_off_still_learns_the_policy_precision() -> None:
    """The two mechanisms must be separately attributable."""

    cfg = PainterConfig(modality_precision_beliefs_enabled=False)
    ledger = PrecisionLedger(cfg)
    modality = ledger.observe("transition", [1.0, 2.0, 3.0], [0.4, 0.9, 1.6])
    assert modality.status == "disabled"
    assert ledger.mean("transition") == cfg.transition_precision
    policy = ledger.observe_policy([1.0, 2.0, 3.0], [0.4, 0.9, 1.6])
    assert policy.status in {"updated", "clamped"}


def _disagreeing_pair(scale: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 1.0, size=24)
    base = base - base.mean()
    G = base * scale
    return G, -0.5 * G


def test_declared_bounded_support_stops_a_precision_deleting_a_preference() -> None:
    """MEASURED HAZARD: unbounded, disagreeing evidence collapses gamma.

    Measured on a 24-candidate vector with std 6.9 nats, F = -0.5 G drives the
    unbounded precision to 0.0679 -- a 15x attenuation of whatever modality it
    weights. Applied to the terminal-coverage modality that is a 15x attenuation
    of the declared C matrix inside G, i.e. outcome data quietly switching a
    preference most of the way off. The declared bounded support is a charter
    requirement, not a convenience.
    """

    G, F = _disagreeing_pair(8.0)
    unbounded = learn_precision(G, F, beta0=1.0, n_iter=512)
    assert unbounded.gamma < 0.1
    assert unbounded.gamma == pytest.approx(0.0679, abs=5e-4)

    cfg = PainterConfig(precision_belief_min_ratio=0.1, precision_belief_max_ratio=10.0)
    ledger = PrecisionLedger(cfg)
    prior = ledger.prior_mean("terminal_coverage")
    statuses = set()
    for _ in range(20):
        statuses.add(ledger.observe("terminal_coverage", G, F).status)
    assert "clamped" in statuses
    assert ledger.mean("terminal_coverage") == pytest.approx(0.1 * prior, abs=1e-12)


def test_bounded_support_also_caps_agreement() -> None:
    """The ceiling is declared too, so agreement cannot inflate a modality freely."""

    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 1.0, size=24)
    G = (base - base.mean()) * 1.7
    cfg = PainterConfig(precision_belief_min_ratio=0.1, precision_belief_max_ratio=1.05)
    ledger = PrecisionLedger(cfg)
    prior = ledger.prior_mean("transition")
    update = ledger.observe("transition", G, 0.5 * G)
    assert update.status == "clamped"
    assert ledger.mean("transition") == pytest.approx(1.05 * prior, abs=1e-12)


def test_too_few_candidates_is_refused() -> None:
    ledger = PrecisionLedger(PainterConfig())
    update = ledger.observe("transition", [1.0], [0.5])
    assert update.status == "too_few_candidates"


def test_resolvable_uncertainty_is_a_constant_diagnostic() -> None:
    """The Gamma SHAPE is declared and never updated, so this cannot be novelty.

    Crediting a constant as information gain would be an unearned epistemic
    bonus, so it is exposed for diagnostics and forbidden from every EFE path.
    """

    belief = GammaPrecisionBelief.from_prior(1.0, alpha0=1.0)
    before = belief.resolvable_uncertainty()
    belief.update(np.array([1.0, 2.0, 3.0]), np.array([0.4, 0.9, 1.6]))
    assert belief.resolvable_uncertainty() == before
    assert before == pytest.approx(0.5 * math.log1p(1.0))


def test_snapshot_restore_round_trip_preserves_the_posterior() -> None:
    cfg = PainterConfig()
    ledger = PrecisionLedger(cfg)
    ledger.observe_policy([1.0, 2.0, 3.0], [0.4, 0.9, 1.6])
    ledger.observe("transition", [1.0, 2.0, 3.0], [0.4, 0.9, 1.6])
    payload = ledger.snapshot()
    expected = {name: ledger.mean(name) for name in LEDGER_KEYS}

    restored = PrecisionLedger(cfg)
    restored.restore(payload)
    for name in LEDGER_KEYS:
        assert restored.mean(name) == pytest.approx(expected[name], abs=1e-12)

    # A payload from before this feature (or a corrupted one) must be inert.
    fresh = PrecisionLedger(cfg)
    fresh.restore(None)
    fresh.restore({"transition": {"nonsense": 1.0}})
    assert fresh.mean("transition") == cfg.transition_precision


def test_summary_is_json_serializable_with_only_finite_builtin_floats() -> None:
    ledger = PrecisionLedger(PainterConfig())
    ledger.observe_policy([1.0, 900.0, 3.0], [0.4, -80.0, 1.6])
    for name in MODALITY_NAMES:
        ledger.observe(name, [1.0, 900.0, 3.0], [0.4, -80.0, 1.6])
    payload = ledger.summary()
    encoded = json.dumps(payload)
    assert "NaN" not in encoded and "Infinity" not in encoded
    for name, entry in payload.items():
        assert set(entry) >= {"gamma", "priorGamma", "gradient", "observations", "status"}
        assert math.isfinite(float(entry["gamma"]))
        assert isinstance(entry["status"], str)
        if name in NORMALIZER_NAMES:
            assert entry["normalizerName"] == NORMALIZER_NAMES[name]


# --------------------------------------------------------------------------
# GapIncrementBelief
# --------------------------------------------------------------------------


def test_unobserved_gap_belief_contributes_exactly_zero_to_the_stop_prior() -> None:
    """The zero is EXACT, which is what preserves log(0.5) at the midpoint."""

    cfg = PainterConfig()
    belief = GapIncrementBelief.from_config(cfg)
    assert not belief.has_observations()
    assert belief.stop_log_prior_term(cfg) == 0.0
    # First reading only anchors the difference; it is not an observation.
    assert belief.observe(0.4, 3) is False
    assert belief.stop_log_prior_term(cfg) == 0.0
    assert belief.observe(0.5, 4) is True
    assert belief.stop_log_prior_term(cfg) != 0.0


def test_gap_progress_stop_term_is_never_positive_and_rises_as_progress_falls() -> None:
    """A PRIOR factor, so it can only make stopping less unlikely."""

    cfg = PainterConfig()
    terms = []
    for increment in (0.5, 0.2, 0.05, 0.0, -0.05, -0.4):
        belief = GapIncrementBelief.from_config(cfg)
        belief.observe(0.0, 0)
        belief.observe(increment, 1)
        term = belief.stop_log_prior_term(cfg)
        assert term <= 0.0
        terms.append(term)
    assert terms == sorted(terms), terms


def test_gap_progress_flag_off_is_exactly_zero() -> None:
    cfg = PainterConfig(gap_progress_stop_enabled=False)
    belief = GapIncrementBelief.from_config(cfg)
    belief.observe(0.0, 0)
    belief.observe(0.4, 1)
    assert belief.has_observations()
    assert belief.stop_log_prior_term(cfg) == 0.0


def test_per_mark_denominator_divides_by_the_marks_actually_elapsed() -> None:
    """Same total gap change over 4 marks must read as a quarter of the increment."""

    cfg = PainterConfig(
        gap_increment_prior_mean=0.0,
        gap_increment_prior_std=10.0,
        gap_increment_process_std=10.0,
        gap_increment_observation_std=1e-4,
    )
    one = GapIncrementBelief.from_config(cfg)
    one.observe(0.0, 0)
    one.observe(0.4, 1)
    four = GapIncrementBelief.from_config(cfg)
    four.observe(0.0, 0)
    four.observe(0.4, 4)
    assert one.posterior_mean() == pytest.approx(0.4, rel=1e-3)
    assert four.posterior_mean() == pytest.approx(0.1, rel=1e-3)

    # A second reading at the SAME mark index is composition-model training
    # drift, not a per-mark increment, so it must not be counted -- otherwise
    # learning progress would be attributed to mark-making. It still advances the
    # anchor, so the drift is discarded rather than folded into the next mark.
    same = GapIncrementBelief.from_config(cfg)
    same.observe(0.0, 7)
    assert same.observe(0.2, 7) is False
    assert not same.has_observations()
    assert same.last_gap == pytest.approx(0.2)
    assert same.observe(0.5, 8) is True
    assert same.posterior_mean() == pytest.approx(0.3, rel=1e-3)


def test_gap_belief_snapshot_restore_round_trip() -> None:
    cfg = PainterConfig()
    belief = GapIncrementBelief.from_config(cfg)
    belief.observe(0.1, 1)
    belief.observe(0.3, 5)
    payload = belief.snapshot()
    restored = GapIncrementBelief.from_config(cfg)
    restored.restore(payload)
    assert restored.posterior_mean() == pytest.approx(belief.posterior_mean(), abs=1e-12)
    assert restored.observations == belief.observations
    assert restored.last_mark_index == belief.last_mark_index
    assert restored.stop_log_prior_term(cfg) == pytest.approx(
        belief.stop_log_prior_term(cfg), abs=1e-12
    )

    fresh = GapIncrementBelief.from_config(cfg)
    fresh.restore(None)
    assert not fresh.has_observations()
    assert json.dumps(fresh.summary())
