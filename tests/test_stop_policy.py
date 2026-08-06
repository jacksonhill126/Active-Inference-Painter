import math

import pytest
import torch

from active_painter.config import PainterConfig
from active_painter.efe import ExpectedFreeEnergy
from active_painter.env import StrokeAction
from active_painter.models import GaussianBelief, ObservationModel
from active_painter.policies import Policy
from active_painter.preferences import TerminalCoveragePreference


class DeterministicCoverageDynamics:
    def predictive_moments(self, state: torch.Tensor, action: torch.Tensor):
        # amount is action index 5; each unit adds 0.25 coverage in this mock model.
        next_state = state.clone()
        next_state[..., 0] = torch.clamp(state[..., 0] + 0.25 * action[..., 5], 0, 1)
        variance = torch.full_like(next_state, 1e-5)
        return next_state, variance, torch.zeros_like(next_state)


def make_efe() -> ExpectedFreeEnergy:
    cfg = PainterConfig()
    return ExpectedFreeEnergy(cfg, DeterministicCoverageDynamics(), ObservationModel(cfg), TerminalCoveragePreference(cfg))


def test_stop_is_preferred_near_target_over_overshoot() -> None:
    efe = make_efe()
    belief = GaussianBelief(torch.tensor([0.87, 0, 0, 0, 0, 0], dtype=torch.float32), torch.full((6,), -12.0))
    stop = Policy((StrokeAction.stop_action(),))
    overshoot_stroke = StrokeAction(0, 0, 1, 1, 0.1, 0.7, 1.0)
    continue_then_stop = Policy((overshoot_stroke, StrokeAction.stop_action()))
    assert efe.evaluate(belief, stop).total < efe.evaluate(belief, continue_then_stop).total


def test_continuation_is_preferred_when_it_reaches_target() -> None:
    efe = make_efe()
    belief = GaussianBelief(torch.tensor([0.69, 0, 0, 0, 0, 0], dtype=torch.float32), torch.full((6,), -12.0))
    stop = Policy((StrokeAction.stop_action(),))
    stroke = StrokeAction(0, 0, 1, 1, 0.1, 0.72, 1.0)  # +0.18 -> 0.87
    continue_then_stop = Policy((stroke, StrokeAction.stop_action()))
    assert efe.evaluate(belief, continue_then_stop).total < efe.evaluate(belief, stop).total


def test_terminal_risk_is_low_in_target_band() -> None:
    efe = make_efe()
    variance = torch.full((6,), 1e-5).log()
    stop = Policy((StrokeAction.stop_action(),))
    target = efe.evaluate(
        GaussianBelief(torch.tensor([0.87, 0, 0, 0, 0, 0], dtype=torch.float32), variance),
        stop,
    )
    low = efe.evaluate(
        GaussianBelief(torch.tensor([0.55, 0, 0, 0, 0, 0], dtype=torch.float32), variance),
        stop,
    )
    high = efe.evaluate(
        GaussianBelief(torch.tensor([0.98, 0, 0, 0, 0, 0], dtype=torch.float32), variance),
        stop,
    )
    assert target.terminal_risk < low.terminal_risk
    assert target.terminal_risk < high.terminal_risk


@pytest.mark.parametrize("precision", [1.0, 0.37, 2.9, 0.1, 10.0])
def test_terminal_risk_decomposes_into_entropy_and_pragmatic_value(precision: float) -> None:
    # The identity risk == -entropy - pragmatic_value must hold at ANY precision,
    # including a non-unit Gamma-belief posterior mean and both clamp boundaries
    # (0.1x and 10x the prior mean under the declared bounded support). It is
    # guaranteed because terminal_preference_terms scales all three returns by
    # the same factor -- a precision belief must not be able to break a
    # decomposition identity.
    cfg = PainterConfig(terminal_risk_precision=precision)
    efe = ExpectedFreeEnergy(
        cfg,
        DeterministicCoverageDynamics(),
        ObservationModel(cfg),
        TerminalCoveragePreference(cfg),
    )
    variance = torch.full((6,), 1e-5).log()
    stop = Policy((StrokeAction.stop_action(),))
    components = efe.evaluate(
        GaussianBelief(torch.tensor([0.87, 0, 0, 0, 0, 0], dtype=torch.float32), variance),
        stop,
    )
    assert components.terminal_risk == pytest.approx(
        -components.terminal_entropy - components.pragmatic_value
    )
    assert components.terminal_coverage_precision == pytest.approx(precision)


@pytest.mark.parametrize("normalization_enabled", [False, True])
def test_base_observation_entropy_does_not_reward_extra_steps(
    normalization_enabled: bool,
) -> None:
    # Run under BOTH normalization settings: the normalizers must be purely
    # multiplicative. An additive offset would be an unowned constant inside G
    # and would break exactly these three equalities -- a zero-amount extra
    # stroke must still cost exactly zero.
    cfg = PainterConfig(modality_normalization_enabled=normalization_enabled)
    efe = ExpectedFreeEnergy(
        cfg,
        DeterministicCoverageDynamics(),
        ObservationModel(cfg),
        TerminalCoveragePreference(cfg),
    )
    belief = GaussianBelief(
        torch.tensor([0.87, 0, 0, 0, 0, 0], dtype=torch.float32),
        torch.log(torch.full((6,), 1e-5)),
    )
    no_change_stroke = StrokeAction(0, 0, 1, 1, 0.1, 0.0, 1.0)
    continue_then_stop = Policy((no_change_stroke, StrokeAction.stop_action()))
    continued = efe.evaluate(belief, continue_then_stop)
    assert continued.ambiguity == pytest.approx(0.0)
    assert continued.transition_risk + continued.transition_ambiguity == pytest.approx(0.0)
    assert continued.total - continued.terminal_risk == pytest.approx(0.0)


@pytest.mark.parametrize("precision", [0.1, 1.0, 7.3])
def test_non_unit_precision_leaves_a_structurally_zero_term_exactly_zero(
    precision: float,
) -> None:
    # A learned precision multiplied into an identically-zero modality must stay
    # exactly 0.0 and finite -- never a NaN from a zero denominator.
    cfg = PainterConfig(ambiguity_precision=precision, transition_precision=precision)
    efe = ExpectedFreeEnergy(
        cfg,
        DeterministicCoverageDynamics(),
        ObservationModel(cfg),
        TerminalCoveragePreference(cfg),
    )
    belief = GaussianBelief(
        torch.tensor([0.87, 0, 0, 0, 0, 0], dtype=torch.float32),
        torch.log(torch.full((6,), 1e-5)),
    )
    continued = efe.evaluate(
        belief,
        Policy((StrokeAction(0, 0, 1, 1, 0.1, 0.0, 1.0), StrokeAction.stop_action())),
    )
    assert continued.ambiguity == 0.0
    assert math.isfinite(continued.total)
    assert continued.motor_risk == 0.0
    assert continued.motor_epistemic_value == 0.0
