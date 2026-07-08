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


def test_terminal_risk_decomposes_into_entropy_and_pragmatic_value() -> None:
    efe = make_efe()
    variance = torch.full((6,), 1e-5).log()
    stop = Policy((StrokeAction.stop_action(),))
    components = efe.evaluate(
        GaussianBelief(torch.tensor([0.87, 0, 0, 0, 0, 0], dtype=torch.float32), variance),
        stop,
    )
    assert components.terminal_risk == pytest.approx(
        -components.terminal_entropy - components.pragmatic_value
    )


def test_base_observation_entropy_does_not_reward_extra_steps() -> None:
    efe = make_efe()
    belief = GaussianBelief(
        torch.tensor([0.87, 0, 0, 0, 0, 0], dtype=torch.float32),
        torch.log(torch.full((6,), 1e-5)),
    )
    stop = Policy((StrokeAction.stop_action(),))
    no_change_stroke = StrokeAction(0, 0, 1, 1, 0.1, 0.0, 1.0)
    continue_then_stop = Policy((no_change_stroke, StrokeAction.stop_action()))
    continued = efe.evaluate(belief, continue_then_stop)
    assert continued.ambiguity == pytest.approx(0.0)
    assert continued.transition_risk + continued.transition_ambiguity == pytest.approx(0.0)
    assert continued.total - continued.terminal_risk == pytest.approx(0.0)
