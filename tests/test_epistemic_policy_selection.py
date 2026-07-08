from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from active_painter.config import PainterConfig
from active_painter.efe import EFEComponents, ExpectedFreeEnergy
from active_painter.env import StrokeAction
from active_painter.models import DynamicsEnsemble, GaussianBelief, ObservationModel
from active_painter.policies import Policy
from active_painter.preferences import TerminalCoveragePreference


class FixedResidualTransition(nn.Module):
    """Deterministic test approximation to one learned ensemble member."""

    def __init__(self, member_offset: float, uncertainty_scale: float) -> None:
        super().__init__()
        self.member_offset = member_offset
        self.uncertainty_scale = uncertainty_scale

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        next_state = state.clone()
        next_state[..., 0] = torch.clamp(state[..., 0] + action[..., 5], 1e-4, 1.0 - 1e-4)

        unfamiliar_region = (action[..., 0:1] > 0.5).to(dtype=state.dtype)
        residual = unfamiliar_region * self.member_offset * self.uncertainty_scale
        next_state[..., 1:] = state[..., 1:] + residual

        logvar = torch.full_like(next_state, math.log(1e-7))
        return next_state, logvar


class DecreasingCoverageDynamics:
    def predictive_moments(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        next_mean = state.clone()
        next_mean[..., 0] = state[..., 0] - 0.05
        variance = torch.full_like(next_mean, 1e-6)
        epistemic = torch.zeros_like(next_mean)
        return next_mean, variance, epistemic


@pytest.fixture
def belief() -> GaussianBelief:
    return GaussianBelief(
        torch.tensor([0.82, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.full((6,), -16.0),
    )


def ensemble_with_uncertainty(uncertainty_scale: float) -> DynamicsEnsemble:
    cfg = PainterConfig(ensemble_size=5)
    dynamics = DynamicsEnsemble(cfg)
    dynamics.members = nn.ModuleList(
        FixedResidualTransition(offset, uncertainty_scale)
        for offset in (-1.0, -0.5, 0.0, 0.5, 1.0)
    )
    return dynamics


def make_efe(cfg: PainterConfig, dynamics: DynamicsEnsemble) -> ExpectedFreeEnergy:
    return ExpectedFreeEnergy(cfg, dynamics, ObservationModel(cfg), TerminalCoveragePreference(cfg))


def stroke_policy(action: StrokeAction) -> Policy:
    return Policy((action, StrokeAction.stop_action()))


def evaluate_deterministically(
    efe: ExpectedFreeEnergy,
    belief: GaussianBelief,
    policy: Policy,
) -> EFEComponents:
    torch.manual_seed(1234)
    return efe.evaluate(belief, policy)


def ensemble_parameter_information_gain(
    dynamics: DynamicsEnsemble,
    belief: GaussianBelief,
    action: StrokeAction,
) -> torch.Tensor:
    state = belief.mean.unsqueeze(0)
    action_tensor = torch.from_numpy(action.vector()).unsqueeze(0)
    member_means, member_logvars = dynamics(state, action_tensor)
    member_variances = member_logvars.exp()
    aleatoric = member_variances.mean(dim=0)
    epistemic = member_means.var(dim=0, unbiased=False)
    marginal_variance = aleatoric + epistemic
    return 0.5 * (torch.log(marginal_variance).sum(dim=-1).mean() - torch.log(member_variances).sum(dim=-1).mean())


def assert_decompositions_are_not_mixed(components: EFEComponents) -> None:
    assert components.total == pytest.approx(
        components.terminal_risk
        + components.ambiguity
        + components.transition_risk
        + components.transition_ambiguity
    )
    assert components.transition_risk + components.transition_ambiguity == pytest.approx(
        -components.epistemic_value
    )


def test_higher_learned_model_uncertainty_increases_epistemic_value(
    belief: GaussianBelief,
) -> None:
    cfg = PainterConfig()
    uncertain_action = StrokeAction(0.8, 0.2, 0.9, 0.8, 0.08, 0.05, 1.0)
    policy = stroke_policy(uncertain_action)

    low_dynamics = ensemble_with_uncertainty(uncertainty_scale=0.01)
    high_dynamics = ensemble_with_uncertainty(uncertainty_scale=0.08)
    action = torch.from_numpy(uncertain_action.vector()).unsqueeze(0)
    state = belief.mean.unsqueeze(0)
    _, _, low_epistemic_variance = low_dynamics.predictive_moments(state, action)
    _, _, high_epistemic_variance = high_dynamics.predictive_moments(state, action)

    assert high_epistemic_variance[..., 1:].mean() > low_epistemic_variance[..., 1:].mean()

    low = evaluate_deterministically(make_efe(cfg, low_dynamics), belief, policy)
    high = evaluate_deterministically(make_efe(cfg, high_dynamics), belief, policy)

    assert high.epistemic_value > low.epistemic_value
    assert high.epistemic_value == pytest.approx(
        float(ensemble_parameter_information_gain(high_dynamics, belief, uncertain_action))
    )
    assert high.terminal_coverage_mean == pytest.approx(low.terminal_coverage_mean)
    assert_decompositions_are_not_mixed(high)


def test_uncertainty_gated_epistemic_value_can_change_policy_preference(
    belief: GaussianBelief,
) -> None:
    certain_target_action = StrokeAction(0.2, 0.2, 0.4, 0.8, 0.08, 0.05, 1.0)
    uncertain_overshoot_action = StrokeAction(0.8, 0.2, 0.9, 0.8, 0.08, 0.12, 1.0)
    certain_policy = stroke_policy(certain_target_action)
    uncertain_policy = stroke_policy(uncertain_overshoot_action)

    low_uncertainty_cfg = PainterConfig()
    low_uncertainty_dynamics = ensemble_with_uncertainty(uncertainty_scale=0.0)
    low_uncertainty_efe = make_efe(low_uncertainty_cfg, low_uncertainty_dynamics)
    certain_without_epistemic = evaluate_deterministically(low_uncertainty_efe, belief, certain_policy)
    uncertain_without_epistemic = evaluate_deterministically(low_uncertainty_efe, belief, uncertain_policy)

    assert certain_without_epistemic.terminal_coverage_mean == pytest.approx(0.87)
    assert uncertain_without_epistemic.terminal_coverage_mean == pytest.approx(0.94)
    assert certain_without_epistemic.total < uncertain_without_epistemic.total

    epistemic_cfg = PainterConfig()
    dynamics = ensemble_with_uncertainty(uncertainty_scale=0.08)
    epistemic_efe = make_efe(epistemic_cfg, dynamics)
    certain_with_epistemic = evaluate_deterministically(epistemic_efe, belief, certain_policy)
    uncertain_with_epistemic = evaluate_deterministically(epistemic_efe, belief, uncertain_policy)

    assert uncertain_with_epistemic.epistemic_value > certain_with_epistemic.epistemic_value
    assert uncertain_with_epistemic.total < certain_with_epistemic.total
    assert_decompositions_are_not_mixed(uncertain_with_epistemic)

    policy_totals = torch.tensor([certain_with_epistemic.total, uncertain_with_epistemic.total])
    posterior = torch.softmax(-epistemic_cfg.policy_precision * (policy_totals - policy_totals.min()), dim=0)
    assert posterior[1] > posterior[0]


class DivergingCoverageTransition(nn.Module):
    """Deterministic member whose coverage increment is member-specific."""

    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = increment

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        next_state = state.clone()
        next_state[..., 0] = torch.clamp(state[..., 0] + self.increment, 0.0, 1.0)
        logvar = torch.full_like(next_state, math.log(1e-7))
        return next_state, logvar


def diverging_ensemble() -> DynamicsEnsemble:
    cfg = PainterConfig(ensemble_size=5)
    dynamics = DynamicsEnsemble(cfg)
    dynamics.members = nn.ModuleList(
        DivergingCoverageTransition(increment) for increment in (0.0, 0.01, 0.02, 0.03, 0.04)
    )
    return dynamics


def test_member_trajectory_rollouts_accumulate_parameter_uncertainty_over_depth() -> None:
    # Member-wise trajectory sampling: each ensemble member propagates its own
    # particle, so disagreement compounds over the policy horizon instead of
    # being collapsed to a mixture after every step.
    cfg = PainterConfig()
    efe = make_efe(cfg, diverging_ensemble())
    belief = GaussianBelief(
        torch.tensor([0.10, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.full((6,), -16.0),
    )
    stroke = StrokeAction(0.2, 0.2, 0.4, 0.4, 0.08, 0.1, 1.0)
    one_step = Policy((stroke, StrokeAction.stop_action()))
    two_step = Policy((stroke, stroke, StrokeAction.stop_action()))

    shallow = efe.evaluate(belief, one_step)
    deep = efe.evaluate(belief, two_step)

    assert shallow.terminal_coverage_mean == pytest.approx(0.12, abs=1e-4)
    assert deep.terminal_coverage_mean == pytest.approx(0.14, abs=1e-4)
    assert deep.terminal_coverage_std > 1.8 * shallow.terminal_coverage_std


def test_evaluate_batch_matches_single_policy_evaluation(belief: GaussianBelief) -> None:
    cfg = PainterConfig()
    efe = make_efe(cfg, ensemble_with_uncertainty(uncertainty_scale=0.05))
    stroke = StrokeAction(0.8, 0.2, 0.9, 0.8, 0.08, 0.05, 1.0)
    policies = [
        Policy((StrokeAction.stop_action(),)),
        Policy((stroke, StrokeAction.stop_action())),
        Policy((stroke, stroke, StrokeAction.stop_action())),
    ]

    batched = efe.evaluate_batch(belief, policies)
    singles = [efe.evaluate(belief, policy) for policy in policies]

    for batch_components, single_components in zip(batched, singles):
        assert batch_components.total == pytest.approx(single_components.total)
        assert batch_components.terminal_risk == pytest.approx(single_components.terminal_risk)
        assert batch_components.ambiguity == pytest.approx(single_components.ambiguity)
        assert batch_components.epistemic_value == pytest.approx(single_components.epistemic_value)
        assert batch_components.terminal_coverage_mean == pytest.approx(single_components.terminal_coverage_mean)
        assert batch_components.terminal_coverage_std == pytest.approx(single_components.terminal_coverage_std)


def test_efe_rollout_projects_painting_transition_to_monotone_material_coverage() -> None:
    cfg = PainterConfig()
    low_coverage_belief = GaussianBelief(
        torch.tensor([0.12, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.full((6,), -16.0),
    )
    efe = ExpectedFreeEnergy(
        cfg,
        DecreasingCoverageDynamics(),
        ObservationModel(cfg),
        TerminalCoveragePreference(cfg),
    )
    policy = stroke_policy(StrokeAction(0.2, 0.2, 0.3, 0.3, 0.08, 0.4, 1.0))

    components = evaluate_deterministically(efe, low_coverage_belief, policy)

    assert components.terminal_coverage_mean == pytest.approx(0.12)
