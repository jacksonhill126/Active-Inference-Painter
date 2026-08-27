import types

import numpy as np
import pytest
import torch

from active_painter.agent import ActiveInferencePainter
from active_painter.config import PainterConfig
from active_painter.efe import EFEComponents
from active_painter.env import StrokeAction
from active_painter.models import GaussianBelief
from active_painter.policies import Policy, policy_stop_log_prior
from active_painter.precision_beliefs import GapIncrementBelief


def make_agent_with_stop_favoring_efe(believed_coverage: float) -> ActiveInferencePainter:
    """Agent whose (stubbed) EFE always scores immediate stop lowest.

    This isolates the declared stop prior: any demotion of stop in the policy
    posterior must come from log p(pi), not from expected free energy.
    """

    cfg = PainterConfig(candidate_policies=8, planning_horizon=1)
    agent = ActiveInferencePainter(cfg, seed=3, device="cpu")
    agent.belief = GaussianBelief(
        torch.tensor([believed_coverage, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.full((6,), -12.0),
    )

    def evaluate_batch(belief, policies):
        return [
            EFEComponents(
                total=0.0 if policy.actions[0].stop else 1.0,
                terminal_risk=0.0,
                ambiguity=0.0,
                epistemic_value=0.0,
                terminal_coverage_mean=believed_coverage,
                terminal_coverage_std=0.01,
            )
            for policy in policies
        ]

    agent.efe = types.SimpleNamespace(evaluate_batch=evaluate_batch)
    return agent


def test_premature_stop_is_suppressed_by_declared_stop_prior() -> None:
    agent = make_agent_with_stop_favoring_efe(0.05)
    torch.manual_seed(0)

    _, _, ranked = agent.infer_policy()

    assert not ranked[0][0].actions[0].stop
    stop_probability = next(prob for policy, _, prob in ranked if policy.actions[0].stop)
    assert stop_probability < 1e-6


def test_stop_prior_is_neutral_near_target_coverage() -> None:
    agent = make_agent_with_stop_favoring_efe(0.87)
    torch.manual_seed(0)

    _, _, ranked = agent.infer_policy()

    assert ranked[0][0].actions[0].stop


def test_policy_stop_log_prior_is_monotone_and_flat_for_continuations() -> None:
    cfg = PainterConfig()
    stop = Policy((StrokeAction.stop_action(),))
    continuation = Policy((StrokeAction(0.1, 0.1, 0.9, 0.9, 0.08, 0.5, 1.0), StrokeAction.stop_action()))

    assert policy_stop_log_prior(continuation, 0.05, cfg) == 0.0
    low = policy_stop_log_prior(stop, 0.05, cfg)
    mid = policy_stop_log_prior(stop, cfg.minimum_stop_coverage, cfg)
    high = policy_stop_log_prior(stop, 0.90, cfg)

    assert low < mid < high <= 0.0
    assert low < -20.0
    assert high > -0.01
    assert mid == pytest.approx(float(np.log(0.5)))


def _observed_gap_belief(cfg: PainterConfig, increment: float) -> GapIncrementBelief:
    belief = GapIncrementBelief.from_config(cfg)
    belief.observe(0.0, 0)
    belief.observe(increment, 1)
    assert belief.has_observations()
    return belief


def test_gap_progress_is_a_separable_second_prior_factor() -> None:
    """The stop prior is a PRODUCT of two priors, hence a sum of two logs.

    The original intent -- the coverage factor alone equals log(0.5) at the
    midpoint -- is preserved by DECOMPOSITION rather than deleted: the total
    minus the progress factor must still be exactly log(0.5).
    """

    cfg = PainterConfig(gap_progress_stop_enabled=True)
    stop = Policy((StrokeAction.stop_action(),))
    belief = _observed_gap_belief(cfg, 0.30)
    progress = belief.stop_log_prior_term(cfg)

    total = policy_stop_log_prior(stop, cfg.minimum_stop_coverage, cfg, belief)
    assert progress < 0.0
    assert total - progress == pytest.approx(float(np.log(0.5)))
    assert total <= 0.0


def test_gap_progress_factor_rises_as_believed_progress_falls() -> None:
    cfg = PainterConfig(gap_progress_stop_enabled=True)
    stop = Policy((StrokeAction.stop_action(),))
    values = [
        policy_stop_log_prior(stop, 0.5, cfg, _observed_gap_belief(cfg, increment))
        for increment in (0.4, 0.2, 0.05, 0.0, -0.2)
    ]
    assert all(value <= 0.0 for value in values)
    assert values == sorted(values), values


def test_continuations_stay_exactly_flat_with_a_fully_observed_gap_belief() -> None:
    cfg = PainterConfig(gap_progress_stop_enabled=True)
    continuation = Policy(
        (StrokeAction(0.1, 0.1, 0.9, 0.9, 0.08, 0.5, 1.0), StrokeAction.stop_action())
    )
    belief = _observed_gap_belief(cfg, 0.4)
    assert policy_stop_log_prior(continuation, 0.05, cfg, belief) == 0.0
    assert policy_stop_log_prior(continuation, 0.95, cfg, belief) == 0.0


def test_stop_stays_admissible_at_every_coverage_with_the_progress_factor() -> None:
    # The progress factor is bounded above by 0 and below by a finite value, so
    # it can never veto stopping -- only make it a priori less likely.
    cfg = PainterConfig(gap_progress_stop_enabled=True)
    stop = Policy((StrokeAction.stop_action(),))
    belief = _observed_gap_belief(cfg, 5.0)
    for coverage in (0.0, 0.05, 0.5, cfg.minimum_stop_coverage, 0.9, 1.0):
        value = policy_stop_log_prior(stop, coverage, cfg, belief)
        assert np.isfinite(value)
        assert value <= 0.0
