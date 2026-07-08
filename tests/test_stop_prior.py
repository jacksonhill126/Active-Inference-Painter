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
