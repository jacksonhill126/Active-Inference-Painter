from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from .action_encoding import encoded_action_vector
from .config import PainterConfig
from .efe import EFEComponents, ExpectedFreeEnergy
from .env import PaintCanvasEnv, StrokeAction
from .inference import VariationalStateEstimator
from .models import DynamicsEnsemble, GaussianBelief, ObservationModel
from .policies import (
    MotorPrimitiveLatent,
    Policy,
    PolicySampler,
    policy_posterior_from_efe,
    policy_stop_log_prior,
)
from .precision_beliefs import (
    POLICY_PRECISION_KEY,
    GapIncrementBelief,
    PrecisionLedger,
)
from .preferences import TerminalCoveragePreference
from .replay import ReplayBuffer


class ActiveInferencePainter:
    def __init__(
        self,
        config: PainterConfig,
        seed: int = 0,
        device: str | None = None,
        precision_ledger: PrecisionLedger | None = None,
        gap_increment: GapIncrementBelief | None = None,
    ) -> None:
        self.cfg = config
        self.precision_ledger = precision_ledger if precision_ledger is not None else PrecisionLedger(config)
        self.gap_increment = (
            gap_increment if gap_increment is not None else GapIncrementBelief.from_config(config)
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.dynamics = DynamicsEnsemble(config).to(self.device)
        self.observation_model = ObservationModel(config).to(self.device)
        self.estimator = VariationalStateEstimator(config, self.observation_model)
        self.preference = TerminalCoveragePreference(config)
        self.efe = ExpectedFreeEnergy(
            config,
            self.dynamics,
            self.observation_model,
            self.preference,
            precision_ledger=self.precision_ledger,
        )
        # No learned proposal here, by declared config rather than by omission:
        # the summary planner has no canvas or relational posterior to condition
        # `q_proposal(z_pi | belief)` on, so it keeps the hand-written proposal and
        # the driver reports `policyProposal.status = "unavailable_summary_planner"`.
        # `config.learned_proposal_enabled` is gated on the spatial planner.
        self.policy_sampler = PolicySampler(config, seed=seed)
        self.replay = ReplayBuffer(config.replay_capacity, seed=seed)
        self.optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=config.model_lr)
        self.belief = GaussianBelief(
            torch.zeros(config.state_dim, device=self.device),
            torch.full((config.state_dim,), -4.5, device=self.device),
        )

    def reset_episode_prior(self) -> None:
        """Discard the painting-local posterior without touching learned state."""

        self.belief = GaussianBelief(
            torch.zeros(self.cfg.state_dim, device=self.device),
            torch.full((self.cfg.state_dim,), -4.5, device=self.device),
        )

    @property
    def last_vfe(self):
        return self.estimator.last_vfe

    def reset_belief(self, observation: np.ndarray) -> None:
        o = torch.tensor(observation, device=self.device)
        prior = GaussianBelief(o.clone(), torch.full_like(o, -4.0))
        self.belief = self.estimator.infer(prior, o)

    def update_belief(
        self,
        previous_action: StrokeAction,
        observation: np.ndarray,
        motor_primitive: MotorPrimitiveLatent | None = None,
    ) -> None:
        a = torch.from_numpy(encoded_action_vector(previous_action, self.cfg, motor_primitive)).to(self.device).unsqueeze(0)
        with torch.no_grad():
            mean, aleatoric, epistemic = self.dynamics.predictive_moments(self.belief.mean.unsqueeze(0), a)
        prior = GaussianBelief(mean.squeeze(0), torch.log(torch.clamp(aleatoric + epistemic, min=1e-7)).squeeze(0))
        o = torch.tensor(observation, device=self.device)
        self.belief = self.estimator.infer(prior, o)

    def infer_policy(self) -> tuple[Policy, EFEComponents, list[tuple[Policy, EFEComponents, float]]]:
        """Sample candidates, score them, and draw from Q(pi).

        The policy precision is the posterior mean of the ledger's Gamma belief.
        Its prior mean is exactly `config.policy_precision`, and this bare-agent
        path deliberately does NOT call `observe_policy`: there is no
        policy-dependent variational free energy here (`last_vfe` is a
        state-inference VFE for the single realized observation, not a
        per-candidate vector). Measured on the reference implementation, a flat
        or absent F makes dF/dgamma exactly 0.0 while reporting
        converged=True -- a degenerate no-op dressed as a fixed point. So this
        path stays bit-identically at the declared prior mean and says so,
        rather than passing F=0 and calling the result learning.
        """

        policies = self.policy_sampler.sample()
        components = self.efe.evaluate_batch(self.belief, policies)
        g = torch.tensor([c.total for c in components], device=self.device)
        believed_coverage = float(self.belief.mean[0].item())
        log_prior = torch.tensor(
            [
                policy_stop_log_prior(policy, believed_coverage, self.cfg, self.gap_increment)
                for policy in policies
            ],
            device=self.device,
        )
        posterior = policy_posterior_from_efe(
            g,
            log_prior,
            self.precision_ledger.mean(POLICY_PRECISION_KEY),
        )
        index = int(torch.multinomial(posterior, 1).item())
        ranked = sorted(
            zip(policies, components, posterior.detach().cpu().tolist()),
            key=lambda item: item[2],
            reverse=True,
        )
        return policies[index], components[index], ranked

    def train_dynamics(self, gradient_steps: int = 1) -> float | None:
        if len(self.replay) < self.cfg.batch_size:
            return None
        total = 0.0
        for _ in range(gradient_steps):
            batch = self.replay.sample(self.cfg.batch_size, self.device)
            loss = self.dynamics.nll(batch.state, batch.action, batch.next_state)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 5.0)
            self.optimizer.step()
            total += float(loss.item())
        return total / gradient_steps

    def collect_random_transition(self, env: PaintCanvasEnv, action: StrokeAction) -> None:
        state = env.latent_state().copy()
        observation, done, _ = env.step(action)
        next_state = env.latent_state().copy()
        self.replay.add(state, encoded_action_vector(action, self.cfg), next_state)
        if done:
            env.reset()
        _ = observation

    @staticmethod
    def policy_dict(policy: Policy) -> list[dict[str, float | bool]]:
        return [asdict(a) for a in policy.actions]
