from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.distributions import Normal, kl_divergence

from .config import PainterConfig
from .models import GaussianBelief, ObservationModel


@dataclass(frozen=True, slots=True)
class VFEComponents:
    total: float
    complexity: float
    negative_log_likelihood: float
    expected_log_likelihood: float
    units: str
    approximation: str = ""


class VariationalStateEstimator:
    """Infer q(s_t) by minimizing variational free energy."""

    def __init__(self, config: PainterConfig, observation_model: ObservationModel) -> None:
        self.cfg = config
        self.observation_model = observation_model
        self.last_vfe: VFEComponents | None = None

    def infer(self, prior: GaussianBelief, observation: torch.Tensor) -> GaussianBelief:
        mean = prior.mean.detach().clone().requires_grad_(True)
        logvar = prior.logvar.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([mean, logvar], lr=self.cfg.inference_lr)

        for _ in range(self.cfg.inference_steps):
            q = Normal(mean, torch.exp(0.5 * logvar))
            prior_dist = Normal(prior.mean.detach(), torch.exp(0.5 * prior.logvar.detach()))
            samples = q.rsample((8,))
            log_likelihood = self.observation_model.distribution(samples).log_prob(observation).sum(dim=-1).mean()
            complexity = kl_divergence(q, prior_dist).sum()
            free_energy = complexity - log_likelihood
            optimizer.zero_grad()
            free_energy.backward()
            optimizer.step()
            with torch.no_grad():
                logvar.clamp_(-9.0, 1.0)

        with torch.no_grad():
            q = Normal(mean, torch.exp(0.5 * logvar))
            prior_dist = Normal(prior.mean.detach(), torch.exp(0.5 * prior.logvar.detach()))
            samples = q.rsample((32,))
            expected_log_likelihood = self.observation_model.distribution(samples).log_prob(observation).sum(dim=-1).mean()
            complexity = kl_divergence(q, prior_dist).sum()
            negative_log_likelihood = -expected_log_likelihood
            total = complexity + negative_log_likelihood
            self.last_vfe = VFEComponents(
                total=float(total.item()),
                complexity=float(complexity.item()),
                negative_log_likelihood=float(negative_log_likelihood.item()),
                expected_log_likelihood=float(expected_log_likelihood.item()),
                units="nats",
                approximation="Monte Carlo expectation with 32 posterior state samples",
            )

        return GaussianBelief(mean.detach(), logvar.detach())


def normal_entropy_from_variance(variance: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.log(2.0 * math.pi * math.e * torch.clamp(variance, min=1e-9)).sum(dim=-1)
