"""Hierarchical composition layer: a compression-gap preference.

This is the single declared structural prior over terminal canvases:
p*(s_T) is proportional to exp(kappa * gap(s_T)), where

    gap(s) = ELBO_hierarchical(s) - log p_flat(s)

in nats per cell-channel. The hierarchical code is a latent composition
variable z with a learned decoder over the spatial material fields; the flat
code is the best context-free iid-cell Gaussian for that specific image. Both
codes share one quantization floor (SIGMA_FLOOR) so neither earns free nats
from continuous-density resolution. The hierarchy pays for its latent code via
the KL term, so:

- a blank canvas scores ~0 (the flat code is already perfect);
- iid noise scores <= 0 (nothing spatial to exploit);
- canvases whose parts predict each other score positive once the hierarchy
  has learned their regularities.

No content term appears anywhere: the preference references only how well the
agent's own hierarchical model explains the canvas beyond a flat code.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from .config import PainterConfig

SIGMA_FLOOR = 0.02
LOGVAR_FLOOR = 2.0 * math.log(SIGMA_FLOOR)


class CompositionHierarchy(nn.Module):
    """Latent composition code over spatial material fields."""

    def __init__(self, config: PainterConfig) -> None:
        super().__init__()
        grid = config.spatial_grid_size
        if grid % 4 != 0:
            raise ValueError("composition hierarchy requires spatial_grid_size divisible by 4.")
        channels = config.spatial_material_channels
        hidden = config.composition_hidden_channels
        latent_dim = config.composition_latent_dim
        self.grid = grid
        self.channels = channels
        self.hidden = hidden
        self.cell_count = float(channels * grid * grid)

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 2 * hidden, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
        )
        bottleneck = 2 * hidden * (grid // 4) ** 2
        self.to_mu = nn.Linear(bottleneck, latent_dim)
        self.to_logvar = nn.Linear(bottleneck, latent_dim)
        self.from_latent = nn.Linear(latent_dim, bottleneck)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2 * hidden, hidden, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden, hidden, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 2 * channels, kernel_size=3, padding=1),
        )

    def encode(self, fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bottleneck = self.encoder(fields)
        return self.to_mu(bottleneck), torch.clamp(self.to_logvar(bottleneck), -9.0, 2.0)

    def decode(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grid4 = self.grid // 4
        raw = self.decoder(self.from_latent(latent).reshape(-1, 2 * self.hidden, grid4, grid4))
        mean, raw_logvar = raw.chunk(2, dim=1)
        return mean, torch.clamp(raw_logvar, LOGVAR_FLOOR, 2.0)

    def _reconstruction_log_likelihood(self, fields: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.decode(latent)
        return Normal(mean, torch.exp(0.5 * logvar)).log_prob(fields).mean(dim=(1, 2, 3))

    @staticmethod
    def _latent_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(dim=-1)

    def elbo(self, fields: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Evidence lower bound in nats per cell-channel."""

        mu, logvar = self.encode(fields)
        if sample:
            latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            latent = mu
        reconstruction = self._reconstruction_log_likelihood(fields, latent)
        return reconstruction - self._latent_kl(mu, logvar) / self.cell_count

    @staticmethod
    def flat_log_likelihood(fields: torch.Tensor) -> torch.Tensor:
        """Best context-free iid-cell Gaussian code for each image.

        Per-image per-channel mean and variance, variance floored at the
        shared quantization floor so blank canvases are coded exactly as well
        as a perfect hierarchical reconstruction, not infinitely better.
        """

        mean = fields.mean(dim=(2, 3), keepdim=True)
        variance = torch.clamp(fields.var(dim=(2, 3), unbiased=False, keepdim=True), min=SIGMA_FLOOR**2)
        return Normal(mean, torch.sqrt(variance)).log_prob(fields).mean(dim=(1, 2, 3))

    @torch.no_grad()
    def compression_gap(self, fields: torch.Tensor) -> torch.Tensor:
        """gap(s) = ELBO_hier(s) - log p_flat(s), nats per cell-channel."""

        return self.elbo(fields, sample=False) - self.flat_log_likelihood(fields)

    def training_loss(self, fields: torch.Tensor) -> torch.Tensor:
        return -self.elbo(fields, sample=True).mean()
