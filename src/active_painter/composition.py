"""Hierarchical composition layer: a compression-gap preference.

This is the single declared structural prior over terminal canvases:
p*(s_T) is proportional to exp(kappa * gap(s_T)), where

    gap(s) = ELBO_hierarchical(s) - max_m log p_m(s)

in nats per cell-channel. The hierarchical code is a latent composition
variable z with a learned decoder over the spatial material fields. The
opponent is a declared, hand-written, parameter-free BASELINE FAMILY, and the
gap is measured against the BEST member, so it only credits structure that NO
member of the family can explain:

- the iid member: the best per-image, per-channel Gaussian (context-free);
- the local member: a 3x3 hollow-neighbourhood Markov code. Each cell is
  predicted by the mean of its eight neighbours EXCLUDING itself, with
  replicate padding, per-image per-channel residual variance, and no learned
  parameters at all.

Every code here (the ELBO decoder and both baseline members) shares one
quantization floor (SIGMA_FLOOR) so none earns free nats from continuous-
density resolution. The hierarchy pays for its latent code via the KL term,
so:

- a blank or spatially constant canvas scores <= 0 (both members sit exactly
  on the shared floor, so the baseline is already perfect);
- iid noise scores <= 0 (nothing local or global to exploit; the iid member
  wins the family max);
- a locally smooth but globally unstructured canvas (a soft blob, low-pass
  noise) scores <= 0 (the local member explains it, so the hierarchy gets no
  credit for mere smoothness);
- only canvases whose distant parts predict each other score positive, once
  the hierarchy has learned their regularities.

The family maximum is a best-of-family code, not a normalized mixture: it
omits the log(family size) model-index cost. See the approximation register in
docs/GENERATIVE_MODEL_SPEC.md.

No content term appears anywhere: the preference references only how well the
agent's own hierarchical model explains the canvas beyond a fixed,
hand-written family of context-free and local codes.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from .config import PainterConfig

SIGMA_FLOOR = 0.02
LOGVAR_FLOOR = 2.0 * math.log(SIGMA_FLOOR)

# Both baseline members and the ELBO decoder are clamped to the same
# quantization floor, so no code can exceed this per-cell-channel ceiling.
# A blank canvas puts every member exactly on it, which is why the gap there
# is bounded above by zero minus the hierarchy's latent KL.
BASELINE_CEILING_NATS = -0.5 * math.log(2.0 * math.pi * SIGMA_FLOOR**2)
IID_BASELINE_MEMBER = "per-image per-channel iid-cell Gaussian"
LOCAL_BASELINE_MEMBER = "parameter-free 3x3 hollow-neighbourhood local Markov Gaussian"
BASELINE_FAMILY_MEMBERS = (IID_BASELINE_MEMBER, LOCAL_BASELINE_MEMBER)


def flat_log_likelihood(fields: torch.Tensor) -> torch.Tensor:
    """Baseline family member 1: best context-free iid-cell Gaussian.

    A genuine normalized density over the field, not a score: per-image
    per-channel mean and variance, variance floored at the shared quantization
    floor so blank canvases are coded exactly as well as a perfect
    hierarchical reconstruction, not infinitely better.
    """

    mean = fields.mean(dim=(2, 3), keepdim=True)
    variance = torch.clamp(fields.var(dim=(2, 3), unbiased=False, keepdim=True), min=SIGMA_FLOOR**2)
    return Normal(mean, torch.sqrt(variance)).log_prob(fields).mean(dim=(1, 2, 3))


def _hollow_neighbourhood_mean(fields: torch.Tensor) -> torch.Tensor:
    """Mean of the eight 3x3 neighbours, EXCLUDING the centre cell.

    Hollow (centre weight zero) so this is a genuine predictive code and not
    the identity: a code that could see the cell it predicts would assign it
    unbounded density and could not be a legitimate opponent. groups=channels
    so channels never mix, which means the deterministic material channels
    cannot leak predictive information about the independent ones. Replicate
    padding, so no artificial border edge is manufactured.

    The kernel is built inline from the input's dtype/device rather than
    registered as a buffer: a buffer would add a state_dict key and break
    strict loading of existing checkpoints.
    """

    channels = fields.shape[-3]
    kernel = torch.full((3, 3), 0.125, dtype=fields.dtype, device=fields.device)
    kernel[1, 1] = 0.0
    kernel = kernel.reshape(1, 1, 3, 3).expand(channels, 1, 3, 3)
    padded = F.pad(fields, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(padded, kernel, groups=channels)


def local_smoothness_log_likelihood(fields: torch.Tensor) -> torch.Tensor:
    """Baseline family member 2: fixed parameter-free local Markov code.

    Predicts each cell from its hollow 3x3 neighbourhood and codes the
    residual with a per-image per-channel variance floored at the same
    SIGMA_FLOOR the iid member and the ELBO decoder use. No learned or fitted
    parameters: the only image-dependent quantity is the residual variance,
    which the iid member is also allowed. This is a likelihood, log p_local(s),
    with every cell's Gaussian individually normalized.
    """

    prediction = _hollow_neighbourhood_mean(fields)
    residual = fields - prediction
    variance = torch.clamp(
        residual.var(dim=(2, 3), unbiased=False, keepdim=True), min=SIGMA_FLOOR**2
    )
    return Normal(prediction, torch.sqrt(variance)).log_prob(fields).mean(dim=(1, 2, 3))


def baseline_log_likelihood(fields: torch.Tensor, *, local_enabled: bool = True) -> torch.Tensor:
    """Best-of-family context-free baseline, nats per cell-channel.

    The elementwise (per-image) maximum over the declared family, so the
    compression gap only credits structure NO member can explain. This is the
    reference measure of the composition preference: it is never itself
    preferred, minimized, or fit to outcomes. With `local_enabled=False` the
    family is the iid member alone, which is the pre-baseline-family behaviour
    exactly.
    """

    flat = flat_log_likelihood(fields)
    if not local_enabled:
        return flat
    return torch.maximum(flat, local_smoothness_log_likelihood(fields))


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
        # Declared baseline-family membership. A plain attribute, never a
        # buffer, so no state_dict key is added and existing checkpoints keep
        # loading. This selects a reference code; it learns nothing.
        self.local_baseline_enabled = bool(config.composition_local_baseline_enabled)

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
        """Baseline family member 1, exposed for diagnostics.

        Delegates to the module-level function, which is the single shared
        implementation for this class and HierarchicalCanvasModel. (Inside a
        method body the bare name resolves to the module global, not the class
        attribute: Python method bodies do not see the class namespace.)
        """

        return flat_log_likelihood(fields)

    @staticmethod
    def local_smoothness_log_likelihood(fields: torch.Tensor) -> torch.Tensor:
        """Baseline family member 2, exposed for diagnostics."""

        return local_smoothness_log_likelihood(fields)

    def baseline_log_likelihood(self, fields: torch.Tensor) -> torch.Tensor:
        """Best-of-family baseline under this model's declared family."""

        return baseline_log_likelihood(fields, local_enabled=self.local_baseline_enabled)

    @torch.no_grad()
    def compression_gap(self, fields: torch.Tensor) -> torch.Tensor:
        """gap(s) = ELBO_hier(s) - max_m log p_m(s), nats per cell-channel.

        Evaluated under no_grad: the family maximum is non-differentiable at
        member ties (a blank canvas sits exactly on a tie), so this must never
        be differentiated.
        """

        return self.elbo(fields, sample=False) - self.baseline_log_likelihood(fields)

    def training_loss(self, fields: torch.Tensor) -> torch.Tensor:
        return -self.elbo(fields, sample=True).mean()
