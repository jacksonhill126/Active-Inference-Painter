from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from .action_encoding import coerce_action_raster, coerce_action_tensor
from .config import PainterConfig
from .efe_common import project_material_support


def _bootstrap_masked_nll(per_element_nll: torch.Tensor, keep_probability: float) -> torch.Tensor:
    """Reduce a per-member, per-sample NLL under Bernoulli bootstrap masks.

    `per_element_nll` has shape [members, batch]. Each member sees its own
    random subset of the batch so ensemble members stay dispersed and their
    disagreement remains usable as an approximate parameter posterior. A
    member whose mask comes up empty falls back to the full batch.
    """

    if keep_probability >= 1.0:
        return per_element_nll.mean()
    mask = torch.rand(per_element_nll.shape, device=per_element_nll.device) < keep_probability
    mask = mask | (mask.sum(dim=1, keepdim=True) == 0)
    mask = mask.to(per_element_nll.dtype)
    return (per_element_nll * mask).sum() / mask.sum().clamp(min=1.0)


class TransitionMember(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(torch.cat([state, action], dim=-1))
        delta_mean, raw_logvar = raw.chunk(2, dim=-1)
        logvar = -11.0 + 6.0 * torch.sigmoid(raw_logvar)
        return state + delta_mean, logvar


class DynamicsEnsemble(nn.Module):
    """Learned transition density p_theta(s[t+1] | s[t], a[t])."""

    def __init__(self, config: PainterConfig) -> None:
        super().__init__()
        self.bootstrap_probability = config.ensemble_bootstrap_probability
        self.action_dim = config.action_dim
        self.members = nn.ModuleList(
            [
                TransitionMember(config.state_dim, config.action_dim, config.hidden_dim)
                for _ in range(config.ensemble_size)
            ]
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action = coerce_action_tensor(action, self.action_dim)
        means, logvars = zip(*(m(state, action) for m in self.members))
        return torch.stack(means, dim=0), torch.stack(logvars, dim=0)

    def predictive_moments(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means, logvars = self(state, action)
        aleatoric = logvars.exp().mean(dim=0)
        epistemic = means.var(dim=0, unbiased=False)
        mean = means.mean(dim=0)
        return mean, aleatoric, epistemic

    def nll(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        means, logvars = self(state, action)
        target = next_state.unsqueeze(0).expand_as(means)
        nll = 0.5 * (((target - means) ** 2) / logvars.exp() + logvars)
        return _bootstrap_masked_nll(nll.mean(dim=-1), self.bootstrap_probability)


class SpatialResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

    def forward_masked(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x * mask
        hidden = self.net[0](residual) * mask
        hidden = self.net[1](hidden) * mask
        hidden = self.net[2](hidden) * mask
        return (residual + hidden) * mask


class SpatialTransitionMember(nn.Module):
    """CNN transition density for explicit spatial material fields."""

    def __init__(
        self,
        material_channels: int,
        action_channels: int,
        hidden_channels: int,
        residual_blocks: int,
        thickness_scale: float,
        ground_tone: float,
        paint_presence_threshold: float = 0.0001,
    ) -> None:
        super().__init__()
        input_channels = material_channels + action_channels
        blocks: list[nn.Module] = [
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        blocks.extend(SpatialResidualBlock(hidden_channels) for _ in range(residual_blocks))
        blocks.extend(
            [
                nn.SiLU(),
                nn.Conv2d(hidden_channels, 2 * material_channels, kernel_size=3, padding=1),
            ]
        )
        self.net = nn.Sequential(*blocks)
        self.material_channels = material_channels
        self.thickness_scale = thickness_scale
        self.ground_tone = ground_tone
        self.paint_presence_threshold = paint_presence_threshold

    def forward(self, material: torch.Tensor, action_raster: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(torch.cat([material, action_raster], dim=1))
        return self._distribution_from_raw(material, raw)

    def forward_masked(
        self,
        material: torch.Tensor,
        action_raster: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = mask.to(device=material.device, dtype=material.dtype)
        hidden = torch.cat([material * mask, action_raster * mask], dim=1)
        for layer in self.net:
            if isinstance(layer, SpatialResidualBlock):
                hidden = layer.forward_masked(hidden, mask)
            else:
                hidden = layer(hidden) * mask
        next_mean, logvar = self._distribution_from_raw(material * mask, hidden)
        return next_mean * mask, logvar * mask

    def _distribution_from_raw(
        self,
        material: torch.Tensor,
        raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_mean, raw_logvar = raw.chunk(2, dim=1)
        next_mean = self._project_material_support(
            material,
            material + delta_mean,
            self.thickness_scale,
            self.ground_tone,
            self.paint_presence_threshold,
        )
        logvar = -11.0 + 6.0 * torch.sigmoid(raw_logvar)
        return next_mean, logvar

    @staticmethod
    def _project_material_support(
        current: torch.Tensor,
        proposed: torch.Tensor,
        thickness_scale: float = 0.005,
        ground_tone: float = 0.34,
        paint_presence_threshold: float = 0.0001,
    ) -> torch.Tensor:
        return project_material_support(
            current,
            proposed,
            thickness_scale,
            ground_tone,
            paint_presence_threshold,
        )


class SpatialDynamicsEnsemble(nn.Module):
    """Learned p_theta(s_grid[t+1] | s_grid[t], a_raster[t]).

    This model is deliberately spatial and material-only: it predicts local
    thickness, wetness, conserved pigment mass, and surface optics. It does not
    contain aesthetic reward terms or composition heuristics.
    """

    def __init__(self, config: PainterConfig) -> None:
        super().__init__()
        self.bootstrap_probability = config.ensemble_bootstrap_probability
        self.action_channels = config.spatial_action_channels
        self.members = nn.ModuleList(
            [
                SpatialTransitionMember(
                    config.spatial_material_channels,
                    config.spatial_action_channels,
                    config.spatial_hidden_channels,
                    config.spatial_residual_blocks,
                    config.thickness_scale,
                    config.canvas_ground_tone,
                    config.paint_presence_threshold,
                )
                for _ in range(config.spatial_ensemble_size)
            ]
        )

    def forward(self, material: torch.Tensor, action_raster: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_raster = coerce_action_raster(action_raster, self.action_channels)
        means, logvars = zip(*(member(material, action_raster) for member in self.members))
        return torch.stack(means, dim=0), torch.stack(logvars, dim=0)

    def forward_masked(
        self,
        material: torch.Tensor,
        action_raster: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_raster = coerce_action_raster(action_raster, self.action_channels)
        means, logvars = zip(
            *(member.forward_masked(material, action_raster, mask) for member in self.members)
        )
        return torch.stack(means, dim=0), torch.stack(logvars, dim=0)

    def predictive_moments(
        self,
        material: torch.Tensor,
        action_raster: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means, logvars = self(material, action_raster)
        aleatoric = logvars.exp().mean(dim=0)
        epistemic = means.var(dim=0, unbiased=False)
        mean = means.mean(dim=0)
        return mean, aleatoric, epistemic

    def nll(
        self,
        material: torch.Tensor,
        action_raster: torch.Tensor,
        next_material: torch.Tensor,
    ) -> torch.Tensor:
        means, logvars = self(material, action_raster)
        target = next_material.unsqueeze(0).expand_as(means)
        nll = 0.5 * (((target - means) ** 2) / logvars.exp() + logvars)
        return _bootstrap_masked_nll(nll.mean(dim=(2, 3, 4)), self.bootstrap_probability)


class LocalSpatialDynamicsEnsemble(SpatialDynamicsEnsemble):
    """Local patch transition density p_theta(s_patch_next | s_patch, a_patch).

    The network architecture is convolutional and therefore accepts variable
    patch sizes. Training batches are padded; `mask` declares which cells are
    real observations so padding never becomes evidence in the likelihood.
    """

    def nll(
        self,
        material: torch.Tensor,
        action_raster: torch.Tensor,
        next_material: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        per_sample = self.per_sample_nll(material, action_raster, next_material, mask)
        bootstrap_mask = self.sample_bootstrap_mask(
            per_sample.shape[1],
            per_sample.device,
            per_sample.dtype,
        )
        return (per_sample * bootstrap_mask).sum() / bootstrap_mask.sum().clamp(min=1.0)

    def per_sample_nll(
        self,
        material: torch.Tensor,
        action_raster: torch.Tensor,
        next_material: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        material = material * mask
        action_raster = action_raster * mask
        next_material = next_material * mask
        means, logvars = self.forward_masked(material, action_raster, mask)
        target = next_material.unsqueeze(0).expand_as(means)
        nll = 0.5 * (((target - means) ** 2) / logvars.exp() + logvars)
        valid = mask.to(nll.dtype).unsqueeze(0)
        if valid.ndim == 5 and valid.shape[2] == 1:
            valid = valid.expand(-1, -1, nll.shape[2], -1, -1)
        valid_count = valid.sum(dim=(2, 3, 4)).clamp(min=1.0)
        return (nll * valid).sum(dim=(2, 3, 4)) / valid_count

    def sample_bootstrap_mask(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        shape = (len(self.members), int(batch_size))
        if self.bootstrap_probability >= 1.0:
            return torch.ones(shape, device=device, dtype=dtype)
        mask = torch.rand(shape, device=device) < self.bootstrap_probability
        mask = mask | (mask.sum(dim=1, keepdim=True) == 0)
        return mask.to(dtype)


class ObservationModel(nn.Module):
    """Explicit likelihood p(o | s) with context-dependent ambiguity."""

    def __init__(self, config: PainterConfig) -> None:
        super().__init__()
        self.cfg = config

    def std(self, state: torch.Tensor) -> torch.Tensor:
        smear = torch.clamp(0.55 * state[..., 1] + 0.75 * state[..., 3] + 0.35 * state[..., 4], 0.0, 2.0)
        scales = torch.tensor([0.55, 0.7, 1.0, 0.8, 0.8, 0.9], device=state.device, dtype=state.dtype)
        return self.cfg.base_observation_std + self.cfg.smear_observation_std * smear.unsqueeze(-1) * scales

    def distribution(self, state: torch.Tensor) -> Normal:
        return Normal(state, self.std(state))

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        return self.distribution(state).entropy().sum(dim=-1)

    def ambiguity(self, state: torch.Tensor) -> torch.Tensor:
        """Observation ambiguity above the dry-canvas likelihood baseline.

        Differential entropy depends on the units of continuous observations.
        The planner therefore uses the excess entropy induced by wet/thick paint
        instead of rewarding extra policy steps with a negative base entropy.
        """

        entropy = self.entropy(state)
        base_std = torch.full_like(state, self.cfg.base_observation_std)
        base_entropy = Normal(state, base_std).entropy().sum(dim=-1)
        return torch.clamp(entropy - base_entropy, min=0.0)


@dataclass(slots=True)
class GaussianBelief:
    mean: torch.Tensor
    logvar: torch.Tensor
