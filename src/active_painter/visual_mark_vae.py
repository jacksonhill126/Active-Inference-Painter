"""Action-conditioned visual mark-consequence VAE.

The model learns a normalized image likelihood for a registered post-action
camera crop conditioned on the fresh pre-action crop, the selected mark,
conditional motor realization, camera identity, and compact brush belief.
The posterior encoder may see the post image during training; counterfactual
prediction samples only from the learned conditional prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .env import mark_path_points
from .visual_trajectory_corpus import (
    VisualTrajectoryShard,
    load_visual_trajectory_shard,
)


VISUAL_MARK_VAE_MODEL_ID = "action-conditioned-visual-mark-cvae-v0"
VISUAL_MARK_VAE_STATUS = "provisional_simulation_only_not_hardware_calibrated"
VISUAL_MOTOR_KINDS = (
    "cartesian_ik",
    "upper_arm_fixed_roll_positive",
    "upper_arm_fixed_roll_negative",
    "joint_spline",
    "elbow_pivot",
    "upper_arm_roll_positive",
    "upper_arm_roll_negative",
)


@dataclass(frozen=True, slots=True)
class VisualMarkVAEConfig:
    patch_size: int = 64
    latent_dim: int = 16
    base_channels: int = 24
    condition_channels: int = 16
    camera_names: tuple[str, ...] = (
        "camera_canvas_left",
        "camera_canvas_right",
    )
    motor_kinds: tuple[str, ...] = VISUAL_MOTOR_KINDS
    crop_min_span: float = 0.24
    crop_width_context: float = 4.0
    likelihood_epsilon: float = 1.0e-4
    minimum_beta_concentration: float = 2.0
    maximum_beta_concentration: float = 200.0

    def __post_init__(self) -> None:
        if self.patch_size < 16 or self.patch_size % 16:
            raise ValueError("patch_size must be at least 16 and divisible by 16")
        if self.latent_dim <= 0 or self.base_channels <= 0:
            raise ValueError("latent_dim and base_channels must be positive")
        if not 0.0 < self.crop_min_span <= 1.0:
            raise ValueError("crop_min_span must lie in (0, 1]")
        if self.crop_width_context <= 0.0:
            raise ValueError("crop_width_context must be positive")
        if not 0.0 < self.likelihood_epsilon < 0.5:
            raise ValueError("likelihood_epsilon must lie in (0, 0.5)")
        if self.minimum_beta_concentration <= 0.0:
            raise ValueError("minimum_beta_concentration must be positive")
        if self.maximum_beta_concentration <= self.minimum_beta_concentration:
            raise ValueError("maximum concentration must exceed minimum")

    @property
    def condition_dim(self) -> int:
        # Crop-local stroke (8), brush belief (5), camera and motor one-hots,
        # plus crop center (2) and span (1).
        return 16 + len(self.camera_names) + len(self.motor_kinds)


@dataclass(frozen=True, slots=True)
class VisualMarkExample:
    before: np.ndarray
    after: np.ndarray
    validity: np.ndarray
    condition: np.ndarray
    trajectory_id: str
    transition_index: int
    camera_name: str


@dataclass(frozen=True, slots=True)
class VisualMarkBatch:
    before: torch.Tensor
    after: torch.Tensor
    validity: torch.Tensor
    condition: torch.Tensor

    def with_condition(self, condition: torch.Tensor) -> "VisualMarkBatch":
        return VisualMarkBatch(
            before=self.before,
            after=self.after,
            validity=self.validity,
            condition=condition,
        )


@dataclass(frozen=True, slots=True)
class VisualVFEComponents:
    reconstruction_nll: torch.Tensor
    latent_kl: torch.Tensor
    free_energy: torch.Tensor
    free_energy_per_observed_pixel: torch.Tensor
    valid_pixel_count: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_logvar: torch.Tensor
    prior_mean: torch.Tensor
    prior_logvar: torch.Tensor
    predicted_mean: torch.Tensor
    predicted_concentration: torch.Tensor


def _bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = image.shape
    x = np.clip(u * (width - 1), 0.0, width - 1)
    y = np.clip(v * (height - 1), 0.0, height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    return (
        image[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + image[y0, x1] * wx * (1.0 - wy)
        + image[y1, x0] * (1.0 - wx) * wy
        + image[y1, x1] * wx * wy
    )


def _nearest_sample(mask: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    x = np.rint(np.clip(u, 0.0, 1.0) * (width - 1)).astype(np.int64)
    y = np.rint(np.clip(v, 0.0, 1.0) * (height - 1)).astype(np.int64)
    result = mask[y, x].astype(np.bool_)
    result &= (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    return result


def _crop_geometry(
    action: np.ndarray,
    config: VisualMarkVAEConfig,
) -> tuple[float, float, float]:
    # The sampled quadratic support preserves signed curvature when declaring
    # the observation crop.  This is an evidence-extraction operation, not a
    # policy preference or a painting score.
    from .env import StrokeAction

    stroke = StrokeAction(*[float(value) for value in action[:7]], curvature=float(action[7]))
    path = mark_path_points(stroke, np.linspace(0.0, 1.0, 65))
    minimum = path.min(axis=0)
    maximum = path.max(axis=0)
    width_context = max(float(action[4]) * config.crop_width_context, 0.04)
    span = max(
        config.crop_min_span,
        float(np.max(maximum - minimum)) + width_context,
    )
    span = min(1.0, span)
    center = 0.5 * (minimum + maximum)
    # Shift, rather than shrink, boundary crops so angles and widths retain the
    # same isotropic scale as interior crops.
    center = np.clip(center, 0.5 * span, 1.0 - 0.5 * span)
    return float(center[0]), float(center[1]), float(span)


def _extract_crop(
    image: np.ndarray,
    validity: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    span: float,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = (np.arange(patch_size, dtype=np.float64) + 0.5) / patch_size
    u = center_x + (coordinates - 0.5) * span
    v = center_y + (coordinates - 0.5) * span
    uu, vv = np.meshgrid(u, v)
    crop = _bilinear_sample(np.asarray(image, dtype=np.float32), uu, vv)
    mask = _nearest_sample(np.asarray(validity, dtype=np.bool_), uu, vv)
    return crop.astype(np.float32), mask


def _condition_vector(
    action: np.ndarray,
    brush: np.ndarray,
    camera_name: str,
    motor_kind: str,
    geometry: tuple[float, float, float],
    config: VisualMarkVAEConfig,
) -> np.ndarray:
    center_x, center_y, span = geometry
    local = np.asarray(action, dtype=np.float32).copy()
    local[0] = (local[0] - (center_x - 0.5 * span)) / span
    local[1] = (local[1] - (center_y - 0.5 * span)) / span
    local[2] = (local[2] - (center_x - 0.5 * span)) / span
    local[3] = (local[3] - (center_y - 0.5 * span)) / span
    local[4] = local[4] / span
    camera = np.zeros(len(config.camera_names), dtype=np.float32)
    if camera_name in config.camera_names:
        camera[config.camera_names.index(camera_name)] = 1.0
    motor = np.zeros(len(config.motor_kinds), dtype=np.float32)
    if motor_kind in config.motor_kinds:
        motor[config.motor_kinds.index(motor_kind)] = 1.0
    elif "cartesian_ik" in config.motor_kinds:
        motor[config.motor_kinds.index("cartesian_ik")] = 1.0
    return np.concatenate(
        [
            local,
            np.asarray(brush, dtype=np.float32),
            camera,
            motor,
            np.asarray((center_x, center_y, span), dtype=np.float32),
        ]
    ).astype(np.float32)


def visual_mark_examples_from_shard(
    shard: VisualTrajectoryShard,
    config: VisualMarkVAEConfig,
) -> list[VisualMarkExample]:
    examples: list[VisualMarkExample] = []
    for index in range(shard.example_count):
        action = shard.action[index]
        geometry = _crop_geometry(action, config)
        before, before_valid = _extract_crop(
            shard.pre_image[index],
            shard.pre_validity[index],
            center_x=geometry[0],
            center_y=geometry[1],
            span=geometry[2],
            patch_size=config.patch_size,
        )
        after, after_valid = _extract_crop(
            shard.post_image[index],
            shard.post_validity[index],
            center_x=geometry[0],
            center_y=geometry[1],
            span=geometry[2],
            patch_size=config.patch_size,
        )
        examples.append(
            VisualMarkExample(
                before=before[None, ...],
                after=after[None, ...],
                validity=(before_valid & after_valid)[None, ...].astype(np.float32),
                condition=_condition_vector(
                    action,
                    shard.brush_condition[index],
                    str(shard.camera_name[index]),
                    str(shard.motor_kind[index]),
                    geometry,
                    config,
                ),
                trajectory_id=shard.trajectory_id,
                transition_index=int(shard.transition_index[index]),
                camera_name=str(shard.camera_name[index]),
            )
        )
    return examples


def visual_mark_examples_from_paths(
    paths: Sequence[Path | str],
    config: VisualMarkVAEConfig,
) -> list[VisualMarkExample]:
    return [
        example
        for path in paths
        for example in visual_mark_examples_from_shard(
            load_visual_trajectory_shard(path), config
        )
    ]


def make_visual_mark_batch(
    examples: Sequence[VisualMarkExample],
    device: torch.device | str,
) -> VisualMarkBatch:
    if not examples:
        raise ValueError("cannot make an empty visual mark batch")
    return VisualMarkBatch(
        before=torch.as_tensor(
            np.stack([item.before for item in examples]),
            dtype=torch.float32,
            device=device,
        ),
        after=torch.as_tensor(
            np.stack([item.after for item in examples]),
            dtype=torch.float32,
            device=device,
        ),
        validity=torch.as_tensor(
            np.stack([item.validity for item in examples]),
            dtype=torch.float32,
            device=device,
        ),
        condition=torch.as_tensor(
            np.stack([item.condition for item in examples]),
            dtype=torch.float32,
            device=device,
        ),
    )


class _ImagePyramid(nn.Module):
    def __init__(self, input_channels: int, base: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(input_channels, base, 4, 2, 1), nn.SiLU()),
                nn.Sequential(nn.Conv2d(base, base * 2, 4, 2, 1), nn.SiLU()),
                nn.Sequential(nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.SiLU()),
                nn.Sequential(nn.Conv2d(base * 4, base * 4, 4, 2, 1), nn.SiLU()),
            ]
        )

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features: list[torch.Tensor] = []
        for layer in self.layers:
            value = layer(value)
            features.append(value)
        return tuple(features)


class VisualMarkVAE(nn.Module):
    """Conditional-prior VAE with a Beta observation likelihood."""

    def __init__(self, config: VisualMarkVAEConfig) -> None:
        super().__init__()
        self.config = config
        base = config.base_channels
        condition_channels = config.condition_channels
        self.condition_embedding = nn.Sequential(
            nn.Linear(config.condition_dim, 64),
            nn.SiLU(),
            nn.Linear(64, condition_channels),
        )
        self.prior_pyramid = _ImagePyramid(1 + condition_channels, base)
        self.posterior_pyramid = _ImagePyramid(2 + condition_channels, base)
        final_side = config.patch_size // 16
        flat_dim = base * 4 * final_side * final_side
        self.prior_head = nn.Linear(flat_dim, 2 * config.latent_dim)
        self.posterior_head = nn.Linear(flat_dim, 2 * config.latent_dim)
        self.latent_map = nn.Linear(config.latent_dim, base * 4 * final_side * final_side)
        self.decode4 = nn.Sequential(nn.Conv2d(base * 8, base * 4, 3, 1, 1), nn.SiLU())
        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 4, 2, 1)
        self.decode3 = nn.Sequential(nn.Conv2d(base * 8, base * 4, 3, 1, 1), nn.SiLU())
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1)
        self.decode2 = nn.Sequential(nn.Conv2d(base * 4, base * 2, 3, 1, 1), nn.SiLU())
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.decode1 = nn.Sequential(nn.Conv2d(base * 2, base, 3, 1, 1), nn.SiLU())
        self.up0 = nn.ConvTranspose2d(base, base, 4, 2, 1)
        self.output = nn.Sequential(
            nn.Conv2d(base + 1 + condition_channels, base, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(base, 2, 1),
        )

    def _condition_map(self, condition: torch.Tensor, side: int) -> torch.Tensor:
        embedded = self.condition_embedding(condition)
        return embedded[:, :, None, None].expand(-1, -1, side, side)

    def prior(self, before: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
        condition_map = self._condition_map(condition, before.shape[-1])
        features = self.prior_pyramid(torch.cat([before, condition_map], dim=1))
        moments = self.prior_head(features[-1].flatten(1))
        mean, logvar = moments.chunk(2, dim=1)
        return mean, logvar.clamp(-10.0, 6.0), features, condition_map

    def posterior(self, batch: VisualMarkBatch) -> tuple[torch.Tensor, torch.Tensor]:
        condition_map = self._condition_map(batch.condition, batch.before.shape[-1])
        features = self.posterior_pyramid(
            torch.cat([batch.before, batch.after, condition_map], dim=1)
        )
        moments = self.posterior_head(features[-1].flatten(1))
        mean, logvar = moments.chunk(2, dim=1)
        return mean, logvar.clamp(-10.0, 6.0)

    def decode(
        self,
        before: torch.Tensor,
        condition: torch.Tensor,
        latent: torch.Tensor,
        *,
        prior_features: tuple[torch.Tensor, ...] | None = None,
        condition_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prior_features is None or condition_map is None:
            _, _, prior_features, condition_map = self.prior(before, condition)
        final_side = self.config.patch_size // 16
        latent_map = self.latent_map(latent).reshape(
            latent.shape[0], self.config.base_channels * 4, final_side, final_side
        )
        value = self.decode4(torch.cat([prior_features[3], latent_map], dim=1))
        value = self.decode3(torch.cat([F.silu(self.up3(value)), prior_features[2]], dim=1))
        value = self.decode2(torch.cat([F.silu(self.up2(value)), prior_features[1]], dim=1))
        value = self.decode1(torch.cat([F.silu(self.up1(value)), prior_features[0]], dim=1))
        value = F.silu(self.up0(value))
        raw = self.output(torch.cat([value, before, condition_map], dim=1))
        epsilon = self.config.likelihood_epsilon
        baseline_logit = torch.logit(before.clamp(epsilon, 1.0 - epsilon))
        mean = torch.sigmoid(baseline_logit + 2.0 * torch.tanh(raw[:, :1]))
        concentration = F.softplus(raw[:, 1:2]) + self.config.minimum_beta_concentration
        concentration = concentration.clamp(max=self.config.maximum_beta_concentration)
        return mean.clamp(epsilon, 1.0 - epsilon), concentration

    def _beta_log_prob(
        self,
        target: torch.Tensor,
        mean: torch.Tensor,
        concentration: torch.Tensor,
    ) -> torch.Tensor:
        epsilon = self.config.likelihood_epsilon
        alpha = (mean * concentration).clamp(min=epsilon)
        beta = ((1.0 - mean) * concentration).clamp(min=epsilon)
        target = target.clamp(epsilon, 1.0 - epsilon)
        return torch.distributions.Beta(alpha, beta).log_prob(target)

    @staticmethod
    def _diagonal_gaussian_kl(
        q_mean: torch.Tensor,
        q_logvar: torch.Tensor,
        p_mean: torch.Tensor,
        p_logvar: torch.Tensor,
    ) -> torch.Tensor:
        return 0.5 * torch.sum(
            p_logvar - q_logvar
            + (q_logvar.exp() + (q_mean - p_mean).square()) / p_logvar.exp()
            - 1.0,
            dim=1,
        )

    def vfe_components(
        self,
        batch: VisualMarkBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> VisualVFEComponents:
        p_mean, p_logvar, features, condition_map = self.prior(
            batch.before, batch.condition
        )
        q_mean, q_logvar = self.posterior(batch)
        noise = torch.randn(
            q_mean.shape,
            dtype=q_mean.dtype,
            device=q_mean.device,
            generator=generator,
        )
        latent = q_mean + torch.exp(0.5 * q_logvar) * noise
        predicted_mean, concentration = self.decode(
            batch.before,
            batch.condition,
            latent,
            prior_features=features,
            condition_map=condition_map,
        )
        log_prob = self._beta_log_prob(batch.after, predicted_mean, concentration)
        valid_count = batch.validity.sum(dim=(1, 2, 3)).clamp(min=1.0)
        reconstruction = -(log_prob * batch.validity).sum(dim=(1, 2, 3))
        kl = self._diagonal_gaussian_kl(q_mean, q_logvar, p_mean, p_logvar)
        free_energy = reconstruction + kl
        return VisualVFEComponents(
            reconstruction_nll=reconstruction,
            latent_kl=kl,
            free_energy=free_energy,
            free_energy_per_observed_pixel=free_energy / valid_count,
            valid_pixel_count=valid_count,
            posterior_mean=q_mean,
            posterior_logvar=q_logvar,
            prior_mean=p_mean,
            prior_logvar=p_logvar,
            predicted_mean=predicted_mean,
            predicted_concentration=concentration,
        )

    def prior_predictive_mean(
        self,
        batch: VisualMarkBatch,
        *,
        samples: int = 8,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if samples <= 0:
            raise ValueError("samples must be positive")
        p_mean, p_logvar, features, condition_map = self.prior(
            batch.before, batch.condition
        )
        predictions = []
        for _ in range(samples):
            noise = torch.randn(
                p_mean.shape,
                dtype=p_mean.dtype,
                device=p_mean.device,
                generator=generator,
            )
            latent = p_mean + torch.exp(0.5 * p_logvar) * noise
            mean, _ = self.decode(
                batch.before,
                batch.condition,
                latent,
                prior_features=features,
                condition_map=condition_map,
            )
            predictions.append(mean)
        return torch.stack(predictions).mean(dim=0)

    def importance_weighted_nll(
        self,
        batch: VisualMarkBatch,
        *,
        samples: int = 8,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if samples <= 0:
            raise ValueError("samples must be positive")
        p_mean, p_logvar, features, condition_map = self.prior(
            batch.before, batch.condition
        )
        q_mean, q_logvar = self.posterior(batch)
        weights = []
        for _ in range(samples):
            noise = torch.randn(
                q_mean.shape,
                dtype=q_mean.dtype,
                device=q_mean.device,
                generator=generator,
            )
            latent = q_mean + torch.exp(0.5 * q_logvar) * noise
            mean, concentration = self.decode(
                batch.before,
                batch.condition,
                latent,
                prior_features=features,
                condition_map=condition_map,
            )
            log_px = (
                self._beta_log_prob(batch.after, mean, concentration) * batch.validity
            ).sum(dim=(1, 2, 3))
            log_pz = -0.5 * (
                np.log(2.0 * np.pi)
                + p_logvar
                + (latent - p_mean).square() / p_logvar.exp()
            ).sum(dim=1)
            log_qz = -0.5 * (
                np.log(2.0 * np.pi)
                + q_logvar
                + (latent - q_mean).square() / q_logvar.exp()
            ).sum(dim=1)
            weights.append(log_px + log_pz - log_qz)
        log_marginal = torch.logsumexp(torch.stack(weights), dim=0) - np.log(samples)
        valid_count = batch.validity.sum(dim=(1, 2, 3)).clamp(min=1.0)
        return -log_marginal / valid_count
