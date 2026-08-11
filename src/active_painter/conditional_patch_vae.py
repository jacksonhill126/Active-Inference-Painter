"""Conditional local material-transition VAE for shadow/offline evaluation.

This module is a learned likelihood family for

    p_theta(x_patch[t+1] | x_patch[t], a_patch[t], q(brush)[t], z[t]).

It is deliberately not wired into policy selection.  Its training inputs are
camera-derived spatial posterior means, the selected painting action, the
conditional motor realization, and (when available) the compact inferred
pre-stroke brush posterior.  Exact simulator material or bristle state is not
accepted by the corpus adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from .config import PainterConfig
from .efe_common import project_material_support
from .env import StrokeAction
from .local_spatial import (
    LocalPatchBounds,
    crop_patch,
    local_patch_bounds_for_action,
)
from .policies import MotorPrimitiveLatent
from .spatial_state import (
    independent_material_channel_count,
    rasterize_stroke_action,
)
from .trajectory_corpus import TrajectoryShard


LOG_2_PI = math.log(2.0 * math.pi)
PATCH_CVAE_MODEL_ID = "conditional-local-material-transition-cvae-v0"
PATCH_CVAE_STATUS = "shadow_offline_not_policy_active"


@dataclass(frozen=True, slots=True)
class ConditionalPatchVAEConfig:
    material_channels: int = 6
    action_channels: int = 12
    brush_context_dim: int = 5
    hidden_channels: int = 32
    residual_blocks: int = 2
    latent_dim: int = 16
    ensemble_size: int = 3
    bootstrap_probability: float = 0.8
    thickness_scale: float = 0.005
    ground_tone: float = 0.34
    paint_presence_threshold: float = 0.0001
    minimum_logvar: float = -11.0
    maximum_logvar: float = -3.0

    def __post_init__(self) -> None:
        integer_fields = (
            self.material_channels,
            self.action_channels,
            self.brush_context_dim,
            self.hidden_channels,
            self.latent_dim,
            self.ensemble_size,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("cVAE channel, latent, hidden, and ensemble sizes must be positive")
        if self.residual_blocks < 0:
            raise ValueError("residual_blocks must be non-negative")
        if not 0.0 < self.bootstrap_probability <= 1.0:
            raise ValueError("bootstrap_probability must lie in (0, 1]")
        if self.minimum_logvar >= self.maximum_logvar:
            raise ValueError("minimum_logvar must be below maximum_logvar")

    @classmethod
    def from_painter_config(
        cls,
        config: PainterConfig,
        *,
        hidden_channels: int | None = None,
        residual_blocks: int = 2,
        latent_dim: int = 16,
        ensemble_size: int | None = None,
    ) -> "ConditionalPatchVAEConfig":
        return cls(
            material_channels=int(config.spatial_material_channels),
            action_channels=int(config.spatial_action_channels),
            hidden_channels=(
                int(config.spatial_hidden_channels)
                if hidden_channels is None
                else int(hidden_channels)
            ),
            residual_blocks=int(residual_blocks),
            latent_dim=int(latent_dim),
            ensemble_size=(
                int(config.spatial_ensemble_size)
                if ensemble_size is None
                else int(ensemble_size)
            ),
            bootstrap_probability=float(config.ensemble_bootstrap_probability),
            thickness_scale=float(config.thickness_scale),
            ground_tone=float(config.canvas_ground_tone),
            paint_presence_threshold=float(config.paint_presence_threshold),
        )


@dataclass(frozen=True, slots=True)
class ConditionalPatchExample:
    bounds: LocalPatchBounds
    material: np.ndarray
    material_logvar: np.ndarray
    action: np.ndarray
    brush_condition: np.ndarray
    next_material: np.ndarray
    next_material_logvar: np.ndarray
    trajectory_id: str


@dataclass(slots=True)
class ConditionalPatchBatch:
    material: torch.Tensor
    material_logvar: torch.Tensor
    action: torch.Tensor
    brush_condition: torch.Tensor
    next_material: torch.Tensor
    next_material_logvar: torch.Tensor
    mask: torch.Tensor
    trajectory_ids: tuple[str, ...] = ()

    def with_conditions(
        self,
        *,
        action: torch.Tensor | None = None,
        brush_condition: torch.Tensor | None = None,
    ) -> "ConditionalPatchBatch":
        return ConditionalPatchBatch(
            material=self.material,
            material_logvar=self.material_logvar,
            action=self.action if action is None else action,
            brush_condition=(
                self.brush_condition
                if brush_condition is None
                else brush_condition
            ),
            next_material=self.next_material,
            next_material_logvar=self.next_material_logvar,
            mask=self.mask,
            trajectory_ids=self.trajectory_ids,
        )


@dataclass(slots=True)
class PatchVFEComponents:
    """Per-sample negative-ELBO/VFE decomposition in nats."""

    reconstruction_nll: torch.Tensor
    latent_kl: torch.Tensor
    negative_elbo: torch.Tensor
    valid_element_count: torch.Tensor
    free_energy_per_element: torch.Tensor


@dataclass(slots=True)
class PatchPredictiveMoments:
    """Prior-predictive uncertainty split for one cVAE ensemble.

    `likelihood_variance` is the decoder's conditional observation variance.
    `latent_variance` is variation across z within a fixed member.
    `epistemic_variance` is disagreement between bootstrap members after
    marginalizing their z samples.  The split is an approximation because a
    finite number of latent samples is used.
    """

    mean: torch.Tensor
    likelihood_variance: torch.Tensor
    latent_variance: torch.Tensor
    epistemic_variance: torch.Tensor

    @property
    def outcome_variance(self) -> torch.Tensor:
        return self.likelihood_variance + self.latent_variance

    @property
    def total_variance(self) -> torch.Tensor:
        return self.outcome_variance + self.epistemic_variance


def conditional_patch_examples_from_shards(
    shards: Iterable[TrajectoryShard],
    config: PainterConfig,
) -> list[ConditionalPatchExample]:
    """Extract local patches only after trajectory-level splitting.

    The adapter intentionally names every accepted field.  It never reads raw
    camera-process images, `ArmPainterSim`, exact material state, held paint, or
    bristle microstructure.  A shard that claims process truth was used as a
    training input is rejected rather than silently mixed into the likelihood.
    """

    examples: list[ConditionalPatchExample] = []
    for shard in shards:
        if shard.metadata.get("process_truth_used_as_training_input") is not False:
            raise ValueError(
                f"trajectory {shard.trajectory_id!r} does not declare a process-truth-free training boundary"
            )
        for index in range(shard.transition_count):
            values = np.asarray(shard.action[index], dtype=np.float32).reshape(-1)
            if values.shape != (8,):
                raise ValueError("stored stroke action must have eight values")
            action = StrokeAction(
                x0=float(values[0]),
                y0=float(values[1]),
                x1=float(values[2]),
                y1=float(values[3]),
                width=float(values[4]),
                amount=float(values[5]),
                tone=float(values[6]),
                curvature=float(values[7]),
            )
            grid_size = int(shard.state_material[index].shape[-1])
            bounds = local_patch_bounds_for_action(action, grid_size, config)
            if bounds is None:
                continue
            primitive = MotorPrimitiveLatent(kind=str(shard.motor_kind[index]))
            action_raster = rasterize_stroke_action(
                action,
                grid_size,
                motor_primitive=primitive,
                config=config,
            )
            examples.append(
                ConditionalPatchExample(
                    bounds=bounds,
                    material=crop_patch(shard.state_material[index], bounds),
                    material_logvar=crop_patch(
                        shard.state_logvar[index], bounds
                    ),
                    action=crop_patch(action_raster, bounds),
                    brush_condition=np.asarray(
                        shard.brush_condition[index], dtype=np.float32
                    ).copy(),
                    next_material=crop_patch(
                        shard.next_material[index], bounds
                    ),
                    next_material_logvar=crop_patch(
                        shard.next_logvar[index], bounds
                    ),
                    trajectory_id=shard.trajectory_id,
                )
            )
    return examples


def make_conditional_patch_batch(
    examples: Sequence[ConditionalPatchExample],
    device: torch.device | str,
) -> ConditionalPatchBatch:
    if not examples:
        raise ValueError("cannot batch an empty conditional-patch sequence")
    height = max(example.bounds.height for example in examples)
    width = max(example.bounds.width for example in examples)
    batch_size = len(examples)
    material_channels = int(examples[0].material.shape[0])
    action_channels = int(examples[0].action.shape[0])
    context_dim = int(examples[0].brush_condition.shape[0])
    material = np.zeros(
        (batch_size, material_channels, height, width), dtype=np.float32
    )
    material_logvar = np.zeros_like(material)
    action = np.zeros(
        (batch_size, action_channels, height, width), dtype=np.float32
    )
    context = np.zeros((batch_size, context_dim), dtype=np.float32)
    next_material = np.zeros_like(material)
    next_material_logvar = np.zeros_like(material)
    mask = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    for row, example in enumerate(examples):
        h, w = example.bounds.height, example.bounds.width
        if example.material.shape[0] != material_channels:
            raise ValueError("material channels differ inside one cVAE batch")
        if example.action.shape[0] != action_channels:
            raise ValueError("action channels differ inside one cVAE batch")
        if example.brush_condition.shape != (context_dim,):
            raise ValueError("brush context dimensions differ inside one cVAE batch")
        material[row, :, :h, :w] = example.material
        material_logvar[row, :, :h, :w] = example.material_logvar
        action[row, :, :h, :w] = example.action
        context[row] = example.brush_condition
        next_material[row, :, :h, :w] = example.next_material
        next_material_logvar[row, :, :h, :w] = example.next_material_logvar
        mask[row, :, :h, :w] = 1.0
    return ConditionalPatchBatch(
        material=torch.as_tensor(material, device=device),
        material_logvar=torch.as_tensor(material_logvar, device=device),
        action=torch.as_tensor(action, device=device),
        brush_condition=torch.as_tensor(context, device=device),
        next_material=torch.as_tensor(next_material, device=device),
        next_material_logvar=torch.as_tensor(
            next_material_logvar, device=device
        ),
        mask=torch.as_tensor(mask, device=device),
        trajectory_ids=tuple(example.trajectory_id for example in examples),
    )


class _MaskedResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.second = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, fields: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.first(fields * mask)) * mask
        hidden = self.second(hidden) * mask
        return (fields + hidden) * mask


class ConditionalPatchVAE(nn.Module):
    """One conditional VAE member with q_phi(z|before, after, conditions)."""

    def __init__(self, config: ConditionalPatchVAEConfig) -> None:
        super().__init__()
        self.config = config
        encoder_channels = (
            4 * config.material_channels
            + config.action_channels
            + config.brush_context_dim
        )
        decoder_channels = (
            2 * config.material_channels
            + config.action_channels
            + config.brush_context_dim
            + config.latent_dim
        )
        self.encoder_input = nn.Conv2d(
            encoder_channels, config.hidden_channels, kernel_size=3, padding=1
        )
        self.encoder_blocks = nn.ModuleList(
            _MaskedResidualBlock(config.hidden_channels)
            for _ in range(config.residual_blocks)
        )
        self.posterior_head = nn.Linear(config.hidden_channels, 2 * config.latent_dim)
        self.decoder_input = nn.Conv2d(
            decoder_channels, config.hidden_channels, kernel_size=3, padding=1
        )
        self.decoder_blocks = nn.ModuleList(
            _MaskedResidualBlock(config.hidden_channels)
            for _ in range(config.residual_blocks)
        )
        self.decoder_output = nn.Conv2d(
            config.hidden_channels,
            2 * config.material_channels,
            kernel_size=3,
            padding=1,
        )

    def _validate(
        self,
        material: torch.Tensor,
        material_logvar: torch.Tensor,
        action: torch.Tensor,
        brush_condition: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if material.ndim != 4 or action.ndim != 4 or mask.ndim != 4:
            raise ValueError("material, action, and mask must be BCHW tensors")
        if material.shape[1] != self.config.material_channels:
            raise ValueError("unexpected material channel count")
        if material_logvar.shape != material.shape:
            raise ValueError("material log variance must match material")
        if action.shape[1] != self.config.action_channels:
            raise ValueError("unexpected action channel count")
        if brush_condition.shape != (
            material.shape[0],
            self.config.brush_context_dim,
        ):
            raise ValueError("unexpected brush-condition shape")
        if mask.shape != (material.shape[0], 1, *material.shape[-2:]):
            raise ValueError("mask must have shape [batch, 1, height, width]")
        if action.shape[0] != material.shape[0] or action.shape[-2:] != material.shape[-2:]:
            raise ValueError("action and material batch/spatial shapes must agree")

    @staticmethod
    def _broadcast(vector: torch.Tensor, spatial: tuple[int, int]) -> torch.Tensor:
        return vector[:, :, None, None].expand(-1, -1, *spatial)

    def posterior_parameters(
        self,
        material: torch.Tensor,
        material_logvar: torch.Tensor,
        action: torch.Tensor,
        brush_condition: torch.Tensor,
        next_material: torch.Tensor,
        next_material_logvar: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(
            material, material_logvar, action, brush_condition, mask
        )
        if next_material.shape != material.shape:
            raise ValueError("next material must match current material")
        if next_material_logvar.shape != material.shape:
            raise ValueError("next material log variance must match current material")
        context = self._broadcast(brush_condition, material.shape[-2:])
        fields = torch.cat(
            [
                material,
                material_logvar,
                next_material,
                next_material_logvar,
                action,
                context,
            ],
            dim=1,
        ) * mask
        hidden = torch.nn.functional.silu(self.encoder_input(fields)) * mask
        for block in self.encoder_blocks:
            hidden = block(hidden, mask)
        count = mask.sum(dim=(2, 3)).clamp(min=1.0)
        pooled = (hidden * mask).sum(dim=(2, 3)) / count
        mean, raw_logvar = self.posterior_head(pooled).chunk(2, dim=1)
        logvar = torch.clamp(raw_logvar, min=-10.0, max=6.0)
        return mean, logvar

    def decode(
        self,
        material: torch.Tensor,
        material_logvar: torch.Tensor,
        action: torch.Tensor,
        brush_condition: torch.Tensor,
        latent: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(
            material, material_logvar, action, brush_condition, mask
        )
        if latent.shape != (material.shape[0], self.config.latent_dim):
            raise ValueError("unexpected latent shape")
        context = self._broadcast(brush_condition, material.shape[-2:])
        latent_fields = self._broadcast(latent, material.shape[-2:])
        fields = torch.cat(
            [
                material,
                material_logvar,
                action,
                context,
                latent_fields,
            ],
            dim=1,
        ) * mask
        hidden = torch.nn.functional.silu(self.decoder_input(fields)) * mask
        for block in self.decoder_blocks:
            hidden = block(hidden, mask)
        raw = self.decoder_output(hidden) * mask
        delta, raw_logvar = raw.chunk(2, dim=1)
        proposed = material + delta
        mean = project_material_support(
            material,
            proposed,
            self.config.thickness_scale,
            self.config.ground_tone,
            self.config.paint_presence_threshold,
        ) * mask
        span = self.config.maximum_logvar - self.config.minimum_logvar
        logvar = (
            self.config.minimum_logvar + span * torch.sigmoid(raw_logvar)
        ) * mask
        independent = independent_material_channel_count(
            self.config.material_channels
        )
        if independent < self.config.material_channels:
            logvar = torch.cat(
                [
                    logvar[:, :independent],
                    torch.full_like(logvar[:, independent:], -20.0) * mask,
                ],
                dim=1,
            )
        return mean, logvar

    def vfe_components(
        self,
        batch: ConditionalPatchBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> PatchVFEComponents:
        posterior_mean, posterior_logvar = self.posterior_parameters(
            batch.material,
            batch.material_logvar,
            batch.action,
            batch.brush_condition,
            batch.next_material,
            batch.next_material_logvar,
            batch.mask,
        )
        noise = torch.randn(
            posterior_mean.shape,
            device=posterior_mean.device,
            dtype=posterior_mean.dtype,
            generator=generator,
        )
        latent = posterior_mean + (0.5 * posterior_logvar).exp() * noise
        mean, logvar = self.decode(
            batch.material,
            batch.material_logvar,
            batch.action,
            batch.brush_condition,
            latent,
            batch.mask,
        )
        independent = independent_material_channel_count(mean.shape[1])
        target = batch.next_material[:, :independent]
        mean = mean[:, :independent]
        logvar = logvar[:, :independent]
        valid = batch.mask.expand(-1, independent, -1, -1)
        element_nll = 0.5 * (
            (target - mean).square() / logvar.exp() + logvar + LOG_2_PI
        )
        reconstruction_nll = (element_nll * valid).sum(dim=(1, 2, 3))
        latent_kl = 0.5 * (
            posterior_mean.square() + posterior_logvar.exp() - posterior_logvar - 1.0
        ).sum(dim=1)
        negative_elbo = reconstruction_nll + latent_kl
        valid_count = valid.sum(dim=(1, 2, 3)).clamp(min=1.0)
        return PatchVFEComponents(
            reconstruction_nll=reconstruction_nll,
            latent_kl=latent_kl,
            negative_elbo=negative_elbo,
            valid_element_count=valid_count,
            free_energy_per_element=negative_elbo / valid_count,
        )

    def importance_weighted_nll(
        self,
        batch: ConditionalPatchBatch,
        *,
        samples: int = 8,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Importance-sampled held-out -log p(after|conditions), per element."""

        if samples <= 0:
            raise ValueError("importance sample count must be positive")
        posterior_mean, posterior_logvar = self.posterior_parameters(
            batch.material,
            batch.material_logvar,
            batch.action,
            batch.brush_condition,
            batch.next_material,
            batch.next_material_logvar,
            batch.mask,
        )
        independent = independent_material_channel_count(batch.material.shape[1])
        target = batch.next_material[:, :independent]
        valid = batch.mask.expand(-1, independent, -1, -1)
        valid_count = valid.sum(dim=(1, 2, 3)).clamp(min=1.0)
        log_weights: list[torch.Tensor] = []
        for _ in range(int(samples)):
            noise = torch.randn(
                posterior_mean.shape,
                device=posterior_mean.device,
                dtype=posterior_mean.dtype,
                generator=generator,
            )
            latent = posterior_mean + (0.5 * posterior_logvar).exp() * noise
            mean, logvar = self.decode(
                batch.material,
                batch.material_logvar,
                batch.action,
                batch.brush_condition,
                latent,
                batch.mask,
            )
            mean = mean[:, :independent]
            logvar = logvar[:, :independent]
            log_likelihood = -(
                0.5
                * (
                    (target - mean).square() / logvar.exp()
                    + logvar
                    + LOG_2_PI
                )
                * valid
            ).sum(dim=(1, 2, 3))
            log_prior = -0.5 * (latent.square() + LOG_2_PI).sum(dim=1)
            log_posterior = -0.5 * (
                (latent - posterior_mean).square() / posterior_logvar.exp()
                + posterior_logvar
                + LOG_2_PI
            ).sum(dim=1)
            log_weights.append(log_likelihood + log_prior - log_posterior)
        stacked = torch.stack(log_weights, dim=0)
        log_evidence = torch.logsumexp(stacked, dim=0) - math.log(float(samples))
        return -log_evidence / valid_count

    def prior_predictions(
        self,
        batch: ConditionalPatchBatch,
        *,
        samples: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if samples <= 0:
            raise ValueError("prior sample count must be positive")
        means: list[torch.Tensor] = []
        logvars: list[torch.Tensor] = []
        shape = (batch.material.shape[0], self.config.latent_dim)
        for _ in range(int(samples)):
            latent = torch.randn(
                shape,
                device=batch.material.device,
                dtype=batch.material.dtype,
                generator=generator,
            )
            mean, logvar = self.decode(
                batch.material,
                batch.material_logvar,
                batch.action,
                batch.brush_condition,
                latent,
                batch.mask,
            )
            means.append(mean)
            logvars.append(logvar)
        return torch.stack(means, dim=0), torch.stack(logvars, dim=0)


class ConditionalPatchVAEEnsemble(nn.Module):
    """Bootstrap cVAE ensemble used only as a shadow transition likelihood."""

    def __init__(self, config: ConditionalPatchVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.members = nn.ModuleList(
            ConditionalPatchVAE(config) for _ in range(config.ensemble_size)
        )

    def training_loss(
        self,
        batch: ConditionalPatchBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        per_member = torch.stack(
            [
                member.vfe_components(batch, generator=generator).free_energy_per_element
                for member in self.members
            ],
            dim=0,
        )
        if self.config.bootstrap_probability >= 1.0:
            bootstrap = torch.ones_like(per_member)
        else:
            bootstrap = (
                torch.rand(
                    per_member.shape,
                    device=per_member.device,
                    generator=generator,
                )
                < self.config.bootstrap_probability
            )
            bootstrap = bootstrap | (bootstrap.sum(dim=1, keepdim=True) == 0)
            bootstrap = bootstrap.to(per_member.dtype)
        return (per_member * bootstrap).sum() / bootstrap.sum().clamp(min=1.0)

    @torch.no_grad()
    def predictive_moments(
        self,
        batch: ConditionalPatchBatch,
        *,
        latent_samples: int = 8,
        generator: torch.Generator | None = None,
    ) -> PatchPredictiveMoments:
        member_sample_means: list[torch.Tensor] = []
        member_sample_logvars: list[torch.Tensor] = []
        for member in self.members:
            means, logvars = member.prior_predictions(
                batch,
                samples=latent_samples,
                generator=generator,
            )
            member_sample_means.append(means)
            member_sample_logvars.append(logvars)
        means = torch.stack(member_sample_means, dim=0)
        logvars = torch.stack(member_sample_logvars, dim=0)
        member_means = means.mean(dim=1)
        mean = member_means.mean(dim=0)
        likelihood_variance = logvars.exp().mean(dim=(0, 1))
        latent_variance = means.var(dim=1, unbiased=False).mean(dim=0)
        epistemic_variance = member_means.var(dim=0, unbiased=False)
        valid = batch.mask.to(mean.dtype)
        return PatchPredictiveMoments(
            mean=mean * valid,
            likelihood_variance=likelihood_variance * valid,
            latent_variance=latent_variance * valid,
            epistemic_variance=epistemic_variance * valid,
        )

    @torch.no_grad()
    def mean_member_importance_weighted_nll(
        self,
        batch: ConditionalPatchBatch,
        *,
        samples: int = 8,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return torch.stack(
            [
                member.importance_weighted_nll(
                    batch,
                    samples=samples,
                    generator=generator,
                )
                for member in self.members
            ],
            dim=0,
        ).mean(dim=0)
