"""Shadow two-component local material-transition likelihood.

The existing local CNN assigns one Gaussian outcome per bootstrap member.  The
AI-107 residuals instead show a large near-identity mass plus rarer material
changes.  This module tests that likelihood-family hypothesis directly with a
normalized conditional mixture:

    p(s' | s, a, theta_e) = sum_k pi_k(s, a) N(s'; mu_k(s, a), var_k(s, a))

Component zero is anchored to the physically projected identity transition;
component one predicts a material consequence.  Both component variance and
mixing probability are learned.  The model is an offline shadow likelihood:
it is not a reward, preference, policy prior, EFE term, or live runtime model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .action_encoding import coerce_action_raster
from .conditional_patch_vae import (
    ConditionalPatchExample,
    conditional_patch_examples_from_shards,
    make_conditional_patch_batch,
)
from .efe_common import project_material_support
from .models import SpatialResidualBlock
from .offline_train import _config_from_json, _pooled_training_config_payloads
from .spatial_state import independent_material_channel_count
from .trajectory_corpus import SPLIT_NAMES, load_trajectory_shard
from .conditional_vae_train import _manifest_paths


MIXTURE_MODEL_ID = "conditional-local-material-transition-mixture-v0"
MIXTURE_CHECKPOINT_SCHEMA = "conditional-local-mixture-checkpoint-v0"
MIXTURE_REPORT_SCHEMA = "conditional-local-mixture-training-report-v0"
LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass(frozen=True, slots=True)
class LocalMixtureConfig:
    material_channels: int = 6
    action_channels: int = 12
    hidden_channels: int = 32
    residual_blocks: int = 2
    ensemble_size: int = 3
    bootstrap_probability: float = 0.7
    thickness_scale: float = 0.005
    ground_tone: float = 0.34
    paint_presence_threshold: float = 0.0001
    minimum_logvar: float = -13.0
    maximum_logvar: float = -3.0

    def __post_init__(self) -> None:
        if min(
            self.material_channels,
            self.action_channels,
            self.hidden_channels,
            self.ensemble_size,
        ) <= 0:
            raise ValueError("mixture model dimensions must be positive")
        if self.residual_blocks < 0:
            raise ValueError("residual block count must be non-negative")
        if not 0.0 < self.bootstrap_probability <= 1.0:
            raise ValueError("bootstrap probability must lie in (0, 1]")
        if self.minimum_logvar >= self.maximum_logvar:
            raise ValueError("minimum log variance must be below maximum")


@dataclass(slots=True)
class MixturePredictiveMoments:
    mean: torch.Tensor
    likelihood_variance: torch.Tensor
    epistemic_variance: torch.Tensor

    @property
    def total_variance(self) -> torch.Tensor:
        return self.likelihood_variance + self.epistemic_variance


class LocalMixtureMember(nn.Module):
    def __init__(self, config: LocalMixtureConfig) -> None:
        super().__init__()
        self.config = config
        input_channels = config.material_channels + config.action_channels
        blocks: list[nn.Module] = [
            nn.Conv2d(input_channels, config.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        blocks.extend(
            SpatialResidualBlock(config.hidden_channels)
            for _ in range(config.residual_blocks)
        )
        blocks.extend(
            [
                nn.SiLU(),
                # consequence delta, two log variances, and one binary logit
                nn.Conv2d(
                    config.hidden_channels,
                    4 * config.material_channels,
                    kernel_size=3,
                    padding=1,
                ),
            ]
        )
        self.net = nn.Sequential(*blocks)

    def forward_masked(
        self,
        material: torch.Tensor,
        action: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = mask.to(material)
        hidden = torch.cat([material * mask, action * mask], dim=1)
        for layer in self.net:
            if isinstance(layer, SpatialResidualBlock):
                hidden = layer.forward_masked(hidden, mask)
            else:
                hidden = layer(hidden) * mask
        delta, raw_identity_logvar, raw_consequence_logvar, consequence_logit = hidden.chunk(
            4, dim=1
        )
        identity = project_material_support(
            material,
            material,
            self.config.thickness_scale,
            self.config.ground_tone,
            self.config.paint_presence_threshold,
        )
        consequence = project_material_support(
            material,
            material + delta,
            self.config.thickness_scale,
            self.config.ground_tone,
            self.config.paint_presence_threshold,
        )
        span = self.config.maximum_logvar - self.config.minimum_logvar
        identity_logvar = self.config.minimum_logvar + span * torch.sigmoid(
            raw_identity_logvar
        )
        consequence_logvar = self.config.minimum_logvar + span * torch.sigmoid(
            raw_consequence_logvar
        )
        independent = independent_material_channel_count(material.shape[1])
        if independent < material.shape[1]:
            identity_logvar = torch.cat(
                [
                    identity_logvar[:, :independent],
                    torch.full_like(identity_logvar[:, independent:], -20.0),
                ],
                dim=1,
            )
            consequence_logvar = torch.cat(
                [
                    consequence_logvar[:, :independent],
                    torch.full_like(consequence_logvar[:, independent:], -20.0),
                ],
                dim=1,
            )
        means = torch.stack([identity, consequence], dim=0) * mask.unsqueeze(0)
        logvars = torch.stack([identity_logvar, consequence_logvar], dim=0) * mask.unsqueeze(0)
        binary_logits = torch.stack(
            [torch.zeros_like(consequence_logit), consequence_logit], dim=0
        )
        log_weights = torch.log_softmax(binary_logits, dim=0) * mask.unsqueeze(0)
        return means, logvars, log_weights


class LocalMixtureDynamicsEnsemble(nn.Module):
    """Bootstrap ensemble whose members each own a normalized two-Gaussian mixture."""

    def __init__(self, config: LocalMixtureConfig) -> None:
        super().__init__()
        self.config = config
        self.members = nn.ModuleList(
            LocalMixtureMember(config) for _ in range(config.ensemble_size)
        )

    def forward_masked(
        self,
        material: torch.Tensor,
        action: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action = coerce_action_raster(action, self.config.action_channels)
        values = [member.forward_masked(material, action, mask) for member in self.members]
        return tuple(torch.stack([value[index] for value in values], dim=0) for index in range(3))  # type: ignore[return-value]

    def per_member_sample_nll(
        self,
        material: torch.Tensor,
        action: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        means, logvars, log_weights = self.forward_masked(material, action, mask)
        independent = independent_material_channel_count(means.shape[3])
        means = means[:, :, :, :independent]
        logvars = logvars[:, :, :, :independent]
        log_weights = log_weights[:, :, :, :independent]
        expanded_target = target[:, :independent].unsqueeze(0).unsqueeze(0)
        component_log_density = -0.5 * (
            (expanded_target - means).square() / logvars.exp()
            + logvars
            + LOG_TWO_PI
        ) + log_weights
        member_nll = -torch.logsumexp(component_log_density, dim=1)
        valid = mask.unsqueeze(0).expand(-1, -1, independent, -1, -1)
        count = valid.sum(dim=(2, 3, 4)).clamp(min=1.0)
        return (member_nll * valid).sum(dim=(2, 3, 4)) / count

    def training_loss(
        self,
        material: torch.Tensor,
        action: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        per_member = self.per_member_sample_nll(material, action, target, mask)
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
        material: torch.Tensor,
        action: torch.Tensor,
        mask: torch.Tensor,
    ) -> MixturePredictiveMoments:
        means, logvars, log_weights = self.forward_masked(material, action, mask)
        weights = log_weights.exp()
        member_means = (weights * means).sum(dim=1)
        member_variance = (
            weights
            * (logvars.exp() + (means - member_means.unsqueeze(1)).square())
        ).sum(dim=1)
        mean = member_means.mean(dim=0)
        likelihood = member_variance.mean(dim=0)
        epistemic = member_means.var(dim=0, unbiased=False)
        return MixturePredictiveMoments(
            mean=mean * mask,
            likelihood_variance=likelihood * mask,
            epistemic_variance=epistemic * mask,
        )

    @torch.no_grad()
    def exact_ensemble_mixture_nll(
        self,
        material: torch.Tensor,
        action: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        means, logvars, log_weights = self.forward_masked(material, action, mask)
        independent = independent_material_channel_count(means.shape[3])
        target = target[:, :independent].unsqueeze(0).unsqueeze(0)
        log_density = -0.5 * (
            (target - means[:, :, :, :independent]).square()
            / logvars[:, :, :, :independent].exp()
            + logvars[:, :, :, :independent]
            + LOG_TWO_PI
        ) + log_weights[:, :, :, :independent]
        return -torch.logsumexp(log_density.flatten(0, 1), dim=0) + math.log(
            len(self.members)
        )


@torch.no_grad()
def evaluate_mixture(
    model: LocalMixtureDynamicsEnsemble,
    examples: Sequence[ConditionalPatchExample],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int] | None:
    if not examples:
        return None
    total_nll = 0.0
    total_elements = 0.0
    squared_error = 0.0
    absolute_error = 0.0
    for start in range(0, len(examples), max(1, int(batch_size))):
        batch = make_conditional_patch_batch(
            examples[start : start + max(1, int(batch_size))], device
        )
        nll = model.exact_ensemble_mixture_nll(
            batch.material, batch.action, batch.next_material, batch.mask
        )
        moments = model.predictive_moments(batch.material, batch.action, batch.mask)
        independent = independent_material_channel_count(batch.material.shape[1])
        valid = batch.mask.expand(-1, independent, -1, -1)
        residual = batch.next_material[:, :independent] - moments.mean[:, :independent]
        total_nll += float((nll * valid).sum().item())
        squared_error += float((residual.square() * valid).sum().item())
        absolute_error += float((residual.abs() * valid).sum().item())
        total_elements += float(valid.sum().item())
    return {
        "transition_count": len(examples),
        "element_count": int(total_elements),
        "exact_ensemble_mixture_nll_nats_per_element": total_nll / total_elements,
        "rmse": math.sqrt(squared_error / total_elements),
        "mae": absolute_error / total_elements,
    }


def train_mixture_from_manifest(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(
        args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
    )
    paths, manifest = _manifest_paths(args.manifest)
    shards = {
        split: [load_trajectory_shard(path) for path in paths[split]]
        for split in SPLIT_NAMES
    }
    all_shards = [shard for split in SPLIT_NAMES for shard in shards[split]]
    painter_payload = _pooled_training_config_payloads(
        [dict(shard.metadata["config"]) for shard in all_shards]
    )
    painter = _config_from_json(painter_payload)
    examples = {
        split: conditional_patch_examples_from_shards(shards[split], painter)
        for split in SPLIT_NAMES
    }
    config = LocalMixtureConfig(
        material_channels=int(painter.spatial_material_channels),
        action_channels=int(painter.spatial_action_channels),
        hidden_channels=int(args.hidden_channels),
        residual_blocks=int(args.residual_blocks),
        ensemble_size=int(args.ensemble_size),
        bootstrap_probability=float(painter.ensemble_bootstrap_probability),
        thickness_scale=float(painter.thickness_scale),
        ground_tone=float(painter.canvas_ground_tone),
        paint_presence_threshold=float(painter.paint_presence_threshold),
    )
    model = LocalMixtureDynamicsEnsemble(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rng = np.random.default_rng(int(args.seed) + 17)
    generator = torch.Generator(device=device.type).manual_seed(int(args.seed) + 31)
    history: list[float] = []
    started = time.perf_counter()
    model.train()
    for _ in range(max(0, int(args.gradient_steps))):
        indices = rng.integers(
            0,
            len(examples["train"]),
            size=min(max(1, int(args.batch_size)), len(examples["train"])),
        )
        batch = make_conditional_patch_batch(
            [examples["train"][int(index)] for index in indices], device
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(
            batch.material,
            batch.action,
            batch.next_material,
            batch.mask,
            generator=generator,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        history.append(float(loss.detach().cpu().item()))
    training_seconds = time.perf_counter() - started
    model.eval()
    evaluation = {
        split: evaluate_mixture(
            model, examples[split], device=device, batch_size=int(args.evaluation_batch_size)
        )
        for split in SPLIT_NAMES
    }
    checkpoint_path = Path(args.output_checkpoint).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema": MIXTURE_REPORT_SCHEMA,
        "model_id": MIXTURE_MODEL_ID,
        "status": "shadow_offline_not_policy_active",
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_schema": manifest.get("schema"),
        "split_unit": "whole trajectory before patch extraction",
        "trajectory_counts": {split: len(shards[split]) for split in SPLIT_NAMES},
        "transition_counts": {split: len(examples[split]) for split in SPLIT_NAMES},
        "gradient_steps": int(args.gradient_steps),
        "training_seconds": training_seconds,
        "training_loss_last": history[-1] if history else None,
        "training_loss_recent_mean": float(np.mean(history[-50:])) if history else None,
        "parameter_count": int(parameter_count),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "model_config": asdict(config),
        "evaluation": evaluation,
        "likelihood": (
            "normalized two-Gaussian mixture per material cell-channel; identity "
            "component plus learned consequence component"
        ),
        "claim_boundary": (
            "uncalibrated simulation-only shadow likelihood-family experiment; "
            "not policy active, not a preference, and not hardware calibrated"
        ),
    }
    payload = {
        "schema": MIXTURE_CHECKPOINT_SCHEMA,
        "model_id": MIXTURE_MODEL_ID,
        "model_config": asdict(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_provenance": {
            "manifest": str(Path(args.manifest).resolve()),
            "training_split_only": True,
            "process_truth_used_as_training_input": False,
            "policy_influence": "none",
        },
        "report": report,
    }
    temporary = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    report["checkpoint"] = str(checkpoint_path)
    report_path = (
        Path(args.report_path)
        if args.report_path
        else checkpoint_path.with_suffix(".report.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(f"{report_path.suffix}.tmp")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary_report.replace(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the shadow local mixture likelihood")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--gradient-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--ensemble-size", type=int, default=3)
    return parser


def main() -> None:
    report = train_mixture_from_manifest(build_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
