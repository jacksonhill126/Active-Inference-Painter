"""Offline trainer and evidence report for the shadow conditional patch VAE."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch

from .conditional_patch_vae import (
    PATCH_CVAE_MODEL_ID,
    PATCH_CVAE_STATUS,
    ConditionalPatchBatch,
    ConditionalPatchExample,
    ConditionalPatchVAEConfig,
    ConditionalPatchVAEEnsemble,
    conditional_patch_examples_from_shards,
    make_conditional_patch_batch,
)
from .offline_train import (
    _config_from_json,
    _pooled_training_config_payloads,
)
from .spatial_state import independent_material_channel_count
from .trajectory_corpus import SPLIT_NAMES, load_trajectory_shard


TRAINING_REPORT_SCHEMA = "conditional-patch-cvae-shadow-training-report-v0"
CHECKPOINT_SCHEMA = "conditional-patch-cvae-shadow-checkpoint-v0"


def _manifest_paths(
    manifest_path: Path | str,
) -> tuple[dict[str, list[Path]], dict[str, object]]:
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "trajectory-split-manifest-v1":
        raise ValueError("unsupported trajectory split manifest")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("trajectory split manifest has no file mapping")
    root = Path(str(payload["root"]))
    if not root.is_absolute():
        root = (manifest_path.parent / root).resolve()
    paths = {
        name: [root / str(item) for item in files.get(name, [])]
        for name in SPLIT_NAMES
    }
    ids = payload.get("trajectory_ids", {})
    flattened = [str(item) for name in SPLIT_NAMES for item in ids.get(name, [])]
    if len(flattened) != len(set(flattened)):
        raise ValueError("a trajectory appears in more than one dataset split")
    return paths, payload


def _seeded_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def _batch_chunks(
    examples: Sequence[ConditionalPatchExample],
    batch_size: int,
    device: torch.device,
):
    step = max(1, int(batch_size))
    for start in range(0, len(examples), step):
        yield make_conditional_patch_batch(examples[start : start + step], device)


def _ablated_batch(
    batch: ConditionalPatchBatch,
    kind: str,
) -> ConditionalPatchBatch:
    if kind == "shuffled_action":
        action = (
            torch.roll(batch.action, shifts=1, dims=0)
            if batch.action.shape[0] > 1
            else torch.flip(batch.action, dims=(-1,))
        )
        return batch.with_conditions(action=action)
    if kind == "motor_ablated":
        action = batch.action.clone()
        if action.shape[1] > 7:
            action[:, 7:] = 0.0
        return batch.with_conditions(action=action)
    if kind == "shuffled_brush":
        context = (
            torch.roll(batch.brush_condition, shifts=1, dims=0)
            if batch.brush_condition.shape[0] > 1
            else batch.brush_condition.clone()
        )
        return batch.with_conditions(brush_condition=context)
    raise ValueError(f"unknown conditioning ablation: {kind}")


@torch.no_grad()
def evaluate_shadow_model(
    model: ConditionalPatchVAEEnsemble,
    examples: Sequence[ConditionalPatchExample],
    *,
    device: torch.device,
    batch_size: int,
    importance_samples: int,
    prior_samples: int,
    seed: int,
) -> dict[str, object] | None:
    if not examples:
        return None
    model.eval()
    sample_count = 0
    negative_elbo = 0.0
    reconstruction = 0.0
    latent_kl = 0.0
    iwae_nll = 0.0
    shuffled_action_iwae = 0.0
    motor_ablated_iwae = 0.0
    shuffled_brush_iwae = 0.0
    valid_brush_count = 0
    likelihood_variance_sum = 0.0
    latent_variance_sum = 0.0
    epistemic_variance_sum = 0.0
    variance_element_count = 0.0
    standardized_count = 0.0
    within_50 = 0.0
    within_90 = 0.0
    brush_contexts = np.stack(
        [example.brush_condition for example in examples]
    ).astype(np.float32)
    valid_brush_indices = np.flatnonzero(brush_contexts[:, -1] > 0.5)
    shuffled_brush_contexts = brush_contexts.copy()
    if valid_brush_indices.size >= 2:
        shuffled_brush_contexts[valid_brush_indices] = brush_contexts[
            np.roll(valid_brush_indices, 1)
        ]
    cursor = 0
    for batch_index, batch in enumerate(_batch_chunks(examples, batch_size, device)):
        count = int(batch.material.shape[0])
        sample_count += count
        pair_seed = int(seed) + 1009 * batch_index
        member_terms = [
            member.vfe_components(
                batch,
                generator=_seeded_generator(device, pair_seed + 1),
            )
            for member in model.members
        ]
        negative_elbo += float(
            torch.stack(
                [term.free_energy_per_element for term in member_terms]
            ).mean(dim=0).sum().item()
        )
        reconstruction += float(
            torch.stack(
                [
                    term.reconstruction_nll / term.valid_element_count
                    for term in member_terms
                ]
            ).mean(dim=0).sum().item()
        )
        latent_kl += float(
            torch.stack([term.latent_kl for term in member_terms])
            .mean(dim=0)
            .sum()
            .item()
        )
        iwae_nll += float(
            model.mean_member_importance_weighted_nll(
                batch,
                samples=importance_samples,
                generator=_seeded_generator(device, pair_seed + 2),
            ).sum().item()
        )
        shuffled_action_iwae += float(
            model.mean_member_importance_weighted_nll(
                _ablated_batch(batch, "shuffled_action"),
                samples=importance_samples,
                generator=_seeded_generator(device, pair_seed + 2),
            ).sum().item()
        )
        motor_ablated_iwae += float(
            model.mean_member_importance_weighted_nll(
                _ablated_batch(batch, "motor_ablated"),
                samples=importance_samples,
                generator=_seeded_generator(device, pair_seed + 2),
            ).sum().item()
        )
        brush_valid = batch.brush_condition[:, -1] > 0.5
        valid_brush_count += int(brush_valid.sum().item())
        if valid_brush_indices.size >= 2:
            shuffled_context = torch.as_tensor(
                shuffled_brush_contexts[cursor : cursor + count],
                device=device,
            )
            shuffled_brush_iwae += float(
                model.mean_member_importance_weighted_nll(
                    batch.with_conditions(brush_condition=shuffled_context),
                    samples=importance_samples,
                    generator=_seeded_generator(device, pair_seed + 2),
                ).sum().item()
            )
        moments = model.predictive_moments(
            batch,
            latent_samples=prior_samples,
            generator=_seeded_generator(device, pair_seed + 3),
        )
        independent = independent_material_channel_count(batch.material.shape[1])
        valid = batch.mask.expand(-1, independent, -1, -1)
        likelihood_variance_sum += float(
            (moments.likelihood_variance[:, :independent] * valid).sum().item()
        )
        latent_variance_sum += float(
            (moments.latent_variance[:, :independent] * valid).sum().item()
        )
        epistemic_variance_sum += float(
            (moments.epistemic_variance[:, :independent] * valid).sum().item()
        )
        variance_element_count += float(valid.sum().item())
        total_variance = moments.total_variance[:, :independent].clamp(min=1e-8)
        standardized = (
            (batch.next_material[:, :independent] - moments.mean[:, :independent]).abs()
            / total_variance.sqrt()
        )
        standardized_count += float(valid.sum().item())
        within_50 += float(((standardized <= 0.67449) * valid).sum().item())
        within_90 += float(((standardized <= 1.64485) * valid).sum().item())
        cursor += count

    denominator = float(max(1, sample_count))
    variance_denominator = max(1.0, variance_element_count)
    calibration_denominator = max(1.0, standardized_count)
    correct_iwae = iwae_nll / denominator
    return {
        "sample_count": sample_count,
        "mean_member_negative_elbo_nats_per_observed_element": negative_elbo
        / denominator,
        "mean_member_reconstruction_nll_nats_per_observed_element": reconstruction
        / denominator,
        "mean_member_latent_kl_nats_per_patch": latent_kl / denominator,
        "mean_member_importance_weighted_nll_nats_per_observed_element": correct_iwae,
        "conditioning_checks": {
            "shuffled_action_iwae_nll": shuffled_action_iwae / denominator,
            "shuffled_action_minus_correct": shuffled_action_iwae / denominator
            - correct_iwae,
            "motor_channels_ablated_iwae_nll": motor_ablated_iwae / denominator,
            "motor_ablated_minus_correct": motor_ablated_iwae / denominator
            - correct_iwae,
            "valid_brush_context_samples": valid_brush_count,
            "shuffled_brush_iwae_nll": (
                shuffled_brush_iwae / denominator
                if valid_brush_count >= 2
                else None
            ),
            "shuffled_brush_minus_correct": (
                shuffled_brush_iwae / denominator - correct_iwae
                if valid_brush_count >= 2
                else None
            ),
            "interpretation": (
                "positive ablation-minus-correct gaps are evidence that the "
                "held-out conditional density uses that condition; non-positive "
                "gaps fail the corresponding capability check"
            ),
        },
        "prior_predictive_variance": {
            "decoder_likelihood_mean": likelihood_variance_sum
            / variance_denominator,
            "within_member_latent_mean": latent_variance_sum
            / variance_denominator,
            "between_member_epistemic_mean": epistemic_variance_sum
            / variance_denominator,
            "finite_latent_sample_approximation": True,
        },
        "gaussian_moment_calibration": {
            "empirical_within_nominal_50_percent": within_50
            / calibration_denominator,
            "empirical_within_nominal_90_percent": within_90
            / calibration_denominator,
            "approximation": (
                "coverage of a moment-matched Gaussian over the cVAE mixture; "
                "reported as a capability diagnostic, not a hardware calibration"
            ),
        },
    }


def train_conditional_vae_from_manifest(
    args: argparse.Namespace,
) -> dict[str, object]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    paths, manifest = _manifest_paths(args.manifest)
    shards = {
        name: [load_trajectory_shard(path) for path in paths[name]]
        for name in SPLIT_NAMES
    }
    all_shards = [shard for name in SPLIT_NAMES for shard in shards[name]]
    if not all_shards:
        raise ValueError("the trajectory manifest contains no shards")
    config_payload = _pooled_training_config_payloads(
        [dict(shard.metadata["config"]) for shard in all_shards]
    )
    painter_config = _config_from_json(config_payload)
    examples = {
        name: conditional_patch_examples_from_shards(shards[name], painter_config)
        for name in SPLIT_NAMES
    }
    if not examples["train"]:
        raise ValueError("the training split contains no local mark transitions")
    model_config = ConditionalPatchVAEConfig.from_painter_config(
        painter_config,
        hidden_channels=int(args.hidden_channels),
        residual_blocks=int(args.residual_blocks),
        latent_dim=int(args.latent_dim),
        ensemble_size=int(args.ensemble_size),
    )
    model = ConditionalPatchVAEEnsemble(model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    if args.input_checkpoint:
        payload = torch.load(
            Path(args.input_checkpoint), map_location=device, weights_only=False
        )
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported conditional-VAE checkpoint")
        if payload.get("model_config") != asdict(model_config):
            raise ValueError("input checkpoint cVAE architecture does not match")
        model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])

    evaluation_before = {
        name: evaluate_shadow_model(
            model,
            examples[name],
            device=device,
            batch_size=int(args.evaluation_batch_size),
            importance_samples=int(args.importance_samples),
            prior_samples=int(args.prior_samples),
            seed=int(args.seed) + 101 + index,
        )
        for index, name in enumerate(("validation", "test"))
    }
    rng = np.random.default_rng(int(args.seed) + 17)
    train_generator = _seeded_generator(device, int(args.seed) + 31)
    model.train()
    loss_history: list[float] = []
    started = time.perf_counter()
    for _ in range(max(0, int(args.gradient_steps))):
        indices = rng.integers(
            0,
            len(examples["train"]),
            size=min(max(1, int(args.batch_size)), len(examples["train"])),
        )
        batch = make_conditional_patch_batch(
            [examples["train"][int(index)] for index in indices],
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(batch, generator=train_generator)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        loss_history.append(float(loss.detach().cpu().item()))
    training_seconds = time.perf_counter() - started
    evaluation_after = {
        name: evaluate_shadow_model(
            model,
            examples[name],
            device=device,
            batch_size=int(args.evaluation_batch_size),
            importance_samples=int(args.importance_samples),
            prior_samples=int(args.prior_samples),
            seed=int(args.seed) + 201 + index,
        )
        for index, name in enumerate(SPLIT_NAMES)
    }
    report: dict[str, object] = {
        "schema": TRAINING_REPORT_SCHEMA,
        "model_id": PATCH_CVAE_MODEL_ID,
        "status": PATCH_CVAE_STATUS,
        "policy_influence": "none",
        "objective": (
            "standard beta=1 conditional VAE negative ELBO; reconstruction is "
            "the camera-posterior local material-transition likelihood and KL "
            "is q_phi(z|before,after,conditions) versus N(0,I)"
        ),
        "model_config": asdict(model_config),
        "device": str(device),
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_schema": manifest.get("schema"),
        "split_unit": "entire_trajectory_before_patch_extraction",
        "trajectory_counts": {name: len(shards[name]) for name in SPLIT_NAMES},
        "patch_counts": {name: len(examples[name]) for name in SPLIT_NAMES},
        "gradient_steps": int(args.gradient_steps),
        "parameter_count": int(parameter_count),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "training_loss_last": loss_history[-1] if loss_history else None,
        "training_loss_recent_mean": (
            float(np.mean(loss_history[-min(50, len(loss_history)) :]))
            if loss_history
            else None
        ),
        "training_seconds": training_seconds,
        "heldout_before": evaluation_before,
        "heldout_after": evaluation_after,
        "training_inputs": (
            "camera-derived material posterior mean, selected painting action, "
            "conditional motor realization, and optional compact inferred "
            "pre-stroke brush posterior"
        ),
        "forbidden_inputs": (
            "exact simulator canvas state, exact contact state, held paint, "
            "bristle microstructure, and future policy outcomes"
        ),
        "leakage_guard": "trajectory split completed before local patch extraction",
        "claim_boundary": (
            "uncalibrated simulation-only shadow likelihood; not policy active, "
            "not a composition model, not a reward, and not hardware calibrated"
        ),
    }
    output_checkpoint = Path(args.output_checkpoint).resolve()
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model_id": PATCH_CVAE_MODEL_ID,
        "status": PATCH_CVAE_STATUS,
        "model_config": asdict(model_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_provenance": {
            "manifest": str(Path(args.manifest).resolve()),
            "training_split_only": True,
            "policy_influence": "none",
            "process_truth_used_as_training_input": False,
            "split_unit": "entire_trajectory_before_patch_extraction",
        },
        "report": report,
    }
    temp_checkpoint = output_checkpoint.with_suffix(
        f"{output_checkpoint.suffix}.tmp"
    )
    torch.save(checkpoint_payload, temp_checkpoint)
    temp_checkpoint.replace(output_checkpoint)
    report["checkpoint"] = str(output_checkpoint)
    report_path = (
        Path(args.report_path)
        if args.report_path
        else output_checkpoint.with_suffix(".report.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_report = report_path.with_suffix(f"{report_path.suffix}.tmp")
    temp_report.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    temp_report.replace(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a conditional local transition VAE in shadow "
            "mode from a trajectory-isolated sensor-posterior corpus."
        )
    )
    parser.add_argument("--manifest", default="runs/corpus/split_manifest.json")
    parser.add_argument(
        "--output-checkpoint", default="runs/checkpoints/conditional_patch_cvae.pt"
    )
    parser.add_argument("--input-checkpoint", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument("--gradient-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--importance-samples", type=int, default=8)
    parser.add_argument("--prior-samples", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.gradient_steps < 0:
        raise SystemExit("gradient steps must be non-negative")
    if args.importance_samples <= 0 or args.prior_samples <= 0:
        raise SystemExit("evaluation sample counts must be positive")
    report = train_conditional_vae_from_manifest(args)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
