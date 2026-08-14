"""Resumable offline trainer for the registered visual mark VAE."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from .trajectory_corpus import SPLIT_NAMES
from .visual_mark_vae import (
    VISUAL_MARK_VAE_MODEL_ID,
    VISUAL_MARK_VAE_STATUS,
    VisualMarkBatch,
    VisualMarkExample,
    VisualMarkVAE,
    VisualMarkVAEConfig,
    make_visual_mark_batch,
    visual_mark_examples_from_paths,
)
from .visual_trajectory_corpus import visual_manifest_paths


VISUAL_VAE_CHECKPOINT_SCHEMA = "registered-visual-mark-cvae-checkpoint-v0"
VISUAL_VAE_REPORT_SCHEMA = "registered-visual-mark-cvae-training-report-v0"


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def _chunks(
    examples: Sequence[VisualMarkExample],
    batch_size: int,
    device: torch.device,
):
    for start in range(0, len(examples), max(1, int(batch_size))):
        yield make_visual_mark_batch(examples[start : start + batch_size], device)


def _sobel(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_kernel = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=value.device,
        dtype=value.dtype,
    ).reshape(1, 1, 3, 3) / 8.0
    y_kernel = x_kernel.transpose(-1, -2)
    return (
        F.conv2d(value, x_kernel, padding=1),
        F.conv2d(value, y_kernel, padding=1),
    )


def _ablate_action(batch: VisualMarkBatch) -> VisualMarkBatch:
    condition = batch.condition.clone()
    # Camera pairs from one transition legitimately have identical actions, so
    # within-batch shuffling can be a no-op.  Removing the declared stroke is a
    # deterministic held-out likelihood ablation for every batching pattern.
    condition[:, :8] = 0.0
    return batch.with_condition(condition)


@torch.no_grad()
def evaluate_visual_model(
    model: VisualMarkVAE,
    examples: Sequence[VisualMarkExample],
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
    count = 0
    negative_elbo = 0.0
    reconstruction = 0.0
    latent_kl = 0.0
    iwae = 0.0
    ablated_iwae = 0.0
    mae = 0.0
    identity_mae = 0.0
    edge_mae = 0.0
    identity_edge_mae = 0.0
    camera_sums: dict[str, list[float]] = {}
    cursor = 0
    for batch_index, batch in enumerate(_chunks(examples, batch_size, device)):
        batch_count = int(batch.before.shape[0])
        pair_seed = int(seed) + 1009 * batch_index
        terms = model.vfe_components(
            batch, generator=_generator(device, pair_seed + 1)
        )
        prediction = model.prior_predictive_mean(
            batch,
            samples=prior_samples,
            generator=_generator(device, pair_seed + 2),
        )
        correct_iwae = model.importance_weighted_nll(
            batch,
            samples=importance_samples,
            generator=_generator(device, pair_seed + 3),
        )
        wrong_iwae = model.importance_weighted_nll(
            _ablate_action(batch),
            samples=importance_samples,
            generator=_generator(device, pair_seed + 3),
        )
        valid = batch.validity
        valid_count = valid.sum(dim=(1, 2, 3)).clamp(min=1.0)
        sample_mae = ((prediction - batch.after).abs() * valid).sum(
            dim=(1, 2, 3)
        ) / valid_count
        sample_identity_mae = ((batch.before - batch.after).abs() * valid).sum(
            dim=(1, 2, 3)
        ) / valid_count
        pred_gx, pred_gy = _sobel(prediction)
        target_gx, target_gy = _sobel(batch.after)
        before_gx, before_gy = _sobel(batch.before)
        edge_valid = F.avg_pool2d(valid, 3, stride=1, padding=1)
        sample_edge = (
            (pred_gx - target_gx).abs() + (pred_gy - target_gy).abs()
        )
        sample_identity_edge = (
            (before_gx - target_gx).abs() + (before_gy - target_gy).abs()
        )
        edge_count = edge_valid.sum(dim=(1, 2, 3)).clamp(min=1.0)
        sample_edge_mae = (sample_edge * edge_valid).sum(dim=(1, 2, 3)) / edge_count
        sample_identity_edge_mae = (sample_identity_edge * edge_valid).sum(
            dim=(1, 2, 3)
        ) / edge_count
        count += batch_count
        negative_elbo += float(terms.free_energy_per_observed_pixel.sum().item())
        reconstruction += float(
            (terms.reconstruction_nll / terms.valid_pixel_count).sum().item()
        )
        latent_kl += float(terms.latent_kl.sum().item())
        iwae += float(correct_iwae.sum().item())
        ablated_iwae += float(wrong_iwae.sum().item())
        mae += float(sample_mae.sum().item())
        identity_mae += float(sample_identity_mae.sum().item())
        edge_mae += float(sample_edge_mae.sum().item())
        identity_edge_mae += float(sample_identity_edge_mae.sum().item())
        for local_index in range(batch_count):
            camera = examples[cursor + local_index].camera_name
            values = camera_sums.setdefault(camera, [0.0, 0.0])
            values[0] += float(sample_mae[local_index].item())
            values[1] += 1.0
        cursor += batch_count
    denominator = float(max(1, count))
    correct_iwae = iwae / denominator
    return {
        "sample_count": count,
        "negative_elbo_nats_per_observed_pixel": negative_elbo / denominator,
        "reconstruction_nll_nats_per_observed_pixel": reconstruction / denominator,
        "latent_kl_nats_per_crop": latent_kl / denominator,
        "importance_weighted_nll_nats_per_observed_pixel": correct_iwae,
        "prior_predictive_mae": mae / denominator,
        "fresh_image_identity_mae": identity_mae / denominator,
        "prior_predictive_minus_identity_mae": (mae - identity_mae) / denominator,
        "oriented_edge_component_mae": edge_mae / denominator,
        "fresh_image_identity_edge_component_mae": identity_edge_mae / denominator,
        "conditioning_checks": {
            "action_ablated_iwae_nll": ablated_iwae / denominator,
            "action_ablated_minus_correct": ablated_iwae / denominator - correct_iwae,
            "interpretation": (
                "a positive held-out gap is evidence that the normalized visual "
                "likelihood uses the selected mark condition"
            ),
        },
        "per_camera_prior_predictive_mae": {
            name: values[0] / max(1.0, values[1])
            for name, values in sorted(camera_sums.items())
        },
        "likelihood_family": "pixelwise_beta_on_registered_normalized_intensity",
        "edge_metric_role": "held_out_diagnostic_not_training_reward_or_preference",
    }


def _save_panel(
    path: Path,
    model: VisualMarkVAE,
    examples: Sequence[VisualMarkExample],
    device: torch.device,
    *,
    seed: int,
    maximum: int = 8,
) -> None:
    if not examples:
        return
    selected = list(examples[:maximum])
    batch = make_visual_mark_batch(selected, device)
    model.eval()
    with torch.no_grad():
        prediction = model.prior_predictive_mean(
            batch, samples=8, generator=_generator(device, seed)
        ).cpu().numpy()
    rows: list[Image.Image] = []
    for index, example in enumerate(selected):
        arrays = [
            example.before[0],
            example.after[0],
            prediction[index, 0],
            np.abs(prediction[index, 0] - example.after[0]),
        ]
        tiles = [
            Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255.0), mode="L")
            .resize((128, 128), Image.Resampling.NEAREST)
            .convert("RGB")
            for array in arrays
        ]
        row = Image.new("RGB", (512, 148), "white")
        for tile_index, tile in enumerate(tiles):
            row.paste(tile, (128 * tile_index, 20))
        draw = ImageDraw.Draw(row)
        draw.text((3, 3), f"{example.camera_name} transition {example.transition_index}", fill="black")
        rows.append(row)
    panel = Image.new("RGB", (512, len(rows) * 148), "white")
    for index, row in enumerate(rows):
        panel.paste(row, (0, index * 148))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, 511, 18), fill="white")
    draw.text((3, 3), "before | observed after | prior prediction | absolute error", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temp)
    temp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_visual_vae(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))
    torch.set_num_threads(max(1, int(args.torch_threads)))
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    paths, manifest = visual_manifest_paths(args.manifest)
    camera_names = tuple(str(item) for item in manifest.get("camera_names", ()))
    motor_kinds = tuple(str(item) for item in manifest.get("motor_kinds", ()))
    config = VisualMarkVAEConfig(
        patch_size=int(args.patch_size),
        latent_dim=int(args.latent_dim),
        base_channels=int(args.base_channels),
        condition_channels=int(args.condition_channels),
        camera_names=camera_names or VisualMarkVAEConfig().camera_names,
        motor_kinds=motor_kinds or VisualMarkVAEConfig().motor_kinds,
    )
    examples = {
        name: visual_mark_examples_from_paths(paths[name], config)
        for name in SPLIT_NAMES
    }
    if not examples["train"]:
        raise ValueError("the visual training split contains no camera transitions")
    model = VisualMarkVAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    checkpoint_path = Path(args.checkpoint)
    start_epoch = 0
    history: list[dict[str, object]] = []
    if args.resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved.get("schema") != VISUAL_VAE_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported visual VAE checkpoint schema")
        if saved.get("config") != asdict(config):
            raise ValueError("checkpoint model configuration does not match this run")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        start_epoch = int(saved["epoch"])
        history = list(saved.get("history", ()))
    started = time.perf_counter()
    best_validation = min(
        (
            float(item["validation_negative_elbo"])
            for item in history
            if item.get("validation_negative_elbo") is not None
        ),
        default=float("inf"),
    )
    best_path = checkpoint_path.with_name(f"{checkpoint_path.stem}.best{checkpoint_path.suffix}")
    for epoch in range(start_epoch, int(args.epochs)):
        model.train()
        order = np.random.default_rng(int(args.seed) + epoch).permutation(
            len(examples["train"])
        )
        train_loss = 0.0
        train_batches = 0
        for batch_index, start in enumerate(
            range(0, len(order), max(1, int(args.batch_size)))
        ):
            indices = order[start : start + int(args.batch_size)]
            batch = make_visual_mark_batch(
                [examples["train"][int(index)] for index in indices], device
            )
            terms = model.vfe_components(
                batch,
                generator=_generator(
                    device, int(args.seed) + epoch * 100_003 + batch_index
                ),
            )
            loss = terms.free_energy.sum() / terms.valid_pixel_count.sum().clamp(min=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            train_loss += float(loss.detach().item())
            train_batches += 1
        validation = evaluate_visual_model(
            model,
            examples["validation"],
            device=device,
            batch_size=int(args.batch_size),
            importance_samples=max(1, int(args.importance_samples)),
            prior_samples=max(1, int(args.prior_samples)),
            seed=int(args.seed) + epoch * 200_003,
        )
        validation_elbo = (
            None
            if validation is None
            else float(validation["negative_elbo_nats_per_observed_pixel"])
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_negative_elbo": train_loss / max(1, train_batches),
                "validation_negative_elbo": validation_elbo,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        checkpoint = {
            "schema": VISUAL_VAE_CHECKPOINT_SCHEMA,
            "model_id": VISUAL_MARK_VAE_MODEL_ID,
            "status": VISUAL_MARK_VAE_STATUS,
            "epoch": epoch + 1,
            "config": asdict(config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "manifest_path": str(Path(args.manifest).resolve()),
            "manifest_sha256": _sha256(Path(args.manifest)),
            "process_truth_used_as_training_input": False,
            "likelihood_family": "pixelwise_beta_on_registered_normalized_intensity",
        }
        _atomic_torch_save(checkpoint, checkpoint_path)
        if validation_elbo is not None and validation_elbo < best_validation:
            best_validation = validation_elbo
            _atomic_torch_save(checkpoint, best_path)
        if args.panel and (
            (epoch + 1) % max(1, int(args.panel_every_epochs)) == 0
            or epoch + 1 == int(args.epochs)
        ):
            _save_panel(
                Path(args.panel),
                model,
                examples["validation"] or examples["train"],
                device,
                seed=int(args.seed) + epoch,
            )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    evaluations = {
        name: evaluate_visual_model(
            model,
            examples[name],
            device=device,
            batch_size=int(args.batch_size),
            importance_samples=max(1, int(args.importance_samples)),
            prior_samples=max(1, int(args.prior_samples)),
            seed=int(args.seed) + 900_001 + index * 10_007,
        )
        for index, name in enumerate(SPLIT_NAMES)
    }
    report = {
        "schema": VISUAL_VAE_REPORT_SCHEMA,
        "model_id": VISUAL_MARK_VAE_MODEL_ID,
        "status": VISUAL_MARK_VAE_STATUS,
        "training_role": (
            "low-level action-conditioned visual mark likelihood; not a reward, "
            "aesthetic score, or persistent canvas-wide material state"
        ),
        "device": str(device),
        "config": asdict(config),
        "manifest": str(Path(args.manifest).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "completed_epochs": int(args.epochs),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "example_counts": {name: len(examples[name]) for name in SPLIT_NAMES},
        "history": history,
        "evaluation": evaluations,
        "admission_status": (
            "engineering_smoke_only; no counterfactual planner integration or "
            "hardware calibration claim"
        ),
        "process_truth_used_as_training_input": False,
        "approximations": [
            "registered images come from the provisional simulation camera likelihood",
            "local square crops are derived from selected mark support plus context",
            "pixelwise Beta likelihood omits explicit spatial pixel covariance",
            "compact brush posterior conditions predictions but is re-inferred upstream",
        ],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp = report_path.with_suffix(f"{report_path.suffix}.tmp")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--panel")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--condition-channels", type=int, default=16)
    parser.add_argument("--importance-samples", type=int, default=8)
    parser.add_argument("--prior-samples", type=int, default=8)
    parser.add_argument("--panel-every-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = train_visual_vae(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
