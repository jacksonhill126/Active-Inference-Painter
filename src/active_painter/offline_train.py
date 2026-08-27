"""Centralized batched training from trajectory-isolated posterior shards.

This is conventional parameter learning around an active-inference runtime.
It pools only shared generative-model parameters. Per-worker canvas beliefs,
precision beliefs, brush state, passage history, and slow online latents are
not merged into the output checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from typing import Iterable, Sequence

import numpy as np
import torch

from .arm_agent_driver import (
    SENSOR_OBSERVATION_ACCESS_MODE,
    ArmActiveInferenceDriver,
)
from .config import PainterConfig
from .canvas_hierarchy import relational_observation_vector
from .env import StrokeAction
from .local_spatial import (
    LocalPatchReplayBuffer,
    LocalPatchTransition,
    local_patch_transition_from_states,
)
from .learning_lifecycle import (
    CONTINUE_SHARED_PRETRAINING,
    SHARED_PRETRAINING,
    normalized_training_provenance,
)
from .models import LocalSpatialDynamicsEnsemble
from .policies import MotorPrimitiveLatent
from .spatial_agent import SpatialActiveInferencePainter
from .spatial_state import SpatialCanvasState
from .trajectory_corpus import (
    SPLIT_NAMES,
    TERMINATION_POLICY_STOP,
    TrajectoryShard,
    load_trajectory_shard,
)


# These fields deliberately vary between independent simulation workers. They
# select frozen process microstructure/tooth realizations; they do not change
# the shared conditional likelihood architecture or training objective.
_WORKER_RANDOMNESS_CONFIG_FIELDS = frozenset(
    {
        "brush_seed",
        "canvas_grain_seed",
    }
)

# These fields describe which conditional motor counterfactuals were offered
# during collection. The selected motor kind is stored on every transition,
# so pooled likelihood training may span these collection profiles as long as
# one canonical motor-conditioning vocabulary fits the declared action raster.
_COLLECTION_POLICY_CONFIG_FIELDS = frozenset(
    {
        "motor_forecast_workers",
        "motor_realization_candidate_limit",
        "motor_realization_kinds",
    }
)
_CANONICAL_POOLED_MOTOR_KIND_ORDER = (
    "cartesian_ik",
    "upper_arm_fixed_roll_positive",
    "upper_arm_fixed_roll_negative",
    "joint_spline",
    "elbow_pivot",
    "upper_arm_roll_positive",
    "upper_arm_roll_negative",
)


def _shared_training_config_payload(raw: dict[str, object]) -> dict[str, object]:
    """Return the configuration fields that must match for pooled training."""

    return {
        name: value
        for name, value in raw.items()
        if name
        not in (_WORKER_RANDOMNESS_CONFIG_FIELDS | _COLLECTION_POLICY_CONFIG_FIELDS)
    }


def _pooled_training_config_payloads(
    raw_payloads: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Return one model config for compatible collection-policy profiles."""

    if not raw_payloads:
        raise ValueError("at least one PainterConfig payload is required")
    shared = _shared_training_config_payload(raw_payloads[0])
    for candidate in raw_payloads[1:]:
        if _shared_training_config_payload(candidate) != shared:
            raise ValueError(
                "all shards in one training run must share model-relevant "
                "PainterConfig fields"
            )

    observed_kinds = {
        str(kind)
        for payload in raw_payloads
        for kind in payload.get("motor_realization_kinds", ())
    }
    ordered_kinds = [
        kind for kind in _CANONICAL_POOLED_MOTOR_KIND_ORDER if kind in observed_kinds
    ]
    ordered_kinds.extend(sorted(observed_kinds.difference(ordered_kinds)))
    motor_channel_capacity = max(
        0,
        int(raw_payloads[0].get("spatial_action_channels", 7)) - 7,
    )
    if len(ordered_kinds) > motor_channel_capacity:
        raise ValueError(
            "pooled collection profiles require more motor-condition channels "
            "than the shared spatial action raster declares"
        )

    pooled = dict(raw_payloads[0])
    pooled["motor_realization_kinds"] = tuple(ordered_kinds)
    pooled["motor_realization_candidate_limit"] = len(ordered_kinds)
    pooled["motor_forecast_workers"] = max(
        int(payload.get("motor_forecast_workers", 1)) for payload in raw_payloads
    )
    return pooled


def _config_from_json(raw: dict[str, object]) -> PainterConfig:
    defaults = PainterConfig()
    normalized = dict(raw)
    for name, default_value in asdict(defaults).items():
        if isinstance(default_value, tuple) and name in normalized:
            normalized[name] = tuple(normalized[name])  # type: ignore[arg-type]
    return PainterConfig(**normalized)


def _load_manifest(path: Path | str) -> tuple[dict[str, list[Path]], dict[str, object]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "trajectory-split-manifest-v1":
        raise ValueError("unsupported split manifest")
    root = Path(str(payload["root"]))
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("split manifest has no file mapping")
    splits = {
        name: [root / str(item) for item in files.get(name, [])]
        for name in SPLIT_NAMES
    }
    ids = payload.get("trajectory_ids", {})
    flattened = [str(item) for name in SPLIT_NAMES for item in ids.get(name, [])]
    if len(flattened) != len(set(flattened)):
        raise ValueError("a trajectory appears in more than one dataset split")
    return splits, payload


def _stroke_action(vector: np.ndarray) -> StrokeAction:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.shape[0] != 8:
        raise ValueError("stored stroke action must have eight values")
    return StrokeAction(
        x0=float(values[0]),
        y0=float(values[1]),
        x1=float(values[2]),
        y1=float(values[3]),
        width=float(values[4]),
        amount=float(values[5]),
        tone=float(values[6]),
        curvature=float(values[7]),
    )


def _state(
    material: np.ndarray,
    logvar: np.ndarray,
    revision: int,
    model_id: str,
) -> SpatialCanvasState:
    return SpatialCanvasState(
        material=np.asarray(material, dtype=np.float32),
        logvar=np.asarray(logvar, dtype=np.float32),
        posterior_revision=max(0, int(revision)),
        inference_model_id=str(model_id) or "trajectory-corpus-v1:unknown",
        calibration_status="provisional_simulation_only_not_hardware_calibrated",
    )


def _local_transitions(
    shards: Iterable[TrajectoryShard],
    config: PainterConfig,
) -> list[LocalPatchTransition]:
    transitions: list[LocalPatchTransition] = []
    for shard in shards:
        for index in range(shard.transition_count):
            state = _state(
                shard.state_material[index],
                shard.state_logvar[index],
                int(shard.state_revision[index]),
                str(shard.state_model_id[index]),
            )
            next_state = _state(
                shard.next_material[index],
                shard.next_logvar[index],
                int(shard.next_revision[index]),
                str(shard.next_model_id[index]),
            )
            action = _stroke_action(shard.action[index])
            primitive = MotorPrimitiveLatent(kind=str(shard.motor_kind[index]))
            transition = local_patch_transition_from_states(
                state, action, next_state, config, primitive
            )
            if transition is not None:
                transitions.append(transition)
    return transitions


@torch.no_grad()
def _heldout_nll(
    dynamics: LocalSpatialDynamicsEnsemble,
    transitions: Sequence[LocalPatchTransition],
    *,
    device: torch.device,
    batch_size: int,
) -> float | None:
    if not transitions:
        return None
    total = 0.0
    count = 0
    for start in range(0, len(transitions), max(1, int(batch_size))):
        chunk = list(transitions[start : start + max(1, int(batch_size))])
        indexed = list(enumerate(chunk))
        batch = LocalPatchReplayBuffer._make_batch(
            indexed,
            device,
            max(item.bounds.height for item in chunk),
            max(item.bounds.width for item in chunk),
        )
        per_member_sample = dynamics.per_sample_nll(
            batch.material,
            batch.action,
            batch.next_material,
            batch.mask,
        )
        total += float(per_member_sample.mean(dim=0).sum().item())
        count += len(chunk)
    return total / max(1, count)


def _terminal_training_data(
    agent: SpatialActiveInferencePainter,
    shards: Iterable[TrajectoryShard],
) -> tuple[np.ndarray, np.ndarray]:
    fields: list[np.ndarray] = []
    relations: list[np.ndarray] = []
    for shard in shards:
        if shard.metadata.get("termination") != TERMINATION_POLICY_STOP:
            continue
        canvas = SpatialCanvasState(
            material=shard.final_coarse_material.astype(np.float32),
            logvar=shard.final_coarse_logvar.astype(np.float32),
            inference_model_id="trajectory-corpus-v1:terminal-camera-posterior",
            calibration_status="provisional_simulation_only_not_hardware_calibrated",
        )
        agent.add_composition_canvas(canvas)
        fields.append(canvas.material.astype(np.float32, copy=True))
        # Hard region clustering is deterministic but CPU-bound. Compute it
        # once per terminal trajectory, outside the GPU gradient loop.
        relations.append(relational_observation_vector(canvas.material, agent.cfg))
    if not fields:
        return (
            np.empty(
                (
                    0,
                    agent.cfg.spatial_material_channels,
                    agent.cfg.spatial_grid_size,
                    agent.cfg.spatial_grid_size,
                ),
                dtype=np.float32,
            ),
            np.empty((0, 0), dtype=np.float32),
        )
    return np.stack(fields), np.stack(relations)


def _train_composition_precomputed(
    agent: SpatialActiveInferencePainter,
    fields: np.ndarray,
    relations: np.ndarray,
    *,
    gradient_steps: int,
    batch_size: int,
    seed: int,
) -> float | None:
    model = agent.composition
    optimizer = agent.composition_optimizer
    if model is None or optimizer is None or fields.shape[0] == 0 or gradient_steps <= 0:
        return None
    rng = np.random.default_rng(seed)
    last_loss: float | None = None
    effective_batch = max(1, min(int(batch_size), int(fields.shape[0])))
    for _ in range(int(gradient_steps)):
        indices = rng.integers(0, fields.shape[0], size=effective_batch)
        field_batch = torch.tensor(fields[indices], device=agent.device)
        relation_batch = torch.tensor(relations[indices], device=agent.device)
        loss = model.training_loss(
            field_batch,
            relational_observations=relation_batch,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last_loss = float(loss.item())
    agent.last_composition_loss = last_loss
    return last_loss


def train_from_manifest(args: argparse.Namespace) -> dict[str, object]:
    splits, manifest = _load_manifest(args.manifest)
    loaded = {
        name: [load_trajectory_shard(path) for path in paths]
        for name, paths in splits.items()
    }
    all_shards = [shard for name in SPLIT_NAMES for shard in loaded[name]]
    if not all_shards:
        raise ValueError("the corpus manifest contains no trajectories")
    config_payloads: list[dict[str, object]] = []
    for shard in all_shards:
        candidate = shard.metadata.get("config")
        if not isinstance(candidate, dict):
            raise ValueError("trajectory has no PainterConfig payload")
        config_payloads.append(candidate)
    config_payload = _pooled_training_config_payloads(config_payloads)
    base_config = _config_from_json(config_payload)
    train_shards = loaded["train"]
    train_transition_count = sum(shard.transition_count for shard in train_shards)
    if train_transition_count <= 0:
        raise ValueError("the training split has no observed transitions")
    batch_size = min(max(1, int(args.batch_size)), train_transition_count)
    config_overrides: dict[str, object] = {
        "batch_size": batch_size,
        "replay_capacity": max(base_config.replay_capacity, train_transition_count + 1),
    }
    if getattr(args, "hidden_channels", None) is not None:
        config_overrides["spatial_hidden_channels"] = int(args.hidden_channels)
    if getattr(args, "residual_blocks", None) is not None:
        config_overrides["spatial_residual_blocks"] = int(args.residual_blocks)
    if getattr(args, "ensemble_size", None) is not None:
        config_overrides["spatial_ensemble_size"] = int(args.ensemble_size)
    config = replace(
        base_config,
        **config_overrides,
    )
    input_checkpoint = Path(args.input_checkpoint) if args.input_checkpoint else None
    output_checkpoint = Path(args.output_checkpoint).resolve()
    driver = ArmActiveInferenceDriver(
        config=config,
        bootstrap_transitions=0,
        bootstrap_train_steps=0,
        checkpoint_path=input_checkpoint,
        checkpoint_load_mode=CONTINUE_SHARED_PRETRAINING,
        observation_access_mode=SENSOR_OBSERVATION_ACCESS_MODE,
        provisional_sensor_policy=True,
        seed=int(args.seed),
        device=args.device,
    )
    agent = driver.agent
    if not isinstance(agent, SpatialActiveInferencePainter) or not isinstance(
        agent.dynamics, LocalSpatialDynamicsEnsemble
    ):
        raise RuntimeError("central trainer requires the spatial local-patch model")
    parameter_count = sum(parameter.numel() for parameter in agent.dynamics.parameters())
    if agent.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(agent.device)

    # A supplied shared checkpoint continues shared parameters/optimizer
    # moments, never its sampling pool. Replay may have unknown split
    # provenance, so the lifecycle load mode already resets it; these clears
    # make that boundary locally visible before this manifest is materialized.
    agent.replay.data.clear()
    agent.composition_replay.data.clear()
    agent.passage_replay.data.clear()
    agent.passage_step_replay.data.clear()

    transitions = {
        name: _local_transitions(loaded[name], config) for name in SPLIT_NAMES
    }
    replay = agent.replay
    if not isinstance(replay, LocalPatchReplayBuffer):
        raise RuntimeError("central trainer requires LocalPatchReplayBuffer")
    for transition in transitions["train"]:
        replay.add(transition)

    evaluation_before = {
        name: _heldout_nll(
            agent.dynamics,
            transitions[name],
            device=agent.device,
            batch_size=int(args.evaluation_batch_size),
        )
        for name in ("validation", "test")
    }
    started = time.perf_counter()
    dynamics_loss = (
        agent.train_dynamics(gradient_steps=int(args.dynamics_steps))
        if int(args.dynamics_steps) > 0
        else None
    )
    dynamics_seconds = time.perf_counter() - started

    terminal_fields, terminal_relations = _terminal_training_data(agent, train_shards)
    terminal_canvas_count = int(terminal_fields.shape[0])
    composition_started = time.perf_counter()
    composition_loss = _train_composition_precomputed(
        agent,
        terminal_fields,
        terminal_relations,
        gradient_steps=int(args.composition_steps),
        batch_size=batch_size,
        seed=int(args.seed) + 701,
    )
    composition_seconds = time.perf_counter() - composition_started
    evaluation_after = {
        name: _heldout_nll(
            agent.dynamics,
            transitions[name],
            device=agent.device,
            batch_size=int(args.evaluation_batch_size),
        )
        for name in SPLIT_NAMES
    }

    driver.trained_transitions = train_transition_count
    driver.checkpoint_path = output_checkpoint
    driver.checkpoint_provenance = normalized_training_provenance({
        "training_role": SHARED_PRETRAINING,
        "mode": "shared_pretraining",
        "objective": "conditional likelihood/VFE parameter learning",
        "manifest": str(Path(args.manifest).resolve()),
        "split_unit": "entire_trajectory_before_patch_extraction",
        "training_split_only": True,
        "pooled": "shared generative-model parameters and training replay",
        "not_pooled": (
            "online canvas posterior, precision beliefs, brush state, passage "
            "history, persistent canvas latent, persistent relational latent"
        ),
        "validation_or_test_used_for_gradient_updates": False,
        "allowed_worker_config_variation": sorted(
            _WORKER_RANDOMNESS_CONFIG_FIELDS | _COLLECTION_POLICY_CONFIG_FIELDS
        ),
        "simulation_claim": "uncalibrated simulation-only integration baseline",
    })
    driver._save_checkpoint_if_due(force=True)
    if driver.checkpoint_status != "saved":
        raise RuntimeError(
            f"checkpoint save failed: {driver.checkpoint_last_error or driver.checkpoint_status}"
        )
    report = {
        "schema": "centralized-training-report-v1",
        "mode": "shared_pretraining",
        "checkpoint": str(output_checkpoint),
        "device": str(agent.device),
        "trajectory_counts": {name: len(loaded[name]) for name in SPLIT_NAMES},
        "transition_counts": {name: len(transitions[name]) for name in SPLIT_NAMES},
        "terminal_training_canvases": terminal_canvas_count,
        "dynamics_gradient_steps": int(args.dynamics_steps),
        "dynamics_parameter_count": int(parameter_count),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(agent.device))
            if agent.device.type == "cuda"
            else None
        ),
        "architecture": {
            "hidden_channels": int(config.spatial_hidden_channels),
            "residual_blocks": int(config.spatial_residual_blocks),
            "ensemble_size": int(config.spatial_ensemble_size),
        },
        "composition_gradient_steps_requested": int(args.composition_steps),
        "dynamics_loss_last_budget_mean": dynamics_loss,
        "composition_loss_last": composition_loss,
        "heldout_nll_before": evaluation_before,
        "heldout_nll_after": evaluation_after,
        "nll_units": (
            "mean conditional Gaussian negative log density without the constant "
            "0.5*log(2*pi), per valid independent material cell-channel, "
            "averaged across ensemble members"
        ),
        "timing_seconds": {
            "dynamics": dynamics_seconds,
            "composition": composition_seconds,
        },
        "manifest_schema": manifest.get("schema"),
        "leakage_guard": "trajectory split completed before local patch extraction",
        "validation_or_test_used_for_gradient_updates": False,
        "allowed_worker_config_variation": sorted(
            _WORKER_RANDOMNESS_CONFIG_FIELDS | _COLLECTION_POLICY_CONFIG_FIELDS
        ),
        "claim_boundary": "uncalibrated simulation-only integration baseline",
    }
    report_path = Path(args.report_path) if args.report_path else output_checkpoint.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train shared generative-model parameters centrally from a "
            "trajectory-first corpus."
        )
    )
    parser.add_argument("--manifest", default="runs/corpus/split_manifest.json")
    parser.add_argument("--output-checkpoint", default="runs/checkpoints/shared_pretraining.pt")
    parser.add_argument("--input-checkpoint", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--dynamics-steps", type=int, default=1000)
    parser.add_argument("--composition-steps", type=int, default=200)
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--residual-blocks", type=int, default=None)
    parser.add_argument("--ensemble-size", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dynamics_steps < 0 or args.composition_steps < 0:
        raise SystemExit("gradient step budgets must be non-negative")
    report = train_from_manifest(args)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
