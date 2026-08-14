from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from active_painter.brush_loading import BrushLoadBelief
from active_painter.config import PainterConfig
from active_painter.canvas_hierarchy import (
    HierarchicalCanvasModel,
    relational_observation_vector,
)
from active_painter.env import StrokeAction
from active_painter.offline_train import (
    _pooled_training_config_payloads,
    _shared_training_config_payload,
    _train_composition_precomputed,
    train_from_manifest,
)
from active_painter.policies import MotorPrimitiveLatent
from active_painter.parallel_collect import _stop_evidence_summary, _worker_specs
from active_painter.spatial_state import SpatialCanvasState
from active_painter.spatial_agent import SpatialActiveInferencePainter
from active_painter.trajectory_corpus import (
    TERMINATION_FIXED_HORIZON,
    TERMINATION_POLICY_STOP,
    SPLIT_STRATEGY,
    TrajectoryRecorder,
    corpus_condition_audit,
    discover_trajectory_shards,
    load_trajectory_shard,
    split_trajectory_paths,
    transition_condition_labels,
    write_split_manifest,
)


def _config() -> PainterConfig:
    return PainterConfig(
        canvas_size=8,
        planner_state_kind="spatial_material",
        spatial_grid_size=8,
        spatial_hidden_channels=4,
        spatial_residual_blocks=1,
        spatial_ensemble_size=1,
        composition_enabled=False,
        composition_gap_precision=0.0,
        candidate_policies=2,
        planning_horizon=1,
        batch_size=2,
        replay_capacity=64,
    )


def _state(value: float, revision: int) -> SpatialCanvasState:
    material = np.zeros((6, 8, 8), dtype=np.float32)
    material[0] = value
    material[3] = 0.7
    material[5] = float(value > 0.02)
    return SpatialCanvasState(
        material=material,
        logvar=np.full_like(material, -5.0),
        posterior_revision=revision,
        inference_model_id=f"test-camera-posterior:{revision}",
        calibration_status="provisional_simulation_only_not_hardware_calibrated",
    )


def _write_trajectories(root: Path, count: int = 6) -> list[Path]:
    base_config = _config()
    paths: list[Path] = []
    for index in range(count):
        config = replace(
            base_config,
            canvas_grain_seed=3000 + index,
            brush_seed=4000 + index,
        )
        recorder = TrajectoryRecorder(
            root,
            config,
            worker_id=index % 2,
            seed=1000 + index,
            provenance={
                "observation_access_mode": "sensor_equivalent",
                "process_truth_role": "not stored and not used as a training input",
            },
        )
        before = _state(0.01 * index, 2 * index)
        after = _state(0.04 + 0.01 * index, 2 * index + 1)
        action = StrokeAction(
            0.15,
            0.20 + 0.05 * (index % 2),
            0.80,
            0.75,
            0.08,
            0.55,
            float(index % 2),
            curvature=(-0.15, 0.0, 0.15)[index % 3],
        )
        recorder.record_transition(
            before,
            action,
            MotorPrimitiveLatent(
                kind=(
                    "cartesian_ik"
                    if index % 2 == 0
                    else "upper_arm_fixed_roll_positive"
                )
            ),
            after,
        )
        paths.append(
            recorder.complete(
                after,
                termination=(
                    TERMINATION_POLICY_STOP
                    if index < count - 1
                    else TERMINATION_FIXED_HORIZON
                ),
                painting_index=index + 1 if index < count - 1 else None,
            )
        )
    return paths


def test_shared_training_config_allows_declared_worker_and_collection_variation() -> None:
    baseline = asdict(_config())
    independent_worker = dict(baseline)
    independent_worker["canvas_grain_seed"] = 91
    independent_worker["brush_seed"] = 92

    assert _shared_training_config_payload(
        independent_worker
    ) == _shared_training_config_payload(baseline)

    different_model = dict(independent_worker)
    different_model["spatial_hidden_channels"] = (
        int(different_model["spatial_hidden_channels"]) + 1
    )
    assert _shared_training_config_payload(
        different_model
    ) != _shared_training_config_payload(baseline)

    fixed_profile = dict(baseline)
    fixed_profile["motor_realization_kinds"] = (
        "cartesian_ik",
        "upper_arm_fixed_roll_positive",
        "upper_arm_fixed_roll_negative",
    )
    fixed_profile["motor_realization_candidate_limit"] = 3
    fixed_profile["motor_forecast_workers"] = 3
    full_roll_profile = dict(baseline)
    full_roll_profile["motor_realization_kinds"] = (
        *fixed_profile["motor_realization_kinds"],
        "upper_arm_roll_positive",
        "upper_arm_roll_negative",
    )
    full_roll_profile["motor_realization_candidate_limit"] = 5
    full_roll_profile["motor_forecast_workers"] = 5
    pooled = _pooled_training_config_payloads(
        [fixed_profile, full_roll_profile]
    )
    assert pooled["motor_realization_kinds"] == full_roll_profile[
        "motor_realization_kinds"
    ]
    assert pooled["motor_realization_candidate_limit"] == 5


def test_trajectory_shards_round_trip_full_posteriors_and_split_without_leakage(
    tmp_path: Path,
) -> None:
    paths = _write_trajectories(tmp_path)
    shard = load_trajectory_shard(paths[0])

    assert shard.metadata["split_unit"] == "entire_trajectory_before_patch_extraction"
    assert shard.metadata["process_truth_used_as_training_input"] is False
    assert shard.transition_count == 1
    assert shard.state_material.shape == (1, 6, 8, 8)
    assert shard.state_coarse_material.shape == (1, 6, 8, 8)
    assert shard.action.shape == (1, 8)
    assert shard.motor_kind.tolist() == ["cartesian_ik"]

    splits = split_trajectory_paths(
        paths,
        seed=41,
        ratios=(0.5, 0.25, 0.25),
    )
    split_sets = [set(items) for items in splits.values()]
    assert set.union(*split_sets) == set(paths)
    assert not (split_sets[0] & split_sets[1])
    assert not (split_sets[0] & split_sets[2])
    assert not (split_sets[1] & split_sets[2])
    assert all(split_sets)

    manifest_path = write_split_manifest(
        tmp_path / "split_manifest.json",
        splits,
        seed=41,
        ratios=(0.5, 0.25, 0.25),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = [
        trajectory_id
        for split_ids in manifest["trajectory_ids"].values()
        for trajectory_id in split_ids
    ]
    assert len(ids) == len(set(ids)) == len(paths)
    assert manifest["split_strategy"] == SPLIT_STRATEGY
    assert manifest["condition_audit"]["process_truth_used_for_labels"] is False
    assert manifest["condition_audit"]["checks"][
        "trajectory_ids_unique_across_splits"
    ] is True
    assert manifest["condition_audit"]["checks"][
        "all_shards_deny_process_truth_training_input"
    ] is True
    assert manifest["provenance_summary"]["canvas_sizes"] == [8]
    assert manifest["provenance_summary"][
        "all_trajectories_use_live_scale_256_canvas"
    ] is False
    assert all(
        len(item["sha256"]) == 64
        for entries in manifest["file_integrity"].values()
        for item in entries
    )

    interrupted_temp = paths[0].with_name(f"{paths[0].stem}.tmp.npz")
    interrupted_temp.write_bytes(paths[0].read_bytes())
    assert discover_trajectory_shards(tmp_path) == sorted(paths)


def test_trajectory_records_stop_posterior_trace_and_resets_between_episodes(
    tmp_path: Path,
) -> None:
    config = _config()
    recorder = TrajectoryRecorder(
        tmp_path,
        config,
        worker_id=3,
        seed=17,
        provenance={"process_truth_role": "not stored"},
    )
    first = {
        "schema": "stop-decision-diagnostic-v1",
        "planning_revision": 1,
        "stroke_count": 0,
        "believed_material_coverage": 0.12,
        "selected_stop": False,
        "stop_rank": 4,
        "stop_posterior": 0.02,
        "stop_had_lowest_efe_but_prior_demoted": False,
        "stop_log_posterior_odds_vs_best_continuation": -3.2,
    }
    second = {
        **first,
        "planning_revision": 2,
        "stroke_count": 1,
        "believed_material_coverage": 0.22,
        "stop_rank": 2,
        "stop_posterior": 0.30,
        "stop_had_lowest_efe_but_prior_demoted": True,
        "stop_log_posterior_odds_vs_best_continuation": -0.4,
    }
    recorder.record_stop_decision(first)
    recorder.record_stop_decision(second)
    path = recorder.complete(
        _state(0.22, 2),
        termination=TERMINATION_FIXED_HORIZON,
    )
    evidence = load_trajectory_shard(path).metadata["stop_decision_evidence"]

    assert evidence["decision_count"] == 2
    assert evidence["selected_stop_count"] == 0
    assert evidence["prior_demoted_lowest_efe_stop_count"] == 1
    assert evidence["maximum_stop_posterior"] == 0.30
    assert evidence["minimum_stop_rank"] == 2
    assert evidence["initial_believed_material_coverage"] == 0.12
    assert evidence["final_believed_material_coverage"] == 0.22
    assert evidence["trace"] == [first, second]

    collection = _stop_evidence_summary([path])
    assert collection["terminations"] == {"fixed_horizon_truncation": 1}
    assert collection["decision_count"] == 2
    assert collection["maximum_stop_posterior"] == 0.30

    next_path = recorder.complete(
        _state(0.0, 3),
        termination=TERMINATION_FIXED_HORIZON,
    )
    next_evidence = load_trajectory_shard(next_path).metadata[
        "stop_decision_evidence"
    ]
    assert next_evidence["decision_count"] == 0
    assert next_evidence["trace"] == []


def test_central_trainer_uses_train_split_and_labels_shared_pretraining(
    tmp_path: Path,
) -> None:
    paths = _write_trajectories(tmp_path / "corpus")
    splits = split_trajectory_paths(
        paths,
        seed=7,
        ratios=(0.5, 0.25, 0.25),
    )
    manifest = write_split_manifest(
        tmp_path / "corpus" / "split_manifest.json",
        splits,
        seed=7,
        ratios=(0.5, 0.25, 0.25),
    )
    checkpoint = tmp_path / "shared.pt"
    report = train_from_manifest(
        argparse.Namespace(
            manifest=str(manifest),
            output_checkpoint=str(checkpoint),
            input_checkpoint=None,
            report_path=str(tmp_path / "training_report.json"),
            device="cpu",
            seed=17,
            batch_size=2,
            evaluation_batch_size=2,
            dynamics_steps=1,
            composition_steps=0,
        )
    )

    assert checkpoint.is_file()
    assert report["mode"] == "shared_pretraining"
    assert report["validation_or_test_used_for_gradient_updates"] is False
    assert report["transition_counts"]["train"] == len(splits["train"])
    assert report["transition_counts"]["validation"] == len(splits["validation"])
    assert report["transition_counts"]["test"] == len(splits["test"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    provenance = payload["training_provenance"]
    assert provenance["mode"] == "shared_pretraining"
    assert provenance["training_split_only"] is True
    assert "online canvas posterior" in provenance["not_pooled"]


def test_precomputed_relations_bypass_repeated_cpu_region_extraction(
    monkeypatch,
) -> None:
    config = PainterConfig(
        canvas_size=8,
        planner_state_kind="spatial_material",
        spatial_grid_size=8,
        spatial_hidden_channels=4,
        spatial_residual_blocks=1,
        spatial_ensemble_size=1,
        composition_hidden_channels=4,
        canvas_latent_channels=2,
        relational_latent_dim=4,
        hierarchy_hidden_dim=8,
        candidate_policies=2,
        planning_horizon=1,
        batch_size=2,
    )
    agent = SpatialActiveInferencePainter(config, seed=3, device="cpu")
    fields = np.stack([_state(0.06, 0).material, _state(0.09, 1).material])
    relations = np.stack(
        [relational_observation_vector(field, config) for field in fields]
    )

    def fail_if_recomputed(self, batch):
        raise AssertionError("relational observations were recomputed inside the gradient loop")

    monkeypatch.setattr(
        HierarchicalCanvasModel,
        "relational_observations",
        fail_if_recomputed,
    )
    loss = _train_composition_precomputed(
        agent,
        fields,
        relations,
        gradient_steps=2,
        batch_size=2,
        seed=5,
    )

    assert loss is not None
    assert np.isfinite(loss)


def test_condition_labels_use_stored_posterior_and_brush_belief_not_process_truth(
    tmp_path: Path,
) -> None:
    config = _config()
    recorder = TrajectoryRecorder(
        tmp_path,
        config,
        worker_id=7,
        seed=81,
        provenance={"process_truth_role": "not stored and not used as a training input"},
    )
    before = _state(0.20, 4)
    before.material[1] = 0.3
    before.material[5] = 1.0
    after = _state(0.25, 5)
    action = StrokeAction(
        0.01,
        0.05,
        0.01,
        0.25,
        0.05,
        0.8,
        0.0,
        curvature=-0.18,
    )
    recorder.record_transition(
        before,
        action,
        MotorPrimitiveLatent(kind="upper_arm_fixed_roll_negative"),
        after,
        BrushLoadBelief(0.7, 0.01, 0.1, 0.02),
    )
    path = recorder.complete(
        after,
        termination=TERMINATION_POLICY_STOP,
        painting_index=1,
    )
    shard = load_trajectory_shard(path)
    labels = transition_condition_labels(shard, 0)

    assert labels == labels | {
        "tone": "white",
        "surface": "wet_over_wet",
        "overlap": "overlap",
        "edge": "edge",
        "width": "narrow",
        "length": "short",
        "curvature": "negative_strong",
        "direction": "vertical",
        "reach": "outer",
        "motor": "upper_arm_fixed_roll_negative",
        "brush_context": "available",
    }
    audit = corpus_condition_audit(
        {"train": [path], "validation": [], "test": []}
    )
    assert audit["process_truth_used_for_labels"] is False
    assert audit["overall_transition_counts"]["surface=wet_over_wet"] == 1
    assert audit["checks"]["all_shards_deny_process_truth_training_input"] is True


def test_parallel_collection_recycles_runtime_per_trajectory_by_default() -> None:
    base = argparse.Namespace(
        trajectories=5,
        workers=2,
        seed=101,
        output_dir="runs/test",
        canvas_size=256,
        spatial_grid_size=16,
        stroke_tone_prior="random",
        max_transitions_per_trajectory=8,
        max_wall_seconds=60.0,
        torch_threads_per_worker=1,
    )
    specs = _worker_specs(base)
    assert [spec.trajectory_count for spec in specs] == [1, 1, 1, 1, 1]
    assert len({spec.seed for spec in specs}) == 5

    base.trajectories_per_runtime = 2
    specs = _worker_specs(base)
    assert [spec.trajectory_count for spec in specs] == [2, 2, 1]
