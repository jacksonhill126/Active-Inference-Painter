"""Collect registered visual mark trajectories from isolated MuJoCo runtimes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Sequence

from .arm_agent_driver import SENSOR_OBSERVATION_ACCESS_MODE
from .trajectory_corpus import TERMINATION_FIXED_HORIZON, TERMINATION_POLICY_STOP
from .visual_trajectory_corpus import (
    VisualTrajectoryRecorder,
    load_visual_trajectory_shard,
    split_visual_trajectory_paths,
    write_visual_split_manifest,
)


@dataclass(frozen=True, slots=True)
class VisualWorkerSpec:
    worker_id: int
    seed: int
    trajectory_count: int
    output_dir: str
    canvas_size: int
    spatial_grid_size: int
    max_transitions_per_trajectory: int
    max_wall_seconds: float
    torch_threads: int
    motor_realization_profile: str
    trajectory_start_index: int = 0


def _worker(spec: VisualWorkerSpec) -> dict[str, object]:
    import torch

    from .web_runtime import WebSimRuntime

    torch.set_num_threads(max(1, spec.torch_threads))
    runtime = WebSimRuntime(
        canvas_size=spec.canvas_size,
        speed=1.0,
        planner_state_kind="spatial_material",
        spatial_grid_size=spec.spatial_grid_size,
        save_every_paintings=0,
        restart_on_stop=True,
        archive_dir=Path(spec.output_dir) / "rendered",
        telemetry_sample_period=0.0,
        driver_bootstrap_transitions=0,
        driver_bootstrap_train_steps=0,
        checkpoint_path=None,
        device="cpu",
        plant_backend="mujoco",
        observation_access_mode=SENSOR_OBSERVATION_ACCESS_MODE,
        provisional_sensor_policy=True,
        sensor_motor_realization_profile=spec.motor_realization_profile,
        seed=spec.seed,
        trajectory_start_index=spec.trajectory_start_index,
    )
    recorder = VisualTrajectoryRecorder(
        spec.output_dir,
        worker_id=spec.worker_id,
        seed=spec.seed,
        provenance={
            "collection_mode": "parallel_headless_process",
            "baseline": "uncalibrated_simulation_only_integration_baseline",
            "plant": "mujoco-robstride-electromechanical-v4",
            "observation_access_mode": SENSOR_OBSERVATION_ACCESS_MODE,
            "camera_update_clock": "action-conditioned-camera-update-v0",
            "camera_likelihood": "provisional_simulation_only_not_hardware_calibrated",
            "camera_rig": "provisional-compact-dual-imx296-v1",
            "registered_pre_post_camera_frames_persisted": True,
            "process_material_arrays_persisted": False,
            "exact_sim_state_persisted": False,
            "process_truth_role": "not stored and not used as a training input",
            "process_canvas_size": int(runtime.sim.config.canvas_size),
            "posterior_grid_size": int(runtime.agent_driver.config.spatial_grid_size),
            "motor_realization_profile": spec.motor_realization_profile,
            "code_build": runtime.code_build.version,
            "approximation": (
                "native camera acquisition is reduced for throughput before the "
                "declared canvas-plane registration; simulation-only evidence"
            ),
        },
    )
    runtime.agent_driver.on_visual_observed_transition = recorder.record_transition

    def complete_policy_stop(painting_index: int, _belief) -> None:
        recorder.complete(
            termination=TERMINATION_POLICY_STOP,
            painting_index=painting_index,
        )

    runtime.on_painting_complete = complete_policy_stop
    runtime.max_speed = True
    started = time.perf_counter()
    progress_path = Path(spec.output_dir) / f"visual-worker-{spec.worker_id:03d}-progress.json"

    def progress(status: str, error: str | None = None) -> None:
        payload = {
            "schema": "registered-visual-corpus-worker-progress-v1",
            "status": status,
            "worker_id": spec.worker_id,
            "seed": spec.seed,
            "elapsed_seconds": time.perf_counter() - started,
            "completed_trajectories": len(recorder.completed_paths),
            "requested_trajectories": spec.trajectory_count,
            "trajectory_start_index": spec.trajectory_start_index,
            "pending_transitions": recorder.pending_transition_count,
            "censoring_limit": spec.max_transitions_per_trajectory,
            "error": error,
        }
        temp = progress_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(progress_path)

    runtime.start()
    error: str | None = None
    last_progress = started - 10.0
    try:
        while len(recorder.completed_paths) < spec.trajectory_count:
            now = time.perf_counter()
            if now - started > spec.max_wall_seconds:
                raise TimeoutError(
                    f"visual worker {spec.worker_id} collected "
                    f"{len(recorder.completed_paths)}/{spec.trajectory_count} trajectories"
                )
            pending_error = runtime.agent_driver.diagnostics().get("plannerError")
            if pending_error:
                raise RuntimeError(str(pending_error))
            if (
                spec.max_transitions_per_trajectory > 0
                and recorder.pending_transition_count >= spec.max_transitions_per_trajectory
            ):
                recorder.complete(termination=TERMINATION_FIXED_HORIZON)
                runtime.command({"type": "reset"})
            if now - last_progress >= 10.0:
                progress("running")
                last_progress = now
            time.sleep(0.02)
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        progress("failed" if error else "complete", error)
        runtime.stop()
    return {
        "worker_id": spec.worker_id,
        "seed": spec.seed,
        "trajectory_count": len(recorder.completed_paths),
        "paths": [str(path) for path in recorder.completed_paths],
        "wall_seconds": time.perf_counter() - started,
    }


def collect_visual_corpus(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_count = int(args.trajectories)
    worker_count = min(max(1, int(args.workers)), trajectory_count)
    base, remainder = divmod(trajectory_count, worker_count)
    specs: list[VisualWorkerSpec] = []
    existing_paths: list[Path] = []
    for index in range(worker_count):
        seed = int(args.seed) + 100_003 * index
        requested = base + (1 if index < remainder else 0)
        pattern = f"visual-worker-{index:03d}-seed-{seed}-trajectory-*.npz"
        prior_paths = sorted(output_dir.glob(pattern))
        if prior_paths and not args.resume:
            raise FileExistsError(
                f"{len(prior_paths)} visual shards already exist for worker {index}; "
                "pass --resume to continue without overwriting"
            )
        prior_shards = [load_visual_trajectory_shard(path) for path in prior_paths]
        for shard in prior_shards:
            if int(shard.metadata["worker_id"]) != index or int(
                shard.metadata["worker_seed"]
            ) != seed:
                raise ValueError("existing visual shard worker provenance mismatch")
        prior_indices = [int(shard.metadata["trajectory_index"]) for shard in prior_shards]
        if len(prior_indices) != len(set(prior_indices)):
            raise ValueError("existing visual shards repeat a trajectory index")
        if len(prior_paths) > requested:
            raise ValueError(
                f"worker {index} already has more trajectories than requested"
            )
        existing_paths.extend(prior_paths)
        remaining = requested - len(prior_paths)
        if remaining <= 0:
            continue
        start_index = max(prior_indices, default=-1) + 1
        specs.append(
            VisualWorkerSpec(
                worker_id=index,
                seed=seed,
                trajectory_count=remaining,
                output_dir=str(output_dir),
                canvas_size=int(args.canvas_size),
                spatial_grid_size=int(args.spatial_grid_size),
                max_transitions_per_trajectory=int(args.max_transitions_per_trajectory),
                max_wall_seconds=float(args.max_worker_seconds),
                torch_threads=int(args.torch_threads),
                motor_realization_profile=str(args.motor_realization_profile),
                trajectory_start_index=start_index,
            )
        )
    started = time.perf_counter()
    if len(specs) == 1:
        results = [_worker(specs[0])]
    elif specs:
        context = mp.get_context("spawn")
        results = []
        with ProcessPoolExecutor(max_workers=len(specs), mp_context=context) as executor:
            futures = {executor.submit(_worker, spec): spec.worker_id for spec in specs}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
    else:
        results = []
    paths = existing_paths + [
        Path(path) for result in results for path in result["paths"]
    ]
    ratios = tuple(float(value) for value in args.split_ratios)
    splits = split_visual_trajectory_paths(paths, seed=int(args.seed), ratios=ratios)
    manifest_path = write_visual_split_manifest(
        args.manifest,
        splits,
        seed=int(args.seed),
        ratios=ratios,
    )
    summary = {
        "schema": "registered-visual-corpus-collection-report-v1",
        "output_dir": str(output_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "trajectory_count": len(paths),
        "resumed_existing_trajectory_count": len(existing_paths),
        "camera_example_count": sum(
            int(load_visual_trajectory_shard(path).example_count)
            for path in paths
        ),
        "split_trajectory_counts": {name: len(split) for name, split in splits.items()},
        "worker_results": sorted(results, key=lambda item: int(item["worker_id"])),
        "wall_seconds": time.perf_counter() - started,
        "process_truth_used_as_training_input": False,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp = report_path.with_suffix(f"{report_path.suffix}.tmp")
    temp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(report_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--trajectories", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="audit matching completed shards and continue at the next trajectory index",
    )
    parser.add_argument("--max-transitions-per-trajectory", type=int, default=12)
    parser.add_argument("--max-worker-seconds", type=float, default=14_400.0)
    parser.add_argument("--canvas-size", type=int, default=64)
    parser.add_argument("--spatial-grid-size", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument(
        "--motor-realization-profile",
        default="bounded_fixed_roll",
    )
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=(0.7, 0.15, 0.15),
        metavar=("TRAIN", "VALIDATION", "TEST"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    report = collect_visual_corpus(build_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
