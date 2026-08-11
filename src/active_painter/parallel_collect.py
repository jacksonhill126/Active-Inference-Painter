"""Parallel headless collection for the provisional sensor-simulation path.

Each process owns an independent MuJoCo model, camera clock, canvas, brush,
posterior, replay, hierarchy state, and RNG stream. Workers never share a
writable checkpoint. They emit full posterior trajectories; local training
patches are extracted later, after trajectory-level dataset splitting.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
import time

from .arm_agent_driver import SENSOR_OBSERVATION_ACCESS_MODE
from .trajectory_corpus import (
    TERMINATION_FIXED_HORIZON,
    TERMINATION_POLICY_STOP,
    TrajectoryRecorder,
    discover_trajectory_shards,
    split_trajectory_paths,
    write_split_manifest,
)


SENSOR_MOTOR_REALIZATION_PROFILE_NAMES = (
    "bounded_fixed_roll",
    "research_full_roll",
)


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    worker_id: int
    seed: int
    trajectory_count: int
    output_dir: str
    canvas_size: int
    spatial_grid_size: int
    stroke_tone_prior: float | None
    max_transitions_per_trajectory: int
    max_wall_seconds: float
    torch_threads: int
    motor_realization_profile: str


def _collect_worker(spec: WorkerSpec) -> dict[str, object]:
    # Imports stay inside the spawned process. This prevents a parent-created
    # CUDA/MuJoCo context from being inherited and makes worker ownership clear.
    import torch

    from .web_runtime import WebSimRuntime

    torch.set_num_threads(max(1, int(spec.torch_threads)))
    runtime = WebSimRuntime(
        canvas_size=spec.canvas_size,
        speed=1.0,
        planner_state_kind="spatial_material",
        spatial_grid_size=spec.spatial_grid_size,
        stroke_tone_prior=spec.stroke_tone_prior,
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
    )
    config = runtime.agent_driver.config
    provenance = {
        "collection_mode": "parallel_headless_process",
        "baseline": "uncalibrated_simulation_only_integration_baseline",
        "plant": "mujoco-robstride-electromechanical-v4",
        "observation_access_mode": SENSOR_OBSERVATION_ACCESS_MODE,
        "camera_update_clock": "action-conditioned-camera-update-v0",
        "camera_likelihood": "provisional_simulation_only_not_hardware_calibrated",
        "raw_camera_frames_persisted": False,
        "process_canvas_size": int(runtime.sim.config.canvas_size),
        "inference_driver_canvas_size": int(config.canvas_size),
        "spatial_posterior_grid_size": int(config.spatial_grid_size),
        "motor_realization_profile": spec.motor_realization_profile,
        "reduced_camera_acquisition_approximation": (
            "provisional sensor profile uses reduced native acquisition renders "
            "for throughput; this is not a sensor-equivalent camera realization"
        ),
        "stored_training_variables": (
            "camera-derived spatial posterior, posterior variance, selected mark, "
            "conditional motor realization, and inferred compact pre-stroke "
            "brush-load posterior"
        ),
        "process_truth_role": "not stored and not used as a training input",
        "code_build": runtime.code_build.version,
    }
    recorder = TrajectoryRecorder(
        spec.output_dir,
        config,
        worker_id=spec.worker_id,
        seed=spec.seed,
        provenance=provenance,
    )
    runtime.agent_driver.on_observed_transition = recorder.record_transition

    def complete_policy_stop(painting_index: int, belief) -> None:
        recorder.complete(
            belief,
            termination=TERMINATION_POLICY_STOP,
            painting_index=painting_index,
        )

    runtime.on_painting_complete = complete_policy_stop
    runtime.max_speed = True
    started = time.perf_counter()
    last_completion = started
    observed_completion_count = 0
    runtime.start()
    error: str | None = None
    try:
        while len(recorder.completed_paths) < spec.trajectory_count:
            now = time.perf_counter()
            if now - started > spec.max_wall_seconds:
                raise TimeoutError(
                    f"worker {spec.worker_id} collected "
                    f"{len(recorder.completed_paths)}/{spec.trajectory_count} trajectories "
                    f"within {spec.max_wall_seconds:.1f}s"
                )
            pending_error = runtime.agent_driver.diagnostics().get("plannerError")
            if pending_error:
                raise RuntimeError(str(pending_error))
            if (
                spec.max_transitions_per_trajectory > 0
                and recorder.pending_transition_count
                >= spec.max_transitions_per_trajectory
            ):
                belief = runtime.agent_driver.belief
                recorder.complete(
                    belief,
                    termination=TERMINATION_FIXED_HORIZON,
                    painting_index=None,
                )
                runtime.command({"type": "reset"})
                last_completion = now
            elif len(recorder.completed_paths) > observed_completion_count:
                last_completion = now
            observed_completion_count = len(recorder.completed_paths)
            time.sleep(0.02)
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        runtime.stop()
    return {
        "worker_id": spec.worker_id,
        "seed": spec.seed,
        "trajectory_count": len(recorder.completed_paths),
        "paths": [str(path) for path in recorder.completed_paths],
        "wall_seconds": time.perf_counter() - started,
        "last_completion_seconds": last_completion - started,
        "error": error,
    }


def _worker_specs(args: argparse.Namespace) -> list[WorkerSpec]:
    trajectories = int(args.trajectories)
    trajectories_per_runtime = max(
        1, int(getattr(args, "trajectories_per_runtime", 1))
    )
    counts: list[int] = []
    remaining = trajectories
    while remaining > 0:
        count = min(trajectories_per_runtime, remaining)
        counts.append(count)
        remaining -= count
    tone = {"black": 1.0, "white": 0.0, "random": None}[args.stroke_tone_prior]
    return [
        WorkerSpec(
            worker_id=index,
            seed=int(args.seed) + 100_003 * index,
            trajectory_count=counts[index],
            output_dir=str(Path(args.output_dir).resolve()),
            canvas_size=int(args.canvas_size),
            spatial_grid_size=int(args.spatial_grid_size),
            stroke_tone_prior=tone,
            max_transitions_per_trajectory=int(args.max_transitions_per_trajectory),
            max_wall_seconds=float(args.max_wall_seconds),
            torch_threads=int(args.torch_threads_per_worker),
            motor_realization_profile=str(
                getattr(args, "motor_realization_profile", "bounded_fixed_roll")
            ),
        )
        for index in range(len(counts))
    ]


def collect_parallel(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _worker_specs(args)
    context = multiprocessing.get_context("spawn")
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    concurrency = max(1, min(int(args.workers), len(specs)))
    with ProcessPoolExecutor(max_workers=concurrency, mp_context=context) as executor:
        futures = {executor.submit(_collect_worker, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(
                    {
                        "worker_id": spec.worker_id,
                        "seed": spec.seed,
                        "requested_trajectories": spec.trajectory_count,
                        "error": repr(exc),
                    }
                )
    paths = discover_trajectory_shards(output_dir)
    ratios = tuple(float(value) for value in args.split_ratios)
    splits = split_trajectory_paths(paths, seed=int(args.split_seed), ratios=ratios)
    manifest = write_split_manifest(
        output_dir / "split_manifest.json",
        splits,
        seed=int(args.split_seed),
        ratios=ratios,
    )
    summary = {
        "schema": "parallel-collection-summary-v1",
        "mode": "uncalibrated_simulation_only_integration_baseline",
        "workers": concurrency,
        "runtime_jobs": len(specs),
        "trajectories_per_runtime": int(
            getattr(args, "trajectories_per_runtime", 1)
        ),
        "requested_trajectories": int(args.trajectories),
        "written_trajectories": len(paths),
        "complete": len(paths) == int(args.trajectories) and not failures,
        "split_counts": {name: len(items) for name, items in splits.items()},
        "manifest": str(manifest),
        "worker_results": sorted(results, key=lambda item: int(item["worker_id"])),
        "worker_failures": sorted(failures, key=lambda item: int(item["worker_id"])),
        "requested_configuration": {
            "process_canvas_size": int(args.canvas_size),
            "spatial_posterior_grid_size": int(args.spatial_grid_size),
            "stroke_tone_prior": str(args.stroke_tone_prior),
            "max_transitions_per_trajectory": int(
                args.max_transitions_per_trajectory
            ),
            "plant": "mujoco-robstride-electromechanical-v4",
            "observation_access_mode": SENSOR_OBSERVATION_ACCESS_MODE,
            "motor_realization_profile": str(
                getattr(args, "motor_realization_profile", "bounded_fixed_roll")
            ),
        },
    }
    summary_path = output_dir / "collection_summary.json"
    temp_summary_path = summary_path.with_suffix(".json.tmp")
    temp_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    temp_summary_path.replace(summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect independent headless MuJoCo/camera posterior trajectories "
            "and split them before local-patch extraction."
        )
    )
    default_workers = max(1, min(8, (os.cpu_count() or 8) // 2))
    parser.add_argument("--output-dir", default="runs/corpus")
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument(
        "--trajectories-per-runtime",
        type=int,
        default=1,
        help=(
            "Recycle the complete simulator runtime after this many trajectories; "
            "one avoids the measured long-run slowdown."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--canvas-size", type=int, default=256)
    parser.add_argument("--spatial-grid-size", type=int, default=16)
    parser.add_argument(
        "--motor-realization-profile",
        choices=SENSOR_MOTOR_REALIZATION_PROFILE_NAMES,
        default="bounded_fixed_roll",
        help=(
            "Conditional motor support. research_full_roll adds symmetric dynamic "
            "roll sweeps for corpus evidence; it does not change actuator assignment."
        ),
    )
    parser.add_argument(
        "--stroke-tone-prior", choices=("black", "white", "random"), default="random"
    )
    parser.add_argument("--max-transitions-per-trajectory", type=int, default=128)
    parser.add_argument("--max-wall-seconds", type=float, default=3600.0)
    parser.add_argument("--torch-threads-per-worker", type=int, default=2)
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    parser.add_argument("--split-seed", type=int, default=104729)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.trajectories <= 0
        or args.workers <= 0
        or args.trajectories_per_runtime <= 0
    ):
        raise SystemExit(
            "--trajectories, --workers, and --trajectories-per-runtime must be positive"
        )
    summary = collect_parallel(args)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not bool(summary["complete"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
