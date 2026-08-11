"""Reproducible scaling benchmark for the headless collection pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from .parallel_collect import collect_parallel
from .trajectory_corpus import discover_trajectory_shards, load_trajectory_shard


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = Path(args.output_dir).resolve() / f"parallel-{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, object]] = []
    for worker_count in args.worker_counts:
        run_dir = root / f"workers-{int(worker_count):02d}"
        collect_args = argparse.Namespace(
            output_dir=str(run_dir),
            trajectories=int(worker_count) * int(args.trajectories_per_worker),
            workers=int(worker_count),
            trajectories_per_runtime=1,
            motor_realization_profile="bounded_fixed_roll",
            seed=int(args.seed),
            canvas_size=int(args.canvas_size),
            spatial_grid_size=int(args.spatial_grid_size),
            stroke_tone_prior=args.stroke_tone_prior,
            max_transitions_per_trajectory=int(args.transitions_per_trajectory),
            max_wall_seconds=float(args.max_wall_seconds),
            torch_threads_per_worker=int(args.torch_threads_per_worker),
            split_ratios=(0.8, 0.1, 0.1),
            split_seed=int(args.seed) + 19,
        )
        started = time.perf_counter()
        summary = collect_parallel(collect_args)
        wall_seconds = time.perf_counter() - started
        shards = [load_trajectory_shard(path) for path in discover_trajectory_shards(run_dir)]
        transition_count = sum(shard.transition_count for shard in shards)
        results.append(
            {
                "workers": int(worker_count),
                "wall_seconds": wall_seconds,
                "trajectories": len(shards),
                "transitions": transition_count,
                "trajectories_per_second": len(shards) / max(wall_seconds, 1e-9),
                "transitions_per_second": transition_count / max(wall_seconds, 1e-9),
                "collection_summary": summary,
            }
        )
    baseline = float(results[0]["transitions_per_second"])
    for result in results:
        throughput = float(result["transitions_per_second"])
        speedup = throughput / baseline if baseline > 0.0 else None
        result["speedup_vs_first"] = speedup
        result["parallel_efficiency"] = (
            speedup / int(result["workers"]) if speedup is not None else None
        )
    report = {
        "schema": "parallel-collection-benchmark-v1",
        "claim_boundary": "uncalibrated simulation-only throughput benchmark",
        "trajectory_termination": "fixed_horizon_truncation",
        "results": results,
    }
    report_path = root / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark one or more independent headless collector counts."
    )
    parser.add_argument("--output-dir", default="runs/benchmarks")
    parser.add_argument("--worker-counts", type=int, nargs="+", default=(1, 4, 6, 8))
    parser.add_argument("--trajectories-per-worker", type=int, default=1)
    parser.add_argument("--transitions-per-trajectory", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--canvas-size", type=int, default=256)
    parser.add_argument("--spatial-grid-size", type=int, default=16)
    parser.add_argument(
        "--stroke-tone-prior", choices=("black", "white", "random"), default="random"
    )
    parser.add_argument("--max-wall-seconds", type=float, default=1800.0)
    parser.add_argument("--torch-threads-per-worker", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.worker_counts or min(args.worker_counts) <= 0:
        raise SystemExit("worker counts must be positive")
    if args.trajectories_per_worker <= 0 or args.transitions_per_trajectory <= 0:
        raise SystemExit("trajectory and transition counts must be positive")
    if max(args.worker_counts) > (os.cpu_count() or max(args.worker_counts)):
        print("warning: requested workers exceed logical CPU count", flush=True)
    report = run_benchmark(args)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
