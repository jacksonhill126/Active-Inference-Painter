"""Reproducible AI-109 predictive learning-curve experiment runner.

The runner creates nested whole-trajectory training subsets while leaving the
accepted validation and test trajectories untouched. It trains/evaluates:

* Gaussian CNN data, capacity, and ensemble-size curves;
* the current conditional patch cVAE at full data across seeds;
* the identity-plus-consequence mixture hypothesis at full data across seeds.

Every subprocess writes an atomic checkpoint/report and is resumable. This is
conventional generative-model estimation and evaluation around the active-
inference painter; none of the curve metrics enter painting policy selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable, Mapping, Sequence

import numpy as np

from .conditional_vae_train import _manifest_paths
from .uncertainty_calibration import CALIBRATION_REPORT_SCHEMA
from .trajectory_corpus import (
    SPLIT_NAMES,
    TrajectoryShard,
    load_trajectory_shard,
    transition_condition_labels,
    write_split_manifest,
)


LEARNING_CURVE_REPORT_SCHEMA = "predictive-learning-curves-v1"


def nested_training_order(shards: Sequence[TrajectoryShard]) -> list[TrajectoryShard]:
    """Greedily order trajectories for nested, condition-diverse subsets."""

    if not shards:
        return []
    label_sets: dict[str, set[str]] = {}
    global_counts: Counter[str] = Counter()
    for shard in shards:
        labels = {
            f"{family}={label}"
            for index in range(shard.transition_count)
            for family, label in transition_condition_labels(shard, index).items()
            if family not in {"phase", "termination", "corpus_schema"}
        }
        label_sets[shard.trajectory_id] = labels
        global_counts.update(labels)
    remaining = list(shards)
    selected: list[TrajectoryShard] = []
    selected_counts: Counter[str] = Counter()
    while remaining:
        def score(shard: TrajectoryShard) -> tuple[float, float, str]:
            labels = label_sets[shard.trajectory_id]
            novelty = sum(
                1.0 / max(1, global_counts[label])
                for label in labels
                if selected_counts[label] == 0
            )
            balance = sum(
                1.0 / ((1.0 + selected_counts[label]) * max(1, global_counts[label]))
                for label in labels
            )
            return novelty, balance, shard.trajectory_id

        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        selected_counts.update(label_sets[chosen.trajectory_id])
    return selected


def fraction_count(total: int, fraction: float) -> int:
    if total <= 0:
        raise ValueError("training trajectory count must be positive")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("training fractions must lie in (0, 1]")
    return max(1, min(total, int(math.ceil(total * float(fraction)))))


def build_fraction_manifests(
    manifest_path: Path | str,
    output_dir: Path | str,
    fractions: Sequence[float],
    *,
    seed: int,
) -> dict[float, Path]:
    paths, _ = _manifest_paths(manifest_path)
    loaded_train = [load_trajectory_shard(path) for path in paths["train"]]
    order = nested_training_order(loaded_train)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[float, Path] = {}
    for fraction in sorted(set(float(value) for value in fractions)):
        count = fraction_count(len(order), fraction)
        path = output_dir / f"fraction-{count:02d}-of-{len(order):02d}.json"
        splits = {
            "train": [shard.path for shard in order[:count]],
            "validation": paths["validation"],
            "test": paths["test"],
        }
        write_split_manifest(
            path,
            splits,
            seed=int(seed),
            ratios=(count / len(order), 0.0, 0.0),
        )
        result[fraction] = path.resolve()
    order_path = output_dir / "nested-training-order.json"
    order_payload = {
        "schema": "nested-condition-diverse-training-order-v1",
        "source_manifest": str(Path(manifest_path).resolve()),
        "trajectory_ids": [shard.trajectory_id for shard in order],
        "transition_counts": [shard.transition_count for shard in order],
        "selection_inputs": "camera-posterior/action condition labels only",
        "process_truth_used": False,
    }
    _atomic_json(order_path, order_payload)
    return result


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _run_command(command: Sequence[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            list(command),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"learning-curve subprocess failed ({completed.returncode}): "
            f"{' '.join(command)}\n" + "\n".join(tail)
        )
    return elapsed


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _calibration_metrics(payload: Mapping[str, object], family: str) -> dict[str, object]:
    models = payload["models"]
    assert isinstance(models, dict)
    model = models[family]
    assert isinstance(model, dict)
    output: dict[str, object] = {
        "validation_fitted_variance_scale": model["validation_fitted_variance_scale"],
        "validation_fitted_mixture_component_variance_scale": model[
            "validation_fitted_mixture_component_variance_scale"
        ],
        "test_calibration_gate": model["test_provisional_m2_calibration_gate"],
        "amount_support_stress": model["amount_support_stress"],
    }
    for split in SPLIT_NAMES:
        split_payload = model[split]
        assert isinstance(split_payload, dict)
        variants = split_payload["moment_gaussian_variants"]
        assert isinstance(variants, dict)
        scaled = variants["validation_scaled_model_total"]
        mixture_calibration = split_payload["predictive_mixture_calibration"]
        assert isinstance(mixture_calibration, dict)
        output[split] = {
            "mixture_nll_nats_per_element": split_payload[
                "exact_or_sample_mixture_nll_nats_mean"
            ],
            "scaled_moment_metrics": scaled,
            "scaled_predictive_mixture_metrics": mixture_calibration[
                "validation_scaled"
            ],
        }
    output["multi_step_rollout"] = model["multi_step_rollout"]
    return output


def _training_resources(report: Mapping[str, object], family: str) -> dict[str, object]:
    if family == "cnn":
        return {
            "training_seconds": report["timing_seconds"]["dynamics"],  # type: ignore[index]
            "parameter_count": report["dynamics_parameter_count"],
            "peak_cuda_memory_bytes": report["peak_cuda_memory_bytes"],
            "optimizer_steps": report["dynamics_gradient_steps"],
            "architecture": report["architecture"],
        }
    return {
        "training_seconds": report["training_seconds"],
        "parameter_count": report["parameter_count"],
        "peak_cuda_memory_bytes": report["peak_cuda_memory_bytes"],
        "optimizer_steps": report["gradient_steps"],
        "architecture": report["model_config"],
    }


def _run_one(
    *,
    family: str,
    axis: str,
    level: str,
    manifest: Path,
    seed: int,
    steps: int,
    hidden: int,
    blocks: int,
    ensemble: int,
    output_dir: Path,
    device: str,
    batch_size: int,
    calibration_latent_samples: int,
    rollout_latent_samples: int,
) -> dict[str, object]:
    run_id = f"{family}-{axis}-{level}-seed-{int(seed)}"
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint.pt"
    training_report = run_dir / "training_report.json"
    calibration_report = run_dir / "calibration_report.json"
    training_log = run_dir / "training.log"
    calibration_log = run_dir / "calibration.log"
    if family == "cnn":
        train_command = [
            sys.executable,
            "-m",
            "active_painter.offline_train",
            "--manifest",
            str(manifest),
            "--output-checkpoint",
            str(checkpoint),
            "--report-path",
            str(training_report),
            "--device",
            device,
            "--seed",
            str(seed),
            "--batch-size",
            str(batch_size),
            "--evaluation-batch-size",
            str(batch_size),
            "--dynamics-steps",
            str(steps),
            "--composition-steps",
            "0",
            "--hidden-channels",
            str(hidden),
            "--residual-blocks",
            str(blocks),
            "--ensemble-size",
            str(ensemble),
        ]
        checkpoint_flag = "--cnn-checkpoint"
    elif family == "cvae":
        train_command = [
            sys.executable,
            "-m",
            "active_painter.conditional_vae_train",
            "--manifest",
            str(manifest),
            "--output-checkpoint",
            str(checkpoint),
            "--report-path",
            str(training_report),
            "--device",
            device,
            "--seed",
            str(seed),
            "--batch-size",
            str(batch_size),
            "--evaluation-batch-size",
            str(batch_size),
            "--gradient-steps",
            str(steps),
            "--hidden-channels",
            str(hidden),
            "--residual-blocks",
            str(blocks),
            "--latent-dim",
            "16",
            "--ensemble-size",
            str(ensemble),
            "--importance-samples",
            "4",
            "--prior-samples",
            "4",
        ]
        checkpoint_flag = "--cvae-checkpoint"
    elif family == "mixture":
        train_command = [
            sys.executable,
            "-m",
            "active_painter.mixture_transition",
            "--manifest",
            str(manifest),
            "--output-checkpoint",
            str(checkpoint),
            "--report-path",
            str(training_report),
            "--device",
            device,
            "--seed",
            str(seed),
            "--batch-size",
            str(batch_size),
            "--evaluation-batch-size",
            str(batch_size),
            "--gradient-steps",
            str(steps),
            "--hidden-channels",
            str(hidden),
            "--residual-blocks",
            str(blocks),
            "--ensemble-size",
            str(ensemble),
        ]
        checkpoint_flag = "--mixture-checkpoint"
    else:
        raise ValueError(f"unknown learning-curve family: {family}")

    if not checkpoint.exists() or not training_report.exists():
        print(f"training {run_id} ({steps} steps)", flush=True)
        _run_command(train_command, training_log)
    calibration_current = False
    if calibration_report.exists():
        try:
            calibration_current = _json(calibration_report).get("schema") == CALIBRATION_REPORT_SCHEMA
        except (OSError, ValueError, json.JSONDecodeError):
            calibration_current = False
    if not calibration_current:
        print(f"calibrating {run_id}", flush=True)
        calibration_command = [
            sys.executable,
            "-m",
            "active_painter.uncertainty_calibration",
            "--manifest",
            str(manifest),
            checkpoint_flag,
            str(checkpoint),
            "--output",
            str(calibration_report),
            "--device",
            device,
            "--batch-size",
            str(batch_size),
            "--latent-samples",
            str(calibration_latent_samples),
            "--rollout-latent-samples",
            str(rollout_latent_samples),
            "--rollout-horizons",
            "1",
            "2",
            "4",
            "8",
            "--seed",
            str(seed + 10_000),
        ]
        _run_command(calibration_command, calibration_log)
    training = _json(training_report)
    calibration = _json(calibration_report)
    manifest_payload = _json(manifest)
    return {
        "run_id": run_id,
        "family": family,
        "axis": axis,
        "level": level,
        "seed": int(seed),
        "manifest": str(manifest),
        "train_trajectory_count": len(manifest_payload["files"]["train"]),  # type: ignore[index]
        "train_transition_count": int(
            training.get("transition_counts", training.get("patch_counts"))["train"]  # type: ignore[index]
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "resources": _training_resources(training, family),
        "metrics": _calibration_metrics(calibration, family),
    }


def aggregate_runs(runs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for run in runs:
        key = (str(run["family"]), str(run["axis"]), str(run["level"]))
        groups.setdefault(key, []).append(run)
    result: list[dict[str, object]] = []
    for (family, axis, level), members in sorted(groups.items()):
        def values(split: str, metric: str) -> np.ndarray:
            return np.asarray(
                [
                    member["metrics"][split][metric]  # type: ignore[index]
                    for member in members
                ],
                dtype=np.float64,
            )

        test_nll = values("test", "mixture_nll_nats_per_element")
        validation_nll = values("validation", "mixture_nll_nats_per_element")
        train_nll = values("train", "mixture_nll_nats_per_element")
        coverage_90 = np.asarray(
            [
                member["metrics"]["test"]["scaled_predictive_mixture_metrics"]["coverage_90"]  # type: ignore[index]
                for member in members
            ],
            dtype=np.float64,
        )
        horizon_rmse: dict[str, object] = {}
        for horizon in ("1", "2", "4", "8"):
            samples = [
                member["metrics"]["multi_step_rollout"]["test"]["horizons"][horizon]["rmse"]  # type: ignore[index]
                for member in members
            ]
            finite = np.asarray([value for value in samples if value is not None], dtype=np.float64)
            horizon_rmse[horizon] = {
                "mean": float(np.mean(finite)) if finite.size else None,
                "std": float(np.std(finite)) if finite.size else None,
            }
        result.append(
            {
                "family": family,
                "axis": axis,
                "level": level,
                "seed_count": len(members),
                "train_trajectory_count": members[0]["train_trajectory_count"],
                "train_transition_count": members[0]["train_transition_count"],
                "train_nll": {"mean": float(train_nll.mean()), "std": float(train_nll.std())},
                "validation_nll": {
                    "mean": float(validation_nll.mean()),
                    "std": float(validation_nll.std()),
                },
                "test_nll": {"mean": float(test_nll.mean()), "std": float(test_nll.std())},
                "test_coverage_90": {
                    "mean": float(coverage_90.mean()),
                    "std": float(coverage_90.std()),
                },
                "test_multistep_rmse": horizon_rmse,
                "resources": {
                    "parameter_count": members[0]["resources"]["parameter_count"],  # type: ignore[index]
                    "training_seconds_mean": float(
                        np.mean([member["resources"]["training_seconds"] for member in members])  # type: ignore[index]
                    ),
                    "peak_cuda_memory_bytes_max": max(
                        int(member["resources"]["peak_cuda_memory_bytes"] or 0)  # type: ignore[index]
                        for member in members
                    ),
                    "optimizer_steps": members[0]["resources"]["optimizer_steps"],  # type: ignore[index]
                },
            }
        )
    return result


def diagnose(aggregates: Sequence[Mapping[str, object]]) -> dict[str, object]:
    lookup = {
        (str(item["family"]), str(item["axis"]), str(item["level"])): item
        for item in aggregates
    }

    def compare(
        left_key: tuple[str, str, str],
        right_key: tuple[str, str, str],
        minimum: float = 0.05,
    ) -> dict[str, object] | None:
        left = lookup.get(left_key)
        right = lookup.get(right_key)
        if left is None or right is None:
            return None
        left_nll = float(left["test_nll"]["mean"])  # type: ignore[index]
        right_nll = float(right["test_nll"]["mean"])  # type: ignore[index]
        noise = max(
            minimum,
            float(left["test_nll"]["std"]),  # type: ignore[index]
            float(right["test_nll"]["std"]),  # type: ignore[index]
        )
        improvement = left_nll - right_nll
        return {
            "left": left_key,
            "right": right_key,
            "left_test_nll": left_nll,
            "right_test_nll": right_nll,
            "right_improvement_nats": improvement,
            "required_improvement_nats": noise,
            "material_improvement": bool(improvement >= noise),
        }

    data = compare(("cnn", "data", "small"), ("cnn", "data", "full"))
    capacity = compare(("cnn", "capacity", "base"), ("cnn", "capacity", "large"))
    ensemble = compare(("cnn", "ensemble", "one"), ("cnn", "ensemble", "five"))
    mixture = compare(("cnn", "data", "full"), ("mixture", "family", "full"))
    cvae = compare(("cnn", "data", "full"), ("cvae", "family", "full"))

    def overfit(
        simple_key: tuple[str, str, str],
        complex_key: tuple[str, str, str],
    ) -> dict[str, object] | None:
        simple = lookup.get(simple_key)
        complex_model = lookup.get(complex_key)
        if simple is None or complex_model is None:
            return None
        simple_train = float(simple["train_nll"]["mean"])  # type: ignore[index]
        complex_train = float(complex_model["train_nll"]["mean"])  # type: ignore[index]
        simple_test = float(simple["test_nll"]["mean"])  # type: ignore[index]
        complex_test = float(complex_model["test_nll"]["mean"])  # type: ignore[index]
        simple_gap = simple_test - simple_train
        complex_gap = complex_test - complex_train
        return {
            "simple": simple_key,
            "complex": complex_key,
            "train_improvement_nats": simple_train - complex_train,
            "test_improvement_nats": simple_test - complex_test,
            "simple_test_minus_train_gap": simple_gap,
            "complex_test_minus_train_gap": complex_gap,
            "overfit_signal": bool(
                simple_train - complex_train >= 0.05
                and simple_test - complex_test <= -0.05
                and complex_gap - simple_gap >= 0.05
            ),
        }

    def family_shape(
        baseline_key: tuple[str, str, str],
        candidate_key: tuple[str, str, str],
    ) -> dict[str, object] | None:
        baseline = lookup.get(baseline_key)
        candidate = lookup.get(candidate_key)
        if baseline is None or candidate is None:
            return None
        baseline_coverage = float(baseline["test_coverage_90"]["mean"])  # type: ignore[index]
        candidate_coverage = float(candidate["test_coverage_90"]["mean"])  # type: ignore[index]
        return {
            "baseline": baseline_key,
            "candidate": candidate_key,
            "baseline_absolute_90_coverage_error": abs(baseline_coverage - 0.90),
            "candidate_absolute_90_coverage_error": abs(candidate_coverage - 0.90),
            "coverage_shape_improved": bool(
                abs(candidate_coverage - 0.90) + 0.01
                <= abs(baseline_coverage - 0.90)
            ),
        }
    return {
        "decision_rules_declared_in_runner": {
            "material_nll_change": (
                "test NLL improvement must exceed max(0.05 nats, either across-seed std)"
            ),
            "overfit_signal": (
                "training NLL improves while validation/test NLL worsens and the gap grows"
            ),
            "calibration": (
                "use the frozen AI-107 interval gates; NLL improvement alone does not admit a model"
            ),
        },
        "data_sensitivity": data,
        "capacity_sensitivity": capacity,
        "ensemble_sensitivity": ensemble,
        "data_overfit_check": overfit(
            ("cnn", "data", "small"), ("cnn", "data", "full")
        ),
        "capacity_overfit_check": overfit(
            ("cnn", "capacity", "base"), ("cnn", "capacity", "large")
        ),
        "mixture_family_vs_cnn": mixture,
        "mixture_calibration_shape_vs_cnn": family_shape(
            ("cnn", "data", "full"), ("mixture", "family", "full")
        ),
        "cvae_family_vs_cnn": cvae,
        "cvae_calibration_shape_vs_cnn": family_shape(
            ("cnn", "data", "full"), ("cvae", "family", "full")
        ),
        "composition_hierarchy": {
            "status": "structurally_unavailable",
            "reason": (
                "the AI-108 corpus contains fixed-horizon truncations and zero policy-selected "
                "terminal canvases; treating truncations as terminal composition evidence would "
                "violate the model semantics"
            ),
        },
    }


def run_learning_curves(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    fractions = tuple(float(value) for value in args.fractions)
    manifests = build_fraction_manifests(
        args.manifest, root / "manifests", fractions, seed=int(args.seed)
    )
    full_fraction = max(fractions)
    full_manifest = manifests[full_fraction]
    paths, _ = _manifest_paths(full_manifest)
    full_train_count = len(paths["train"])
    fraction_levels = {min(fractions): "small", max(fractions): "full"}
    middle = [value for value in fractions if value not in fraction_levels]
    for value in middle:
        fraction_levels[value] = "medium"

    specs: list[dict[str, object]] = []
    for fraction in sorted(fractions):
        count = len(_manifest_paths(manifests[fraction])[0]["train"])
        steps = max(100, int(round(args.cnn_steps * count / full_train_count)))
        for seed in args.seeds:
            specs.append(
                dict(
                    family="cnn",
                    axis="data",
                    level=fraction_levels[fraction],
                    manifest=manifests[fraction],
                    seed=seed,
                    steps=steps,
                    hidden=32,
                    blocks=2,
                    ensemble=3,
                )
            )
    for level, hidden, blocks, ensemble in (
        ("small", 16, 1, 3),
        ("large", 64, 3, 3),
    ):
        for seed in args.seeds:
            specs.append(
                dict(
                    family="cnn",
                    axis="capacity",
                    level=level,
                    manifest=full_manifest,
                    seed=seed,
                    steps=args.cnn_steps,
                    hidden=hidden,
                    blocks=blocks,
                    ensemble=ensemble,
                )
            )
    # Alias the already scheduled base/full data point in aggregation by adding
    # no duplicate training run; a lightweight alias is created after execution.
    for level, ensemble in (("one", 1), ("five", 5)):
        for seed in args.seeds:
            specs.append(
                dict(
                    family="cnn",
                    axis="ensemble",
                    level=level,
                    manifest=full_manifest,
                    seed=seed,
                    steps=args.cnn_steps,
                    hidden=32,
                    blocks=2,
                    ensemble=ensemble,
                )
            )
    if not args.skip_cvae:
        for seed in args.seeds:
            specs.append(
                dict(
                    family="cvae",
                    axis="family",
                    level="full",
                    manifest=full_manifest,
                    seed=seed,
                    steps=args.cvae_steps,
                    hidden=32,
                    blocks=2,
                    ensemble=3,
                )
            )
    if not args.skip_mixture:
        for seed in args.seeds:
            specs.append(
                dict(
                    family="mixture",
                    axis="family",
                    level="full",
                    manifest=full_manifest,
                    seed=seed,
                    steps=args.mixture_steps,
                    hidden=32,
                    blocks=2,
                    ensemble=3,
                )
            )

    runs: list[dict[str, object]] = []
    for index, spec in enumerate(specs, start=1):
        print(f"AI-109 run {index}/{len(specs)}", flush=True)
        runs.append(
            _run_one(
                **spec,
                output_dir=root,
                device=args.device,
                batch_size=int(args.batch_size),
                calibration_latent_samples=int(args.calibration_latent_samples),
                rollout_latent_samples=int(args.rollout_latent_samples),
            )
        )

    # Reuse base/full runs as the base-capacity and three-member ensemble points.
    base_full = [
        run
        for run in runs
        if run["family"] == "cnn" and run["axis"] == "data" and run["level"] == "full"
    ]
    aliases: list[dict[str, object]] = []
    for run in base_full:
        for axis, level in (("capacity", "base"), ("ensemble", "three")):
            alias = dict(run)
            alias["axis"] = axis
            alias["level"] = level
            alias["run_id"] = f"{run['run_id']}-alias-{axis}-{level}"
            alias["reuses_run_id"] = run["run_id"]
            aliases.append(alias)
    all_runs = runs + aliases
    aggregates = aggregate_runs(all_runs)
    report: dict[str, object] = {
        "schema": LEARNING_CURVE_REPORT_SCHEMA,
        "source_manifest": str(Path(args.manifest).resolve()),
        "output_dir": str(root),
        "seeds": [int(seed) for seed in args.seeds],
        "fractions": list(fractions),
        "run_count_excluding_aliases": len(runs),
        "runs": runs,
        "aggregates": aggregates,
        "diagnosis": diagnose(aggregates),
        "split_use": {
            "train": "gradient updates and train diagnostics",
            "validation": "one scalar variance temperature only",
            "test": "frozen evaluation only",
        },
        "claim_boundary": (
            "uncalibrated simulation-only learning-curve evidence; not policy active, "
            "not painting quality, and not hardware calibration"
        ),
    }
    _atomic_json(root / "learning_curve_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run resumable AI-109 learning curves")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--seeds", type=int, nargs="+", default=(109, 211, 307))
    parser.add_argument("--fractions", type=float, nargs="+", default=(0.3, 0.6, 1.0))
    parser.add_argument("--cnn-steps", type=int, default=900)
    parser.add_argument("--cvae-steps", type=int, default=1200)
    parser.add_argument("--mixture-steps", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--calibration-latent-samples", type=int, default=8)
    parser.add_argument("--rollout-latent-samples", type=int, default=4)
    parser.add_argument("--skip-cvae", action="store_true")
    parser.add_argument("--skip-mixture", action="store_true")
    return parser


def main() -> None:
    report = run_learning_curves(build_parser().parse_args())
    compact = {
        "schema": report["schema"],
        "run_count": report["run_count_excluding_aliases"],
        "report": str(Path(report["output_dir"]) / "learning_curve_report.json"),
        "diagnosis": report["diagnosis"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
