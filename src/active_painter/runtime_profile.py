"""Manifested phase profiler for the M1 formal painting-policy baseline.

This is a computational evidence harness. It does not score painting quality,
alter any policy term, or treat profiling measurements as observations for the
painting model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, TypeVar

import numpy as np
from PIL import Image
import torch

from .arm_agent_driver import (
    ORACLE_OBSERVATION_ACCESS_MODE,
    ArmActiveInferenceDriver,
    execute_stroke_action,
)
from .arm_sim import ArmPainterSim
from .config import (
    M1_FORMAL_POLICY_BASELINE_ID,
    LEGACY_MATERIAL_HIERARCHY_DIAGNOSTIC_ID,
    PainterConfig,
    painting_policy_profile_id,
)
from .learning_lifecycle import ORACLE_DIAGNOSTIC_EXECUTION
from .policies import MotorPrimitiveLatent
from .spatial_agent import SpatialActiveInferencePainter
from .spatial_state import SpatialCanvasState
from .version import package_version, source_fingerprint


AI113_PROFILE_SCHEMA = "ai113-runtime-phase-profile-v1"
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class RuntimeProfileSettings:
    seed: int = 20260827
    planning_repeats: int = 3
    inference_repeats: int = 5
    training_repeats: int = 3
    gradient_steps: int = 2
    serialization_repeats: int = 3
    rendering_repeats: int = 5
    training_batch_size: int = 32
    motor_forecast_workers: int = 1
    device: str = "cpu"
    quick: bool = False


def representative_config(settings: RuntimeProfileSettings) -> PainterConfig:
    """The bounded configuration used by the ordinary spatial web runtime."""

    if settings.quick:
        candidate_policies = 4
        planning_horizon = 1
        motor_forecast_candidates = 1
        motor_forecast_samples = 1
        motor_kinds = ("cartesian_ik",)
    else:
        candidate_policies = 32
        planning_horizon = 4
        motor_forecast_candidates = 2
        motor_forecast_samples = 3
        motor_kinds = (
            "cartesian_ik",
            "joint_spline",
            "elbow_pivot",
            "upper_arm_roll_positive",
            "upper_arm_roll_negative",
        )
    return PainterConfig(
        canvas_size=64,
        planner_state_kind="spatial_material",
        spatial_grid_size=16,
        candidate_policies=candidate_policies,
        planning_horizon=planning_horizon,
        passage_proposal_mix=0.45 if not settings.quick else 0.0,
        passage_plan_proposal_mix=0.15 if not settings.quick else 0.0,
        motor_forecast_candidates=motor_forecast_candidates,
        motor_forecast_samples=motor_forecast_samples,
        motor_forecast_hz=45.0,
        motor_forecast_workers=settings.motor_forecast_workers,
        motor_realization_kinds=motor_kinds,
        motor_realization_candidate_limit=len(motor_kinds),
        batch_size=settings.training_batch_size,
        replay_capacity=max(256, settings.training_batch_size * 4),
        bootstrap_composition_train_steps=0,
        canvas_grain_seed=settings.seed + 101,
        brush_seed=settings.seed + 211,
    )


def _process_memory() -> dict[str, int | None]:
    """Current/peak process memory without an optional psutil dependency."""

    if os.name != "nt":
        return {
            "working_set_bytes": None,
            "peak_working_set_bytes": None,
            "private_bytes": None,
        }

    from ctypes import wintypes

    size_t = ctypes.c_size_t

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", size_t),
            ("WorkingSetSize", size_t),
            ("QuotaPeakPagedPoolUsage", size_t),
            ("QuotaPagedPoolUsage", size_t),
            ("QuotaPeakNonPagedPoolUsage", size_t),
            ("QuotaNonPagedPoolUsage", size_t),
            ("PagefileUsage", size_t),
            ("PeakPagefileUsage", size_t),
            ("PrivateUsage", size_t),
        ]

    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(
        handle,
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        return {
            "working_set_bytes": None,
            "peak_working_set_bytes": None,
            "private_bytes": None,
        }
    return {
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        "private_bytes": int(counters.PrivateUsage),
    }


def measure_phase(
    function: Callable[[], _T],
    *,
    device: str = "cpu",
) -> tuple[_T, dict[str, object]]:
    """Measure wall time, process CPU time, memory, and CUDA allocation."""

    cuda_active = str(device).startswith("cuda")
    if cuda_active and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but torch.cuda is unavailable")
    if cuda_active:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    memory_before = _process_memory()
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    result = function()
    if cuda_active:
        torch.cuda.synchronize()
    wall_seconds = max(0.0, time.perf_counter() - wall_started)
    cpu_seconds = max(0.0, time.process_time() - cpu_started)
    memory_after = _process_memory()
    logical_cpus = max(1, int(os.cpu_count() or 1))
    average_active_cpu_cores = cpu_seconds / max(wall_seconds, 1e-12)
    record: dict[str, object] = {
        "wall_seconds": wall_seconds,
        "process_cpu_seconds": cpu_seconds,
        "average_active_cpu_cores": average_active_cpu_cores,
        "normalized_process_cpu_percent": (
            100.0 * average_active_cpu_cores / logical_cpus
        ),
        "working_set_before_bytes": memory_before["working_set_bytes"],
        "working_set_after_bytes": memory_after["working_set_bytes"],
        "peak_working_set_bytes": memory_after["peak_working_set_bytes"],
        "private_bytes_after": memory_after["private_bytes"],
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if cuda_active else None
        ),
        "cuda_measurement_status": (
            "peak_allocation_measured_utilization_not_available_from_torch"
            if cuda_active
            else "not_used_profile_device_cpu"
        ),
    }
    return result, record


def summarize_samples(samples: list[dict[str, object]]) -> dict[str, object]:
    if not samples:
        raise ValueError("phase summary requires at least one sample")
    numeric_keys = (
        "wall_seconds",
        "process_cpu_seconds",
        "average_active_cpu_cores",
        "normalized_process_cpu_percent",
    )
    summary: dict[str, object] = {"sample_count": len(samples), "samples": samples}
    for key in numeric_keys:
        values = [float(sample[key]) for sample in samples]
        summary[key] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    for key in (
        "working_set_after_bytes",
        "peak_working_set_bytes",
        "private_bytes_after",
        "peak_cuda_memory_bytes",
    ):
        values = [int(sample[key]) for sample in samples if sample.get(key) is not None]
        summary[key] = max(values) if values else None
    return summary


def _repeat_phase(
    repeats: int,
    function: Callable[[], _T],
    *,
    device: str = "cpu",
) -> tuple[list[_T], dict[str, object]]:
    results: list[_T] = []
    samples: list[dict[str, object]] = []
    for _ in range(max(1, int(repeats))):
        result, sample = measure_phase(function, device=device)
        results.append(result)
        samples.append(sample)
    return results, summarize_samples(samples)


def _jsonable_dataclass(value: object) -> object:
    return asdict(value) if is_dataclass(value) else value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_revision(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _environment(root: Path, profile_device: str) -> dict[str, object]:
    cuda = torch.cuda.is_available()
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_version": str(torch.__version__),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "numpy_version": np.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_utilization": (
            "not available from torch; peak allocation and paired wall/CPU time reported"
            if str(profile_device).startswith("cuda")
            else "not sampled: representative profile explicitly uses CPU"
            if cuda
            else "not applicable: CUDA unavailable"
        ),
        "profile_compute_device": profile_device,
        "initial_process_memory": _process_memory(),
        "code_revision": _git_revision(root),
        "source_fingerprint": source_fingerprint(root),
        "package_version": package_version(),
    }


def _collect_observed_training_transitions(
    driver: ArmActiveInferenceDriver,
    sim: ArmPainterSim,
    count: int,
) -> tuple[
    SpatialCanvasState,
    object,
    SpatialCanvasState,
    MotorPrimitiveLatent,
]:
    latest: tuple[
        SpatialCanvasState,
        object,
        SpatialCanvasState,
        MotorPrimitiveLatent,
    ] | None = None
    primitive = MotorPrimitiveLatent("cartesian_ik")
    for index in range(max(1, int(count))):
        if index > 0 and index % 8 == 0:
            sim.reset_pose()
            sim.canvas.clear()
        before = driver._planner_state(sim)
        action = driver.agent.policy_sampler._stroke()
        execute_stroke_action(sim, action, dt=1.0 / 90.0)
        after = driver._planner_state(sim)
        if not isinstance(before, SpatialCanvasState) or not isinstance(
            after, SpatialCanvasState
        ):
            raise RuntimeError("AI-113 requires the spatial planner")
        driver._add_transition_to_agent(
            before,
            action,
            after,
            primitive,
            evidence_source=ORACLE_DIAGNOSTIC_EXECUTION,
        )
        latest = (before, action, after, primitive)
    if latest is None:
        raise RuntimeError("no observed transition was collected")
    return latest


def run_profile(
    settings: RuntimeProfileSettings,
    run_dir: Path,
) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    started_utc = datetime.now(timezone.utc)
    total_cpu_started = time.process_time()
    total_wall_started = time.perf_counter()
    if settings.device not in {"cpu", "cuda"}:
        raise ValueError("profile device must be 'cpu' or 'cuda'")
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profile requested but torch.cuda is unavailable")
    environment = _environment(root, settings.device)
    config = representative_config(settings)
    profile_id = painting_policy_profile_id(config)
    if profile_id != M1_FORMAL_POLICY_BASELINE_ID:
        raise RuntimeError(f"representative profile resolved to {profile_id!r}")
    resolved_config_path = run_dir / "resolved-config.json"
    _write_json(resolved_config_path, asdict(config))

    sim = ArmPainterSim(config)
    driver = ArmActiveInferenceDriver(
        config=config,
        bootstrap_transitions=0,
        bootstrap_train_steps=0,
        observation_access_mode=ORACLE_OBSERVATION_ACCESS_MODE,
        seed=settings.seed,
        device=settings.device,
        checkpoint_provenance={
            "purpose": "AI-113 computational phase profile",
            "claim_boundary": "oracle diagnostic; not painting-quality evidence",
        },
    )
    if not isinstance(driver.agent, SpatialActiveInferencePainter):
        raise RuntimeError("AI-113 requires SpatialActiveInferencePainter")

    latest_box: dict[str, object] = {}

    def collect() -> int:
        latest_box["transition"] = _collect_observed_training_transitions(
            driver,
            sim,
            settings.training_batch_size,
        )
        return len(driver.agent.replay)

    replay_sizes, collection_phase = _repeat_phase(
        1,
        collect,
        device=settings.device,
    )
    transition = latest_box["transition"]
    assert isinstance(transition, tuple)
    before, action, after, primitive = transition
    assert isinstance(before, SpatialCanvasState)
    assert isinstance(after, SpatialCanvasState)
    assert isinstance(primitive, MotorPrimitiveLatent)

    def infer_state() -> object:
        driver._reset_agent_belief(before)
        driver._update_agent_belief(action, after, primitive)  # type: ignore[arg-type]
        driver.belief = driver.agent.belief
        return _jsonable_dataclass(driver.agent.last_vfe)

    inference_results, inference_phase = _repeat_phase(
        settings.inference_repeats,
        infer_state,
        device=settings.device,
    )

    planning_profiles: list[dict[str, object]] = []
    policy_traces: list[dict[str, object]] = []

    def plan() -> dict[str, object]:
        driver._background_plan(after, None, sim)
        if driver._pending_error is not None:
            raise RuntimeError(driver._pending_error)
        profile = dict(driver.last_planning_profile)
        planning_profiles.append(profile)
        policy_traces.append(
            {
                "planning_profile": profile,
                "ranked": [
                    {
                        "policy": asdict(policy),
                        "efe": asdict(efe),
                        "posterior": float(posterior),
                    }
                    for policy, efe, posterior in driver.last_ranked
                ],
            }
        )
        return profile

    _, planning_phase = _repeat_phase(
        settings.planning_repeats,
        plan,
        device=settings.device,
    )

    def train() -> float:
        loss = driver.agent.train_dynamics(gradient_steps=settings.gradient_steps)
        if loss is None:
            raise RuntimeError("training replay did not reach the declared batch size")
        return float(loss)

    training_losses, training_phase = _repeat_phase(
        settings.training_repeats,
        train,
        device=settings.device,
    )

    legacy_config = replace(
        config,
        composition_enabled=True,
        composition_gap_precision=1.0,
        learned_proposal_enabled=False,
    )
    if painting_policy_profile_id(legacy_config) != LEGACY_MATERIAL_HIERARCHY_DIAGNOSTIC_ID:
        raise RuntimeError("legacy hierarchy positive control was not explicitly labelled")
    legacy_agent = SpatialActiveInferencePainter(
        legacy_config,
        seed=settings.seed + 997,
        device=settings.device,
    )
    if legacy_agent.composition is None:
        raise RuntimeError("legacy hierarchy positive control is unavailable")
    hierarchy_fields = torch.tensor(
        after.material,
        dtype=torch.float32,
        device=legacy_agent.device,
    ).unsqueeze(0)

    def hierarchy_positive_control() -> float:
        assert legacy_agent.composition is not None
        return float(legacy_agent.composition.compression_gap(hierarchy_fields).item())

    hierarchy_gaps, hierarchy_phase = _repeat_phase(
        settings.inference_repeats,
        hierarchy_positive_control,
        device=settings.device,
    )

    checkpoint_path = run_dir / "profile-checkpoint.pt"
    driver.checkpoint_path = checkpoint_path
    driver.trained_transitions = int(replay_sizes[-1])

    def save_checkpoint() -> int:
        driver._save_checkpoint_if_due(force=True)
        if driver.checkpoint_status != "saved":
            raise RuntimeError(driver.checkpoint_last_error or driver.checkpoint_status)
        return checkpoint_path.stat().st_size

    checkpoint_sizes, checkpoint_phase = _repeat_phase(
        settings.serialization_repeats,
        save_checkpoint,
        device=settings.device,
    )

    def serialize_diagnostics() -> int:
        return len(
            json.dumps(
                driver.diagnostics(),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        )

    diagnostics_sizes, diagnostics_serialization_phase = _repeat_phase(
        settings.serialization_repeats,
        serialize_diagnostics,
        device=settings.device,
    )

    render_box: dict[str, np.ndarray] = {}

    def render_material_canvas() -> tuple[int, int]:
        tone = sim.canvas.observed_tone()
        gray = np.clip((1.0 - tone) * 255.0, 0.0, 255.0).astype(np.uint8)
        render_box["gray"] = gray
        return gray.shape

    _, rendering_phase = _repeat_phase(
        settings.rendering_repeats,
        render_material_canvas,
        device=settings.device,
    )

    def encode_png() -> int:
        output = io.BytesIO()
        Image.fromarray(render_box["gray"], mode="L").save(output, format="PNG")
        return len(output.getvalue())

    png_sizes, png_serialization_phase = _repeat_phase(
        settings.serialization_repeats,
        encode_png,
        device=settings.device,
    )
    Image.fromarray(render_box["gray"], mode="L").save(
        run_dir / "profile-canvas.png",
        format="PNG",
    )

    driver_phase_keys = (
        "beliefUpdateSeconds",
        "policySampleSeconds",
        "baseEFESeconds",
        "motorForecastSeconds",
        "motorEFERescoreSeconds",
        "posteriorSeconds",
        "compositionDiagnosticSeconds",
        "selectedForecastSeconds",
        "publishSeconds",
        "totalSeconds",
    )
    driver_phase_summary: dict[str, object] = {}
    for key in driver_phase_keys:
        values = [float(profile.get(key, 0.0)) for profile in planning_profiles]
        driver_phase_summary[key] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
        }

    report: dict[str, object] = {
        "schema": AI113_PROFILE_SCHEMA,
        "status": "completed",
        "claim_boundary": (
            "uncalibrated native/oracle computational profile; no painting-quality, "
            "sensor-equivalence, or hardware-calibration claim"
        ),
        "painting_policy_profile": profile_id,
        "observation_access_mode": ORACLE_OBSERVATION_ACCESS_MODE,
        "settings": asdict(settings),
        "environment": environment,
        "phases": {
            "observed_training_data_collection": collection_phase,
            "state_inference": inference_phase,
            "planning_total": planning_phase,
            "planning_driver_decomposition": driver_phase_summary,
            "gradient_training": training_phase,
            "formal_hierarchy_evaluation": {
                "status": "disabled_by_m1_formal_policy_baseline",
                "policy_influence": False,
                "bookkeeping_timer": driver_phase_summary[
                    "compositionDiagnosticSeconds"
                ],
            },
            "legacy_hierarchy_positive_control": {
                "status": "noncanonical_diagnostic_only",
                "painting_policy_profile": LEGACY_MATERIAL_HIERARCHY_DIAGNOSTIC_ID,
                "policy_influence_in_measured_formal_run": False,
                "timing": hierarchy_phase,
                "compression_gap_values": hierarchy_gaps,
            },
            "checkpoint_serialization": checkpoint_phase,
            "diagnostics_json_serialization": diagnostics_serialization_phase,
            "material_canvas_rendering": rendering_phase,
            "png_serialization": png_serialization_phase,
        },
        "measurements": {
            "training_losses": training_losses,
            "checkpoint_sizes_bytes": checkpoint_sizes,
            "diagnostics_json_sizes_bytes": diagnostics_sizes,
            "png_sizes_bytes": png_sizes,
            "replay_size": len(driver.agent.replay),
            "replay_evidence_counts": dict(driver.replay_evidence_counts),
            "state_inference_vfe": inference_results,
            "planning_profiles": planning_profiles,
        },
        "parallelism_taxonomy": {
            "parallel_policy_rollouts": (
                "same agent and frozen belief; scheduling-only batching of independent "
                "motor counterfactuals; representative profile uses one worker"
            ),
            "parallel_data_generation": (
                "independent simulation workers with isolated process and belief state; "
                "AI-108 evidence, not remeasured here"
            ),
            "batched_training": (
                "conventional vectorized gradient learning over observed replay; no "
                "painting-policy selection"
            ),
            "independent_research_replicas": (
                "separate fixed-seed experimental units; never worker threads or "
                "particles within one posterior"
            ),
        },
        "batchable_without_equation_change": [
            "base-EFE evaluation over a fixed sampled candidate set",
            "independent motor forecast particles and motor realizations",
            "local transition-likelihood minibatches grouped by patch bucket",
            "held-out likelihood evaluation over fixed trajectory splits",
        ],
        "not_batchable_as_one_shared_state": [
            "independent research replicas",
            "parallel collection workers with separate online beliefs and replay",
            "causally ordered camera updates and passage-boundary posterior updates",
        ],
        "limitations": [
            "CPU utilization is process CPU time divided by wall time; it is not a system-wide utilization sample.",
            "GPU utilization percentage is unavailable from Torch; CUDA variants report peak allocation and paired wall/CPU time rather than inventing utilization.",
            "The registered-camera likelihood and MuJoCo camera renderer are outside this native/oracle M1 timing profile.",
            "The transition model begins from the declared seed and is trained only for the profiling steps; timings are computational evidence, not predictive-quality evidence.",
        ],
    }
    _write_json(run_dir / "policy-efe-trace.json", policy_traces)
    _write_json(run_dir / "state-vfe-trace.json", inference_results)
    _write_json(run_dir / "environment.json", environment)
    _write_json(run_dir / "version-manifest.json", environment)
    (run_dir / "failure-log.jsonl").write_text("", encoding="utf-8")

    ended_utc = datetime.now(timezone.utc)
    total_wall_seconds = time.perf_counter() - total_wall_started
    total_cpu_seconds = time.process_time() - total_cpu_started
    report["run_dir"] = str(run_dir)
    report["total_wall_seconds"] = total_wall_seconds
    report["total_process_cpu_seconds"] = total_cpu_seconds
    profile_path = run_dir / "profile.json"
    _write_json(profile_path, report)
    manifest = {
        "schema": "experiment-manifest-v1",
        "manifest_revision": 1,
        "run_id": run_dir.name,
        "run_kind": "performance_profile",
        "status": "completed",
        "evidence_level": "provisional",
        "start_utc": started_utc.isoformat(),
        "end_utc": ended_utc.isoformat(),
        "painting_policy_profile": profile_id,
        "versions": "version-manifest.json",
        "observation_access": {
            "mode": ORACLE_OBSERVATION_ACCESS_MODE,
            "claim": "explicit oracle diagnostic; not sensor-equivalent cognition",
        },
        "randomness": {"seed": settings.seed},
        "learning": {
            "online_updates_profiled": True,
            "replay_sources": dict(driver.replay_evidence_counts),
            "inheritance": "fresh individual; no input checkpoint",
        },
        "timing": {
            "total_wall_seconds": total_wall_seconds,
            "total_process_cpu_seconds": total_cpu_seconds,
            "phase_profile": "profile.json",
        },
        "artifacts": {
            "resolved_config": {
                "path": "resolved-config.json",
                "sha256": _sha256(resolved_config_path),
                "status": "present",
            },
            "profile": {
                "path": "profile.json",
                "sha256": _sha256(profile_path),
                "status": "present",
            },
            "version_manifest": {
                "path": "version-manifest.json",
                "sha256": _sha256(run_dir / "version-manifest.json"),
                "status": "present",
            },
            "vfe_trace": {"path": "state-vfe-trace.json", "status": "present"},
            "efe_trace": {"path": "policy-efe-trace.json", "status": "present"},
            "checkpoint": {
                "path": "profile-checkpoint.pt",
                "sha256": _sha256(checkpoint_path),
                "status": "present",
            },
            "canvas": {"path": "profile-canvas.png", "status": "present"},
            "failure_log": {"path": "failure-log.jsonl", "status": "present"},
            "registered_camera_observations": {
                "status": "not_applicable",
                "reason": "native/oracle computational profile",
            },
            "telemetry": {
                "status": "not_applicable",
                "reason": "phase microbenchmark, not an executed painting episode",
            },
        },
        "active_inference": {
            "profiled": [
                "state inference",
                "base expected free energy",
                "motor expected-free-energy augmentation",
                "sampled-candidate policy posterior",
            ],
            "formal_hierarchy": "disabled",
        },
        "conventional_support": [
            "native motor counterfactual integration",
            "gradient optimization",
            "checkpoint and JSON serialization",
            "material canvas rendering and PNG encoding",
        ],
        "approximations": [
            "oracle diagnostic observation access",
            "native abstract plant",
            "process CPU utilization proxy rather than system sampler",
        ],
    }
    manifest_path = run_dir / "experiment-manifest.json"
    _write_json(manifest_path, manifest)
    report["manifest_path"] = str(manifest_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile M1 inference, EFE, motor forecast, learning, serialization, and rendering phases."
    )
    parser.add_argument("--output-dir", default="runs/ai113-runtime-profiles")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--planning-repeats", type=int, default=3)
    parser.add_argument("--inference-repeats", type=int, default=5)
    parser.add_argument("--training-repeats", type=int, default=3)
    parser.add_argument("--gradient-steps", type=int, default=2)
    parser.add_argument("--serialization-repeats", type=int, default=3)
    parser.add_argument("--rendering-repeats", type=int, default=5)
    parser.add_argument("--training-batch-size", type=int, default=32)
    parser.add_argument("--motor-forecast-workers", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = RuntimeProfileSettings(
        seed=args.seed,
        planning_repeats=args.planning_repeats,
        inference_repeats=args.inference_repeats,
        training_repeats=args.training_repeats,
        gradient_steps=args.gradient_steps,
        serialization_repeats=args.serialization_repeats,
        rendering_repeats=args.rendering_repeats,
        training_batch_size=args.training_batch_size,
        motor_forecast_workers=args.motor_forecast_workers,
        device=args.device,
        quick=args.quick,
    )
    if min(
        settings.planning_repeats,
        settings.inference_repeats,
        settings.training_repeats,
        settings.gradient_steps,
        settings.serialization_repeats,
        settings.rendering_repeats,
        settings.training_batch_size,
        settings.motor_forecast_workers,
    ) <= 0:
        raise SystemExit("all profile counts must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    label = "quick" if settings.quick else "representative"
    run_dir = Path(args.output_dir) / (
        f"ai113-{label}-{settings.device}-workers-{settings.motor_forecast_workers}-"
        f"{stamp}-seed-{settings.seed}"
    )
    report = run_profile(settings, run_dir)
    output = (
        {
            "run_dir": report["run_dir"],
            "manifest_path": report["manifest_path"],
            "total_wall_seconds": report["total_wall_seconds"],
        }
        if args.quiet
        else report
    )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
