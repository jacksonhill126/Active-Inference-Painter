"""Trajectory-first corpus records for parallel sensor-simulation learning.

The split unit is an entire painting trajectory. Full spatial posteriors are
stored before local-patch extraction so neighboring cells and consecutive
marks from one physical curve can never cross train/validation/test boundaries.
The records contain agent beliefs and action conditions, not privileged live
process material state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Iterable, Mapping, Sequence

import numpy as np

from .action_encoding import motor_kind
from .brush_loading import BrushLoadBelief
from .config import PainterConfig
from .env import StrokeAction
from .local_spatial import pixel_logvar_from_state, pixel_material_from_state
from .policies import MotorPrimitiveLatent
from .spatial_state import SpatialCanvasState, rasterize_stroke_action


TRAJECTORY_CORPUS_SCHEMA = "trajectory-posterior-corpus-v2"
LEGACY_TRAJECTORY_CORPUS_SCHEMAS = frozenset({"trajectory-posterior-corpus-v1"})
SPLIT_MANIFEST_SCHEMA = "trajectory-split-manifest-v1"
CONDITION_AUDIT_SCHEMA = "trajectory-condition-audit-v1"
SPLIT_STRATEGY = "deterministic-greedy-multilabel-transition-balance-v1"
TERMINATION_POLICY_STOP = "policy_selected_stop"
TERMINATION_FIXED_HORIZON = "fixed_horizon_truncation"
SPLIT_NAMES = ("train", "validation", "test")

# Declared normalized-action/posterior thresholds for corpus evidence. These
# labels are evaluation/stratification bins, never policy preferences, rewards,
# likelihood terms, or privileged process labels.
CONDITION_THRESHOLDS: dict[str, float] = {
    "core_footprint": 0.10,
    "overlap_coverage": 0.10,
    "wetness": 1.0e-4,
    "narrow_width_max": 0.07,
    "broad_width_min": 0.16,
    "short_length_max": 0.30,
    "long_length_min": 0.48,
    "strong_curvature_min": 0.12,
    "straight_curvature_max": 1.0e-4,
    "axis_direction_ratio": 2.0,
    "center_reach_max": 0.20,
    "outer_reach_min": 0.40,
}

REQUIRED_OVERALL_CONDITIONS: tuple[str, ...] = (
    "tone=white",
    "tone=black",
    "surface=dry_blank",
    "surface=dry_existing",
    "overlap=fresh",
    "overlap=overlap",
    "edge=interior",
    "edge=edge",
    "width=narrow",
    "width=broad",
    "length=short",
    "length=long",
    "curvature=negative_gentle",
    "curvature=negative_strong",
    "curvature=straight",
    "curvature=positive_gentle",
    "curvature=positive_strong",
    "direction=vertical",
    "reach=center",
    "reach=outer",
    "motor=cartesian_ik",
    "motor=upper_arm_fixed_roll_positive",
    "motor=upper_arm_fixed_roll_negative",
    "motor=upper_arm_roll_positive",
    "motor=upper_arm_roll_negative",
)


@dataclass(frozen=True, slots=True)
class RecordedTransition:
    state_coarse_material: np.ndarray
    state_coarse_logvar: np.ndarray
    state_material: np.ndarray
    state_logvar: np.ndarray
    action: np.ndarray
    motor_kind: str
    brush_condition: np.ndarray
    brush_inference_model_id: str
    brush_calibration_status: str
    next_material: np.ndarray
    next_logvar: np.ndarray
    next_coarse_material: np.ndarray
    next_coarse_logvar: np.ndarray
    state_revision: int
    next_revision: int
    state_model_id: str
    next_model_id: str


@dataclass(frozen=True, slots=True)
class TrajectoryShard:
    path: Path
    metadata: dict[str, object]
    state_coarse_material: np.ndarray
    state_coarse_logvar: np.ndarray
    state_material: np.ndarray
    state_logvar: np.ndarray
    action: np.ndarray
    motor_kind: np.ndarray
    brush_condition: np.ndarray
    brush_inference_model_id: np.ndarray
    brush_calibration_status: np.ndarray
    next_material: np.ndarray
    next_logvar: np.ndarray
    next_coarse_material: np.ndarray
    next_coarse_logvar: np.ndarray
    state_revision: np.ndarray
    next_revision: np.ndarray
    state_model_id: np.ndarray
    next_model_id: np.ndarray
    final_material: np.ndarray
    final_logvar: np.ndarray
    final_coarse_material: np.ndarray
    final_coarse_logvar: np.ndarray

    @property
    def trajectory_id(self) -> str:
        return str(self.metadata["trajectory_id"])

    @property
    def transition_count(self) -> int:
        return int(self.action.shape[0])


class TrajectoryRecorder:
    """Thread-safe accumulator attached to one isolated simulation worker."""

    def __init__(
        self,
        output_dir: Path | str,
        config: PainterConfig,
        *,
        worker_id: int,
        seed: int,
        provenance: Mapping[str, object],
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config
        self.worker_id = int(worker_id)
        self.seed = int(seed)
        self.provenance = dict(provenance)
        self._lock = threading.Lock()
        self._transitions: list[RecordedTransition] = []
        self._trajectory_index = 0
        self.completed_paths: list[Path] = []

    @property
    def pending_transition_count(self) -> int:
        with self._lock:
            return len(self._transitions)

    def record_transition(
        self,
        state: np.ndarray | SpatialCanvasState,
        action: StrokeAction,
        primitive: MotorPrimitiveLatent | None,
        next_state: np.ndarray | SpatialCanvasState,
        brush_belief: BrushLoadBelief | None = None,
    ) -> None:
        if not isinstance(state, SpatialCanvasState) or not isinstance(
            next_state, SpatialCanvasState
        ):
            raise TypeError(
                "trajectory-posterior-corpus-v2 records spatial posteriors only"
            )
        record = RecordedTransition(
            state_coarse_material=state.material.astype(np.float32, copy=True),
            state_coarse_logvar=state.logvar.astype(np.float32, copy=True),
            state_material=pixel_material_from_state(state).astype(
                np.float32, copy=True
            ),
            state_logvar=pixel_logvar_from_state(state, self.config).astype(
                np.float32, copy=True
            ),
            action=action.vector().astype(np.float32, copy=True),
            motor_kind=motor_kind(primitive),
            brush_condition=brush_condition_vector(brush_belief),
            brush_inference_model_id=(
                "unavailable:legacy-or-non-sensor-transition"
                if brush_belief is None
                else str(brush_belief.inference_model_id)
            ),
            brush_calibration_status=(
                "unavailable"
                if brush_belief is None
                else str(brush_belief.calibration_status)
            ),
            next_material=pixel_material_from_state(next_state).astype(
                np.float32, copy=True
            ),
            next_logvar=pixel_logvar_from_state(next_state, self.config).astype(
                np.float32, copy=True
            ),
            next_coarse_material=next_state.material.astype(np.float32, copy=True),
            next_coarse_logvar=next_state.logvar.astype(np.float32, copy=True),
            state_revision=int(state.posterior_revision),
            next_revision=int(next_state.posterior_revision),
            state_model_id=str(state.inference_model_id),
            next_model_id=str(next_state.inference_model_id),
        )
        if record.state_material.shape != record.next_material.shape:
            raise ValueError("transition posterior shapes differ")
        with self._lock:
            self._transitions.append(record)

    def complete(
        self,
        final_belief: SpatialCanvasState,
        *,
        termination: str,
        painting_index: int | None = None,
    ) -> Path:
        if termination not in {
            TERMINATION_POLICY_STOP,
            TERMINATION_FIXED_HORIZON,
        }:
            raise ValueError(f"unsupported trajectory termination: {termination!r}")
        with self._lock:
            records = self._transitions
            self._transitions = []
            trajectory_index = self._trajectory_index
            self._trajectory_index += 1

        trajectory_id = (
            f"worker-{self.worker_id:03d}-seed-{self.seed}-"
            f"trajectory-{trajectory_index:06d}"
        )
        metadata: dict[str, object] = {
            "schema": TRAJECTORY_CORPUS_SCHEMA,
            "trajectory_id": trajectory_id,
            "worker_id": self.worker_id,
            "worker_seed": self.seed,
            "trajectory_index": trajectory_index,
            "painting_index": painting_index,
            "termination": termination,
            "transition_count": len(records),
            "split_unit": "entire_trajectory_before_patch_extraction",
            "process_truth_used_as_training_input": False,
            "config": asdict(self.config),
            "provenance": self.provenance,
        }
        path = self.output_dir / f"{trajectory_id}.npz"
        _write_shard(path, metadata, records, final_belief, self.config)
        self.completed_paths.append(path)
        return path


def _write_shard(
    path: Path,
    metadata: Mapping[str, object],
    records: Sequence[RecordedTransition],
    final_belief: SpatialCanvasState,
    config: PainterConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    final_material = pixel_material_from_state(final_belief).astype(
        np.float32, copy=True
    )
    final_logvar = pixel_logvar_from_state(final_belief, config).astype(
        np.float32, copy=True
    )
    field_shape = final_material.shape

    def stack(name: str) -> np.ndarray:
        if not records:
            return np.empty((0, *field_shape), dtype=np.float32)
        arrays = [getattr(record, name) for record in records]
        if any(array.shape != field_shape for array in arrays):
            raise ValueError(f"{name} fields do not share one trajectory shape")
        return np.stack(arrays).astype(np.float32, copy=False)

    payload = {
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            dtype=np.str_,
        ),
        "state_coarse_material": (
            np.stack([record.state_coarse_material for record in records]).astype(np.float32)
            if records
            else np.empty((0, *final_belief.material.shape), dtype=np.float32)
        ),
        "state_coarse_logvar": (
            np.stack([record.state_coarse_logvar for record in records]).astype(np.float32)
            if records
            else np.empty((0, *final_belief.logvar.shape), dtype=np.float32)
        ),
        "state_material": stack("state_material"),
        "state_logvar": stack("state_logvar"),
        "action": (
            np.stack([record.action for record in records]).astype(np.float32)
            if records
            else np.empty((0, 8), dtype=np.float32)
        ),
        "motor_kind": np.asarray(
            [record.motor_kind for record in records], dtype=np.str_
        ),
        "brush_condition": (
            np.stack([record.brush_condition for record in records]).astype(np.float32)
            if records
            else np.empty((0, 5), dtype=np.float32)
        ),
        "brush_inference_model_id": np.asarray(
            [record.brush_inference_model_id for record in records], dtype=np.str_
        ),
        "brush_calibration_status": np.asarray(
            [record.brush_calibration_status for record in records], dtype=np.str_
        ),
        "next_material": stack("next_material"),
        "next_logvar": stack("next_logvar"),
        "next_coarse_material": (
            np.stack([record.next_coarse_material for record in records]).astype(np.float32)
            if records
            else np.empty((0, *final_belief.material.shape), dtype=np.float32)
        ),
        "next_coarse_logvar": (
            np.stack([record.next_coarse_logvar for record in records]).astype(np.float32)
            if records
            else np.empty((0, *final_belief.logvar.shape), dtype=np.float32)
        ),
        "state_revision": np.asarray(
            [record.state_revision for record in records], dtype=np.int64
        ),
        "next_revision": np.asarray(
            [record.next_revision for record in records], dtype=np.int64
        ),
        "state_model_id": np.asarray(
            [record.state_model_id for record in records], dtype=np.str_
        ),
        "next_model_id": np.asarray(
            [record.next_model_id for record in records], dtype=np.str_
        ),
        "final_material": final_material,
        "final_logvar": final_logvar,
        "final_coarse_material": final_belief.material.astype(np.float32, copy=True),
        "final_coarse_logvar": final_belief.logvar.astype(np.float32, copy=True),
    }
    temp_path = path.with_name(f"{path.stem}.tmp.npz")
    np.savez_compressed(temp_path, **payload)
    temp_path.replace(path)


def load_trajectory_shard(path: Path | str) -> TrajectoryShard:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        arrays = {name: payload[name].copy() for name in payload.files if name != "metadata_json"}
    schema = str(metadata.get("schema", ""))
    if schema not in {TRAJECTORY_CORPUS_SCHEMA, *LEGACY_TRAJECTORY_CORPUS_SCHEMAS}:
        raise ValueError(f"unsupported trajectory corpus schema in {path}")
    required = {
        "state_coarse_material",
        "state_coarse_logvar",
        "state_material",
        "state_logvar",
        "action",
        "motor_kind",
        "next_material",
        "next_logvar",
        "next_coarse_material",
        "next_coarse_logvar",
        "state_revision",
        "next_revision",
        "state_model_id",
        "next_model_id",
        "final_material",
        "final_logvar",
        "final_coarse_material",
        "final_coarse_logvar",
    }
    optional_brush = {
        "brush_condition",
        "brush_inference_model_id",
        "brush_calibration_status",
    }
    unexpected = set(arrays).difference(required | optional_brush)
    if unexpected:
        raise ValueError(
            f"trajectory shard {path} contains unsupported arrays {sorted(unexpected)}"
        )
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"trajectory shard {path} is missing {sorted(missing)}")
    count = int(arrays["action"].shape[0])
    for name in required.difference(
        {
            "final_material",
            "final_logvar",
            "final_coarse_material",
            "final_coarse_logvar",
        }
    ):
        if int(arrays[name].shape[0]) != count:
            raise ValueError(f"trajectory shard {path} has inconsistent {name}")
    if int(metadata.get("transition_count", -1)) != count:
        raise ValueError(f"trajectory shard {path} transition count disagrees with metadata")
    missing_brush = optional_brush.difference(arrays)
    if schema == TRAJECTORY_CORPUS_SCHEMA and missing_brush:
        raise ValueError(
            f"trajectory shard {path} is missing v2 brush fields {sorted(missing_brush)}"
        )
    # v1 shards predate compact brush-posterior capture.  A fifth validity
    # field keeps that absence explicit instead of inventing a zero-load
    # observation.  The cVAE may learn from the other conditions while the
    # invalid brush fields remain masked by this flag.
    arrays.setdefault(
        "brush_condition",
        np.zeros((count, 5), dtype=np.float32),
    )
    arrays.setdefault(
        "brush_inference_model_id",
        np.full(
            count,
            "unavailable:trajectory-posterior-corpus-v1",
            dtype=np.str_,
        ),
    )
    arrays.setdefault(
        "brush_calibration_status",
        np.full(count, "unavailable", dtype=np.str_),
    )
    if arrays["brush_condition"].shape != (count, 5):
        raise ValueError(f"trajectory shard {path} has invalid brush_condition")
    for name in ("brush_inference_model_id", "brush_calibration_status"):
        if arrays[name].shape != (count,):
            raise ValueError(f"trajectory shard {path} has invalid {name}")
    return TrajectoryShard(path=path, metadata=metadata, **arrays)


def brush_condition_vector(belief: BrushLoadBelief | None) -> np.ndarray:
    """Compact sensor-compatible q(brush) condition plus an availability bit.

    The four moments are inferred belief state, not exact held paint or bristle
    microstructure.  Standard deviations are used rather than variances so all
    four physical fields remain on roughly comparable normalized scales.
    """

    if belief is None:
        return np.zeros(5, dtype=np.float32)
    return np.asarray(
        [
            belief.load_mean,
            np.sqrt(belief.load_variance),
            belief.black_fraction_mean,
            np.sqrt(belief.black_fraction_variance),
            1.0,
        ],
        dtype=np.float32,
    )


def discover_trajectory_shards(root: Path | str) -> list[Path]:
    return sorted(
        path
        for path in Path(root).glob("worker-*-trajectory-*.npz")
        if not path.name.endswith(".tmp.npz")
    )


def _stroke_action(vector: np.ndarray) -> StrokeAction:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.shape != (8,):
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


def transition_condition_labels(
    shard: TrajectoryShard,
    transition_index: int,
) -> dict[str, str]:
    """Evaluation-only labels derived from stored posterior/action evidence.

    Material labels use the *pre-action camera-derived posterior* in the shard,
    never exact process canvas state. Wetness currently remains largely driven
    by the transition prior because the camera likelihood does not directly
    observe bulk wetness; the audit reports coverage of that belief condition,
    not physically calibrated wet-paint ground truth.
    """

    index = int(transition_index)
    if index < 0 or index >= shard.transition_count:
        raise IndexError("transition index is outside the trajectory")
    action = _stroke_action(shard.action[index])
    state = np.asarray(shard.state_material[index], dtype=np.float32)
    grid_size = int(state.shape[-1])
    raster = rasterize_stroke_action(action, grid_size)
    core = raster[0] >= CONDITION_THRESHOLDS["core_footprint"]
    if not bool(np.any(core)):
        row = min(grid_size - 1, max(0, int(action.y0 * grid_size)))
        col = min(grid_size - 1, max(0, int(action.x0 * grid_size)))
        core = np.zeros((grid_size, grid_size), dtype=bool)
        core[row, col] = True

    if state.shape[0] > 5:
        local_coverage = float(np.mean(state[5][core]))
    else:
        local_coverage = float(np.mean(state[0][core] > 1.0e-4))
    local_wetness = float(np.mean(state[1][core])) if state.shape[0] > 1 else 0.0
    overlap = local_coverage >= CONDITION_THRESHOLDS["overlap_coverage"]
    wet = local_wetness >= CONDITION_THRESHOLDS["wetness"]
    if overlap and wet:
        surface = "wet_over_wet"
    elif overlap:
        surface = "dry_existing"
    elif wet:
        surface = "wet_residual_uncovered"
    else:
        surface = "dry_blank"

    tone = "white" if action.tone < 0.25 else "black" if action.tone > 0.75 else "mixed"
    if action.width <= CONDITION_THRESHOLDS["narrow_width_max"]:
        width = "narrow"
    elif action.width >= CONDITION_THRESHOLDS["broad_width_min"]:
        width = "broad"
    else:
        width = "medium"

    dx = float(action.x1 - action.x0)
    dy = float(action.y1 - action.y0)
    length_value = float(np.hypot(dx, dy))
    if length_value <= CONDITION_THRESHOLDS["short_length_max"]:
        length = "short"
    elif length_value >= CONDITION_THRESHOLDS["long_length_min"]:
        length = "long"
    else:
        length = "medium"

    curvature_value = float(action.curvature)
    curvature_abs = abs(curvature_value)
    if curvature_abs <= CONDITION_THRESHOLDS["straight_curvature_max"]:
        curvature = "straight"
    else:
        strength = (
            "strong"
            if curvature_abs >= CONDITION_THRESHOLDS["strong_curvature_min"]
            else "gentle"
        )
        curvature = f"{'positive' if curvature_value > 0.0 else 'negative'}_{strength}"

    ratio = CONDITION_THRESHOLDS["axis_direction_ratio"]
    if abs(dy) >= ratio * max(abs(dx), 1.0e-8):
        direction = "vertical"
    elif abs(dx) >= ratio * max(abs(dy), 1.0e-8):
        direction = "horizontal"
    else:
        direction = "diagonal"

    center_x = 0.5 * float(action.x0 + action.x1)
    center_y = 0.5 * float(action.y0 + action.y1)
    col_band = "left" if center_x < 1.0 / 3.0 else "right" if center_x >= 2.0 / 3.0 else "center"
    row_band = "top" if center_y < 1.0 / 3.0 else "bottom" if center_y >= 2.0 / 3.0 else "middle"
    radius = float(np.hypot(center_x - 0.5, center_y - 0.5))
    if radius <= CONDITION_THRESHOLDS["center_reach_max"]:
        reach = "center"
    elif radius >= CONDITION_THRESHOLDS["outer_reach_min"]:
        reach = "outer"
    else:
        reach = "mid"

    edge = bool(
        np.any(core[0])
        or np.any(core[-1])
        or np.any(core[:, 0])
        or np.any(core[:, -1])
    )
    amount = "low" if action.amount < 1.0 / 3.0 else "high" if action.amount >= 2.0 / 3.0 else "medium"
    phase_fraction = (index + 0.5) / max(1, shard.transition_count)
    phase = "early" if phase_fraction < 1.0 / 3.0 else "late" if phase_fraction >= 2.0 / 3.0 else "middle"
    brush_available = bool(float(shard.brush_condition[index, 4]) >= 0.5)

    return {
        "tone": tone,
        "surface": surface,
        "overlap": "overlap" if overlap else "fresh",
        "edge": "edge" if edge else "interior",
        "width": width,
        "length": length,
        "amount": amount,
        "curvature": curvature,
        "direction": direction,
        "region": f"{row_band}_{col_band}",
        "reach": reach,
        "motor": str(shard.motor_kind[index]),
        "brush_context": "available" if brush_available else "unavailable",
        "phase": phase,
        "termination": str(shard.metadata.get("termination", "unknown")),
        "corpus_schema": str(shard.metadata.get("schema", "unknown")),
    }


def trajectory_condition_counts(shard: TrajectoryShard) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for index in range(shard.transition_count):
        for family, label in transition_condition_labels(shard, index).items():
            family_counts = counts.setdefault(family, {})
            family_counts[label] = family_counts.get(label, 0) + 1
    return counts


def _flat_condition_counts(shard: TrajectoryShard) -> dict[str, int]:
    return {
        f"{family}={label}": count
        for family, labels in trajectory_condition_counts(shard).items()
        for label, count in labels.items()
    }


def _ratio_counts(total: int, ratio: np.ndarray) -> np.ndarray:
    raw = ratio * int(total)
    counts = np.floor(raw).astype(int)
    remainder = int(total) - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    for index in order[:remainder]:
        counts[int(index)] += 1
    return counts


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def corpus_provenance_summary(
    splits: Mapping[str, Sequence[Path | str]],
) -> dict[str, object]:
    """Summarize reproducibility and evidence boundaries across all shards."""

    shards = [
        load_trajectory_shard(path)
        for split_name in SPLIT_NAMES
        for path in splits.get(split_name, ())
    ]

    def metadata_values(key: str) -> list[str]:
        return sorted({str(shard.metadata.get(key, "unavailable")) for shard in shards})

    def provenance_values(key: str) -> list[str]:
        return sorted(
            {
                str(dict(shard.metadata.get("provenance", {})).get(key, "unavailable"))
                for shard in shards
            }
        )

    configs = [dict(shard.metadata.get("config", {})) for shard in shards]
    config_fingerprints = sorted({_metadata_fingerprint(config) for config in configs})
    canvas_sizes = sorted(
        {int(config["canvas_size"]) for config in configs if "canvas_size" in config}
    )
    spatial_grid_sizes = sorted(
        {
            int(config["spatial_grid_size"])
            for config in configs
            if "spatial_grid_size" in config
        }
    )
    worker_seeds = sorted(
        {int(shard.metadata["worker_seed"]) for shard in shards if "worker_seed" in shard.metadata}
    )
    transition_count = sum(shard.transition_count for shard in shards)
    brush_context_count = sum(
        int(np.count_nonzero(shard.brush_condition[:, 4] >= 0.5)) for shard in shards
    )
    process_canvas_sizes = sorted(
        {
            int(dict(shard.metadata.get("provenance", {}))["process_canvas_size"])
            for shard in shards
            if "process_canvas_size"
            in dict(shard.metadata.get("provenance", {}))
        }
    )
    process_canvas_provenance_complete = bool(shards) and all(
        "process_canvas_size" in dict(shard.metadata.get("provenance", {}))
        for shard in shards
    )
    return {
        "trajectory_count": len(shards),
        "transition_count": transition_count,
        "corpus_schemas": metadata_values("schema"),
        "inference_driver_canvas_sizes": canvas_sizes,
        # Kept for readers of the initial audit draft. This field is the
        # inference-driver config, not the process canvas resolution.
        "canvas_sizes": canvas_sizes,
        "process_canvas_sizes": process_canvas_sizes,
        "process_canvas_provenance_complete": process_canvas_provenance_complete,
        "spatial_grid_sizes": spatial_grid_sizes,
        "all_trajectories_use_live_scale_256_canvas": (
            process_canvas_provenance_complete and process_canvas_sizes == [256]
        ),
        "worker_seeds": worker_seeds,
        "config_fingerprints_sha256": config_fingerprints,
        "code_builds": provenance_values("code_build"),
        "plants": provenance_values("plant"),
        "observation_access_modes": provenance_values("observation_access_mode"),
        "camera_likelihoods": provenance_values("camera_likelihood"),
        "raw_camera_frames_persisted": provenance_values("raw_camera_frames_persisted"),
        "reduced_camera_acquisition_approximations": provenance_values(
            "reduced_camera_acquisition_approximation"
        ),
        "brush_context_transition_count": brush_context_count,
        "brush_context_fraction": (
            brush_context_count / transition_count if transition_count else None
        ),
        "termination_modes": sorted(
            {str(shard.metadata.get("termination", "unknown")) for shard in shards}
        ),
    }


def corpus_condition_audit(
    splits: Mapping[str, Sequence[Path | str]],
) -> dict[str, object]:
    """Measured transition and trajectory-presence counts for every split."""

    split_payload: dict[str, object] = {}
    all_ids: list[str] = []
    overall_counts: dict[str, int] = {}
    process_truth_flags: list[bool] = []
    for split_name in SPLIT_NAMES:
        shards = [load_trajectory_shard(path) for path in splits.get(split_name, ())]
        transition_counts: dict[str, int] = {}
        trajectory_presence: dict[str, int] = {}
        terminations: dict[str, int] = {}
        for shard in shards:
            all_ids.append(shard.trajectory_id)
            process_truth_flags.append(
                bool(shard.metadata.get("process_truth_used_as_training_input", True))
            )
            termination = str(shard.metadata.get("termination", "unknown"))
            terminations[termination] = terminations.get(termination, 0) + 1
            flat = _flat_condition_counts(shard)
            for label, count in flat.items():
                transition_counts[label] = transition_counts.get(label, 0) + int(count)
                overall_counts[label] = overall_counts.get(label, 0) + int(count)
                trajectory_presence[label] = trajectory_presence.get(label, 0) + 1
        split_payload[split_name] = {
            "trajectory_count": len(shards),
            "transition_count": sum(shard.transition_count for shard in shards),
            "terminations": dict(sorted(terminations.items())),
            "transition_counts": dict(sorted(transition_counts.items())),
            "trajectory_presence": dict(sorted(trajectory_presence.items())),
            "missing_required_conditions": sorted(
                label for label in REQUIRED_OVERALL_CONDITIONS if transition_counts.get(label, 0) == 0
            ),
        }
    duplicate_ids = sorted({item for item in all_ids if all_ids.count(item) > 1})
    return {
        "schema": CONDITION_AUDIT_SCHEMA,
        "label_source": (
            "stored pre-action camera-derived posterior plus selected normalized mark, "
            "conditional motor realization, compact brush belief availability, and trajectory metadata"
        ),
        "process_truth_used_for_labels": False,
        "wetness_caveat": (
            "wetness is a posterior/transition-prior condition and is not yet directly identified "
            "by the camera likelihood or physically calibrated"
        ),
        "structurally_unobserved_material_conditions": [
            (
                "camera evidence does not currently identify bulk wetness; "
                "wet_over_wet cannot be required as sensor-corpus ground truth"
            )
        ],
        "reach_caveat": (
            "reach is normalized canvas radial target position, not measured joint-space effort"
        ),
        "thresholds": dict(CONDITION_THRESHOLDS),
        "required_overall_conditions": list(REQUIRED_OVERALL_CONDITIONS),
        "missing_overall_conditions": sorted(
            label for label in REQUIRED_OVERALL_CONDITIONS if overall_counts.get(label, 0) == 0
        ),
        "overall_transition_counts": dict(sorted(overall_counts.items())),
        "splits": split_payload,
        "checks": {
            "trajectory_ids_unique_across_splits": not duplicate_ids,
            "duplicate_trajectory_ids": duplicate_ids,
            "all_shards_deny_process_truth_training_input": not any(process_truth_flags),
        },
    }


def trajectory_stratum(shard: TrajectoryShard) -> str:
    """Coarse multi-condition label used for best-effort group stratification."""

    actions = shard.action
    if actions.shape[0] == 0:
        tone = "none"
        curvature = "none"
        region = "none"
    else:
        tone_mean = float(actions[:, 6].mean())
        tone = "white" if tone_mean < 0.25 else "black" if tone_mean > 0.75 else "mixed"
        curve = actions[:, 7]
        has_negative = bool(np.any(curve < -1e-4))
        has_positive = bool(np.any(curve > 1e-4))
        has_straight = bool(np.any(np.abs(curve) <= 1e-4))
        curvature = f"n{int(has_negative)}s{int(has_straight)}p{int(has_positive)}"
        center = 0.5 * (actions[:, :2] + actions[:, 2:4])
        mean_center = center.mean(axis=0)
        region = f"x{int(mean_center[0] >= 0.5)}y{int(mean_center[1] >= 0.5)}"
    motors = "+".join(sorted(set(shard.motor_kind.astype(str).tolist()))) or "none"
    final = shard.final_material
    coverage = final[5] if final.shape[0] > 5 else (final[0] > 0.02)
    coverage_mean = float(np.mean(coverage))
    coverage_band = "low" if coverage_mean < 0.30 else "mid" if coverage_mean < 0.70 else "high"
    return f"tone={tone}|curve={curvature}|region={region}|motor={motors}|coverage={coverage_band}"


def split_trajectory_paths(
    paths: Iterable[Path | str],
    *,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[Path]]:
    """Assign whole trajectory shards, never patches, to deterministic splits."""

    ratio = np.asarray(ratios, dtype=np.float64)
    if ratio.shape != (3,) or np.any(ratio < 0.0) or not np.isfinite(ratio).all():
        raise ValueError("split ratios must be three finite non-negative values")
    if float(ratio.sum()) <= 0.0:
        raise ValueError("at least one split ratio must be positive")
    ratio /= ratio.sum()
    shards = [load_trajectory_shard(path) for path in paths]
    result = {name: [] for name in SPLIT_NAMES}
    if not shards:
        return result

    features = {shard.trajectory_id: _flat_condition_counts(shard) for shard in shards}
    total_features: dict[str, int] = {}
    presence: dict[str, int] = {}
    for flat in features.values():
        for label, count in flat.items():
            total_features[label] = total_features.get(label, 0) + int(count)
            presence[label] = presence.get(label, 0) + 1

    # Rarest condition-bearing trajectories are placed first. The hash is a
    # deterministic tie-breaker, so a manifest can be exactly reproduced.
    ordered = sorted(
        shards,
        key=lambda shard: (
            -sum(1.0 / presence[label] for label in features[shard.trajectory_id]),
            hashlib.sha256(f"{seed}:{shard.trajectory_id}".encode("utf-8")).digest(),
        ),
    )
    capacities = _ratio_counts(len(shards), ratio)
    assigned_features = {name: {} for name in SPLIT_NAMES}
    for shard in ordered:
        flat = features[shard.trajectory_id]
        candidates: list[tuple[float, int, str]] = []
        for split_index, split_name in enumerate(SPLIT_NAMES):
            capacity = int(capacities[split_index])
            if capacity <= 0 or len(result[split_name]) >= capacity:
                continue
            size_fill = (len(result[split_name]) + 1.0) / capacity
            feature_fill = []
            for label, count in flat.items():
                target = max(1.0e-9, float(total_features[label]) * float(ratio[split_index]))
                current = float(assigned_features[split_name].get(label, 0))
                feature_fill.append((current + int(count)) / target)
            score = 0.5 * size_fill + (
                float(np.mean(feature_fill)) if feature_fill else 0.0
            )
            candidates.append((score, split_index, split_name))
        if not candidates:
            raise RuntimeError("no split capacity remains for a trajectory")
        _, _, selected = min(candidates)
        result[selected].append(shard.path)
        for label, count in flat.items():
            assigned_features[selected][label] = (
                assigned_features[selected].get(label, 0) + int(count)
            )
    return {name: sorted(items) for name, items in result.items()}


def write_split_manifest(
    output_path: Path | str,
    splits: Mapping[str, Sequence[Path]],
    *,
    seed: int,
    ratios: tuple[float, float, float],
) -> Path:
    output_path = Path(output_path)
    all_paths = [Path(path) for paths in splits.values() for path in paths]
    resolved_paths = [path.resolve() for path in all_paths]
    common_root = (
        Path(os.path.commonpath([str(path.parent) for path in resolved_paths]))
        if resolved_paths
        else output_path.parent.resolve()
    )
    ids: dict[str, list[str]] = {}
    files: dict[str, list[str]] = {}
    strata: dict[str, dict[str, int]] = {}
    integrity: dict[str, list[dict[str, str]]] = {}
    corpus_schemas: set[str] = set()
    for split_name in SPLIT_NAMES:
        split_paths = [Path(path) for path in splits.get(split_name, ())]
        loaded = [load_trajectory_shard(path) for path in split_paths]
        corpus_schemas.update(str(shard.metadata["schema"]) for shard in loaded)
        ids[split_name] = [shard.trajectory_id for shard in loaded]
        files[split_name] = [
            str(path.resolve().relative_to(common_root)) for path in split_paths
        ]
        integrity[split_name] = [
            {
                "file": str(path.resolve().relative_to(common_root)),
                "sha256": _file_sha256(path),
            }
            for path in split_paths
        ]
        counts: dict[str, int] = {}
        for shard in loaded:
            label = trajectory_stratum(shard)
            counts[label] = counts.get(label, 0) + 1
        strata[split_name] = counts
    audit = corpus_condition_audit(splits)
    payload = {
        "schema": SPLIT_MANIFEST_SCHEMA,
        "corpus_schema": (
            next(iter(corpus_schemas))
            if len(corpus_schemas) == 1
            else "mixed" if corpus_schemas else TRAJECTORY_CORPUS_SCHEMA
        ),
        "corpus_schemas": sorted(corpus_schemas or {TRAJECTORY_CORPUS_SCHEMA}),
        "split_unit": "entire_trajectory_before_patch_extraction",
        "split_strategy": SPLIT_STRATEGY,
        "seed": int(seed),
        "ratios": dict(zip(SPLIT_NAMES, ratios, strict=True)),
        "root": str(common_root),
        "trajectory_ids": ids,
        "files": files,
        "file_integrity": integrity,
        "strata": strata,
        "condition_audit": audit,
        "provenance_summary": corpus_provenance_summary(splits),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(output_path)
    return output_path
