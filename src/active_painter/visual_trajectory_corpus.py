"""Registered pre/post camera trajectories for visual mark-model learning.

The unit of dataset splitting is a complete painting trajectory.  Each row is
one camera view of one executed mark, paired across the action-conditioned
camera clock.  The corpus intentionally stores no simulator material arrays,
contact truth, segmentation, or exact robot state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Mapping, Sequence

import numpy as np

from .action_encoding import motor_kind
from .brush_loading import BrushLoadBelief
from .camera_observation import (
    CAMERA_OBSERVATION_INTERFACE_VERSION,
    GLOBAL_CANVAS_PRODUCT,
    CameraFrame,
    CameraObservationBundle,
)
from .env import StrokeAction
from .policies import MotorPrimitiveLatent
from .trajectory_corpus import (
    SPLIT_NAMES,
    TERMINATION_FIXED_HORIZON,
    TERMINATION_POLICY_STOP,
    brush_condition_vector,
)


VISUAL_TRAJECTORY_CORPUS_SCHEMA = "registered-visual-trajectory-corpus-v1"
VISUAL_SPLIT_MANIFEST_SCHEMA = "registered-visual-trajectory-split-manifest-v1"
VISUAL_TRANSITION_INTERFACE = "action-conditioned-registered-visual-transition-v1"


@dataclass(frozen=True, slots=True)
class VisualTransitionRecord:
    transition_index: int
    camera_name: str
    pre_image: np.ndarray
    post_image: np.ndarray
    pre_validity: np.ndarray
    post_validity: np.ndarray
    action: np.ndarray
    motor_kind: str
    brush_condition: np.ndarray
    brush_inference_model_id: str
    brush_calibration_status: str
    pre_sequence: int
    post_sequence: int
    pre_capture_time_s: float
    post_capture_time_s: float
    calibration_revision: str
    observation_model: str
    registration: str
    pre_product_id: str
    post_product_id: str


@dataclass(frozen=True, slots=True)
class VisualTrajectoryShard:
    path: Path
    metadata: dict[str, object]
    transition_index: np.ndarray
    camera_name: np.ndarray
    pre_image: np.ndarray
    post_image: np.ndarray
    pre_validity: np.ndarray
    post_validity: np.ndarray
    action: np.ndarray
    motor_kind: np.ndarray
    brush_condition: np.ndarray
    brush_inference_model_id: np.ndarray
    brush_calibration_status: np.ndarray
    pre_sequence: np.ndarray
    post_sequence: np.ndarray
    pre_capture_time_s: np.ndarray
    post_capture_time_s: np.ndarray
    calibration_revision: np.ndarray
    observation_model: np.ndarray
    registration: np.ndarray
    pre_product_id: np.ndarray
    post_product_id: np.ndarray

    @property
    def trajectory_id(self) -> str:
        return str(self.metadata["trajectory_id"])

    @property
    def example_count(self) -> int:
        return int(self.action.shape[0])


def _registered_global_frames(
    bundle: CameraObservationBundle,
) -> dict[str, CameraFrame]:
    if bundle.interface_version != CAMERA_OBSERVATION_INTERFACE_VERSION:
        raise ValueError("unsupported camera observation interface")
    frames: dict[str, CameraFrame] = {}
    for frame in bundle.frames:
        if frame.product_kind != GLOBAL_CANVAS_PRODUCT:
            continue
        if frame.registration != "canvas_plane_homography":
            raise ValueError("global visual training products must be canvas-registered")
        if frame.camera_name in frames:
            raise ValueError("a bundle contains duplicate global products for one camera")
        if not np.all(np.isfinite(frame.grayscale)):
            raise ValueError("camera images must be finite")
        frames[frame.camera_name] = frame
    return frames


class VisualTrajectoryRecorder:
    """Thread-safe recorder for camera evidence with whole-trajectory writes."""

    def __init__(
        self,
        output_dir: Path | str,
        *,
        worker_id: int,
        seed: int,
        provenance: Mapping[str, object],
        trajectory_start_index: int = 0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.worker_id = int(worker_id)
        self.seed = int(seed)
        self.provenance = dict(provenance)
        self._lock = threading.Lock()
        self._records: list[VisualTransitionRecord] = []
        self._action_count = 0
        if trajectory_start_index < 0:
            raise ValueError("trajectory_start_index must be non-negative")
        self._trajectory_index = int(trajectory_start_index)
        self.completed_paths: list[Path] = []

    @property
    def pending_transition_count(self) -> int:
        with self._lock:
            return self._action_count

    def record_transition(
        self,
        before: CameraObservationBundle,
        action: StrokeAction,
        primitive: MotorPrimitiveLatent | None,
        after: CameraObservationBundle,
        brush_belief: BrushLoadBelief | None = None,
    ) -> None:
        if action.stop:
            raise ValueError("stop has no visual mark transition")
        pre = _registered_global_frames(before)
        post = _registered_global_frames(after)
        camera_names = sorted(set(pre).intersection(post))
        if not camera_names:
            raise ValueError("pre/post bundles share no registered global camera")
        records: list[VisualTransitionRecord] = []
        with self._lock:
            transition_index = self._action_count
            self._action_count += 1
        for name in camera_names:
            first = pre[name]
            second = post[name]
            if second.capture_time_s + 1e-12 < first.capture_time_s:
                raise ValueError("post-action exposure precedes pre-action exposure")
            if first.grayscale.shape != second.grayscale.shape:
                raise ValueError("pre/post registered images have different shapes")
            if first.calibration_revision != second.calibration_revision:
                raise ValueError("calibration revision changed inside a visual transition")
            if first.observation_model != second.observation_model:
                raise ValueError("observation model changed inside a visual transition")
            records.append(
                VisualTransitionRecord(
                    transition_index=transition_index,
                    camera_name=name,
                    pre_image=first.grayscale.astype(np.float16, copy=True),
                    post_image=second.grayscale.astype(np.float16, copy=True),
                    pre_validity=first.calibration_validity.astype(np.bool_, copy=True),
                    post_validity=second.calibration_validity.astype(np.bool_, copy=True),
                    action=action.vector().astype(np.float32, copy=True),
                    motor_kind=motor_kind(primitive),
                    brush_condition=brush_condition_vector(brush_belief),
                    brush_inference_model_id=(
                        "unavailable" if brush_belief is None else brush_belief.inference_model_id
                    ),
                    brush_calibration_status=(
                        "unavailable" if brush_belief is None else brush_belief.calibration_status
                    ),
                    pre_sequence=int(first.sequence),
                    post_sequence=int(second.sequence),
                    pre_capture_time_s=float(first.capture_time_s),
                    post_capture_time_s=float(second.capture_time_s),
                    calibration_revision=first.calibration_revision,
                    observation_model=first.observation_model,
                    registration=first.registration,
                    pre_product_id=first.product_id,
                    post_product_id=second.product_id,
                )
            )
        with self._lock:
            self._records.extend(records)

    def complete(
        self,
        *,
        termination: str,
        painting_index: int | None = None,
    ) -> Path:
        if termination not in {TERMINATION_POLICY_STOP, TERMINATION_FIXED_HORIZON}:
            raise ValueError(f"unsupported trajectory termination: {termination!r}")
        with self._lock:
            records = self._records
            action_count = self._action_count
            self._records = []
            self._action_count = 0
            trajectory_index = self._trajectory_index
            self._trajectory_index += 1
        trajectory_id = (
            f"visual-worker-{self.worker_id:03d}-seed-{self.seed}-"
            f"trajectory-{trajectory_index:06d}"
        )
        metadata: dict[str, object] = {
            "schema": VISUAL_TRAJECTORY_CORPUS_SCHEMA,
            "transition_interface": VISUAL_TRANSITION_INTERFACE,
            "trajectory_id": trajectory_id,
            "worker_id": self.worker_id,
            "worker_seed": self.seed,
            "trajectory_index": trajectory_index,
            "painting_index": painting_index,
            "termination": termination,
            "transition_count": action_count,
            "camera_example_count": len(records),
            "split_unit": "entire_trajectory_before_crop_extraction",
            "process_truth_used_as_training_input": False,
            "stored_observation_product": GLOBAL_CANVAS_PRODUCT,
            "persistent_material_latents_stored": False,
            "provenance": self.provenance,
        }
        path = self.output_dir / f"{trajectory_id}.npz"
        _write_visual_shard(path, metadata, records)
        self.completed_paths.append(path)
        return path


def _stack(records: Sequence[VisualTransitionRecord], name: str, *, dtype) -> np.ndarray:
    if not records:
        return np.empty((0, 0, 0), dtype=dtype)
    arrays = [np.asarray(getattr(record, name)) for record in records]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError(f"{name} images do not share one registered shape")
    return np.stack(arrays).astype(dtype, copy=False)


def _write_visual_shard(
    path: Path,
    metadata: Mapping[str, object],
    records: Sequence[VisualTransitionRecord],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata_json": np.asarray(
            json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")),
            dtype=np.str_,
        ),
        "transition_index": np.asarray(
            [record.transition_index for record in records], dtype=np.int64
        ),
        "camera_name": np.asarray([record.camera_name for record in records], dtype=np.str_),
        "pre_image": _stack(records, "pre_image", dtype=np.float16),
        "post_image": _stack(records, "post_image", dtype=np.float16),
        "pre_validity": _stack(records, "pre_validity", dtype=np.bool_),
        "post_validity": _stack(records, "post_validity", dtype=np.bool_),
        "action": (
            np.stack([record.action for record in records]).astype(np.float32)
            if records else np.empty((0, 8), dtype=np.float32)
        ),
        "motor_kind": np.asarray([record.motor_kind for record in records], dtype=np.str_),
        "brush_condition": (
            np.stack([record.brush_condition for record in records]).astype(np.float32)
            if records else np.empty((0, 5), dtype=np.float32)
        ),
        "brush_inference_model_id": np.asarray(
            [record.brush_inference_model_id for record in records], dtype=np.str_
        ),
        "brush_calibration_status": np.asarray(
            [record.brush_calibration_status for record in records], dtype=np.str_
        ),
        "pre_sequence": np.asarray([record.pre_sequence for record in records], dtype=np.int64),
        "post_sequence": np.asarray([record.post_sequence for record in records], dtype=np.int64),
        "pre_capture_time_s": np.asarray(
            [record.pre_capture_time_s for record in records], dtype=np.float64
        ),
        "post_capture_time_s": np.asarray(
            [record.post_capture_time_s for record in records], dtype=np.float64
        ),
        "calibration_revision": np.asarray(
            [record.calibration_revision for record in records], dtype=np.str_
        ),
        "observation_model": np.asarray(
            [record.observation_model for record in records], dtype=np.str_
        ),
        "registration": np.asarray([record.registration for record in records], dtype=np.str_),
        "pre_product_id": np.asarray([record.pre_product_id for record in records], dtype=np.str_),
        "post_product_id": np.asarray([record.post_product_id for record in records], dtype=np.str_),
    }
    temp_path = path.with_name(f"{path.stem}.tmp.npz")
    np.savez_compressed(temp_path, **payload)
    temp_path.replace(path)


def load_visual_trajectory_shard(path: Path | str) -> VisualTrajectoryShard:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        arrays = {
            name: payload[name].copy()
            for name in payload.files
            if name != "metadata_json"
        }
    if metadata.get("schema") != VISUAL_TRAJECTORY_CORPUS_SCHEMA:
        raise ValueError(f"unsupported visual trajectory corpus schema in {path}")
    required = set(VisualTrajectoryShard.__dataclass_fields__) - {"path", "metadata"}
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"visual trajectory shard is missing {sorted(missing)}")
    count = int(arrays["action"].shape[0])
    if any(int(arrays[name].shape[0]) != count for name in required):
        raise ValueError("visual trajectory arrays disagree on example count")
    if arrays["pre_image"].shape != arrays["post_image"].shape:
        raise ValueError("visual pre/post arrays have different shapes")
    if arrays["pre_validity"].shape != arrays["pre_image"].shape:
        raise ValueError("visual pre validity has the wrong shape")
    if arrays["post_validity"].shape != arrays["post_image"].shape:
        raise ValueError("visual post validity has the wrong shape")
    if count and np.any(arrays["post_capture_time_s"] < arrays["pre_capture_time_s"]):
        raise ValueError("visual shard contains time-reversed transitions")
    return VisualTrajectoryShard(path=path, metadata=metadata, **arrays)


def split_visual_trajectory_paths(
    paths: Sequence[Path | str],
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, list[Path]]:
    if len(ratios) != 3 or any(value < 0.0 for value in ratios):
        raise ValueError("split ratios must be three non-negative values")
    total_ratio = float(sum(ratios))
    if total_ratio <= 0.0:
        raise ValueError("at least one split ratio must be positive")
    normalized = np.asarray(ratios, dtype=np.float64) / total_ratio
    loaded = [load_visual_trajectory_shard(path) for path in paths]
    ids = [shard.trajectory_id for shard in loaded]
    if len(ids) != len(set(ids)):
        raise ValueError("visual trajectory ids must be unique before splitting")
    order = np.random.default_rng(int(seed)).permutation(len(loaded)).tolist()
    counts = np.floor(normalized * len(loaded)).astype(int)
    remainder = len(loaded) - int(counts.sum())
    fractions = normalized * len(loaded) - counts
    for index in np.argsort(-fractions, kind="stable")[:remainder]:
        counts[index] += 1
    if len(loaded) >= 3:
        for index in range(3):
            if ratios[index] > 0.0 and counts[index] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] += 1
    result: dict[str, list[Path]] = {name: [] for name in SPLIT_NAMES}
    cursor = 0
    for name, count in zip(SPLIT_NAMES, counts.tolist(), strict=True):
        selected = order[cursor : cursor + count]
        result[name] = sorted(loaded[index].path for index in selected)
        cursor += count
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_visual_split_manifest(
    output_path: Path | str,
    splits: Mapping[str, Sequence[Path | str]],
    *,
    seed: int,
    ratios: tuple[float, float, float],
) -> Path:
    output_path = Path(output_path)
    all_paths = [Path(path).resolve() for name in SPLIT_NAMES for path in splits.get(name, ())]
    root = (
        Path(os.path.commonpath([str(path.parent) for path in all_paths]))
        if all_paths else output_path.parent.resolve()
    )
    ids: dict[str, list[str]] = {}
    files: dict[str, list[str]] = {}
    integrity: dict[str, list[dict[str, str]]] = {}
    camera_names: set[str] = set()
    motor_kinds: set[str] = set()
    for name in SPLIT_NAMES:
        paths = [Path(path).resolve() for path in splits.get(name, ())]
        shards = [load_visual_trajectory_shard(path) for path in paths]
        ids[name] = [shard.trajectory_id for shard in shards]
        files[name] = [str(path.relative_to(root)) for path in paths]
        integrity[name] = [
            {"file": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in paths
        ]
        for shard in shards:
            camera_names.update(str(item) for item in shard.camera_name.tolist())
            motor_kinds.update(str(item) for item in shard.motor_kind.tolist())
    flattened = [item for name in SPLIT_NAMES for item in ids[name]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("a visual trajectory appears in more than one split")
    payload = {
        "schema": VISUAL_SPLIT_MANIFEST_SCHEMA,
        "corpus_schema": VISUAL_TRAJECTORY_CORPUS_SCHEMA,
        "split_unit": "entire_trajectory_before_crop_extraction",
        "seed": int(seed),
        "ratios": dict(zip(SPLIT_NAMES, ratios, strict=True)),
        "root": str(root),
        "trajectory_ids": ids,
        "files": files,
        "file_integrity": integrity,
        "camera_names": sorted(camera_names),
        "motor_kinds": sorted(motor_kinds),
        "process_truth_used_as_training_input": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(output_path)
    return output_path


def visual_manifest_paths(
    manifest_path: Path | str,
) -> tuple[dict[str, list[Path]], dict[str, object]]:
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != VISUAL_SPLIT_MANIFEST_SCHEMA:
        raise ValueError("unsupported visual trajectory split manifest")
    root = Path(str(payload["root"]))
    if not root.is_absolute():
        root = (manifest_path.parent / root).resolve()
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("visual split manifest has no file mapping")
    paths = {
        name: [root / str(item) for item in files.get(name, ())]
        for name in SPLIT_NAMES
    }
    ids = payload.get("trajectory_ids", {})
    flattened = [str(item) for name in SPLIT_NAMES for item in ids.get(name, ())]
    if len(flattened) != len(set(flattened)):
        raise ValueError("a visual trajectory appears in more than one split")
    return paths, payload
