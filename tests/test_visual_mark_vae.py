from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import uuid
from argparse import Namespace
import json

import numpy as np
import pytest
import torch

from active_painter.camera_observation import (
    GLOBAL_CANVAS_PRODUCT,
    CameraFrame,
    CameraObservationBundle,
)
from active_painter.env import StrokeAction
from active_painter.visual_mark_vae import (
    VisualMarkVAE,
    VisualMarkVAEConfig,
    make_visual_mark_batch,
    visual_mark_examples_from_paths,
)
from active_painter.visual_trajectory_corpus import (
    VISUAL_TRAJECTORY_CORPUS_SCHEMA,
    VisualTrajectoryRecorder,
    VisualTrajectoryShard,
    load_visual_trajectory_shard,
    split_visual_trajectory_paths,
    write_visual_split_manifest,
)
from active_painter.visual_vae_train import (
    VISUAL_VAE_CHECKPOINT_SCHEMA,
    train_visual_vae,
)
from active_painter.visual_collect import collect_visual_corpus


@pytest.fixture
def visual_tmp_path() -> Path:
    # Python 3.14's temporary-directory helpers apply POSIX mode bits that can
    # create an unreadable ACL in this Windows workspace.  Plain mkdir inherits
    # the workspace ACL; these ignored test artifacts are cleaned by CI/workspace
    # housekeeping rather than by chmod-based tempfile cleanup.
    path = Path("data") / f"visual-vae-test-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _frame(
    value: float,
    *,
    camera_name: str,
    sequence: int,
    capture_time_s: float,
) -> CameraFrame:
    image = np.full((48, 64), value, dtype=np.float32)
    return CameraFrame(
        camera_name=camera_name,
        role="contact_tracking",
        sequence=sequence,
        product_kind=GLOBAL_CANVAS_PRODUCT,
        product_id="global",
        capture_time_s=capture_time_s,
        available_time_s=capture_time_s + 0.01,
        calibration_revision="test-calibration-v1",
        observation_model="test-camera-likelihood-v1",
        registration="canvas_plane_homography",
        sampling_kind="native_to_canvas_homography",
        source_resolution_px=(64, 48),
        declared_acquisition_resolution_px=(64, 48),
        grayscale=image,
        calibration_validity=np.ones_like(image, dtype=np.bool_),
    )


def _bundle(value: float, sequence: int, capture_time_s: float) -> CameraObservationBundle:
    frames = tuple(
        _frame(
            value + 0.02 * index,
            camera_name=name,
            sequence=sequence,
            capture_time_s=capture_time_s,
        )
        for index, name in enumerate(("left", "right"))
    )
    return CameraObservationBundle(
        monotonic_time_s=capture_time_s + 0.01,
        frames=frames,
    )


def _write_trajectory(
    root: Path,
    *,
    worker_id: int,
    seed: int,
    before: float = 0.8,
    after: float = 0.3,
) -> Path:
    recorder = VisualTrajectoryRecorder(
        root,
        worker_id=worker_id,
        seed=seed,
        provenance={"test": True},
    )
    recorder.record_transition(
        _bundle(before, 1, 1.0),
        StrokeAction(0.2, 0.3, 0.8, 0.7, 0.08, 0.5, 1.0, curvature=0.12),
        None,
        _bundle(after, 2, 2.0),
    )
    return recorder.complete(termination="fixed_horizon_truncation")


def test_visual_corpus_records_registered_images_without_material_state(visual_tmp_path: Path) -> None:
    path = _write_trajectory(visual_tmp_path, worker_id=0, seed=11)
    shard = load_visual_trajectory_shard(path)

    assert shard.metadata["schema"] == VISUAL_TRAJECTORY_CORPUS_SCHEMA
    assert shard.metadata["process_truth_used_as_training_input"] is False
    assert shard.metadata["persistent_material_latents_stored"] is False
    assert shard.example_count == 2
    assert shard.pre_image.dtype == np.float16
    assert shard.post_capture_time_s.min() >= shard.pre_capture_time_s.max()
    forbidden = {"material", "wetness", "thickness", "contact", "qpos"}
    assert forbidden.isdisjoint(field.name for field in fields(VisualTrajectoryShard))


def test_visual_split_keeps_complete_trajectories_disjoint(visual_tmp_path: Path) -> None:
    paths = [
        _write_trajectory(visual_tmp_path, worker_id=index, seed=100 + index)
        for index in range(6)
    ]
    splits = split_visual_trajectory_paths(paths, seed=19)
    manifest_path = write_visual_split_manifest(
        visual_tmp_path / "manifest.json",
        splits,
        seed=19,
        ratios=(0.7, 0.15, 0.15),
    )
    ids = {
        name: {load_visual_trajectory_shard(path).trajectory_id for path in selected}
        for name, selected in splits.items()
    }

    assert all(ids[name] for name in ("train", "validation", "test"))
    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])
    assert manifest_path.exists()
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["motor_kinds"] == ["cartesian_ik"]


def test_visual_vae_logs_normalized_likelihood_and_uses_conditional_prior(visual_tmp_path: Path) -> None:
    path = _write_trajectory(visual_tmp_path, worker_id=0, seed=7)
    config = VisualMarkVAEConfig(
        patch_size=32,
        latent_dim=4,
        base_channels=4,
        condition_channels=4,
        camera_names=("left", "right"),
    )
    examples = visual_mark_examples_from_paths([path], config)
    batch = make_visual_mark_batch(examples, "cpu")
    model = VisualMarkVAE(config)
    terms = model.vfe_components(batch, generator=torch.Generator().manual_seed(3))

    assert torch.isfinite(terms.free_energy).all()
    assert torch.allclose(
        terms.free_energy,
        terms.reconstruction_nll + terms.latent_kl,
        atol=1e-5,
    )
    assert (terms.valid_pixel_count > 0).all()
    prediction = model.prior_predictive_mean(
        batch, samples=2, generator=torch.Generator().manual_seed(4)
    )
    assert prediction.shape == batch.after.shape
    assert torch.isfinite(prediction).all()
    assert torch.all((prediction > 0.0) & (prediction < 1.0))
    iwae = model.importance_weighted_nll(
        batch, samples=2, generator=torch.Generator().manual_seed(5)
    )
    assert iwae.shape == (len(examples),)
    assert torch.isfinite(iwae).all()


def test_visual_recorder_rejects_unpaired_camera_evidence(visual_tmp_path: Path) -> None:
    recorder = VisualTrajectoryRecorder(
        visual_tmp_path,
        worker_id=0,
        seed=1,
        provenance={},
    )
    before = CameraObservationBundle(
        monotonic_time_s=1.01,
        frames=(_frame(0.8, camera_name="left", sequence=1, capture_time_s=1.0),),
    )
    after = CameraObservationBundle(
        monotonic_time_s=2.01,
        frames=(_frame(0.2, camera_name="right", sequence=2, capture_time_s=2.0),),
    )
    with pytest.raises(ValueError, match="share no"):
        recorder.record_transition(
            before,
            StrokeAction(0.2, 0.2, 0.8, 0.8, 0.1, 0.5, 1.0),
            None,
            after,
        )


def test_visual_training_checkpoint_resumes_to_total_epoch(visual_tmp_path: Path) -> None:
    paths = [
        _write_trajectory(visual_tmp_path, worker_id=index, seed=30 + index)
        for index in range(3)
    ]
    splits = split_visual_trajectory_paths(paths, seed=8)
    manifest = write_visual_split_manifest(
        visual_tmp_path / "manifest.json",
        splits,
        seed=8,
        ratios=(0.7, 0.15, 0.15),
    )
    checkpoint = visual_tmp_path / "model.pt"
    common = dict(
        manifest=str(manifest),
        checkpoint=str(checkpoint),
        report=str(visual_tmp_path / "report.json"),
        panel=None,
        batch_size=2,
        learning_rate=2.0e-4,
        patch_size=16,
        latent_dim=2,
        base_channels=4,
        condition_channels=4,
        importance_samples=1,
        prior_samples=1,
        panel_every_epochs=1,
        seed=9,
        device="cpu",
        torch_threads=1,
    )
    train_visual_vae(Namespace(**common, resume=False, epochs=1))
    train_visual_vae(Namespace(**common, resume=True, epochs=2))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert saved["schema"] == VISUAL_VAE_CHECKPOINT_SCHEMA
    assert saved["epoch"] == 2
    assert len(saved["history"]) == 2
    assert saved["process_truth_used_as_training_input"] is False


def test_visual_collection_resume_reuses_complete_shards(visual_tmp_path: Path) -> None:
    existing = _write_trajectory(visual_tmp_path, worker_id=0, seed=77)
    report = collect_visual_corpus(
        Namespace(
            output_dir=str(visual_tmp_path),
            manifest=str(visual_tmp_path / "resumed-manifest.json"),
            report=str(visual_tmp_path / "resumed-report.json"),
            trajectories=1,
            workers=1,
            resume=True,
            max_transitions_per_trajectory=8,
            max_worker_seconds=60.0,
            canvas_size=64,
            spatial_grid_size=16,
            torch_threads=1,
            seed=77,
            motor_realization_profile="bounded_fixed_roll",
            split_ratios=(1.0, 0.0, 0.0),
        )
    )

    assert report["trajectory_count"] == 1
    assert report["resumed_existing_trajectory_count"] == 1
    assert Path(existing).exists()
