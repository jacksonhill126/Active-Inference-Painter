from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from active_painter.arm_agent_driver import ArmActiveInferenceDriver
from active_painter.brush_loading import BrushLoadBelief
from active_painter.conditional_patch_vae import (
    ConditionalPatchBatch,
    ConditionalPatchVAE,
    ConditionalPatchVAEConfig,
    ConditionalPatchVAEEnsemble,
    conditional_patch_examples_from_shards,
)
from active_painter.conditional_vae_train import (
    CHECKPOINT_SCHEMA,
    train_conditional_vae_from_manifest,
)
from active_painter.config import PainterConfig
from active_painter.env import StrokeAction
from active_painter.learning_lifecycle import ORACLE_DIAGNOSTIC_EXECUTION
from active_painter.policies import MotorPrimitiveLatent
from active_painter.spatial_state import SpatialCanvasState
from active_painter.trajectory_corpus import (
    TERMINATION_POLICY_STOP,
    TrajectoryRecorder,
    load_trajectory_shard,
    write_split_manifest,
)


def _model_config(*, ensemble_size: int = 1) -> ConditionalPatchVAEConfig:
    return ConditionalPatchVAEConfig(
        material_channels=6,
        action_channels=12,
        brush_context_dim=5,
        hidden_channels=4,
        residual_blocks=1,
        latent_dim=3,
        ensemble_size=ensemble_size,
        bootstrap_probability=1.0,
    )


def _batch(batch_size: int = 2) -> ConditionalPatchBatch:
    material = torch.zeros(batch_size, 6, 7, 9)
    material[:, 3] = 0.34
    next_material = material.clone()
    next_material[:, 0, 2:5, 2:7] = 0.02
    next_material[:, 1, 2:5, 2:7] = 0.01
    next_material[:, 2, 2:5, 2:7] = 0.012
    next_material[:, 3, 2:5, 2:7] = 0.8
    next_material[:, 4, 2:5, 2:7] = 0.46
    next_material[:, 5, 2:5, 2:7] = 1.0
    action = torch.zeros(batch_size, 12, 7, 9)
    action[:, 0, 2:5, 2:7] = 1.0
    action[:, 4] = 0.6
    action[:, 5] = 1.0
    action[:, 7] = 1.0
    brush = torch.tensor([[0.8, 0.1, 0.9, 0.05, 1.0]]).repeat(
        batch_size, 1
    )
    mask = torch.ones(batch_size, 1, 7, 9)
    return ConditionalPatchBatch(
        material=material,
        material_logvar=torch.full_like(material, -6.0),
        action=action,
        brush_condition=brush,
        next_material=next_material,
        next_material_logvar=torch.full_like(next_material, -6.0),
        mask=mask,
    )


def test_standard_beta_one_vfe_decomposition_is_explicit() -> None:
    torch.manual_seed(3)
    model = ConditionalPatchVAE(_model_config())
    batch = _batch()
    components = model.vfe_components(
        batch, generator=torch.Generator().manual_seed(8)
    )

    assert torch.allclose(
        components.negative_elbo,
        components.reconstruction_nll + components.latent_kl,
    )
    assert torch.allclose(
        components.free_energy_per_element,
        components.negative_elbo / components.valid_element_count,
    )
    assert torch.all(components.latent_kl >= 0.0)
    assert torch.isfinite(components.free_energy_per_element).all()


def test_decoder_is_conditioned_on_action_and_compact_brush_posterior() -> None:
    torch.manual_seed(4)
    model = ConditionalPatchVAE(_model_config())
    batch = _batch(batch_size=1)
    latent = torch.zeros(1, model.config.latent_dim)
    baseline_mean, baseline_logvar = model.decode(
        batch.material,
        batch.material_logvar,
        batch.action,
        batch.brush_condition,
        latent,
        batch.mask,
    )
    changed_action = batch.action.clone()
    changed_action[:, 4] = 0.1
    action_mean, action_logvar = model.decode(
        batch.material,
        batch.material_logvar,
        changed_action,
        batch.brush_condition,
        latent,
        batch.mask,
    )
    changed_brush = batch.brush_condition.clone()
    changed_brush[:, 0] = 0.1
    brush_mean, brush_logvar = model.decode(
        batch.material,
        batch.material_logvar,
        batch.action,
        changed_brush,
        latent,
        batch.mask,
    )

    assert not (
        torch.allclose(baseline_mean, action_mean)
        and torch.allclose(baseline_logvar, action_logvar)
    )
    assert not (
        torch.allclose(baseline_mean, brush_mean)
        and torch.allclose(baseline_logvar, brush_logvar)
    )


class _KnownPriorMember(nn.Module):
    def __init__(self, member_offset: float) -> None:
        super().__init__()
        self.member_offset = float(member_offset)

    def prior_predictions(self, batch, *, samples, generator=None):
        del generator
        assert samples == 2
        base = torch.zeros_like(batch.material) + self.member_offset
        means = torch.stack([base - 1.0, base + 1.0], dim=0)
        logvars = torch.full_like(means, float(np.log(2.0)))
        return means, logvars


def test_predictive_uncertainty_separates_likelihood_latent_and_member_terms() -> None:
    ensemble = ConditionalPatchVAEEnsemble(_model_config(ensemble_size=2))
    ensemble.members = nn.ModuleList([_KnownPriorMember(0.0), _KnownPriorMember(2.0)])
    moments = ensemble.predictive_moments(_batch(batch_size=1), latent_samples=2)

    assert torch.allclose(moments.mean, torch.ones_like(moments.mean))
    assert torch.allclose(
        moments.likelihood_variance,
        torch.full_like(moments.likelihood_variance, 2.0),
    )
    assert torch.allclose(
        moments.latent_variance,
        torch.ones_like(moments.latent_variance),
    )
    assert torch.allclose(
        moments.epistemic_variance,
        torch.ones_like(moments.epistemic_variance),
    )
    assert torch.allclose(
        moments.total_variance,
        torch.full_like(moments.total_variance, 4.0),
    )


def _painter_config() -> PainterConfig:
    return PainterConfig(
        canvas_size=8,
        planner_state_kind="spatial_material",
        spatial_grid_size=8,
        material_pyramid_levels=(8,),
        spatial_hidden_channels=4,
        spatial_residual_blocks=1,
        spatial_ensemble_size=1,
        local_patch_margin_cells=1,
        local_patch_min_cells=4,
        composition_enabled=False,
        composition_gap_precision=0.0,
        candidate_policies=2,
        planning_horizon=1,
        batch_size=1,
    )


def _state(value: float, revision: int) -> SpatialCanvasState:
    material = np.zeros((6, 8, 8), dtype=np.float32)
    material[0, 2:6, 2:6] = value
    material[1, 2:6, 2:6] = value * 0.5
    material[2, 2:6, 2:6] = value * 0.7
    material[3] = 0.34
    material[3, 2:6, 2:6] = 0.8
    material[4, 2:6, 2:6] = 0.46
    material[5, 2:6, 2:6] = float(value >= 0.0001)
    return SpatialCanvasState(
        material=material,
        logvar=np.full_like(material, -6.0),
        posterior_revision=revision,
        inference_model_id=f"test-camera-posterior:{revision}",
        calibration_status="provisional_simulation_only_not_hardware_calibrated",
    )


def _record_one(root: Path, index: int) -> Path:
    config = _painter_config()
    recorder = TrajectoryRecorder(
        root,
        config,
        worker_id=index,
        seed=100 + index,
        provenance={"process_truth_role": "not stored and not used as a training input"},
    )
    before = _state(0.0, 2 * index)
    after = _state(0.02 + 0.005 * index, 2 * index + 1)
    action = StrokeAction(0.2, 0.25, 0.75, 0.7, 0.08, 0.6, float(index % 2))
    brush = BrushLoadBelief(
        load_mean=0.8 - 0.1 * index,
        load_variance=0.01,
        black_fraction_mean=float(index % 2),
        black_fraction_variance=0.0025,
        revision=index,
    )
    recorder.record_transition(
        before,
        action,
        MotorPrimitiveLatent(kind="cartesian_ik"),
        after,
        brush,
    )
    return recorder.complete(
        after, termination=TERMINATION_POLICY_STOP, painting_index=index
    )


def test_corpus_adapter_uses_inferred_brush_context_and_rejects_truth_claims(
    tmp_path: Path,
) -> None:
    shard = load_trajectory_shard(_record_one(tmp_path, 0))
    assert np.allclose(shard.brush_condition[0], [0.8, 0.1, 0.0, 0.05, 1.0])
    examples = conditional_patch_examples_from_shards([shard], _painter_config())
    assert len(examples) == 1
    assert examples[0].brush_condition[-1] == 1.0
    assert examples[0].trajectory_id == shard.trajectory_id

    invalid = replace(
        shard,
        metadata={**shard.metadata, "process_truth_used_as_training_input": True},
    )
    with pytest.raises(ValueError, match="process-truth-free"):
        conditional_patch_examples_from_shards([invalid], _painter_config())


def test_driver_observed_transition_callback_carries_pre_stroke_brush_belief() -> None:
    driver = ArmActiveInferenceDriver(
        config=_painter_config(),
        bootstrap_transitions=0,
        bootstrap_train_steps=0,
    )
    before = _state(0.0, 0)
    after = _state(0.03, 1)
    action = StrokeAction(0.2, 0.25, 0.75, 0.7, 0.08, 0.6, 1.0)
    primitive = MotorPrimitiveLatent(kind="cartesian_ik")
    brush = BrushLoadBelief(
        load_mean=0.75,
        load_variance=0.01,
        black_fraction_mean=1.0,
        black_fraction_variance=0.0025,
        revision=3,
    )
    observed: list[tuple[object, ...]] = []
    driver.on_observed_transition = lambda *values: observed.append(values)

    driver._add_transition_to_agent(
        before,
        action,
        after,
        primitive,
        brush,
        evidence_source=ORACLE_DIAGNOSTIC_EXECUTION,
    )

    assert len(observed) == 1
    assert observed[0][0] is before
    assert observed[0][1] is action
    assert observed[0][2] is primitive
    assert observed[0][3] is after
    assert observed[0][4] is brush


def test_shadow_trainer_writes_isolated_checkpoint_and_capability_report(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    paths = [_record_one(corpus, index) for index in range(3)]
    splits = {
        "train": [paths[0]],
        "validation": [paths[1]],
        "test": [paths[2]],
    }
    manifest = write_split_manifest(
        corpus / "split_manifest.json",
        splits,
        seed=9,
        ratios=(1 / 3, 1 / 3, 1 / 3),
    )
    checkpoint = tmp_path / "shadow_cvae.pt"
    report = train_conditional_vae_from_manifest(
        argparse.Namespace(
            manifest=str(manifest),
            output_checkpoint=str(checkpoint),
            input_checkpoint=None,
            report_path=str(tmp_path / "shadow_report.json"),
            device="cpu",
            seed=11,
            batch_size=1,
            evaluation_batch_size=1,
            gradient_steps=1,
            learning_rate=1e-3,
            hidden_channels=4,
            residual_blocks=0,
            latent_dim=2,
            ensemble_size=1,
            importance_samples=1,
            prior_samples=2,
        )
    )

    assert checkpoint.is_file()
    assert report["status"] == "shadow_offline_not_policy_active"
    assert report["policy_influence"] == "none"
    assert report["patch_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert report["heldout_after"]["validation"]["conditioning_checks"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["schema"] == CHECKPOINT_SCHEMA
    assert payload["training_provenance"]["training_split_only"] is True
    assert payload["training_provenance"]["policy_influence"] == "none"
    assert payload["training_provenance"]["process_truth_used_as_training_input"] is False
