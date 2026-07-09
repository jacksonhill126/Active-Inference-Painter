import numpy as np
import torch
from torch import nn

from active_painter.arm_sim import ArmPainterSim
from active_painter.config import PainterConfig
from active_painter.efe_common import project_material_support
from active_painter.env import StrokeAction
from active_painter.local_spatial import (
    local_patch_bounds_for_action,
    local_patch_bounds_for_raster,
    local_patch_transition_from_states,
    paste_patch,
)
from active_painter.models import LocalSpatialDynamicsEnsemble
from active_painter.policies import Policy
from active_painter.preferences import TerminalCoveragePreference
from active_painter.spatial_efe import SpatialExpectedFreeEnergy
from active_painter.spatial_state import rasterize_stroke_action, spatial_canvas_state


class DeterministicPatchDynamics:
    def __init__(self, deposit_scale: float = 0.01) -> None:
        self.deposit_scale = deposit_scale

    def predictive_moments(
        self,
        material: torch.Tensor,
        action_raster: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        footprint = action_raster[:, 0:1]
        amount = action_raster[:, 4:5]
        tone = action_raster[:, 5:6]
        deposited = self.deposit_scale * amount * footprint
        delta = torch.zeros_like(material)
        delta[:, 0:1] = deposited
        delta[:, 1:2] = 0.5 * deposited
        delta[:, 2:3] = tone * deposited
        next_material = project_material_support(material, material + delta, 0.005, 0.34)
        aleatoric = torch.full_like(next_material, 1e-6)
        epistemic = torch.zeros_like(next_material)
        return next_material, aleatoric, epistemic


class DivergentPatchMember(nn.Module):
    def __init__(self, delta_scale: float) -> None:
        super().__init__()
        self.delta_scale = delta_scale

    def forward(self, material: torch.Tensor, action_raster: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = torch.zeros_like(material)
        delta[:, 0:1] = self.delta_scale * action_raster[:, 0:1]
        next_material = project_material_support(material, material + delta, 0.005, 0.34)
        return next_material, torch.full_like(next_material, -10.0)


def test_local_patch_bounds_cover_compact_action_support_and_clip_edges() -> None:
    cfg = PainterConfig(canvas_size=32, local_patch_margin_cells=2, local_patch_min_cells=8)
    action = StrokeAction(0.01, 0.02, 0.22, 0.08, 0.08, 0.5, 1.0)

    bounds = local_patch_bounds_for_action(action, 32, cfg)
    assert bounds is not None
    raster = rasterize_stroke_action(action, 32)
    support = np.any(raster[:3] > 0.0, axis=0)
    covered = np.zeros_like(support)
    row_slice, col_slice = bounds.slices()
    covered[row_slice, col_slice] = True

    assert np.all(covered[support])
    assert bounds.row0 == 0
    assert bounds.col0 == 0
    assert bounds.height >= cfg.local_patch_min_cells
    assert bounds.width >= cfg.local_patch_min_cells


def test_local_patch_bounds_from_raster_match_action_bounds() -> None:
    cfg = PainterConfig(canvas_size=32, local_patch_margin_cells=3, local_patch_min_cells=10)
    action = StrokeAction(0.15, 0.2, 0.65, 0.48, 0.11, 0.5, 1.0)
    raster = rasterize_stroke_action(action, 32, config=cfg)

    from_action = local_patch_bounds_for_action(action, 32, cfg)
    from_raster = local_patch_bounds_for_raster(raster, 32, cfg)

    assert from_action == from_raster


def test_stop_action_has_no_local_patch() -> None:
    cfg = PainterConfig(canvas_size=32)

    assert local_patch_bounds_for_action(StrokeAction.stop_action(), 32, cfg) is None


def test_paste_patch_leaves_outside_support_unchanged() -> None:
    cfg = PainterConfig(canvas_size=16, local_patch_margin_cells=1, local_patch_min_cells=4)
    field = np.arange(6 * 16 * 16, dtype=np.float32).reshape(6, 16, 16)
    action = StrokeAction(0.4, 0.4, 0.6, 0.6, 0.05, 0.5, 1.0)
    bounds = local_patch_bounds_for_action(action, 16, cfg)
    assert bounds is not None
    patch = np.zeros((6, bounds.height, bounds.width), dtype=np.float32)

    pasted = paste_patch(field, patch, bounds)
    outside = np.ones((16, 16), dtype=bool)
    row_slice, col_slice = bounds.slices()
    outside[row_slice, col_slice] = False

    assert np.allclose(pasted[:, outside], field[:, outside])
    assert np.allclose(pasted[:, row_slice, col_slice], 0.0)


def test_local_transition_keeps_white_material_coverage_on_pixel_patch() -> None:
    cfg = PainterConfig(canvas_size=32, spatial_grid_size=8, local_patch_margin_cells=4, local_patch_min_cells=8)
    sim = ArmPainterSim(cfg)
    before = spatial_canvas_state(sim, cfg)
    sim.canvas.paint_at(np.asarray([0.0, sim.canvas.distance, 0.0]), pressure=0.8, tone=0.0, dt=0.2)
    after = spatial_canvas_state(sim, cfg)
    action = StrokeAction(0.45, 0.5, 0.55, 0.5, 0.12, 0.6, 0.0)

    transition = local_patch_transition_from_states(before, action, after, cfg)

    assert transition is not None
    assert transition.next_material[5].mean() > transition.material[5].mean()
    assert np.isclose(transition.next_material[2].mean(), transition.material[2].mean())


def test_local_patch_nll_masks_padded_inputs_and_targets() -> None:
    cfg = PainterConfig(
        spatial_transition_mode="local_patch",
        spatial_hidden_channels=8,
        spatial_residual_blocks=1,
        spatial_ensemble_size=2,
        ensemble_bootstrap_probability=1.0,
    )
    model = LocalSpatialDynamicsEnsemble(cfg)
    material = torch.zeros(2, cfg.spatial_material_channels, 8, 8)
    action = torch.zeros(2, cfg.spatial_action_channels, 8, 8)
    next_material = torch.zeros_like(material)
    mask = torch.zeros(2, 1, 8, 8)
    mask[:, :, :4, :4] = 1.0
    material_variant = material.clone()
    action_variant = action.clone()
    next_variant = next_material.clone()
    material_variant[:, :, 4:, :] = 99.0
    action_variant[:, :, :, 4:] = -99.0
    next_variant[:, :, 4:, 4:] = 42.0

    loss = model.nll(material, action, next_material, mask)
    variant_loss = model.nll(material_variant, action_variant, next_variant, mask)

    assert torch.isclose(loss, variant_loss)


def test_local_pixel_rollout_matches_dense_terminal_consequence_on_same_grid() -> None:
    cfg = PainterConfig(
        canvas_size=16,
        spatial_grid_size=16,
        spatial_transition_mode="local_patch",
        transition_precision=0.0,
        ambiguity_precision=0.0,
        composition_gap_precision=0.0,
        local_patch_margin_cells=0,
        local_patch_min_cells=1,
    )
    dense_cfg = PainterConfig(
        canvas_size=16,
        spatial_grid_size=16,
        spatial_transition_mode="dense_grid",
        transition_precision=0.0,
        ambiguity_precision=0.0,
        composition_gap_precision=0.0,
        local_patch_margin_cells=0,
        local_patch_min_cells=1,
    )
    belief = spatial_canvas_state(ArmPainterSim(cfg), cfg, logvar_value=float(np.log(1e-6)))
    policy = Policy((StrokeAction(0.25, 0.25, 0.75, 0.75, 0.08, 0.6, 1.0), StrokeAction.stop_action()))

    local = SpatialExpectedFreeEnergy(cfg, DeterministicPatchDynamics(), TerminalCoveragePreference(cfg)).evaluate(
        belief, policy
    )
    dense = SpatialExpectedFreeEnergy(dense_cfg, DeterministicPatchDynamics(), TerminalCoveragePreference(dense_cfg)).evaluate(
        belief, policy
    )

    assert local.rollout_mode == "local_patch"
    assert local.rollout_grid_size == 16
    assert np.isclose(local.terminal_coverage_mean, dense.terminal_coverage_mean)
    assert np.isclose(local.terminal_risk, dense.terminal_risk)


def test_local_ensemble_disagreement_contributes_epistemic_value_on_patch_support() -> None:
    cfg = PainterConfig(
        canvas_size=24,
        spatial_grid_size=8,
        spatial_transition_mode="local_patch",
        ambiguity_precision=0.0,
        composition_gap_precision=0.0,
        local_patch_margin_cells=1,
        local_patch_min_cells=4,
        spatial_ensemble_size=2,
    )
    action = StrokeAction(0.35, 0.35, 0.65, 0.65, 0.1, 0.7, 1.0)
    policy = Policy((action, StrokeAction.stop_action()))
    belief = spatial_canvas_state(ArmPainterSim(cfg), cfg)
    same = LocalSpatialDynamicsEnsemble(cfg)
    same.members = nn.ModuleList([DivergentPatchMember(0.01), DivergentPatchMember(0.01)])
    divergent = LocalSpatialDynamicsEnsemble(cfg)
    divergent.members = nn.ModuleList([DivergentPatchMember(0.0), DivergentPatchMember(0.03)])

    same_components = SpatialExpectedFreeEnergy(cfg, same, TerminalCoveragePreference(cfg)).evaluate(belief, policy)
    divergent_components = SpatialExpectedFreeEnergy(cfg, divergent, TerminalCoveragePreference(cfg)).evaluate(belief, policy)

    assert divergent_components.active_patch_area_fraction > 0.0
    assert divergent_components.epistemic_value > same_components.epistemic_value


def test_batched_local_ensemble_rollout_matches_single_policy_evaluation() -> None:
    cfg = PainterConfig(
        canvas_size=24,
        spatial_grid_size=8,
        spatial_transition_mode="local_patch",
        ambiguity_precision=0.0,
        composition_gap_precision=0.0,
        local_patch_margin_cells=1,
        local_patch_min_cells=4,
        spatial_ensemble_size=2,
    )
    policies = [
        Policy((StrokeAction(0.2, 0.2, 0.55, 0.35, 0.08, 0.7, 1.0), StrokeAction.stop_action())),
        Policy(
            (
                StrokeAction(0.35, 0.65, 0.75, 0.65, 0.10, 0.5, 0.0),
                StrokeAction(0.70, 0.30, 0.88, 0.45, 0.06, 0.4, 0.0),
                StrokeAction.stop_action(),
            )
        ),
    ]
    belief = spatial_canvas_state(ArmPainterSim(cfg), cfg)
    dynamics = LocalSpatialDynamicsEnsemble(cfg)
    dynamics.members = nn.ModuleList([DivergentPatchMember(0.0), DivergentPatchMember(0.03)])
    efe = SpatialExpectedFreeEnergy(cfg, dynamics, TerminalCoveragePreference(cfg))

    batched = efe.evaluate_batch(belief, policies)
    singles = [efe._evaluate_local_ensemble_policy(belief, policy) for policy in policies]

    for batch_component, single_component in zip(batched, singles):
        assert np.isclose(batch_component.total, single_component.total)
        assert np.isclose(batch_component.terminal_coverage_mean, single_component.terminal_coverage_mean)
        assert np.isclose(batch_component.transition_risk, single_component.transition_risk)
        assert np.isclose(batch_component.transition_ambiguity, single_component.transition_ambiguity)
        assert np.isclose(batch_component.epistemic_value, single_component.epistemic_value)
        assert batch_component.local_transition_steps == single_component.local_transition_steps
        assert np.isclose(batch_component.active_patch_area_fraction, single_component.active_patch_area_fraction)


def test_local_composition_risk_uses_coarse_grained_terminal_fields() -> None:
    class ShapeComposition:
        def __init__(self) -> None:
            self.shape: tuple[int, ...] | None = None

        def compression_gap(self, fields: torch.Tensor) -> torch.Tensor:
            self.shape = tuple(fields.shape)
            return torch.ones(fields.shape[0], device=fields.device)

    cfg = PainterConfig(
        canvas_size=32,
        spatial_grid_size=8,
        spatial_transition_mode="local_patch",
        composition_gap_precision=1.0,
        transition_precision=0.0,
        ambiguity_precision=0.0,
    )
    composition = ShapeComposition()
    belief = spatial_canvas_state(ArmPainterSim(cfg), cfg)
    policy = Policy((StrokeAction(0.2, 0.2, 0.8, 0.2, 0.08, 0.5, 1.0), StrokeAction.stop_action()))

    components = SpatialExpectedFreeEnergy(
        cfg,
        DeterministicPatchDynamics(),
        TerminalCoveragePreference(cfg),
        composition=composition,
    ).evaluate(belief, policy)

    assert composition.shape is not None
    assert composition.shape[-2:] == (cfg.spatial_grid_size, cfg.spatial_grid_size)
    assert np.isclose(components.composition_risk, -cfg.composition_gap_precision)
