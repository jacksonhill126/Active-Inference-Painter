import numpy as np
import torch
from torch import nn

from active_painter.arm_sim import ArmPainterSim
from active_painter.config import PainterConfig
from active_painter.efe_common import project_material_support
from active_painter.env import StrokeAction
from active_painter.local_spatial import (
    LocalPatchBounds,
    LocalPatchReplayBuffer,
    LocalPatchTransition,
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


class RecordingPatchMember(DivergentPatchMember):
    def __init__(self) -> None:
        super().__init__(0.0)
        self.batch_sizes: list[int] = []

    def forward(self, material: torch.Tensor, action_raster: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.batch_sizes.append(int(material.shape[0]))
        return super().forward(material, action_raster)


def test_material_support_counts_painted_cells_once_regardless_of_layer_thickness() -> None:
    cfg = PainterConfig()
    current = torch.zeros(1, cfg.spatial_material_channels, 4, 4)
    first = current.clone()
    first[:, 0, 1:3, 1:3] = 2.0 * cfg.paint_presence_threshold
    layered = first.clone()
    layered[:, 0, 1:3, 1:3] *= 50.0

    projected_first = project_material_support(
        current,
        first,
        cfg.thickness_scale,
        cfg.canvas_ground_tone,
        cfg.paint_presence_threshold,
    )
    projected_layered = project_material_support(
        projected_first,
        layered,
        cfg.thickness_scale,
        cfg.canvas_ground_tone,
        cfg.paint_presence_threshold,
    )

    assert torch.equal(projected_first[:, 5], projected_layered[:, 5])
    assert projected_first[:, 5].mean().item() == 0.25


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


def test_local_replay_buckets_one_draw_without_largest_patch_padding() -> None:
    class FixedRng:
        @staticmethod
        def integers(_low: int, _high: int, size: int) -> np.ndarray:
            return np.arange(size, dtype=np.int64)

    replay = LocalPatchReplayBuffer(capacity=8)
    replay.rng = FixedRng()  # type: ignore[assignment]
    for index, (height, width) in enumerate(((8, 7), (13, 11), (100, 96))):
        bounds = LocalPatchBounds(0, height, 0, width, 128)
        material = np.full((6, height, width), 0.01 * index, dtype=np.float32)
        replay.add(
            LocalPatchTransition(
                bounds=bounds,
                material=material,
                action=np.zeros((9, height, width), dtype=np.float32),
                next_material=material.copy(),
            )
        )

    batches = replay.sample_buckets(
        batch_size=3,
        device=torch.device("cpu"),
        bucket_cells=16,
        sequential_cell_limit=8192,
    )

    assert sorted(batch.material.shape[0] for batch in batches) == [1, 2]
    assert sorted((batch.material.shape[-2], batch.material.shape[-1]) for batch in batches) == [
        (16, 16),
        (100, 96),
    ]
    assert sorted(index for batch in batches for index in batch.sample_indices) == [0, 1, 2]


def test_bucketed_patch_likelihood_matches_individual_observation_mean() -> None:
    class FixedRng:
        @staticmethod
        def integers(_low: int, _high: int, size: int) -> np.ndarray:
            return np.arange(size, dtype=np.int64)

    torch.manual_seed(8)
    cfg = PainterConfig(
        spatial_hidden_channels=8,
        spatial_residual_blocks=1,
        spatial_ensemble_size=2,
        ensemble_bootstrap_probability=1.0,
    )
    replay = LocalPatchReplayBuffer(capacity=8)
    replay.rng = FixedRng()  # type: ignore[assignment]
    transitions = []
    for height, width in ((7, 9), (13, 12), (21, 18)):
        bounds = LocalPatchBounds(0, height, 0, width, 32)
        material = np.random.default_rng(height).random((6, height, width), dtype=np.float32)
        action = np.random.default_rng(width).random((9, height, width), dtype=np.float32)
        transition = LocalPatchTransition(bounds, material, action, material.copy())
        replay.add(transition)
        transitions.append(transition)
    model = LocalSpatialDynamicsEnsemble(cfg)
    batches = replay.sample_buckets(3, torch.device("cpu"), bucket_cells=16, sequential_cell_limit=8192)

    bucketed = torch.cat(
        [
            model.per_sample_nll(batch.material, batch.action, batch.next_material, batch.mask)
            for batch in batches
        ],
        dim=1,
    ).mean()
    individual = torch.cat(
        [
            model.per_sample_nll(
                torch.tensor(transition.material).unsqueeze(0),
                torch.tensor(transition.action).unsqueeze(0),
                torch.tensor(transition.next_material).unsqueeze(0),
                torch.ones(1, 1, transition.bounds.height, transition.bounds.width),
            )
            for transition in transitions
        ],
        dim=1,
    ).mean()

    assert torch.allclose(bucketed, individual, atol=1e-6, rtol=1e-5)


def test_masked_padded_transition_matches_exact_patch_prediction() -> None:
    torch.manual_seed(4)
    cfg = PainterConfig(
        spatial_hidden_channels=8,
        spatial_residual_blocks=2,
        spatial_ensemble_size=1,
    )
    member = LocalSpatialDynamicsEnsemble(cfg).members[0]
    material = torch.rand(2, cfg.spatial_material_channels, 11, 19)
    action = torch.rand(2, cfg.spatial_action_channels, 11, 19)
    exact_mean, exact_logvar = member(material, action)
    padded_material = torch.zeros(2, cfg.spatial_material_channels, 16, 32)
    padded_action = torch.zeros(2, cfg.spatial_action_channels, 16, 32)
    mask = torch.zeros(2, 1, 16, 32)
    padded_material[:, :, :11, :19] = material
    padded_action[:, :, :11, :19] = action
    mask[:, :, :11, :19] = 1.0

    padded_mean, padded_logvar = member.forward_masked(padded_material, padded_action, mask)

    assert torch.allclose(exact_mean, padded_mean[:, :, :11, :19], atol=1e-6, rtol=1e-5)
    assert torch.allclose(exact_logvar, padded_logvar[:, :, :11, :19], atol=1e-6, rtol=1e-5)


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


def test_canonical_patch_buckets_batch_different_support_shapes() -> None:
    cfg = PainterConfig(
        canvas_size=32,
        spatial_grid_size=8,
        spatial_transition_mode="local_patch",
        local_patch_batch_bucket_cells=32,
        local_patch_margin_cells=0,
        local_patch_min_cells=1,
        spatial_ensemble_size=1,
    )
    policies = [
        Policy((StrokeAction(0.20, 0.20, 0.35, 0.25, 0.04, 0.6, 0.0), StrokeAction.stop_action())),
        Policy((StrokeAction(0.55, 0.55, 0.85, 0.80, 0.10, 0.6, 1.0), StrokeAction.stop_action())),
    ]
    belief = spatial_canvas_state(ArmPainterSim(cfg), cfg)
    dynamics = LocalSpatialDynamicsEnsemble(cfg)
    member = RecordingPatchMember()
    dynamics.members = nn.ModuleList([member])

    SpatialExpectedFreeEnergy(cfg, dynamics, TerminalCoveragePreference(cfg)).evaluate_batch(belief, policies)

    assert member.batch_sizes == [2]


def test_oversized_local_patches_use_sequential_policy_batches() -> None:
    cfg = PainterConfig(
        canvas_size=24,
        spatial_grid_size=8,
        spatial_transition_mode="local_patch",
        local_patch_sequential_cell_limit=1,
        local_patch_margin_cells=1,
        local_patch_min_cells=4,
        spatial_ensemble_size=1,
    )
    policies = [
        Policy((StrokeAction(0.2, 0.2, 0.7, 0.3, 0.08, 0.6, tone), StrokeAction.stop_action()))
        for tone in (0.0, 1.0)
    ]
    belief = spatial_canvas_state(ArmPainterSim(cfg), cfg)
    dynamics = LocalSpatialDynamicsEnsemble(cfg)
    member = RecordingPatchMember()
    dynamics.members = nn.ModuleList([member])

    components = SpatialExpectedFreeEnergy(cfg, dynamics, TerminalCoveragePreference(cfg)).evaluate_batch(
        belief,
        policies,
    )

    assert member.batch_sizes == [1, 1]
    assert [component.sequential_patch_steps for component in components] == [1, 1]


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


def test_conditioned_motor_transition_remains_sparse_and_keeps_efe_terms_separate() -> None:
    cfg = PainterConfig(
        canvas_size=32,
        spatial_grid_size=8,
        spatial_transition_mode="local_patch",
        local_patch_margin_cells=2,
        local_patch_min_cells=4,
        spatial_ensemble_size=2,
    )
    belief = spatial_canvas_state(ArmPainterSim(cfg), cfg)
    action = StrokeAction(0.25, 0.35, 0.55, 0.45, 0.06, 0.6, 1.0)
    policy = Policy((action, StrokeAction.stop_action()))
    material = torch.tensor(max(belief.pyramid, key=lambda level: level.grid_size).material)
    raster = torch.tensor(rasterize_stroke_action(action, cfg.canvas_size, config=cfg))
    support = raster[0] > 0.0
    next_material = material.clone()
    next_material[0, support] += 0.01
    next_material[1, support] += 0.005
    next_material[2, support] += 0.006
    next_material = project_material_support(
        material.unsqueeze(0),
        next_material.unsqueeze(0),
        cfg.thickness_scale,
        cfg.canvas_ground_tone,
    )[0]
    material_delta = torch.zeros_like(next_material)
    material_delta[:, support] = next_material[:, support] - material[:, support]
    next_material[0, 0, 0] += 0.001
    dynamics = LocalSpatialDynamicsEnsemble(cfg)
    dynamics.members = nn.ModuleList([DivergentPatchMember(0.0), DivergentPatchMember(0.0)])
    efe = SpatialExpectedFreeEnergy(cfg, dynamics, TerminalCoveragePreference(cfg))

    components = efe.evaluate_with_first_transition(
        belief,
        policy,
        next_material,
        torch.full_like(next_material, 1e-5),
        next_material_delta=material_delta,
        execution_uncertainty=0.2,
        contact_loss_probability=0.1,
        motor_overshoot=0.05,
        motor_feasible=True,
        motor_risk=0.4,
        motor_ambiguity=0.2,
        motor_epistemic_value=0.1,
        motor_efe_approximation="test motor likelihood",
    )

    expected_total = (
        components.terminal_risk
        + components.ambiguity
        + components.transition_risk
        + components.transition_ambiguity
        + components.composition_risk
        + components.motor_risk
        + components.motor_ambiguity
        - components.motor_epistemic_value
    )
    assert components.execution_forecast_used
    assert components.rollout_mode == "local_patch"
    assert components.local_transition_steps == 1
    expected_bounds = local_patch_bounds_for_raster(raster.numpy(), cfg.canvas_size, cfg)
    assert expected_bounds is not None
    assert np.isclose(components.active_patch_area_fraction, expected_bounds.area_fraction)
    assert np.isclose(components.motor_risk, 0.4)
    assert np.isclose(components.motor_ambiguity, 0.2)
    assert np.isclose(components.motor_epistemic_value, 0.1)
    assert components.motor_efe_approximation == "test motor likelihood"
    assert np.isclose(components.total, expected_total)

    stopped = efe.evaluate_with_first_transition(
        belief,
        Policy((StrokeAction.stop_action(),)),
        next_material,
        torch.full_like(next_material, 1e-5),
        execution_uncertainty=0.2,
        contact_loss_probability=0.1,
        motor_overshoot=0.05,
        motor_feasible=True,
        motor_risk=0.4,
        motor_ambiguity=0.2,
        motor_epistemic_value=0.1,
    )
    assert not stopped.execution_forecast_used
    assert stopped.local_transition_steps == 0
    assert stopped.motor_risk == 0.0


def test_batched_conditioned_transitions_match_individual_motor_rescoring() -> None:
    cfg = PainterConfig(
        canvas_size=32,
        spatial_grid_size=8,
        spatial_transition_mode="local_patch",
        local_patch_batch_bucket_cells=32,
        local_patch_margin_cells=1,
        local_patch_min_cells=4,
        spatial_ensemble_size=2,
    )
    belief = spatial_canvas_state(ArmPainterSim(cfg), cfg)
    material = torch.tensor(max(belief.pyramid, key=lambda level: level.grid_size).material)
    first_action = StrokeAction(0.20, 0.30, 0.55, 0.40, 0.07, 0.6, 1.0)
    policies = [
        Policy((first_action, StrokeAction(0.45, 0.45, 0.70, 0.55, 0.06, 0.5, tone), StrokeAction.stop_action()))
        for tone in (0.0, 1.0)
    ]
    first_transitions = []
    for index in range(2):
        raster = torch.tensor(rasterize_stroke_action(first_action, cfg.canvas_size, config=cfg))
        support = raster[0] > 0.0
        next_material = material.clone()
        next_material[0, support] += 0.008 + 0.002 * index
        next_material[1, support] += 0.004
        next_material = project_material_support(
            material.unsqueeze(0),
            next_material.unsqueeze(0),
            cfg.thickness_scale,
            cfg.canvas_ground_tone,
        )[0]
        delta = torch.zeros_like(material)
        delta[:, support] = next_material[:, support] - material[:, support]
        first_transitions.append((next_material, torch.full_like(material, 1e-5), delta))
    dynamics = LocalSpatialDynamicsEnsemble(cfg)
    dynamics.members = nn.ModuleList([DivergentPatchMember(0.0), DivergentPatchMember(0.02)])
    efe = SpatialExpectedFreeEnergy(cfg, dynamics, TerminalCoveragePreference(cfg))
    diagnostics = {
        "execution_uncertainties": [0.1, 0.2],
        "contact_loss_probabilities": [0.03, 0.07],
        "motor_overshoots": [0.01, 0.02],
        "motor_feasibilities": [True, False],
        "motor_risks": [0.2, 0.4],
        "motor_ambiguities": [0.1, 0.15],
        "motor_epistemic_values": [0.02, 0.05],
        "motor_efe_approximations": ["motor-0", "motor-1"],
    }

    batched = efe.evaluate_batch_with_first_transitions(
        belief,
        policies,
        first_transitions,
        **diagnostics,
    )
    singles = [
        efe.evaluate_with_first_transition(
            belief,
            policies[index],
            first_transitions[index][0],
            first_transitions[index][1],
            next_material_delta=first_transitions[index][2],
            execution_uncertainty=diagnostics["execution_uncertainties"][index],
            contact_loss_probability=diagnostics["contact_loss_probabilities"][index],
            motor_overshoot=diagnostics["motor_overshoots"][index],
            motor_feasible=diagnostics["motor_feasibilities"][index],
            motor_risk=diagnostics["motor_risks"][index],
            motor_ambiguity=diagnostics["motor_ambiguities"][index],
            motor_epistemic_value=diagnostics["motor_epistemic_values"][index],
            motor_efe_approximation=diagnostics["motor_efe_approximations"][index],
        )
        for index in range(2)
    ]

    for batch_component, single_component in zip(batched, singles):
        assert np.isclose(batch_component.total, single_component.total)
        assert np.isclose(batch_component.terminal_coverage_mean, single_component.terminal_coverage_mean)
        assert np.isclose(batch_component.transition_risk, single_component.transition_risk)
        assert np.isclose(batch_component.transition_ambiguity, single_component.transition_ambiguity)
        assert batch_component.execution_uncertainty == single_component.execution_uncertainty
        assert batch_component.motor_feasible == single_component.motor_feasible
        assert batch_component.motor_efe_approximation == single_component.motor_efe_approximation
