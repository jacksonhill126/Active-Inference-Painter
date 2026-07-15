import numpy as np
import torch

from active_painter.arm_agent_driver import canvas_summary_state
from active_painter.arm_sim import ArmPainterSim
from active_painter.config import PainterConfig
from active_painter.env import StrokeAction
from active_painter.models import SpatialDynamicsEnsemble, SpatialTransitionMember
from active_painter.policies import Policy
from active_painter.spatial_inference import SpatialVariationalStateEstimator
from active_painter.preferences import TerminalCoveragePreference
from active_painter.spatial_efe import SpatialExpectedFreeEnergy
from active_painter.spatial_state import (
    ACTION_CHANNELS,
    MATERIAL_CHANNELS,
    coverage_from_thickness,
    material_grid_from_canvas,
    material_pyramid_from_canvas,
    rasterize_stroke_action,
    spatial_canvas_state,
)


class DeterministicFootprintDynamics:
    def __init__(self, deposit_scale: float = 0.012) -> None:
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
        delta[:, 1:2] = 0.65 * deposited
        delta[:, 2:3] = tone * deposited
        next_material = material + delta
        if next_material.shape[1] > 3:
            thickness = torch.clamp(next_material[:, 0], min=0.0)
            opacity = 1.0 - torch.exp(-thickness / 0.005)
            coverage = (thickness >= 0.0001).to(thickness.dtype)
            next_material[:, 3:4] = torch.where(deposited > 0.0, tone, material[:, 3:4])
        if next_material.shape[1] > 4:
            observed = (1.0 - opacity) * 0.34 + opacity * next_material[:, 3]
            next_material[:, 4] = torch.abs(observed - 0.34)
        if next_material.shape[1] > 5:
            next_material[:, 5] = coverage
        aleatoric = torch.full_like(next_material, 2e-5)
        epistemic = torch.zeros_like(next_material)
        return next_material, aleatoric, epistemic


def test_spatial_state_distinguishes_equal_summary_canvases_at_different_locations() -> None:
    cfg = PainterConfig(canvas_size=32, spatial_grid_size=16)
    left = ArmPainterSim(cfg)
    right = ArmPainterSim(cfg)
    left.canvas.thickness[4:12, 4:12] = 0.02
    right.canvas.thickness[20:28, 20:28] = 0.02

    assert np.allclose(canvas_summary_state(left), canvas_summary_state(right))

    left_state = spatial_canvas_state(left, cfg)
    right_state = spatial_canvas_state(right, cfg)

    assert left_state.material.shape == (len(MATERIAL_CHANNELS), 16, 16)
    assert right_state.material.shape == (len(MATERIAL_CHANNELS), 16, 16)
    assert not np.allclose(left_state.material, right_state.material)
    assert np.isclose(
        left_state.material_coverage_mean(cfg.paint_presence_threshold),
        right_state.material_coverage_mean(cfg.paint_presence_threshold),
    )


def test_spatial_material_coverage_is_derived_from_thickness_not_visible_tone() -> None:
    cfg = PainterConfig(canvas_size=32, spatial_grid_size=16)
    sim = ArmPainterSim(cfg)
    before = spatial_canvas_state(sim, cfg)

    sim.canvas.paint_at(
        np.asarray([0.0, sim.canvas.distance, 0.0]),
        pressure=0.8,
        tone=0.0,
        dt=0.2,
    )
    after = spatial_canvas_state(sim, cfg)

    assert after.material_coverage_mean(cfg.paint_presence_threshold) > before.material_coverage_mean(
        cfg.paint_presence_threshold
    )
    assert after.material[0].mean() > before.material[0].mean()
    assert after.material[2].mean() == before.material[2].mean()
    assert after.material[3].mean() == before.material[3].mean()
    assert after.material[4].mean() > before.material[4].mean()
    assert after.material[4].mean() > before.material[4].mean()


def test_spatial_posterior_combines_transition_prior_and_material_likelihood() -> None:
    cfg = PainterConfig(canvas_size=16, spatial_grid_size=8, local_identity_logvar=-10.0)
    sim = ArmPainterSim(cfg)
    initial_observation = spatial_canvas_state(sim, cfg)
    estimator = SpatialVariationalStateEstimator(cfg, torch.device("cpu"))
    prior = estimator.initialize(initial_observation)
    action = StrokeAction(0.25, 0.5, 0.75, 0.5, 0.1, 0.5, tone=1.0)
    sim.canvas.paint_at(
        np.asarray([0.0, sim.canvas.distance, 0.0]),
        pressure=0.7,
        tone=1.0,
        dt=1.0 / 60.0,
    )
    observation = spatial_canvas_state(sim, cfg)

    posterior = estimator.infer(prior, action, observation, DeterministicFootprintDynamics())

    assert posterior.pixel_logvar is not None
    assert posterior.pixel_logvar.shape == (cfg.spatial_material_channels, cfg.canvas_size, cfg.canvas_size)
    assert np.isfinite(posterior.pixel_logvar).all()
    assert estimator.last_vfe is not None
    assert np.isclose(
        estimator.last_vfe.total,
        estimator.last_vfe.complexity + estimator.last_vfe.negative_log_likelihood,
    )
    assert estimator.last_vfe.complexity >= 0.0


def test_spatial_state_includes_surface_tone_contrast_and_derived_material_coverage_fields() -> None:
    cfg = PainterConfig(canvas_size=32, spatial_grid_size=16)
    sim = ArmPainterSim(cfg)
    sim.canvas.paint_at(
        np.asarray([0.0, sim.canvas.distance, 0.0]),
        pressure=0.8,
        tone=1.0,
        dt=0.2,
    )

    state = spatial_canvas_state(sim, cfg)

    assert list(MATERIAL_CHANNELS) == [
        "thickness",
        "wetness",
        "black_mass",
        "surface_tone",
        "ground_contrast",
        "material_coverage",
    ]
    assert state.material.shape == (len(MATERIAL_CHANNELS), 16, 16)
    assert state.material[3].max() > 0.2
    assert state.material[4].max() > 0.0
    assert np.allclose(state.material[5], state.coverage(cfg.paint_presence_threshold))


def test_material_pyramid_includes_pixel_tile_and_planner_levels() -> None:
    cfg = PainterConfig(canvas_size=32, spatial_grid_size=8, material_pyramid_levels=(16, 8))
    sim = ArmPainterSim(cfg)
    sim.canvas.thickness[3:11, 5:17] = 0.02
    sim.canvas.wetness[3:11, 5:17] = 0.01
    sim.canvas.black_mass[3:11, 5:17] = 0.012

    state = spatial_canvas_state(sim, cfg)
    pyramid = material_pyramid_from_canvas(sim.canvas, cfg)
    coverage_mean = float(sim.canvas.coverage_field().mean())

    assert [level.grid_size for level in pyramid] == [32, 16, 8]
    assert [level.name for level in pyramid] == ["pixel", "tile_16", "planner"]
    assert [level.grid_size for level in state.pyramid] == [32, 16, 8]
    assert all(level.material.shape == (len(MATERIAL_CHANNELS), level.grid_size, level.grid_size) for level in pyramid)
    assert all(
        np.isclose(level.coverage(cfg.paint_presence_threshold).mean(), coverage_mean)
        for level in pyramid
    )


def test_coarse_material_coverage_preserves_pixel_coverage_not_blurred_thickness() -> None:
    cfg = PainterConfig(canvas_size=4, spatial_grid_size=1)
    sim = ArmPainterSim(cfg)
    sim.canvas.thickness[0, 0] = 0.03

    coarse = material_grid_from_canvas(sim.canvas, 1)
    pixel_coverage_mean = float(sim.canvas.coverage_field().mean())
    coverage_from_mean_thickness = float(
        coverage_from_thickness(coarse[0], cfg.paint_presence_threshold).item()
    )

    assert np.isclose(float(coarse[5, 0, 0]), pixel_coverage_mean)
    assert not np.isclose(float(coarse[5, 0, 0]), coverage_from_mean_thickness)


def test_spatial_downsample_preserves_material_means_for_divisible_grid() -> None:
    cfg = PainterConfig(canvas_size=32, spatial_grid_size=16)
    sim = ArmPainterSim(cfg)
    sim.canvas.thickness[:, :] = np.linspace(0.0, 0.03, 32 * 32, dtype=np.float32).reshape(32, 32)
    sim.canvas.wetness[:, :] = 0.5 * sim.canvas.thickness
    sim.canvas.black_mass[:, :] = 0.25 * sim.canvas.thickness

    state = spatial_canvas_state(sim, cfg)

    assert np.isclose(state.material[0].mean(), sim.canvas.thickness.mean())
    assert np.isclose(state.material[1].mean(), sim.canvas.wetness.mean())
    assert np.isclose(state.material[2].mean(), sim.canvas.black_mass.mean())
    assert np.allclose(state.material[5], state.coverage(cfg.paint_presence_threshold))


def test_stroke_action_rasterization_is_spatial_and_deterministic() -> None:
    action = StrokeAction(0.15, 0.25, 0.85, 0.75, 0.08, 0.7, 1.0)

    first = rasterize_stroke_action(action, 16)
    second = rasterize_stroke_action(action, 16)

    assert first.shape == (len(ACTION_CHANNELS), 16, 16)
    assert np.allclose(first, second)
    assert first[0, 4, 3] > first[0, 15, 0]
    assert first[1].max() > 0.5
    assert first[2].max() > 0.5
    assert np.allclose(first[3], action.width)
    assert np.allclose(first[4], action.amount)
    assert np.allclose(first[5], action.tone)


def test_spatial_dynamics_ensemble_predicts_material_field_density_shapes() -> None:
    cfg = PainterConfig(spatial_grid_size=16, spatial_ensemble_size=3, spatial_hidden_channels=16)
    model = SpatialDynamicsEnsemble(cfg)
    material = torch.zeros(2, cfg.spatial_material_channels, cfg.spatial_grid_size, cfg.spatial_grid_size)
    action = torch.zeros(2, cfg.spatial_action_channels, cfg.spatial_grid_size, cfg.spatial_grid_size)
    action[:, 0, 5:9, 5:9] = 1.0

    means, logvars = model(material, action)
    mean, aleatoric, epistemic = model.predictive_moments(material, action)
    loss = model.nll(material, action, torch.zeros_like(material))

    expected = (
        cfg.spatial_ensemble_size,
        2,
        cfg.spatial_material_channels,
        cfg.spatial_grid_size,
        cfg.spatial_grid_size,
    )
    assert means.shape == expected
    assert logvars.shape == expected
    assert mean.shape == material.shape
    assert aleatoric.shape == material.shape
    assert epistemic.shape == material.shape
    assert torch.all(aleatoric > 0.0)
    assert torch.isfinite(loss)


def test_spatial_dynamics_projection_supports_backpropagation() -> None:
    cfg = PainterConfig(spatial_grid_size=8, spatial_ensemble_size=2, spatial_hidden_channels=8, spatial_residual_blocks=1)
    model = SpatialDynamicsEnsemble(cfg)
    material = torch.zeros(2, cfg.spatial_material_channels, cfg.spatial_grid_size, cfg.spatial_grid_size)
    action = torch.zeros(2, cfg.spatial_action_channels, cfg.spatial_grid_size, cfg.spatial_grid_size)
    action[:, 0, 2:6, 2:6] = 1.0
    next_material = torch.zeros_like(material)
    next_material[:, 0, 2:6, 2:6] = 0.01
    next_material[:, 1, 2:6, 2:6] = 0.004
    next_material[:, 2, 2:6, 2:6] = 0.006
    thickness = torch.clamp(next_material[:, 0], min=0.0)
    opacity = 1.0 - torch.exp(-thickness / cfg.thickness_scale)
    coverage = (thickness >= cfg.paint_presence_threshold).to(thickness.dtype)
    next_material[:, 3] = 0.6
    observed_tone = (1.0 - opacity) * cfg.canvas_ground_tone + opacity * next_material[:, 3]
    next_material[:, 4] = torch.abs(observed_tone - cfg.canvas_ground_tone)
    next_material[:, 5] = coverage

    loss = model.nll(material, action, next_material)
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_spatial_material_support_preserves_persistent_wetness_and_material() -> None:
    current = torch.tensor([[[[0.4]], [[0.3]], [[0.2]], [[0.0]], [[0.0]], [[0.0]]]])
    proposed = torch.tensor([[[[0.1]], [[0.05]], [[0.0]], [[0.7]], [[0.9]], [[0.2]]]])

    projected = SpatialTransitionMember._project_material_support(current, proposed)

    assert projected[0, 0, 0, 0] == current[0, 0, 0, 0]
    assert projected[0, 1, 0, 0] == current[0, 1, 0, 0]
    assert projected[0, 2, 0, 0] == current[0, 2, 0, 0]
    expected_coverage = torch.tensor(1.0, dtype=projected.dtype)
    assert torch.isclose(projected[0, 3, 0, 0], torch.tensor(0.7))
    assert torch.isclose(projected[0, 4, 0, 0], torch.tensor(abs(0.7 - 0.34), dtype=projected.dtype))
    assert torch.isclose(projected[0, 5, 0, 0], expected_coverage)


def test_spatial_efe_uses_realized_spatial_consequences_not_global_heuristics() -> None:
    cfg = PainterConfig(canvas_size=32, spatial_grid_size=16, target_coverage=0.45, terminal_concentration=80.0)
    sim = ArmPainterSim(cfg)
    sim.canvas.thickness[6:18, 2:14] = 0.04
    belief = spatial_canvas_state(sim, cfg, logvar_value=-12.0)
    efe = SpatialExpectedFreeEnergy(cfg, DeterministicFootprintDynamics(), TerminalCoveragePreference(cfg))

    already_painted = Policy((StrokeAction(0.12, 0.18, 0.36, 0.42, 0.16, 0.8, 1.0), StrokeAction.stop_action()))
    empty_region = Policy((StrokeAction(0.62, 0.58, 0.86, 0.82, 0.16, 0.8, 1.0), StrokeAction.stop_action()))

    painted_components = efe.evaluate(belief, already_painted)
    empty_components = efe.evaluate(belief, empty_region)

    assert empty_components.terminal_coverage_mean > painted_components.terminal_coverage_mean
    assert empty_components.terminal_risk < painted_components.terminal_risk
    assert np.isclose(
        empty_components.total,
        empty_components.terminal_risk
        + empty_components.ambiguity
        + empty_components.transition_risk
        + empty_components.transition_ambiguity,
    )
