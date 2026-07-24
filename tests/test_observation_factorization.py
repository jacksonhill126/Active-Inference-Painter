import math

import numpy as np
import torch

from active_painter.arm_sim import ArmPainterSim
from active_painter.config import PainterConfig
from active_painter.env import StrokeAction
from active_painter.models import LocalSpatialDynamicsEnsemble, SpatialDynamicsEnsemble
from active_painter.preferences import TerminalCoveragePreference
from active_painter.spatial_efe import SpatialExpectedFreeEnergy
from active_painter.spatial_inference import (
    SpatialVariationalStateEstimator,
    _diagonal_gaussian_kl,
    _expected_negative_log_likelihood,
)
from active_painter.spatial_state import (
    DERIVED_MATERIAL_CHANNELS,
    INDEPENDENT_MATERIAL_CHANNELS,
    MaterialPyramidLevel,
    SpatialCanvasState,
    spatial_canvas_state,
)


class IdentitySpatialDynamics:
    def predictive_moments(
        self,
        material: torch.Tensor,
        action_raster: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        variance = torch.full_like(material, 1e-6)
        return material, variance, torch.zeros_like(material)


def _override_derived_observation(
    state: SpatialCanvasState,
    contrast: float,
    coverage: float,
) -> SpatialCanvasState:
    def override(material: np.ndarray) -> np.ndarray:
        changed = material.copy()
        changed[4] = contrast
        changed[5] = coverage
        return changed

    return SpatialCanvasState(
        material=override(state.material),
        logvar=state.logvar.copy(),
        pyramid=tuple(
            MaterialPyramidLevel(level.name, level.grid_size, override(level.material))
            for level in state.pyramid
        ),
        pixel_logvar=None if state.pixel_logvar is None else state.pixel_logvar.copy(),
    )


def _efe(config: PainterConfig) -> SpatialExpectedFreeEnergy:
    return SpatialExpectedFreeEnergy(
        config,
        IdentitySpatialDynamics(),
        TerminalCoveragePreference(config),
    )


def test_material_factorization_declares_primary_and_deterministic_channels() -> None:
    assert INDEPENDENT_MATERIAL_CHANNELS == (
        "thickness",
        "wetness",
        "black_mass",
        "surface_tone",
    )
    assert DERIVED_MATERIAL_CHANNELS == ("ground_contrast", "material_coverage")


def test_spatial_posterior_does_not_double_count_deterministic_observation_channels() -> None:
    cfg = PainterConfig(canvas_size=8, spatial_grid_size=4, local_identity_logvar=-10.0)
    observation = spatial_canvas_state(ArmPainterSim(cfg), cfg)
    contradictory = _override_derived_observation(observation, contrast=0.97, coverage=1.0)
    action = StrokeAction.stop_action()

    reference_estimator = SpatialVariationalStateEstimator(cfg, torch.device("cpu"))
    reference_prior = reference_estimator.initialize(observation)
    reference = reference_estimator.infer(
        reference_prior,
        action,
        observation,
        IdentitySpatialDynamics(),
    )

    changed_estimator = SpatialVariationalStateEstimator(cfg, torch.device("cpu"))
    changed_prior = changed_estimator.initialize(observation)
    changed = changed_estimator.infer(
        changed_prior,
        action,
        contradictory,
        IdentitySpatialDynamics(),
    )

    assert np.array_equal(reference.material, changed.material)
    assert np.array_equal(reference.pixel_logvar, changed.pixel_logvar)
    assert reference_estimator.last_vfe is not None
    assert changed_estimator.last_vfe is not None
    assert reference_estimator.last_vfe.total == changed_estimator.last_vfe.total


def test_dense_transition_likelihood_ignores_deterministic_target_channels() -> None:
    torch.manual_seed(3)
    cfg = PainterConfig(
        spatial_grid_size=8,
        spatial_hidden_channels=8,
        spatial_residual_blocks=1,
        spatial_ensemble_size=2,
        ensemble_bootstrap_probability=1.0,
    )
    model = SpatialDynamicsEnsemble(cfg)
    material = torch.rand(2, cfg.spatial_material_channels, 8, 8)
    action = torch.rand(2, cfg.spatial_action_channels, 8, 8)
    target = torch.rand_like(material)
    contradictory = target.clone()
    contradictory[:, 4] = 1000.0
    contradictory[:, 5] = -1000.0

    assert torch.equal(model.nll(material, action, target), model.nll(material, action, contradictory))


def test_local_transition_likelihood_ignores_deterministic_target_channels() -> None:
    torch.manual_seed(4)
    cfg = PainterConfig(
        spatial_hidden_channels=8,
        spatial_residual_blocks=1,
        spatial_ensemble_size=2,
        ensemble_bootstrap_probability=1.0,
    )
    model = LocalSpatialDynamicsEnsemble(cfg)
    material = torch.rand(2, cfg.spatial_material_channels, 7, 9)
    action = torch.rand(2, cfg.spatial_action_channels, 7, 9)
    target = torch.rand_like(material)
    contradictory = target.clone()
    contradictory[:, 4] = 1000.0
    contradictory[:, 5] = -1000.0
    mask = torch.ones(2, 1, 7, 9)

    reference = model.per_sample_nll(material, action, target, mask)
    changed = model.per_sample_nll(material, action, contradictory, mask)

    assert torch.equal(reference, changed)


def test_transition_density_assigns_no_independent_variance_to_derived_channels() -> None:
    cfg = PainterConfig(
        spatial_grid_size=8,
        spatial_hidden_channels=8,
        spatial_residual_blocks=1,
        spatial_ensemble_size=2,
    )
    model = SpatialDynamicsEnsemble(cfg)
    material = torch.zeros(1, cfg.spatial_material_channels, 8, 8)
    action = torch.zeros(1, cfg.spatial_action_channels, 8, 8)

    _, logvars = model(material, action)

    assert torch.all(logvars[:, :, 4:] == -20.0)


def test_efe_uncertainty_terms_ignore_deterministic_material_channels() -> None:
    cfg = PainterConfig()
    evaluator = _efe(cfg)
    mean = torch.rand(2, cfg.spatial_material_channels, 5, 7)
    variance = torch.full_like(mean, 2e-4)
    changed_mean = mean.clone()
    changed_variance = variance.clone()
    changed_mean[:, 4:] = 500.0
    changed_variance[:, 4:] = 500.0

    assert torch.equal(
        evaluator._observation_ambiguity(mean),
        evaluator._observation_ambiguity(changed_mean),
    )
    assert torch.equal(
        evaluator._material_entropy_per_cell_channel(variance),
        evaluator._material_entropy_per_cell_channel(changed_variance),
    )


def test_spatial_vfe_and_entropy_normalization_are_invariant_to_repeated_cells() -> None:
    cfg = PainterConfig()
    evaluator = _efe(cfg)
    posterior_mean = np.asarray([[[0.2]], [[0.3]], [[0.4]], [[0.5]]], dtype=np.float32)
    posterior_variance = np.asarray([[[0.08]], [[0.09]], [[0.1]], [[0.11]]], dtype=np.float32)
    prior_mean = np.asarray([[[0.1]], [[0.2]], [[0.3]], [[0.4]]], dtype=np.float32)
    prior_variance = np.asarray([[[0.2]], [[0.2]], [[0.2]], [[0.2]]], dtype=np.float32)
    observation = np.asarray([[[0.25]], [[0.35]], [[0.45]], [[0.55]]], dtype=np.float32)
    observation_variance = np.asarray([[[0.03]], [[0.04]], [[0.05]], [[0.06]]], dtype=np.float32)

    def tile(value: np.ndarray) -> np.ndarray:
        return np.tile(value, (1, 6, 9))

    assert _diagonal_gaussian_kl(
        posterior_mean,
        posterior_variance,
        prior_mean,
        prior_variance,
    ) == _diagonal_gaussian_kl(
        tile(posterior_mean),
        tile(posterior_variance),
        tile(prior_mean),
        tile(prior_variance),
    )
    assert _expected_negative_log_likelihood(
        posterior_mean,
        posterior_variance,
        observation,
        observation_variance,
    ) == _expected_negative_log_likelihood(
        tile(posterior_mean),
        tile(posterior_variance),
        tile(observation),
        tile(observation_variance),
    )

    base_entropy = evaluator._material_entropy_per_cell_channel(torch.tensor(posterior_variance))
    tiled_entropy = evaluator._material_entropy_per_cell_channel(torch.tensor(tile(posterior_variance)))
    assert torch.allclose(base_entropy, tiled_entropy, atol=1e-7, rtol=1e-7)


def test_gaussian_vfe_coordinate_scaling_changes_only_the_nll_density_constant() -> None:
    posterior_mean = np.asarray([0.15, 0.4, 0.7], dtype=np.float64)
    posterior_variance = np.asarray([0.03, 0.05, 0.08], dtype=np.float64)
    prior_mean = np.asarray([0.1, 0.5, 0.6], dtype=np.float64)
    prior_variance = np.asarray([0.2, 0.2, 0.2], dtype=np.float64)
    observation = np.asarray([0.2, 0.45, 0.75], dtype=np.float64)
    observation_variance = np.asarray([0.04, 0.06, 0.09], dtype=np.float64)
    scale = 37.0

    original_kl = _diagonal_gaussian_kl(
        posterior_mean,
        posterior_variance,
        prior_mean,
        prior_variance,
    )
    scaled_kl = _diagonal_gaussian_kl(
        scale * posterior_mean,
        scale**2 * posterior_variance,
        scale * prior_mean,
        scale**2 * prior_variance,
    )
    original_nll = _expected_negative_log_likelihood(
        posterior_mean,
        posterior_variance,
        observation,
        observation_variance,
    )
    scaled_nll = _expected_negative_log_likelihood(
        scale * posterior_mean,
        scale**2 * posterior_variance,
        scale * observation,
        scale**2 * observation_variance,
    )

    assert math.isclose(original_kl, scaled_kl, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(scaled_nll - original_nll, math.log(scale), rel_tol=1e-12, abs_tol=1e-12)
