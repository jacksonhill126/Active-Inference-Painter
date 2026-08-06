from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import torch

from .config import PainterConfig
from .env import StrokeAction
from .inference import VFEComponents
from .local_spatial import local_patch_bounds_for_raster, pixel_logvar_from_state, pixel_material_from_state
from .models import LocalSpatialDynamicsEnsemble
from .policies import MotorPrimitiveLatent
from .spatial_state import (
    SpatialCanvasState,
    independent_material_channel_count,
    rasterize_stroke_action,
    spatial_state_from_pixel_posterior,
)


SPATIAL_VARIATIONAL_INFERENCE_VERSION = "spatial-variational-state-v0"
SPATIAL_TRANSITION_PRIOR_VERSION = "spatial-action-transition-prior-v0"
SPATIAL_TRANSITION_PRIOR_CALIBRATION_STATUS = (
    "provisional_simulation_trained_not_hardware_calibrated"
)


class SpatialTransitionModel(Protocol):
    def predictive_moments(
        self,
        material: torch.Tensor,
        action_raster: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


def spatial_observation_variance(material: np.ndarray, config: PainterConfig) -> np.ndarray:
    thickness = np.clip(material[0], 0.0, None)
    wetness = np.clip(material[1], 0.0, None)
    pigment = np.clip(material[2], 0.0, None)
    smear = np.clip(0.5 * thickness + 0.75 * wetness + 0.35 * pigment, 0.0, 2.0)
    scale_values = np.asarray([0.7, 0.9, 0.8, 0.6, 0.7, 0.5], dtype=np.float32)[: material.shape[0]]
    std = config.base_observation_std + config.smear_observation_std * smear[None, ...] * scale_values[:, None, None]
    return np.clip(std * std, 1e-12, 1e6).astype(np.float32)


class SpatialVariationalStateEstimator:
    """Diagonal pixel posterior q(s_t) from transition and observation densities."""

    def __init__(self, config: PainterConfig, device: torch.device) -> None:
        self.cfg = config
        self.device = device
        self.last_vfe: VFEComponents | None = None

    def initialize(self, observation: SpatialCanvasState) -> SpatialCanvasState:
        observed = pixel_material_from_state(observation)
        observation_variance = spatial_observation_variance(observed, self.cfg)
        posterior = spatial_state_from_pixel_posterior(
            observed,
            observation_variance,
            self.cfg,
            posterior_revision=observation.posterior_revision + 1,
            inference_model_id=(
                f"{SPATIAL_VARIATIONAL_INFERENCE_VERSION}:"
                f"{observation.inference_model_id}"
            ),
            calibration_status=observation.calibration_status,
        )
        independent_channels = independent_material_channel_count(observed.shape[0])
        negative_log_likelihood = _expected_negative_log_likelihood(
            observed[:independent_channels],
            observation_variance[:independent_channels],
            observed[:independent_channels],
            observation_variance[:independent_channels],
        )
        self.last_vfe = VFEComponents(
            total=negative_log_likelihood,
            complexity=0.0,
            negative_log_likelihood=negative_log_likelihood,
            expected_log_likelihood=-negative_log_likelihood,
            units="nats_per_independent_cell_channel",
            approximation=(
                "Initial spatial posterior is anchored to the first oracle material observation; "
                "no transition prior is available, and deterministic contrast/coverage channels "
                "do not contribute separate likelihood evidence."
            ),
        )
        return posterior

    @torch.no_grad()
    def predict(
        self,
        previous: SpatialCanvasState,
        action: StrokeAction,
        dynamics: SpatialTransitionModel,
        motor_primitive: MotorPrimitiveLatent | None = None,
    ) -> SpatialCanvasState:
        """Propagate q(s_t) through p_theta(s_t+1 | s_t, a_t, m_t).

        This is a transition prior, not an observation update, so it does not
        create or overwrite a VFE record. The next delivered camera likelihood
        is responsible for the posterior update and its VFE decomposition.
        """

        prior_mean, prior_variance = self._transition_moments(
            previous,
            action,
            dynamics,
            motor_primitive,
        )
        return spatial_state_from_pixel_posterior(
            prior_mean,
            prior_variance,
            self.cfg,
            posterior_revision=previous.posterior_revision + 1,
            inference_model_id=SPATIAL_TRANSITION_PRIOR_VERSION,
            calibration_status=SPATIAL_TRANSITION_PRIOR_CALIBRATION_STATUS,
        )

    @torch.no_grad()
    def infer(
        self,
        previous: SpatialCanvasState,
        action: StrokeAction,
        observation: SpatialCanvasState,
        dynamics: SpatialTransitionModel,
        motor_primitive: MotorPrimitiveLatent | None = None,
    ) -> SpatialCanvasState:
        prior_mean, prior_variance = self._transition_moments(
            previous,
            action,
            dynamics,
            motor_primitive,
        )

        observed = pixel_material_from_state(observation)
        observation_variance = spatial_observation_variance(observed, self.cfg)
        prior_variance = np.clip(prior_variance, 1e-12, 1e6)
        independent_channels = independent_material_channel_count(observed.shape[0])
        independent = slice(0, independent_channels)
        posterior_mean = prior_mean.copy()
        posterior_variance = prior_variance.copy()
        posterior_precision = (
            1.0 / prior_variance[independent] + 1.0 / observation_variance[independent]
        )
        posterior_variance[independent] = 1.0 / posterior_precision
        posterior_mean[independent] = posterior_variance[independent] * (
            prior_mean[independent] / prior_variance[independent]
            + observed[independent] / observation_variance[independent]
        )

        complexity = _diagonal_gaussian_kl(
            posterior_mean[independent],
            posterior_variance[independent],
            prior_mean[independent],
            prior_variance[independent],
        )
        negative_log_likelihood = _expected_negative_log_likelihood(
            posterior_mean[independent],
            posterior_variance[independent],
            observed[independent],
            observation_variance[independent],
        )
        self.last_vfe = VFEComponents(
            total=complexity + negative_log_likelihood,
            complexity=complexity,
            negative_log_likelihood=negative_log_likelihood,
            expected_log_likelihood=-negative_log_likelihood,
            units="nats_per_independent_cell_channel",
            approximation=(
                "Diagonal Gaussian pixel posterior; transition moments are evaluated at the previous posterior mean, "
                "only primary material factors enter VFE, and deterministic contrast/coverage channels are projected "
                "after Gaussian fusion with provisional propagated variance."
            ),
        )
        return spatial_state_from_pixel_posterior(
            posterior_mean,
            posterior_variance,
            self.cfg,
            posterior_revision=previous.posterior_revision + 1,
            inference_model_id=SPATIAL_VARIATIONAL_INFERENCE_VERSION,
            calibration_status=observation.calibration_status,
        )

    def _transition_moments(
        self,
        previous: SpatialCanvasState,
        action: StrokeAction,
        dynamics: SpatialTransitionModel,
        motor_primitive: MotorPrimitiveLatent | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        current_mean = pixel_material_from_state(previous)
        current_variance = np.exp(
            np.clip(pixel_logvar_from_state(previous, self.cfg), -30.0, 20.0)
        ).astype(np.float32)
        prior_mean = current_mean.copy()
        identity_variance = float(
            np.exp(np.clip(self.cfg.local_identity_logvar, -30.0, 20.0))
        )
        prior_variance = current_variance + identity_variance

        raster = rasterize_stroke_action(
            action,
            current_mean.shape[-1],
            motor_primitive=motor_primitive,
            config=self.cfg,
        )
        if isinstance(dynamics, LocalSpatialDynamicsEnsemble):
            bounds = local_patch_bounds_for_raster(
                raster,
                current_mean.shape[-1],
                self.cfg,
            )
            if bounds is not None:
                row_slice, col_slice = bounds.slices()
                material_t = torch.tensor(
                    current_mean[:, row_slice, col_slice],
                    device=self.device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                action_t = torch.tensor(
                    raster[:, row_slice, col_slice],
                    device=self.device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                mean_t, aleatoric_t, epistemic_t = dynamics.predictive_moments(
                    material_t,
                    action_t,
                )
                prior_mean[:, row_slice, col_slice] = mean_t[0].cpu().numpy()
                prior_variance[:, row_slice, col_slice] = (
                    aleatoric_t[0] + epistemic_t[0]
                ).cpu().numpy() + current_variance[:, row_slice, col_slice]
        else:
            material_t = torch.tensor(
                current_mean,
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            action_t = torch.tensor(
                raster,
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            mean_t, aleatoric_t, epistemic_t = dynamics.predictive_moments(
                material_t,
                action_t,
            )
            prior_mean = mean_t[0].cpu().numpy()
            prior_variance = (
                aleatoric_t[0] + epistemic_t[0]
            ).cpu().numpy() + current_variance
        return prior_mean, np.clip(prior_variance, 1e-12, 1e6)


def _diagonal_gaussian_kl(
    posterior_mean: np.ndarray,
    posterior_variance: np.ndarray,
    prior_mean: np.ndarray,
    prior_variance: np.ndarray,
) -> float:
    value = 0.5 * (
        np.log(prior_variance / posterior_variance)
        + (posterior_variance + (posterior_mean - prior_mean) ** 2) / prior_variance
        - 1.0
    )
    return float(np.mean(value, dtype=np.float64))


def _expected_negative_log_likelihood(
    posterior_mean: np.ndarray,
    posterior_variance: np.ndarray,
    observation: np.ndarray,
    observation_variance: np.ndarray,
) -> float:
    value = 0.5 * (
        np.log(2.0 * math.pi * observation_variance)
        + ((observation - posterior_mean) ** 2 + posterior_variance) / observation_variance
    )
    return float(np.mean(value, dtype=np.float64))
