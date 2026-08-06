from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .camera_geometry import ROBOT_MODEL_PATH, CameraRigSpec, CameraSpec, load_camera_rig
from .camera_observation import (
    CAMERA_OBSERVATION_INTERFACE_VERSION,
    FOVEA_CANVAS_PRODUCT,
    GLOBAL_CANVAS_PRODUCT,
    CameraFrame,
    CameraObservationBundle,
)
from .config import PainterConfig
from .local_spatial import pixel_logvar_from_state, pixel_material_from_state
from .spatial_state import SpatialCanvasState, spatial_state_from_pixel_posterior


CAMERA_SPATIAL_LIKELIHOOD_VERSION = "camera-spatial-likelihood-v0"
_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class CameraVFEFactor:
    """One conditionally independent camera-exposure likelihood factor."""

    camera_name: str
    sequence: int
    capture_time_s: float
    product_ids: tuple[str, ...]
    observed_cell_count: int
    mean_inlier_probability: float
    state_complexity: float
    occlusion_complexity: float
    negative_log_likelihood: float

    @property
    def complexity(self) -> float:
        return self.state_complexity + self.occlusion_complexity

    @property
    def total(self) -> float:
        return self.complexity + self.negative_log_likelihood


@dataclass(frozen=True, slots=True)
class CameraVFEComponents:
    """Separately logged camera contribution to variational free energy."""

    total: float
    complexity: float
    negative_log_likelihood: float
    expected_log_likelihood: float
    factors: tuple[CameraVFEFactor, ...]
    assimilated_products: tuple[str, ...]
    ignored_products: tuple[str, ...]
    units: str = "nats_per_observed_canvas_cell"
    approximation: str = (
        "Per-cell first-order Gaussian update of the nonlinear superficial-"
        "grayscale likelihood; diagonal material covariance and mean-field "
        "camera-occlusion responsibilities. Global and foveal products from "
        "one native exposure are mosaicked before one likelihood update so "
        "correlated pixels are not counted twice."
    )
    version: str = CAMERA_SPATIAL_LIKELIHOOD_VERSION


class CameraSpatialLikelihood:
    """Infer a spatial material posterior from permitted camera products.

    The likelihood is

        p(o | s, z=inlier) = Normal(g(s), sigma_camera^2)

    where ``g(s)`` is the superficial grayscale predicted from latent paint
    thickness and surface tone. A broad, state-independent outlier component
    represents occlusion and other gross image mismatch. Its responsibility
    is inferred from pixels; no segmentation, exact visibility, contact,
    material array, or simulator pose enters this class.

    Wetness and black-pigment mass have zero direct image Jacobian and remain
    transition-prior beliefs. White paint whose visible tone equals the ground
    also supplies no thickness evidence, as required by the present grayscale
    observability assumption.
    """

    def __init__(
        self,
        config: PainterConfig,
        rig: CameraRigSpec | None = None,
        *,
        model_path: Path | str = ROBOT_MODEL_PATH,
    ) -> None:
        self.cfg = config
        self.rig = rig or load_camera_rig(model_path)
        self.last_vfe: CameraVFEComponents | None = None

    def infer(
        self,
        prior: SpatialCanvasState,
        observation: CameraObservationBundle,
    ) -> SpatialCanvasState:
        if observation.interface_version != CAMERA_OBSERVATION_INTERFACE_VERSION:
            raise ValueError(
                "camera bundle interface does not match the spatial likelihood"
            )
        mean = pixel_material_from_state(prior).astype(np.float64, copy=True)
        variance = np.exp(
            np.clip(pixel_logvar_from_state(prior, self.cfg), -30.0, 20.0)
        ).astype(np.float64)
        if mean.shape[0] < 4:
            raise ValueError(
                "camera likelihood requires thickness and surface_tone channels"
            )
        if mean.shape != variance.shape or mean.shape[-2] != mean.shape[-1]:
            raise ValueError("camera likelihood requires square aligned material fields")

        eligible = tuple(
            frame
            for frame in observation.frames
            if frame.product_kind in {GLOBAL_CANVAS_PRODUCT, FOVEA_CANVAS_PRODUCT}
        )
        ignored = tuple(
            _product_key(frame)
            for frame in observation.frames
            if frame.product_kind not in {GLOBAL_CANVAS_PRODUCT, FOVEA_CANVAS_PRODUCT}
        )
        exposures: dict[tuple[str, int], list[CameraFrame]] = {}
        for frame in eligible:
            exposures.setdefault((frame.camera_name, frame.sequence), []).append(frame)

        factors: list[CameraVFEFactor] = []
        assimilated: list[str] = []
        for camera_name, sequence in sorted(
            exposures,
            key=lambda key: (
                min(frame.capture_time_s for frame in exposures[key]),
                key[0],
                key[1],
            ),
        ):
            frames = tuple(exposures[(camera_name, sequence)])
            camera = self.rig.camera(camera_name)
            self._validate_exposure(camera, frames)
            pixels, valid, product_ids = _registered_exposure_grid(
                frames,
                mean.shape[-1],
            )
            if not np.any(valid):
                continue
            mean, variance, factor = self._update_exposure(
                mean,
                variance,
                pixels,
                valid,
                camera,
                sequence=sequence,
                capture_time_s=frames[0].capture_time_s,
                product_ids=product_ids,
            )
            factors.append(factor)
            assimilated.extend(
                f"{camera_name}:{sequence}:{product_id}"
                for product_id in product_ids
            )

        complexity = float(sum(factor.complexity for factor in factors))
        negative_log_likelihood = float(
            sum(factor.negative_log_likelihood for factor in factors)
        )
        self.last_vfe = CameraVFEComponents(
            total=complexity + negative_log_likelihood,
            complexity=complexity,
            negative_log_likelihood=negative_log_likelihood,
            expected_log_likelihood=-negative_log_likelihood,
            factors=tuple(factors),
            assimilated_products=tuple(assimilated),
            ignored_products=ignored,
        )
        if not factors:
            return prior
        return spatial_state_from_pixel_posterior(
            mean.astype(np.float32),
            np.clip(variance, 1e-12, 1e6).astype(np.float32),
            self.cfg,
            posterior_revision=prior.posterior_revision + 1,
            inference_model_id=(
                f"{CAMERA_SPATIAL_LIKELIHOOD_VERSION}:"
                f"{self.rig.likelihood_model}"
            ),
            calibration_status=self.rig.likelihood_status,
        )

    def _validate_exposure(
        self,
        camera: CameraSpec,
        frames: Sequence[CameraFrame],
    ) -> None:
        if camera.registration != "canvas_plane_homography":
            raise ValueError(
                f"camera {camera.name!r} is not registered to canvas UV"
            )
        if sum(frame.product_kind == GLOBAL_CANVAS_PRODUCT for frame in frames) > 1:
            raise ValueError("an exposure may contain at most one global product")
        for frame in frames:
            if frame.calibration_revision != self.rig.version:
                raise ValueError("camera frame calibration revision is incompatible")
            if frame.observation_model != self.rig.observation_model:
                raise ValueError("camera frame observation model is incompatible")
            if frame.registration != "canvas_plane_homography":
                raise ValueError("camera likelihood requires canvas-registered products")
            if frame.capture_time_s != frames[0].capture_time_s:
                raise ValueError("products in one exposure must share capture time")

    def _update_exposure(
        self,
        prior_mean: np.ndarray,
        prior_variance: np.ndarray,
        observation: np.ndarray,
        valid: np.ndarray,
        camera: CameraSpec,
        *,
        sequence: int,
        capture_time_s: float,
        product_ids: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray, CameraVFEFactor]:
        predicted, jacobian = self._predict_grayscale(prior_mean)
        measurement_variance = (
            camera.likelihood_model_error_std**2
            + camera.read_noise_std**2
            + camera.signal_noise_std**2 * np.clip(predicted, 0.0, 1.0)
        )
        projected_prior_variance = np.sum(
            np.square(jacobian) * prior_variance,
            axis=0,
        )
        predictive_variance = np.clip(
            measurement_variance + projected_prior_variance,
            _EPSILON,
            1e12,
        )
        residual = observation - predicted
        log_inlier = (
            math.log(camera.likelihood_inlier_probability)
            - 0.5
            * (
                np.log(2.0 * math.pi * predictive_variance)
                + np.square(residual) / predictive_variance
            )
        )
        outlier_variance = camera.likelihood_outlier_std**2
        log_outlier = (
            math.log(1.0 - camera.likelihood_inlier_probability)
            - 0.5
            * (
                math.log(2.0 * math.pi * outlier_variance)
                + np.square(observation - 0.5) / outlier_variance
            )
        )
        inlier_probability = np.exp(
            log_inlier - np.logaddexp(log_inlier, log_outlier)
        )
        inlier_probability = np.where(valid, inlier_probability, 0.0)

        effective_variance = measurement_variance / np.clip(
            inlier_probability,
            1e-6,
            1.0,
        )
        innovation_variance = np.clip(
            effective_variance + projected_prior_variance,
            _EPSILON,
            1e12,
        )
        gain = prior_variance * jacobian / innovation_variance[None, ...]
        gain[:, ~valid] = 0.0
        posterior_mean = prior_mean + gain * residual[None, ...]
        posterior_variance = prior_variance * (
            1.0 - gain * jacobian
        )
        posterior_variance = np.clip(posterior_variance, 1e-12, 1e6)

        gaussian_kl = 0.5 * np.sum(
            np.log(prior_variance / posterior_variance)
            + (
                posterior_variance
                + np.square(posterior_mean - prior_mean)
            )
            / prior_variance
            - 1.0,
            axis=0,
        )
        pi = camera.likelihood_inlier_probability
        gamma = np.clip(inlier_probability, 1e-12, 1.0 - 1e-12)
        occlusion_kl = (
            gamma * np.log(gamma / pi)
            + (1.0 - gamma) * np.log((1.0 - gamma) / (1.0 - pi))
        )
        posterior_prediction = predicted + np.sum(
            jacobian * (posterior_mean - prior_mean),
            axis=0,
        )
        posterior_projected_variance = np.sum(
            np.square(jacobian) * posterior_variance,
            axis=0,
        )
        inlier_nll = 0.5 * (
            np.log(2.0 * math.pi * measurement_variance)
            + (
                np.square(observation - posterior_prediction)
                + posterior_projected_variance
            )
            / measurement_variance
        )
        outlier_nll = 0.5 * (
            math.log(2.0 * math.pi * outlier_variance)
            + np.square(observation - 0.5) / outlier_variance
        )
        expected_nll = gamma * inlier_nll + (1.0 - gamma) * outlier_nll
        state_complexity = float(np.mean(gaussian_kl[valid], dtype=np.float64))
        occlusion_complexity = float(
            np.mean(occlusion_kl[valid], dtype=np.float64)
        )
        negative_log_likelihood = float(
            np.mean(expected_nll[valid], dtype=np.float64)
        )
        return posterior_mean, posterior_variance, CameraVFEFactor(
            camera_name=camera.name,
            sequence=sequence,
            capture_time_s=capture_time_s,
            product_ids=product_ids,
            observed_cell_count=int(np.count_nonzero(valid)),
            mean_inlier_probability=float(
                np.mean(inlier_probability[valid], dtype=np.float64)
            ),
            state_complexity=state_complexity,
            occlusion_complexity=occlusion_complexity,
            negative_log_likelihood=negative_log_likelihood,
        )

    def _predict_grayscale(
        self,
        material_mean: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        thickness = np.clip(material_mean[0], 0.0, None)
        surface_tone = np.clip(material_mean[3], 0.0, 1.0)
        scale = max(self.cfg.thickness_scale, 1e-8)
        transmittance = np.exp(-thickness / scale)
        opacity = 1.0 - transmittance
        observed_tone = (
            transmittance * self.cfg.canvas_ground_tone
            + opacity * surface_tone
        )
        grayscale = np.clip(1.0 - observed_tone, 0.0, 1.0)
        jacobian = np.zeros_like(material_mean, dtype=np.float64)
        jacobian[0] = transmittance * (
            self.cfg.canvas_ground_tone - surface_tone
        ) / scale
        jacobian[3] = -opacity
        return grayscale, jacobian


def _product_key(frame: CameraFrame) -> str:
    return f"{frame.camera_name}:{frame.sequence}:{frame.product_id}"


def _registered_exposure_grid(
    frames: Sequence[CameraFrame],
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    centers = (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size
    uu, vv = np.meshgrid(centers, centers)
    pixels = np.zeros((grid_size, grid_size), dtype=np.float64)
    valid = np.zeros((grid_size, grid_size), dtype=np.bool_)
    product_ids: list[str] = []

    global_frames = sorted(
        (frame for frame in frames if frame.product_kind == GLOBAL_CANVAS_PRODUCT),
        key=lambda frame: frame.product_id,
    )
    if global_frames:
        frame = global_frames[0]
        pixels[:] = _bilinear_sample(frame.grayscale, uu, vv)
        valid[:] = _conservative_validity(frame.calibration_validity, uu, vv)
        product_ids.append(frame.product_id)

    foveae = sorted(
        (frame for frame in frames if frame.product_kind == FOVEA_CANVAS_PRODUCT),
        key=lambda frame: frame.product_id,
    )
    for frame in foveae:
        assert frame.center_canvas_uv is not None
        assert frame.span_canvas_uv is not None
        center_u, center_v = frame.center_canvas_uv
        span_u, span_v = frame.span_canvas_uv
        local_u = (uu - (center_u - span_u / 2.0)) / span_u
        local_v = (vv - (center_v - span_v / 2.0)) / span_v
        inside = (
            (local_u >= 0.0)
            & (local_u <= 1.0)
            & (local_v >= 0.0)
            & (local_v <= 1.0)
        )
        fovea_valid = inside & _conservative_validity(
            frame.calibration_validity,
            local_u,
            local_v,
        )
        sampled = _bilinear_sample(frame.grayscale, local_u, local_v)
        pixels[fovea_valid] = sampled[fovea_valid]
        valid[fovea_valid] = True
        product_ids.append(frame.product_id)
    return pixels, valid, tuple(product_ids)


def _bilinear_sample(
    image: np.ndarray,
    normalized_u: np.ndarray,
    normalized_v: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    x = np.clip(normalized_u, 0.0, 1.0) * (width - 1)
    y = np.clip(normalized_v, 0.0, 1.0) * (height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    return (
        image[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + image[y0, x1] * wx * (1.0 - wy)
        + image[y1, x0] * (1.0 - wx) * wy
        + image[y1, x1] * wx * wy
    )


def _conservative_validity(
    validity: np.ndarray,
    normalized_u: np.ndarray,
    normalized_v: np.ndarray,
) -> np.ndarray:
    sampled = _bilinear_sample(
        validity.astype(np.float32),
        normalized_u,
        normalized_v,
    )
    return sampled >= 1.0 - 1e-6
