from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import PainterConfig
from .env import StrokeAction
from .local_spatial import pixel_material_from_state
from .policies import PassageLatent
from .spatial_state import SpatialCanvasState


PASSAGE_PARAMETER_NAMES = (
    "center_x",
    "center_y",
    "direction",
    "length",
    "spacing",
    "width",
    "amount",
)


@dataclass(frozen=True, slots=True)
class PassageObservation:
    mean: np.ndarray
    variance: np.ndarray
    observed: np.ndarray
    black_probability: float
    tone_precision: float
    approximation: str


@dataclass(frozen=True, slots=True)
class PassageBelief:
    """Slow passage posterior q(z_passage) with a Bernoulli tone factor."""

    template: PassageLatent
    mean: np.ndarray
    variance: np.ndarray
    black_alpha: float
    black_beta: float
    update_count: int = 0
    approximation: str = (
        "diagonal Gaussian passage posterior plus beta-Bernoulli tone; "
        "pixel material deltas provide local mark likelihoods"
    )

    @classmethod
    def from_latent(cls, latent: PassageLatent, config: PainterConfig) -> "PassageBelief":
        mean = _latent_vector(latent)
        variance = np.asarray(
            [
                config.passage_belief_center_std**2,
                config.passage_belief_center_std**2,
                config.passage_belief_direction_std**2,
                config.passage_belief_geometry_std**2,
                config.passage_belief_geometry_std**2,
                config.passage_belief_geometry_std**2,
                config.passage_belief_geometry_std**2,
            ],
            dtype=np.float64,
        )
        if latent.tone >= 0.5:
            alpha, beta = 19.0, 1.0
        else:
            alpha, beta = 1.0, 19.0
        return cls(latent, mean, variance, alpha, beta)

    @property
    def black_probability(self) -> float:
        return float(self.black_alpha / max(1e-8, self.black_alpha + self.black_beta))

    def update(self, observation: PassageObservation, config: PainterConfig) -> "PassageBelief":
        transition_variance = float(config.passage_belief_transition_std) ** 2
        prior_variance = self.variance + transition_variance
        observation_variance = np.maximum(observation.variance, 1e-8)
        prior_precision = 1.0 / prior_variance
        likelihood_precision = observation.observed.astype(np.float64) / observation_variance
        posterior_variance = 1.0 / np.maximum(prior_precision + likelihood_precision, 1e-8)
        innovation = observation.mean - self.mean
        innovation[2] = _wrapped_angle(innovation[2])
        posterior_mean = self.mean + posterior_variance * likelihood_precision * innovation
        posterior_mean[0:2] = np.clip(posterior_mean[0:2], 0.03, 0.97)
        posterior_mean[2] = posterior_mean[2] % (2.0 * np.pi)
        posterior_mean[3:] = np.clip(posterior_mean[3:], 0.01, 1.0)
        tone_weight = max(0.0, float(observation.tone_precision))
        alpha = self.black_alpha + tone_weight * float(np.clip(observation.black_probability, 0.0, 1.0))
        beta = self.black_beta + tone_weight * (1.0 - float(np.clip(observation.black_probability, 0.0, 1.0)))
        return replace(
            self,
            mean=posterior_mean,
            variance=posterior_variance,
            black_alpha=alpha,
            black_beta=beta,
            update_count=self.update_count + 1,
        )

    def mean_latent(self) -> PassageLatent:
        return _vector_to_latent(self.mean, self.template, float(self.black_probability >= 0.5))

    def sample_latent(self, rng: np.random.Generator, tone: float | None = None) -> PassageLatent:
        sample = rng.normal(self.mean, np.sqrt(np.maximum(self.variance, 1e-8)))
        sample[0:2] = np.clip(sample[0:2], 0.03, 0.97)
        sample[2] %= 2.0 * np.pi
        sample[3:] = np.clip(sample[3:], 0.01, 1.0)
        sampled_tone = float(rng.uniform() < self.black_probability) if tone is None else float(tone)
        return _vector_to_latent(sample, self.template, sampled_tone)

    def transition_log_prior(self, latent: PassageLatent) -> float:
        difference = _latent_vector(latent) - self.mean
        difference[2] = _wrapped_angle(difference[2])
        gaussian = -0.5 * float(np.sum(difference * difference / np.maximum(self.variance, 1e-8)))
        tone_probability = self.black_probability if latent.tone >= 0.5 else 1.0 - self.black_probability
        return gaussian + float(np.log(max(tone_probability, 1e-8)))

    def diagnostics(self) -> dict[str, object]:
        return {
            "parameterNames": list(PASSAGE_PARAMETER_NAMES),
            "mean": self.mean.astype(float).tolist(),
            "std": np.sqrt(self.variance).astype(float).tolist(),
            "blackProbability": self.black_probability,
            "updateCount": self.update_count,
            "approximation": self.approximation,
        }


def infer_passage_observation(
    before: np.ndarray | SpatialCanvasState,
    after: np.ndarray | SpatialCanvasState,
    action: StrokeAction,
    passage: PassageLatent,
    stroke_index: int,
    config: PainterConfig,
) -> PassageObservation:
    action_center = np.asarray([0.5 * (action.x0 + action.x1), 0.5 * (action.y0 + action.y1)])
    direction = float(np.arctan2(action.y1 - action.y0, action.x1 - action.x0) % (2.0 * np.pi))
    length = float(np.hypot(action.x1 - action.x0, action.y1 - action.y0))
    mark_center = action_center
    black_probability = float(action.tone >= 0.5)
    tone_precision = 0.5
    center_variance = max(config.passage_belief_observation_std**2, 1e-8)
    approximation = "action-conditioned mark observation; no pixel material delta available"

    if isinstance(before, SpatialCanvasState) and isinstance(after, SpatialCanvasState):
        before_material = pixel_material_from_state(before)
        after_material = pixel_material_from_state(after)
        delta = np.clip(after_material[0] - before_material[0], 0.0, None)
        mass = float(delta.sum())
        if mass > 1e-8:
            rows, cols = np.indices(delta.shape)
            weights = delta / mass
            mark_center = np.asarray(
                [
                    float(np.sum(weights * (cols + 0.5) / delta.shape[1])),
                    float(np.sum(weights * (rows + 0.5) / delta.shape[0])),
                ]
            )
            if after_material.shape[0] > 3:
                black_probability = float(np.clip(np.sum(weights * after_material[3]), 0.0, 1.0))
                tone_precision = min(12.0, 1.0 + 4.0 * mass / max(config.thickness_scale, 1e-8))
            center_variance = max(1e-6, config.passage_belief_observation_std**2 / (1.0 + mass))
            approximation = "pixel thickness-delta centroid and surface-tone likelihood; geometry uses executed mark"

    midpoint = 0.5 * (passage.stroke_count - 1)
    offset = float(stroke_index - midpoint) * passage.spacing
    passage_direction = np.asarray([np.cos(direction), np.sin(direction)])
    passage_normal = np.asarray([-passage_direction[1], passage_direction[0]])
    offset_direction = passage_direction if passage.kind == "chain" else passage_normal
    inferred_center = mark_center - offset * offset_direction
    mean = np.asarray(
        [
            inferred_center[0],
            inferred_center[1],
            direction,
            length,
            passage.spacing,
            action.width,
            action.amount,
        ],
        dtype=np.float64,
    )
    geometry_variance = max(config.passage_belief_observation_std**2, 1e-8)
    variance = np.asarray(
        [
            center_variance,
            center_variance,
            geometry_variance,
            geometry_variance,
            1.0,
            geometry_variance,
            geometry_variance,
        ],
        dtype=np.float64,
    )
    observed = np.asarray([True, True, True, True, False, True, True], dtype=bool)
    return PassageObservation(mean, variance, observed, black_probability, tone_precision, approximation)


def _latent_vector(latent: PassageLatent) -> np.ndarray:
    return np.asarray(
        [
            latent.center_x,
            latent.center_y,
            latent.direction,
            latent.length,
            latent.spacing,
            latent.width,
            latent.amount,
        ],
        dtype=np.float64,
    )


def _vector_to_latent(values: np.ndarray, template: PassageLatent, tone: float) -> PassageLatent:
    return replace(
        template,
        center_x=float(values[0]),
        center_y=float(values[1]),
        direction=float(values[2]),
        length=float(values[3]),
        spacing=float(values[4]),
        width=float(values[5]),
        amount=float(values[6]),
        tone=float(tone),
    )


def _wrapped_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)
